#include "inpaint.h"

#include <algorithm>
#include <array>
#include <iostream>

namespace
{
    std::vector<double> make_distance2similarity()
    {
        constexpr std::array<double, 11> base = {1.0, 0.99, 0.96, 0.83, 0.38, 0.11, 0.02, 0.005, 0.0006, 0.0001, 0};
        const std::size_t length = static_cast<std::size_t>(PatchDistanceMetric::kDistanceScale) + 1;
        std::vector<double> table(length);
        for (std::size_t i = 0; i < length; ++i)
        {
            const double t = static_cast<double>(i) / static_cast<double>(length);
            const auto j = static_cast<std::size_t>(100 * t);
            if (j >= base.size() - 1)
                continue; // the table is zero from here on
            table[i] = base[j] + (100 * t - static_cast<double>(j)) * (base[j + 1] - base[j]);
        }
        return table;
    }

    // Built on first use. The initialization of a local static is thread-safe, which
    // matters because the Python bindings release the GIL during inpainting.
    const std::vector<double> &distance2similarity()
    {
        static const std::vector<double> table = make_distance2similarity();
        return table;
    }

    inline void _weighted_copy(
        const MaskedImage &source, int ys, int xs, cv::Mat &target, int yt, int xt, double weight)
    {
        if (source.is_masked(ys, xs))
            return;
        if (source.is_globally_masked(ys, xs))
            return;

        auto source_ptr = source.get_image(ys, xs);
        auto target_ptr = target.ptr<double>(yt, xt);

        for (int c = 0; c < 3; ++c)
            target_ptr[c] += static_cast<double>(source_ptr[c]) * weight;
        target_ptr[3] += weight;
    }

    // Replaces the vote of the pixel if the new one weighs more. It is stored like the
    // sums of _weighted_copy, so the maximization step divides it by its weight.
    inline void _best_copy(const MaskedImage &source, int ys, int xs, cv::Mat &target, int yt, int xt, double weight)
    {
        if (source.is_masked(ys, xs))
            return;
        if (source.is_globally_masked(ys, xs))
            return;

        auto source_ptr = source.get_image(ys, xs);
        auto target_ptr = target.ptr<double>(yt, xt);
        if (weight <= target_ptr[3])
            return;

        for (int c = 0; c < 3; ++c)
            target_ptr[c] = static_cast<double>(source_ptr[c]) * weight;
        target_ptr[3] = weight;
    }
} // namespace

/**
 * This algorithm uses a version proposed by Xavier Philippeau.
 */

Inpainting::Inpainting(cv::Mat image, cv::Mat mask, const PatchDistanceMetric *metric)
    : m_initial(image, mask), m_pyramid(), m_source2target(), m_target2source(), m_distance_metric(metric)
{
    _initialize_pyramid();
}

Inpainting::Inpainting(cv::Mat image, cv::Mat mask, cv::Mat global_mask, const PatchDistanceMetric *metric)
    : m_initial(image, mask, global_mask), m_pyramid(), m_source2target(), m_target2source(), m_distance_metric(metric)
{
    _initialize_pyramid();
}

void Inpainting::_initialize_pyramid()
{
    auto source = m_initial;
    m_pyramid.push_back(source);
    while (source.size().height > m_distance_metric->patch_size() &&
           source.size().width > m_distance_metric->patch_size())
    {
        source = source.downsample();
        m_pyramid.push_back(source);
    }
}

cv::Mat Inpainting::run(bool verbose, unsigned int random_seed)
{
    NearestNeighborField::seed_random(random_seed);
    const auto nr_levels = static_cast<int>(m_pyramid.size());

    MaskedImage source, target;
    for (int level = nr_levels - 1; level >= 0; --level)
    {
        if (verbose)
            std::cerr << "Inpainting level: " << level << std::endl;

        source = m_pyramid[level];

        if (level == nr_levels - 1)
        {
            target = source.clone();
            target.clear_mask();
            m_source2target = NearestNeighborField(source, target, m_distance_metric);
            m_target2source = NearestNeighborField(target, source, m_distance_metric);
        }
        else
        {
            m_source2target = NearestNeighborField(source, target, m_distance_metric, m_source2target);
            m_target2source = NearestNeighborField(target, source, m_distance_metric, m_target2source);
        }

        if (verbose)
            std::cerr << "Initialization done." << std::endl;

        target = _expectation_maximization(source, target, level, verbose);
    }

    // Only the holes are filled. The maximization step of level 0 keeps the known pixels,
    // but the pyramid leaves globally masked pixels black, which are neither filled nor
    // used as a source. Copy every pixel outside the holes from the input.
    // A plain loop instead of OpenCV matrix expressions: those pull OpenCV's whole
    // expression module into the statically linked library, 40 % more on disk.
    cv::Mat result = target.image();
    for (int y = 0; y < result.rows; ++y)
    {
        for (int x = 0; x < result.cols; ++x)
        {
            if (m_initial.is_masked(y, x) && !m_initial.is_globally_masked(y, x))
                continue;
            const unsigned char *input = m_initial.get_image(y, x);
            std::copy(input, input + 3, result.ptr<unsigned char>(y, x));
        }
    }
    return result;
}

// EM-Like algorithm (see "PatchMatch" - page 6).
// Returns a double sized target image (unless level = 0).
MaskedImage Inpainting::_expectation_maximization(MaskedImage source, MaskedImage target, int level, bool verbose)
{
    const int nr_iters_em = 1 + 2 * level;
    const int nr_iters_nnf = static_cast<int>(std::min(7, 1 + level));
    const int patch_size = m_distance_metric->patch_size();

    MaskedImage new_source, new_target;

    for (int iter_em = 0; iter_em < nr_iters_em; ++iter_em)
    {
        if (iter_em != 0)
        {
            m_source2target.set_target(new_target);
            m_target2source.set_source(new_target);
            target = new_target;
        }

        if (verbose)
            std::cerr << "EM Iteration: " << iter_em << std::endl;

        auto size = source.size();
        for (int i = 0; i < size.height; ++i)
        {
            for (int j = 0; j < size.width; ++j)
            {
                if (!source.contains_mask(i, j, patch_size))
                {
                    m_source2target.set_identity(i, j);
                    m_target2source.set_identity(i, j);
                }
            }
        }
        if (verbose)
            std::cerr << "  NNF minimization started." << std::endl;
        m_source2target.minimize(nr_iters_nnf);
        m_target2source.minimize(nr_iters_nnf);
        if (verbose)
            std::cerr << "  NNF minimization finished." << std::endl;

        // Instead of upsizing the final target, we build the last target from the next level source image.
        // Thus, the final target is less blurry (see "Space-Time Video Completion" - page 5).
        bool upscaled = false;
        if (level >= 1 && iter_em == nr_iters_em - 1)
        {
            new_source = m_pyramid[level - 1];
            new_target =
                target.upsample(new_source.size().width, new_source.size().height, m_pyramid[level - 1].global_mask());
            upscaled = true;
        }
        else
        {
            new_source = m_pyramid[level];
            new_target = target.clone();
        }

        auto vote = cv::Mat(new_target.size(), CV_64FC4);
        vote.setTo(cv::Scalar::all(0));

        // The weighted mean of all overlapping patches blurs the fill, so in the last
        // iteration of the finest level each pixel takes the value of its most similar
        // patch instead (similar to "Space-Time Completion of Video", which takes the
        // mode of the votes). On coarser levels the mean is kept: it lays out the smooth
        // structure that the finer levels refine, the best vote there makes seams.
        const bool best_only = level == 0 && iter_em == nr_iters_em - 1;

        // Votes for best patch from NNF Source->Target (completeness) and Target->Source (coherence).
        _expectation_step(m_source2target, true, vote, new_source, upscaled, best_only);
        if (verbose)
            std::cerr << "  Expectation source to target finished." << std::endl;
        _expectation_step(m_target2source, false, vote, new_source, upscaled, best_only);
        if (verbose)
            std::cerr << "  Expectation target to source finished." << std::endl;

        // Compile votes and update pixel values.
        // Known pixels are kept on the two finest levels only. On coarser levels those
        // next to the holes average only the known part of their downsampling kernel,
        // estimates that reach into the holes; keeping them there made the fills copy
        // smooth regions, e.g. a concrete wall into the foliage above it.
        const int new_level = upscaled ? level - 1 : level;
        _maximization_step(new_target, vote, new_source, new_level <= 1);
        if (verbose)
            std::cerr << "  Minimization step finished." << std::endl;
    }

    return new_target;
}

// Expectation step: vote for best estimations of each pixel.
void Inpainting::_expectation_step(
    const NearestNeighborField &nnf, bool source2target, cv::Mat &vote, const MaskedImage &source, bool upscaled,
    bool best_only)
{
    auto source_size = nnf.source_size();
    auto target_size = nnf.target_size();
    const int patch_size = m_distance_metric->patch_size();
    const auto &kDistance2Similarity = distance2similarity();

    double w = 0;
    auto copy = [&](int ys, int xs, int yt, int xt) {
        if (best_only)
            _best_copy(source, ys, xs, vote, yt, xt, w);
        else
            _weighted_copy(source, ys, xs, vote, yt, xt, w);
    };

    for (int i = 0; i < source_size.height; ++i)
    {
        for (int j = 0; j < source_size.width; ++j)
        {
            if (nnf.source().is_globally_masked(i, j))
                continue;
            int yp = nnf.at(i, j, 0), xp = nnf.at(i, j, 1), dp = nnf.at(i, j, 2);
            w = kDistance2Similarity[dp];

            for (int di = -patch_size; di <= patch_size; ++di)
            {
                for (int dj = -patch_size; dj <= patch_size; ++dj)
                {
                    int ys = i + di, xs = j + dj, yt = yp + di, xt = xp + dj;
                    if (!(ys >= 0 && ys < source_size.height && xs >= 0 && xs < source_size.width))
                        continue;
                    if (nnf.source().is_globally_masked(ys, xs))
                        continue;
                    if (!(yt >= 0 && yt < target_size.height && xt >= 0 && xt < target_size.width))
                        continue;
                    if (nnf.target().is_globally_masked(yt, xt))
                        continue;

                    if (!source2target)
                    {
                        std::swap(ys, yt);
                        std::swap(xs, xt);
                    }

                    if (upscaled)
                    {
                        for (int uy = 0; uy < 2; ++uy)
                        {
                            for (int ux = 0; ux < 2; ++ux)
                            {
                                copy(2 * ys + uy, 2 * xs + ux, 2 * yt + uy, 2 * xt + ux);
                            }
                        }
                    }
                    else
                    {
                        copy(ys, xs, yt, xt);
                    }
                }
            }
        }
    }
}

// Maximization Step: maximum likelihood of target pixel.
void Inpainting::_maximization_step(
    MaskedImage &target, const cv::Mat &vote, const MaskedImage &source, bool keep_known) const
{
    auto target_size = target.size();
    for (int i = 0; i < target_size.height; ++i)
    {
        for (int j = 0; j < target_size.width; ++j)
        {
            const double *source_ptr = vote.ptr<double>(i, j);
            unsigned char *target_ptr = target.get_mutable_image(i, j);

            if (target.is_globally_masked(i, j))
            {
                continue;
            }

            // Known pixels keep their values. Otherwise the votes of patches that overlap
            // a hole change them, the holes are filled to match the changed surroundings,
            // and a seam appears once the known pixels are restored at the end.
            if (keep_known && !source.is_masked(i, j))
            {
                const unsigned char *known = source.get_image(i, j);
                std::copy(known, known + 3, target_ptr);
                continue;
            }

            if (source_ptr[3] > 0)
            {
                unsigned char r = cv::saturate_cast<unsigned char>(source_ptr[0] / source_ptr[3]);
                unsigned char g = cv::saturate_cast<unsigned char>(source_ptr[1] / source_ptr[3]);
                unsigned char b = cv::saturate_cast<unsigned char>(source_ptr[2] / source_ptr[3]);
                target_ptr[0] = r, target_ptr[1] = g, target_ptr[2] = b;
            }
            else
            {
                target.set_mask(i, j, 0);
            }
        }
    }
}
