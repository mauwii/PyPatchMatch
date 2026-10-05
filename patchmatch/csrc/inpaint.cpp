#include "inpaint.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>

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

    // Calls visit(y, x) for the pixels of the image within radius of (cy, cx).
    template <typename Visit>
    void for_window(cv::Size size, int cy, int cx, int radius, Visit visit)
    {
        for (int y = std::max(0, cy - radius); y <= std::min(size.height - 1, cy + radius); ++y)
            for (int x = std::max(0, cx - radius); x <= std::min(size.width - 1, cx + radius); ++x)
                visit(y, x);
    }

    // Mean of value(y, x) over the pixels within radius of (cy, cx) for which select(y, x) holds.
    template <typename Select, typename Value>
    cv::Vec3d window_mean(cv::Size size, int cy, int cx, int radius, Select select, Value value)
    {
        cv::Vec3d sum;
        int count = 0;
        for_window(size, cy, cx, radius, [&sum, &count, &select, &value](int y, int x) {
            if (select(y, x))
            {
                sum += value(y, x);
                ++count;
            }
        });
        return count > 0 ? sum / count : sum;
    }

    bool is_known(const MaskedImage &image, int y, int x)
    {
        return !image.is_masked(y, x) && !image.is_globally_masked(y, x);
    }

    bool is_hole(const MaskedImage &image, int y, int x)
    {
        return image.is_masked(y, x) && !image.is_globally_masked(y, x);
    }

    // Chessboard distance of the hole pixels to the nearest known pixel, 0 beyond radius.
    cv::Mat hole_depths(const MaskedImage &image, int radius)
    {
        const auto size = image.size();
        cv::Mat depth(size, CV_32S, cv::Scalar(0));
        for (int cy = 0; cy < size.height; ++cy)
        {
            for (int cx = 0; cx < size.width; ++cx)
            {
                if (!is_hole(image, cy, cx))
                    continue;
                int &d = depth.at<int>(cy, cx);
                for_window(size, cy, cx, radius, [&d, &image, cy, cx](int y, int x) {
                    const int distance = std::max(std::abs(y - cy), std::abs(x - cx));
                    if (is_known(image, y, x) && (d == 0 || distance < d))
                        d = distance;
                });
            }
        }
        return depth;
    }

    // Shifts the hole pixels near the border by the color step across it, which showed as an outline.
    void blend_hole_borders(cv::Mat &result, const MaskedImage &initial)
    {
        constexpr int kWidth = 4;
        constexpr int kStepRadius = 1;
        constexpr int kSpreadRadius = 4;
        const auto size = initial.size();
        const auto known = [&initial](int y, int x) { return is_known(initial, y, x); };
        const auto hole = [&initial](int y, int x) { return is_hole(initial, y, x); };
        const auto color = [&result](int y, int x) {
            const auto *pixel = result.ptr<unsigned char>(y, x);
            return cv::Vec3d(pixel[0], pixel[1], pixel[2]);
        };

        const cv::Mat depth = hole_depths(initial, kWidth);
        const auto border = [&depth](int y, int x) { return depth.at<int>(y, x) == 1; };

        cv::Mat steps(size, CV_64FC3, cv::Scalar::all(0));
        for (int y = 0; y < size.height; ++y)
        {
            for (int x = 0; x < size.width; ++x)
            {
                if (border(y, x))
                    steps.at<cv::Vec3d>(y, x) = window_mean(size, y, x, kStepRadius, known, color) -
                                                window_mean(size, y, x, kStepRadius, hole, color);
            }
        }

        const auto step = [&steps](int y, int x) { return steps.at<cv::Vec3d>(y, x); };
        for (int y = 0; y < size.height; ++y)
        {
            for (int x = 0; x < size.width; ++x)
            {
                const int d = depth.at<int>(y, x);
                if (d == 0)
                    continue;
                const cv::Vec3d shift =
                    window_mean(size, y, x, kSpreadRadius, border, step) * (kWidth + 1 - d) / kWidth;
                auto *pixel = result.ptr<unsigned char>(y, x);
                for (int c = 0; c < 3; ++c)
                    pixel[c] = cv::saturate_cast<unsigned char>(pixel[c] + shift[c]);
            }
        }
    }

    template <typename... Args>
    void trace(bool verbose, const Args &...args)
    {
        if (verbose)
            (std::cerr << ... << args) << std::endl;
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

        auto source_ptr = source.get_image(ys, xs);
        auto target_ptr = target.ptr<double>(yt, xt);
        if (weight <= target_ptr[3])
            return;

        for (int c = 0; c < 3; ++c)
            target_ptr[c] = static_cast<double>(source_ptr[c]) * weight;
        target_ptr[3] = weight;
    }

    // Counts the holes of an image in rectangles, from the number of holes above and left of each pixel.
    class HoleCounter
    {
    public:
        explicit HoleCounter(const MaskedImage &image)
            : m_size(image.size()), m_sums(static_cast<std::size_t>(m_size.height + 1) * (m_size.width + 1), 0)
        {
            for (int y = 0; y < m_size.height; ++y)
            {
                for (int x = 0; x < m_size.width; ++x)
                {
                    const std::int64_t hole = image.is_masked(y, x) ? 1 : 0;
                    sum(y + 1, x + 1) = hole + sum(y, x + 1) + sum(y + 1, x) - sum(y, x);
                }
            }
        }

        // Whether the rectangle from (y0, x0) to (y1, x1), clipped to the image, contains a hole.
        bool any(std::int64_t y0, std::int64_t x0, std::int64_t y1, std::int64_t x1) const
        {
            const auto clip = [](std::int64_t value, int size) {
                return static_cast<int>(std::clamp<std::int64_t>(value, 0, size));
            };
            const int top = clip(y0, m_size.height);
            const int bottom = clip(y1 + 1, m_size.height);
            const int left = clip(x0, m_size.width);
            const int right = clip(x1 + 1, m_size.width);
            return sum(bottom, right) - sum(top, right) - sum(bottom, left) + sum(top, left) > 0;
        }

    private:
        std::int64_t &sum(int y, int x)
        {
            return m_sums[static_cast<std::size_t>(y) * (m_size.width + 1) + x];
        }
        std::int64_t sum(int y, int x) const
        {
            return m_sums[static_cast<std::size_t>(y) * (m_size.width + 1) + x];
        }

        cv::Size m_size;
        std::vector<std::int64_t> m_sums;
    };

    // Casts the votes of one nearest-neighbor field into the vote image.
    class VoteCaster
    {
    public:
        // With holes, only the patches whose votes reach one of them are cast.
        VoteCaster(
            const NearestNeighborField &nnf, bool source2target, cv::Mat &vote, const MaskedImage &source,
            bool upscaled, bool best_only, const HoleCounter *holes)
            : m_nnf(&nnf), m_vote(&vote), m_source(&source), m_holes(holes), m_source_size(nnf.source_size()),
              m_target_size(nnf.target_size()), m_source2target(source2target), m_upscaled(upscaled),
              m_best_only(best_only)
        {
        }

        // Every pixel of the patch around (i, j) votes for the pixel at the same offset
        // in the patch around its nearest neighbor.
        void cast_patch(int i, int j, int patch_size, double weight) const
        {
            const int yp = m_nnf->at(i, j, 0);
            const int xp = m_nnf->at(i, j, 1);
            if (m_holes && !votes_for_hole(m_source2target ? yp : i, m_source2target ? xp : j, patch_size))
                return;
            for (int di = -patch_size; di <= patch_size; ++di)
            {
                for (int dj = -patch_size; dj <= patch_size; ++dj)
                {
                    const int ys = i + di;
                    const int xs = j + dj;
                    const int yt = yp + di;
                    const int xt = xp + dj;
                    if (ys < 0 || ys >= m_source_size.height || xs < 0 || xs >= m_source_size.width ||
                        m_nnf->source().is_globally_masked(ys, xs))
                        continue;
                    if (yt < 0 || yt >= m_target_size.height || xt < 0 || xt >= m_target_size.width ||
                        m_nnf->target().is_globally_masked(yt, xt))
                        continue;
                    // Near ties go to the patch centered closest, not the first cast, which made blocks.
                    const double offset2 = static_cast<double>(di) * di + static_cast<double>(dj) * dj;
                    cast_pair(ys, xs, yt, xt, m_best_only ? weight * (1 - 1e-5 * offset2) : weight);
                }
            }
        }

    private:
        // Whether the patch around (y, x) of the voted image, scaled like the votes, has a hole.
        bool votes_for_hole(int y, int x, int patch_size) const
        {
            const std::int64_t scale = m_upscaled ? 2 : 1;
            return m_holes->any(
                scale * (std::int64_t{y} - patch_size), scale * (std::int64_t{x} - patch_size),
                scale * (std::int64_t{y} + patch_size + 1) - 1, scale * (std::int64_t{x} + patch_size + 1) - 1);
        }

        void cast_pair(int ys, int xs, int yt, int xt, double weight) const
        {
            if (!m_source2target)
            {
                std::swap(ys, yt);
                std::swap(xs, xt);
            }
            if (!m_upscaled)
            {
                copy(ys, xs, yt, xt, weight);
                return;
            }
            for (int uy = 0; uy < 2; ++uy)
                for (int ux = 0; ux < 2; ++ux)
                    copy(2 * ys + uy, 2 * xs + ux, 2 * yt + uy, 2 * xt + ux, weight);
        }

        void copy(int ys, int xs, int yt, int xt, double weight) const
        {
            if (m_best_only)
                _best_copy(*m_source, ys, xs, *m_vote, yt, xt, weight);
            else
                _weighted_copy(*m_source, ys, xs, *m_vote, yt, xt, weight);
        }

        const NearestNeighborField *m_nnf;
        cv::Mat *m_vote;
        const MaskedImage *m_source;
        const HoleCounter *m_holes;
        cv::Size m_source_size;
        cv::Size m_target_size;
        bool m_source2target;
        bool m_upscaled;
        bool m_best_only;
    };
} // namespace

/**
 * This algorithm uses a version proposed by Xavier Philippeau.
 */

Inpainting::Inpainting(const cv::Mat &image, const cv::Mat &mask, const PatchDistanceMetric *metric)
    : m_initial(image, mask), m_distance_metric(metric)
{
    _initialize_pyramid();
}

Inpainting::Inpainting(
    const cv::Mat &image, const cv::Mat &mask, const cv::Mat &global_mask, const PatchDistanceMetric *metric)
    : m_initial(image, mask, global_mask), m_distance_metric(metric)
{
    _initialize_pyramid();
}

void Inpainting::_initialize_pyramid()
{
    // The colors under the holes would reach the fill through the gradients of their
    // neighbors and the target of a single-level pyramid. Black, like on coarser levels.
    auto source = m_initial.clone();
    const auto size = source.size();
    for (int y = 0; y < size.height; ++y)
    {
        for (int x = 0; x < size.width; ++x)
        {
            if (source.is_masked(y, x))
                std::fill_n(source.get_mutable_image(y, x), 3, 0);
        }
    }
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

    MaskedImage source;
    MaskedImage target;
    for (int level = nr_levels - 1; level >= 0; --level)
    {
        trace(verbose, "Inpainting level: ", level);

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

        trace(verbose, "Initialization done.");

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
    blend_hole_borders(result, m_initial);
    return result;
}

// EM-Like algorithm (see "PatchMatch" - page 6).
// Returns a double sized target image (unless level = 0).
MaskedImage Inpainting::_expectation_maximization(
    const MaskedImage &source, MaskedImage target, int level, bool verbose)
{
    const int nr_iters_em = 1 + 2 * level;
    const int nr_iters_nnf = std::min(7, 1 + level);

    MaskedImage new_source;
    MaskedImage new_target;

    for (int iter_em = 0; iter_em < nr_iters_em; ++iter_em)
    {
        if (iter_em != 0)
        {
            m_source2target.set_target(new_target);
            m_target2source.set_source(new_target);
            target = new_target;
        }

        trace(verbose, "EM Iteration: ", iter_em);

        _link_patches_without_holes(source);
        trace(verbose, "  NNF minimization started.");
        m_source2target.minimize(nr_iters_nnf);
        m_target2source.minimize(nr_iters_nnf);
        trace(verbose, "  NNF minimization finished.");

        // Instead of upsizing the final target, we build the last target from the next level source image.
        // Thus, the final target is less blurry (see "Space-Time Video Completion" - page 5).
        const bool upscaled = level >= 1 && iter_em == nr_iters_em - 1;
        if (upscaled)
        {
            new_source = m_pyramid[level - 1];
            new_target =
                target.upsample(new_source.size().width, new_source.size().height, m_pyramid[level - 1].global_mask());
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

        // Known pixels are kept on the two finest levels only. On coarser levels those
        // next to the holes average only the known part of their downsampling kernel,
        // estimates that reach into the holes; keeping them there made the fills copy
        // smooth regions, e.g. a concrete wall into the foliage above it.
        const int new_level = upscaled ? level - 1 : level;
        const bool keep_known = new_level <= 1;

        // Votes for best patch from NNF Source->Target (completeness) and Target->Source (coherence).
        _expectation_step(m_source2target, true, vote, new_source, upscaled, best_only, keep_known);
        trace(verbose, "  Expectation source to target finished.");
        _expectation_step(m_target2source, false, vote, new_source, upscaled, best_only, keep_known);
        trace(verbose, "  Expectation target to source finished.");

        // Compile votes and update pixel values.
        _maximization_step(new_target, vote, new_source, keep_known);
        trace(verbose, "  Minimization step finished.");
    }

    return new_target;
}

// Patches without holes are their own nearest neighbors.
void Inpainting::_link_patches_without_holes(const MaskedImage &source)
{
    const int patch_size = m_distance_metric->patch_size();
    const auto size = source.size();
    for (int i = 0; i < size.height; ++i)
    {
        for (int j = 0; j < size.width; ++j)
        {
            if (source.contains_mask(i, j, patch_size))
                continue;
            m_source2target.set_identity(i, j);
            m_target2source.set_identity(i, j);
        }
    }
}

// Expectation step: vote for best estimations of each pixel.
void Inpainting::_expectation_step(
    const NearestNeighborField &nnf, bool source2target, cv::Mat &vote, const MaskedImage &source, bool upscaled,
    bool best_only, bool keep_known) const
{
    // Votes for known pixels do not count where the maximization step keeps them.
    std::optional<HoleCounter> holes;
    if (keep_known)
        holes.emplace(source);
    const VoteCaster caster(
        nnf, source2target, vote, source, upscaled, best_only, holes.has_value() ? &holes.value() : nullptr);
    const int patch_size = m_distance_metric->patch_size();
    const auto &kDistance2Similarity = distance2similarity();
    const auto source_size = nnf.source_size();
    for (int i = 0; i < source_size.height; ++i)
    {
        for (int j = 0; j < source_size.width; ++j)
        {
            if (!nnf.source().is_globally_masked(i, j))
                caster.cast_patch(i, j, patch_size, kDistance2Similarity[nnf.at(i, j, 2)]);
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
                // All three before the stores: they may alias the votes, which changes the fast-math divisions.
                const auto r = cv::saturate_cast<unsigned char>(source_ptr[0] / source_ptr[3]);
                const auto g = cv::saturate_cast<unsigned char>(source_ptr[1] / source_ptr[3]);
                const auto b = cv::saturate_cast<unsigned char>(source_ptr[2] / source_ptr[3]);
                target_ptr[0] = r;
                target_ptr[1] = g;
                target_ptr[2] = b;
            }
            else
            {
                target.set_mask(i, j, false);
            }
        }
    }
}
