#include "inpaint.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <utility>

#include "holes.h"
#include "parallel.h"
#include "peel.h"
#include "votes.h"
#include "window.h"

namespace
{
    // The pixel loops read three channels and index the masks with the image coordinates.
    void check_inputs(const MaskedImage &image)
    {
        if (image.image().empty())
            return;
        CV_Assert(image.image().type() == CV_8UC3);
        CV_Assert(image.mask().type() == CV_8UC1 && image.mask().size() == image.size());
        if (image.has_global_mask())
            CV_Assert(image.global_mask().type() == CV_8UC1 && image.global_mask().size() == image.size());
    }

    // The gradient magnitude at a known pixel, summed over the channels, or -1 if it or a neighbor is unknown.
    double gradient_magnitude(const MaskedImage &image, int y, int x)
    {
        const auto size = image.size();
        const int up = std::max(y - 1, 0);
        const int down = std::min(y + 1, size.height - 1);
        const int left = std::max(x - 1, 0);
        const int right = std::min(x + 1, size.width - 1);
        if (!is_known(image, y, x) || !is_known(image, up, x) || !is_known(image, down, x) ||
            !is_known(image, y, left) || !is_known(image, y, right))
            return -1;
        int magnitude = 0;
        for (int c = 0; c < 3; ++c)
            magnitude += std::abs(image.get_image(down, x)[c] - image.get_image(up, x)[c]) +
                         std::abs(image.get_image(y, right)[c] - image.get_image(y, left)[c]);
        return magnitude / 2.0;
    }

    // The mean gradient magnitude around each known pixel, -1 elsewhere or without one. Gradients
    // across unknown pixels are left out, so the colors under the holes do not matter.
    cv::Mat texture_energy(const MaskedImage &image)
    {
        constexpr int kRadius = 2;
        const auto size = image.size();
        // above and left of each pixel: the sum of the magnitudes and their number
        cv::Mat sums(size.height + 1, size.width + 1, CV_64FC2, cv::Scalar::all(0));
        for (int y = 0; y < size.height; ++y)
        {
            for (int x = 0; x < size.width; ++x)
            {
                const double magnitude = gradient_magnitude(image, y, x);
                const cv::Vec2d measured = magnitude >= 0 ? cv::Vec2d(magnitude, 1) : cv::Vec2d();
                sums.at<cv::Vec2d>(y + 1, x + 1) =
                    measured + sums.at<cv::Vec2d>(y, x + 1) + sums.at<cv::Vec2d>(y + 1, x) - sums.at<cv::Vec2d>(y, x);
            }
        }

        cv::Mat energy(size, CV_64F, cv::Scalar(-1));
        for (int y = 0; y < size.height; ++y)
        {
            const int top = std::max(y - kRadius, 0);
            const int bottom = std::min(y + kRadius + 1, size.height);
            for (int x = 0; x < size.width; ++x)
            {
                const int left = std::max(x - kRadius, 0);
                const int right = std::min(x + kRadius + 1, size.width);
                const cv::Vec2d window = sums.at<cv::Vec2d>(bottom, right) - sums.at<cv::Vec2d>(top, right) -
                                         sums.at<cv::Vec2d>(bottom, left) + sums.at<cv::Vec2d>(top, left);
                if (is_known(image, y, x) && window[1] > 0)
                    energy.at<double>(y, x) = window[0] / window[1];
            }
        }
        return energy;
    }

    // The mean of the values >= 0 in the 2x2 block from (y, x), -1 if there is none.
    double block_mean(const cv::Mat &values, int y, int x)
    {
        double sum = 0;
        int count = 0;
        for (int yy = y; yy < std::min(y + 2, values.rows); ++yy)
        {
            for (int xx = x; xx < std::min(x + 2, values.cols); ++xx)
            {
                if (values.at<double>(yy, xx) >= 0)
                {
                    sum += values.at<double>(yy, xx);
                    ++count;
                }
            }
        }
        return count > 0 ? sum / count : -1;
    }

    // The value of coarse, which has half the resolution, at the center of pixel (y, x), bilinear.
    double interpolate(const cv::Mat &coarse, int y, int x)
    {
        const double cy = std::clamp((y + 0.5) / 2 - 0.5, 0.0, coarse.rows - 1.0);
        const double cx = std::clamp((x + 0.5) / 2 - 0.5, 0.0, coarse.cols - 1.0);
        const auto y0 = static_cast<int>(cy);
        const auto x0 = static_cast<int>(cx);
        const int y1 = std::min(y0 + 1, coarse.rows - 1);
        const int x1 = std::min(x0 + 1, coarse.cols - 1);
        const double fy = cy - y0;
        const double fx = cx - x0;
        return (1 - fy) * ((1 - fx) * coarse.at<double>(y0, x0) + fx * coarse.at<double>(y0, x1)) +
               fy * ((1 - fx) * coarse.at<double>(y1, x0) + fx * coarse.at<double>(y1, x1));
    }

    // Not std::all_of: libstdc++ subtracts the iterators, which divides by zero for an empty cv::Mat.
    bool has_unknown(const cv::Mat &values)
    {
        for (int y = 0; y < values.rows; ++y)
            for (int x = 0; x < values.cols; ++x)
                if (values.at<double>(y, x) < 0)
                    return true;
        return false;
    }

    // Replaces the values below 0 from a pyramid of the others (pull-push), which carries the
    // values around a hole smoothly into it.
    void pull_push(cv::Mat &values)
    {
        if (!has_unknown(values))
            return;
        if (values.total() == 1)
        {
            values.at<double>(0, 0) = 0; // nothing is known
            return;
        }
        cv::Mat coarse((values.rows + 1) / 2, (values.cols + 1) / 2, CV_64F);
        for (int y = 0; y < coarse.rows; ++y)
            for (int x = 0; x < coarse.cols; ++x)
                coarse.at<double>(y, x) = block_mean(values, 2 * y, 2 * x);
        pull_push(coarse);
        for (int y = 0; y < values.rows; ++y)
        {
            for (int x = 0; x < values.cols; ++x)
            {
                if (values.at<double>(y, x) < 0)
                    values.at<double>(y, x) = interpolate(coarse, y, x);
            }
        }
    }

    // The texture of the next level: the means of the 2x2 blocks.
    cv::Mat halve(const cv::Mat &values, cv::Size size)
    {
        cv::Mat half(size, CV_64F);
        for (int y = 0; y < size.height; ++y)
            for (int x = 0; x < size.width; ++x)
                half.at<double>(y, x) = block_mean(values, 2 * y, 2 * x);
        return half;
    }

    cv::Mat to_bytes(const cv::Mat &values)
    {
        cv::Mat bytes(values.size(), CV_8U);
        for (int y = 0; y < values.rows; ++y)
            for (int x = 0; x < values.cols; ++x)
                bytes.at<unsigned char>(y, x) = cv::saturate_cast<unsigned char>(values.at<double>(y, x));
        return bytes;
    }

    // A patch wider than a sixth of a level found no sources there inside the image and outside
    // the holes, and the fill stayed flat. Patches up to 7 keep their size: smaller ones lost the joints of bricks.
    constexpr int kLevelSidesPerPatch = 6;
    constexpr int kMinFittingPatchSize = 7;

    int nnf_iterations(int level)
    {
        return std::min(7, 1 + level);
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
    check_inputs(m_initial);

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

    // The coarser levels cannot tell blurred from sharp sources by their colors, so every level
    // compares the texture of the finest one, carried into the holes from their surroundings.
    cv::Mat texture = texture_energy(m_pyramid[0]);
    pull_push(texture);
    for (std::size_t level = 0; level < m_pyramid.size(); ++level)
    {
        if (level > 0)
            texture = halve(texture, m_pyramid[level].size());
        m_pyramid[level].set_texture(to_bytes(texture));
    }

    for (const MaskedImage &image : m_pyramid)
    {
        const int side = std::min(image.size().height, image.size().width);
        const int fitting = std::max(kMinFittingPatchSize, (side / kLevelSidesPerPatch - 1) / 2);
        m_fitting_metrics.push_back(
            fitting < m_distance_metric->patch_size() ? m_distance_metric->with_patch_size(fitting) : nullptr);
    }
    if (m_distance_metric->patch_size() < kCoarsePatchSize)
    {
        m_hole_metric = m_distance_metric->with_patch_size(kCoarsePatchSize);
        m_peel_labels = peel_labels(m_pyramid, 0, std::numeric_limits<int>::max(), m_peel_reach);
        for (std::size_t level = 0; level < m_pyramid.size(); ++level)
            m_hole_pixels.push_back(
                coarse_pixels(m_pyramid[level], m_peel_labels[level], m_peel_reach[level], static_cast<int>(level)));
        return;
    }
    m_hole_metric = m_distance_metric->with_patch_size(kLinePatchSize);
    m_peel_labels = peel_labels(m_pyramid, kLeastLineReach, kMostLineReach, m_peel_reach);
    for (std::size_t level = 0; level < m_pyramid.size(); ++level)
    {
        const auto &fitting = m_fitting_metrics[level];
        m_hole_pixels.push_back(pixels_around(
            m_pyramid[level], m_peel_labels[level], m_peel_reach[level], static_cast<int>(level) + 1,
            fitting ? fitting->patch_size() : m_distance_metric->patch_size()));
    }
}

cv::Mat Inpainting::run(bool verbose, unsigned int random_seed)
{
    NearestNeighborField::seed_random(random_seed);
    MaskedImage target = _run_levels(verbose);

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

// Fills the holes level by level, from the coarsest one down, and returns level 0.
MaskedImage Inpainting::_run_levels(bool verbose)
{
    MaskedImage target;
    for (int level = static_cast<int>(m_pyramid.size()) - 1; level >= 0; --level)
    {
        trace(verbose, "Inpainting level: ", level);

        const MaskedImage &source = m_pyramid[level];
        m_level_metric = m_fitting_metrics[level] ? m_fitting_metrics[level].get() : m_distance_metric;
        // Below kCoarsePatchSize the larger patches only around the holes that end on this level: below, the holes are
        // wider than them, and they took flat sources in foliage.
        cv::Mat hole_pixels = _hole_pixels(level);
        const bool coarsest = level == static_cast<int>(m_pyramid.size()) - 1;
        if (coarsest)
        {
            target = source.clone();
            target.clear_mask();
        }
        PeelResult peeled;
        if (_peels(level))
        {
            peeled = _peel(level, target);
            target = std::move(peeled.image);
        }
        if (m_distance_metric->patch_size() >= kCoarsePatchSize)
            hole_pixels = around_lines(hole_pixels, peeled.lines, m_level_metric->patch_size());
        // The votes of the completeness field keep the known pixels near their input on the
        // coarse levels; on levels 0 and 1, which keep the known pixels, they change nothing.
        const PatchDistanceMetric *hole_metric = m_hole_metric.get();
        if (level < 2)
            m_source2target.reset();
        else
            m_source2target =
                coarsest
                    ? NearestNeighborField(source, target, m_level_metric, hole_metric, hole_pixels)
                    : NearestNeighborField(source, target, m_level_metric, *m_source2target, hole_metric, hole_pixels);
        m_target2source =
            coarsest ? NearestNeighborField(target, source, m_level_metric, hole_metric, hole_pixels)
                     : NearestNeighborField(target, source, m_level_metric, m_target2source, hole_metric, hole_pixels);
        _link_peeled(peeled.matches);

        trace(verbose, "Initialization done.");

        target = _expectation_maximization(source, target, level, verbose);
    }
    return target;
}

// The levels where a hole's fill is laid out: the coarsest that it reaches and the one below.
bool Inpainting::_peels(int level) const
{
    return level < static_cast<int>(m_peel_labels.size()) && !m_peel_labels[level].empty();
}

// The pixels that search with m_hole_metric on level before the peel, or an empty matrix.
cv::Mat Inpainting::_hole_pixels(int level) const
{
    return level < static_cast<int>(m_hole_pixels.size()) ? m_hole_pixels[level] : cv::Mat();
}

PeelResult Inpainting::_peel(int level, const MaskedImage &target) const
{
    const MaskedImage &source = m_pyramid[level];
    const cv::Mat &holes = m_peel_labels[level];
    const std::vector<int> &reach = m_peel_reach[level];
    const int patch_size =
        m_distance_metric->patch_size() < kCoarsePatchSize ? m_distance_metric->patch_size() : kLinePatchSize;
    if (level == static_cast<int>(m_pyramid.size()) - 1)
        return peel_holes(source, target, holes, reach, nullptr, patch_size, level);
    // The field that the search of this level starts from: the links of the coarser one are off by a pixel here,
    // so the peel deviated from patterns that the coarser levels had locked onto.
    NearestNeighborField guide(
        target, source, m_level_metric, m_target2source, m_hole_metric.get(), _hole_pixels(level));
    guide.minimize(nnf_iterations(level));
    return peel_holes(source, target, holes, reach, &guide, patch_size, level);
}

// The search does not find the few sources of a line that the coarser levels lost.
void Inpainting::_link_peeled(const cv::Mat &matches)
{
    for (int y = 0; y < matches.rows; ++y)
    {
        for (int x = 0; x < matches.cols; ++x)
        {
            if (const auto match = matches.at<cv::Vec2i>(y, x); match[0] >= 0)
                m_target2source.link_if_closer(y, x, match[0], match[1]);
        }
    }
}

// EM-Like algorithm (see "PatchMatch" - page 6).
// Returns a double sized target image (unless level = 0).
MaskedImage Inpainting::_expectation_maximization(
    const MaskedImage &source, MaskedImage target, int level, bool verbose)
{
    const int nr_iters_em = 1 + 2 * level;
    const int nr_iters_nnf = nnf_iterations(level);
    // Not the larger patches of m_hole_metric below kCoarsePatchSize: they would average more votes, which blurred the
    // fills. Its smaller ones from kCoarsePatchSize on, though, which keep the lines.
    const int vote_patch_size = std::min(m_distance_metric->patch_size(), m_level_metric->patch_size());

    MaskedImage new_source;
    MaskedImage new_target;

    for (int iter_em = 0; iter_em < nr_iters_em; ++iter_em)
    {
        if (iter_em != 0)
        {
            if (m_source2target)
                m_source2target->set_target(new_target);
            m_target2source.set_source(new_target);
            target = new_target;
        }

        trace(verbose, "EM Iteration: ", iter_em);

        _update_links(source, iter_em != 0);
        trace(verbose, "  NNF minimization started.");
        if (m_source2target)
            m_source2target->minimize(nr_iters_nnf);
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

        // The weighted mean of all overlapping patches blurs the fill, so in the last
        // iteration of the finest level each pixel takes the value of its most similar
        // patch instead (similar to "Space-Time Completion of Video", which takes the
        // mode of the votes). On coarser levels the best vote makes seams.
        const bool best_only = level == 0 && iter_em == nr_iters_em - 1;

        // Known pixels are kept on the two finest levels only. On coarser levels those
        // next to the holes average only the known part of their downsampling kernel,
        // estimates that reach into the holes; keeping them there made the fills copy
        // smooth regions, e.g. a concrete wall into the foliage above it.
        const int new_level = upscaled ? level - 1 : level;
        const bool keep_known = new_level <= 1;

        // The other votes for these levels average only their low frequencies and add the
        // detail of the best vote: with the plain mean the fill stayed smooth, and smooth
        // fills match smooth sources best. On coarser levels the mean lays out the structure.
        const bool split = keep_known && !best_only;
        auto vote = cv::Mat(new_target.size(), CV_64FC(split ? kSplitChannels : 4), cv::Scalar::all(0));

        // Votes for best patch from NNF Source->Target (completeness) and Target->Source (coherence).
        const VoteSettings settings{upscaled, best_only, keep_known, vote_patch_size};
        if (m_source2target)
        {
            cast_votes(*m_source2target, true, vote, new_source, settings);
            trace(verbose, "  Expectation source to target finished.");
        }
        cast_votes(m_target2source, false, vote, new_source, settings);
        trace(verbose, "  Expectation target to source finished.");

        // Compile votes and update pixel values.
        maximize_votes(new_target, vote, new_source, keep_known);
        trace(verbose, "  Minimization step finished.");

        // The fill expects the texture of its level: the known one, or that around the holes. On level 0
        // that of the sources chosen on level 1: with the texture around the holes, the search there took
        // blurred sources beside detailed surroundings, and parts of the fill stayed flat.
        new_target.set_texture(upscaled && level == 1 ? voted_texture(new_source, vote) : new_source.texture());
    }

    return new_target;
}

// Patches without holes are their own nearest neighbors; the others keep their links,
// measured again on a changed image. Globally masked pixels have no link.
void Inpainting::_update_links(const MaskedImage &source, bool image_changed)
{
    const auto for_each_field = [this](const auto &update) {
        if (m_source2target)
            update(*m_source2target);
        update(m_target2source);
    };
    for_each_field([](const NearestNeighborField &field) { field.prepare(); });
    source.compute_hole_sums();
    const auto size = source.size();
    parallel_for(size.height, [this, &source, &for_each_field, image_changed, &size](int i) {
        for (int j = 0; j < size.width; ++j)
        {
            if (!source.contains_mask(i, j, m_target2source.patch_size(i, j)))
                for_each_field([i, j](NearestNeighborField &field) { field.set_identity(i, j); });
            else if (image_changed && !source.is_globally_masked(i, j))
                for_each_field([i, j](NearestNeighborField &field) { field.update_distance(i, j); });
        }
    });
}
