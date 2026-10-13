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

    bool is_known(const MaskedImage &image, int y, int x)
    {
        return !image.is_masked(y, x) && !image.is_globally_masked(y, x);
    }

    bool is_hole(const MaskedImage &image, int y, int x)
    {
        return image.is_masked(y, x) && !image.is_globally_masked(y, x);
    }

    bool has_hole(const MaskedImage &image)
    {
        for (int y = 0; y < image.size().height; ++y)
        {
            for (int x = 0; x < image.size().width; ++x)
            {
                if (is_hole(image, y, x))
                    return true;
            }
        }
        return false;
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
        const int y0 = static_cast<int>(cy);
        const int x0 = static_cast<int>(cx);
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

    // A patch smaller than a structure cannot tell where it continues: on the coarsest level
    // with a hole, holes wider than the patches were filled from several shifted copies.
    constexpr int kCoarsePatchSize = 5;

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

    // A split vote: the weighted low frequencies and their weight, the detail of the best vote and its weight.
    constexpr int kSplitChannels = 8;

    // The mean color of the known pixels around each pixel, the part of the split votes that is averaged.
    cv::Mat low_pass(const MaskedImage &image)
    {
        const auto size = image.size();
        const auto known = [&image](int y, int x) { return is_known(image, y, x); };
        const auto color = [&image](int y, int x) {
            const auto *pixel = image.get_image(y, x);
            return cv::Vec3d(pixel[0], pixel[1], pixel[2]);
        };
        cv::Mat low(size, CV_8UC3);
        for (int y = 0; y < size.height; ++y)
        {
            for (int x = 0; x < size.width; ++x)
            {
                const cv::Vec3d mean = window_mean(size, y, x, 1, known, color);
                auto *pixel = low.ptr<unsigned char>(y, x);
                for (int c = 0; c < 3; ++c)
                    pixel[c] = cv::saturate_cast<unsigned char>(mean[c]);
            }
        }
        return low;
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
              m_best_only(best_only), m_split(vote.channels() == kSplitChannels),
              m_low(m_split ? low_pass(source) : cv::Mat())
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
                    cast_pair(ys, xs, yt, xt, weight, weight * (1 - 1e-5 * offset2));
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

        void cast_pair(int ys, int xs, int yt, int xt, double weight, double best_weight) const
        {
            if (!m_source2target)
            {
                std::swap(ys, yt);
                std::swap(xs, xt);
            }
            if (!m_upscaled)
            {
                copy(ys, xs, yt, xt, weight, best_weight);
                return;
            }
            for (int uy = 0; uy < 2; ++uy)
                for (int ux = 0; ux < 2; ++ux)
                    copy(2 * ys + uy, 2 * xs + ux, 2 * yt + uy, 2 * xt + ux, weight, best_weight);
        }

        void copy(int ys, int xs, int yt, int xt, double weight, double best_weight) const
        {
            if (m_best_only)
                _best_copy(*m_source, ys, xs, *m_vote, yt, xt, best_weight);
            else if (m_split)
                split_copy(ys, xs, yt, xt, weight, best_weight);
            else
                _weighted_copy(*m_source, ys, xs, *m_vote, yt, xt, weight);
        }

        // Adds the low frequencies of the source pixel to the sums, and keeps the rest if the vote weighs most.
        void split_copy(int ys, int xs, int yt, int xt, double weight, double best_weight) const
        {
            if (!is_known(*m_source, ys, xs))
                return;

            const auto *color = m_source->get_image(ys, xs);
            const auto *low = m_low.ptr<unsigned char>(ys, xs);
            auto *vote = m_vote->ptr<double>(yt, xt);
            for (int c = 0; c < 3; ++c)
                vote[c] += low[c] * weight;
            vote[3] += weight;
            if (best_weight <= vote[7])
                return;
            for (int c = 0; c < 3; ++c)
                vote[4 + c] = color[c] - low[c];
            vote[7] = best_weight;
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
        // cv::Mat::empty() is not inline in OpenCV 5, too slow to call for every vote
        bool m_split;
        cv::Mat m_low; // split votes only
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

    for (int level = 0; level < static_cast<int>(m_pyramid.size()); ++level)
    {
        if (has_hole(m_pyramid[level]))
            m_coarsest_hole_level = level;
        const int side = std::min(m_pyramid[level].size().height, m_pyramid[level].size().width);
        const int fitting = std::max(kMinFittingPatchSize, (side / kLevelSidesPerPatch - 1) / 2);
        m_fitting_metrics.push_back(
            fitting < m_distance_metric->patch_size() ? m_distance_metric->with_patch_size(fitting) : nullptr);
    }
    if (m_distance_metric->patch_size() < kCoarsePatchSize)
        m_coarse_metric = m_distance_metric->with_patch_size(kCoarsePatchSize);
}

cv::Mat Inpainting::run(bool verbose, unsigned int random_seed)
{
    NearestNeighborField::seed_random(random_seed);

    int level = static_cast<int>(m_pyramid.size()) - 1;
    MaskedImage target;
    // Below level 2 no level is left to compare the fills on.
    if (m_coarse_metric && m_coarsest_hole_level >= 2)
    {
        target = _run_levels(level, m_coarsest_hole_level + 1, target, m_distance_metric, verbose);
        target = _fill_coarse_levels(target, verbose);
        level = m_coarsest_hole_level - 2;
    }
    target = _run_levels(level, 0, target, m_distance_metric, verbose);

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

// Runs the levels from `from` down to `to`; target is empty on the coarsest level.
MaskedImage Inpainting::_run_levels(
    int from, int to, MaskedImage target, const PatchDistanceMetric *metric, bool verbose)
{
    for (int level = from; level >= to; --level)
    {
        trace(verbose, "Inpainting level: ", level);

        const MaskedImage &source = m_pyramid[level];
        m_level_metric = m_fitting_metrics[level] ? m_fitting_metrics[level].get() : metric;
        const bool coarsest = level == static_cast<int>(m_pyramid.size()) - 1;
        if (coarsest)
        {
            target = source.clone();
            target.clear_mask();
        }
        // The votes of the completeness field keep the known pixels near their input on the
        // coarse levels; on levels 0 and 1, which keep the known pixels, they change nothing.
        if (level < 2)
            m_source2target.reset();
        else
            m_source2target = coarsest ? NearestNeighborField(source, target, m_level_metric)
                                       : NearestNeighborField(source, target, m_level_metric, *m_source2target);
        m_target2source = coarsest ? NearestNeighborField(target, source, m_level_metric)
                                   : NearestNeighborField(target, source, m_level_metric, m_target2source);

        trace(verbose, "Initialization done.");

        target = _expectation_maximization(source, target, level, verbose);
    }
    return target;
}

// Fills the coarsest level with a hole and the next one with patch_size and with the
// larger patches, and keeps the fill with the lower energy on the following level. The
// larger patches lock onto repeating structures but pick flat sources elsewhere, which
// made fills of foliage gray.
MaskedImage Inpainting::_fill_coarse_levels(const MaskedImage &target, bool verbose)
{
    const int level = m_coarsest_hole_level;
    const auto source2target = m_source2target;
    const auto target2source = m_target2source;

    const MaskedImage plain = _run_levels(level, level - 1, target, m_distance_metric, verbose);
    const double plain_energy = _hole_energy(level - 2, plain);
    const auto plain_source2target = m_source2target;
    const auto plain_target2source = m_target2source;

    m_source2target = source2target;
    m_target2source = target2source;
    MaskedImage coarse = _run_levels(level, level - 1, target, m_coarse_metric.get(), verbose);
    const double coarse_energy = _hole_energy(level - 2, coarse);
    trace(verbose, "Energy with patch_size: ", plain_energy, ", with larger patches: ", coarse_energy);
    m_level_metric = m_distance_metric;
    if (coarse_energy < plain_energy)
        return coarse;
    m_source2target = plain_source2target;
    m_target2source = plain_target2source;
    return plain;
}

// The mean distance of the patches with holes after a search with patch_size on the level,
// started from the field of the level above, like the first iteration of that level.
double Inpainting::_hole_energy(int level, const MaskedImage &target) const
{
    const MaskedImage &source = m_pyramid[level];
    NearestNeighborField field(target, source, m_distance_metric, m_target2source);
    field.minimize(nnf_iterations(level));

    const int patch_size = m_distance_metric->patch_size();
    const auto size = source.size();
    double sum = 0;
    int count = 0;
    for (int i = 0; i < size.height; ++i)
    {
        for (int j = 0; j < size.width; ++j)
        {
            if (source.contains_mask(i, j, patch_size) && !source.is_globally_masked(i, j))
            {
                sum += field.at(i, j, 2);
                ++count;
            }
        }
    }
    return count > 0 ? sum / count : 0;
}

// EM-Like algorithm (see "PatchMatch" - page 6).
// Returns a double sized target image (unless level = 0).
MaskedImage Inpainting::_expectation_maximization(
    const MaskedImage &source, MaskedImage target, int level, bool verbose)
{
    const int nr_iters_em = 1 + 2 * level;
    const int nr_iters_nnf = nnf_iterations(level);

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
        // The fill expects the texture of its level: the known one, or that around the holes.
        new_target.set_texture(new_source.texture());

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
        if (m_source2target)
        {
            _expectation_step(*m_source2target, true, vote, new_source, upscaled, best_only, keep_known);
            trace(verbose, "  Expectation source to target finished.");
        }
        _expectation_step(m_target2source, false, vote, new_source, upscaled, best_only, keep_known);
        trace(verbose, "  Expectation target to source finished.");

        // Compile votes and update pixel values.
        _maximization_step(new_target, vote, new_source, keep_known);
        trace(verbose, "  Minimization step finished.");
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
    const int patch_size = m_level_metric->patch_size();
    const auto size = source.size();
    for (int i = 0; i < size.height; ++i)
    {
        for (int j = 0; j < size.width; ++j)
        {
            if (!source.contains_mask(i, j, patch_size))
                for_each_field([i, j](NearestNeighborField &field) { field.set_identity(i, j); });
            else if (image_changed && !source.is_globally_masked(i, j))
                for_each_field([i, j](NearestNeighborField &field) { field.update_distance(i, j); });
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
    // The larger patches of m_coarse_metric would average more votes, which blurred the fills.
    const int patch_size = std::min(m_distance_metric->patch_size(), m_level_metric->patch_size());
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
    constexpr std::array<double, 3> kNoDetail = {0, 0, 0};
    const bool split = vote.channels() == kSplitChannels;
    auto target_size = target.size();
    for (int i = 0; i < target_size.height; ++i)
    {
        for (int j = 0; j < target_size.width; ++j)
        {
            const double *source_ptr = vote.ptr<double>(i, j);
            const double *detail = split ? source_ptr + 4 : kNoDetail.data();
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
                const auto r = cv::saturate_cast<unsigned char>(source_ptr[0] / source_ptr[3] + detail[0]);
                const auto g = cv::saturate_cast<unsigned char>(source_ptr[1] / source_ptr[3] + detail[1]);
                const auto b = cv::saturate_cast<unsigned char>(source_ptr[2] / source_ptr[3] + detail[2]);
                target_ptr[0] = r;
                target_ptr[1] = g;
                target_ptr[2] = b;
            }
        }
    }
}
