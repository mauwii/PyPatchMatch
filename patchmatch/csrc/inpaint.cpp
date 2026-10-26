#include "inpaint.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>

#include "parallel.h"

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

    // Gives the hole pixels 8-connected to (y, x), the neighborhood of the peel's front, the label.
    void label_hole(const MaskedImage &image, cv::Mat &labels, int y, int x, int label)
    {
        labels.at<int>(y, x) = label;
        std::vector<cv::Point> stack{{x, y}};
        while (!stack.empty())
        {
            const cv::Point p = stack.back();
            stack.pop_back();
            for_window(image.size(), p.y, p.x, 1, [&image, &labels, &stack, label](int yy, int xx) {
                if (is_hole(image, yy, xx) && labels.at<int>(yy, xx) == 0)
                {
                    labels.at<int>(yy, xx) = label;
                    stack.emplace_back(xx, yy);
                }
            });
        }
    }

    // Per pixel the label of its hole from 1 on, 0 outside the holes.
    cv::Mat label_holes(const MaskedImage &image, int &count)
    {
        cv::Mat labels(image.size(), CV_32S, cv::Scalar(0));
        count = 0;
        for (int y = 0; y < labels.rows; ++y)
        {
            for (int x = 0; x < labels.cols; ++x)
            {
                if (is_hole(image, y, x) && labels.at<int>(y, x) == 0)
                    label_hole(image, labels, y, x, ++count);
            }
        }
        return labels;
    }

    // Raises the reach of the labeled holes in the window to top.
    void raise_reach(const cv::Mat &labels, const cv::Rect &window, int top, std::vector<int> &reach)
    {
        for (int y = window.y; y < window.y + window.height; ++y)
        {
            for (int x = window.x; x < window.x + window.width; ++x)
            {
                auto &hole = reach[static_cast<std::size_t>(labels.at<int>(y, x))];
                hole = std::max(hole, top);
            }
        }
    }

    // Raises the reach of each labeled hole to that of the holes of the coarser level that it averages into.
    void reach_coarser(const cv::Mat &coarser_reach, const cv::Mat &labels, std::vector<int> &reach)
    {
        const cv::Rect image({}, labels.size());
        for (int y = 0; y < coarser_reach.rows; ++y)
        {
            for (int x = 0; x < coarser_reach.cols; ++x)
            {
                if (const int top = coarser_reach.at<int>(y, x); top >= 0)
                    raise_reach(labels, downsample_window(y, x) & image, top, reach);
            }
        }
    }

    // The labels of the holes that reach at most one level coarser than level, 0 elsewhere, or an empty matrix; and
    // per pixel the reach of its hole, -1 outside the holes.
    cv::Mat peeled_holes(const cv::Mat &labels, const std::vector<int> &reach, int level, cv::Mat &pixel_reach)
    {
        pixel_reach = cv::Mat(labels.size(), CV_32S, cv::Scalar(-1));
        cv::Mat peeled(labels.size(), CV_32S, cv::Scalar(0));
        bool peels = false;
        for (int y = 0; y < labels.rows; ++y)
        {
            for (int x = 0; x < labels.cols; ++x)
            {
                const int label = labels.at<int>(y, x);
                if (label == 0)
                    continue;
                const int top = reach[static_cast<std::size_t>(label)];
                pixel_reach.at<int>(y, x) = top;
                if (top <= level + 1)
                {
                    peeled.at<int>(y, x) = label;
                    peels = true;
                }
            }
        }
        return peels ? peeled : cv::Mat();
    }

    // Per level the labels of the holes that the peel fills there, 0 elsewhere, or an empty matrix: each hole on the
    // coarsest level that it reaches and on the one below. On the levels of the largest hole only, smaller holes
    // elsewhere broke their lines off.
    std::vector<cv::Mat> peel_labels(const std::vector<MaskedImage> &pyramid)
    {
        std::vector<cv::Mat> peel(pyramid.size());
        cv::Mat coarser_reach; // per pixel the coarsest level that its hole reaches, -1 outside the holes
        for (int level = static_cast<int>(pyramid.size()) - 1; level >= 0; --level)
        {
            int count = 0;
            const cv::Mat labels = label_holes(pyramid[level], count);
            // reach[0] belongs to the globally masked pixels in the windows
            std::vector reach(static_cast<std::size_t>(count) + 1, level);
            reach_coarser(coarser_reach, labels, reach);
            peel[static_cast<std::size_t>(level)] = peeled_holes(labels, reach, level, coarser_reach);
        }
        return peel;
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

    // A split vote: the weighted low frequencies and their weight, the detail of the best vote and its weight,
    // and the weighted texture of the source pixels.
    constexpr int kSplitChannels = 9;

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
        parallel_for(size.height, [&low, &size, &known, &color](int y) {
            for (int x = 0; x < size.width; ++x)
            {
                const cv::Vec3d mean = window_mean(size, y, x, 1, known, color);
                auto *pixel = low.ptr<unsigned char>(y, x);
                for (int c = 0; c < 3; ++c)
                    pixel[c] = cv::saturate_cast<unsigned char>(mean[c]);
            }
        });
        return low;
    }

    // Replaces the vote of the pixel if the new one weighs more. It is stored like the
    // sums of _weighted_copy, so the maximization step divides it by its weight.
    inline void _best_copy(
        const unsigned char *color, const cv::Vec3d &shift, cv::Mat &target, int yt, int xt, double weight)
    {
        auto target_ptr = target.ptr<double>(yt, xt);
        if (weight <= target_ptr[3])
            return;

        for (int c = 0; c < 3; ++c)
            target_ptr[c] = (static_cast<double>(color[c]) + shift[c]) * weight;
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
        // in the patch around its nearest neighbor; only the votes for rows of the vote image
        // are cast, which start at a multiple of the scale.
        void cast_patch(int i, int j, int patch_size, double weight, const cv::Range &rows) const
        {
            const int yp = m_nnf->at(i, j, 0);
            const int xp = m_nnf->at(i, j, 1);
            // The voted image is the source of the coherence field and the target of the completeness field.
            const int y = m_source2target ? yp : i;
            const int scale = m_upscaled ? 2 : 1;
            const int first = std::max(-patch_size, rows.start / scale - y);
            const int last = std::min(patch_size, (rows.end - 1) / scale - y);
            if (first > last)
                return;
            if (m_holes && !votes_for_hole(y, m_source2target ? xp : j, patch_size))
                return;
            const cv::Vec3d shift = m_holes ? color_shift(i, j, yp, xp, patch_size) : cv::Vec3d();
            for (int di = first; di <= last; ++di)
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
                    cast_pair(ys, xs, yt, xt, weight, weight * (1 - 1e-5 * offset2), shift);
                }
            }
        }

    private:
        // The mean color difference of the voted image to the voting source over the pixels known in both patches.
        // The votes of the finest levels add it: their sources lie elsewhere in the image, and with their colors
        // the fill of a vignetted photo came out lighter than the colors the coarser levels had laid out.
        cv::Vec3d color_shift(int i, int j, int yp, int xp, int patch_size) const
        {
            const MaskedImage &source = m_nnf->source();
            const MaskedImage &target = m_nnf->target();
            cv::Vec3d sum;
            int count = 0;
            for (int di = -patch_size; di <= patch_size; ++di)
            {
                for (int dj = -patch_size; dj <= patch_size; ++dj)
                {
                    const int ys = i + di;
                    const int xs = j + dj;
                    const int yt = yp + di;
                    const int xt = xp + dj;
                    if (ys < 0 || ys >= m_source_size.height || xs < 0 || xs >= m_source_size.width || yt < 0 ||
                        yt >= m_target_size.height || xt < 0 || xt >= m_target_size.width)
                        continue;
                    if (!is_known(source, ys, xs) || !is_known(target, yt, xt))
                        continue;
                    const auto *s = source.get_image(ys, xs);
                    const auto *t = target.get_image(yt, xt);
                    for (int c = 0; c < 3; ++c)
                        sum[c] += s[c] - t[c];
                    ++count;
                }
            }
            if (count == 0)
                return {};
            // The voted image is the source of the coherence field and the target of the completeness field.
            return sum * ((m_source2target ? -1.0 : 1.0) / count);
        }

        // Whether the patch around (y, x) of the voted image, scaled like the votes, has a hole.
        bool votes_for_hole(int y, int x, int patch_size) const
        {
            const std::int64_t scale = m_upscaled ? 2 : 1;
            return m_holes->any(
                scale * (std::int64_t{y} - patch_size), scale * (std::int64_t{x} - patch_size),
                scale * (std::int64_t{y} + patch_size + 1) - 1, scale * (std::int64_t{x} + patch_size + 1) - 1);
        }

        void cast_pair(int ys, int xs, int yt, int xt, double weight, double best_weight, const cv::Vec3d &shift) const
        {
            if (!m_source2target)
            {
                std::swap(ys, yt);
                std::swap(xs, xt);
            }
            if (!m_upscaled)
            {
                copy(ys, xs, yt, xt, weight, best_weight, shift);
                return;
            }
            for (int uy = 0; uy < 2; ++uy)
                for (int ux = 0; ux < 2; ++ux)
                    copy(2 * ys + uy, 2 * xs + ux, 2 * yt + uy, 2 * xt + ux, weight, best_weight, shift);
        }

        void copy(int ys, int xs, int yt, int xt, double weight, double best_weight, const cv::Vec3d &shift) const
        {
            if (m_best_only)
            {
                if (!m_source->is_masked(ys, xs))
                    _best_copy(m_source->get_image(ys, xs), shift, *m_vote, yt, xt, best_weight);
            }
            else if (m_split)
                split_copy(ys, xs, yt, xt, weight, best_weight, shift);
            else
                _weighted_copy(*m_source, ys, xs, *m_vote, yt, xt, weight);
        }

        // Adds the shifted low frequencies and the texture of the source pixel to the sums, and keeps the rest if
        // the vote weighs most.
        void split_copy(int ys, int xs, int yt, int xt, double weight, double best_weight, const cv::Vec3d &shift) const
        {
            if (!is_known(*m_source, ys, xs))
                return;

            const auto *color = m_source->get_image(ys, xs);
            const auto *low = m_low.ptr<unsigned char>(ys, xs);
            auto *vote = m_vote->ptr<double>(yt, xt);
            for (int c = 0; c < 3; ++c)
                vote[c] += (low[c] + shift[c]) * weight;
            vote[3] += weight;
            vote[8] += m_source->texture().at<unsigned char>(ys, xs) * weight;
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

    // The known texture, and in the holes twice the mean texture of the pixels that the split votes came from:
    // the distances charge a deficit below half of it, so a fill expects the full texture of its sources.
    cv::Mat voted_texture(const MaskedImage &source, const cv::Mat &vote)
    {
        cv::Mat texture = source.texture().clone();
        for (int y = 0; y < texture.rows; ++y)
        {
            for (int x = 0; x < texture.cols; ++x)
            {
                const auto *sums = vote.ptr<double>(y, x);
                if (is_hole(source, y, x) && sums[3] > 0)
                    texture.at<unsigned char>(y, x) = cv::saturate_cast<unsigned char>(2 * sums[8] / sums[3]);
            }
        }
        return texture;
    }

    std::int64_t squared_difference(const cv::Vec3b &a, const cv::Vec3b &b)
    {
        std::int64_t sum = 0;
        for (int c = 0; c < 3; ++c)
            sum += (a[c] - b[c]) * (a[c] - b[c]);
        return sum;
    }

    // Fills the holes of a level from their border inwards, pixel by pixel in the order of Criminisi et al. 2004,
    // each from the center of a source patch that fits the known and filled pixels around it. A line that enters
    // a hole fades on the coarser levels, and the search only refines the fill it starts from.
    class Peel
    {
    public:
        // holes: the labels of the holes to fill (CV_32S), 0 elsewhere; guide: the field of this level that the search
        // would start from, nullptr on the coarsest level.
        Peel(
            const MaskedImage &source, const MaskedImage &target, const cv::Mat &holes,
            const NearestNeighborField *guide, int patch_size)
            : m_source(source), m_target(target), m_holes(holes), m_guide(guide), m_patch_size(patch_size),
              m_size(source.size()), m_colors(source.image().clone()), m_filled(m_size, CV_8U, cv::Scalar(0)),
              m_front_mask(m_size, CV_8U, cv::Scalar(0)), m_confidence(m_size, CV_64F, cv::Scalar(0)),
              m_matches(m_size, CV_32SC2, cv::Scalar::all(-1))
        {
            double holes_count = 0;
            cv::minMaxLoc(holes, nullptr, &holes_count);
            m_fits.resize(static_cast<std::size_t>(holes_count) + 1);
            for (int y = 0; y < m_size.height; ++y)
            {
                for (int x = 0; x < m_size.width; ++x)
                {
                    if (is_hole(source, y, x))
                        m_colors.at<cv::Vec3b>(y, x) = target.image().at<cv::Vec3b>(y, x);
                    if (is_known(source, y, x))
                    {
                        m_filled.at<unsigned char>(y, x) = 1;
                        m_confidence.at<double>(y, x) = 1;
                    }
                }
            }
            for (int y = 0; y < m_size.height; ++y)
                for (int x = 0; x < m_size.width; ++x)
                    _enter_front(y, x);
        }

        // The target with the peeled holes, and per peeled pixel the center of its source patch, -1 elsewhere.
        MaskedImage run(cv::Mat &matches)
        {
            while (!m_front.empty())
            {
                // the first of the highest
                const auto next = static_cast<std::size_t>(
                    std::max_element(m_priorities.begin(), m_priorities.end()) - m_priorities.begin());
                const cv::Point p = m_front[next];
                m_front[next] = m_front.back();
                m_front.pop_back();
                m_priorities[next] = m_priorities.back();
                m_priorities.pop_back();
                _fill(p, _choose(p));
                _update_priorities(p);
            }

            MaskedImage peeled = m_target.clone();
            for (int y = 0; y < m_size.height; ++y)
                for (int x = 0; x < m_size.width; ++x)
                    if (m_matches.at<cv::Vec2i>(y, x)[0] >= 0)
                        std::copy_n(m_colors.ptr<unsigned char>(y, x), 3, peeled.get_mutable_image(y, x));
            matches = m_matches;
            return peeled;
        }

    private:
        struct Candidate
        {
            int y;
            int x;
            std::int64_t fit;   // to the known and filled pixels around the pixel
            std::int64_t guide; // to the coarser fill of the other hole pixels around it
        };

        // Near ties of the fit go to the patch closest to the coarser fill, which lays out repeating structures.
        static constexpr double kNearTie = 0.3;
        // The guide's own link wins unless the known pixels clearly contradict it, as at the end of a line.
        static constexpr double kGuideTolerance = 2;
        // More where the coarser levels locked onto a structure, which the peel broke into pieces of the wrong phase:
        // the guide's links around the pixel share one offset, and its source is about as varied as the best fit's,
        // unlike a coherent background across which a line continues.
        static constexpr double kLockedGuideTolerance = 4;
        static constexpr double kLockedVariance = 0.75;
        // A fit this much worse than the mean of the pixels of its hole before calls for a search of all sources: the
        // random search does not find the few sources of a line. The first pixels of the level search all, and the
        // first of each hole: shared by all holes, the warm-up left the lines of the later ones broken. The mean
        // forgets slowly.
        static constexpr double kPoorFit = 3;
        static constexpr int kWarmUp = 8;
        static constexpr int kHoleWarmUp = 2;
        static constexpr double kFitMemory = 0.9;
        // Without an isophote, the confidence orders the pixels.
        static constexpr double kFlatData = 0.001;

        bool _filled(int y, int x) const
        {
            return y >= 0 && y < m_size.height && x >= 0 && x < m_size.width && m_filled.at<unsigned char>(y, x);
        }

        void _enter_front(int y, int x)
        {
            if (m_holes.at<int>(y, x) == 0 || m_filled.at<unsigned char>(y, x) || m_front_mask.at<unsigned char>(y, x))
                return;
            bool next_to_filled = false;
            for_window(m_size, y, x, 1, [this, &next_to_filled](int yy, int xx) {
                next_to_filled = next_to_filled || m_filled.at<unsigned char>(yy, xx);
            });
            if (!next_to_filled)
                return;
            m_front_mask.at<unsigned char>(y, x) = 1;
            m_front.emplace_back(x, y);
            m_priorities.push_back(_priority({x, y}));
        }

        // Filling p changes the priorities of the front only within its confidence window and the isophotes around it.
        void _update_priorities(cv::Point p)
        {
            const int reach = std::max(m_patch_size, 2);
            for (std::size_t i = 0; i < m_front.size(); ++i)
            {
                if (std::abs(m_front[i].y - p.y) <= reach && std::abs(m_front[i].x - p.x) <= reach)
                    m_priorities[i] = _priority(m_front[i]);
            }
        }

        double _luminance(int y, int x) const
        {
            const auto &color = m_colors.at<cv::Vec3b>(y, x);
            return (color[0] + color[1] + color[2]) / 3.0;
        }

        // The confidence of the window times the strongest isophote that hits the front at p.
        double _priority(cv::Point p) const
        {
            double confidence = 0;
            for_window(m_size, p.y, p.x, m_patch_size, [this, &confidence](int y, int x) {
                confidence += m_confidence.at<double>(y, x);
            });
            const double side = 2.0 * m_patch_size + 1;
            confidence /= side * side;

            double ny = 0;
            double nx = 0;
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    if (!_filled(p.y + dy, p.x + dx))
                        continue;
                    ny += dy * (dx == 0 ? 2 : 1);
                    nx += dx * (dy == 0 ? 2 : 1);
                }
            }
            if (const double norm = std::hypot(ny, nx); norm > 0)
            {
                ny /= norm;
                nx /= norm;
            }

            double isophote = 0;
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    const int y = p.y + dy;
                    const int x = p.x + dx;
                    if (!_filled(y, x) || !_filled(y - 1, x) || !_filled(y + 1, x) || !_filled(y, x - 1) ||
                        !_filled(y, x + 1))
                        continue;
                    const double gy = (_luminance(y + 1, x) - _luminance(y - 1, x)) / 2;
                    const double gx = (_luminance(y, x + 1) - _luminance(y, x - 1)) / 2;
                    isophote = std::max(isophote, std::fabs(gx * ny - gy * nx));
                }
            }
            return confidence * (isophote / 255 + kFlatData);
        }

        // Adds the source patch around (qy, qx) as a candidate for p unless it overlaps a hole or the global mask,
        // or its fit is beyond the near ties of the best one so far.
        void _evaluate(cv::Point p, int qy, int qx)
        {
            if (const auto candidate = _fit(p, qy, qx, m_best))
                _add(*candidate);
        }

        void _add(const Candidate &candidate)
        {
            if (static_cast<double>(candidate.fit) > _near_ties(m_best))
                return;
            m_best = std::min(m_best, candidate.fit);
            m_candidates.push_back(candidate);
        }

        static double _near_ties(std::int64_t best)
        {
            return best == kNone ? std::numeric_limits<double>::max() : (1 + kNearTie) * static_cast<double>(best) + 1;
        }

        // The source patch around (qy, qx) as a candidate for p, unless it overlaps a hole or the global mask, or its
        // fit is beyond the near ties of best.
        std::optional<Candidate> _fit(cv::Point p, int qy, int qx, std::int64_t best) const
        {
            if (qy < m_patch_size || qy >= m_size.height - m_patch_size || qx < m_patch_size ||
                qx >= m_size.width - m_patch_size || m_source.contains_mask(qy, qx, m_patch_size))
                return std::nullopt;
            const double bound = _near_ties(best);
            std::int64_t fit = 0;
            std::int64_t guide = 0;
            for (int dy = -m_patch_size; dy <= m_patch_size; ++dy)
            {
                if (static_cast<double>(fit) > bound)
                    return std::nullopt;
                for (int dx = -m_patch_size; dx <= m_patch_size; ++dx)
                {
                    if (m_source.is_globally_masked(qy + dy, qx + dx))
                        return std::nullopt;
                    const Role role = _role(p.y + dy, p.x + dx);
                    if (role == Role::kIgnored)
                        continue;
                    const std::int64_t difference = squared_difference(
                        m_colors.at<cv::Vec3b>(p.y + dy, p.x + dx), m_source.image().at<cv::Vec3b>(qy + dy, qx + dx));
                    (role == Role::kFit ? fit : guide) += difference;
                }
            }
            if (static_cast<double>(fit) > bound)
                return std::nullopt;
            return Candidate{qy, qx, fit, guide};
        }

        enum class Role
        {
            kIgnored,
            kFit,   // a known or filled pixel
            kGuide, // a hole pixel with the coarser fill
        };

        Role _role(int y, int x) const
        {
            if (y < 0 || y >= m_size.height || x < 0 || x >= m_size.width)
                return Role::kIgnored;
            if (m_filled.at<unsigned char>(y, x))
                return Role::kFit;
            return m_guide && is_hole(m_source, y, x) ? Role::kGuide : Role::kIgnored;
        }

        // The center of the source patch for p, or (-1, -1) if no patch fits.
        cv::Vec2i _choose(cv::Point p)
        {
            m_candidates.clear();
            m_best = kNone;
            // first, so that the bound of the near ties never drops it
            if (m_guide)
                _evaluate(p, m_guide->at(p.y, p.x, 0), m_guide->at(p.y, p.x, 1));
            const bool guided = !m_candidates.empty();
            _evaluate_neighbors(p);
            _search_randomly(p);

            int filled = 0;
            for_window(m_size, p.y, p.x, m_patch_size, [this, &filled](int y, int x) {
                filled += m_filled.at<unsigned char>(y, x);
            });
            filled = std::max(filled, 1);
            Fits &fits = m_fits[static_cast<std::size_t>(m_holes.at<int>(p.y, p.x))];
            if (m_candidates.empty() || m_fitted < kWarmUp || fits.count < kHoleWarmUp ||
                static_cast<double>(m_best) / filled > kPoorFit * fits.mean)
                _search_everywhere(p);
            if (m_candidates.empty())
                return {-1, -1};

            const Candidate &chosen = _pick(p, guided);
            const double fit = static_cast<double>(chosen.fit) / filled;
            fits.mean = fits.count == 0 ? fit : kFitMemory * fits.mean + (1 - kFitMemory) * fit;
            ++fits.count;
            ++m_fitted;
            return {chosen.y, chosen.x};
        }

        // The matches of the peeled neighbors, shifted to p, which continue the regions they copy.
        void _evaluate_neighbors(cv::Point p)
        {
            for_window(m_size, p.y, p.x, 1, [this, p](int y, int x) {
                const auto match = m_matches.at<cv::Vec2i>(y, x);
                if (match[0] >= 0)
                    _evaluate(p, match[0] + p.y - y, match[1] + p.x - x);
            });
        }

        // Twice around the best fit so far, in windows of decreasing size.
        void _search_randomly(cv::Point p)
        {
            for (int round = 0; round < 2; ++round)
            {
                const Candidate *best = _best_fit();
                const int cy = best ? best->y : m_size.height / 2;
                const int cx = best ? best->x : m_size.width / 2;
                for (int radius = std::max(m_size.height, m_size.width) / 2; radius >= 1; radius /= 2)
                {
                    const int top = std::max(cy - radius, 0);
                    const int left = std::max(cx - radius, 0);
                    const int bottom = std::min(cy + radius, m_size.height - 1);
                    const int right = std::min(cx + radius, m_size.width - 1);
                    // Two statements: the order of evaluation of arguments differs between compilers.
                    const int qy = top + NearestNeighborField::random_index(bottom - top + 1);
                    const int qx = left + NearestNeighborField::random_index(right - left + 1);
                    _evaluate(p, qy, qx);
                }
            }
        }

        // A row of sources per task. Each keeps the near ties of its own best fit, among them all that the scan in
        // order keeps, so that adding them in that order gives the same candidates and best fit.
        void _search_everywhere(cv::Point p)
        {
            const int rows = std::max(m_size.height - 2 * m_patch_size, 0);
            std::vector<std::vector<Candidate>> found(static_cast<std::size_t>(rows));
            m_source.compute_hole_sums(); // before the rows read it
            parallel_for(rows, [this, p, &found](int row) {
                std::int64_t best = m_best;
                const int qy = m_patch_size + row;
                for (int qx = m_patch_size; qx < m_size.width - m_patch_size; ++qx)
                {
                    if (const auto candidate = _fit(p, qy, qx, best))
                    {
                        best = std::min(best, candidate->fit);
                        found[static_cast<std::size_t>(row)].push_back(*candidate);
                    }
                }
            });
            for (const auto &row : found)
                for (const auto &candidate : row)
                    _add(candidate);
        }

        // The guide's link, otherwise among the near ties of the best fit the patch closest to the coarser fill.
        const Candidate &_pick(cv::Point p, bool guided) const
        {
            const auto best = static_cast<double>(m_best);
            if (guided)
            {
                const auto fit = static_cast<double>(m_candidates[0].fit);
                if (fit <= (1 + kGuideTolerance) * best + 1 ||
                    (fit <= (1 + kLockedGuideTolerance) * best + 1 && _locked(p) &&
                     _variance(m_candidates[0]) >= kLockedVariance * _variance(*_best_fit())))
                    return m_candidates[0];
            }
            const double near = (1 + kNearTie) * best + 1;
            // the first best fit, which is a near tie itself
            auto chosen =
                std::min_element(m_candidates.begin(), m_candidates.end(), [](const Candidate &a, const Candidate &b) {
                    return a.fit < b.fit;
                });
            for (auto candidate = m_candidates.begin(); candidate != m_candidates.end(); ++candidate)
            {
                if (static_cast<double>(candidate->fit) <= near &&
                    (candidate->guide < chosen->guide ||
                     (candidate->guide == chosen->guide && candidate->fit < chosen->fit)))
                    chosen = candidate;
            }
            return *chosen;
        }

        // Whether the guide's links of p and its eight neighbors share one offset.
        bool _locked(cv::Point p) const
        {
            const int dy = m_guide->at(p.y, p.x, 0) - p.y;
            const int dx = m_guide->at(p.y, p.x, 1) - p.x;
            int shared = 0;
            for_window(m_size, p.y, p.x, 1, [this, dy, dx, &shared](int y, int x) {
                if (m_guide->at(y, x, 0) - y == dy && m_guide->at(y, x, 1) - x == dx)
                    ++shared;
            });
            return shared == 9;
        }

        // The variance of the colors of a candidate's source patch.
        double _variance(const Candidate &candidate) const
        {
            cv::Vec3d sum;
            double squares = 0;
            for_window(m_size, candidate.y, candidate.x, m_patch_size, [this, &sum, &squares](int y, int x) {
                const auto &color = m_source.image().at<cv::Vec3b>(y, x);
                for (int c = 0; c < 3; ++c)
                {
                    sum[c] += color[c];
                    squares += static_cast<double>(color[c]) * color[c];
                }
            });
            const double count = (2.0 * m_patch_size + 1) * (2.0 * m_patch_size + 1);
            return (squares - sum.dot(sum) / count) / count;
        }

        const Candidate *_best_fit() const
        {
            const Candidate *best = nullptr;
            for (const auto &candidate : m_candidates)
                if (!best || candidate.fit < best->fit)
                    best = &candidate;
            return best;
        }

        // Copies the source pixel, shifted by the mean color difference of the filled pixels to their sources:
        // raw colors from elsewhere lost the tone that the coarser levels laid out, as in a vignetted photo.
        void _fill(cv::Point p, cv::Vec2i match)
        {
            if (match[0] >= 0)
            {
                const ColorDifferences differences = _differences(p, match);
                const auto &color = m_source.image().at<cv::Vec3b>(match[0], match[1]);
                auto &value = m_colors.at<cv::Vec3b>(p.y, p.x);
                for (int c = 0; c < 3; ++c)
                    value[c] = cv::saturate_cast<unsigned char>(
                        color[c] + differences.sum[c] / std::max(differences.count, 1));
                m_matches.at<cv::Vec2i>(p.y, p.x) = match;
            }

            double confidence = 0;
            for_window(m_size, p.y, p.x, m_patch_size, [this, &confidence](int y, int x) {
                confidence += m_confidence.at<double>(y, x);
            });
            const double side = 2.0 * m_patch_size + 1;
            m_confidence.at<double>(p.y, p.x) = confidence / (side * side);
            m_filled.at<unsigned char>(p.y, p.x) = 1;
            for_window(m_size, p.y, p.x, 1, [this](int y, int x) { _enter_front(y, x); });
        }

        struct ColorDifferences
        {
            std::array<double, 3> sum{}; // not cv::Vec3d, whose move is not noexcept
            int count = 0;
        };

        // The color differences of the filled pixels around p to those around the match.
        ColorDifferences _differences(cv::Point p, cv::Vec2i match) const
        {
            ColorDifferences differences;
            for (int dy = -m_patch_size; dy <= m_patch_size; ++dy)
            {
                for (int dx = -m_patch_size; dx <= m_patch_size; ++dx)
                {
                    if (!_filled(p.y + dy, p.x + dx))
                        continue;
                    const auto &a = m_colors.at<cv::Vec3b>(p.y + dy, p.x + dx);
                    const auto &b = m_source.image().at<cv::Vec3b>(match[0] + dy, match[1] + dx);
                    for (int c = 0; c < 3; ++c)
                        differences.sum[c] += a[c] - b[c];
                    ++differences.count;
                }
            }
            return differences;
        }

        static constexpr std::int64_t kNone = std::numeric_limits<std::int64_t>::max();

        const MaskedImage &m_source;
        const MaskedImage &m_target;
        const cv::Mat &m_holes;
        const NearestNeighborField *m_guide;
        int m_patch_size;
        cv::Size m_size;
        cv::Mat m_colors;
        cv::Mat m_filled;
        cv::Mat m_front_mask;
        cv::Mat m_confidence;
        cv::Mat m_matches;
        std::vector<cv::Point> m_front;
        std::vector<double> m_priorities; // of the front, in its order
        std::vector<Candidate> m_candidates;
        std::int64_t m_best = kNone;
        // The fits chosen so far per hole, by its label.
        struct Fits
        {
            double mean = 0;
            int count = 0;
        };
        std::vector<Fits> m_fits;
        int m_fitted = 0;
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
    // No peel either from kCoarsePatchSize on: the searches of the finer levels wiped the peeled lines out again.
    if (m_distance_metric->patch_size() < kCoarsePatchSize)
    {
        m_coarse_metric = m_distance_metric->with_patch_size(kCoarsePatchSize);
        m_peel_labels = peel_labels(m_pyramid);
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
        // The larger patches on the coarsest level with a hole only: below, the holes are wider than them, and they
        // took flat sources in foliage.
        const PatchDistanceMetric *metric =
            m_coarse_metric && level == m_coarsest_hole_level ? m_coarse_metric.get() : m_distance_metric;
        m_level_metric = m_fitting_metrics[level] ? m_fitting_metrics[level].get() : metric;
        const bool coarsest = level == static_cast<int>(m_pyramid.size()) - 1;
        if (coarsest)
        {
            target = source.clone();
            target.clear_mask();
        }
        cv::Mat peeled;
        if (_peels(level))
            target = _peel(level, target, peeled);
        // The votes of the completeness field keep the known pixels near their input on the
        // coarse levels; on levels 0 and 1, which keep the known pixels, they change nothing.
        if (level < 2)
            m_source2target.reset();
        else
            m_source2target = coarsest ? NearestNeighborField(source, target, m_level_metric)
                                       : NearestNeighborField(source, target, m_level_metric, *m_source2target);
        m_target2source = coarsest ? NearestNeighborField(target, source, m_level_metric)
                                   : NearestNeighborField(target, source, m_level_metric, m_target2source);
        _link_peeled(peeled);

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

MaskedImage Inpainting::_peel(int level, const MaskedImage &target, cv::Mat &matches) const
{
    const MaskedImage &source = m_pyramid[level];
    const cv::Mat &holes = m_peel_labels[level];
    const int patch_size = m_distance_metric->patch_size();
    if (level == static_cast<int>(m_pyramid.size()) - 1)
        return Peel(source, target, holes, nullptr, patch_size).run(matches);
    // The field that the search of this level starts from: the links of the coarser one are off by a pixel here,
    // so the peel deviated from patterns that the coarser levels had locked onto.
    NearestNeighborField guide(target, source, m_level_metric, m_target2source);
    guide.minimize(nnf_iterations(level));
    return Peel(source, target, holes, &guide, patch_size).run(matches);
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
    const int patch_size = m_level_metric->patch_size();
    const auto size = source.size();
    parallel_for(size.height, [&source, &for_each_field, image_changed, patch_size, &size](int i) {
        for (int j = 0; j < size.width; ++j)
        {
            if (!source.contains_mask(i, j, patch_size))
                for_each_field([i, j](NearestNeighborField &field) { field.set_identity(i, j); });
            else if (image_changed && !source.is_globally_masked(i, j))
                for_each_field([i, j](NearestNeighborField &field) { field.update_distance(i, j); });
        }
    });
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

    // Bands of vote rows, each cast by one thread in the order of the pixels, so that every vote gets its terms in
    // the sequential order. Four per thread balance holes that some bands lack, and the patches across a border,
    // which both bands visit, stay a minority. A band starts at a multiple of the scale, so that both rows of an
    // upscaled vote fall into it.
    const int scale = upscaled ? 2 : 1;
    const int threads = thread_count();
    const int band =
        threads > 1 ? scale * std::max(2 * (2 * patch_size + 1), vote.rows / scale / (4 * threads)) : vote.rows;
    parallel_for(
        (vote.rows + band - 1) / band,
        [&caster, &nnf, &kDistance2Similarity, &source_size, &vote, band, scale, patch_size, source2target](int b) {
            const cv::Range rows(b * band, std::min(vote.rows, (b + 1) * band));
            // The coherence field votes around its pixels, the completeness field around their links.
            const int first = source2target ? 0 : std::max(0, rows.start / scale - patch_size);
            const int last = source2target ? source_size.height
                                           : std::min(source_size.height, (rows.end - 1) / scale + patch_size + 1);
            for (int i = first; i < last; ++i)
            {
                for (int j = 0; j < source_size.width; ++j)
                {
                    if (!nnf.source().is_globally_masked(i, j))
                        caster.cast_patch(i, j, patch_size, kDistance2Similarity[nnf.at(i, j, 2)], rows);
                }
            }
        });
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
