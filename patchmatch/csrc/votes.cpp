#include "votes.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <utility>

#include "parallel.h"
#include "window.h"

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
} // namespace

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

void cast_votes(
    const NearestNeighborField &nnf, bool source2target, cv::Mat &vote, const MaskedImage &source,
    const VoteSettings &settings)
{
    // Votes for known pixels do not count where the maximization step keeps them.
    std::optional<HoleCounter> holes;
    if (settings.keep_known)
        holes.emplace(source);
    const VoteCaster caster(
        nnf, source2target, vote, source, settings.upscaled, settings.best_only,
        holes.has_value() ? &holes.value() : nullptr);
    const int patch_size = settings.patch_size;
    const auto &kDistance2Similarity = distance2similarity();
    const auto source_size = nnf.source_size();

    // Bands of vote rows, each cast by one thread in the order of the pixels, so that every vote gets its terms in
    // the sequential order. Four per thread balance holes that some bands lack, and the patches across a border,
    // which both bands visit, stay a minority. A band starts at a multiple of the scale, so that both rows of an
    // upscaled vote fall into it.
    const int scale = settings.upscaled ? 2 : 1;
    const int threads = thread_count();
    const int band = threads > 1 ? scale * std::max(2 * (2 * patch_size + 1), vote.rows / scale / (4 * threads))
                                 : std::max(1, vote.rows);
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
                        caster.cast_patch(
                            i, j, std::min(patch_size, nnf.patch_size(i, j)), kDistance2Similarity[nnf.at(i, j, 2)],
                            rows);
                }
            }
        });
}

void maximize_votes(MaskedImage &target, const cv::Mat &vote, const MaskedImage &source, bool keep_known)
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
