#include "peel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <utility>

#include "parallel.h"
#include "window.h"

namespace
{
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
        // holes: the labels of the holes to fill (CV_32S), 0 elsewhere; reach: per label the coarsest level that the
        // hole reaches; guide: the field of this level that the search would start from, nullptr on the coarsest level.
        Peel(
            const MaskedImage &source, const MaskedImage &target, const cv::Mat &holes, const std::vector<int> &reach,
            const NearestNeighborField *guide, int patch_size, int level)
            : m_source(source), m_target(target), m_holes(holes), m_reach(reach), m_guide(guide),
              m_patch_size(patch_size), m_level(level), m_size(source.size()), m_colors(source.image().clone()),
              m_filled(m_size, CV_8U, cv::Scalar(0)), m_front_mask(m_size, CV_8U, cv::Scalar(0)),
              m_confidence(m_size, CV_64F, cv::Scalar(0)), m_matches(m_size, CV_32SC2, cv::Scalar::all(-1)),
              m_lines(m_size, CV_8U, cv::Scalar(0))
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

        PeelResult run()
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
            return {std::move(peeled), m_matches, m_lines};
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
        // The guide's own link wins unless the known pixels clearly contradict it, as at the end of a line. On the two
        // finest levels only if its source is about as varied as the best fit's: the coarser fill lost the lines a few
        // pixels wide there, and the link copied the background across them.
        static constexpr double kGuideTolerance = 2;
        static constexpr double kGuideVariance = 0.75;
        // More where the coarser levels locked onto a structure, which the peel broke into pieces of the wrong phase:
        // the guide's links around the pixel share one offset, and its source is about as varied as the best fit's,
        // unlike a coherent background across which a line continues.
        static constexpr double kLockedGuideTolerance = 4;
        // A fit this much worse than the mean of the pixels of its hole before calls for a search of all sources: the
        // random search does not find the few sources of a line. The first pixels of the level search all, and the
        // first of each hole: shared by all holes, the warm-up left the lines of the later ones broken. The mean
        // forgets slowly.
        static constexpr double kPoorFit = 3;
        static constexpr int kWarmUp = 8;
        static constexpr int kHoleWarmUp = 2;
        static constexpr double kFitMemory = 0.9;
        // Where the coarser fill is the first of a hole and this much flatter than the filled pixels around the pixel,
        // it lost a line, and the near ties closest to it ended or bent the line: only sources about as varied as the
        // best fit's count there.
        static constexpr double kFlatGuide = 0.15;
        // There, a best fit this much flatter than the filled pixels around the pixel lost a line too: the sources
        // nearby are searched for its few ones. In the whole image, that doubled the time of large images.
        static constexpr double kFlatFit = 0.3;
        static constexpr int kLineSearchRadius = 16;
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
                _search(p, _sources());
            else if (_first_fill(p) && _variance(*_best_fit()) < kFlatFit * _variance_around(p, Role::kFit))
            {
                constexpr int side = 2 * kLineSearchRadius + 1;
                _search(p, _sources() & cv::Rect(p.x - kLineSearchRadius, p.y - kLineSearchRadius, side, side));
            }
            if (m_candidates.empty())
                return {-1, -1};

            const Candidate &chosen = _pick(p, guided);
            m_lines.at<unsigned char>(p.y, p.x) = _lost_line(p);
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

        // The centers of the source patches inside the image.
        cv::Rect _sources() const
        {
            return {
                m_patch_size, m_patch_size, std::max(m_size.width - 2 * m_patch_size, 0),
                std::max(m_size.height - 2 * m_patch_size, 0)};
        }

        // Every source with its center in centers, a row per task. Each keeps the near ties of its own best fit, among
        // them all that the scan in order keeps, so that adding them in that order gives the same candidates and best
        // fit.
        void _search(cv::Point p, const cv::Rect &centers)
        {
            std::vector<std::vector<Candidate>> found(static_cast<std::size_t>(centers.height));
            m_source.compute_hole_sums(); // before the rows read it
            parallel_for(centers.height, [this, p, &centers, &found](int row) {
                std::int64_t best = m_best;
                const int qy = centers.y + row;
                for (int qx = centers.x; qx < centers.x + centers.width; ++qx)
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

        // The guide's link, otherwise among the near ties of the best fit the patch closest to the coarser fill; where
        // that lost a line, only among those about as varied as the best fit.
        const Candidate &_pick(cv::Point p, bool guided) const
        {
            const auto best = static_cast<double>(m_best);
            if (guided)
            {
                const auto fit = static_cast<double>(m_candidates[0].fit);
                if ((fit <= (1 + kGuideTolerance) * best + 1 && (m_level > 1 || _varied())) ||
                    (fit <= (1 + kLockedGuideTolerance) * best + 1 && _locked(p) && _varied()))
                    return m_candidates[0];
            }
            const double near = (1 + kNearTie) * best + 1;
            // the first best fit, which is a near tie itself
            auto chosen =
                std::min_element(m_candidates.begin(), m_candidates.end(), [](const Candidate &a, const Candidate &b) {
                    return a.fit < b.fit;
                });
            const bool varied_only = _lost_line(p);
            const double least_variance = varied_only ? kGuideVariance * _variance(*chosen) : 0;
            for (auto candidate = m_candidates.begin(); candidate != m_candidates.end(); ++candidate)
            {
                if (static_cast<double>(candidate->fit) <= near &&
                    (candidate->guide < chosen->guide ||
                     (candidate->guide == chosen->guide && candidate->fit < chosen->fit)) &&
                    (!varied_only || _variance(*candidate) >= least_variance))
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

        // Whether the source of the guide's link is about as varied as that of the best fit.
        bool _varied() const
        {
            return _variance(m_candidates[0]) >= kGuideVariance * _variance(*_best_fit());
        }

        // The variance of the colors of a candidate's source patch.
        double _variance(const Candidate &candidate) const
        {
            return window_variance(
                m_source.image(), candidate.y, candidate.x, m_patch_size, [](int, int) { return true; });
        }

        // Whether the coarser fill around p is the first of its hole and much flatter than the filled pixels around p.
        bool _lost_line(cv::Point p) const
        {
            return _first_fill(p) && _variance_around(p, Role::kGuide) < kFlatGuide * _variance_around(p, Role::kFit);
        }

        // Whether the coarser fill around p is the first of its hole, on the two finest levels: it lost the lines a few
        // pixels wide there.
        bool _first_fill(cv::Point p) const
        {
            return m_level <= 1 && m_reach[static_cast<std::size_t>(m_holes.at<int>(p.y, p.x))] > m_level;
        }

        // The variance of the colors around p of the pixels in role: the filled ones, of which a pixel of the front has
        // one, or the coarser fill, which includes p.
        double _variance_around(cv::Point p, Role role) const
        {
            return window_variance(
                m_colors, p.y, p.x, m_patch_size, [this, role](int y, int x) { return _role(y, x) == role; });
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
        const std::vector<int> &m_reach;
        const NearestNeighborField *m_guide;
        int m_patch_size;
        int m_level;
        cv::Size m_size;
        cv::Mat m_colors;
        cv::Mat m_filled;
        cv::Mat m_front_mask;
        cv::Mat m_confidence;
        cv::Mat m_matches;
        cv::Mat m_lines;
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

PeelResult peel_holes(
    const MaskedImage &source, const MaskedImage &target, const cv::Mat &holes, const std::vector<int> &reach,
    const NearestNeighborField *guide, int patch_size, int level)
{
    return Peel(source, target, holes, reach, guide, patch_size, level).run();
}
