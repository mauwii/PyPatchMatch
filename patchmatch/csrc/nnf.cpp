#include "nnf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <random>
#include <typeinfo>

#include "masked_image.h"

/**
 * Nearest-Neighbor Field (see PatchMatch algorithm).
 * This algorithm uses a version proposed by Xavier Philippeau.
 *
 */

namespace
{
    constexpr int kMaxRetry = 20;

    // One generator per thread, so concurrent inpaintings with the same seed give the
    // same results. It only drives the randomized search, which has to be reproducible
    // from a seed, so a standard generator is the right choice. SonarCloud's PRNG rule
    // (cpp:S2245) flags every declaration of its type, hence the NOSONAR markers.
    std::mt19937 &random_engine() // NOSONAR
    {
        thread_local std::mt19937 engine; // NOSONAR
        return engine;
    }

    // The modulo instead of std::uniform_int_distribution keeps the sequence identical
    // on all platforms.
    inline int random_int(int n)
    {
        return static_cast<int>(random_engine()() % static_cast<unsigned int>(n));
    }

    inline int pow2(int i)
    {
        return i * i;
    }

    struct Position
    {
        int y;
        int x;
    };

    // A source and a target image with their sizes, which are read once per patch.
    struct ImagePair
    {
        ImagePair(const MaskedImage &source_image, const MaskedImage &target_image)
            : source(&source_image), target(&target_image), source_size(source_image.size()),
              target_size(target_image.size())
        {
        }

        bool rows_inside(int ys, int yt) const
        {
            return ys >= 0 && ys < source_size.height && yt >= 0 && yt < target_size.height;
        }

        const MaskedImage *source;
        const MaskedImage *target;
        cv::Size source_size;
        cv::Size target_size;
    };

    // One row of a source and a target image, both inside their images. The gradients must be computed.
    class RowPair
    {
    public:
        RowPair(const ImagePair &images, int ys, int yt)
            : m_source_width(images.source_size.width), m_target_width(images.target_size.width),
              m_si(images.source->image().ptr<unsigned char>(ys, 0)),
              m_ti(images.target->image().ptr<unsigned char>(yt, 0)),
              m_sm(images.source->mask().ptr<unsigned char>(ys, 0)),
              m_tm(images.target->mask().ptr<unsigned char>(yt, 0)),
              m_sgm(
                  images.source->has_global_mask() ? images.source->global_mask().ptr<unsigned char>(ys, 0) : nullptr),
              m_tgm(
                  images.target->has_global_mask() ? images.target->global_mask().ptr<unsigned char>(yt, 0) : nullptr),
              m_sgy(images.source->grady().ptr<unsigned char>(ys, 0)),
              m_tgy(images.target->grady().ptr<unsigned char>(yt, 0)),
              m_sgx(images.source->gradx().ptr<unsigned char>(ys, 0)),
              m_tgx(images.target->gradx().ptr<unsigned char>(yt, 0))
        {
        }

        // The SSD over the colors and gradients, kSSDScale where a pixel is masked or outside its image.
        int cost(int xs, int xt) const
        {
            // The bounds first: the masks are read only inside the image.
            if (xs < 0 || xs >= m_source_width || xt < 0 || xt >= m_target_width || m_sm[xs] || m_tm[xt] ||
                (m_sgm && m_sgm[xs]) || (m_tgm && m_tgm[xt]))
                return PatchSSDDistanceMetric::kSSDScale;

            int ssd = 0;
            for (int c = 0; c < 3; ++c)
            {
                ssd += pow2(m_si[xs * 3 + c] - m_ti[xt * 3 + c]);
                ssd += pow2(m_sgx[xs * 3 + c] - m_tgx[xt * 3 + c]);
                ssd += pow2(m_sgy[xs * 3 + c] - m_tgy[xt * 3 + c]);
            }
            return ssd;
        }

    private:
        int m_source_width;
        int m_target_width;
        const unsigned char *m_si;
        const unsigned char *m_ti;
        const unsigned char *m_sm;
        const unsigned char *m_tm;
        const unsigned char *m_sgm;
        const unsigned char *m_tgm;
        const unsigned char *m_sgy;
        const unsigned char *m_tgy;
        const unsigned char *m_sgx;
        const unsigned char *m_tgx;
    };

    // The costs of the row of the patches around s and t.
    std::int64_t row_sum(const ImagePair &images, Position s, Position t, int patch_size)
    {
        if (!images.rows_inside(s.y, t.y))
            return std::int64_t{PatchSSDDistanceMetric::kSSDScale} * (2 * patch_size + 1);
        const RowPair row(images, s.y, t.y);
        std::int64_t sum = 0;
        for (int dx = -patch_size; dx <= patch_size; ++dx)
            sum += row.cost(s.x + dx, t.x + dx);
        return sum;
    }

    // The costs of the column of the patches around s and t.
    std::int64_t column_sum(const ImagePair &images, Position s, Position t, int patch_size)
    {
        std::int64_t sum = 0;
        for (int dy = -patch_size; dy <= patch_size; ++dy)
        {
            if (images.rows_inside(s.y + dy, t.y + dy))
                sum += RowPair(images, s.y + dy, t.y + dy).cost(s.x, t.x);
            else
                sum += PatchSSDDistanceMetric::kSSDScale;
        }
        return sum;
    }

    // The costs of the patches around s and t, or -1 once their distance is certain to exceed bound.
    std::int64_t patch_sum(
        const MaskedImage &source, Position s, const MaskedImage &target, Position t, int patch_size, int bound)
    {
        // Exact integer sums: long double is emulated in software on Linux aarch64
        std::int64_t sum = 0;

        // Above this sum the result is at least bound + 1, a margin for the rounding of the doubles.
        const double side = 2.0 * patch_size + 1;
        const double limit =
            (bound + 1.0) * PatchSSDDistanceMetric::kSSDScale * side * side / PatchDistanceMetric::kDistanceScale;

        source.compute_image_gradients();
        target.compute_image_gradients();

        const ImagePair images(source, target);
        for (int dy = -patch_size; dy <= patch_size; ++dy)
        {
            if (static_cast<double>(sum) > limit)
                return -1;
            sum += row_sum(images, {s.y + dy, s.x}, {t.y + dy, t.x}, patch_size);
        }
        return sum;
    }

    // Scales a sum of costs to [0, kDistanceScale].
    int scale_sum(std::int64_t sum, int patch_size)
    {
        const std::int64_t side = 2 * patch_size + 1;
        const double scaled = static_cast<double>(sum) / PatchSSDDistanceMetric::kSSDScale;
        const auto res =
            static_cast<int>(PatchDistanceMetric::kDistanceScale * scaled / static_cast<double>(side * side));
        if (res < 0 || res > PatchDistanceMetric::kDistanceScale)
            return PatchDistanceMetric::kDistanceScale;
        return res;
    }
} // namespace

void NearestNeighborField::seed_random(unsigned int seed)
{
    random_engine().seed(seed);
}

void NearestNeighborField::_allocate_sums()
{
    // Only the plain SSD adds up line by line; a subclass may measure differently.
    if (typeid(*m_distance_metric) == typeid(PatchSSDDistanceMetric))
        m_sums.assign(static_cast<std::size_t>(m_field.rows) * m_field.cols, -1);
}

NearestNeighborField::Measured NearestNeighborField::_measure(int y, int x, int y_target, int x_target, int bound) const
{
    if (m_sums.empty())
        return {(*m_distance_metric)(m_source, y, x, m_target, y_target, x_target), -1};
    const int patch_size = m_distance_metric->patch_size();
    const std::int64_t sum = patch_sum(m_source, {y, x}, m_target, {y_target, x_target}, patch_size, bound);
    if (sum < 0)
        return {PatchDistanceMetric::kDistanceScale, -1};
    return {scale_sum(sum, patch_size), sum};
}

void NearestNeighborField::_randomize_field(bool reset)
{
    auto this_size = source_size();
    for (int i = 0; i < this_size.height; ++i)
    {
        for (int j = 0; j < this_size.width; ++j)
        {
            if (m_source.is_globally_masked(i, j))
                continue;
            if (!reset && at(i, j, 2) < PatchDistanceMetric::kDistanceScale)
                continue;
            _randomize_link(i, j);
        }
    }
}

void NearestNeighborField::_randomize_link(int y, int x)
{
    auto this_target_size = target_size();
    int y_target = 0;
    int x_target = 0;
    Measured measured{PatchDistanceMetric::kDistanceScale, -1};
    for (int t = 0; t < kMaxRetry; ++t)
    {
        y_target = random_int(this_target_size.height);
        x_target = random_int(this_target_size.width);
        if (m_target.is_globally_masked(y_target, x_target))
        {
            measured.sum = -1; // the distance of an earlier try, which is kDistanceScale
            continue;
        }

        measured = _measure(y, x, y_target, x_target, PatchDistanceMetric::kDistanceScale);
        if (measured.distance < PatchDistanceMetric::kDistanceScale)
            break;
    }
    _set(y, x, y_target, x_target, measured);
}

void NearestNeighborField::_initialize_field_from(const NearestNeighborField &other)
{
    const auto &this_size = source_size();
    const auto &other_size = other.source_size();
    double fi = static_cast<double>(this_size.height) / other_size.height;
    double fj = static_cast<double>(this_size.width) / other_size.width;

    for (int i = 0; i < this_size.height; ++i)
    {
        for (int j = 0; j < this_size.width; ++j)
        {
            if (m_source.is_globally_masked(i, j))
                continue;

            auto ilow = static_cast<int>(std::min(i / fi, static_cast<double>(other_size.height - 1)));
            auto jlow = static_cast<int>(std::min(j / fj, static_cast<double>(other_size.width - 1)));
            auto other_value = other.ptr(ilow, jlow);

            // Keep the offset within the coarse pixel, so that neighbors still point to
            // neighbors. Without it, both pixels of a pair point to the same target pixel,
            // which makes the upscaled fill blocky.
            const int y_target =
                std::clamp(static_cast<int>(other_value[0] * fi + (i - ilow * fi)), 0, target_size().height - 1);
            const int x_target =
                std::clamp(static_cast<int>(other_value[1] * fj + (j - jlow * fj)), 0, target_size().width - 1);
            const Measured measured = _measure(i, j, y_target, x_target, PatchDistanceMetric::kDistanceScale);
            _set(i, j, y_target, x_target, measured);

            // A hole that is new on this level: search near the patch before a neighbor's link replaces its own.
            const bool linked_to_itself = other_value[0] == ilow && other_value[1] == jlow;
            if (linked_to_itself && measured.distance > 0 &&
                m_target.contains_mask(y_target, x_target, m_distance_metric->patch_size()))
                _random_search(i, j);
        }
    }

    _randomize_field(false);
}

void NearestNeighborField::minimize(int nr_pass)
{
    const auto &this_size = source_size();
    while (nr_pass--)
    {
        for (int i = 0; i < this_size.height; ++i)
            for (int j = 0; j < this_size.width; ++j)
                _minimize_link(i, j, +1);
        for (int i = this_size.height - 1; i >= 0; --i)
            for (int j = this_size.width - 1; j >= 0; --j)
                _minimize_link(i, j, -1);
    }
}

void NearestNeighborField::_minimize_link(int y, int x, int direction)
{
    // Globally masked pixels have no link, and a distance of 0 cannot improve.
    if (m_source.is_globally_masked(y, x) || at(y, x, 2) <= 0)
        return;

    const auto &this_size = source_size();

    // propagation along the y direction.
    if (y - direction >= 0 && y - direction < this_size.height && !m_source.is_globally_masked(y - direction, x))
        _propagate(y, x, direction, 0);

    // propagation along the x direction.
    if (x - direction >= 0 && x - direction < this_size.width && !m_source.is_globally_masked(y, x - direction))
        _propagate(y, x, 0, direction);

    _random_search(y, x);
}

// Tries the link of the neighbor (y - dy, x - dx), shifted by (dy, dx). With the neighbor's sum,
// its distance takes the line that leaves the patch and the one that enters it (paper 3.3).
void NearestNeighborField::_propagate(int y, int x, int dy, int dx)
{
    const int y_target = at(y - dy, x - dx, 0) + dy;
    const int x_target = at(y - dy, x - dx, 1) + dx;
    const std::int64_t neighbor_sum = m_sums.empty() ? -1 : m_sums[_index(y - dy, x - dx)];
    if (neighbor_sum < 0)
    {
        _link_if_closer(y, x, y_target, x_target);
        return;
    }
    if (y_target == at(y, x, 0) && x_target == at(y, x, 1))
        return;

    const int patch_size = m_distance_metric->patch_size();
    const ImagePair images(m_source, m_target);
    const auto line_sum = [&images, dy, patch_size](Position s, Position t) {
        return dy != 0 ? row_sum(images, s, t, patch_size) : column_sum(images, s, t, patch_size);
    };
    const int leaving = patch_size + 1;
    const std::int64_t sum =
        neighbor_sum -
        line_sum({y - dy * leaving, x - dx * leaving}, {y_target - dy * leaving, x_target - dx * leaving}) +
        line_sum({y + dy * patch_size, x + dx * patch_size}, {y_target + dy * patch_size, x_target + dx * patch_size});
    // The sanitizer build keeps the asserts, so the tests compare every sum with a full measurement.
    assert(
        sum ==
        patch_sum(m_source, {y, x}, m_target, {y_target, x_target}, patch_size, PatchDistanceMetric::kDistanceScale));
    const int distance = scale_sum(sum, patch_size);
    if (distance < at(y, x, 2))
        _set(y, x, y_target, x_target, {distance, sum});
}

// Random search around the link in windows of decreasing size, clamped to the image.
void NearestNeighborField::_random_search(int y, int x)
{
    const auto size = target_size();
    const int *link = ptr(y, x);
    for (int radius = (std::min(size.height, size.width) - 1) / 2; radius > 0; radius /= 2)
    {
        // A propagated link can lie outside the image, which would leave the window empty.
        const int yc = std::clamp(link[0], 0, size.height - 1);
        const int xc = std::clamp(link[1], 0, size.width - 1);
        const int top = std::max(yc - radius, 0);
        const int left = std::max(xc - radius, 0);
        const int yp = top + random_int(std::min(yc + radius, size.height - 1) - top + 1);
        const int xp = left + random_int(std::min(xc + radius, size.width - 1) - left + 1);
        if (!m_target.is_globally_masked(yp, xp))
            _link_if_closer(y, x, yp, xp);
    }
}

const int PatchDistanceMetric::kDistanceScale = 65535;
const int PatchSSDDistanceMetric::kSSDScale = 9 * 255 * 255;

namespace
{
    // Returns kDistanceScale once the result is certain to reach bound, never for bound kDistanceScale.
    int distance_masked_images(
        const MaskedImage &source, Position source_center, const MaskedImage &target, Position target_center,
        int patch_size, int bound)
    {
        const std::int64_t sum = patch_sum(source, source_center, target, target_center, patch_size, bound);
        return sum < 0 ? PatchDistanceMetric::kDistanceScale : scale_sum(sum, patch_size);
    }
} // namespace

int PatchSSDDistanceMetric::operator()(
    const MaskedImage &source, int source_y, int source_x, const MaskedImage &target, int target_y, int target_x) const
{
    return distance_masked_images(
        source, {source_y, source_x}, target, {target_y, target_x}, patch_size(), PatchDistanceMetric::kDistanceScale);
}

int RegularityGuidedPatchDistanceMetricV2::operator()(
    const MaskedImage &source, int source_y, int source_x, const MaskedImage &target, int target_y, int target_x) const
{
    if (target_y < 0 || target_y >= target.size().height || target_x < 0 || target_x >= target.size().width)
        return PatchDistanceMetric::kDistanceScale;

    // Map pyramid-level coordinates to the full-resolution ijmap, height and width
    // independently; a coordinate inside the image stays inside the map. In 64 bits:
    // the products overflow an int from 46,342 pixels on.
    const int map_h = m_ijmap.size().height;
    const int map_w = m_ijmap.size().width;
    auto map_y = [&](const MaskedImage &img, int y) {
        return static_cast<int>(std::int64_t{y} * map_h / img.size().height);
    };
    auto map_x = [&](const MaskedImage &img, int x) {
        return static_cast<int>(std::int64_t{x} * map_w / img.size().width);
    };

    double score1 = PatchDistanceMetric::kDistanceScale;
    if (!source.is_globally_masked(source_y, source_x) && !target.is_globally_masked(target_y, target_x))
    {
        auto source_ij = m_ijmap.ptr<float>(map_y(source, source_y), map_x(source, source_x));
        auto target_ij = m_ijmap.ptr<float>(map_y(target, target_y), map_x(target, target_x));

        float di = std::fabs(source_ij[0] - target_ij[0]);
        if (di > 0.5F)
            di = 1.0F - di;
        float dj = std::fabs(source_ij[1] - target_ij[1]);
        if (dj > 0.5F)
            dj = 1.0F - dj;
        score1 = sqrt(di * di + dj * dj) / 0.707;
        if (score1 < 0 || score1 > 1)
            score1 = 1;
        score1 *= PatchDistanceMetric::kDistanceScale;
    }

    double score2 = distance_masked_images(
        source, {source_y, source_x}, target, {target_y, target_x}, patch_size(), PatchDistanceMetric::kDistanceScale);
    double score = (score1 * m_weight + score2) / (1 + m_weight);
    // The distance indexes the similarity table in Inpainting, so keep it in range.
    if (!(score > 0))
        return 0;
    if (score >= PatchDistanceMetric::kDistanceScale)
        return PatchDistanceMetric::kDistanceScale;
    return static_cast<int>(score);
}
