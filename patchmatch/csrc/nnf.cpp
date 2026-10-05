#include "nnf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>

#include "masked_image.h"

/**
 * Nearest-Neighbor Field (see PatchMatch algorithm).
 * This algorithm uses a version proposed by Xavier Philippeau.
 *
 */

template <typename T>
T clamp(T value, T min_value, T max_value)
{
    return std::min(std::max(value, min_value), max_value);
}

namespace
{
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
} // namespace

void NearestNeighborField::seed_random(unsigned int seed)
{
    random_engine().seed(seed);
}

void NearestNeighborField::_randomize_field(int max_retry, bool reset)
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
            _randomize_link(i, j, max_retry);
        }
    }
}

void NearestNeighborField::_randomize_link(int y, int x, int max_retry)
{
    auto this_target_size = target_size();
    int y_target = 0;
    int x_target = 0;
    int distance = PatchDistanceMetric::kDistanceScale;
    for (int t = 0; t < max_retry; ++t)
    {
        y_target = random_int(this_target_size.height);
        x_target = random_int(this_target_size.width);
        if (m_target.is_globally_masked(y_target, x_target))
            continue;

        distance = _distance(y, x, y_target, x_target);
        if (distance < PatchDistanceMetric::kDistanceScale)
            break;
    }
    _set(y, x, y_target, x_target, distance);
}

void NearestNeighborField::_initialize_field_from(const NearestNeighborField &other, int max_retry)
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
            auto this_value = mutable_ptr(i, j);
            auto other_value = other.ptr(ilow, jlow);

            // Keep the offset within the coarse pixel, so that neighbors still point to
            // neighbors. Without it, both pixels of a pair point to the same target pixel,
            // which makes the upscaled fill blocky.
            this_value[0] = clamp(static_cast<int>(other_value[0] * fi + (i - ilow * fi)), 0, target_size().height - 1);
            this_value[1] = clamp(static_cast<int>(other_value[1] * fj + (j - jlow * fj)), 0, target_size().width - 1);
            this_value[2] = _distance(i, j, this_value[0], this_value[1]);

            // A hole that is new on this level: search near the patch before a neighbor's link replaces its own.
            const bool linked_to_itself = other_value[0] == ilow && other_value[1] == jlow;
            if (linked_to_itself && this_value[2] > 0 &&
                m_target.contains_mask(this_value[0], this_value[1], m_distance_metric->patch_size()))
                _random_search(i, j);
        }
    }

    _randomize_field(max_retry, false);
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
    {
        int yp = at(y - direction, x, 0) + direction;
        int xp = at(y - direction, x, 1);
        _link_if_closer(y, x, yp, xp);
    }

    // propagation along the x direction.
    if (x - direction >= 0 && x - direction < this_size.width && !m_source.is_globally_masked(y, x - direction))
    {
        int yp = at(y, x - direction, 0);
        int xp = at(y, x - direction, 1) + direction;
        _link_if_closer(y, x, yp, xp);
    }

    _random_search(y, x);
}

// Random search around the link with a progressive step size.
void NearestNeighborField::_random_search(int y, int x)
{
    const auto &this_target_size = target_size();
    const int *this_ptr = ptr(y, x);

    int random_scale = (std::min(this_target_size.height, this_target_size.width) - 1) / 2;
    while (random_scale > 0)
    {
        int yp = this_ptr[0] + (random_int(2 * random_scale + 1) - random_scale);
        int xp = this_ptr[1] + (random_int(2 * random_scale + 1) - random_scale);
        yp = clamp(yp, 0, target_size().height - 1);
        xp = clamp(xp, 0, target_size().width - 1);

        if (m_target.is_globally_masked(yp, xp))
        {
            random_scale /= 2;
            continue;
        }

        _link_if_closer(y, x, yp, xp);
        random_scale /= 2;
    }
}

const int PatchDistanceMetric::kDistanceScale = 65535;
const int PatchSSDDistanceMetric::kSSDScale = 9 * 255 * 255;

namespace
{

    inline int pow2(int i)
    {
        return i * i;
    }

    struct Position
    {
        int y;
        int x;
    };

    // Returns kDistanceScale once the result is certain to reach bound, never for bound kDistanceScale.
    int distance_masked_images(
        const MaskedImage &source, Position source_center, const MaskedImage &target, Position target_center,
        int patch_size, int bound)
    {
        const auto [ys, xs] = source_center;
        const auto [yt, xt] = target_center;

        // Exact integer sums: long double is emulated in software on Linux aarch64
        std::int64_t distance = 0;
        std::int64_t wsum = 0;

        // Above this sum the result is at least bound + 1, a margin for the rounding of the doubles.
        const double side = 2.0 * patch_size + 1;
        const double limit =
            (bound + 1.0) * PatchSSDDistanceMetric::kSSDScale * side * side / PatchDistanceMetric::kDistanceScale;

        source.compute_image_gradients();
        target.compute_image_gradients();

        auto source_size = source.size();
        auto target_size = target.size();

        for (int dy = -patch_size; dy <= patch_size; ++dy)
        {
            if (static_cast<double>(distance) > limit)
                return PatchDistanceMetric::kDistanceScale;

            const int yys = ys + dy;
            const int yyt = yt + dy;

            if (yys <= 0 || yys >= source_size.height - 1 || yyt <= 0 || yyt >= target_size.height - 1)
            {
                distance += std::int64_t{PatchSSDDistanceMetric::kSSDScale} * (2 * patch_size + 1);
                wsum += 2 * patch_size + 1;
                continue;
            }

            const auto *p_si = source.image().ptr<unsigned char>(yys, 0);
            const auto *p_ti = target.image().ptr<unsigned char>(yyt, 0);
            const auto *p_sm = source.mask().ptr<unsigned char>(yys, 0);
            const auto *p_tm = target.mask().ptr<unsigned char>(yyt, 0);

            const unsigned char *p_sgm = nullptr;
            const unsigned char *p_tgm = nullptr;
            if (source.has_global_mask())
                p_sgm = source.global_mask().ptr<unsigned char>(yys, 0);
            if (target.has_global_mask())
                p_tgm = target.global_mask().ptr<unsigned char>(yyt, 0);

            const auto *p_sgy = source.grady().ptr<unsigned char>(yys, 0);
            const auto *p_tgy = target.grady().ptr<unsigned char>(yyt, 0);
            const auto *p_sgx = source.gradx().ptr<unsigned char>(yys, 0);
            const auto *p_tgx = target.gradx().ptr<unsigned char>(yyt, 0);

            for (int dx = -patch_size; dx <= patch_size; ++dx)
            {
                const int xxs = xs + dx;
                const int xxt = xt + dx;
                wsum += 1;

                // The bounds first: the masks are read only inside the image.
                if (xxs <= 0 || xxs >= source_size.width - 1 || xxt <= 0 || xxt >= target_size.width - 1 || p_sm[xxs] ||
                    p_tm[xxt] || (p_sgm && p_sgm[xxs]) || (p_tgm && p_tgm[xxt]))
                {
                    distance += PatchSSDDistanceMetric::kSSDScale;
                    continue;
                }

                int ssd = 0;
                for (int c = 0; c < 3; ++c)
                {
                    int s_value = p_si[xxs * 3 + c];
                    int t_value = p_ti[xxt * 3 + c];
                    int s_gy = p_sgy[xxs * 3 + c];
                    int t_gy = p_tgy[xxt * 3 + c];
                    int s_gx = p_sgx[xxs * 3 + c];
                    int t_gx = p_tgx[xxt * 3 + c];

                    ssd += pow2(s_value - t_value);
                    ssd += pow2(s_gx - t_gx);
                    ssd += pow2(s_gy - t_gy);
                }
                distance += ssd;
            }
        }

        const double scaled = static_cast<double>(distance) / PatchSSDDistanceMetric::kSSDScale;
        const auto res = static_cast<int>(PatchDistanceMetric::kDistanceScale * scaled / static_cast<double>(wsum));
        if (res < 0 || res > PatchDistanceMetric::kDistanceScale)
            return PatchDistanceMetric::kDistanceScale;
        return res;
    }

} // namespace

int PatchSSDDistanceMetric::operator()(
    const MaskedImage &source, int source_y, int source_x, const MaskedImage &target, int target_y, int target_x) const
{
    return distance_masked_images(
        source, {source_y, source_x}, target, {target_y, target_x}, patch_size(), PatchDistanceMetric::kDistanceScale);
}

int PatchSSDDistanceMetric::distance_below(
    const MaskedImage &source, int source_y, int source_x, const MaskedImage &target, int target_y, int target_x,
    int bound) const
{
    return distance_masked_images(source, {source_y, source_x}, target, {target_y, target_x}, patch_size(), bound);
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
