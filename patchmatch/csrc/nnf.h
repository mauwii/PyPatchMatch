#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <vector>

#include "masked_image.h"

class PatchDistanceMetric
{
public:
    explicit PatchDistanceMetric(int patch_size) : m_patch_size(patch_size)
    {
        // 0 crashes, negative sizes never terminate, and 2 * patch_size + 1 has to fit an int.
        CV_Assert(patch_size >= 1 && patch_size <= (std::numeric_limits<int>::max() - 1) / 2);
    }
    virtual ~PatchDistanceMetric() = default;

    int patch_size() const
    {
        return m_patch_size;
    }
    virtual int operator()(
        const MaskedImage &source, int source_y, int source_x, const MaskedImage &target, int target_y,
        int target_x) const = 0;
    // The same metric with another patch size, or nullptr if it has none.
    virtual std::unique_ptr<PatchDistanceMetric> with_patch_size(int) const
    {
        return nullptr;
    }
    static constexpr int kDistanceScale = 65535;

private:
    int m_patch_size;
};

class NearestNeighborField
{
public:
    NearestNeighborField() = default;
    // hole_metric, of the type of metric, measures the pixels of the source that hole_pixels (CV_8U) sets.
    NearestNeighborField(
        const MaskedImage &source, const MaskedImage &target, const PatchDistanceMetric *metric,
        const PatchDistanceMetric *hole_metric = nullptr, const cv::Mat &hole_pixels = cv::Mat())
        : m_source(source), m_target(target), m_distance_metric(metric),
          m_hole_metric(hole_pixels.empty() ? nullptr : hole_metric), m_hole_pixels(hole_pixels)
    {
        m_field = cv::Mat(m_source.size(), CV_32SC3);
        _allocate_sums();
        _randomize_field(true);
    }
    NearestNeighborField(
        const MaskedImage &source, const MaskedImage &target, const PatchDistanceMetric *metric,
        const NearestNeighborField &other, const PatchDistanceMetric *hole_metric = nullptr,
        const cv::Mat &hole_pixels = cv::Mat())
        : m_source(source), m_target(target), m_distance_metric(metric),
          m_hole_metric(hole_pixels.empty() ? nullptr : hole_metric), m_hole_pixels(hole_pixels)
    {
        m_field = cv::Mat(m_source.size(), CV_32SC3);
        _allocate_sums();
        _initialize_field_from(other);
    }

    const MaskedImage &source() const
    {
        return m_source;
    }
    const MaskedImage &target() const
    {
        return m_target;
    }
    cv::Size source_size() const
    {
        return m_source.size();
    }
    cv::Size target_size() const
    {
        return m_target.size();
    }
    void set_source(const MaskedImage &source)
    {
        m_source = source;
        std::fill(m_sums.begin(), m_sums.end(), -1);
    }
    void set_target(const MaskedImage &target)
    {
        m_target = target;
        std::fill(m_sums.begin(), m_sums.end(), -1);
    }

    const int *ptr(int y, int x) const
    {
        return m_field.ptr<int>(y, x);
    }

    int at(int y, int x, int c) const
    {
        return m_field.ptr<int>(y, x)[c];
    }
    // The patch size that the pixel of the source is measured with.
    int patch_size(int y, int x) const
    {
        return _metric(y, x).patch_size();
    }
    // The distance 0 is not measured, so the sum stays unknown.
    void set_identity(int y, int x)
    {
        _set(y, x, y, x, {0, -1});
    }
    // Computes the caches of both images, which the parallel loops read.
    void prepare() const
    {
        m_source.compute_features();
        m_target.compute_features();
        m_source.compute_hole_sums();
        m_target.compute_hole_sums();
    }
    // Measures the link again after an image changed.
    void update_distance(int y, int x)
    {
        _set(
            y, x, at(y, x, 0), at(y, x, 1),
            _measure(y, x, at(y, x, 0), at(y, x, 1), PatchDistanceMetric::kDistanceScale));
    }

    void minimize(int nr_pass);

    void link_if_closer(int y, int x, int y_target, int x_target)
    {
        // In coherent regions most propagated candidates are the link itself, measured in full otherwise.
        if (y_target == at(y, x, 0) && x_target == at(y, x, 1))
            return;
        const int current = at(y, x, 2);
        const Measured measured = _measure(y, x, y_target, x_target, current);
        if (measured.distance < current)
            _set(y, x, y_target, x_target, measured);
    }

    // Seeds the random search of the calling thread.
    static void seed_random(unsigned int seed);
    // A number in [0, n) from the generator of the calling thread.
    static int random_index(int n);

private:
    // A distance and the sum of the pixel costs behind it, -1 if unknown.
    struct Measured
    {
        int distance;
        std::int64_t sum;
    };

    const PatchDistanceMetric &_metric(int y, int x) const
    {
        return m_hole_metric && m_hole_pixels.at<unsigned char>(y, x) ? *m_hole_metric : *m_distance_metric;
    }
    // The distance if it is below bound, otherwise any value >= bound.
    Measured _measure(int y, int x, int y_target, int x_target, int bound) const;
    void _set(int y, int x, int y_target, int x_target, Measured measured)
    {
        auto ptr = m_field.ptr<int>(y, x);
        ptr[0] = y_target;
        ptr[1] = x_target;
        ptr[2] = measured.distance;
        if (!m_sums.empty())
            m_sums[_index(y, x)] = measured.sum;
    }
    std::size_t _index(int y, int x) const
    {
        return static_cast<std::size_t>(y) * m_field.cols + x;
    }

    void _randomize_field(bool reset);
    void _randomize_link(int y, int x);
    void _initialize_field_from(const NearestNeighborField &other);
    bool _initialize_link_from(const NearestNeighborField &other, int i, int j, double fi, double fj);
    void _allocate_sums();
    void _pass(int direction);
    void _draw(
        int direction, int top, int rows, std::vector<std::size_t> &first_draws,
        std::vector<std::uint32_t> &draws) const;
    void _minimize_tile(
        int direction, int top, const cv::Rect &tile, const std::size_t *first_draws, const std::uint32_t *draws);
    // Globally masked pixels have no link, and a distance of 0 cannot improve.
    bool _searches(int y, int x) const
    {
        return !m_source.is_globally_masked(y, x) && at(y, x, 2) > 0;
    }
    // draw() gives the next number of a generator.
    template <typename Draw>
    void _minimize_link(int y, int x, int direction, Draw &draw);
    void _propagate(int y, int x, int dy, int dx);
    template <typename Draw>
    void _random_search(int y, int x, Draw &draw);

    MaskedImage m_source;
    MaskedImage m_target;
    // per pixel: the target y and x and the scaled distance to it
    cv::Mat m_field;
    // per pixel: the sum behind the distance, for updates in O(patch_size); empty for other metrics
    std::vector<std::int64_t> m_sums;
    const PatchDistanceMetric *m_distance_metric = nullptr;
    const PatchDistanceMetric *m_hole_metric = nullptr; // nullptr without hole pixels
    cv::Mat m_hole_pixels;
};

class PatchSSDDistanceMetric : public PatchDistanceMetric
{
public:
    using PatchDistanceMetric::PatchDistanceMetric;
    int operator()(
        const MaskedImage &source, int source_y, int source_x, const MaskedImage &target, int target_y,
        int target_x) const override;
    std::unique_ptr<PatchDistanceMetric> with_patch_size(int patch_size) const override
    {
        return std::make_unique<PatchSSDDistanceMetric>(patch_size);
    }
    static constexpr int kSSDScale = 9 * 255 * 255;
};

class RegularityGuidedPatchDistanceMetricV2 : public PatchDistanceMetric
{
public:
    RegularityGuidedPatchDistanceMetricV2(int patch_size, const cv::Mat &ijmap, double weight)
        : PatchDistanceMetric(patch_size), m_ijmap(ijmap), m_weight(weight)
    {
        // The distance is divided by 1 + weight, so a negative weight flips its sign.
        if (!(weight >= 0))
            throw std::invalid_argument("the guide weight must be >= 0");
        // operator() reads the first two channels at coordinates scaled to the map.
        CV_Assert(!ijmap.empty() && ijmap.depth() == CV_32F && ijmap.channels() >= 2);
    }
    int operator()(
        const MaskedImage &source, int source_y, int source_x, const MaskedImage &target, int target_y,
        int target_x) const override;
    std::unique_ptr<PatchDistanceMetric> with_patch_size(int patch_size) const override
    {
        return std::make_unique<RegularityGuidedPatchDistanceMetricV2>(patch_size, m_ijmap, m_weight);
    }

private:
    cv::Mat m_ijmap;
    double m_weight;
};
