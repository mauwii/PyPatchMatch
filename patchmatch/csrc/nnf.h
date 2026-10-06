#pragma once

#include <algorithm>
#include <cstdint>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <vector>

#include "masked_image.h"

class PatchDistanceMetric
{
public:
    explicit PatchDistanceMetric(int patch_size) : m_patch_size(patch_size) {}
    virtual ~PatchDistanceMetric() = default;

    int patch_size() const
    {
        return m_patch_size;
    }
    virtual int operator()(
        const MaskedImage &source, int source_y, int source_x, const MaskedImage &target, int target_y,
        int target_x) const = 0;
    // The distance if it is below bound, otherwise any value >= bound.
    virtual int distance_below(
        const MaskedImage &source, int source_y, int source_x, const MaskedImage &target, int target_y, int target_x,
        int /* bound */) const
    {
        return (*this)(source, source_y, source_x, target, target_y, target_x);
    }
    static const int kDistanceScale;

private:
    int m_patch_size;
};

class NearestNeighborField
{
public:
    NearestNeighborField() = default;
    NearestNeighborField(const MaskedImage &source, const MaskedImage &target, const PatchDistanceMetric *metric)
        : m_source(source), m_target(target), m_distance_metric(metric)
    {
        m_field = cv::Mat(m_source.size(), CV_32SC3);
        _allocate_sums();
        _randomize_field(true);
    }
    NearestNeighborField(
        const MaskedImage &source, const MaskedImage &target, const PatchDistanceMetric *metric,
        const NearestNeighborField &other)
        : m_source(source), m_target(target), m_distance_metric(metric)
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
    // The distance 0 is not measured, so the sum stays unknown.
    void set_identity(int y, int x)
    {
        _set(y, x, y, x, {0, -1});
    }
    // Measures the link again after an image changed.
    void update_distance(int y, int x)
    {
        _set(
            y, x, at(y, x, 0), at(y, x, 1),
            _measure(y, x, at(y, x, 0), at(y, x, 1), PatchDistanceMetric::kDistanceScale));
    }

    void minimize(int nr_pass);

    // Seeds the random search of the calling thread.
    static void seed_random(unsigned int seed);

private:
    // Private: a link written from outside would not update its sum.
    int *mutable_ptr(int y, int x)
    {
        return m_field.ptr<int>(y, x);
    }

    // A distance and the sum of the pixel costs behind it, -1 if unknown.
    struct Measured
    {
        int distance;
        std::int64_t sum;
    };

    // The distance if it is below bound, otherwise any value >= bound.
    Measured _measure(int y, int x, int y_target, int x_target, int bound) const;
    void _link_if_closer(int y, int x, int y_target, int x_target)
    {
        // In coherent regions most propagated candidates are the link itself, measured in full otherwise.
        if (y_target == at(y, x, 0) && x_target == at(y, x, 1))
            return;
        const int current = at(y, x, 2);
        const Measured measured = _measure(y, x, y_target, x_target, current);
        if (measured.distance < current)
            _set(y, x, y_target, x_target, measured);
    }
    void _set(int y, int x, int y_target, int x_target, Measured measured)
    {
        auto ptr = mutable_ptr(y, x);
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
    void _allocate_sums();
    void _minimize_link(int y, int x, int direction);
    void _propagate(int y, int x, int dy, int dx);
    void _random_search(int y, int x);

    MaskedImage m_source;
    MaskedImage m_target;
    // per pixel: the target y and x and the scaled distance to it
    cv::Mat m_field;
    // per pixel: the sum behind the distance, for updates in O(patch_size); empty for other metrics
    std::vector<std::int64_t> m_sums;
    const PatchDistanceMetric *m_distance_metric = nullptr;
};

class PatchSSDDistanceMetric : public PatchDistanceMetric
{
public:
    using PatchDistanceMetric::PatchDistanceMetric;
    int operator()(
        const MaskedImage &source, int source_y, int source_x, const MaskedImage &target, int target_y,
        int target_x) const override;
    int distance_below(
        const MaskedImage &source, int source_y, int source_x, const MaskedImage &target, int target_y, int target_x,
        int bound) const override;
    static const int kSSDScale;
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
    }
    int operator()(
        const MaskedImage &source, int source_y, int source_x, const MaskedImage &target, int target_y,
        int target_x) const override;

private:
    cv::Mat m_ijmap;
    double m_weight;
};
