#pragma once

#include <opencv2/core.hpp>
#include <stdexcept>

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
        _randomize_field(true);
    }
    NearestNeighborField(
        const MaskedImage &source, const MaskedImage &target, const PatchDistanceMetric *metric,
        const NearestNeighborField &other)
        : m_source(source), m_target(target), m_distance_metric(metric)
    {
        m_field = cv::Mat(m_source.size(), CV_32SC3);
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
    }
    void set_target(const MaskedImage &target)
    {
        m_target = target;
    }

    int *mutable_ptr(int y, int x)
    {
        return m_field.ptr<int>(y, x);
    }
    const int *ptr(int y, int x) const
    {
        return m_field.ptr<int>(y, x);
    }

    int at(int y, int x, int c) const
    {
        return m_field.ptr<int>(y, x)[c];
    }
    void set_identity(int y, int x)
    {
        auto ptr = mutable_ptr(y, x);
        ptr[0] = y;
        ptr[1] = x;
        ptr[2] = 0;
    }
    // Measures the link again after an image changed.
    void update_distance(int y, int x)
    {
        auto ptr = mutable_ptr(y, x);
        ptr[2] = _distance(y, x, ptr[0], ptr[1]);
    }

    void minimize(int nr_pass);

    // Seeds the random search of the calling thread.
    static void seed_random(unsigned int seed);

private:
    int _distance(int source_y, int source_x, int target_y, int target_x) const
    {
        return (*m_distance_metric)(m_source, source_y, source_x, m_target, target_y, target_x);
    }
    void _link_if_closer(int y, int x, int y_target, int x_target)
    {
        // In coherent regions most propagated candidates are the link itself, measured in full otherwise.
        if (y_target == at(y, x, 0) && x_target == at(y, x, 1))
            return;
        const int current = at(y, x, 2);
        const int distance = m_distance_metric->distance_below(m_source, y, x, m_target, y_target, x_target, current);
        if (distance < current)
            _set(y, x, y_target, x_target, distance);
    }
    void _set(int y, int x, int y_target, int x_target, int distance)
    {
        auto ptr = mutable_ptr(y, x);
        ptr[0] = y_target;
        ptr[1] = x_target;
        ptr[2] = distance;
    }

    void _randomize_field(bool reset);
    void _randomize_link(int y, int x);
    void _initialize_field_from(const NearestNeighborField &other);
    void _minimize_link(int y, int x, int direction);
    void _random_search(int y, int x);

    MaskedImage m_source;
    MaskedImage m_target;
    // per pixel: the target y and x and the scaled distance to it
    cv::Mat m_field;
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
