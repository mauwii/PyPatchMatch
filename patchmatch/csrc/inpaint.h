#pragma once

#include <vector>

#include "masked_image.h"
#include "nnf.h"

class Inpainting
{
public:
    Inpainting(const cv::Mat &image, const cv::Mat &mask, const PatchDistanceMetric *metric);
    Inpainting(
        const cv::Mat &image, const cv::Mat &mask, const cv::Mat &global_mask, const PatchDistanceMetric *metric);
    cv::Mat run(bool verbose = false, unsigned int random_seed = 1212);

private:
    void _initialize_pyramid(void);
    MaskedImage _expectation_maximization(const MaskedImage &source, MaskedImage target, int level, bool verbose);
    void _update_links(const MaskedImage &source, bool image_changed);
    void _expectation_step(
        const NearestNeighborField &nnf, bool source2target, cv::Mat &vote, const MaskedImage &source, bool upscaled,
        bool best_only, bool keep_known) const;
    void _maximization_step(MaskedImage &target, const cv::Mat &vote, const MaskedImage &source, bool keep_known) const;

    MaskedImage m_initial;
    std::vector<MaskedImage> m_pyramid;

    NearestNeighborField m_source2target;
    NearestNeighborField m_target2source;
    const PatchDistanceMetric *m_distance_metric;
};
