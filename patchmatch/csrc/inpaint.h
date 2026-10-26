#pragma once

#include <memory>
#include <optional>
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
    MaskedImage _run_levels(bool verbose);
    bool _peels(int level) const;
    MaskedImage _peel(int level, const MaskedImage &target, cv::Mat &matches) const;
    void _link_peeled(const cv::Mat &matches);
    MaskedImage _expectation_maximization(const MaskedImage &source, MaskedImage target, int level, bool verbose);
    void _update_links(const MaskedImage &source, bool image_changed);
    void _expectation_step(
        const NearestNeighborField &nnf, bool source2target, cv::Mat &vote, const MaskedImage &source, bool upscaled,
        bool best_only, bool keep_known) const;
    void _maximization_step(MaskedImage &target, const cv::Mat &vote, const MaskedImage &source, bool keep_known) const;

    MaskedImage m_initial;
    std::vector<MaskedImage> m_pyramid;

    std::optional<NearestNeighborField> m_source2target;
    NearestNeighborField m_target2source;
    const PatchDistanceMetric *m_distance_metric;
    // Searches on the coarsest level with a hole, if patch_size is smaller than kCoarsePatchSize.
    std::unique_ptr<PatchDistanceMetric> m_coarse_metric;
    // Per level: smaller patches where patch_size does not fit, otherwise nullptr.
    std::vector<std::unique_ptr<PatchDistanceMetric>> m_fitting_metrics;
    int m_coarsest_hole_level = -1;
    // Per level the labels of the holes that the peel fills, or an empty matrix; none from kCoarsePatchSize on.
    std::vector<cv::Mat> m_peel_labels;
    // Searches on the current level; the votes use the smaller of its patch size and patch_size.
    const PatchDistanceMetric *m_level_metric = nullptr;
};
