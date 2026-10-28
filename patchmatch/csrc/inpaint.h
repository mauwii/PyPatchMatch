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
    cv::Mat _hole_pixels(int level) const;
    MaskedImage _peel(int level, const MaskedImage &target, cv::Mat &matches, cv::Mat &lines) const;
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
    // Searches around each hole on the coarsest level that it reaches with kCoarsePatchSize if patch_size is smaller,
    // otherwise around the lines that the peel continues with kLinePatchSize.
    std::unique_ptr<PatchDistanceMetric> m_hole_metric;
    // Per level the pixels that search with it (CV_8U), or an empty matrix; from kCoarsePatchSize on around the holes
    // whose first fill is on the level, which the peel narrows to its lines.
    std::vector<cv::Mat> m_hole_pixels;
    // Per level: smaller patches where patch_size does not fit, otherwise nullptr.
    std::vector<std::unique_ptr<PatchDistanceMetric>> m_fitting_metrics;
    // Per level the labels of the holes that the peel fills, or an empty matrix.
    std::vector<cv::Mat> m_peel_labels;
    // Per level and label the coarsest level that the hole reaches.
    std::vector<std::vector<int>> m_peel_reach;
    // Searches on the current level; the votes use the smaller of its patch size and patch_size.
    const PatchDistanceMetric *m_level_metric = nullptr;
};
