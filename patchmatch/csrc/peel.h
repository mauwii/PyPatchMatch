#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "masked_image.h"
#include "nnf.h"

struct PeelResult
{
    MaskedImage image; // the target with the peeled holes
    cv::Mat matches;   // per peeled pixel the center of its source patch, -1 elsewhere (CV_32SC2)
    cv::Mat lines;     // the peeled pixels where the coarser fill lost a line (CV_8U)
};

// Fills the holes of a level from their border inwards, see Peel in peel.cpp. holes: the labels of the holes to fill
// (CV_32S), 0 elsewhere; reach: per label the coarsest level that the hole reaches; guide: the field of this level that
// the search would start from, nullptr on the coarsest level.
PeelResult peel_holes(
    const MaskedImage &source, const MaskedImage &target, const cv::Mat &holes, const std::vector<int> &reach,
    const NearestNeighborField *guide, int patch_size, int level);
