#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "masked_image.h"
#include "nnf.h"

// Fills the holes of a level from their border inwards, see Peel in peel.cpp. holes: the labels of the holes to fill
// (CV_32S), 0 elsewhere; reach: per label the coarsest level that the hole reaches; guide: the field of this level that
// the search would start from, nullptr on the coarsest level. Returns the target with the peeled holes, and sets
// matches to the center of the source patch per peeled pixel, -1 elsewhere, and lines to the peeled pixels where the
// coarser fill lost a line (CV_8U).
MaskedImage peel_holes(
    const MaskedImage &source, const MaskedImage &target, const cv::Mat &holes, const std::vector<int> &reach,
    const NearestNeighborField *guide, int patch_size, int level, cv::Mat &matches, cv::Mat &lines);
