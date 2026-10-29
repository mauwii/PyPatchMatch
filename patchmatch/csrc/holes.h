#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "masked_image.h"

// A patch smaller than a structure cannot tell where it continues: on the coarsest level
// that a hole reaches, holes wider than the patches were filled from several shifted copies.
inline constexpr int kCoarsePatchSize = 5;

// From kCoarsePatchSize on, the peel continues lines with these patches, and the pixels around the lines search and
// vote with them as well: the larger patches of the level wiped the lines out again.
inline constexpr int kLinePatchSize = 3;
// From kCoarsePatchSize on, only the holes whose first fill is on level 1 or 0 are peeled, where the coarser fill
// loses the lines: the peel of larger holes made bricks worse, and a hole that reaches level 0 only has no coarser
// fill.
inline constexpr int kLeastLineReach = 1;
inline constexpr int kMostLineReach = 2;

// Per level the labels of the holes that the peel fills there, 0 elsewhere, or an empty matrix: each hole on the
// coarsest level that it reaches and on the one below. On the levels of the largest hole only, smaller holes
// elsewhere broke their lines off. Only the holes that reach between least and most. reaches: per level and label
// the coarsest level that the hole reaches.
std::vector<cv::Mat> peel_labels(
    const std::vector<MaskedImage> &pyramid, int least, int most, std::vector<std::vector<int>> &reaches);

// The pixels whose patches of radius overlap a hole that reaches top, outside the holes that reach another level,
// or an empty matrix. peeled and reach as in peel_labels.
cv::Mat pixels_around(
    const MaskedImage &image, const cv::Mat &peeled, const std::vector<int> &reach, int top, int radius);

// The pixels whose larger patches overlap a hole on the coarsest level that it reaches, outside the holes that
// reach a coarser one, or an empty matrix. On the coarsest level of the largest hole only, the smaller holes
// elsewhere searched with patch_size, unlike alone, and broke more lines. On level 0 only if no hole reaches a
// coarser one: beside larger holes, the scratches of camera got a worse seam with them and took longer.
// peeled and reach as in peel_labels.
cv::Mat coarse_pixels(const MaskedImage &image, const cv::Mat &peeled, const std::vector<int> &reach, int level);

// Those of pixels within radius of the lines (CV_8U), or an empty matrix. Only around them: patches of
// kLinePatchSize everywhere around the holes moved the edges of regions, such as a hedge above a meadow, and lost
// the short joints of bricks.
cv::Mat around_lines(const cv::Mat &pixels, const cv::Mat &lines, int radius);
