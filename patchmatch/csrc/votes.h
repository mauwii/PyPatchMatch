#pragma once

#include <opencv2/core.hpp>

#include "masked_image.h"
#include "nnf.h"

// A split vote: the weighted low frequencies and their weight, the detail of the best vote and its weight,
// and the weighted texture of the source pixels.
inline constexpr int kSplitChannels = 9;

// The known texture, and in the holes twice the mean texture of the pixels that the split votes came from:
// the distances charge a deficit below half of it, so a fill expects the full texture of its sources.
cv::Mat voted_texture(const MaskedImage &source, const cv::Mat &vote);

// Expectation step: casts the votes of the patches of patch_size of a field into vote.
void cast_votes(
    const NearestNeighborField &nnf, bool source2target, cv::Mat &vote, const MaskedImage &source, bool upscaled,
    bool best_only, bool keep_known, int patch_size);

// Maximization step: the holes of target, and the known pixels unless keep_known, take their votes.
void maximize_votes(MaskedImage &target, const cv::Mat &vote, const MaskedImage &source, bool keep_known);
