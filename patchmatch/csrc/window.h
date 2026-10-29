#pragma once

#include <algorithm>
#include <opencv2/core.hpp>

#include "masked_image.h"

// Calls visit(y, x) for the pixels of the image within radius of (cy, cx).
template <typename Visit>
void for_window(cv::Size size, int cy, int cx, int radius, Visit visit)
{
    for (int y = std::max(0, cy - radius); y <= std::min(size.height - 1, cy + radius); ++y)
        for (int x = std::max(0, cx - radius); x <= std::min(size.width - 1, cx + radius); ++x)
            visit(y, x);
}

// Mean of value(y, x) over the pixels within radius of (cy, cx) for which select(y, x) holds.
template <typename Select, typename Value>
cv::Vec3d window_mean(cv::Size size, int cy, int cx, int radius, Select select, Value value)
{
    cv::Vec3d sum;
    int count = 0;
    for_window(size, cy, cx, radius, [&sum, &count, &select, &value](int y, int x) {
        if (select(y, x))
        {
            sum += value(y, x);
            ++count;
        }
    });
    return count > 0 ? sum / count : sum;
}

// Variance of the colors of image within radius of (cy, cx) where select(y, x) holds, which it does at least once.
template <typename Select>
double window_variance(const cv::Mat &image, int cy, int cx, int radius, Select select)
{
    cv::Vec3d sum;
    double squares = 0;
    int count = 0;
    for_window(image.size(), cy, cx, radius, [&image, &select, &sum, &squares, &count](int y, int x) {
        if (!select(y, x))
            return;
        const auto &color = image.at<cv::Vec3b>(y, x);
        for (int c = 0; c < 3; ++c)
        {
            sum[c] += color[c];
            squares += static_cast<double>(color[c]) * color[c];
        }
        ++count;
    });
    return (squares - sum.dot(sum) / count) / count;
}

inline bool is_known(const MaskedImage &image, int y, int x)
{
    return !image.is_masked(y, x) && !image.is_globally_masked(y, x);
}

inline bool is_hole(const MaskedImage &image, int y, int x)
{
    return image.is_masked(y, x) && !image.is_globally_masked(y, x);
}
