#include "holes.h"

#include <algorithm>
#include <cstddef>
#include <utility>

#include "window.h"

namespace
{
    // Gives the hole pixels 8-connected to (y, x), the neighborhood of the peel's front, the label.
    void label_hole(const MaskedImage &image, cv::Mat &labels, int y, int x, int label)
    {
        labels.at<int>(y, x) = label;
        std::vector<cv::Point> stack{{x, y}};
        while (!stack.empty())
        {
            const cv::Point p = stack.back();
            stack.pop_back();
            for_window(image.size(), p.y, p.x, 1, [&image, &labels, &stack, label](int yy, int xx) {
                if (is_hole(image, yy, xx) && labels.at<int>(yy, xx) == 0)
                {
                    labels.at<int>(yy, xx) = label;
                    stack.emplace_back(xx, yy);
                }
            });
        }
    }

    // Per pixel the label of its hole from 1 on, 0 outside the holes.
    cv::Mat label_holes(const MaskedImage &image, int &count)
    {
        cv::Mat labels(image.size(), CV_32S, cv::Scalar(0));
        count = 0;
        for (int y = 0; y < labels.rows; ++y)
        {
            for (int x = 0; x < labels.cols; ++x)
            {
                if (is_hole(image, y, x) && labels.at<int>(y, x) == 0)
                    label_hole(image, labels, y, x, ++count);
            }
        }
        return labels;
    }

    // Raises the reach of the labeled holes in the window to top.
    void raise_reach(const cv::Mat &labels, const cv::Rect &window, int top, std::vector<int> &reach)
    {
        for (int y = window.y; y < window.y + window.height; ++y)
        {
            for (int x = window.x; x < window.x + window.width; ++x)
            {
                auto &hole = reach[static_cast<std::size_t>(labels.at<int>(y, x))];
                hole = std::max(hole, top);
            }
        }
    }

    // Raises the reach of each labeled hole to that of the holes of the coarser level that it averages into.
    void reach_coarser(const cv::Mat &coarser_reach, const cv::Mat &labels, std::vector<int> &reach)
    {
        const cv::Rect image({}, labels.size());
        for (int y = 0; y < coarser_reach.rows; ++y)
        {
            for (int x = 0; x < coarser_reach.cols; ++x)
            {
                if (const int top = coarser_reach.at<int>(y, x); top >= 0)
                    raise_reach(labels, downsample_window(y, x) & image, top, reach);
            }
        }
    }

    // The labels of the holes that reach at most one level coarser than level and between least and most, 0 elsewhere,
    // or an empty matrix; and per pixel the reach of its hole, -1 outside the holes.
    cv::Mat peeled_holes(
        const cv::Mat &labels, const std::vector<int> &reach, int level, int least, int most, cv::Mat &pixel_reach)
    {
        pixel_reach = cv::Mat(labels.size(), CV_32S, cv::Scalar(-1));
        cv::Mat peeled(labels.size(), CV_32S, cv::Scalar(0));
        bool peels = false;
        for (int y = 0; y < labels.rows; ++y)
        {
            for (int x = 0; x < labels.cols; ++x)
            {
                const int label = labels.at<int>(y, x);
                if (label == 0)
                    continue;
                const int top = reach[static_cast<std::size_t>(label)];
                pixel_reach.at<int>(y, x) = top;
                if (top <= level + 1 && top >= least && top <= most)
                {
                    peeled.at<int>(y, x) = label;
                    peels = true;
                }
            }
        }
        return peels ? peeled : cv::Mat();
    }
} // namespace

std::vector<cv::Mat> peel_labels(
    const std::vector<MaskedImage> &pyramid, int least, int most, std::vector<std::vector<int>> &reaches)
{
    std::vector<cv::Mat> peel(pyramid.size());
    reaches.resize(pyramid.size());
    cv::Mat coarser_reach; // per pixel the coarsest level that its hole reaches, -1 outside the holes
    for (int level = static_cast<int>(pyramid.size()) - 1; level >= 0; --level)
    {
        int count = 0;
        const cv::Mat labels = label_holes(pyramid[level], count);
        // reach[0] belongs to the globally masked pixels in the windows
        std::vector reach(static_cast<std::size_t>(count) + 1, level);
        reach_coarser(coarser_reach, labels, reach);
        peel[static_cast<std::size_t>(level)] = peeled_holes(labels, reach, level, least, most, coarser_reach);
        reaches[static_cast<std::size_t>(level)] = std::move(reach);
    }
    return peel;
}

cv::Mat pixels_around(
    const MaskedImage &image, const cv::Mat &peeled, const std::vector<int> &reach, int top, int radius)
{
    if (peeled.empty())
        return {};
    const auto reaches_top = [&peeled, &reach, top](int y, int x) {
        const int label = peeled.at<int>(y, x);
        return label > 0 && reach[static_cast<std::size_t>(label)] == top;
    };
    cv::Mat pixels(image.size(), CV_8U, cv::Scalar(0));
    bool any = false;
    for (int y = 0; y < pixels.rows; ++y)
    {
        for (int x = 0; x < pixels.cols; ++x)
        {
            if (!reaches_top(y, x))
                continue;
            any = true;
            for_window(image.size(), y, x, radius, [&pixels](int yy, int xx) { pixels.at<unsigned char>(yy, xx) = 1; });
        }
    }
    for (int y = 0; y < pixels.rows; ++y)
    {
        for (int x = 0; x < pixels.cols; ++x)
        {
            if (is_hole(image, y, x) && !reaches_top(y, x))
                pixels.at<unsigned char>(y, x) = 0;
        }
    }
    return any ? pixels : cv::Mat();
}

cv::Mat coarse_pixels(const MaskedImage &image, const cv::Mat &peeled, const std::vector<int> &reach, int level)
{
    // reach[0] belongs to no hole
    if (level == 0 && std::any_of(reach.begin() + 1, reach.end(), [](int top) { return top > 0; }))
        return {};
    return pixels_around(image, peeled, reach, level, kCoarsePatchSize);
}

cv::Mat around_lines(const cv::Mat &pixels, const cv::Mat &lines, int radius)
{
    if (pixels.empty() || lines.empty())
        return {};
    cv::Mat around(pixels.size(), CV_8U, cv::Scalar(0));
    bool any = false;
    for (int y = 0; y < lines.rows; ++y)
    {
        for (int x = 0; x < lines.cols; ++x)
        {
            if (!lines.at<unsigned char>(y, x))
                continue;
            for_window(pixels.size(), y, x, radius, [&around, &pixels, &any](int yy, int xx) {
                around.at<unsigned char>(yy, xx) = pixels.at<unsigned char>(yy, xx);
                any = any || pixels.at<unsigned char>(yy, xx);
            });
        }
    }
    return any ? around : cv::Mat();
}
