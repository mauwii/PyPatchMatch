#pragma once

#include <algorithm>
#include <atomic>
#include <opencv2/core.hpp>

// The most threads of a parallel loop, 0 for OpenCV's default. PM_set_num_threads sets it while other threads may be
// inpainting.
inline std::atomic<int> &thread_limit()
{
    static std::atomic value{0};
    return value;
}

// Calls body(i) for every i in [0, n) on at most thread_limit() threads, which take the next i when done. Not
// cv::setNumThreads: Grand Central Dispatch (macOS) ignores limits above 1. The results must not depend on the thread.
template <typename Body>
void parallel_for(int n, const Body &body)
{
    if (n <= 0)
        return;
    static const int kDefaultThreads = std::max(1, cv::getNumThreads());
    const int limit = thread_limit().load();
    const int stripes = std::min(n, limit > 0 ? limit : kDefaultThreads);
    std::atomic<int> next{0};
    cv::parallel_for_(
        cv::Range(0, stripes),
        [&next, n, &body](const cv::Range &) {
            for (int i = next++; i < n; i = next++)
                body(i);
        },
        stripes);
}
