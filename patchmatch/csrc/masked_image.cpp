#include "masked_image.h"

#include <algorithm>
#include <array>
#include <cstdint>

#include "parallel.h"

namespace
{
    constexpr std::array<int, 6> kDownsampleKernel = {1, 5, 10, 10, 5, 1};

    struct KernelSum
    {
        int r = 0;
        int g = 0;
        int b = 0;
        int weight = 0;
        bool globally_masked = true;
    };

    // The known pixels under the downsampling kernel at (y, x), weighted by the kernel.
    KernelSum sum_kernel(const MaskedImage &image, int y, int x)
    {
        constexpr int half = static_cast<int>(kDownsampleKernel.size()) / 2;
        const auto size = image.size();

        KernelSum sum;
        for (int dy = -half + 1; dy <= half; ++dy)
        {
            for (int dx = -half + 1; dx <= half; ++dx)
            {
                const int yy = y + dy;
                const int xx = x + dx;
                if (yy < 0 || yy >= size.height || xx < 0 || xx >= size.width)
                    continue;
                if (image.is_globally_masked(yy, xx))
                    continue;
                sum.globally_masked = false;
                if (image.is_masked(yy, xx))
                    continue;

                const auto *source_ptr = image.get_image(yy, xx);
                const int k = kDownsampleKernel[half - 1 + dy] * kDownsampleKernel[half - 1 + dx];
                sum.r += source_ptr[0] * k;
                sum.g += source_ptr[1] * k;
                sum.b += source_ptr[2] * k;
                sum.weight += k;
            }
        }
        return sum;
    }
} // namespace

void MaskedImage::compute_hole_sums() const
{
    if (m_hole_sums_computed)
        return;
    const auto size = this->size();
    m_hole_sums = cv::Mat(size.height + 1, size.width + 1, CV_32S, cv::Scalar(0));
    for (int y = 0; y < size.height; ++y)
    {
        for (int x = 0; x < size.width; ++x)
        {
            const int hole = is_masked(y, x) && !is_globally_masked(y, x) ? 1 : 0;
            m_hole_sums.at<int>(y + 1, x + 1) =
                hole + m_hole_sums.at<int>(y, x + 1) + m_hole_sums.at<int>(y + 1, x) - m_hole_sums.at<int>(y, x);
        }
    }
    m_hole_sums_computed = true;
}

bool MaskedImage::contains_mask(int y, int x, int patch_size) const
{
    compute_hole_sums();
    const auto size = this->size();
    // In 64 bits: patch_size may be close to the largest int.
    const auto clip = [](std::int64_t value, int limit) {
        return static_cast<int>(std::clamp<std::int64_t>(value, 0, limit));
    };
    const int top = clip(std::int64_t{y} - patch_size, size.height);
    const int bottom = clip(std::int64_t{y} + patch_size + 1, size.height);
    const int left = clip(std::int64_t{x} - patch_size, size.width);
    const int right = clip(std::int64_t{x} + patch_size + 1, size.width);
    const auto sum = [this](int yy, int xx) { return m_hole_sums.at<int>(yy, xx); };
    return sum(bottom, right) - sum(top, right) - sum(bottom, left) + sum(top, left) > 0;
}

MaskedImage MaskedImage::downsample() const
{
    const auto size = this->size();
    auto ret = MaskedImage(size.width / 2, size.height / 2);
    if (m_has_global_mask)
        ret.init_global_mask_mat();
    for (int y = 0; y < size.height - 1; y += 2)
    {
        for (int x = 0; x < size.width - 1; x += 2)
        {
            const auto sum = sum_kernel(*this, y, x);
            if (m_has_global_mask)
                ret.set_global_mask(y / 2, x / 2, sum.globally_masked);
            if (sum.weight == 0)
            {
                ret.set_mask(y / 2, x / 2, true);
                continue;
            }

            auto *target_ptr = ret.get_mutable_image(y / 2, x / 2);
            target_ptr[0] = static_cast<unsigned char>(sum.r / sum.weight);
            target_ptr[1] = static_cast<unsigned char>(sum.g / sum.weight);
            target_ptr[2] = static_cast<unsigned char>(sum.b / sum.weight);
            ret.set_mask(y / 2, x / 2, false);
        }
    }

    return ret;
}

MaskedImage MaskedImage::upsample(int new_w, int new_h) const
{
    const auto size = this->size();
    auto ret = MaskedImage(new_w, new_h);
    if (m_has_global_mask)
        ret.init_global_mask_mat();
    for (int y = 0; y < new_h; ++y)
    {
        for (int x = 0; x < new_w; ++x)
        {
            // In 64 bits: the products overflow an int from 65,537 pixels on.
            const auto yy = static_cast<int>(std::int64_t{y} * size.height / new_h);
            const auto xx = static_cast<int>(std::int64_t{x} * size.width / new_w);

            if (is_globally_masked(yy, xx))
            {
                ret.set_global_mask(y, x, true);
                ret.set_mask(y, x, true);
                continue;
            }
            if (m_has_global_mask)
                ret.set_global_mask(y, x, false);
            if (is_masked(yy, xx))
            {
                ret.set_mask(y, x, true);
                continue;
            }

            std::copy_n(get_image(yy, xx), 3, ret.get_mutable_image(y, x));
            ret.set_mask(y, x, false);
        }
    }

    return ret;
}

MaskedImage MaskedImage::upsample(int new_w, int new_h, const cv::Mat &new_global_mask) const
{
    auto ret = upsample(new_w, new_h);
    ret.set_global_mask_mat(new_global_mask);
    return ret;
}

void MaskedImage::compute_features() const
{
    if (m_features_computed)
    {
        return;
    }

    const auto size = m_image.size();
    m_features = cv::Mat(size, CV_8UC(kFeatureChannels));

    // The border replicates its pixels, so that patches over the outermost line compare it as well.
    parallel_for(size.height, [this, &size](int y) {
        const int up = std::max(y - 1, 0);
        const int down = std::min(y + 1, size.height - 1);
        for (int x = 0; x < size.width; ++x)
        {
            const int left = std::max(x - 1, 0);
            const int right = std::min(x + 1, size.width - 1);
            // A neutral gradient next to a globally masked pixel keeps its color out of the distances.
            const bool skip_y = is_globally_masked(down, x) || is_globally_masked(up, x);
            const bool skip_x = is_globally_masked(y, right) || is_globally_masked(y, left);
            const unsigned char *below = get_image(down, x);
            const unsigned char *above = get_image(up, x);
            const unsigned char *next = get_image(y, right);
            const unsigned char *previous = get_image(y, left);
            auto *features = m_features.ptr<unsigned char>(y, x);
            std::fill_n(features, kFeatureChannels, 0);
            const unsigned char *color = get_image(y, x);
            for (int c = 0; c < 3; ++c)
            {
                features[c] = color[c];
                features[3 + c] = static_cast<unsigned char>(skip_x ? 128 : next[c] / 2 - previous[c] / 2 + 128);
                features[6 + c] = static_cast<unsigned char>(skip_y ? 128 : below[c] / 2 - above[c] / 2 + 128);
            }
            if (m_has_texture)
                features[kTextureFeature] = m_texture.at<unsigned char>(y, x);
            features[kMaskedFeature] = static_cast<unsigned char>(is_masked(y, x) || is_globally_masked(y, x));
        }
    });

    m_features_computed = true;
}
