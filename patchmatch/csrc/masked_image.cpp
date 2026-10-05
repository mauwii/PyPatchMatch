#include "masked_image.h"

#include <algorithm>
#include <cstdint>

const cv::Size MaskedImage::kDownsampleKernelSize = cv::Size(6, 6);
const std::array<int, 6> MaskedImage::kDownsampleKernel = {1, 5, 10, 10, 5, 1};

namespace
{
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
        const auto &kernel_size = MaskedImage::kDownsampleKernelSize;
        const auto &kernel = MaskedImage::kDownsampleKernel;
        const auto size = image.size();

        KernelSum sum;
        for (int dy = -kernel_size.height / 2 + 1; dy <= kernel_size.height / 2; ++dy)
        {
            for (int dx = -kernel_size.width / 2 + 1; dx <= kernel_size.width / 2; ++dx)
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
                const int k = kernel[kernel_size.height / 2 - 1 + dy] * kernel[kernel_size.width / 2 - 1 + dx];
                sum.r += source_ptr[0] * k;
                sum.g += source_ptr[1] * k;
                sum.b += source_ptr[2] * k;
                sum.weight += k;
            }
        }
        return sum;
    }
} // namespace

bool MaskedImage::contains_mask(int y, int x, int patch_size) const
{
    auto mask_size = size();
    for (int dy = -patch_size; dy <= patch_size; ++dy)
    {
        for (int dx = -patch_size; dx <= patch_size; ++dx)
        {
            const int yy = y + dy;
            const int xx = x + dx;
            if (yy >= 0 && yy < mask_size.height && xx >= 0 && xx < mask_size.width && is_masked(yy, xx) &&
                !is_globally_masked(yy, xx))
                return true;
        }
    }
    return false;
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

void MaskedImage::compute_image_gradients() const
{
    if (m_image_grad_computed)
    {
        return;
    }

    const auto size = m_image.size();
    m_image_grady = cv::Mat(size, CV_8UC3);
    m_image_gradx = cv::Mat(size, CV_8UC3);

    // The border replicates its pixels, so that patches over the outermost line compare it as well.
    for (int y = 0; y < size.height; ++y)
    {
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
            auto *grady = m_image_grady.ptr<unsigned char>(y, x);
            auto *gradx = m_image_gradx.ptr<unsigned char>(y, x);
            for (int c = 0; c < 3; ++c)
            {
                grady[c] = static_cast<unsigned char>(skip_y ? 128 : below[c] / 2 - above[c] / 2 + 128);
                gradx[c] = static_cast<unsigned char>(skip_x ? 128 : next[c] / 2 - previous[c] / 2 + 128);
            }
        }
    }

    m_image_grad_computed = true;
}
