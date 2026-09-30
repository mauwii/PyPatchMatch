#include "masked_image.h"

#include <algorithm>

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
    if (!m_global_mask.empty())
        ret.init_global_mask_mat();
    for (int y = 0; y < size.height - 1; y += 2)
    {
        for (int x = 0; x < size.width - 1; x += 2)
        {
            const auto sum = sum_kernel(*this, y, x);
            if (!m_global_mask.empty())
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
    if (!m_global_mask.empty())
        ret.init_global_mask_mat();
    for (int y = 0; y < new_h; ++y)
    {
        for (int x = 0; x < new_w; ++x)
        {
            const int yy = y * size.height / new_h;
            const int xx = x * size.width / new_w;

            if (is_globally_masked(yy, xx))
            {
                ret.set_global_mask(y, x, true);
                ret.set_mask(y, x, true);
                continue;
            }
            if (!m_global_mask.empty())
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
    m_image_grady = cv::Scalar::all(0);
    m_image_gradx = cv::Scalar::all(0);

    for (int i = 1; i < size.height - 1; ++i)
    {
        const auto *ptry1 = m_image.ptr<unsigned char>(i + 1, 0);
        const auto *ptry2 = m_image.ptr<unsigned char>(i - 1, 0);
        const auto *ptrx1 = m_image.ptr<unsigned char>(i, 0) + 3;
        const auto *ptrx2 = m_image.ptr<unsigned char>(i, 0) - 3;
        auto *mptry = m_image_grady.ptr<unsigned char>(i, 0);
        auto *mptrx = m_image_gradx.ptr<unsigned char>(i, 0);
        for (int j = 3; j < size.width * 3 - 3; ++j)
        {
            // A neutral gradient next to a globally masked pixel keeps its color out of the distances.
            const int x = j / 3;
            const bool skip_y = is_globally_masked(i + 1, x) || is_globally_masked(i - 1, x);
            const bool skip_x = is_globally_masked(i, x + 1) || is_globally_masked(i, x - 1);
            mptry[j] = skip_y ? 128 : (ptry1[j] / 2 - ptry2[j] / 2) + 128;
            mptrx[j] = skip_x ? 128 : (ptrx1[j] / 2 - ptrx2[j] / 2) + 128;
        }
    }

    m_image_grad_computed = true;
}
