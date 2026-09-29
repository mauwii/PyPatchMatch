#pragma once

#include <array>
#include <cassert>
#include <opencv2/core.hpp>

class MaskedImage
{
public:
    MaskedImage() = default;
    MaskedImage(const cv::Mat &image, const cv::Mat &mask) : m_image(image), m_mask(mask) {}
    MaskedImage(const cv::Mat &image, const cv::Mat &mask, const cv::Mat &global_mask)
        : m_image(image), m_mask(mask), m_global_mask(global_mask)
    {
    }
    MaskedImage(
        const cv::Mat &image, const cv::Mat &mask, const cv::Mat &global_mask, const cv::Mat &grady,
        const cv::Mat &gradx, bool grad_computed)
        : m_image(image), m_mask(mask), m_global_mask(global_mask), m_image_grady(grady), m_image_gradx(gradx),
          m_image_grad_computed(grad_computed)
    {
    }
    MaskedImage(int width, int height)
        : m_image(cv::Size(width, height), CV_8UC3, cv::Scalar::all(0)),
          m_mask(cv::Size(width, height), CV_8U, cv::Scalar::all(0))
    {
    }
    // cv::Mat's move assignment is not noexcept, so the implicit one would not be
    MaskedImage(const MaskedImage &) = default;
    MaskedImage(MaskedImage &&) noexcept = default;
    MaskedImage &operator=(const MaskedImage &) = default;
    MaskedImage &operator=(MaskedImage &&) noexcept = default;
    ~MaskedImage() = default;

    inline MaskedImage clone() const
    {
        return MaskedImage(
            m_image.clone(), m_mask.clone(), m_global_mask.clone(), m_image_grady.clone(), m_image_gradx.clone(),
            m_image_grad_computed);
    }

    inline cv::Size size() const
    {
        return m_image.size();
    }
    inline const cv::Mat &image() const
    {
        return m_image;
    }
    inline const cv::Mat &mask() const
    {
        return m_mask;
    }
    inline const cv::Mat &global_mask() const
    {
        return m_global_mask;
    }
    inline const cv::Mat &grady() const
    {
        assert(m_image_grad_computed);
        return m_image_grady;
    }
    inline const cv::Mat &gradx() const
    {
        assert(m_image_grad_computed);
        return m_image_gradx;
    }

    inline void init_global_mask_mat()
    {
        m_global_mask = cv::Mat(m_mask.size(), CV_8U);
        m_global_mask.setTo(cv::Scalar(0));
    }
    inline void set_global_mask_mat(const cv::Mat &other)
    {
        m_global_mask = other;
    }

    inline bool is_masked(int y, int x) const
    {
        return static_cast<bool>(m_mask.at<unsigned char>(y, x));
    }
    inline bool is_globally_masked(int y, int x) const
    {
        return !m_global_mask.empty() && static_cast<bool>(m_global_mask.at<unsigned char>(y, x));
    }
    inline void set_mask(int y, int x, bool value)
    {
        m_mask.at<unsigned char>(y, x) = static_cast<unsigned char>(value);
    }
    inline void set_global_mask(int y, int x, bool value)
    {
        m_global_mask.at<unsigned char>(y, x) = static_cast<unsigned char>(value);
    }
    inline void clear_mask()
    {
        m_mask.setTo(cv::Scalar(0));
    }

    inline const unsigned char *get_image(int y, int x) const
    {
        return m_image.ptr<unsigned char>(y, x);
    }
    inline unsigned char *get_mutable_image(int y, int x)
    {
        return m_image.ptr<unsigned char>(y, x);
    }

    bool contains_mask(int y, int x, int patch_size) const;
    MaskedImage downsample() const;
    MaskedImage upsample(int new_w, int new_h) const;
    MaskedImage upsample(int new_w, int new_h, const cv::Mat &new_global_mask) const;
    void compute_image_gradients() const;

    static const cv::Size kDownsampleKernelSize;
    static const std::array<int, 6> kDownsampleKernel;

private:
    cv::Mat m_image;
    cv::Mat m_mask;
    cv::Mat m_global_mask;
    // computed on first use by compute_image_gradients
    mutable cv::Mat m_image_grady;
    mutable cv::Mat m_image_gradx;
    mutable bool m_image_grad_computed = false;
};
