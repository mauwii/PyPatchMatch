#pragma once

#include <cassert>
#include <opencv2/core.hpp>

class MaskedImage
{
public:
    MaskedImage() = default;
    MaskedImage(const cv::Mat &image, const cv::Mat &mask) : m_image(image), m_mask(mask) {}
    MaskedImage(const cv::Mat &image, const cv::Mat &mask, const cv::Mat &global_mask)
        : m_image(image), m_mask(mask), m_global_mask(global_mask), m_has_global_mask(!global_mask.empty())
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

    // Without the gradients, which would be outdated once the clone is changed.
    MaskedImage clone() const
    {
        return MaskedImage(m_image.clone(), m_mask.clone(), m_global_mask.clone());
    }

    cv::Size size() const
    {
        return m_image.size();
    }
    const cv::Mat &image() const
    {
        return m_image;
    }
    const cv::Mat &mask() const
    {
        return m_mask;
    }
    const cv::Mat &global_mask() const
    {
        return m_global_mask;
    }
    bool has_global_mask() const
    {
        return m_has_global_mask;
    }
    const cv::Mat &grady() const
    {
        assert(m_image_grad_computed);
        return m_image_grady;
    }
    const cv::Mat &gradx() const
    {
        assert(m_image_grad_computed);
        return m_image_gradx;
    }

    void init_global_mask_mat()
    {
        m_global_mask = cv::Mat(m_mask.size(), CV_8U);
        m_global_mask.setTo(cv::Scalar(0));
        m_has_global_mask = !m_global_mask.empty();
    }
    void set_global_mask_mat(const cv::Mat &other)
    {
        m_global_mask = other;
        m_has_global_mask = !other.empty();
    }

    bool is_masked(int y, int x) const
    {
        return static_cast<bool>(m_mask.at<unsigned char>(y, x));
    }
    bool is_globally_masked(int y, int x) const
    {
        return m_has_global_mask && static_cast<bool>(m_global_mask.at<unsigned char>(y, x));
    }
    void set_mask(int y, int x, bool value)
    {
        m_mask.at<unsigned char>(y, x) = static_cast<unsigned char>(value);
    }
    void set_global_mask(int y, int x, bool value)
    {
        m_global_mask.at<unsigned char>(y, x) = static_cast<unsigned char>(value);
    }
    void clear_mask()
    {
        m_mask.setTo(cv::Scalar(0));
    }

    const unsigned char *get_image(int y, int x) const
    {
        return m_image.ptr<unsigned char>(y, x);
    }
    unsigned char *get_mutable_image(int y, int x)
    {
        return m_image.ptr<unsigned char>(y, x);
    }

    bool contains_mask(int y, int x, int patch_size) const;
    MaskedImage downsample() const;
    MaskedImage upsample(int new_w, int new_h) const;
    MaskedImage upsample(int new_w, int new_h, const cv::Mat &new_global_mask) const;
    void compute_image_gradients() const;

private:
    cv::Mat m_image;
    cv::Mat m_mask;
    cv::Mat m_global_mask;
    // cv::Mat::empty() is not inline in OpenCV 5, too slow to call for every pixel
    bool m_has_global_mask = false;
    // computed on first use by compute_image_gradients
    mutable cv::Mat m_image_grady;
    mutable cv::Mat m_image_gradx;
    mutable bool m_image_grad_computed = false;
};
