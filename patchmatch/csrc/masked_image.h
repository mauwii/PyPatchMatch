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

    // Without the gradients, which would be outdated once the clone is changed. It shares the texture.
    MaskedImage clone() const
    {
        MaskedImage copy(m_image.clone(), m_mask.clone(), m_global_mask.clone());
        copy.set_texture(m_texture);
        return copy;
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
    // Per pixel the colors, the gradients along x and y, the texture and whether it is masked, packed so that
    // a distance reads one place per pixel.
    static constexpr int kFeatureChannels = 12;
    static constexpr int kColorAndGradientFeatures = 9;
    static constexpr int kTextureFeature = 9;
    static constexpr int kMaskedFeature = 10;
    const cv::Mat &features() const
    {
        assert(m_features_computed);
        return m_features;
    }

    // The texture energy that the distances compare (CV_8UC1, see Inpainting), or empty.
    // It is never changed, and the resampling does not carry it over.
    const cv::Mat &texture() const
    {
        return m_texture;
    }
    bool has_texture() const
    {
        return m_has_texture;
    }
    void set_texture(const cv::Mat &texture)
    {
        m_texture = texture;
        m_has_texture = !texture.empty();
        m_features_computed = false;
    }

    void init_global_mask_mat()
    {
        m_global_mask = cv::Mat(m_mask.size(), CV_8U);
        m_global_mask.setTo(cv::Scalar(0));
        m_has_global_mask = !m_global_mask.empty();
        m_hole_sums_computed = false;
        m_features_computed = false;
    }
    void set_global_mask_mat(const cv::Mat &other)
    {
        m_global_mask = other;
        m_has_global_mask = !other.empty();
        m_hole_sums_computed = false;
        m_features_computed = false;
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
        m_hole_sums_computed = false;
        m_features_computed = false;
    }
    void set_global_mask(int y, int x, bool value)
    {
        m_global_mask.at<unsigned char>(y, x) = static_cast<unsigned char>(value);
        m_hole_sums_computed = false;
        m_features_computed = false;
    }
    void clear_mask()
    {
        m_mask.setTo(cv::Scalar(0));
        m_hole_sums_computed = false;
        m_features_computed = false;
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
    // The caches that the distances and contains_mask compute on first use; a parallel loop needs them computed.
    void compute_features() const;
    void compute_hole_sums() const;

private:
    void _compute_features_at(int y, int x) const;

    cv::Mat m_image;
    cv::Mat m_mask;
    cv::Mat m_global_mask;
    cv::Mat m_texture;
    // cv::Mat::empty() is not inline in OpenCV 5, too slow to call for every pixel
    bool m_has_global_mask = false;
    bool m_has_texture = false;
    // computed on first use by compute_features
    mutable cv::Mat m_features;
    mutable bool m_features_computed = false;
    // computed on first use by compute_hole_sums: the masked pixels outside the global mask
    // above and left of each pixel. Copies share them with the masks, so change a mask only
    // through an image that nothing has copied yet, as downsample and upsample do.
    mutable cv::Mat m_hole_sums;
    mutable bool m_hole_sums_computed = false;
};
