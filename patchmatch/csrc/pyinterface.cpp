#include "pyinterface.h"

#include <atomic>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

#include "inpaint.h"

int _dtype_py_to_cv(PM_dtype_e dtype_py);
cv::Mat _py_to_cv2(PM_mat_t pymat);
PM_mat_t _cv2_to_py(cv::Mat cvmat);

namespace
{
    // The settings are set by one thread while others may be inpainting, since ctypes
    // releases the GIL.
    std::atomic<unsigned int> &random_seed()
    {
        static std::atomic value{1212u};
        return value;
    }

    std::atomic<bool> &verbose()
    {
        static std::atomic value{false};
        return value;
    }

    // Message of the last failed call on the calling thread, see PM_last_error.
    std::string &last_error()
    {
        thread_local std::string message;
        return message;
    }

    void set_last_error(const char *message) noexcept
    {
        try
        {
            last_error() = message;
        }
        catch (...)
        {
            last_error().clear();
        }
    }

    // A C++ exception must not propagate through the C interface into ctypes, which
    // would abort the process. Report it with an empty result instead.
    template <typename Func>
    PM_mat_t guarded(Func &&func) noexcept
    {
        try
        {
            return func();
        }
        catch (const std::exception &e)
        {
            set_last_error(e.what());
        }
        catch (...)
        {
            set_last_error("unknown error");
        }
        return PM_mat_t{nullptr, {0, 0, 0}, PM_dtype_e::PM_UINT8};
    }

    PM_mat_t inpaint(
        PM_mat_t source_py, PM_mat_t mask_py, const PM_mat_t *global_mask_py, const PatchDistanceMetric &metric)
    {
        cv::Mat source = _py_to_cv2(source_py);
        cv::Mat mask = _py_to_cv2(mask_py);
        cv::Mat global_mask = global_mask_py ? _py_to_cv2(*global_mask_py) : cv::Mat();
        cv::Mat result = Inpainting(source, mask, global_mask, &metric).run(verbose().load(), random_seed().load());
        return _cv2_to_py(result);
    }
} // namespace

const char *PM_last_error()
{
    return last_error().c_str();
}

void PM_set_random_seed(unsigned int seed)
{
    random_seed().store(seed);
}

void PM_set_verbose(int value)
{
    verbose().store(value != 0);
}

void PM_free_pymat(PM_mat_t pymat)
{
    // Frees the buffer that _cv2_to_py released.
    std::default_delete<unsigned char[]>()(static_cast<unsigned char *>(pymat.data_ptr));
}

PM_mat_t PM_inpaint(PM_mat_t source_py, PM_mat_t mask_py, int patch_size)
{
    return guarded([source_py, mask_py, patch_size] {
        return inpaint(source_py, mask_py, nullptr, PatchSSDDistanceMetric(patch_size));
    });
}

PM_mat_t PM_inpaint_regularity(
    PM_mat_t source_py, PM_mat_t mask_py, PM_mat_t ijmap_py, int patch_size, float guide_weight)
{
    return guarded([source_py, mask_py, ijmap_py, patch_size, guide_weight] {
        return inpaint(
            source_py, mask_py, nullptr,
            RegularityGuidedPatchDistanceMetricV2(patch_size, _py_to_cv2(ijmap_py), guide_weight));
    });
}

PM_mat_t PM_inpaint2(PM_mat_t source_py, PM_mat_t mask_py, PM_mat_t global_mask_py, int patch_size)
{
    return guarded([source_py, mask_py, global_mask_py, patch_size] {
        return inpaint(source_py, mask_py, &global_mask_py, PatchSSDDistanceMetric(patch_size));
    });
}

PM_mat_t PM_inpaint2_regularity(
    PM_mat_t source_py, PM_mat_t mask_py, PM_mat_t global_mask_py, PM_mat_t ijmap_py, int patch_size,
    float guide_weight)
{
    return guarded([source_py, mask_py, global_mask_py, ijmap_py, patch_size, guide_weight] {
        return inpaint(
            source_py, mask_py, &global_mask_py,
            RegularityGuidedPatchDistanceMetricV2(patch_size, _py_to_cv2(ijmap_py), guide_weight));
    });
}

int _dtype_py_to_cv(PM_dtype_e dtype_py)
{
    switch (dtype_py)
    {
    case PM_dtype_e::PM_UINT8:
        return CV_8U;
    case PM_dtype_e::PM_FLOAT32:
        return CV_32F;
    default:
        throw std::invalid_argument("unsupported dtype");
    }
}

cv::Mat _py_to_cv2(PM_mat_t pymat)
{
    int dtype = CV_MAKETYPE(_dtype_py_to_cv(pymat.dtype), pymat.shape.channels);
    return cv::Mat(cv::Size(pymat.shape.width, pymat.shape.height), dtype, pymat.data_ptr).clone();
}

PM_mat_t _cv2_to_py(cv::Mat cvmat)
{
    if (!cvmat.isContinuous())
        cvmat = cvmat.clone();

    CV_Assert(cvmat.depth() == CV_8U);
    PM_shape_t shape = {cvmat.size().width, cvmat.size().height, cvmat.channels()};
    size_t dsize = cvmat.total() * cvmat.elemSize();

    // A null pointer reports an error to Python, so allocate at least one byte. A failed
    // allocation throws std::bad_alloc, which guarded() reports.
    auto data = std::make_unique<unsigned char[]>(dsize > 0 ? dsize : 1);
    if (dsize > 0)
        std::memcpy(data.get(), cvmat.data, dsize);

    // Python passes the buffer back to PM_free_pymat, which frees it.
    return PM_mat_t{data.release(), shape, PM_dtype_e::PM_UINT8};
}
