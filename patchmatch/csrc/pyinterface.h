#pragma once

extern "C"
{

    struct PM_shape_t
    {
        int width;
        int height;
        int channels;
    };

    enum class PM_dtype_e : int
    {
        PM_UINT8,
        PM_FLOAT32,
    };

    struct PM_mat_t
    {
        void *data_ptr;
        PM_shape_t shape;
        int dtype;
    };

    // The PM_inpaint* functions return a null data_ptr on failure; this returns the
    // error message of the last failure on the calling thread.
    const char *PM_last_error(void);

    void PM_set_random_seed(unsigned int seed);
    void PM_set_verbose(int value);

    void PM_free_pymat(PM_mat_t pymat);
    PM_mat_t PM_inpaint(PM_mat_t image, PM_mat_t mask, int patch_size);
    PM_mat_t PM_inpaint_regularity(PM_mat_t image, PM_mat_t mask, PM_mat_t ijmap, int patch_size, float guide_weight);
    PM_mat_t PM_inpaint2(PM_mat_t image, PM_mat_t mask, PM_mat_t global_mask, int patch_size);
    PM_mat_t PM_inpaint2_regularity(
        PM_mat_t image, PM_mat_t mask, PM_mat_t global_mask, PM_mat_t ijmap, int patch_size, float guide_weight);

} /*  extern "C" */
