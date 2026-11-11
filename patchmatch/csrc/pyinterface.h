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
        PM_dtype_e dtype;
    };

    // The PM_inpaint* functions return a null data_ptr on failure; this returns the
    // error message of the last failure on the calling thread.
    const char *PM_last_error(void);

    void PM_set_random_seed(unsigned int seed);
    void PM_set_verbose(int value);
    // The most threads of the following inpaintings, 0 for one per core; their results do not depend on it.
    void PM_set_num_threads(int threads);

    void PM_free_pymat(PM_mat_t pymat);
    PM_mat_t PM_inpaint(PM_mat_t image, PM_mat_t mask, int patch_size);
    PM_mat_t PM_inpaint_regularity(PM_mat_t image, PM_mat_t mask, PM_mat_t ijmap, int patch_size, float guide_weight);
    PM_mat_t PM_inpaint2(PM_mat_t image, PM_mat_t mask, PM_mat_t global_mask, int patch_size);
    PM_mat_t PM_inpaint2_regularity(
        PM_mat_t image, PM_mat_t mask, PM_mat_t global_mask, PM_mat_t ijmap, int patch_size, float guide_weight);
    // Like the functions above with the seed of this call instead of PM_set_random_seed's. A null global_mask or
    // ijmap leaves it out; without ijmap, guide_weight is unused.
    PM_mat_t PM_inpaint_seeded(
        PM_mat_t image, PM_mat_t mask, const PM_mat_t *global_mask, const PM_mat_t *ijmap, int patch_size,
        float guide_weight, unsigned int seed);

} /*  extern "C" */
