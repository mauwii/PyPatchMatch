"""Property-based tests: the invariants of the inpainting on random inputs."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import patchmatch

sizes = st.integers(1, 24)
patch_sizes = st.integers(1, 4)


@st.composite
def images_and_masks(draw):
    """An image, a hole mask (None: the white pixels) and an optional global mask."""
    shape = (draw(sizes), draw(sizes))
    image = draw(hnp.arrays(np.uint8, (*shape, 3)))
    mask = draw(st.none() | hnp.arrays(np.bool_, shape))
    global_mask = draw(st.none() | hnp.arrays(np.bool_, shape))
    return image, mask, global_mask


def assert_only_holes_changed(result, image, mask, global_mask):
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    holes = (image == 255).all(axis=-1) if mask is None else mask
    if global_mask is not None:
        holes = holes & ~global_mask
    np.testing.assert_array_equal(result[~holes], image[~holes])


@settings(deadline=None)
@given(images_and_masks(), patch_sizes)
def test_inpaint(inputs, patch_size):
    image, mask, global_mask = inputs

    def inpaint():
        patchmatch.set_random_seed(0)
        return patchmatch.inpaint(
            image, mask, global_mask=global_mask, patch_size=patch_size
        )

    result = inpaint()
    assert_only_holes_changed(result, image, mask, global_mask)
    np.testing.assert_array_equal(inpaint(), result)


@settings(deadline=None)
@given(
    images_and_masks(),
    patch_sizes,
    hnp.arrays(
        np.float32,
        st.tuples(sizes, sizes, st.just(3)),
        elements=st.floats(-2, 2, width=32),
    ),
    st.floats(0, 10),
)
def test_inpaint_regularity(inputs, patch_size, ijmap, guide_weight):
    image, mask, global_mask = inputs
    result = patchmatch.inpaint_regularity(
        image,
        mask,
        ijmap,
        global_mask=global_mask,
        patch_size=patch_size,
        guide_weight=guide_weight,
    )
    assert_only_holes_changed(result, image, mask, global_mask)


@settings(deadline=None)
@given(images_and_masks(), hnp.arrays(np.uint8, (24, 24, 3)), patch_sizes)
def test_colors_under_global_mask_do_not_matter(inputs, colors, patch_size):
    image, mask, global_mask = inputs
    if global_mask is None:
        global_mask = np.zeros(image.shape[:2], dtype=bool)
    if mask is None:
        mask = (image == 255).all(axis=-1)
    colors = colors[: image.shape[0], : image.shape[1]]
    recolored = np.where(global_mask[..., None], colors, image)
    ijmap = image.astype(np.float32) / 255

    def fills(source):
        patchmatch.set_random_seed(0)
        kwargs = {"global_mask": global_mask, "patch_size": patch_size}
        return (
            patchmatch.inpaint(source, mask, **kwargs),
            patchmatch.inpaint_regularity(source, mask, ijmap, **kwargs),
        )

    holes = mask & ~global_mask
    for result, expected in zip(fills(recolored), fills(image), strict=True):
        np.testing.assert_array_equal(result[holes], expected[holes])
