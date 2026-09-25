"""Unit tests of the Python API on small synthetic images."""

import ctypes

import numpy as np
import pytest
from PIL import Image

import patchmatch
from patchmatch import _lib, patch_match

HEIGHT, WIDTH = 48, 64
HOLE = (slice(20, 28), slice(24, 36))


@pytest.fixture
def image() -> np.ndarray:
    """Smooth color gradient with a little noise and a white rectangular hole."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    rgb = np.stack([4 * x, 5 * y, 2 * (x + y)], axis=-1)
    noise = np.random.default_rng(0).integers(0, 8, rgb.shape)
    img = np.clip(rgb + noise, 0, 200).astype(np.uint8)
    img[HOLE] = 255
    return img


@pytest.fixture
def hole_mask() -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[HOLE] = 1
    return mask


@pytest.fixture
def ijmap() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)


@pytest.fixture(autouse=True)
def seed():
    patchmatch.set_random_seed(0)


def assert_filled(result: np.ndarray, source: np.ndarray) -> None:
    assert result.shape == source.shape
    assert result.dtype == np.uint8
    assert not (result[HOLE] == 255).all(axis=-1).any()


# --- package ----------------------------------------------------------------


def test_library_available():
    assert patchmatch.patchmatch_available


def test_top_level_exports():
    for name in patch_match.__all__:
        assert getattr(patchmatch, name) is getattr(patch_match, name)


def test_version():
    assert isinstance(patchmatch.__version__, str)
    assert patchmatch.__version__[0].isdigit()


# --- inpaint ----------------------------------------------------------------


def test_inpaint_default_mask_fills_white_pixels(image):
    assert_filled(patchmatch.inpaint(image, patch_size=3), image)


def test_inpaint_explicit_mask(image, hole_mask):
    image[HOLE] = 0  # not white: only the explicit mask marks the hole
    result = patchmatch.inpaint(image, hole_mask, patch_size=3)
    assert result.shape == image.shape
    assert (result[HOLE] != 0).any()


def test_inpaint_global_mask(image, hole_mask):
    global_mask = np.zeros_like(hole_mask)
    global_mask[:8] = 1
    assert_filled(
        patchmatch.inpaint(image, global_mask=global_mask, patch_size=3), image
    )


@pytest.mark.parametrize("patch_size", [1, 3, 7])
def test_inpaint_patch_sizes(image, patch_size):
    assert_filled(patchmatch.inpaint(image, patch_size=patch_size), image)


def test_inpaint_does_not_modify_inputs(image, hole_mask):
    image_before, mask_before = image.copy(), hole_mask.copy()
    patchmatch.inpaint(image, hole_mask, global_mask=hole_mask, patch_size=3)
    np.testing.assert_array_equal(image, image_before)
    np.testing.assert_array_equal(hole_mask, mask_before)


def test_inpaint_is_deterministic_with_seed(image):
    patchmatch.set_random_seed(42)
    first = patchmatch.inpaint(image, patch_size=3)
    patchmatch.set_random_seed(42)
    second = patchmatch.inpaint(image, patch_size=3)
    np.testing.assert_array_equal(first, second)


def test_set_verbose_prints_progress(image, capfd):
    patchmatch.set_verbose(True)
    try:
        patchmatch.inpaint(image, patch_size=3)
    finally:
        patchmatch.set_verbose(False)
    assert "Inpainting level" in capfd.readouterr().err

    patchmatch.inpaint(image, patch_size=3)
    assert capfd.readouterr().err == ""


# --- inpaint_regularity -----------------------------------------------------


@pytest.mark.parametrize("use_mask", [False, True], ids=["white", "mask"])
@pytest.mark.parametrize("use_global_mask", [False, True], ids=["", "global"])
@pytest.mark.parametrize("guide_weight", [0.0, 0.25, 1.0])
def test_inpaint_regularity(
    image, hole_mask, ijmap, use_mask, use_global_mask, guide_weight
):
    result = patchmatch.inpaint_regularity(
        image,
        hole_mask if use_mask else None,
        ijmap,
        global_mask=np.zeros_like(hole_mask) if use_global_mask else None,
        patch_size=3,
        guide_weight=guide_weight,
    )
    assert_filled(result, image)


def test_inpaint_regularity_scales_smaller_ijmap(image):
    """The map is scaled to the image, see RegularityGuidedPatchDistanceMetricV2."""
    ijmap = np.zeros((HEIGHT // 2, WIDTH // 3, 3), dtype=np.float32)
    assert_filled(
        patchmatch.inpaint_regularity(image, None, ijmap, patch_size=3), image
    )


# --- input conversion and validation ----------------------------------------


def test_inpaint_accepts_pil_images(image, hole_mask):
    result = patchmatch.inpaint(
        Image.fromarray(image), Image.fromarray(hole_mask), patch_size=3
    )
    assert_filled(result, image)


def test_inpaint_accepts_non_contiguous_image(image):
    wide = np.repeat(image, 2, axis=1)[:, ::2]
    assert not wide.flags.c_contiguous
    assert_filled(patchmatch.inpaint(wide, patch_size=3), image)


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 1), dtype=np.uint8),
        np.zeros((10, 10), dtype=bool),
        Image.new("L", (10, 10)),
        Image.new("1", (10, 10)),
    ],
    ids=["2d", "3d", "bool", "pil-L", "pil-1"],
)
def test_canonize_mask_array(mask):
    result = patch_match._canonize_mask_array(mask)
    assert result.shape == (10, 10, 1)
    assert result.dtype == np.uint8
    assert result.flags.c_contiguous


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.zeros((10, 10), dtype=np.float32),
        np.zeros(10, dtype=np.uint8),
    ],
    ids=["3-channel", "float", "1d"],
)
def test_canonize_mask_array_invalid(mask):
    with pytest.raises(ValueError, match="mask"):
        patch_match._canonize_mask_array(mask)


@pytest.mark.parametrize(
    "bad_image",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
    ],
    ids=["gray", "rgba", "float"],
)
def test_inpaint_invalid_image(bad_image):
    with pytest.raises(ValueError, match="image"):
        patchmatch.inpaint(bad_image)


@pytest.mark.parametrize("argument", ["mask", "global_mask"])
@pytest.mark.parametrize(
    "func",
    [patchmatch.inpaint, patchmatch.inpaint_regularity],
    ids=lambda f: f.__name__,
)
def test_mask_size_must_match_image(image, ijmap, func, argument):
    kwargs = {"mask": None, argument: np.zeros((HEIGHT // 2, WIDTH), dtype=np.uint8)}
    if func is patchmatch.inpaint_regularity:
        kwargs["ijmap"] = ijmap
    with pytest.raises(ValueError, match=f"{argument} must have the same height"):
        func(image, **kwargs)


@pytest.mark.parametrize(
    "bad_ijmap",
    [
        np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64),
        np.zeros((HEIGHT, WIDTH, 2), dtype=np.float32),
        [[[0.0, 0.0, 0.0]]],
    ],
    ids=["float64", "2-channel", "list"],
)
def test_inpaint_regularity_invalid_ijmap(image, bad_ijmap):
    with pytest.raises(ValueError, match="ijmap"):
        patchmatch.inpaint_regularity(image, None, bad_ijmap)


# --- missing native library -------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda img: patchmatch.set_random_seed(0),
        lambda img: patchmatch.set_verbose(False),
        lambda img: patchmatch.inpaint(img),
        lambda img: patchmatch.inpaint_regularity(
            img, None, np.zeros(img.shape, dtype=np.float32)
        ),
    ],
    ids=["set_random_seed", "set_verbose", "inpaint", "inpaint_regularity"],
)
def test_unavailable_library_raises(monkeypatch, image, call):
    monkeypatch.setattr(patch_match, "_lib", None)
    with pytest.raises(RuntimeError, match="not available"):
        call(image)


def test_find_library_reports_missing_file(monkeypatch):
    monkeypatch.setattr(_lib, "LIBRARY_NAME", "does-not-exist.so")
    with pytest.raises(OSError, match=r"does-not-exist\.so not found"):
        _lib.find_library()


# --- ctypes conversion ------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [np.uint8, np.int8, np.uint16, np.int16, np.int32, np.float32, np.float64],
)
def test_pymat_roundtrip(dtype):
    npmat = (np.arange(10 * 20 * 3) % 100).astype(dtype).reshape(10, 20, 3)
    pymat = _lib.np_to_pymat(npmat)
    assert (pymat.shape.height, pymat.shape.width, pymat.shape.channels) == (10, 20, 3)

    result = _lib.pymat_to_np(pymat)
    assert result.dtype == dtype
    np.testing.assert_array_equal(result, npmat)
    assert not np.shares_memory(result, npmat)


def test_np_to_pymat_does_not_copy():
    npmat = np.zeros((2, 3, 1), dtype=np.uint8)
    pymat = _lib.np_to_pymat(npmat)
    assert pymat.data_ptr == npmat.ctypes.data


def test_np_to_pymat_unsupported_dtype():
    with pytest.raises(TypeError, match="int64"):
        _lib.np_to_pymat(np.zeros((2, 2, 1), dtype=np.int64))


def test_np_to_pymat_wrong_ndim():
    with pytest.raises(ValueError, match="3-dimensional"):
        _lib.np_to_pymat(np.zeros((2, 2), dtype=np.uint8))


def test_np_to_pymat_non_contiguous():
    with pytest.raises(ValueError, match="contiguous"):
        _lib.np_to_pymat(np.zeros((10, 10, 3), dtype=np.uint8)[::2])


def test_ctypes_structures_backwards_compatible():
    """CMatT/CShapeT stay importable from patch_match with the C layout."""
    assert patch_match.CMatT is _lib.CMatT
    assert patch_match.CShapeT is _lib.CShapeT
    assert ctypes.sizeof(_lib.CShapeT) == 3 * ctypes.sizeof(ctypes.c_int)
