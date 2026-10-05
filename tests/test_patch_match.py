"""Unit tests of the Python API on small synthetic images."""

import ctypes
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image

import patchmatch
from patchmatch import _lib, patch_match

HEIGHT, WIDTH = 48, 64
HOLE = (slice(20, 28), slice(24, 36))
AROUND_HOLE = (slice(14, 34), slice(18, 42))


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
def regular_image() -> np.ndarray:
    """Noisy pattern that repeats every 8 pixels, as ijmap describes, with the hole.

    The regularity term copies from whole periods away, which in the gradient of
    ``image`` shifted the fill by up to 35 color values.
    """
    rng = np.random.default_rng(0)
    tile = rng.integers(80, 120, (8, 8, 3))
    noise = rng.integers(0, 8, (HEIGHT, WIDTH, 3))
    img = (np.tile(tile, (HEIGHT // 8, WIDTH // 8, 1)) + noise).astype(np.uint8)
    img[HOLE] = 255
    return img


@pytest.fixture
def hole_mask() -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[HOLE] = 1
    return mask


@pytest.fixture
def ijmap() -> np.ndarray:
    """Regularity coordinates of a pattern that repeats every 8 pixels."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    ij = np.stack([y % 8 / 8, x % 8 / 8, np.zeros_like(y)], axis=-1)
    return ij.astype(np.float32)


@pytest.fixture(autouse=True)
def seed():
    patchmatch.set_random_seed(0)


def assert_filled(result: np.ndarray, source: np.ndarray) -> None:
    """The hole is filled to blend in, and all other pixels keep their values."""
    assert result.shape == source.shape
    assert result.dtype == np.uint8
    known = np.ones(source.shape[:2], dtype=bool)
    known[HOLE] = False
    np.testing.assert_array_equal(result[known], source[known])

    # a hole that is not filled stays black, see _initialize_pyramid
    ring = np.zeros_like(known)
    ring[AROUND_HOLE] = True
    ring[HOLE] = False
    fill = result[HOLE].reshape(-1, 3).mean(axis=0)
    color_diff = np.abs(fill - source[ring].mean(axis=0))
    assert color_diff.max() < 10, color_diff


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
    assert_filled(patchmatch.inpaint(image, hole_mask, patch_size=3), image)


def test_inpaint_global_mask(image, hole_mask):
    global_mask = np.zeros_like(hole_mask)
    global_mask[:8] = 1
    result = patchmatch.inpaint(image, global_mask=global_mask, patch_size=3)
    assert_filled(result, image)
    # excluded pixels keep their values instead of the zeros of the pyramid
    np.testing.assert_array_equal(result[:8], image[:8])


def test_inpaint_ignores_colors_under_global_mask(image, hole_mask):
    global_mask = np.zeros_like(hole_mask)
    global_mask[16:32, :24] = 1  # touches the hole
    results = []
    for color in (0, 255):
        image[global_mask == 1] = color
        patchmatch.set_random_seed(0)
        results.append(
            patchmatch.inpaint(image, hole_mask, global_mask=global_mask, patch_size=3)
        )
    np.testing.assert_array_equal(results[0][HOLE], results[1][HOLE])


@pytest.mark.parametrize("shape", [(0, 0, 3), (0, 10, 3), (10, 0, 3)])
def test_inpaint_empty_image(shape, ijmap):
    empty = np.zeros(shape, dtype=np.uint8)
    global_mask = np.zeros(shape[:2], dtype=np.uint8)
    results = [
        patchmatch.inpaint(empty, patch_size=3),
        patchmatch.inpaint(empty, global_mask=global_mask),
        patchmatch.inpaint_regularity(empty, None, ijmap, patch_size=3),
        patchmatch.inpaint_regularity(empty, None, ijmap, global_mask=global_mask),
    ]
    for result in results:
        assert result.shape == shape
        assert result is not empty


@pytest.mark.parametrize("patch_size", [1, 3, 7])
def test_inpaint_patch_sizes(image, patch_size):
    assert_filled(patchmatch.inpaint(image, patch_size=patch_size), image)


def test_inpaint_thin_hole_takes_the_colors_beside_it():
    """A thin hole that runs from a color gradient into a plain region.

    The coarser levels do not have it, and its plain part was filled with the
    colors of the gradient in a third of the seeds.
    """
    plain = (150, 170, 90)
    y, x = np.mgrid[0:96, 0:48]
    gradient = np.stack([10 + 3 * x, 200 - 3 * x, np.full_like(x, 60)], axis=-1)
    noise = np.random.default_rng(0).integers(-3, 4, gradient.shape)
    image = (np.where(y[..., None] < 72, gradient, plain) + noise).astype(np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[48:, 22:26] = 1
    image[mask == 1] = 255

    for seed in range(20):
        patchmatch.set_random_seed(seed)
        result = patchmatch.inpaint(image, mask, patch_size=3)
        fill = result[76:, 22:26].reshape(-1, 3).mean(axis=0)
        assert np.abs(fill - plain).max() < 10, (seed, fill)


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


def test_concurrent_inpaint_is_deterministic(image):
    # the native code runs without the GIL, so the threads really overlap
    expected = patchmatch.inpaint(image, patch_size=3)
    with ThreadPoolExecutor(4) as pool:
        results = list(
            pool.map(lambda _: patchmatch.inpaint(image, patch_size=3), range(16))
        )
    for result in results:
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("verbose", [True, False])
def test_set_verbose_prints_progress(verbose):
    # The native code writes to its own C runtime's stderr. On Windows the DLL
    # links the static runtime, whose stderr is not redirected by pytest's capfd,
    # so the output is checked in a separate process.
    code = (
        "import numpy as np, patchmatch\n"
        "img = np.full((32, 32, 3), 100, dtype=np.uint8)\n"
        "img[8:16, 8:16] = 255\n"
        f"patchmatch.set_verbose({verbose})\n"
        "patchmatch.inpaint(img, patch_size=3)\n"
    )
    process = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert ("Inpainting level" in process.stderr) is verbose


# --- inpaint_regularity -----------------------------------------------------


@pytest.mark.parametrize("use_mask", [False, True], ids=["white", "mask"])
@pytest.mark.parametrize("use_global_mask", [False, True], ids=["", "global"])
@pytest.mark.parametrize("guide_weight", [0.0, 0.25, 1.0])
def test_inpaint_regularity(
    regular_image, hole_mask, ijmap, use_mask, use_global_mask, guide_weight
):
    global_mask = np.zeros_like(hole_mask)
    global_mask[:8] = 1
    result = patchmatch.inpaint_regularity(
        regular_image,
        hole_mask if use_mask else None,
        ijmap,
        global_mask=global_mask if use_global_mask else None,
        patch_size=3,
        guide_weight=guide_weight,
    )
    assert_filled(result, regular_image)


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


# 2**31 wrapped to a negative C int, which never terminated
@pytest.mark.parametrize("patch_size", [0, -1, 2**30, 2**31])
@pytest.mark.parametrize(
    "func",
    [patchmatch.inpaint, patchmatch.inpaint_regularity],
    ids=lambda f: f.__name__,
)
def test_invalid_patch_size(image, ijmap, func, patch_size):
    kwargs = {"ijmap": ijmap} if func is patchmatch.inpaint_regularity else {}
    with pytest.raises(ValueError, match="patch_size"):
        func(image, None, patch_size=patch_size, **kwargs)


@pytest.mark.parametrize("patch_size", [3.0, "3", None])
def test_patch_size_must_be_an_integer(image, patch_size):
    with pytest.raises(TypeError, match="patch_size must be an integer"):
        patchmatch.inpaint(image, patch_size=patch_size)


def test_patch_size_accepts_numpy_integers(image):
    expected = patchmatch.inpaint(image, patch_size=3)
    np.testing.assert_array_equal(
        patchmatch.inpaint(image, patch_size=np.int64(3)), expected
    )


@pytest.mark.parametrize(
    "bad_ijmap",
    [
        np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64),
        np.zeros((HEIGHT, WIDTH, 2), dtype=np.float32),
        np.zeros((0, WIDTH, 3), dtype=np.float32),
        [[[0.0, 0.0, 0.0]]],
        np.full((HEIGHT, WIDTH, 3), np.nan, dtype=np.float32),
        np.full((HEIGHT, WIDTH, 3), np.inf, dtype=np.float32),
    ],
    ids=["float64", "2-channel", "empty", "list", "nan", "inf"],
)
def test_inpaint_regularity_invalid_ijmap(image, bad_ijmap):
    with pytest.raises(ValueError, match="ijmap"):
        patchmatch.inpaint_regularity(image, None, bad_ijmap)


@pytest.mark.parametrize("guide_weight", [-1.0, -0.5, float("nan"), float("inf")])
def test_inpaint_regularity_invalid_guide_weight(image, ijmap, guide_weight):
    # a weight of -1 used to crash the native code, other negative weights made it
    # read outside of the similarity table
    with pytest.raises(ValueError, match="guide_weight"):
        patchmatch.inpaint_regularity(image, None, ijmap, guide_weight=guide_weight)


# --- native errors ----------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["PM_inpaint", "PM_inpaint2", "PM_inpaint_regularity", "PM_inpaint2_regularity"],
)
def test_native_exception_raises_runtime_error(name):
    # A null data pointer with a non-empty shape fails an OpenCV assertion. The
    # exception must not propagate through the C interface and abort the process.
    func = getattr(patch_match._get_lib(), name)
    bad = _lib.CMatT(None, _lib.CShapeT(4, 4, 3), 0)
    args = [bad if argtype is _lib.CMatT else argtype(3) for argtype in func.argtypes]
    with pytest.raises(RuntimeError, match=r"patchmatch failed: .*Assertion failed"):
        patch_match._call(func, *args)


def test_native_rejects_negative_guide_weight(image, hole_mask, ijmap):
    """The C++ metric checks the weight as well, for callers that bypass Python."""
    lib = patch_match._get_lib()
    mask = hole_mask[..., np.newaxis]
    args = (image, mask, ijmap, ctypes.c_int(3), ctypes.c_float(-1))
    with pytest.raises(RuntimeError, match="guide weight must be >= 0"):
        patch_match._call(lib.PM_inpaint_regularity, *args)


def test_native_accepts_empty_image():
    # Python returns empty images before the native call; C callers still get an
    # empty result instead of a failure. OpenCV 4.6 returns it with shape (0, 0, 1).
    lib = patch_match._get_lib()
    empty = np.zeros((0, 10, 3), dtype=np.uint8)
    mask = np.zeros((0, 10, 1), dtype=np.uint8)
    assert patch_match._call(lib.PM_inpaint, empty, mask, ctypes.c_int(3)).size == 0


def test_native_rejects_unsupported_dtype(image, hole_mask):
    lib = patch_match._get_lib()
    source = _lib.np_to_pymat(image)
    source.dtype = len(_lib._PYMAT_DTYPES)
    args = (source, hole_mask[..., np.newaxis], ctypes.c_int(3))
    with pytest.raises(RuntimeError, match="unsupported dtype"):
        patch_match._call(lib.PM_inpaint, *args)


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


def test_import_without_native_library():
    """A failed build from the sdist installs the package without the library.

    The import has to work and explain how to fix it, see the overrides in
    pyproject.toml. The module is reloaded in a separate process, so the other
    tests keep the loaded library.
    """
    code = (
        "import importlib\n"
        "from patchmatch import _lib, patch_match\n"
        "_lib.LIBRARY_NAME = 'missing-library'\n"
        "importlib.reload(patch_match)\n"
        "assert not patch_match.patchmatch_available\n"
    )
    process = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert "missing-library not found" in process.stderr
    assert "--no-cache-dir" in process.stderr


# --- ctypes conversion ------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
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
