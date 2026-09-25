from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import patchmatch
from patchmatch import patch_match

IMAGES = Path(__file__).parents[1] / "examples" / "images"


@pytest.fixture
def input_image():
    return Image.open(IMAGES / "forest_pruned.bmp")


@pytest.fixture
def ijmap(input_image):
    return np.zeros((input_image.height, input_image.width, 3), dtype=np.float32)


@pytest.fixture
def mask(input_image):
    return np.zeros((input_image.height, input_image.width), dtype=np.uint8)


def test_library_available():
    assert patch_match.patchmatch_available


def test_top_level_exports():
    for name in patch_match.__all__:
        assert getattr(patchmatch, name) is getattr(patch_match, name)


def test_inpaint_fills_white_pixels(input_image):
    source = np.array(input_image)
    holes = (source == 255).all(axis=2)
    assert holes.any()

    result = patch_match.inpaint(input_image, patch_size=3)

    assert result.shape == source.shape
    assert result.dtype == np.uint8
    assert not (result[holes] == 255).all(axis=1).all()


def test_inpaint_custom_patch_size(input_image, mask):
    result = patch_match.inpaint(input_image, mask, patch_size=1)
    assert isinstance(result, np.ndarray)


def test_inpaint_global_mask(input_image):
    source = np.array(input_image)
    source[:100, :100] = 255
    global_mask = np.zeros_like(source[..., 0])
    global_mask[:100, :100] = 1
    result = patch_match.inpaint(source, global_mask=global_mask, patch_size=3)
    assert result.shape == source.shape


def test_inpaint_is_deterministic_with_seed(input_image):
    patch_match.set_random_seed(42)
    first = patch_match.inpaint(input_image, patch_size=3)
    patch_match.set_random_seed(42)
    second = patch_match.inpaint(input_image, patch_size=3)
    np.testing.assert_array_equal(first, second)


def test_set_verbose():
    patch_match.set_verbose(False)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
    ],
)
def test_inpaint_invalid_image(image):
    with pytest.raises(ValueError, match="image"):
        patch_match.inpaint(image)


def test_np_to_pymat():
    npmat = np.zeros((10, 20, 3), dtype=np.uint8)
    result = patch_match.np_to_pymat(npmat)
    assert isinstance(result, patch_match.CMatT)
    assert (result.shape.height, result.shape.width) == (10, 20)


def test_np_to_pymat_unsupported_dtype():
    with pytest.raises(TypeError):
        patch_match.np_to_pymat(np.zeros((2, 2, 1), dtype=np.int64))


def test_np_to_pymat_non_contiguous():
    with pytest.raises(ValueError, match="contiguous"):
        patch_match.np_to_pymat(np.zeros((10, 10, 3), dtype=np.uint8)[::2])


def test_pymat_to_np_copies_data():
    npmat = np.arange(10 * 20 * 3, dtype=np.float32).reshape(10, 20, 3)
    result = patch_match.pymat_to_np(patch_match.np_to_pymat(npmat))
    np.testing.assert_array_equal(result, npmat)
    assert not np.shares_memory(result, npmat)


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 1), dtype=np.uint8),
        Image.new("L", (10, 10)),
    ],
)
def test_canonize_mask_array(mask):
    result = patch_match._canonize_mask_array(mask)
    assert result.shape == (10, 10, 1)
    assert result.dtype == np.uint8


def test_canonize_mask_array_invalid_input():
    with pytest.raises(ValueError, match="mask"):
        patch_match._canonize_mask_array(np.zeros((10, 10, 3), dtype=np.uint8))


def test_inpaint_regularity(input_image, ijmap):
    result = patch_match.inpaint_regularity(input_image, None, ijmap, patch_size=3)
    assert result.shape == (input_image.height, input_image.width, 3)


def test_inpaint_regularity_custom_guide_weight(input_image, ijmap):
    result = patch_match.inpaint_regularity(
        input_image, None, ijmap, patch_size=5, guide_weight=0.5
    )
    assert isinstance(result, np.ndarray)


def test_inpaint_regularity_custom_global_mask(input_image, ijmap, mask):
    result = patch_match.inpaint_regularity(
        input_image, None, ijmap, global_mask=mask, patch_size=3
    )
    assert isinstance(result, np.ndarray)


def test_inpaint_regularity_custom_mask(input_image, ijmap, mask):
    result = patch_match.inpaint_regularity(input_image, mask, ijmap, patch_size=3)
    assert isinstance(result, np.ndarray)


def test_inpaint_regularity_invalid_ijmap(input_image, ijmap):
    with pytest.raises(ValueError, match="ijmap"):
        patch_match.inpaint_regularity(input_image, None, ijmap.astype(np.float64))


def test_unavailable_library_raises(monkeypatch):
    monkeypatch.setattr(patch_match, "_lib", None)
    with pytest.raises(RuntimeError, match="not available"):
        patch_match.set_random_seed(0)
