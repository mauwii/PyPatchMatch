import os

import numpy as np
import pytest
from PIL import Image

from patchmatch import patch_match


@pytest.fixture
def input_image():
    return Image.open("./examples/images/forest_pruned.bmp")


@pytest.fixture
def ijmap(input_image):
    return np.zeros((input_image.height, input_image.width, 3), dtype=np.float32)


@pytest.fixture
def mask(input_image):
    return np.zeros((input_image.height, input_image.width), dtype=np.uint8)


@pytest.fixture
def temp_file():
    filename = "temp.txt"
    with open(filename, "w") as f:
        f.write("test")
    yield filename
    os.remove(filename)


def test_inpaint_custom_patch_size(input_image, mask):
    result = patch_match.inpaint(input_image, mask, patch_size=1)
    assert isinstance(result, np.ndarray)


def test_inpaint_global_mask(input_image):
    source = np.array(input_image)
    source[:100, :100] = 255
    global_mask = np.zeros_like(source[..., 0])
    global_mask[:100, :100] = 1
    result = Image.fromarray(
        patch_match.inpaint(image=source, global_mask=global_mask, patch_size=3)
    )
    assert isinstance(result, Image.Image)


def test_np_to_pymat():
    npmat = np.zeros((10, 10, 3), dtype=np.uint8)
    result = patch_match.np_to_pymat(npmat)
    assert isinstance(result, patch_match.CMatT)


def test_pymat_to_np():
    pymat = patch_match.CMatT(None, patch_match.CShapeT(10, 10, 3), 0)
    npmat = np.zeros((10, 10, 3), dtype=np.uint8)
    pymat.data_ptr = npmat.ctypes.data
    result = patch_match.pymat_to_np(pymat)
    assert isinstance(result, np.ndarray)
    assert result.shape == npmat.shape
    assert result.dtype == npmat.dtype
    assert np.allclose(result, npmat)


def test_canonize_mask_array_2d_uint8():
    mask = np.zeros((10, 10), dtype=np.uint8)
    result = patch_match._canonize_mask_array(mask)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 3
    assert result.shape[2] == 1
    assert result.dtype == np.uint8


def test_canonize_mask_array_3d_uint8():
    mask = np.zeros((10, 10, 1), dtype=np.uint8)
    result = patch_match._canonize_mask_array(mask)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 3
    assert result.shape[2] == 1
    assert result.dtype == np.uint8


def test_canonize_mask_array_pil_image():
    mask = Image.new("L", (10, 10))
    result = patch_match._canonize_mask_array(mask)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 3
    assert result.shape[2] == 1
    assert result.dtype == np.uint8


def test_canonize_mask_array_invalid_input():
    with pytest.raises(AssertionError):
        mask = np.zeros((10, 10, 3), dtype=np.uint8)
        patch_match._canonize_mask_array(mask)


def test_inpaint_regularity(input_image, ijmap):
    result = patch_match.inpaint_regularity(input_image, None, ijmap, patch_size=3)
    assert isinstance(result, np.ndarray)


def test_inpaint_regularity_custom_patch_size(input_image, ijmap):
    result = patch_match.inpaint_regularity(input_image, None, ijmap, patch_size=5)
    assert isinstance(result, np.ndarray)


def test_inpaint_regularity_custom_guide_weight(input_image, ijmap):
    result = patch_match.inpaint_regularity(
        input_image, None, ijmap, patch_size=3, guide_weight=0.5
    )
    assert isinstance(result, np.ndarray)


def test_inpaint_regularity_custom_global_mask(input_image, ijmap, mask):
    global_mask = mask
    result = patch_match.inpaint_regularity(
        input_image, None, ijmap, global_mask=global_mask, patch_size=3
    )
    assert isinstance(result, np.ndarray)


def test_inpaint_regularity_custom_mask(input_image, ijmap, mask):
    result = patch_match.inpaint_regularity(input_image, mask, ijmap, patch_size=3)
    assert isinstance(result, np.ndarray)


def test_download_url_to_file(temp_file):
    url = "https://www.google.com"
    patch_match.download_url_to_file(url, temp_file)
    assert os.path.exists(temp_file)
