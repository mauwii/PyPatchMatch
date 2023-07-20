import tempfile

import numpy as np
import pytest
from PIL import Image

from patchmatch import patch_match


@pytest.fixture
def input_image():
    return Image.open("./examples/images/forest_pruned.bmp")


@pytest.fixture
def mask(input_image):
    return np.zeros((input_image.height, input_image.width), dtype=np.uint8)


@pytest.fixture
def ijmap(input_image):
    return np.zeros((input_image.height, input_image.width, 3), dtype=np.float32)


def test_inpaint_custom_patch_size(input_image):
    result = Image.fromarray(patch_match.inpaint(image=input_image, patch_size=3))
    assert isinstance(result, Image.Image)


def test_inpaint_global_mask(input_image):
    patch_match.set_verbose(True)
    source = np.array(input_image)
    source[:100, :100] = 255
    global_mask = np.zeros_like(source[..., 0])
    global_mask[:100, :100] = 1
    result = Image.fromarray(
        patch_match.inpaint(image=source, global_mask=global_mask, patch_size=3)
    )
    assert isinstance(result, Image.Image)


def test_download_url_to_file():
    url = "https://github.com/mauwii/PyPatchMatch/raw/main/examples/images/forest.bmp"
    dst = tempfile.mkstemp().__dir__
    patch_match.download_url_to_file(url=url, dst=str(dst))
    read_image = Image.open(str(dst))
    assert isinstance(read_image, Image.Image)


def test_inpaint_regularity(input_image, ijmap):
    ijmap = ijmap
    result = patch_match.inpaint_regularity(input_image, None, ijmap, patch_size=3)
    assert isinstance(result, np.ndarray)


def test_inpaint_regularity_custom_guide_weight(input_image, ijmap):
    ijmap = ijmap
    result = patch_match.inpaint_regularity(input_image, None, ijmap, guide_weight=0.5)
    assert isinstance(result, np.ndarray)


def test_inpaint_regularity_custom_global_mask(input_image, mask, ijmap):
    ijmap = ijmap
    global_mask = mask
    result = patch_match.inpaint_regularity(
        input_image, None, ijmap, global_mask=global_mask
    )
    assert isinstance(result, np.ndarray)


def test_inpaint_regularity_custom_mask(input_image, mask, ijmap):
    ijmap = ijmap
    mask = mask
    result = patch_match.inpaint_regularity(input_image, mask, ijmap)
    assert isinstance(result, np.ndarray)


def test_canonize_mask_array_2d_uint8():
    mask = np.zeros((10, 10), dtype=np.uint8)
    result = patch_match._canonize_mask_array(mask)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 3
    assert result.shape[2] == 1
    assert result.dtype == np.uint8


def test_canonize_mask_array_invalid_input():
    with pytest.raises(AssertionError):
        mask = np.zeros((10, 10, 3), dtype=np.uint8)
        patch_match._canonize_mask_array(mask)
