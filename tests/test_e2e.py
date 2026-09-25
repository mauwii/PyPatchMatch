"""End-to-end tests of the user workflows shown in examples/ and the README."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

import patchmatch

IMAGES = Path(__file__).parents[1] / "examples" / "images"

pytestmark = pytest.mark.e2e


def white_pixels(image: np.ndarray) -> np.ndarray:
    return (image == 255).all(axis=2)


def surrounding(mask: np.ndarray, width: int = 10) -> np.ndarray:
    """Band of ``width`` pixels around ``mask``."""
    grown = Image.fromarray(mask.astype(np.uint8) * 255).filter(
        ImageFilter.MaxFilter(2 * width + 1)
    )
    return (np.array(grown) > 0) & ~mask


def assert_plausible_fill(source: np.ndarray, result: np.ndarray) -> None:
    """Check an inpainting result of ``source`` without a ground truth image."""
    holes = white_pixels(source)
    assert holes.any()
    assert result.shape == source.shape
    assert result.dtype == np.uint8

    assert not white_pixels(result)[holes].any(), "holes were not filled"

    # the known part of the image is reconstructed almost unchanged
    outside = np.abs(result[~holes].astype(float) - source[~holes]).mean()
    assert outside < 1, outside

    # the filling blends in with its surroundings
    band = surrounding(holes)
    color_diff = np.abs(result[holes].mean(axis=0) - source[band].mean(axis=0))
    assert color_diff.max() < 10, color_diff


def coarse_error(image: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """Mean error inside ``mask`` on a downscaled image (texture, not pixels)."""

    def reduce(a: np.ndarray) -> np.ndarray:
        return np.array(Image.fromarray(a).reduce(8)).astype(float)

    cells = reduce(mask.astype(np.uint8) * 255) > 200
    return float(np.abs(reduce(image)[cells] - reduce(truth)[cells]).mean())


@pytest.fixture
def pruned_image() -> Image.Image:
    return Image.open(IMAGES / "forest_pruned.bmp")


@pytest.fixture(autouse=True)
def seed():
    patchmatch.set_random_seed(0)


@pytest.mark.parametrize("as_array", [False, True], ids=["pil", "numpy"])
def test_remove_white_regions(pruned_image, tmp_path, as_array):
    """examples/py_example.py: white pixels are the holes, the result is saved."""
    image = np.array(pruned_image) if as_array else pruned_image

    result = patchmatch.inpaint(image, patch_size=3)

    output = tmp_path / "forest_recovered.bmp"
    Image.fromarray(result).save(output)
    saved = np.array(Image.open(output))
    np.testing.assert_array_equal(saved, result)
    assert_plausible_fill(np.array(pruned_image), saved)


def test_global_mask(pruned_image, capfd):
    """examples/py_example_global_mask.py: extra hole excluded as patch source."""
    source = np.array(pruned_image)
    source[:100, :100] = 255
    global_mask = np.zeros_like(source[..., 0])
    global_mask[:100, :100] = 1

    patchmatch.set_verbose(True)
    try:
        result = patchmatch.inpaint(source, global_mask=global_mask, patch_size=3)
    finally:
        patchmatch.set_verbose(False)

    assert_plausible_fill(source, result)
    assert "Inpainting level" in capfd.readouterr().err


@pytest.mark.parametrize("hole_value", [1, 255])
def test_explicit_mask_matches_implicit_white_mask(pruned_image, hole_value):
    """README: an explicit mask behaves like the default mask of white pixels."""
    holes = white_pixels(np.array(pruned_image))
    mask = Image.fromarray(holes.astype(np.uint8) * hole_value)

    explicit = patchmatch.inpaint(pruned_image, mask, patch_size=3)
    implicit = patchmatch.inpaint(pruned_image, patch_size=3)

    np.testing.assert_array_equal(explicit, implicit)


# Regions of pure background in forest_pruned.bmp as (y, x, height, width).
BACKGROUND_REGIONS = [(20, 400, 60, 80), (40, 520, 60, 60)]


@pytest.mark.parametrize("region", BACKGROUND_REGIONS)
def test_reconstructs_known_background(pruned_image, region):
    """Cut a hole into known background and compare with the original pixels."""
    truth = np.array(pruned_image)
    y, x, height, width = region
    mask = np.zeros(truth.shape[:2], dtype=bool)
    mask[y : y + height, x : x + width] = True
    assert not white_pixels(truth)[mask].any()

    source = truth.copy()
    source[mask] = 255
    result = patchmatch.inpaint(source, mask.astype(np.uint8), patch_size=3)

    # baseline: fill the hole with the mean color of its surroundings
    naive = source.copy()
    naive[mask] = source[surrounding(mask)].mean(axis=0).astype(np.uint8)

    error = coarse_error(result, truth, mask)
    naive_error = coarse_error(naive, truth, mask)
    assert error < 0.8 * naive_error, (error, naive_error)
