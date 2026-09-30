"""End-to-end tests of the user workflows shown in examples/ and the README."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

import patchmatch

IMAGES = Path(__file__).parents[1] / "examples" / "images"
SCRIPTS = Path(__file__).parents[1] / "scripts"

pytestmark = pytest.mark.e2e


def white_pixels(image: np.ndarray) -> np.ndarray:
    return (image == 255).all(axis=2)


def surrounding(mask: np.ndarray, width: int = 10) -> np.ndarray:
    """Band of ``width`` pixels around ``mask``."""
    grown = Image.fromarray(mask.astype(np.uint8) * 255).filter(
        ImageFilter.MaxFilter(2 * width + 1)
    )
    return (np.array(grown) > 0) & ~mask


def assert_plausible_fill(
    source: np.ndarray, result: np.ndarray, excluded: np.ndarray | None = None
) -> None:
    """Check an inpainting result of ``source`` without a ground truth image.

    ``excluded`` marks the pixels of a global mask, which are not filled.
    """
    if excluded is None:
        excluded = np.zeros(source.shape[:2], dtype=bool)
    holes = white_pixels(source) & ~excluded
    assert holes.any()
    assert result.shape == source.shape
    assert result.dtype == np.uint8

    assert not white_pixels(result)[holes].any(), "holes were not filled"

    # only the holes are filled
    np.testing.assert_array_equal(result[~holes], source[~holes])

    # the filling blends in with its surroundings
    band = surrounding(holes) & ~excluded
    color_diff = np.abs(result[holes].mean(axis=0) - source[band].mean(axis=0))
    assert color_diff.max() < 10, color_diff


def load_evaluation():
    """scripts/evaluate_inpainting.py, whose measures the quality tests share."""
    path = SCRIPTS / "evaluate_inpainting.py"
    spec = importlib.util.spec_from_file_location("evaluate_inpainting", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = load_evaluation()


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


def test_global_mask(pruned_image):
    """examples/py_example_global_mask.py: the globally masked plant is kept."""
    source = np.array(pruned_image)
    global_mask = np.zeros_like(source[..., 0])
    global_mask[290:, 100:180] = 1

    result = patchmatch.inpaint(source, global_mask=global_mask, patch_size=3)

    excluded = global_mask.astype(bool)
    np.testing.assert_array_equal(result[excluded], source[excluded])
    assert_plausible_fill(source, result, excluded)


def test_global_mask_is_not_used_as_source(pruned_image):
    """A red block next to a hole fills it, unless the block is globally masked."""
    source = np.array(pruned_image)
    source[200:300, 20:120] = (255, 0, 0)
    mask = np.zeros(source.shape[:2], dtype=np.uint8)
    mask[210:290, 120:160] = 1
    global_mask = np.zeros_like(mask)
    global_mask[200:300, 20:120] = 1

    def red_in_hole(result: np.ndarray) -> bool:
        red = (result[..., 0] > 150) & (result[..., 1] < 80) & (result[..., 2] < 80)
        return bool(red[mask == 1].any())

    assert red_in_hole(patchmatch.inpaint(source, mask, patch_size=3))

    result = patchmatch.inpaint(source, mask, global_mask=global_mask, patch_size=3)
    assert not red_in_hole(result)
    excluded = global_mask == 1
    np.testing.assert_array_equal(result[excluded], source[excluded])


def test_fill_is_as_detailed_as_its_surroundings(pruned_image):
    """The weighted mean of all overlapping patches blurred the fill (detail 0.69)."""
    source = np.array(pruned_image)
    holes = white_pixels(source)

    result = patchmatch.inpaint(source, patch_size=3)

    assert evaluation.measure(result, holes)["detail"] > 0.8


def test_no_seam_at_the_hole_border(pruned_image):
    """Known pixels changed during the iterations and restored at the end left a seam.

    The difference across the border of the holes was about twice the difference of
    neighbors on either side of it (seam 2.03).
    """
    source = np.array(pruned_image)
    holes = white_pixels(source)

    result = patchmatch.inpaint(source, patch_size=3)

    assert evaluation.measure(result, holes)["seam"] < 1.9


@pytest.mark.parametrize(
    ("seed", "use_global_mask"), [(6, False), (11, False), (28, True)]
)
def test_stem_is_filled_with_meadow(pruned_image, seed, use_global_mask):
    """The thin stem was filled with dark forest for these seeds.

    It is a small part of the holes, so assert_plausible_fill did not notice.
    """
    source = np.array(pruned_image)
    stem = white_pixels(source)
    stem[:315] = False
    beside = surrounding(stem, width=20)
    beside[:315] = False
    global_mask = None
    if use_global_mask:
        global_mask = np.zeros_like(source[..., 0])
        global_mask[290:, 100:180] = 1

    patchmatch.set_random_seed(seed)
    result = patchmatch.inpaint(source, global_mask=global_mask, patch_size=3)

    assert result[stem].mean() > 0.85 * source[beside].mean()


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

    error = evaluation.coarse_error(result, truth, mask)
    naive_error = evaluation.coarse_error(naive, truth, mask)
    assert error < 0.8 * naive_error, (error, naive_error)
