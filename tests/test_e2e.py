"""End-to-end tests of the user workflows shown in examples/ and the README."""

import functools
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import patchmatch

IMAGES = Path(__file__).parents[1] / "examples" / "images"
SCRIPTS = Path(__file__).parents[1] / "scripts"

pytestmark = pytest.mark.e2e


def load_evaluation():
    """scripts/evaluate_inpainting.py, whose measures and masks the tests share."""
    path = SCRIPTS / "evaluate_inpainting.py"
    spec = importlib.util.spec_from_file_location("evaluate_inpainting", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = load_evaluation()


def surrounding(mask: np.ndarray, width: int = 10) -> np.ndarray:
    """Band of ``width`` pixels around ``mask``."""
    return evaluation.grow(mask, width) & ~mask


def assert_plausible_fill(
    source: np.ndarray,
    result: np.ndarray,
    excluded: np.ndarray | None = None,
    holes: np.ndarray | None = None,
) -> None:
    """Check an inpainting result of ``source`` without a ground truth image.

    ``excluded`` marks the pixels of a global mask, which are not filled. ``holes``
    defaults to the white pixels of ``source``.
    """
    if excluded is None:
        excluded = np.zeros(source.shape[:2], dtype=bool)
    if holes is None:
        holes = evaluation.white(source)
    holes = holes & ~excluded
    assert holes.any()
    assert result.shape == source.shape
    assert result.dtype == np.uint8

    assert not evaluation.white(result)[holes].any(), "holes were not filled"

    # only the holes are filled
    np.testing.assert_array_equal(result[~holes], source[~holes])

    # the filling blends in with its surroundings
    band = surrounding(holes) & ~excluded
    color_diff = np.abs(result[holes].mean(axis=0) - source[band].mean(axis=0))
    assert color_diff.max() < 10, color_diff


@functools.cache
def fill(
    image: str, hole: tuple[int, int, int, int] | None, patch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """``image`` with white holes and its fill with seed 0, shared by several tests."""
    source = evaluation.load(IMAGES / image)
    if hole is not None:
        source[evaluation.ellipse(source.shape, *hole)] = 255
    patchmatch.set_random_seed(0)
    result = patchmatch.inpaint(source, patch_size=patch_size)
    source.setflags(write=False)
    result.setflags(write=False)
    return source, result


@pytest.fixture
def pruned_image() -> Image.Image:
    return Image.open(IMAGES / "forest_pruned.bmp")


@pytest.fixture(autouse=True)
def seed():
    patchmatch.set_random_seed(0)


def test_remove_white_regions(tmp_path):
    """examples/py_example.py: white pixels are the holes, the result is saved."""
    source, result = fill("forest_pruned.bmp", None, 3)

    output = tmp_path / "forest_recovered.bmp"
    Image.fromarray(result).save(output)
    saved = np.array(Image.open(output))
    np.testing.assert_array_equal(saved, result)
    assert_plausible_fill(source, saved)


def test_fill_does_not_depend_on_threads():
    source, result = fill("forest_pruned.bmp", None, 3)
    patchmatch.set_num_threads(1)
    try:
        np.testing.assert_array_equal(patchmatch.inpaint(source, patch_size=3), result)
    finally:
        patchmatch.set_num_threads(0)


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


def test_restore_aged_photo():
    """Scratches, holes and a torn corner, light but not white, given as a mask.

    The finest levels averaged the colors of sources from elsewhere in the vignetted
    photo, so the torn corner came out lighter (error 4.6 to 6.1 with ten seeds).
    """
    source = Image.open(IMAGES / "camera_damaged.png")
    mask = Image.open(IMAGES / "camera_damaged_mask.png")
    holes = np.array(mask) > 0

    result = patchmatch.inpaint(source, mask, patch_size=3)

    assert_plausible_fill(np.array(source), result, holes=holes)
    truth = evaluation.load(IMAGES / "camera_aged.png")
    error = evaluation.coarse_error(result, truth, holes)
    assert error < 4.2, error


@pytest.mark.parametrize(
    ("image", "hole", "patch_size", "limit"),
    [
        ("forest_pruned.bmp", None, 3, 0.8),
        ("chelsea.png", evaluation.HOLES["chelsea"], 3, 0.7),
        ("chelsea.png", evaluation.HOLES["chelsea"], 15, 0.4),
    ],
)
def test_fill_is_as_detailed_as_its_surroundings(image, hole, patch_size, limit):
    """The weighted mean of all overlapping patches blurred the fill (detail 0.69).

    On the finer levels it kept the fill of the fur in chelsea smooth, which then
    matched smooth sources best (detail 0.40). With patch size 15, the coarser levels
    already took smooth sources, which look like fur once downsampled (detail 0.29).
    """
    source, result = fill(image, hole, patch_size)
    assert evaluation.measure(result, evaluation.white(source))["detail"] > limit


def test_no_part_of_the_fill_is_flat():
    """The finest level linked the left of the hole in chelsea to the blurred fur at
    the border of the image. That part stayed flat (detail 0.30 in the flattest 16x16
    block) while the whole fill reached 0.70.
    """
    source, result = fill("chelsea.png", evaluation.HOLES["chelsea"], 7)
    holes = evaluation.white(source)
    inside = holes & ~evaluation.grow(~holes, 2)
    around = evaluation.grow(holes, 24) & ~evaluation.grow(holes, 2)
    ys, xs = np.nonzero(inside)
    details = []
    for y in range(ys.min(), ys.max() + 1, 16):
        for x in range(xs.min(), xs.max() + 1, 16):
            block = np.zeros_like(inside)
            block[y : y + 16, x : x + 16] = True
            block &= inside
            if block.sum() > 128:
                details.append(evaluation.neighbor_difference(result, block, block))
    reference = evaluation.neighbor_difference(result, around, around)
    assert min(details) / reference > 0.5


@pytest.mark.parametrize(
    ("image", "hole", "patch_size", "limit"),
    [
        ("forest_pruned.bmp", None, 3, 1.5),
        ("chelsea.png", evaluation.HOLES["chelsea"], 15, 2.0),
    ],
)
def test_no_seam_at_the_hole_border(image, hole, patch_size, limit):
    """The border of the holes stood out as an outline.

    Known pixels changed during the iterations and restored at the end made the
    difference across the border about twice the difference of neighbors on either
    side of it (seam 2.03). Keeping them still left the colors of the fill off along
    the border: seam 1.73 for forest with patch size 3, 3.24 for chelsea with 15.
    """
    source, result = fill(image, hole, patch_size)
    assert evaluation.measure(result, evaluation.white(source))["seam"] < limit


@pytest.mark.parametrize(
    ("seed", "use_global_mask"), [(6, False), (11, False), (28, True)]
)
def test_stem_is_filled_with_meadow(pruned_image, seed, use_global_mask):
    """The thin stem was filled with dark forest for these seeds.

    It is a small part of the holes, so assert_plausible_fill did not notice.
    """
    source = np.array(pruned_image)
    stem = evaluation.white(source)
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


@pytest.mark.parametrize(
    ("name", "column", "limit"), [("tripod", "edges", 3.0), ("pole", "error", 24.0)]
)
def test_line_goes_on_through_the_hole(name, column, limit):
    """The coarser levels lost the leg of the tripod and the pole of the birdhouses,
    and the fills broke them off: edges 4.2 to 4.8 and error 27 to 33 with ten seeds.
    """
    filename, hole = evaluation.LINES[name]
    truth = evaluation.load(IMAGES / filename)
    holes = evaluation.ellipse(truth.shape, *hole)
    source = truth.copy()
    source[holes] = 255

    result = patchmatch.inpaint(source, holes.astype(np.uint8), patch_size=3)

    assert evaluation.measure(result, holes, truth)[column] < limit


def test_line_goes_on_beside_a_larger_hole():
    """The peel ran on the levels of the largest hole only: a hole in the sky broke the
    leg of the tripod off (edges 8.7 to 11.9 with ten seeds, 1.4 to 2.6 without it).
    """
    truth = evaluation.load(IMAGES / "camera_aged.png")
    leg = evaluation.ellipse(truth.shape, 398, 344, 11, 11)
    holes = leg | evaluation.ellipse(truth.shape, 60, 420, 30, 30)
    source = truth.copy()
    source[holes] = 255

    result = patchmatch.inpaint(source, holes.astype(np.uint8), patch_size=3)

    assert evaluation.edge_error(result, truth, leg) < 5.0


def test_line_goes_on_through_a_small_hole():
    """The coarser fill lost the leg of the tripod, and the guide's link copied the
    background across it: edges 3.0 to 10.3 with ten seeds, above 8 in nine of them
    (2.5 to 3.6 when the link needs a source as varied as the best fit's).
    """
    truth = evaluation.load(IMAGES / "camera_aged.png")
    leg = evaluation.ellipse(truth.shape, 425, 261, 14, 14)
    source = truth.copy()
    source[leg] = 255

    result = patchmatch.inpaint(source, leg.astype(np.uint8), patch_size=3)

    assert evaluation.edge_error(result, truth, leg) < 5.0


def test_lines_go_on_through_many_holes():
    """The coarser fill lost the column and the legs of the tripod in the largest holes
    of tripod-holes, and the near ties closest to it ended or bent them: edges of the
    column 7.7 to 8.7 and of the legs up to 5.9 and 8.4 with ten seeds (2.7 to 6.5, 3.5
    and 3.0 with sources as varied as the best fit's and a search nearby).
    """
    truth = evaluation.load(IMAGES / "camera_aged.png")
    holes = np.logical_or.reduce(
        [
            evaluation.ellipse(truth.shape, y, x, r, r)
            for y, x, r in evaluation.TRIPOD_HOLES
        ]
    )
    source = truth.copy()
    source[holes] = 255

    result = patchmatch.inpaint(source, holes.astype(np.uint8), patch_size=3)

    column = evaluation.ellipse(truth.shape, 452, 294, 15, 15)
    assert evaluation.edge_error(result, truth, column) < 7.0
    for y, x, r in [(442, 367, 15), (425, 261, 14)]:
        leg = evaluation.ellipse(truth.shape, y, x, r, r)
        assert evaluation.edge_error(result, truth, leg) < 4.0


def test_explicit_mask_matches_implicit_white_mask(pruned_image):
    """README: an explicit mask behaves like the default mask of white pixels."""
    holes = evaluation.white(np.array(pruned_image))
    mask = Image.fromarray(holes.astype(np.uint8) * 255)

    explicit = patchmatch.inpaint(pruned_image, mask, patch_size=3)

    np.testing.assert_array_equal(explicit, fill("forest_pruned.bmp", None, 3)[1])


# Regions of pure background in forest_pruned.bmp as (y, x, height, width), with the
# patch size to fill them.
BACKGROUND_REGIONS = [
    ((40, 520, 60, 60), 3),
    ((25, 505, 90, 110), 15),
]


@pytest.mark.parametrize(("region", "patch_size"), BACKGROUND_REGIONS)
def test_reconstructs_known_background(pruned_image, region, patch_size):
    """Cut a hole into known background and compare with the original pixels.

    With patch size 15, the coarser levels found no sources inside the image and
    outside the hole, and the trees got a lighter patch (0.90 of the naive error).
    """
    truth = np.array(pruned_image)
    y, x, height, width = region
    mask = np.zeros(truth.shape[:2], dtype=bool)
    mask[y : y + height, x : x + width] = True
    assert not evaluation.white(truth)[mask].any()

    source = truth.copy()
    source[mask] = 255
    result = patchmatch.inpaint(source, mask.astype(np.uint8), patch_size=patch_size)

    # baseline: fill the hole with the mean color of its surroundings
    naive = source.copy()
    naive[mask] = source[surrounding(mask)].mean(axis=0).astype(np.uint8)

    error = evaluation.coarse_error(result, truth, mask)
    naive_error = evaluation.coarse_error(naive, truth, mask)
    assert error < 0.8 * naive_error, (error, naive_error)
