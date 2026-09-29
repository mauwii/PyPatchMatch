"""Measure how well patchmatch fills holes, to compare versions of the library.

The holes are cut into images whose content is known: forest.bmp and the images listed
in examples/images/SOURCES.md. forest_pruned.bmp and every ``*_pruned.*`` image in the
git-ignored examples/images/local/ are added as object removals without a ground
truth; their pure white pixels are the holes.

Usage:
    uv run python scripts/evaluate_inpainting.py
    uv run python scripts/evaluate_inpainting.py --library main/libpatchmatch.dylib
    uv run python scripts/evaluate_inpainting.py --patch-size 15 --seeds 1 --cases brick
    uv run python scripts/evaluate_inpainting.py --save  # fills to examples/images/local/out

The e2e tests use measure() to check the seam and the detail of forest_pruned.bmp.

Columns, means over the seeds:
    seam    color difference of neighbors across the border of the holes, relative to
            neighbors on either side of it; about 1.0 in the original images
    detail  color difference of neighbors inside the holes, relative to their
            surroundings; 0.85 to 1.1 in the original images, lower is blurrier
    error   mean color error against the original on images reduced to 1/8, which
            compares the structure of the fill rather than its pixels
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

import patchmatch
from patchmatch import _lib, patch_match

IMAGES = Path(__file__).resolve().parents[1] / "examples" / "images"
LOCAL = IMAGES / "local"

# (center y, center x, radius y, radius x) of an elliptic hole
HOLES = {
    "brick": (256, 256, 55, 55),
    "grass": (256, 256, 60, 60),
    "gravel": (256, 256, 60, 60),
    "coffee": (260, 85, 55, 60),
    "chelsea": (52, 215, 24, 40),
}


class Case(NamedTuple):
    name: str
    image: np.ndarray
    holes: np.ndarray
    truth: np.ndarray | None


def load(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def white(image: np.ndarray) -> np.ndarray:
    return (image == 255).all(axis=2)


def ellipse(shape: tuple[int, ...], cy: int, cx: int, ry: int, rx: int) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1


def cut(name: str, truth: np.ndarray, holes: np.ndarray) -> Case:
    image = truth.copy()
    image[holes] = 255
    return Case(name, image, holes, truth)


def cases() -> Iterator[Case]:
    forest = load(IMAGES / "forest.bmp")
    pruned = load(IMAGES / "forest_pruned.bmp")
    yield Case("forest", pruned, white(pruned), None)
    yield cut("trees", forest, ellipse(forest.shape, 70, 560, 45, 55))
    meadow = np.zeros(forest.shape[:2], dtype=bool)
    meadow[318:350, 380:470] = True
    yield cut("meadow", forest, meadow)
    for name, hole in HOLES.items():
        truth = load(IMAGES / f"{name}.png")
        yield cut(name, truth, ellipse(truth.shape, *hole))
    for path in sorted(LOCAL.glob("*_pruned.*")):
        image = load(path)
        yield Case(path.stem.removesuffix("_pruned"), image, white(image), None)


def grow(mask: np.ndarray, radius: int) -> np.ndarray:
    grown = mask.copy()
    for _ in range(radius):
        step = grown.copy()
        step[1:] |= grown[:-1]
        step[:-1] |= grown[1:]
        step[:, 1:] |= grown[:, :-1]
        step[:, :-1] |= grown[:, 1:]
        grown = step
    return grown


def neighbor_difference(
    image: np.ndarray, first: np.ndarray, second: np.ndarray
) -> float:
    """Mean color difference of adjacent pixels, one in ``first``, one in ``second``."""
    pixels = image.astype(int)
    dx = np.abs(pixels[:, 1:] - pixels[:, :-1]).sum(axis=2)
    dy = np.abs(pixels[1:] - pixels[:-1]).sum(axis=2)
    pairs_x = (first[:, 1:] & second[:, :-1]) | (second[:, 1:] & first[:, :-1])
    pairs_y = (first[1:] & second[:-1]) | (second[1:] & first[:-1])
    return float(np.concatenate([dx[pairs_x], dy[pairs_y]]).mean())


def coarse_error(image: np.ndarray, truth: np.ndarray, holes: np.ndarray) -> float:
    def reduce(array: np.ndarray) -> np.ndarray:
        return np.array(Image.fromarray(array).reduce(8)).astype(float)

    cells = reduce(holes.astype(np.uint8) * 255) > 200
    return float(np.abs(reduce(image)[cells] - reduce(truth)[cells]).mean())


def measure(
    result: np.ndarray, holes: np.ndarray, truth: np.ndarray | None = None
) -> dict[str, float]:
    """The columns of the table for the fill ``result`` of ``holes``."""
    known = ~holes
    inner_ring = holes & grow(known, 8)
    outer_ring = known & grow(holes, 8)
    inside = holes & ~grow(known, 2)
    around = grow(holes, 24) & ~grow(holes, 2)
    along = (
        neighbor_difference(result, inner_ring, inner_ring)
        + neighbor_difference(result, outer_ring, outer_ring)
    ) / 2
    values = {
        "seam": neighbor_difference(result, holes, known) / along,
        "detail": neighbor_difference(result, inside, inside)
        / neighbor_difference(result, around, around),
    }
    if truth is not None:
        values["error"] = coarse_error(result, truth, holes)
    return values


def use_library(path: Path) -> None:
    """Load another build of the library instead of the installed one."""
    _lib.find_library = lambda: path
    patch_match._lib = _lib.load_library()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", type=Path, help="library file to load instead")
    parser.add_argument("--patch-size", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--cases", help="comma-separated names, default: all")
    parser.add_argument("--save", action="store_true", help="save the fills of seed 0")
    args = parser.parse_args()

    if args.library:
        use_library(args.library.resolve())
    selected = set(args.cases.split(",")) if args.cases else None
    # A fixed, git-ignored directory rather than one from the command line.
    out = LOCAL / "out"
    if args.save:
        out.mkdir(parents=True, exist_ok=True)

    print(f"patch_size={args.patch_size} seeds={args.seeds}")
    print(f"{'case':10} {'seam':>6} {'detail':>7} {'error':>6} {'time':>7}")
    for case in cases():
        if selected and case.name not in selected:
            continue
        rows = []
        start = time.perf_counter()
        for seed in range(args.seeds):
            patchmatch.set_random_seed(seed)
            mask = case.holes.astype(np.uint8)
            result = patchmatch.inpaint(case.image, mask, patch_size=args.patch_size)
            rows.append(measure(result, case.holes, case.truth))
            if args.save and seed == 0:
                Image.fromarray(result).save(out / f"{case.name}.png")
        seconds = (time.perf_counter() - start) / args.seeds
        mean = {key: np.mean([row[key] for row in rows]) for key in rows[0]}
        error = f"{mean['error']:6.2f}" if "error" in mean else f"{'':6}"
        print(
            f"{case.name:10} {mean['seam']:6.2f} {mean['detail']:7.2f} {error} "
            f"{seconds:6.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
