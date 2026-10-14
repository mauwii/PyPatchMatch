"""Measure how well patchmatch fills holes, to compare versions of the library.

The holes are cut into images whose content is known: forest.bmp and the images listed
in examples/images/SOURCES.md. forest_pruned.bmp and every ``*_pruned.*`` image in the
git-ignored examples/images/local/ are added as object removals without a ground
truth; their pure white pixels are the holes. camera restores the damage of
camera_damaged.png, which camera_damaged_mask.png marks, against camera_aged.png; its
error covers the larger holes and the torn corner, the scratches are narrower than the
cells it compares. grass-edge cuts a band at the right border, as in outpainting;
brick-large and grass-large cut a circle of radius 242 into the center, 70 % of the
image. pattern repeats almost exactly, so its error tells a fill that continues it (2 to
6) from a broken one; pattern-wide widens the hole to a radius of 60, where patch size 3
mixed the rings before 2.2 (error 37). forest-global and brick-global add a global mask,
darkened in the report: the plant of examples/py_example_global_mask.py and the left
third of brick.png.

Usage:
    uv run python scripts/evaluate_inpainting.py
    uv run python scripts/evaluate_inpainting.py --baseline main/libpatchmatch.dylib
    uv run python scripts/evaluate_inpainting.py --patch-size 15 --seeds 1 --cases brick

The inpaintings run in parallel processes, one per CPU unless --jobs says otherwise.
They slow each other down, so time a speed-up with --jobs 1.

--markdown prints the table as Markdown and moves the plain one to stderr; the CI
workflow "Fill quality" redirects it to the job summary.

Besides the table, it writes examples/images/evaluation.html (git-ignored): the holes,
the fills (of --baseline too) and the originals side by side, cut to the holes. It
embeds the images of examples/images/local/ as well, so keep it out of the repository.

The e2e tests use measure() to check the seam and the detail of forest_pruned.bmp.

Columns, means over the seeds:
    seam    color difference of neighbors across the border of the holes, relative to
            neighbors on either side of it; about 1.0 in the original images
    detail  color difference of neighbors inside the holes, relative to their
            surroundings; 0.85 to 1.1 in the original images, lower is blurrier
    error   mean color error against the original on images reduced to 1/8, which
            compares the structure of the fill rather than its pixels; it rates a flat
            fill better than a texture whose structures are shifted
    edges   difference of the gradient directions from the original, in histograms
            weighted by magnitude per 32x32 cell of the holes; lowest for the right
            texture, also when shifted, highest for a flat fill
    variety difference of the 8x8 patches of the holes, without their mean color, to
            the closest one that does not overlap them, relative to the original;
            lower where the fill repeats its sources or blurs (the tiles of
            grass-large: about 0.25), higher for fragments
    fills   with --baseline: whether the fills of all seeds are byte-identical to the
            baseline's, as they have to be after a change that only makes it faster
"""

from __future__ import annotations

import argparse
import base64
import functools
import hashlib
import html
import io
import sys
import time
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, TextIO

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

import patchmatch
from patchmatch import _lib, patch_match

IMAGES = Path(__file__).resolve().parents[1] / "examples" / "images"
LOCAL = IMAGES / "local"
REPORT = IMAGES / "evaluation.html"

# (center y, center x, radius y, radius x) of an elliptic hole
HOLES = {
    "brick": (256, 256, 55, 55),
    "grass": (256, 256, 60, 60),
    "gravel": (256, 256, 60, 60),
    "coffee": (260, 85, 55, 60),
    "chelsea": (52, 215, 24, 40),
    "pattern": (320, 320, 40, 40),
}
COLUMNS = ("seam", "detail", "error", "edges", "variety")


class Case(NamedTuple):
    name: str
    image: np.ndarray
    holes: np.ndarray
    truth: np.ndarray | None
    global_mask: np.ndarray | None = None


class Result(NamedTuple):
    values: dict[str, float]  # means over the seeds
    seconds: float  # per seed
    fill: np.ndarray  # of seed 0
    digest: str  # of the fills of all seeds


def load(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def white(image: np.ndarray) -> np.ndarray:
    return (image == 255).all(axis=2)


def ellipse(shape: tuple[int, ...], cy: int, cx: int, ry: int, rx: int) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1


def cut(
    name: str,
    truth: np.ndarray,
    holes: np.ndarray,
    global_mask: np.ndarray | None = None,
) -> Case:
    image = truth.copy()
    image[holes] = 255
    return Case(name, image, holes, truth, global_mask)


def cases() -> Iterator[Case]:
    forest = load(IMAGES / "forest.bmp")
    pruned = load(IMAGES / "forest_pruned.bmp")
    yield Case("forest", pruned, white(pruned), None)
    damaged = load(IMAGES / "camera_damaged.png")
    damage = np.array(Image.open(IMAGES / "camera_damaged_mask.png")) > 0
    yield Case("camera", damaged, damage, load(IMAGES / "camera_aged.png"))
    yield cut("trees", forest, ellipse(forest.shape, 70, 560, 45, 55))
    meadow = np.zeros(forest.shape[:2], dtype=bool)
    meadow[318:350, 380:470] = True
    yield cut("meadow", forest, meadow)
    for name, hole in HOLES.items():
        truth = load(IMAGES / f"{name}.png")
        yield cut(name, truth, ellipse(truth.shape, *hole))
    grass = load(IMAGES / "grass.png")
    edge = np.zeros(grass.shape[:2], dtype=bool)
    edge[:, -64:] = True
    yield cut("grass-edge", grass, edge)
    for name in ("brick", "grass"):
        truth = load(IMAGES / f"{name}.png")
        yield cut(f"{name}-large", truth, ellipse(truth.shape, 256, 256, 242, 242))
    pattern = load(IMAGES / "pattern.png")
    yield cut("pattern-wide", pattern, ellipse(pattern.shape, 320, 320, 60, 60))
    plant = np.zeros(pruned.shape[:2], dtype=bool)
    plant[290:, 100:180] = True
    yield Case("forest-global", pruned, white(pruned), None, plant)
    brick = load(IMAGES / "brick.png")
    left = np.zeros(brick.shape[:2], dtype=bool)
    left[:, :180] = True
    yield cut("brick-global", brick, ellipse(brick.shape, *HOLES["brick"]), left)
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


EDGE_CELL = 32
EDGE_BINS = 8


def edge_error(image: np.ndarray, truth: np.ndarray, holes: np.ndarray) -> float:
    """Difference of the gradient directions, weighted by magnitude, per cell of holes."""
    ys, xs = np.nonzero(holes)
    cells = (ys // EDGE_CELL) * (holes.shape[1] // EDGE_CELL + 1) + xs // EDGE_CELL
    length = (int(cells.max()) + 1) * EDGE_BINS

    def histograms(array: np.ndarray) -> np.ndarray:
        gy, gx = np.gradient(array.astype(float).mean(axis=2))
        bins = (np.arctan2(gy, gx) % np.pi * EDGE_BINS / np.pi).astype(int) % EDGE_BINS
        weights = np.hypot(gx, gy)[holes]
        return np.bincount(cells * EDGE_BINS + bins[holes], weights, length)

    return float(np.abs(histograms(image) - histograms(truth)).sum() / len(ys))


PATCH = 8


def patches(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The PATCHxPATCH patches inside ``mask`` on a grid, flattened, and their corners."""
    grid = (slice(None, None, PATCH // 2),) * 2
    windows = sliding_window_view(image, (PATCH, PATCH), axis=(0, 1))[grid]
    inside = sliding_window_view(mask, (PATCH, PATCH))[grid].all(axis=(2, 3))
    flat = windows[inside].reshape(int(inside.sum()), 3, -1).astype(np.float32)
    # without their mean color, which the votes shift for repeated sources too
    flat -= flat.mean(axis=2, keepdims=True)
    return flat.reshape(len(flat), -1), np.argwhere(inside) * (PATCH // 2)


def variety(image: np.ndarray, truth: np.ndarray, holes: np.ndarray) -> float:
    def mean(array: np.ndarray) -> float:
        """Mean distance of the patches to the closest one that does not overlap them."""
        found, corners = patches(array, holes)
        norms = (found**2).sum(axis=1)
        closest = []
        for start in range(0, len(found), 512):  # bounds the memory of the distances
            chunk, where = found[start : start + 512], corners[start : start + 512]
            distances = norms[start : start + 512, None] + norms - 2 * chunk @ found.T
            distances[np.abs(where[:, None] - corners).max(axis=2) < PATCH] = np.inf
            closest.append(distances.min(axis=1))
        return float(np.sqrt(np.maximum(np.concatenate(closest), 0)).mean())

    return mean(image) / mean(truth)


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
        values["edges"] = edge_error(result, truth, holes)
        # holes thinner than a patch, such as scratches, have none to compare
        if sliding_window_view(holes, (PATCH, PATCH)).all(axis=(2, 3)).any():
            values["variety"] = variety(result, truth, holes)
    return values


Seed = tuple[np.ndarray, dict[str, float], float]  # fill, measures, seconds

library = functools.cache(_lib.load_library)


def fill_seed(path: str, case: Case, patch_size: int, seed: int) -> Seed:
    """One inpainting, in a worker process."""
    # processes, not threads: the seed and patch_match._lib are global
    patch_match._lib = library(Path(path))
    patchmatch.set_random_seed(seed)
    start = time.perf_counter()
    fill = patchmatch.inpaint(
        case.image,
        case.holes.astype(np.uint8),
        global_mask=case.global_mask,
        patch_size=patch_size,
    )
    seconds = time.perf_counter() - start
    return fill, measure(fill, case.holes, case.truth), seconds


def evaluate(jobs: list[Future[Seed]]) -> Result:
    """The result of one case and library from the jobs of all seeds, in order."""
    fills, rows, seconds = zip(*(job.result() for job in jobs), strict=True)
    values = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    digest = hashlib.sha256(b"".join(fill.tobytes() for fill in fills)).hexdigest()
    return Result(values, float(np.mean(seconds)), fills[0], digest)


def identical(results: list[Result]) -> str:
    """Whether the fills match those of the baseline, empty without one."""
    if len(results) < 2:
        return ""
    return "identical" if results[0].digest == results[1].digest else "changed"


def report_columns(rows: list[tuple[Case, list[Result]]]) -> list[str]:
    baseline = any(len(results) > 1 for _, results in rows)
    return [*COLUMNS, "time", *(["fills"] if baseline else [])]


def print_row(name: str, results: list[Result], out: TextIO) -> None:
    """One line of the table, with ``old -> new`` for a baseline."""
    cells = []
    for key in COLUMNS:
        found = [
            f"{result.values[key]:.2f}" for result in results if key in result.values
        ]
        cells.append(f"{' -> '.join(found):16}")
    seconds = " -> ".join(f"{result.seconds:.1f}s" for result in results)
    line = f"{name:14} {''.join(cells)}{seconds:16}{identical(results)}"
    print(line.rstrip(), file=out, flush=True)


# Report ------------------------------------------------------------------------------

STYLE = """
:root { color-scheme: light dark; --bg: #f7f7f5; --fg: #1d1d1b; --muted: #6f6f6a;
  --card: #fff; --line: #e3e3de; --good: #1b7f3b; --bad: #b42318; }
@media (prefers-color-scheme: dark) { :root { --bg: #161615; --fg: #efefec;
  --muted: #a3a39c; --card: #22221f; --line: #3a3a36; --good: #5bd181;
  --bad: #ff8a7a; } }
* { box-sizing: border-box; }
body { margin: 0; padding: 0 16px 64px; background: var(--bg); color: var(--fg);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1440px; margin: 0 auto; }
header { padding: 24px 0 8px; }
h1 { font-size: 22px; margin: 0 0 4px; }
.meta, .hint { color: var(--muted); font-size: 13px; margin: 0; }
.controls { position: sticky; top: 0; z-index: 1; display: flex; flex-wrap: wrap;
  gap: 8px 24px; padding: 10px 0; background: var(--bg);
  border-bottom: 1px solid var(--line); }
.controls label { cursor: pointer; }
table { border-collapse: collapse; margin: 16px 0 28px;
  font-variant-numeric: tabular-nums; }
th, td { padding: 4px 18px 4px 0; text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 500; border-bottom: 1px solid var(--line); }
a { color: inherit; }
.better { color: var(--good); font-weight: 600; }
.worse { color: var(--bad); font-weight: 600; }
section { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  padding: 12px 16px 16px; margin: 0 0 20px; }
section h2 { font-size: 17px; margin: 0; display: inline; }
section .meta { display: inline; margin-left: 10px; }
.versions { display: grid; gap: 12px; margin-top: 10px;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 240px), 1fr)); }
figure { margin: 0; cursor: zoom-in; }
figcaption { font-size: 13px; color: var(--muted); margin-top: 4px;
  font-variant-numeric: tabular-nums; }
figcaption b { color: var(--fg); font-weight: 600; margin-right: 6px; }
figure img { display: block; width: 100%; height: auto; }
body.pixels img { image-rendering: pixelated; }
#viewer { position: fixed; inset: 0; z-index: 10; display: flex;
  flex-direction: column; align-items: center; justify-content: center; gap: 10px;
  background: rgb(0 0 0 / 0.9); cursor: zoom-out; }
#viewer[hidden] { display: none; }
#viewer img { width: 94vw; height: 86vh; object-fit: contain; cursor: pointer; }
#viewer p { color: #ddd; font-size: 14px; margin: 0; }
"""

SCRIPT = """
const viewer = document.getElementById("viewer");
let figures = [];
let index = 0;
function show(i) {
  index = (i + figures.length) % figures.length;
  const figure = figures[index];
  viewer.querySelector(".slot").replaceChildren(
    figure.querySelector("img").cloneNode());
  viewer.querySelector("p").textContent = figure.dataset.title
    + "  (" + (index + 1) + "/" + figures.length + ")";
}
for (const figure of document.querySelectorAll("figure")) {
  figure.addEventListener("click", () => {
    figures = Array.from(figure.parentElement.querySelectorAll("figure"));
    viewer.hidden = false;
    show(figures.indexOf(figure));
  });
}
viewer.addEventListener("click", (event) => {
  if (event.target.closest("img")) show(index + 1);
  else viewer.hidden = true;
});
document.addEventListener("keydown", (event) => {
  if (viewer.hidden) return;
  if (event.key === "ArrowRight" || event.key === " ") show(index + 1);
  else if (event.key === "ArrowLeft") show(index - 1);
  else if (event.key === "Escape") viewer.hidden = true;
  else return;
  event.preventDefault();
});
const pixels = document.getElementById("pixels");
const apply = () => document.body.classList.toggle("pixels", pixels.checked);
pixels.addEventListener("change", apply);
apply();  // browsers restore the checkbox on reload
"""


def png(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def crop_box(holes: np.ndarray) -> tuple[int, int, int, int]:
    """(x, y, width, height) around the holes, or the whole image if they fill most."""
    height, width = holes.shape
    if not holes.any():
        return 0, 0, width, height
    ys, xs = np.nonzero(holes)
    margin = max(16, (max(np.ptp(ys), np.ptp(xs)) + 1) // 4)
    x0, x1 = max(0, xs.min() - margin), min(width, xs.max() + 1 + margin)
    y0, y1 = max(0, ys.min() - margin), min(height, ys.max() + 1 + margin)
    if (x1 - x0) * (y1 - y0) > 0.6 * width * height:
        return 0, 0, width, height
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def verdict(key: str, old: float, new: float) -> str:
    """CSS class of ``new`` compared with ``old``: better, worse or none."""
    if abs(new - old) <= (0.1 if key == "time" else 0.02) * abs(old):
        return ""
    if key in ("seam", "detail", "variety"):  # best at 1.0, as in the original images
        old, new = abs(old - 1), abs(new - 1)
    return "better" if new < old else "worse"


def column(key: str, results: list[Result]) -> tuple[list[float], list[str]]:
    """The values of a column and their texts, one per library that has the value."""
    if key == "time":
        values = [result.seconds for result in results]
        return values, [f"{value:.1f} s" for value in values]
    values = [result.values[key] for result in results if key in result.values]
    return values, [f"{value:.2f}" for value in values]


def cell(key: str, results: list[Result]) -> str:
    if key == "fills":
        return f"<td>{identical(results)}</td>"
    values, text = column(key, results)
    if not values:
        return "<td></td>"
    if len(values) == 1:
        return f"<td>{text[0]}</td>"
    css = verdict(key, values[0], values[1])
    return f'<td>{text[0]} &rarr; <span class="{css}">{text[1]}</span></td>'


def caption(label: str, result: Result | None) -> str:
    parts = [f"<b>{label}</b>"]
    if result is not None:
        parts += [f"{key} {value:.2f}" for key, value in result.values.items()]
    return " ".join(parts)


def section(case: Case, results: list[Result], labels: list[str]) -> str:
    x, y, w, h = crop_box(case.holes)
    shown = case.image.copy()
    if case.global_mask is not None:
        shown[case.global_mask] //= 2
    versions = [("holes", shown, None)]
    versions += [(label, r.fill, r) for label, r in zip(labels, results, strict=True)]
    if case.truth is not None:
        versions.append(("original", case.truth, None))
    figures = "".join(
        f'<figure data-title="{html.escape(case.name)}: {label}">'
        f'<img alt="" src="data:image/png;base64,{png(image[y : y + h, x : x + w])}">'
        f"<figcaption>{caption(label, result)}</figcaption></figure>"
        for label, image, result in versions
    )
    height, width = case.holes.shape
    meta = f"{width}&times;{height}, holes {case.holes.mean():.1%}"
    if case.global_mask is not None:
        meta += f", global mask {case.global_mask.mean():.1%}"
    name = html.escape(case.name)
    return (
        f'<section id="{name}"><h2>{name}</h2><p class="meta">{meta}</p>'
        f'<div class="versions">{figures}</div></section>'
    )


def write_report(
    rows: list[tuple[Case, list[Result]]], labels: list[str], meta: str
) -> None:
    columns = report_columns(rows)
    head = "".join(f"<th>{key}</th>" for key in ["case", *columns])
    body = "".join(
        f'<tr><td><a href="#{html.escape(case.name)}">{html.escape(case.name)}</a></td>'
        + "".join(cell(key, results) for key in columns)
        + "</tr>"
        for case, results in rows
    )
    sections = "".join(section(case, results, labels) for case, results in rows)
    with REPORT.open("w", encoding="utf-8") as report:
        report.write(
            "<!doctype html>\n"
            '<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>patchmatch evaluation</title><style>{STYLE}</style></head><body>"
            f"<main><header><h1>patchmatch evaluation</h1>{meta}</header>"
            '<div class="controls">'
            '<label><input type="checkbox" id="pixels"> show pixels</label>'
            '<p class="hint">Click an image to enlarge it; click it, &larr; or &rarr; '
            "switches between the versions, Esc closes.</p></div>"
            f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
            f'{sections}</main><div id="viewer" hidden><div class="slot"></div><p></p>'
            f"</div><script>{SCRIPT}</script></body></html>\n"
        )


MARKS = {"better": " 🟢", "worse": " 🔴", "": ""}


def markdown_cell(key: str, results: list[Result]) -> str:
    if key == "fills":
        return identical(results)
    values, text = column(key, results)
    if len(values) < 2:
        return "".join(text)
    return f"{text[0]} → {text[1]}{MARKS[verdict(key, values[0], values[1])]}"


def markdown(rows: list[tuple[Case, list[Result]]], title: str) -> str:
    columns = report_columns(rows)
    lines = [
        f"### Fill quality ({title})",
        "",
        f"| case | {' | '.join(columns)} |",
        f"| --- |{' ---: |' * len(columns)}",
    ]
    for case, results in rows:
        cells = [markdown_cell(key, results) for key in columns]
        lines.append(f"| {case.name} | {' | '.join(cells)} |")
    if "fills" in columns:
        same = sum(identical(results) == "identical" for _, results in rows)
        lines += [
            "",
            f"The fills of all seeds are byte-identical to the baseline in {same} of "
            f"{len(rows)} cases.",
        ]
    lines += [
        "",
        "Means over the seeds. seam, detail and variety are best at 1.0, as in the "
        "original images, error and edges at 0. With a baseline: baseline → current, "
        "🟢 better and 🔴 worse by more than 2 % (time: 10 %).",
    ]
    return "\n".join(lines)


def describe(args: argparse.Namespace, labels: list[str], paths: list[str]) -> str:
    lines = [
        f"{datetime.now():%Y-%m-%d %H:%M}, patch_size {args.patch_size}, "
        f"{args.seeds} seed(s), fills of seed 0",
        *(f"{label}: {path}" for label, path in zip(labels, paths, strict=True)),
    ]
    return "".join(f'<p class="meta">{html.escape(line)}</p>' for line in lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", type=Path, help="library file to load instead")
    parser.add_argument("--baseline", type=Path, help="library file to compare with")
    parser.add_argument("--patch-size", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--cases", help="comma-separated names, default: all")
    parser.add_argument("--jobs", type=int, help="processes, default: one per CPU")
    parser.add_argument(
        "--markdown", action="store_true", help="print the table as Markdown"
    )
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")

    current = (
        _lib.load_library(args.library.resolve()) if args.library else patch_match._lib
    )
    if current is None:
        parser.error("the installed package has no native library, pass --library")
    libraries, labels = [current], ["fill"]
    if args.baseline:
        libraries.insert(0, _lib.load_library(args.baseline.resolve()))
        labels = ["baseline", "current"]
    paths = [lib._name for lib in libraries]
    selected = set(args.cases.split(",")) if args.cases else None
    chosen = [case for case in cases() if not selected or case.name in selected]

    out = sys.stderr if args.markdown else sys.stdout
    print(f"patch_size={args.patch_size} seeds={args.seeds}", file=out)
    fills = "fills" if args.baseline else ""
    header = f"{'case':14} {''.join(f'{key:16}' for key in COLUMNS)}{'time':16}{fills}"
    print(header.rstrip(), file=out)
    rows = []
    pool = ProcessPoolExecutor(args.jobs)
    try:
        # all at once, the seeds outside, so that the libraries run under equal load
        jobs = [
            {
                (index, seed): pool.submit(fill_seed, path, case, args.patch_size, seed)
                for seed in range(args.seeds)
                for index, path in enumerate(paths)
            }
            for case in chosen
        ]
        for case, futures in zip(chosen, jobs, strict=True):
            results = [
                evaluate([futures[index, seed] for seed in range(args.seeds)])
                for index in range(len(paths))
            ]
            print_row(case.name, results, out)
            rows.append((case, results))
    finally:
        pool.shutdown(cancel_futures=True)  # Ctrl+C stops after the running jobs

    meta = describe(args, labels, paths)
    write_report(rows, labels, meta)
    print(f"report: {REPORT}", file=out)
    if args.markdown:
        print(markdown(rows, f"patch_size {args.patch_size}, {args.seeds} seed(s)"))


if __name__ == "__main__":
    main()
