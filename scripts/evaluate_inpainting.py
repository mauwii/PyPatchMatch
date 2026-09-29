"""Measure how well patchmatch fills holes, to compare versions of the library.

The holes are cut into images whose content is known: forest.bmp and the images listed
in examples/images/SOURCES.md. forest_pruned.bmp and every ``*_pruned.*`` image in the
git-ignored examples/images/local/ are added as object removals without a ground
truth; their pure white pixels are the holes.

Usage:
    uv run python scripts/evaluate_inpainting.py
    uv run python scripts/evaluate_inpainting.py --baseline main/libpatchmatch.dylib
    uv run python scripts/evaluate_inpainting.py --patch-size 15 --seeds 1 --cases brick

Besides the table, it writes examples/images/evaluation.html (git-ignored): the holes,
the fills (of --baseline too) and the originals side by side. It embeds the images of
examples/images/local/ as well, so keep it out of the repository.

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
import base64
import ctypes
import html
import io
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
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
}
COLUMNS = ("seam", "detail", "error")


class Case(NamedTuple):
    name: str
    image: np.ndarray
    holes: np.ndarray
    truth: np.ndarray | None


class Result(NamedTuple):
    values: dict[str, float]  # means over the seeds
    seconds: float  # per seed
    fill: np.ndarray  # of seed 0


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


def load_library(path: Path) -> ctypes.CDLL:
    """Load another build of the library than the installed one."""
    _lib.find_library = lambda: path
    return _lib.load_library()


def evaluate(library: ctypes.CDLL, case: Case, patch_size: int, seeds: int) -> Result:
    patch_match._lib = library
    mask = case.holes.astype(np.uint8)
    rows, fills = [], []
    start = time.perf_counter()
    for seed in range(seeds):
        patchmatch.set_random_seed(seed)
        fills.append(patchmatch.inpaint(case.image, mask, patch_size=patch_size))
        rows.append(measure(fills[-1], case.holes, case.truth))
    seconds = (time.perf_counter() - start) / seeds
    values = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    return Result(values, seconds, fills[0])


def print_row(name: str, results: list[Result]) -> None:
    """One line of the table, with ``old -> new`` for a baseline."""
    cells = []
    for key in COLUMNS:
        found = [
            f"{result.values[key]:.2f}" for result in results if key in result.values
        ]
        cells.append(f"{' -> '.join(found):16}")
    seconds = " -> ".join(f"{result.seconds:.1f}s" for result in results)
    print(f"{name:10} {''.join(cells)}{seconds}", flush=True)


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
.frame { position: relative; overflow: hidden; width: 100%;
  aspect-ratio: var(--crop); background: var(--line); }
.frame img { position: absolute; display: block; max-width: none; height: auto;
  left: var(--left); top: var(--top); width: var(--width); }
body.whole .frame { aspect-ratio: var(--full); }
body.whole .frame img { left: 0; top: 0; width: 100%; }
body.pixels img { image-rendering: pixelated; }
#viewer { position: fixed; inset: 0; z-index: 10; display: flex;
  flex-direction: column; align-items: center; justify-content: center; gap: 10px;
  background: rgb(0 0 0 / 0.9); cursor: zoom-out; }
#viewer[hidden] { display: none; }
#viewer .frame { width: min(94vw, calc(86vh * var(--crop))); cursor: pointer; }
body.whole #viewer .frame { width: min(94vw, calc(86vh * var(--full))); }
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
    figure.querySelector(".frame").cloneNode(true));
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
  if (event.target.closest(".frame")) show(index + 1);
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
for (const name of ["whole", "pixels"]) {
  const box = document.getElementById(name);
  const apply = () => document.body.classList.toggle(name, box.checked);
  box.addEventListener("change", apply);
  apply();  // browsers restore the checkboxes on reload
}
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


def frame(image: np.ndarray, box: tuple[int, int, int, int]) -> str:
    """The image, cut to ``box`` by CSS unless the whole images are shown."""
    height, width = image.shape[:2]
    x, y, w, h = box
    style = (
        f"--crop:{w / h:.5f};--full:{width / height:.5f};--width:{100 * width / w:.4f}%;"
        f"--left:{-100 * x / w:.4f}%;--top:{-100 * y / h:.4f}%"
    )
    source = f"data:image/png;base64,{png(image)}"
    return f'<div class="frame" style="{style}"><img alt="" src="{source}"></div>'


def verdict(key: str, old: float, new: float) -> str:
    """CSS class of ``new`` compared with ``old``: better, worse or none."""
    if abs(new - old) <= (0.1 if key == "time" else 0.02) * abs(old):
        return ""
    if key in ("seam", "detail"):  # best at 1.0, as in the original images
        old, new = abs(old - 1), abs(new - 1)
    return "better" if new < old else "worse"


def cell(key: str, results: list[Result]) -> str:
    if key == "time":
        values = [result.seconds for result in results]
        text = [f"{value:.1f} s" for value in values]
    else:
        values = [result.values[key] for result in results if key in result.values]
        text = [f"{value:.2f}" for value in values]
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
    box = crop_box(case.holes)
    versions = [("holes", case.image, None)]
    versions += [(label, r.fill, r) for label, r in zip(labels, results, strict=True)]
    if case.truth is not None:
        versions.append(("original", case.truth, None))
    figures = "".join(
        f'<figure data-title="{html.escape(case.name)}: {label}">'
        f"{frame(image, box)}<figcaption>{caption(label, result)}</figcaption></figure>"
        for label, image, result in versions
    )
    height, width = case.holes.shape
    meta = f"{width}&times;{height}, holes {case.holes.mean():.1%}"
    name = html.escape(case.name)
    return (
        f'<section id="{name}"><h2>{name}</h2><p class="meta">{meta}</p>'
        f'<div class="versions">{figures}</div></section>'
    )


def write_report(
    rows: list[tuple[Case, list[Result]]], labels: list[str], meta: str
) -> None:
    columns = [*COLUMNS, "time"]
    head = "".join(f"<th>{key}</th>" for key in ["case", *columns])
    body = "".join(
        f'<tr><td><a href="#{html.escape(case.name)}">{html.escape(case.name)}</a></td>'
        + "".join(cell(key, results) for key in columns)
        + "</tr>"
        for case, results in rows
    )
    sections = "".join(section(case, results, labels) for case, results in rows)
    REPORT.write_text(
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>patchmatch evaluation</title><style>{STYLE}</style></head><body>"
        f"<main><header><h1>patchmatch evaluation</h1>{meta}</header>"
        '<div class="controls">'
        '<label><input type="checkbox" id="whole"> whole images</label>'
        '<label><input type="checkbox" id="pixels"> show pixels</label>'
        '<p class="hint">Click an image to enlarge it; click it, &larr; or &rarr; '
        "switches between the versions, Esc closes.</p></div>"
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
        f'{sections}</main><div id="viewer" hidden><div class="slot"></div><p></p>'
        f"</div><script>{SCRIPT}</script></body></html>\n",
        encoding="utf-8",
    )


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
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")

    current = load_library(args.library.resolve()) if args.library else patch_match._lib
    if current is None:
        parser.error("the installed package has no native library, pass --library")
    libraries, labels = [current], ["fill"]
    if args.baseline:
        libraries.insert(0, load_library(args.baseline.resolve()))
        labels = ["baseline", "current"]
    selected = set(args.cases.split(",")) if args.cases else None

    print(f"patch_size={args.patch_size} seeds={args.seeds}")
    print(f"{'case':10} {'seam':16}{'detail':16}{'error':16}time")
    rows = []
    for case in cases():
        if selected and case.name not in selected:
            continue
        results = [
            evaluate(lib, case, args.patch_size, args.seeds) for lib in libraries
        ]
        print_row(case.name, results)
        rows.append((case, results))

    meta = describe(args, labels, [lib._name for lib in libraries])
    write_report(rows, labels, meta)
    print(f"report: {REPORT}")


if __name__ == "__main__":
    main()
