"""Render examples/images/readme_example.jpg, the picture at the top of the README.

It runs examples/py_example.py and shows, from left to right, the photo forest.bmp,
the input forest_pruned.bmp with the birdhouses painted out in white, and the result.
The PyPI description links the picture at the release tag, so rerun the script and
commit the picture after changes that affect the fills.

Usage:
    uv run python scripts/render_readme_example.py
"""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
from PIL import Image

import patchmatch

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
IMAGES = EXAMPLES / "images"

if __name__ == "__main__":
    patchmatch.set_random_seed(0)
    runpy.run_path(str(EXAMPLES / "py_example.py"), run_name="__main__")
    panels = [
        np.asarray(Image.open(IMAGES / f"forest{suffix}.bmp"))
        for suffix in ("", "_pruned", "_recovered")
    ]
    gap = np.full((panels[0].shape[0], 8, 3), 255, dtype=np.uint8)
    picture = np.hstack([panels[0], gap, panels[1], gap, panels[2]])
    Image.fromarray(picture).save(
        IMAGES / "readme_example.jpg", quality=85, optimize=True
    )
