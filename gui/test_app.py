from __future__ import annotations

import numpy as np
import pytest
from app import MAX_ZOOM, SHAPES, clamp, extent, stroke
from PIL import Image, ImageDraw


@pytest.mark.parametrize("shape", SHAPES)
def test_stroke_leaves_no_gaps(shape):
    mask = Image.new("L", (16, 16))
    draw = ImageDraw.Draw(mask)
    stroke(draw, (5, 6), (10, 6), 4, shape)  # one motion event over several pixels
    stroke(draw, (8, 14), (8, 14), 0, shape)  # a click, zoomed in
    holes = np.asarray(mask) > 0
    assert holes[6, 5:11].all()
    assert holes[14, 8]
    assert holes[13:].sum() == 1
    ys, xs = np.nonzero(holes[:13])
    hx, hy = extent(shape, 4)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (5 - hx, 10 + hx, 6 - hy, 6 + hy)


def test_clamp_keeps_the_view_inside_the_image():
    image, view = (640, 360), (320, 180)  # fits at zoom 0.5
    assert clamp(0.1, -5, -5, image, view) == (0.5, 0, 0)
    assert clamp(2, 1e9, 1e9, image, view) == (2, 480, 270)
    assert clamp(1e9, 0, 0, image, view)[0] == MAX_ZOOM
    assert clamp(1, 0, 0, (4, 4), (640, 640))[0] == 160  # tiny images fill the view
