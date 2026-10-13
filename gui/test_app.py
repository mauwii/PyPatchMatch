from __future__ import annotations

import numpy as np
from app import editor_mask


def test_editor_mask_unions_layers_by_alpha():
    a = np.zeros((4, 4, 4), np.uint8)
    b = np.zeros((4, 4, 4), np.uint8)
    a[0, 0, 3] = 1  # faint anti-aliased pixel still counts
    b[3, 3, 3] = 255
    b[2, 2, :3] = 255  # colour without alpha is not a hole
    mask = editor_mask({"background": np.zeros((4, 4, 4), np.uint8), "layers": [a, b]})
    assert mask.sum() == 2 and mask[0, 0] and mask[3, 3]
