#!/usr/bin/env python3
# Author : Jiayuan Mao <maojiayuan@gmail.com>, updated by Matthias Wild
#
# Distributed under terms of the MIT license.

from pathlib import Path
from typing import Any

import numpy as np
from numpy import dtype, ndarray
from numpy._typing._array_like import NDArray
from numpy._typing._shape import _AnyShape
from PIL import Image

import patchmatch

IMAGES: Path = Path(__file__).parent / "images"

if __name__ == "__main__":
    patchmatch.set_verbose(verbose=True)
    source: NDArray[Any] = np.array(Image.open(IMAGES / "forest_pruned.bmp"))
    # The global mask covers the plant at the bottom left: it stays in the image, but
    # is not used as a source, so it is not copied into the holes.
    global_mask: ndarray[_AnyShape, dtype[Any]] = np.zeros_like(source[..., 0])
    global_mask[290:, 100:180] = 1
    result: ndarray[_AnyShape, dtype[Any]] = patchmatch.inpaint(
        image=source, global_mask=global_mask, patch_size=3
    )
    Image.fromarray(obj=result).save(fp=IMAGES / "forest_recovered.bmp")
