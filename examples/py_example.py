#!/usr/bin/env python3
# Author : Jiayuan Mao <maojiayuan@gmail.com>, updated by Matthias Wild
#
# Distributed under terms of the MIT license.

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from PIL.ImageFile import ImageFile

import patchmatch

IMAGES: Path = Path(__file__).parent / "images"

if __name__ == "__main__":
    source: ImageFile = Image.open(fp=IMAGES / "forest_pruned.bmp")
    result: NDArray[np.uint8] = patchmatch.inpaint(image=source, patch_size=3)
    Image.fromarray(obj=result).save(fp=IMAGES / "forest_recovered.bmp")
