#!/usr/bin/env python3
# Author : Jiayuan Mao <maojiayuan@gmail.com>, updated by Matthias Wild
#
# Distributed under terms of the MIT license.

from pathlib import Path

from PIL import Image

import patchmatch

IMAGES = Path(__file__).parent / "images"

if __name__ == "__main__":
    source = Image.open(IMAGES / "forest_pruned.bmp")
    result = patchmatch.inpaint(source, patch_size=3)
    Image.fromarray(result).save(IMAGES / "forest_recovered.bmp")
