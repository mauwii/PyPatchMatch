#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/09/2020
#
# updated by Matthias Wild <mauwii@outlook.de>
#
# Distributed under terms of the MIT license.

import numpy as np
from PIL import Image

from patchmatch import patch_match

if __name__ == "__main__":
    patch_match.set_verbose(True)
    source = Image.open("./examples/images/forest_pruned.bmp")
    source = np.array(source)
    source[:100, :100] = 255
    global_mask = np.zeros_like(source[..., 0])
    global_mask[:100, :100] = 1
    result = patch_match.inpaint(source, global_mask=global_mask, patch_size=3)
    Image.fromarray(result).save("./examples/images/forest_recovered.bmp")
