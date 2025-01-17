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

from PIL import Image

from patchmatch import patch_match

if __name__ == "__main__":
    source = Image.open("./examples/images/forest_pruned.bmp")
    result = patch_match.inpaint(source, patch_size=3)
    Image.fromarray(result).save("./examples/images/forest_recovered.bmp")
