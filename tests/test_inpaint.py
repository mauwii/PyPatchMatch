#!/usr/bin/env python
"""
Unit tests for inpainting
"""
import unittest

from PIL import Image

from patchmatch import patch_match

img_input = Image.open("./examples/images/forest_pruned.bmp")
result = patch_match.inpaint(img_input, patch_size=3)
img_result = Image.fromarray(result)


class TestInpaint(unittest.TestCase):
    def test_result_type(self, img=img_result):
        assert isinstance(img, Image.Image)

    def test_result_size(self, image=img_input, img=img_result):
        assert img.size == image.size


if __name__ == "__main__":
    unittest.main()
