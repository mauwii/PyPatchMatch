# File   : patch_match.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/09/2020
#
# Distributed under terms of the MIT license.
#
# Additional modifications by Kyle Schouviller and Matthias Wild

"""PatchMatch based inpainting.

PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing.
C. Barnes, E. Shechtman, A. Finkelstein and Dan B. Goldman. SIGGRAPH 2009.
"""

from __future__ import annotations

import ctypes
import logging
from collections.abc import Callable
from typing import TypeAlias

import numpy as np
from PIL import Image

from ._lib import CMatT, load_library, np_to_pymat, pymat_to_np
from ._lib import CShapeT as CShapeT  # re-exported for backwards compatibility

__all__ = [
    "inpaint",
    "inpaint_regularity",
    "patchmatch_available",
    "set_random_seed",
    "set_verbose",
]

logger = logging.getLogger(__name__)

ImageLike: TypeAlias = np.ndarray | Image.Image

try:
    _lib: ctypes.CDLL | None = load_library()
except OSError as e:
    _lib = None
    logger.warning("patchmatch failed to load: %s", e)

# True if the native library was loaded successfully.
patchmatch_available = _lib is not None


def _get_lib() -> ctypes.CDLL:
    if _lib is None:
        raise RuntimeError("the patchmatch native library is not available")
    return _lib


def set_random_seed(seed: int) -> None:
    _get_lib().PM_set_random_seed(ctypes.c_uint(seed))


def set_verbose(verbose: bool) -> None:
    _get_lib().PM_set_verbose(ctypes.c_int(verbose))


def inpaint(
    image: ImageLike,
    mask: ImageLike | None = None,
    *,
    global_mask: ImageLike | None = None,
    patch_size: int = 15,
) -> np.ndarray:
    """Fill the masked regions of ``image`` using PatchMatch.

    Args:
        image: 3-channel uint8 RGB/BGR image.
        mask: 1-channel mask of the hole(s) to fill (non-zero = hole). If ``None``,
            all pure white pixels (255, 255, 255) are treated as holes.
        global_mask: 1-channel mask of pixels that must not be used as a source.
        patch_size: patch size for the inpainting algorithm.

    Returns:
        The repaired image, with the same shape as ``image``.
    """
    lib = _get_lib()
    image = _canonize_image_array(image)
    mask = _default_mask(image) if mask is None else _canonize_mask_array(mask)

    if global_mask is None:
        return _call(lib.PM_inpaint, image, mask, ctypes.c_int(patch_size))
    return _call(
        lib.PM_inpaint2,
        image,
        mask,
        _canonize_mask_array(global_mask),
        ctypes.c_int(patch_size),
    )


def inpaint_regularity(
    image: ImageLike,
    mask: ImageLike | None,
    ijmap: np.ndarray,
    *,
    global_mask: ImageLike | None = None,
    patch_size: int = 15,
    guide_weight: float = 0.25,
) -> np.ndarray:
    """Like :func:`inpaint`, additionally guided by a regularity map.

    Args:
        ijmap: HxWx3 float32 array with the regularity coordinates of each pixel.
        guide_weight: weight of the regularity term relative to the patch distance.
    """
    lib = _get_lib()
    image = _canonize_image_array(image)
    mask = _default_mask(image) if mask is None else _canonize_mask_array(mask)

    if not (
        isinstance(ijmap, np.ndarray)
        and ijmap.ndim == 3
        and ijmap.shape[2] == 3
        and ijmap.dtype == np.float32
    ):
        raise ValueError("ijmap must be an HxWx3 float32 array")
    ijmap = np.ascontiguousarray(ijmap)

    args = (ijmap, ctypes.c_int(patch_size), ctypes.c_float(guide_weight))
    if global_mask is None:
        return _call(lib.PM_inpaint_regularity, image, mask, *args)
    return _call(
        lib.PM_inpaint2_regularity,
        image,
        mask,
        _canonize_mask_array(global_mask),
        *args,
    )


def _call(func: Callable[..., CMatT], *args: object) -> np.ndarray:
    """Call ``func`` with arrays converted to pymats and copy the result."""
    c_args = [np_to_pymat(a) if isinstance(a, np.ndarray) else a for a in args]
    ret = func(*c_args)
    try:
        return pymat_to_np(ret)
    finally:
        _get_lib().PM_free_pymat(ret)


def _canonize_image_array(image: ImageLike) -> np.ndarray:
    image = np.asarray(image)
    if not (image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8):
        raise ValueError(
            "image must be an HxWx3 uint8 array, "
            f"got shape {image.shape} and dtype {image.dtype}"
        )
    return np.ascontiguousarray(image)


def _canonize_mask_array(mask: ImageLike) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 2:
        mask = mask[..., np.newaxis]
    if not (mask.ndim == 3 and mask.shape[2] == 1 and mask.dtype == np.uint8):
        raise ValueError(
            "mask must be an HxW or HxWx1 uint8 array, "
            f"got shape {mask.shape} and dtype {mask.dtype}"
        )
    return np.ascontiguousarray(mask)


def _default_mask(image: np.ndarray) -> np.ndarray:
    """Treat all pure white pixels as holes."""
    mask = (image == 255).all(axis=2, keepdims=True).astype(np.uint8)
    return np.ascontiguousarray(mask)
