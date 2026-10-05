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
import operator
from collections.abc import Callable
from typing import SupportsIndex, TypeAlias

import numpy as np
from PIL import Image

# CMatT and CShapeT are re-exported for backwards compatibility
from ._lib import CMatT as CMatT
from ._lib import CShapeT as CShapeT
from ._lib import load_library, np_to_pymat, pymat_to_np

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
except (OSError, AttributeError) as e:
    # AttributeError: a library without the PM_* functions, e.g. one that 1.x
    # compiled into the package directory and that pip does not remove on upgrade
    _lib = None
    logger.warning("patchmatch failed to load: %s", e)

# True if the native library was loaded successfully.
patchmatch_available = _lib is not None

# The native code computes 2 * patch_size + 1 as a C int.
_MAX_PATCH_SIZE = (2**31 - 2) // 2

# The native code takes a C float, which turns larger weights into infinity.
_MAX_GUIDE_WEIGHT = float(np.finfo(np.float32).max)


def _get_lib() -> ctypes.CDLL:
    if _lib is None:
        raise RuntimeError("the patchmatch native library is not available")
    return _lib


def set_random_seed(seed: int) -> None:
    """Set the seed of the randomized search of all following inpaintings.

    Every inpainting starts from this seed, so equal inputs give equal results. The
    native code keeps it as a 32-bit unsigned int, so ``seed`` is taken modulo
    ``2**32``: ``2**32`` behaves like 0 and -1 like ``2**32 - 1``.
    """
    _get_lib().PM_set_random_seed(ctypes.c_uint(seed))


def set_verbose(verbose: bool) -> None:
    """Print the progress of all following inpaintings to stderr if ``verbose``."""
    _get_lib().PM_set_verbose(ctypes.c_int(verbose))


def inpaint(
    image: ImageLike,
    mask: ImageLike | None = None,
    *,
    global_mask: ImageLike | None = None,
    patch_size: SupportsIndex = 15,
) -> np.ndarray:
    """Fill the masked regions of ``image`` using PatchMatch.

    Args:
        image: 3-channel uint8 RGB/BGR image.
        mask: 1-channel uint8 or bool mask of the hole(s) to fill (non-zero = hole),
            with the same height and width as ``image``. If ``None``, all pure
            white pixels (255, 255, 255) are treated as holes.
        global_mask: mask like ``mask`` of pixels that are neither filled nor used as
            a source; they keep their values from ``image``.
        patch_size: radius of the compared patches, which span
            ``(2 * patch_size + 1) ** 2`` pixels. Larger patches follow larger
            structures but are much slower; the examples use 3.

    Returns:
        The repaired image, with the same shape as ``image``. Only the holes are
        filled; all other pixels keep their values from ``image``.
    """
    lib = _get_lib()
    patch_size = _check_patch_size(patch_size)
    image, mask, global_mask = _prepare_inputs(image, mask, global_mask)
    if not _has_holes(mask, global_mask):
        return image.copy()

    if global_mask is None:
        return _call(lib.PM_inpaint, image, mask, ctypes.c_int(patch_size))
    return _call(lib.PM_inpaint2, image, mask, global_mask, ctypes.c_int(patch_size))


def inpaint_regularity(
    image: ImageLike,
    mask: ImageLike | None,
    ijmap: np.ndarray,
    *,
    global_mask: ImageLike | None = None,
    patch_size: SupportsIndex = 15,
    guide_weight: float = 0.25,
) -> np.ndarray:
    """Like :func:`inpaint`, additionally guided by a regularity map.

    Args:
        ijmap: HxWx3 float32 array of finite values with the regularity coordinates
            of each pixel in the first two channels; the third channel is unused.
            A map of a different size is scaled to the image.
        guide_weight: non-negative weight of the regularity term relative to the
            patch distance.
    """
    lib = _get_lib()
    patch_size = _check_patch_size(patch_size)
    image, mask, global_mask = _prepare_inputs(image, mask, global_mask)

    if not (
        isinstance(ijmap, np.ndarray)
        and ijmap.ndim == 3
        and ijmap.shape[2] == 3
        and ijmap.dtype == np.float32
        and ijmap.size > 0
    ):
        raise ValueError("ijmap must be a non-empty HxWx3 float32 array")
    # NaN and infinity turn into out-of-range patch distances in the native code
    if not np.isfinite(ijmap).all():
        raise ValueError("ijmap must only contain finite values")
    ijmap = np.ascontiguousarray(ijmap)

    # the native code divides by 1 + guide_weight and indexes a table with the result;
    # the comparison also rejects NaN
    guide_weight = float(guide_weight)
    if not 0 <= guide_weight <= _MAX_GUIDE_WEIGHT:
        raise ValueError(
            f"guide_weight must be between 0 and {_MAX_GUIDE_WEIGHT:g}, "
            f"got {guide_weight}"
        )

    if not _has_holes(mask, global_mask):
        return image.copy()

    args = (ijmap, ctypes.c_int(patch_size), ctypes.c_float(guide_weight))
    if global_mask is None:
        return _call(lib.PM_inpaint_regularity, image, mask, *args)
    return _call(lib.PM_inpaint2_regularity, image, mask, global_mask, *args)


def _check_patch_size(patch_size: SupportsIndex) -> int:
    try:
        patch_size = operator.index(patch_size)
    except TypeError:
        raise TypeError(
            f"patch_size must be an integer, got {type(patch_size).__name__}"
        ) from None
    # The native code crashes for 0 and never terminates for negative sizes. ctypes
    # silently wraps larger values into the C int range, e.g. 2**31 to a negative size.
    if not 1 <= patch_size <= _MAX_PATCH_SIZE:
        raise ValueError(
            f"patch_size must be between 1 and {_MAX_PATCH_SIZE}, got {patch_size}"
        )
    return patch_size


def _prepare_inputs(
    image: ImageLike,
    mask: ImageLike | None,
    global_mask: ImageLike | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate the inputs and convert them to contiguous arrays."""
    image = _canonize_image_array(image)
    mask = _default_mask(image) if mask is None else _canonize_mask_array(mask)
    if global_mask is not None:
        global_mask = _canonize_mask_array(global_mask)

    # the native code indexes the masks with the image coordinates
    for name, m in (("mask", mask), ("global_mask", global_mask)):
        if m is not None and m.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"{name} must have the same height and width as the image, "
                f"got {m.shape[:2]} for an image of {image.shape[:2]}"
            )
    return image, mask, global_mask


def _has_holes(mask: np.ndarray, global_mask: np.ndarray | None) -> bool:
    """Whether a hole lies outside the global mask.

    Otherwise the result equals the input, but the native code still runs every
    level of the pyramid. This also covers empty images, which OpenCV 4.6 would
    return with the shape (0, 0, 1).
    """
    holes = mask != 0
    if global_mask is not None:
        holes &= global_mask == 0
    return bool(holes.any())


def _call(func: Callable[..., CMatT], *args: object) -> np.ndarray:
    """Call ``func`` with arrays converted to pymats and copy the result."""
    lib = _get_lib()
    c_args = [np_to_pymat(a) if isinstance(a, np.ndarray) else a for a in args]
    ret = func(*c_args)
    if not ret.data_ptr:
        # the error message is kept per thread, like the call itself
        message = lib.PM_last_error().decode(errors="replace").strip()
        raise RuntimeError(f"patchmatch failed: {message}")
    try:
        return pymat_to_np(ret)
    finally:
        lib.PM_free_pymat(ret)


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
    if mask.dtype == np.bool_:  # e.g. boolean arrays or PIL images in mode "1"
        mask = mask.astype(np.uint8)
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
