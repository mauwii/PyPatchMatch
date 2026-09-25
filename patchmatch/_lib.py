"""ctypes bindings for the native patchmatch library."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    LIBRARY_NAME = "patchmatch.dll"
elif sys.platform == "darwin":
    LIBRARY_NAME = "libpatchmatch.dylib"
else:
    LIBRARY_NAME = "libpatchmatch.so"


class CShapeT(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("channels", ctypes.c_int),
    ]


class CMatT(ctypes.Structure):
    _fields_ = [
        ("data_ptr", ctypes.c_void_p),
        ("shape", CShapeT),
        ("dtype", ctypes.c_int),
    ]


# Order matches the PM_dtype_e enum in csrc/pyinterface.h.
_PYMAT_DTYPES = [
    np.dtype(np.uint8),
    np.dtype(np.int8),
    np.dtype(np.uint16),
    np.dtype(np.int16),
    np.dtype(np.int32),
    np.dtype(np.float32),
    np.dtype(np.float64),
]
_PYMAT_DTYPE_IDS = {dtype: i for i, dtype in enumerate(_PYMAT_DTYPES)}


def np_to_pymat(npmat: np.ndarray) -> CMatT:
    """Wrap a contiguous HxWxC array without copying it."""
    if npmat.ndim != 3:
        raise ValueError(f"expected a 3-dimensional array, got {npmat.ndim} dims")
    if not npmat.flags.c_contiguous:
        raise ValueError("expected a C-contiguous array")
    try:
        dtype_id = _PYMAT_DTYPE_IDS[npmat.dtype]
    except KeyError:
        raise TypeError(f"unsupported dtype {npmat.dtype}") from None
    height, width, channels = npmat.shape
    return CMatT(
        ctypes.c_void_p(npmat.ctypes.data),
        CShapeT(width, height, channels),
        dtype_id,
    )


def pymat_to_np(pymat: CMatT) -> np.ndarray:
    """Copy the data referenced by ``pymat`` into a new numpy array."""
    dtype = _PYMAT_DTYPES[pymat.dtype]
    shape = (pymat.shape.height, pymat.shape.width, pymat.shape.channels)
    buffer = (ctypes.c_byte * (int(np.prod(shape)) * dtype.itemsize)).from_address(
        pymat.data_ptr
    )
    return np.frombuffer(buffer, dtype=dtype).reshape(shape).copy()


def find_library() -> Path:
    """Locate the shared library that is installed next to this package."""
    package = sys.modules[__package__]
    for directory in package.__path__:
        candidate = Path(directory) / LIBRARY_NAME
        if candidate.is_file():
            return candidate
    raise OSError(
        f"{LIBRARY_NAME} not found in {list(package.__path__)}; "
        "reinstall PyPatchMatch or build it from source with OpenCV available"
    )


def load_library() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(find_library()))

    lib.PM_set_random_seed.argtypes = [ctypes.c_uint]
    lib.PM_set_random_seed.restype = None
    lib.PM_set_verbose.argtypes = [ctypes.c_int]
    lib.PM_set_verbose.restype = None
    lib.PM_free_pymat.argtypes = [CMatT]
    lib.PM_free_pymat.restype = None

    lib.PM_inpaint.argtypes = [CMatT, CMatT, ctypes.c_int]
    lib.PM_inpaint2.argtypes = [CMatT, CMatT, CMatT, ctypes.c_int]
    lib.PM_inpaint_regularity.argtypes = [
        CMatT,
        CMatT,
        CMatT,
        ctypes.c_int,
        ctypes.c_float,
    ]
    lib.PM_inpaint2_regularity.argtypes = [
        CMatT,
        CMatT,
        CMatT,
        CMatT,
        ctypes.c_int,
        ctypes.c_float,
    ]
    for func in (
        lib.PM_inpaint,
        lib.PM_inpaint2,
        lib.PM_inpaint_regularity,
        lib.PM_inpaint2_regularity,
    ):
        func.restype = CMatT

    return lib
