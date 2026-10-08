---
paths:
  - "patchmatch/_lib.py"
  - "patchmatch/patch_match.py"
  - "patchmatch/csrc/pyinterface.*"
---

# The C interface contract

A change on one side has to be mirrored on the other:

- `PM_mat_t`/`PM_shape_t` in `pyinterface.h` and `CMatT`/`CShapeT` in `_lib.py` (field
  order and types).
- The order of `PM_dtype_e` and of the `_PYMAT_DTYPES` list: uint8 and float32, the
  only types the native code reads; `_py_to_cv2` rejects others.
- Every `PM_*` function needs its `argtypes` and `restype` in `load_library()`; without
  them ctypes silently assumes `int`.
- Symbols are exported by `extern "C"` and `CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS`, there are
  no export macros.

A new entry point is declared in `pyinterface.h`, implemented in `pyinterface.cpp`
as a call of the shared `inpaint` helper inside `guarded(...)`, registered in
`load_library()` and called through `_call()`.

Memory ownership: `np_to_pymat` wraps a C-contiguous array without copying it, so
callers convert with `np.ascontiguousarray` first and keep the array alive during the
call. `_py_to_cv2` clones it into a `cv::Mat`, so the inputs are never modified.
`_cv2_to_py` hands a heap buffer (a released `std::unique_ptr<unsigned char[]>`) to
Python; `pymat_to_np` copies it, and `_call` always returns it through `PM_free_pymat` in
a `finally`. It allocates at least one byte, because a null `data_ptr` signals failure.

Errors: a C++ exception that reaches ctypes aborts the process. `guarded()` catches
everything, stores the message in a thread-local string and returns a null `data_ptr`;
`_call` turns that into `RuntimeError("patchmatch failed: <PM_last_error()>")`. Input
validation lives in Python, because the native code crashes or hangs on bad input; the
comments at the checks in `patch_match.py` say how.
