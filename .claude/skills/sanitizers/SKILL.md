---
name: sanitizers
description: Build the library with ASan and UBSan and run the tests under them, like the CI job sanitizers. Use after changes to memory handling, indexing, integer arithmetic or the incremental distances of the propagation in patchmatch/csrc, or to reproduce a failure of that job.
---

# Sanitizers

ASan and UBSan, like the CI job `sanitizers` (Linux):

```sh
SKBUILD_CMAKE_DEFINE=PATCHMATCH_SANITIZE=ON uv sync --reinstall-package pypatchmatch
ASAN="$(gcc -print-file-name=libasan.so) $(gcc -print-file-name=libstdc++.so)"
ASAN_OPTIONS=detect_leaks=0 uv run env LD_PRELOAD="$ASAN" pytest --no-cov --capture=sys
uv sync --reinstall-package pypatchmatch             # back to the normal library
```

- Python is not instrumented, so the ASan runtime has to be loaded at startup; without
  it every test aborts with `Interceptors are not working`. On Linux `libstdc++` has to
  follow it: ASan looks up the real `__cxa_throw` at startup, and without it the first
  C++ exception aborts with `CHECK failed: ... real___cxa_throw`. The leak check would
  report the interpreter's own allocations.
- `--capture=sys` is required: pytest's default capture of the file descriptors keeps
  the report of an abort, and only `Fatal Python error: Aborted` reaches the terminal.
- On macOS, inject the runtime with
  `DYLD_INSERT_LIBRARIES="$(xcrun clang -print-resource-dir)/lib/darwin/libclang_rt.asan_osx_dynamic.dylib"`,
  add `strip_env=0` to `ASAN_OPTIONS` (ASan removes the variable, and the tests that
  start a subprocess abort), and start pytest as `.venv/bin/python .venv/bin/pytest`:
  SIP drops the variable for `/usr/bin/env` and `/bin/sh`, which uv uses as the
  shebang of scripts in long paths.
- Only this build keeps the asserts (`-UNDEBUG`), which compare every distance sum of
  the propagation with a full measurement.
- The weekly and manual CI runs pass `--hypothesis-profile=deep` (5000 examples per
  property test, `tests/conftest.py`). A failure there prints a `@reproduce_failure`
  decorator; add it to the test to rerun that input locally.
- The run takes about 80 s locally. Calling `PM_inpaint` through ctypes with a shape
  larger than the buffer (ASan) or a `patch_size` of `2**31 - 1` (UBSan, `nnf.cpp`)
  shows that both report and abort.
