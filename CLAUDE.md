# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

## Project

PyPatchMatch implements PatchMatch based image inpainting (Barnes et al., SIGGRAPH 2009):
a C++17 library on top of OpenCV core, and a thin Python package that loads it through
`ctypes`. It is published on PyPI as `PyPatchMatch` (import name `patchmatch`) and used
by InvokeAI. This repository (`mauwii/PyPatchMatch`) is the maintained fork of
`invoke-ai/PyPatchMatch`, which derives from `vacancy/PyPatchMatch` and
`younesse-cv/PatchMatch`.

## Commands

The project is managed with uv (`.python-version` pins 3.12 for the local venv). Building
needs a C++17 compiler, CMake >= 3.21 and an OpenCV that `find_package(OpenCV)` finds:
Homebrew `opencv`, `libopencv-dev`, or an installation prefix in `OpenCV_ROOT`.

```sh
uv sync                                    # .venv, editable install, compiles the library
uv run pre-commit install
uv run pytest                              # all tests, with the coverage gate
uv run pytest -m "not e2e"                 # fast unit tests only
uv run pytest -m e2e --no-cov              # end-to-end tests only
uv run pytest tests/test_patch_match.py::test_inpaint_global_mask --no-cov
uv run pytest -k regularity --no-cov
uv run mypy                                # strict, also a pre-commit hook
uv run pre-commit run --all-files          # what the CI lint job runs
uv run ruff check --fix . && uv run ruff format .
uv build                                   # sdist and wheel into dist/
```

- Pass `--no-cov` for any partial run: `addopts` always enables coverage, and
  `fail_under = 95` (branch coverage, subprocesses included through
  `patch = ["subprocess"]`) fails the run otherwise.
- Use `uv run pytest`, not `python -m pytest`. The latter puts the repository root on
  `sys.path`, which imports the source tree `patchmatch/` without the compiled library;
  the symptoms are the warning `patchmatch failed to load: libpatchmatch.* not found`
  and `RuntimeError: the patchmatch native library is not available` in every test.
- `--cov=patchmatch` measures the source tree, so coverage only works with the editable
  install that `uv sync` creates. With a non-editable install (`uv sync --no-editable`,
  `uv pip install .`) it reports 0 % and the gate fails.
- `uv sync` rebuilds the library only when a `tool.uv.cache-keys` entry changes
  (`CMakeLists.txt`, `patchmatch/csrc/*`, `pyproject.toml`, git commit or tags). Force it
  with `uv sync --reinstall-package pypatchmatch`.
- CI builds the library with warnings as errors. Locally:
  `SKBUILD_CMAKE_DEFINE=CMAKE_COMPILE_WARNING_AS_ERROR=ON uv sync --reinstall-package pypatchmatch`.

Build against the same static, core-only OpenCV as the release wheels:

```sh
export OpenCV_ROOT=/tmp/opencv
uv run --no-project python scripts/build_opencv.py   # pinned version and SHA-256
uv sync --reinstall-package pypatchmatch
```

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
- The run takes about 80 s locally. Calling `PM_inpaint` through ctypes with a shape
  larger than the buffer (ASan) or a `patch_size` of `2**31 - 1` (UBSan, `nnf.cpp`)
  shows that both report and abort.

C++ coverage, like the CI job `coverage` (gcovr comes from the `coverage` group):

```sh
SKBUILD_CMAKE_DEFINE=PATCHMATCH_COVERAGE=ON uv sync --reinstall-package pypatchmatch
find build -name '*.gcda' -delete                    # the counts add up across runs
uv run --group coverage pytest --no-cov
uv run --group coverage gcovr --filter patchmatch/csrc/ --print-summary \
    --exclude-throw-branches --exclude-unreachable-branches \
    --gcov-suspicious-hits-threshold 0
uv sync --reinstall-package pypatchmatch             # back to the normal library
```

- The Python tests are the C++ tests: there is no C++ test framework, the coverage of
  the native code is measured while they run.
- The build uses `-Og` with atomic counters (a test inpaints in several threads at once);
  the run takes about 90 s locally. `-O0` reports one more line (the masked branch of
  `MaskedImage::upsample`, which `-Og` merges) but triples the time. The distance loop
  runs more than 2**32 times, which gcovr rejects as a gcov bug without
  `--gcov-suspicious-hits-threshold 0`.
- On macOS add `--gcov-executable "xcrun llvm-cov gcov"`. Clang's counts are not
  reliable: they report a closing brace after a `return` as a missed line, and hundreds
  of millions of returns through the clamp at the end of `distance_masked_images`,
  which recomputing the distances showed to be unreachable. Judge by GCC (the CI job,
  or Docker with `ubuntu:24.04`).
- Missed on purpose: the `catch (...)` blocks of `guarded()` and `set_last_error()`,
  the `clone()` of a non-continuous result in `_cv2_to_py`, the clamps at the end of
  `distance_masked_images` and for targets outside the image in
  `RegularityGuidedPatchDistanceMetricV2`, and the masked branch of
  `MaskedImage::upsample` (only called on targets, which have no holes). GCC also
  reports the closing braces of the functions that return a `MaskedImage`, where it
  puts the cleanup for exceptions.
- GCC 13 (Docker arm64, `-Og`): 97.0 % of the lines, 95.2 % of the branches. The gates
  are these values rounded down to 5 % with at least two points of margin.

Fill quality (`scripts/evaluate_inpainting.py`, its docstring explains the columns):

```sh
uv run --no-sync python scripts/evaluate_inpainting.py --baseline <library of main>
uv run --no-sync python scripts/evaluate_inpainting.py --patch-size 15 --seeds 1 \
    --cases forest,brick
```

- It cuts holes into `forest.bmp` and the CC0 images of `examples/images` (sources in
  `SOURCES.md`) and adds `forest_pruned.bmp` and every `*_pruned.*` of the git-ignored
  `examples/images/local/` as object removals without a ground truth. That folder is
  for your own images; never commit them.
- For `--baseline`, build the library of `main` outside the checkout:

  ```sh
  mkdir -p /tmp/main-lib
  git archive main CMakeLists.txt patchmatch/csrc | tar -x -C /tmp/main-lib
  cmake -S /tmp/main-lib -B /tmp/main-lib/build -DCMAKE_BUILD_TYPE=Release
  cmake --build /tmp/main-lib/build   # libpatchmatch.so, .dylib or patchmatch.dll
  ```

- Judge changes on all cases and look at the fills in `examples/images/evaluation.html`
  (git-ignored, it embeds the images of `examples/images/local/` too): fixes for one
  image regressed others before.

Other entry points:

- `examples/cpp_example_run.sh` builds the library and `examples/cpp_example.cpp` with
  CMake into `build/cpp` and runs it. It needs OpenCV `imgcodecs`, so a full OpenCV, not
  the core-only build of the script.
- `uv run python examples/py_example.py` (and `py_example_global_mask.py`) write
  `examples/images/forest_recovered.bmp`, which is git-ignored.
- CMake options: `PATCHMATCH_BUILD_EXAMPLES` (OFF), `PATCHMATCH_FAST_MATH` (ON,
  `-ffast-math` or `/fp:fast`), `PATCHMATCH_SANITIZE` (OFF, ASan and UBSan with
  `float-cast-overflow`, every finding aborts; turns off fast math, which lets the
  compiler drop checks; GCC and Clang only), `PATCHMATCH_COVERAGE` (OFF, gcov counters
  at `-Og`; GCC and Clang only).
- The fallback wheel of the CI job `sdist-fallback`:
  `uv build --wheel dist/<sdist>.tar.gz --out-dir fallback -C cmake.define.CMAKE_DISABLE_FIND_PACKAGE_OpenCV=ON`
  must produce a `py3-none-any` wheel with the Python package but without the library.

## Architecture

### Layers, in the call order of `patchmatch.inpaint`

1. `patchmatch/__init__.py` re-exports the public API of `patch_match.py` and
   `__version__` from the generated `_version.py`. `test_top_level_exports` checks that
   every name in `patch_match.__all__` is re-exported.
2. `patchmatch/patch_match.py` is the public API. It validates and canonicalizes the
   inputs (`_prepare_inputs`, `_check_patch_size`, the `ijmap` and `guide_weight`
   checks), picks the entry point (`PM_inpaint`, `PM_inpaint2` with a global mask, and
   their `_regularity` variants) and runs it through `_call`. It loads the library at
   import time; on `OSError` it logs a warning and sets `patchmatch_available = False`,
   and every function then raises `RuntimeError` through `_get_lib()`.
3. `patchmatch/_lib.py` is the ctypes layer: the `CShapeT`/`CMatT` structures, the dtype
   table, `np_to_pymat`/`pymat_to_np`, `find_library` (looks for `libpatchmatch.so`,
   `libpatchmatch.dylib` or `patchmatch.dll` in every directory of `patchmatch.__path__`)
   and `load_library` (argtypes and restype of every `PM_*` function).
4. `patchmatch/csrc/pyinterface.{h,cpp}` is the `extern "C"` API: the `PM_*` functions,
   the conversion between `PM_mat_t` and `cv::Mat`, the global settings and the error
   reporting.
5. `csrc/inpaint.{h,cpp}`: `Inpainting`, the image pyramid and the EM loop.
6. `csrc/nnf.{h,cpp}`: `NearestNeighborField` (propagation and random search) and the
   distance metrics `PatchSSDDistanceMetric` and `RegularityGuidedPatchDistanceMetricV2`.
7. `csrc/masked_image.{h,cpp}`: `MaskedImage`, an image with its hole mask and optional
   global mask, down- and upsampling, gradients.

The C++ headers double as the C++ API: `examples/cpp_example.cpp` uses `Inpainting` and
`PatchSSDDistanceMetric` directly.

### ctypes instead of a CPython extension

The library does not use the CPython ABI, so the wheels are tagged
`py3-none-<platform>` (`wheel.py-api = "py3"`): one wheel per platform serves all Python
versions, and cibuildwheel only builds `cp310-*`. Keep `csrc` free of `Python.h`,
pybind11 or nanobind. The C++ sources are excluded from the wheel (`wheel.exclude`);
CMake installs the shared library into the package directory.

### The C interface contract

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
inside `guarded(...)`, registered in `load_library()` and called through `_call()`.

Memory ownership: `np_to_pymat` wraps a C-contiguous array without copying it, so
callers convert with `np.ascontiguousarray` first and keep the array alive during the
call. `_py_to_cv2` clones it into a `cv::Mat`, so the inputs are never modified.
`_cv2_to_py` hands a heap buffer (a released `std::unique_ptr<unsigned char[]>`) to
Python; `pymat_to_np` copies it, and `_call` always returns it through `PM_free_pymat` in
a `finally`. It allocates at least one byte, because a null `data_ptr` signals failure.

Errors: a C++ exception that reaches ctypes aborts the process. `guarded()` catches
everything, stores the message in a thread-local string and returns a null `data_ptr`;
`_call` turns that into `RuntimeError("patchmatch failed: <PM_last_error()>")`. Most
input validation lives in Python because the native code crashes or hangs on bad input:
`patch_size` 0 crashes, negative sizes never terminate, sizes >= 2**31 wrap to negative
C ints, negative guide weights index outside the similarity table, NaN or infinity in the
`ijmap` produce out-of-range distances. `RegularityGuidedPatchDistanceMetricV2` also
throws on a negative weight, for C++ callers.

### Threading and determinism

ctypes releases the GIL during native calls, so inpaintings run truly concurrently
(`test_concurrent_inpaint_is_deterministic`). Therefore:

- The seed and the verbose flag are `std::atomic` globals, read when a call starts.
- The random generator is a function-local `thread_local std::mt19937`, seeded at the
  start of every `Inpainting::run`, so each call is reproducible from the seed on any
  thread.
- `random_int` uses `engine() % n` instead of `std::uniform_int_distribution` to keep
  the sequence identical on all platforms and standard libraries. Keep it that way.
- Lookup tables are function-local statics (thread-safe initialization). Do not add
  mutable shared state.

### Algorithm

- `patch_size` is a radius: patches span `(2 * patch_size + 1)` pixels in each
  direction. The pyramid halves the image until one side is <= `patch_size`.
- From the coarsest level to the finest, two nearest-neighbor fields are kept:
  source to target (completeness) and target to source (coherence). They start random
  at the coarsest level and are upscaled from the previous level afterwards, keeping the
  offset within the coarse pixel so that neighbors still point to neighbors.
- Each level runs `1 + 2 * level` EM iterations with `min(7, 1 + level)` NNF passes
  (propagation and random search), an expectation step (votes of both fields, weighted
  through the `distance2similarity` table) and a maximization step, which sets each
  pixel to the weighted mean of its votes. On levels 0 and 1 the known pixels keep their
  input values instead; without that the votes change the pixels around the holes, and a
  seam appears when `run()` restores them. On coarser levels the known pixels next to
  the holes are averages of partially masked kernels, and keeping them made fills copy
  smooth regions into textured ones (57 % instead of 17 % of the fill in one case). The
  last iteration of a level votes directly into the upsampled image of the next level,
  which is less blurry than upsampling the result. The last iteration at level 0 keeps
  only the heaviest vote of each pixel instead of the mean, which blurs.
- The coarse levels mark a pixel as a hole only if its whole downsampling kernel is
  one, so the holes shrink quickly and are filled smoothly from their border; the finer
  levels add the texture. Marking every pixel that touches a hole instead was tried and
  made the fills worse: it leaves the coarsest levels without valid source patches,
  because the similarity table gives zero weight to patches with more than about 10 %
  masked or border pixels.
- Distances are ints in `[0, PatchDistanceMetric::kDistanceScale]` (65535): an SSD over
  the colors and the x/y gradients, where masked pixels and pixels at the border count
  as maximal. The distance indexes `distance2similarity`, so every metric has to clamp
  its result to that range.
- Mask semantics: `mask` non-zero marks the holes (by default all pure white pixels,
  computed in Python by `_default_mask`); `global_mask` marks pixels that are neither
  filled nor used as a patch source. At the end of `run()` every pixel that is not a
  hole, including the globally masked ones, is copied back from the input, so only the
  holes change. The tests assert this invariant (`assert_filled`,
  `assert_plausible_fill`).

### OpenCV usage and library size

The release wheels link a static, core-only OpenCV (`scripts/build_opencv.py`:
`BUILD_LIST=core`, every optional backend off). For changes to the C++ code:

- The library links only `opencv_core`; `imgcodecs` is only used by the example.
- Use plain pixel loops instead of `cv::Mat` matrix expressions such as `a == b` or
  `m1 | m2`: they pull OpenCV's expression module, and with it `FileStorage`, PCA and
  `SparseMat` code, into the static library, which grew by about 40 % before commit
  e50c613 replaced them.
- On Windows the static OpenCV uses the static MSVC runtime (`BUILD_WITH_STATIC_CRT`),
  so the library does too (`MSVC_RUNTIME_LIBRARY` in `CMakeLists.txt`). Its stderr is
  then not captured by pytest's `capfd`, so tests of native output run in a subprocess
  (`test_set_verbose_prints_progress`).
- With a static OpenCV, CMake installs its licenses, which the build script collects in
  `<prefix>/licenses`, into `.dist-info/licenses/opencv` of the wheel, and the CycloneDX
  SBOM it writes to `<prefix>/sboms/opencv.cdx.json` into `.dist-info/sboms` (PEP 770).
  The static library hides OpenCV from scanners; grype matches the SBOM through its CPE.
- The SBOM leaves out OpenCV's bundled zlib: only `FileStorage` uses it, and none of it
  is linked into the library. The wheel job checks that every wheel has the SBOM and,
  except on Windows, that `nm` finds no zlib symbol in the library. `write_sbom` fails
  if the build bundles another 3rdparty library; add a new one to the SBOM.
- To update OpenCV, change `OPENCV_VERSION` and `OPENCV_SHA256` together. The CI cache of
  the OpenCV build is keyed on the hash of the script.

### Packaging and versioning

- Build backend: scikit-build-core with setuptools-scm. The version comes from the git
  tags; `patchmatch/_version.py` is generated and git-ignored. Without git metadata the
  version falls back to `0.0.0`, so CI checks out with `fetch-depth: 0`.
- The release wheels are built from the sdist, not from the checkout, and cibuildwheel
  runs `pytest {project}/tests --no-cov` against the sdist contents. Everything the tests
  need has to be in the sdist: `sdist.include` adds `examples/images/forest_pruned.bmp`,
  the rest of `examples/` is excluded. New test data has to be added there as well.
- Fallback: if a build from the sdist fails (e.g. no compiler, CMake or OpenCV on a
  platform without a wheel), the `tool.scikit-build.overrides` entry builds a pure wheel
  without the library, so the installation succeeds with `patchmatch_available` False and
  the error message of `find_library` explains how to reinstall. Builds from the source
  tree still fail. The CI job `sdist-fallback` and `test_import_without_native_library`
  keep this path working.
- Wheels: manylinux x86_64 and aarch64, macOS arm64 and x86_64 (deployment target 11.0),
  Windows x64. musllinux is skipped.

## CI

- `ci.yml`: `lint` runs `pre-commit run --all-files`; `test` runs `uv run pytest` on
  Ubuntu, macOS and Windows with Python 3.10 and 3.14, after building (or restoring) the
  OpenCV of the build script.
- pre-commit runs, besides the file checks, ruff and uv-lock: clang-format
  (`.clang-format`), shellcheck, markdownlint-cli2 (`.markdownlint.yaml`, 88 columns),
  typos (fixes in place), actionlint (checks `run:` scripts with shellcheck only if it
  is on `PATH`) and zizmor (offline, default persona), and mypy as a local hook
  (`uv run --locked --no-build --only-group dev mypy`, like the `lint` job, so the
  library is not built; numpy and pillow are in the `dev` group for their types).
  MegaLinter was considered and rejected: it duplicates pre-commit in a multi-GB Docker
  image and bundles the linter versions instead of pinning each hook for Dependabot.
- Dependabot waits 7 days after a release (`cooldown`, required by zizmor).
- `test` sets `SKBUILD_CMAKE_DEFINE=CMAKE_COMPILE_WARNING_AS_ERROR=ON`, so the warning
  flags of `CMakeLists.txt` (`-Wall -Wextra -Wpedantic -Wshadow`, `/W4` on MSVC) fail the
  build. Only there: builds from the sdist and the wheels must not break when a newer
  compiler adds a warning. The OpenCV headers are system includes and do not count.
- `sanitizers` builds with `PATCHMATCH_SANITIZE=ON` on Ubuntu and runs the tests with
  the preloaded ASan runtime (see Commands). It reuses the checkout and OpenCV steps of
  `test` through YAML anchors, so change them there.
- `coverage` builds with `PATCHMATCH_COVERAGE=ON` on Ubuntu, runs the tests and fails
  below 95 % of the C++ lines or 90 % of the branches (gcovr, see Commands).
  It reuses the same anchors. It also writes `coverage.xml` (Python) and
  `coverage-cpp.xml` (`gcovr --sonarqube`) and exports `build/compile_commands.json`,
  and then runs the SonarCloud scan (`sonar-project.properties`, `SONAR_TOKEN`, skipped
  without the secret, e.g. for Dependabot and forks). Automatic analysis in SonarCloud
  is disabled; it cannot import coverage and would conflict with the CI scan. `scripts/`
  has no tests and is excluded from the coverage on new code (80 % in "Sonar way").
- The `uv run` steps pass `--locked` and `--no-build` (SonarCloud S8544, S8541).
  `--no-build` only forbids building dependencies from source; uv still builds the
  project itself. The OpenCV step runs with `--only-group dev`: the project cannot be
  built before OpenCV is installed, and `--no-project` would lock nothing.
- `wheels.yml`: sdist, then cibuildwheel on five runners, then `sdist-fallback`;
  `publish` uploads to PyPI with trusted publishing (environment `pypi`) only for a
  published GitHub release of `mauwii/PyPatchMatch`.
- `codeql.yml` analyzes actions, C/C++ and Python.
- `ci-ok` and `wheels-ok` (re-actors/alls-green) are the required status checks; add new
  jobs to their `needs`.
- Actions are pinned to full commit SHAs with a `# vX.Y.Z` comment, and checkouts use
  `persist-credentials: false`. Dependabot updates actions, uv and pre-commit weekly in
  groups.

## Conventions

- Python >= 3.10: keep `from __future__ import annotations`. Ruff targets py310 with the
  rules B, C4, E, F, I, RUF, SIM, UP and W, and formats at 88 columns. `patchmatch/csrc`
  is excluded from ruff. C++ is formatted by clang-format (Microsoft style, 120 columns,
  includes grouped as own header, `<...>`, `"..."`); naming is not checked: `m_`
  members, `k` constants. Commit a reformat separately and add its hash to
  `.git-blame-ignore-revs`.
- The package ships `py.typed`; public functions are fully annotated. mypy checks
  `patchmatch` and `tests` in strict mode (`[tool.mypy]`); the tests have to use the
  API correctly but need no annotations of their own. A deliberately wrong argument
  goes through a test parameter instead of a `# type: ignore`.
- Backwards compatibility: `from patchmatch import patch_match` and
  `patch_match.CShapeT`/`CMatT` keep working (tested).
- Comments explain why: which crash a check prevents, why a construct is avoided.
- No `NOSONAR` markers: change the code so that the SonarCloud rule is satisfied, and ask
  when that seems impossible. The only exception is `nnf.cpp`: cpp:S2245 flags every
  declaration of `std::mt19937`, which only drives the randomized search.
- Tests check invariants instead of golden images: holes filled, all other pixels
  unchanged, determinism with `set_random_seed`, fills that blend in with their
  surroundings. `test_patch_match.py` and `test_e2e.py` seed through an autouse fixture.
  Unit tests use a small synthetic image; the `e2e` tests use
  `examples/images/forest_pruned.bmp` and mirror the workflows of the README and the
  examples. `test_properties.py` checks the invariants on random inputs with Hypothesis
  (in the `test` group, so the wheel tests run it too); it seeds inside the tests,
  because Hypothesis rejects function-scoped fixtures, and has no deadline, because
  the sanitizer and coverage builds are much slower.
- The README documents the behavior (mask semantics, `patch_size` as a radius, the
  fallback installation); update it together with API or installation changes.
- Commit messages: imperative subject in sentence case without a prefix (e.g. "Reject
  invalid patch sizes and empty regularity maps"), and a body that explains the reason,
  with measurements where relevant. Changes go through PRs from branches such as
  `fix/<topic>`.

## Git and GitHub

- `origin` is `mauwii/PyPatchMatch`, where PRs and releases go; `upstream` is
  `invoke-ai/PyPatchMatch`. `gh` resolves the default repository to upstream, so pass
  `-R mauwii/PyPatchMatch`.
- Releases: creating a GitHub release with a tag like `v2.0.0` (or `v2.0.0rc2` for
  pre-releases) builds the wheels and publishes them to PyPI.
