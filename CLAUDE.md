# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository. The details of the native code, the C interface, packaging and CI are
path-scoped rules in `.claude/rules/`, which load when a matching file is read or
edited; read the rule before changing such a file through Bash. The sanitizer, C++
coverage and fill-quality runs are skills in `.claude/skills/`.

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
versions, and cibuildwheel only builds `cp311-*`. Keep `csrc` free of `Python.h`,
pybind11 or nanobind. The C++ sources are excluded from the wheel (`wheel.exclude`);
CMake installs the shared library into the package directory.

## Conventions

- Python >= 3.11: keep `from __future__ import annotations`. Ruff targets py311 with the
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
