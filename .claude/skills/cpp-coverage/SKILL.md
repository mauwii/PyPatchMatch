---
name: cpp-coverage
description: Measure the line and branch coverage of patchmatch/csrc with gcovr, like the CI job coverage. Use to check the coverage gates or to find untested C++ code.
---

# C++ coverage

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
  of millions of returns through the clamp at the end of `scale_sum`,
  which recomputing the distances showed to be unreachable. Judge by GCC (the CI job,
  or Docker with `ubuntu:24.04`).
- Missed on purpose: the `catch (...)` blocks of `guarded()` and `set_last_error()`, the
  clamps at the end of `scale_sum` and for targets outside the image in
  `RegularityGuidedPatchDistanceMetricV2`, `operator()` of `PatchSSDDistanceMetric`
  (C++ API; the field measures its sums directly), and the masked branch of
  `MaskedImage::upsample` (only called on targets, which have no holes). GCC also
  reports the closing braces of the functions that return a `MaskedImage`, where it
  puts the cleanup for exceptions.
- GCC 13 (Docker arm64, `-Og`): 97.0 % of the lines, 95.2 % of the branches. The gates
  are these values rounded down to 5 % with at least two points of margin.
