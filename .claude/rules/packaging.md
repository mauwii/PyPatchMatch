---
paths:
  - "pyproject.toml"
  - "CMakeLists.txt"
  - "scripts/build_opencv.py"
  - ".github/workflows/wheels.yml"
---

# Packaging and versioning

The comments in `pyproject.toml`, `CMakeLists.txt`, `wheels.yml` and the build script
explain their settings; this lists what they do not say.

- The PyPI description rewrites relative image paths of `README.md` to
  raw.githubusercontent.com URLs at the tag `v$HFPR_VERSION`, so release tags must be
  `v` plus the normalized version (`v2.0.0rc6`).
- cibuildwheel tests the sdist contents, so new test data has to be added to
  `sdist.include` as well.
- The fallback without the native library (the `if.failed` override) is kept working by
  the CI job `sdist-fallback` and `test_import_without_native_library`.
- The wheels ship OpenCV's licenses and a CycloneDX SBOM (PEP 770), which grype matches
  through its CPE. The SBOM leaves out the bundled zlib, which is not linked; the wheel
  job checks both.
- To update OpenCV, change the default of `OPENCV_VERSION` and add its hash to
  `OPENCV_SHA256S`. The CI cache of the OpenCV build is keyed on the hash of the
  script. `PATCHMATCH_OPENCV_VERSION` selects another pinned version, e.g. 4.14.0 of
  the 2.0.0 wheels to compare the fills. Builds from the sdist link whatever OpenCV the
  system has; CI tests 5.0 through the script and 4.6 in `system-opencv`.
