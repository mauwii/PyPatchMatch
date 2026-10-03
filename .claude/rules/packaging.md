---
paths:
  - "pyproject.toml"
  - "CMakeLists.txt"
  - "scripts/build_opencv.py"
  - ".github/workflows/wheels.yml"
---

# Packaging and versioning

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
- The fallback wheel of the CI job `sdist-fallback`:
  `uv build --wheel dist/<sdist>.tar.gz --out-dir fallback -C cmake.define.CMAKE_DISABLE_FIND_PACKAGE_OpenCV=ON`
  must produce a `py3-none-any` wheel with the Python package but without the library.
- Wheels: manylinux x86_64 and aarch64, macOS arm64 and x86_64 (deployment target 11.0),
  Windows x64. musllinux is skipped.

## The static OpenCV

- With a static OpenCV, CMake installs its licenses, which the build script collects in
  `<prefix>/licenses`, into `.dist-info/licenses/opencv` of the wheel, and the CycloneDX
  SBOM it writes to `<prefix>/sboms/opencv.cdx.json` into `.dist-info/sboms` (PEP 770).
  The static library hides OpenCV from scanners; grype matches the SBOM through its CPE.
- The SBOM leaves out OpenCV's bundled zlib: only `FileStorage` uses it, and none of it
  is linked into the library. The wheel job checks that every wheel has the SBOM and,
  except on Windows, that `nm` finds no zlib symbol in the library. `write_sbom` fails
  if the build bundles another 3rdparty library; add a new one to the SBOM.
- To update OpenCV, change the default of `OPENCV_VERSION` and add its hash to
  `OPENCV_SHA256S`. `PATCHMATCH_OPENCV_VERSION` selects another pinned version: the
  CI job `opencv5` builds against OpenCV 5 (Homebrew's `opencv` since 2026), because
  builds from the sdist link whatever OpenCV the system has. The CI cache of the
  OpenCV build is keyed on the hash of the script.
