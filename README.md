# PatchMatch based Inpainting

[![CI](https://github.com/mauwii/PyPatchMatch/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mauwii/PyPatchMatch/actions/workflows/ci.yml)
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=mauwii_PyPatchMatch&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=mauwii_PyPatchMatch)
[![License: MIT](https://img.shields.io/badge/License-MIT-blueviolet.svg)](https://github.com/mauwii/PyPatchMatch/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/PyPatchMatch)](https://pypi.org/project/PyPatchMatch/)
[![Downloads](https://static.pepy.tech/badge/pypatchmatch)](https://pepy.tech/projects/pypatchmatch)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

This library implements the PatchMatch based inpainting algorithm. It provides both C++
and Python interfaces. This implementation is heavily based on the implementation by
Younesse ANDAM: [younesse-cv/PatchMatch](https://github.com/younesse-cv/PatchMatch),
with some bug fixes, and updates.

![Birdhouses removed from a photo by patchmatch.inpaint](examples/images/readme_example.jpg)

From left to right: the photo, the input of
[examples/py_example.py](https://github.com/mauwii/PyPatchMatch/blob/main/examples/py_example.py)
with the birdhouses painted out in white, and its result.

## Installation

```sh
pip install PyPatchMatch
```

Wheels for Linux (glibc, x86_64 and aarch64), macOS 11+ (arm64 and x86_64) and Windows
(x64) ship the compiled library with a statically linked OpenCV, so no compiler or
OpenCV installation is needed. On other platforms, e.g. Alpine Linux or Windows on ARM,
pip builds from the source distribution, which requires a C++17 compiler, CMake and the
OpenCV development files (e.g. `apt install libopencv-dev` or `brew install opencv`).

If that build fails, PyPatchMatch is installed without the native library instead of
blocking the installation: `patchmatch.patchmatch_available` is `False` and the
inpainting functions raise `RuntimeError`. After installing the missing tools, reinstall
it without the cached build:

```sh
pip install --force-reinstall --no-deps --no-cache-dir PyPatchMatch
```

## Usage

Python (see
[examples/py_example.py](https://github.com/mauwii/PyPatchMatch/blob/main/examples/py_example.py)):

```python
import patchmatch

image = ...  # HxWx3 uint8 numpy array or PIL image
mask = ...  # HxW uint8 or bool numpy array or PIL image, non-zero marks the holes
result = patchmatch.inpaint(image, mask, patch_size=3)
```

The mask must have the same height and width as the image. If `mask` is omitted, all
pure white pixels are treated as holes. Only the holes are filled; all other pixels keep
their values, and the colors under the holes have no influence on the fill. Holes need
hard edges: an anti-aliased brush leaves light pixels along the edge of its strokes that
are not pure white, so they count as known and the fill continues them. Paint without
anti-aliasing or pass a mask that covers those pixels as well. The optional keyword
argument `global_mask`, in the same format, marks pixels that are neither filled nor
used as a source; they keep their values as well, and their colors have no influence on
the fill (see
[examples/py_example_global_mask.py](https://github.com/mauwii/PyPatchMatch/blob/main/examples/py_example_global_mask.py)).
`patchmatch.patchmatch_available` tells whether the native library could be loaded.
The previous import path `from patchmatch import patch_match` keeps working.

`patch_size` is the radius of the compared patches, which span `2 * patch_size + 1`
pixels in each direction. Larger patches follow larger structures but are much slower;
the default of 15 compares 31x31 patches, the examples use 3.

`patchmatch.set_random_seed(seed)` sets the seed of the randomized search, modulo
`2**32`, and `patchmatch.set_verbose(True)` prints the progress of the native code to
stderr. `patchmatch.inpaint_regularity(image, mask, ijmap)` additionally guides the
search with a regularity map, an HxWx3 float32 array with the regularity coordinates of
each pixel in its first two channels. Its `guide_weight` must be between 0 and the
largest float32, about 3.4e38.

C++ (see
[examples/cpp_example.cpp](https://github.com/mauwii/PyPatchMatch/blob/main/examples/cpp_example.cpp),
build and run it with
[examples/cpp_example_run.sh](https://github.com/mauwii/PyPatchMatch/blob/main/examples/cpp_example_run.sh)):

```cpp
#include "inpaint.h"

int main() {
    cv::Mat image = ...;
    cv::Mat mask = ...;

    auto metric = PatchSSDDistanceMetric(5);
    cv::Mat result = Inpainting(image, mask, &metric).run();
}
```

The library is built with CMake; `PATCHMATCH_BUILD_EXAMPLES` builds the example.

## Development

The project is managed with [uv](https://docs.astral.sh/uv/):

```sh
uv sync                        # create .venv, build the library, install dev deps
uv run pre-commit install      # enable ruff and the other hooks on commit
uv run pytest                  # run the test suite
uv run pytest -m "not e2e"     # only the fast unit tests
uv run python scripts/evaluate_inpainting.py  # measure the fill quality
```

The evaluation cuts holes into the images of `examples/images`, compares the fills with
the original content and writes `examples/images/evaluation.html`, which shows them side
by side. `--baseline` evaluates another build of the library as well, e.g. of the main
branch; the docstring of the script explains the columns.
Pull requests that change the C++ code get this comparison with their base commit in the
summary of the workflow "Fill quality", and the HTML report as its artifact.

Releases are published to PyPI by creating a GitHub release; the version is taken from
its tag (e.g. `v2.0.0`). Every released wheel and sdist carries a build provenance
attestation: `gh attestation verify <file> -R mauwii/PyPatchMatch`.

## License

PyPatchMatch is released under the
[MIT License](https://github.com/mauwii/PyPatchMatch/blob/main/LICENSE). The wheels
contain a statically linked build of OpenCV core (Apache-2.0) and its bundled
third-party code; their licenses are included in the `licenses/opencv` directory of the
wheel's `.dist-info`, and a CycloneDX SBOM of OpenCV (PEP 770) in
`sboms/opencv.cdx.json`, so that vulnerability scanners can find it.

## README and COPYRIGHT by Younesse ANDAM

@Author: Younesse ANDAM

@Contact: <younesse.andam@gmail.com>

Description:

This project is a personal implementation of an algorithm called PATCHMATCH
that restores missing areas in an image. The algorithm is presented in the following
paper PatchMatch A Randomized Correspondence Algorithm for Structural Image Editing by
C.Barnes, E.Shechtman, A.Finkelstein and Dan B.Goldman ACM Transactions on Graphics
(Proc. SIGGRAPH), vol.28, aug-2009

For more information please refer to
<https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/>

Copyright (c) 2010-2011
