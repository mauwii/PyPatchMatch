# PatchMatch based Inpainting

[![License: MIT](https://img.shields.io/badge/License-MIT-blueviolet.svg)](https://github.com/mauwii/PyPatchMatch/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/PyPatchMatch)](https://pypi.org/project/PyPatchMatch/)
[![Downloads](https://static.pepy.tech/badge/pypatchmatch)](https://pepy.tech/project/pypatchmatch)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

This library implements the PatchMatch based inpainting algorithm. It provides both C++
and Python interfaces. This implementation is heavily based on the implementation by
Younesse ANDAM: [younesse-cv/PatchMatch](https://github.com/younesse-cv/PatchMatch),
with some bug fixes, and updates.

## Installation

```sh
pip install PyPatchMatch
```

Wheels for Linux, macOS and Windows ship the compiled library including the required
OpenCV runtime, so no compiler or OpenCV installation is needed. On other platforms pip
builds from the source distribution, which requires a C++17 compiler, CMake and the
OpenCV development files (e.g. `apt install libopencv-dev` or `brew install opencv`).

## Usage

Python (see [examples/py_example.py](examples/py_example.py)):

```python
import patchmatch

image = ...  # HxWx3 uint8 numpy array or PIL image
mask = ...  # HxW uint8 or bool numpy array or PIL image, non-zero marks the holes
result = patchmatch.inpaint(image, mask, patch_size=3)
```

The mask must have the same height and width as the image. If `mask` is omitted, all
pure white pixels are treated as holes.
`patchmatch.patchmatch_available` tells whether the native library could be loaded.
The previous import path `from patchmatch import patch_match` keeps working.

C++ (see [examples/cpp_example.cpp](examples/cpp_example.cpp), build and run it with
[examples/cpp_example_run.sh](examples/cpp_example_run.sh)):

```cpp
#include "inpaint.h"

int main() {
    cv::Mat image = ...;
    cv::Mat mask = ...;

    auto metric = PatchSSDDistanceMetric(5);
    cv::Mat result = Inpainting(image, mask, &metric).run();
}
```

The library is built with CMake; `PATCHMATCH_BUILD_EXAMPLES` builds the example and
`PATCHMATCH_WITH_HIGHGUI` enables the debug visualization of `Inpainting::run`.

## Development

The project is managed with [uv](https://docs.astral.sh/uv/):

```sh
uv sync                        # create .venv, build the library, install dev deps
uv run pre-commit install      # enable ruff and the other hooks on commit
uv run pytest                  # run the test suite
uv run pytest -m "not e2e"     # only the fast unit tests
```

Releases are published to PyPI by creating a GitHub release; the version is taken from
its tag (e.g. `v1.1.0`).

## README and COPYRIGHT by Younesse ANDAM

@Author: Younesse ANDAM

@Contact: younesse.andam@gmail.com

Description:

This project is a personal implementation of an algorithm called PATCHMATCH
that restores missing areas in an image. The algorithm is presented in the following
paper PatchMatch A Randomized Correspondence Algorithm for Structural Image Editing by
C.Barnes, E.Shechtman, A.Finkelstein and Dan B.Goldman ACM Transactions on Graphics
(Proc. SIGGRAPH), vol.28, aug-2009

For more information please refer to
http://www.cs.princeton.edu/gfx/pubs/Barnes_2009_PAR/index.php

Copyright (c) 2010-2011

## Requirements

To run the project you need to install Opencv library and link it to your project.
Opencv can be download it here
http://opencv.org/downloads.html
