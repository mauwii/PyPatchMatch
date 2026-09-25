# PatchMatch based Inpainting

[![License: MIT](https://img.shields.io/badge/License-MIT-blueviolet.svg)](https://github.com/mauwii/PyPatchMatch/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/PyPatchMatch)](https://pypi.org/project/PyPatchMatch/)
[![Downloads](https://static.pepy.tech/badge/pypatchmatch)](https://pepy.tech/projects/pypatchmatch)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

This library implements the PatchMatch based inpainting algorithm. It provides both C++
and Python interfaces. This implementation is heavily based on the implementation by
Younesse ANDAM: [younesse-cv/PatchMatch](https://github.com/younesse-cv/PatchMatch),
with some bug fixes, and updates.

## Installation

```sh
pip install PyPatchMatch
```

Wheels for Linux (glibc, x86_64 and aarch64), macOS 11+ (arm64 and x86_64) and Windows
(x64) ship the compiled library with a statically linked OpenCV, so no compiler or
OpenCV installation is needed. On other platforms, e.g. Alpine Linux or Windows on ARM,
pip builds from the source distribution, which requires a C++17 compiler, CMake and the
OpenCV development files (e.g. `apt install libopencv-dev` or `brew install opencv`).

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
pure white pixels are treated as holes. The optional keyword argument `global_mask`, in
the same format, marks pixels that are neither filled nor used as a source; they keep
their values (see
[examples/py_example_global_mask.py](https://github.com/mauwii/PyPatchMatch/blob/main/examples/py_example_global_mask.py)).
`patchmatch.patchmatch_available` tells whether the native library could be loaded.
The previous import path `from patchmatch import patch_match` keeps working.

`patchmatch.set_random_seed(seed)` sets the seed of the randomized search and
`patchmatch.set_verbose(True)` prints the progress of the native code to stderr.
`patchmatch.inpaint_regularity(image, mask, ijmap)` additionally guides the search with
a regularity map, an HxWx3 float32 array with the regularity coordinates of each pixel.

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
its tag (e.g. `v2.0.0`).

## License

PyPatchMatch is released under the
[MIT License](https://github.com/mauwii/PyPatchMatch/blob/main/LICENSE). The wheels
contain a statically linked build of OpenCV core (Apache-2.0) and its bundled
third-party code; their licenses are included in the `licenses/opencv` directory of the
wheel's `.dist-info`.

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
