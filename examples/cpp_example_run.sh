#!/usr/bin/env bash
#
# Build the library and the C++ example with CMake and run it.
#
# Distributed under terms of the MIT license.

set -euo pipefail

cd "$(dirname "$0")"

cmake -S .. -B ../build/cpp -DPATCHMATCH_BUILD_EXAMPLES=ON
cmake --build ../build/cpp --parallel

time ../build/cpp/cpp_example
