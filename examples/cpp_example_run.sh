#! /bin/bash
#
# cpp_example_run.sh
# Copyright (C) 2020 Jiayuan Mao <maojiayuan@gmail.com>
#
# updated by Matthias Wild <mauwii@outlook.de>
#
# Distributed under terms of the MIT license.
#

set -x

cd "$(dirname "$0")" || exit 1

CFLAGS=("-std=c++17" "-Ofast")
pkg-config --cflags opencv &>/dev/null && CFLAGS+=("$(pkg-config --cflags opencv)")
pkg-config --cflags opencv4 &>/dev/null && CFLAGS+=("$(pkg-config --cflags opencv4)")
pkg-config --libs opencv &>/dev/null && LDFLAGS="$(pkg-config --libs opencv)"
pkg-config --libs opencv4 &>/dev/null && LDFLAGS="$(pkg-config --libs opencv4)"
c++ "${CFLAGS[@]}" cpp_example.cpp -I../patchmatch/csrc/ -L../patchmatch/ -lpatchmatch $LDFLAGS -o cpp_example.exe

if [[ "$(uname -s)" == "Darwin" ]]; then
    export DYLD_LIBRARY_PATH="../patchmatch/${DYLD_LIBRARY_PATH:+:DYLD_LIBRARY_PATH}"  # For macOS
elif [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH="../patchmatch/${LD_LIBRARY_PATH:+:LD_LIBRARY_PATH}"  # For Linux
fi

time ./cpp_example.exe
