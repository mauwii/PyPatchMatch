"""Build a minimal static OpenCV (core module only) for the release wheels.

The patchmatch library only needs opencv_core. Linking it statically keeps the
wheels small and free of OpenCV's GUI/codec dependencies.

Usage: python build_opencv.py  (installs into $OpenCV_ROOT)
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

OPENCV_VERSION = "4.14.0"
OPENCV_URL = (
    f"https://github.com/opencv/opencv/archive/refs/tags/{OPENCV_VERSION}.tar.gz"
)
# The wheels link this code statically, so the archive is pinned by its SHA-256.
# Update it together with the version, e.g. with `sha256sum` or `shasum -a 256`.
OPENCV_SHA256 = "ee8fb9b30eb60850431b4656447080e3737b56e45719c92b67f245950609f86e"

CMAKE_OPTIONS = {
    "CMAKE_BUILD_TYPE": "Release",
    "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
    "BUILD_LIST": "core",
    "BUILD_SHARED_LIBS": "OFF",
    # patchmatch links the static MSVC runtime as well, see CMakeLists.txt
    "BUILD_WITH_STATIC_CRT": "ON",
    "BUILD_ZLIB": "ON",
    "BUILD_TESTS": "OFF",
    "BUILD_PERF_TESTS": "OFF",
    "BUILD_EXAMPLES": "OFF",
    "BUILD_DOCS": "OFF",
    "BUILD_JAVA": "OFF",
    "BUILD_opencv_apps": "OFF",
    "BUILD_opencv_python3": "OFF",
    "OPENCV_GENERATE_PKGCONFIG": "OFF",
    # CMakeLists.txt ships the licenses from here with the wheels
    "OPENCV_LICENSES_INSTALL_PATH": "licenses",
    # only core is built, so disable every optional backend and codec
    "WITH_ADE": "OFF",
    "WITH_AVIF": "OFF",
    # ARM HAL libraries, patchmatch uses no function they accelerate
    "WITH_CAROTENE": "OFF",
    "WITH_KLEIDICV": "OFF",
    "WITH_EIGEN": "OFF",
    "WITH_FFMPEG": "OFF",
    "WITH_GSTREAMER": "OFF",
    "WITH_GTK": "OFF",
    "WITH_JASPER": "OFF",
    "WITH_JPEG": "OFF",
    "WITH_OPENEXR": "OFF",
    "WITH_OPENGL": "OFF",
    "WITH_OPENJPEG": "OFF",
    "WITH_PNG": "OFF",
    "WITH_PROTOBUF": "OFF",
    "WITH_QT": "OFF",
    "WITH_TIFF": "OFF",
    "WITH_V4L": "OFF",
    "WITH_WEBP": "OFF",
    "WITH_IPP": "OFF",
    "WITH_ITT": "OFF",
    "WITH_LAPACK": "OFF",
    "WITH_OPENCL": "OFF",
    "WITH_OPENMP": "OFF",
    "WITH_TBB": "OFF",
}

# Install a plain CMake config instead of OpenCV's "Windows pack" wrapper, which
# guesses the <arch>/<vc runtime>/ subdirectory from the consuming compiler and
# fails for newer MSVC versions and static builds. The config must not be in the
# install root: CMake would then resolve the import prefix one level too high.
WINDOWS_CMAKE_OPTIONS = {
    "OPENCV_CONFIG_INSTALL_PATH": "cmake",
    "OPENCV_INSTALL_BINARIES_PREFIX": "",
    "OPENCV_SKIP_CMAKE_ROOT_CONFIG": "ON",
}


def find_cmake() -> str:
    cmake = shutil.which("cmake")
    if cmake:
        return cmake
    subprocess.run([sys.executable, "-m", "pip", "install", "cmake"], check=True)
    scripts = Path(sys.executable).parent
    return str(scripts / ("cmake.exe" if os.name == "nt" else "cmake"))


def verify_archive(archive: Path) -> None:
    sha256 = hashlib.sha256()
    with archive.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha256.update(chunk)
    if sha256.hexdigest() != OPENCV_SHA256:
        sys.exit(
            f"SHA-256 mismatch for {OPENCV_URL}: "
            f"expected {OPENCV_SHA256}, got {sha256.hexdigest()}"
        )


def write_sbom(build: Path, path: Path) -> None:
    # Scanners cannot see statically linked code, so the wheels ship a CycloneDX SBOM
    # (PEP 770). The bundled zlib is left out: only FileStorage uses it, and the wheel
    # job checks that none of it is linked. A new 3rdparty library has to be added.
    bundled = {
        lib.stem.removeprefix("lib")
        for lib in (build / "3rdparty" / "lib").rglob("*")
        if lib.suffix in {".a", ".lib"}
    }
    if bundled - {"zlib"}:
        sys.exit(f"Add the bundled 3rdparty libraries {sorted(bundled)} to the SBOM")
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "components": [
            {
                "type": "library",
                "bom-ref": "opencv",
                "name": "opencv",
                "version": OPENCV_VERSION,
                "description": "OpenCV core module, linked statically",
                "licenses": [{"license": {"id": "Apache-2.0"}}],
                "purl": f"pkg:github/opencv/opencv@{OPENCV_VERSION}",
                "cpe": f"cpe:2.3:a:opencv:opencv:{OPENCV_VERSION}:*:*:*:*:*:*:*",
                "hashes": [{"alg": "SHA-256", "content": OPENCV_SHA256}],
                "externalReferences": [{"type": "distribution", "url": OPENCV_URL}],
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sbom, indent=2) + "\n")


def main() -> None:
    # OpenCV_ROOT is the variable CMake's find_package(OpenCV) looks for.
    root = os.environ.get("OpenCV_ROOT")  # noqa: SIM112
    if not root:
        sys.exit("OpenCV_ROOT must be set to the installation prefix")
    prefix = Path(root)
    if any(prefix.rglob("OpenCVConfig.cmake")):
        print(f"OpenCV already installed in {prefix}")
        return

    cmake = find_cmake()
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "opencv.tar.gz"
        print(f"Downloading {OPENCV_URL}", flush=True)
        urllib.request.urlretrieve(OPENCV_URL, archive)
        verify_archive(archive)
        with tarfile.open(archive) as tar:
            # extraction filters are missing on older patch releases (3.11 < 3.11.4)
            if hasattr(tarfile, "data_filter"):
                tar.extractall(tmp, filter="data")
            else:
                tar.extractall(tmp)

        source = Path(tmp) / f"opencv-{OPENCV_VERSION}"
        build = Path(tmp) / "build"
        options = {**CMAKE_OPTIONS, "CMAKE_INSTALL_PREFIX": prefix.as_posix()}
        if sys.platform == "win32":
            options.update(WINDOWS_CMAKE_OPTIONS)
        defines = [f"-D{key}={value}" for key, value in options.items()]
        subprocess.run([cmake, "-S", source, "-B", build, *defines], check=True)
        # without a count, `make -j` starts all compilations at once
        jobs = str(os.cpu_count() or 1)
        subprocess.run(
            [cmake, "--build", build, "--config", "Release", "--parallel", jobs],
            check=True,
        )
        subprocess.run([cmake, "--install", build, "--config", "Release"], check=True)

        # OpenCV installs only the licenses of its 3rdparty code
        licenses = prefix / CMAKE_OPTIONS["OPENCV_LICENSES_INSTALL_PATH"]
        licenses.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "LICENSE", licenses / "opencv-LICENSE")
        write_sbom(build, prefix / "sboms" / "opencv.cdx.json")


if __name__ == "__main__":
    main()
