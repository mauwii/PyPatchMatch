#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : patch_match.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/09/2020
#
# Distributed under terms of the MIT license.

# Additional modifications by Kyle Schouviller

# Reformated by Matthias Wild


import ctypes
import json
import logging
import os
import os.path as osp
import platform
import shutil
import tempfile
from typing import Optional, Union
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
from tqdm import tqdm

# This will be True if patchmatch was loaded successfully
patchmatch_available = False

# The GitHub release information
repo = "https://api.github.com/repos/invoke-ai/PyPatchMatch/"
release_id = "tags/0.1.1"
release_url = f"{repo}releases/{release_id}"

install_help_location = (
    "https://invoke-ai.github.io/InvokeAI/installation/060_INSTALL_PATCHMATCH/"
)

# Create the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)  # TODO: make this user-configurable
stream_format = logging.Formatter(">> %(name)s: %(levelname)s - %(message)s")
stream_handler.setFormatter(stream_format)
logger.addHandler(stream_handler)


__all__ = ["set_random_seed", "set_verbose", "inpaint", "inpaint_regularity"]


class CShapeT(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("channels", ctypes.c_int),
    ]


class CMatT(ctypes.Structure):
    _fields_ = [
        ("data_ptr", ctypes.c_void_p),
        ("shape", CShapeT),
        ("dtype", ctypes.c_int),
    ]


def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    r"""Download object at the given URL to a local path.

    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. ``/tmp/temporary_file``
        hash_prefix (string, optional): If not None, the SHA256 downloaded file should start with ``hash_prefix``. Default: None progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True
    https://pytorch.org/docs/stable/_modules/torch/hub.html#load_state_dict_from_url
    """
    file_size = None
    req = Request(url)
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, "getheaders"):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        with tqdm(
            total=file_size,
            disable=not progress,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                pbar.update(len(buffer))

        f.close()
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


try:
    # Get release information from github
    release_response = urlopen(release_url)
    release_json = json.loads(release_response.read())

    # Filter to assets for platform
    platform_slug = f"{platform.system().lower()}_{platform.machine().lower()}"
    platform_assets = list(
        filter(lambda a: platform_slug in a["name"], release_json["assets"])
    )
    if "windows" in platform_slug:
        platform_assets.extend(
            filter(
                lambda a: a["name"] == "opencv_world460.dll",
                release_json["assets"],
            )
        )

    # Get assets
    pypatchmatch_lib = None
    for asset in platform_assets:
        lib_name = asset["name"]
        lib_url = asset["browser_download_url"]

        if not os.path.exists(osp.join(osp.dirname(__file__), lib_name)):
            logger.info(
                f"Downloading patchmatch libraries from github release {lib_url}"
            )
            download_url_to_file(
                url=lib_url, dst=osp.join(osp.dirname(__file__), lib_name)
            )

        # Store patchmatch library name
        if lib_name.startswith("libpatchmatch_"):
            pypatchmatch_lib = lib_name

    # Compile if we didn't find a platform-compatible version (and it's not compiled already)
    if pypatchmatch_lib is None:
        pypatchmatch_lib = "libpatchmatch.so"
        if not os.path.exists(osp.join(osp.dirname(__file__), pypatchmatch_lib)):
            import subprocess

            # Streams make will write to
            # TODO: use user-configured log-level to control this
            make_stdout = subprocess.DEVNULL
            make_stderr = subprocess.DEVNULL

            if os.environ.get("INVOKEAI_DEBUG_PATCHMATCH"):
                make_stdout = None
                make_stderr = None

            logger.info(
                'Compiling and loading c extensions from "{}".'.format(
                    osp.realpath(osp.dirname(__file__))
                )
            )
            # subprocess.check_call(['./travis.sh'], cwd=osp.dirname(__file__))
            # TODO: pipe output to logger instead of just swallowing it
            subprocess.run(
                "make clean && make",
                cwd=osp.dirname(__file__),
                shell=True,
                check=True,
                stdout=make_stdout,
                stderr=make_stderr,
            )

    PMLIB = ctypes.CDLL(osp.join(osp.dirname(__file__), pypatchmatch_lib))
    patchmatch_available = True

    PMLIB.PM_set_random_seed.argtypes = [ctypes.c_uint]
    PMLIB.PM_set_verbose.argtypes = [ctypes.c_int]
    PMLIB.PM_free_pymat.argtypes = [CMatT]
    PMLIB.PM_inpaint.argtypes = [CMatT, CMatT, ctypes.c_int]
    PMLIB.PM_inpaint.restype = CMatT
    PMLIB.PM_inpaint_regularity.argtypes = [
        CMatT,
        CMatT,
        CMatT,
        ctypes.c_int,
        ctypes.c_float,
    ]
    PMLIB.PM_inpaint_regularity.restype = CMatT
    PMLIB.PM_inpaint2.argtypes = [CMatT, CMatT, CMatT, ctypes.c_int]
    PMLIB.PM_inpaint2.restype = CMatT
    PMLIB.PM_inpaint2_regularity.argtypes = [
        CMatT,
        CMatT,
        CMatT,
        CMatT,
        ctypes.c_int,
        ctypes.c_float,
    ]
    PMLIB.PM_inpaint2_regularity.restype = CMatT

    def set_random_seed(seed: int):
        PMLIB.PM_set_random_seed(ctypes.c_uint(seed))

    def set_verbose(verbose: bool):
        PMLIB.PM_set_verbose(ctypes.c_int(verbose))

    def inpaint(
        image: Union[np.ndarray, Image.Image],
        mask: Optional[Union[np.ndarray, Image.Image]] = None,
        *,
        global_mask: Optional[Union[np.ndarray, Image.Image]] = None,
        patch_size: int = 15,
    ) -> np.ndarray:
        """
        PatchMatch based inpainting proposed in:

            PatchMatch : A Randomized Correspondence Algorithm for Structural
            Image Editing
            C.Barnes, E.Shechtman, A.Finkelstein and Dan B.Goldman
            SIGGRAPH 2009

        Args:
            image (Union[np.ndarray, Image.Image]): the input image, should be 3-channel RGB/BGR.
            mask (Union[np.array, Image.Image], optional): the mask of the hole(s) to be filled, should be 1-channel.
            If not provided (None), the algorithm will treat all purely white pixels as the holes (255, 255, 255).
            global_mask (Union[np.array, Image.Image], optional): the target mask of the output image.
            patch_size (int): the patch size for the inpainting algorithm.

        Return:
            result (np.ndarray): the repaired image, of the same size as the input image.
        """

        if isinstance(image, Image.Image):
            image = np.array(image)
        image = np.ascontiguousarray(image)
        assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == "uint8"

        if mask is None:
            mask = (image == (255, 255, 255)).all(axis=2, keepdims=True).astype("uint8")
            mask = np.ascontiguousarray(mask)
        else:
            mask = _canonize_mask_array(mask)

        if global_mask is None:
            ret_pymat = PMLIB.PM_inpaint(
                np_to_pymat(image), np_to_pymat(mask), ctypes.c_int(patch_size)
            )
        else:
            global_mask = _canonize_mask_array(global_mask)
            ret_pymat = PMLIB.PM_inpaint2(
                np_to_pymat(image),
                np_to_pymat(mask),
                np_to_pymat(global_mask),
                ctypes.c_int(patch_size),
            )

        ret_npmat = pymat_to_np(ret_pymat)
        PMLIB.PM_free_pymat(ret_pymat)

        return ret_npmat

    def inpaint_regularity(
        image: Union[np.ndarray, Image.Image],
        mask: Optional[Union[np.ndarray, Image.Image]],
        ijmap: np.ndarray,
        *,
        global_mask: Optional[Union[np.ndarray, Image.Image]] = None,
        patch_size: int = 15,
        guide_weight: float = 0.25,
    ) -> np.ndarray:
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = np.ascontiguousarray(image)

        assert (
            isinstance(ijmap, np.ndarray)
            and ijmap.ndim == 3
            and ijmap.shape[2] == 3
            and ijmap.dtype == "float32"
        )
        ijmap = np.ascontiguousarray(ijmap)

        assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == "uint8"
        if mask is None:
            mask = (image == (255, 255, 255)).all(axis=2, keepdims=True).astype("uint8")
            mask = np.ascontiguousarray(mask)
        else:
            mask = _canonize_mask_array(mask)

        if global_mask is None:
            ret_pymat = PMLIB.PM_inpaint_regularity(
                np_to_pymat(image),
                np_to_pymat(mask),
                np_to_pymat(ijmap),
                ctypes.c_int(patch_size),
                ctypes.c_float(guide_weight),
            )
        else:
            global_mask = _canonize_mask_array(global_mask)
            ret_pymat = PMLIB.PM_inpaint2_regularity(
                np_to_pymat(image),
                np_to_pymat(mask),
                np_to_pymat(global_mask),
                np_to_pymat(ijmap),
                ctypes.c_int(patch_size),
                ctypes.c_float(guide_weight),
            )

        ret_npmat = pymat_to_np(ret_pymat)
        PMLIB.PM_free_pymat(ret_pymat)

        return ret_npmat

    def _canonize_mask_array(mask):
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        if mask.ndim == 2 and mask.dtype == "uint8":
            mask = mask[..., np.newaxis]
        assert mask.ndim == 3 and mask.shape[2] == 1 and mask.dtype == "uint8"
        return np.ascontiguousarray(mask)

    dtype_pymat_to_ctypes = [
        ctypes.c_uint8,
        ctypes.c_int8,
        ctypes.c_uint16,
        ctypes.c_int16,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_double,
    ]

    dtype_np_to_pymat = {
        "uint8": 0,
        "int8": 1,
        "uint16": 2,
        "int16": 3,
        "int32": 4,
        "float32": 5,
        "float64": 6,
    }

    def np_to_pymat(npmat):
        assert npmat.ndim == 3
        return CMatT(
            ctypes.cast(npmat.ctypes.data, ctypes.c_void_p),
            CShapeT(npmat.shape[1], npmat.shape[0], npmat.shape[2]),
            dtype_np_to_pymat[str(npmat.dtype)],
        )

    def pymat_to_np(pymat):
        npmat = np.ctypeslib.as_array(
            ctypes.cast(
                pymat.data_ptr,
                ctypes.POINTER(dtype_pymat_to_ctypes[pymat.dtype]),
            ),
            (pymat.shape.height, pymat.shape.width, pymat.shape.channels),
        )
        ret = np.empty(npmat.shape, npmat.dtype)
        ret[:] = npmat
        return ret

except Exception as e:
    logger.error(f"patchmatch failed to load or compile ({e}).")
    logger.info(f"Refer to {install_help_location} for installation instructions.")
