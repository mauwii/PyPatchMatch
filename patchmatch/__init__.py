"""PatchMatch based inpainting."""

from ._version import __version__
from .patch_match import (
    inpaint,
    inpaint_regularity,
    patchmatch_available,
    set_random_seed,
    set_verbose,
)

__all__ = [
    "__version__",
    "inpaint",
    "inpaint_regularity",
    "patchmatch_available",
    "set_random_seed",
    "set_verbose",
]
