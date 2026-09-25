"""PatchMatch based inpainting."""

__version__ = "1.1.0"

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
