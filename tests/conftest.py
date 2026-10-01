from __future__ import annotations

from hypothesis import settings

# nightly CI; print_blob shows how to reproduce a failure locally
settings.register_profile("deep", max_examples=5000, print_blob=True)
