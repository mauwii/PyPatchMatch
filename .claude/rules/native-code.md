---
paths:
  - "patchmatch/csrc/**"
  - "CMakeLists.txt"
  - "examples/cpp_example.cpp"
---

# Native code

The comments in the code explain how it works; this lists the invariants and the
alternatives that were measured and rejected. Measure against these before trying one
again.

## Invariants

- ctypes releases the GIL, so inpaintings run concurrently
  (`test_concurrent_inpaint_is_deterministic`). Keep all mutable state per call or per
  thread, so that a fill depends only on its input and the seed. `random_int` stays
  `engine() % n` instead of `std::uniform_int_distribution`, which gives the same
  sequence on every platform.
- Only the holes change, and the colors under the holes and under `global_mask` do not
  influence the fill (`assert_filled`, `assert_plausible_fill`,
  `test_colors_under_the_holes_do_not_matter`,
  `test_colors_under_global_mask_do_not_matter`).
- Every distance metric returns an int in `[0, PatchDistanceMetric::kDistanceScale]`,
  because the distance indexes `distance2similarity`. The table is zero from 10 % of the
  scale on, which a patch reaches with about 10 % of masked pixels or pixels outside the
  image: such patches cast no votes.

## OpenCV and library size

- The release wheels link a static, core-only OpenCV (`scripts/build_opencv.py`). Use
  plain pixel loops instead of `cv::Mat` matrix expressions such as `a == b` or
  `m1 | m2`: they pull OpenCV's expression module, and with it `FileStorage`, PCA and
  `SparseMat`, into the library (about 40 % larger).
- On Windows the library uses the static MSVC runtime of that OpenCV, whose stderr
  pytest's `capfd` does not capture, so tests of native output run in a subprocess
  (`test_set_verbose_prints_progress`).

## Tried and rejected

- The completeness field (source to target) on no level, as in the paper (section 4):
  it is left out on levels 0 and 1 only, where its votes changed nothing and cost a
  quarter of the time. On the coarser levels they keep the known pixels near their
  input; without them the error of the fills rose by up to 69 % (trees, `patch_size`
  15).
- Marking every coarse pixel that touches a hole as a hole, instead of those whose whole
  downsampling kernel is one: it leaves the coarsest levels without valid source patches
  (the 10 % above). The holes are meant to shrink quickly, be filled smoothly from their
  border, and get their texture on the finer levels.
- Keeping the distances of the links of patches with holes after an EM iteration: worse
  candidates replaced better links, and the votes were weighted with outdated distances.
- A stronger preference for the patch centered closest in the best vote of level 0 than
  `1 - 1e-5 * (di² + dj²)`, such as `1e-4`, a Gaussian or `1 / (1 + r²)`: it overrides
  real differences, and on the color gradient of the unit tests a patch copied colors
  from far away (`test_inpaint_patch_sizes[7]`).
- The weighted mean at level 0, or a blend of it with the best vote: with the known
  pixels kept, the mean of `patch_size` 15 is flat (detail 0.08 for chelsea).
- A width of `blend_hole_borders` that grows with `patch_size`: worse than 4 pixels.
- Propagating before searching around a patch next to a hole that is new on its level
  (`_initialize_field_from`): its link to itself points into the hole, the first
  propagation replaces it with a neighbor's link, and the random search around that does
  not find the way back to the surroundings of the patch; the stem of
  `forest_pruned.bmp` was filled with forest instead of meadow
  (`test_inpaint_thin_hole_takes_the_colors_beside_it`).
  Also rejected: searching before propagating in every pass, which no longer refines a
  propagated link along a structure; the search radius of the larger image side from
  the paper, which blurs the fills; a radius of 1 on level 0 (paper section 4.3), which
  saved at most 10 % and made the seam worse with `patch_size` 7 and 15.
- Counting the outermost line of the image as maximal instead of replicating its pixels
  for the gradients: up to `patch_size` 4 every patch over it is above the 10 %, so that
  line of a hole at the image border never got a vote and stayed flat
  (`test_inpaint_textures_the_image_border`).
- More, with measurements, in the issues closed as not planned: #115 (start radius of
  the new-hole search, merging random fields), #120 (several cores) and #124 (confidence
  weighting for large holes).
