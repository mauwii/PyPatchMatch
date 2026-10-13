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
- The larger patches of `_fill_coarse_levels` without the energy comparison: they lock
  onto repeating patterns but take flat sources in foliage, which filled holes in
  `forest.bmp` gray (11 of 20 positions worse, `test_reconstructs_known_background`
  5.9 → 8.4). Also rejected: voting with them (coffee detail 0.34 → 0.29), a radius of
  7 (coffee error 17.5 → 20.0, up to 3× the time) or 4 (a fifth faster, but the
  pattern with a hole of radius 50 stayed broken), the energy on the level right below
  them (border patches dominate it), a margin in their favor (from 1.05 on, forest
  fills got worse), and choosing by the depth of the hole (the same for the failing
  pattern and the forest holes).
- Split votes (mean of the low frequencies plus the detail of the best vote) on coarser
  levels than the two finest: from level 2 on they helped coffee with `patch_size` 3, but
  raised the error of brick with 7 from 6.96 to 8.99 and of brick-global from 6.30 to
  7.82. A low pass wider than 3x3 gained nothing with `patch_size` 7. Also rejected
  against #148 (smooth sources for smooth fills): comparing the hole pixels of the fill
  with the low pass of the sources and without gradients (brick 8.6 → 15.7, the mortar
  got lost), and a penalty for sources with less gradient energy than the ring around
  the holes (coffee with 7 improved, chelsea 14.8 → 19.8 and grass 18.9 → 25.4). With
  the energy of the known pixels around each patch as the reference instead, it copied
  salient sources such as a clover into grass (25.8) on the coarse levels; on levels 0
  and 1 alone it had the best numbers with 7 (coffee detail 0.35 → 1.09, error 19.2 →
  17.8, brick 7.6 → 9.0), but filled the holes with hard-edged blocks of differently
  oriented fur and specks in the bricks. A band that also penalizes sources with more
  energy did not help grass.
- Other forms of the texture deficit in `RowPair::cost`: a texture channel voted like
  the colors on every level (Newson et al. 2014) takes the texture of whatever the
  coarse levels chose (coffee with 7 copied the reddish wood beside the saucer, error
  19.2 → 22.4); a
  two-sided squared difference prescribes the mean texture and drops sparse structures
  such as mortar joints (brick with 3: 10.0 → 11.8), also with windows of 13x13 and
  25x25. A margin of 3/4 or a weight of 4 broke pattern with 3 (5.4 → 14.1); a weight
  of 1 and windows from 9x9 to 33x33 were within the noise. brick-global with 3 looked
  worse with 10 seeds (5.8 → 6.8), but main has 6.8 ± 0.3 with 40.
- Other references for level 0 than twice the voted texture of the sources of level 1:
  the texture around the holes with a margin of 1 or a weight of 8 there broke pattern
  (with 3: 5.2 → 7.0, with 7: 2.6 → 4.0); voting it into level 1 as well raised the
  detail of coffee with 3 to 1.27 and broke pattern with 7 (3.3); the voted texture
  itself left chelsea with 15 flat (detail 0.70), 1.5 times it helped less, 3 times it
  raised brick with 15 from 6.1 to 6.4; the texture of the best vote instead of the
  mean was slightly worse.
- The fitting metrics without the minimum of 7: with `patch_size` 7, brick rose from
  6.8 to 8.2 and grass from 18.8 to 21.3. A fourth or an eighth of the smaller side
  instead of a sixth: mean error ratio with 15 of 0.849 and 0.846 instead of 0.835.
- More, with measurements, in the issues closed as not planned: #115 (start radius of
  the new-hole search, merging random fields), #120 (several cores) and #124 (confidence
  weighting for large holes).
