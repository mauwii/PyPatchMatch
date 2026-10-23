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
- The larger patches of `m_coarse_metric` on the level below the coarsest with a hole
  as well: the holes there are wider than them, and they take flat sources in foliage,
  which filled holes in `forest.bmp` gray (11 of 20 positions worse,
  `test_reconstructs_known_background` 5.9 → 8.4). Until 2.2 both levels were filled
  with patch_size and with the larger patches, and the fill with the lower mean
  distance with patch_size on the next level was kept. That distance does not see
  whether structures line up: right in 173 of 280 seeds (10 seeds of 19 cases and 20
  holes in `forest.bmp` where the errors differ by 5 %), brick and pattern were better
  with the larger patches in 10 of 10 seeds but got them in 5 and 1. On the coarsest
  level alone and without a choice (patch size 3, 10 seeds): error −5.9 %, edges
  −1.9 %, 10 to 50 % faster; on other holes and seeds −6.9 % on the cases, −8.2 % on 20
  holes in `forest.bmp` and −5.1 % on 12 in coffee, chelsea and camera, but their detail
  −3.6 % and variety −6.3 %: chelsea copies one region of fur (variety 0.81 → 0.45).
  Two parts of its hole copy the same source 12 by 24 to 28 pixels apart (28 to 46 % of
  the 8x8 patches have a twin), laid out on the coarsest level and grown by the EM of the
  next two; every variant without the repetition raised its error by 8 to 12 %:
  patch_size there, patch size 4 or 5 on the level below (holes in `forest.bmp` +2.5
  and +5 %), a radius capped by the side of the level (brick 6.76 → 9.60).
  With 30 seeds brick 10.3 → 7.5 and pattern 5.3 → 3.6, pattern-wide 5.2 → 5.9. Holes
  that leave level 1 or 0 changed within 1 %. Other choices between the two fills: the
  distance with patch sizes 2 to 7, of the patches inside the holes only, with twice
  the passes, and their sums gained at most 2 % on the cases; vetoes by the detail or
  the tone of a fill against its surroundings did worse; the plain fill against the
  new one by the distance with patch size 4 (−2.7 % on other holes and seeds) or a
  choice among all three (−3.5 %) cost the second path. Also rejected, measured with
  the comparison: voting with the larger patches (coffee detail 0.34 → 0.29), a
  radius of 7 (coffee error 17.5 → 20.0, up to 3× the time) or 4 (a fifth faster, but
  the pattern with a hole of radius 50 stayed broken), a margin in their favor (from
  1.05 on, forest fills got worse), and choosing by the depth of the hole (the same for
  the failing pattern and the forest holes).
- Split votes (mean of the low frequencies plus the detail of the best vote) on coarser
  levels than the two finest: from level 2 on they helped coffee with `patch_size` 3,
  but raised the error of brick with 7 from 6.96 to 8.99 and of brick-global from 6.30
  to 7.82; `edges` agrees (#157, with 15: coffee 6.59 → 8.59, grass 7.00 → 8.28). A low
  pass wider than 3x3 gained nothing with `patch_size` 7. Also rejected against #148
  (smooth sources for smooth fills): comparing the hole pixels of the fill with the low
  pass of the sources and without gradients (brick 8.6 → 15.7, the mortar got lost), and
  a penalty for sources with less gradient energy than the ring around the holes (coffee
  with 7 improved, chelsea 14.8 → 19.8 and grass 18.9 → 25.4). With the energy of the
  known pixels around each patch as the reference instead, it copied salient sources
  such as a clover into grass (25.8) on the coarse levels; on levels 0 and 1 alone it
  had the best numbers with 7 (coffee detail 0.35 → 1.09, error 19.2 → 17.8, brick 7.6 →
  9.0), but filled the holes with hard-edged blocks of differently oriented fur and
  specks in the bricks. A band that also penalizes sources with more energy did not help
  grass.
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
  6.8 to 8.2 and grass from 18.8 to 21.3. A minimum of 4 did not help brick-large
  either (#157, `edges` with 15: 4.59 → 5.11; brick with 7: 2.96 → 3.88). A fourth or
  an eighth of the smaller side instead of a sixth: mean error ratio with 15 of 0.849
  and 0.846 instead of 0.835.
- Rotated sources (generalized PatchMatch, Barnes et al. 2010) for the perspective wall
  of brick-large, simulated with rotated copies of the image as extra sources: the
  stripes kept their missing joints, and the error rose with every patch size.
- A penalty for the distance between a patch and its source, so that the inside of a
  large hole continues the nearest border: brick-large continued each side but got
  seams where they met; with a weight of 200 grass, trees, pattern-wide and
  brick-global got worse, with 500 pattern-wide with 3 broke (error 5.31 → 44.50).
- Weighting the votes with `exp(-d / 2p)`, p a percentile of the distances of the field
  (Wexler et al. 2007), instead of `distance2similarity`, which gives every match up to
  an RMS error of about 40 per channel a weight of 0.90 or more: on all averaged votes
  or on the coarse levels only, with the median or the 75th percentile, the mean error
  stayed within 2 % (10 seeds). Like the split votes on coarser levels, it helped brick
  with 3 (10.66 → 9.07) and hurt it with 7 (6.75 → 7.67, brick-global 6.06 → 7.14).
- Spreading the color steps at the border over the whole hole with `pull_push` (a
  membrane, as in seamless cloning) instead of the ramp of `blend_hole_borders`: camera
  improved a lot (with 7: 7.94 → 4.28, with 15: 8.89 → 3.79), so its fill is off in
  color as a whole, but in textures the steps are noise that shifted the whole fill
  (grass with 15: 18.56 → 23.66, pattern with 7: 2.50 → 4.38); the mean error rose by
  0.5, 3.2 and 6.3 % with 3, 7 and 15. With the raw steps instead of their mean along
  the border, the seam fell below that of the originals (0.69 to 0.76).
- Other forms of the color shift of the votes for levels 0 and 1: also compensating it
  in the distances (Image Melding, Darabi et al. 2012) did worse than the votes alone,
  and in the distances alone the fill took the colors of brighter or darker sources
  (camera with 3: 5.1 → 29.8). On the coarser levels too it lost the layout (coffee
  with 15: 18.7 → 21.4, on them alone 25.3). The median of the differences kept
  pattern but broke bricks (brick-global with 3: 6.9 → 9.0, brick-large with 15: 16.0
  → 19.4); a shift of the luminance only, or a limit of 2 or 4, gained less for the
  same losses. The split votes or the best vote alone gained half (error with 7 −2.3
  and −2.6 % instead of −5.3 %).
- More EM iterations or NNF passes on level 0 (1 and 1) or level 1 (3 and 2): level 0
  is converged; 3 iterations with 2 passes stayed within the noise (error +0.2 to
  +0.5 %, 3 seeds) for up to 95 % more time, and 5 iterations on level 1 raised the
  error by up to 1.9 %. A single iteration on level 1 saved 8 to 21 % of the time but
  broke the patterns (pattern-wide with 15: 3.41 → 4.08) and brick with 15 (6.13 →
  6.62).
- Preferring the best votes of patches with known pixels on level 0, weight ×
  (1 + β · their known fraction): at the border of stochastic textures the best vote
  comes from the patch shifted farthest into the hole, which matches the fill copied
  from it (grass with 7: offset 6.88 of 7, 59 % with a known pixel; brick picks
  centered patches). With β 0.02 the means improved (10 seeds: seam −7 to −10 %, error
  up to −5.9 %), but the seam moved inward: a ring of splinters in gravel, a block of
  fur in chelsea, fragments in trees and meadow, which `seam` does not see. Favoring
  every patch with a known pixel equally let those with one known row win, and the seam
  stayed: each patch with more known pixels matches worse, so the choice of the vote
  only moves the seam. Neither boosting only the pixels within patch_size of the border
  nor anchoring the split votes too, with 3 iterations for the inside to adapt, helped.
- Other weights of the features in `RowPair::cost`: luminance and half the chroma
  instead of the channels (coffee with 15: 18.7 → 27.1), or the gradients weighted 2 or
  4 times, did worse; the gradients at a half, a quarter or not at all stayed within
  about 1 % of the mean error, but without them the joints of brick-global with 3
  kinked (5.6 → 7.3, 10 seeds) and pattern with 3 got worse (4.1 → 5.0). Not measured:
  leaving them out from patch size 7 on as a speed-up (3 instead of 9 channels).
- Drawing each candidate of the random search again, up to 4 or 16 times, while its
  patch has a hole: within the noise (error ±2 %), large holes unchanged, up to 11 %
  slower. The search does not lack candidates.
- An occurrence penalty (Kaspar et al. 2015) in the search of the coherence field,
  λ · (link centers per pixel of the source window − 1): large holes gained (brick-large
  `edges` with 7: 5.10 → 3.42, grass-large error 19.5 → 17.0), but small holes have to
  reuse their few fitting sources (coffee with 7 and 15: 18 → 23.5), and the fills got
  blurrier (detail −2 to −6 %). On the coarse levels alone as well.
- Other forms of the peel (`Peel`, patch size 3, 10 seeds, 30 for the patterns and
  bricks). Peeling only the coarsest level with a hole left the pole broken (error 26
  to 30): its sources in front of the forest exist one level below. Peeling only that
  level, or choosing the path before the peel and peeling the chosen one, kept the
  fills of tripod and pole, but the bricks got worse (brick-global +10 to +20 %), and
  peeling only the level below also tilted every selection of pattern to the plain
  path (4.37 → 5.70). Onion-peel order without the isophotes: the sides filled the
  middle of the hole before the leg grew more than 3 rows. A full search for every
  pixel: 10 to 50 times slower, and the pole was worse (17.5 against 11.7), as the
  matches of the peeled neighbors continue a line coherently. Without the seeded links
  the search lost what the peel laid out (brick 9.8 → 11.8). Windows of 11x11 or two
  windows of 7x7 and 11x11: the line is a minor term in them (pattern-wide 46). Partly
  unknown windows: chelsea repeated the border of the hole (variety 0.45). An absolute
  tolerance for the near ties lost the low-contrast leg, the guide as a weighted term
  of the distance kept the pole broken, and a regularized or additive priority gained
  little (pole 21 → 16). The guide's link with a tolerance of 0.3 instead of 2:
  pattern-wide +44 %; of 1: brick-global and pattern-wide +8 to +9 %; of 3: meadow
  8.2 → 9.3. Without the tone shift: camera 3.30 → 3.89. With patch sizes 7 and 15
  the searches of the finer levels wipe the peeled lines out again (15: tripod and
  pole unchanged; 7: pole 32 → 20, but forest-frame, meadow and brick-global worse).
  While two fills of the coarse levels were compared, the peel tipped the near tie of
  their distances towards patch_size: pattern +16 % with 30 seeds.
  Against the pieces of the wrong phase that the peel of the level below the coarsest
  hole level put into the rings of pattern-wide (4 of 30 seeds after #188): no peel
  there (error +9.6 % over the cases, pole 33), its guide searched with the larger
  patches (5.87), a tolerance of 4 or 6 everywhere (meadow +17 %, pole 22.6 with 6), of
  3 or 4 on that level alone (with 4: coffee +5.7 %, tripod `edges` 1.55 → 1.97), the
  best fit within 1 or 2 pixels of the guide's link instead of the link (its phase
  drifts: pattern-wide 14.0 and 11.0), and the coherence of the guide's links without
  the variance (the tripod leg broke where the background around it is copied
  coherently).
- Not tried: starting the holes of the coarsest level smoothly instead of black. As the
  downsampling shrinks the holes, only coffee-corner with 15 keeps one there (51 of 216
  pixels); outpainting of a third of the image does with 15, of half with 7, and a hole
  over 75 % of the image with 15.
- More, with measurements, in the issues closed as not planned: #115 (start radius of
  the new-hole search, merging random fields), #120 (several cores), #124 (confidence
  weighting for large holes) and #157 (stripes in brick-large: rotation, locality).
