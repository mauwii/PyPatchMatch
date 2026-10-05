---
paths:
  - "patchmatch/csrc/**"
  - "CMakeLists.txt"
  - "examples/cpp_example.cpp"
---

# Native code

## Threading and determinism

ctypes releases the GIL during native calls, so inpaintings run truly concurrently
(`test_concurrent_inpaint_is_deterministic`). Therefore:

- The seed and the verbose flag are `std::atomic` globals, read when a call starts.
- The random generator is a function-local `thread_local std::mt19937`, seeded at the
  start of every `Inpainting::run`, so each call is reproducible from the seed on any
  thread.
- `random_int` uses `engine() % n` instead of `std::uniform_int_distribution` to keep
  the sequence identical on all platforms and standard libraries. Keep it that way.
- Lookup tables are function-local statics (thread-safe initialization). Do not add
  mutable shared state.

## Algorithm

- `patch_size` is a radius: patches span `(2 * patch_size + 1)` pixels in each
  direction. The pyramid halves the image until one side is <= `patch_size`.
- Two nearest-neighbor fields link the patches: source to target (completeness) and
  target to source (coherence). They start random at the coarsest level and are
  upscaled from the previous level afterwards, keeping the offset within the coarse
  pixel so that neighbors still point to neighbors. Levels 0 and 1 leave the
  completeness field out, which the paper does on all levels (section 4): there its
  votes changed nothing, and it cost a quarter of the time. On the coarser levels they
  keep the known pixels near their input; without them the error of the fills rose by
  up to 69 % (trees with `patch_size` 15).
- Each level runs `1 + 2 * level` EM iterations with `min(7, 1 + level)` NNF passes
  (propagation and random search), an expectation step (votes of the fields, weighted
  through the `distance2similarity` table) and a maximization step, which sets each
  pixel to the weighted mean of its votes. From the second iteration on, the links of
  patches with holes are measured again on the new image before the passes; with the
  distances of the previous image, worse candidates replaced better links, and the
  votes were weighted with outdated distances. On levels 0 and 1 the known pixels keep
  their input values instead; without that the votes change the pixels around the
  holes, and a seam appears when `run()` restores them. On coarser levels the known
  pixels next to the holes are averages of partially masked kernels, and keeping them
  made fills copy smooth regions into textured ones. The last iteration of a level
  votes directly into the upsampled image of the next level, which is less blurry than
  upsampling the result. The last iteration at level 0 keeps only the heaviest vote of
  each pixel instead of the mean, which blurs. Many votes there are near ties (fill
  copied from the source, distance close to 0), and the first one cast won the whole
  area of its patch, which filled large patch sizes with blocks of
  `2 * patch_size + 1` pixels. The weight
  is therefore multiplied by `1 - 1e-5 * (di² + dj²)`, so near ties go to the patch
  centered closest to the pixel. Stronger preferences (`1e-4`, a Gaussian, `1 / (1 +
  r²)`) override real differences: on the color gradient of the unit tests a patch in
  the hole copied colors from far away (`test_inpaint_patch_sizes[7]`).
- The fills match the texture around a hole better than its colors, and the border
  stood out as an outline once the known pixels were restored. At the end of `run()`,
  `blend_hole_borders` shifts the hole pixels up to 4 pixels from the border by the
  local color step across it (3x3 means on either side, averaged over the border
  pixels within 4 pixels), fading out inwards. A width that grows with `patch_size`
  measured worse. Blending the mean and the heaviest vote did not help: with the
  known pixels kept, the mean of `patch_size` 15 is flat (detail 0.08 for chelsea).
- The coarse levels mark a pixel as a hole only if its whole downsampling kernel is
  one, so the holes shrink quickly and are filled smoothly from their border; the finer
  levels add the texture. Marking every pixel that touches a hole instead was tried and
  made the fills worse: it leaves the coarsest levels without valid source patches,
  because the similarity table gives zero weight to patches with more than about 10 %
  masked pixels or pixels outside the image.
- A hole, and later each thin part of it, therefore has a level on which it appears for
  the first time. The patches around it arrive there with the link of a patch without
  holes, the one to itself, which now points into the hole. `_initialize_field_from`
  runs a random search around such a patch before the first pass. Otherwise the first
  propagation replaces the link with the one of a neighbor, the random search goes on
  around that link, and with a radius of half the smaller image side it does not find
  the way back to the surroundings of the patch, e.g. the stem of `forest_pruned.bmp`
  was filled with forest instead of meadow
  (`test_inpaint_thin_hole_takes_the_colors_beside_it`). Tried and rejected: searching
  before propagating in every pass, which no longer refines a propagated link, the way
  a structure is followed along a hole; a search radius of the larger image side, as
  in the paper, which blurs the fills, also since the window rather than each candidate
  is clamped to the image; and a radius of 1 on level 0, as the paper suggests for the
  finest levels (section 4.3), which saved at most 10 % and made the seam worse with
  `patch_size` 7 and 15 (on levels 0 and 1 with every patch size).
- Distances are ints in `[0, PatchDistanceMetric::kDistanceScale]` (65535): an SSD over
  the colors and the x/y gradients, where masked pixels and pixels outside the image
  count as maximal. The distance indexes `distance2similarity`, so every metric has to
  clamp its result to that range. The gradients replicate the outermost pixels. When
  the outermost line counted as maximal too, every patch over it had at least
  1/(2p+1) maximal pixels, above the 10 % at which the table is zero up to
  `patch_size` 4, so that line of a hole at the image border never got a vote and
  stayed flat (`test_inpaint_textures_the_image_border`).
- Mask semantics: `mask` non-zero marks the holes (by default all pure white pixels,
  computed in Python by `_default_mask`); `_initialize_pyramid` makes them black on
  level 0 as on the coarser levels, so their colors do not matter
  (`test_colors_under_the_holes_do_not_matter`); `global_mask` marks pixels that are
  neither filled nor used as a patch source, and whose colors must not influence the
  fill: the downsampling kernel skips them, and gradients next to them are neutral
  (`test_colors_under_global_mask_do_not_matter`). At the end of `run()` every pixel
  that is not a hole, including the globally masked ones, is copied back from the input,
  so only the holes change. The tests assert this invariant (`assert_filled`,
  `assert_plausible_fill`).

## OpenCV usage and library size

The release wheels link a static, core-only OpenCV (`scripts/build_opencv.py`:
`BUILD_LIST=core`, every optional backend off). For changes to the C++ code:

- The library links only `opencv_core`; `imgcodecs` is only used by the example.
- Use plain pixel loops instead of `cv::Mat` matrix expressions such as `a == b` or
  `m1 | m2`: they pull OpenCV's expression module, and with it `FileStorage`, PCA and
  `SparseMat` code, into the static library (about 40 % larger).
- On Windows the static OpenCV uses the static MSVC runtime (`BUILD_WITH_STATIC_CRT`),
  so the library does too (`MSVC_RUNTIME_LIBRARY` in `CMakeLists.txt`). Its stderr is
  then not captured by pytest's `capfd`, so tests of native output run in a subprocess
  (`test_set_verbose_prints_progress`).
