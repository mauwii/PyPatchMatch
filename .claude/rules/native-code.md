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
- From the coarsest level to the finest, two nearest-neighbor fields are kept:
  source to target (completeness) and target to source (coherence). They start random
  at the coarsest level and are upscaled from the previous level afterwards, keeping the
  offset within the coarse pixel so that neighbors still point to neighbors.
- Each level runs `1 + 2 * level` EM iterations with `min(7, 1 + level)` NNF passes
  (propagation and random search), an expectation step (votes of both fields, weighted
  through the `distance2similarity` table) and a maximization step, which sets each
  pixel to the weighted mean of its votes. On levels 0 and 1 the known pixels keep their
  input values instead; without that the votes change the pixels around the holes, and a
  seam appears when `run()` restores them. On coarser levels the known pixels next to
  the holes are averages of partially masked kernels, and keeping them made fills copy
  smooth regions into textured ones (57 % instead of 17 % of the fill in one case). The
  last iteration of a level votes directly into the upsampled image of the next level,
  which is less blurry than upsampling the result. The last iteration at level 0 keeps
  only the heaviest vote of each pixel instead of the mean, which blurs.
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
  the way back to the surroundings of the patch: the stem of `forest_pruned.bmp` was
  filled with forest instead of meadow in 31 of 200 seeds (36 of 48 with `patch_size`
  7), and the `meadow` case had an error above 12 in 9 of 96 seeds, now at most 9.7
  (`test_inpaint_thin_hole_takes_the_colors_beside_it`). Tried and rejected: searching
  before propagating in every pass, which no longer refines a propagated link, the way
  a structure is followed along a hole (brick error +9 %); and a search radius of the
  larger image side, as in the paper, which blurs the fills (coffee detail 0.30 instead
  of 0.53).
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
  `SparseMat` code, into the static library, which grew by about 40 % before commit
  e50c613 replaced them.
- On Windows the static OpenCV uses the static MSVC runtime (`BUILD_WITH_STATIC_CRT`),
  so the library does too (`MSVC_RUNTIME_LIBRARY` in `CMakeLists.txt`). Its stderr is
  then not captured by pytest's `capfd`, so tests of native output run in a subprocess
  (`test_set_verbose_prints_progress`).
