---
name: fill-quality
description: Compare the fills of the inpainting with those of the library of main using scripts/evaluate_inpainting.py. Use for every change to the algorithm in patchmatch/csrc before judging whether it improves the fills.
---

# Fill quality

Fill quality (`scripts/evaluate_inpainting.py`, its docstring explains the columns):

```sh
uv run --no-sync python scripts/evaluate_inpainting.py --baseline <library of main>
uv run --no-sync python scripts/evaluate_inpainting.py --patch-size 15 --seeds 1 \
    --cases forest,brick
```

- It cuts holes into `forest.bmp` and the CC0 images of `examples/images` (sources in
  `SOURCES.md`) and adds `forest_pruned.bmp` and every `*_pruned.*` of the git-ignored
  `examples/images/local/` as object removals without a ground truth. That folder is
  for your own images; never commit them.
- For `--baseline`, build the library of `main` outside the checkout:

  ```sh
  mkdir -p /tmp/main-lib
  git archive main CMakeLists.txt patchmatch/csrc | tar -x -C /tmp/main-lib
  cmake -S /tmp/main-lib -B /tmp/main-lib/build -DCMAKE_BUILD_TYPE=Release
  cmake --build /tmp/main-lib/build   # libpatchmatch.so, .dylib or patchmatch.dll
  ```

- Judge changes on all cases and look at the fills in `examples/images/evaluation.html`
  (git-ignored, it embeds the images of `examples/images/local/` too): fixes for one
  image regressed others before.
