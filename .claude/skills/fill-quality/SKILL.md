---
name: fill-quality
description: Compare the fills of the inpainting with those of the library of main using scripts/evaluate_inpainting.py. Use for every change to the algorithm in patchmatch/csrc, before judging whether it improves the fills or claiming that a speed-up leaves them byte-identical.
---

# Fill quality

Fill quality (`scripts/evaluate_inpainting.py`, its docstring explains the cases and the
columns):

```sh
uv run --no-sync python scripts/evaluate_inpainting.py --baseline <library of main>
uv run --no-sync python scripts/evaluate_inpainting.py --patch-size 15 --seeds 1 \
    --cases forest,brick
```

- For `--baseline`, build the library of `main` outside the checkout:

  ```sh
  mkdir -p /tmp/main-lib
  git archive main CMakeLists.txt patchmatch/csrc | tar -x -C /tmp/main-lib
  cmake -S /tmp/main-lib -B /tmp/main-lib/build -DCMAKE_BUILD_TYPE=Release
  cmake --build /tmp/main-lib/build   # libpatchmatch.so, .dylib or patchmatch.dll
  ```

- A speed-up has to keep the fills byte-identical; the column `fills` reports it per
  case, and a few seeds suffice.
- Judge a change of the fills on all cases with 10 seeds (`--seeds 10`) and patch sizes
  3, 7 and 15: the mean error of brick varied from 10.2 to 12.5 between sets of 3 seeds,
  and fixes for one image regressed others before. Look at the fills in
  `examples/images/evaluation.html`, which every run overwrites.
- `examples/images/local/` is for your own images, which the script adds as object
  removals; never commit them, nor `evaluation.html`, which embeds them.
- The workflow `Fill quality` compares every PR that changes the native code with its
  base branch (job summary, artifact `fill-quality`); a report, not a gate.
