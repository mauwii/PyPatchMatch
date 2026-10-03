---
paths:
  - ".github/**"
  - ".pre-commit-config.yaml"
  - "sonar-project.properties"
---

# CI

- `ci.yml`: `lint` runs `pre-commit run --all-files`; `test` runs `uv run pytest` on
  Ubuntu, macOS and Windows with Python 3.11, 3.14 and 3.15, after building (or
  restoring) the OpenCV of the build script. It syncs only the `test` group: the `dev`
  group pulls in pyyaml through pre-commit, which had no wheel for 3.15 yet, and
  `--no-build` refuses to build it.
- pre-commit runs, besides the file checks, ruff and uv-lock: clang-format
  (`.clang-format`), shellcheck, markdownlint-cli2 (`.markdownlint.yaml`, 88 columns),
  typos (fixes in place), actionlint (checks `run:` scripts with shellcheck only if it
  is on `PATH`) and zizmor (offline, default persona), and mypy as a local hook
  (`uv run --locked --no-build --only-group dev mypy`, like the `lint` job, so the
  library is not built; numpy and pillow are in the `dev` group for their types).
  MegaLinter was considered and rejected: it duplicates pre-commit in a multi-GB Docker
  image and bundles the linter versions instead of pinning each hook for Dependabot.
- Dependabot waits 7 days after a release (`cooldown`, required by zizmor).
- `test` sets `SKBUILD_CMAKE_DEFINE=CMAKE_COMPILE_WARNING_AS_ERROR=ON`, so the warning
  flags of `CMakeLists.txt` (`-Wall -Wextra -Wpedantic -Wshadow`, `/W4` on MSVC) fail the
  build. Only there: builds from the sdist and the wheels must not break when a newer
  compiler adds a warning. The OpenCV headers are system includes and do not count.
- `sanitizers` builds with `PATCHMATCH_SANITIZE=ON` on Ubuntu and runs the tests with
  the preloaded ASan runtime (skill `sanitizers`). It reuses the checkout and OpenCV
  steps of `test` through YAML anchors, so change them there.
  Scheduled (weekly) and manual runs of `ci.yml` pass `--hypothesis-profile=deep`
  (`tests/conftest.py`: 5000 examples per property test, about 90 s locally without
  sanitizers) and allow the job 90 minutes; a failure prints a `@reproduce_failure`
  blob. The property tests found real bugs, so they get the most search time.
- `coverage` builds with `PATCHMATCH_COVERAGE=ON` on Ubuntu, runs the tests and fails
  below 95 % of the C++ lines or 90 % of the branches (gcovr, skill `cpp-coverage`).
  It reuses the same anchors. It also writes `coverage.xml` (Python) and
  `coverage-cpp.xml` (`gcovr --sonarqube`) and exports `build/compile_commands.json`,
  and then runs the SonarCloud scan (`sonar-project.properties`, `SONAR_TOKEN`, skipped
  without the secret, e.g. for Dependabot and forks). Automatic analysis in SonarCloud
  is disabled; it cannot import coverage and would conflict with the CI scan. `scripts/`
  has no tests and is excluded from the coverage on new code (80 % in "Sonar way").
  `sonar.qualitygate.wait=true` makes the scan fail on a red quality gate, so the gate
  is part of `ci-ok` and auto-merge waits for it; the skipped scan of Dependabot and
  fork PRs blocks nothing, unlike a required "SonarCloud Code Analysis" check, which
  would never appear there.
- The `uv run` steps pass `--locked` and `--no-build` (SonarCloud S8544, S8541).
  `--no-build` only forbids building dependencies from source; uv still builds the
  project itself. The OpenCV step runs with `--only-group test`: the project cannot be
  built before OpenCV is installed, `--no-project` would lock nothing, and the `dev`
  group may lack wheels for the newest Python of the matrix.
- Every setup-uv step sets `version: latest-known`: the newest uv whose checksum is
  bundled with the pinned setup-uv. Without it, CI installs each uv release minutes
  after publication; this way uv only moves when Dependabot updates the action, with
  its cooldown. That uv can be a few releases older than the one of the `uv-lock` hook.
  `prune-cache: true` keeps only wheels built from source in the uv cache; downloaded
  wheels restore no faster than they download.
- `system-opencv` installs `libopencv-dev` on `ubuntu-24.04` (OpenCV 4.6) instead of
  running the build script and runs the tests, because builds from the sdist link
  whatever OpenCV the system has. The runner is pinned so that the OpenCV version only
  changes on purpose. The Ubuntu mirror is sometimes very slow (180 MB at under
  300 kB/s), so the job caches the downloaded `.deb` files, keyed on the hash of
  `apt-get install --print-uris`, with `restore-keys` for the files that did not change.
  apt checks them against the signed index and dpkg installs them as usual. The job
  downloads first (`--download-only`) and caches only the files of the current list.
- `wheels.yml`: sdist, then cibuildwheel on five runners, then `sdist-fallback`;
  `publish` attests the build provenance of all files (`actions/attest`) and uploads
  them to PyPI with trusted publishing (environment `pypi`, deployable only from tags
  `v*`) only for a published GitHub release of `mauwii/PyPatchMatch`. It downloads the
  artifacts named `dist-*`, so other artifacts of the run never reach PyPI. The wheel
  jobs cache the OpenCV prefix, keyed on `build_opencv.py` and `pyproject.toml` from the
  sdist (there is no checkout); on Linux a volume makes `/tmp/opencv` of the container
  visible to the cache step. Runs on a tag, and therefore releases, skip the cache and
  build OpenCV from scratch.
  The build and test dependencies of the distributions are not locked, so the workflow
  sets `UV_EXCLUDE_NEWER: 7 days` (the cooldown of `dependabot.yml`): `uv build` and
  cibuildwheel's `build-frontend = "build[uv]"` resolve only releases older than a
  week. On Linux `environment-pass` hands the variable into the container; on macOS
  and Windows cibuildwheel uses the uv of a setup-uv step.
- `fill-quality.yml` runs for pull requests that change `CMakeLists.txt`,
  `patchmatch/csrc` or the evaluation script: it builds the library of the base commit,
  runs `scripts/evaluate_inpainting.py --baseline` against it (patch size 3, 3 seeds),
  writes the table to the job summary (`--markdown`) and uploads `evaluation.html` as the
  artifact `fill-quality`. It is a report, not a gate: the measures vary over the seeds,
  so it is not in the `needs` of `ci-ok`. It repeats the OpenCV steps of `ci.yml` with
  the same cache key, because anchors do not reach across files.
- `codeql.yml` analyzes actions, C/C++ and Python.
- `cache-cleanup.yml` deletes the caches of a pull request when it is closed: only that
  pull request can read them, and at the 10 GB limit GitHub evicts the least recently
  used caches, often those of `main`. For pull requests from forks the token is
  read-only; their caches expire after 7 days without access.
- `ci-ok` and `wheels-ok` (re-actors/alls-green) are the required status checks; add new
  jobs to their `needs`. The ruleset of `main` also has a `code_scanning` rule: CodeQL
  alerts of severity error or security severity high and above block the merge (CodeQL
  uploads its results for Dependabot PRs too, so they are not stuck).
- Actions are pinned to full commit SHAs with a `# vX.Y.Z` comment, and checkouts use
  `persist-credentials: false`. The pre-commit hooks are pinned the same way
  (`rev: <sha>  # frozen: vX.Y.Z`; `pre-commit autoupdate --freeze` by hand), because a
  moved tag would otherwise reach CI and fresh clones without review. Except typos,
  which is pinned by tag: its repository tags other crates too, and Dependabot matched
  the frozen comment against one of them (#92). Dependabot updates actions and
  pre-commit weekly in groups, SHA and comment together. The repository is a GitHub
  fork, on which Dependabot version updates are off by default; they are enabled in
  Insights → Dependency graph → Dependabot.
- `uv.lock` only pins development tools and is updated by hand (`uv lock --upgrade`).
  Dependabot cannot: its `uv lock --upgrade-package` builds the project, which fails
  without OpenCV, because scikit-build-core hides the `prepare_metadata_for_build_*`
  hooks when an override uses `if.failed` (the fallback of the sdist).
