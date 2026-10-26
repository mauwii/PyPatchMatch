---
paths:
  - ".github/**"
  - ".pre-commit-config.yaml"
  - "sonar-project.properties"
---

# CI

The workflow files explain their steps in comments; this lists what they do not say.

- `ci-ok` and `wheels-ok` (re-actors/alls-green) are the required status checks; add new
  jobs to their `needs`. The ruleset of `main` also has a `code_scanning` rule: CodeQL
  alerts of severity error or security severity high and above block the merge.
- `sanitizers`, `coverage` and `lowest` reuse the checkout and OpenCV steps of `test`
  through YAML anchors, so change them there. `fill-quality.yml` repeats them with the
  same cache key, because anchors do not reach across files.
- SonarCloud: automatic analysis is disabled (it cannot import coverage and would
  conflict with the scan of the `coverage` job). `sonar.qualitygate.wait=true` makes a
  red gate fail `ci-ok`, so auto-merge waits for it; without `SONAR_TOKEN` (Dependabot,
  forks) the scan is skipped and blocks nothing. Sonar way passes code smells of any
  severity below 5 % debt ratio, and a custom gate needs a paid plan, so a step after
  the scan fails on every unresolved issue of the PR or branch; accept an issue in
  SonarCloud when it cannot be fixed. A required check "SonarCloud Code Analysis"
  would never report on Dependabot PRs and block them.
- The `uv run` steps pass `--locked` and `--no-build` (SonarCloud S8544, S8541);
  `--no-build` only forbids building dependencies, uv still builds the project. The
  OpenCV step runs with `--only-group test`: the project cannot be built before OpenCV
  is installed, and the `dev` group may lack wheels for the newest Python.
- Actions are pinned to full commit SHAs with a `# vX.Y.Z` comment. The pre-commit hooks
  are frozen the same way, and Dependabot (ecosystem `pre-commit`) keeps them frozen,
  except typos, which is pinned by tag: Dependabot matched the frozen comment against
  the tags of its other crates (#92). ruff and mypy are local hooks from `uv.lock`, so
  they move with `uv lock --upgrade`, not with Dependabot. MegaLinter was rejected: it
  duplicates pre-commit in a multi-GB image.
- Dependabot version updates are off by default on forks; they are enabled in Insights →
  Dependency graph → Dependabot. `uv.lock` is updated by hand (`uv lock --upgrade`,
  with the same 7-day cooldown through `exclude-newer`): Dependabot's `uv lock` builds
  the project, which fails without OpenCV, because scikit-build-core hides the
  `prepare_metadata_for_build_*` hooks when an override uses `if.failed`.
- The environment `pypi` deploys only from tags `v*`.
