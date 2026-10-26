# Corrections completed — 26 October 2026

The companion suite passes all 84 tests.
It requires no market CSV or network access.
Both notebook sources match the lecture repository.
Companion notebook outputs remain cleared.

Metrics include initial capital and the first return.
Comparisons use matching dates.
Disjoint sampling and chronological helpers have regression coverage.
Unsafe evaluation normalisation is rejected.
New checkpoints retain fitted inference statistics.
The data-cache tool verifies the frozen snapshot checksum.
The metadata records known provenance limits.

The historical profiles remain fixed reference profiles.
They are not presented as verified sweep winners.
The companion repository does not bundle historical fitted runs.
Exact lecture-result reconstruction requires those saved inputs.

The implementation checkpoint is `60c086b`.
The earlier review remains below for context.

---

# Companion code review — 26 October 2026

Lecture: **27 October 2026**. This assessment supersedes the historical
test results retained below. The full local review and proposed changes are
in [`../rl4am/REVIEW.md`](../rl4am/REVIEW.md), in the sibling lecture repository;
that report is not included in a standalone clone of this repository.

## Corrections and current validation

- Both notebooks now fall back to existing reference configurations.
- DQN inspection uses the main evaluation seed; A2C narrative correctly
  identifies mode-policy evaluation. Companion outputs remain cleared.
- Both notebooks passed structural validation and code-cell syntax checks.
  Active Python files parsed successfully.
- Package sources/tests match the lecture repository. Configuration files
  differ only by existing trailing whitespace in `configs/default.yml`.
- Pytest: **70 passed and three failed**. `test_load_default_config` and
  `test_load_market_data_falls_back_to_remote` assume the omitted local CSV
  exists. `test_load_market_data_for_symbol` also needs that file or network
  access; its network fallback was unavailable in this environment.

Substantial fixes remain proposed: historical train/test overlap disclosure,
baseline/agent horizon alignment, first-period metric handling, offline data
and persisted normalisation, optional walk-forward/sampling paths, and
mathematical documentation. Publication-builder tools and canonical result
artefacts are not included in this distribution. No training, algorithms,
saved numerical results, or publication outputs were changed.

---

## Historical review — superseded by the assessment above

## Scope

Companion code repository for the RL4AM article. Contains the `rl4am`
Python package, test suite, and two workbench notebooks.

## Status

All Python sources compile cleanly. 71/73 tests pass; 2 failures are due
to a gitignored data file (`data/eod_data.csv`) that is intentionally not
bundled — the tests assume local data availability and fall back to a
remote URL when it is absent.

## Open Items

| ID | Description |
| --- | --- |
| D1 | `data/eod_data.csv` is gitignored; two tests (`test_config.py::test_load_default_config`, `test_data.py::test_load_market_data_falls_back_to_remote`) fail on a fresh clone without the data file. One passing test (`test_data.py::test_load_market_data_for_symbol`) is silently network-dependent. |
| S1 | 29 Python lines exceed 85 characters (specials bar), concentrated in `cli.py`, `notebooks/common.py`, `training/a2c.py`, `slices.py`. |
| S2 | 46 type-annotated default parameters use `: type = value` instead of the house-mandated `: type=value` across 13 files. |

## Resolved Items

| ID | Resolution |
| --- | --- |
| R1 | Non-breaking hyphen (U+2011) in `README.md` "decision‑making" replaced with ASCII hyphen. |
| R2 | Trailing whitespace in `configs/default.yml` `symbol: EURUSD` removed. |
| R3 | Two notebook `CONFIG_PATH` ternary lines (137 chars) wrapped to ≤80 chars. |
