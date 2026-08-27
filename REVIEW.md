# rl4amcode Review

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
