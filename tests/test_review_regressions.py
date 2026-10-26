from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl4am.baselines import simulate_constant_mix
from rl4am.config import SamplingConfig, build_config
from rl4am.data import MarketData
from rl4am.env import SingleAssetAllocationEnv
from rl4am.metrics import performance_summary
from rl4am.notebooks.walk_forward_workbench import (
    prepare_walk_forward_context, run_walk_forward,
)
from rl4am.reporting.comparison import compare_strategy_paths
from rl4am.reporting.tables import _table_to_tex
from rl4am.results import save_strategy_result
from rl4am.slices import sample_market_slices


def test_growth_includes_first_return_and_actual_period_count():
    summary = performance_summary([0.10, -0.10], periods_per_year=2)
    assert summary["annualised_return"] == pytest.approx(-0.01)
    assert summary["terminal_equity"] == pytest.approx(0.99)
    single = performance_summary([0.10], periods_per_year=1)
    assert single["annualised_return"] == pytest.approx(0.10)


def test_drawdown_includes_initial_capital():
    assert performance_summary([-0.10, 0.10])["max_drawdown"] == (
        pytest.approx(-0.10)
    )


def test_common_span_metrics_and_terminal_equity_share_first_return(tmp_path):
    dates = pd.date_range("2026-01-01", periods=3)
    for name, returns, span in [
        ("long", [0.50, -0.10, 0.10], dates),
        ("short", [-0.10, 0.10], dates[1:]),
    ]:
        result = simulate_constant_mix(np.array(returns), 1.0, name=name)
        save_strategy_result(result, tmp_path / name, "baseline", span)
    comparison = compare_strategy_paths(tmp_path / "long", tmp_path / "short")
    assert comparison.rows == 2
    assert comparison.left_terminal_equity == pytest.approx(0.99)
    assert comparison.left_terminal_equity == comparison.right_terminal_equity
    assert comparison.left_metrics == comparison.right_metrics


def _market(returns):
    dates = pd.date_range("2026-01-01", periods=len(returns))
    series = pd.Series(returns, index=dates)
    return MarketData("synthetic", "test", (1 + series).cumprod(), series)


def test_nonoverlap_applies_across_training_and_test():
    config = SamplingConfig(
        train_slices=3, test_slices=2, trading_days_per_slice=4,
        overlap=False, seed=9,
    )
    data = _market(np.zeros(23))
    slices = sample_market_slices(data, config)
    intervals = sorted((s.start, s.stop) for s in slices.train + slices.test)
    assert all(left[1] <= right[0]
               for left, right in zip(intervals, intervals[1:]))
    again = sample_market_slices(data, config)
    assert [s.start for s in slices.test] == [s.start for s in again.test]
    with pytest.raises(ValueError, match="Not enough rows"):
        sample_market_slices(_market(np.zeros(19)), config)


def test_log_data_is_converted_before_portfolio_accounting():
    data = _market(np.array([0.01, -0.02, 0.03, 0.04]))
    log_data = replace(data, returns=np.log1p(data.returns), return_type="log")
    config = SamplingConfig(
        train_slices=1, test_slices=0, trading_days_per_slice=4, seed=1,
    )
    actual = sample_market_slices(log_data, config).train[0].returns
    np.testing.assert_allclose(actual, data.returns)


def test_normalized_environment_cannot_fit_on_future_evaluation_returns():
    with pytest.raises(ValueError, match="training-fitted"):
        SingleAssetAllocationEnv(
            np.array([0.01, 0.02, 100.0]), 2,
            state_features={"normalize": True},
        )
    env = SingleAssetAllocationEnv(
        np.array([0.01, 0.02, 100.0]), 2,
        state_features={"normalize": True}, normalization=False,
    )
    np.testing.assert_allclose(env.reset()[:2], [0.01, 0.02])


def test_generic_tex_table_escapes_strategy_and_metric_identifiers():
    table = pd.DataFrame({"metric": ["ann_return"], "a2c_mode": [0.10]})
    tex = _table_to_tex(table)
    assert r"a2c\_mode" in tex and r"ann\_return" in tex


def test_walk_forward_runs_with_aligned_windows(tmp_path: Path):
    # Synthetic prices exercise the complete window path without live data.
    prices = 100 * np.cumprod(1 + np.sin(np.arange(40)) / 100)
    source = tmp_path / "synthetic.csv"
    pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=40), "asset": prices,
    }).to_csv(source, index=False)
    raw = {
        "project": {"seed": 1, "device": "cpu"},
        "data": {"source": str(source), "symbol": "asset"},
        "environment": {"window": 2, "state_features": {"normalize": True}},
        "sampling": {
            "mode": "walk_forward", "train_days": 8, "test_days": 6,
            "step_days": 6, "max_windows": 2, "normalization": "none",
        },
        "experiments": {
            "a2c": {"model": {"hidden_units": 4},
                    "optimisation": {"updates": 1}},
            "dqn": {
                "model": {"hidden_units": 4},
                "action_grid": {"bins": 3},
                "optimisation": {
                    "updates": 1, "batch_size": 1, "min_replay_size": 1,
                },
            },
        },
    }
    config = build_config(raw, tmp_path / "config.yml")
    context = prepare_walk_forward_context(
        root=tmp_path, config_path=config.path, config=config, config_notes=(),
    )
    result = run_walk_forward(context, show_progress=False)
    assert len(result.test_dates) == 8
    assert result.test_dates.is_unique
    assert len(result.a2c_result.net_returns) == len(result.test_dates)
    assert len(result.dqn_result.net_returns) == len(result.test_dates)
    assert all(len(b.net_returns) == len(result.test_dates)
               for b in result.baseline_results.values())


def test_minimal_saved_run_retains_each_slice_first_return(tmp_path):
    from types import SimpleNamespace
    from rl4am.notebooks.saved_workbench import _saved_slices

    result = simulate_constant_mix(np.array([.10, -.10, -.20, .20]), 1.0)
    save_strategy_result(result, tmp_path / "policy", "policy")
    windows = sample_market_slices(
        _market(np.zeros(8)),
        SamplingConfig(train_slices=1, test_slices=2,
                       trading_days_per_slice=4, seed=2),
    ).test
    loaded = _saved_slices(
        tmp_path / "policy", SimpleNamespace(eval_slices=windows),
    )
    assert [x.start_index for x in loaded] == [2, 2]
    assert loaded[0].net_metrics["terminal_equity"] == pytest.approx(.99)
    assert loaded[1].net_metrics["terminal_equity"] == pytest.approx(.96)
    assert loaded[1].net_metrics["max_drawdown"] == pytest.approx(-.20)


def test_saved_normalization_round_trips_without_evaluation_data():
    from rl4am.provenance import normalization_from_contract

    stats = {
        "normalization": {"return_mean": .01, "return_std": .02,
                          "feature_means": [], "feature_stds": []},
    }
    transform = normalization_from_contract(stats)
    env = SingleAssetAllocationEnv(
        np.array([.01, .03, 100.0]), 2,
        state_features={"normalize": True}, normalization=transform,
    )
    np.testing.assert_allclose(env.reset()[:2], [0.0, 1.0], atol=1e-7)


def test_snapshot_mismatch_never_overwrites_existing_data(tmp_path):
    from tools.prepare_data import prepare

    destination = tmp_path / "data.csv"
    original = b"existing data must survive"
    destination.write_bytes(original)
    with pytest.raises(ValueError, match="checksum differs"):
        prepare("not-read", destination)
    assert destination.read_bytes() == original
