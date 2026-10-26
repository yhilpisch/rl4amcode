"""Load audited saved workbench results without repeating training."""

from dataclasses import dataclass, replace
import json

import numpy as np
import pandas as pd

from rl4am.baselines import BaselineResult, aggregate_baseline_results
from rl4am.evaluation import EvaluationResult, aggregate_evaluations
from rl4am.metrics import performance_summary
from rl4am.notebooks.common import BaselineRunArtifacts, WorkbenchContext
from rl4am.reporting.slices import (
    build_slice_metric_frame, build_slice_summary_table,
)
from rl4am.results import load_strategy_result
from rl4am.provenance import inference_contract


@dataclass(frozen=True)
class SavedTrainingInfo:
    double_dqn: bool | None


@dataclass(frozen=True)
class SavedWorkbenchArtifacts:
    training: SavedTrainingInfo
    evaluation: EvaluationResult
    evaluation_slices: list[EvaluationResult]
    slice_summary: pd.DataFrame
    selected_slice_metrics: pd.DataFrame
    history_payload: list[dict]
    history_frame: pd.DataFrame


def evaluation_from_saved(path) -> EvaluationResult:
    result = load_strategy_result(path)
    gross = result.returns["gross_return"].to_numpy(dtype=float)
    net = result.returns["net_return"].to_numpy(dtype=float)
    return EvaluationResult(
        name=result.name, start_index=0,
        weights=result.allocation["risky_weight"].to_numpy(dtype=float),
        gross_returns=gross, net_returns=net,
        gross_equity=np.cumprod(1 + gross),
        net_equity=np.cumprod(1 + net),
        turnover=result.turnover["turnover"].to_numpy(dtype=float),
        # Rewards are not stored by the historical result contract.
        rewards=np.full(len(net), np.nan),
        gross_metrics=performance_summary(gross),
        net_metrics=performance_summary(net),
    )


def load_saved_policy(context: WorkbenchContext) -> SavedWorkbenchArtifacts:
    """Load period returns and historical training diagnostics."""
    _validate_contract(context)
    evaluations = _saved_slices(context.policy_dir, context)
    evaluation = aggregate_evaluations(evaluations, evaluations[0].name)
    history = json.loads(
        (context.policy_dir / "training_history.json").read_text()
    )
    return SavedWorkbenchArtifacts(
        training=SavedTrainingInfo(
            context.config.experiments.get("dqn", {}).get(
                "optimisation", {}
            ).get("double_dqn", True)
            if context.policy_dir.name == "dqn" else None
        ),
        evaluation=evaluation, evaluation_slices=evaluations,
        slice_summary=build_slice_summary_table(
            {evaluation.name: evaluations},
        ),
        selected_slice_metrics=pd.DataFrame([
            evaluations[context.selected_test_slice].net_metrics,
        ]),
        history_payload=history, history_frame=pd.DataFrame(history),
    )


def load_saved_baselines(context: WorkbenchContext) -> BaselineRunArtifacts:
    _validate_contract(context)
    names = sorted(p.name for p in context.baseline_dir.iterdir()
                   if (p / "manifest.json").is_file())
    saved_by_name = {
        name: _saved_slices(context.baseline_dir / name, context)
        for name in names
    }
    per_slice = []
    weights = {}
    for index, s in enumerate(context.eval_slices):
        results = {}
        for name in names:
            saved = saved_by_name[name][index]
            weight = float(saved.weights[0])
            weights[name] = weight
            values = vars(saved).copy()
            for key in ("start_index", "rewards"):
                values.pop(key)
            results[name] = BaselineResult(weight=weight, **values)
        per_slice.append(results)
    aggregate = {
        name: aggregate_baseline_results([s[name] for s in per_slice], name)
        for name in names
    }
    selected = context.eval_slices[context.selected_test_slice]
    return BaselineRunArtifacts(
        aggregate_results=aggregate, per_slice_results=per_slice,
        baseline_setup=pd.DataFrame([
            {"strategy": name, "weight": weight,
             "calibration": "saved mean training-slice weights"}
            for name, weight in weights.items()
        ]),
        baseline_slice_frames={
            name: build_slice_metric_frame([s[name] for s in per_slice])
            for name in names
        },
        baseline_slice_summary=build_slice_summary_table({
            name: [s[name] for s in per_slice] for name in names
        }),
        selected_slice_manifest={
            "slice_id": selected.slice_id, "split": selected.split,
            "start_date": selected.start_date, "end_date": selected.end_date,
        },
        selected_slice_metrics=pd.DataFrame({
            name: per_slice[context.selected_test_slice][name].net_metrics
            for name in names
        }).T,
        boundary_note=None,
    )


def _validate_contract(context: WorkbenchContext) -> None:
    manifest = json.loads((context.policy_dir / "manifest.json").read_text())
    saved = manifest["metadata"]["inference_contract"]
    current = inference_contract(context.config, context.slices)
    for key in ("data_sha256", "configuration_sha256"):
        if saved[key] != current[key]:
            raise ValueError(f"Saved workbench contract differs: {key}")


def _saved_slices(path, context: WorkbenchContext) -> list[EvaluationResult]:
    """Support both complete and minimal per-slice result contracts."""
    whole = evaluation_from_saved(path)
    count = len(context.eval_slices)
    rows, remainder = divmod(len(whole.net_returns), count)
    if remainder:
        raise ValueError("Saved rows do not divide into evaluation slices")
    results = []
    for index, window in enumerate(context.eval_slices):
        part = slice(index * rows, (index + 1) * rows)
        gross = whole.gross_returns[part]
        net = whole.net_returns[part]
        results.append(replace(
            whole, start_index=len(window.returns) - rows,
            weights=whole.weights[part], turnover=whole.turnover[part],
            gross_returns=gross, net_returns=net,
            gross_equity=np.cumprod(1 + gross),
            net_equity=np.cumprod(1 + net), rewards=np.full(rows, np.nan),
            gross_metrics=performance_summary(gross),
            net_metrics=performance_summary(net),
        ))
    return results
