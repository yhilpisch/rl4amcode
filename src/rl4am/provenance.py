"""Persist the effective configuration and fitted inference inputs."""

from dataclasses import asdict
import hashlib
from pathlib import Path
import numpy as np

from rl4am.config import AppConfig
from rl4am.env import StateNormalization
from rl4am.slices import SliceSet, slice_manifest_payload
from rl4am.training.common import resolve_normalization


def inference_contract(config: AppConfig, slices: SliceSet) -> dict:
    """Record enough information to reconstruct the observation transform."""
    effective = asdict(config)
    effective.pop("path")
    source = Path(config.data.source)
    digest = (
        hashlib.sha256(source.read_bytes()).hexdigest()
        if source.is_file() else None
    )
    normalization = resolve_normalization(slices, config)
    stats = None
    if isinstance(normalization, StateNormalization):
        stats = {
            "return_mean": normalization.return_mean,
            "return_std": normalization.return_std,
            "feature_means": normalization.feature_means.tolist(),
            "feature_stds": normalization.feature_stds.tolist(),
        }
    return {
        "schema_version": 1,
        "metric_convention": "initial_capital_v2",
        "effective_config": effective,
        "data_sha256": digest,
        "configuration_sha256": _config_digest(effective),
        "normalization": stats,
        "normalization_enabled": stats is not None,
        "sampling": slice_manifest_payload(slices),
    }


def _config_digest(config: dict) -> str:
    import json

    payload = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalization_from_contract(payload: dict):
    """Restore the exact fitted transform without reading evaluation data."""
    stats = payload.get("normalization")
    if stats is None:
        return False
    return StateNormalization(
        return_mean=stats["return_mean"], return_std=stats["return_std"],
        feature_means=np.asarray(stats["feature_means"], dtype=np.float32),
        feature_stds=np.asarray(stats["feature_stds"], dtype=np.float32),
    )
