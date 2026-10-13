#!/usr/bin/env python3
"""Mid/late hierarchical live-heat correction layer.

Offline validation: `analysis/offline_intrinsic_live/experiment_live_layer.py`
(strictly causal time-blocked replay, event-weighted MAPE, paired bootstrap by
event). Variant L1 (current observation only) gives

    progress 15-40%  -0.61pp  CI [-1.17, -0.08]
    progress 45-75%  -0.54pp  CI [-0.77, -0.32]
    progress 80-95%  -0.22pp  CI [-0.35, -0.10]
    progress  5-10%   0.00pp  (gated off)

Idea: the shape prior implies a final score; the model implies another one.
Their disagreement, pooled across the ranks of the same event/character inside
a rank band, is a real signal about the event's heat. It is only trusted once
the event is far enough along (gate) and in proportion to the evidence.

This module is deliberately standalone: it takes the prediction rows produced by
`predict_live_event.py` and mutates them in place. It never reads production
state and is disabled unless the caller enables it.
"""
from __future__ import annotations

import collections
import math

GATE_PROGRESS = 0.15
MAX_LOG_CORRECTION = 0.80
POOL_SHRINKAGE = 3.0
PROGRESS_BUCKET = 0.05
RANK_BANDS = ((0, 100), (100, 1000), (1000, 10 ** 12))

SHAPE_KEYS = ("jp_shape_projection", "shape_first_prediction")


def rank_band(rank: int) -> int:
    for index, (low, high) in enumerate(RANK_BANDS):
        if low <= rank < high:
            return index
    return len(RANK_BANDS) - 1


def progress_bucket(progress: float) -> float:
    bucket = math.floor(max(progress, 0.0) / PROGRESS_BUCKET) * PROGRESS_BUCKET
    return round(min(max(bucket, GATE_PROGRESS), 0.95), 2)


def shape_reference(row: dict) -> float | None:
    """The shape prior's implied final score, if the row carries one."""
    for key in SHAPE_KEYS:
        value = row.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value
    return None


def apply_live_heat(rows: list[dict], *, enabled: bool = True,
                    gate: float = GATE_PROGRESS,
                    max_correction: float = MAX_LOG_CORRECTION,
                    shrinkage: float = POOL_SHRINKAGE) -> dict:
    """Apply the hierarchical live-heat correction in place; return a summary."""
    summary = {
        "enabled": bool(enabled),
        "policy": "live_heat_v1",
        "gate_progress": gate,
        "rows": len(rows),
        "candidates": 0,
        "applied": 0,
        "mean_abs_log_correction": 0.0,
    }
    if not enabled or not rows:
        return summary

    # per-line disagreement between the shape-implied final and the model final
    per_line: list[float | None] = []
    weights: list[float] = []
    keys: list[tuple | None] = []
    for row in rows:
        per_line.append(None)
        weights.append(0.0)
        keys.append(None)
        if row.get("event_finished_prediction"):
            continue
        try:
            progress = float(row.get("progress") or 0.0)
            current = float(row.get("current_score") or 0.0)
            prediction = float(row.get("model_prediction") or 0.0)
            rank = int(row.get("rank") or 0)
        except (TypeError, ValueError):
            continue
        shape = shape_reference(row)
        if shape is None or current <= 0.0 or prediction <= 0.0 or progress < gate:
            continue
        disagreement = math.log(shape) - math.log(prediction)
        disagreement = max(min(disagreement, max_correction), -max_correction)
        index = len(per_line) - 1
        per_line[index] = disagreement
        try:
            weight = float(row.get("history_points") or 1.0)
        except (TypeError, ValueError):
            weight = 1.0
        weights[index] = max(weight, 1.0)
        keys[index] = (int(row.get("character_id") or 0), rank_band(rank),
                       progress_bucket(progress))
        summary["candidates"] += 1

    total: dict[object, float] = collections.defaultdict(float)
    mass: dict[object, float] = collections.defaultdict(float)
    square: dict[object, float] = collections.defaultdict(float)
    for value, weight, key in zip(per_line, weights, keys):
        if key is None or value is None:
            continue
        total[key] += weight * value
        mass[key] += weight
        square[key] += weight * weight

    pooled: dict[object, float] = {}
    for key, weight in mass.items():
        if weight <= 0.0:
            continue
        effective = weight * weight / max(square[key], 1e-12)
        evidence = effective / (effective + shrinkage)
        pooled[key] = evidence * max(min(total[key] / weight, max_correction), -max_correction)

    corrections = []
    for row, value, key in zip(rows, per_line, keys):
        if value is None or key is None or key not in pooled:
            continue
        correction = pooled[key]
        current = float(row["current_score"])
        prediction = float(row["model_prediction"])
        remaining = max(prediction - current, 0.0)
        adjusted = current + remaining * math.exp(correction)
        if not math.isfinite(adjusted):
            continue
        row["live_heat_policy"] = "live_heat_v1"
        row["live_heat_key"] = list(key)
        row["live_heat_per_line_log"] = value
        row["live_heat_pooled_log"] = correction
        row["live_heat_input_prediction"] = prediction
        row["live_heat_applied"] = adjusted != prediction
        row["model_prediction"] = adjusted
        row["pressure_adjusted_prediction"] = adjusted
        corrections.append(correction)
        if adjusted != prediction:
            summary["applied"] += 1
    if corrections:
        summary["mean_abs_log_correction"] = sum(abs(value) for value in corrections) / len(corrections)
    return summary
