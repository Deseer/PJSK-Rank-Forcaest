#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_CACHE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "cache"
    / "world_bloom_rank_timeseries_cn_jp_100_1000.csv.gz"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "wl_character_prior_gain_validation.json"
JP_WL2_EVENTS = frozenset({163, 167, 170, 171, 176, 179})
CN_WL1_EVENTS = frozenset({112, 118, 124, 130, 137, 140})


def load_training_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("train_baseline_module", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_baseline_module"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_cached_rows(path: Path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({
                "region": row["region"],
                "event_id": int(row["event_id"]),
                "series_type": row["series_type"],
                "character_id": int(row["character_id"]),
                "rank": int(row["rank"]),
                "timestamp": int(row["timestamp"]),
                "score": int(row["score"]),
            })
    return rows


def install_cached_iterator(module, rows):
    rows_by_region = defaultdict(list)
    for row in rows:
        rows_by_region[row["region"]].append(row)

    def iter_cached_rows(_data_root, region, selected_ranks, _legacy_event_files):
        selected = set(selected_ranks)
        for row in rows_by_region.get(region, []):
            if row["rank"] in selected:
                yield dict(row)

    module.iter_all_rank_timeseries_rows = iter_cached_rows


def build_finals(rows):
    finals = {}
    for row in rows:
        key = (row["region"], row["event_id"], row["character_id"], row["rank"])
        if key not in finals or row["timestamp"] > finals[key][1]:
            finals[key] = (row["score"], row["timestamp"])
    return finals


def build_coefficients(
    finals,
    *,
    source_region: str,
    source_events: set[int],
    rank_weights: dict[int, float],
) -> dict[int, float]:
    grouped = defaultdict(list)
    for (region, event_id, character_id, rank), (score, _timestamp) in finals.items():
        if region != source_region or event_id not in source_events or rank not in rank_weights:
            continue
        grouped[(event_id, rank)].append((character_id, max(int(score), 1)))

    residuals = defaultdict(list)
    for (_event_id, rank), items in grouped.items():
        event_log_median = float(np.median(np.asarray([math.log(score) for _cid, score in items], dtype=np.float64)))
        for character_id, score in items:
            residuals[(character_id, rank)].append(math.log(score) - event_log_median)

    coefficients = {}
    character_ids = sorted({character_id for character_id, _rank in residuals})
    for character_id in character_ids:
        total = 0.0
        weight_total = 0.0
        for rank, weight in rank_weights.items():
            values = residuals.get((character_id, rank), [])
            if not values:
                continue
            total += float(np.mean(np.asarray(values, dtype=np.float64))) * weight
            weight_total += weight
        if weight_total > 0.0:
            coefficients[character_id] = total / weight_total
    return coefficients


def progress_weight(progress: float, mode: str) -> float:
    progress = min(max(float(progress), 0.0), 1.0)
    if mode == "constant":
        return 1.0
    if mode == "linear":
        return 1.0 - progress
    if mode == "sqrt":
        return math.sqrt(max(1.0 - progress, 0.0))
    if mode == "early75":
        return min(max((0.75 - progress) / 0.70, 0.0), 1.0)
    raise ValueError(f"unknown progress weight mode: {mode}")


def score_metrics(module, records, prediction_key: str):
    true_scores = np.asarray([record["final_score"] for record in records], dtype=np.float64)
    predictions = np.asarray([record[prediction_key] for record in records], dtype=np.float64)
    return module.score_metrics(true_scores, predictions)


def apply_character_prior(records, coefficients: dict[int, float], *, alpha: float, mode: str, centered: bool, base_key: str):
    event_means = {}
    if centered:
        grouped = defaultdict(list)
        for record in records:
            coeff = coefficients.get(int(record["character_id"]), 0.0)
            grouped[(record["region"], record["event_id"])].append(coeff)
        event_means = {
            key: float(np.mean(np.asarray(values, dtype=np.float64)))
            for key, values in grouped.items()
            if values
        }

    predictions = []
    for record in records:
        coeff = coefficients.get(int(record["character_id"]), 0.0)
        if centered:
            coeff -= event_means.get((record["region"], record["event_id"]), 0.0)
        weight = progress_weight(float(record["progress"]), mode)
        factor = math.exp(float(alpha) * coeff * weight)
        predictions.append(max(float(record["current_score"]), float(record[base_key]) * factor))
    return predictions


def metrics_for_adjustment(module, records, coefficients, *, alpha: float, mode: str, centered: bool, base_key: str):
    adjusted_records = [dict(record) for record in records]
    predictions = apply_character_prior(
        adjusted_records,
        coefficients,
        alpha=alpha,
        mode=mode,
        centered=centered,
        base_key=base_key,
    )
    for record, prediction in zip(adjusted_records, predictions, strict=True):
        record["prediction"] = prediction
    return module.score_metrics(
        np.asarray([record["final_score"] for record in adjusted_records], dtype=np.float64),
        np.asarray(predictions, dtype=np.float64),
    )


def by_rank_metrics(module, records, predictions):
    grouped = defaultdict(list)
    for record, prediction in zip(records, predictions, strict=True):
        grouped[int(record["rank"])].append((record["final_score"], prediction))
    result = {}
    for rank, items in sorted(grouped.items()):
        true_scores = np.asarray([item[0] for item in items], dtype=np.float64)
        pred_scores = np.asarray([item[1] for item in items], dtype=np.float64)
        result[str(rank)] = module.score_metrics(true_scores, pred_scores)
        result[str(rank)]["samples"] = len(items)
    return result


def build_eval_records(
    module,
    *,
    data_root: Path,
    event_meta,
    finals,
    event_feature_contexts,
    selected_ranks: set[int],
    target_region: str,
    target_events: set[int],
) -> list[dict[str, float | int | str]]:
    records = []
    for holdout_event in sorted(target_events):
        validation_events = {region: set() for region in module.REGIONS}
        validation_events[target_region] = {holdout_event}
        jp_shape_priors = module.build_jp_shape_priors(
            data_root=data_root,
            event_meta=event_meta,
            finals=finals,
            validation_events=validation_events,
            selected_ranks=selected_ranks,
            legacy_event_files={region: [] for region in module.REGIONS},
            model_series_types=module.WL_MODEL_SERIES_TYPES,
        )
        train_x, train_y, train_rows, val_x, val_y, val_rows = module.build_dataset(
            data_root=data_root,
            event_meta=event_meta,
            finals=finals,
            validation_events=validation_events,
            selected_ranks=selected_ranks,
            event_feature_contexts=event_feature_contexts,
            character_heat_priors={},
            jp_shape_priors=jp_shape_priors,
            boundary_pressure_contexts={},
            legacy_event_files={region: [] for region in module.REGIONS},
            model_series_types=module.WL_MODEL_SERIES_TYPES,
        )
        _ = (train_x, train_y, train_rows, val_x, val_y)
        for row in val_rows:
            if row["region"] != target_region or int(row["event_id"]) != holdout_event:
                continue
            records.append(dict(row))
    return records


def summarize_grid(module, records, coefficients, *, base_key: str):
    base_metrics = score_metrics(module, records, base_key)
    configs = []
    for centered in (False, True):
        for mode in ("constant", "linear", "sqrt", "early75"):
            for alpha in (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
                metrics = metrics_for_adjustment(
                    module,
                    records,
                    coefficients,
                    alpha=alpha,
                    mode=mode,
                    centered=centered,
                    base_key=base_key,
                )
                configs.append({
                    "alpha": alpha,
                    "mode": mode,
                    "centered": centered,
                    "metrics": metrics,
                })
    best = min(configs, key=lambda item: item["metrics"]["mape"])
    fixed = {}
    for alpha in (0.5, 1.0):
        key = f"alpha_{alpha:g}_linear"
        fixed[key] = {
            "alpha": alpha,
            "mode": "linear",
            "centered": False,
            "metrics": metrics_for_adjustment(
                module,
                records,
                coefficients,
                alpha=alpha,
                mode="linear",
                centered=False,
                base_key=base_key,
            ),
        }
    best_predictions = apply_character_prior(
        records,
        coefficients,
        alpha=float(best["alpha"]),
        mode=str(best["mode"]),
        centered=bool(best["centered"]),
        base_key=base_key,
    )
    fixed_predictions = apply_character_prior(
        records,
        coefficients,
        alpha=1.0,
        mode="linear",
        centered=False,
        base_key=base_key,
    )
    return {
        "samples": len(records),
        "base": base_metrics,
        "base_by_rank": by_rank_metrics(module, records, [record[base_key] for record in records]),
        "fixed": fixed,
        "fixed_alpha_1_linear_by_rank": by_rank_metrics(module, records, fixed_predictions),
        "oracle_best": best,
        "oracle_best_by_rank": by_rank_metrics(module, records, best_predictions),
    }


def prediction_delta_for_wl3(prediction_path: Path, coefficients: dict[int, float], *, alpha: float, mode: str, base_key: str):
    if not prediction_path.exists():
        return []
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    rows = payload.get("predictions", [])
    result = []
    for row in rows:
        if int(row.get("rank", 0)) not in (100, 1000):
            continue
        character_id = int(row["character_id"])
        coeff = coefficients.get(character_id, 0.0)
        weight = progress_weight(float(row["progress"]), mode)
        base = float(row[base_key])
        adjusted = max(float(row["current_score"]), base * math.exp(alpha * coeff * weight))
        result.append({
            "character_id": character_id,
            "rank": int(row["rank"]),
            "progress": float(row["progress"]),
            "current_score": int(row["current_score"]),
            "base_prediction": base,
            "adjusted_prediction": adjusted,
            "delta": adjusted - base,
            "coefficient": coeff,
            "progress_weight": weight,
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate whether WL character coefficients improve later WL forecasts.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--data-root", type=Path, default=Path("/Users/deseer/PycharmProjects/Haruki-Tracker-Exports"))
    parser.add_argument("--train-script", type=Path, default=Path(__file__).resolve().parent / "train_baseline.py")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--wl3-prediction", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "jp_202_prediction_world_bloom.json")
    args = parser.parse_args()

    module = load_training_module(args.train_script.resolve())
    rows = load_cached_rows(args.cache)
    install_cached_iterator(module, rows)
    data_root = args.data_root.resolve()
    selected_ranks = {100, 1000}
    finished_event_ids = module.load_finished_event_ids()
    event_meta = module.load_event_meta(data_root, finished_event_ids)
    event_feature_contexts = module.load_event_feature_contexts()
    finals = build_finals(rows)
    rank_weights = {100: 0.6, 1000: 0.4}

    jp_wl2_coefficients = build_coefficients(
        finals,
        source_region="jp",
        source_events=set(JP_WL2_EVENTS),
        rank_weights=rank_weights,
    )
    cn_wl1_coefficients = build_coefficients(
        finals,
        source_region="cn",
        source_events=set(CN_WL1_EVENTS),
        rank_weights=rank_weights,
    )
    cn_records = build_eval_records(
        module,
        data_root=data_root,
        event_meta=event_meta,
        finals=finals,
        event_feature_contexts=event_feature_contexts,
        selected_ranks=selected_ranks,
        target_region="cn",
        target_events=set(CN_WL1_EVENTS),
    )
    jp_records = build_eval_records(
        module,
        data_root=data_root,
        event_meta=event_meta,
        finals=finals,
        event_feature_contexts=event_feature_contexts,
        selected_ranks=selected_ranks,
        target_region="jp",
        target_events=set(JP_WL2_EVENTS),
    )
    base_keys = ("jp_shape_projection", "hybrid_projection", "pace_6h_projection", "linear_projection")
    cn_summaries = {
        base_key: summarize_grid(
            module,
            cn_records,
            jp_wl2_coefficients,
            base_key=base_key,
        )
        for base_key in base_keys
    }
    jp_summaries = {
        base_key: summarize_grid(
            module,
            jp_records,
            cn_wl1_coefficients,
            base_key=base_key,
        )
        for base_key in base_keys
    }
    output = {
        "definition": {
            "base_predictions": list(base_keys),
            "adjustment": "base * exp(alpha * character_coefficient * progress_weight)",
            "fixed_configs_are_not_tuned_on_cn": True,
            "rank_weights": rank_weights,
        },
        "cn_wl1_test_using_jp_wl2_coefficients": cn_summaries,
        "jp_wl2_test_using_cn_wl1_coefficients": jp_summaries,
        "jp_wl3_current_projection_using_jp_wl2_coefficients": prediction_delta_for_wl3(
            args.wl3_prediction,
            jp_wl2_coefficients,
            alpha=1.0,
            mode="linear",
            base_key="jp_shape_projection",
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
