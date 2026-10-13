#!/usr/bin/env python3

from __future__ import annotations

import argparse
import bisect
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
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "world_bloom_shape_optimization_validation.json"
)
JP_WL2_EVENTS = (163, 167, 170, 171, 176, 179)
CN_WL1_EVENTS = (112, 118, 124, 130, 137, 140)


def load_training_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("train_baseline_module", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_baseline_module"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_cached_rows(path: Path) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({
                "region": str(row["region"]),
                "event_id": int(row["event_id"]),
                "series_type": str(row["series_type"]),
                "character_id": int(row["character_id"]),
                "rank": int(row["rank"]),
                "timestamp": int(row["timestamp"]),
                "score": int(row["score"]),
            })
    return rows


def install_cached_iterator(module, rows: list[dict[str, int | str]]) -> None:
    rows_by_region: dict[str, list[dict[str, int | str]]] = defaultdict(list)
    for row in rows:
        rows_by_region[str(row["region"])].append(row)

    def iter_cached_rows(_data_root, region, selected_ranks, _legacy_event_files):
        selected = set(selected_ranks)
        for row in rows_by_region.get(region, []):
            if int(row["rank"]) in selected:
                yield dict(row)

    module.iter_all_rank_timeseries_rows = iter_cached_rows


def build_finals(rows: list[dict[str, int | str]]) -> dict[tuple[str, int, int, int], tuple[int, int]]:
    finals: dict[tuple[str, int, int, int], tuple[int, int]] = {}
    for row in rows:
        key = (
            str(row["region"]),
            int(row["event_id"]),
            int(row["character_id"]),
            int(row["rank"]),
        )
        timestamp = int(row["timestamp"])
        score = int(row["score"])
        if key not in finals or timestamp > finals[key][1]:
            finals[key] = (score, timestamp)
    return finals


def metrics(module, rows: list[dict[str, object]], predictions: np.ndarray) -> dict[str, float]:
    return module.score_metrics(
        np.asarray([float(row["final_score"]) for row in rows], dtype=np.float64),
        np.asarray(predictions, dtype=np.float64),
    )


def grouped_metrics(module, rows: list[dict[str, object]], predictions: np.ndarray, key_name: str):
    grouped: dict[object, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for row, prediction in zip(rows, predictions, strict=True):
        preds, trues = grouped[row[key_name]]
        preds.append(float(prediction))
        trues.append(float(row["final_score"]))
    return {
        str(key): module.score_metrics(
            np.asarray(trues, dtype=np.float64),
            np.asarray(preds, dtype=np.float64),
        )
        | {"samples": len(trues)}
        for key, (preds, trues) in sorted(grouped.items(), key=lambda item: str(item[0]))
    }


def progress_band(progress: float) -> str:
    if progress < 0.30:
        return "early"
    if progress < 0.65:
        return "mid"
    if progress < 0.85:
        return "late"
    return "final"


def calibration_keys(row: dict[str, object]):
    region = str(row["region"])
    rank = int(row["rank"])
    progress = f"{float(row['progress']):.2f}"
    band = progress_band(float(row["progress"]))
    return (
        ("region_rank_progress", region, rank, progress),
        ("region_rank_band", region, rank, band),
        ("region_rank", region, rank),
        ("rank_progress", rank, progress),
        ("rank_band", rank, band),
        ("rank", rank),
        ("global",),
    )


def fit_log_residual_calibrator(
    rows: list[dict[str, object]],
    base_key: str,
) -> dict[tuple[object, ...], dict[str, float]]:
    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        actual = max(float(row["final_score"]), 1.0)
        predicted = max(float(row[base_key]), 1.0)
        residual = math.log(actual / predicted)
        for key in calibration_keys(row):
            grouped[key].append(residual)
    return {
        key: {
            "count": float(len(values)),
            "median_log_ratio": float(np.median(np.asarray(values, dtype=np.float64))),
        }
        for key, values in grouped.items()
    }


def apply_log_residual_calibration(
    rows: list[dict[str, object]],
    base_predictions: np.ndarray,
    calibrator: dict[tuple[object, ...], dict[str, float]],
    *,
    min_count: int,
    shrinkage: float,
) -> np.ndarray:
    output = np.asarray(base_predictions, dtype=np.float64).copy()
    for index, row in enumerate(rows):
        for key in calibration_keys(row):
            item = calibrator.get(key)
            if item is None:
                continue
            count = int(item["count"])
            if key[0] != "global" and count < min_count:
                continue
            shrink = count / (count + shrinkage)
            output[index] *= math.exp(float(item["median_log_ratio"]) * shrink)
            break
        output[index] = max(output[index], float(row["current_score"]))
    return output


def log_blend(left: np.ndarray, right: np.ndarray, left_weight: float) -> np.ndarray:
    weight = min(max(float(left_weight), 0.0), 1.0)
    return np.exp(
        np.log(np.maximum(np.asarray(left, dtype=np.float64), 1.0)) * weight
        + np.log(np.maximum(np.asarray(right, dtype=np.float64), 1.0)) * (1.0 - weight)
    )


def static_t100_pressure_raw(module, row: dict[str, object]) -> float:
    if int(row["rank"]) != 100:
        return 0.0
    progress = module.clamp(float(row["progress"]), 0.0, 1.0)
    late_strength = module.clamp((progress - 0.55) / 0.45, 0.0, 1.0)
    final_strength = module.clamp((progress - 0.82) / 0.18, 0.0, 1.0)
    weekend_strength = module.rank_boundary_weekend_flag(row)
    return (
        0.010
        + 0.035 * late_strength
        + 0.080 * final_strength
        + 0.020 * late_strength * weekend_strength
        + 0.045 * final_strength * weekend_strength
    )


def apply_static_t100_pressure(
    module,
    rows: list[dict[str, object]],
    predictions: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    adjusted = np.asarray(predictions, dtype=np.float64).copy()
    for index, row in enumerate(rows):
        raw = static_t100_pressure_raw(module, row)
        if raw <= 0.0:
            continue
        adjusted[index] = max(float(row["current_score"]), adjusted[index] * min(1.35, 1.0 + scale * raw))
    return adjusted


def best_static_pressure_scale(module, rows: list[dict[str, object]], predictions: np.ndarray) -> float:
    candidates = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
    target_rows = [row for row in rows if int(row["rank"]) == 100]
    if not target_rows:
        return 0.0
    best_scale = 0.0
    best_score = float("inf")
    for scale in candidates:
        adjusted = apply_static_t100_pressure(module, rows, predictions, scale=scale)
        score = metrics(module, rows, adjusted)["mape"]
        if score < best_score:
            best_score = float(score)
            best_scale = scale
    return best_scale


def load_jp_exact_boundary_pressure(module) -> dict[tuple[int, int, int], object]:
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "sekai_best_jp_wl2"
        / "jp_wl2_chapter_rank_timeseries.csv.gz"
    )
    if not path.exists():
        return {}
    snapshots: dict[tuple[int, int, int], list[tuple[int, frozenset[str]]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rank = int(row["rank"])
            if rank != 100:
                continue
            user_id = str(row.get("user_id") or "")
            if not user_id:
                continue
            timestamp = module.parse_iso_timestamp_seconds(str(row["timestamp"]))
            snapshots[(int(row["event_id"]), int(row["character_id"]), rank)].append(
                (timestamp, frozenset({user_id}))
            )
    return {
        key: module.boundary_pressure_series_from_snapshots(value)
        for key, value in snapshots.items()
    }


def pressure_scalar_for_row(
    row: dict[str, object],
    jp_pressure: dict[tuple[int, int, int], object],
) -> float:
    if str(row["region"]) != "jp" or int(row["rank"]) != 100:
        return 0.0
    series = jp_pressure.get((int(row["event_id"]), int(row["character_id"]), int(row["rank"])))
    if series is None:
        return 0.0
    index = bisect.bisect_right(series.timestamps, int(row["current_timestamp"])) - 1
    if index < 0:
        return 0.0
    feature = series.features[index]
    unique_6h_ratio = float(feature[4])
    unique_24h_ratio = float(feature[5])
    turnover = float(feature[6])
    return max(0.0, unique_6h_ratio - 1.0) * 0.45 + max(0.0, unique_24h_ratio - 1.0) * 0.12 + turnover


def fit_pressure_normalizer(
    rows: list[dict[str, object]],
    jp_pressure: dict[tuple[int, int, int], object],
) -> tuple[float, float]:
    values = np.asarray(
        [
            pressure_scalar_for_row(row, jp_pressure)
            for row in rows
            if str(row["region"]) == "jp" and int(row["rank"]) == 100
        ],
        dtype=np.float64,
    )
    if values.size == 0:
        return 0.0, 1.0
    median = float(np.median(values))
    q75 = float(np.percentile(values, 75))
    q25 = float(np.percentile(values, 25))
    scale = max(q75 - q25, 1.0)
    return median, scale


def turnover_progress_weight(progress: float, mode: str) -> float:
    progress = min(max(progress, 0.0), 1.0)
    if mode == "all":
        return 1.0
    if mode == "mid_late":
        return min(max((progress - 0.35) / 0.65, 0.0), 1.0)
    if mode == "late":
        return min(max((progress - 0.55) / 0.45, 0.0), 1.0)
    if mode == "final":
        return min(max((progress - 0.75) / 0.25, 0.0), 1.0)
    raise ValueError(mode)


def apply_turnover_pressure(
    rows: list[dict[str, object]],
    predictions: np.ndarray,
    jp_pressure: dict[tuple[int, int, int], object],
    *,
    alpha: float,
    mode: str,
    median: float,
    scale: float,
) -> np.ndarray:
    adjusted = np.asarray(predictions, dtype=np.float64).copy()
    for index, row in enumerate(rows):
        pressure = pressure_scalar_for_row(row, jp_pressure)
        if pressure <= 0.0:
            continue
        normalized = min(max((pressure - median) / scale, 0.0), 3.0)
        if normalized <= 0.0:
            continue
        weight = turnover_progress_weight(float(row["progress"]), mode)
        factor = math.exp(alpha * normalized * weight)
        adjusted[index] = max(float(row["current_score"]), adjusted[index] * min(factor, 1.30))
    return adjusted


def best_turnover_pressure_config(
    module,
    rows: list[dict[str, object]],
    predictions: np.ndarray,
    jp_pressure: dict[tuple[int, int, int], object],
) -> dict[str, float | str]:
    if not jp_pressure or not any(str(row["region"]) == "jp" and int(row["rank"]) == 100 for row in rows):
        return {"alpha": 0.0, "mode": "late", "median": 0.0, "scale": 1.0}
    median, scale = fit_pressure_normalizer(rows, jp_pressure)
    best = {"alpha": 0.0, "mode": "late", "median": median, "scale": scale}
    best_score = float("inf")
    for mode in ("all", "mid_late", "late", "final"):
        for alpha in (0.0, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.14, 0.18):
            adjusted = apply_turnover_pressure(
                rows,
                predictions,
                jp_pressure,
                alpha=alpha,
                mode=mode,
                median=median,
                scale=scale,
            )
            score = metrics(module, rows, adjusted)["mape"]
            if score < best_score:
                best_score = float(score)
                best = {"alpha": alpha, "mode": mode, "median": median, "scale": scale}
    return best


def scenario_records(
    module,
    *,
    data_root: Path,
    rows: list[dict[str, int | str]],
    event_meta,
    finals,
    event_feature_contexts,
    target_region: str,
    target_events: tuple[int, ...],
    selected_ranks: set[int],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for holdout_event in target_events:
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
        _ = (train_x, train_y, val_x, val_y, rows)
        for row in val_rows:
            if str(row["region"]) != target_region or int(row["event_id"]) != holdout_event:
                continue
            item = dict(row)
            item["holdout_event"] = holdout_event
            item["_train_rows"] = train_rows
            records.append(item)
    return records


def evaluate_holdout_models(
    module,
    records: list[dict[str, object]],
    jp_pressure,
) -> dict[str, object]:
    by_holdout: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in records:
        by_holdout[int(row["holdout_event"])].append(row)

    model_predictions: dict[str, list[float]] = defaultdict(list)
    tuned_configs: dict[str, list[dict[str, object]]] = defaultdict(list)
    ordered_rows: list[dict[str, object]] = []

    for holdout_event, val_rows in sorted(by_holdout.items()):
        ordered_rows.extend(val_rows)
        train_rows = list(val_rows[0]["_train_rows"]) if val_rows else []
        for row in val_rows:
            row.pop("_train_rows", None)

        base = np.asarray([float(row["jp_shape_projection"]) for row in val_rows], dtype=np.float64)
        model_predictions["jp_shape"].extend(base.tolist())

        calibrator = fit_log_residual_calibrator(train_rows, "jp_shape_projection")
        calibrated = apply_log_residual_calibration(
            val_rows,
            base,
            calibrator,
            min_count=8,
            shrinkage=0.0,
        )
        model_predictions["shape_residual_calibrated"].extend(calibrated.tolist())

        hybrid = np.asarray([float(row["hybrid_projection"]) for row in val_rows], dtype=np.float64)
        calibrated_hybrid = log_blend(calibrated, hybrid, 0.9)
        calibrated_hybrid = np.maximum(
            calibrated_hybrid,
            np.asarray([float(row["current_score"]) for row in val_rows], dtype=np.float64),
        )
        model_predictions["calibrated_shape_hybrid_0_9"].extend(calibrated_hybrid.tolist())

        train_base = np.asarray([float(row["jp_shape_projection"]) for row in train_rows], dtype=np.float64)
        pressure_scale = best_static_pressure_scale(module, train_rows, train_base)
        static_pressure = apply_static_t100_pressure(module, val_rows, base, scale=pressure_scale)
        model_predictions["t100_static_pressure"].extend(static_pressure.tolist())
        tuned_configs["t100_static_pressure"].append({
            "holdout_event": holdout_event,
            "scale": pressure_scale,
        })

        combined_static = apply_static_t100_pressure(module, val_rows, calibrated, scale=pressure_scale)
        model_predictions["calibrated_plus_t100_static"].extend(combined_static.tolist())

        turnover_config = best_turnover_pressure_config(module, train_rows, train_base, jp_pressure)
        turnover = apply_turnover_pressure(
            val_rows,
            base,
            jp_pressure,
            alpha=float(turnover_config["alpha"]),
            mode=str(turnover_config["mode"]),
            median=float(turnover_config["median"]),
            scale=float(turnover_config["scale"]),
        )
        model_predictions["t100_turnover_pressure"].extend(turnover.tolist())
        tuned_configs["t100_turnover_pressure"].append({
            "holdout_event": holdout_event,
            **turnover_config,
        })

        combined_turnover = apply_turnover_pressure(
            val_rows,
            calibrated,
            jp_pressure,
            alpha=float(turnover_config["alpha"]),
            mode=str(turnover_config["mode"]),
            median=float(turnover_config["median"]),
            scale=float(turnover_config["scale"]),
        )
        model_predictions["calibrated_plus_t100_turnover"].extend(combined_turnover.tolist())

    rows_without_train = [
        {key: value for key, value in row.items() if key != "_train_rows"}
        for row in ordered_rows
    ]
    summary = {}
    for name, predictions in model_predictions.items():
        arr = np.asarray(predictions, dtype=np.float64)
        summary[name] = {
            "overall": metrics(module, rows_without_train, arr),
            "by_rank": grouped_metrics(module, rows_without_train, arr, "rank"),
            "by_event": grouped_metrics(module, rows_without_train, arr, "holdout_event"),
        }
    return {
        "samples": len(rows_without_train),
        "models": summary,
        "tuned_configs": tuned_configs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate shape-first World Bloom forecast optimizations.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/deseer/PycharmProjects/Haruki-Tracker-Exports"),
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path(__file__).resolve().parent / "train_baseline.py",
    )
    args = parser.parse_args()

    module = load_training_module(args.train_script.resolve())
    rows = load_cached_rows(args.cache)
    selected_ranks = {100, 1000}
    rows = [row for row in rows if int(row["rank"]) in selected_ranks]
    install_cached_iterator(module, rows)
    finals = build_finals(rows)
    data_root = args.data_root.resolve()
    finished_event_ids = module.load_finished_event_ids()
    event_meta = module.load_event_meta(data_root, finished_event_ids)
    event_feature_contexts = module.load_event_feature_contexts()
    jp_pressure = load_jp_exact_boundary_pressure(module)

    cn_records = scenario_records(
        module,
        data_root=data_root,
        rows=rows,
        event_meta=event_meta,
        finals=finals,
        event_feature_contexts=event_feature_contexts,
        target_region="cn",
        target_events=CN_WL1_EVENTS,
        selected_ranks=selected_ranks,
    )
    jp_records = scenario_records(
        module,
        data_root=data_root,
        rows=rows,
        event_meta=event_meta,
        finals=finals,
        event_feature_contexts=event_feature_contexts,
        target_region="jp",
        target_events=JP_WL2_EVENTS,
        selected_ranks=selected_ranks,
    )

    output = {
        "definition": {
            "base": "jp_shape_projection",
            "validation": "leave-one-event; every tuning step fits only on the corresponding training fold",
            "selected_ranks": sorted(selected_ranks),
            "cn_events": list(CN_WL1_EVENTS),
            "jp_events": list(JP_WL2_EVENTS),
            "jp_turnover_pressure_source": "sekai.best exact rank-100 user id changes",
        },
        "cn_wl1": evaluate_holdout_models(module, cn_records, jp_pressure),
        "jp_wl2": evaluate_holdout_models(module, jp_records, jp_pressure),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
