#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_baseline.py"
PREDICT_SCRIPT = REPO_ROOT / "scripts" / "predict_live_event.py"

PROGRESS_BUCKETS = (
    ("00-05", 0.00, 0.05),
    ("05-10", 0.05, 0.10),
    ("10-20", 0.10, 0.20),
    ("20-35", 0.20, 0.35),
    ("35-60", 0.35, 0.60),
    ("60-85", 0.60, 0.85),
    ("85-100", 0.85, 1.01),
)

RANK_BUCKETS = (
    ("top_10_50", 1, 50),
    ("t100", 51, 100),
    ("t500", 101, 500),
    ("t1000", 501, 1000),
    ("t5000", 1001, 5000),
    ("t10000", 5001, 10000),
    ("tail", 10001, 10**9),
)

HEAT_SCALE_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def env_from_config(config: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    required = config.get("required_local_paths", {})
    mapping = {
        "jp_masterdata": "HARUKI_FORECAST_MASTERDATA_JP",
        "en_masterdata": "HARUKI_FORECAST_MASTERDATA_EN",
        "tw_masterdata": "HARUKI_FORECAST_MASTERDATA_TW",
        "kr_masterdata": "HARUKI_FORECAST_MASTERDATA_KR",
        "cn_masterdata": "HARUKI_FORECAST_MASTERDATA_CN",
    }
    for key, env_name in mapping.items():
        value = required.get(key)
        if value:
            env[env_name] = str(Path(value))
    cache_file = required.get("jp_wl2_sekai_best_cache")
    if cache_file:
        env["HARUKI_FORECAST_SEKAI_BEST_JP_WL2_DIR"] = str(Path(cache_file).parent)
    world_bloom_cache_dir = required.get("world_bloom_sekai_best_cache_dir")
    if world_bloom_cache_dir:
        env["HARUKI_FORECAST_SEKAI_BEST_WORLD_BLOOM_DIR"] = str(Path(world_bloom_cache_dir))
    return env


def parse_float(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def parse_int(value: object, default: int | None = None) -> int | None:
    parsed = parse_float(value, None)
    if parsed is None:
        return default
    return int(parsed)


def log_error(prediction: float, final_score: float) -> float:
    if prediction <= 0.0 or final_score <= 0.0:
        return 0.0
    return abs(math.log(prediction / final_score))


def signed_log_error(prediction: float, final_score: float) -> float:
    if prediction <= 0.0 or final_score <= 0.0:
        return 0.0
    return math.log(prediction / final_score)


def log_blend(left: float, right: float, left_weight: float) -> float:
    weight = min(max(float(left_weight), 0.0), 1.0)
    return math.exp(math.log(max(float(left), 1.0)) * weight + math.log(max(float(right), 1.0)) * (1.0 - weight))


def progress_bucket(progress: float) -> str:
    for name, lower, upper in PROGRESS_BUCKETS:
        if lower <= progress < upper:
            return name
    return "other"


def rank_bucket(rank: int) -> str:
    rank = int(rank)
    for name, lower, upper in RANK_BUCKETS:
        if lower <= rank <= upper:
            return name
    return "other"


def iter_history_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) == 3:
                rank, timestamp, score = row
                yield 0, int(rank), int(timestamp), int(score)
            elif len(row) == 4:
                character_id, rank, timestamp, score = row
                yield int(character_id), int(rank), int(timestamp), int(score)


def load_final_scores(history_dir: Path) -> dict[tuple[str, int, str, int, int], int]:
    finals: dict[tuple[str, int, str, int, int], int] = {}
    for path in history_dir.glob("*_history.tsv"):
        stem = path.stem
        if not stem.endswith("_history"):
            continue
        name = stem.removesuffix("_history")
        scope = "chapter"
        if name.endswith("_total"):
            scope = "total"
            name = name.removesuffix("_total")
        parts = name.split("_")
        if len(parts) != 2:
            continue
        region, event_id_text = parts
        event_id = int(event_id_text)
        latest: dict[tuple[int, int], tuple[int, int]] = {}
        for character_id, rank, timestamp, score in iter_history_rows(path):
            key = (character_id, rank)
            old = latest.get(key)
            if old is None or timestamp >= old[0]:
                latest[key] = (timestamp, score)
        for (character_id, rank), (_timestamp, score) in latest.items():
            finals[(region, event_id, scope, character_id, rank)] = score
    return finals


def load_prediction_rows(record_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(record_dir.glob("*/*/*.tsv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if row.get("series_type") != "normal":
                    continue
                if str(row.get("leaderboard_scope") or "total") != "total":
                    continue
                prediction = parse_float(row.get("model_prediction"))
                progress = parse_float(row.get("progress"))
                rank = parse_int(row.get("rank"))
                event_id = parse_int(row.get("event_id"))
                current_score = parse_float(row.get("current_score"))
                if prediction is None or progress is None or rank is None or event_id is None or current_score is None:
                    continue
                if prediction <= 0.0:
                    continue
                rows.append({
                    "archived_at": parse_int(row.get("archived_at"), 0) or 0,
                    "region": str(row.get("region") or ""),
                    "event_id": event_id,
                    "character_id": parse_int(row.get("character_id"), 0) or 0,
                    "rank": rank,
                    "progress": progress,
                    "current_score": current_score,
                    "model_prediction": prediction,
                    "pressure_adjusted_prediction": parse_float(row.get("pressure_adjusted_prediction"), prediction) or prediction,
                    "calibrated_model_prediction": parse_float(row.get("calibrated_model_prediction"), prediction) or prediction,
                    "boundary_pressure_multiplier": parse_float(row.get("boundary_pressure_multiplier"), 1.0) or 1.0,
                    "live_calibration_multiplier": parse_float(row.get("live_calibration_multiplier"), 1.0) or 1.0,
                    "jp_shape_projection": parse_float(row.get("jp_shape_projection"), 0.0) or 0.0,
                    "hybrid_projection": parse_float(row.get("hybrid_projection"), 0.0) or 0.0,
                    "linear_projection": parse_float(row.get("linear_projection"), 0.0) or 0.0,
                    "pace_6h_projection": parse_float(row.get("pace_6h_projection"), 0.0) or 0.0,
                    "event_finished_prediction": str(row.get("event_finished_prediction") or "").lower() in {"true", "1", "yes"},
                })
    return rows


def load_event_meta_from_prediction_rows(module, rows: list[dict[str, Any]]):
    finished: dict[str, set[int]] = {region: set() for region in module.REGIONS}
    for row in rows:
        region = str(row["region"])
        if region in finished:
            finished[region].add(int(row["event_id"]))
    return module.load_event_meta(Path("."), finished)


def build_finals_from_history(
    module,
    event_meta,
    final_scores: dict[tuple[str, int, str, int, int], int],
) -> dict[tuple[str, int, int, int], tuple[int, int]]:
    finals: dict[tuple[str, int, int, int], tuple[int, int]] = {}
    for (region, event_id, scope, character_id, rank), score in final_scores.items():
        if scope != "total":
            continue
        meta = event_meta.get((region, event_id))
        if meta is None or meta.series_type != "normal":
            continue
        finals[(region, event_id, int(character_id), int(rank))] = (int(score), int(meta.end_timestamp))
    return finals


def apply_event_heat_prior(module, row: dict[str, Any], prediction: float, payload: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    prior_prediction, debug = module.event_heat_prior_prediction(
        row,
        payload.get("live_event_heat_priors"),
        payload.get("live_character_heat_priors", {}),
        payload.get("live_event_feature_contexts", {}),
    )
    blend_weight = module.event_heat_prior_blend_weight(float(row["progress"]), int(row["rank"]))
    debug["event_heat_prior_blend_weight"] = blend_weight
    if prior_prediction is None or blend_weight <= 0.0:
        return float(prediction), debug
    adjusted = log_blend(float(prediction), max(float(prior_prediction), float(row["current_score"])), 1.0 - blend_weight)
    return max(float(row["current_score"]), adjusted), debug


def pre_early_stage_prediction(row: dict[str, Any]) -> float:
    prediction = float(row["calibrated_model_prediction"])
    prediction *= max(float(row.get("boundary_pressure_multiplier") or 1.0), 0.0)
    prediction *= max(float(row.get("live_calibration_multiplier") or 1.0), 0.0)
    return max(float(row["current_score"]), prediction)


def zero_heat_prior(heat_prior: dict[str, float]) -> dict[str, float]:
    zeroed = dict(heat_prior)
    for key in (
        "event_heat",
        "unit_heat",
        "main_heat",
        "character_heat",
        "line_heat",
        "unit_popularity_heat",
        "main_popularity_heat",
        "character_popularity_heat",
        "line_popularity_heat",
    ):
        zeroed[key] = 0.0
    return zeroed


def scale_heat_prior(heat_prior: dict[str, float], scale: float) -> dict[str, float]:
    scaled = dict(heat_prior)
    for key in (
        "event_heat",
        "unit_heat",
        "main_heat",
        "character_heat",
        "line_heat",
        "unit_popularity_heat",
        "main_popularity_heat",
        "character_popularity_heat",
        "line_popularity_heat",
    ):
        scaled[key] = float(scaled.get(key, 0.0)) * float(scale)
    return scaled


def early_stage_replay_predictions(
    predict_module,
    train_module,
    row: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    baselines = {
        "linear_projection": float(row.get("linear_projection") or 0.0),
        "pace_6h_projection": float(row.get("pace_6h_projection") or 0.0),
        "jp_shape_projection": float(row.get("jp_shape_projection") or 0.0),
        "hybrid_projection": float(row.get("hybrid_projection") or 0.0),
    }
    pre_stage = pre_early_stage_prediction(row)
    heat_prior = predict_module.early_event_heat_prior_log(
        train_module,
        str(row["region"]),
        int(row["event_id"]),
        int(row["character_id"]),
        payload.get("live_event_feature_contexts", {}),
        payload.get("live_character_heat_priors", {}),
    )
    output: dict[str, Any] = {
        "pre_stage_prediction": pre_stage,
        "early_heat_log": heat_prior.get("event_heat", 0.0),
    }
    for scale in HEAT_SCALE_GRID:
        prediction, debug = predict_module.early_stage_prior_adjustment(
            pre_stage,
            int(float(row["current_score"])),
            int(row["rank"]),
            float(row["progress"]),
            baselines,
            scale_heat_prior(heat_prior, scale),
            "normal",
        )
        suffix = int(round(scale * 100))
        output[f"early_heat_scale_{suffix}_prediction"] = prediction
        output[f"early_heat_scale_{suffix}_multiplier"] = debug.get("early_heat_multiplier", 1.0)
        output[f"early_heat_scale_{suffix}_projection_target"] = debug.get("early_projection_target")
        if scale == 0.0:
            output["early_no_heat_prediction"] = prediction
            output["early_no_heat_projection_target"] = debug.get("early_projection_target")
        if scale == 1.0:
            output["early_heat_prediction"] = prediction
            output["early_heat_multiplier"] = debug.get("early_heat_multiplier", 1.0)
            output["early_projection_strength"] = debug.get("early_projection_strength", 0.0)
            output["early_heat_projection_target"] = debug.get("early_projection_target")
    return output


def shape_envelope_anchor(current_score: float, baselines: dict[str, float]) -> dict[str, float]:
    current = float(current_score)
    shape = max(float(baselines.get("jp_shape_projection") or 0.0), current, 1.0)
    hybrid = max(float(baselines.get("hybrid_projection") or 0.0), current, 1.0)
    linear = max(float(baselines.get("linear_projection") or 0.0), current, 1.0)
    anchor = log_blend(shape, hybrid, 0.55)
    return {
        "shape": shape,
        "hybrid": hybrid,
        "linear": linear,
        "anchor": anchor,
        "early_anchor": log_blend(anchor, linear, 0.82),
        "opening_anchor": log_blend(hybrid, linear, 0.72),
    }


def replay_full_postprocess(
    predict_module,
    early_stage_samples: list[dict[str, Any]],
    variant: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in early_stage_samples:
        anchors = shape_envelope_anchor(
            float(sample["current_score"]),
            {
                "jp_shape_projection": float(sample.get("jp_shape_projection") or 0.0),
                "hybrid_projection": float(sample.get("hybrid_projection") or 0.0),
                "linear_projection": float(sample.get("linear_projection") or 0.0),
            },
        )
        prediction = float(sample[variant])
        rows.append({
            **sample,
            "model_prediction": prediction,
            "pressure_adjusted_prediction": prediction,
            "shape_envelope_anchor": anchors["anchor"],
            "shape_envelope_early_anchor": anchors["early_anchor"],
            "shape_envelope_opening_anchor": anchors["opening_anchor"],
            "shape_envelope_target": None,
            "shape_envelope_strength": 0.0,
            "shape_envelope_mode": "pending",
            "shape_envelope_applied_prediction": None,
            "front_parking_cap": None,
            "front_parking_cap_applied": None,
            "rank_consistency_floor": None,
            "series_type": "normal",
            "event_finished_prediction": bool(sample.get("event_finished_prediction")),
        })
    by_snapshot: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_snapshot[
            (
                str(row["region"]),
                int(row["event_id"]),
                int(row.get("character_id") or 0),
                int(row.get("archived_at") or 0),
            )
        ].append(row)
    for snapshot_rows in by_snapshot.values():
        predict_module.apply_shape_envelope_adjustments(snapshot_rows)
        predict_module.apply_rank_consistency_smoothing(snapshot_rows)
    return rows


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {
            "count": 0,
            "baseline_abs_log": None,
            "event_heat_abs_log": None,
            "delta_abs_log": None,
            "baseline_signed_log": None,
            "event_heat_signed_log": None,
            "improved_share": None,
        }
    baseline_abs = [float(item["baseline_abs_log"]) for item in samples]
    adjusted_abs = [float(item["event_heat_abs_log"]) for item in samples]
    baseline_signed = [float(item["baseline_signed_log"]) for item in samples]
    adjusted_signed = [float(item["event_heat_signed_log"]) for item in samples]
    improved = sum(1 for base, adjusted in zip(baseline_abs, adjusted_abs) if adjusted < base)
    return {
        "count": len(samples),
        "baseline_abs_log": sum(baseline_abs) / len(baseline_abs),
        "event_heat_abs_log": sum(adjusted_abs) / len(adjusted_abs),
        "delta_abs_log": (sum(adjusted_abs) - sum(baseline_abs)) / len(samples),
        "baseline_signed_log": sum(baseline_signed) / len(baseline_signed),
        "event_heat_signed_log": sum(adjusted_signed) / len(adjusted_signed),
        "improved_share": improved / len(samples),
    }


def summarize_variants(samples: list[dict[str, Any]], variants: tuple[str, ...]) -> dict[str, Any]:
    if not samples:
        return {
            variant: {
                "count": 0,
                "abs_log": None,
                "signed_log": None,
                "delta_abs_log_vs_recorded": None,
                "improved_share_vs_recorded": None,
            }
            for variant in variants
        }
    recorded_abs = [
        log_error(float(item["recorded_prediction"]), float(item["final_score"]))
        for item in samples
    ]
    summary: dict[str, Any] = {}
    for variant in variants:
        abs_values = [
            log_error(float(item[variant]), float(item["final_score"]))
            for item in samples
        ]
        signed_values = [
            signed_log_error(float(item[variant]), float(item["final_score"]))
            for item in samples
        ]
        improved = sum(1 for value, recorded in zip(abs_values, recorded_abs) if value < recorded)
        summary[variant] = {
            "count": len(samples),
            "abs_log": sum(abs_values) / len(abs_values),
            "signed_log": sum(signed_values) / len(signed_values),
            "delta_abs_log_vs_recorded": (sum(abs_values) - sum(recorded_abs)) / len(samples),
            "improved_share_vs_recorded": improved / len(samples),
        }
    return summary


def summarize_variant_groups(
    samples: list[dict[str, Any]],
    variants: tuple[str, ...],
) -> dict[str, Any]:
    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_rank_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_region_progress: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_region_rank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        region = str(sample["region"])
        bucket = str(sample["progress_bucket"])
        rank_group = str(sample["rank_bucket"])
        event_key = f"{region}:{int(sample['event_id'])}"
        by_region[region].append(sample)
        by_bucket[bucket].append(sample)
        by_rank_bucket[rank_group].append(sample)
        by_region_progress[f"{region}:{bucket}"].append(sample)
        by_region_rank[f"{region}:{rank_group}"].append(sample)
        by_event[event_key].append(sample)
    return {
        "overall": summarize_variants(samples, variants),
        "by_region": {key: summarize_variants(value, variants) for key, value in sorted(by_region.items())},
        "by_progress_bucket": {key: summarize_variants(by_bucket.get(key, []), variants) for key, _lower, _upper in PROGRESS_BUCKETS},
        "by_rank_bucket": {key: summarize_variants(by_rank_bucket.get(key, []), variants) for key, _lower, _upper in RANK_BUCKETS},
        "by_region_progress_bucket": {key: summarize_variants(value, variants) for key, value in sorted(by_region_progress.items())},
        "by_region_rank_bucket": {key: summarize_variants(value, variants) for key, value in sorted(by_region_rank.items())},
        "by_event": {key: summarize_variants(value, variants) for key, value in sorted(by_event.items())},
    }


def summarize_prediction_samples(samples: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    if not samples:
        return {
            "count": 0,
            "abs_log": None,
            "signed_log": None,
            "delta_abs_log_vs_recorded": None,
            "improved_share_vs_recorded": None,
        }
    abs_values = [
        log_error(float(item[prediction_key]), float(item["final_score"]))
        for item in samples
    ]
    signed_values = [
        signed_log_error(float(item[prediction_key]), float(item["final_score"]))
        for item in samples
    ]
    recorded_abs = [
        log_error(float(item["recorded_prediction"]), float(item["final_score"]))
        for item in samples
    ]
    improved = sum(1 for value, recorded in zip(abs_values, recorded_abs) if value < recorded)
    return {
        "count": len(samples),
        "abs_log": sum(abs_values) / len(abs_values),
        "signed_log": sum(signed_values) / len(signed_values),
        "delta_abs_log_vs_recorded": (sum(abs_values) - sum(recorded_abs)) / len(samples),
        "improved_share_vs_recorded": improved / len(samples),
    }


def summarize_postprocessed_variants(
    postprocessed: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "overall": {},
        "by_region": defaultdict(dict),
        "by_progress_bucket": defaultdict(dict),
        "by_rank_bucket": defaultdict(dict),
        "by_region_progress_bucket": defaultdict(dict),
        "by_region_rank_bucket": defaultdict(dict),
        "by_event": defaultdict(dict),
    }
    for variant, rows in postprocessed.items():
        result["overall"][variant] = summarize_prediction_samples(rows, "model_prediction")
        groupers = {
            "by_region": lambda item: str(item["region"]),
            "by_progress_bucket": lambda item: str(item["progress_bucket"]),
            "by_rank_bucket": lambda item: str(item["rank_bucket"]),
            "by_region_progress_bucket": lambda item: f"{item['region']}:{item['progress_bucket']}",
            "by_region_rank_bucket": lambda item: f"{item['region']}:{item['rank_bucket']}",
            "by_event": lambda item: f"{item['region']}:{int(item['event_id'])}",
        }
        for group_name, key_func in groupers.items():
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                grouped[key_func(row)].append(row)
            for key, value in sorted(grouped.items()):
                result[group_name][key][variant] = summarize_prediction_samples(value, "model_prediction")
    return {
        key: dict(value) if isinstance(value, defaultdict) else value
        for key, value in result.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay prediction records with the event heat prior in read-only mode.")
    parser.add_argument("--config", type=Path, default=None, help="Production config. Used only for masterdata env paths.")
    parser.add_argument("--record-dir", type=Path, default=REPO_ROOT / "prediction_records")
    parser.add_argument("--history-dir", type=Path, default=REPO_ROOT / "live_history")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "analysis" / "event_heat_prior_validation.json")
    parser.add_argument("--max-progress", type=float, default=0.35)
    args = parser.parse_args()

    if args.config is not None:
        os.environ.update(env_from_config(read_json(args.config)))

    module = load_module("train_baseline_module", TRAIN_SCRIPT)
    predict_module = load_module("predict_live_event_module", PREDICT_SCRIPT)
    rows = load_prediction_rows(args.record_dir)
    final_scores = load_final_scores(args.history_dir)
    event_meta = load_event_meta_from_prediction_rows(module, rows)
    event_feature_contexts = module.load_event_feature_contexts()
    finals = build_finals_from_history(module, event_meta, final_scores)
    validation_events = {region: set() for region in module.REGIONS}
    full_character_heat_priors = module.build_character_heat_priors(
        event_meta=event_meta,
        finals=finals,
        validation_events=validation_events,
        event_feature_contexts=event_feature_contexts,
        model_series_types=module.NORMAL_MODEL_SERIES_TYPES,
    )
    full_event_heat_priors = module.build_event_heat_final_priors(
        event_meta=event_meta,
        finals=finals,
        validation_events=validation_events,
        event_feature_contexts=event_feature_contexts,
        character_heat_priors=full_character_heat_priors,
        model_series_types=module.NORMAL_MODEL_SERIES_TYPES,
    )

    eligible_rows: list[tuple[dict[str, Any], int]] = []
    skipped_missing_final = 0
    skipped_no_context = 0
    for row in rows:
        if float(row["progress"]) > args.max_progress:
            continue
        key = (str(row["region"]), int(row["event_id"]), "total", int(row["character_id"]), int(row["rank"]))
        final_score = final_scores.get(key)
        if final_score is None:
            skipped_missing_final += 1
            continue
        if (str(row["region"]), int(row["event_id"])) not in event_feature_contexts:
            skipped_no_context += 1
            continue
        eligible_rows.append((row, final_score))

    leave_one_payloads: dict[tuple[str, int], dict[str, Any]] = {}
    for region, event_id in sorted({(str(row["region"]), int(row["event_id"])) for row, _final in eligible_rows}):
        leave_one_validation_events = {name: set() for name in module.REGIONS}
        leave_one_validation_events[region].add(event_id)
        character_heat_priors = module.build_character_heat_priors(
            event_meta=event_meta,
            finals=finals,
            validation_events=leave_one_validation_events,
            event_feature_contexts=event_feature_contexts,
            model_series_types=module.NORMAL_MODEL_SERIES_TYPES,
        )
        event_heat_priors = module.build_event_heat_final_priors(
            event_meta=event_meta,
            finals=finals,
            validation_events=leave_one_validation_events,
            event_feature_contexts=event_feature_contexts,
            character_heat_priors=character_heat_priors,
            model_series_types=module.NORMAL_MODEL_SERIES_TYPES,
        )
        leave_one_payloads[(region, event_id)] = {
            "live_character_heat_priors": character_heat_priors,
            "live_event_heat_priors": event_heat_priors,
            "live_event_feature_contexts": event_feature_contexts,
        }

    samples: list[dict[str, Any]] = []
    early_stage_samples: list[dict[str, Any]] = []
    for row, final_score in eligible_rows:
        payload = leave_one_payloads[(str(row["region"]), int(row["event_id"]))]
        adjusted, debug = apply_event_heat_prior(module, row, float(row["model_prediction"]), payload)
        early_stage = early_stage_replay_predictions(predict_module, module, row, payload)
        samples.append({
            **row,
            "final_score": final_score,
            "event_heat_prediction": adjusted,
            "baseline_abs_log": log_error(float(row["model_prediction"]), float(final_score)),
            "event_heat_abs_log": log_error(adjusted, float(final_score)),
            "baseline_signed_log": signed_log_error(float(row["model_prediction"]), float(final_score)),
            "event_heat_signed_log": signed_log_error(adjusted, float(final_score)),
            "event_heat_prior_source": debug.get("event_heat_prior_source"),
            "event_heat_prior_count": debug.get("event_heat_prior_count"),
            "event_heat_prior_log": debug.get("event_heat_prior_log"),
            "event_heat_prior_blend_weight": debug.get("event_heat_prior_blend_weight"),
            "progress_bucket": progress_bucket(float(row["progress"])),
            "rank_bucket": rank_bucket(int(row["rank"])),
        })
        early_stage_samples.append({
            **row,
            "final_score": final_score,
            "recorded_prediction": float(row["model_prediction"]),
            "progress_bucket": progress_bucket(float(row["progress"])),
            "rank_bucket": rank_bucket(int(row["rank"])),
            **early_stage,
        })

    early_variants = (
        "recorded_prediction",
        "pre_stage_prediction",
        "early_no_heat_prediction",
        "early_heat_scale_25_prediction",
        "early_heat_scale_50_prediction",
        "early_heat_scale_75_prediction",
        "early_heat_prediction",
    )
    postprocess_variants = {
        variant: replay_full_postprocess(predict_module, early_stage_samples, variant)
        for variant in early_variants
        if variant != "recorded_prediction"
    }

    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_region_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        region = str(sample["region"])
        bucket = str(sample["progress_bucket"])
        event_key = f"{region}:{int(sample['event_id'])}"
        by_region[region].append(sample)
        by_bucket[bucket].append(sample)
        by_region_bucket[f"{region}:{bucket}"].append(sample)
        by_event[event_key].append(sample)

    regressions = sorted(
        (
            {
                "event": key,
                **summarize(event_samples),
            }
            for key, event_samples in by_event.items()
        ),
        key=lambda item: float(item.get("delta_abs_log") or 0.0),
        reverse=True,
    )

    output = {
        "settings": {
            "max_progress": args.max_progress,
            "record_dir": str(args.record_dir),
            "history_dir": str(args.history_dir),
            "normal_events": sorted(f"{region}:{event_id}" for region, event_id in event_meta),
            "event_heat_prior_condition_groups": len(full_event_heat_priors.get("condition_histories", {})),
            "event_heat_prior_region_groups": len(full_event_heat_priors.get("region_histories", {})),
            "event_heat_prior_global_groups": len(full_event_heat_priors.get("global_histories", {})),
            "leave_one_events": sorted(f"{region}:{event_id}" for region, event_id in leave_one_payloads),
            "skipped_missing_final": skipped_missing_final,
            "skipped_no_context": skipped_no_context,
        },
        "overall": summarize(samples),
        "by_region": {key: summarize(value) for key, value in sorted(by_region.items())},
        "by_progress_bucket": {key: summarize(by_bucket.get(key, [])) for key, _lower, _upper in PROGRESS_BUCKETS},
        "by_region_progress_bucket": {key: summarize(value) for key, value in sorted(by_region_bucket.items())},
        "by_event": {key: summarize(value) for key, value in sorted(by_event.items())},
        "early_stage_replay": summarize_variant_groups(
            early_stage_samples,
            early_variants,
        ),
        "postprocess_replay": summarize_postprocessed_variants(postprocess_variants),
        "worst_event_regressions": regressions[:10],
        "best_event_improvements": list(reversed(regressions[-10:])),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "overall": output["overall"],
        "by_region": output["by_region"],
        "by_progress_bucket": output["by_progress_bucket"],
        "early_stage_replay": output["early_stage_replay"]["overall"],
        "postprocess_replay": output["postprocess_replay"]["overall"],
        "worst_event_regressions": output["worst_event_regressions"][:5],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
