#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_baseline.py"
SHAPE_VALIDATION_SCRIPT = REPO_ROOT / "scripts" / "validate_world_bloom_shape_optimizations.py"


def parse_csv_ints(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_config_env(config_path: Path | None) -> None:
    if config_path is None:
        return
    config = load_json(config_path)
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
            os.environ[env_name] = str(Path(value))
    cache_file = required.get("jp_wl2_sekai_best_cache")
    if cache_file:
        os.environ["HARUKI_FORECAST_SEKAI_BEST_JP_WL2_DIR"] = str(Path(cache_file).parent)
    world_bloom_cache_dir = required.get("world_bloom_sekai_best_cache_dir")
    if world_bloom_cache_dir:
        os.environ["HARUKI_FORECAST_SEKAI_BEST_WORLD_BLOOM_DIR"] = str(Path(world_bloom_cache_dir))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def complete_external_rows(module, rows: list[dict[str, int | str]], event_meta, min_final_progress: float):
    grouped: dict[tuple[str, int, int, int], list[dict[str, int | str]]] = defaultdict(list)
    for row in rows:
        grouped[(
            str(row["region"]),
            int(row["event_id"]),
            int(row["character_id"]),
            int(row["rank"]),
        )].append(row)

    kept_keys: set[tuple[str, int, int, int]] = set()
    dropped: list[dict[str, object]] = []
    for key, series_rows in sorted(grouped.items()):
        region, event_id, character_id, rank = key
        meta = event_meta.get((region, event_id))
        if meta is None or not module.has_event_line_window(meta, character_id):
            dropped.append({
                "region": region,
                "event_id": event_id,
                "character_id": character_id,
                "rank": rank,
                "reason": "missing_event_meta_or_chapter_window",
                "rows": len(series_rows),
                "max_progress": 0.0,
            })
            continue
        max_progress = max(
            module.event_line_progress(meta, character_id, int(row["timestamp"]))
            for row in series_rows
        )
        if max_progress >= min_final_progress:
            kept_keys.add(key)
            continue
        dropped.append({
            "region": region,
            "event_id": event_id,
            "character_id": character_id,
            "rank": rank,
            "reason": "incomplete_final_observation",
            "rows": len(series_rows),
            "max_progress": max_progress,
        })

    filtered = [
        row
        for row in rows
        if (
            str(row["region"]),
            int(row["event_id"]),
            int(row["character_id"]),
            int(row["rank"]),
        ) in kept_keys
    ]
    return filtered, dropped


def progress_band(progress: float) -> str:
    if progress < 0.30:
        return "early<30%"
    if progress < 0.65:
        return "mid30-65%"
    if progress < 0.85:
        return "late65-85%"
    return "final85%+"


def score_metrics(module, rows: list[dict[str, object]], predictions: list[float] | np.ndarray):
    if not rows:
        return {
            "mae": None,
            "rmse": None,
            "mape": None,
            "smape": None,
            "within_5pct": None,
            "within_10pct": None,
            "within_20pct": None,
        }
    return module.score_metrics(
        np.asarray([float(row["final_score"]) for row in rows], dtype=np.float64),
        np.asarray(predictions, dtype=np.float64),
    )


def grouped_metrics(module, rows: list[dict[str, object]], predictions: list[float], key_name: str):
    grouped: dict[object, tuple[list[dict[str, object]], list[float]]] = defaultdict(lambda: ([], []))
    for row, prediction in zip(rows, predictions, strict=True):
        key = progress_band(float(row["progress"])) if key_name == "progress_band" else row[key_name]
        group_rows, group_predictions = grouped[key]
        group_rows.append(row)
        group_predictions.append(float(prediction))
    return {
        str(key): {
            "samples": len(group_rows),
            **score_metrics(module, group_rows, group_predictions),
        }
        for key, (group_rows, group_predictions) in sorted(grouped.items(), key=lambda item: str(item[0]))
    }


def build_validation_rows(
    module,
    validation_module,
    *,
    data_root: Path,
    event_meta,
    finals,
    event_feature_contexts,
    selected_ranks: set[int],
    target_region: str,
    target_events: list[int],
    strict_same_event: bool,
):
    all_rows: list[dict[str, object]] = []
    predictions: dict[str, list[float]] = defaultdict(list)
    legacy_event_files = {region: [] for region in module.REGIONS}

    for holdout_event in target_events:
        validation_events = {region: set() for region in module.REGIONS}
        if strict_same_event:
            for region in ("en", "tw", "kr"):
                validation_events[region].add(holdout_event)
        else:
            validation_events[target_region].add(holdout_event)

        jp_shape_priors = module.build_jp_shape_priors(
            data_root=data_root,
            event_meta=event_meta,
            finals=finals,
            validation_events=validation_events,
            selected_ranks=selected_ranks,
            legacy_event_files=legacy_event_files,
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
            legacy_event_files=legacy_event_files,
            model_series_types=module.WL_MODEL_SERIES_TYPES,
        )
        _ = (train_x, train_y, val_x, val_y)
        target_rows = [
            dict(row, holdout_event=holdout_event)
            for row in val_rows
            if str(row["region"]) == target_region and int(row["event_id"]) == holdout_event
        ]
        base = np.asarray([float(row["jp_shape_projection"]) for row in target_rows], dtype=np.float64)
        calibrator = validation_module.fit_log_residual_calibrator(train_rows, "jp_shape_projection")
        calibrated = validation_module.apply_log_residual_calibration(
            target_rows,
            base,
            calibrator,
            min_count=8,
            shrinkage=0.0,
        )
        hybrid = np.asarray([float(row["hybrid_projection"]) for row in target_rows], dtype=np.float64)
        calibrated_hybrid = validation_module.log_blend(calibrated, hybrid, 0.9)
        calibrated_hybrid = np.maximum(
            calibrated_hybrid,
            np.asarray([float(row["current_score"]) for row in target_rows], dtype=np.float64),
        )

        all_rows.extend(target_rows)
        predictions["jp_shape"].extend(base.tolist())
        predictions["shape_residual_calibrated"].extend(calibrated.tolist())
        predictions["calibrated_shape_hybrid_0_9"].extend(calibrated_hybrid.tolist())

    return all_rows, predictions


def summarize_model(module, rows: list[dict[str, object]], predictions: list[float]) -> dict[str, object]:
    return {
        "overall": score_metrics(module, rows, predictions),
        "by_rank": grouped_metrics(module, rows, predictions, "rank"),
        "by_progress_band": grouped_metrics(module, rows, predictions, "progress_band"),
        "by_event": grouped_metrics(module, rows, predictions, "holdout_event"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate World Bloom shape forecasts for overseas regions.")
    parser.add_argument("--config", type=Path, default=None, help="Optional production config for path env vars.")
    parser.add_argument("--regions", default="en,tw,kr", help="Comma-separated target regions.")
    parser.add_argument("--ranks", default="100,1000", help="Comma-separated ranks.")
    parser.add_argument(
        "--min-final-progress",
        type=float,
        default=None,
        help="Drop external WL series whose last observation is before this chapter progress.",
    )
    parser.add_argument("--allow-same-event-training", action="store_true")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/deseer/PycharmProjects/Haruki-Tracker-Exports"),
    )
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "world_bloom_region_validation.json")
    args = parser.parse_args()

    apply_config_env(args.config.resolve() if args.config else None)
    module = load_module("train_baseline_module", TRAIN_SCRIPT)
    validation_module = load_module("shape_validation_module", SHAPE_VALIDATION_SCRIPT)
    selected_ranks = parse_csv_ints(args.ranks)
    target_regions = parse_csv_strings(args.regions)
    min_final_progress = (
        float(args.min_final_progress)
        if args.min_final_progress is not None
        else float(module.WORLD_BLOOM_MIN_FINAL_PROGRESS)
    )

    data_root = args.data_root.resolve()
    event_meta = module.load_event_meta(data_root, module.load_finished_event_ids())
    rows: list[dict[str, int | str]] = []
    source_regions = ["jp", *target_regions]
    for region in dict.fromkeys(source_regions):
        rows.extend(module.iter_external_world_bloom_rank_timeseries_rows(region, selected_ranks))
    rows.sort(key=lambda row: (
        str(row["region"]),
        int(row["event_id"]),
        int(row["character_id"]),
        int(row["rank"]),
        int(row["timestamp"]),
    ))
    filtered_rows, dropped_series = complete_external_rows(
        module,
        rows,
        event_meta,
        min_final_progress,
    )
    validation_module.install_cached_iterator(module, filtered_rows)
    finals = validation_module.build_finals(filtered_rows)
    event_feature_contexts = module.load_event_feature_contexts()

    output: dict[str, object] = {
        "definition": {
            "selected_ranks": sorted(selected_ranks),
            "target_regions": target_regions,
            "source_regions": list(dict.fromkeys(source_regions)),
            "min_final_progress": min_final_progress,
            "strict_same_event": not args.allow_same_event_training,
            "models": [
                "jp_shape",
                "shape_residual_calibrated",
                "calibrated_shape_hybrid_0_9",
            ],
        },
        "source": {
            "raw_rows": len(rows),
            "filtered_rows": len(filtered_rows),
            "raw_series": len({
                (row["region"], row["event_id"], row["character_id"], row["rank"])
                for row in rows
            }),
            "filtered_series": len({
                (row["region"], row["event_id"], row["character_id"], row["rank"])
                for row in filtered_rows
            }),
            "dropped_series": dropped_series,
        },
        "regions": {},
    }

    for region in target_regions:
        target_events = sorted({int(row["event_id"]) for row in filtered_rows if str(row["region"]) == region})
        validation_rows, model_predictions = build_validation_rows(
            module,
            validation_module,
            data_root=data_root,
            event_meta=event_meta,
            finals=finals,
            event_feature_contexts=event_feature_contexts,
            selected_ranks=selected_ranks,
            target_region=region,
            target_events=target_events,
            strict_same_event=not args.allow_same_event_training,
        )
        output["regions"][region] = {
            "events": target_events,
            "samples": len(validation_rows),
            "models": {
                name: summarize_model(module, validation_rows, predictions)
                for name, predictions in model_predictions.items()
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
