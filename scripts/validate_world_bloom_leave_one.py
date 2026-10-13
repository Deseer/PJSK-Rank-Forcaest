#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_training_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("train_baseline_module", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_baseline_module"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_event_ids(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_ranks(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def parse_regions(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def world_bloom_legacy_only(legacy_event_files):
    return {
        region: [
            (event_id, series_type, files)
            for event_id, series_type, files in entries
            if series_type == "world_bloom"
        ]
        for region, entries in legacy_event_files.items()
    }


def build_world_bloom_finals(module, data_root: Path, event_meta, selected_ranks, legacy_event_files):
    finals = {}
    for region in module.REGIONS:
        for row in module.iter_all_rank_timeseries_rows(data_root, region, selected_ranks, legacy_event_files):
            meta = event_meta.get((region, row["event_id"]))
            if meta is None or meta.series_type not in module.WL_MODEL_SERIES_TYPES:
                continue
            finals[(region, row["event_id"], row["character_id"], row["rank"])] = (
                row["score"],
                row["timestamp"],
            )
    return finals


def cache_name_for_ranks(selected_ranks: set[int], train_regions: set[str]) -> str:
    region_label = "_".join(sorted(train_regions))
    rank_label = "_".join(str(rank) for rank in sorted(selected_ranks))
    return f"world_bloom_rank_timeseries_{region_label}_{rank_label}.csv.gz"


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


def write_cached_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["region", "event_id", "series_type", "character_id", "rank", "timestamp", "score"],
        )
        writer.writeheader()
        writer.writerows(rows)


def load_or_build_world_bloom_rows(module, data_root: Path, event_meta, selected_ranks, legacy_event_files, cache_path: Path, refresh: bool):
    if cache_path.exists() and not refresh:
        rows = load_cached_rows(cache_path)
        print(f"loaded cached rows: {cache_path} rows={len(rows)}", flush=True)
        return rows

    rows = []
    regions_with_legacy = {
        region
        for region, entries in legacy_event_files.items()
        if entries
    }
    for region in module.REGIONS:
        if region not in regions_with_legacy and region != "jp":
            continue
        for row in module.iter_all_rank_timeseries_rows(data_root, region, selected_ranks, legacy_event_files):
            meta = event_meta.get((region, row["event_id"]))
            if meta is None or meta.series_type not in module.WL_MODEL_SERIES_TYPES:
                continue
            rows.append({
                "region": region,
                "event_id": int(row["event_id"]),
                "series_type": "world_bloom",
                "character_id": int(row["character_id"]),
                "rank": int(row["rank"]),
                "timestamp": int(row["timestamp"]),
                "score": int(row["score"]),
            })
    rows.sort(key=lambda row: (
        row["region"],
        row["event_id"],
        row["character_id"],
        row["rank"],
        row["timestamp"],
    ))
    write_cached_rows(cache_path, rows)
    print(f"built cached rows: {cache_path} rows={len(rows)}", flush=True)
    return rows


def install_cached_iterator(module, rows):
    rows_by_region = defaultdict(list)
    for row in rows:
        rows_by_region[row["region"]].append(row)

    def iter_cached_rows(_data_root, region, selected_ranks, _legacy_event_files):
        rank_filter = set(selected_ranks)
        for row in rows_by_region.get(region, []):
            if row["rank"] in rank_filter:
                yield dict(row)

    module.iter_all_rank_timeseries_rows = iter_cached_rows


def grouped_metrics(module, rows, predictions, targets, key_name: str):
    grouped = defaultdict(lambda: ([], []))
    for row, prediction, target in zip(rows, predictions, targets, strict=True):
        key = row[key_name]
        preds, trues = grouped[key]
        preds.append(float(prediction))
        trues.append(float(target))
    return {
        str(key): module.score_metrics(
            np.asarray(trues, dtype=np.float64),
            np.asarray(preds, dtype=np.float64),
        )
        for key, (preds, trues) in sorted(grouped.items(), key=lambda item: int(item[0]))
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Leave-one validation for World Bloom forecasts.")
    parser.add_argument("--region", default="cn")
    parser.add_argument("--events", default="112,118,124,130,137,140")
    parser.add_argument("--ranks", default="100,1000")
    parser.add_argument("--train-regions", default="jp,cn")
    parser.add_argument("--boundary-pressure", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "cn_wl1_leave_one.json",
    )
    args = parser.parse_args()

    module = load_training_module(args.train_script.resolve())
    data_root = args.data_root.resolve()
    selected_ranks = parse_ranks(args.ranks)
    train_regions = parse_regions(args.train_regions)
    finished_event_ids = module.load_finished_event_ids()
    event_meta = module.load_event_meta(data_root, finished_event_ids)
    normalized_event_keys = {region: set() for region in module.REGIONS}
    legacy_event_files = world_bloom_legacy_only(
        module.load_legacy_event_files(data_root, normalized_event_keys)
    )
    legacy_event_files = {
        region: entries if region in train_regions else []
        for region, entries in legacy_event_files.items()
    }
    if "jp" in train_regions:
        # JP WL2 is sourced from the sekai.best cache, so avoid mixing older
        # local legacy dumps for the same chapters.
        legacy_event_files["jp"] = []
    cache_path = Path(__file__).resolve().parents[1] / "data" / "cache" / cache_name_for_ranks(selected_ranks, train_regions)
    cached_rows = load_or_build_world_bloom_rows(
        module,
        data_root,
        event_meta,
        selected_ranks,
        legacy_event_files,
        cache_path,
        args.refresh_cache,
    )
    install_cached_iterator(module, cached_rows)
    finals = build_world_bloom_finals(module, data_root, event_meta, selected_ranks, legacy_event_files)
    event_feature_contexts = module.load_event_feature_contexts()
    boundary_pressure_contexts = (
        module.build_boundary_pressure_contexts(
            data_root=data_root,
            event_meta=event_meta,
            selected_ranks=selected_ranks,
            model_series_types=module.WL_MODEL_SERIES_TYPES,
        )
        if args.boundary_pressure
        else {}
    )

    results = []
    for holdout_event in parse_event_ids(args.events):
        validation_events = {region: set() for region in module.REGIONS}
        validation_events[args.region] = {holdout_event}
        character_heat_priors = module.build_character_heat_priors(
            event_meta=event_meta,
            finals=finals,
            validation_events=validation_events,
            event_feature_contexts=event_feature_contexts,
            model_series_types=module.WL_MODEL_SERIES_TYPES,
        )
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
            character_heat_priors=character_heat_priors,
            jp_shape_priors=jp_shape_priors,
            boundary_pressure_contexts=boundary_pressure_contexts,
            legacy_event_files=legacy_event_files,
            model_series_types=module.WL_MODEL_SERIES_TYPES,
        )
        result = module.train_pace_bias_forecast_model(
            train_x,
            train_rows,
            val_x,
            val_y,
            val_rows,
            model_series_types=module.WL_MODEL_SERIES_TYPES,
        )
        predictions = np.asarray(result["predictions"], dtype=np.float64)
        true_scores = np.expm1(val_y)
        payload = {
            "holdout_event": holdout_event,
            "selected_ranks": sorted(selected_ranks),
            "train_samples": int(train_x.shape[0]),
            "validation_samples": int(val_x.shape[0]),
            "metrics": module.score_metrics(true_scores, predictions),
            "baseline_metrics": {
                "pace_6h": result["metrics"]["pace_6h_baseline"],
                "bias_calibrated": result["metrics"]["pace_6h_bias_without_wl2_prior"],
                "linear": result["metrics"]["linear_baseline"],
                "jp_shape": result["metrics"]["jp_shape_baseline"],
                "hybrid": result["metrics"]["hybrid_baseline"],
            },
            "by_rank": grouped_metrics(module, val_rows, predictions, true_scores, "rank"),
            "by_character": grouped_metrics(module, val_rows, predictions, true_scores, "character_id"),
        }
        results.append(payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
