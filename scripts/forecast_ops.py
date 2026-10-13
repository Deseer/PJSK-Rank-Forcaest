#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_baseline.py"
PREDICT_SCRIPT = REPO_ROOT / "scripts" / "predict_live_event.py"
FETCH_WORLD_BLOOM_SCRIPT = REPO_ROOT / "scripts" / "fetch_sekai_best_world_bloom.py"
VALIDATE_WORLD_BLOOM_SCRIPT = REPO_ROOT / "scripts" / "validate_world_bloom_regions.py"
FETCH_EVENT_TRACKER_HISTORY_SCRIPT = REPO_ROOT / "scripts" / "fetch_event_tracker_history.py"
DEFAULT_LIVE_CALIBRATION = {
    "min_count": 8,
    "shrinkage": 24.0,
    "max_abs_log_adjustment": 0.08,
    "max_abs_sample_log_ratio": 0.35,
    "min_progress": 0.05,
    "max_progress": 0.995,
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def encode_calibration_key(key: tuple[object, ...]) -> str:
    return json.dumps(list(key), ensure_ascii=False, separators=(",", ":"))


def prediction_archive_dir(config: dict[str, Any]) -> Path:
    return Path(config.get(
        "prediction_archive_dir",
        str(Path(config["results_dir"]).parent / "prediction_archive"),
    ))


def prediction_record_dir(config: dict[str, Any]) -> Path:
    return Path(config.get(
        "prediction_record_dir",
        str(Path(config["results_dir"]).parent / "prediction_records"),
    ))


def prediction_cold_archive_dir(config: dict[str, Any]) -> Path | None:
    value = config.get("prediction_cold_archive_dir")
    return Path(value) if value else None


def live_calibration_path(config: dict[str, Any]) -> Path:
    return Path(config.get(
        "live_calibration_file",
        str(Path(config["results_dir"]) / "live_calibration.json"),
    ))


def live_calibration_settings(config: dict[str, Any]) -> dict[str, float]:
    raw = config.get("live_calibration", {})
    settings = dict(DEFAULT_LIVE_CALIBRATION)
    if isinstance(raw, dict):
        for key in settings:
            if key in raw:
                settings[key] = raw[key]
    return {
        "min_count": int(settings["min_count"]),
        "shrinkage": float(settings["shrinkage"]),
        "max_abs_log_adjustment": float(settings["max_abs_log_adjustment"]),
        "max_abs_sample_log_ratio": float(settings["max_abs_sample_log_ratio"]),
        "min_progress": float(settings["min_progress"]),
        "max_progress": float(settings["max_progress"]),
    }


def default_total_output_path(output: Path) -> Path:
    if output.stem.endswith("_prediction"):
        return output.with_name(output.stem.removesuffix("_prediction") + "_total_prediction" + output.suffix)
    return output.with_name(output.stem + "_total" + output.suffix)


def retarget_event_path(path: str | Path, old_event_id: int, new_event_id: int) -> str:
    value = str(path)
    old = str(old_event_id)
    new = str(new_event_id)
    replacements = (
        (f"_{old}_", f"_{new}_"),
        (f"_{old}.", f"_{new}."),
        (f"/{old}/", f"/{new}/"),
    )
    for source, target in replacements:
        value = value.replace(source, target)
    return value


def masterdata_root(config: dict[str, Any], region: str) -> Path | None:
    required = config.get("required_local_paths", {})
    root_value = required.get(f"{region}_masterdata")
    if not root_value:
        return None
    root = Path(root_value)
    return root if root.exists() else None


def current_event_from_masterdata(
    config: dict[str, Any],
    region: str,
    *,
    now_ms: int | None = None,
) -> dict[str, Any] | None:
    root = masterdata_root(config, region)
    if root is None:
        return None
    events_path = root / "events.json"
    if not events_path.exists():
        return None
    now_value = int(now_ms if now_ms is not None else time.time() * 1000)
    current_rows: list[dict[str, Any]] = []
    latest_started: dict[str, Any] | None = None
    for row in load_json(events_path):
        event_id = int(row.get("id", -1))
        start_at = row.get("startAt")
        aggregate_at = row.get("aggregateAt")
        if event_id < 0 or start_at is None or aggregate_at is None:
            continue
        start_value = int(start_at)
        aggregate_value = int(aggregate_at)
        if start_value <= now_value < aggregate_value:
            current_rows.append(row)
        elif start_value <= now_value and (
            latest_started is None or start_value > int(latest_started.get("startAt") or 0)
        ):
            latest_started = row
    if current_rows:
        return max(current_rows, key=lambda item: int(item.get("startAt") or 0))
    # Do not switch to a published-but-not-started event: EventTracker does not
    # have live rows before the opening timestamp, which would produce empty
    # predictions. Keep the latest started event until the next one actually
    # begins.
    return latest_started


def resolve_active_events(config: dict[str, Any]) -> list[dict[str, Any]]:
    configured_events = config.get("active_events", [])
    if not bool(config.get("auto_active_events", True)):
        return list(configured_events)

    resolved: list[dict[str, Any]] = []
    seen_regions: set[str] = set()
    for event in configured_events:
        region = str(event["region"])
        seen_regions.add(region)
        current = current_event_from_masterdata(config, region)
        if current is None:
            resolved.append(dict(event))
            continue
        old_event_id = int(event["event_id"])
        new_event_id = int(current["id"])
        updated = dict(event)
        updated["event_id"] = new_event_id
        updated["series_type"] = "world_bloom" if str(current.get("eventType")) == "world_bloom" else "normal"
        if "output" in updated:
            updated["output"] = retarget_event_path(str(updated["output"]), old_event_id, new_event_id)
        if "total_output" in updated:
            updated["total_output"] = retarget_event_path(str(updated["total_output"]), old_event_id, new_event_id)
        resolved.append(updated)

    # If active_events omits a supported region, auto-discover it when
    # masterdata is configured. Existing config order remains authoritative.
    for key in sorted(config.get("required_local_paths", {})):
        if not key.endswith("_masterdata"):
            continue
        region = key.removesuffix("_masterdata")
        if region in seen_regions:
            continue
        current = current_event_from_masterdata(config, region)
        if current is None:
            continue
        event_id = int(current["id"])
        output = Path(config.get("results_dir", REPO_ROOT / "results")).parent / "live" / f"{region}_{event_id}_prediction.json"
        resolved.append({
            "region": region,
            "event_id": event_id,
            "series_type": "world_bloom" if str(current.get("eventType")) == "world_bloom" else "normal",
            "output": str(output),
        })
    return resolved


def event_masterdata_series_type(config: dict[str, Any], event: dict[str, Any]) -> str | None:
    region = str(event["region"])
    root = masterdata_root(config, region)
    if root is None:
        return None
    events_path = root / "events.json"
    if not events_path.exists():
        return None
    event_id = int(event["event_id"])
    for row in load_json(events_path):
        if int(row.get("id", -1)) == event_id:
            return "world_bloom" if str(row.get("eventType")) == "world_bloom" else "normal"
    return None


def has_world_bloom_character_chapters(config: dict[str, Any], event: dict[str, Any]) -> bool:
    root = masterdata_root(config, str(event["region"]))
    path = root / "worldBlooms.json" if root else None
    if path is None or not path.exists():
        return False
    event_id = int(event["event_id"])
    return any(
        int(row.get("eventId", -1)) == event_id
        and row.get("gameCharacterId") is not None
        and int(row["gameCharacterId"]) > 0
        for row in load_json(path)
    )


def prediction_jobs(config: dict[str, Any], event: dict[str, Any]) -> list[dict[str, Any]]:
    output = Path(event["output"])
    region = str(event["region"])
    event_id = int(event["event_id"])
    series_type = str(event.get("series_type") or event_masterdata_series_type(config, event) or "normal")
    if series_type != "world_bloom":
        return [{
            "region": region,
            "event_id": event_id,
            "series_type": series_type,
            "leaderboard_scope": "total",
            "output": output,
            "event": event,
        }]

    total_output = Path(event.get("total_output") or default_total_output_path(output))
    if not has_world_bloom_character_chapters(config, event):
        return [{
            "region": region,
            "event_id": event_id,
            "series_type": "world_bloom",
            "leaderboard_scope": "total",
            "output": total_output,
            "event": event,
        }]
    return [
        {
            "region": region,
            "event_id": event_id,
            "series_type": "world_bloom",
            "leaderboard_scope": "chapter",
            "output": output,
            "event": event,
        },
        {
            "region": region,
            "event_id": event_id,
            "series_type": "world_bloom",
            "leaderboard_scope": "total",
            "output": total_output,
            "event": event,
        },
    ]


def env_from_config(config: dict[str, Any]) -> dict[str, str]:
    required = config.get("required_local_paths", {})
    env = os.environ.copy()
    if config.get("shape_live_model_pkl"):
        env["HARUKI_FORECAST_SHAPE_LIVE_MODEL"] = str(Path(config["shape_live_model_pkl"]))
    if config.get("opening_blend_file") and config.get("opening_blend_enabled", True):
        env["HARUKI_FORECAST_OPENING_BLEND"] = str(Path(config["opening_blend_file"]))
    if config.get("live_heat_enabled"):
        # mid/late hierarchical live-heat correction layer, see
        # analysis/offline_intrinsic_live/STRATEGY.md and scripts/live_heat_layer.py
        env["HARUKI_FORECAST_LIVE_HEAT"] = "1"
    quantile_point_min = config.get("quantile_point_min_progress")
    if quantile_point_min:
        # from this progress on, take the point estimate from the median-loss
        # model; validated through the full production chain in
        # analysis/offline_intrinsic_live/replay_chain.py
        env["HARUKI_FORECAST_QUANTILE_POINT_MIN"] = str(quantile_point_min)
    history_dir = config.get("event_tracker", {}).get("history_dir")
    if history_dir:
        env["HARUKI_FORECAST_HISTORY_DIR"] = str(Path(history_dir))
    token_file = config.get("event_tracker", {}).get("token_file")
    if token_file:
        env["HARUKI_FORECAST_TRACKER_TOKEN_FILE"] = str(Path(token_file))
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


def run_command(args: list[str], *, stdout_path: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(args), flush=True)
    if stdout_path is None:
        subprocess.run(args, cwd=REPO_ROOT, check=True, env=env)
        return
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = stdout_path.with_name(f".{stdout_path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            subprocess.run(args, cwd=REPO_ROOT, check=True, stdout=handle, env=env)
        tmp_path.replace(stdout_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def config_path_entries(config: dict[str, Any]) -> list[tuple[str, Path, bool]]:
    required = config.get("required_local_paths", {})
    entries: list[tuple[str, Path, bool]] = [
        ("data_root", Path(config["data_root"]), True),
    ]
    for key in (
        "jp_masterdata",
        "en_masterdata",
        "tw_masterdata",
        "kr_masterdata",
        "cn_masterdata",
        "world_bloom_sekai_best_cache_dir",
    ):
        value = required.get(key)
        if value:
            entries.append((f"required_local_paths.{key}", Path(value), True))
    cache_file = required.get("jp_wl2_sekai_best_cache")
    if cache_file:
        entries.append(("required_local_paths.jp_wl2_sekai_best_cache", Path(cache_file), True))
    for region, value in config.get("current_sql_zst", {}).items():
        entries.append((f"current_sql_zst.{region}", Path(value), False))
    return entries


def validate_config(config: dict[str, Any]) -> None:
    missing_keys = [
        key
        for key in ("data_root", "results_dir", "model_pkl", "state_file")
        if not config.get(key)
    ]
    if missing_keys:
        raise SystemExit(f"Missing required config keys: {', '.join(missing_keys)}")
    active_events = config.get("active_events", [])
    if not isinstance(active_events, list) or not active_events:
        raise SystemExit("Config must contain at least one active_events entry.")
    for index, event in enumerate(active_events):
        for key in ("region", "event_id", "output"):
            if key not in event:
                raise SystemExit(f"active_events[{index}] missing {key}")

    required_paths = config.get("required_local_paths", {})
    missing_path_keys = [
        key
        for key in (
            "jp_masterdata",
            "en_masterdata",
            "tw_masterdata",
            "kr_masterdata",
            "cn_masterdata",
            "jp_wl2_sekai_best_cache",
            "world_bloom_sekai_best_cache_dir",
        )
        if not required_paths.get(key)
    ]
    if missing_path_keys:
        raise SystemExit(
            "Missing required_local_paths entries: " + ", ".join(missing_path_keys)
        )

    errors: list[str] = []
    warnings: list[str] = []
    for label, path, required in config_path_entries(config):
        if path.exists():
            continue
        message = f"{label}: {path}"
        if required:
            errors.append(message)
        else:
            warnings.append(message)

    for label, value in (
        ("results_dir", config.get("results_dir")),
        ("model_pkl parent", Path(config["model_pkl"]).parent if config.get("model_pkl") else None),
        ("state_file parent", Path(config["state_file"]).parent if config.get("state_file") else None),
    ):
        if value is None:
            continue
        path = Path(value)
        if not path.exists():
            warnings.append(f"{label} does not exist yet and will be created by the relevant command: {path}")

    if warnings:
        print("Config warnings:", flush=True)
        for warning in warnings:
            print(f"  - {warning}", flush=True)
    if errors:
        print("Config errors:", flush=True)
        for error in errors:
            print(f"  - {error}", flush=True)
        raise SystemExit(1)
    print("Config OK.", flush=True)


def load_training_module():
    spec = importlib.util.spec_from_file_location("train_baseline_module", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_baseline_module"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def train(config: dict[str, Any]) -> None:
    validate_config(config)
    results_dir = Path(config["results_dir"])
    args = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--data-root",
        str(Path(config["data_root"])),
        "--results-dir",
        str(results_dir),
        "--holdout-count",
        str(int(config.get("holdout_count", 2))),
    ]
    run_command(args, env=env_from_config(config))

    if config.get("shape_live_model_pkl"):
        shape_config = results_dir / ".shape-live-training-config.json"
        write_json(shape_config, config)
        try:
            run_command([sys.executable, str(REPO_ROOT / "scripts" / "train_shape_live.py"),
                         "--config", str(shape_config), "--output", str(config["shape_live_model_pkl"]),
                         "--cache", str(results_dir / "shape_live_training_cache.pkl")], env=env_from_config(config))
        finally:
            shape_config.unlink(missing_ok=True)

    if config.get("opening_blend_file"):
        opening_config = results_dir / ".opening-blend-training-config.json"
        write_json(opening_config, config)
        try:
            run_command([sys.executable, str(REPO_ROOT / "scripts" / "train_opening_blend.py"),
                         "--config", str(opening_config), "--output", str(config["opening_blend_file"])],
                        env=env_from_config(config))
        finally:
            opening_config.unlink(missing_ok=True)


PREDICTION_RECORD_FIELDS = [
    "shape_live_version",
    "shape_live_prediction",
    "previous_chain_prediction",
    "archived_at",
    "archive_path",
    "source_output",
    "region",
    "event_id",
    "series_type",
    "event_series_type",
    "leaderboard_scope",
    "model_name",
    "character_id",
    "rank",
    "progress",
    "current_timestamp",
    "current_score",
    "end_timestamp",
    "model_prediction",
    "pressure_adjusted_prediction",
    "raw_model_prediction",
    "calibrated_model_prediction",
    "shape_first_prediction",
    "linear_projection",
    "pace_6h_projection",
    "jp_shape_projection",
    "hybrid_projection",
    "shape_envelope_anchor",
    "shape_envelope_early_anchor",
    "shape_envelope_opening_anchor",
    "shape_envelope_target",
    "shape_envelope_strength",
    "shape_envelope_mode",
    "shape_envelope_applied_prediction",
    "shape_envelope_signal_log_gap",
    "shape_envelope_signal_positive_share",
    "shape_envelope_signal_shape_model_gap",
    "shape_envelope_signal_pace_model_gap",
    "rank_boundary_multiplier",
    "boundary_pressure_multiplier",
    "wl2_character_prior_multiplier",
    "event_heat_prior_prediction",
    "event_heat_prior_log",
    "event_heat_prior_source",
    "event_heat_prior_count",
    "event_heat_prior_blend_weight",
    "early_heat_log",
    "early_heat_rank_strength",
    "early_unit_popularity_heat_log",
    "early_main_popularity_heat_log",
    "early_character_popularity_heat_log",
    "early_projection_target",
    "early_projection_strength",
    "world_bloom_total_weighted_progress",
    "world_bloom_total_projection_target",
    "world_bloom_total_projection_strength",
    "world_bloom_total_active_character_id",
    "world_bloom_total_active_chapter_progress",
    "world_bloom_total_chapter_sum_projection",
    "world_bloom_total_chapter_sum_conversion",
    "world_bloom_total_active_chapter_projection",
    "live_calibration_key",
    "live_calibration_multiplier",
    "event_finished_prediction",
    "synthetic_rank",
    "opening_prior_applied",
    "opening_prior_score",
    "opening_prior_weight",
    "opening_prior_event_count",
    "regional_shape_prediction",
    "regional_shape_weight",
    "regional_shape_event_count",
    "median_heat_heat",
    "median_heat_daily",
    "median_heat_slope",
    "median_heat_cumulative_heat",
    "median_heat_rhythm_log_residual",
    "median_heat_reliability",
    "median_heat_reference_count",
    "median_heat_prediction",
    "median_heat_blend_weight",
    "opening_blend_version",
    "opening_blend_prediction",
    "opening_blend_weight",
    "opening_blend_chain_prediction",
    "opening_blend_prior",
    "opening_blend_shape",
    "opening_blend_interpolated",
    "opening_blend_cross_server_heat",
]


def append_prediction_record_rows(
    config: dict[str, Any],
    job: dict[str, Any],
    payload: dict[str, Any],
    *,
    archive_path: Path,
    archived_at: int,
    source_output: Path,
) -> Path | None:
    if not bool(config.get("prediction_record_enabled", True)):
        return None
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        return None

    region = str(job["region"])
    event_id = int(job["event_id"])
    leaderboard_scope = str(job.get("leaderboard_scope") or payload.get("leaderboard_scope") or "total")
    path = prediction_record_dir(config) / region / str(event_id) / f"{leaderboard_scope}.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    fields = PREDICTION_RECORD_FIELDS
    if not write_header:
        with path.open("r", encoding="utf-8", newline="") as existing:
            fields = next(csv.reader(existing, delimiter="\t"))
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for item in predictions:
            if not isinstance(item, dict):
                continue
            writer.writerow({
                "archived_at": archived_at,
                "archive_path": str(archive_path),
                "source_output": str(source_output),
                "region": item.get("region", payload.get("region", region)),
                "event_id": item.get("event_id", payload.get("event_id", event_id)),
                "series_type": item.get("series_type", payload.get("event_series_type")),
                "event_series_type": item.get("event_series_type", payload.get("event_series_type")),
                "leaderboard_scope": item.get("leaderboard_scope", payload.get("leaderboard_scope", leaderboard_scope)),
                "model_name": payload.get("model_name"),
                "character_id": item.get("character_id"),
                "rank": item.get("rank"),
                "progress": item.get("progress"),
                "current_timestamp": item.get("current_timestamp"),
                "current_score": item.get("current_score"),
                "end_timestamp": item.get("end_timestamp"),
                "model_prediction": item.get("model_prediction"),
                "pressure_adjusted_prediction": item.get("pressure_adjusted_prediction"),
                "raw_model_prediction": item.get("raw_model_prediction"),
                "calibrated_model_prediction": item.get("calibrated_model_prediction"),
                "shape_first_prediction": item.get("shape_first_prediction"),
                "linear_projection": item.get("linear_projection"),
                "pace_6h_projection": item.get("pace_6h_projection"),
                "jp_shape_projection": item.get("jp_shape_projection"),
                "hybrid_projection": item.get("hybrid_projection"),
                "shape_envelope_anchor": item.get("shape_envelope_anchor"),
                "shape_envelope_early_anchor": item.get("shape_envelope_early_anchor"),
                "shape_envelope_opening_anchor": item.get("shape_envelope_opening_anchor"),
                "shape_envelope_target": item.get("shape_envelope_target"),
                "shape_envelope_strength": item.get("shape_envelope_strength"),
                "shape_envelope_mode": item.get("shape_envelope_mode"),
                "shape_envelope_applied_prediction": item.get("shape_envelope_applied_prediction"),
                "shape_envelope_signal_log_gap": item.get("shape_envelope_signal_log_gap"),
                "shape_envelope_signal_positive_share": item.get("shape_envelope_signal_positive_share"),
                "shape_envelope_signal_shape_model_gap": item.get("shape_envelope_signal_shape_model_gap"),
                "shape_envelope_signal_pace_model_gap": item.get("shape_envelope_signal_pace_model_gap"),
                "rank_boundary_multiplier": item.get("rank_boundary_multiplier"),
                "boundary_pressure_multiplier": item.get("boundary_pressure_multiplier"),
                "wl2_character_prior_multiplier": item.get("wl2_character_prior_multiplier"),
                "event_heat_prior_prediction": item.get("event_heat_prior_prediction"),
                "event_heat_prior_log": item.get("event_heat_prior_log"),
                "event_heat_prior_source": item.get("event_heat_prior_source"),
                "event_heat_prior_count": item.get("event_heat_prior_count"),
                "event_heat_prior_blend_weight": item.get("event_heat_prior_blend_weight"),
                "early_heat_log": item.get("early_heat_log"),
                "early_heat_rank_strength": item.get("early_heat_rank_strength"),
                "early_unit_popularity_heat_log": item.get("early_unit_popularity_heat_log"),
                "early_main_popularity_heat_log": item.get("early_main_popularity_heat_log"),
                "early_character_popularity_heat_log": item.get("early_character_popularity_heat_log"),
                "early_projection_target": item.get("early_projection_target"),
                "early_projection_strength": item.get("early_projection_strength"),
                "world_bloom_total_weighted_progress": item.get("world_bloom_total_weighted_progress"),
                "world_bloom_total_projection_target": item.get("world_bloom_total_projection_target"),
                "world_bloom_total_projection_strength": item.get("world_bloom_total_projection_strength"),
                "world_bloom_total_active_character_id": item.get("world_bloom_total_active_character_id"),
                "world_bloom_total_active_chapter_progress": item.get("world_bloom_total_active_chapter_progress"),
                "world_bloom_total_chapter_sum_projection": item.get("world_bloom_total_chapter_sum_projection"),
                "world_bloom_total_chapter_sum_conversion": item.get("world_bloom_total_chapter_sum_conversion"),
                "world_bloom_total_active_chapter_projection": item.get("world_bloom_total_active_chapter_projection"),
                "live_calibration_key": item.get("live_calibration_key"),
                "live_calibration_multiplier": item.get("live_calibration_multiplier"),
                "event_finished_prediction": item.get("event_finished_prediction"),
                "synthetic_rank": item.get("synthetic_rank"),
                "opening_prior_applied": item.get("opening_prior_applied"),
                "opening_prior_score": item.get("opening_prior_score"),
                "opening_prior_weight": item.get("opening_prior_weight"),
                "opening_prior_event_count": item.get("opening_prior_event_count"),
                "regional_shape_prediction": item.get("regional_shape_prediction"),
                "regional_shape_weight": item.get("regional_shape_weight"),
                "regional_shape_event_count": item.get("regional_shape_event_count"),
                **{key: item.get(key) for key in PREDICTION_RECORD_FIELDS if key.startswith("median_heat_")},
                **{key: item.get(key) for key in PREDICTION_RECORD_FIELDS if key.startswith("opening_blend_")},
            })
    return path


def archive_prediction_output(config: dict[str, Any], job: dict[str, Any], output: Path) -> Path | None:
    if not bool(config.get("prediction_archive_enabled", True)):
        return None
    if not output.exists() or output.stat().st_size == 0:
        return None
    region = str(job["region"])
    event_id = int(job["event_id"])
    leaderboard_scope = str(job.get("leaderboard_scope") or "total")
    archive_root = prediction_archive_dir(config) / region / str(event_id) / leaderboard_scope
    archive_root.mkdir(parents=True, exist_ok=True)
    payload = load_json(output)
    archived_at = int(time.time())
    payload.setdefault("archive", {})
    if isinstance(payload["archive"], dict):
        payload["archive"].update({
            "archived_at": archived_at,
            "source_output": str(output),
            "leaderboard_scope": leaderboard_scope,
        })
    destination = archive_root / f"{archived_at}.json"
    suffix = 1
    while destination.exists():
        destination = archive_root / f"{archived_at}_{suffix}.json"
        suffix += 1
    write_json(destination, payload)
    record_path = append_prediction_record_rows(
        config,
        job,
        payload,
        archive_path=destination,
        archived_at=archived_at,
        source_output=output,
    )
    if record_path is not None and isinstance(payload["archive"], dict):
        payload["archive"]["prediction_record"] = str(record_path)
        write_json(destination, payload)
    return destination


def predict(config: dict[str, Any], *, only_region: str | None = None) -> None:
    validate_config(config)
    model_pkl = Path(config["model_pkl"])
    current_sql_zst = config.get("current_sql_zst", {})
    for event in resolve_active_events(config):
        region = str(event["region"])
        if only_region is not None and region != only_region:
            continue
        for job in prediction_jobs(config, event):
            output = Path(job["output"])
            args = [
                sys.executable,
                str(PREDICT_SCRIPT),
                "--region",
                region,
                "--event-id",
                str(int(event["event_id"])),
                "--data-root",
                str(Path(config["data_root"])),
                "--model-pkl",
                str(model_pkl),
                "--leaderboard-scope",
                str(job["leaderboard_scope"]),
            ]
            calibration_path = live_calibration_path(config)
            if config.get('live_speed_enabled', False):
                args.extend(['--speed-api-base-url',str(config['event_tracker']['base_url'])])
            if calibration_path.exists():
                args.extend(["--live-calibration", str(calibration_path)])
            history_tsv = fetch_event_tracker_history(config, job)
            if history_tsv is not None:
                args.extend(["--history-tsv", str(history_tsv)])
            use_current_sql_zst = bool(event.get(
                "use_current_sql_zst",
                config.get("use_current_sql_zst", False),
            ))
            sql_path = current_sql_zst.get(region) if use_current_sql_zst else None
            if sql_path:
                args.extend(["--current-sql-zst", str(Path(sql_path))])
            run_command(args, stdout_path=output, env=env_from_config(config))
            archive_path = archive_prediction_output(config, job, output)
            if archive_path is not None:
                print(f"Archived prediction: {archive_path}", flush=True)


def finished_event_signature(config: dict[str, Any]) -> dict[str, list[int]]:
    os.environ.update(env_from_config(config))
    module = load_training_module()
    finished = module.load_finished_event_ids(
        now_ms=int((time.time() - int(config.get("retrain_after_event_end_delay_seconds", 3600))) * 1000)
    )
    return {region: sorted(event_ids) for region, event_ids in finished.items()}


def training_input_signature(config, finished):
    files = []
    root = Path(config["data_root"])
    for region in finished:
        files.extend((root / "regions" / region / "training").glob("*.csv*"))
        files.extend((root / "regions" / region / "raw" / "legacy").rglob("*.sql"))
    history_dir = Path(config.get("event_tracker", {}).get("history_dir", str(REPO_ROOT / "live_history")))
    for path in history_dir.glob("*_history.tsv"):
        parts = path.name.split("_")
        if len(parts) >= 3 and parts[1].isdigit() and int(parts[1]) in finished.get(parts[0], []):
            files.append(path)
    entries = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(files) if p.exists()]
    return hashlib.sha256(json.dumps(["live-lines-v1", entries]).encode()).hexdigest()


def maybe_train(config: dict[str, Any]) -> bool:
    validate_config(config)
    state_path = Path(config["state_file"])
    previous = load_json(state_path) if state_path.exists() else {}
    signature = finished_event_signature(config)
    now = int(time.time())
    input_signature = training_input_signature(config, signature)
    if previous.get("finished_events") == signature and previous.get("training_input_signature") == input_signature:
        print("No newly finished events; skip retrain.", flush=True)
        return False

    model_pkl = Path(config["model_pkl"])
    retry_after_seconds = int(config.get("failed_retrain_retry_after_seconds", 86400))
    failed_retrain = previous.get("failed_retrain")
    if (
        model_pkl.exists()
        and isinstance(failed_retrain, dict)
        and failed_retrain.get("finished_events") == signature
        and failed_retrain.get("training_input_signature") == input_signature
        and now - int(failed_retrain.get("updated_at", 0)) < retry_after_seconds
    ):
        retry_at = int(failed_retrain.get("updated_at", 0)) + retry_after_seconds
        print(
            "Previous retrain failed for the same finished-event signature; "
            f"keep existing model and retry after {retry_at}.",
            flush=True,
        )
        return False

    try:
        train(config)
    except subprocess.CalledProcessError as exc:
        if not model_pkl.exists():
            raise
        write_json(
            state_path,
            {
                **previous,
                "failed_retrain": {
                    "finished_events": signature,
                    "training_input_signature": input_signature,
                    "updated_at": now,
                    "returncode": exc.returncode,
                    "retry_after_seconds": retry_after_seconds,
                },
            },
        )
        print(
            "Retrain failed; keep existing model and continue maintenance. "
            f"Command returned {exc.returncode}.",
            flush=True,
        )
        return False

    write_json(
        state_path,
        {
            "finished_events": signature,
            "training_input_signature": input_signature,
            "updated_at": now,
            "failed_retrain": None,
        },
    )
    return True


def event_aggregate_at_ms(module, region: str, event_id: int) -> int | None:
    base = module.MASTERDATA_ROOTS.get(region)
    if base is None or not base.exists():
        return None
    for row in module.load_json(base / "events.json"):
        if int(row.get("id", -1)) != event_id:
            continue
        value = module.parse_int(row.get("aggregateAt"))
        return int(value) if value is not None else None
    return None


def event_finished_for_calibration(
    module,
    config: dict[str, Any],
    region: str,
    event_id: int,
    *,
    now_ms: int | None = None,
) -> bool:
    aggregate_at = event_aggregate_at_ms(module, region, event_id)
    if aggregate_at is None:
        return False
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    delay_ms = int(config.get("retrain_after_event_end_delay_seconds", 3600)) * 1000
    return aggregate_at + delay_ms <= now_ms


def _tree_manifest(root: Path) -> tuple[dict[str, tuple[int, str]], int]:
    files: dict[str, tuple[int, str]] = {}
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        files[relative] = (size, digest.hexdigest())
        total_bytes += size
    return files, total_bytes


def _latest_tree_mtime(root: Path) -> float:
    latest = root.stat().st_mtime
    for path in root.rglob("*"):
        try:
            latest = max(latest, path.stat().st_mtime)
        except FileNotFoundError:
            return time.time()
    return latest


def _append_cold_archive_manifest(cold_root: Path, payload: dict[str, Any]) -> None:
    manifest = cold_root / "migration-manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _archive_finished_event_dir(
    source: Path,
    destination: Path,
    *,
    cold_root: Path,
    region: str,
    event_id: int,
) -> dict[str, Any]:
    source_before, source_bytes = _tree_manifest(source)
    if not source_before:
        return {"region": region, "event_id": event_id, "status": "empty"}

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination_manifest, _ = _tree_manifest(destination)
        if destination_manifest != source_before:
            raise RuntimeError(f"cold archive destination differs: {destination}")
    else:
        temporary = destination.with_name(
            f".{destination.name}.migrating-{os.getpid()}-{int(time.time())}"
        )
        if temporary.exists():
            raise RuntimeError(f"cold archive temporary path exists: {temporary}")
        try:
            shutil.copytree(source, temporary, copy_function=shutil.copy2)
            copied_manifest, _ = _tree_manifest(temporary)
            source_after, _ = _tree_manifest(source)
            if source_after != source_before:
                raise RuntimeError(f"source changed during cold archive copy: {source}")
            if copied_manifest != source_before:
                raise RuntimeError(f"cold archive SHA-256 verification failed: {source}")
            temporary.rename(destination)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    backup = source.with_name(f".{source.name}.cold-archive-backup-{int(time.time())}")
    source.rename(backup)
    try:
        source.symlink_to(destination, target_is_directory=True)
        linked_manifest, _ = _tree_manifest(source)
        if linked_manifest != source_before:
            raise RuntimeError(f"cold archive symlink verification failed: {source}")
    except Exception:
        if source.is_symlink():
            source.unlink()
        backup.rename(source)
        raise
    shutil.rmtree(backup)

    payload = {
        "archived_at": int(time.time()),
        "region": region,
        "event_id": event_id,
        "source": str(source),
        "destination": str(destination),
        "file_count": len(source_before),
        "bytes": source_bytes,
        "tree_sha256": hashlib.sha256(
            "\n".join(
                f"{name}\t{size}\t{digest}"
                for name, (size, digest) in sorted(source_before.items())
            ).encode("utf-8")
        ).hexdigest(),
        "status": "archived",
    }
    _append_cold_archive_manifest(cold_root, payload)
    return payload


def _archive_history_file(source: Path, destination: Path, cold_root: Path, region: str, event_id: int) -> dict[str, Any]:
    before = (source.stat().st_size, hashlib.sha256(source.read_bytes()).hexdigest())
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.migrating-{os.getpid()}")
    if destination.exists():
        if (destination.stat().st_size, hashlib.sha256(destination.read_bytes()).hexdigest()) != before:
            raise RuntimeError(f"cold archive destination differs: {destination}")
    else:
        try:
            shutil.copy2(source, temporary)
            if (temporary.stat().st_size, hashlib.sha256(temporary.read_bytes()).hexdigest()) != before:
                raise RuntimeError("history copy checksum mismatch")
            if (source.stat().st_size, hashlib.sha256(source.read_bytes()).hexdigest()) != before:
                raise RuntimeError("history source changed during copy")
            temporary.rename(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    backup = source.with_name(f".{source.name}.cold-archive-backup-{os.getpid()}")
    source.rename(backup)
    try:
        source.symlink_to(destination)
        if hashlib.sha256(source.read_bytes()).hexdigest() != before[1]:
            raise RuntimeError("history symlink checksum mismatch")
    except Exception:
        if source.is_symlink():
            source.unlink()
        backup.rename(source)
        raise
    backup.unlink()
    result = dict(archived_at=int(time.time()), region=region, event_id=event_id,
                  source=str(source), destination=str(destination), bytes=before[0],
                  sha256=before[1], status="archived", kind="history")
    _append_cold_archive_manifest(cold_root, result)
    return result


def _archive_related_data(config, module, cold_root, active_events, now, quiet_seconds, archived, skipped, errors):
    records_root = prediction_record_dir(config)
    record_cold = Path(config.get("prediction_record_cold_archive_dir", str(cold_root.parent / "Haruki-Tracker-Prediction-Records")))
    history_root = Path(config.get("event_tracker", {}).get("history_dir", str(Path(config["results_dir"]).parent / "live_history")))
    history_cold = Path(config.get("history_cold_archive_dir", str(cold_root.parent / "Haruki-Tracker-History-Archive")))
    candidates = []
    if records_root.exists():
        for region in records_root.iterdir():
            if not region.is_dir() or region.is_symlink():
                continue
            for source in region.iterdir():
                if source.name.isdigit():
                    candidates.append((source, record_cold / region.name / source.name, record_cold, region.name, int(source.name), "records"))
    if history_root.exists():
        for source in history_root.glob("*_history.tsv"):
            parts = source.name.split("_")
            if len(parts) >= 3 and parts[1].isdigit():
                candidates.append((source, history_cold / parts[0] / parts[1] / source.name, history_cold, parts[0], int(parts[1]), "history"))
    for source, destination, archive_base, region, event_id, kind in candidates:
        if source.is_symlink():
            continue
        if (region, event_id) in active_events or not event_finished_for_calibration(module, config, region, event_id):
            continue
        modified = _latest_tree_mtime(source) if source.is_dir() else source.stat().st_mtime
        if modified > now - quiet_seconds:
            continue
        # Related destinations must live on the same verified WD volume.
        required_mount = Path(config.get("prediction_cold_archive_mount", cold_root.parent))
        if not archive_base.resolve().is_relative_to(required_mount.resolve()):
            errors.append(dict(source=str(source), error="related archive is outside configured volume"))
            continue
        try:
            if kind == "records":
                item = _archive_finished_event_dir(source, destination, cold_root=archive_base, region=region, event_id=event_id)
                item["kind"] = kind
            else:
                item = _archive_history_file(source, destination, archive_base, region, event_id)
            archived.append(item)
        except Exception as exc:
            errors.append(dict(source=str(source), destination=str(destination), error=str(exc)))


def archive_finished_prediction_events(config: dict[str, Any]) -> dict[str, Any]:
    if not bool(config.get("prediction_cold_archive_enabled", False)):
        return {"status": "disabled", "archived": [], "skipped": []}
    cold_root = prediction_cold_archive_dir(config)
    if cold_root is None:
        return {"status": "not_configured", "archived": [], "skipped": []}

    default_mount = (
        f"/Volumes/{cold_root.parts[2]}"
        if len(cold_root.parts) > 2 and cold_root.parts[1] == "Volumes"
        else "/"
    )
    required_mount = Path(config.get("prediction_cold_archive_mount", default_mount))
    if bool(config.get("prediction_cold_archive_require_mount", True)) and not required_mount.is_mount():
        result = {
            "status": "mount_missing",
            "required_mount": str(required_mount),
            "archived": [],
            "skipped": [],
        }
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result

    archive_root = prediction_archive_dir(config)
    if not archive_root.exists():
        return {"status": "source_missing", "archived": [], "skipped": []}
    cold_root.mkdir(parents=True, exist_ok=True)

    os.environ.update(env_from_config(config))
    module = load_training_module()
    active_events = {
        (str(event["region"]), int(event["event_id"]))
        for event in resolve_active_events(config)
    }
    quiet_seconds = int(config.get(
        "prediction_cold_archive_quiet_seconds",
        config.get("retrain_after_event_end_delay_seconds", 3600),
    ))
    now = time.time()
    lock_path = Path(config.get(
        "prediction_cold_archive_lock_file",
        str(Path(config["state_file"]).with_name("prediction-cold-archive.lock")),
    ))
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    archived: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        for region_dir in sorted(archive_root.iterdir()):
            if not region_dir.is_dir() or region_dir.is_symlink():
                continue
            region = region_dir.name
            for source in sorted(region_dir.iterdir()):
                if source.is_symlink():
                    skipped.append({"region": region, "event_id": source.name, "reason": "already_archived"})
                    continue
                if not source.is_dir():
                    continue
                try:
                    event_id = int(source.name)
                except ValueError:
                    skipped.append({"region": region, "event_id": source.name, "reason": "invalid_event_id"})
                    continue
                if (region, event_id) in active_events:
                    skipped.append({"region": region, "event_id": event_id, "reason": "active"})
                    continue
                if not event_finished_for_calibration(module, config, region, event_id):
                    skipped.append({"region": region, "event_id": event_id, "reason": "not_finished"})
                    continue
                if _latest_tree_mtime(source) > now - quiet_seconds:
                    skipped.append({"region": region, "event_id": event_id, "reason": "not_quiet"})
                    continue
                destination = cold_root / region / str(event_id)
                try:
                    archived.append(_archive_finished_event_dir(
                        source,
                        destination,
                        cold_root=cold_root,
                        region=region,
                        event_id=event_id,
                    ))
                except Exception as exc:  # noqa: BLE001 - one bad event must not block maintenance.
                    errors.append({
                        "region": region,
                        "event_id": event_id,
                        "source": str(source),
                        "destination": str(destination),
                        "error": str(exc),
                    })
        if config.get("related_cold_archive_enabled", False):
            _archive_related_data(config, module, cold_root, active_events, now, quiet_seconds, archived, skipped, errors)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    result = {
        "status": "partial" if errors else "ok",
        "archived": archived,
        "skipped": skipped,
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def row_prediction_value(row: dict[str, Any]) -> float | None:
    for key in (
        "model_prediction",
        "prediction",
        "pressure_adjusted_prediction",
        "shape_first_prediction",
        "calibrated_model_prediction",
    ):
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


def final_scores_from_prediction(payload: dict[str, Any]) -> dict[tuple[int, int], float]:
    finals: dict[tuple[int, int], float] = {}
    for row in payload.get("predictions", []):
        if not isinstance(row, dict):
            continue
        rank = row.get("rank")
        if rank is None:
            continue
        character_id = int(row.get("character_id") or 0)
        current_score = row.get("current_score")
        value = float(current_score) if current_score is not None else row_prediction_value(row)
        if value is None or value <= 0:
            continue
        finals[(character_id, int(rank))] = value
    return finals


def final_scores_from_history_tsv(path: Path | None) -> dict[tuple[int, int], float]:
    if path is None or not path.exists():
        return {}
    latest: dict[tuple[int, int], tuple[int, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw_line in handle:
            row = raw_line.rstrip("\n").split("\t")
            if len(row) >= 4:
                character_id = int(row[0])
                rank = int(row[1])
                timestamp = int(row[2])
                score = float(row[3])
            elif len(row) >= 3:
                character_id = 0
                rank = int(row[0])
                timestamp = int(row[1])
                score = float(row[2])
            else:
                continue
            key = (character_id, rank)
            if key not in latest or timestamp >= latest[key][0]:
                latest[key] = (timestamp, score)
    return {
        key: score
        for key, (_timestamp, score) in latest.items()
        if score > 0
    }


def load_calibration_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": 1,
            "event_groups": [],
            "calibrator": {},
        }
    payload = load_json(path)
    if not isinstance(payload.get("event_groups"), list):
        payload["event_groups"] = []
    if not isinstance(payload.get("calibrator"), dict):
        payload["calibrator"] = {}
    return payload


def weighted_median(values: list[tuple[float, int]]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values, key=lambda item: item[0])
    total = sum(max(int(weight), 0) for _value, weight in ordered)
    if total <= 0:
        return float(statistics.median(value for value, _weight in ordered))
    midpoint = total / 2.0
    cumulative = 0
    for value, weight in ordered:
        cumulative += max(int(weight), 0)
        if cumulative >= midpoint:
            return float(value)
    return float(ordered[-1][0])


def rebuild_live_calibrator(document: dict[str, Any], settings: dict[str, float]) -> dict[str, Any]:
    grouped: dict[str, list[tuple[float, int, str]]] = defaultdict(list)
    for item in document.get("event_groups", []):
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if not isinstance(key, str):
            continue
        count = int(item.get("count") or 0)
        if count <= 0:
            continue
        grouped[key].append((
            float(item.get("median_log_ratio") or 0.0),
            count,
            str(item.get("event_key") or ""),
        ))

    calibrator: dict[str, dict[str, Any]] = {}
    max_abs_sample = float(settings["max_abs_sample_log_ratio"])
    for key, values in grouped.items():
        median = weighted_median([(value, count) for value, count, _event_key in values])
        median = max(min(median, max_abs_sample), -max_abs_sample)
        calibrator[key] = {
            "count": int(sum(count for _value, count, _event_key in values)),
            "event_count": int(len({event_key for _value, _count, event_key in values if event_key})),
            "median_log_ratio": median,
            "event_median_log_ratio": max(min(statistics.median(value for value, _count, _key in values), max_abs_sample), -max_abs_sample),
        }
    return calibrator


def build_finished_event_groups(
    module,
    config: dict[str, Any],
    job: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    region = str(job["region"])
    event_id = int(job["event_id"])
    leaderboard_scope = str(job.get("leaderboard_scope") or "total")
    event_key = f"{region}:{event_id}:{leaderboard_scope}"
    output = Path(job["output"])
    final_payload = load_json(output) if output.exists() else {
        "region": region,
        "event_id": event_id,
        "leaderboard_scope": leaderboard_scope,
    }
    history_tsv = None
    try:
        history_tsv = fetch_event_tracker_history(config, job)
    except Exception as exc:  # noqa: BLE001 - calibration should fall back to the latest prediction output.
        print(f"Failed to refresh final history for {event_key}: {exc}", flush=True)
    finals = final_scores_from_history_tsv(history_tsv)
    if not finals:
        finals = final_scores_from_prediction(final_payload)
    if not finals:
        return event_key, [], {"error": "no final scores found in prediction output"}

    settings = live_calibration_settings(config)
    archive_root = prediction_archive_dir(config) / region / str(event_id) / leaderboard_scope
    if not archive_root.exists() and leaderboard_scope == "chapter":
        archive_root = prediction_archive_dir(config) / region / str(event_id)
    if not archive_root.exists():
        return event_key, [], {"error": f"archive directory missing: {archive_root}"}

    grouped: dict[str, list[float]] = defaultdict(list)
    sample_count = 0
    archive_count = 0
    min_progress = float(settings["min_progress"])
    max_progress = float(settings["max_progress"])
    max_abs_sample = float(settings["max_abs_sample_log_ratio"])
    for archive_path in sorted(archive_root.glob("*.json")):
        try:
            payload = load_json(archive_path)
        except json.JSONDecodeError:
            continue
        archive_count += 1
        for row in payload.get("predictions", []):
            if not isinstance(row, dict) or bool(row.get("event_finished_prediction")):
                continue
            rank = row.get("rank")
            if rank is None:
                continue
            character_id = int(row.get("character_id") or 0)
            actual = finals.get((character_id, int(rank)))
            prediction = row_prediction_value(row)
            if actual is None or prediction is None or actual <= 0.0 or prediction <= 0.0:
                continue
            progress = float(row.get("progress") or 0.0)
            if progress < min_progress or progress > max_progress:
                continue
            current_timestamp = int(row.get("current_timestamp") or row.get("timestamp") or 0)
            end_timestamp = int(row.get("end_timestamp") or 0)
            if end_timestamp > 0 and current_timestamp >= end_timestamp:
                continue
            log_ratio = max(min(math.log(actual / prediction), max_abs_sample), -max_abs_sample)
            calibration_row = {
                "region": region,
                "event_id": event_id,
                "series_type": str(row.get("series_type") or final_payload.get("series_type") or "normal"),
                "leaderboard_scope": str(row.get("leaderboard_scope") or final_payload.get("leaderboard_scope") or leaderboard_scope),
                "character_id": character_id,
                "progress": progress,
                "rank": int(rank),
                "current_timestamp": current_timestamp or None,
                "end_timestamp": end_timestamp or None,
                "current_calendar_segment": row.get("current_calendar_segment"),
                "end_calendar_segment": row.get("end_calendar_segment"),
            }
            for key in module.calibration_keys(calibration_row):
                grouped[encode_calibration_key(key)].append(log_ratio)
            sample_count += 1

    event_groups = [
        {
            "event_key": event_key,
            "key": key,
            "count": len(values),
            "median_log_ratio": float(statistics.median(values)),
        }
        for key, values in sorted(grouped.items())
        if values
    ]
    return event_key, event_groups, {
        "archive_count": archive_count,
        "final_lines": len(finals),
        "sample_count": sample_count,
        "group_count": len(event_groups),
    }


def calibrate_finished_events(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    validate_config(config)
    os.environ.update(env_from_config(config))
    module = load_training_module()
    settings = live_calibration_settings(config)
    path = live_calibration_path(config)
    document = load_calibration_document(path)
    previous_groups = [
        item for item in document.get("event_groups", [])
        if isinstance(item, dict) and isinstance(item.get("event_key"), str)
    ]
    existing_event_keys = {str(item["event_key"]) for item in previous_groups}

    summaries: list[dict[str, Any]] = []
    current_groups = previous_groups
    for event in resolve_active_events(config):
        region = str(event["region"])
        event_id = int(event["event_id"])
        if not event_finished_for_calibration(module, config, region, event_id):
            summaries.append({"event_key": f"{region}:{event_id}", "status": "active"})
            continue
        for job in prediction_jobs(config, event):
            event_key = f"{region}:{event_id}:{job['leaderboard_scope']}"
            if event_key in existing_event_keys and not force:
                summaries.append({"event_key": event_key, "status": "already_calibrated"})
                continue
            event_key, event_groups, summary = build_finished_event_groups(module, config, job)
            if not event_groups:
                summaries.append({"event_key": event_key, "status": "skipped", **summary})
                continue
            current_groups = [
                item for item in current_groups
                if str(item.get("event_key")) != event_key
            ]
            current_groups.extend(event_groups)
            summaries.append({"event_key": event_key, "status": "calibrated", **summary})

    document = {
        "version": 1,
        "updated_at": int(time.time()),
        "settings": settings,
        "event_groups": current_groups,
        "calibrator": rebuild_live_calibrator({"event_groups": current_groups}, settings),
        "events": summaries,
    }
    write_json(path, document)
    print(json.dumps({
        "live_calibration_file": str(path),
        "calibrator_keys": len(document["calibrator"]),
        "event_groups": len(current_groups),
        "events": summaries,
    }, ensure_ascii=False, indent=2), flush=True)
    return document


def world_bloom_validation_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("world_bloom_validation", {})


def csv_or_list(value: Any, default: list[Any]) -> list[Any]:
    if value is None:
        return default
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return value
    return list(value)


def event_tracker_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("event_tracker", {})


def fetch_event_tracker_history(config: dict[str, Any], event: dict[str, Any]) -> Path | None:
    tracker = event_tracker_config(config)
    if not bool(tracker.get("enabled", False)):
        return None
    base_url = tracker.get("base_url")
    if not base_url:
        raise SystemExit("event_tracker.base_url is required when event_tracker.enabled=true")

    region = str(event["region"])
    event_id = int(event["event_id"])
    source_event = event.get("event") if isinstance(event.get("event"), dict) else event
    leaderboard_scope = str(event.get("leaderboard_scope") or source_event.get("leaderboard_scope") or "auto")
    history_dir = Path(tracker.get("history_dir", str(Path(config["results_dir"]) / "live_history")))
    suffix = "" if leaderboard_scope in {"auto", "chapter"} else f"_{leaderboard_scope}"
    output = history_dir / f"{region}_{event_id}{suffix}_history.tsv"
    args = [
        sys.executable,
        str(FETCH_EVENT_TRACKER_HISTORY_SCRIPT),
        "--base-url",
        str(base_url),
        "--region",
        region,
        "--event-id",
        str(event_id),
        "--data-root",
        str(Path(config["data_root"])),
        "--output",
        str(output),
        "--leaderboard-scope",
        leaderboard_scope,
    ]
    ranks = event.get("ranks") or source_event.get("ranks") or tracker.get("ranks")
    if ranks:
        args.extend(["--ranks", ",".join(str(rank) for rank in csv_or_list(ranks, []))])
    characters = event.get("characters") or source_event.get("characters")
    if characters:
        args.extend(["--characters", ",".join(str(character) for character in csv_or_list(characters, []))])
    series_type = event.get("series_type") or source_event.get("series_type") or tracker.get("series_type")
    if series_type:
        args.extend(["--series-type", str(series_type)])
    timeout = tracker.get("timeout_seconds")
    if timeout is not None:
        args.extend(["--timeout", str(float(timeout))])
    min_points = tracker.get("min_points")
    if min_points is not None:
        args.extend(["--min-points", str(int(min_points))])
    run_command(args, env=env_from_config(config))
    if not output.exists() or output.stat().st_size == 0:
        print(
            f"EventTracker returned no usable rows for {region} event {event_id}; falling back to current SQL dump.",
            flush=True,
        )
        return None
    return output


def validate_world_bloom_cache(config: dict[str, Any]) -> None:
    validate_config(config)
    validation = world_bloom_validation_config(config)
    output = Path(validation.get(
        "output",
        str(Path(config["results_dir"]) / "world_bloom_region_validation.json"),
    ))
    regions = csv_or_list(validation.get("regions"), ["en", "tw", "kr"])
    ranks = csv_or_list(validation.get("ranks"), [100, 1000])
    args = [
        sys.executable,
        str(VALIDATE_WORLD_BLOOM_SCRIPT),
        "--config",
        str(Path(config.get("_config_path", REPO_ROOT / "config" / "forecast.production.json"))),
        "--data-root",
        str(Path(config["data_root"])),
        "--regions",
        ",".join(str(region) for region in regions),
        "--ranks",
        ",".join(str(rank) for rank in ranks),
        "--output",
        str(output),
    ]
    min_final_progress = validation.get("min_final_progress")
    if min_final_progress is not None:
        args.extend(["--min-final-progress", str(float(min_final_progress))])
    if bool(validation.get("allow_same_event_training", False)):
        args.append("--allow-same-event-training")
    run_command(args, env=env_from_config(config))


def fetch_world_bloom_cache(config: dict[str, Any], *, regions: str | None, events: str | None, ranks: str | None) -> None:
    validate_config(config)
    required = config.get("required_local_paths", {})
    output_dir = required.get("world_bloom_sekai_best_cache_dir")
    if not output_dir:
        raise SystemExit("required_local_paths.world_bloom_sekai_best_cache_dir is required for fetch-wl-cache")
    validation = world_bloom_validation_config(config)
    args = [
        sys.executable,
        str(FETCH_WORLD_BLOOM_SCRIPT),
        "--output-dir",
        str(Path(output_dir)),
        "--regions",
        regions or ",".join(str(region) for region in csv_or_list(validation.get("regions"), ["en", "tw", "kr"])),
        "--events",
        events or "finished",
        "--ranks",
        ranks or ",".join(str(rank) for rank in csv_or_list(validation.get("ranks"), [100, 1000])),
    ]
    run_command(args, env=env_from_config(config))


def main() -> int:
    parser = argparse.ArgumentParser(description="Production helpers for Haruki tracker forecasts.")
    parser.add_argument(
        "command",
        choices=(
            "validate-config",
            "fetch-wl-cache",
            "validate-wl-cache",
            "train",
            "predict",
            "maybe-train",
            "calibrate-ended",
            "archive-ended",
            "maintain",
            "maybe-train-and-predict",
            "train-and-predict",
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "forecast.production.json",
        help="Production JSON config. Copy config/forecast.production.example.json first.",
    )
    parser.add_argument("--region", default=None, help="Optional region filter for predict.")
    parser.add_argument("--wl-regions", default=None, help="Optional comma-separated regions for fetch-wl-cache.")
    parser.add_argument("--wl-events", default=None, help="Optional event ids or 'finished' for fetch-wl-cache.")
    parser.add_argument("--wl-ranks", default=None, help="Optional comma-separated ranks for fetch-wl-cache.")
    parser.add_argument("--force-calibration", action="store_true", help="Rebuild calibration groups for finished active events.")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_json(config_path)
    config["_config_path"] = str(config_path)
    if args.command == "validate-config":
        validate_config(config)
    elif args.command == "fetch-wl-cache":
        fetch_world_bloom_cache(
            config,
            regions=args.wl_regions,
            events=args.wl_events,
            ranks=args.wl_ranks,
        )
    elif args.command == "validate-wl-cache":
        validate_world_bloom_cache(config)
    elif args.command == "train":
        train(config)
    elif args.command == "predict":
        predict(config, only_region=args.region)
    elif args.command == "maybe-train":
        maybe_train(config)
    elif args.command == "calibrate-ended":
        calibrate_finished_events(config, force=args.force_calibration)
    elif args.command == "archive-ended":
        archive_finished_prediction_events(config)
    elif args.command == "maintain":
        maybe_train(config)
        calibrate_finished_events(config, force=args.force_calibration)
        archive_finished_prediction_events(config)
    elif args.command == "maybe-train-and-predict":
        maybe_train(config)
        calibrate_finished_events(config, force=args.force_calibration)
        archive_finished_prediction_events(config)
        predict(config, only_region=args.region)
    elif args.command == "train-and-predict":
        train(config)
        predict(config, only_region=args.region)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
