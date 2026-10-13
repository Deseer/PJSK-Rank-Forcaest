#!/usr/bin/env python3

from __future__ import annotations

import argparse
import bisect
import csv
import io
import importlib.util
import json
import math
import os
import pickle
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from adaptive_forecast import trend_diagnostics


LIVE_PRESSURE_RANKS = frozenset((
    100,
    200,
    300,
    400,
    500,
    1000,
    2000,
    3000,
    4000,
    5000,
    10000,
    100000,
))


PRESSURE_CALIBRATION_BASE = {
    100: {"max_extra": 0.22, "tail_extra": 0.04, "cap": 1.24},
    200: {"max_extra": 0.07, "tail_extra": 0.02, "cap": 1.10},
    300: {"max_extra": 0.06, "tail_extra": 0.018, "cap": 1.09},
    400: {"max_extra": 0.05, "tail_extra": 0.016, "cap": 1.08},
    500: {"max_extra": 0.045, "tail_extra": 0.014, "cap": 1.07},
    1000: {"max_extra": 0.075, "tail_extra": 0.025, "cap": 1.12},
    2000: {"max_extra": 0.04, "tail_extra": 0.014, "cap": 1.07},
    3000: {"max_extra": 0.03, "tail_extra": 0.012, "cap": 1.055},
    4000: {"max_extra": 0.028, "tail_extra": 0.01, "cap": 1.05},
    5000: {"max_extra": 0.022, "tail_extra": 0.008, "cap": 1.04},
    10000: {"max_extra": 0.014, "tail_extra": 0.006, "cap": 1.03},
    100000: {"max_extra": 0.008, "tail_extra": 0.004, "cap": 1.02},
}

PRESSURE_REGION_THRESHOLDS = {
    "jp": {
        100: (12.0, 26.0, 22.73, 57.0),
        200: (26.0, 50.0, 74.0, 135.0),
        300: (34.0, 58.0, 100.0, 168.0),
        400: (42.0, 62.0, 124.0, 189.0),
        500: (45.0, 64.0, 142.0, 203.0),
        1000: (57.0, 69.0, 184.0, 242.0),
        2000: (68.0, 72.0, 241.0, 269.0),
        3000: (70.0, 72.0, 264.0, 277.0),
        4000: (72.0, 72.0, 273.0, 281.0),
        5000: (72.0, 72.0, 277.0, 283.0),
        10000: (72.0, 72.0, 283.0, 286.0),
        100000: (72.0, 73.0, 288.0, 288.0),
    },
    "en": {
        100: (5.0, 23.0, 14.5, 41.0),
        200: (24.0, 48.0, 70.0, 139.0),
        300: (34.0, 57.0, 101.0, 176.0),
        400: (43.0, 64.0, 132.0, 202.6),
        500: (50.0, 66.0, 154.0, 221.0),
        1000: (65.0, 71.0, 219.0, 259.0),
        2000: (70.0, 72.0, 260.0, 273.0),
        3000: (72.0, 72.0, 270.0, 279.0),
        4000: (72.0, 72.0, 275.0, 281.0),
        5000: (72.0, 72.0, 276.0, 283.0),
        10000: (72.0, 72.0, 281.0, 285.0),
        100000: (72.0, 72.0, 287.0, 287.0),
    },
    "tw": {
        100: (6.5, 28.0, 21.0, 57.0),
        200: (27.0, 49.0, 78.0, 135.0),
        300: (38.0, 58.0, 115.0, 172.0),
        400: (46.0, 64.0, 141.0, 200.0),
        500: (52.0, 67.0, 159.0, 216.0),
        1000: (66.0, 71.0, 219.0, 256.0),
        2000: (70.0, 72.0, 255.0, 274.0),
        3000: (71.0, 72.0, 269.0, 278.0),
        4000: (72.0, 72.0, 273.0, 281.0),
        5000: (72.0, 72.0, 275.0, 282.0),
        10000: (72.0, 72.0, 280.0, 284.0),
    },
    "kr": {
        100: (3.73, 24.0, 13.0, 45.0),
        200: (25.0, 50.0, 72.0, 134.0),
        300: (38.0, 57.0, 111.0, 171.0),
        400: (48.0, 63.0, 140.0, 199.0),
        500: (53.0, 67.0, 162.0, 218.0),
        1000: (65.0, 71.0, 214.0, 253.0),
        2000: (70.0, 72.0, 252.0, 270.0),
        3000: (71.0, 72.0, 264.0, 277.0),
        4000: (71.0, 72.0, 270.0, 280.0),
        5000: (72.0, 72.0, 273.0, 282.0),
        10000: (72.0, 72.0, 277.0, 284.0),
        100000: (72.0, 72.0, 285.0, 287.0),
    },
    "cn": {
        100: (15.0, 34.64, 28.0, 97.0),
        200: (27.0, 49.0, 75.0, 132.0),
        300: (35.0, 58.0, 103.0, 166.0),
        400: (43.0, 62.0, 125.0, 186.0),
        500: (48.0, 65.0, 138.0, 204.0),
        1000: (62.0, 70.0, 195.0, 242.0),
        2000: (68.0, 71.0, 236.0, 269.0),
        3000: (71.0, 72.0, 261.0, 279.0),
        4000: (72.0, 72.0, 273.0, 282.0),
        5000: (72.0, 72.0, 276.0, 283.0),
        10000: (72.0, 72.0, 283.0, 286.0),
        100000: (72.0, 73.0, 288.0, 288.0),
    },
}

WORLD_BLOOM_PRESSURE_EXTRA_SCALE = 0.65
WORLD_BLOOM_SHAPE_HYBRID_WEIGHT = 0.9
WORLD_BLOOM_TOTAL_SERIES_TYPE = "world_bloom_total"
WORLD_BLOOM_TOTAL_SHAPE_WEIGHT = 0.88
WORLD_BLOOM_TOTAL_LINEAR_WEIGHT = 0.10
WORLD_BLOOM_TOTAL_EARLY_STRENGTH = 1.08
WORLD_BLOOM_TOTAL_STRENGTH_DECAY_PROGRESS = 0.55
WORLD_BLOOM_TOTAL_SUPPORT_DECAY_PROGRESS = 0.92
WORLD_BLOOM_TOTAL_SUPPORT_STRENGTH = 0.72
WORLD_BLOOM_TOTAL_TARGET_FLOOR_MIN = 0.84
WORLD_BLOOM_TOTAL_TARGET_FLOOR_MAX = 0.96
TAIL_PACE_GUARD_SECONDS = 3 * 3600
TAIL_PACE_GUARD_FORCE_SECONDS = 3600
TAIL_PACE_VISIBLE_UPLIFT = 0.015
LATE_PROGRESS_GUARD_START = 0.85
LATE_PROGRESS_GUARD_REGION_START = {
    "jp": 0.95,
    "en": 0.95,
}
CN_170_WL_OPENING_EVENT_ID = 170
CN_170_WL_OPENING_PROGRESS_END = 0.05
FRONT_PARKING_MIN_PROGRESS = 0.35
FRONT_PARKING_MAX_PROGRESS = 0.85
MID_STAGE_SHAPE_FLOOR_START = 0.16
MID_STAGE_SHAPE_FLOOR_END = 0.40
MID_STAGE_SHAPE_FLOOR_TRIGGER = 1.28
SHAPE_ENVELOPE_SIGNAL_RANKS = frozenset({10, 20, 30, 50, 100, 200, 500, 1000, 5000, 10000})
SHAPE_ENVELOPE_SIGNAL_GAP = math.log(1.08)
SHAPE_ENVELOPE_SIGNAL_STRONG_GAP = math.log(1.16)
SHAPE_ENVELOPE_SIGNAL_SHARE = 0.58
SHAPE_ENVELOPE_SIGNAL_STRONG_SHARE = 0.70
SHAPE_ENVELOPE_MID_STRONG_GAP = math.log(1.34)
SHAPE_ENVELOPE_MID_PACE_SUPPORT = math.log(0.92)
SHAPE_ENVELOPE_SHAPE_SUPPORT_GAP = math.log(1.22)
LIVE_CALIBRATION_DEFAULT_MIN_COUNT = 8
LIVE_CALIBRATION_DEFAULT_SHRINKAGE = 24.0
LIVE_CALIBRATION_DEFAULT_MAX_ABS_LOG_ADJUSTMENT = 0.08
EARLY_PRIOR_FULL_PROGRESS = 0.015
EARLY_PRIOR_DECAY_PROGRESS = 0.18
EARLY_HEAT_FULL_PROGRESS = 0.05
EARLY_HEAT_DECAY_PROGRESS = 0.10
EARLY_PRIOR_MAX_POSITIVE_LOG_ADJUSTMENT = 0.32
EARLY_PRIOR_MAX_NEGATIVE_LOG_ADJUSTMENT = -0.18
EARLY_PACE_TRUST_FULL_PROGRESS = 0.18
EARLY_HEAT_SCALE = float(os.environ.get("HARUKI_FORECAST_EARLY_HEAT_SCALE", "1.0"))
EARLY_HEAT_SCALE_BY_REGION = {
    item.split(":", 1)[0].strip(): float(item.split(":", 1)[1])
    for item in os.environ.get("HARUKI_FORECAST_EARLY_HEAT_SCALE_BY_REGION", "").split(",")
    if ":" in item and item.split(":", 1)[0].strip()
}
POPULARITY_PRIOR_LIVE_WEIGHT_BY_CONDITION = {
    "world_bloom": 0.42,
    "box": 0.42,
    "mixed": 0.40,
    "other": 0.34,
}
UNIT_PRIOR_LIVE_WEIGHT_BY_CONDITION = {
    "box": 0.46,
    "mixed": 0.34,
    "other": 0.34,
    "world_bloom": 0.0,
}
NORMAL_CHARACTER_POPULARITY_PRIOR = {
    # Normal events need a weak popularity anchor because short regional
    # histories can confuse event strength with numerical inflation.
    1: -0.020,  # Ichika
    2: 0.010,   # Saki
    3: -0.055,  # Honami
    4: 0.030,   # Shiho
    5: -0.010,  # Minori
    6: 0.010,   # Haruka
    7: -0.005,  # Airi
    8: -0.015,  # Shizuku
    9: 0.025,   # Kohane
    10: -0.055, # An
    11: 0.050,  # Akito
    12: 0.060,  # Toya
    13: 0.075,  # Tsukasa
    14: 0.018,  # Emu
    15: 0.030,  # Nene
    16: 0.075,  # Rui
    17: 0.070,  # Kanade
    18: 0.045,  # Mafuyu
    19: 0.065,  # Ena
    20: 0.080,  # Mizuki
}


def character_popularity_prior_for_live(module, condition: str, character_id: int) -> float:
    character_id = int(character_id)
    if character_id <= 0:
        return 0.0
    if condition == "world_bloom":
        return float(module.CHARACTER_POPULARITY_PRIOR.get(character_id, 0.0))
    if character_id in module.VOCALOID_CHARACTER_IDS:
        return 0.0
    return float(NORMAL_CHARACTER_POPULARITY_PRIOR.get(character_id, 0.0))


def load_training_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("train_baseline_module", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_baseline_module"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def encode_calibration_key(key: tuple[object, ...]) -> str:
    return json.dumps(list(key), ensure_ascii=False, separators=(",", ":"))


def load_live_calibration(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {}
    # Existing calibration files predate the equal-event statistic. Derive it
    # in memory so deployment does not need to rewrite production state.
    event_values: dict[str, dict[str, float]] = {}
    for group in payload.get("event_groups", []):
        if isinstance(group, dict) and group.get("key") and group.get("event_key"):
            event_values.setdefault(group["key"], {})[group["event_key"]] = float(group.get("median_log_ratio") or 0)
    for key, values in event_values.items():
        stats = payload.get("calibrator", {}).get(key)
        if isinstance(stats, dict) and values:
            stats["event_median_log_ratio"] = statistics.median(values.values())
            stats["event_count"] = len(values)
    return payload


def live_calibration_multiplier(
    module,
    row: dict[str, object],
    calibration: dict[str, object],
) -> tuple[float, str | None, float | None]:
    calibrator = calibration.get("calibrator")
    if not isinstance(calibrator, dict):
        return 1.0, None, None
    settings = calibration.get("settings")
    if not isinstance(settings, dict):
        settings = {}
    min_count = int(settings.get("min_count", LIVE_CALIBRATION_DEFAULT_MIN_COUNT))
    shrinkage = float(settings.get("shrinkage", LIVE_CALIBRATION_DEFAULT_SHRINKAGE))
    max_abs = float(settings.get(
        "max_abs_log_adjustment",
        LIVE_CALIBRATION_DEFAULT_MAX_ABS_LOG_ADJUSTMENT,
    ))

    use_v2 = use_increment_calibration(row)
    # Online corrections must retain region, series and stage. Character-only
    # fallbacks can otherwise apply one foreign opening curve to a closing rank.
    allowed_keys = {"region_series_character_progress_rank", "region_series_progress_rank",
                    "region_series_progress"}
    for key in module.calibration_keys(row):
        if use_v2 and key[0] not in allowed_keys:
            continue
        stats = calibrator.get(encode_calibration_key(key))
        if not isinstance(stats, dict):
            continue
        count = int(float(stats.get("count", 0)))
        if count < min_count:
            continue
        if use_v2:
            event_count = int(stats.get("event_count", 0))
            if event_count < int(settings.get("min_event_count", 3)):
                continue
            shrink = event_count / (event_count + float(settings.get("event_shrinkage", 6.0)))
            median = float(stats.get("event_median_log_ratio", stats.get("median_log_ratio", 0.0)))
        else:
            shrink = count / (count + shrinkage)
            median = float(stats.get("median_log_ratio", 0.0))
        log_adjustment = median * shrink
        log_adjustment = max(min(log_adjustment, max_abs), -max_abs)
        return math.exp(log_adjustment), str(key[0]), log_adjustment
    return 1.0, None, None


def apply_live_calibration(prediction: float, current_score: float, factor: float) -> float:
    """Calibrate only the unobserved increment; never discount earned score."""
    return float(current_score) + max(float(prediction) - float(current_score), 0.0) * factor


def use_increment_calibration(row: dict[str, object]) -> bool:
    # Replay supports all-region WL closing stages; earlier stages regressed
    # in several regions and stay on the published policy for now.
    return row.get("series_type") == "world_bloom" and float(row.get("progress") or 0) >= 0.85


def cn_170_wl_opening_rank_boost(rank: int, progress: float) -> float:
    if progress >= CN_170_WL_OPENING_PROGRESS_END:
        return 1.0
    if rank <= 100:
        base = 1.18
    elif rank <= 1000:
        base = 1.14
    elif rank <= 5000:
        base = 1.08
    else:
        base = 1.04
    # Fade out smoothly before the first 5% of the chapter. The first two CN170
    # chapters caught up by 5-10%, so this temporary boost must stay narrow.
    strength = max(
        0.0,
        (CN_170_WL_OPENING_PROGRESS_END - float(progress))
        / CN_170_WL_OPENING_PROGRESS_END,
    )
    return math.exp(math.log(base) * strength)


def is_cn_170_wl_opening_chapter(
    region: str,
    event_id: int,
    leaderboard_scope: str,
    progress: float,
) -> bool:
    return (
        region == "cn"
        and int(event_id) == CN_170_WL_OPENING_EVENT_ID
        and leaderboard_scope == "chapter"
        and float(progress) < CN_170_WL_OPENING_PROGRESS_END
    )


def load_event_meta_from_masterdata(module, region: str, event_id: int):
    base = module.MASTERDATA_ROOTS[region]
    chapter_windows = module.load_world_bloom_chapter_windows(base)
    for row in module.load_json(base / "events.json"):
        if int(row["id"]) != event_id:
            continue
        start_timestamp = int(row["startAt"]) // 1000
        end_timestamp = int(row["aggregateAt"]) // 1000
        series_type = "world_bloom" if str(row.get("eventType")) == "world_bloom" else "normal"
        return module.EventMeta(
            region=region,
            event_id=event_id,
            series_type=series_type,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            chapter_windows=chapter_windows.get(event_id, {}),
        )
    raise ValueError(f"event {event_id} not found for region {region}")


def load_event_meta_from_manifest(data_root: Path, module, region: str, event_id: int):
    path = data_root / "regions" / region / "training" / "event_manifest.csv"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["event_id"]) != event_id:
                continue
            return module.EventMeta(
                region=region,
                event_id=event_id,
                series_type=str(row["series_type"]),
                start_timestamp=int(row["start_timestamp"]),
                end_timestamp=int(row["end_timestamp"]),
                chapter_windows=module.load_world_bloom_chapter_windows(
                    module.MASTERDATA_ROOTS.get(region, Path())
                ).get(event_id, {}),
            )
    return None


def normalize_leaderboard_scope(series_type: str, value: str) -> str:
    if value != "auto":
        return value
    return "chapter" if series_type == "world_bloom" else "total"


def meta_for_leaderboard_scope(module, meta, leaderboard_scope: str):
    if meta.series_type == "world_bloom" and leaderboard_scope == "total":
        return module.EventMeta(
            region=meta.region,
            event_id=meta.event_id,
            series_type="normal",
            start_timestamp=meta.start_timestamp,
            end_timestamp=meta.end_timestamp,
            chapter_windows={},
        )
    return meta


def load_rank_history(path: Path):
    per_line: dict[tuple[int, int], list[tuple[int, int]]] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            if len(row) >= 4:
                character_id = int(row[0])
                rank = int(row[1])
                timestamp = int(row[2])
                score = int(row[3])
            else:
                character_id = 0
                rank = int(row[0])
                timestamp = int(row[1])
                score = int(row[2])
            per_line.setdefault((character_id, rank), []).append((timestamp, score))
    for key in per_line:
        per_line[key].sort()
    return per_line


def interpolate_missing_rank_history(per_rank: dict[int, list[tuple[int, int]]], target_rank: int) -> bool:
    if target_rank in per_rank:
        return False
    lower_candidates = [rank for rank in per_rank if rank < target_rank]
    upper_candidates = [rank for rank in per_rank if rank > target_rank]
    if not lower_candidates or not upper_candidates:
        return False
    lower_rank = max(lower_candidates)
    upper_rank = min(upper_candidates)
    lower_points = dict(per_rank[lower_rank])
    upper_points = dict(per_rank[upper_rank])
    common_timestamps = sorted(set(lower_points) & set(upper_points))
    if len(common_timestamps) < 2 or lower_rank <= 0 or upper_rank <= lower_rank:
        return False
    weight = (math.log(target_rank) - math.log(lower_rank)) / (math.log(upper_rank) - math.log(lower_rank))
    per_rank[target_rank] = [
        (
            timestamp,
            int(round(math.exp(
                math.log(max(float(lower_points[timestamp]), 1.0)) * (1.0 - weight)
                + math.log(max(float(upper_points[timestamp]), 1.0)) * weight
            ))),
        )
        for timestamp in common_timestamps
    ]
    return True


def interpolate_missing_line_histories(per_line: dict[tuple[int, int], list[tuple[int, int]]], target_rank: int) -> set[tuple[int, int]]:
    synthetic_keys: set[tuple[int, int]] = set()
    character_ids = sorted({character_id for character_id, _rank in per_line})
    for character_id in character_ids:
        per_rank = {
            rank: points
            for (line_character_id, rank), points in per_line.items()
            if line_character_id == character_id
        }
        if interpolate_missing_rank_history(per_rank, target_rank):
            per_line[(character_id, target_rank)] = per_rank[target_rank]
            synthetic_keys.add((character_id, target_rank))
    return synthetic_keys


def live_pressure_target_rank(rank: int) -> int | None:
    if 90 <= rank <= 100:
        return 100
    if rank in LIVE_PRESSURE_RANKS:
        return rank
    return None


def iter_zstd_text_lines(path: Path):
    zstdcat = shutil.which("zstdcat")
    if zstdcat is None:
        for candidate in ("/opt/homebrew/bin/zstdcat", "/usr/local/bin/zstdcat"):
            if Path(candidate).exists():
                zstdcat = candidate
                break
    if zstdcat:
        process = subprocess.Popen(
            [zstdcat, str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        try:
            yield from process.stdout
        finally:
            process.stdout.close()
            process.wait()
        return

    try:
        import zstandard as zstd
    except ImportError as exc:
        raise RuntimeError("zstdcat is not installed and Python package zstandard is unavailable") from exc

    with path.open("rb") as handle:
        reader = zstd.ZstdDecompressor().stream_reader(handle)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        try:
            yield from text
        finally:
            text.close()


def iter_sql_table_rows(module, path: Path, table_name: str):
    if not path.exists():
        return
    prefix = f"INSERT INTO `{table_name}` VALUES"
    for line in iter_zstd_text_lines(path):
        if not line.startswith(prefix):
            continue
        parsed = module.parse_insert_statement_blob(line)
        if parsed is None:
            continue
        _table_name, values_sql = parsed
        yield from module.iter_mixed_rows(values_sql)


def current_table_names(event_id: int, series_type: str) -> tuple[str, tuple[str, ...]]:
    prefix = "wl" if series_type == "world_bloom" else "event"
    table_name = f"{prefix}_{event_id}"
    time_table_names = (f"{table_name}_time_id",)
    if series_type == "world_bloom":
        time_table_names = (*time_table_names, f"event_{event_id}_time_id")
    return table_name, time_table_names


def iter_current_rank_rows(module, current_sql_zst: Path, event_id: int, series_type: str):
    table_name, time_table_names = current_table_names(event_id, series_type)
    time_by_id: dict[int, int] = {}
    for time_table_name in time_table_names:
        for row in iter_sql_table_rows(module, current_sql_zst, time_table_name):
            if len(row) >= 2:
                time_by_id[int(row[0])] = int(row[1])
        if time_by_id:
            break
    if not time_by_id:
        return

    for row in iter_sql_table_rows(module, current_sql_zst, table_name):
        if series_type == "world_bloom":
            if len(row) < 5:
                continue
            timestamp = time_by_id.get(int(row[0]))
            if timestamp is None:
                continue
            yield {
                "timestamp": timestamp,
                "user_id": str(row[1]),
                "character_id": int(row[2]),
                "score": int(row[3]),
                "rank": int(row[4]),
            }
        else:
            if len(row) < 4:
                continue
            timestamp = time_by_id.get(int(row[0]))
            if timestamp is None:
                continue
            yield {
                "timestamp": timestamp,
                "user_id": str(row[1]),
                "character_id": 0,
                "score": int(row[2]),
                "rank": int(row[3]),
            }


def load_current_rank_history(module, region: str, event_id: int, series_type: str, current_sql_zst: Path | None):
    _ = region
    if current_sql_zst is None or not current_sql_zst.exists():
        return {}
    per_line_points: dict[tuple[int, int], dict[int, int]] = {}
    for row in iter_current_rank_rows(module, current_sql_zst, event_id, series_type):
        rank = int(row["rank"])
        if rank not in module.COMMON_RANKS:
            continue
        key = (int(row["character_id"]), rank)
        timestamp = int(row["timestamp"])
        score = int(row["score"])
        per_line_points.setdefault(key, {})[timestamp] = max(score, per_line_points.setdefault(key, {}).get(timestamp, 0))
    return {
        key: sorted(points.items())
        for key, points in per_line_points.items()
    }


def load_current_boundary_pressure_contexts(
    module,
    region: str,
    event_id: int,
    series_type: str,
    current_sql_zst: Path | None,
):
    if current_sql_zst is None or not current_sql_zst.exists():
        return {}

    snapshot_users: dict[tuple[int, int, int], set[str]] = {}
    for row in iter_current_rank_rows(module, current_sql_zst, event_id, series_type):
        target_rank = live_pressure_target_rank(int(row["rank"]))
        if target_rank is None:
            continue
        snapshot_users.setdefault(
            (int(row["character_id"]), target_rank, int(row["timestamp"])),
            set(),
        ).add(str(row["user_id"]))

    grouped: dict[tuple[int, int], list[tuple[int, frozenset[str]]]] = {}
    for (character_id, target_rank, timestamp), users in snapshot_users.items():
        grouped.setdefault((character_id, target_rank), []).append((timestamp, frozenset(users)))

    return {
        (region, event_id, character_id, target_rank): module.boundary_pressure_series_from_snapshots(snapshots)
        for (character_id, target_rank), snapshots in grouped.items()
    }


def live_boundary_pressure_features(
    module,
    event_key: tuple[str, int],
    character_id: int,
    rank: int,
    timestamp: int,
    boundary_pressure_contexts,
) -> list[float]:
    series = boundary_pressure_contexts.get((event_key[0], event_key[1], character_id, rank))
    if series is None and character_id != 0:
        series = boundary_pressure_contexts.get((event_key[0], event_key[1], 0, rank))
    if series is None:
        return [0.0] * module.BOUNDARY_PRESSURE_FEATURE_COUNT
    index = bisect.bisect_right(series.timestamps, timestamp) - 1
    if index < 0:
        return [0.0] * module.BOUNDARY_PRESSURE_FEATURE_COUNT
    return list(series.features[index])


def percentile_pressure(value: float, p75: float, p95: float) -> float:
    if p95 <= p75:
        return 0.0
    return max(min((float(value) - p75) / (p95 - p75), 1.0), 0.0)


def pressure_calibration_profile(region: str, rank: int, series_type: str) -> dict[str, float] | None:
    base = PRESSURE_CALIBRATION_BASE.get(rank)
    thresholds = PRESSURE_REGION_THRESHOLDS.get(region, {}).get(rank)
    if base is None or thresholds is None:
        return None
    p75_6h, p95_6h, p75_24h, p95_24h = thresholds
    scale = WORLD_BLOOM_PRESSURE_EXTRA_SCALE if is_world_bloom_like(series_type) else 1.0
    return {
        "p75_6h": float(p75_6h),
        "p95_6h": float(p95_6h),
        "p75_24h": float(p75_24h),
        "p95_24h": float(p95_24h),
        "max_extra": float(base["max_extra"]) * scale,
        "tail_extra": float(base["tail_extra"]) * scale,
        "cap": 1.0 + (float(base["cap"]) - 1.0) * scale,
    }


def live_boundary_pressure_multiplier(
    region: str,
    rank: int,
    progress: float,
    pressure_features: list[float],
    series_type: str,
) -> float:
    if not pressure_features or pressure_features[0] <= 0.0:
        return 1.0
    profile = pressure_calibration_profile(region, rank, series_type)
    if profile is None:
        return 1.0
    unique_6h_ratio = float(pressure_features[4])
    unique_24h_ratio = float(pressure_features[5])
    tail_strength = max(min((float(progress) - 0.70) / 0.30, 1.0), 0.0)
    pressure = max(
        percentile_pressure(unique_6h_ratio, float(profile["p75_6h"]), float(profile["p95_6h"])),
        percentile_pressure(unique_24h_ratio, float(profile["p75_24h"]), float(profile["p95_24h"])),
    )
    extra = float(profile["max_extra"]) * pressure + float(profile["tail_extra"]) * pressure * tail_strength
    return min(1.0 + extra, float(profile["cap"]))


def log_blend(left: float, right: float, left_weight: float) -> float:
    weight = max(min(float(left_weight), 1.0), 0.0)
    left_value = max(float(left), 1.0)
    right_value = max(float(right), 1.0)
    return math.exp(math.log(left_value) * weight + math.log(right_value) * (1.0 - weight))


def world_bloom_shape_first_prediction(current_score: int, baselines: dict[str, float]) -> float:
    prediction = log_blend(
        float(baselines["jp_shape_projection"]),
        float(baselines["hybrid_projection"]),
        WORLD_BLOOM_SHAPE_HYBRID_WEIGHT,
    )
    return max(float(current_score), prediction)


def front_rank_shape_floor(
    prediction: float,
    current_score: int,
    rank: int,
    progress: float,
    baselines: dict[str, float],
    series_type: str,
) -> float:
    if is_world_bloom_like(series_type) or rank > 100 or progress < 0.70:
        return max(float(prediction), float(current_score))
    shape_projection = float(baselines.get("jp_shape_projection") or 0.0)
    if shape_projection <= 0.0:
        return max(float(prediction), float(current_score))
    return max(float(prediction), float(current_score), shape_projection)


def mean_or_zero(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def clamp(value: float, lower: float, upper: float) -> float:
    return max(min(float(value), float(upper)), float(lower))

def is_world_bloom_like(series_type: str) -> bool:
    return str(series_type) in {"world_bloom", WORLD_BLOOM_TOTAL_SERIES_TYPE}


def is_world_bloom_total(series_type: str) -> bool:
    return str(series_type) == WORLD_BLOOM_TOTAL_SERIES_TYPE


def late_progress_guard_start(region: str) -> float:
    return LATE_PROGRESS_GUARD_REGION_START.get(region, LATE_PROGRESS_GUARD_START)


def late_stage_rank_profile(rank: int) -> tuple[float, float, float, float]:
    rank = int(rank)
    if rank <= 100:
        return 0.68, 0.86, 0.020, 0.004
    if rank <= 200:
        return 0.65, 0.82, 0.024, 0.006
    if rank <= 500:
        return 0.60, 0.70, 0.024, 0.008
    if rank <= 1000:
        return 0.44, 0.48, 0.028, 0.010
    if rank <= 2000:
        return 0.54, 0.58, 0.026, 0.008
    return 0.55, 0.62, 0.020, 0.006


def early_progress_strength(progress: float) -> float:
    progress = float(progress)
    if progress <= EARLY_PRIOR_FULL_PROGRESS:
        return 1.0
    if progress >= EARLY_PRIOR_DECAY_PROGRESS:
        return 0.0
    raw = (EARLY_PRIOR_DECAY_PROGRESS - progress) / (EARLY_PRIOR_DECAY_PROGRESS - EARLY_PRIOR_FULL_PROGRESS)
    return clamp(raw, 0.0, 1.0) ** 1.15


def early_heat_progress_strength(progress: float) -> float:
    progress = float(progress)
    if progress <= EARLY_HEAT_FULL_PROGRESS:
        return 1.0
    if progress >= EARLY_HEAT_DECAY_PROGRESS:
        return 0.0
    raw = (EARLY_HEAT_DECAY_PROGRESS - progress) / (EARLY_HEAT_DECAY_PROGRESS - EARLY_HEAT_FULL_PROGRESS)
    return clamp(raw, 0.0, 1.0) ** 1.35


def early_rank_strength(rank: int) -> float:
    rank = int(rank)
    if rank <= 20:
        return 1.0
    if rank <= 50:
        return 0.92
    if rank <= 100:
        return 0.82
    if rank <= 500:
        return 0.68
    if rank <= 1000:
        return 0.58
    if rank <= 5000:
        return 0.45
    if rank <= 20000:
        return 0.32
    return 0.20


def early_heat_rank_strength(rank: int) -> float:
    rank = int(rank)
    if rank <= 20:
        return 0.78
    if rank <= 50:
        return 0.30
    if rank <= 100:
        return 1.00
    if rank <= 500:
        return 0.45
    if rank <= 1000:
        return 0.35
    if rank <= 5000:
        return 0.60
    if rank <= 20000:
        return 0.42
    return 0.26


def early_pace_trust(progress: float) -> float:
    progress = float(progress)
    if progress <= 0.0:
        return 0.0
    if progress >= EARLY_PACE_TRUST_FULL_PROGRESS:
        return 1.0
    return clamp(progress / EARLY_PACE_TRUST_FULL_PROGRESS, 0.0, 1.0) ** 1.85


def early_shape_hybrid_weight(rank: int, progress: float) -> float:
    if progress >= 0.10:
        return 0.12 if int(rank) <= 100 else 0.10
    opening_weight = 0.42 if int(rank) <= 100 else 0.48
    progress_decay = clamp(progress / 0.10, 0.0, 1.0) ** 0.85
    return opening_weight * (1.0 - progress_decay) + (0.12 if int(rank) <= 100 else 0.10) * progress_decay


def early_linear_projection_weight(
    rank_weight: float,
    pace_trust: float,
    positive_heat_weight: float,
    series_type: str,
) -> float:
    # Opening snapshots are useful evidence, but their instantaneous pace is
    # noisy. Treat linear pace as a late-rising correction to the shape anchor,
    # not as the anchor itself.
    linear_weight = 0.040 + 0.090 * rank_weight + 0.165 * pace_trust + 0.035 * positive_heat_weight
    if is_world_bloom_total(series_type):
        linear_weight *= 0.55
    elif is_world_bloom_like(series_type):
        linear_weight *= 0.70
    return clamp(linear_weight, 0.040, 0.36)


def mid_stage_shape_floor(
    prediction: float,
    current_score: int,
    rank: int,
    progress: float,
    baselines: dict[str, float],
    heat_prior: dict[str, float],
    series_type: str,
) -> float | None:
    if (
        is_world_bloom_like(series_type)
        or progress < MID_STAGE_SHAPE_FLOOR_START
        or progress > MID_STAGE_SHAPE_FLOOR_END
        or prediction <= float(current_score)
    ):
        return None

    linear_projection = max(float(baselines.get("linear_projection") or 0.0), float(current_score), 1.0)
    pace_projection = max(float(baselines.get("pace_6h_projection") or 0.0), float(current_score), 1.0)
    hybrid_projection = max(float(baselines.get("hybrid_projection") or 0.0), float(current_score), 1.0)
    shape_projection = max(float(baselines.get("jp_shape_projection") or 0.0), float(current_score), 1.0)
    shape_anchor = log_blend(shape_projection, hybrid_projection, 0.70)
    pace_anchor = log_blend(linear_projection, pace_projection, 0.55)
    floor_target = log_blend(shape_anchor, pace_anchor, 0.70)

    if floor_target <= float(prediction) * MID_STAGE_SHAPE_FLOOR_TRIGGER:
        return None

    progress_window = 1.0
    if progress > 0.30:
        progress_window = clamp((MID_STAGE_SHAPE_FLOOR_END - progress) / 0.10, 0.0, 1.0)
    if progress_window <= 0.0:
        return None

    gap = floor_target / max(float(prediction), 1.0)
    gap_strength = clamp((math.log(gap) - math.log(MID_STAGE_SHAPE_FLOOR_TRIGGER)) / math.log(1.75 / MID_STAGE_SHAPE_FLOOR_TRIGGER), 0.0, 1.0)
    heat_support = clamp((float(heat_prior.get("event_heat", 0.0)) + 0.04) / 0.18, 0.0, 1.0)
    rank_support = early_rank_strength(rank)
    strength = (0.36 + 0.24 * gap_strength + 0.12 * heat_support) * (0.72 + 0.28 * rank_support)
    strength = clamp(strength * progress_window, 0.0, 0.72)
    if strength <= 0.0:
        return None

    floored = log_blend(float(prediction), floor_target, 1.0 - strength)
    floored = max(float(current_score), floored)
    if floored <= float(prediction):
        return None
    return floored


def shape_envelope_anchor(
    current_score: int,
    baselines: dict[str, float],
) -> dict[str, float]:
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


def shape_envelope_target_and_strength(
    progress: float,
    anchors: dict[str, float],
    signal: dict[str, float],
) -> tuple[float | None, float, str]:
    progress = float(progress)
    if progress >= 0.95:
        return None, 0.0, "off"

    gap = float(signal.get("median_log_gap", 0.0))
    share = float(signal.get("positive_share", 0.0))
    if progress >= 0.80:
        strong = gap > math.log(1.14) and share >= 0.50
        active = gap > math.log(1.07) and share >= 0.45
    else:
        strong = gap > SHAPE_ENVELOPE_SIGNAL_STRONG_GAP and share >= SHAPE_ENVELOPE_SIGNAL_STRONG_SHARE
        active = gap > SHAPE_ENVELOPE_SIGNAL_GAP and share >= SHAPE_ENVELOPE_SIGNAL_SHARE
    if progress < 0.60 and float(signal.get("shape_model_gap", 0.0)) < SHAPE_ENVELOPE_SHAPE_SUPPORT_GAP:
        active = False
    if 0.20 <= progress < 0.40 and (
        gap < SHAPE_ENVELOPE_MID_STRONG_GAP
        or float(signal.get("pace_model_gap", 0.0)) < SHAPE_ENVELOPE_MID_PACE_SUPPORT
    ):
        active = False
    if not active:
        return None, 0.0, "off"

    if progress < 0.05:
        return None, 0.0, "off"
    if progress < 0.10:
        return anchors["shape"], 0.75 if strong else 0.55, "early_shape"
    if progress < 0.20:
        return anchors["shape"], 0.74, "early_shape"
    if progress < 0.40:
        return anchors["anchor"], 0.92, "mid_anchor"
    if progress < 0.80:
        return max(anchors["shape"], anchors["hybrid"]), 1.0, "shape_hybrid"
    return anchors["shape"], 1.0, "late_shape"


def apply_shape_envelope_adjustments(predictions: list[dict[str, object]]) -> None:
    by_line: dict[int, list[dict[str, object]]] = {}
    for row in predictions:
        by_line.setdefault(int(row.get("character_id") or 0), []).append(row)

    for rows in by_line.values():
        if not rows or any(is_world_bloom_like(str(row.get("series_type", ""))) for row in rows):
            continue
        signal_logs = []
        shape_model_logs = []
        pace_model_logs = []
        for row in rows:
            rank = int(row.get("rank") or 0)
            progress = float(row.get("progress") or 0.0)
            if rank not in SHAPE_ENVELOPE_SIGNAL_RANKS or progress >= 0.98:
                continue
            prediction = max(float(row.get("model_prediction") or 0.0), 1.0)
            anchor = float(row.get("shape_envelope_anchor") or 0.0)
            if anchor <= 0.0:
                continue
            signal_logs.append(math.log(anchor / prediction))
            shape_projection = max(float(row.get("jp_shape_projection") or 0.0), 1.0)
            pace_projection = max(float(row.get("pace_6h_projection") or 0.0), 1.0)
            shape_model_logs.append(math.log(shape_projection / prediction))
            pace_model_logs.append(math.log(pace_projection / prediction))

        signal_logs.sort()
        shape_model_logs.sort()
        pace_model_logs.sort()
        if signal_logs:
            median_log_gap = signal_logs[len(signal_logs) // 2]
            positive_share = sum(1 for value in signal_logs if value > math.log(1.05)) / len(signal_logs)
        else:
            median_log_gap = 0.0
            positive_share = 0.0
        shape_model_gap = shape_model_logs[len(shape_model_logs) // 2] if shape_model_logs else 0.0
        pace_model_gap = pace_model_logs[len(pace_model_logs) // 2] if pace_model_logs else 0.0
        signal = {
            "median_log_gap": median_log_gap,
            "positive_share": positive_share,
            "shape_model_gap": shape_model_gap,
            "pace_model_gap": pace_model_gap,
        }

        for row in rows:
            row["shape_envelope_signal_log_gap"] = median_log_gap
            row["shape_envelope_signal_positive_share"] = positive_share
            row["shape_envelope_signal_shape_model_gap"] = shape_model_gap
            row["shape_envelope_signal_pace_model_gap"] = pace_model_gap
            if bool(row.get("event_finished_prediction")):
                row["shape_envelope_target"] = None
                row["shape_envelope_strength"] = 0.0
                row["shape_envelope_mode"] = "finished"
                continue
            target, strength, mode = shape_envelope_target_and_strength(
                float(row.get("progress") or 0.0),
                {
                    "shape": float(row.get("jp_shape_projection") or 0.0),
                    "hybrid": float(row.get("hybrid_projection") or 0.0),
                    "linear": float(row.get("linear_projection") or 0.0),
                    "anchor": float(row.get("shape_envelope_anchor") or 0.0),
                    "early_anchor": float(row.get("shape_envelope_early_anchor") or 0.0),
                    "opening_anchor": float(row.get("shape_envelope_opening_anchor") or 0.0),
                },
                signal,
            )
            row["shape_envelope_target"] = target
            row["shape_envelope_strength"] = strength
            row["shape_envelope_mode"] = mode
            row["shape_envelope_applied_prediction"] = None
            if target is None or strength <= 0.0:
                continue
            prediction = float(row.get("model_prediction") or 0.0)
            adjusted = max(prediction, log_blend(prediction, float(target), 1.0 - strength))
            if adjusted > prediction:
                row["shape_envelope_applied_prediction"] = adjusted
                row["pressure_adjusted_prediction"] = adjusted
                row["model_prediction"] = adjusted


def early_event_heat_prior_log(
    module,
    region: str,
    event_id: int,
    character_id: int,
    event_feature_contexts,
    character_heat_priors,
) -> dict[str, float]:
    context = event_feature_contexts.get((region, event_id), module.EventFeatureContext())
    condition = str(context.condition)
    unit_values = [
        module.get_unit_heat_prior(character_heat_priors, region, condition, str(unit_label))
        for unit_label in context.unit_labels
    ]
    main_values = [
        module.get_main_character_heat_prior(character_heat_priors, region, condition, int(cid))
        for cid in context.main_character_ids
    ]
    character_values = [
        module.get_character_heat_prior(character_heat_priors, region, condition, int(cid))
        for cid in context.character_ids
    ]
    line_value = (
        module.get_character_heat_prior(character_heat_priors, region, condition, int(character_id))
        if int(character_id) > 0
        else 0.0
    )
    unit_heat = mean_or_zero(unit_values)
    main_heat = mean_or_zero(main_values)
    character_heat = mean_or_zero(character_values)
    unit_popularity_heat = mean_or_zero([
        float(module.UNIT_HEAT_PRIOR.get(str(unit_label), 0.0))
        for unit_label in context.unit_labels
    ])
    main_popularity_heat = mean_or_zero([
        character_popularity_prior_for_live(module, condition, int(cid))
        for cid in context.main_character_ids
        if condition == "world_bloom" or int(cid) not in module.VOCALOID_CHARACTER_IDS
    ])
    character_popularity_heat = mean_or_zero([
        character_popularity_prior_for_live(module, condition, int(cid))
        for cid in context.character_ids
        if condition == "world_bloom" or int(cid) not in module.VOCALOID_CHARACTER_IDS
    ])
    line_popularity_heat = (
        character_popularity_prior_for_live(module, condition, int(character_id))
        if int(character_id) > 0
        else 0.0
    )

    popularity_weight = float(POPULARITY_PRIOR_LIVE_WEIGHT_BY_CONDITION.get(condition, 0.25))
    unit_weight = float(UNIT_PRIOR_LIVE_WEIGHT_BY_CONDITION.get(condition, 0.25))
    if unit_popularity_heat:
        unit_heat = unit_heat * (1.0 - unit_weight) + unit_popularity_heat * unit_weight
    if main_popularity_heat:
        main_heat = main_heat * (1.0 - popularity_weight) + main_popularity_heat * popularity_weight
    if character_popularity_heat:
        character_heat = character_heat * (1.0 - popularity_weight) + character_popularity_heat * popularity_weight
    if line_popularity_heat and (condition == "world_bloom" or int(character_id) not in module.VOCALOID_CHARACTER_IDS):
        line_value = line_value * (1.0 - popularity_weight) + line_popularity_heat * popularity_weight

    if condition == "box":
        # Box events are often driven by both unit demand and the banner
        # character. Use the stronger signal so one cold historical banner does
        # not erase a hot unit in the first few hours.
        event_heat = max(unit_heat, 0.55 * unit_heat + 0.45 * main_heat, character_heat)
    elif condition == "world_bloom":
        event_heat = max(line_value, main_heat, character_heat)
    elif condition == "mixed":
        event_heat = 0.50 * character_heat + 0.30 * main_heat + 0.20 * unit_heat
    else:
        event_heat = 0.45 * character_heat + 0.35 * main_heat + 0.20 * unit_heat

    return {
        "event_heat": clamp(event_heat, -0.22, 0.28),
        "unit_heat": unit_heat,
        "main_heat": main_heat,
        "character_heat": character_heat,
        "line_heat": line_value,
        "unit_popularity_heat": unit_popularity_heat,
        "main_popularity_heat": main_popularity_heat,
        "character_popularity_heat": character_popularity_heat,
        "line_popularity_heat": line_popularity_heat,
    }


def early_stage_prior_adjustment(
    prediction: float,
    current_score: int,
    rank: int,
    progress: float,
    baselines: dict[str, float],
    heat_prior: dict[str, float],
    series_type: str,
) -> tuple[float, dict[str, float | None]]:
    return early_stage_prior_adjustment_with_region(
        prediction,
        current_score,
        rank,
        progress,
        baselines,
        heat_prior,
        series_type,
        "",
    )


def early_stage_prior_adjustment_with_region(
    prediction: float,
    current_score: int,
    rank: int,
    progress: float,
    baselines: dict[str, float],
    heat_prior: dict[str, float],
    series_type: str,
    region: str,
) -> tuple[float, dict[str, float | None]]:
    progress_weight = early_progress_strength(progress)
    if progress_weight <= 0.0:
        return float(prediction), {
            "early_progress_strength": 0.0,
            "early_heat_progress_strength": 0.0,
            "early_rank_strength": early_rank_strength(rank),
            "early_heat_log": heat_prior["event_heat"],
            "early_heat_multiplier": 1.0,
            "early_projection_target": None,
        }

    rank_weight = early_rank_strength(rank)
    heat_rank_weight = early_heat_rank_strength(rank)
    heat_progress_weight = early_heat_progress_strength(progress)
    linear_projection = max(float(baselines.get("linear_projection") or 0.0), float(current_score), 1.0)
    hybrid_projection = max(float(baselines.get("hybrid_projection") or 0.0), float(current_score), 1.0)
    shape_projection = max(float(baselines.get("jp_shape_projection") or 0.0), float(current_score), 1.0)
    hybrid_weight = early_shape_hybrid_weight(rank, progress)
    shape_base = log_blend(shape_projection, hybrid_projection, 1.0 - hybrid_weight)

    pace_trust = early_pace_trust(progress)
    heat_strength = progress_weight * heat_progress_weight * (0.30 + 0.24 * heat_rank_weight)
    if is_world_bloom_total(series_type):
        heat_strength *= 0.95
    elif is_world_bloom_like(series_type):
        heat_strength *= 0.75
    heat_strength *= float(EARLY_HEAT_SCALE_BY_REGION.get(str(region), EARLY_HEAT_SCALE))
    heat_log_adjustment = clamp(
        heat_prior["event_heat"] * heat_strength,
        EARLY_PRIOR_MAX_NEGATIVE_LOG_ADJUSTMENT,
        EARLY_PRIOR_MAX_POSITIVE_LOG_ADJUSTMENT,
    )
    heat_multiplier = math.exp(heat_log_adjustment)
    shape_prior_target = max(float(current_score), shape_base * heat_multiplier)

    # The first few hours can overstate the true final pace because many users
    # start playing at once. Trust the linear pace only gradually; early output
    # should mostly follow shape/hybrid priors plus event heat.
    positive_heat_weight = clamp(max(heat_prior["event_heat"], 0.0) / 0.20, 0.0, 1.0) * heat_progress_weight
    linear_weight = early_linear_projection_weight(
        rank_weight,
        pace_trust,
        positive_heat_weight,
        series_type,
    )
    future_target = log_blend(linear_projection, shape_prior_target, linear_weight)

    projection_strength = progress_weight * (0.46 + 0.32 * rank_weight)
    if is_world_bloom_total(series_type):
        projection_strength *= 0.82
    elif is_world_bloom_like(series_type):
        projection_strength *= 0.72
    projection_strength = clamp(projection_strength, 0.0, 0.86)
    adjusted = log_blend(float(prediction), future_target, 1.0 - projection_strength)
    adjusted = max(float(current_score), adjusted)
    return adjusted, {
        "early_progress_strength": progress_weight,
        "early_heat_progress_strength": heat_progress_weight,
        "early_rank_strength": rank_weight,
        "early_heat_rank_strength": heat_rank_weight,
        "early_pace_trust": pace_trust,
        "early_heat_log": heat_prior["event_heat"],
        "early_unit_heat_log": heat_prior["unit_heat"],
        "early_main_heat_log": heat_prior["main_heat"],
        "early_character_heat_log": heat_prior["character_heat"],
        "early_line_heat_log": heat_prior["line_heat"],
        "early_unit_popularity_heat_log": heat_prior.get("unit_popularity_heat", 0.0),
        "early_main_popularity_heat_log": heat_prior.get("main_popularity_heat", 0.0),
        "early_character_popularity_heat_log": heat_prior.get("character_popularity_heat", 0.0),
        "early_line_popularity_heat_log": heat_prior.get("line_popularity_heat", 0.0),
        "early_heat_multiplier": heat_multiplier,
        "early_heat_scale": float(EARLY_HEAT_SCALE_BY_REGION.get(str(region), EARLY_HEAT_SCALE)),
        "early_shape_hybrid_weight": hybrid_weight,
        "early_shape_prior_target": shape_prior_target,
        "early_projection_target": future_target,
        "early_projection_strength": projection_strength,
        "early_projection_linear_weight": linear_weight,
    }


def event_heat_prior_live_adjustment(
    module,
    prediction: float,
    current_score: int,
    rank: int,
    progress: float,
    prediction_row: dict[str, object],
    payload: dict[str, object],
) -> tuple[float, dict[str, float | int | str | None]]:
    debug: dict[str, float | int | str | None] = {
        "event_heat_prior_prediction": None,
        "event_heat_prior_log": 0.0,
        "event_heat_prior_source": None,
        "event_heat_prior_count": 0,
        "event_heat_prior_blend_weight": 0.0,
    }
    if is_world_bloom_like(str(prediction_row.get("series_type", ""))):
        return float(prediction), debug
    if not bool(payload.get("event_heat_prior_enabled", False)):
        return float(prediction), debug

    prior_prediction, prior_debug = module.event_heat_prior_prediction(
        prediction_row,
        payload.get("live_event_heat_priors"),
        payload.get("live_character_heat_priors", {}),
        payload.get("live_event_feature_contexts", {}),
    )
    debug.update(prior_debug)
    blend_weight = module.event_heat_prior_blend_weight(progress, rank)
    debug["event_heat_prior_blend_weight"] = blend_weight
    if prior_prediction is None or blend_weight <= 0.0:
        return float(prediction), debug

    adjusted = log_blend(float(prediction), max(float(prior_prediction), float(current_score)), 1.0 - blend_weight)
    return max(float(current_score), adjusted), debug


def world_bloom_total_chapter_weight(
    module,
    region: str,
    character_id: int,
    rank: int,
    character_heat_priors,
    payload: dict[str, object] | None,
    future_chapter: bool = False,
) -> tuple[float, float]:
    heat = module.get_character_heat_prior(
        character_heat_priors,
        region,
        "world_bloom",
        int(character_id),
    )
    popularity_heat = float(module.CHARACTER_POPULARITY_PRIOR.get(int(character_id), 0.0))
    if popularity_heat:
        if future_chapter:
            heat = heat * 0.35 + popularity_heat * 0.65
        else:
            heat = heat * 0.58 + popularity_heat * 0.42

    if payload is not None and isinstance(payload.get("wl2_character_prior_region_weights"), dict):
        raw_region_weights = payload["wl2_character_prior_region_weights"]
        region_weight = float(raw_region_weights.get(region, 0.0)) if isinstance(raw_region_weights, dict) else 0.0
    else:
        region_weight = float(module.WORLD_BLOOM_WL2_CHARACTER_PRIOR_REGION_WEIGHTS.get(region, 0.0))
    rank_weight = module.wl2_character_prior_rank_weight(rank, payload)
    future_boost = 1.18 if future_chapter else 1.0
    effective_heat = clamp(float(heat) * max(region_weight, 0.35) * max(rank_weight, 0.20) * future_boost, -0.32, 0.32)
    return math.exp(effective_heat), float(heat)


def world_bloom_total_weighted_progress(
    module,
    meta,
    rank: int,
    current_timestamp: int,
    character_heat_priors,
    payload: dict[str, object] | None,
) -> dict[str, float | int | None]:
    chapters = sorted(
        (
            (int(start_timestamp), int(end_timestamp), int(character_id))
            for character_id, (start_timestamp, end_timestamp) in meta.chapter_windows.items()
            if int(start_timestamp) < int(end_timestamp)
        ),
        key=lambda item: (item[0], item[2]),
    )
    if not chapters:
        return {
            "weighted_progress": module.event_line_progress(meta, 0, current_timestamp),
            "active_character_id": None,
            "active_chapter_progress": None,
            "total_chapter_weight": 0.0,
            "elapsed_chapter_weight": 0.0,
            "future_chapter_weight": 0.0,
            "future_chapter_heat": 0.0,
            "future_chapter_count": 0,
        }

    total_weight = 0.0
    elapsed_weight = 0.0
    future_weight = 0.0
    future_heat_sum = 0.0
    future_count = 0
    active_character_id: int | None = None
    active_chapter_progress: float | None = None
    active_chapter_heat = 0.0
    active_chapter_weight = 0.0
    for start_timestamp, end_timestamp, character_id in chapters:
        future_chapter = current_timestamp < start_timestamp
        weight, heat = world_bloom_total_chapter_weight(
            module,
            meta.region,
            character_id,
            rank,
            character_heat_priors,
            payload,
            future_chapter=future_chapter,
        )
        total_weight += weight
        if future_chapter:
            future_weight += weight
            future_heat_sum += float(heat)
            future_count += 1
        if current_timestamp >= end_timestamp:
            elapsed_weight += weight
            continue
        if current_timestamp >= start_timestamp:
            chapter_progress = clamp(
                (int(current_timestamp) - start_timestamp) / max(end_timestamp - start_timestamp, 1),
                0.0,
                1.0,
            )
            elapsed_weight += weight * chapter_progress
            active_character_id = character_id
            active_chapter_progress = chapter_progress
            active_chapter_heat = heat
            active_chapter_weight = weight

    weighted_progress = elapsed_weight / max(total_weight, 1e-6)
    return {
        "weighted_progress": clamp(weighted_progress, 0.0, 1.0),
        "active_character_id": active_character_id,
        "active_chapter_progress": active_chapter_progress,
        "active_chapter_heat": active_chapter_heat,
        "active_chapter_weight": active_chapter_weight,
        "total_chapter_weight": total_weight,
        "elapsed_chapter_weight": elapsed_weight,
        "future_chapter_weight": future_weight,
        "future_chapter_heat": future_heat_sum / future_count if future_count else 0.0,
        "future_chapter_count": future_count,
        "chapter_count": len(chapters),
    }


def world_bloom_total_chapter_sum_conversion(rank: int, heat_bias: float) -> float:
    rank = int(rank)
    if rank <= 10:
        base = 0.96
    elif rank <= 50:
        base = 0.93
    elif rank <= 100:
        base = 0.88
    elif rank <= 500:
        base = 0.74
    elif rank <= 1000:
        base = 0.62
    elif rank <= 5000:
        base = 0.42
    elif rank <= 10000:
        base = 0.34
    else:
        base = 0.24
    heat_uplift = clamp(float(heat_bias) / 0.08, -0.5, 1.0) * 0.06
    return clamp(base + heat_uplift, 0.18, 0.98)


def world_bloom_total_chapter_sum_projection(
    module,
    current_score: int,
    rank: int,
    current_timestamp: int,
    live_meta,
    progress_info: dict[str, float | int | None],
    character_heat_priors,
    jp_shape_priors,
    payload: dict[str, object] | None,
) -> tuple[float | None, dict[str, float | int | None]]:
    debug: dict[str, float | int | None] = {
        "world_bloom_total_chapter_sum_projection": None,
        "world_bloom_total_chapter_sum_conversion": None,
        "world_bloom_total_active_chapter_projection": None,
    }
    active_character_id = progress_info.get("active_character_id")
    active_progress = progress_info.get("active_chapter_progress")
    active_weight = float(progress_info.get("active_chapter_weight") or 0.0)
    total_weight = float(progress_info.get("total_chapter_weight") or 0.0)
    elapsed_weight = float(progress_info.get("elapsed_chapter_weight") or 0.0)
    if active_character_id is None or active_progress is None or active_weight <= 0.0 or total_weight <= 0.0:
        return None, debug

    # Once previous chapters have contributed materially, total score no longer
    # lets us isolate the active chapter. Then the full-event shape is safer.
    completed_weight = max(elapsed_weight - active_weight * float(active_progress), 0.0)
    if completed_weight > total_weight * 0.08:
        return None, debug

    chapter_progress = clamp(float(active_progress), 0.01, 0.98)
    linear_projection = max(float(current_score), float(current_score) / chapter_progress)
    chapter_meta = module.EventMeta(
        region=live_meta.region,
        event_id=live_meta.event_id,
        series_type="world_bloom",
        start_timestamp=live_meta.start_timestamp,
        end_timestamp=live_meta.end_timestamp,
        chapter_windows={int(active_character_id): live_meta.chapter_windows[int(active_character_id)]},
    )
    _features, chapter_baselines = module.jp_shape_features(
        meta=chapter_meta,
        rank=rank,
        bucket_progress=chapter_progress,
        current_timestamp=current_timestamp,
        current_score=current_score,
        linear_projection=linear_projection,
        jp_shape_priors=jp_shape_priors,
    )
    active_projection = world_bloom_shape_first_prediction(current_score, chapter_baselines)

    estimated_sum = 0.0
    heat_values: list[float] = []
    for character_id in sorted(int(cid) for cid in live_meta.chapter_windows):
        start_timestamp, _end_timestamp = live_meta.chapter_windows[int(character_id)]
        weight, heat = world_bloom_total_chapter_weight(
            module,
            live_meta.region,
            character_id,
            rank,
            character_heat_priors,
            payload,
            future_chapter=current_timestamp < int(start_timestamp),
        )
        estimated_sum += active_projection * weight / max(active_weight, 1e-6)
        heat_values.append(float(heat))

    heat_bias = mean_or_zero(heat_values)
    conversion = world_bloom_total_chapter_sum_conversion(rank, heat_bias)
    projection = max(float(current_score), estimated_sum * conversion)
    debug.update({
        "world_bloom_total_chapter_sum_projection": projection,
        "world_bloom_total_chapter_sum_conversion": conversion,
        "world_bloom_total_active_chapter_projection": active_projection,
    })
    return projection, debug


def world_bloom_total_shape_adjustment(
    module,
    prediction: float,
    current_score: int,
    rank: int,
    current_timestamp: int,
    live_meta,
    baselines: dict[str, float],
    character_heat_priors,
    jp_shape_priors,
    payload: dict[str, object] | None,
) -> tuple[float, dict[str, float | int | None]]:
    debug: dict[str, float | int | None] = {
        "world_bloom_total_weighted_progress": None,
        "world_bloom_total_projection_target": None,
        "world_bloom_total_projection_strength": 0.0,
        "world_bloom_total_active_character_id": None,
        "world_bloom_total_active_chapter_progress": None,
        "world_bloom_total_active_chapter_heat": None,
        "world_bloom_total_future_chapter_weight": None,
        "world_bloom_total_future_chapter_heat": None,
        "world_bloom_total_future_chapter_count": None,
        "world_bloom_total_chapter_sum_projection": None,
        "world_bloom_total_chapter_sum_conversion": None,
        "world_bloom_total_active_chapter_projection": None,
        "world_bloom_total_full_event_shape_target": None,
        "world_bloom_total_support_strength": 0.0,
        "world_bloom_total_target_floor": None,
    }
    if live_meta.series_type != "world_bloom" or not live_meta.chapter_windows:
        return float(prediction), debug

    progress_info = world_bloom_total_weighted_progress(
        module,
        live_meta,
        rank,
        current_timestamp,
        character_heat_priors,
        payload,
    )
    weighted_progress = float(progress_info["weighted_progress"] or 0.0)
    debug.update({
        "world_bloom_total_weighted_progress": weighted_progress,
        "world_bloom_total_active_character_id": progress_info.get("active_character_id"),
        "world_bloom_total_active_chapter_progress": progress_info.get("active_chapter_progress"),
        "world_bloom_total_active_chapter_heat": progress_info.get("active_chapter_heat"),
        "world_bloom_total_future_chapter_weight": progress_info.get("future_chapter_weight"),
        "world_bloom_total_future_chapter_heat": progress_info.get("future_chapter_heat"),
        "world_bloom_total_future_chapter_count": progress_info.get("future_chapter_count"),
        "world_bloom_total_chapter_count": progress_info.get("chapter_count"),
    })
    if weighted_progress <= 0.0 or weighted_progress >= 0.985:
        return float(prediction), debug

    linear_projection = max(float(current_score), float(current_score) / max(weighted_progress, 0.01))
    shape_meta = module.EventMeta(
        region=live_meta.region,
        event_id=live_meta.event_id,
        series_type="world_bloom",
        start_timestamp=live_meta.start_timestamp,
        end_timestamp=live_meta.end_timestamp,
        chapter_windows={},
    )
    _features, shape_baselines = module.jp_shape_features(
        meta=shape_meta,
        rank=rank,
        bucket_progress=weighted_progress,
        current_timestamp=current_timestamp,
        current_score=current_score,
        linear_projection=linear_projection,
        jp_shape_priors=jp_shape_priors,
    )
    shape_target = log_blend(
        float(shape_baselines["jp_shape_projection"]),
        float(shape_baselines["hybrid_projection"]),
        WORLD_BLOOM_TOTAL_SHAPE_WEIGHT,
    )
    target = log_blend(shape_target, linear_projection, 1.0 - WORLD_BLOOM_TOTAL_LINEAR_WEIGHT)
    full_event_shape = max(float(baselines.get("jp_shape_projection") or 0.0), float(current_score), 1.0)
    full_event_hybrid = max(float(baselines.get("hybrid_projection") or 0.0), float(current_score), 1.0)
    full_event_linear = max(float(baselines.get("linear_projection") or 0.0), float(current_score), 1.0)
    full_event_shape_target = log_blend(
        log_blend(full_event_shape, full_event_hybrid, WORLD_BLOOM_TOTAL_SHAPE_WEIGHT),
        full_event_linear,
        1.0 - WORLD_BLOOM_TOTAL_LINEAR_WEIGHT,
    )
    target = max(target, full_event_shape_target)
    target = max(float(current_score), target)
    chapter_sum_target, chapter_sum_debug = world_bloom_total_chapter_sum_projection(
        module,
        current_score,
        rank,
        current_timestamp,
        live_meta,
        progress_info,
        character_heat_priors,
        jp_shape_priors,
        payload,
    )
    debug.update(chapter_sum_debug)
    if chapter_sum_target is not None:
        chapter_weight = clamp(
            (0.30 - weighted_progress) / 0.30,
            0.0,
            1.0,
        ) ** 0.75
        target = log_blend(target, chapter_sum_target, 1.0 - chapter_weight)

    progress_strength = clamp(
        (WORLD_BLOOM_TOTAL_STRENGTH_DECAY_PROGRESS - weighted_progress)
        / WORLD_BLOOM_TOTAL_STRENGTH_DECAY_PROGRESS,
        0.0,
        1.0,
    )
    strength = WORLD_BLOOM_TOTAL_EARLY_STRENGTH * progress_strength * (0.72 + 0.28 * early_rank_strength(rank))
    strength = clamp(strength, 0.0, 0.94)
    support_window = clamp(
        (WORLD_BLOOM_TOTAL_SUPPORT_DECAY_PROGRESS - weighted_progress)
        / WORLD_BLOOM_TOTAL_SUPPORT_DECAY_PROGRESS,
        0.0,
        1.0,
    )
    support_strength = 0.0
    target_floor = None
    prediction_floor_base = max(float(prediction), float(current_score), 1.0)
    if target > prediction_floor_base and support_window > 0.0:
        gap = math.log(target / prediction_floor_base)
        gap_weight = clamp(
            (gap - math.log(1.10)) / math.log(2.20 / 1.10),
            0.0,
            1.0,
        )
        support_strength = (
            WORLD_BLOOM_TOTAL_SUPPORT_STRENGTH
            * (support_window ** 0.35)
            * (0.70 + 0.30 * early_rank_strength(rank))
            * gap_weight
        )
        strength = max(strength, support_strength)
        floor_ratio = WORLD_BLOOM_TOTAL_TARGET_FLOOR_MIN + (
            WORLD_BLOOM_TOTAL_TARGET_FLOOR_MAX - WORLD_BLOOM_TOTAL_TARGET_FLOOR_MIN
        ) * (support_window ** 0.45) * (0.82 + 0.18 * early_rank_strength(rank))
        target_floor = target * clamp(
            floor_ratio,
            WORLD_BLOOM_TOTAL_TARGET_FLOOR_MIN,
            WORLD_BLOOM_TOTAL_TARGET_FLOOR_MAX,
        )
    if strength <= 0.0:
        return float(prediction), debug

    adjusted = log_blend(float(prediction), target, 1.0 - strength)
    if target_floor is not None:
        adjusted = max(adjusted, target_floor)
    adjusted = max(float(current_score), adjusted)
    debug.update({
        "world_bloom_total_projection_target": target,
        "world_bloom_total_shape_projection": shape_baselines["jp_shape_projection"],
        "world_bloom_total_hybrid_projection": shape_baselines["hybrid_projection"],
        "world_bloom_total_linear_projection": linear_projection,
        "world_bloom_total_full_event_shape_target": full_event_shape_target,
        "world_bloom_total_projection_strength": strength,
        "world_bloom_total_support_strength": support_strength,
        "world_bloom_total_target_floor": target_floor,
    })
    return adjusted, debug


def front_parking_prediction_cap(
    prediction: float,
    current_score: int,
    rank: int,
    progress: float,
    baselines: dict[str, float],
    series_type: str,
) -> float | None:
    if (
        is_world_bloom_like(series_type)
        or rank >= 100
        or progress < FRONT_PARKING_MIN_PROGRESS
        or progress >= FRONT_PARKING_MAX_PROGRESS
        or prediction <= float(current_score)
    ):
        return None

    linear_projection = max(float(baselines.get("linear_projection") or 0.0), float(current_score), 1.0)
    pace_projection = max(float(baselines.get("pace_6h_projection") or 0.0), float(current_score))
    pace_ratio = pace_projection / linear_projection
    slowdown = max(min((1.0 - pace_ratio) / 0.18, 1.0), 0.0)
    progress_weight = max(min((float(progress) - FRONT_PARKING_MIN_PROGRESS) / 0.45, 1.0), 0.0)

    if rank <= 20:
        rank_weight = 0.78
        slowdown_weight = max(slowdown, 0.72)
    elif rank <= 30:
        rank_weight = 0.58
        slowdown_weight = max(slowdown, 0.45)
    else:
        # T40-T99 can still be a real race. Only damp clearly optimistic
        # projections instead of treating them as safe parking by default.
        if slowdown < 0.10 or prediction < linear_projection * 1.02:
            return None
        rank_weight = max((100.0 - float(rank)) / 70.0, 0.0) * 0.75
        slowdown_weight = slowdown

    strength = max(min(rank_weight * slowdown_weight * progress_weight, 0.82), 0.0)
    if strength <= 0.0:
        return None

    pace_ceiling = max(float(current_score), min(pace_projection, linear_projection))
    baseline_current_weight = 0.58 + 0.22 * slowdown
    parking_baseline = log_blend(float(current_score), pace_ceiling, baseline_current_weight)
    cap = log_blend(parking_baseline, float(prediction), strength)
    cap = max(float(current_score), cap)
    if cap >= float(prediction):
        return None
    return cap


def late_stage_pace_cap(
    prediction: float,
    current_score: int,
    rank: int,
    progress: float,
    wall_progress: float,
    baselines: dict[str, float],
    pressure_multiplier: float,
    current_wall_timestamp: int,
    line_end_timestamp: int,
    region: str,
    series_type: str,
) -> float:
    seconds_remaining = line_end_timestamp - current_wall_timestamp
    if seconds_remaining < 0:
        return float(current_score)

    progress_value = max(clamp(progress, 0.0, 1.0), clamp(wall_progress, 0.0, 1.0))
    time_guard_active = seconds_remaining <= TAIL_PACE_GUARD_SECONDS
    progress_guard_start = late_progress_guard_start(region)
    progress_guard_active = progress_value >= progress_guard_start
    if not time_guard_active and not progress_guard_active:
        return float(prediction)

    capped_prediction = float(prediction)
    pace_projection = max(float(current_score), float(baselines.get("pace_6h_projection") or 0.0))
    if pace_projection <= float(current_score):
        capped_prediction = max(float(current_score), min(capped_prediction, pace_projection))

    linear_projection = max(float(current_score), float(baselines.get("linear_projection") or 0.0), 1.0)
    pace_uplift = pace_projection / linear_projection
    force_guard = seconds_remaining <= TAIL_PACE_GUARD_FORCE_SECONDS
    if time_guard_active and (
        force_guard or pace_uplift >= 1.0 + TAIL_PACE_VISIBLE_UPLIFT
    ):
        window_strength = 1.0 - max(seconds_remaining, 0) / float(TAIL_PACE_GUARD_SECONDS)
        guard_strength = max(min((pace_uplift - (1.0 + TAIL_PACE_VISIBLE_UPLIFT)) / 0.12, 1.0), 0.0)
        if force_guard:
            guard_strength = max(guard_strength, 0.6)
        # The recent 6h pace already includes part of the closing rush. Keep
        # only a small remaining-tail margin so late predictions do not count
        # the same rush twice.
        margin_start = 0.045 if is_world_bloom_like(series_type) else 0.060
        margin_end = 0.012 if is_world_bloom_like(series_type) else 0.018
        margin = margin_start * (1.0 - window_strength) + margin_end * window_strength
        margin += (1.0 - guard_strength) * (0.015 if is_world_bloom_like(series_type) else 0.022)
        reserve_by_rank = {
            100: 0.006,
            200: 0.010,
            300: 0.013,
            400: 0.016,
            500: 0.020,
            1000: 0.030,
            2000: 0.026,
            3000: 0.022,
            4000: 0.020,
            5000: 0.018,
            10000: 0.014,
            100000: 0.008,
        }
        pressure_reserve = max(min(float(pressure_multiplier) - 1.0, 0.12), 0.0)
        pressure_reserve *= 0.50 if int(rank) <= 1000 else 0.35
        reserve_base = reserve_by_rank.get(int(rank), 0.012) + pressure_reserve
        if is_world_bloom_like(series_type):
            reserve_base *= WORLD_BLOOM_PRESSURE_EXTRA_SCALE
        # The recent pace already contains part of the tail rush. Keep only a
        # remaining-tail reserve, and shrink it when the 6h pace is visibly lifted.
        reserve_window = max(min(seconds_remaining / float(TAIL_PACE_GUARD_SECONDS), 1.0), 0.0)
        tail_reserve = float(current_score) * reserve_base * reserve_window * (1.0 - 0.75 * guard_strength)
        cap = float(current_score) + (pace_projection - float(current_score)) * (1.0 + margin) + tail_reserve
        capped_prediction = min(capped_prediction, cap)

    if progress_guard_active:
        progress_strength = clamp(
            (progress_value - progress_guard_start) / max(1.0 - progress_guard_start, 1e-6),
            0.0,
            1.0,
        )
        hybrid_projection = max(float(current_score), float(baselines.get("hybrid_projection") or 0.0))
        shape_projection = max(float(current_score), float(baselines.get("jp_shape_projection") or 0.0))
        shape_base = max(hybrid_projection, shape_projection)
        start_pace_weight, end_pace_weight, reserve_start, reserve_end = late_stage_rank_profile(rank)
        pace_weight = start_pace_weight * (1.0 - progress_strength) + end_pace_weight * progress_strength
        progress_cap = log_blend(pace_projection, shape_base, pace_weight)
        reserve = reserve_start * (1.0 - progress_strength) + reserve_end
        reserve += max(min(float(pressure_multiplier) - 1.0, 0.08), 0.0) * (
            0.30 if int(rank) <= 1000 else 0.20
        )
        if is_world_bloom_like(series_type):
            reserve *= WORLD_BLOOM_PRESSURE_EXTRA_SCALE
        progress_cap = max(float(current_score), progress_cap * (1.0 + reserve))
        capped_prediction = min(capped_prediction, progress_cap)

    return max(float(current_score), capped_prediction)


def rank_consistency_gap(
    better_rank_row: dict[str, object],
    lower_rank_row: dict[str, object],
) -> float:
    better_current = float(better_rank_row.get("current_score") or 0.0)
    lower_current = float(lower_rank_row.get("current_score") or 0.0)
    current_gap = 0.0
    if better_current > lower_current > 0.0:
        current_gap = math.log(better_current / lower_current)
    rank_gap = math.log(
        max(float(lower_rank_row["rank"]), 1.0) / max(float(better_rank_row["rank"]), 1.0)
    )
    return min(max(current_gap * 0.35, rank_gap * 0.006, 0.0015), 0.04)


def apply_rank_consistency_smoothing(predictions: list[dict[str, object]]) -> None:
    by_line: dict[int, list[dict[str, object]]] = {}
    for row in predictions:
        by_line.setdefault(int(row.get("character_id") or 0), []).append(row)

    for rows in by_line.values():
        if all(bool(row.get("event_finished_prediction")) for row in rows):
            for row in rows:
                row["rank_consistency_floor"] = None
            continue
        lower_rank_row: dict[str, object] | None = None
        lower_rank_floor = 0.0
        for row in sorted(rows, key=lambda item: int(item["rank"]), reverse=True):
            prediction = float(row["model_prediction"])
            required = lower_rank_floor
            if lower_rank_row is not None:
                required = lower_rank_floor * math.exp(rank_consistency_gap(row, lower_rank_row))
            cap = row.get("front_parking_cap")
            envelope_floor = float(row.get("shape_envelope_applied_prediction") or 0.0)
            if cap is not None:
                capped_prediction = min(prediction, max(float(cap), envelope_floor))
            else:
                capped_prediction = prediction
            adjusted_prediction = max(required, capped_prediction)
            if adjusted_prediction != prediction:
                row["rank_consistency_floor"] = required if adjusted_prediction == required else None
                row["front_parking_cap_applied"] = adjusted_prediction if adjusted_prediction < prediction else None
                row["pressure_adjusted_prediction"] = adjusted_prediction
                row["model_prediction"] = adjusted_prediction
                lower_rank_floor = adjusted_prediction
            else:
                row["rank_consistency_floor"] = None
                row["front_parking_cap_applied"] = None
                lower_rank_floor = prediction
            lower_rank_row = row


def select_payload_for_series(payload: dict[str, object], series_type: str) -> dict[str, object]:
    series_payloads = payload.get("series_payloads")
    if isinstance(series_payloads, dict):
        series_payload = series_payloads.get(series_type)
        if isinstance(series_payload, dict):
            return series_payload
    return payload


def deserialize_shape_priors(module, payload: object):
    if not isinstance(payload, dict):
        return None
    priors = {}
    for group, raw_curves in payload.items():
        if not isinstance(group, str) or not isinstance(raw_curves, dict):
            return None
        curves = {}
        for key, raw_curve in raw_curves.items():
            if not isinstance(raw_curve, dict):
                return None
            curve = module.ShapeCurve(
                progresses=[float(item) for item in raw_curve.get("progresses", [])],
                log_multipliers=[float(item) for item in raw_curve.get("log_multipliers", [])],
                counts=[int(item) for item in raw_curve.get("counts", [])],
            )
            curves[key] = curve
        priors[group] = curves
    return priors


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict current event lines using the trained baseline model.")
    parser.add_argument('--speed-api-base-url', default=None)
    parser.add_argument('--speed-state-dir', type=Path, default=Path(__file__).resolve().parents[1]/'state'/'live_speed')
    parser.add_argument("--region", default="cn")
    parser.add_argument("--event-id", type=int, required=True)
    parser.add_argument(
        "--history-tsv",
        type=Path,
        default=None,
        help="Optional TSV: rank,timestamp,score or character_id,rank,timestamp,score. Defaults to current SQL dump.",
    )
    parser.add_argument(
        "--model-pkl",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "baseline_model.pkl",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path(__file__).resolve().parents[0] / "train_baseline.py",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/deseer/PycharmProjects/Haruki-Tracker-Exports"),
    )
    parser.add_argument(
        "--current-sql-zst",
        type=Path,
        default=None,
        help="Optional current tracker SQL dump for live boundary user-turnover features.",
    )
    parser.add_argument(
        "--now-timestamp",
        type=int,
        default=None,
        help="Override current wall-clock timestamp for replay/debug runs.",
    )
    parser.add_argument(
        "--live-calibration",
        type=Path,
        default=None,
        help="Optional online calibration JSON built from finished-event replay errors.",
    )
    parser.add_argument(
        "--leaderboard-scope",
        choices=("auto", "total", "chapter"),
        default="auto",
        help="For World Bloom, predict total ranking or per-chapter character ranking.",
    )
    args = parser.parse_args()

    module = load_training_module(args.train_script.resolve())
    with open(args.model_pkl, "rb") as handle:
        payload = pickle.load(handle)
    live_meta = load_event_meta_from_masterdata(module, args.region, args.event_id)
    if live_meta is None:
        live_meta = load_event_meta_from_manifest(args.data_root.resolve(), module, args.region, args.event_id)
    event_series_type = live_meta.series_type
    leaderboard_scope = normalize_leaderboard_scope(event_series_type, args.leaderboard_scope)
    is_wl_total = event_series_type == "world_bloom" and leaderboard_scope == "total"
    model_meta = meta_for_leaderboard_scope(module, live_meta, leaderboard_scope)
    effective_payload = select_payload_for_series(payload, model_meta.series_type)
    world_bloom_payload = select_payload_for_series(payload, "world_bloom")
    adjustment_series_type = WORLD_BLOOM_TOTAL_SERIES_TYPE if is_wl_total else model_meta.series_type
    model_name = effective_payload["model_name"]
    model_series_types = frozenset(str(item) for item in effective_payload.get("model_series_types", [model_meta.series_type]))
    live_calibration = load_live_calibration(args.live_calibration.resolve() if args.live_calibration else None)

    event_feature_contexts = module.load_event_feature_contexts()
    adjustment_event_feature_contexts = dict(event_feature_contexts)
    if is_wl_total:
        key = (args.region, args.event_id)
        current_context = event_feature_contexts.get(key, module.EventFeatureContext())
        event_feature_contexts[key] = module.EventFeatureContext(
            character_ids=current_context.character_ids,
            unit_labels=current_context.unit_labels,
            main_character_ids=current_context.main_character_ids,
            condition="other",
        )
    payload_event_feature_contexts = module.deserialize_event_feature_contexts(
        effective_payload.get("live_event_feature_contexts")
    )
    if payload_event_feature_contexts:
        payload_event_feature_contexts.update(event_feature_contexts)
    else:
        payload_event_feature_contexts = event_feature_contexts
    effective_payload = dict(effective_payload)
    effective_payload["live_event_feature_contexts"] = payload_event_feature_contexts
    character_heat_priors = effective_payload.get("live_character_heat_priors")
    jp_shape_priors = deserialize_shape_priors(module, effective_payload.get("live_jp_shape_priors"))
    adjustment_character_heat_priors = (
        world_bloom_payload.get("live_character_heat_priors")
        if is_wl_total and isinstance(world_bloom_payload, dict)
        else character_heat_priors
    )
    adjustment_jp_shape_priors = (
        deserialize_shape_priors(module, world_bloom_payload.get("live_jp_shape_priors"))
        if is_wl_total and isinstance(world_bloom_payload, dict)
        else jp_shape_priors
    )
    if not isinstance(character_heat_priors, dict) or jp_shape_priors is None:
        finished_event_ids = module.load_finished_event_ids()
        event_meta = module.load_event_meta(args.data_root.resolve(), finished_event_ids)
        normalized_event_keys = module.load_normalized_rank_timeseries_event_keys(args.data_root.resolve())
        legacy_event_files = module.load_legacy_event_files(args.data_root.resolve(), normalized_event_keys)
        selected_ranks = set(module.COMMON_RANKS)
        finals = module.build_final_targets(args.data_root.resolve(), event_meta, selected_ranks, legacy_event_files)
        # Live prediction should use every completed event as prior data. The
        # holdout split is only for offline validation. New model files cache
        # these priors so routine live refreshes do not rescan all history.
        validation_events = {region: set() for region in module.REGIONS}
        character_heat_priors = module.build_character_heat_priors(
            event_meta=event_meta,
            finals=finals,
            validation_events=validation_events,
            event_feature_contexts=event_feature_contexts,
            model_series_types=model_series_types,
        )
        jp_shape_priors = module.build_jp_shape_priors(
            data_root=args.data_root.resolve(),
            event_meta=event_meta,
            finals=finals,
            validation_events=validation_events,
            selected_ranks=selected_ranks,
            legacy_event_files=legacy_event_files,
            model_series_types=model_series_types,
        )
    if not isinstance(adjustment_character_heat_priors, dict):
        adjustment_character_heat_priors = character_heat_priors
    if adjustment_jp_shape_priors is None:
        adjustment_jp_shape_priors = jp_shape_priors

    current_sql_zst = args.current_sql_zst
    if current_sql_zst is None:
        current_sql_zst = (
            args.data_root.resolve()
            / "regions"
            / args.region
            / "raw"
            / "current"
            / f"{args.region}_event_db.sql.zst"
        )
    if args.history_tsv is not None:
        rank_history = load_rank_history(args.history_tsv.resolve())
    else:
        rank_history = load_current_rank_history(
            module,
            args.region,
            args.event_id,
            model_meta.series_type,
            current_sql_zst,
        )
    synthetic_ranks = set()
    for rank in module.COMMON_RANKS:
        synthetic_ranks.update(interpolate_missing_line_histories(rank_history, int(rank)))
    boundary_pressure_contexts = (
        {}
        if model_meta.series_type == "world_bloom"
        else load_current_boundary_pressure_contexts(
            module,
            args.region,
            args.event_id,
            model_meta.series_type,
            current_sql_zst,
        )
    )

    predictions = []
    current_wall_timestamp = int(args.now_timestamp or time.time())
    for (character_id, rank), history_points in sorted(rank_history.items(), key=lambda item: (item[0][0], item[0][1])):
        if not module.has_event_line_window(model_meta, character_id):
            continue
        if len(history_points) < 2:
            continue
        current_timestamp, current_score = history_points[-1]
        history_state = module.HistoryState(
            timestamps=[ts for ts, _ in history_points[:-1]],
            scores=[sc for _, sc in history_points[:-1]],
            next_bucket_index=0,
        )
        progress = module.event_line_progress(model_meta, character_id, current_timestamp)
        line_end_timestamp = module.event_line_end_timestamp(model_meta, character_id)
        wall_progress = module.event_line_progress(
            model_meta,
            character_id,
            min(current_wall_timestamp, line_end_timestamp),
        )
        event_finished = current_wall_timestamp >= line_end_timestamp
        row = {
            "region": args.region,
            "event_id": args.event_id,
            "series_type": model_meta.series_type,
            "character_id": character_id,
            "rank": rank,
            "timestamp": current_timestamp,
            "score": current_score,
        }
        feature_vector, baselines = module.make_feature_vector(
            row=row,
            meta=model_meta,
            history=history_state,
            bucket_progress=progress,
            event_feature_contexts=event_feature_contexts,
            character_heat_priors=character_heat_priors,
            jp_shape_priors=jp_shape_priors,
            boundary_pressure_contexts=boundary_pressure_contexts,
        )
        x = np.asarray([feature_vector], dtype=np.float64)
        prediction_row = {
            "region": args.region,
            "event_id": args.event_id,
            "series_type": model_meta.series_type,
            "event_series_type": event_series_type,
            "leaderboard_scope": leaderboard_scope,
            "character_id": character_id,
            "progress": progress,
            "wall_progress": wall_progress,
            "rank": rank,
            "current_timestamp": current_timestamp,
            "end_timestamp": line_end_timestamp,
            "current_calendar_segment": module.local_calendar_segment(args.region, current_timestamp),
            "end_calendar_segment": module.local_calendar_segment(args.region, line_end_timestamp - 1),
        }
        uncalibrated_payload = dict(effective_payload)
        uncalibrated_payload["rank_boundary_calibration"] = None
        uncalibrated_payload["wl2_character_prior_enabled"] = False
        raw_pred = float(module.predict_payload_scores(uncalibrated_payload, x, [prediction_row])[0])
        calibrated_pred = float(module.predict_payload_scores(effective_payload, x, [prediction_row])[0])
        quantile_scores = module.predict_quantile_scores(
            effective_payload, x, [prediction_row], np.asarray([float(current_score)]))
        if quantile_scores is not None and 0.1 in quantile_scores and 0.9 in quantile_scores:
            # never publish an interval that contradicts the point prediction
            median = float(quantile_scores.get(0.5, quantile_scores[0.1])[0])
            if not (0.5 * calibrated_pred <= median <= 2.0 * calibrated_pred):
                quantile_scores = None
        quantile_point_min = float(os.environ.get("HARUKI_FORECAST_QUANTILE_POINT_MIN", "0") or 0.0)
        if (
            quantile_scores is not None
            and quantile_point_min > 0.0
            and progress >= quantile_point_min
            and 0.5 in quantile_scores
        ):
            # offline replay: the median-loss model beats squared error late
            # (analysis/offline_intrinsic_live/experiment_structure.py)
            calibrated_pred = max(float(quantile_scores[0.5][0]), float(current_score))
        wl2_prior_multiplier = module.wl2_character_prior_multiplier(prediction_row, effective_payload)
        final_pred = max(calibrated_pred, float(current_score))
        calibration_config = payload.get("rank_boundary_calibration")
        boundary_multiplier = (
            module.rank_boundary_calibration_multiplier(prediction_row, calibration_config)
            if isinstance(calibration_config, dict)
            else 1.0
        )
        pressure_features = live_boundary_pressure_features(
            module,
            event_key=(args.region, args.event_id),
            character_id=character_id,
            rank=rank,
            timestamp=current_timestamp,
            boundary_pressure_contexts=boundary_pressure_contexts,
        )
        pressure_multiplier = live_boundary_pressure_multiplier(
            args.region,
            rank,
            progress,
            pressure_features,
            adjustment_series_type,
        )
        shape_first_pred = world_bloom_shape_first_prediction(current_score, baselines)
        shape_envelope_debug = shape_envelope_anchor(current_score, baselines)
        world_bloom_model_floor = None
        if model_meta.series_type == "world_bloom":
            final_pred = max(shape_first_pred, calibrated_pred)
            if final_pred > shape_first_pred:
                world_bloom_model_floor = final_pred
            pressure_multiplier = 1.0
        pressure_adjusted_pred = max(final_pred * pressure_multiplier, float(current_score))
        front_rank_shape_floor_value = None
        if model_meta.series_type != "world_bloom":
            adjusted_pred = front_rank_shape_floor(
                pressure_adjusted_pred,
                current_score,
                rank,
                progress,
                baselines,
                adjustment_series_type,
            )
            if adjusted_pred > pressure_adjusted_pred:
                front_rank_shape_floor_value = adjusted_pred
                pressure_adjusted_pred = adjusted_pred
        live_calibration_key = None
        live_calibration_log_adjustment = None
        live_calibration_factor = 1.0
        cn_170_wl_negative_calibration_skipped = False
        if live_calibration and not is_wl_total:
            live_calibration_factor, live_calibration_key, live_calibration_log_adjustment = live_calibration_multiplier(
                module,
                prediction_row,
                live_calibration,
            )
            if (
                live_calibration_factor < 1.0
                and is_cn_170_wl_opening_chapter(
                    args.region,
                    args.event_id,
                    leaderboard_scope,
                    progress,
                )
            ):
                cn_170_wl_negative_calibration_skipped = True
                live_calibration_factor = 1.0
                live_calibration_log_adjustment = 0.0
            if use_increment_calibration(prediction_row):
                pressure_adjusted_pred = apply_live_calibration(pressure_adjusted_pred, current_score, live_calibration_factor)
            else:
                pressure_adjusted_pred = max(float(current_score), pressure_adjusted_pred * live_calibration_factor)
        heat_prior = early_event_heat_prior_log(
            module,
            args.region,
            args.event_id,
            character_id,
            adjustment_event_feature_contexts,
            adjustment_character_heat_priors,
        )
        event_heat_prior_debug: dict[str, float | int | str | None]
        pressure_adjusted_pred, event_heat_prior_debug = event_heat_prior_live_adjustment(
            module,
            pressure_adjusted_pred,
            current_score,
            rank,
            progress,
            prediction_row,
            effective_payload,
        )
        early_prior_debug: dict[str, float | None]
        pressure_adjusted_pred, early_prior_debug = early_stage_prior_adjustment_with_region(
            pressure_adjusted_pred,
            current_score,
            rank,
            progress,
            baselines,
            heat_prior,
            adjustment_series_type,
            args.region,
        )
        world_bloom_total_debug: dict[str, float | int | None] = {}
        if is_wl_total:
            pressure_adjusted_pred, world_bloom_total_debug = world_bloom_total_shape_adjustment(
                module,
                pressure_adjusted_pred,
                current_score,
                rank,
                current_timestamp,
                live_meta,
                baselines,
                adjustment_character_heat_priors,
                adjustment_jp_shape_priors,
                world_bloom_payload if isinstance(world_bloom_payload, dict) else None,
            )
        cn_170_wl_opening_boost = 1.0
        if is_cn_170_wl_opening_chapter(
            args.region,
            args.event_id,
            leaderboard_scope,
            progress,
        ):
            cn_170_wl_opening_boost = cn_170_wl_opening_rank_boost(rank, progress)
            pressure_adjusted_pred = max(
                float(current_score),
                pressure_adjusted_pred * cn_170_wl_opening_boost,
            )
        mid_stage_shape_floor_value = None
        mid_stage_shaped_pred = mid_stage_shape_floor(
            pressure_adjusted_pred,
            current_score,
            rank,
            progress,
            baselines,
            heat_prior,
            adjustment_series_type,
        )
        if mid_stage_shaped_pred is not None:
            mid_stage_shape_floor_value = mid_stage_shaped_pred
            pressure_adjusted_pred = mid_stage_shaped_pred
        front_parking_cap_value = front_parking_prediction_cap(
            pressure_adjusted_pred,
            current_score,
            rank,
            progress,
            baselines,
            adjustment_series_type,
        )
        late_stage_pace_cap_value = None
        capped_pred = late_stage_pace_cap(
            pressure_adjusted_pred,
            current_score,
            rank,
            progress,
            wall_progress,
            baselines,
            pressure_multiplier,
            current_wall_timestamp,
            line_end_timestamp,
            args.region,
            adjustment_series_type,
        )
        if capped_pred < pressure_adjusted_pred:
            late_stage_pace_cap_value = capped_pred
            pressure_adjusted_pred = capped_pred
        if event_finished:
            pressure_adjusted_pred = float(current_score)
            quantile_scores = None  # the outcome is already observed; no interval
            world_bloom_model_floor = None
            front_rank_shape_floor_value = None
            late_stage_pace_cap_value = None
            front_parking_cap_value = None
            mid_stage_shape_floor_value = None
            live_calibration_key = None
            live_calibration_log_adjustment = None
            live_calibration_factor = 1.0
            event_heat_prior_debug = {
                **event_heat_prior_debug,
                "event_heat_prior_prediction": None,
                "event_heat_prior_blend_weight": 0.0,
            }
            early_prior_debug = {
                **early_prior_debug,
                "early_heat_multiplier": 1.0,
                "early_projection_target": None,
                "early_projection_strength": 0.0,
                **world_bloom_total_debug,
                "world_bloom_total_projection_target": None,
                "world_bloom_total_projection_strength": 0.0,
            }
        predictions.append(
            {
                "rank": rank,
                "character_id": character_id,
                "synthetic_rank": (character_id, rank) in synthetic_ranks,
                "event_series_type": event_series_type,
                "series_type": adjustment_series_type,
                "model_series_type": model_meta.series_type,
                "leaderboard_scope": leaderboard_scope,
                "progress": progress,
                "wall_progress": wall_progress,
                "current_timestamp": current_timestamp,
                "end_timestamp": line_end_timestamp,
                "current_score": current_score,
                "boundary_pressure_features": pressure_features,
                "linear_projection": baselines["linear_projection"],
                "pace_6h_projection": baselines["pace_6h_projection"],
                "jp_shape_projection": baselines["jp_shape_projection"],
                "hybrid_projection": baselines["hybrid_projection"],
                "shape_envelope_anchor": shape_envelope_debug["anchor"],
                "shape_envelope_early_anchor": shape_envelope_debug["early_anchor"],
                "shape_envelope_opening_anchor": shape_envelope_debug["opening_anchor"],
                "shape_envelope_target": None,
                "shape_envelope_strength": 0.0,
                "shape_envelope_mode": "pending",
                "shape_envelope_applied_prediction": None,
                "shape_envelope_signal_log_gap": 0.0,
                "shape_envelope_signal_positive_share": 0.0,
                "shape_envelope_signal_shape_model_gap": 0.0,
                "shape_envelope_signal_pace_model_gap": 0.0,
                "shape_first_prediction": shape_first_pred,
                "raw_model_prediction": raw_pred,
                "rank_boundary_multiplier": boundary_multiplier,
                "wl2_character_prior_multiplier": wl2_prior_multiplier,
                "calibrated_model_prediction": calibrated_pred,
                "prediction_p10": (float(quantile_scores[0.1][0])
                                   if quantile_scores and 0.1 in quantile_scores else None),
                "prediction_p50": (float(quantile_scores[0.5][0])
                                   if quantile_scores and 0.5 in quantile_scores else None),
                "prediction_p90": (float(quantile_scores[0.9][0])
                                   if quantile_scores and 0.9 in quantile_scores else None),
                "boundary_pressure_multiplier": pressure_multiplier,
                "world_bloom_model_floor": world_bloom_model_floor,
                "front_rank_shape_floor": front_rank_shape_floor_value,
                "mid_stage_shape_floor": mid_stage_shape_floor_value,
                "front_parking_cap": front_parking_cap_value,
                "front_parking_cap_applied": None,
                "late_stage_pace_cap": late_stage_pace_cap_value,
                "live_calibration_key": live_calibration_key,
                "live_calibration_log_adjustment": live_calibration_log_adjustment,
                "live_calibration_multiplier": live_calibration_factor,
                "live_calibration_policy": "remaining_increment_v2" if use_increment_calibration(prediction_row) else "legacy_total_v1",
                "short_term_trend": trend_diagnostics(history_points, current_wall_timestamp),
                "cn_170_wl_negative_calibration_skipped": cn_170_wl_negative_calibration_skipped,
                "cn_170_wl_opening_boost": cn_170_wl_opening_boost,
                **event_heat_prior_debug,
                **early_prior_debug,
                **world_bloom_total_debug,
                "event_finished_prediction": event_finished,
                "pressure_adjusted_prediction": pressure_adjusted_pred,
                "model_prediction": pressure_adjusted_pred,
            }
        )

    apply_shape_envelope_adjustments(predictions)
    apply_rank_consistency_smoothing(predictions)

    live_heat_summary = None
    if os.environ.get("HARUKI_FORECAST_LIVE_HEAT", "").lower() in {"1", "true", "yes", "on"}:
        from live_heat_layer import apply_live_heat
        live_heat_summary = apply_live_heat(predictions, enabled=True)
        # the layer can move different rank bands by different amounts, so restore
        # the per-character rank monotonicity the smoothing above had enforced
        apply_rank_consistency_smoothing(predictions)

    from regional_shape import projection as regional_shape_projection, adjust as regional_shape_adjust
    from opening_prior import adjust as opening_prior_adjust
    for row in predictions:
        start_timestamp, _ = module.event_line_window(live_meta, int(row["character_id"]))
        prior_context = adjustment_event_feature_contexts.get((args.region, args.event_id))
        value, debug = opening_prior_adjust(
            float(row["model_prediction"]), float(row["current_score"]), args.region, args.event_id,
            int(row["character_id"]), int(row["rank"]), adjustment_series_type,
            float(row["progress"]), start_timestamp,
            payload.get("opening_final_history", []) if args.region in payload.get("opening_prior_regions", ["cn"]) and adjustment_series_type == "normal" else [],
            condition=prior_context.condition if prior_context else None,
            scope_kind=("chapters" if live_meta.chapter_windows else "finale") if is_wl_total else None)
        row["model_prediction"] = row["pressure_adjusted_prediction"] = value
        row.update(debug)
        shape_weights = payload.get("regional_shape_weights", {}).get(args.region)
        if shape_weights and adjustment_series_type == "normal":
            context = adjustment_event_feature_contexts.get((args.region, args.event_id))
            shape, count = regional_shape_projection(
                payload.get("regional_shape_history", []), args.region, args.event_id,
                int(row["character_id"]), int(row["rank"]), adjustment_series_type,
                float(row["progress"]), start_timestamp, float(row["current_score"]),
                context.condition if context else "other")
            value, weight = regional_shape_adjust(value, float(row["current_score"]), shape,
                                                 float(row["progress"]), shape_weights)
            row["model_prediction"] = row["pressure_adjusted_prediction"] = value
            row.update(regional_shape_prediction=shape, regional_shape_weight=weight,
                       regional_shape_event_count=count)
        if debug.get("opening_prior_applied") or row.get("regional_shape_weight", 0) > 0:
            # The extrapolated model interval does not cover this historical cold-start prior.
            for key in ("prediction_p10", "prediction_p50", "prediction_p90"):
                row[key] = None
    if args.region == "cn" and adjustment_series_type == "normal":
        from median_heat import reference as heat_reference, posterior as heat_posterior, speed_heat_index
        final_history = payload.get("opening_final_history", [])
        historical_finals = {(r["region"], r["event_id"], r["rank"]): r["score"]
                             for r in final_history if r["series_type"] == "normal" and r["character_id"] == 0}
        heat_settings = payload.get("median_heat_settings", {})
        historical_meta = module.load_event_meta(args.data_root.resolve(), module.load_finished_event_ids())
        historical_meta[(args.region, args.event_id)] = live_meta
        for row in predictions:
            if row.get("synthetic_rank") or row.get("event_finished_prediction"):
                continue
            char, rank = int(row["character_id"]), int(row["rank"])
            seen = rank_history.get((char, rank), [])
            if len(seen) < 2:
                continue
            line_start, line_end = module.event_line_window(live_meta, char)
            context = adjustment_event_feature_contexts.get((args.region, args.event_id))
            ref = heat_reference(payload.get("regional_shape_history", []), historical_finals,
                                 args.region, args.event_id, rank, line_start,
                                 context.condition if context else "other", .25,
                                 adjustment_event_feature_contexts, historical_meta,
                                 heat_settings.get("unit_match", False), heat_settings.get("duration_power", 0.))
            target, debug = heat_posterior(seen, line_start, line_end, ref, 1., .25)
            row.update({"median_heat_" + k: v for k, v in debug.items()})
            row["median_heat_prediction"] = target
            daily = speed_heat_index(seen, line_start, line_end, ref, "daily")
            slope = speed_heat_index(seen, line_start, line_end, ref, "slope")
            row["median_heat_daily"] = daily.get("heat")
            row["median_heat_slope"] = slope.get("heat")
            row["median_heat_daily_windows"] = len(daily.get("windows", []))
            cumulative = debug.get("cumulative_heat")
            row["median_heat_rhythm_log_residual"] = (math.log(slope["heat"] / cumulative)
                if slope.get("heat") and cumulative else None)
            weights = heat_settings.get("weights", [0., 0., 0., 0.])
            p = float(row["progress"])
            stage = 0 if p < .02 else 1 if p < .2 else 2 if p < .75 else 3
            weight = float(weights[stage]) if heat_settings.get("enabled") else 0.
            if target is not None and weight > 0.:
                value = log_blend(float(row["model_prediction"]), target, 1. - weight)
                row["model_prediction"] = row["pressure_adjusted_prediction"] = max(float(row["current_score"]), value)
                for key in ("prediction_p10", "prediction_p50", "prediction_p90"):
                    row[key] = None
            row["median_heat_blend_weight"] = weight
    shape_live_summary = None
    segmented_heat_summary = None
    shape_live_path = os.environ.get("HARUKI_FORECAST_SHAPE_LIVE_MODEL")
    if shape_live_path and args.region == "cn" and adjustment_series_type == "normal":
        from shape_live_model import predict_rows as shape_live_predict, segment_diagnostics
        with Path(shape_live_path).open("rb") as handle:
            shape_live_payload = pickle.load(handle)
        current_context = adjustment_event_feature_contexts.get((args.region, args.event_id))
        if current_context is not None:
            shape_live_payload["contexts"][(args.region, args.event_id)] = {
                "condition": current_context.condition, "units": list(current_context.unit_labels),
                "main_characters": list(current_context.main_character_ids),
            }
        for event_row in module.load_json(module.MASTERDATA_ROOTS[args.region] / "events.json"):
            if int(event_row["id"]) == args.event_id:
                shape_live_payload["assets"][(args.region, args.event_id)] = event_row.get("assetbundleName")
                break
        line_start, line_end = module.event_line_window(live_meta, 0)
        count = shape_live_predict(shape_live_payload, args.region, args.event_id, line_start, line_end,
                                   current_wall_timestamp, rank_history, predictions)
        shape_live_summary = {"version": shape_live_payload["version"], "lines": count,
                              "rank_reconciliation": shape_live_payload.get("rank_reconciliation", "legacy"),
                              "training_events": shape_live_payload["training_events"],
                              "training_samples": shape_live_payload["training_samples"]}
        segmented_heat_summary = segment_diagnostics(shape_live_payload, args.region, args.event_id,
                                                      line_start, line_end, current_wall_timestamp, rank_history)
        if count:
            model_name += "+" + shape_live_payload["version"]
    opening_blend_summary = None
    opening_blend_path = os.environ.get("HARUKI_FORECAST_OPENING_BLEND")
    if opening_blend_path and Path(opening_blend_path).exists():
        import opening_blend
        opening_artifact = opening_blend.load(opening_blend_path)
        opening_context = adjustment_event_feature_contexts.get((args.region, args.event_id))
        count = opening_blend.apply(opening_artifact, predictions, args.region, args.event_id,
                                    lambda character_id: module.event_line_window(live_meta, character_id),
                                    opening_context.condition if opening_context else "other",
                                    list(opening_context.unit_labels) if opening_context else [],
                                    current_wall_timestamp, len(live_meta.chapter_windows))
        opening_blend_summary = {"version": opening_artifact["version"], "lines": count,
                                 "built_at": opening_artifact.get("built_at"),
                                 "training_events": opening_artifact.get("events")}
        if count:
            model_name += "+" + opening_artifact["version"]
    if shape_live_summary is not None and shape_live_summary["lines"] and shape_live_summary["rank_reconciliation"] == "isotonic":
        from shape_live_model import reconcile as shape_live_reconcile
        shape_live_reconcile(predictions)
    else:
        apply_rank_consistency_smoothing(predictions)

    from forecast_safety import protect
    previous = []
    elapsed = None
    suffix = '_total' if leaderboard_scope == 'total' and event_series_type == 'world_bloom' else ''
    previous_path = Path(__file__).resolve().parents[1] / 'live' / f'{args.region}_{args.event_id}{suffix}_prediction.json'
    try:
        previous_payload = json.loads(previous_path.read_text())
        # Replay must never consult a future live output.
        generated = previous_payload.get('generated_at')
        if generated is not None and 0 < current_wall_timestamp-float(generated) <= 1800:
            previous = previous_payload.get('predictions', [])
            elapsed = current_wall_timestamp-float(generated)
    except (OSError, ValueError, TypeError):
        pass
    protect(predictions, previous, elapsed)
    if args.speed_api_base_url and args.now_timestamp is None:
        from live_speed import collect, adjust
        try:
            collect(args.speed_api_base_url,args.region,args.event_id,predictions,args.speed_state_dir,args.history_tsv)
            adjust(predictions,int(time.time()),live_meta.chapter_windows)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            for row in predictions:
                row['live_speed_error']=type(exc).__name__
        # Reconcile downward only; speed caps must not be undone by rank floors.
        ceilings={}
        for row in sorted(predictions,key=lambda r:r['rank']):
            char=row['character_id'];ceiling=ceilings.get(char,float('inf'))
            row['model_prediction']=max(row['current_score'],min(row['model_prediction'],ceiling))
            row['pressure_adjusted_prediction']=row['model_prediction']
            remaining_hours=max(0.,(row['end_timestamp']-row['current_timestamp'])/3600)
            row['forecast_required_pace']=(row['model_prediction']-row['current_score'])/remaining_hours if remaining_hours else None
            ceilings[char]=row['model_prediction']

    median_heat_enabled = bool(payload.get("median_heat_settings", {}).get("enabled", False))
    payload = {
        "generated_at": current_wall_timestamp,
        "region": args.region,
        "event_id": args.event_id,
        "event_series_type": event_series_type,
        "leaderboard_scope": leaderboard_scope,
        "model_name": model_name,
        "predictions": predictions,
    }
    heat_rows = [r for r in predictions if r.get("median_heat_heat") is not None and not r.get("synthetic_rank")]
    if heat_rows:
        def median_heat_field(key):
            values = sorted(float(r[key]) for r in heat_rows if r.get(key) is not None)
            return float(np.median(values)) if values else None
        payload["median_heat"] = {
            "reference": 1.0,
            "estimated_heat": median_heat_field("median_heat_heat"),
            "daily_heat": median_heat_field("median_heat_daily"),
            "slope_heat": median_heat_field("median_heat_slope"),
            "rhythm_log_residual": median_heat_field("median_heat_rhythm_log_residual"),
            "reliability": median_heat_field("median_heat_reliability"),
            "lines": len(heat_rows),
            "point_correction_enabled": median_heat_enabled,
        }
    if live_heat_summary is not None:
        payload["live_heat"] = live_heat_summary
    if shape_live_summary is not None:
        payload["shape_live"] = shape_live_summary
    if segmented_heat_summary is not None:
        payload["segmented_heat"] = segmented_heat_summary
    if opening_blend_summary is not None:
        payload["opening_blend"] = opening_blend_summary
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
