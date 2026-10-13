#!/usr/bin/env python3

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import math
import os
import pickle
import re
import time
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_training_history import live_index, iter_rows as iter_live_history_rows
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


REGIONS = ("jp", "en", "tw", "kr", "cn")
REGION_INDEX = {region: idx for idx, region in enumerate(REGIONS)}
REGION_UTC_OFFSETS = {
    "jp": 9,
    "en": 0,
    "tw": 8,
    "kr": 9,
    "cn": 8,
}
REGION_BLEND_WEIGHTS = {
    "jp": 0.45,
    "en": 0.70,
    "tw": 0.70,
    "kr": 1.00,
    "cn": 0.65,
}
# Weight is applied to the linear residual branch. Lower values lean more on the
# shape-hybrid branch, which should dominate once the opening burst settles.
REGION_PROGRESS_BLEND_WEIGHTS = {
    "jp": (
        (0.08, 0.60),
        (0.18, 0.45),
        (0.75, 0.35),
        (1.01, 0.25),
    ),
    "en": (
        (0.08, 0.75),
        (0.18, 0.60),
        (0.75, 0.50),
        (1.01, 0.40),
    ),
    "tw": (
        (0.08, 0.75),
        (0.18, 0.60),
        (0.75, 0.50),
        (1.01, 0.40),
    ),
    "kr": (
        (0.08, 0.90),
        (0.18, 0.75),
        (0.75, 0.65),
        (1.01, 0.50),
    ),
    "cn": (
        (0.08, 0.70),
        (0.18, 0.45),
        (0.50, 0.25),
        (0.75, 0.15),
        (1.01, 0.05),
    ),
}
CN_PROGRESS_BLEND_WEIGHTS = (
    (0.08, 0.70),
    (0.18, 0.45),
    (0.50, 0.25),
    (0.75, 0.15),
    (1.01, 0.05),
)
CN_SIDECAR_REGION = "cn"
CN_SIDECAR_APPLY_MIN_RANK = 5000
CN_SIDECAR_BASE_WEIGHT = 0.60
CHARACTER_HEAT_LOCAL_WINDOW = 9
UNIT_HEAT_SHRINKAGE = 4.0
UNIT_HEAT_PRIOR_WEIGHT = 4.0
USE_UNIT_HEAT_PRIORS_IN_MODEL = False
UNIT_HEAT_PRIOR_WEIGHT_BY_CONDITION = {
    "box": 24.0,
    "mixed": 6.0,
    "other": 12.0,
    "world_bloom": 12.0,
}
UNIT_HEAT_MIN_ORDER_GAP = 0.006
CHARACTER_HEAT_SHRINKAGE = 5.0
CHARACTER_HEAT_PRIOR_WEIGHT = 2.0
MAIN_CHARACTER_HEAT_PRIOR_WEIGHT = 2.5
CHARACTER_RESIDUAL_CLIP = 0.75
EVENT_HEAT_PRIOR_RECENT_WINDOW = 8
EVENT_HEAT_PRIOR_CONDITION_MIN_COUNT = 3
EVENT_HEAT_PRIOR_RECENCY_DECAY = 0.82
EVENT_HEAT_PRIOR_ENABLED = os.environ.get(
    "HARUKI_FORECAST_ENABLE_EVENT_HEAT_PRIOR",
    "",
).lower() in {"1", "true", "yes", "on"}
USE_POPULARITY_PRIORS_IN_MODEL = False
UNIT_COMPONENT_SUBTRACTION_BY_CONDITION = {
    "mixed": 1.00,
    "box": 0.35,
    "other": 0.50,
    "world_bloom": 0.35,
}
HEAT_EVENT_RESIDUAL_ABS_LIMIT = 4.0
HEAT_EVENT_VALUE_CLIP = 1.5
VOCALOID_CHARACTER_IDS = frozenset({21, 22, 23, 24, 25, 26})
UNIT_HEAT_ORDER = ("school_refusal", "theme_park", "street", "light_sound", "idol")
UNIT_HEAT_PRIOR = {
    "school_refusal": 0.180,
    "theme_park": 0.140,
    "street": 0.100,
    "light_sound": 0.055,
    "idol": 0.015,
}
JP_WL2_T100_SCORES = {
    1: 142_687_808,
    2: 137_018_749,
    3: 93_001_027,
    4: 101_828_138,
    5: 152_400_414,
    6: 165_535_105,
    7: 133_430_319,
    8: 141_161_556,
    9: 118_686_000,
    10: 128_942_204,
    11: 164_144_815,
    12: 132_479_680,
    13: 169_486_295,
    14: 110_147_110,
    15: 158_816_100,
    16: 171_188_883,
    17: 161_609_433,
    18: 122_735_877,
    19: 166_999_355,
    20: 170_030_587,
}
JP_WL2_CHARACTER_PRIOR_SCALE = 0.30
JP_WL2_CHARACTER_PRIOR_CLIP = 0.12
_JP_WL2_LOG_SCORES = sorted(math.log(float(score)) for score in JP_WL2_T100_SCORES.values())
_JP_WL2_LOG_MEDIAN = (
    _JP_WL2_LOG_SCORES[len(_JP_WL2_LOG_SCORES) // 2 - 1]
    + _JP_WL2_LOG_SCORES[len(_JP_WL2_LOG_SCORES) // 2]
) / 2.0
CHARACTER_POPULARITY_PRIOR = {
    character_id: min(
        max(
            (math.log(float(score)) - _JP_WL2_LOG_MEDIAN) * JP_WL2_CHARACTER_PRIOR_SCALE,
            -JP_WL2_CHARACTER_PRIOR_CLIP,
        ),
        JP_WL2_CHARACTER_PRIOR_CLIP,
    )
    for character_id, score in JP_WL2_T100_SCORES.items()
}
WORLD_BLOOM_WL2_EVENT_IDS = frozenset({163, 167, 170, 171, 176, 179})
WORLD_BLOOM_WL3_EVENT_IDS = frozenset({202})
WORLD_BLOOM_WL2_CHARACTER_PRIOR_ENABLED = False
WORLD_BLOOM_WL2_CHARACTER_PRIOR_REGION_WEIGHTS = {
    "jp": 1.00,
    "en": 0.65,
    "tw": 0.35,
    "kr": 0.60,
    "cn": 0.75,
}
WORLD_BLOOM_WL2_CHARACTER_PRIOR_RANK_WEIGHTS = (
    (100, 0.75),
    (300, 0.65),
    (500, 0.55),
    (1000, 0.40),
    (5000, 0.25),
    (10000, 0.18),
    (10**9, 0.10),
)
WORLD_BLOOM_WL2_CHARACTER_PRIOR_PROGRESS_FLOOR = 0.55
WORLD_BLOOM_WL2_CHARACTER_PRIOR_PROGRESS_DECAY = 0.45
BASE_NUMERIC_FEATURE_COUNT = 23
TEMPORAL_FEATURE_COUNT = 15
PACE_TREND_FEATURE_COUNT = 5
EDGE_FEATURE_COUNT = 8
HEAT_FEATURE_COUNT = 24
BOUNDARY_PRESSURE_FEATURE_COUNT = 8
SHAPE_HYBRID_LOG_OFFSET = 7
CURRENT_SCORE_LOG_FEATURE_INDEX = 13
LINEAR_LOG_FEATURE_INDEX = 14
PACE_6H_PROJECTION_LOG_FEATURE_INDEX = 16
HOURS_ELAPSED_FEATURE_INDEX = 10
HOURS_REMAINING_FEATURE_INDEX = 11
PACE_1H_FEATURE_INDEX = 21
PACE_6H_FEATURE_INDEX = 22
PACE_24H_FEATURE_INDEX = 23
PREV_PACE_FEATURE_INDEX = 24
HYBRID_LOG_FEATURE_INDEX = (
    len(REGIONS)
    + BASE_NUMERIC_FEATURE_COUNT
    + TEMPORAL_FEATURE_COUNT
    + PACE_TREND_FEATURE_COUNT
    + EDGE_FEATURE_COUNT
    + HEAT_FEATURE_COUNT
    + SHAPE_HYBRID_LOG_OFFSET
)
NORMAL_MODEL_SERIES_TYPES = frozenset({"normal"})
WL_MODEL_SERIES_TYPES = frozenset({"world_bloom"})
WORLD_BLOOM_MIN_FINAL_PROGRESS = 0.98
MANUAL_MAIN_CHARACTER_OVERRIDES = {
    ("cn", 150): frozenset({19, 20}),
}
def env_path(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default)))


MASTERDATA_ROOTS = {
    "jp": env_path("HARUKI_FORECAST_MASTERDATA_JP", "/Users/deseer/PycharmProjects/haruki-sekai-master/master"),
    "en": env_path("HARUKI_FORECAST_MASTERDATA_EN", "/Users/deseer/PycharmProjects/haruki-sekai-en-master/master"),
    "tw": env_path("HARUKI_FORECAST_MASTERDATA_TW", "/Users/deseer/PycharmProjects/haruki-sekai-tc-master/master"),
    "kr": env_path("HARUKI_FORECAST_MASTERDATA_KR", "/Users/deseer/PycharmProjects/haruki-sekai-kr-master/master"),
    "cn": env_path("HARUKI_FORECAST_MASTERDATA_CN", "/Users/deseer/PycharmProjects/haruki-sekai-sc-master/master"),
}
SEKAI_BEST_JP_WL2_DATA_ROOT = env_path(
    "HARUKI_FORECAST_SEKAI_BEST_JP_WL2_DIR",
    Path(__file__).resolve().parents[1] / "data" / "sekai_best_jp_wl2",
)
SEKAI_BEST_WORLD_BLOOM_DATA_ROOT = env_path(
    "HARUKI_FORECAST_SEKAI_BEST_WORLD_BLOOM_DIR",
    Path(__file__).resolve().parents[1] / "data" / "sekai_best_world_bloom",
)
COMMON_RANKS = (
    1,
    2,
    3,
    10,
    20,
    30,
    50,
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
    20000,
    30000,
    50000,
    100000,
    200000,
    300000,
)
BUCKETS = tuple(round(i * 0.05, 2) for i in range(1, 20))
BOUNDARY_PRESSURE_RANKS = frozenset({100, 400, 1000, 4000})
BOUNDARY_PRESSURE_TOP100_MIN_RANK = 90
LOOKBACK_WINDOWS = (
    ("1h", 3600),
    ("6h", 6 * 3600),
    ("24h", 24 * 3600),
)
LEGACY_SOURCES = ("downloads-backup-2", "downloads-backup")
LEGACY_EVENT_RE = re.compile(r"(?:^|\.)(event_\d+|wl_\d+)\.(\d{5})\.sql$")
LEGACY_NORMAL_ROW_RE = re.compile(
    r"\((\d+),(?:\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'),(\d+),(\d+)\)"
)
LEGACY_WL_ROW_RE = re.compile(
    r"\((\d+),(?:\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'),(\d+),(\d+),(\d+)\)"
)
RANK_BOUNDARY_CALIBRATION = {
    100: {
        "base": 0.040,
        "late_start": 0.70,
        "late": 0.006,
        "final_start": 0.90,
        "final": 0.024,
        "weekend_late": 0.004,
        "weekend_final": 0.014,
        "region_weights": {"jp": 0.00, "en": 0.18, "tw": 1.25, "kr": 0.78, "cn": 1.32},
        "max_multiplier": 1.18,
    },
    1000: {
        "base": 0.026,
        "late_start": 0.70,
        "late": 0.005,
        "final_start": 0.90,
        "final": 0.024,
        "weekend_late": 0.003,
        "weekend_final": 0.012,
        "region_weights": {"jp": 0.55, "en": 0.18, "tw": 0.68, "kr": 1.40, "cn": 2.05},
        "max_multiplier": 1.16,
    },
}


@dataclass
class EventMeta:
    region: str
    event_id: int
    series_type: str
    start_timestamp: int
    end_timestamp: int
    chapter_windows: dict[int, tuple[int, int]] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> int:
        return max(self.end_timestamp - self.start_timestamp, 1)

    @property
    def duration_hours(self) -> float:
        return self.duration_seconds / 3600.0


def event_line_window(meta: EventMeta, character_id: int) -> tuple[int, int]:
    if meta.series_type == "world_bloom":
        window = meta.chapter_windows.get(int(character_id))
        if window is not None and window[0] < window[1]:
            return window
    return meta.start_timestamp, meta.end_timestamp


def has_event_line_window(meta: EventMeta, character_id: int) -> bool:
    if meta.series_type != "world_bloom":
        return True
    window = meta.chapter_windows.get(int(character_id))
    return window is not None and window[0] < window[1]


def event_line_duration_seconds(meta: EventMeta, character_id: int) -> int:
    start_timestamp, end_timestamp = event_line_window(meta, character_id)
    return max(end_timestamp - start_timestamp, 1)


def event_line_progress(meta: EventMeta, character_id: int, timestamp: int) -> float:
    start_timestamp, end_timestamp = event_line_window(meta, character_id)
    duration_seconds = max(end_timestamp - start_timestamp, 1)
    return min(max((int(timestamp) - start_timestamp) / duration_seconds, 0.0), 1.0)


def event_line_end_timestamp(meta: EventMeta, character_id: int) -> int:
    return event_line_window(meta, character_id)[1]


def is_world_bloom_series_complete(
    meta: EventMeta,
    character_id: int,
    timestamp: int,
    min_final_progress: float = WORLD_BLOOM_MIN_FINAL_PROGRESS,
) -> bool:
    if meta.series_type != "world_bloom":
        return True
    if not has_event_line_window(meta, character_id):
        return False
    return event_line_progress(meta, character_id, timestamp) >= min_final_progress


@dataclass
class HistoryState:
    timestamps: list[int] = field(default_factory=list)
    scores: list[int] = field(default_factory=list)
    next_bucket_index: int = 0


@dataclass
class ShapeCurve:
    progresses: list[float] = field(default_factory=list)
    log_multipliers: list[float] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)


@dataclass
class BoundaryPressureSeries:
    timestamps: list[int] = field(default_factory=list)
    features: list[tuple[float, ...]] = field(default_factory=list)


@dataclass
class EventFeatureContext:
    character_ids: tuple[int, ...] = field(default_factory=tuple)
    unit_labels: tuple[str, ...] = field(default_factory=tuple)
    main_character_ids: tuple[int, ...] = field(default_factory=tuple)
    condition: str = "other"

    @property
    def featured_character_count(self) -> int:
        return len(self.character_ids)

    @property
    def featured_unit_count(self) -> int:
        return len(self.unit_labels)

    @property
    def main_character_count(self) -> int:
        return len(self.main_character_ids)


def parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(text)


def masterdata_row_value(row: Any, key: str, list_index: int) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    if isinstance(row, (list, tuple)) and len(row) > list_index:
        return row[list_index]
    return None


def safe_log1p(value: float) -> float:
    return math.log1p(max(value, 0.0))


def signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def np_safe_log1p(values: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.log1p(np.maximum(arr, 0.0))


def np_signed_log1p(values: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.sign(arr) * np.log1p(np.abs(arr))


def shrink_mean(values: list[float], shrinkage: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    shrink = float(arr.size) / (float(arr.size) + float(shrinkage))
    return float(np.mean(arr) * shrink)


def clipped_mean(values: list[float], clip_abs: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    if arr.size >= 4:
        lower, upper = np.quantile(arr, [0.15, 0.85])
        arr = np.clip(arr, lower, upper)
    arr = np.clip(arr, -float(clip_abs), float(clip_abs))
    return float(np.mean(arr))


def regularized_heat_mean(
    values: list[float],
    shrinkage: float,
    *,
    prior: float = 0.0,
    prior_weight: float = 0.0,
    clip_abs: float = CHARACTER_RESIDUAL_CLIP,
) -> float:
    if not values and prior_weight <= 0.0:
        return 0.0
    data_weight = float(len(values))
    total_weight = data_weight + float(shrinkage) + float(prior_weight)
    if total_weight <= 0.0:
        return 0.0
    return float((clipped_mean(values, clip_abs) * data_weight + float(prior) * float(prior_weight)) / total_weight)


def enforce_unit_heat_order(unit_values: dict[tuple[object, ...], float]) -> dict[tuple[object, ...], float]:
    ordered_values = dict(unit_values)
    grouped_prefixes: set[tuple[object, ...]] = {
        tuple(key[:-1])
        for key in ordered_values
        if len(key) >= 2
    }
    for prefix in grouped_prefixes:
        previous: float | None = None
        for unit_label in UNIT_HEAT_ORDER:
            key = prefix + (unit_label,)
            if key not in ordered_values:
                continue
            value = float(ordered_values[key])
            if previous is not None:
                value = min(value, previous - UNIT_HEAT_MIN_ORDER_GAP)
            ordered_values[key] = value
            previous = value
    return ordered_values


def unit_component_subtraction_weight(condition: str) -> float:
    return float(
        UNIT_COMPONENT_SUBTRACTION_BY_CONDITION.get(
            condition,
            UNIT_COMPONENT_SUBTRACTION_BY_CONDITION["other"],
        )
    )


def unit_heat_prior_weight(condition: str) -> float:
    return float(
        UNIT_HEAT_PRIOR_WEIGHT_BY_CONDITION.get(
            condition,
            UNIT_HEAT_PRIOR_WEIGHT,
        )
    )


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


def coerce_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return parse_int(str(value))
    except ValueError:
        return None


def is_world_bloom_wl2_prior_event(
    row: dict[str, float | int | str],
    payload: dict[str, object] | None = None,
) -> bool:
    event_id = coerce_int(row.get("event_id"))
    if event_id is None:
        return False
    wl2_event_ids = WORLD_BLOOM_WL2_EVENT_IDS
    wl3_event_ids = WORLD_BLOOM_WL3_EVENT_IDS
    if payload is not None:
        raw_wl2_event_ids = payload.get("wl2_event_ids")
        raw_wl3_event_ids = payload.get("wl3_event_ids")
        if isinstance(raw_wl2_event_ids, (list, tuple, set, frozenset)):
            wl2_event_ids = frozenset(
                event_id
                for event_id in (coerce_int(item) for item in raw_wl2_event_ids)
                if event_id is not None
            )
        if isinstance(raw_wl3_event_ids, (list, tuple, set, frozenset)):
            wl3_event_ids = frozenset(
                event_id
                for event_id in (coerce_int(item) for item in raw_wl3_event_ids)
                if event_id is not None
            )
    if event_id in wl3_event_ids:
        return False
    return str(row.get("series_type")) == "world_bloom" and event_id in wl2_event_ids


def wl2_character_prior_rank_weight(
    rank: int,
    payload: dict[str, object] | None = None,
) -> float:
    raw_weights = (
        payload.get("wl2_character_prior_rank_weights")
        if payload is not None
        else None
    )
    weights = raw_weights if isinstance(raw_weights, (list, tuple)) else WORLD_BLOOM_WL2_CHARACTER_PRIOR_RANK_WEIGHTS
    for max_rank, weight in weights:
        if int(rank) <= int(max_rank):
            return float(weight)
    return 0.0


def wl2_character_prior_multiplier(
    row: dict[str, float | int | str],
    payload: dict[str, object] | None = None,
) -> float:
    enabled = (
        bool(payload.get("wl2_character_prior_enabled", WORLD_BLOOM_WL2_CHARACTER_PRIOR_ENABLED))
        if payload is not None
        else WORLD_BLOOM_WL2_CHARACTER_PRIOR_ENABLED
    )
    if not enabled:
        return 1.0
    if not is_world_bloom_wl2_prior_event(row, payload):
        return 1.0
    character_id = coerce_int(row.get("character_id"))
    if character_id is None or character_id in VOCALOID_CHARACTER_IDS:
        return 1.0
    prior_map = CHARACTER_POPULARITY_PRIOR
    if payload is not None and isinstance(payload.get("wl2_character_prior"), dict):
        raw_map = payload["wl2_character_prior"]
        prior = raw_map.get(character_id) if isinstance(raw_map, dict) else None
        if prior is None and isinstance(raw_map, dict):
            prior = raw_map.get(str(character_id))
    else:
        prior = prior_map.get(character_id)
    if prior is None:
        return 1.0
    region = str(row.get("region"))
    if payload is not None and isinstance(payload.get("wl2_character_prior_region_weights"), dict):
        raw_region_weights = payload["wl2_character_prior_region_weights"]
        region_weight = float(raw_region_weights.get(region, 0.0)) if isinstance(raw_region_weights, dict) else 0.0
    else:
        region_weight = float(WORLD_BLOOM_WL2_CHARACTER_PRIOR_REGION_WEIGHTS.get(region, 0.0))
    if region_weight <= 0.0:
        return 1.0
    rank = coerce_int(row.get("rank")) or 0
    rank_weight = wl2_character_prior_rank_weight(rank, payload)
    progress = clamp(float(row.get("progress", 0.0)), 0.0, 1.0)
    progress_floor = (
        float(payload.get("wl2_character_prior_progress_floor", WORLD_BLOOM_WL2_CHARACTER_PRIOR_PROGRESS_FLOOR))
        if payload is not None
        else WORLD_BLOOM_WL2_CHARACTER_PRIOR_PROGRESS_FLOOR
    )
    progress_decay = (
        float(payload.get("wl2_character_prior_progress_decay", WORLD_BLOOM_WL2_CHARACTER_PRIOR_PROGRESS_DECAY))
        if payload is not None
        else WORLD_BLOOM_WL2_CHARACTER_PRIOR_PROGRESS_DECAY
    )
    progress_weight = max(progress_floor, 1.0 - progress_decay * progress)
    return math.exp(float(prior) * region_weight * rank_weight * progress_weight)


def apply_wl2_character_prior_batch(
    rows: list[dict[str, float | int | str]],
    predictions: np.ndarray,
    payload: dict[str, object] | None = None,
) -> np.ndarray:
    return np.asarray(
        [
            max(float(prediction), 1.0) * wl2_character_prior_multiplier(row, payload)
            for row, prediction in zip(rows, predictions, strict=True)
        ],
        dtype=np.float64,
    )


def region_one_hot(region: str) -> list[float]:
    vector = [0.0] * len(REGIONS)
    vector[REGION_INDEX[region]] = 1.0
    return vector


def local_temporal_features(region: str, timestamp: int) -> list[float]:
    offset_hours = REGION_UTC_OFFSETS.get(region, 0)
    local_timestamp = timestamp + offset_hours * 3600
    local_hour = (local_timestamp % 86400) / 3600.0
    local_weekday = int(time.gmtime(local_timestamp).tm_wday)
    reset_hour = (local_hour - 4.0) % 24.0
    hour_angle = 2.0 * math.pi * (local_hour / 24.0)
    weekday_angle = 2.0 * math.pi * (local_weekday / 7.0)
    reset_angle = 2.0 * math.pi * (reset_hour / 24.0)
    hour_sin = math.sin(hour_angle)
    hour_cos = math.cos(hour_angle)
    reset_sin = math.sin(reset_angle)
    reset_cos = math.cos(reset_angle)
    weekend_flag = 1.0 if local_weekday >= 5 else 0.0
    weekday_flag = 1.0 - weekend_flag
    return [
        hour_sin,
        hour_cos,
        math.sin(weekday_angle),
        math.cos(weekday_angle),
        weekend_flag,
        reset_sin,
        reset_cos,
        weekday_flag * hour_sin,
        weekday_flag * hour_cos,
        weekend_flag * hour_sin,
        weekend_flag * hour_cos,
        weekday_flag * reset_sin,
        weekday_flag * reset_cos,
        weekend_flag * reset_sin,
        weekend_flag * reset_cos,
    ]


def local_calendar_segment(region: str, timestamp: int) -> str:
    offset_hours = REGION_UTC_OFFSETS.get(region, 0)
    local_timestamp = timestamp + offset_hours * 3600
    local_weekday = int(time.gmtime(local_timestamp).tm_wday)
    return "weekend" if local_weekday >= 5 else "weekday"


def is_local_weekend(region: str, timestamp: int) -> bool:
    return local_calendar_segment(region, timestamp) == "weekend"


def weekend_overlap_flag(region: str, start_timestamp: int, end_timestamp: int) -> float:
    if end_timestamp <= start_timestamp:
        return 1.0 if is_local_weekend(region, start_timestamp) else 0.0
    probe = start_timestamp
    step_seconds = 6 * 3600
    while probe <= end_timestamp:
        if is_local_weekend(region, probe):
            return 1.0
        probe += step_seconds
    return 1.0 if is_local_weekend(region, end_timestamp) else 0.0


def next_score_milestone(score: float) -> tuple[float, float]:
    current = max(float(score), 0.0)
    if current <= 0.0:
        return 0.0, 1.0
    exponent = math.floor(math.log10(current))
    milestones: list[float] = []
    for exp in range(max(exponent - 1, 0), exponent + 3):
        base = 10.0 ** exp
        milestones.extend(multiplier * base for multiplier in (1.0, 2.0, 3.0, 5.0, 10.0))
    ordered = sorted(set(value for value in milestones if value >= 1.0))
    previous = 0.0
    for value in ordered:
        if value > current:
            return previous, value
        previous = value
    return previous, max(previous * 2.0, current + 1.0)


def score_edge_features(current_score: int, linear_projection: float, hours_remaining: float) -> list[float]:
    previous, next_value = next_score_milestone(float(current_score))
    gap = max(next_value - float(current_score), 0.0)
    band_width = max(next_value - previous, 1.0)
    gap_ratio = gap / max(next_value, 1.0)
    position = min(max((float(current_score) - previous) / band_width, 0.0), 1.0)
    required_pace = gap / max(hours_remaining, 1e-6)
    crossed_milestones = 0
    probe_previous, probe_next = previous, next_value
    while probe_next <= max(float(linear_projection), float(current_score)) and crossed_milestones < 5:
        crossed_milestones += 1
        probe_previous, probe_next = next_score_milestone(probe_next)
        if probe_next <= probe_previous:
            break
    return [
        safe_log1p(next_value),
        gap_ratio,
        position,
        safe_log1p(required_pace),
        1.0 if gap_ratio <= 0.02 else 0.0,
        1.0 if gap_ratio <= 0.05 else 0.0,
        signed_log1p(float(linear_projection) - next_value),
        float(crossed_milestones),
    ]


def make_boosting_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=8,
        max_iter=500,
        min_samples_leaf=20,
        l2_regularization=0.05,
        random_state=42,
        early_stopping=False,
    )


def make_cn_sidecar_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=3,
        max_iter=60,
        min_samples_leaf=80,
        random_state=42,
        early_stopping=False,
    )


def get_residual_blend_weight(region: str, progress: float) -> float:
    normalized_progress = min(max(float(progress), 0.0), 1.0)
    progress_weights = REGION_PROGRESS_BLEND_WEIGHTS.get(region)
    if progress_weights:
        for threshold, weight in progress_weights:
            if normalized_progress < threshold:
                return float(weight)
    return float(REGION_BLEND_WEIGHTS.get(region, 0.80))


def progress_bucket_name(progress: float) -> str:
    if progress < 0.25:
        return "early_5_25"
    if progress < 0.50:
        return "mid_25_50"
    if progress < 0.75:
        return "mid_50_75"
    return "late_75_95"


def rank_bucket_name(rank: int) -> str:
    if rank <= 10:
        return "top10"
    if rank <= 100:
        return "top100"
    if rank <= 5000:
        return "mid"
    return "tail"


def load_finished_event_ids(now_ms: int | None = None) -> dict[str, set[int]]:
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    finished: dict[str, set[int]] = {region: set() for region in REGIONS}
    for region, base in MASTERDATA_ROOTS.items():
        if not base.exists():
            continue
        for row in load_json(base / "events.json"):
            event_id = parse_int(row.get("id"))
            aggregate_at = parse_int(row.get("aggregateAt"))
            if event_id is None or aggregate_at is None:
                continue
            if aggregate_at <= now_ms:
                finished[region].add(event_id)
    return finished


def load_world_bloom_chapter_windows(base: Path) -> dict[int, dict[int, tuple[int, int]]]:
    path = base / "worldBlooms.json"
    if not path.exists():
        return {}
    windows: dict[int, dict[int, tuple[int, int]]] = defaultdict(dict)
    for row in load_json(path):
        event_id = parse_int(row.get("eventId"))
        character_id = parse_int(row.get("gameCharacterId"))
        start_at = parse_int(row.get("chapterStartAt"))
        aggregate_at = parse_int(row.get("aggregateAt")) or parse_int(row.get("chapterEndAt"))
        if event_id is None or character_id is None or start_at is None or aggregate_at is None:
            continue
        start_timestamp = start_at // 1000
        end_timestamp = aggregate_at // 1000
        if start_timestamp < end_timestamp:
            windows[event_id][character_id] = (start_timestamp, end_timestamp)
    return {event_id: dict(character_windows) for event_id, character_windows in windows.items()}


def load_event_meta(data_root: Path, finished_event_ids: dict[str, set[int]]) -> dict[tuple[str, int], EventMeta]:
    _ = data_root
    event_meta: dict[tuple[str, int], EventMeta] = {}
    for region in REGIONS:
        base = MASTERDATA_ROOTS.get(region)
        if base is None or not base.exists():
            continue
        chapter_windows = load_world_bloom_chapter_windows(base)
        for row in load_json(base / "events.json"):
            event_id = parse_int(row.get("id"))
            start_at = parse_int(row.get("startAt"))
            aggregate_at = parse_int(row.get("aggregateAt"))
            if event_id is None or start_at is None or aggregate_at is None:
                continue
            if event_id not in finished_event_ids.get(region, set()):
                continue
            start_timestamp = start_at // 1000
            end_timestamp = aggregate_at // 1000
            if start_timestamp >= end_timestamp:
                continue
            series_type = "world_bloom" if str(row.get("eventType")) == "world_bloom" else "normal"
            event_meta[(region, event_id)] = EventMeta(
                region=region,
                event_id=event_id,
                series_type=series_type,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                chapter_windows=chapter_windows.get(event_id, {}),
            )
    return event_meta


def select_validation_events(
    event_meta: dict[tuple[str, int], EventMeta],
    holdout_count: int,
    model_series_types: frozenset[str] = NORMAL_MODEL_SERIES_TYPES,
    *,
    available_event_ids: dict[str, set[int]] | None = None,
) -> dict[str, set[int]]:
    by_region: dict[str, list[int]] = defaultdict(list)
    for (region, event_id), meta in event_meta.items():
        if meta.series_type not in model_series_types:
            continue
        if available_event_ids is not None and event_id not in available_event_ids.get(region, set()):
            continue
        by_region[region].append(event_id)
    validation: dict[str, set[int]] = {}
    for region, event_ids in by_region.items():
        validation[region] = set(sorted(event_ids)[-holdout_count:])
    return validation


def iter_rank_timeseries_rows(
    data_root: Path,
    region: str,
    selected_ranks: set[int],
    model_series_types: frozenset[str] | None = None,
):
    path = data_root / "regions" / region / "training" / "rank_timeseries.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            series_type = row["series_type"]
            if model_series_types is not None and series_type not in model_series_types:
                continue
            rank = parse_int(row.get("rank"))
            timestamp = parse_int(row.get("timestamp"))
            score = parse_int(row.get("score"))
            event_id = parse_int(row.get("event_id"))
            if rank not in selected_ranks or timestamp is None or score is None or event_id is None:
                continue
            yield {
                "region": region,
                "event_id": event_id,
                "series_type": series_type,
                "character_id": parse_int(row.get("character_id")) or 0,
                "rank": rank,
                "timestamp": timestamp,
                "score": score,
            }


def iter_legacy_rank_timeseries_rows(
    region: str,
    selected_ranks: set[int],
    legacy_event_files: dict[str, list[tuple[int, str, list[Path]]]],
    model_series_types: frozenset[str] | None = None,
):
    for event_id, series_type, files in legacy_event_files.get(region, []):
        if model_series_types is not None and series_type not in model_series_types:
            continue
        for path in files:
            sql_text = path.read_text(encoding="utf-8", errors="replace")
            row_re = LEGACY_WL_ROW_RE if series_type == "world_bloom" else LEGACY_NORMAL_ROW_RE
            fast_match_count = 0
            for match in row_re.finditer(sql_text):
                fast_match_count += 1
                if series_type == "world_bloom":
                    timestamp, score, rank, character_id = match.groups()
                else:
                    timestamp, score, rank = match.groups()
                    character_id = 0
                rank_value = int(rank)
                if rank_value not in selected_ranks:
                    continue
                yield {
                    "region": region,
                    "event_id": event_id,
                    "series_type": series_type,
                    "character_id": int(character_id),
                    "rank": rank_value,
                    "timestamp": int(timestamp),
                    "score": int(score),
                }
            if fast_match_count > 0:
                continue
            parsed = parse_insert_statement_blob(sql_text)
            if parsed is None:
                continue
            _table_name, values_sql = parsed
            rows = list(iter_mixed_rows(values_sql))
            if series_type == "world_bloom":
                rows.sort(key=lambda row: (int(row[0]), int(row[4]), int(row[3])))
            else:
                rows.sort(key=lambda row: (int(row[0]), int(row[3])))
            for row in rows:
                if series_type == "world_bloom":
                    if len(row) < 5:
                        continue
                    timestamp, _user_id, score, rank, character_id = row
                else:
                    if len(row) < 4:
                        continue
                    timestamp, _user_id, score, rank = row
                    character_id = 0
                if timestamp is None or score is None or rank is None or int(rank) not in selected_ranks:
                    continue
                yield {
                    "region": region,
                    "event_id": event_id,
                    "series_type": series_type,
                    "character_id": int(character_id) if character_id is not None else 0,
                    "rank": int(rank),
                    "timestamp": int(timestamp),
                    "score": int(score),
                }


def parse_iso_timestamp_seconds(value: str) -> int:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return int(datetime.fromisoformat(text).astimezone(timezone.utc).timestamp())


def sekai_best_jp_wl2_timeseries_path() -> Path:
    return SEKAI_BEST_JP_WL2_DATA_ROOT / "jp_wl2_chapter_rank_timeseries.csv.gz"


def sekai_best_world_bloom_timeseries_path(region: str) -> Path:
    return (
        SEKAI_BEST_WORLD_BLOOM_DATA_ROOT
        / region
        / f"{region}_world_bloom_chapter_rank_timeseries.csv.gz"
    )


def external_world_bloom_timeseries_paths(region: str) -> list[Path]:
    paths: list[Path] = []
    generic_path = sekai_best_world_bloom_timeseries_path(region)
    if generic_path.exists():
        paths.append(generic_path)
    legacy_jp_path = sekai_best_jp_wl2_timeseries_path()
    if region == "jp" and legacy_jp_path.exists() and legacy_jp_path not in paths:
        paths.append(legacy_jp_path)
    return paths


def external_world_bloom_event_ids(region: str) -> set[int]:
    event_ids: set[int] = set()
    for path in external_world_bloom_timeseries_paths(region):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                event_id = parse_int(row.get("event_id"))
                if event_id is not None:
                    event_ids.add(event_id)
    return event_ids


def iter_external_world_bloom_rank_timeseries_rows(region: str, selected_ranks: set[int]):
    for path in external_world_bloom_timeseries_paths(region):
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rank = parse_int(row.get("rank"))
                event_id = parse_int(row.get("event_id"))
                character_id = parse_int(row.get("character_id"))
                score = parse_int(row.get("score"))
                timestamp_text = row.get("timestamp")
                row_region = str(row.get("region") or region)
                if (
                    rank not in selected_ranks
                    or event_id is None
                    or character_id is None
                    or score is None
                    or not timestamp_text
                ):
                    continue
                yield {
                    "region": row_region,
                    "event_id": event_id,
                    "series_type": "world_bloom",
                    "character_id": character_id,
                    "rank": rank,
                    "timestamp": parse_iso_timestamp_seconds(timestamp_text),
                    "score": score,
                }


def iter_sekai_best_jp_wl2_rank_timeseries_rows(selected_ranks: set[int]):
    path = sekai_best_jp_wl2_timeseries_path()
    if not path.exists():
        return
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rank = parse_int(row.get("rank"))
            event_id = parse_int(row.get("event_id"))
            character_id = parse_int(row.get("character_id"))
            score = parse_int(row.get("score"))
            timestamp_text = row.get("timestamp")
            if (
                rank not in selected_ranks
                or event_id not in WORLD_BLOOM_WL2_EVENT_IDS
                or character_id is None
                or score is None
                or not timestamp_text
            ):
                continue
            yield {
                "region": "jp",
                "event_id": event_id,
                "series_type": "world_bloom",
                "character_id": character_id,
                "rank": rank,
                "timestamp": parse_iso_timestamp_seconds(timestamp_text),
                "score": score,
            }


def iter_all_rank_timeseries_rows(
    data_root: Path,
    region: str,
    selected_ranks: set[int],
    legacy_event_files: dict[str, list[tuple[int, str, list[Path]]]],
    model_series_types: frozenset[str] | None = None,
):
    live_keys = live_index(sys.modules[__name__], region)
    def replaced(row):
        return (row["event_id"], row["series_type"], row["character_id"], row["rank"]) in live_keys

    include_world_bloom = model_series_types is None or "world_bloom" in model_series_types
    external_wl_event_ids = external_world_bloom_event_ids(region) if include_world_bloom else set()
    for row in iter_legacy_rank_timeseries_rows(
        region,
        selected_ranks,
        legacy_event_files,
        model_series_types,
    ):
        if row["series_type"] == "world_bloom" and row["event_id"] in external_wl_event_ids:
            continue
        if not replaced(row):
            yield row
    if include_world_bloom:
        for row in iter_external_world_bloom_rank_timeseries_rows(region, selected_ranks):
            if not replaced(row):
                yield row
    for row in iter_rank_timeseries_rows(data_root, region, selected_ranks, model_series_types):
        if row["series_type"] == "world_bloom" and row["event_id"] in external_wl_event_ids:
            continue
        if not replaced(row):
            yield row

    yield from iter_live_history_rows(sys.modules[__name__], region, selected_ranks, model_series_types)


def boundary_pressure_target_rank(rank: int) -> int | None:
    if BOUNDARY_PRESSURE_TOP100_MIN_RANK <= rank <= 100:
        return 100
    if rank in BOUNDARY_PRESSURE_RANKS:
        return rank
    return None


def iter_leaderboard_observation_rows(data_root: Path, region: str):
    path = data_root / "regions" / region / "training" / "leaderboard_observations.csv.gz"
    if not path.exists():
        return
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rank = parse_int(row.get("rank"))
            target_rank = boundary_pressure_target_rank(rank or 0)
            if target_rank is None:
                continue
            timestamp = parse_int(row.get("timestamp"))
            event_id = parse_int(row.get("event_id"))
            score = parse_int(row.get("score"))
            user_id_key = row.get("user_id") or row.get("user_id_key")
            if timestamp is None or event_id is None or score is None or not user_id_key:
                continue
            yield {
                "region": region,
                "event_id": event_id,
                "series_type": row.get("series_type") or "normal",
                "character_id": parse_int(row.get("character_id")) or 0,
                "rank": rank,
                "target_rank": target_rank,
                "timestamp": timestamp,
                "score": score,
                "user_id": str(user_id_key),
            }


def boundary_pressure_series_from_snapshots(
    snapshots: list[tuple[int, frozenset[str]]],
) -> BoundaryPressureSeries:
    series = BoundaryPressureSeries()
    window_6h: deque[tuple[int, frozenset[str]]] = deque()
    window_24h: deque[tuple[int, frozenset[str]]] = deque()
    counts_6h: Counter[str] = Counter()
    counts_24h: Counter[str] = Counter()
    previous_users: frozenset[str] = frozenset()

    def add_users(counter: Counter[str], users: frozenset[str]) -> None:
        for user_id in users:
            counter[user_id] += 1

    def remove_users(counter: Counter[str], users: frozenset[str]) -> None:
        for user_id in users:
            counter[user_id] -= 1
            if counter[user_id] <= 0:
                del counter[user_id]

    for timestamp, active_users in sorted(snapshots, key=lambda item: item[0]):
        while window_6h and window_6h[0][0] < timestamp - 6 * 3600:
            _old_timestamp, old_users = window_6h.popleft()
            remove_users(counts_6h, old_users)
        while window_24h and window_24h[0][0] < timestamp - 24 * 3600:
            _old_timestamp, old_users = window_24h.popleft()
            remove_users(counts_24h, old_users)

        window_6h.append((timestamp, active_users))
        window_24h.append((timestamp, active_users))
        add_users(counts_6h, active_users)
        add_users(counts_24h, active_users)

        active_count = float(len(active_users))
        unique_6h = float(len(counts_6h))
        unique_24h = float(len(counts_24h))
        denominator = max(active_count, 1.0)
        if active_users or previous_users:
            overlap = len(active_users & previous_users) / max(len(active_users | previous_users), 1)
            turnover = 1.0 - overlap
        else:
            turnover = 0.0
        series.timestamps.append(timestamp)
        series.features.append(
            (
                1.0,
                safe_log1p(active_count),
                safe_log1p(unique_6h),
                safe_log1p(unique_24h),
                unique_6h / denominator,
                unique_24h / denominator,
                turnover,
                float(len(window_6h)) / 6.0,
            )
        )
        previous_users = active_users

    return series


def build_boundary_pressure_contexts(
    data_root: Path,
    event_meta: dict[tuple[str, int], EventMeta],
    selected_ranks: set[int],
    model_series_types: frozenset[str] = NORMAL_MODEL_SERIES_TYPES,
) -> dict[tuple[str, int, int, int], BoundaryPressureSeries]:
    snapshot_users: dict[tuple[str, int, int, int, int], set[str]] = defaultdict(set)
    for region in REGIONS:
        for row in iter_leaderboard_observation_rows(data_root, region):
            target_rank = int(row["target_rank"])
            if target_rank not in selected_ranks:
                continue
            event_key = (region, int(row["event_id"]))
            meta = event_meta.get(event_key)
            if meta is None or meta.series_type not in model_series_types:
                continue
            snapshot_key = (
                region,
                int(row["event_id"]),
                int(row["character_id"]),
                target_rank,
                int(row["timestamp"]),
            )
            snapshot_users[snapshot_key].add(str(row["user_id"]))

    grouped: dict[tuple[str, int, int, int], list[tuple[int, frozenset[str]]]] = defaultdict(list)
    for (region, event_id, character_id, target_rank, timestamp), users in snapshot_users.items():
        grouped[(region, event_id, character_id, target_rank)].append((timestamp, frozenset(users)))

    return {
        key: boundary_pressure_series_from_snapshots(snapshots)
        for key, snapshots in grouped.items()
    }


def boundary_pressure_features(
    event_key: tuple[str, int],
    character_id: int,
    rank: int,
    timestamp: int,
    boundary_pressure_contexts: dict[tuple[str, int, int, int], BoundaryPressureSeries] | None,
) -> list[float]:
    if boundary_pressure_contexts is None or rank not in BOUNDARY_PRESSURE_RANKS:
        return [0.0] * BOUNDARY_PRESSURE_FEATURE_COUNT
    region, event_id = event_key
    series = boundary_pressure_contexts.get((region, event_id, character_id, rank))
    if series is None and character_id != 0:
        series = boundary_pressure_contexts.get((region, event_id, 0, rank))
    if series is None:
        return [0.0] * BOUNDARY_PRESSURE_FEATURE_COUNT
    index = bisect.bisect_right(series.timestamps, timestamp) - 1
    if index < 0:
        return [0.0] * BOUNDARY_PRESSURE_FEATURE_COUNT
    return list(series.features[index])


def build_final_targets(
    data_root: Path,
    event_meta: dict[tuple[str, int], EventMeta],
    selected_ranks: set[int],
    legacy_event_files: dict[str, list[tuple[int, str, list[Path]]]],
) -> dict[tuple[str, int, int, int], tuple[int, int]]:
    finals: dict[tuple[str, int, int, int], tuple[int, int]] = {}
    for region in REGIONS:
        for row in iter_all_rank_timeseries_rows(data_root, region, selected_ranks, legacy_event_files):
            key = (region, row["event_id"], row["character_id"], row["rank"])
            meta = event_meta.get((region, row["event_id"]))
            if meta is None:
                continue
            finals[key] = (row["score"], row["timestamp"])
    for key, (_score, timestamp) in list(finals.items()):
        region, event_id, character_id, _rank = key
        meta = event_meta.get((region, event_id))
        if meta is None:
            continue
        if not is_world_bloom_series_complete(meta, character_id, timestamp):
            del finals[key]
    return finals


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_insert_statement_blob(sql_text: str) -> tuple[str, str] | None:
    prefix = "INSERT INTO `"
    start = sql_text.find(prefix)
    if start == -1:
        return None
    table_start = start + len(prefix)
    table_end = sql_text.find("`", table_start)
    if table_end == -1:
        return None
    values_marker = " VALUES"
    values_index = sql_text.find(values_marker, table_end)
    if values_index == -1:
        return None
    table_name = sql_text[table_start:table_end]
    rest = sql_text[values_index + len(values_marker):]
    return table_name, rest


def iter_mixed_rows(values_sql: str):
    index = 0
    length = len(values_sql)
    while index < length:
        while index < length and values_sql[index] in " \t\r\n,":
            index += 1
        if index >= length or values_sql[index] == ";":
            return
        if values_sql[index] != "(":
            index += 1
            continue
        index += 1
        row: list[object | None] = []
        while index < length:
            while index < length and values_sql[index] in " \t\r\n":
                index += 1
            if index >= length:
                break
            if values_sql[index] in ("'", '"'):
                quote = values_sql[index]
                index += 1
                chars: list[str] = []
                while index < length:
                    char = values_sql[index]
                    if char == "\\":
                        index += 1
                        if index >= length:
                            break
                        chars.append(values_sql[index])
                        index += 1
                        continue
                    if char == quote:
                        index += 1
                        break
                    chars.append(char)
                    index += 1
                row.append("".join(chars))
            else:
                start = index
                while index < length and values_sql[index] not in ",)":
                    index += 1
                token = values_sql[start:index].strip()
                if token.upper() == "NULL" or token == "":
                    row.append(None)
                elif token.lstrip("-").isdigit():
                    row.append(int(token))
                else:
                    row.append(token)
            if index < length and values_sql[index] == ",":
                index += 1
                continue
            if index < length and values_sql[index] == ")":
                index += 1
                break
        yield tuple(row)
        while index < length and values_sql[index] in " \t\r\n,":
            index += 1
        if index < length and values_sql[index] == ";":
            return


def load_event_feature_contexts() -> dict[tuple[str, int], EventFeatureContext]:
    event_contexts: dict[tuple[str, int], EventFeatureContext] = {}
    for region, base in MASTERDATA_ROOTS.items():
        if not base.exists():
            continue
        unit_rows = load_json(base / "gameCharacterUnits.json")
        bonus_rows = load_json(base / "eventDeckBonuses.json")
        event_story_rows = load_json(base / "eventStories.json")
        world_bloom_rows = load_json(base / "worldBlooms.json")

        unit_to_character: dict[int, int] = {}
        unit_to_unit_label: dict[int, str] = {}
        for row in unit_rows:
            unit_id = parse_int(masterdata_row_value(row, "id", 0))
            game_character_id = parse_int(masterdata_row_value(row, "gameCharacterId", 1))
            if unit_id is None or game_character_id is None:
                continue
            unit_to_character[unit_id] = game_character_id
            unit_to_unit_label[unit_id] = str(masterdata_row_value(row, "unit", 2) or "none")
        per_event_characters: dict[int, set[int]] = defaultdict(set)
        per_event_units: dict[int, set[str]] = defaultdict(set)
        per_event_main_characters: dict[int, set[int]] = defaultdict(set)

        for row in bonus_rows:
            event_id = parse_int(masterdata_row_value(row, "eventId", 1))
            game_character_unit_id = parse_int(masterdata_row_value(row, "gameCharacterUnitId", 2))
            if event_id is None:
                continue
            if game_character_unit_id is None:
                continue
            game_character_id = unit_to_character.get(game_character_unit_id)
            if game_character_id in VOCALOID_CHARACTER_IDS:
                continue
            if game_character_id is not None:
                per_event_characters[event_id].add(game_character_id)
            unit_label = unit_to_unit_label.get(game_character_unit_id)
            if unit_label:
                per_event_units[event_id].add(unit_label)

        for row in event_story_rows:
            event_id = parse_int(masterdata_row_value(row, "eventId", 1))
            banner_unit_id = parse_int(masterdata_row_value(row, "bannerGameCharacterUnitId", 3))
            if event_id is None:
                continue
            if banner_unit_id is None:
                continue
            character_id = unit_to_character.get(banner_unit_id)
            if character_id is None or character_id in VOCALOID_CHARACTER_IDS:
                continue
            per_event_main_characters[event_id].add(character_id)

        for override_key, override_characters in MANUAL_MAIN_CHARACTER_OVERRIDES.items():
            override_region, override_event_id = override_key
            if override_region != region:
                continue
            per_event_main_characters[override_event_id].update(
                character_id
                for character_id in override_characters
                if character_id not in VOCALOID_CHARACTER_IDS
            )

        for event_id, characters in per_event_characters.items():
            unit_labels = tuple(sorted(per_event_units.get(event_id, set())))
            if unit_labels and len(unit_labels) <= 1:
                condition = "box"
            elif len(unit_labels) > 1:
                condition = "mixed"
            else:
                condition = "other"
            event_contexts[(region, event_id)] = EventFeatureContext(
                character_ids=tuple(sorted(characters)),
                unit_labels=unit_labels,
                main_character_ids=tuple(sorted(per_event_main_characters.get(event_id, set()))),
                condition=condition,
            )

        for row in world_bloom_rows:
            event_id = parse_int(masterdata_row_value(row, "eventId", 1))
            game_character_id = parse_int(masterdata_row_value(row, "gameCharacterId", 2))
            if event_id is None:
                continue
            if game_character_id is None:
                continue
            current = event_contexts.get((region, event_id), EventFeatureContext(condition="world_bloom"))
            event_contexts[(region, event_id)] = EventFeatureContext(
                character_ids=tuple(sorted({*current.character_ids, game_character_id})),
                unit_labels=tuple(),
                main_character_ids=current.main_character_ids,
                condition="world_bloom",
            )
    return event_contexts


def load_event_character_map(
    event_feature_contexts: dict[tuple[str, int], EventFeatureContext],
) -> dict[tuple[str, int], tuple[int, ...]]:
    return {
        event_key: context.character_ids
        for event_key, context in event_feature_contexts.items()
    }


def load_current_event_ids(data_root: Path) -> dict[str, set[int]]:
    current_event_ids: dict[str, set[int]] = {region: set() for region in REGIONS}
    for region in REGIONS:
        manifest_path = data_root / "regions" / region / "training" / "event_manifest.csv"
        with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                event_id = parse_int(row.get("event_id"))
                if event_id is not None:
                    current_event_ids[region].add(event_id)
    return current_event_ids


def load_normalized_rank_timeseries_event_keys(data_root: Path) -> dict[str, set[tuple[str, int]]]:
    event_keys: dict[str, set[tuple[str, int]]] = {region: set() for region in REGIONS}
    for region in REGIONS:
        path = data_root / "regions" / region / "training" / "rank_timeseries.csv.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                event_id = parse_int(row.get("event_id"))
                if event_id is None:
                    continue
                event_keys[region].add((row.get("series_type") or "normal", event_id))
    return event_keys


def load_legacy_event_files(
    data_root: Path,
    normalized_event_keys: dict[str, set[tuple[str, int]]],
) -> dict[str, list[tuple[int, str, list[Path]]]]:
    legacy_event_files: dict[str, list[tuple[int, str, list[Path]]]] = {region: [] for region in REGIONS}
    for region in REGIONS:
        chosen: dict[tuple[str, int], list[Path]] = {}
        for legacy_source in LEGACY_SOURCES:
            base = data_root / "regions" / region / "raw" / "legacy" / legacy_source
            if not base.exists():
                continue
            grouped: dict[tuple[str, int], list[Path]] = defaultdict(list)
            for path in sorted(base.glob("*.sql")):
                match = LEGACY_EVENT_RE.search(path.name)
                if not match:
                    continue
                table_name = match.group(1)
                if "_names" in table_name:
                    continue
                series_type = "world_bloom" if table_name.startswith("wl_") else "normal"
                event_id = int(table_name.split("_", 1)[1])
                if (series_type, event_id) in normalized_event_keys.get(region, set()):
                    continue
                grouped[(series_type, event_id)].append(path)
            for key, files in grouped.items():
                if key not in chosen:
                    chosen[key] = files
        legacy_event_files[region] = [
            (event_id, series_type, files)
            for (series_type, event_id), files in sorted(chosen.items(), key=lambda item: item[0][1])
        ]
    return legacy_event_files


def collect_available_training_event_ids(
    normalized_event_keys: dict[str, set[tuple[str, int]]],
    legacy_event_files: dict[str, list[tuple[int, str, list[Path]]]],
    model_series_types: frozenset[str] = NORMAL_MODEL_SERIES_TYPES,
) -> dict[str, set[int]]:
    available: dict[str, set[int]] = {region: set() for region in REGIONS}
    for region, keys in normalized_event_keys.items():
        for series_type, event_id in keys:
            if series_type in model_series_types:
                available[region].add(event_id)
    for region, entries in legacy_event_files.items():
        for event_id, series_type, files in entries:
            if files and series_type in model_series_types:
                available[region].add(event_id)
    for region in REGIONS:
        for event_id, series, char, rank in live_index(sys.modules[__name__], region):
            if series in model_series_types and rank in COMMON_RANKS:
                available[region].add(event_id)
    return available


def build_character_heat_priors(
    event_meta: dict[tuple[str, int], EventMeta],
    finals: dict[tuple[str, int, int, int], tuple[int, int]],
    validation_events: dict[str, set[int]],
    event_feature_contexts: dict[tuple[str, int], EventFeatureContext],
    model_series_types: frozenset[str] = NORMAL_MODEL_SERIES_TYPES,
) -> dict[str, dict[object, float]]:
    baseline_groups: dict[tuple[str, str, int], list[tuple[int, float]]] = defaultdict(list)
    for (region, event_id, _character_id, rank), (score, _timestamp) in finals.items():
        if event_id in validation_events[region]:
            continue
        meta = event_meta.get((region, event_id))
        if meta is None:
            continue
        if meta.series_type not in model_series_types:
            continue
        context = event_feature_contexts.get((region, event_id))
        if context is None:
            continue
        baseline_groups[(region, context.condition, rank)].append((event_id, safe_log1p(score)))

    local_baselines: dict[tuple[str, str, int, int], float] = {}
    half_window = CHARACTER_HEAT_LOCAL_WINDOW // 2
    for key, values in baseline_groups.items():
        ordered_values = sorted(values, key=lambda item: item[0])
        ordered_scores = [score for _, score in ordered_values]
        for index, (event_id, _score) in enumerate(ordered_values):
            left = max(0, index - half_window)
            right = min(len(ordered_values), index + half_window + 1)
            local_baselines[(key[0], key[1], event_id, key[2])] = float(
                np.median(np.asarray(ordered_scores[left:right], dtype=np.float64))
            )

    per_event_residuals: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for (region, event_id, _character_id, rank), (score, _timestamp) in finals.items():
        if event_id in validation_events[region]:
            continue
        meta = event_meta.get((region, event_id))
        if meta is None or meta.series_type not in model_series_types:
            continue
        context = event_feature_contexts.get((region, event_id))
        if context is None:
            continue
        baseline = local_baselines.get((region, context.condition, event_id, rank))
        if baseline is None:
            continue
        per_event_residuals[(region, event_id, context.condition)].append(safe_log1p(score) - baseline)

    per_character_by_region: dict[tuple[str, int], list[float]] = defaultdict(list)
    per_character_by_region_condition: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    per_main_character_by_region: dict[tuple[str, int], list[float]] = defaultdict(list)
    per_main_character_by_region_condition: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    per_unit_by_region: dict[tuple[str, str], list[float]] = defaultdict(list)
    per_unit_by_region_condition: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    event_heat_values: dict[tuple[str, int, str], float] = {}
    filtered_extreme_events: list[tuple[str, int, str, float]] = []
    for (region, event_id, condition), residuals in per_event_residuals.items():
        context = event_feature_contexts.get((region, event_id))
        if context is None or not residuals:
            continue
        residual_arr = np.asarray(residuals, dtype=np.float64)
        median_residual = float(np.median(residual_arr))
        if abs(median_residual) > HEAT_EVENT_RESIDUAL_ABS_LIMIT:
            filtered_extreme_events.append((region, event_id, condition, median_residual))
            continue
        event_heat = clamp(
            float(np.mean(residual_arr)),
            -HEAT_EVENT_VALUE_CLIP,
            HEAT_EVENT_VALUE_CLIP,
        )
        event_heat_values[(region, event_id, condition)] = event_heat
        for unit_label in context.unit_labels:
            per_unit_by_region[(region, unit_label)].append(event_heat)
            per_unit_by_region_condition[(region, condition, unit_label)].append(event_heat)

    if USE_UNIT_HEAT_PRIORS_IN_MODEL:
        unit_by_region = enforce_unit_heat_order({
            key: regularized_heat_mean(
                values,
                UNIT_HEAT_SHRINKAGE,
                prior=UNIT_HEAT_PRIOR.get(key[1], 0.0),
                prior_weight=UNIT_HEAT_PRIOR_WEIGHT,
                clip_abs=0.60,
            )
            for key, values in per_unit_by_region.items()
            if values
        })
        unit_by_region_condition = enforce_unit_heat_order({
            key: regularized_heat_mean(
                values,
                UNIT_HEAT_SHRINKAGE,
                prior=UNIT_HEAT_PRIOR.get(key[2], 0.0),
                prior_weight=unit_heat_prior_weight(str(key[1])),
                clip_abs=0.60,
            )
            for key, values in per_unit_by_region_condition.items()
            if values
        })
    else:
        unit_by_region = {
            key: shrink_mean(values, UNIT_HEAT_SHRINKAGE)
            for key, values in per_unit_by_region.items()
            if values
        }
        unit_by_region_condition = {
            key: shrink_mean(values, UNIT_HEAT_SHRINKAGE)
            for key, values in per_unit_by_region_condition.items()
            if values
        }

    for (region, event_id, condition), event_heat in event_heat_values.items():
        context = event_feature_contexts.get((region, event_id))
        characters = context.character_ids if context is not None else ()
        if not context or not characters:
            continue
        unit_component = 0.0
        if context.unit_labels:
            unit_component = float(np.mean(
                np.asarray(
                    [
                        unit_by_region_condition.get(
                            (region, condition, unit_label),
                            unit_by_region.get((region, unit_label), 0.0),
                        )
                        for unit_label in context.unit_labels
                    ],
                    dtype=np.float64,
                )
            ))
        character_residual = event_heat - unit_component * unit_component_subtraction_weight(condition)
        if USE_POPULARITY_PRIORS_IN_MODEL:
            character_residual = clamp(character_residual, -CHARACTER_RESIDUAL_CLIP, CHARACTER_RESIDUAL_CLIP)
        for character_id in context.main_character_ids:
            per_main_character_by_region[(region, character_id)].append(character_residual)
            per_main_character_by_region_condition[(region, condition, character_id)].append(character_residual)
        if condition == "box" and len(characters) > 1:
            continue
        for character_id in characters:
            per_character_by_region[(region, character_id)].append(character_residual)
            per_character_by_region_condition[(region, condition, character_id)].append(character_residual)

    return {
        "character_by_region": {
            key: (
                regularized_heat_mean(
                    values,
                    CHARACTER_HEAT_SHRINKAGE,
                    prior=CHARACTER_POPULARITY_PRIOR.get(key[1], 0.0),
                    prior_weight=CHARACTER_HEAT_PRIOR_WEIGHT if key[1] in CHARACTER_POPULARITY_PRIOR else 0.0,
                )
                if USE_POPULARITY_PRIORS_IN_MODEL
                else shrink_mean(values, CHARACTER_HEAT_SHRINKAGE)
            )
            for key, values in per_character_by_region.items()
            if values
        },
        "character_by_region_condition": {
            key: (
                regularized_heat_mean(
                    values,
                    CHARACTER_HEAT_SHRINKAGE,
                    prior=CHARACTER_POPULARITY_PRIOR.get(key[2], 0.0),
                    prior_weight=CHARACTER_HEAT_PRIOR_WEIGHT if key[2] in CHARACTER_POPULARITY_PRIOR else 0.0,
                )
                if USE_POPULARITY_PRIORS_IN_MODEL
                else shrink_mean(values, CHARACTER_HEAT_SHRINKAGE)
            )
            for key, values in per_character_by_region_condition.items()
            if values
        },
        "main_character_by_region": {
            key: (
                regularized_heat_mean(
                    values,
                    CHARACTER_HEAT_SHRINKAGE,
                    prior=CHARACTER_POPULARITY_PRIOR.get(key[1], 0.0),
                    prior_weight=MAIN_CHARACTER_HEAT_PRIOR_WEIGHT if key[1] in CHARACTER_POPULARITY_PRIOR else 0.0,
                )
                if USE_POPULARITY_PRIORS_IN_MODEL
                else shrink_mean(values, CHARACTER_HEAT_SHRINKAGE)
            )
            for key, values in per_main_character_by_region.items()
            if values
        },
        "main_character_by_region_condition": {
            key: (
                regularized_heat_mean(
                    values,
                    CHARACTER_HEAT_SHRINKAGE,
                    prior=CHARACTER_POPULARITY_PRIOR.get(key[2], 0.0),
                    prior_weight=MAIN_CHARACTER_HEAT_PRIOR_WEIGHT if key[2] in CHARACTER_POPULARITY_PRIOR else 0.0,
                )
                if USE_POPULARITY_PRIORS_IN_MODEL
                else shrink_mean(values, CHARACTER_HEAT_SHRINKAGE)
            )
            for key, values in per_main_character_by_region_condition.items()
            if values
        },
        "unit_by_region": unit_by_region,
        "unit_by_region_condition": unit_by_region_condition,
        "diagnostics": {
            "filtered_extreme_event_count": float(len(filtered_extreme_events)),
            "heat_event_residual_abs_limit": HEAT_EVENT_RESIDUAL_ABS_LIMIT,
            "heat_event_value_clip": HEAT_EVENT_VALUE_CLIP,
            "use_popularity_priors_in_model": USE_POPULARITY_PRIORS_IN_MODEL,
            "use_unit_heat_priors_in_model": USE_UNIT_HEAT_PRIORS_IN_MODEL,
            "unit_component_subtraction_by_condition": UNIT_COMPONENT_SUBTRACTION_BY_CONDITION,
            "unit_heat_prior_weight": UNIT_HEAT_PRIOR_WEIGHT,
            "unit_heat_prior_weight_by_condition": UNIT_HEAT_PRIOR_WEIGHT_BY_CONDITION,
            "unit_heat_min_order_gap": UNIT_HEAT_MIN_ORDER_GAP,
            "character_heat_prior_weight": CHARACTER_HEAT_PRIOR_WEIGHT,
            "main_character_heat_prior_weight": MAIN_CHARACTER_HEAT_PRIOR_WEIGHT,
            "character_residual_clip": CHARACTER_RESIDUAL_CLIP,
        },
    }


def get_character_heat_prior(
    character_heat_priors: dict[str, dict[object, float]],
    region: str,
    condition: str,
    character_id: int,
) -> float:
    if condition != "world_bloom" and character_id in VOCALOID_CHARACTER_IDS:
        return 0.0
    by_region_condition = character_heat_priors.get("character_by_region_condition", {})
    if (region, condition, character_id) in by_region_condition:
        return float(by_region_condition[(region, condition, character_id)])
    return float(character_heat_priors.get("character_by_region", {}).get((region, character_id), 0.0))


def get_main_character_heat_prior(
    character_heat_priors: dict[str, dict[object, float]],
    region: str,
    condition: str,
    character_id: int,
) -> float:
    if condition != "world_bloom" and character_id in VOCALOID_CHARACTER_IDS:
        return 0.0
    by_region_condition = character_heat_priors.get("main_character_by_region_condition", {})
    if (region, condition, character_id) in by_region_condition:
        return float(by_region_condition[(region, condition, character_id)])
    return float(character_heat_priors.get("main_character_by_region", {}).get((region, character_id), 0.0))


def get_unit_heat_prior(
    character_heat_priors: dict[str, dict[object, float]],
    region: str,
    condition: str,
    unit_label: str,
) -> float:
    by_region_condition = character_heat_priors.get("unit_by_region_condition", {})
    if (region, condition, unit_label) in by_region_condition:
        return float(by_region_condition[(region, condition, unit_label)])
    return float(character_heat_priors.get("unit_by_region", {}).get((region, unit_label), 0.0))


def event_heat_rank_weight(rank: int) -> float:
    rank = int(rank)
    if rank <= 100:
        return 1.00
    if rank <= 1000:
        return 0.82
    if rank <= 10000:
        return 0.58
    if rank <= 50000:
        return 0.34
    return 0.18


def aggregate_event_heat_log(
    character_heat_priors: dict[str, dict[object, float]],
    region: str,
    condition: str,
    context: EventFeatureContext,
    line_character_id: int = 0,
) -> float:
    unit_values = [
        get_unit_heat_prior(character_heat_priors, region, condition, str(unit_label))
        for unit_label in context.unit_labels
    ]
    main_values = [
        get_main_character_heat_prior(character_heat_priors, region, condition, int(cid))
        for cid in context.main_character_ids
    ]
    character_values = [
        get_character_heat_prior(character_heat_priors, region, condition, int(cid))
        for cid in context.character_ids
    ]
    line_value = (
        get_character_heat_prior(character_heat_priors, region, condition, int(line_character_id))
        if int(line_character_id) > 0
        else 0.0
    )
    unit_heat = float(np.mean(np.asarray(unit_values, dtype=np.float64))) if unit_values else 0.0
    main_heat = float(np.mean(np.asarray(main_values, dtype=np.float64))) if main_values else 0.0
    character_heat = float(np.mean(np.asarray(character_values, dtype=np.float64))) if character_values else 0.0

    if condition == "box":
        event_heat = max(unit_heat, 0.55 * unit_heat + 0.45 * main_heat, character_heat)
    elif condition == "world_bloom":
        event_heat = max(line_value, main_heat, character_heat)
    elif condition == "mixed":
        event_heat = 0.50 * character_heat + 0.30 * main_heat + 0.20 * unit_heat
    else:
        event_heat = 0.45 * character_heat + 0.35 * main_heat + 0.20 * unit_heat
    return clamp(event_heat, -0.22, 0.28)


def weighted_recent_log_mean(values: list[tuple[int, float]], window: int = EVENT_HEAT_PRIOR_RECENT_WINDOW) -> tuple[float, int]:
    if not values:
        return 0.0, 0
    recent = sorted(values, key=lambda item: item[0])[-int(window):]
    weights = np.asarray(
        [
            EVENT_HEAT_PRIOR_RECENCY_DECAY ** (len(recent) - index - 1)
            for index in range(len(recent))
        ],
        dtype=np.float64,
    )
    scores = np.asarray([score for _event_id, score in recent], dtype=np.float64)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return float(np.mean(scores)), len(recent)
    return float(np.sum(scores * weights) / weight_sum), len(recent)


def build_event_heat_final_priors(
    event_meta: dict[tuple[str, int], EventMeta],
    finals: dict[tuple[str, int, int, int], tuple[int, int]],
    validation_events: dict[str, set[int]],
    event_feature_contexts: dict[tuple[str, int], EventFeatureContext],
    character_heat_priors: dict[str, dict[object, float]],
    model_series_types: frozenset[str] = NORMAL_MODEL_SERIES_TYPES,
) -> dict[str, object]:
    condition_histories: dict[tuple[str, str, int], list[tuple[int, float]]] = defaultdict(list)
    region_histories: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
    global_histories: dict[tuple[int], list[tuple[int, float]]] = defaultdict(list)
    examples: list[dict[str, object]] = []

    for (region, event_id, character_id, rank), (score, _timestamp) in finals.items():
        if event_id in validation_events[region]:
            continue
        meta = event_meta.get((region, event_id))
        if meta is None or meta.series_type not in model_series_types:
            continue
        context = event_feature_contexts.get((region, event_id))
        if context is None:
            continue
        if meta.series_type != "world_bloom" and character_id != 0:
            continue
        if score <= 0:
            continue
        log_score = safe_log1p(score)
        condition = str(context.condition)
        condition_histories[(region, condition, int(rank))].append((int(event_id), log_score))
        region_histories[(region, int(rank))].append((int(event_id), log_score))
        global_histories[(int(rank),)].append((int(event_id), log_score))

    for values in (condition_histories, region_histories, global_histories):
        for key in values:
            values[key].sort(key=lambda item: item[0])

    def serialize_history_map(
        mapping: dict[tuple[object, ...], list[tuple[int, float]]],
    ) -> dict[tuple[object, ...], list[tuple[int, float]]]:
        return {
            key: list(values)
            for key, values in mapping.items()
            if values
        }

    for (region, event_id), meta in sorted(event_meta.items()):
        if event_id in validation_events[region] or meta.series_type not in model_series_types:
            continue
        context = event_feature_contexts.get((region, event_id))
        if context is None:
            continue
        event_heat = aggregate_event_heat_log(
            character_heat_priors,
            region,
            str(context.condition),
            context,
        )
        examples.append({
            "region": region,
            "event_id": event_id,
            "series_type": meta.series_type,
            "condition": context.condition,
            "main_character_ids": list(context.main_character_ids),
            "character_ids": list(context.character_ids),
            "unit_labels": list(context.unit_labels),
            "event_heat_log": event_heat,
        })

    return {
        "version": 1,
        "recent_window": EVENT_HEAT_PRIOR_RECENT_WINDOW,
        "condition_min_count": EVENT_HEAT_PRIOR_CONDITION_MIN_COUNT,
        "recency_decay": EVENT_HEAT_PRIOR_RECENCY_DECAY,
        "rank_weights": {
            int(rank): event_heat_rank_weight(int(rank))
            for rank in COMMON_RANKS
        },
        "condition_histories": serialize_history_map(condition_histories),
        "region_histories": serialize_history_map(region_histories),
        "global_histories": serialize_history_map(global_histories),
        "event_examples": examples,
    }


def serialize_event_feature_contexts(
    event_feature_contexts: dict[tuple[str, int], EventFeatureContext],
) -> dict[tuple[str, int], dict[str, object]]:
    return {
        key: {
            "character_ids": list(context.character_ids),
            "unit_labels": list(context.unit_labels),
            "main_character_ids": list(context.main_character_ids),
            "condition": context.condition,
        }
        for key, context in event_feature_contexts.items()
    }


def deserialize_event_feature_contexts(
    raw_contexts: object,
) -> dict[tuple[str, int], EventFeatureContext]:
    if not isinstance(raw_contexts, dict):
        return {}
    contexts: dict[tuple[str, int], EventFeatureContext] = {}
    for raw_key, raw_context in raw_contexts.items():
        if not isinstance(raw_context, dict):
            continue
        key: tuple[str, int] | None = None
        if isinstance(raw_key, tuple) and len(raw_key) == 2:
            region = str(raw_key[0])
            event_id = coerce_int(raw_key[1])
            if event_id is not None:
                key = (region, event_id)
        elif isinstance(raw_key, str):
            try:
                decoded = json.loads(raw_key)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list) and len(decoded) == 2:
                event_id = coerce_int(decoded[1])
                if event_id is not None:
                    key = (str(decoded[0]), event_id)
        if key is None:
            continue
        contexts[key] = EventFeatureContext(
            character_ids=tuple(int(item) for item in raw_context.get("character_ids", [])),
            unit_labels=tuple(str(item) for item in raw_context.get("unit_labels", [])),
            main_character_ids=tuple(int(item) for item in raw_context.get("main_character_ids", [])),
            condition=str(raw_context.get("condition") or "other"),
        )
    return contexts


def write_event_heat_prior_tables(results_dir: Path, event_heat_priors: dict[str, object]) -> None:
    with open(results_dir / "event_heat_prior_examples.tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "region",
            "event_id",
            "series_type",
            "condition",
            "event_heat_log",
            "main_character_ids",
            "character_ids",
            "unit_labels",
        ])
        for row in event_heat_priors.get("event_examples", []):
            if not isinstance(row, dict):
                continue
            writer.writerow([
                row.get("region"),
                row.get("event_id"),
                row.get("series_type"),
                row.get("condition"),
                f"{float(row.get('event_heat_log') or 0.0):.6f}",
                ",".join(str(item) for item in row.get("main_character_ids", [])),
                ",".join(str(item) for item in row.get("character_ids", [])),
                ",".join(str(item) for item in row.get("unit_labels", [])),
            ])

    with open(results_dir / "event_heat_prior_rank_baselines.tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["group", "region", "condition", "rank", "count", "recent_log_score", "recent_score"])
        for group_name, mapping_name in (
            ("condition", "condition_histories"),
            ("region", "region_histories"),
            ("global", "global_histories"),
        ):
            mapping = event_heat_priors.get(mapping_name, {})
            if not isinstance(mapping, dict):
                continue
            for key, values in sorted(mapping.items(), key=lambda item: str(item[0])):
                if not isinstance(values, list):
                    continue
                recent_log, count = weighted_recent_log_mean(values)
                if group_name == "condition":
                    region, condition, rank = key
                elif group_name == "region":
                    region, rank = key
                    condition = ""
                else:
                    (rank,) = key
                    region = ""
                    condition = ""
                writer.writerow([
                    group_name,
                    region,
                    condition,
                    rank,
                    count,
                    f"{recent_log:.6f}",
                    int(round(math.expm1(recent_log))),
                ])


def load_character_names() -> dict[int, str]:
    base = MASTERDATA_ROOTS.get("jp")
    if base is None or not base.exists():
        return {}
    names: dict[int, str] = {}
    for row in load_json(base / "gameCharacters.json"):
        character_id = parse_int(row.get("id"))
        if character_id is None:
            continue
        first_name = str(row.get("firstName") or "")
        given_name = str(row.get("givenName") or "")
        names[character_id] = f"{first_name}{given_name}" or str(character_id)
    return names


def write_heat_prior_tables(results_dir: Path, character_heat_priors: dict[str, dict[object, float]]) -> None:
    names = load_character_names()

    with open(results_dir / "wl2_character_heat_prior.tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["character_id", "name", "jp_wl2_t100_score", "prior"])
        for character_id, prior in sorted(
            CHARACTER_POPULARITY_PRIOR.items(),
            key=lambda item: -float(item[1]),
        ):
            writer.writerow([
                character_id,
                names.get(int(character_id), ""),
                JP_WL2_T100_SCORES.get(int(character_id), ""),
                f"{float(prior):.6f}",
            ])

    with open(results_dir / "unit_popularity_prior.tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["unit", "prior"])
        for unit_label in UNIT_HEAT_ORDER:
            writer.writerow([unit_label, f"{float(UNIT_HEAT_PRIOR.get(unit_label, 0.0)):.6f}"])

    with open(results_dir / "character_heat_priors_global.tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["region", "character_id", "name", "heat"])
        for (region, character_id), heat in sorted(
            character_heat_priors.get("character_by_region", {}).items(),
            key=lambda item: (str(item[0][0]), -float(item[1])),
        ):
            writer.writerow([region, character_id, names.get(int(character_id), ""), f"{float(heat):.6f}"])

    with open(results_dir / "character_heat_priors.tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["region", "character_id", "name", "heat"])
        for (region, character_id), heat in sorted(
            character_heat_priors.get("character_by_region", {}).items(),
            key=lambda item: (str(item[0][0]), -float(item[1])),
        ):
            writer.writerow([region, character_id, names.get(int(character_id), ""), f"{float(heat):.6f}"])

    with open(results_dir / "character_heat_priors_by_condition.tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["region", "condition", "character_id", "name", "heat"])
        for (region, condition, character_id), heat in sorted(
            character_heat_priors.get("character_by_region_condition", {}).items(),
            key=lambda item: (str(item[0][0]), str(item[0][1]), -float(item[1])),
        ):
            writer.writerow([region, condition, character_id, names.get(int(character_id), ""), f"{float(heat):.6f}"])

    with open(results_dir / "main_character_heat_priors.tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["region", "character_id", "name", "heat"])
        for (region, character_id), heat in sorted(
            character_heat_priors.get("main_character_by_region", {}).items(),
            key=lambda item: (str(item[0][0]), -float(item[1])),
        ):
            writer.writerow([region, character_id, names.get(int(character_id), ""), f"{float(heat):.6f}"])

    with open(results_dir / "main_character_heat_priors_by_condition.tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["region", "condition", "character_id", "name", "heat"])
        for (region, condition, character_id), heat in sorted(
            character_heat_priors.get("main_character_by_region_condition", {}).items(),
            key=lambda item: (str(item[0][0]), str(item[0][1]), -float(item[1])),
        ):
            writer.writerow([region, condition, character_id, names.get(int(character_id), ""), f"{float(heat):.6f}"])

    with open(results_dir / "unit_heat_priors_global.tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["region", "unit", "heat"])
        for (region, unit_label), heat in sorted(
            character_heat_priors.get("unit_by_region", {}).items(),
            key=lambda item: (str(item[0][0]), -float(item[1])),
        ):
            writer.writerow([region, unit_label, f"{float(heat):.6f}"])

    with open(results_dir / "unit_heat_priors_by_condition.tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["region", "condition", "unit", "heat"])
        for (region, condition, unit_label), heat in sorted(
            character_heat_priors.get("unit_by_region_condition", {}).items(),
            key=lambda item: (str(item[0][0]), str(item[0][1]), -float(item[1])),
        ):
            writer.writerow([region, condition, unit_label, f"{float(heat):.6f}"])


def finalize_shape_curves(
    bucket_values: dict[tuple[object, ...], list[float]],
) -> dict[tuple[object, ...], ShapeCurve]:
    curves: dict[tuple[object, ...], ShapeCurve] = {}
    grouped: dict[tuple[object, ...], list[tuple[float, list[float]]]] = defaultdict(list)
    for key, values in bucket_values.items():
        prefix = key[:-1]
        bucket_progress = float(key[-1])
        grouped[prefix].append((bucket_progress, values))
    for key, entries in grouped.items():
        curve = ShapeCurve()
        for bucket_progress, values in sorted(entries, key=lambda item: item[0]):
            arr = np.asarray(values, dtype=np.float64)
            curve.progresses.append(bucket_progress)
            curve.log_multipliers.append(float(np.median(arr)))
            curve.counts.append(int(arr.size))
        curves[key] = curve
    return curves


def build_jp_shape_priors(
    data_root: Path,
    event_meta: dict[tuple[str, int], EventMeta],
    finals: dict[tuple[str, int, int, int], tuple[int, int]],
    validation_events: dict[str, set[int]],
    selected_ranks: set[int],
    legacy_event_files: dict[str, list[tuple[int, str, list[Path]]]],
    model_series_types: frozenset[str] = NORMAL_MODEL_SERIES_TYPES,
) -> dict[str, dict[tuple[object, ...], ShapeCurve]]:
    series_bucket_values: dict[tuple[object, ...], list[float]] = defaultdict(list)
    global_bucket_values: dict[tuple[object, ...], list[float]] = defaultdict(list)
    calendar_bucket_values: dict[tuple[object, ...], list[float]] = defaultdict(list)
    series_calendar_bucket_values: dict[tuple[object, ...], list[float]] = defaultdict(list)
    current_event: tuple[str, int] | None = None
    histories: dict[tuple[str, int, int, int], HistoryState] = {}

    for row in iter_all_rank_timeseries_rows(data_root, "jp", selected_ranks, legacy_event_files):
        event_key = ("jp", row["event_id"])
        if row["event_id"] in validation_events.get("jp", set()):
            continue
        meta = event_meta.get(event_key)
        if meta is None:
            continue
        if meta.series_type not in model_series_types:
            continue
        if not has_event_line_window(meta, row["character_id"]):
            continue
        if current_event != event_key:
            histories.clear()
            current_event = event_key

        series_key = ("jp", row["event_id"], row["character_id"], row["rank"])
        final_score, final_timestamp = finals.get(series_key, (None, None))
        if final_score is None or final_timestamp is None:
            continue
        line_end_timestamp = event_line_end_timestamp(meta, row["character_id"])
        if row["timestamp"] >= final_timestamp or row["timestamp"] >= line_end_timestamp:
            histories.setdefault(series_key, HistoryState()).timestamps.append(row["timestamp"])
            histories[series_key].scores.append(row["score"])
            continue

        progress = event_line_progress(meta, row["character_id"], row["timestamp"])
        history = histories.setdefault(series_key, HistoryState())
        current_score = max(int(row["score"]), 1)
        final_value = max(int(final_score), current_score)

        while history.next_bucket_index < len(BUCKETS) and progress >= BUCKETS[history.next_bucket_index]:
            bucket_progress = BUCKETS[history.next_bucket_index]
            log_multiplier = math.log(final_value / current_score)
            calendar_segment = local_calendar_segment("jp", row["timestamp"])
            series_bucket_values[(meta.series_type, row["rank"], bucket_progress)].append(log_multiplier)
            global_bucket_values[(row["rank"], bucket_progress)].append(log_multiplier)
            calendar_bucket_values[(calendar_segment, row["rank"], bucket_progress)].append(log_multiplier)
            series_calendar_bucket_values[(meta.series_type, calendar_segment, row["rank"], bucket_progress)].append(log_multiplier)
            history.next_bucket_index += 1

        history.timestamps.append(row["timestamp"])
        history.scores.append(row["score"])

    return {
        "series": finalize_shape_curves(series_bucket_values),
        "global": finalize_shape_curves(global_bucket_values),
        "calendar": finalize_shape_curves(calendar_bucket_values),
        "series_calendar": finalize_shape_curves(series_calendar_bucket_values),
    }


def serialize_shape_priors(
    priors: dict[str, dict[tuple[object, ...], ShapeCurve]],
) -> dict[str, dict[tuple[object, ...], dict[str, list[float] | list[int]]]]:
    serialized: dict[str, dict[tuple[object, ...], dict[str, list[float] | list[int]]]] = {}
    for group, curves in priors.items():
        serialized[group] = {
            key: {
                "progresses": list(curve.progresses),
                "log_multipliers": list(curve.log_multipliers),
                "counts": list(curve.counts),
            }
            for key, curve in curves.items()
        }
    return serialized


def interpolate_shape_curve(curve: ShapeCurve | None, progress: float) -> tuple[float | None, float]:
    if curve is None or not curve.progresses:
        return None, 0.0
    if progress <= curve.progresses[0]:
        return curve.log_multipliers[0], float(curve.counts[0])
    if progress >= curve.progresses[-1]:
        return curve.log_multipliers[-1], float(curve.counts[-1])

    index = bisect.bisect_left(curve.progresses, progress)
    left_progress = curve.progresses[index - 1]
    right_progress = curve.progresses[index]
    left_value = curve.log_multipliers[index - 1]
    right_value = curve.log_multipliers[index]
    left_count = float(curve.counts[index - 1])
    right_count = float(curve.counts[index])
    if right_progress <= left_progress:
        return right_value, right_count
    weight = (progress - left_progress) / (right_progress - left_progress)
    return (
        left_value + (right_value - left_value) * weight,
        left_count + (right_count - left_count) * weight,
    )


def projection_from_log_multiplier(current_score: int, log_multiplier: float | None) -> float:
    if log_multiplier is None or current_score <= 0:
        return float(current_score)
    return max(float(current_score), float(current_score) * math.exp(log_multiplier))


def geometric_mean_projection(left: float, right: float, floor: float) -> float:
    left_value = max(left, floor, 1.0)
    right_value = max(right, floor, 1.0)
    return max(float(floor), math.sqrt(left_value * right_value))


def weighted_shape_projection(shape_projection: float, linear_projection: float, floor: float, shape_weight: float) -> float:
    shape_value = max(shape_projection, floor, 1.0)
    linear_value = max(linear_projection, floor, 1.0)
    weight = min(max(float(shape_weight), 0.0), 1.0)
    return max(float(floor), math.exp(math.log(shape_value) * weight + math.log(linear_value) * (1.0 - weight)))


def jp_transfer_shape_weight(region: str, support: float, progress: float) -> float:
    if support <= 0.0:
        return 0.50
    if region == "jp":
        return 0.55
    if progress < 0.25:
        return 0.75
    if progress < 0.75:
        return 0.70
    return 0.60


def jp_shape_features(
    meta: EventMeta,
    rank: int,
    bucket_progress: float,
    current_timestamp: int,
    current_score: int,
    linear_projection: float,
    jp_shape_priors: dict[str, dict[tuple[object, ...], ShapeCurve]],
) -> tuple[list[float], dict[str, float]]:
    series_curve = jp_shape_priors.get("series", {}).get((meta.series_type, rank))
    global_curve = jp_shape_priors.get("global", {}).get((rank,))
    calendar_segment = local_calendar_segment(meta.region, current_timestamp)
    series_calendar_curve = jp_shape_priors.get("series_calendar", {}).get((meta.series_type, calendar_segment, rank))
    calendar_curve = jp_shape_priors.get("calendar", {}).get((calendar_segment, rank))
    series_log_multiplier, series_support = interpolate_shape_curve(series_curve, bucket_progress)
    global_log_multiplier, global_support = interpolate_shape_curve(global_curve, bucket_progress)
    series_calendar_log_multiplier, series_calendar_support = interpolate_shape_curve(series_calendar_curve, bucket_progress)
    calendar_log_multiplier, calendar_support = interpolate_shape_curve(calendar_curve, bucket_progress)

    if series_log_multiplier is None:
        series_log_multiplier = global_log_multiplier
        series_support = global_support
    if global_log_multiplier is None:
        global_log_multiplier = series_log_multiplier
        global_support = series_support
    if series_calendar_log_multiplier is None:
        series_calendar_log_multiplier = calendar_log_multiplier
        series_calendar_support = calendar_support
    if calendar_log_multiplier is None:
        calendar_log_multiplier = series_calendar_log_multiplier
        calendar_support = series_calendar_support

    series_projection = projection_from_log_multiplier(current_score, series_log_multiplier)
    global_projection = projection_from_log_multiplier(current_score, global_log_multiplier)
    series_calendar_projection = projection_from_log_multiplier(current_score, series_calendar_log_multiplier)
    calendar_projection = projection_from_log_multiplier(current_score, calendar_log_multiplier)
    shape_projection = series_projection if series_support > 0 else global_projection
    calendar_shape_projection = series_calendar_projection if series_calendar_support > 0 else calendar_projection
    transfer_shape_projection = geometric_mean_projection(shape_projection, calendar_shape_projection, current_score)
    transfer_support = max(series_support, global_support, series_calendar_support, calendar_support)
    hybrid_projection = weighted_shape_projection(
        transfer_shape_projection,
        linear_projection,
        current_score,
        jp_transfer_shape_weight(meta.region, transfer_support, bucket_progress),
    )

    features = [
        safe_log1p(series_projection),
        safe_log1p(global_projection),
        float(series_log_multiplier or 0.0),
        float(global_log_multiplier or 0.0),
        float(series_support),
        float(global_support),
        safe_log1p(shape_projection),
        safe_log1p(hybrid_projection),
        safe_log1p(calendar_shape_projection),
        float(series_calendar_log_multiplier or 0.0),
        float(calendar_log_multiplier or 0.0),
        float(series_calendar_support),
        float(calendar_support),
        signed_log1p(calendar_shape_projection - linear_projection),
        signed_log1p(transfer_shape_projection - linear_projection),
    ]
    baselines = {
        "jp_series_projection": series_projection,
        "jp_global_projection": global_projection,
        "jp_calendar_projection": calendar_shape_projection,
        "jp_shape_projection": shape_projection,
        "hybrid_projection": hybrid_projection,
    }
    return features, baselines


def aggregate_character_heat_features(
    event_key: tuple[str, int],
    line_character_id: int,
    event_feature_contexts: dict[tuple[str, int], EventFeatureContext],
    character_heat_priors: dict[str, dict[object, float]],
) -> list[float]:
    region = event_key[0]
    context = event_feature_contexts.get(event_key, EventFeatureContext())
    condition = context.condition
    characters = context.character_ids
    units = context.unit_labels
    main_characters = context.main_character_ids
    character_by_region = character_heat_priors.get("character_by_region", {})
    character_by_region_condition = character_heat_priors.get("character_by_region_condition", {})
    main_character_by_region = character_heat_priors.get("main_character_by_region", {})
    main_character_by_region_condition = character_heat_priors.get("main_character_by_region_condition", {})
    heats = [
        get_character_heat_prior(character_heat_priors, region, condition, cid)
        for cid in characters
        if (region, cid) in character_by_region
        or (region, condition, cid) in character_by_region_condition
    ]
    unit_heats = [
        get_unit_heat_prior(character_heat_priors, region, condition, unit_label)
        for unit_label in units
        if (region, unit_label) in character_heat_priors.get("unit_by_region", {})
        or (region, condition, unit_label) in character_heat_priors.get("unit_by_region_condition", {})
    ]
    main_heats = [
        get_main_character_heat_prior(character_heat_priors, region, condition, cid)
        for cid in main_characters
        if (region, cid) in main_character_by_region
        or (region, condition, cid) in main_character_by_region_condition
    ]
    total_count = len(characters)
    known_count = len(heats)
    if heats:
        arr = np.asarray(heats, dtype=np.float64)
        mean_heat = float(np.mean(arr))
        max_heat = float(np.max(arr))
        min_heat = float(np.min(arr))
        std_heat = float(np.std(arr))
    else:
        mean_heat = 0.0
        max_heat = 0.0
        min_heat = 0.0
        std_heat = 0.0
    unit_total_count = len(units)
    unit_known_count = len(unit_heats)
    if unit_heats:
        unit_arr = np.asarray(unit_heats, dtype=np.float64)
        unit_mean_heat = float(np.mean(unit_arr))
        unit_max_heat = float(np.max(unit_arr))
        unit_min_heat = float(np.min(unit_arr))
        unit_std_heat = float(np.std(unit_arr))
    else:
        unit_mean_heat = 0.0
        unit_max_heat = 0.0
        unit_min_heat = 0.0
        unit_std_heat = 0.0
    main_total_count = len(main_characters)
    main_known_count = len(main_heats)
    if main_heats:
        main_arr = np.asarray(main_heats, dtype=np.float64)
        main_mean_heat = float(np.mean(main_arr))
        main_max_heat = float(np.max(main_arr))
        main_min_heat = float(np.min(main_arr))
        main_std_heat = float(np.std(main_arr))
    else:
        main_mean_heat = 0.0
        main_max_heat = 0.0
        main_min_heat = 0.0
        main_std_heat = 0.0
    line_character_heat = (
        get_character_heat_prior(character_heat_priors, region, condition, line_character_id)
        if line_character_id > 0
        else 0.0
    )
    line_main_character_heat = (
        get_main_character_heat_prior(character_heat_priors, region, condition, line_character_id)
        if line_character_id > 0
        else 0.0
    )
    return [
        float(total_count),
        float(known_count),
        float(known_count / total_count) if total_count > 0 else 0.0,
        mean_heat,
        max_heat,
        min_heat,
        std_heat,
        line_character_heat,
        float(unit_total_count),
        float(unit_known_count),
        float(unit_known_count / unit_total_count) if unit_total_count > 0 else 0.0,
        unit_mean_heat,
        unit_max_heat,
        unit_min_heat,
        unit_std_heat,
        unit_mean_heat - mean_heat,
        float(main_total_count),
        float(main_known_count),
        float(main_known_count / main_total_count) if main_total_count > 0 else 0.0,
        main_mean_heat,
        main_max_heat,
        main_min_heat,
        main_std_heat,
        line_main_character_heat,
    ]


def event_condition_features(
    event_key: tuple[str, int],
    event_feature_contexts: dict[tuple[str, int], EventFeatureContext],
) -> list[float]:
    context = event_feature_contexts.get(event_key, EventFeatureContext())
    return [
        1.0 if context.condition == "box" else 0.0,
        1.0 if context.condition == "mixed" else 0.0,
        1.0 if context.condition == "world_bloom" else 0.0,
        float(context.featured_unit_count),
    ]


def lookback_features(history: HistoryState, current_timestamp: int, current_score: int) -> dict[str, float]:
    features: dict[str, float] = {}
    for name, seconds in LOOKBACK_WINDOWS:
        idx = bisect.bisect_right(history.timestamps, current_timestamp - seconds) - 1
        has_window = 1.0 if idx >= 0 else 0.0
        if idx >= 0:
            old_timestamp = history.timestamps[idx]
            old_score = history.scores[idx]
            delta_score = float(current_score - old_score)
            delta_hours = max((current_timestamp - old_timestamp) / 3600.0, 1e-6)
            pace = delta_score / delta_hours
        else:
            delta_score = 0.0
            delta_hours = 0.0
            pace = 0.0
        features[f"has_{name}"] = has_window
        features[f"delta_score_{name}"] = delta_score
        features[f"delta_hours_{name}"] = delta_hours
        features[f"pace_{name}"] = pace
    if history.timestamps:
        prev_delta_score = float(current_score - history.scores[-1])
        prev_delta_hours = max((current_timestamp - history.timestamps[-1]) / 3600.0, 1e-6)
    else:
        prev_delta_score = 0.0
        prev_delta_hours = 0.0
    features["prev_delta_score"] = prev_delta_score
    features["prev_delta_hours"] = prev_delta_hours
    features["prev_pace"] = prev_delta_score / prev_delta_hours if prev_delta_hours > 0 else 0.0
    return features


def projection_from_pace(current_score: int, pace: float, hours_remaining: float) -> float:
    return max(float(current_score), float(current_score) + pace * max(hours_remaining, 0.0))


def make_feature_vector(
    row: dict,
    meta: EventMeta,
    history: HistoryState,
    bucket_progress: float,
    event_feature_contexts: dict[tuple[str, int], EventFeatureContext],
    character_heat_priors: dict[str, dict[object, float]],
    jp_shape_priors: dict[str, dict[tuple[object, ...], ShapeCurve]],
    boundary_pressure_contexts: dict[tuple[str, int, int, int], BoundaryPressureSeries] | None = None,
) -> tuple[list[float], dict[str, float]]:
    current_score = row["score"]
    current_timestamp = row["timestamp"]
    line_start_timestamp, line_end_timestamp = event_line_window(meta, row["character_id"])
    line_duration_seconds = event_line_duration_seconds(meta, row["character_id"])
    hours_elapsed = max((current_timestamp - line_start_timestamp) / 3600.0, 0.0)
    hours_remaining = max((line_end_timestamp - current_timestamp) / 3600.0, 0.0)
    linear_projection = current_score / max(bucket_progress, 0.01)
    lb = lookback_features(history, current_timestamp, current_score)
    proj_1h = projection_from_pace(current_score, lb["pace_1h"], hours_remaining)
    proj_6h = projection_from_pace(current_score, lb["pace_6h"], hours_remaining)
    proj_24h = projection_from_pace(current_score, lb["pace_24h"], hours_remaining)
    temporal_features = local_temporal_features(meta.region, current_timestamp)
    character_heat_features = aggregate_character_heat_features(
        event_key=(meta.region, meta.event_id),
        line_character_id=row["character_id"],
        event_feature_contexts=event_feature_contexts,
        character_heat_priors=character_heat_priors,
    )
    shape_features, shape_baselines = jp_shape_features(
        meta=meta,
        rank=row["rank"],
        bucket_progress=bucket_progress,
        current_timestamp=current_timestamp,
        current_score=current_score,
        linear_projection=linear_projection,
        jp_shape_priors=jp_shape_priors,
    )
    pace_trend_features = [
        signed_log1p(lb["pace_1h"] - lb["pace_6h"]),
        signed_log1p(lb["pace_6h"] - lb["pace_24h"]),
        signed_log1p(lb["prev_pace"] - lb["pace_1h"]),
        signed_log1p(shape_baselines["jp_shape_projection"] - linear_projection),
        signed_log1p(proj_6h - linear_projection),
    ]
    edge_features = score_edge_features(current_score, linear_projection, hours_remaining)
    pressure_features = boundary_pressure_features(
        event_key=(meta.region, meta.event_id),
        character_id=row["character_id"],
        rank=row["rank"],
        timestamp=current_timestamp,
        boundary_pressure_contexts=boundary_pressure_contexts,
    )
    feature_vector = (
        region_one_hot(meta.region)
        + [
            1.0 if meta.series_type == "world_bloom" else 0.0,
            1.0 if row["character_id"] > 0 else 0.0,
            float(row["character_id"]),
            math.log10(row["rank"] + 1.0),
            bucket_progress,
            hours_elapsed,
            hours_remaining,
            line_duration_seconds / 3600.0,
            safe_log1p(current_score),
            safe_log1p(linear_projection),
            safe_log1p(proj_1h),
            safe_log1p(proj_6h),
            safe_log1p(proj_24h),
            safe_log1p(lb["delta_score_1h"]),
            safe_log1p(lb["delta_score_6h"]),
            safe_log1p(lb["delta_score_24h"]),
            lb["pace_1h"],
            lb["pace_6h"],
            lb["pace_24h"],
            lb["prev_pace"],
            lb["has_1h"],
            lb["has_6h"],
            lb["has_24h"],
        ]
        + temporal_features
        + pace_trend_features
        + edge_features
        + character_heat_features
        + shape_features
        + event_condition_features((meta.region, meta.event_id), event_feature_contexts)
        + pressure_features
    )
    baselines = {
        "linear_projection": linear_projection,
        "pace_6h_projection": proj_6h,
        **shape_baselines,
    }
    return feature_vector, baselines


def build_dataset(
    data_root: Path,
    event_meta: dict[tuple[str, int], EventMeta],
    finals: dict[tuple[str, int, int, int], tuple[int, int]],
    validation_events: dict[str, set[int]],
    selected_ranks: set[int],
    event_feature_contexts: dict[tuple[str, int], EventFeatureContext],
    character_heat_priors: dict[str, dict[object, float]],
    jp_shape_priors: dict[str, dict[tuple[object, ...], ShapeCurve]],
    boundary_pressure_contexts: dict[tuple[str, int, int, int], BoundaryPressureSeries],
    legacy_event_files: dict[str, list[tuple[int, str, list[Path]]]],
    model_series_types: frozenset[str] = NORMAL_MODEL_SERIES_TYPES,
):
    train_x: list[list[float]] = []
    train_y: list[float] = []
    train_rows: list[dict[str, float | int | str]] = []
    val_x: list[list[float]] = []
    val_y: list[float] = []
    val_rows: list[dict[str, float | int | str]] = []

    for region in REGIONS:
        current_event: tuple[str, int] | None = None
        histories: dict[tuple[str, int, int, int], HistoryState] = {}
        for row in iter_all_rank_timeseries_rows(
            data_root,
            region,
            selected_ranks,
            legacy_event_files,
            model_series_types,
        ):
            event_key = (region, row["event_id"])
            meta = event_meta.get(event_key)
            if meta is None:
                continue
            if meta.series_type not in model_series_types:
                continue
            if not has_event_line_window(meta, row["character_id"]):
                continue
            if current_event != event_key:
                histories.clear()
                current_event = event_key
            series_key = (region, row["event_id"], row["character_id"], row["rank"])
            final_score, final_timestamp = finals.get(series_key, (None, None))
            if final_score is None or final_timestamp is None:
                continue
            line_end_timestamp = event_line_end_timestamp(meta, row["character_id"])
            if row["timestamp"] >= final_timestamp or row["timestamp"] >= line_end_timestamp:
                histories.setdefault(series_key, HistoryState()).timestamps.append(row["timestamp"])
                histories[series_key].scores.append(row["score"])
                continue

            progress = event_line_progress(meta, row["character_id"], row["timestamp"])
            history = histories.setdefault(series_key, HistoryState())

            while history.next_bucket_index < len(BUCKETS) and progress >= BUCKETS[history.next_bucket_index]:
                bucket_progress = BUCKETS[history.next_bucket_index]
                if progress - bucket_progress > 0.025:
                    history.next_bucket_index += 1
                    continue
                feature_vector, baselines = make_feature_vector(
                    row,
                    meta,
                    history,
                    bucket_progress,
                    event_feature_contexts,
                    character_heat_priors,
                    jp_shape_priors,
                    boundary_pressure_contexts,
                )
                context = event_feature_contexts.get((region, row["event_id"]), EventFeatureContext())
                target = safe_log1p(final_score)
                sample_row = {
                    "region": region,
                    "event_id": row["event_id"],
                    "series_type": meta.series_type,
                    "event_condition": context.condition,
                    "character_id": row["character_id"],
                    "rank": row["rank"],
                    "progress": bucket_progress,
                    "current_timestamp": row["timestamp"],
                    "end_timestamp": event_line_end_timestamp(meta, row["character_id"]),
                    "current_calendar_segment": local_calendar_segment(region, row["timestamp"]),
                    "end_calendar_segment": local_calendar_segment(
                        region,
                        event_line_end_timestamp(meta, row["character_id"]) - 1,
                    ),
                    "current_score": row["score"],
                    "final_score": final_score,
                    "linear_projection": baselines["linear_projection"],
                    "pace_6h_projection": baselines["pace_6h_projection"],
                    "jp_shape_projection": baselines["jp_shape_projection"],
                    "hybrid_projection": baselines["hybrid_projection"],
                    "event_featured_character_count": context.featured_character_count,
                    "event_featured_unit_count": context.featured_unit_count,
                    "line_character_heat": get_character_heat_prior(
                        character_heat_priors,
                        region,
                        context.condition,
                        row["character_id"],
                    ),
                }
                sample_row["wl2_character_prior_multiplier"] = wl2_character_prior_multiplier(sample_row)
                if row["event_id"] in validation_events[region]:
                    val_x.append(feature_vector)
                    val_y.append(target)
                    val_rows.append(sample_row)
                else:
                    train_x.append(feature_vector)
                    train_y.append(target)
                    train_rows.append(sample_row)
                history.next_bucket_index += 1

            history.timestamps.append(row["timestamp"])
            history.scores.append(row["score"])

    return (
        np.asarray(train_x, dtype=np.float64),
        np.asarray(train_y, dtype=np.float64),
        train_rows,
        np.asarray(val_x, dtype=np.float64),
        np.asarray(val_y, dtype=np.float64),
        val_rows,
    )


def calibration_keys(row: dict[str, float | int | str]) -> list[tuple[object, ...]]:
    region = str(row["region"])
    series_type = str(row["series_type"])
    character_id = int(row.get("character_id", 0))
    progress_name = progress_bucket_name(float(row["progress"]))
    rank_name = rank_bucket_name(int(row["rank"]))
    keys = []
    if character_id > 0:
        keys.extend([
            ("region_series_character_progress_rank", region, series_type, character_id, progress_name, rank_name),
            ("region_series_character_rank", region, series_type, character_id, rank_name),
            ("region_series_character", region, series_type, character_id),
            ("series_character_rank", series_type, character_id, rank_name),
            ("series_character", series_type, character_id),
        ])
    keys.extend([
        ("region_series_progress_rank", region, series_type, progress_name, rank_name),
        ("region_progress_rank", region, progress_name, rank_name),
        ("region_series_progress", region, series_type, progress_name),
        ("region_progress", region, progress_name),
        ("progress_rank", progress_name, rank_name),
        ("region_rank", region, rank_name),
        ("region_series", region, series_type),
        ("region", region),
        ("progress", progress_name),
        ("rank", rank_name),
        ("global",),
    ])
    return keys


def assign_event_folds(
    rows: list[dict[str, float | int | str]],
    fold_count: int,
) -> list[int]:
    event_indices: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        event_indices[str(row["region"])].append(int(row["event_id"]))

    per_region_event_folds: dict[tuple[str, int], int] = {}
    for region, event_ids in event_indices.items():
        unique_ids = sorted(set(event_ids))
        splits = np.array_split(np.asarray(unique_ids, dtype=np.int64), min(fold_count, len(unique_ids)))
        for fold_idx, split in enumerate(splits):
            for event_id in split.tolist():
                per_region_event_folds[(region, int(event_id))] = fold_idx

    return [per_region_event_folds[(str(row["region"]), int(row["event_id"]))] for row in rows]


def build_oof_residual_predictions(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_rows: list[dict[str, float | int | str]],
    baseline_log_feature_index: int,
    fold_count: int = 4,
) -> np.ndarray:
    fold_ids = np.asarray(assign_event_folds(train_rows, fold_count), dtype=np.int64)
    predictions = np.zeros(train_y.shape[0], dtype=np.float64)
    for fold in sorted(set(fold_ids.tolist())):
        val_mask = fold_ids == fold
        train_mask = ~val_mask
        model = make_boosting_regressor()
        model.fit(
            train_x[train_mask],
            train_y[train_mask] - train_x[train_mask, baseline_log_feature_index],
        )
        fold_log_predictions = model.predict(train_x[val_mask]) + train_x[val_mask, baseline_log_feature_index]
        predictions[val_mask] = np.expm1(fold_log_predictions)
    return predictions


def predict_direct_scores(model: HistGradientBoostingRegressor, x: np.ndarray) -> np.ndarray:
    return np.expm1(model.predict(x))


def predict_residual_scores(
    model: HistGradientBoostingRegressor,
    x: np.ndarray,
    baseline_log_feature_index: int,
) -> np.ndarray:
    return np.expm1(model.predict(x) + x[:, baseline_log_feature_index])


def log_blend_predictions(
    left_predictions: np.ndarray,
    right_predictions: np.ndarray,
    left_weight: float | np.ndarray,
) -> np.ndarray:
    weights = np.asarray(left_weight, dtype=np.float64)
    return np.expm1(
        np.log1p(np.maximum(left_predictions, 0.0)) * weights
        + np.log1p(np.maximum(right_predictions, 0.0)) * (1.0 - weights)
    )


def predict_residual_blend_scores(
    model_bundle: dict[str, HistGradientBoostingRegressor],
    x: np.ndarray,
    rows: list[dict[str, float | int | str]],
    linear_log_feature_index: int,
    hybrid_log_feature_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    linear_predictions = predict_residual_scores(
        model_bundle["residual_linear_model"],
        x,
        linear_log_feature_index,
    )
    hybrid_predictions = predict_residual_scores(
        model_bundle["residual_hybrid_model"],
        x,
        hybrid_log_feature_index,
    )
    weights = np.asarray(
        [
            get_residual_blend_weight(str(row["region"]), float(row["progress"]))
            for row in rows
        ],
        dtype=np.float64,
    )
    return (
        log_blend_predictions(linear_predictions, hybrid_predictions, weights),
        linear_predictions,
        hybrid_predictions,
    )


def build_cn_sidecar_features(
    x: np.ndarray,
    base_predictions: np.ndarray,
    linear_predictions: np.ndarray,
    hybrid_predictions: np.ndarray,
) -> np.ndarray:
    current_scores = np.expm1(x[:, CURRENT_SCORE_LOG_FEATURE_INDEX])
    linear_projection = np.expm1(x[:, LINEAR_LOG_FEATURE_INDEX])
    hours_elapsed = np.maximum(x[:, HOURS_ELAPSED_FEATURE_INDEX], 1e-6)
    hours_remaining = np.maximum(x[:, HOURS_REMAINING_FEATURE_INDEX], 1e-6)
    pace_1h = x[:, PACE_1H_FEATURE_INDEX]
    pace_6h = x[:, PACE_6H_FEATURE_INDEX]
    pace_24h = x[:, PACE_24H_FEATURE_INDEX]
    prev_pace = x[:, PREV_PACE_FEATURE_INDEX]

    current_avg_pace = current_scores / hours_elapsed
    linear_required_pace = np.maximum(linear_projection - current_scores, 0.0) / hours_remaining
    base_required_pace = np.maximum(np.asarray(base_predictions, dtype=np.float64) - current_scores, 0.0) / hours_remaining
    hybrid_required_pace = np.maximum(np.asarray(hybrid_predictions, dtype=np.float64) - current_scores, 0.0) / hours_remaining

    base_log_predictions = np_safe_log1p(base_predictions).reshape(-1, 1)
    linear_hybrid_gap = (
        np_safe_log1p(linear_predictions) - np_safe_log1p(hybrid_predictions)
    ).reshape(-1, 1)
    pace_context = np.column_stack(
        (
            np_safe_log1p(current_avg_pace),
            np_safe_log1p(linear_required_pace),
            np_safe_log1p(base_required_pace),
            np_safe_log1p(hybrid_required_pace),
            np_signed_log1p(pace_1h - current_avg_pace),
            np_signed_log1p(pace_6h - current_avg_pace),
            np_signed_log1p(pace_24h - current_avg_pace),
            np_signed_log1p(prev_pace - current_avg_pace),
            np_signed_log1p(pace_1h - base_required_pace),
            np_signed_log1p(pace_6h - hybrid_required_pace),
        )
    )
    return np.concatenate((x, base_log_predictions, linear_hybrid_gap, pace_context), axis=1)


def should_apply_cn_tail_sidecar(row: dict[str, float | int | str]) -> bool:
    return str(row["region"]) == CN_SIDECAR_REGION and int(row["rank"]) > CN_SIDECAR_APPLY_MIN_RANK


def apply_cn_tail_sidecar(
    base_predictions: np.ndarray,
    sidecar_predictions: np.ndarray,
    rows: list[dict[str, float | int | str]],
) -> tuple[np.ndarray, np.ndarray]:
    adjusted = np.asarray(base_predictions, dtype=np.float64).copy()
    mask = np.asarray([should_apply_cn_tail_sidecar(row) for row in rows], dtype=bool)
    if np.any(mask):
        adjusted[mask] = log_blend_predictions(
            adjusted[mask],
            np.asarray(sidecar_predictions, dtype=np.float64)[mask],
            CN_SIDECAR_BASE_WEIGHT,
        )
    return adjusted, mask


def rank_boundary_weekend_flag(row: dict[str, float | int | str]) -> float:
    region = str(row.get("region", ""))
    current_timestamp = row.get("current_timestamp")
    end_timestamp = row.get("end_timestamp")
    if current_timestamp is not None and end_timestamp is not None:
        final_window_start = max(int(current_timestamp), int(end_timestamp) - 36 * 3600)
        return weekend_overlap_flag(region, final_window_start, int(end_timestamp) - 1)
    current_segment = str(row.get("current_calendar_segment", ""))
    end_segment = str(row.get("end_calendar_segment", ""))
    return 1.0 if current_segment == "weekend" or end_segment == "weekend" else 0.0


def rank_boundary_calibration_multiplier(
    row: dict[str, float | int | str],
    config: dict[object, object] | None = None,
) -> float:
    if config is None:
        config = RANK_BOUNDARY_CALIBRATION
    rank = int(row.get("rank", 0))
    params = config.get(rank) or config.get(str(rank))
    if not isinstance(params, dict):
        return 1.0
    progress = clamp(float(row.get("progress", 0.0)), 0.0, 1.0)
    late_start = float(params.get("late_start", 0.65))
    final_start = float(params.get("final_start", 0.88))
    late_strength = clamp((progress - late_start) / max(1.0 - late_start, 1e-6), 0.0, 1.0)
    final_strength = clamp((progress - final_start) / max(1.0 - final_start, 1e-6), 0.0, 1.0)
    weekend_strength = rank_boundary_weekend_flag(row)
    multiplier = (
        1.0
        + float(params.get("base", 0.0))
        + float(params.get("late", 0.0)) * late_strength
        + float(params.get("final", 0.0)) * final_strength
        + float(params.get("weekend_late", 0.0)) * late_strength * weekend_strength
        + float(params.get("weekend_final", 0.0)) * final_strength * weekend_strength
    )
    region_weights = params.get("region_weights", {})
    if isinstance(region_weights, dict):
        region_weight = float(region_weights.get(str(row.get("region", "")), 1.0))
        multiplier = 1.0 + (multiplier - 1.0) * region_weight
    return clamp(multiplier, 1.0, float(params.get("max_multiplier", multiplier)))


def apply_rank_boundary_calibration_batch(
    rows: list[dict[str, float | int | str]],
    predictions: np.ndarray,
    config: dict[object, object] | None = None,
) -> np.ndarray:
    if config is None:
        return np.asarray(predictions, dtype=np.float64)
    return np.asarray(
        [
            max(float(prediction), 1.0) * rank_boundary_calibration_multiplier(row, config)
            for row, prediction in zip(rows, predictions, strict=True)
        ],
        dtype=np.float64,
    )


def predict_payload_scores(
    payload: dict[str, object],
    x: np.ndarray,
    rows: list[dict[str, float | int | str]],
) -> np.ndarray:
    model_name = str(payload["model_name"])
    if model_name == "region_routed_model":
        region_model_names = payload.get("region_model_names", {})
        candidate_payloads = payload.get("candidate_payloads", {})
        fallback_model_name = str(payload.get("fallback_model_name", ""))
        routed = np.zeros(len(rows), dtype=np.float64)
        for idx, row in enumerate(rows):
            region = str(row["region"])
            candidate_name = str(region_model_names.get(region, fallback_model_name))
            candidate_payload = candidate_payloads.get(candidate_name)
            if not isinstance(candidate_payload, dict):
                raise ValueError(f"missing routed payload for region={region} model={candidate_name}")
            routed[idx] = predict_payload_scores(candidate_payload, x[idx : idx + 1], [row])[0]
        return routed

    linear_log_feature_index = int(payload["linear_log_feature_index"])
    hybrid_log_feature_index = int(payload["hybrid_log_feature_index"])
    pace_6h_log_feature_index = int(payload.get("pace_6h_log_feature_index", PACE_6H_PROJECTION_LOG_FEATURE_INDEX))
    model = payload.get("model")
    model_bundle = payload.get("model_bundle", {})

    if model_name == "direct_model":
        assert isinstance(model, HistGradientBoostingRegressor)
        predictions = predict_direct_scores(model, x)
    elif model_name == "residual_linear_model":
        assert isinstance(model, HistGradientBoostingRegressor)
        predictions = predict_residual_scores(model, x, linear_log_feature_index)
    elif model_name == "residual_hybrid_model":
        assert isinstance(model, HistGradientBoostingRegressor)
        predictions = predict_residual_scores(model, x, hybrid_log_feature_index)
    elif model_name == "residual_pace_6h_model":
        assert isinstance(model, HistGradientBoostingRegressor)
        predictions = predict_residual_scores(model, x, pace_6h_log_feature_index)
    elif model_name == "pace_6h_bias_calibrated_model":
        base_log_feature_index = int(payload.get("base_log_feature_index", pace_6h_log_feature_index))
        predictions = np.expm1(x[:, base_log_feature_index])
        predictions = apply_bias_calibration_batch(
            rows,
            predictions,
            payload.get("bias_calibrator"),
            min_count=int(payload.get("bias_calibration_min_count", 40)),
            shrinkage=float(payload.get("bias_calibration_shrinkage", 80.0)),
        )
    elif model_name == "residual_blend_model":
        blend_predictions, _, _ = predict_residual_blend_scores(
            model_bundle,
            x,
            rows,
            linear_log_feature_index,
            hybrid_log_feature_index,
        )
        predictions = blend_predictions
    elif model_name == "regional_calibrated_blend_model":
        blend_predictions, _, _ = predict_residual_blend_scores(
            model_bundle,
            x,
            rows,
            linear_log_feature_index,
            hybrid_log_feature_index,
        )
        predictions = apply_bias_calibration_batch(
            rows,
            blend_predictions,
            payload.get("post_blend_calibrator"),
            min_count=int(payload.get("post_blend_calibration_min_count", 24)),
            shrinkage=float(payload.get("post_blend_calibration_shrinkage", 60.0)),
        )
    elif model_name == "cn_tail_sidecar_model":
        base_predictions, linear_predictions, hybrid_predictions = predict_residual_blend_scores(
            model_bundle,
            x,
            rows,
            linear_log_feature_index,
            hybrid_log_feature_index,
        )
        sidecar_model = model_bundle["cn_sidecar_model"]
        sidecar_features = build_cn_sidecar_features(
            x,
            base_predictions,
            linear_predictions,
            hybrid_predictions,
        )
        sidecar_predictions = predict_direct_scores(sidecar_model, sidecar_features)
        adjusted_predictions, _ = apply_cn_tail_sidecar(base_predictions, sidecar_predictions, rows)
        predictions = adjusted_predictions
    else:
        raise ValueError(f"unsupported model name: {model_name}")

    predictions = apply_wl2_character_prior_batch(rows, predictions, payload)

    calibration_config = payload.get("rank_boundary_calibration")
    if isinstance(calibration_config, dict):
        predictions = apply_rank_boundary_calibration_batch(rows, predictions, calibration_config)
    return predictions


def event_heat_history_recent_log_score(
    event_heat_priors: dict[str, object],
    region: str,
    condition: str,
    rank: int,
) -> tuple[float | None, str | None, int]:
    for mapping_name, key, source in (
        ("condition_histories", (region, condition, int(rank)), "condition"),
        ("region_histories", (region, int(rank)), "region"),
        ("global_histories", (int(rank),), "global"),
    ):
        mapping = event_heat_priors.get(mapping_name, {})
        if not isinstance(mapping, dict):
            continue
        values = mapping.get(key)
        if values is None:
            values = mapping.get(tuple(str(item) for item in key))
        if not isinstance(values, list) or not values:
            continue
        recent_log, count = weighted_recent_log_mean(values)
        if source == "condition" and count < EVENT_HEAT_PRIOR_CONDITION_MIN_COUNT:
            continue
        return recent_log, source, count
    return None, None, 0


def event_heat_prior_prediction(
    row: dict[str, float | int | str],
    event_heat_priors: dict[str, object] | None,
    character_heat_priors: dict[str, dict[object, float]],
    event_feature_contexts: dict[tuple[str, int], EventFeatureContext],
) -> tuple[float | None, dict[str, float | str | int | None]]:
    debug: dict[str, float | str | int | None] = {
        "event_heat_prior_prediction": None,
        "event_heat_prior_log": 0.0,
        "event_heat_prior_source": None,
        "event_heat_prior_count": 0,
    }
    if not isinstance(event_heat_priors, dict):
        return None, debug
    region = str(row["region"])
    event_id = int(row["event_id"])
    rank = int(row["rank"])
    context = event_feature_contexts.get((region, event_id), EventFeatureContext())
    condition = str(context.condition)
    base_log_score, source, count = event_heat_history_recent_log_score(
        event_heat_priors,
        region,
        condition,
        rank,
    )
    if base_log_score is None:
        return None, debug
    event_heat_log = aggregate_event_heat_log(
        character_heat_priors,
        region,
        condition,
        context,
        int(row.get("character_id", 0)),
    )
    rank_weights = event_heat_priors.get("rank_weights", {})
    rank_weight = 1.0
    if isinstance(rank_weights, dict):
        rank_weight = float(rank_weights.get(rank, rank_weights.get(str(rank), event_heat_rank_weight(rank))))
    adjusted_log_score = float(base_log_score) + float(event_heat_log) * rank_weight
    prediction = max(1.0, math.expm1(adjusted_log_score))
    debug.update({
        "event_heat_prior_prediction": prediction,
        "event_heat_prior_log": event_heat_log,
        "event_heat_prior_source": source,
        "event_heat_prior_count": count,
    })
    return prediction, debug


def event_heat_prior_blend_weight(progress: float, rank: int) -> float:
    progress = clamp(float(progress), 0.0, 1.0)
    if progress >= 0.18:
        return 0.0
    if progress < 0.03:
        base = 0.84
    elif progress < 0.05:
        base = 0.76
    elif progress < 0.10:
        base = 0.50 - 0.32 * ((progress - 0.05) / 0.05)
    else:
        base = 0.18 * (0.18 - progress) / 0.08
    return clamp(base * (0.72 + 0.28 * event_heat_rank_weight(int(rank))), 0.0, 0.86)


def apply_event_heat_prior_batch(
    rows: list[dict[str, float | int | str]],
    predictions: np.ndarray,
    payload: dict[str, object],
) -> tuple[np.ndarray, list[dict[str, float | str | int | None]]]:
    event_heat_priors = payload.get("live_event_heat_priors")
    character_heat_priors = payload.get("live_character_heat_priors", {})
    event_feature_contexts = payload.get("live_event_feature_contexts", {})
    if not isinstance(event_heat_priors, dict) or not isinstance(character_heat_priors, dict) or not isinstance(event_feature_contexts, dict):
        return np.asarray(predictions, dtype=np.float64), [
            {
                "event_heat_prior_prediction": None,
                "event_heat_prior_log": 0.0,
                "event_heat_prior_source": None,
                "event_heat_prior_count": 0,
                "event_heat_prior_blend_weight": 0.0,
            }
            for _row in rows
        ]

    adjusted = np.asarray(predictions, dtype=np.float64).copy()
    debug_rows: list[dict[str, float | str | int | None]] = []
    for index, row in enumerate(rows):
        prior_prediction, debug = event_heat_prior_prediction(
            row,
            event_heat_priors,
            character_heat_priors,
            event_feature_contexts,
        )
        blend_weight = event_heat_prior_blend_weight(float(row.get("progress", 0.0)), int(row.get("rank", 0)))
        debug["event_heat_prior_blend_weight"] = blend_weight
        if prior_prediction is not None and blend_weight > 0.0:
            adjusted[index] = log_blend_predictions(
                np.asarray([adjusted[index]], dtype=np.float64),
                np.asarray([prior_prediction], dtype=np.float64),
                1.0 - blend_weight,
            )[0]
        debug_rows.append(debug)
    return adjusted, debug_rows


def fit_bias_calibrator(
    rows: list[dict[str, float | int | str]],
    predictions: np.ndarray,
) -> dict[tuple[object, ...], dict[str, float]]:
    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row, prediction in zip(rows, predictions, strict=True):
        actual = max(float(row["final_score"]), 1.0)
        predicted = max(float(prediction), 1.0)
        log_ratio = math.log(actual / predicted)
        for key in calibration_keys(row):
            grouped[key].append(log_ratio)

    calibrator: dict[tuple[object, ...], dict[str, float]] = {}
    for key, values in grouped.items():
        arr = np.asarray(values, dtype=np.float64)
        calibrator[key] = {
            "count": float(arr.size),
            "median_log_ratio": float(np.median(arr)),
        }
    return calibrator


def apply_bias_calibration(
    row: dict[str, float | int | str],
    prediction: float,
    calibrator: dict[tuple[object, ...], dict[str, float]] | None,
    *,
    min_count: int = 40,
    shrinkage: float = 80.0,
) -> float:
    if not calibrator:
        return float(prediction)
    base_prediction = max(float(prediction), 1.0)
    for key in calibration_keys(row):
        stats = calibrator.get(key)
        if stats is None:
            continue
        count = int(stats["count"])
        if key[0] != "global" and count < min_count:
            continue
        shrink = count / (count + shrinkage)
        factor = math.exp(float(stats["median_log_ratio"]) * shrink)
        return base_prediction * factor
    return base_prediction


def apply_bias_calibration_batch(
    rows: list[dict[str, float | int | str]],
    predictions: np.ndarray,
    calibrator: dict[tuple[object, ...], dict[str, float]] | None,
    *,
    min_count: int = 40,
    shrinkage: float = 80.0,
) -> np.ndarray:
    return np.asarray(
        [
            apply_bias_calibration(
                row,
                float(prediction),
                calibrator,
                min_count=min_count,
                shrinkage=shrinkage,
            )
            for row, prediction in zip(rows, predictions, strict=True)
        ],
        dtype=np.float64,
    )


def score_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    abs_err = np.abs(y_pred - y_true)
    denom = np.maximum(y_true, 1.0)
    rel_err = abs_err / denom
    smape = np.mean(2.0 * abs_err / np.maximum(np.abs(y_true) + np.abs(y_pred), 1.0))
    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(np.square(abs_err)))),
        "mape": float(np.mean(rel_err)),
        "smape": float(smape),
        "within_5pct": float(np.mean(rel_err <= 0.05)),
        "within_10pct": float(np.mean(rel_err <= 0.10)),
        "within_20pct": float(np.mean(rel_err <= 0.20)),
    }


def bucketed_metrics(val_rows: list[dict[str, float | int | str]], predictions: np.ndarray) -> dict[str, dict[str, float]]:
    groups = {
        "early_5_25": [],
        "mid_25_50": [],
        "mid_50_75": [],
        "late_75_95": [],
    }
    for idx, row in enumerate(val_rows):
        progress = float(row["progress"])
        if progress < 0.25:
            groups["early_5_25"].append(idx)
        elif progress < 0.50:
            groups["mid_25_50"].append(idx)
        elif progress < 0.75:
            groups["mid_50_75"].append(idx)
        else:
            groups["late_75_95"].append(idx)
    metrics = {}
    finals = np.asarray([float(row["final_score"]) for row in val_rows], dtype=np.float64)
    for name, indices in groups.items():
        if not indices:
            continue
        metrics[name] = score_metrics(finals[indices], predictions[indices])
        metrics[name]["samples"] = len(indices)
    return metrics


def region_metrics(val_rows: list[dict[str, float | int | str]], predictions: np.ndarray) -> dict[str, dict[str, float]]:
    indices_by_region: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(val_rows):
        indices_by_region[str(row["region"])].append(idx)
    finals = np.asarray([float(row["final_score"]) for row in val_rows], dtype=np.float64)
    result = {}
    for region, indices in indices_by_region.items():
        result[region] = score_metrics(finals[indices], predictions[indices])
        result[region]["samples"] = len(indices)
    return result


def rank_metrics(val_rows: list[dict[str, float | int | str]], predictions: np.ndarray) -> dict[str, dict[str, float]]:
    indices_by_rank: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(val_rows):
        indices_by_rank[str(int(row["rank"]))].append(idx)
    finals = np.asarray([float(row["final_score"]) for row in val_rows], dtype=np.float64)
    result = {}
    for rank, indices in sorted(indices_by_rank.items(), key=lambda item: int(item[0])):
        result[rank] = score_metrics(finals[indices], predictions[indices])
        result[rank]["samples"] = len(indices)
    return result


def boundary_rank_metrics(
    val_rows: list[dict[str, float | int | str]],
    predictions: np.ndarray,
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(val_rows):
        rank = int(row["rank"])
        if rank not in (100, 1000):
            continue
        progress = float(row["progress"])
        groups[f"rank_{rank}"].append(idx)
        if progress >= 0.75:
            groups[f"rank_{rank}_late"].append(idx)
        if rank_boundary_weekend_flag(row) > 0.0 and progress >= 0.65:
            groups[f"rank_{rank}_late_weekend"].append(idx)
    finals = np.asarray([float(row["final_score"]) for row in val_rows], dtype=np.float64)
    result = {}
    for name, indices in sorted(groups.items()):
        if not indices:
            continue
        result[name] = score_metrics(finals[indices], predictions[indices])
        result[name]["samples"] = len(indices)
    return result


def write_predictions(
    path: Path,
    val_rows: list[dict[str, float | int | str]],
    model_predictions: np.ndarray,
    rank_boundary_calibration: dict[object, object] | None = RANK_BOUNDARY_CALIBRATION,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "region",
            "event_id",
            "series_type",
            "event_condition",
            "character_id",
            "rank",
            "progress",
            "current_timestamp",
            "end_timestamp",
            "current_calendar_segment",
            "end_calendar_segment",
            "current_score",
            "final_score",
            "linear_projection",
            "pace_6h_projection",
            "jp_shape_projection",
            "hybrid_projection",
            "event_featured_character_count",
            "event_featured_unit_count",
            "line_character_heat",
            "wl2_character_prior_multiplier",
            "rank_boundary_multiplier",
            "model_prediction",
            "model_abs_error",
            "model_rel_error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, pred in zip(val_rows, model_predictions, strict=True):
            final_score = float(row["final_score"])
            abs_error = abs(float(pred) - final_score)
            rel_error = abs_error / max(final_score, 1.0)
            output = dict(row)
            output["rank_boundary_multiplier"] = round(
                rank_boundary_calibration_multiplier(row, rank_boundary_calibration),
                6,
            )
            output["model_prediction"] = round(float(pred), 3)
            output["model_abs_error"] = round(abs_error, 3)
            output["model_rel_error"] = round(rel_error, 6)
            writer.writerow(output)


def growth_targets(rows: list[dict]) -> np.ndarray:
    """log(final / current): the same target the offline quantile replay used."""
    final = np.asarray([float(row["final_score"]) for row in rows], dtype=np.float64)
    current = np.asarray([float(row["current_score"]) for row in rows], dtype=np.float64)
    return np.log(np.maximum(final, 1.0) / np.maximum(current, 1.0))


def train_quantile_models(
    x: np.ndarray,
    y: np.ndarray,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> dict[float, HistGradientBoostingRegressor]:
    """Quantile boosters on the log remaining-growth target.

    Offline replay (analysis/offline_intrinsic_live/experiment_structure.py) showed
    the median-loss model beating the squared-error model at 80-95% progress
    (-0.20pp causal, -0.32pp random folds, 85% of events improved), and the
    quantiles are what the interval output needs.
    """
    models: dict[float, HistGradientBoostingRegressor] = {}
    for quantile in quantiles:
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            learning_rate=0.05,
            max_depth=8,
            max_iter=400,
            min_samples_leaf=20,
            l2_regularization=0.05,
            random_state=42,
            early_stopping=False,
        )
        model.fit(x, y)
        models[float(quantile)] = model
    return models


def predict_quantile_scores(
    payload: dict[str, object],
    x: np.ndarray,
    rows: list[dict[str, float | int | str]],
    current_scores: np.ndarray,
) -> dict[float, np.ndarray] | None:
    """Quantile predictions of the final score, or None when the payload has none."""
    models = payload.get("quantile_models")
    if not isinstance(models, dict) or not models:
        return None
    raw: dict[float, np.ndarray] = {}
    for quantile, model in models.items():
        if not isinstance(model, HistGradientBoostingRegressor):
            return None
        raw[float(quantile)] = np.maximum(current_scores * np.exp(model.predict(x)), current_scores)
    levels = sorted(raw)
    stacked = np.stack([raw[level] for level in levels])
    stacked.sort(axis=0)  # independent boosters can cross; enforce p10 <= p50 <= p90
    return {level: stacked[index] for index, level in enumerate(levels)}


def train_forecast_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_rows: list[dict[str, float | int | str]],
    val_x: np.ndarray,
    val_y: np.ndarray,
    val_rows: list[dict[str, float | int | str]],
    *,
    model_series_types: frozenset[str],
    rank_boundary_calibration: dict[object, object] | None,
    include_cn_sidecar: bool,
) -> dict[str, object]:
    if train_x.size == 0 or val_x.size == 0:
        raise ValueError(f"not enough samples for {sorted(model_series_types)}")

    direct_model = make_boosting_regressor()
    residual_linear_model = make_boosting_regressor()
    residual_hybrid_model = make_boosting_regressor()
    cn_sidecar_model = make_cn_sidecar_regressor() if include_cn_sidecar else None
    linear_log_feature_index = LINEAR_LOG_FEATURE_INDEX
    hybrid_log_feature_index = HYBRID_LOG_FEATURE_INDEX
    pace_6h_log_feature_index = PACE_6H_PROJECTION_LOG_FEATURE_INDEX

    direct_model.fit(train_x, train_y)
    linear_log_train = train_x[:, linear_log_feature_index]
    hybrid_log_train = train_x[:, hybrid_log_feature_index]
    residual_linear_model.fit(train_x, train_y - linear_log_train)
    residual_hybrid_model.fit(train_x, train_y - hybrid_log_train)

    direct_predictions = predict_direct_scores(direct_model, val_x)
    residual_linear_predictions = predict_residual_scores(
        residual_linear_model,
        val_x,
        linear_log_feature_index,
    )
    residual_blend_bundle = {
        "residual_linear_model": residual_linear_model,
        "residual_hybrid_model": residual_hybrid_model,
    }
    residual_blend_predictions, residual_blend_linear_predictions, residual_blend_hybrid_predictions = predict_residual_blend_scores(
        residual_blend_bundle,
        val_x,
        val_rows,
        linear_log_feature_index,
        hybrid_log_feature_index,
    )
    residual_hybrid_predictions = residual_blend_hybrid_predictions

    uncalibrated_model_predictions = {
        "direct_model": direct_predictions.copy(),
        "residual_linear_model": residual_linear_predictions.copy(),
        "residual_hybrid_model": residual_hybrid_predictions.copy(),
        "residual_blend_model": residual_blend_predictions.copy(),
    }

    cn_tail_sidecar_predictions = residual_blend_predictions.copy()
    cn_tail_sidecar_mask = np.zeros(len(val_rows), dtype=bool)
    cn_sidecar_train_mask = np.zeros(len(train_rows), dtype=bool)
    train_linear_oof_predictions = build_oof_residual_predictions(
        train_x,
        train_y,
        train_rows,
        linear_log_feature_index,
    )
    train_hybrid_oof_predictions = build_oof_residual_predictions(
        train_x,
        train_y,
        train_rows,
        hybrid_log_feature_index,
    )
    train_blend_weights = np.asarray(
        [
            get_residual_blend_weight(str(row["region"]), float(row["progress"]))
            for row in train_rows
        ],
        dtype=np.float64,
    )
    train_residual_blend_oof_predictions = log_blend_predictions(
        train_linear_oof_predictions,
        train_hybrid_oof_predictions,
        train_blend_weights,
    )
    if include_cn_sidecar and cn_sidecar_model is not None:
        cn_sidecar_train_mask = np.asarray(
            [str(row["region"]) == CN_SIDECAR_REGION for row in train_rows],
            dtype=bool,
        )
        cn_tail_sidecar_mask = np.asarray(
            [should_apply_cn_tail_sidecar(row) for row in val_rows],
            dtype=bool,
        )
        if np.any(cn_sidecar_train_mask):
            cn_sidecar_train_x = build_cn_sidecar_features(
                train_x[cn_sidecar_train_mask],
                train_residual_blend_oof_predictions[cn_sidecar_train_mask],
                train_linear_oof_predictions[cn_sidecar_train_mask],
                train_hybrid_oof_predictions[cn_sidecar_train_mask],
            )
            cn_sidecar_model.fit(cn_sidecar_train_x, train_y[cn_sidecar_train_mask])
            cn_sidecar_val_x = build_cn_sidecar_features(
                val_x,
                residual_blend_predictions,
                residual_blend_linear_predictions,
                residual_blend_hybrid_predictions,
            )
            cn_sidecar_raw_predictions = predict_direct_scores(cn_sidecar_model, cn_sidecar_val_x)
            cn_tail_sidecar_predictions, cn_tail_sidecar_mask = apply_cn_tail_sidecar(
                residual_blend_predictions,
                cn_sidecar_raw_predictions,
                val_rows,
            )
            uncalibrated_model_predictions["cn_tail_sidecar_model"] = cn_tail_sidecar_predictions.copy()

    post_blend_calibrator = fit_bias_calibrator(train_rows, train_residual_blend_oof_predictions)
    regional_calibrated_blend_uncalibrated_predictions = apply_bias_calibration_batch(
        val_rows,
        residual_blend_predictions,
        post_blend_calibrator,
        min_count=24,
        shrinkage=60.0,
    )
    uncalibrated_model_predictions["regional_calibrated_blend_model"] = (
        regional_calibrated_blend_uncalibrated_predictions.copy()
    )

    true_scores = np.expm1(val_y)
    linear_predictions = np.asarray([float(row["linear_projection"]) for row in val_rows], dtype=np.float64)
    pace_6h_predictions = np.asarray([float(row["pace_6h_projection"]) for row in val_rows], dtype=np.float64)
    jp_shape_predictions = np.asarray([float(row["jp_shape_projection"]) for row in val_rows], dtype=np.float64)
    hybrid_predictions = np.asarray([float(row["hybrid_projection"]) for row in val_rows], dtype=np.float64)

    uncalibrated_model_metrics = {
        name: score_metrics(true_scores, predictions)
        for name, predictions in uncalibrated_model_predictions.items()
    }
    direct_predictions = apply_rank_boundary_calibration_batch(
        val_rows,
        direct_predictions,
        rank_boundary_calibration,
    )
    residual_linear_predictions = apply_rank_boundary_calibration_batch(
        val_rows,
        residual_linear_predictions,
        rank_boundary_calibration,
    )
    residual_hybrid_predictions = apply_rank_boundary_calibration_batch(
        val_rows,
        residual_hybrid_predictions,
        rank_boundary_calibration,
    )
    residual_blend_predictions = apply_rank_boundary_calibration_batch(
        val_rows,
        residual_blend_predictions,
        rank_boundary_calibration,
    )
    regional_calibrated_blend_predictions = apply_rank_boundary_calibration_batch(
        val_rows,
        regional_calibrated_blend_uncalibrated_predictions,
        rank_boundary_calibration,
    )
    if "cn_tail_sidecar_model" in uncalibrated_model_predictions:
        cn_tail_sidecar_predictions = apply_rank_boundary_calibration_batch(
            val_rows,
            cn_tail_sidecar_predictions,
            rank_boundary_calibration,
        )

    direct_metrics = score_metrics(true_scores, direct_predictions)
    residual_metrics = score_metrics(true_scores, residual_linear_predictions)
    residual_hybrid_metrics = score_metrics(true_scores, residual_hybrid_predictions)
    residual_blend_metrics = score_metrics(true_scores, residual_blend_predictions)
    regional_calibrated_blend_metrics = score_metrics(true_scores, regional_calibrated_blend_predictions)
    train_pace_6h_predictions = np.expm1(train_x[:, pace_6h_log_feature_index])
    pace_6h_bias_calibrator = fit_bias_calibrator(train_rows, train_pace_6h_predictions)
    pace_6h_bias_uncalibrated_predictions = apply_bias_calibration_batch(
        val_rows,
        pace_6h_predictions,
        pace_6h_bias_calibrator,
        min_count=8 if "world_bloom" in model_series_types else 40,
        shrinkage=30.0 if "world_bloom" in model_series_types else 80.0,
    )
    uncalibrated_model_predictions["pace_6h_bias_calibrated_model"] = pace_6h_bias_uncalibrated_predictions.copy()
    pace_6h_bias_predictions = apply_rank_boundary_calibration_batch(
        val_rows,
        pace_6h_bias_uncalibrated_predictions,
        rank_boundary_calibration,
    )
    pace_6h_bias_metrics = score_metrics(true_scores, pace_6h_bias_predictions)
    candidates = [
        {
            "name": "direct_model",
            "predictions": direct_predictions,
            "metrics": direct_metrics,
            "payload": {"model": direct_model},
        },
        {
            "name": "residual_linear_model",
            "predictions": residual_linear_predictions,
            "metrics": residual_metrics,
            "payload": {"model": residual_linear_model},
        },
        {
            "name": "residual_hybrid_model",
            "predictions": residual_hybrid_predictions,
            "metrics": residual_hybrid_metrics,
            "payload": {"model": residual_hybrid_model},
        },
        {
            "name": "pace_6h_bias_calibrated_model",
            "predictions": pace_6h_bias_predictions,
            "metrics": pace_6h_bias_metrics,
            "payload": {
                "base_log_feature_index": pace_6h_log_feature_index,
                "bias_calibrator": pace_6h_bias_calibrator,
                "bias_calibration_min_count": 8 if "world_bloom" in model_series_types else 40,
                "bias_calibration_shrinkage": 30.0 if "world_bloom" in model_series_types else 80.0,
            },
        },
        {
            "name": "residual_blend_model",
            "predictions": residual_blend_predictions,
            "metrics": residual_blend_metrics,
            "payload": {
                "model_bundle": {
                    "residual_linear_model": residual_linear_model,
                    "residual_hybrid_model": residual_hybrid_model,
                },
                "residual_blend_weights": {
                    "region_defaults": REGION_BLEND_WEIGHTS,
                    "cn_progress_overrides": CN_PROGRESS_BLEND_WEIGHTS,
                },
            },
        },
        {
            "name": "regional_calibrated_blend_model",
            "predictions": regional_calibrated_blend_predictions,
            "metrics": regional_calibrated_blend_metrics,
            "payload": {
                "model_bundle": {
                    "residual_linear_model": residual_linear_model,
                    "residual_hybrid_model": residual_hybrid_model,
                },
                "residual_blend_weights": {
                    "region_defaults": REGION_BLEND_WEIGHTS,
                    "cn_progress_overrides": CN_PROGRESS_BLEND_WEIGHTS,
                },
                "post_blend_calibrator": post_blend_calibrator,
                "post_blend_calibration_min_count": 24,
                "post_blend_calibration_shrinkage": 60.0,
            },
        },
    ]
    if "cn_tail_sidecar_model" in uncalibrated_model_predictions and cn_sidecar_model is not None:
        cn_tail_sidecar_metrics = score_metrics(true_scores, cn_tail_sidecar_predictions)
        candidates.append(
            {
                "name": "cn_tail_sidecar_model",
                "predictions": cn_tail_sidecar_predictions,
                "metrics": cn_tail_sidecar_metrics,
                "payload": {
                    "model_bundle": {
                        "residual_linear_model": residual_linear_model,
                        "residual_hybrid_model": residual_hybrid_model,
                        "cn_sidecar_model": cn_sidecar_model,
                    },
                    "residual_blend_weights": {
                        "region_defaults": REGION_BLEND_WEIGHTS,
                        "region_progress": {
                            region: [
                                {"progress_lt": threshold, "weight": weight}
                                for threshold, weight in weights
                            ]
                            for region, weights in REGION_PROGRESS_BLEND_WEIGHTS.items()
                        },
                        "cn_progress_overrides": CN_PROGRESS_BLEND_WEIGHTS,
                    },
                    "cn_sidecar": {
                        "region": CN_SIDECAR_REGION,
                        "apply_rank_gt": CN_SIDECAR_APPLY_MIN_RANK,
                        "base_weight": CN_SIDECAR_BASE_WEIGHT,
                        "apply_validation_samples": int(np.sum(cn_tail_sidecar_mask)),
                        "train_region_samples": int(np.sum(cn_sidecar_train_mask)),
                    },
                },
            }
        )

    def full_candidate_payload(candidate: dict[str, object]) -> dict[str, object]:
        return {
            "model_name": str(candidate["name"]),
            "regions": REGIONS,
            "selected_ranks": COMMON_RANKS,
            "progress_buckets": BUCKETS,
            "feature_count": int(train_x.shape[1]),
            "model_series_types": sorted(model_series_types),
            "linear_log_feature_index": linear_log_feature_index,
            "hybrid_log_feature_index": hybrid_log_feature_index,
            "pace_6h_log_feature_index": pace_6h_log_feature_index,
            "rank_boundary_calibration": rank_boundary_calibration,
            "event_heat_prior_enabled": EVENT_HEAT_PRIOR_ENABLED,
            **candidate["payload"],
        }

    fallback_candidate = min(candidates, key=lambda item: item["metrics"]["mape"])
    region_model_names: dict[str, str] = {}
    region_route_metrics: dict[str, dict[str, object]] = {}
    routed_predictions = np.zeros(len(val_rows), dtype=np.float64)
    for region in REGIONS:
        region_indices = [idx for idx, row in enumerate(val_rows) if str(row["region"]) == region]
        if not region_indices:
            continue
        best_candidate = min(
            candidates,
            key=lambda candidate: score_metrics(
                true_scores[region_indices],
                np.asarray(candidate["predictions"], dtype=np.float64)[region_indices],
            )["mape"],
        )
        best_predictions = np.asarray(best_candidate["predictions"], dtype=np.float64)
        routed_predictions[region_indices] = best_predictions[region_indices]
        region_model_names[region] = str(best_candidate["name"])
        region_route_metrics[region] = {
            "selected_model": str(best_candidate["name"]),
            "metrics": score_metrics(true_scores[region_indices], best_predictions[region_indices]),
            "samples": len(region_indices),
        }

    candidates.append({
        "name": "region_routed_model",
        "predictions": routed_predictions,
        "metrics": score_metrics(true_scores, routed_predictions),
        "payload": {
            "fallback_model_name": str(fallback_candidate["name"]),
            "region_model_names": region_model_names,
            "region_route_metrics": region_route_metrics,
            "candidate_payloads": {
                str(candidate["name"]): full_candidate_payload(candidate)
                for candidate in candidates
            },
        },
    })
    uncalibrated_model_predictions["region_routed_model"] = routed_predictions.copy()

    chosen = min(candidates, key=lambda item: item["metrics"]["mape"])
    chosen_name = str(chosen["name"])
    chosen_predictions = np.asarray(chosen["predictions"], dtype=np.float64)
    chosen_payload = full_candidate_payload(chosen)
    metrics = {
        "selected_model": chosen_name,
        "model_series_types": sorted(model_series_types),
        "train_samples": int(train_x.shape[0]),
        "validation_samples": int(val_x.shape[0]),
        "feature_count": int(train_x.shape[1]),
        "direct_model": direct_metrics,
        "residual_linear_model": residual_metrics,
        "pace_6h_bias_calibrated_model": pace_6h_bias_metrics,
        "residual_hybrid_model": residual_hybrid_metrics,
        "residual_blend_model": residual_blend_metrics,
        "regional_calibrated_blend_model": regional_calibrated_blend_metrics,
        "region_routed_model": score_metrics(true_scores, routed_predictions),
        "region_model_names": region_model_names,
        "region_route_metrics": region_route_metrics,
        "uncalibrated_model_metrics": uncalibrated_model_metrics,
        "rank_boundary_calibration": rank_boundary_calibration,
        "linear_baseline": score_metrics(true_scores, linear_predictions),
        "pace_6h_baseline": score_metrics(true_scores, pace_6h_predictions),
        "jp_shape_baseline": score_metrics(true_scores, jp_shape_predictions),
        "hybrid_baseline": score_metrics(true_scores, hybrid_predictions),
        "by_progress_bucket": bucketed_metrics(val_rows, chosen_predictions),
        "by_region": region_metrics(val_rows, chosen_predictions),
        "by_rank": rank_metrics(val_rows, chosen_predictions),
        "boundary_ranks": boundary_rank_metrics(val_rows, chosen_predictions),
        "boundary_ranks_uncalibrated": boundary_rank_metrics(
            val_rows,
            uncalibrated_model_predictions[chosen_name],
        ),
    }
    if "cn_tail_sidecar_model" in uncalibrated_model_predictions:
        metrics["cn_tail_sidecar_model"] = score_metrics(true_scores, cn_tail_sidecar_predictions)
        metrics["cn_sidecar"] = {
            "region": CN_SIDECAR_REGION,
            "apply_rank_gt": CN_SIDECAR_APPLY_MIN_RANK,
            "base_weight": CN_SIDECAR_BASE_WEIGHT,
            "train_region_samples": int(np.sum(cn_sidecar_train_mask)),
            "validation_apply_samples": int(np.sum(cn_tail_sidecar_mask)),
        }
    return {
        "payload": chosen_payload,
        "metrics": metrics,
        "predictions": chosen_predictions,
        "uncalibrated_model_predictions": uncalibrated_model_predictions,
    }


def train_pace_bias_forecast_model(
    train_x: np.ndarray,
    train_rows: list[dict[str, float | int | str]],
    val_x: np.ndarray,
    val_y: np.ndarray,
    val_rows: list[dict[str, float | int | str]],
    *,
    model_series_types: frozenset[str],
) -> dict[str, object]:
    if train_x.size == 0 or val_x.size == 0:
        raise ValueError(f"not enough samples for {sorted(model_series_types)}")
    pace_6h_log_feature_index = PACE_6H_PROJECTION_LOG_FEATURE_INDEX
    train_pace_6h_predictions = np.expm1(train_x[:, pace_6h_log_feature_index])
    pace_6h_predictions = np.asarray([float(row["pace_6h_projection"]) for row in val_rows], dtype=np.float64)
    calibrator = fit_bias_calibrator(train_rows, train_pace_6h_predictions)
    predictions = apply_bias_calibration_batch(
        val_rows,
        pace_6h_predictions,
        calibrator,
        min_count=80,
        shrinkage=400.0,
    )
    bias_only_predictions = predictions.copy()
    prediction_payload = {
        "wl2_character_prior_enabled": WORLD_BLOOM_WL2_CHARACTER_PRIOR_ENABLED,
        "wl2_character_prior": CHARACTER_POPULARITY_PRIOR,
        "wl2_character_prior_region_weights": WORLD_BLOOM_WL2_CHARACTER_PRIOR_REGION_WEIGHTS,
        "wl2_character_prior_progress_floor": WORLD_BLOOM_WL2_CHARACTER_PRIOR_PROGRESS_FLOOR,
        "wl2_character_prior_progress_decay": WORLD_BLOOM_WL2_CHARACTER_PRIOR_PROGRESS_DECAY,
    }
    predictions = apply_wl2_character_prior_batch(val_rows, predictions, prediction_payload)
    true_scores = np.expm1(val_y)
    linear_predictions = np.asarray([float(row["linear_projection"]) for row in val_rows], dtype=np.float64)
    jp_shape_predictions = np.asarray([float(row["jp_shape_projection"]) for row in val_rows], dtype=np.float64)
    hybrid_predictions = np.asarray([float(row["hybrid_projection"]) for row in val_rows], dtype=np.float64)
    metrics = {
        "selected_model": "pace_6h_bias_calibrated_model",
        "model_series_types": sorted(model_series_types),
        "train_samples": int(train_x.shape[0]),
        "validation_samples": int(val_x.shape[0]),
        "feature_count": int(train_x.shape[1]),
        "pace_6h_bias_calibrated_model": score_metrics(true_scores, predictions),
        "pace_6h_bias_without_wl2_prior": score_metrics(true_scores, bias_only_predictions),
        "linear_baseline": score_metrics(true_scores, linear_predictions),
        "pace_6h_baseline": score_metrics(true_scores, pace_6h_predictions),
        "jp_shape_baseline": score_metrics(true_scores, jp_shape_predictions),
        "hybrid_baseline": score_metrics(true_scores, hybrid_predictions),
        "by_progress_bucket": bucketed_metrics(val_rows, predictions),
        "by_region": region_metrics(val_rows, predictions),
        "by_rank": rank_metrics(val_rows, predictions),
        "boundary_ranks": boundary_rank_metrics(val_rows, predictions),
        "boundary_ranks_uncalibrated": boundary_rank_metrics(val_rows, predictions),
    }
    payload = {
        "model_name": "pace_6h_bias_calibrated_model",
        "regions": REGIONS,
        "selected_ranks": COMMON_RANKS,
        "progress_buckets": BUCKETS,
        "feature_count": int(train_x.shape[1]),
        "model_series_types": sorted(model_series_types),
        "linear_log_feature_index": LINEAR_LOG_FEATURE_INDEX,
        "hybrid_log_feature_index": HYBRID_LOG_FEATURE_INDEX,
        "pace_6h_log_feature_index": pace_6h_log_feature_index,
        "rank_boundary_calibration": None,
        "base_log_feature_index": pace_6h_log_feature_index,
        "bias_calibrator": calibrator,
        "bias_calibration_min_count": 80,
        "bias_calibration_shrinkage": 400.0,
        "wl2_character_prior_enabled": WORLD_BLOOM_WL2_CHARACTER_PRIOR_ENABLED,
        "wl2_event_ids": sorted(WORLD_BLOOM_WL2_EVENT_IDS),
        "wl3_event_ids": sorted(WORLD_BLOOM_WL3_EVENT_IDS),
        "wl2_character_prior_source": "jp_wl2_t100_manual",
        "wl2_character_prior": CHARACTER_POPULARITY_PRIOR,
        "wl2_character_prior_region_weights": WORLD_BLOOM_WL2_CHARACTER_PRIOR_REGION_WEIGHTS,
        "wl2_character_prior_rank_weights": WORLD_BLOOM_WL2_CHARACTER_PRIOR_RANK_WEIGHTS,
        "wl2_character_prior_progress_floor": WORLD_BLOOM_WL2_CHARACTER_PRIOR_PROGRESS_FLOOR,
        "wl2_character_prior_progress_decay": WORLD_BLOOM_WL2_CHARACTER_PRIOR_PROGRESS_DECAY,
    }
    return {
        "payload": payload,
        "metrics": metrics,
        "predictions": predictions,
        "uncalibrated_model_predictions": {
            "pace_6h_bias_calibrated_model": predictions.copy(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a baseline event result forecaster.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/deseer/PycharmProjects/Haruki-Tracker-Exports"),
        help="Path to the normalized tracker export project.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
        help="Where to write model outputs.",
    )
    parser.add_argument(
        "--holdout-count",
        type=int,
        default=2,
        help="How many latest events per region to keep for validation.",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    finished_event_ids = load_finished_event_ids()
    event_meta = load_event_meta(data_root, finished_event_ids)
    selected_ranks = set(COMMON_RANKS)
    normalized_event_keys = load_normalized_rank_timeseries_event_keys(data_root)
    legacy_event_files = load_legacy_event_files(data_root, normalized_event_keys)
    normal_available_event_ids = collect_available_training_event_ids(
        normalized_event_keys,
        legacy_event_files,
        NORMAL_MODEL_SERIES_TYPES,
    )
    wl_available_event_ids = collect_available_training_event_ids(
        normalized_event_keys,
        legacy_event_files,
        WL_MODEL_SERIES_TYPES,
    )
    normal_validation_events = select_validation_events(
        event_meta,
        args.holdout_count,
        NORMAL_MODEL_SERIES_TYPES,
        available_event_ids=normal_available_event_ids,
    )
    wl_validation_events = select_validation_events(
        event_meta,
        args.holdout_count,
        WL_MODEL_SERIES_TYPES,
        available_event_ids=wl_available_event_ids,
    )
    finals = build_final_targets(data_root, event_meta, selected_ranks, legacy_event_files)
    event_feature_contexts = load_event_feature_contexts()
    normal_character_heat_priors = build_character_heat_priors(
        event_meta=event_meta,
        finals=finals,
        validation_events=normal_validation_events,
        event_feature_contexts=event_feature_contexts,
        model_series_types=NORMAL_MODEL_SERIES_TYPES,
    )
    write_heat_prior_tables(results_dir, normal_character_heat_priors)
    live_validation_events = {region: set() for region in REGIONS}
    live_normal_character_heat_priors = build_character_heat_priors(
        event_meta=event_meta,
        finals=finals,
        validation_events=live_validation_events,
        event_feature_contexts=event_feature_contexts,
        model_series_types=NORMAL_MODEL_SERIES_TYPES,
    )
    live_normal_event_heat_priors = build_event_heat_final_priors(
        event_meta=event_meta,
        finals=finals,
        validation_events=live_validation_events,
        event_feature_contexts=event_feature_contexts,
        character_heat_priors=live_normal_character_heat_priors,
        model_series_types=NORMAL_MODEL_SERIES_TYPES,
    )
    write_event_heat_prior_tables(results_dir, live_normal_event_heat_priors)
    wl_character_heat_priors = build_character_heat_priors(
        event_meta=event_meta,
        finals=finals,
        validation_events=wl_validation_events,
        event_feature_contexts=event_feature_contexts,
        model_series_types=WL_MODEL_SERIES_TYPES,
    )
    wl_results_dir = results_dir / "world_bloom"
    wl_results_dir.mkdir(parents=True, exist_ok=True)
    write_heat_prior_tables(wl_results_dir, wl_character_heat_priors)
    normal_boundary_pressure_contexts = build_boundary_pressure_contexts(
        data_root=data_root,
        event_meta=event_meta,
        selected_ranks=selected_ranks,
        model_series_types=NORMAL_MODEL_SERIES_TYPES,
    )
    wl_boundary_pressure_contexts = build_boundary_pressure_contexts(
        data_root=data_root,
        event_meta=event_meta,
        selected_ranks=selected_ranks,
        model_series_types=WL_MODEL_SERIES_TYPES,
    )
    normal_jp_shape_priors = build_jp_shape_priors(
        data_root=data_root,
        event_meta=event_meta,
        finals=finals,
        validation_events=normal_validation_events,
        selected_ranks=selected_ranks,
        legacy_event_files=legacy_event_files,
        model_series_types=NORMAL_MODEL_SERIES_TYPES,
    )
    wl_jp_shape_priors = build_jp_shape_priors(
        data_root=data_root,
        event_meta=event_meta,
        finals=finals,
        validation_events=wl_validation_events,
        selected_ranks=selected_ranks,
        legacy_event_files=legacy_event_files,
        model_series_types=WL_MODEL_SERIES_TYPES,
    )
    train_x, train_y, train_rows, val_x, val_y, val_rows = build_dataset(
        data_root=data_root,
        event_meta=event_meta,
        finals=finals,
        validation_events=normal_validation_events,
        selected_ranks=selected_ranks,
        event_feature_contexts=event_feature_contexts,
        character_heat_priors=normal_character_heat_priors,
        jp_shape_priors=normal_jp_shape_priors,
        boundary_pressure_contexts=normal_boundary_pressure_contexts,
        legacy_event_files=legacy_event_files,
        model_series_types=NORMAL_MODEL_SERIES_TYPES,
    )
    wl_train_x, wl_train_y, wl_train_rows, wl_val_x, wl_val_y, wl_val_rows = build_dataset(
        data_root=data_root,
        event_meta=event_meta,
        finals=finals,
        validation_events=wl_validation_events,
        selected_ranks=selected_ranks,
        event_feature_contexts=event_feature_contexts,
        character_heat_priors=wl_character_heat_priors,
        jp_shape_priors=wl_jp_shape_priors,
        boundary_pressure_contexts=wl_boundary_pressure_contexts,
        legacy_event_files=legacy_event_files,
        model_series_types=WL_MODEL_SERIES_TYPES,
    )

    np.savez_compressed(
        results_dir / "prepared_dataset.npz",
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
    )
    np.savez_compressed(
        results_dir / "prepared_dataset_world_bloom.npz",
        train_x=wl_train_x,
        train_y=wl_train_y,
        val_x=wl_val_x,
        val_y=wl_val_y,
    )

    normal_result = train_forecast_model(
        train_x,
        train_y,
        train_rows,
        val_x,
        val_y,
        val_rows,
        model_series_types=NORMAL_MODEL_SERIES_TYPES,
        rank_boundary_calibration=RANK_BOUNDARY_CALIBRATION,
        include_cn_sidecar=True,
    )
    wl_result = train_pace_bias_forecast_model(
        wl_train_x,
        wl_train_rows,
        wl_val_x,
        wl_val_y,
        wl_val_rows,
        model_series_types=WL_MODEL_SERIES_TYPES,
    )
    normal_metrics = normal_result["metrics"]
    wl_metrics = wl_result["metrics"]
    chosen_name = str(normal_metrics["selected_model"])
    chosen_predictions = np.asarray(normal_result["predictions"], dtype=np.float64)

    metrics = {
        "dataset": {
            "train_samples": int(train_x.shape[0]),
            "validation_samples": int(val_x.shape[0]),
            "world_bloom_train_samples": int(wl_train_x.shape[0]),
            "world_bloom_validation_samples": int(wl_val_x.shape[0]),
            "feature_count": int(train_x.shape[1]),
            "world_bloom_feature_count": int(wl_train_x.shape[1]),
            "model_series_types": sorted(NORMAL_MODEL_SERIES_TYPES),
            "world_bloom_model_series_types": sorted(WL_MODEL_SERIES_TYPES),
            "character_heat_priors_by_region": int(len(normal_character_heat_priors.get("character_by_region", {}))),
            "character_heat_priors_by_region_condition": int(len(normal_character_heat_priors.get("character_by_region_condition", {}))),
            "main_character_heat_priors_by_region": int(len(normal_character_heat_priors.get("main_character_by_region", {}))),
            "main_character_heat_priors_by_region_condition": int(len(normal_character_heat_priors.get("main_character_by_region_condition", {}))),
            "unit_heat_priors_by_region": int(len(normal_character_heat_priors.get("unit_by_region", {}))),
            "unit_heat_priors_by_region_condition": int(len(normal_character_heat_priors.get("unit_by_region_condition", {}))),
            "heat_prior_diagnostics": normal_character_heat_priors.get("diagnostics", {}),
            "event_heat_prior_condition_groups": int(len(live_normal_event_heat_priors.get("condition_histories", {}))),
            "event_heat_prior_region_groups": int(len(live_normal_event_heat_priors.get("region_histories", {}))),
            "event_heat_prior_global_groups": int(len(live_normal_event_heat_priors.get("global_histories", {}))),
            "world_bloom_character_heat_priors_by_region": int(len(wl_character_heat_priors.get("character_by_region", {}))),
            "world_bloom_character_heat_priors_by_region_condition": int(len(wl_character_heat_priors.get("character_by_region_condition", {}))),
            "world_bloom_heat_prior_diagnostics": wl_character_heat_priors.get("diagnostics", {}),
            "jp_shape_series_curves": int(len(normal_jp_shape_priors.get("series", {}))),
            "jp_shape_global_curves": int(len(normal_jp_shape_priors.get("global", {}))),
            "jp_shape_calendar_curves": int(len(normal_jp_shape_priors.get("calendar", {}))),
            "jp_shape_series_calendar_curves": int(len(normal_jp_shape_priors.get("series_calendar", {}))),
            "world_bloom_jp_shape_series_curves": int(len(wl_jp_shape_priors.get("series", {}))),
            "world_bloom_jp_shape_global_curves": int(len(wl_jp_shape_priors.get("global", {}))),
            "world_bloom_jp_shape_calendar_curves": int(len(wl_jp_shape_priors.get("calendar", {}))),
            "world_bloom_jp_shape_series_calendar_curves": int(len(wl_jp_shape_priors.get("series_calendar", {}))),
            "live_history_line_counts": {r: len(live_index(sys.modules[__name__], r)) for r in REGIONS},
            "finished_event_counts": {region: len(event_ids) for region, event_ids in finished_event_ids.items()},
            "legacy_event_counts": {region: len(legacy_event_files.get(region, [])) for region in REGIONS},
            "holdout_events": {region: sorted(event_ids) for region, event_ids in normal_validation_events.items()},
            "world_bloom_holdout_events": {region: sorted(event_ids) for region, event_ids in wl_validation_events.items()},
            "selected_ranks": list(COMMON_RANKS),
            "progress_buckets": list(BUCKETS),
        },
        "selected_model": chosen_name,
        "normal_model": normal_metrics,
        "world_bloom_model": wl_metrics,
        "direct_model": normal_metrics["direct_model"],
        "residual_linear_model": normal_metrics["residual_linear_model"],
        "pace_6h_bias_calibrated_model": normal_metrics["pace_6h_bias_calibrated_model"],
        "residual_hybrid_model": normal_metrics["residual_hybrid_model"],
        "residual_blend_model": normal_metrics["residual_blend_model"],
        "regional_calibrated_blend_model": normal_metrics["regional_calibrated_blend_model"],
        "region_routed_model": normal_metrics["region_routed_model"],
        "region_model_names": normal_metrics["region_model_names"],
        "region_route_metrics": normal_metrics["region_route_metrics"],
        "cn_tail_sidecar_model": normal_metrics.get("cn_tail_sidecar_model"),
        "uncalibrated_model_metrics": normal_metrics["uncalibrated_model_metrics"],
        "rank_boundary_calibration": RANK_BOUNDARY_CALIBRATION,
        "residual_blend_weights": {
            "region_defaults": REGION_BLEND_WEIGHTS,
            "region_progress": {
                region: [
                    {"progress_lt": threshold, "weight": weight}
                    for threshold, weight in weights
                ]
                for region, weights in REGION_PROGRESS_BLEND_WEIGHTS.items()
            },
            "cn_progress_overrides": [
                {"progress_lt": threshold, "weight": weight}
                for threshold, weight in CN_PROGRESS_BLEND_WEIGHTS
            ],
        },
        "cn_sidecar": normal_metrics.get("cn_sidecar"),
        "linear_baseline": normal_metrics["linear_baseline"],
        "pace_6h_baseline": normal_metrics["pace_6h_baseline"],
        "jp_shape_baseline": normal_metrics["jp_shape_baseline"],
        "hybrid_baseline": normal_metrics["hybrid_baseline"],
        "by_progress_bucket": normal_metrics["by_progress_bucket"],
        "by_region": normal_metrics["by_region"],
        "by_rank": normal_metrics["by_rank"],
        "boundary_ranks": normal_metrics["boundary_ranks"],
        "boundary_ranks_uncalibrated": normal_metrics["boundary_ranks_uncalibrated"],
    }

    metrics_path = results_dir / "latest_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_predictions(
        results_dir / "latest_predictions.csv",
        val_rows,
        chosen_predictions,
        RANK_BOUNDARY_CALIBRATION,
    )
    write_predictions(
        results_dir / "world_bloom_predictions.csv",
        wl_val_rows,
        np.asarray(wl_result["predictions"], dtype=np.float64),
        None,
    )

    from opening_prior import build_history as build_opening_history
    opening_history = build_opening_history(sys.modules[__name__], event_meta, finals)
    (results_dir / "opening_final_history.json").write_text(json.dumps(opening_history))

    from regional_shape import build_history as build_regional_shape_history, build_export_history, merge_histories
    regional_shape_history = merge_histories(
        build_export_history(sys.modules[__name__], event_meta, data_root),
        build_regional_shape_history(sys.modules[__name__], event_meta))
    (results_dir / "regional_shape_history.json").write_text(json.dumps(regional_shape_history))

    # atomic replace: the live service reads this file on every prediction
    model_tmp = results_dir / "baseline_model.pkl.tmp"
    with open(model_tmp, "wb") as handle:
        normal_payload = dict(normal_result["payload"])
        wl_payload = dict(wl_result["payload"])
        serialized_event_feature_contexts = serialize_event_feature_contexts(event_feature_contexts)
        normal_payload["live_character_heat_priors"] = live_normal_character_heat_priors
        normal_payload["live_event_heat_priors"] = live_normal_event_heat_priors
        normal_payload["live_event_feature_contexts"] = serialized_event_feature_contexts
        normal_payload["live_jp_shape_priors"] = serialize_shape_priors(normal_jp_shape_priors)
        wl_payload["live_character_heat_priors"] = wl_character_heat_priors
        wl_payload["live_event_feature_contexts"] = serialized_event_feature_contexts
        wl_payload["live_jp_shape_priors"] = serialize_shape_priors(wl_jp_shape_priors)
        normal_payload["quantile_models"] = train_quantile_models(
            train_x, growth_targets(train_rows))
        wl_payload["quantile_models"] = train_quantile_models(
            wl_train_x, growth_targets(wl_train_rows))
        payload = dict(normal_payload)
        payload["opening_final_history"] = opening_history
        payload["opening_prior_regions"] = ["cn"]
        payload["median_heat_settings"] = {"enabled": False, "weights": [0., 0., 0., 0.], "unit_match": False, "duration_power": 0.}
        payload["regional_shape_history"] = regional_shape_history
        payload["regional_shape_weights"] = {"cn": [0.25, 0.5, 0.0, 0.25, 0.75]}
        payload["series_payloads"] = {
            "normal": normal_payload,
            "world_bloom": wl_payload,
        }
        pickle.dump(
            payload,
            handle,
        )
    os.replace(model_tmp, results_dir / "baseline_model.pkl")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
