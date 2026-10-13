#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageDraw, ImageFilter, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
FORECAST_OPS = REPO_ROOT / "scripts" / "forecast_ops.py"
DEFAULT_CONFIG = REPO_ROOT / "config" / "forecast.production.json"

REGIONS = ("jp", "en", "tw", "kr", "cn")
REGION_TITLES = {
    "jp": "JP",
    "en": "EN",
    "tw": "TW",
    "kr": "KR",
    "cn": "CN",
}
DISPLAY_RANKS = (
    10,
    20,
    30,
    40,
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
    40000,
    50000,
    100000,
    200000,
)
LOCAL_TZ = ZoneInfo("Asia/Shanghai")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def localized_path(path: str | Path) -> Path:
    raw = Path(path)
    if raw.exists():
        return raw
    text = str(raw)
    remote_project = "/Users/deseer/project/Haruki-Tracker-Forecast"
    if text.startswith(remote_project):
        local = REPO_ROOT / Path(text.removeprefix(remote_project).lstrip("/"))
        if local.exists():
            return local
    remote_exports = "/Users/deseer/project/Haruki-Tracker-Exports"
    if text.startswith(remote_exports):
        local = Path("/Users/deseer/PycharmProjects/Haruki-Tracker-Exports") / Path(
            text.removeprefix(remote_exports).lstrip("/")
        )
        if local.exists():
            return local
    remote_master = "/Users/deseer/project/masterdata"
    if text.startswith(remote_master):
        parts = Path(text.removeprefix(remote_master).lstrip("/")).parts
        if parts:
            region = parts[0]
            local_roots = {
                "jp": "/Users/deseer/PycharmProjects/haruki-sekai-master",
                "en": "/Users/deseer/PycharmProjects/haruki-sekai-en-master",
                "tw": "/Users/deseer/PycharmProjects/haruki-sekai-tc-master",
                "kr": "/Users/deseer/PycharmProjects/haruki-sekai-kr-master",
                "cn": "/Users/deseer/PycharmProjects/haruki-sekai-cn-master",
            }
            root = local_roots.get(region)
            if root:
                suffix = Path(*parts[1:]) if len(parts) > 1 else Path()
                local = Path(root) / suffix
                if local.exists():
                    return local
    return raw


def prediction_value(row: dict[str, Any]) -> float | None:
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


def current_timestamp(row: dict[str, Any]) -> int | None:
    for key in ("current_timestamp", "timestamp"):
        value = row.get(key)
        if value is not None:
            return int(value)
    return None


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def interpolate_between(left: float | None, right: float | None, weight: float) -> int | None:
    if left is None or right is None or left <= 0 or right <= 0:
        return None
    return int(round(math.exp(math.log(left) * (1.0 - weight) + math.log(right) * weight)))


def interpolate_rank(rows_by_rank: dict[int, dict[str, Any]], rank: int) -> dict[str, Any] | None:
    lower = [candidate for candidate in rows_by_rank if candidate < rank]
    upper = [candidate for candidate in rows_by_rank if candidate > rank]
    if not lower or not upper:
        return None
    left_rank = max(lower)
    right_rank = min(upper)
    if left_rank <= 0 or right_rank <= left_rank:
        return None
    weight = (math.log(rank) - math.log(left_rank)) / (math.log(right_rank) - math.log(left_rank))
    left = rows_by_rank[left_rank]
    right = rows_by_rank[right_rank]
    current_score = interpolate_between(
        float(left["current_score"]) if left.get("current_score") is not None else None,
        float(right["current_score"]) if right.get("current_score") is not None else None,
        weight,
    )
    prediction = interpolate_between(prediction_value(left), prediction_value(right), weight)
    timestamp_left = current_timestamp(left)
    timestamp_right = current_timestamp(right)
    return {
        "rank": rank,
        "character_id": as_int(left.get("character_id")),
        "synthetic_rank": True,
        "progress": left.get("progress"),
        "current_timestamp": max(timestamp for timestamp in (timestamp_left, timestamp_right) if timestamp is not None)
        if timestamp_left is not None or timestamp_right is not None
        else None,
        "end_timestamp": left.get("end_timestamp") or right.get("end_timestamp"),
        "current_score": current_score,
        "model_prediction": prediction,
    }


def normalize_line_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_rank = {as_int(row.get("rank")): row for row in rows if row.get("rank") is not None}
    normalized: list[dict[str, Any]] = []
    for rank in DISPLAY_RANKS:
        source = rows_by_rank.get(rank)
        synthetic = False
        if source is None:
            source = interpolate_rank(rows_by_rank, rank)
            synthetic = True
        if source is None:
            normalized.append({
                "rank": rank,
                "current_score": None,
                "prediction": None,
                "synthetic_rank": True,
            })
            continue
        normalized.append({
            "rank": rank,
            "current_score": as_int(source.get("current_score")) if source.get("current_score") is not None else None,
            "prediction": int(round(prediction_value(source))) if prediction_value(source) is not None else None,
            "synthetic_rank": bool(source.get("synthetic_rank", synthetic)),
            "progress": source.get("progress"),
            "current_timestamp": current_timestamp(source),
            "end_timestamp": source.get("end_timestamp"),
        })
    return normalized


def seconds_to_text(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    if seconds < 60:
        return f"{seconds}秒前"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}分钟前"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}小时前"
    return f"{hours // 24}天前"


def remaining_to_text(seconds: float) -> str:
    seconds = int(seconds)
    if seconds <= 0:
        return "已结束"
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    if days > 0:
        return f"距离活动结束还有{days}天{hours}小时{minutes}分钟"
    if hours > 0:
        return f"距离活动结束还有{hours}小时{minutes}分钟"
    return f"距离活动结束还有{minutes}分钟"


def format_datetime(timestamp: int | None) -> str:
    if timestamp is None:
        return "-"
    return datetime.fromtimestamp(timestamp, LOCAL_TZ).strftime("%Y-%m-%d %H:%M")


def format_score(value: int | float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) / 10000:.4f}w"


def normalize_scope(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"total", "all", "总榜"}:
        return "total"
    if normalized in {"chapter", "character", "single", "singleline", "单人榜", "个人榜", "章节"}:
        return "chapter"
    return None


def default_total_output_path(output: Path) -> Path:
    if output.stem.endswith("_prediction"):
        return output.with_name(output.stem.removesuffix("_prediction") + "_total_prediction" + output.suffix)
    return output.with_name(output.stem + "_total" + output.suffix)


def retarget_event_path(path: str | Path, old_event_id: int, new_event_id: int) -> str:
    value = str(path)
    old = str(old_event_id)
    new = str(new_event_id)
    for source, target in (
        (f"_{old}_", f"_{new}_"),
        (f"_{old}.", f"_{new}."),
        (f"/{old}/", f"/{new}/"),
    ):
        value = value.replace(source, target)
    return value


def fit_text(text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    if font.getlength(text) <= max_width:
        return text
    ellipsis = "..."
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if font.getlength(text[:mid] + ellipsis) <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + ellipsis


def rounded_glass(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int, fill: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=(255, 255, 255, 160), width=2)


class ForecastService:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = load_json(self.config_path)
        self.interval_seconds = int(os.environ.get(
            "HARUKI_FORECAST_INTERVAL_SECONDS",
            self.config.get("prediction_interval_seconds", 1800),
        ))
        self.region_timeout_seconds = int(os.environ.get("HARUKI_FORECAST_REGION_TIMEOUT_SECONDS", "900"))
        self.max_parallel_regions = max(
            1,
            int(os.environ.get(
                "HARUKI_FORECAST_MAX_PARALLEL_REGIONS",
                self.config.get("max_parallel_prediction_regions", 5),
            )),
        )
        self.prediction_worker_threads = max(
            1,
            int(os.environ.get(
                "HARUKI_FORECAST_PREDICTION_WORKER_THREADS",
                self.config.get("prediction_worker_threads", 1),
            )),
        )
        self.image_dir = localized_path(
            self.config.get("image_dir")
            or str(REPO_ROOT / "live_images")
        )
        self.refresh_lock = asyncio.Lock()
        self.last_refresh: dict[str, Any] = {
            "status": "idle",
            "updated_at": None,
            "error": None,
            "output_tail": "",
        }
        self._background_task: asyncio.Task[None] | None = None

    def next_refresh_delay(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        normal = max(self.interval_seconds, 60)
        urgent = max(60, min(normal, int(self.config.get("final_hours_prediction_interval_seconds", 300))))
        delay = float(normal)
        # Inspect all published chapter/total windows. An old, already ended
        # chapter must not leave the service permanently in urgent mode.
        for region in REGIONS:
            try:
                for scope in self.prediction_scopes_for_region(region):
                    path = self.prediction_path(region, scope)
                    if not path.exists():
                        continue
                    for row in load_json(path).get("predictions", []):
                        remaining = float(row.get("end_timestamp") or 0) - now
                        if remaining > 0 and row.get("series_type") in {"world_bloom", "world_bloom_total"}:
                            delay = min(delay, urgent)
                        if 0 < remaining <= 6 * 3600:
                            delay = min(delay, urgent)
                        elif remaining > 6 * 3600:
                            delay = min(delay, max(60, remaining - 6 * 3600))
            except (OSError, ValueError, KeyError, HTTPException):
                continue
        return delay

    def current_event_from_masterdata(self, region: str) -> dict[str, Any] | None:
        root = self.masterdata_root(region)
        if root is None:
            return None
        events_path = root / "events.json"
        if not events_path.exists():
            return None

        now_ms = int(time.time() * 1000)
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
            if start_value <= now_ms < aggregate_value:
                current_rows.append(row)
            elif start_value <= now_ms and (
                latest_started is None or start_value > int(latest_started.get("startAt") or 0)
            ):
                latest_started = row
        if current_rows:
            return max(current_rows, key=lambda item: int(item.get("startAt") or 0))
        # Do not switch to a published-but-not-started event: EventTracker does
        # not have live rows before the opening timestamp, which would produce
        # empty predictions. Keep the latest started event until the next one
        # actually begins.
        return latest_started

    def resolve_active_event(self, event: dict[str, Any]) -> dict[str, Any]:
        if not bool(self.config.get("auto_active_events", True)):
            return dict(event)
        region = str(event["region"])
        current = self.current_event_from_masterdata(region)
        if current is None:
            return dict(event)
        old_event_id = int(event["event_id"])
        new_event_id = int(current["id"])
        resolved = dict(event)
        resolved["event_id"] = new_event_id
        resolved["series_type"] = "world_bloom" if str(current.get("eventType")) == "world_bloom" else "normal"
        if "output" in resolved:
            resolved["output"] = retarget_event_path(str(resolved["output"]), old_event_id, new_event_id)
        if "total_output" in resolved:
            resolved["total_output"] = retarget_event_path(str(resolved["total_output"]), old_event_id, new_event_id)
        return resolved

    def active_events(self) -> dict[str, dict[str, Any]]:
        return {
            str(event["region"]): event
            for event in (self.resolve_active_event(item) for item in self.config.get("active_events", []))
        }

    def event_for_region(self, region: str) -> dict[str, Any]:
        event = self.active_events().get(region)
        if event is None:
            raise HTTPException(status_code=404, detail=f"unknown active region: {region}")
        return event

    def default_scope_for_region(self, region: str) -> str:
        event = self.event_for_region(region)
        meta = self.load_event_meta(region, int(event["event_id"]))
        return "chapter" if meta.get("series_type") == "world_bloom" and meta.get("chapters") else "total"

    def prediction_path(self, region: str, scope: str | None = None) -> Path:
        event = self.event_for_region(region)
        normalized_scope = normalize_scope(scope) or self.default_scope_for_region(region)
        meta = self.load_event_meta(region, int(event["event_id"]))
        configured = (
            localized_path(event.get("total_output") or str(default_total_output_path(localized_path(event["output"]))))
            if normalized_scope == "total" and meta.get("series_type") == "world_bloom"
            else localized_path(event["output"])
        )
        if configured.exists():
            return configured
        event_id = int(event["event_id"])
        if normalized_scope == "total" and meta.get("series_type") == "world_bloom":
            candidates = (
                REPO_ROOT / "live" / f"{region}_{event_id}_total_prediction.json",
                REPO_ROOT / "live" / f"{region}_{event_id}_prediction_total.json",
                REPO_ROOT / "results" / f"{region}_{event_id}_total_prediction.json",
            )
        else:
            candidates = (
                REPO_ROOT / "live" / f"{region}_{event_id}_prediction.json",
                REPO_ROOT / "results" / f"{region}_{event_id}_prediction_world_bloom.json",
                REPO_ROOT / "results" / f"{region}_{event_id}_prediction_pressure_batch.json",
                REPO_ROOT / "results" / f"{region}_{event_id}_prediction_region_pressure.json",
                REPO_ROOT / "results" / f"{region}_{event_id}_prediction.json",
            )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return configured

    def prediction_scopes_for_region(self, region: str) -> list[str]:
        event = self.event_for_region(region)
        meta = self.load_event_meta(region, int(event["event_id"]))
        if meta.get("series_type") == "world_bloom" and meta.get("chapters"):
            return ["chapter", "total"]
        return ["total"]

    def region_needs_refresh(self, region: str) -> bool:
        max_age = max(int(self.interval_seconds), 60)
        now = time.time()
        for scope in self.prediction_scopes_for_region(region):
            path = self.prediction_path(region, scope)
            if not path.exists() or path.stat().st_size <= 0:
                return True
            if now - path.stat().st_mtime > max_age:
                return True
        return False

    def regions_needing_refresh(self) -> list[str]:
        active = self.active_events()
        return [
            region
            for region in ("cn", "en", "tw", "kr", "jp")
            if region in active and self.region_needs_refresh(region)
        ]

    def masterdata_root(self, region: str) -> Path | None:
        required = self.config.get("required_local_paths", {})
        path = required.get(f"{region}_masterdata")
        if not path:
            return None
        root = localized_path(path)
        return root if root.exists() else None

    def load_event_meta(self, region: str, event_id: int) -> dict[str, Any]:
        root = self.masterdata_root(region)
        if root is None:
            return {
                "id": event_id,
                "name": f"Event {event_id}",
                "series_type": "unknown",
                "start_timestamp": None,
                "end_timestamp": None,
                "characters": {},
            }
        events_path = root / "events.json"
        event = None
        if events_path.exists():
            for row in load_json(events_path):
                if int(row.get("id", -1)) == event_id:
                    event = row
                    break
        characters = self.load_character_names(root)
        chapters = self.load_world_bloom_chapters(root, event_id)
        if event is None:
            return {
                "id": event_id,
                "name": f"Event {event_id}",
                "series_type": "unknown",
                "start_timestamp": None,
                "end_timestamp": None,
                "characters": characters,
                "chapters": chapters,
            }
        return {
            "id": event_id,
            "name": str(event.get("name") or f"Event {event_id}"),
            "assetbundle_name": event.get("assetbundleName"),
            "unit": event.get("unit"),
            "event_type": event.get("eventType"),
            "series_type": "world_bloom" if str(event.get("eventType")) == "world_bloom" else "normal",
            "start_timestamp": int(event["startAt"]) // 1000 if event.get("startAt") is not None else None,
            "end_timestamp": int(event["aggregateAt"]) // 1000 if event.get("aggregateAt") is not None else None,
            "characters": characters,
            "chapters": chapters,
        }

    @staticmethod
    def load_character_names(root: Path) -> dict[int, str]:
        path = root / "gameCharacters.json"
        if not path.exists():
            return {}
        output: dict[int, str] = {}
        for row in load_json(path):
            character_id = as_int(row.get("id"))
            if character_id <= 0:
                continue
            first_name = str(row.get("firstName") or "")
            given_name = str(row.get("givenName") or "")
            output[character_id] = (first_name + given_name).strip() or f"角色{character_id}"
        return output

    @staticmethod
    def load_world_bloom_chapters(root: Path, event_id: int) -> dict[int, dict[str, Any]]:
        path = root / "worldBlooms.json"
        if not path.exists():
            return {}
        output: dict[int, dict[str, Any]] = {}
        for row in load_json(path):
            if as_int(row.get("eventId"), -1) != event_id:
                continue
            character_id = as_int(row.get("gameCharacterId"))
            if character_id <= 0:
                continue
            output[character_id] = {
                "chapter_no": as_int(row.get("chapterNo")),
                "start_timestamp": as_int(row.get("chapterStartAt")) // 1000 if row.get("chapterStartAt") is not None else None,
                "end_timestamp": as_int(row.get("aggregateAt")) // 1000 if row.get("aggregateAt") is not None else None,
            }
        return output

    def load_prediction(self, region: str, scope: str | None = None) -> dict[str, Any]:
        normalized_scope = normalize_scope(scope) or self.default_scope_for_region(region)
        path = self.prediction_path(region, normalized_scope)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"prediction output not found: {path}")
        try:
            payload = load_json(path)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail=f"prediction output is not valid JSON: {path}") from exc
        event_id = int(payload.get("event_id") or self.event_for_region(region)["event_id"])
        meta = self.load_event_meta(region, event_id)
        leaderboard_scope = str(payload.get("leaderboard_scope") or normalized_scope)
        groups: dict[int, list[dict[str, Any]]] = {}
        for row in payload.get("predictions", []):
            if not isinstance(row, dict):
                continue
            character_id = as_int(row.get("character_id"))
            groups.setdefault(character_id, []).append(row)
        lines = []
        for character_id, rows in sorted(groups.items()):
            line_rows = normalize_line_rows(rows)
            latest_ts = max((row.get("current_timestamp") or 0 for row in line_rows), default=0)
            end_ts = next((row.get("end_timestamp") for row in line_rows if row.get("end_timestamp")), None)
            progress = next((row.get("progress") for row in line_rows if row.get("progress") is not None), None)
            chapter = meta.get("chapters", {}).get(character_id, {}) if isinstance(meta.get("chapters"), dict) else {}
            if end_ts is None:
                end_ts = chapter.get("end_timestamp") or meta.get("end_timestamp")
            lines.append({
                "character_id": character_id,
                "character_name": meta.get("characters", {}).get(character_id) if character_id else None,
                "chapter_no": chapter.get("chapter_no") if isinstance(chapter, dict) else None,
                "leaderboard_scope": leaderboard_scope,
                "progress": progress,
                "current_timestamp": latest_ts or None,
                "end_timestamp": end_ts,
                "rows": line_rows,
            })
        now = time.time()
        updated_at = path.stat().st_mtime
        return {
            "region": region,
            "region_title": REGION_TITLES.get(region, region.upper()),
            "event_id": event_id,
            "event_name": meta.get("name"),
            "leaderboard_scope": leaderboard_scope,
            "series_type": meta.get("series_type") or payload.get("series_type"),
            "prediction_series_type": payload.get("series_type"),
            "event_type": meta.get("event_type"),
            "assetbundle_name": meta.get("assetbundle_name"),
            "start_timestamp": meta.get("start_timestamp"),
            "end_timestamp": meta.get("end_timestamp"),
            "model_name": payload.get("model_name"),
            "median_heat": payload.get("median_heat"),
            "shape_live": payload.get("shape_live"),
            "segmented_heat": payload.get("segmented_heat"),
            "prediction_path": str(path),
            "updated_at": int(updated_at),
            "updated_age_seconds": max(int(now - updated_at), 0),
            "stale": now - updated_at > max(self.interval_seconds * 2, 3600),
            "display_ranks": list(DISPLAY_RANKS),
            "lines": lines,
            "refresh": self.last_refresh,
        }

    @staticmethod
    def select_line(payload: dict[str, Any], character_id: int | None) -> dict[str, Any]:
        lines = payload.get("lines") or []
        if not lines:
            raise HTTPException(status_code=404, detail="prediction has no rank lines")
        if character_id is not None:
            for line in lines:
                if as_int(line.get("character_id")) == character_id:
                    return line
            raise HTTPException(status_code=404, detail=f"character_id {character_id} is not in this prediction")
        if len(lines) == 1:
            return lines[0]
        active = [
            line
            for line in lines
            if line.get("current_timestamp") and (line.get("progress") is None or float(line["progress"]) < 1.01)
        ]
        if active:
            return max(active, key=lambda line: (as_int(line.get("current_timestamp")), as_int(line.get("chapter_no"))))
        return max(lines, key=lambda line: as_int(line.get("current_timestamp")))

    async def run_region_prediction(self, region: str) -> dict[str, Any]:
        command = [
            sys.executable,
            str(FORECAST_OPS),
            "predict",
            "--config",
            str(self.config_path),
            "--region",
            region,
        ]
        started_at = int(time.time())
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(REPO_ROOT),
            env=self.prediction_subprocess_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=self.region_timeout_seconds,
            )
        except asyncio.TimeoutError:
            process.kill()
            stdout, _ = await process.communicate()
            output = stdout.decode("utf-8", errors="replace") if stdout else ""
            return {
                "region": region,
                "status": "error",
                "started_at": started_at,
                "updated_at": int(time.time()),
                "returncode": None,
                "error": f"region prediction timed out after {self.region_timeout_seconds}s",
                "output_tail": output[-4000:],
            }

        output = stdout.decode("utf-8", errors="replace") if stdout else ""
        return {
            "region": region,
            "status": "ok" if process.returncode == 0 else "error",
            "started_at": started_at,
            "updated_at": int(time.time()),
            "returncode": process.returncode,
            "error": None if process.returncode == 0 else f"forecast_ops exited with {process.returncode}",
            "output_tail": output[-4000:],
        }

    def prediction_subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        worker_threads = str(self.prediction_worker_threads)
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[key] = worker_threads
        return env

    async def refresh_predictions(
        self,
        region: str | None = None,
        regions_override: list[str] | None = None,
    ) -> dict[str, Any]:
        async with self.refresh_lock:
            if regions_override is not None:
                regions = list(regions_override)
            else:
                regions = [region] if region else [
                    item for item in ("cn", "en", "tw", "kr", "jp") if item in self.active_events()
                ]
            started_at = int(time.time())
            results: list[dict[str, Any]] = []
            self.last_refresh = {
                "status": "running",
                "started_at": started_at,
                "updated_at": started_at,
                "region": region,
                "regions": regions,
                "results": results,
                "error": None,
                "output_tail": "",
            }
            running_regions: set[str] = set()
            semaphore = asyncio.Semaphore(self.max_parallel_regions)

            async def run_tracked(current_region: str) -> dict[str, Any]:
                async with semaphore:
                    running_regions.add(current_region)
                    self.last_refresh = {
                        **self.last_refresh,
                        "status": "running",
                        "updated_at": int(time.time()),
                        "current_region": current_region,
                        "running_regions": sorted(running_regions),
                        "results": results,
                    }
                    try:
                        return await self.run_region_prediction(current_region)
                    except Exception as exc:
                        return {
                            "region": current_region,
                            "status": "error",
                            "started_at": started_at,
                            "updated_at": int(time.time()),
                            "returncode": None,
                            "error": str(exc),
                            "output_tail": "",
                        }
                    finally:
                        running_regions.discard(current_region)

            tasks = [asyncio.create_task(run_tracked(current_region)) for current_region in regions]
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
                self.last_refresh = {
                    **self.last_refresh,
                    "status": "running",
                    "updated_at": int(time.time()),
                    "current_region": None,
                    "running_regions": sorted(running_regions),
                    "results": results,
                }

            failed = [item for item in results if item["status"] != "ok"]
            payload = {
                "status": "partial_error" if failed else "ok",
                "started_at": started_at,
                "updated_at": int(time.time()),
                "region": region,
                "regions": regions,
                "results": results,
                "error": "; ".join(f"{item['region']}: {item['error']}" for item in failed) or None,
                "output_tail": "\n".join(item.get("output_tail", "") for item in results)[-4000:],
            }
            self.last_refresh = payload
            if region is not None and failed:
                raise RuntimeError(payload["error"] or "region prediction failed")
            return payload

    async def refresh_loop(self) -> None:
        try:
            initial_regions = self.regions_needing_refresh()
            if initial_regions:
                await self.refresh_predictions(regions_override=initial_regions)
            else:
                self.last_refresh = {
                    "status": "skipped",
                    "updated_at": int(time.time()),
                    "region": None,
                    "regions": [],
                    "results": [],
                    "error": None,
                    "output_tail": "All active prediction files are fresh; skip startup refresh.",
                }
        except Exception as exc:
            self.last_refresh = {
                "status": "error",
                "updated_at": int(time.time()),
                "error": str(exc),
                "output_tail": "",
            }
        while True:
            # Keep a start-to-start cadence, rather than adding runtime to
            # the configured interval on every cycle.
            started = float(self.last_refresh.get("started_at") or time.time())
            delay = self.next_refresh_delay()
            await asyncio.sleep(max(1, delay - max(time.time() - started, 0)))
            try:
                await self.refresh_predictions()
            except Exception as exc:
                self.last_refresh = {
                    "status": "error",
                    "updated_at": int(time.time()),
                    "error": str(exc),
                    "output_tail": "",
                }

    def start_background(self) -> None:
        if self._background_task is None:
            self._background_task = asyncio.create_task(self.refresh_loop())

    async def stop_background(self) -> None:
        if self._background_task is None:
            return
        self._background_task.cancel()
        await asyncio.gather(self._background_task, return_exceptions=True)
        self._background_task = None

    def render_prediction_image(self, payload: dict[str, Any], character_id: int | None = None) -> bytes:
        line = self.select_line(payload, character_id)
        rows = line["rows"]
        width = 612
        margin = 20
        header_h = 92
        table_top = margin + header_h + 10
        row_h = 35
        footer_rows = 2
        table_rows = 1 + len(rows) + footer_rows
        height = table_top + table_rows * row_h + 28

        image = Image.new("RGBA", (width, height), (239, 230, 248, 255))
        bg = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        bg_draw = ImageDraw.Draw(bg, "RGBA")
        for y in range(height):
            r = int(238 + 10 * y / max(height, 1))
            g = int(232 + 13 * y / max(height, 1))
            b = int(248 - 6 * y / max(height, 1))
            bg_draw.line((0, y, width, y), fill=(r, g, b, 255))
        for box, fill in (
            ((-80, -40, 220, 180), (255, 205, 220, 80)),
            ((390, -60, 700, 210), (188, 209, 255, 95)),
            ((-40, 500, 240, 760), (193, 242, 246, 80)),
            ((380, 600, 720, 980), (255, 220, 170, 65)),
        ):
            bg_draw.ellipse(box, fill=fill)
        image = Image.alpha_composite(image, bg.filter(ImageFilter.GaussianBlur(18)))
        draw = ImageDraw.Draw(image, "RGBA")

        regular, bold = self.fonts()
        title_font = bold(20)
        text_font = regular(18)
        row_font = regular(20)
        row_bold = bold(20)
        small_font = regular(11)

        header_box = (margin, margin, width - margin, margin + header_h)
        rounded_glass(draw, header_box, 10, (255, 255, 255, 178))
        title = f"【{payload['region_title']}-{payload['event_id']}】 {payload.get('event_name') or ''}"
        title_suffix = ""
        if payload.get("leaderboard_scope") == "total":
            title_suffix = "  总榜"
        elif payload.get("series_type") == "world_bloom":
            chapter = line.get("chapter_no")
            character_name = line.get("character_name") or f"角色{line.get('character_id')}"
            title_suffix = f"  Ch.{chapter} {character_name}" if chapter else f"  {character_name}"
        draw.text((margin + 20, margin + 10), fit_text(title + title_suffix, title_font, width - 260), fill=(20, 20, 28, 255), font=title_font)
        time_text = f"{format_datetime(payload.get('start_timestamp'))} ~ {format_datetime(payload.get('end_timestamp'))}"
        draw.text((margin + 10, margin + 41), time_text, fill=(22, 22, 30, 255), font=text_font)
        end_ts = line.get("end_timestamp") or payload.get("end_timestamp")
        remain = remaining_to_text(float(end_ts) - time.time()) if end_ts else "距离活动结束时间未知"
        draw.text((margin + 10, margin + 66), remain, fill=(10, 10, 18, 255), font=bold(17))

        pill = (width - 252, margin + 11, width - 115, margin + 66)
        draw.rounded_rectangle(pill, radius=8, fill=(250, 205, 232, 125), outline=(255, 255, 255, 165), width=1)
        draw.text((pill[0] + 22, pill[1] + 14), "本地预测", fill=(164, 60, 116, 230), font=bold(21))

        table_left = margin
        table_width = width - margin * 2
        col_w = table_width // 3
        table_box = (table_left, table_top, table_left + table_width, table_top + table_rows * row_h)
        rounded_glass(draw, table_box, 8, (255, 255, 255, 132))
        headers = ("排名", "当前榜线", "本地预测")
        for col, header in enumerate(headers):
            x0 = table_left + col * col_w
            draw.rectangle((x0, table_top, x0 + col_w, table_top + row_h), fill=(248, 246, 252, 178))
            draw.text(
                (x0 + col_w / 2, table_top + 7),
                header,
                fill=(14, 14, 18, 255),
                font=row_bold,
                anchor="ma",
            )
        for col in range(1, 3):
            x = table_left + col * col_w
            draw.line((x, table_top, x, table_box[3]), fill=(225, 214, 235, 220), width=5)
        for idx, row in enumerate(rows, start=1):
            y0 = table_top + idx * row_h
            fill = (255, 255, 255, 122) if idx % 2 else (247, 239, 250, 132)
            draw.rectangle((table_left, y0, table_left + table_width, y0 + row_h), fill=fill)
            values = (
                f"{int(row['rank']):,}",
                format_score(row.get("current_score")),
                format_score(row.get("prediction")),
            )
            for col, value in enumerate(values):
                x0 = table_left + col * col_w
                draw.text((x0 + col_w / 2, y0 + 5), value, fill=(18, 18, 24, 255), font=row_font, anchor="ma")
        footer_y = table_top + (1 + len(rows)) * row_h
        latest_ts = line.get("current_timestamp")
        file_age = payload.get("updated_age_seconds")
        footer = (
            ("预测时间", "-", seconds_to_text(float(file_age or 0))),
            ("获取时间", seconds_to_text(time.time() - float(latest_ts)) if latest_ts else "-", "0秒前"),
        )
        for idx, values in enumerate(footer):
            y0 = footer_y + idx * row_h
            draw.rectangle((table_left, y0, table_left + table_width, y0 + row_h), fill=(255, 255, 255, 145))
            for col, value in enumerate(values):
                x0 = table_left + col * col_w
                font = row_bold if col == 0 else row_font
                draw.text((x0 + col_w / 2, y0 + 5), value, fill=(18, 18, 24, 255), font=font, anchor="ma")
        for row_index in range(table_rows + 1):
            y = table_top + row_index * row_h
            draw.line((table_left, y, table_left + table_width, y), fill=(226, 216, 236, 185), width=2)

        watermark = datetime.now(LOCAL_TZ).strftime("DT: %Y-%m-%d %H:%M:%S (Asia/Shanghai)  generated by Haruki Forecast")
        draw.text((width / 2, height - 16), watermark, fill=(255, 255, 255, 230), stroke_width=1, stroke_fill=(110, 94, 128, 155), font=small_font, anchor="ma")
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()

    @staticmethod
    def fonts():
        font_env = os.environ.get("HARUKI_FORECAST_FONT")
        bold_font_env = os.environ.get("HARUKI_FORECAST_BOLD_FONT")
        candidates = [
            Path(font_env) if font_env else None,
            REPO_ROOT / "assets" / "SourceHanSansSC-Regular.otf",
            REPO_ROOT / "assets" / "SourceHanSansSC-Regular.ttf",
            Path("/Users/deseer/PycharmProjects/Haruki-Drawing-API/data/SourceHanSansSC-Regular.otf"),
            Path("/Users/deseer/PycharmProjects/Haruki-Drawing-API/data/SourceHanSansSC-Regular.ttf"),
            Path("/System/Library/Fonts/PingFang.ttc"),
        ]
        bold_candidates = [
            Path(bold_font_env) if bold_font_env else None,
            REPO_ROOT / "assets" / "SourceHanSansSC-Bold.otf",
            REPO_ROOT / "assets" / "SourceHanSansSC-Bold.ttf",
            Path("/Users/deseer/PycharmProjects/Haruki-Drawing-API/data/SourceHanSansSC-Bold.otf"),
            Path("/Users/deseer/PycharmProjects/Haruki-Drawing-API/data/SourceHanSansSC-Bold.ttf"),
            Path("/System/Library/Fonts/PingFang.ttc"),
        ]

        def loader(paths: list[Path | None]):
            existing = next((path for path in paths if path is not None and path.exists()), None)

            def load(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
                if existing is None:
                    return ImageFont.load_default()
                return ImageFont.truetype(str(existing), size=size)

            return load

        return loader(candidates), loader(bold_candidates)

    async def save_image(self, region: str, character_id: int | None = None, scope: str | None = None) -> Path:
        normalized_scope = normalize_scope(scope) or self.default_scope_for_region(region)
        payload = self.load_prediction(region, normalized_scope)
        event_id = int(payload["event_id"])
        scope_suffix = "_total" if normalized_scope == "total" else ""
        suffix = f"_{character_id}" if character_id is not None else scope_suffix
        output = self.image_dir / f"{region}_{event_id}{suffix}_prediction.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(self.render_prediction_image(payload, character_id))
        return output


service = ForecastService(Path(os.environ.get("HARUKI_FORECAST_CONFIG", DEFAULT_CONFIG)))


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.environ.get("HARUKI_FORECAST_DISABLE_BACKGROUND", "").lower() not in {"1", "true", "yes"}:
        service.start_background()
    try:
        yield
    finally:
        await service.stop_background()


app = FastAPI(title="Haruki Tracker Forecast Service", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "regions": list(service.active_events()),
        "interval_seconds": service.interval_seconds,
        "effective_interval_seconds": service.next_refresh_delay(),
        "max_parallel_regions": service.max_parallel_regions,
        "prediction_worker_threads": service.prediction_worker_threads,
        "refresh": service.last_refresh,
    }


@app.get("/regions")
async def regions() -> dict[str, Any]:
    return {"regions": service.active_events()}


@app.post("/refresh")
async def refresh_all() -> dict[str, Any]:
    return await service.refresh_predictions()


@app.post("/refresh/{region}")
async def refresh_region(region: str) -> dict[str, Any]:
    if region not in REGIONS:
        raise HTTPException(status_code=404, detail=f"unknown region: {region}")
    return await service.refresh_predictions(region=region)


@app.get("/prediction/{region}")
async def prediction(
    region: str,
    character_id: int | None = Query(default=None),
    scope: str | None = Query(default=None),
    refresh: bool = Query(default=False),
) -> JSONResponse:
    if region not in REGIONS:
        raise HTTPException(status_code=404, detail=f"unknown region: {region}")
    if refresh:
        await service.refresh_predictions(region=region)
    payload = service.load_prediction(region, scope)
    payload["selected_line"] = service.select_line(payload, character_id)
    return JSONResponse(payload)


@app.get("/prediction/{region}/image")
async def prediction_image(
    region: str,
    character_id: int | None = Query(default=None),
    scope: str | None = Query(default=None),
    refresh: bool = Query(default=False),
) -> Response:
    if region not in REGIONS:
        raise HTTPException(status_code=404, detail=f"unknown region: {region}")
    if refresh:
        await service.refresh_predictions(region=region)
    payload = service.load_prediction(region, scope)
    return Response(service.render_prediction_image(payload, character_id), media_type="image/png")


@app.get("/prediction/{region}/total")
async def prediction_total(
    region: str,
    refresh: bool = Query(default=False),
) -> JSONResponse:
    if region not in REGIONS:
        raise HTTPException(status_code=404, detail=f"unknown region: {region}")
    if refresh:
        await service.refresh_predictions(region=region)
    payload = service.load_prediction(region, "total")
    payload["selected_line"] = service.select_line(payload, None)
    return JSONResponse(payload)


@app.get("/prediction/{region}/total/image")
async def prediction_total_image(
    region: str,
    refresh: bool = Query(default=False),
) -> Response:
    if region not in REGIONS:
        raise HTTPException(status_code=404, detail=f"unknown region: {region}")
    if refresh:
        await service.refresh_predictions(region=region)
    payload = service.load_prediction(region, "total")
    return Response(service.render_prediction_image(payload), media_type="image/png")


@app.get("/prediction/{region}/chapter")
async def prediction_chapter(
    region: str,
    character_id: int | None = Query(default=None),
    refresh: bool = Query(default=False),
) -> JSONResponse:
    if region not in REGIONS:
        raise HTTPException(status_code=404, detail=f"unknown region: {region}")
    if refresh:
        await service.refresh_predictions(region=region)
    payload = service.load_prediction(region, "chapter")
    payload["selected_line"] = service.select_line(payload, character_id)
    return JSONResponse(payload)


@app.get("/prediction/{region}/chapter/image")
async def prediction_chapter_image(
    region: str,
    character_id: int | None = Query(default=None),
    refresh: bool = Query(default=False),
) -> Response:
    if region not in REGIONS:
        raise HTTPException(status_code=404, detail=f"unknown region: {region}")
    if refresh:
        await service.refresh_predictions(region=region)
    payload = service.load_prediction(region, "chapter")
    return Response(service.render_prediction_image(payload, character_id), media_type="image/png")


class ReverseWebSocketActions:
    def __init__(self, websocket: WebSocket, timeout_seconds: float = 10.0) -> None:
        self.websocket = websocket
        self.timeout_seconds = timeout_seconds
        self.pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self.send_lock = asyncio.Lock()

    def resolve_response(self, payload: dict[str, Any]) -> bool:
        echo = payload.get("echo")
        if not isinstance(echo, str):
            return False
        future = self.pending.get(echo)
        if future is None or future.done():
            return False
        future.set_result(payload)
        return True

    async def close(self) -> None:
        error = RuntimeError("reverse websocket connection closed")
        for future in self.pending.values():
            if not future.done():
                future.set_exception(error)
        self.pending.clear()

    async def call(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        echo = f"forecast-ws:{uuid.uuid4().hex}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self.pending[echo] = future
        try:
            async with self.send_lock:
                await self.websocket.send_json({"action": action, "params": params, "echo": echo})
            response = await asyncio.wait_for(future, timeout=self.timeout_seconds)
        finally:
            self.pending.pop(echo, None)
        if response.get("status") != "ok" or response.get("retcode", 0) != 0:
            raise RuntimeError(f"OneBot action failed: {response}")
        return response


def message_text(payload: dict[str, Any]) -> str:
    message = payload.get("message")
    if isinstance(message, str):
        return message
    if not isinstance(message, list):
        return str(payload.get("raw_message") or "")
    parts: list[str] = []
    for segment in message:
        if not isinstance(segment, dict):
            continue
        if segment.get("type") == "text":
            data = segment.get("data")
            if isinstance(data, dict):
                parts.append(str(data.get("text") or ""))
    return "".join(parts) or str(payload.get("raw_message") or "")


COMMAND_RE = re.compile(
    r"^\s*(jp|en|tw|kr|cn)\s*(?:(总榜|单人榜|个人榜|章节|chapter|total)\s*)?预测\s*(?:(总榜|单人榜|个人榜|章节|chapter|total)\s*)?$",
    re.IGNORECASE,
)


def command_scope(*tokens: str | None) -> str | None:
    for token in tokens:
        scope = normalize_scope(token)
        if scope is not None:
            return scope
    return None


async def send_message(actions: ReverseWebSocketActions, payload: dict[str, Any], message: list[dict[str, Any]] | str) -> None:
    message_type = payload.get("message_type")
    if message_type == "group":
        await actions.call("send_group_msg", {"group_id": payload.get("group_id"), "message": message})
        return
    if message_type == "private":
        await actions.call("send_private_msg", {"user_id": payload.get("user_id"), "message": message})
        return


async def handle_ws_payload(payload: dict[str, Any], actions: ReverseWebSocketActions) -> None:
    if payload.get("post_type") != "message":
        return
    text = message_text(payload)
    match = COMMAND_RE.match(text)
    if match is None:
        return
    region = match.group(1).lower()
    scope = command_scope(match.group(2), match.group(3))
    try:
        path = await service.save_image(region, scope=scope)
        encoded_image = base64.b64encode(path.read_bytes()).decode("ascii")
        await send_message(
            actions,
            payload,
            [{"type": "image", "data": {"file": f"base64://{encoded_image}"}}],
        )
    except Exception as exc:
        await send_message(actions, payload, f"{REGION_TITLES.get(region, region.upper())}预测图生成失败：{exc}")


@app.websocket("/ws")
@app.websocket("/onebot/ws")
async def onebot_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    actions = ReverseWebSocketActions(websocket)
    tasks: set[asyncio.Task[None]] = set()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if actions.resolve_response(payload):
                continue
            task = asyncio.create_task(handle_ws_payload(payload, actions))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
    except WebSocketDisconnect:
        pass
    finally:
        for task in list(tasks):
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await actions.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Haruki forecast HTTP and OneBot WS service.")
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("HARUKI_FORECAST_CONFIG", DEFAULT_CONFIG)))
    parser.add_argument("--host", default=os.environ.get("HARUKI_FORECAST_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("HARUKI_FORECAST_PORT", "18746")))
    args = parser.parse_args()

    os.environ["HARUKI_FORECAST_CONFIG"] = str(args.config.resolve())
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
