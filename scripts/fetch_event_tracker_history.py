#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import http.client
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_baseline.py"
DIRECT_OPENER = build_opener(ProxyHandler({}))


def parse_csv_ints(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_training_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("train_baseline_module", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_baseline_module"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def request_json(url: str, *, timeout: float, retries: int = 3):
    headers = {"Accept": "application/json", "Accept-Encoding": "gzip", "User-Agent": "Haruki-Tracker-Forecast/0.1"}
    token_file = os.environ.get("HARUKI_FORECAST_TRACKER_TOKEN_FILE")
    if token_file:
        token = Path(token_file).read_text(encoding="utf-8").strip()
        if not token or not token.isascii() or "\r" in token or "\n" in token:
            raise ValueError("Tracker token file must contain one nonempty ASCII token")
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    last_error: Exception | None = None
    for _attempt in range(max(int(retries), 1)):
        try:
            with DIRECT_OPENER.open(request, timeout=timeout) as response:
                data = response.read()
                if response.headers.get("Content-Encoding", "").lower() == "gzip":
                    data = gzip.decompress(data)
                return json.loads(data.decode("utf-8"))
        except (http.client.IncompleteRead, http.client.RemoteDisconnected, json.JSONDecodeError, gzip.BadGzipFile, EOFError, TimeoutError, URLError) as exc:
            last_error = exc
            time.sleep(0.5)
    assert last_error is not None
    raise last_error


def tracker_url(base_url: str, *parts: object) -> str:
    base = base_url.rstrip("/")
    path = "/".join(quote(str(part).strip("/")) for part in parts)
    return f"{base}/{path}"


def tracker_url_with_query(base_url: str, *parts: object, query: dict[str, object]) -> str:
    base = tracker_url(base_url, *parts)
    pairs = [
        f"{quote(str(key))}={quote(str(value))}"
        for key, value in query.items()
        if value is not None
    ]
    return f"{base}?{'&'.join(pairs)}" if pairs else base


def load_event_meta(module, data_root: Path, region: str, event_id: int):
    from_predict = REPO_ROOT / "scripts" / "predict_live_event.py"
    spec = importlib.util.spec_from_file_location("predict_live_event_module", from_predict)
    predict_module = importlib.util.module_from_spec(spec)
    sys.modules["predict_live_event_module"] = predict_module
    assert spec.loader is not None
    spec.loader.exec_module(predict_module)
    try:
        return predict_module.load_event_meta_from_masterdata(module, region, event_id)
    except (FileNotFoundError, ValueError):
        return predict_module.load_event_meta_from_manifest(data_root, module, region, event_id)


def world_bloom_characters(module, meta) -> list[int]:
    if meta is None or meta.series_type != "world_bloom":
        return []
    now = int(time.time())
    active = [
        int(character_id)
        for character_id, (start_timestamp, end_timestamp) in meta.chapter_windows.items()
        if int(start_timestamp) <= now <= int(end_timestamp)
    ]
    if active:
        return sorted(active)
    return sorted(int(character_id) for character_id in meta.chapter_windows)


def normalize_leaderboard_scope(series_type: str, value: str) -> str:
    if value != "auto":
        return value
    return "chapter" if series_type == "world_bloom" else "total"


def extract_rank_rows(payload) -> list[dict[str, int]]:
    rows = payload.get("rankData") if isinstance(payload, dict) else None
    if rows is None and isinstance(payload, list):
        rows = payload
    if not isinstance(rows, list):
        return []
    output: list[dict[str, int]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        timestamp = row.get("timestamp")
        score = row.get("score")
        rank = row.get("rank")
        if timestamp is None or score is None or rank is None:
            continue
        output.append({
            "timestamp": int(timestamp),
            "score": int(score),
            "rank": int(rank),
        })
    return output


def fetch_normal_rank(base_url: str, region: str, event_id: int, rank: int, *, timeout: float):
    v2_url = tracker_url_with_query(
        base_url,
        "api",
        "v2",
        "cloud",
        "events",
        region,
        event_id,
        "leaderboards",
        "total",
        "sk",
        "trace",
        query={
            "subjectType": "rank",
            "subject": rank,
        },
    )
    try:
        return extract_rank_rows(request_json(v2_url, timeout=timeout))
    except HTTPError as exc:
        if exc.code != 404:
            raise
    legacy_url = tracker_url(base_url, "event", region, event_id, "trace-ranking", "rank", rank)
    return extract_rank_rows(request_json(legacy_url, timeout=timeout))


def fetch_world_bloom_rank(base_url: str, region: str, event_id: int, character_id: int, rank: int, *, timeout: float):
    v2_url = tracker_url_with_query(
        base_url,
        "api",
        "v2",
        "cloud",
        "events",
        region,
        event_id,
        "leaderboards",
        "world-bloom",
        character_id,
        "sk",
        "trace",
        query={
            "subjectType": "rank",
            "subject": rank,
        },
    )
    try:
        return extract_rank_rows(request_json(v2_url, timeout=timeout))
    except HTTPError as exc:
        if exc.code != 404:
            raise
    legacy_url = tracker_url(
        base_url,
        "event",
        region,
        event_id,
        "trace-world-bloom-ranking",
        "character",
        character_id,
        "rank",
        rank,
    )
    return extract_rank_rows(request_json(legacy_url, timeout=timeout))


def dedupe_points(rows: list[dict[str, int]]) -> list[tuple[int, int]]:
    by_timestamp: dict[int, int] = {}
    for row in rows:
        timestamp = int(row["timestamp"])
        score = int(row["score"])
        by_timestamp[timestamp] = max(score, by_timestamp.get(timestamp, 0))
    return sorted(by_timestamp.items())


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch live rank history from Haruki Event Tracker as TSV.")
    parser.add_argument("--base-url", required=True, help="Haruki Event Tracker base URL, e.g. http://127.0.0.1:8080")
    parser.add_argument("--region", required=True, choices=("jp", "en", "tw", "kr", "cn"))
    parser.add_argument("--event-id", type=int, required=True)
    parser.add_argument("--series-type", choices=("auto", "normal", "world_bloom"), default="auto")
    parser.add_argument(
        "--leaderboard-scope",
        choices=("auto", "total", "chapter"),
        default="auto",
        help="For World Bloom, fetch total ranking or per-chapter character ranking.",
    )
    parser.add_argument("--characters", default=None, help="World Bloom character IDs. Defaults to masterdata chapter characters.")
    parser.add_argument("--ranks", default=None, help="Comma-separated ranks. Defaults to train_baseline.COMMON_RANKS.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-points", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--data-root", type=Path, default=Path("/Users/deseer/PycharmProjects/Haruki-Tracker-Exports"))
    parser.add_argument("--train-script", type=Path, default=TRAIN_SCRIPT)
    args = parser.parse_args()

    module = load_training_module(args.train_script.resolve())
    meta = load_event_meta(module, args.data_root.resolve(), args.region, args.event_id)
    series_type = args.series_type
    if series_type == "auto":
        if meta is None:
            raise SystemExit("Cannot infer series type; pass --series-type explicitly.")
        series_type = meta.series_type
    leaderboard_scope = normalize_leaderboard_scope(series_type, args.leaderboard_scope)

    ranks = parse_csv_ints(args.ranks) or list(module.COMMON_RANKS)
    characters = parse_csv_ints(args.characters)
    if series_type == "world_bloom" and leaderboard_scope == "chapter" and not characters:
        characters = world_bloom_characters(module, meta)
    if series_type == "world_bloom" and leaderboard_scope == "chapter" and not characters:
        raise SystemExit("World Bloom history requires --characters or masterdata worldBlooms.json.")

    records: list[tuple[int, int, int, int | None]] = []
    errors: list[str] = []
    fetched_lines = 0
    kept_lines = 0

    if series_type == "world_bloom" and leaderboard_scope == "chapter":
        for character_id in characters:
            for rank in ranks:
                fetched_lines += 1
                try:
                    rows = fetch_world_bloom_rank(
                        args.base_url,
                        args.region,
                        args.event_id,
                        character_id,
                        rank,
                        timeout=args.timeout,
                    )
                except HTTPError as exc:
                    if exc.code != 404:
                        errors.append(f"character={character_id} rank={rank}: HTTP {exc.code}")
                    continue
                except (URLError, TimeoutError, http.client.IncompleteRead, http.client.RemoteDisconnected, json.JSONDecodeError) as exc:
                    errors.append(f"character={character_id} rank={rank}: {exc}")
                    continue
                points = dedupe_points(rows)
                if len(points) < args.min_points:
                    continue
                kept_lines += 1
                for timestamp, score in points:
                    records.append((character_id, rank, timestamp, score))
    else:
        for rank in ranks:
            fetched_lines += 1
            try:
                rows = fetch_normal_rank(args.base_url, args.region, args.event_id, rank, timeout=args.timeout)
            except HTTPError as exc:
                if exc.code != 404:
                    errors.append(f"rank={rank}: HTTP {exc.code}")
                continue
            except (URLError, TimeoutError, http.client.IncompleteRead, http.client.RemoteDisconnected, json.JSONDecodeError) as exc:
                errors.append(f"rank={rank}: {exc}")
                continue
            points = dedupe_points(rows)
            if len(points) < args.min_points:
                continue
            kept_lines += 1
            for timestamp, score in points:
                records.append((0, rank, timestamp, score))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for character_id, rank, timestamp, score in sorted(records):
            if series_type == "world_bloom" and leaderboard_scope == "chapter":
                writer.writerow([character_id, rank, timestamp, score])
            else:
                writer.writerow([rank, timestamp, score])

    print(json.dumps({
        "region": args.region,
        "event_id": args.event_id,
        "series_type": series_type,
        "leaderboard_scope": leaderboard_scope,
        "output": str(args.output),
        "fetched_lines": fetched_lines,
        "kept_lines": kept_lines,
        "rows": len(records),
        "errors": errors[:20],
        "error_count": len(errors),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
