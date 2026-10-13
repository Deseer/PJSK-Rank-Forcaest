#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


API_BASE = "https://api.sekai.best"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "sekai_best_world_bloom"
DEFAULT_RANKS = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
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
    7000,
    10000,
    20000,
    30000,
    40000,
    50000,
    70000,
    100000,
)


def load_training_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("train_baseline_module", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_baseline_module"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_csv_ints(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip() or value.strip().lower() == "all":
        return None
    return tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))


def parse_regions(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def world_bloom_chapters(module, region: str, *, only_finished: bool) -> dict[int, list[int]]:
    base = module.MASTERDATA_ROOTS[region]
    finished = module.load_finished_event_ids().get(region, set()) if only_finished else None
    chapters: dict[int, list[int]] = {}
    for row in load_json(base / "worldBlooms.json"):
        if "gameCharacterId" not in row:
            continue
        event_id = int(row["eventId"])
        if finished is not None and event_id not in finished:
            continue
        chapters.setdefault(event_id, []).append(int(row["gameCharacterId"]))
    return {event_id: sorted(set(character_ids)) for event_id, character_ids in sorted(chapters.items())}


def request_json(url: str, *, retries: int, retry_sleep: float):
    headers = {
        "Accept": "application/json",
        "User-Agent": "Haruki-Tracker-Forecast/1.0 (research; respectful cached fetch)",
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep * (attempt + 1))
    raise RuntimeError(f"request failed after {retries + 1} attempts: {url}") from last_error


def graph_url(event_id: int, character_id: int, rank: int, region: str) -> str:
    query = urllib.parse.urlencode({
        "region": region,
        "charaId": character_id,
        "rank": rank,
    })
    return f"{API_BASE}/event/{event_id}/chapter_rankings/graph?{query}"


def fetch_rank_graph(
    *,
    event_id: int,
    character_id: int,
    rank: int,
    region: str,
    raw_dir: Path,
    delay: float,
    retries: int,
    retry_sleep: float,
    refresh: bool,
) -> dict:
    raw_path = raw_dir / region / f"event_{event_id}" / f"chara_{character_id}" / f"rank_{rank}.json"
    if raw_path.exists() and not refresh:
        return load_json(raw_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    data = request_json(
        graph_url(event_id, character_id, rank, region),
        retries=retries,
        retry_sleep=retry_sleep,
    )
    raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    time.sleep(delay)
    return data


def iter_graph_rows(data: dict, *, region: str, event_id: int, character_id: int, rank: int):
    rankings = ((data.get("data") or {}).get("eventRankings") or [])
    seen: set[tuple[str, int]] = set()
    for row in rankings:
        if not isinstance(row, dict):
            continue
        timestamp = str(row.get("timestamp") or "")
        score = int(row.get("score") or 0)
        key = (timestamp, score)
        if not timestamp or key in seen:
            continue
        seen.add(key)
        yield {
            "region": region,
            "event_id": event_id,
            "character_id": character_id,
            "rank": rank,
            "timestamp": timestamp,
            "score": score,
            "user_id": str(row.get("userId") or ""),
            "user_name": str(row.get("userName") or ""),
        }


def existing_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_region_outputs(output_dir: Path, region: str, rows: list[dict[str, object]]) -> None:
    region_dir = output_dir / region
    region_dir.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (
        int(row["event_id"]),
        int(row["character_id"]),
        int(row["rank"]),
        str(row["timestamp"]),
    ))
    fieldnames = ["region", "event_id", "character_id", "rank", "timestamp", "score", "user_id", "user_name"]
    with gzip.open(region_dir / f"{region}_world_bloom_chapter_rank_timeseries.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    final_rows: dict[tuple[int, int, int], dict[str, object]] = {}
    for row in rows:
        final_rows[(int(row["event_id"]), int(row["character_id"]), int(row["rank"]))] = row
    with (region_dir / f"{region}_world_bloom_chapter_rank_finals.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(final_rows.values(), key=lambda item: (
            int(item["event_id"]),
            int(item["character_id"]),
            int(item["rank"]),
        )):
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch World Bloom chapter ranking graphs from sekai.best.")
    parser.add_argument("--regions", default="en,tw,kr", help="Comma-separated regions. Use jp,en,tw,kr,cn as needed.")
    parser.add_argument("--events", default="finished", help="'finished', 'all', or comma-separated event ids.")
    parser.add_argument("--ranks", default=None, help="Comma-separated ranks. Defaults to tracker ranks.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-script", type=Path, default=Path(__file__).resolve().parent / "train_baseline.py")
    parser.add_argument("--delay", type=float, default=0.12)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    module = load_training_module(args.train_script.resolve())
    ranks = parse_csv_ints(args.ranks) or DEFAULT_RANKS
    event_filter = parse_csv_ints(args.events)
    only_finished = args.events.strip().lower() == "finished"
    raw_dir = args.output_dir / "raw"
    summary: dict[str, object] = {}

    for region in parse_regions(args.regions):
        chapters = world_bloom_chapters(module, region, only_finished=only_finished)
        if event_filter is not None:
            chapters = {event_id: chapters[event_id] for event_id in event_filter if event_id in chapters}
        output_path = args.output_dir / region / f"{region}_world_bloom_chapter_rank_timeseries.csv.gz"
        rows = existing_rows(output_path)
        seen = {
            (
                str(row["region"]),
                int(row["event_id"]),
                int(row["character_id"]),
                int(row["rank"]),
                str(row["timestamp"]),
                int(row["score"]),
            )
            for row in rows
        }
        total = sum(len(character_ids) * len(ranks) for character_ids in chapters.values())
        fetched = 0
        errors: list[str] = []
        index = 0
        for event_id, character_ids in chapters.items():
            for character_id in character_ids:
                for rank in ranks:
                    index += 1
                    print(f"[{region} {index}/{total}] event={event_id} chara={character_id} rank={rank}", flush=True)
                    try:
                        data = fetch_rank_graph(
                            event_id=event_id,
                            character_id=character_id,
                            rank=rank,
                            region=region,
                            raw_dir=raw_dir,
                            delay=args.delay,
                            retries=args.retries,
                            retry_sleep=args.retry_sleep,
                            refresh=args.refresh,
                        )
                    except RuntimeError as exc:
                        errors.append(str(exc))
                        print(f"warning: {exc}", flush=True)
                        if not args.continue_on_error:
                            raise
                        continue
                    for row in iter_graph_rows(data, region=region, event_id=event_id, character_id=character_id, rank=rank):
                        key = (
                            str(row["region"]),
                            int(row["event_id"]),
                            int(row["character_id"]),
                            int(row["rank"]),
                            str(row["timestamp"]),
                            int(row["score"]),
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        rows.append(row)
                        fetched += 1
        write_region_outputs(args.output_dir, region, rows)
        summary[region] = {
            "events": {str(event_id): character_ids for event_id, character_ids in chapters.items()},
            "ranks": list(ranks),
            "rows": len(rows),
            "new_rows": fetched,
            "errors": errors[:20],
            "error_count": len(errors),
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
