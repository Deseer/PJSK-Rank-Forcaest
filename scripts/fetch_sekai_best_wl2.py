#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


API_BASE = "https://api.sekai.best"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "sekai_best_jp_wl2"
JP_WL2_EVENT_IDS = (163, 167, 170, 171, 176, 179)
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
MASTERDATA_ROOT = Path("/Users/deseer/PycharmProjects/haruki-sekai-master/master")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_wl2_chapters(event_ids: tuple[int, ...]) -> dict[int, list[int]]:
    rows = load_json(MASTERDATA_ROOT / "worldBlooms.json")
    chapters: dict[int, list[int]] = {}
    for row in rows:
        event_id = int(row["eventId"])
        if event_id not in event_ids:
            continue
        chapters.setdefault(event_id, []).append(int(row["gameCharacterId"]))
    return {event_id: sorted(set(chapters.get(event_id, []))) for event_id in event_ids}


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
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
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
    raw_path = raw_dir / f"event_{event_id}" / f"chara_{character_id}" / f"rank_{rank}.json"
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


def iter_graph_rows(data: dict, event_id: int, character_id: int, rank: int):
    rankings = ((data.get("data") or {}).get("eventRankings") or [])
    seen: set[tuple[str, int]] = set()
    for row in rankings:
        timestamp = str(row.get("timestamp") or "")
        score = int(row.get("score") or 0)
        key = (timestamp, score)
        if not timestamp or key in seen:
            continue
        seen.add(key)
        yield {
            "region": "jp",
            "event_id": event_id,
            "character_id": character_id,
            "rank": rank,
            "timestamp": timestamp,
            "score": score,
            "user_id": str(row.get("userId") or ""),
            "user_name": str(row.get("userName") or ""),
        }


def write_outputs(output_dir: Path, rows: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (
        int(row["event_id"]),
        int(row["character_id"]),
        int(row["rank"]),
        str(row["timestamp"]),
    ))
    fieldnames = ["region", "event_id", "character_id", "rank", "timestamp", "score", "user_id", "user_name"]
    with gzip.open(output_dir / "jp_wl2_chapter_rank_timeseries.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    final_rows: dict[tuple[int, int, int], dict[str, object]] = {}
    for row in rows:
        final_rows[(int(row["event_id"]), int(row["character_id"]), int(row["rank"]))] = row
    with (output_dir / "jp_wl2_chapter_rank_finals.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(final_rows.values(), key=lambda item: (
            int(item["event_id"]),
            int(item["character_id"]),
            int(item["rank"]),
        )):
            writer.writerow(row)


def parse_ranks(value: str | None) -> tuple[int, ...]:
    if value is None or not value.strip():
        return DEFAULT_RANKS
    return tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch JP WL2 chapter ranking graphs from sekai.best.")
    parser.add_argument("--region", default="jp")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ranks", default=None, help="Comma-separated ranks. Defaults to Sekai Viewer tracker ranks.")
    parser.add_argument("--delay", type=float, default=0.12)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    event_ids = JP_WL2_EVENT_IDS
    ranks = parse_ranks(args.ranks)
    chapters = load_wl2_chapters(event_ids)
    raw_dir = args.output_dir / "raw"
    rows: list[dict[str, object]] = []
    total = sum(len(chapters[event_id]) * len(ranks) for event_id in event_ids)
    index = 0
    for event_id in event_ids:
        for character_id in chapters[event_id]:
            for rank in ranks:
                index += 1
                print(f"[{index}/{total}] event={event_id} chara={character_id} rank={rank}", flush=True)
                data = fetch_rank_graph(
                    event_id=event_id,
                    character_id=character_id,
                    rank=rank,
                    region=args.region,
                    raw_dir=raw_dir,
                    delay=args.delay,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                    refresh=args.refresh,
                )
                rows.extend(iter_graph_rows(data, event_id, character_id, rank))
    write_outputs(args.output_dir, rows)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "events": {str(event_id): chapters[event_id] for event_id in event_ids},
        "ranks": ranks,
        "rows": len(rows),
        "series": total,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
