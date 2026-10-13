"""Finished, complete tracker lines, including files reached through WD symlinks."""
from functools import lru_cache
import csv
import os
from pathlib import Path


def history_root():
    return Path(os.environ.get('HARUKI_FORECAST_HISTORY_DIR', str(Path(__file__).resolve().parents[1] / 'live_history')))


def read_points(path):
    with path.open(encoding='utf-8') as handle:
        for raw in csv.reader(handle, delimiter='\t'):
            try:
                if len(raw) == 3:
                    rank, timestamp, score = map(int, raw)
                    yield 0, rank, timestamp, score
                elif len(raw) == 4:
                    yield tuple(map(int, raw))
            except ValueError:
                continue


@lru_cache(maxsize=5)
def index_region(region, root, master_root):
    import json, time
    events = {int(r['id']): r for r in json.loads((Path(master_root) / 'events.json').read_text())}
    chapters = {}
    chapter_file = Path(master_root) / 'worldBlooms.json'
    if chapter_file.exists():
        for r in json.loads(chapter_file.read_text()):
            if not all(r.get(k) is not None for k in ('eventId', 'gameCharacterId', 'chapterStartAt')):
                continue
            chapters[(int(r['eventId']), int(r['gameCharacterId']))] = (
                int(r['chapterStartAt']) // 1000,
                int(r.get('aggregateAt') or r['chapterEndAt']) // 1000,
            )
    index = {}
    for path in sorted(Path(root).glob(f'{region}_*_history.tsv')):
        event_id = int(path.name.split('_')[1])
        event = events.get(event_id)
        if not event or int(event['aggregateAt']) > time.time() * 1000:
            continue
        series = 'world_bloom' if event['eventType'] == 'world_bloom' else 'normal'
        limits = {}
        for char, rank, timestamp, score in read_points(path):
            if series == 'world_bloom':
                # Total WL lines are used by the opening prior, not chapter training.
                window = chapters.get((event_id, char))
                if window is None:
                    continue
            else:
                if char != 0:
                    continue
                window = (int(event['startAt']) // 1000, int(event['aggregateAt']) // 1000)
            if timestamp < window[0] or score < 0:
                continue
            key = (char, rank)
            lo, hi = limits.get(key, (timestamp, timestamp))
            limits[key] = (min(lo, timestamp), max(hi, timestamp))
        for (char, rank), (lo, hi) in limits.items():
            end = chapters[(event_id, char)][1] if series == 'world_bloom' else int(event['aggregateAt']) // 1000
            # Do not train on a truncated line using its last score as the final label.
            if hi < end - 120 or lo >= end:
                continue
            key = (event_id, series, char, rank)
            if key not in index or '_total_' in path.name:
                index[key] = path
    return index


def live_index(module, region):
    return index_region(region, str(history_root()), str(module.MASTERDATA_ROOTS[region]))


def iter_rows(module, region, selected_ranks, model_series_types=None):
    index = live_index(module, region)
    for event_id in sorted({key[0] for key in index}):
        paths = {path for key, path in index.items() if key[0] == event_id}
        points = {}
        for path in sorted(paths):
            for char, rank, timestamp, score in read_points(path):
                if rank not in selected_ranks:
                    continue
                for series in ('normal', 'world_bloom'):
                    if model_series_types is not None and series not in model_series_types:
                        continue
                    if index.get((event_id, series, char, rank)) != path:
                        continue
                    points[(timestamp, char, rank)] = (series, score)
        for (timestamp, char, rank), (series, score) in sorted(points.items()):
            yield dict(region=region, event_id=event_id, series_type=series,
                       character_id=char, rank=rank, timestamp=timestamp, score=score)
