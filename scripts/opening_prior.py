"""Causal absolute-score prior for cold starts, decaying as actual pace accumulates."""
import math
import statistics
from live_training_history import history_root, read_points


def build_history(module, event_meta, finals):
    history = []
    contexts = module.load_event_feature_contexts()
    for (region, event_id, char, rank), (score, timestamp) in finals.items():
        meta = event_meta[(region, event_id)]
        if score <= 0 or not module.has_event_line_window(meta, char):
            continue
        if timestamp < module.event_line_end_timestamp(meta, char) - 120:
            continue
        history.append(dict(region=region, event_id=event_id, character_id=char, rank=rank,
                            series_type=meta.series_type, end_timestamp=module.event_line_end_timestamp(meta, char), score=score,
                            condition=contexts.get((region,event_id), module.EventFeatureContext()).condition))
    for region in module.REGIONS:
        for path in history_root().glob(f'{region}_*_total_history.tsv'):
            event_id = int(path.name.split('_')[1])
            meta = event_meta.get((region, event_id))
            if not meta or meta.series_type != 'world_bloom':
                continue
            last = {}
            for char, rank, timestamp, score in read_points(path):
                if char == 0 and (rank not in last or timestamp > last[rank][0]):
                    last[rank] = (timestamp, score)
            for rank, (timestamp, score) in last.items():
                if score > 0 and timestamp >= meta.end_timestamp - 120:
                    history.append(dict(region=region, event_id=event_id, character_id=0, rank=rank,
                                        series_type='world_bloom_total', end_timestamp=meta.end_timestamp, score=score,
                                        scope_kind='chapters' if meta.chapter_windows else 'finale'))
    return history


def adjust(prediction, current_score, region, event_id, character_id, rank, series_type, progress, start_timestamp, history,
           decay=0.04, strength=1.0, quantile=0.25, condition=None, scope_kind=None):
    debug = dict(opening_prior_applied=False)
    if progress >= decay or not history:
        return max(float(current_score), prediction), debug
    eligible = [r for r in history if r['region'] == region and r['rank'] == rank
                and r['series_type'] == series_type and r['event_id'] < event_id
                and r['end_timestamp'] < start_timestamp and r['score'] > 0]
    if scope_kind is not None:
        eligible = [r for r in eligible if r.get("scope_kind") == scope_kind]
    if condition is not None:
        same_condition = [r for r in eligible if r.get('condition') == condition]
        if len({r['event_id'] for r in same_condition}) >= 3:
            eligible = same_condition
    if series_type == 'world_bloom':
        same_char = [r for r in eligible if r['character_id'] == character_id]
        if len({r['event_id'] for r in same_char}) >= 3:
            eligible = same_char
    if len({r['event_id'] for r in eligible}) < 3:
        return max(float(current_score), prediction), debug
    # One equally weighted score per event, then the six most recent events.
    by_event = {}
    for r in eligible:
        by_event.setdefault(r['event_id'], []).append(r['score'])
    scores = [statistics.median(by_event[e]) for e in sorted(by_event)[-6:]]
    logs = sorted(math.log(s) for s in scores)
    position = quantile * (len(logs) - 1)
    lower = int(position)
    upper = min(lower + 1, len(logs) - 1)
    prior = math.exp(logs[lower] + (logs[upper] - logs[lower]) * (position - lower))
    weight = strength * min(1.0, max(0.0, (decay - progress) / (decay - 0.005))) ** 1.8
    if prediction < prior and weight > 0:
        prediction = math.exp((1.0 - weight) * math.log(max(prediction, 1.0)) + weight * math.log(prior))
        debug.update(opening_prior_applied=True, opening_prior_score=prior,
                     opening_prior_weight=weight, opening_prior_event_count=len(scores))
    return max(float(current_score), prediction), debug
