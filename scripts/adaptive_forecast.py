"""Small, auditable candidate library shared by live inference and replay."""
from __future__ import annotations
import math
import bisect


def trend_diagnostics(history, now):
    """Observe short-window changes without treating missing reports as parking."""
    if len(history) < 4:
        return {'state': 'insufficient', 'windows': {}}
    points = sorted({int(ts): float(score) for ts, score in history}.items())
    timestamps = [ts for ts, score in points]
    if now - timestamps[-1] > 300:
        return {'state': 'stale', 'age_seconds': now-timestamps[-1], 'windows': {}}
    if any(b[1] < a[1] for a,b in zip(points,points[1:])):
        return {'state': 'invalid_decrease', 'windows': {}}
    end, score = points[-1]
    windows = {}
    for minutes in (5,15,30,60):
        idx = bisect.bisect_right(timestamps, end-minutes*60)-1
        if idx < 0 or len(points)-idx < 3:
            windows[str(minutes)] = None
            continue
        ts, old = points[idx]
        max_gap = max(b[0]-a[0] for a,b in zip(points[idx:],points[idx+1:]))
        if max_gap > 300 or end-ts > minutes*90:
            windows[str(minutes)] = None
        else:
            windows[str(minutes)] = (score-old)*3600/(end-ts)
    fast, medium, slow = (windows.get(k) for k in ('5','15','60'))
    state = 'stable'
    if any(v is None for v in (fast,medium,slow)):
        state = 'insufficient'
    elif slow > 0 and fast > 1.8*slow and medium > 1.3*slow:
        state = 'acceleration'
    elif slow > 0 and fast < .3*slow and medium < .6*slow:
        state = 'slowdown'
    return {'state':state, 'windows':windows, 'age_seconds':now-end,
            'policy':'diagnostic_only_v1'}


def stage(progress):
    return 'early' if progress < .5 else ('middle' if progress < .85 else 'late')


def rank_group(rank):
    return 'top10' if rank <= 10 else ('top100' if rank <= 100 else ('mid' if rank <= 10000 else 'tail'))


def policy_key(region, series, rank, progress):
    return '|'.join([region, series, rank_group(rank), stage(progress)])


def candidates(row, prior=None):
    current = float(row.get('current_score', row.get('current', 0)))
    raw = max(current, float(row.get('calibrated_model_prediction') or row.get('raw_model_prediction') or current))
    shape = max(current, float(row.get('shape_first_prediction') or current))
    hybrid = max(current, float(row.get('hybrid_projection') or current))
    base = max(current, float(row.get('model_prediction') or current))
    result = {'published': base, 'raw': raw, 'shape': shape, 'hybrid': hybrid,
              'raw_shape': math.sqrt(max(raw, 1)*max(shape, 1))}
    if prior and prior > 0:
        # Absolute historical prior remains weak; real current score is a floor.
        result['prior_shape'] = max(current, math.exp(.25*math.log(prior) + .75*math.log(max(shape, 1))))
    return result


def apply_policy(rows, region, document):
    for row in rows:
        if row.get('event_finished_prediction') or row.get('synthetic_rank'):
            continue
        key = policy_key(region, str(row.get('series_type')), int(row['rank']), float(row.get('progress') or 0))
        policy = document.get('policies', {}).get(key)
        row['adaptive_policy_version'] = document.get('version')
        if not policy or policy.get('candidate') == 'published':
            continue
        values = candidates(row, policy.get('prior'))
        name = policy['candidate']
        if name not in values:
            continue
        row['pre_adaptive_prediction'] = row['model_prediction']
        row['adaptive_candidate'] = name
        row['adaptive_evidence_events'] = policy.get('event_count')
        row['pressure_adjusted_prediction'] = row['model_prediction'] = values[name]
