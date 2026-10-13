"""Opening forecast layer for normal events and World Link chapter/total boards.

The main model never sees progress below 5% and extrapolates badly there; World Link
boards are underestimated well into their second half. This layer estimates the final
score from two causal sources and blends it into the chain's output as boards mature:

* prior: duration-adjusted finals of earlier boards of the same kind at the same
  region/rank. Normal events match box/mixed and unit, and outside JP add the relative
  heat the same event had in JP (event ids and content are shared across servers).
  World Link chapters prefer chapters already finished in the same event.
* shape: the current score times the median growth from the same elapsed time to the
  final on those earlier boards.

Normal events combine the two with per-checkpoint coefficients fitted on as-of
observations; World Link boards use fixed progress schedules because far fewer boards
exist. Only boards that ended before the target started are consulted.
"""
import bisect
import json
import math
import os
import statistics
import tempfile
from pathlib import Path

import numpy as np

VERSION = "opening_blend_v2"
NORMAL, CHAPTER, TOTAL = "n", "c", "t"
CHECKPOINTS = (0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.25)
# Elapsed-hour grid for stored curves: 15 min to 6 h, 30 min to 24 h, hourly to 96 h, 2 h to 300 h.
GRID_HOURS = tuple([i / 4 for i in range(1, 25)] + [6 + i / 2 for i in range(1, 37)]
                   + [24 + i for i in range(1, 73)] + [96 + 2 * i for i in range(1, 103)])
DONORS = 8
CONDITION_DONORS = 6
UNIT_DONORS = 4
POOLED_DONORS = 16
MIN_EVENTS = 3
CROSS_SERVER_NEIGHBOURS = 8
MAX_GAP_SECONDS = 3 * 3600
STALE_SECONDS = 1800
# Live rows are estimated as of their own last observation; an hour-long tracker gap on one
# rank should not hand that rank back to the chain in the opening.
APPLY_MAX_AGE_SECONDS = 6 * 3600
# Weight against the existing chain: full until the first value, then linear to zero.
# Since the 2026-10-12 retrain the chain is as good as history late in normal events.
HANDOFF = {NORMAL: (0.10, 0.35), CHAPTER: (0.90, 1.00), TOTAL: (0.75, 1.00)}
# CN normal events keep an ensemble with their own opening layers and shape residual model.
REGION_WEIGHT_CAP = {"cn": 0.7}


def point_slot(elapsed_seconds):
    """5-minute slots for three days, hourly afterwards (later grid points are 2 h apart)."""
    if elapsed_seconds < 72 * 3600:
        return elapsed_seconds // 300
    return 864 + (elapsed_seconds - 72 * 3600) // 3600


def curve_from_points(points, start, duration):
    """Sample a finished line on GRID_HOURS by linear interpolation; long gaps stay unknown."""
    points = sorted(points)
    times = [t for t, _ in points]
    values = []
    for hours in GRID_HOURS:
        t = start + hours * 3600
        if hours * 3600 >= duration:
            break
        i = bisect.bisect_right(times, t)
        if i == 0:
            values.append(None)
        elif i == len(points):
            last_t, last_s = points[-1]
            values.append(int(last_s) if t - last_t <= STALE_SECONDS and last_s > 0 else None)
        else:
            (t0, s0), (t1, s1) = points[i - 1], points[i]
            if t1 - t0 > MAX_GAP_SECONDS:
                values.append(None)
                continue
            value = s0 if t1 == t0 else s0 + (s1 - s0) * (t - t0) / (t1 - t0)
            values.append(int(round(value)) if value > 0 else None)
    while values and values[-1] is None:
        values.pop()
    return values


def checkpoint_observations(points, start, duration):
    """Latest real observation at or before each checkpoint, as a live forecast would see it.

    Fitting on interpolated curve values would let a target peek at its next observation."""
    points = sorted(points)
    times = [t for t, _ in points]
    out = []
    for p in CHECKPOINTS:
        t = start + p * duration
        i = bisect.bisect_right(times, t) - 1
        if i < 0 or t - times[i] > STALE_SECONDS or points[i][1] <= 0 or times[i] <= start:
            out.append(None)
        else:
            out.append([times[i] - start, points[i][1]])
    return out


def curve_at(line, elapsed_seconds):
    hours = elapsed_seconds / 3600
    values = line["v"]
    i = bisect.bisect_left(GRID_HOURS, hours)
    if i < len(values) and abs(GRID_HOURS[i] - hours) < 1e-9:
        return values[i]
    if i == 0 or i >= len(values):
        return None
    left, right = values[i - 1], values[i]
    if left is None or right is None:
        return None
    w = (hours - GRID_HOURS[i - 1]) / (GRID_HOURS[i] - GRID_HOURS[i - 1])
    return left + (right - left) * w


def line_end(line):
    return line["s"] + line["d"]


def build_index(lines):
    own, pooled, jp = {}, {}, {}
    for line in lines:
        kind = line.get("t", NORMAL)
        own.setdefault((line["r"], kind, line["k"]), []).append(line)
        pooled.setdefault((kind, line["k"]), []).append(line)
        if kind == NORMAL and line["r"] == "jp":
            jp[(line["e"], line["k"])] = line
    for group in list(own.values()) + list(pooled.values()):
        group.sort(key=line_end)
    jp_events = sorted({e for e, _ in jp})
    return {"own": own, "pooled": pooled, "jp": jp, "jp_events": jp_events}


def jp_heat(index, event, rank, start):
    """Per-day final of the same event in JP relative to its JP predecessors, if it already ended."""
    def per_day(line):
        return math.log(line["f"] / (line["d"] / 86400))

    same = index["jp"].get((event, rank))
    if same is None or line_end(same) >= start:
        return None
    neighbours = [per_day(index["jp"][(e, rank)]) for e in index["jp_events"]
                  if event - CROSS_SERVER_NEIGHBOURS <= e < event and (e, rank) in index["jp"]
                  and line_end(index["jp"][(e, rank)]) < start]
    if len(neighbours) < MIN_EVENTS:
        return None
    return per_day(same) - statistics.median(neighbours)


def growth(donors, elapsed, duration):
    out = []
    for v in donors:
        if elapsed >= v["d"]:
            continue
        at = curve_at(v, elapsed)
        if at:
            out.append(math.log(v["f"] / at) + math.log(duration / v["d"]))
    return out


def pooled_growth(index, kind, rank, start, duration, chapters, elapsed):
    donors = [v for v in index["pooled"].get((kind, rank), ()) if line_end(v) < start]
    same_structure = [v for v in donors if v.get("n") == chapters and abs(v["d"] - duration) < 86400]
    if len({(v["r"], v["e"]) for v in same_structure}) >= MIN_EVENTS:
        donors = same_structure
    return growth(donors[-POOLED_DONORS:], elapsed, duration)


def prior_and_shape(index, kind, region, event, rank, start, duration, condition, units, elapsed, score,
                    beta=0.0, chapters=None):
    pool = [v for v in index["own"].get((region, kind, rank), ()) if line_end(v) < start]
    events = {v["e"] for v in pool}
    debug = {}
    if kind == NORMAL:
        if len(events) < MIN_EVENTS:
            return None, None, debug
        same = [v for v in pool if v["c"] == condition]
        matched = same[-CONDITION_DONORS:] if len({v["e"] for v in same}) >= MIN_EVENTS else pool[-DONORS:]
        log_prior = statistics.median(math.log(v["f"] * duration / v["d"]) for v in matched)
        unit = [v for v in pool if v["c"] == condition and v["u"] == list(units)]
        if condition == "box" and len(unit) >= 2:
            n = min(len(unit), UNIT_DONORS)
            log_unit = statistics.median(math.log(v["f"] * duration / v["d"]) for v in unit[-UNIT_DONORS:])
            log_prior = n / (n + 2) * log_unit + 2 / (n + 2) * log_prior
            debug["opening_blend_unit_events"] = n
        if region != "jp" and beta:
            heat = jp_heat(index, event, rank, start)
            if heat is not None:
                log_prior += beta * heat
                debug["opening_blend_cross_server_heat"] = heat
        donors = matched
    else:
        if len(events) < (1 if kind == CHAPTER else 2) or len(pool) < (MIN_EVENTS if kind == CHAPTER else 2):
            return None, None, debug
        donors = pool[-DONORS:]
        log_prior = statistics.median(math.log(v["f"] * duration / v["d"]) for v in donors)
        same_event = [v for v in pool if v["e"] == event] if kind == CHAPTER else []
        if same_event:
            # Chapters already finished in this event carry its heat.
            log_same = statistics.median(math.log(v["f"] * duration / v["d"]) for v in same_event)
            weight = len(same_event) / (len(same_event) + 1)
            log_prior = weight * log_same + (1 - weight) * log_prior
            debug["opening_blend_same_event_chapters"] = len(same_event)
    debug["opening_blend_prior_events"] = len({v["e"] for v in donors})
    shape = None
    if score and score > 0:
        rates = growth(donors, elapsed, duration)
        if len(rates) < MIN_EVENTS and kind != NORMAL:
            rates = pooled_growth(index, kind, rank, start, duration, chapters, elapsed)
            debug["opening_blend_shape_pooled"] = True
        if len(rates) >= MIN_EVENTS:
            shape = score * math.exp(statistics.median(rates))
        debug["opening_blend_shape_events"] = len(rates)
    return math.exp(log_prior), shape, debug


def lad_fit(x, y, iterations=30):
    """Least absolute deviations for y = a + b x via IRLS."""
    design = np.vstack([np.ones_like(x), x]).T
    weights = np.ones_like(x)
    solution = np.zeros(2)
    for _ in range(iterations):
        solution = np.linalg.lstsq(design * weights[:, None], y * weights, rcond=None)[0]
        weights = 1 / np.sqrt(np.maximum(np.abs(y - design @ solution), 1e-3))
    return float(solution[0]), float(solution[1])


def normal_targets(lines, before):
    for line in lines:
        if line.get("t", NORMAL) != NORMAL or (before is not None and line_end(line) >= before):
            continue
        yield line


def fit_cross_server_beta(lines, before=None, min_rows=30):
    """Slope of the normal prior residual on the JP same-event heat (non-JP lines)."""
    index = build_index(lines)
    rows = []
    for line in normal_targets(lines, before):
        if line["r"] == "jp":
            continue
        heat = jp_heat(index, line["e"], line["k"], line["s"])
        if heat is None:
            continue
        prior, _, _ = prior_and_shape(index, NORMAL, line["r"], line["e"], line["k"], line["s"], line["d"],
                                      line["c"], line["u"], 0, None)
        if prior is not None:
            rows.append((heat, math.log(line["f"] / prior)))
    if len(rows) < min_rows:
        return 0.0
    _, slope = lad_fit(np.array([a for a, _ in rows]), np.clip(np.array([b for _, b in rows]), -1.2, 1.2))
    return min(1.0, max(0.0, slope))


def fit_coefficients(lines, before=None, beta=0.0, min_rows=40):
    """Per-checkpoint intercept/slope for normal events, pooled over regions and ranks.

    `before` restricts targets to events that ended before that timestamp (replay)."""
    index = build_index(lines)
    samples = {p: [] for p in CHECKPOINTS}
    for line in normal_targets(lines, before):
        for p, observation in zip(CHECKPOINTS, line.get("o") or [None] * len(CHECKPOINTS)):
            if observation is None:
                continue
            elapsed, score = observation
            prior, shape, _ = prior_and_shape(index, NORMAL, line["r"], line["e"], line["k"], line["s"], line["d"],
                                              line["c"], line["u"], elapsed, score, beta)
            if prior is None or shape is None:
                continue
            samples[p].append((math.log(shape / prior), math.log(line["f"] / prior)))
    coefficients = {}
    for p, rows in samples.items():
        if len(rows) < min_rows:
            continue
        x = np.array([a for a, _ in rows])
        y = np.clip(np.array([b for _, b in rows]), -1.2, 1.2)
        a, b = lad_fit(x, y)
        coefficients[str(p)] = [a, min(1.0, max(0.0, b)), len(rows)]
    return coefficients


def coefficient_at(coefficients, progress):
    points = sorted((float(p), v) for p, v in coefficients.items())
    if not points:
        return None
    if progress <= points[0][0]:
        return points[0][1][:2]
    if progress >= points[-1][0]:
        return points[-1][1][:2]
    for (p0, v0), (p1, v1) in zip(points, points[1:]):
        if p0 <= progress <= p1:
            w = (math.log(progress) - math.log(p0)) / (math.log(p1) - math.log(p0))
            return [v0[0] + (v1[0] - v0[0]) * w, v0[1] + (v1[1] - v0[1]) * w]
    return points[-1][1][:2]


def shape_weight(kind, progress):
    """World Link weight on shape vs prior; normal events use fitted coefficients instead."""
    if kind == CHAPTER:
        return min(1.0, 0.5 + 0.5 * progress / 0.35)
    if progress <= 0.10:
        return 0.5
    return min(1.0, 0.5 + 0.5 * (progress - 0.10) / 0.40)


def handoff_weight(progress, region=None, kind=NORMAL):
    start, end = HANDOFF[kind]
    cap = REGION_WEIGHT_CAP.get(region, 1.0) if kind == NORMAL else 1.0
    if progress <= start:
        return cap
    if progress >= end:
        return 0.0
    return cap * (end - progress) / (end - start)


def own_signal_weight(progress):
    """Weight of a normal line's own shape-vs-prior signal against the board median.

    The first hours of a single line are noisy; the board-wide signal is steadier until
    ranks start to diverge."""
    if progress <= 0.015:
        return 0.3
    if progress >= 0.03:
        return 1.0
    return 0.3 + 0.7 * (progress - 0.015) / 0.015


def combine(artifact, kind, progress, prior, shape, board_signal=None):
    if shape is None:
        return prior
    signal = math.log(shape / prior)
    if kind == NORMAL:
        coefficient = coefficient_at(artifact.get("coefficients", {}), progress)
        if coefficient is None:
            return prior
        if board_signal is not None:
            own = own_signal_weight(progress)
            signal = own * signal + (1 - own) * board_signal
        a, b = coefficient
        return prior * math.exp(a + b * signal)
    w = shape_weight(kind, progress)
    return prior * math.exp(w * signal)


def estimate(artifact, index, kind, region, event, rank, start, duration, condition, units, elapsed, score,
             chapters=None, board_signal=None):
    prior, shape, debug = prior_and_shape(index, kind, region, event, rank, start, duration, condition, units,
                                          elapsed, score, artifact.get("cross_server_beta", 0.0), chapters)
    if prior is None:
        return None, debug
    debug.update(opening_blend_prior=prior, opening_blend_shape=shape)
    return combine(artifact, kind, elapsed / duration, prior, shape, board_signal), debug


def row_kind(row, chapters):
    series = row.get("series_type")
    if series == "normal":
        return NORMAL
    if chapters <= 1:
        return None  # finale: a single board with no earlier counterpart yet
    if series == "world_bloom" and int(row.get("character_id") or 0):
        return CHAPTER
    if series == "world_bloom_total":
        return TOTAL
    return None


def apply(artifact, rows, region, event_id, window, condition, units, now, chapters=0):
    """Blend the estimate into rows in place; returns the number of directly estimated lines.

    `window(character_id)` gives the board's (start, end)."""
    index = artifact.get("_index") or build_index(artifact["lines"])
    artifact["_index"] = index
    beta = artifact.get("cross_server_beta", 0.0)
    factors = {}
    pending = []
    candidates = []
    for row in rows:
        if row.get("event_finished_prediction"):
            continue
        kind = row_kind(row, chapters)
        if kind is None:
            continue
        char = int(row.get("character_id") or 0)
        start, end = window(char)
        current, observed = float(row.get("current_score") or 0), int(row.get("current_timestamp") or 0)
        elapsed = observed - start
        if elapsed <= 0 or now - observed > APPLY_MAX_AGE_SECONDS or current <= 0 or end <= start:
            continue
        weight = handoff_weight(elapsed / (end - start), region, kind)
        if weight <= 0:
            continue
        if row.get("synthetic_rank"):
            pending.append((row, char))
            continue
        prior, shape, debug = prior_and_shape(index, kind, region, event_id, int(row["rank"]), start, end - start,
                                              condition, units, elapsed, current, beta, chapters)
        row.update(debug)
        if prior is None:
            pending.append((row, char))
            continue
        row.update(opening_blend_prior=prior, opening_blend_shape=shape)
        candidates.append((row, char, kind, weight, (elapsed / (end - start)), prior, shape))
    signals = {}
    for row, char, kind, _, _, prior, shape in candidates:
        if kind == NORMAL and shape is not None:
            signals.setdefault(char, []).append(math.log(shape / prior))
    for row, char, kind, weight, progress, prior, shape in candidates:
        board = statistics.median(signals[char]) if signals.get(char) else None
        value = combine(artifact, kind, progress, prior, shape, board)
        current = float(row["current_score"])
        chain = max(float(row["model_prediction"]), current, 1.0)
        final = max(current, math.exp(weight * math.log(max(value, 1.0)) + (1 - weight) * math.log(chain)))
        row.update(opening_blend_version=VERSION, opening_blend_prediction=value,
                   opening_blend_weight=weight, opening_blend_chain_prediction=chain)
        factors.setdefault(char, {})[int(row["rank"])] = math.log(final / chain)
        row["model_prediction"] = row["pressure_adjusted_prediction"] = final
    # Ranks without enough history (e.g. 40, synthetic ranks) follow their neighbours.
    for row, char in pending:
        known = factors.get(char)
        if not known:
            continue
        rank = int(row["rank"])
        ranks = sorted(known)
        i = bisect.bisect_left(ranks, rank)
        if i == 0:
            factor = known[ranks[0]]
        elif i == len(ranks):
            factor = known[ranks[-1]]
        else:
            r0, r1 = ranks[i - 1], ranks[i]
            w = (math.log(rank) - math.log(r0)) / (math.log(r1) - math.log(r0))
            factor = known[r0] + (known[r1] - known[r0]) * w
        current = float(row.get("current_score") or 0)
        chain = max(float(row["model_prediction"]), current, 1.0)
        final = max(current, chain * math.exp(factor))
        row.update(opening_blend_version=VERSION, opening_blend_interpolated=True,
                   opening_blend_chain_prediction=chain)
        row["model_prediction"] = row["pressure_adjusted_prediction"] = final
    reconcile(rows, set(factors))
    for row in rows:
        if row.get("opening_blend_version"):
            # The chain's model interval does not describe this estimate.
            for key in ("prediction_p10", "prediction_p50", "prediction_p90"):
                row[key] = None
            hours = max(0.0, (float(row.get("end_timestamp") or 0) - float(row.get("current_timestamp") or 0)) / 3600)
            row["forecast_required_pace"] = ((row["model_prediction"] - float(row.get("current_score") or 0)) / hours
                                             if hours else None)
    return sum(len(v) for v in factors.values())


def reconcile(rows, characters):
    """A better rank never ends below a worse one. No caps: the chain applied its own before."""
    for char in characters:
        floor = 0.0
        group = [r for r in rows if int(r.get("character_id") or 0) == char and not r.get("event_finished_prediction")]
        for row in sorted(group, key=lambda r: int(r["rank"]), reverse=True):
            value = float(row["model_prediction"])
            if value < floor:
                row["model_prediction"] = row["pressure_adjusted_prediction"] = floor
                row["opening_blend_rank_floor"] = floor
            floor = max(floor, float(row["model_prediction"]))


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_atomic(path, artifact):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, separators=(",", ":"))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
