"""Prefix-only ranking-segment heat; log indices are relative to past profiles."""
import math
import statistics

BANDS=((1,100),(101,1000),(1001,10000),(10001,50000),(50001,None))
LABELS=('top100','101_1000','1001_10000','10001_50000','50001_plus')


def band_index(rank):
    if rank<1:raise ValueError('Rank must be positive')
    return next(i for i,(lo,hi) in enumerate(BANDS) if rank>=lo and (hi is None or rank<=hi))


def summarize(scores,reference,previous=None,expected_deltas=None,shrinkage=3.):
    previous=previous or {};expected_deltas=expected_deltas or {}
    result=[]
    for index,(lo,hi) in enumerate(BANDS):
        ranks=[r for r in scores if lo<=r and (hi is None or r<=hi) and scores[r]>=0]
        levels=[math.log((scores[r]+1)/(reference[r]+1)) for r in ranks if r in reference and reference[r]>0]
        # Insufficient support stays unknown; one unusual competitor is not a segment.
        support=len(levels);reliability=support/(support+shrinkage) if support>=2 else 0.
        level=statistics.median(levels) if support>=2 else math.nan
        spread=statistics.median(abs(v-level) for v in levels) if support>=2 else math.nan
        rates=[math.log((max(0,scores[r]-previous[r])+1)/(expected_deltas[r]+1)) for r in ranks
               if r in previous and r in expected_deltas and scores[r]>=previous[r] and expected_deltas[r]>=0]
        pace=statistics.median(rates) if len(rates)>=2 else math.nan
        slopes=[]
        ordered=sorted(ranks)
        for a,b in zip(ordered,ordered[1:]):
            if scores[a]>=scores[b]:slopes.append(math.log((scores[a]+1)/(scores[b]+1))/math.log(b/a))
        slope=statistics.median(slopes) if slopes else math.nan
        common=[r for r in ranks if r in previous]
        fill=[]
        for threshold in (10000,100000,1000000):
            if len(common)<2:
                fill.append(math.nan)
                continue
            # Change in the observed within-segment threshold-count lower bound.
            # Both endpoints have the same rank coverage; the caller uses a 1h window.
            now=max((r-lo+1 for r in common if scores[r]>=threshold),default=0)
            before=max((r-lo+1 for r in common if previous[r]>=threshold),default=0)
            fill.append(math.log1p(now)-math.log1p(before))
        result.append(dict(label=LABELS[index],log_heat=level,shrunk_log_heat=level*reliability,
                           support=support,reliability=reliability,dispersion=spread,
                           log_pace_heat=pace,pace_support=len(rates),rank_score_slope=slope,
                           threshold_fill_log_change=fill))
    return result


def local_features(segments,rank,neighbors=False,global_context=False):
    index=band_index(rank);s=segments[index]
    features=[s[k] for k in ('shrunk_log_heat','reliability','dispersion','log_pace_heat','pace_support','rank_score_slope')]
    features += s['threshold_fill_log_change']
    features += [float(index==i) for i in range(len(BANDS))]
    if neighbors:
        for neighbor in (index-1,index+1):
            other=segments[neighbor] if 0<=neighbor<len(segments) else None
            features += [other['shrunk_log_heat'] if other else math.nan,
                         s['shrunk_log_heat']-other['shrunk_log_heat'] if other else math.nan,
                         other['log_pace_heat'] if other else math.nan]
    if global_context:
        features += [s['shrunk_log_heat'] for s in segments]+[s['log_pace_heat'] for s in segments]
    return features


def contrast_features(segments,rank):
    index=band_index(rank);s=segments[index]
    available=[v['log_heat'] for v in segments if math.isfinite(v['log_heat'])]
    baseline=statistics.median(available) if available else math.nan
    result=[(s['log_heat']-baseline)*s['reliability'],
            s['log_pace_heat']-s['log_heat'],s['reliability'],s['pace_support']]
    for neighbor in (index-1,index+1):
        result.append(s['log_heat']-segments[neighbor]['log_heat'] if 0<=neighbor<len(segments) else math.nan)
    return result
