"""Rank-score profile diagnostics. Missing ranks are never interpreted as zeros.

Threshold counts include observed LOWER BOUNDS: a positive score at rank r
proves at least r scored entries. Upper bounds require synchronized snapshots;
stale below-threshold observations cannot bound the current player count.
"""
import math
import statistics

PAIRS=((100,1000),(1000,5000),(5000,10000),(10000,50000),(50000,100000),(100000,200000))
THRESHOLDS=(1,10000,100000,500000,1000000)


def threshold_bounds(scores,threshold):
    if not scores:return None,None
    above=[rank for rank,score in scores.items() if score>=threshold]
    below=[rank for rank,score in scores.items() if score<threshold]
    lower=max(above,default=0)
    upper=min(below,default=None)
    if upper is not None and upper<=lower:
        return None,None  # asynchronous or inconsistent rank ordering
    return lower,upper


def gaps(scores):
    result=[]
    for a,b in PAIRS:
        sa,sb=scores.get(a),scores.get(b)
        if sa is None or sb is None or sa<sb:
            result += [math.nan]*2
            continue
        result += [(sa-sb)/max(sa,1),math.log((sa+1)/(sb+1))/math.log(b/a)]
    return result


def profile_heat(scores,reference):
    bands=(lambda r:100<=r<=1000,lambda r:5000<=r<=200000)
    heat=[]
    for band in bands:
        values=[math.log((score+1)/(reference[rank]+1)) for rank,score in scores.items()
                if rank in reference and band(rank) and reference[rank]>0]
        heat.append(statistics.median(values) if len(values)>=2 else math.nan)
    return heat+[heat[0]-heat[1]]


def breadth(scores,reference):
    result=[]
    for threshold in THRESHOLDS:
        lo,hi=threshold_bounds(scores,threshold)
        ref_lo,ref_hi=threshold_bounds(reference,threshold)
        result += [math.log1p(lo) if lo is not None else math.nan,
                   math.log((lo+1)/(ref_lo+1)) if lo is not None and ref_lo is not None else math.nan,
                   float(hi is None) if lo is not None else math.nan]
    return result


def filling(scores,previous,hours):
    if hours is None or hours<=0:return [math.nan]*(len(THRESHOLDS)+1)
    common=set(scores)&set(previous)
    if len(common)<3:return [math.nan]*(len(THRESHOLDS)+1)
    now={r:scores[r] for r in common};before={r:previous[r] for r in common}
    result=[]
    for threshold in THRESHOLDS:
        a,_=threshold_bounds(now,threshold);b,_=threshold_bounds(before,threshold)
        result.append((math.log1p(a)-math.log1p(b))/hours if a is not None and b is not None else math.nan)
    rates=[math.log((now[r]+1)/(before[r]+1))/hours for r in common if r>=5000 and now[r]>=before[r]]
    return result+[statistics.median(rates) if len(rates)>=2 else math.nan]


def diagnose(scores,reference,previous=None,hours=None):
    def clean(value):
        if isinstance(value,float) and not math.isfinite(value):return None
        if isinstance(value,list):return [clean(v) for v in value]
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
        return value
    return clean(dict(threshold_counts={str(t):dict(zip(('lower','upper'),threshold_bounds(scores,t))) for t in THRESHOLDS},
                count_scope='provided_snapshot_only',actual_participant_count_identified=False,
                highest_observed_positive_rank=max((r for r,s in scores.items() if s>0),default=None),
                competition_and_breadth_log_heat=profile_heat(scores,reference),
                gap_features=gaps(scores),breadth_features=breadth(scores,reference),
                filling_features=filling(scores,previous or {},hours)))
