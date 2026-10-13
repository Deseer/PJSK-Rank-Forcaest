"""Causal per-region normalized ranking curves from complete ended tracker lines."""
import bisect
import math
import statistics
from collections import defaultdict
from live_training_history import live_index, read_points

GRID=(.001,.003,.005,.01,.015,.02,.03,.05,.075,.1,.15,.2,.25,.35,.5,.65,.75,.85,.9,.95,.98)


def build_history(module,event_meta):
    history=[]
    contexts=module.load_event_feature_contexts()
    for region in module.REGIONS:
        index=live_index(module,region)
        for path in sorted(set(index.values())):
            event_id=int(path.name.split('_')[1]);meta=event_meta.get((region,event_id))
            if meta is None:continue
            groups=defaultdict(list)
            for char,rank,timestamp,score in read_points(path):
                if rank not in module.COMMON_RANKS:continue
                if index.get((event_id,meta.series_type,char,rank))!=path:continue
                start,end=module.event_line_window(meta,char)
                if timestamp>=start and score>=0:groups[(char,rank)].append((timestamp,score))
            for (char,rank),points in groups.items():
                points=sorted(dict(points).items());final=points[-1][1]
                start,end=module.event_line_window(meta,char)
                if final<=0:continue
                times=[p[0] for p in points];curve=[]
                for progress in GRID:
                    target=start+progress*(end-start);i=bisect.bisect_right(times,target)-1
                    if i<0 or target-times[i]>max(120,(end-start)*.003):continue
                    fraction=points[i][1]/final
                    if 0<fraction<=1.05:curve.append((progress,min(fraction,1.)))
                if curve:
                    context=contexts.get((region,event_id))
                    history.append(dict(region=region,event_id=event_id,character_id=char,rank=rank,
                                        series_type=meta.series_type,end_timestamp=end,
                                        condition=context.condition if context else 'other',curve=curve))
    return history


def curve_fraction(curve,progress):
    grid=[r[0] for r in curve];i=bisect.bisect_right(grid,progress)-1
    if i<0 or progress>grid[-1]:return None
    if i==len(grid)-1:return curve[i][1]
    a,b=curve[i],curve[i+1];weight=(progress-a[0])/(b[0]-a[0])
    return math.exp(math.log(a[1])*(1-weight)+math.log(b[1])*weight)


def projection(history,region,event_id,char,rank,series,progress,start,current,condition='other'):
    if current<=0 or progress<.001 or progress>.98:return None,0
    eligible=[r for r in history if r['region']==region and r['event_id']<event_id and r['end_timestamp']<start
              and r['rank']==rank and r['series_type']==series]
    same=[r for r in eligible if r.get('condition')==condition]
    if len({r['event_id'] for r in same})>=4:eligible=same
    if series=='world_bloom':
        same=[r for r in eligible if r['character_id']==char]
        if len({r['event_id'] for r in same})>=3:eligible=same
    by_event=defaultdict(list)
    for r in eligible:
        value=curve_fraction(r['curve'],progress)
        if value is not None:by_event[r['event_id']].append(value)
    fractions=[statistics.median(by_event[e]) for e in sorted(by_event)[-6:]]
    if len(fractions)<3:return None,len(fractions)
    fraction=math.exp(statistics.median(math.log(f) for f in fractions))
    return max(current,current/fraction),len(fractions)


def adjust(prediction,current,shape,progress,weights):
    if shape is None:return prediction,0.
    if progress<.01:weight=weights[0]
    elif progress<.05:weight=weights[1]
    elif progress<.2:weight=weights[2]
    elif progress<.75:weight=weights[3]
    else:weight=weights[4]
    if .2 <= progress < .75 and prediction > shape * 1.1:
        # A materially higher model estimate gets a stronger, downward shape correction.
        weight = max(weight, .75)
    value=math.exp(math.log(max(prediction,1.))*(1-weight)+math.log(max(shape,1.))*weight)
    return max(current,value),weight


def build_export_history(module,event_meta,data_root,regions=('cn',)):
    """Compact grid samples from older standard exports, without loading full traces."""
    history=[];contexts=module.load_event_feature_contexts()
    for region in regions:
        samples={};finals={}
        for row in module.iter_rank_timeseries_rows(data_root,region,set(module.COMMON_RANKS)):
            event_id=row['event_id'];meta=event_meta.get((region,event_id));char=row['character_id'];rank=row['rank']
            if not meta or not module.has_event_line_window(meta,char):continue
            key=(event_id,char,rank);timestamp=row['timestamp'];score=row['score']
            if key not in finals or timestamp>finals[key][0]:finals[key]=(timestamp,score)
            progress=module.event_line_progress(meta,char,timestamp)
            if progress<=0 or progress>.985 or score<=0:continue
            i=bisect.bisect_left(GRID,progress)
            target=min(GRID[max(0,i-1):min(len(GRID),i+1)],key=lambda p:abs(p-progress))
            distance=abs(target-progress)
            if distance>.003:continue
            curve=samples.setdefault(key,{})
            if target not in curve or distance<curve[target][0]:curve[target]=(distance,progress,score)
        for (event_id,char,rank),curve in samples.items():
            meta=event_meta[(region,event_id)];end=module.event_line_end_timestamp(meta,char);timestamp,score=finals[(event_id,char,rank)]
            if timestamp<end-120 or score<=0:continue
            normalized=sorted((progress,min(value/score,1.)) for gap,progress,value in curve.values() if value>0)
            if normalized:
                context=contexts.get((region,event_id))
                history.append(dict(region=region,event_id=event_id,character_id=char,rank=rank,
                                    series_type=meta.series_type,end_timestamp=end,
                                    condition=context.condition if context else 'other',curve=normalized))
    return history


def merge_histories(export,live):
    merged={(r['region'],r['event_id'],r['character_id'],r['rank'],r['series_type']):r for r in export}
    for row in live:
        key=(row['region'],row['event_id'],row['character_id'],row['rank'],row['series_type'])
        merged[key]=row
    return list(merged.values())
