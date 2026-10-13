"""Causal median-shape residual model with a shared training/live feature path."""
import bisect
import collections
import math
import statistics
from datetime import datetime,timezone
import numpy as np

CHECKPOINTS=(.005,.015,.05,.1,.25,.5,.75,.9)
WINDOWS=(1,6,24,48)
REGIONS=('jp','en','tw','kr','cn')
UNITS=('light_sound','idol','street','theme_park','school_refusal')
OFFSETS={'jp':9,'en':0,'tw':8,'kr':9,'cn':8}


def reference_observation(line,boundary):
    grid=line['boundaries'];i=bisect.bisect_left(grid,boundary)
    if i<len(grid) and abs(grid[i]-boundary)<2:
        value=line['observed'][i]
        return value if value and grid[i]-value[0]<=1800 else None
    if i and abs(grid[i-1]-boundary)<2:
        value=line['observed'][i-1]
        return value if value and grid[i-1]-value[0]<=1800 else None
    # Completed historical curves can interpolate. Never interpolate live prefixes.
    if not 0<i<len(grid):return None
    a,b=line['observed'][i-1],line['observed'][i]
    if not a or not b or grid[i-1]-a[0]>1800 or grid[i]-b[0]>1800:return None
    w=(boundary-grid[i-1])/(grid[i]-grid[i-1])
    return boundary,a[1]*(1-w)+b[1]*w


def live_observation(points,boundary):
    i=bisect.bisect_right(points,(boundary,float('inf')))-1
    value=points[i] if i>=0 else None
    return value if value and boundary-value[0]<=1800 else None


def prepare(lines,contexts,assets):
    by_rank=collections.defaultdict(list)
    for line in lines.values():by_rank[(line['region'],line['rank'])].append(line)
    for group in by_rank.values():group.sort(key=lambda v:v['end'])
    def prior(line):
        donors=[v for v in by_rank[(line['region'],line['rank'])] if v['end']<line['start']]
        same=[v for v in donors if v['condition']==line['condition']]
        if len(same)>=3:donors=same
        latest=donors[-8:];unit=[v for v in donors if v['units']==line['units']]
        matched=unit[-6:] if len(unit)>=3 else latest
        def summary(group):
            values=[math.log1p(v['final'][1]) for v in group]
            return [float(np.quantile(values,q)) if values else math.nan for q in (.25,.5,.75)]
        return summary(latest)+summary(matched)+[len(latest),len(matched)]
    priors={key:prior(v) for key,v in lines.items()}
    return dict(reference_lines=lines,contexts=contexts,assets=assets,priors=priors)


def features(payload,line,now,current,cross_scores,earlier):
    t,score=current
    if score<=0 or now-t>1800:return None
    duration=line['end']-line['start'];p=(t-line['start'])/duration
    wall=(now-line['start'])/duration
    for checkpoint in CHECKPOINTS:
        if abs(wall-checkpoint)<2/duration:wall=checkpoint;break
    if p<=0:return None
    lines=payload['reference_lines'];contexts=payload['contexts'];assets=payload['assets']
    context=contexts.get((line['region'],line['event']),{})
    donors=[v for v in lines.values() if v['region']==line['region'] and v['rank']==line['rank'] and v['end']<line['start']]
    donors.sort(key=lambda v:v['end'])
    same=[v for v in donors if v['condition']==line['condition']]
    if len(same)>=3:donors=same
    latest=donors[-8:];unit=[v for v in donors if v['units']==line['units']]
    matched=unit[-6:] if len(unit)>=3 else latest
    def summary(group):
        values=[math.log1p(v['final'][1]) for v in group]
        return [float(np.quantile(values,q)) if values else math.nan for q in (.25,.5,.75)]
    prior=summary(latest)+summary(matched)+[len(latest),len(matched)]
    if len(latest)<3:return None
    logs=math.log1p(score)
    x=[float(line['region']==r) for r in REGIONS]+[float(line['condition']==c) for c in ('box','mixed','other')]
    x += [float(u in line['units']) for u in UNITS]
    x += [float(c in context.get('main_characters',())) for c in range(1,27)]
    local=datetime.fromtimestamp(line['start']+OFFSETS[line['region']]*3600,timezone.utc)
    x += [float(local.weekday()),float(local.hour)]
    x += [math.log10(line['rank']),p,duration/3600,p*duration/3600,(1-p)*duration/3600,logs,logs-math.log(p),(now-t)/3600]
    x += [v-logs if i<6 else v for i,v in enumerate(prior)]
    for h in WINDOWS:
        before=earlier.get(h)
        if before and before[0]<t:
            dt=(t-before[0])/3600
            x += [max(0,score-before[1])/score/dt,(logs-math.log1p(before[1]))/dt,1.]
        else:x += [math.nan,math.nan,0.]
    x += [math.log1p(cross_scores[r])-logs if r in cross_scores else math.nan for r in (10,100,1000,10000,100000)]
    fractions=[];donor_now={}
    for v in latest:
        value=reference_observation(v,int(v['start']+wall*(v['end']-v['start'])))
        if value and value[1]>0:
            fractions.append(value[1]/v['final'][1]);donor_now[(v['region'],v['event'],v['rank'])]=value
    if len(fractions)>=3:
        fraction_logs=np.log(fractions)
        x += [-float(q) for q in np.quantile(fraction_logs,(.25,.5,.75))]+[float(np.std(fraction_logs)),len(fractions)]
    else:x += [math.nan]*4+[len(fractions)]
    asset=assets.get((line['region'],line['event']))
    foreign=[(key,v) for key,v in lines.items() if v['region']!=line['region'] and v['rank']==line['rank'] and v['end']<line['start']
             and asset and assets.get((v['region'],v['event']))==asset and payload['priors'][key][-2]>=3]
    heats=[math.log1p(v['final'][1])-payload['priors'][key][1] for key,v in foreign]
    x += [float(np.median(heats)) if heats else math.nan,float(np.std(heats)) if heats else math.nan,len(heats)]
    ratios=[]
    for h,column in zip(WINDOWS,(57,60,63,66)):
        expected=[]
        for v in latest:
            value=donor_now.get((v['region'],v['event'],v['rank']))
            cutoff=int(v['start']+wall*(v['end']-v['start']))
            before=reference_observation(v,cutoff-h*3600)
            if value and before and value[0]>before[0] and value[1]>before[1]:
                expected.append((value[1]-before[1])/value[1]/((value[0]-before[0])/3600))
        ratios.append(math.log(max(x[column],1e-8)/np.median(expected)) if len(expected)>=3 and math.isfinite(x[column]) else math.nan)
    heat=-x[50]+x[75] if math.isfinite(x[75]) else math.nan
    x += ratios+[ratios[1]-ratios[2],heat]
    return x


def predict_rows(payload,region,event,start,end,now,history,rows):
    context=payload['contexts'].get((region,event),{})
    cross={}
    for (char,rank),points in history.items():
        if char==0:
            value=live_observation(points,now)
            if value:cross[rank]=value[1]
    vectors=[];selected=[]
    for row in rows:
        if row.get('synthetic_rank') or row.get('event_finished_prediction') or row.get('character_id')!=0:continue
        points=history.get((0,int(row['rank'])),[]);current=live_observation(points,now)
        if not current:continue
        wall=(now-start)/(end-start)
        if not .005-2/(end-start)<=wall<=.90+2/(end-start):continue
        line=dict(region=region,event=event,rank=int(row['rank']),start=start,end=end,
                  condition=context.get('condition','other'),units=context.get('units',[]))
        earlier={h:live_observation(points,now-h*3600) for h in WINDOWS}
        vector=features(payload,line,now,current,cross,earlier)
        if vector is not None:vectors.append(vector);selected.append(row)
    if not vectors:return 0
    matrix=np.asarray(vectors);base=np.where(np.isfinite(matrix[:,75]),matrix[:,75],0.)
    growth=np.clip(payload['model'].predict(matrix)+base,0,12)
    for row,g in zip(selected,growth):
        value=max(float(row['current_score']),float(row['current_score'])*math.exp(float(g)))
        row['previous_chain_prediction']=row['model_prediction']
        row['model_prediction']=row['pressure_adjusted_prediction']=value
        row['shape_live_prediction']=value
        row['shape_live_version']=payload['version']
        for key in ('prediction_p10','prediction_p50','prediction_p90'):row[key]=None
    return len(selected)


def segment_diagnostics(payload,region,event,start,end,now,history):
    from segmented_heat import summarize
    context=payload['contexts'].get((region,event),{})
    scores={};previous={};reference={};deltas={};wall=(now-start)/(end-start)
    for (char,rank),points in history.items():
        if char!=0:continue
        current=live_observation(points,now);before=live_observation(points,now-3600)
        if not current:continue
        scores[rank]=current[1]
        if before:previous[rank]=before[1]
        donors=[v for v in payload['reference_lines'].values() if v['region']==region and v['rank']==rank and v['end']<start]
        donors.sort(key=lambda v:v['end']);same=[v for v in donors if v['condition']==context.get('condition','other')]
        if len(same)>=3:donors=same
        levels=[];increments=[]
        for v in donors[-8:]:
            cutoff=int(v['start']+wall*(v['end']-v['start']))
            value=reference_observation(v,cutoff);old=reference_observation(v,cutoff-3600)
            if value:levels.append(value[1])
            if value and old and value[1]>=old[1]:increments.append(value[1]-old[1])
        if len(levels)>=3:reference[rank]=float(np.median(levels))
        if len(increments)>=3:deltas[rank]=float(np.median(increments))
    segments=summarize(scores,reference,previous,deltas)
    for s in segments:
        s['heat_index']=math.exp(s['log_heat']) if math.isfinite(s['log_heat']) else None
        s['pace_heat_index']=math.exp(s['log_pace_heat']) if math.isfinite(s['log_pace_heat']) else None
        for key,value in list(s.items()):
            if isinstance(value,float) and not math.isfinite(value):s[key]=None
            elif isinstance(value,list):s[key]=[v if math.isfinite(v) else None for v in value]
    return dict(reference=1.,segments=segments,point_correction_enabled=False)


def reconcile(rows):
    """Closest monotone log-score sequence, with observed-score lower bounds."""
    from sklearn.isotonic import isotonic_regression
    ordered=sorted(rows,key=lambda r:r['rank'])
    scores=np.asarray([max(float(r['model_prediction']),1.) for r in ordered])
    if np.all(scores[:-1]>=scores[1:]) and all(s>=float(r['current_score']) for s,r in zip(scores,ordered)):return
    values=np.exp(isotonic_regression(np.log(scores),increasing=False))
    values=np.maximum(values,[float(r['current_score']) for r in ordered])
    values=np.maximum.accumulate(values[::-1])[::-1]
    for row,value in zip(ordered,values):
        if abs(value-float(row['model_prediction']))>1e-6:
            row['shape_live_rank_reconciled']=True
            row['model_prediction']=row['pressure_adjusted_prediction']=float(value)
