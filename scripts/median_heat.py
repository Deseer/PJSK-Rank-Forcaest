"""A heat index of one means median historical score scale at the same shape phase."""
import bisect
import math
import statistics


def fraction(curve, progress):
    if progress >= 1:return 1.
    if progress <= 0:return 0.
    # Do not invent the missing opening of a partially recorded historical event.
    if progress < curve[0][0]:return None
    points=curve if curve[-1][0]>=1 else [*curve,(1.,1.)]
    i=bisect.bisect_right([p[0] for p in points],progress)-1
    if i==len(points)-1:return points[i][1]
    a,b=points[i],points[i+1]
    return a[1]+(b[1]-a[1])*(progress-a[0])/(b[0]-a[0])


def quantile(values,q):
    values=sorted(values);p=q*(len(values)-1);i=int(p);j=min(i+1,len(values)-1)
    return values[i]+(values[j]-values[i])*(p-i)


def reference(shapes,finals,region,event,rank,start,condition,prior_quantile=.5,
              contexts=None,event_meta=None,unit_match=False,duration_power=0.):
    eligible=[r for r in shapes if r['region']==region and r['series_type']=='normal' and r['rank']==rank
              and r['event_id']<event and r['end_timestamp']<start
              and (region,r['event_id'],rank) in finals]
    matching=[r for r in eligible if r.get('condition')==condition]
    if len({r['event_id'] for r in matching})>=3:eligible=matching
    if unit_match and contexts:
        current_context=contexts.get((region,event))
        same_units=[r for r in eligible if current_context and contexts.get((region,r['event_id']))
                    and contexts[(region,r['event_id'])].unit_labels==current_context.unit_labels]
        if len(same_units)>=3:eligible=same_units
    eligible=sorted(eligible,key=lambda r:r['event_id'])[-8:]
    if len(eligible)<3:return None
    logs=[]
    current_meta=event_meta.get((region,event)) if event_meta else None
    for r in eligible:
        value=finals[(region,r['event_id'],rank)]
        historical_meta=event_meta.get((region,r['event_id'])) if event_meta else None
        if current_meta and historical_meta and duration_power:
            ratio=(current_meta.end_timestamp-current_meta.start_timestamp)/(historical_meta.end_timestamp-historical_meta.start_timestamp)
            value*=ratio**duration_power
        logs.append(math.log(value))
    median=math.exp(statistics.median(logs));prior=math.exp(quantile(logs,prior_quantile))
    return dict(donors=eligible,median=median,prior_heat=prior/median,count=len(eligible),
                donor_scores={r['event_id']:math.exp(value) for r,value in zip(eligible,logs)})


def score_at(points,timestamp):
    times=[p[0] for p in points];i=bisect.bisect_right(times,timestamp)-1
    if i<0 or timestamp-times[i]>1800:return None
    return points[i][1]


def observed_windows(points,start,end,mode):
    """Windows end at the latest available point, never beyond the forecast input."""
    now=points[-1][0];age=now-start;windows=[]
    if mode=='daily':
        # Consecutive non-overlapping activity days; no fictitious full first day.
        boundaries=[start+i*86400 for i in range(int(age//86400)+1)]
        if boundaries[-1]<now:boundaries.append(now)
        windows=list(zip(boundaries,boundaries[1:]))[-4:]
    else:
        # Shape slope evidence at several elapsed-time scales.
        for hours in [6,24,48]:
            if age>=hours*3600:windows.append((now-hours*3600,now))
        if not windows and age>=3600:windows=[(start,now)]
    result=[]
    for a,b in windows:
        if b-a<3600:continue
        sa=0 if a==start and points[0][0]-start<=120 else score_at(points,a)
        sb=score_at(points,b)
        if sa is None or sb is None or sb<sa:continue
        result.append(((a-start)/(end-start),(b-start)/(end-start),sb-sa,(b-a)/3600))
    return result


def estimate(points,start,end,ref,mode='daily',tau_days=1.,adaptive=False):
    now,current=points[-1];progress=(now-start)/(end-start)
    if not ref or not 0<=progress<1:return None,{}
    windows=observed_windows(points,start,end,mode)
    scored=[]
    for donor in ref['donors']:
        f=fraction(donor['curve'],progress)
        if f is None:continue
        ratios=[]
        for a,b,delta,hours in windows:
            if delta <= 0:continue
            fa=fraction(donor['curve'],a);fb=fraction(donor['curve'],b)
            if fa is None or fb is None or fb-fa<=1e-5:continue
            ratios.append(math.log(delta/(ref['median']*(fb-fa))))
        logheat=statistics.median(ratios) if ratios else math.log(ref['prior_heat'])
        dispersion=statistics.median(abs(r-logheat) for r in ratios) if len(ratios)>1 else 0.
        scored.append((dispersion,logheat,f,len(ratios),donor['event_id']))
    if not scored:
        return max(current,ref['median']*ref['prior_heat']),dict(heat=ref['prior_heat'],reliability=0.,source='opening_prior')
    if adaptive and len(windows)>=2:
        scored=sorted(scored)[:max(3,len(scored)//2)]
    logheat=statistics.median(s[1] for s in scored)
    # Exposure counts distinct elapsed hours, rather than counting overlapping windows as independent days.
    exposure=min((now-start)/86400,4.)
    reliability=exposure/(exposure+tau_days) if windows else 0.
    logheat=(1-reliability)*math.log(ref['prior_heat'])+reliability*max(-1.1,min(1.1,logheat))
    heat=math.exp(logheat);f=statistics.median(s[2] for s in scored)
    # Forecast remaining growth; observed score is not rescaled by heat a second time.
    target=max(current,current+heat*ref['median']*max(0.,1-f))
    return target,dict(heat=heat,median_final=ref['median'],prior_heat=ref['prior_heat'],
                       reliability=reliability,windows=len(windows),used_ratios=statistics.median(s[3] for s in scored),reference_count=ref['count'],
                       donor_events=[s[4] for s in scored],shape_fraction=f,source=mode)


def posterior(points,start,end,ref,prior_hours=2.,pace_weight=0.,pace_mode='daily'):
    """Separate score scale from slope residual, and shrink uncertain openings to a prior."""
    now,current=points[-1];progress=(now-start)/(end-start)
    if not ref or not 0<=progress<1:return None,{}
    available=[fraction(r['curve'],progress) for r in ref['donors']]
    available=[f for f in available if f is not None and f>0]
    prior_score=ref['median']*ref['prior_heat']
    if current <= 0:
        return prior_score,dict(heat=ref['prior_heat'],source='no_score_prior',reliability=0.)
    if not available:return max(current,prior_score),dict(heat=ref['prior_heat'],source='prior',reliability=0.)
    f=statistics.median(available);cumulative_heat=current/max(ref['median']*f,1.)
    age_hours=max(0.,(now-start)/3600)
    opening_reliability=age_hours/(age_hours+prior_hours)
    logheat=(1-opening_reliability)*math.log(ref['prior_heat'])+opening_reliability*math.log(max(cumulative_heat,1e-6))
    pace,debug=estimate(points,start,end,ref,pace_mode,.5,False)
    if debug.get('used_ratios', 0) > 0:
        weight=pace_weight*min(1.,age_hours/48.)
        # A slope residual adjusts remaining growth, never scales already observed points.
        deviation=max(-.5,min(.5,math.log(debug['heat']/max(cumulative_heat,1e-6))))
        logheat+=weight*deviation
    heat=math.exp(logheat)
    value=max(current,current+heat*ref['median']*(1-f))
    return value,dict(heat=heat,cumulative_heat=cumulative_heat,pace_heat=debug.get('heat'),
                      median_final=ref['median'],prior_heat=ref['prior_heat'],reliability=opening_reliability,
                      reference_count=ref['count'],windows=debug.get('windows',0),shape_fraction=f,
                      source='posterior',prior_hours=prior_hours,pace_weight=pace_weight)


def speed_heat_index(points,start,end,ref,mode="daily"):
    """Observed growth divided by the median matched historical growth in the same window."""
    if not ref:return {}
    ratios=[];details=[]
    for a,b,delta,hours in observed_windows(points,start,end,mode):
        expected=[]
        for donor in ref['donors']:
            fa=fraction(donor['curve'],a);fb=fraction(donor['curve'],b)
            if fa is None or fb is None or fb<=fa:continue
            score=ref.get('donor_scores',{}).get(donor['event_id'],ref['median'])
            expected.append(score*(fb-fa))
        if len(expected)<3:continue
        median_growth=statistics.median(expected)
        ratio=delta/median_growth
        ratios.append(ratio)
        details.append(dict(hours=hours,observed_daily_score=delta/hours*24,
                            reference_daily_score=median_growth/hours*24,heat=ratio))
    return dict(heat=statistics.median(ratios),windows=details) if ratios else {}
