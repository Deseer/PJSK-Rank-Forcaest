"""Observed hourly maxima and short-horizon live pace corrections, not physics limits."""
import json
import math
import os
import time
import bisect
import fcntl
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from fetch_event_tracker_history import request_json


def history_maxima(path, rows):
    """Causal bootstrap from local trace; never read points after the forecast input."""
    if path is None:return {}
    groups={}
    try:
        with Path(path).open() as f:
            for line in f:
                try:
                    parts=list(map(int,line.strip().split('\t')))
                    if len(parts)==3:rank,ts,score=parts;char=0
                    else:char,rank,ts,score=parts
                    groups.setdefault((char,rank),{})[ts]=score
                except ValueError:continue
    except OSError:return {}
    result={}
    for r in rows:
        key=(r['character_id'],r['rank'])
        points=sorted((t,s) for t,s in groups.get(key,{}).items()
                      if t<=r['current_timestamp'] and 0<=s<=r['current_score'])
        times=[t for t,s in points]
        for window in (900,3600):
            best=0.;stamp=None
            for i,(t,s) in enumerate(points):
                j=bisect.bisect_right(times,t-window)-1
                if j<0:continue
                elapsed=t-times[j];delta=s-points[j][1]
                if not window<=elapsed<=window+120 or delta<0:continue
                speed=delta*3600/elapsed
                if speed>best:best=speed;stamp=t
            if stamp is not None:result[(key,window)]=(best,stamp)
    return result


def collect(base, region, event, rows, state_dir, history_path=None):
    state_dir=Path(state_dir);state_dir.mkdir(parents=True,exist_ok=True)
    groups={}
    for r in rows:
        if not r.get('event_finished_prediction'):
            groups.setdefault(int(r['character_id']),[]).append(r)
    snapshots={}
    def fetch(task):
        char,window=task
        scope=f'world-bloom/{char}' if char else 'total'
        query=urlencode({'ranks':','.join(str(r['rank']) for r in groups[char]),
                         'interval':window,'unitSeconds':3600,'skipMissing':'true'},safe=',')
        url=f'{base.rstrip("/")}/api/v2/cloud/events/{region}/{event}/leaderboards/{scope}/sk/speed?{query}'
        try:
            return char,window,request_json(url,timeout=8,retries=1)
        except Exception as exc:
            return char,window,{'error':type(exc).__name__}
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses=list(pool.map(fetch,[(c,w) for c in groups for w in (900,3600)]))
    for char,window,payload in responses:
        snapshots[(char,window)]=payload
    now=int(time.time())
    bootstrap=history_maxima(history_path,rows)
    for char,group in groups.items():
        path=state_dir/f'{region}_{event}_{char}.json'
        lock=path.with_suffix('.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:state=json.loads(path.read_text())
        except (OSError,ValueError):state={'version':1,'region':region,'event':event,'character':char,'ranks':{}}
        samples={};errors={}
        for window in (900,3600):
            payload=snapshots[(char,window)]
            if 'error' in payload:errors[str(window)]=payload['error']
            meta=payload.get('meta',{})
            expected_scope=f'world-bloom/{char}' if char else 'total'
            if (meta.get('server')!=region or meta.get('eventId')!=event or meta.get('scope')!=expected_scope
                    or payload.get('intervalSeconds')!=window or payload.get('unitSeconds')!=3600):continue
            for item in payload.get('speeds',[]):
                speed=item.get('speed');ts=item.get('timestamp',0)
                if not isinstance(speed,(float,int)) or not math.isfinite(speed) or speed<0:continue
                score=item.get('score')
                if not isinstance(ts,(int,float)) or not math.isfinite(ts):continue
                if not isinstance(score,(int,float)) or not math.isfinite(score) or score<0:continue
                if not 0<=now-ts<=300:continue
                samples[(item['rank'],window)]={'speed':speed,'timestamp':ts,'score':score,'fetched_at':now}
        for r in group:
            key=str(r['rank']);record=state['ranks'].setdefault(key,{})
            hour=samples.get((r['rank'],3600));short=samples.get((r['rank'],900))
            # A negative correction, future point or outlying isolated short burst is not a new record.
            for window,sample in [(3600,hour),(900,short)]:
                if sample is None:continue
                if sample['score']<r['current_score']:continue
                if sample['timestamp']<r['current_timestamp']:continue
                if window==900 and hour and sample['speed']>max(hour['speed'],1)*2.5:continue
                elapsed=(r['end_timestamp']-r['current_timestamp'])*r['progress']/max(1-r['progress'],1e-6)
                if elapsed<window:continue
                previous=record.get(str(window),{})
                if sample['timestamp']<=previous.get('timestamp',0):continue
                historic,historic_at=bootstrap.get(((char,r['rank']),window),(0,None))
                # Require a second window to corroborate a large new hourly spike.
                known=max(historic,previous.get('max',0))
                if window==3600 and known>0 and sample['speed']>known*1.5:
                    if not short or short['speed']<sample['speed']*.65:continue
                maximum=max(sample['speed'],previous.get('max',0),historic)
                max_timestamp=(sample['timestamp'] if maximum==sample['speed'] else
                               historic_at if maximum==historic else previous.get('max_timestamp'))
                record[str(window)]={**sample,'max':maximum,
                    'max_timestamp':max_timestamp,'bootstrap_max':historic,
                    'count':previous.get('count',0)+1}
            r['live_speed']=dict(record)
            r['live_speed_source']='tracker_window_api_observed_not_theoretical'
            r['live_speed_fetch_errors']=errors
        state['updated_at']=now
        state['fetch_errors']=errors
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,ensure_ascii=False,allow_nan=False));os.replace(tmp,path)
        lock.close()


def adjust(rows, now, chapter_windows=None):
    for r in rows:
        r['live_speed_applied']=False
        if r.get('event_finished_prediction'):continue
        values=r.get('live_speed',{})
        h=values.get('3600',{});s=values.get('900',{})
        fresh=lambda q: q and 0<=now-q.get('timestamp',0)<=300
        if not fresh(h):continue
        if not 0<=now-r['current_timestamp']<=1800:continue
        # The batched speed snapshot may be newer than the long trace download.
        # Re-anchor to its observed score, rather than discarding real-time information.
        latest=max([q for q in (h,s) if fresh(q)],key=lambda q:q['timestamp'])
        new_ts=latest['timestamp'];new_score=latest.get('score',r['current_score'])
        if new_ts>=r['current_timestamp'] and new_ts<r['end_timestamp'] and new_score>=r['current_score']:
            duration=(r['end_timestamp']-r['current_timestamp'])/max(1-r['progress'],1e-6)
            r['live_speed_base_current_score']=r['current_score']
            r['live_speed_base_current_timestamp']=r['current_timestamp']
            r['progress']=min(1.,r['progress']+(new_ts-r['current_timestamp'])/max(duration,1.))
            r['current_score']=new_score;r['current_timestamp']=new_ts
        current=float(r['current_score']);hours=max(0,(r['end_timestamp']-r['current_timestamp'])/3600)
        if hours<=0:continue
        recent=.65*s['speed']+.35*h['speed'] if fresh(s) else h['speed']
        # A historic 15-minute burst must not be treated as sustainable for days.
        observed=max(h['max'],recent)
        prediction=float(r['model_prediction']);required=max(0,(prediction-current)/hours)
        horizon=min(hours,2.)
        total=r.get('series_type')=='world_bloom_total'
        if total:
            # Correct only the active chapter horizon; leave unseen chapters to structural model.
            ends=[end for start,end in (chapter_windows or {}).values() if start<=now<end]
            if not ends:continue
            horizon=min(horizon,max(0,(min(ends)-now)/3600))
        changed=max(current,prediction+.5*(recent-required)*horizon)
        cap=None
        if not total and r['progress']>=.1 and observed>0:
            cap=current+observed*hours
            changed=min(changed,cap)
        r.update(live_speed_applied=True,live_speed_recent=recent,live_speed_observed_max=observed,
                 live_speed_horizon_hours=horizon,live_speed_input_prediction=prediction,
                 live_speed_score_cap=cap,model_prediction=changed,pressure_adjusted_prediction=changed)
