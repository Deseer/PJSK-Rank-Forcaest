"""Train the production CN shape residual sidecar from complete ended lines."""
import argparse
import bisect
import collections
import importlib
import json
import math
import os
import pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import forecast_ops
import train_baseline as module
from live_training_history import read_points,history_root
from shape_live_model import CHECKPOINTS,WINDOWS,prepare,features,reference_observation


def collect(module,config):
    root=Path(config['data_root']);meta=module.load_event_meta(root,module.load_finished_event_ids());contexts=module.load_event_feature_contexts()
    normalized=module.load_normalized_rank_timeseries_event_keys(root);legacy=module.load_legacy_event_files(root,normalized)
    lines={}
    for region in module.REGIONS:
        for row in module.iter_all_rank_timeseries_rows(root,region,set(module.COMMON_RANKS),legacy,frozenset({'normal'})):
            em=meta.get((region,row['event_id']))
            if not em or em.series_type!='normal' or row['character_id']!=0 or row['timestamp']<em.start_timestamp or row['score']<0:continue
            key=(region,row['event_id'],row['rank']);t,score=row['timestamp'],row['score']
            if key not in lines:
                context=contexts.get((region,row['event_id']))
                cutoffs=[int(em.start_timestamp+p*(em.end_timestamp-em.start_timestamp)) for p in CHECKPOINTS]
                boundaries=sorted({c-h*3600 for c in cutoffs for h in (0,*WINDOWS) if c-h*3600>=em.start_timestamp})
                lines[key]=dict(region=region,event=row['event_id'],rank=row['rank'],start=em.start_timestamp,end=em.end_timestamp,
                                units=list(context.unit_labels) if context else [],condition=context.condition if context else 'other',
                                cutoffs=cutoffs,boundaries=boundaries,observed=[None]*len(boundaries),final=None)
            line=lines[key]
            if line['final'] is None or t>=line['final'][0]:line['final']=(t,score)
            for i in range(bisect.bisect_left(line['boundaries'],t),len(line['boundaries'])):
                old=line['observed'][i]
                if old is None or t>=old[0]:line['observed'][i]=(t,score)
        print('collected',region,len(lines),flush=True)
    return lines


def dataset(module,all_lines,data_root=None):
    # Prefix features must include observed ranks even when those ranks have no
    # fresh final label. Production's training iterator filters such live lines.
    # Supplement their observations without inventing settled training labels.
    context_rows=module.load_event_feature_contexts()
    meta=module.load_event_meta(Path(data_root),module.load_finished_event_ids()) if data_root else None
    if meta is not None:
        for path in history_root().glob('*_history.tsv'):
            parts=path.name.split('_');region,event=parts[0],int(parts[1]);em=meta.get((region,event))
            if not em or em.series_type!='normal':continue
            context=context_rows.get((region,event))
            for char,rank,t,score in read_points(path):
                if char!=0 or rank not in module.COMMON_RANKS or t<em.start_timestamp or score<0:continue
                key=(region,event,rank)
                if key not in all_lines:
                    cuts=[int(em.start_timestamp+p*(em.end_timestamp-em.start_timestamp)) for p in CHECKPOINTS]
                    boundaries=sorted({c-h*3600 for c in cuts for h in (0,*WINDOWS) if c-h*3600>=em.start_timestamp})
                    all_lines[key]=dict(region=region,event=event,rank=rank,start=em.start_timestamp,end=em.end_timestamp,
                                        units=list(context.unit_labels) if context else [],condition=context.condition if context else 'other',
                                        cutoffs=cuts,boundaries=boundaries,observed=[None]*len(boundaries),final=(t,score))
                line=all_lines[key]
                if t>=line['final'][0]:line['final']=(t,score)
                for i in range(bisect.bisect_left(line['boundaries'],t),len(line['boundaries'])):
                    value=line['observed'][i]
                    if value is None or t>=value[0]:line['observed'][i]=(t,score)
    accepted={k:v for k,v in all_lines.items() if v['final'][0]>=v['end']-120 and v['final'][1]>0}
    contexts={key:dict(condition=v.condition,units=list(v.unit_labels),main_characters=list(v.main_character_ids)) for key,v in context_rows.items()}
    assets={}
    for region in module.REGIONS:
        for event in json.loads((module.MASTERDATA_ROOTS[region]/'events.json').read_text()):assets[(region,int(event['id']))]=event.get('assetbundleName')
    payload=prepare(accepted,contexts,assets)
    snapshots={}
    for line in all_lines.values():
        for cutoff in line['cutoffs']:
            value=reference_observation(line,cutoff)
            if value:snapshots.setdefault((line['region'],line['event'],cutoff),{})[line['rank']]=value[1]
    rows=[];vectors=[]
    for key,line in sorted(accepted.items()):
        for target,now in zip(CHECKPOINTS,line['cutoffs']):
            current=reference_observation(line,now)
            if not current:continue
            earlier={h:reference_observation(line,now-h*3600) for h in WINDOWS}
            vector=features(payload,line,now,current,snapshots[(line['region'],line['event'],now)],earlier)
            if vector is None:continue
            rows.append(dict(region=line['region'],event=line['event'],rank=line['rank'],start=line['start'],end=line['end'],
                             target=target,current=current[1],final=line['final'][1]))
            vectors.append(vector)
    return payload,rows,np.asarray(vectors)


def train(payload,rows,x,indices=None):
    indices=list(range(len(rows))) if indices is None else indices
    base=np.where(np.isfinite(x[:,75]),x[:,75],0.)
    y=np.asarray([math.log(r['final']/r['current']) for r in rows])-base
    counts=collections.Counter((rows[i]['region'],rows[i]['event']) for i in indices)
    weights=np.asarray([1/counts[(rows[i]['region'],rows[i]['event'])] for i in indices]);weights/=weights.mean()
    model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=180,max_leaf_nodes=15,learning_rate=.06,
                                      min_samples_leaf=30,l2_regularization=10,early_stopping=False,random_state=42)
    model.fit(x[indices],y[indices],sample_weight=weights)
    return dict(payload,model=model,version='cn_shape_residual_v1',regions=['cn'],feature_count=88,rank_reconciliation='legacy',
                training_samples=len(indices),training_events=len(counts),last_train_end=max(rows[i]['end'] for i in indices))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True);args=parser.parse_args()
    config=json.loads(args.config.read_text());os.environ.update(forecast_ops.env_from_config(config));m=importlib.reload(module)
    signature=forecast_ops.training_input_signature(config,m.load_finished_event_ids())
    cached=pickle.load(args.cache.open('rb')) if args.cache.exists() else None
    if cached and cached.get('signature')==signature:all_lines=cached['lines']
    else:
        all_lines=collect(m,config)
        args.cache.parent.mkdir(parents=True,exist_ok=True)
        with args.cache.open('wb') as handle:pickle.dump(dict(signature=signature,lines=all_lines),handle)
    payload,rows,x=dataset(m,all_lines,config['data_root'])
    with args.output.with_suffix('.dataset.pkl').open('wb') as handle:pickle.dump((payload,rows,x),handle)
    print('dataset',len(rows),x.shape,flush=True)
    artifact=train(payload,rows,x);artifact['input_signature']=signature
    args.output.parent.mkdir(parents=True,exist_ok=True);temporary=args.output.with_suffix('.tmp')
    with temporary.open('wb') as handle:pickle.dump(artifact,handle)
    temporary.replace(args.output)
    print('trained',artifact['training_events'],artifact['training_samples'],flush=True)


if __name__=='__main__':main()
