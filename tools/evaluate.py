#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from causal_se2_occ.checkpoint import load_model_checkpoint
from causal_se2_occ.data.cache import load_cache
from causal_se2_occ.data.nuscenes import NuScenesOcc3D,WindowTokens
from causal_se2_occ.inference import forecast_window
from causal_se2_occ.priors.strong_kta import strong_kta_sequence
from causal_se2_occ.metrics.moving import moving_support_for_nuscenes
from causal_se2_occ.metrics.iou import raw_counts,add_counts,metrics_from_horizon_counts
from causal_se2_occ.metrics.bootstrap import paired_scene_bootstrap
from causal_se2_occ.protocol import REPORT_HORIZON_INDICES,REPORT_HORIZONS_S
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataroot',required=True);p.add_argument('--val-cache',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--max-windows',type=int,default=0);p.add_argument('--bootstrap-samples',type=int,default=2000);a=p.parse_args();meta,recs=load_cache(a.val_cache)
 if not a.max_windows and (str(meta.get('split'))!='val' or int(meta.get('scene_count',-1))!=150 or int(meta.get('eligible_windows',-1))!=4369 or len(recs)!=4369):raise RuntimeError('formal evaluation requires full 150-scene/4,369-window val cache')
 if a.max_windows:recs=recs[:a.max_windows]
 src=NuScenesOcc3D(a.dataroot,split='val');_,model=load_model_checkpoint(a.checkpoint,a.device);scene_rows={};scenes=[]
 for wi,r in enumerate(recs,1):
  scene=str(r['scene_name']);scenes.append(scene);w=WindowTokens(scene,tuple(r['history_tokens']),str(r['t0_token']),tuple(r['future_tokens']));hist=[src.semantics(scene,t) for t in w.history_tokens];hp=[src.pose(t) for t in w.history_tokens];fp=[src.pose(t) for t in w.future_tokens];strong,_,_=strong_kta_sequence(hist,hp,fp);pred,_=forecast_window(model,hist,hp,fp,device=a.device)
  for h,hi in REPORT_HORIZON_INDICES.items():
   gt=src.semantics(scene,w.future_tokens[hi]);mov=moving_support_for_nuscenes(src.nusc,w.t0_token,w.future_tokens[hi],h)
   for name,x in [('strong_kta',strong[hi]),('clean_e14',pred[hi])]:
    key=(scene,name,float(h));scene_rows[key]=add_counts(scene_rows.get(key),raw_counts(x,gt,mov))
  if wi==1 or wi%50==0 or wi==len(recs):print(f'eval {wi}/{len(recs)} {scene}',flush=True)
 unique=list(dict.fromkeys(scenes));reports={}
 for name in ('strong_kta','clean_e14'):
  by={float(h):None for h in REPORT_HORIZONS_S}
  for s in unique:
   for h in REPORT_HORIZONS_S:by[h]=add_counts(by[h],scene_rows[(s,name,h)])
  reports[name]=metrics_from_horizon_counts(by)
 result={'protocol':'formal_full_grid_4369_window_v1','num_windows':len(recs),'num_scenes':len(unique),'reports':reports,'bootstrap_clean_minus_strong':paired_scene_bootstrap(scene_rows,'strong_kta','clean_e14',unique,samples=a.bootstrap_samples) if len(unique)>1 else None};Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:{m:v for m,v in q.items() if m!='per_horizon'} for k,q in reports.items()},indent=2))
if __name__=='__main__':main()
