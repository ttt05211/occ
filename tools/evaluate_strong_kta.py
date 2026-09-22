#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from causal_se2_occ.data.nuscenes import NuScenesOcc3D
from causal_se2_occ.priors.strong_kta import strong_kta_sequence
from causal_se2_occ.metrics.moving import moving_support_for_nuscenes
from causal_se2_occ.metrics.iou import raw_counts,add_counts,metrics_from_horizon_counts
from causal_se2_occ.protocol import REPORT_HORIZON_INDICES,REPORT_HORIZONS_S
from causal_se2_occ.io import prepare_output_file
p=argparse.ArgumentParser();p.add_argument('--dataroot',required=True);p.add_argument('--output',required=True);p.add_argument('--max-windows',type=int,default=0);a=p.parse_args();prepare_output_file(a.output);src=NuScenesOcc3D(a.dataroot,split='val');by={float(h):None for h in REPORT_HORIZONS_S};wins=list(src.iter_windows(max_windows=(a.max_windows or None)))
if not a.max_windows and (len({w.scene_name for w in wins}),len(wins))!=(150,4369):raise RuntimeError('formal Strong/KTA evaluation requires 150 scenes / 4,369 windows')
for i,w in enumerate(wins,1):
 hist=[src.semantics(w.scene_name,t) for t in w.history_tokens];hp=[src.pose(t) for t in w.history_tokens];fp=[src.pose(t) for t in w.future_tokens];pred,_,_=strong_kta_sequence(hist,hp,fp)
 for h,hi in REPORT_HORIZON_INDICES.items():by[h]=add_counts(by[h],raw_counts(pred[hi],src.semantics(w.scene_name,w.future_tokens[hi]),moving_support_for_nuscenes(src.nusc,w.t0_token,w.future_tokens[hi],h)))
r=metrics_from_horizon_counts(by);Path(a.output).write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='per_horizon'},indent=2))
