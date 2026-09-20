#!/usr/bin/env python3
from __future__ import annotations
import argparse,numpy as np
from causal_se2_occ.checkpoint import load_model_checkpoint
from causal_se2_occ.data.nuscenes import NuScenesOcc3D
from causal_se2_occ.inference import forecast_window
p=argparse.ArgumentParser();p.add_argument('--dataroot',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--split',default='val');p.add_argument('--window-index',type=int,default=0);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();src=NuScenesOcc3D(a.dataroot,split=a.split);w=next(x for i,x in enumerate(src.iter_windows()) if i==a.window_index);hist=[src.semantics(w.scene_name,t) for t in w.history_tokens];hp=[src.pose(t) for t in w.history_tokens];fp=[src.pose(t) for t in w.future_tokens];ck,m=load_model_checkpoint(a.checkpoint,a.device);pred,diag=forecast_window(m,hist,hp,fp,device=a.device);np.savez_compressed(a.output,prediction=pred,scene_name=w.scene_name,t0_token=w.t0_token,future_tokens=np.asarray(w.future_tokens),existence_logits=diag['existence_logits']);print(a.output)
