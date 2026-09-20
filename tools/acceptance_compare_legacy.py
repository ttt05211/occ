#!/usr/bin/env python3
"""Acceptance-only old/new parity gate. Normal package code never imports the legacy repo."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
from causal_se2_occ.data.cache import load_cache,flatten_supervised
from causal_se2_occ.models.stwm import SourceCenteredSE2Predictor,config_from_mapping
from causal_se2_occ.losses.objective import clean_objective
from causal_se2_occ.protocol import CHECKPOINT_PROTOCOL
def main():
 p=argparse.ArgumentParser();p.add_argument('--legacy-repo',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--cache',required=True);p.add_argument('--sources',type=int,default=8);p.add_argument('--device',default='cpu');a=p.parse_args();legacy=Path(a.legacy_repo).resolve()
 if not (legacy/'real_motion').exists():raise FileNotFoundError(legacy/'real_motion')
 sys.path.insert(0,str(legacy))
 from real_motion.local_st_world_model_v17 import config_from_mapping_v17
 from real_motion.local_st_world_model_v18_se2 import LocalSpatialTemporalWorldModelV18SE2
 from tools.real_motion.train_p0_f9_v18_se2_pair import se2_objective_loss as legacy_loss
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 if str(ck.get('protocol'))!=CHECKPOINT_PROTOCOL:raise RuntimeError('unexpected checkpoint protocol')
 d=torch.device(a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu');new=SourceCenteredSE2Predictor(config_from_mapping(ck.get('model_config'))).to(d);old=LocalSpatialTemporalWorldModelV18SE2(config_from_mapping_v17(ck.get('model_config'))).to(d);new.load_state_dict(ck['state_dict'],strict=True);old.load_state_dict(ck['state_dict'],strict=True)
 if list(new.state_dict())!=list(old.state_dict()):raise RuntimeError('state-dict key order differs')
 _,records=load_cache(a.cache);flat=flatten_supervised(records);n=min(a.sources,len(flat['features']));keys=('features','local_semantic_tube','kta_displacement_xy_m','existence','supervised_source','source_class_id','frame_motion_features','target_source_mask_tube','target_source_displacement_xy_m','target_source_residual_xy_m','target_yaw_rad','yaw_label_valid','se2_target_valid','yaw_enabled');b={k:flat[k][:n].to(d) for k in keys}
 def fwd(m):return m(b['features'],b['local_semantic_tube'],b['kta_displacement_xy_m'],b['frame_motion_features'],b['target_source_mask_tube'])
 with torch.no_grad():oo=fwd(old);nn=fwd(new)
 for k in ('residual_xy_m','existence_logits','yaw_delta_rad'):
  if not torch.equal(oo[k],nn[k]):raise RuntimeError(f'network output mismatch: {k}')
 old.train();new.train();old.zero_grad(set_to_none=True);new.zero_grad(set_to_none=True);oo=fwd(old);nn=fwd(new);lo,_=legacy_loss(oo,b,yaw_weight=19.,shape_weight=.25,patch_resolution_m=.8,safe_weight=0.);ln,_=clean_objective(nn,b,yaw_weight=19.,shape_weight=.25,patch_resolution_m=.8);lo.backward();ln.backward()
 if abs(float(lo.detach())-float(ln.detach()))>1e-7:raise RuntimeError('loss mismatch')
 for (ko,po),(kn,pn) in zip(old.named_parameters(),new.named_parameters()):
  if ko!=kn or (po.grad is None)!=(pn.grad is None):raise RuntimeError(f'gradient structure mismatch {ko}')
  if po.grad is not None and not torch.equal(po.grad,pn.grad):raise RuntimeError(f'gradient mismatch {ko}')
 print(json.dumps({'checkpoint_epoch':ck.get('epoch'),'global_step':ck.get('global_step'),'sources':n,'status':'PASS'},indent=2))
if __name__=='__main__':main()
