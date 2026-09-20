from pathlib import Path
import torch
from causal_se2_occ.models.stwm import SourceCenteredSE2Predictor,ModelConfig
from causal_se2_occ.losses.objective import clean_objective
from causal_se2_occ.data.features import FEATURE_DIM
def make_batch(B=1,H=4):
 v=torch.ones(B,6,dtype=torch.bool);mask=torch.zeros(B,6,H,H,dtype=torch.uint8);mask[:,:,1:3,1:3]=1;return {'features':torch.zeros(B,FEATURE_DIM),'local_semantic_tube':torch.full((B,6,H,H),17,dtype=torch.uint8),'kta_displacement_xy_m':torch.zeros(B,6,2),'frame_motion_features':torch.zeros(B,6,5),'target_source_mask_tube':mask,'target_source_displacement_xy_m':torch.zeros(B,6,2),'target_source_residual_xy_m':torch.ones(B,6,2)*.1,'target_yaw_rad':torch.zeros(B,6),'yaw_enabled':torch.ones(B,dtype=torch.bool),'yaw_label_valid':v,'se2_target_valid':v,'existence':torch.ones(B,6)}
def test_update_restore(tmp_path:Path):
 c=ModelConfig(d_model=16,semantic_dim=8,heads=4,blocks=1,decoder_blocks=1,tube_hw=4);m=SourceCenteredSE2Predictor(c);o=torch.optim.AdamW(m.parameters(),lr=5e-4,weight_decay=1e-4);b=make_batch();out=m(b['features'],b['local_semantic_tube'],b['kta_displacement_xy_m'],b['frame_motion_features'],b['target_source_mask_tube']);loss,_=clean_objective(out,b);loss.backward();o.step();p=tmp_path/'s.pt';torch.save({'state_dict':m.state_dict(),'optimizer':o.state_dict(),'global_step':1},p);m2=SourceCenteredSE2Predictor(c);o2=torch.optim.AdamW(m2.parameters(),lr=1.);ck=torch.load(p,weights_only=False);m2.load_state_dict(ck['state_dict'],strict=True);o2.load_state_dict(ck['optimizer']);assert o2.param_groups[0]['lr']==5e-4 and ck['global_step']==1
