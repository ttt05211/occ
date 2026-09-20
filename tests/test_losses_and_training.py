import torch
from causal_se2_occ.losses.objective import shape_overlap_loss,clean_objective
from causal_se2_occ.models.stwm import SourceCenteredSE2Predictor,ModelConfig
from causal_se2_occ.data.features import FEATURE_DIM
from causal_se2_occ.training import cosine_scale
def batch(B=2,H=4):
 v=torch.ones(B,6,dtype=torch.bool);s=torch.zeros(B,H,H);s[:,1:3,1:3]=1
 return {'features':torch.zeros(B,FEATURE_DIM),'local_semantic_tube':torch.full((B,6,H,H),17,dtype=torch.uint8),'kta_displacement_xy_m':torch.zeros(B,6,2),'frame_motion_features':torch.zeros(B,6,5),'target_source_mask_tube':s[:,None].repeat(1,6,1,1).to(torch.uint8),'target_source_displacement_xy_m':torch.zeros(B,6,2),'target_source_residual_xy_m':torch.zeros(B,6,2),'target_yaw_rad':torch.zeros(B,6),'yaw_enabled':torch.tensor([True,False]),'yaw_label_valid':v.clone(),'se2_target_valid':v,'existence':torch.ones(B,6)}
def test_shape_identity_zeroish():
 b=batch();p=torch.zeros(2,6,2,requires_grad=True);y=torch.zeros(2,6,requires_grad=True);l=shape_overlap_loss(p,b['target_source_displacement_xy_m'],y,b['target_yaw_rad'],b['target_source_mask_tube'][:,-1].float(),b['se2_target_valid'],b['yaw_enabled'],b['yaw_label_valid'],resolution=.8);assert float(l.detach())<1e-6;l.backward();assert torch.isfinite(p.grad).all()
def test_frozen_weights_gradient():
 b=batch();m=SourceCenteredSE2Predictor(ModelConfig(d_model=16,semantic_dim=8,heads=4,blocks=1,decoder_blocks=1,tube_hw=4));o=m(b['features'],b['local_semantic_tube'],b['kta_displacement_xy_m'],b['frame_motion_features'],b['target_source_mask_tube']);total,s=clean_objective(o,b,yaw_weight=19.,shape_weight=.25,patch_resolution_m=.8);expected=s['translation_smooth_l1']+s['existence_bce']+19*s['yaw_periodic_loss']+.25*s['se2_shape_loss'];assert abs(float(total.detach())-expected)<1e-6;total.backward()
def test_cosine_contract():assert cosine_scale(0,100)==1.0 and abs(cosine_scale(100,100)-.1)<1e-12
