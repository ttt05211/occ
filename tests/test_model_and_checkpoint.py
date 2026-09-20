from pathlib import Path
import torch
from causal_se2_occ.models.stwm import SourceCenteredSE2Predictor,ModelConfig
from causal_se2_occ.data.features import FEATURE_DIM
from causal_se2_occ.checkpoint import migrate_legacy_checkpoint,load_model_checkpoint
from causal_se2_occ.protocol import CHECKPOINT_PROTOCOL
def test_frozen_default_parameter_count_and_init():
 m=SourceCenteredSE2Predictor();assert FEATURE_DIM==46;assert sum(p.numel() for p in m.parameters())==2_073_220;assert torch.count_nonzero(m.residual_head.weight)==0;assert torch.count_nonzero(m.yaw_head.weight)==0
def test_forward_shapes():
 c=ModelConfig(d_model=16,semantic_dim=8,heads=4,blocks=1,decoder_blocks=1,tube_hw=4);m=SourceCenteredSE2Predictor(c);B=2;o=m(torch.zeros(B,FEATURE_DIM),torch.full((B,6,4,4),17,dtype=torch.uint8),torch.zeros(B,6,2),torch.zeros(B,6,5),torch.zeros(B,6,4,4,dtype=torch.uint8));assert o['residual_xy_m'].shape==(B,6,2);assert torch.equal(o['residual_xy_m'],torch.zeros_like(o['residual_xy_m']))
def test_identity_migration(tmp_path:Path):
 c=ModelConfig(d_model=16,semantic_dim=8,heads=4,blocks=1,decoder_blocks=1,tube_hw=4);m=SourceCenteredSE2Predictor(c);src=tmp_path/'a.pt';dst=tmp_path/'b.pt';torch.save({'protocol':CHECKPOINT_PROTOCOL,'arm':'Y','model_config':c.__dict__,'state_dict':m.state_dict(),'epoch':14,'global_step':18410},src);migrate_legacy_checkpoint(src,dst);_,m2=load_model_checkpoint(dst);assert list(m.state_dict())==list(m2.state_dict());assert all(torch.equal(v,m2.state_dict()[k]) for k,v in m.state_dict().items())
