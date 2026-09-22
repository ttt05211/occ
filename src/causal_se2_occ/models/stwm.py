from __future__ import annotations
from dataclasses import dataclass,asdict
import torch
import torch.nn as nn
from ..protocol import HISTORY_FRAMES,FUTURE_FRAMES,SEMANTIC_CLASSES
from ..data.features import FEATURE_DIM,FRAME_MOTION_DIM

class SpatialTemporalBlock(nn.Module):
    def __init__(self,dim,heads,mlp_ratio=4,kernel_size=5):
        super().__init__(); pad=kernel_size//2; self.spatial_norm=nn.GroupNorm(1,dim); self.spatial_dw=nn.Conv2d(dim,dim,kernel_size,padding=pad,groups=dim); self.spatial_pw1=nn.Conv2d(dim,dim*mlp_ratio,1); self.spatial_pw2=nn.Conv2d(dim*mlp_ratio,dim,1); self.spatial_act=nn.GELU(); self.temporal_norm=nn.LayerNorm(dim); self.temporal_attn=nn.MultiheadAttention(dim,heads,batch_first=True); self.temporal_ffn_norm=nn.LayerNorm(dim); self.temporal_ffn=nn.Sequential(nn.Linear(dim,dim*mlp_ratio),nn.GELU(),nn.Linear(dim*mlp_ratio,dim))
    def forward(self,x):
        B,T,C,H,W=x.shape; s=x.reshape(B*T,C,H,W); y=self.spatial_pw2(self.spatial_act(self.spatial_pw1(self.spatial_dw(self.spatial_norm(s))))); x=(s+y).reshape(B,T,C,H,W); seq=x.permute(0,3,4,1,2).reshape(B*H*W,T,C); q=self.temporal_norm(seq); y,_=self.temporal_attn(q,q,q,need_weights=False); seq=seq+y; seq=seq+self.temporal_ffn(self.temporal_ffn_norm(seq)); return seq.reshape(B,H,W,T,C).permute(0,3,4,1,2).contiguous()
class FutureQueryBlock(nn.Module):
    def __init__(self,dim,heads,mlp_ratio=4):
        super().__init__(); self.self_norm=nn.LayerNorm(dim); self.self_attn=nn.MultiheadAttention(dim,heads,batch_first=True); self.cross_q_norm=nn.LayerNorm(dim); self.cross_ctx_norm=nn.LayerNorm(dim); self.cross_attn=nn.MultiheadAttention(dim,heads,batch_first=True); self.ffn_norm=nn.LayerNorm(dim); self.ffn=nn.Sequential(nn.Linear(dim,dim*mlp_ratio),nn.GELU(),nn.Linear(dim*mlp_ratio,dim))
    def forward(self,q,context):
        z=self.self_norm(q); y,_=self.self_attn(z,z,z,need_weights=False); q=q+y; y,_=self.cross_attn(self.cross_q_norm(q),self.cross_ctx_norm(context),self.cross_ctx_norm(context),need_weights=False); q=q+y; return q+self.ffn(self.ffn_norm(q))
@dataclass(frozen=True)
class ModelConfig:
    d_model:int=128; semantic_dim:int=32; heads:int=4; blocks:int=4; decoder_blocks:int=2; tube_hw:int=20; history_frames:int=HISTORY_FRAMES; future_frames:int=FUTURE_FRAMES; use_representation:bool=True
class SourceCenteredSE2Predictor(nn.Module):
    """State-dict-compatible Clean-E14 network; existence is output but not a deployment gate."""
    def __init__(self,config=ModelConfig()):
        super().__init__(); cfg=config
        if cfg.history_frames!=6 or cfg.future_frames!=6 or not cfg.use_representation: raise ValueError("frozen 6+6 representation required")
        self.config=cfg; self.v17_config=cfg
        # Construction order intentionally matches the frozen V16 -> V17 -> V18 inheritance chain.
        self.semantic_embedding=nn.Embedding(SEMANTIC_CLASSES,cfg.semantic_dim); self.spatial_stem=nn.Sequential(nn.Conv2d(cfg.semantic_dim,cfg.d_model,3,stride=2,padding=1),nn.GroupNorm(1,cfg.d_model),nn.GELU(),nn.Conv2d(cfg.d_model,cfg.d_model,3,padding=1),nn.GroupNorm(1,cfg.d_model),nn.GELU()); self.kinematic_proj=nn.Sequential(nn.Linear(FEATURE_DIM,cfg.d_model),nn.GELU(),nn.LayerNorm(cfg.d_model)); self.time_embedding=nn.Parameter(torch.zeros(1,6,cfg.d_model,1,1)); stem=cfg.tube_hw//2; self.spatial_embedding=nn.Parameter(torch.zeros(1,1,cfg.d_model,stem,stem)); self.blocks=nn.ModuleList([SpatialTemporalBlock(cfg.d_model,cfg.heads) for _ in range(cfg.blocks)]); self.future_query=nn.Parameter(torch.zeros(1,6,cfg.d_model)); self.future_time_embedding=nn.Parameter(torch.zeros(1,6,cfg.d_model)); self.kta_future_proj=nn.Sequential(nn.Linear(2,cfg.d_model),nn.GELU(),nn.LayerNorm(cfg.d_model)); self.decoder=nn.ModuleList([FutureQueryBlock(cfg.d_model,cfg.heads) for _ in range(cfg.decoder_blocks)]); self.residual_head=nn.Linear(cfg.d_model,2); self.existence_head=nn.Linear(cfg.d_model,1)
        nn.init.trunc_normal_(self.time_embedding,std=.02); nn.init.trunc_normal_(self.spatial_embedding,std=.02); nn.init.trunc_normal_(self.future_query,std=.02); nn.init.trunc_normal_(self.future_time_embedding,std=.02); nn.init.zeros_(self.residual_head.weight); nn.init.zeros_(self.residual_head.bias); nn.init.zeros_(self.existence_head.weight); nn.init.zeros_(self.existence_head.bias)
        self.frame_motion_proj=nn.Sequential(nn.Linear(FRAME_MOTION_DIM,cfg.d_model),nn.GELU(),nn.LayerNorm(cfg.d_model)); self.source_mask_embedding=nn.Embedding(2,cfg.semantic_dim); nn.init.normal_(self.source_mask_embedding.weight,mean=0.,std=.02); self.yaw_head=nn.Linear(cfg.d_model,1); nn.init.zeros_(self.yaw_head.weight); nn.init.zeros_(self.yaw_head.bias)
    def forward(self,features,local_semantic_tube,kta_displacement_xy_m,frame_motion_features,target_source_mask_tube):
        cfg=self.config
        if features.ndim!=2 or features.shape[-1]!=FEATURE_DIM: raise ValueError(f'features must be [B,{FEATURE_DIM}]')
        B=features.shape[0]
        if tuple(local_semantic_tube.shape)!=(B,6,cfg.tube_hw,cfg.tube_hw): raise ValueError('local_semantic_tube must be [B,6,tube_hw,tube_hw]')
        if tuple(target_source_mask_tube.shape)!=tuple(local_semantic_tube.shape): raise ValueError('target_source_mask_tube must match local_semantic_tube')
        if tuple(kta_displacement_xy_m.shape)!=(B,6,2): raise ValueError('kta_displacement_xy_m must be [B,6,2]')
        if tuple(frame_motion_features.shape)!=(B,6,FRAME_MOTION_DIM): raise ValueError('frame_motion_features must be [B,6,5]')
        labels=local_semantic_tube.long(); mask=target_source_mask_tube.long()
        if bool((labels<0).any()) or bool((labels>=SEMANTIC_CLASSES).any()): raise ValueError('semantic tube labels must be in [0,17]')
        if bool((mask<0).any()) or bool((mask>1).any()): raise ValueError('source mask must be binary')
        if B==0: return {'residual_xy_m':features.new_empty((0,6,2)),'existence_logits':features.new_empty((0,6)),'yaw_delta_rad':features.new_empty((0,6))}
        emb=self.semantic_embedding(labels)+self.source_mask_embedding(mask); x=emb.permute(0,1,4,2,3).reshape(B*6,cfg.semantic_dim,cfg.tube_hw,cfg.tube_hw); x=self.spatial_stem(x); Hs,Ws=x.shape[-2:]; x=x.reshape(B,6,cfg.d_model,Hs,Ws); obj=self.kinematic_proj(features).view(B,1,cfg.d_model,1,1); fm=self.frame_motion_proj(frame_motion_features.to(x.dtype)).view(B,6,cfg.d_model,1,1); x=x+obj+fm+self.time_embedding+self.spatial_embedding
        for block in self.blocks: x=block(x)
        context=x.permute(0,1,3,4,2).reshape(B,6*Hs*Ws,cfg.d_model); q=self.future_query.expand(B,-1,-1)+self.future_time_embedding; q=q+self.kinematic_proj(features).unsqueeze(1); q=q+self.kta_future_proj(kta_displacement_xy_m.to(q.dtype)/20.)
        for block in self.decoder: q=block(q,context)
        return {'residual_xy_m':self.residual_head(q),'existence_logits':self.existence_head(q)[...,0],'yaw_delta_rad':self.yaw_head(q)[...,0]}
def config_from_mapping(raw):
    if not raw:return ModelConfig()
    f=ModelConfig.__dataclass_fields__; return ModelConfig(**{k:(bool(raw[k]) if k=='use_representation' else int(raw[k])) for k in f if k in raw})
