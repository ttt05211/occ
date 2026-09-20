from __future__ import annotations

import math
from typing import Mapping, Sequence
import numpy as np
import torch
from ..geometry.grid import OccupancyGrid, transform_points
from ..protocol import DYNAMIC_CLASS_IDS, HISTORY_FRAMES

FRAME_MOTION_DIM = 5
CLASS_TO_SLOT = {int(c): i for i, c in enumerate(DYNAMIC_CLASS_IDS)}
FEATURE_NAMES = (
    "current_x_norm", "current_y_norm",
    "current_vx_norm", "current_vy_norm", "current_speed_norm",
    "log_voxel_count", "extent_x_norm", "extent_y_norm", "extent_z_norm",
    "kta_matched",
    *tuple(f"class_{int(c)}" for c in DYNAMIC_CLASS_IDS),
    *tuple(f"hist_offset_{t}_{a}" for t in range(HISTORY_FRAMES) for a in ("x", "y")),
    *tuple(f"hist_valid_{t}" for t in range(HISTORY_FRAMES)),
    *tuple(f"hist_vel_{t}_{a}" for t in range(HISTORY_FRAMES - 1) for a in ("x", "y")),
)
FEATURE_DIM = len(FEATURE_NAMES)
_NAME_TO_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}

def world_points_to_t0(points_world: np.ndarray, t0_ego_to_world: np.ndarray) -> np.ndarray:
    return transform_points(np.linalg.inv(np.asarray(t0_ego_to_world, dtype=np.float64)), points_world)

def world_vec_to_t0(vec_world: np.ndarray, t0_ego_to_world: np.ndarray) -> np.ndarray:
    R = np.asarray(t0_ego_to_world, dtype=np.float64)[:3, :3]
    return np.asarray(vec_world, dtype=np.float64) @ R

def component_extent_xyz_m(component: Mapping, grid: OccupancyGrid = OccupancyGrid()) -> np.ndarray:
    idx = np.asarray(component["voxel_indices"], dtype=np.int64)
    if len(idx) == 0:
        return np.zeros(3, dtype=np.float32)
    span = idx.max(axis=0) - idx.min(axis=0) + 1
    return (span * np.asarray(grid.voxel_size, dtype=np.float64)).astype(np.float32)

def backward_component_tracks(components_by_frame: Sequence[Sequence[Mapping]], *, frame_dt_s: float = 0.5, max_speed_mps: float = 25.0) -> tuple[np.ndarray, np.ndarray]:
    if len(components_by_frame) != HISTORY_FRAMES:
        raise ValueError(f"expected {HISTORY_FRAMES} history frames")
    current = list(components_by_frame[-1]); n=len(current)
    centers=np.zeros((n,HISTORY_FRAMES,3),dtype=np.float64); valid=np.zeros((n,HISTORY_FRAMES),dtype=bool)
    for i,comp in enumerate(current):
        centers[i,-1]=np.asarray(comp["centroid_world"],dtype=np.float64); valid[i,-1]=True
    gate=float(frame_dt_s)*float(max_speed_mps)
    for t in range(HISTORY_FRAMES-2,-1,-1):
        previous=list(components_by_frame[t]); pairs=[]
        for i,comp0 in enumerate(current):
            if not valid[i,t+1]: continue
            ref=centers[i,t+1]
            for j,comp in enumerate(previous):
                if int(comp["class_id"])!=int(comp0["class_id"]): continue
                p=np.asarray(comp["centroid_world"],dtype=np.float64); d=float(np.linalg.norm(ref[:2]-p[:2]))
                if d<=gate: pairs.append((d,i,j))
        pairs.sort(key=lambda x:(x[0],x[1],x[2])); used_i=set(); used_j=set()
        for _,i,j in pairs:
            if i in used_i or j in used_j: continue
            used_i.add(i); used_j.add(j); centers[i,t]=np.asarray(previous[j]["centroid_world"],dtype=np.float64); valid[i,t]=True
    return centers,valid

def build_source_features(current_components,current_velocities_world,track_centers_world,track_valid,t0_ego_to_world,*,frame_dt_s=.5,grid=OccupancyGrid()):
    n=len(current_components)
    if track_centers_world.shape!=(n,HISTORY_FRAMES,3) or track_valid.shape!=(n,HISTORY_FRAMES): raise ValueError("history-track shape mismatch")
    out=np.zeros((n,FEATURE_DIM),dtype=np.float32)
    for i,comp in enumerate(current_components):
        cur_world=np.asarray(comp["centroid_world"],dtype=np.float64); cur_t0=world_points_to_t0(cur_world[None],t0_ego_to_world)[0]
        v_world=np.asarray(current_velocities_world.get(i,np.zeros(3)),dtype=np.float64); v_t0=world_vec_to_t0(v_world,t0_ego_to_world); extent=component_extent_xyz_m(comp,grid)
        feat=[cur_t0[0]/40.,cur_t0[1]/40.,v_t0[0]/20.,v_t0[1]/20.,float(np.linalg.norm(v_t0[:2]))/20.,math.log1p(float(comp.get("voxel_count",len(comp["voxel_indices"]))))/8.,extent[0]/10.,extent[1]/10.,extent[2]/5.,1. if i in current_velocities_world else 0.]
        cls=[0.]*len(DYNAMIC_CLASS_IDS); cls[CLASS_TO_SLOT[int(comp["class_id"])]]=1.; feat.extend(cls)
        hist_t0=np.zeros((HISTORY_FRAMES,3),dtype=np.float64)
        for t in range(HISTORY_FRAMES):
            if track_valid[i,t]: hist_t0[t]=world_points_to_t0(track_centers_world[i,t][None],t0_ego_to_world)[0]
        offsets=hist_t0[:,:2]-cur_t0[None,:2]; offsets[~track_valid[i]]=0.; feat.extend((offsets/20.).reshape(-1).tolist()); feat.extend(track_valid[i].astype(np.float32).tolist())
        seg=np.zeros((HISTORY_FRAMES-1,2),dtype=np.float64)
        for t in range(HISTORY_FRAMES-1):
            if track_valid[i,t] and track_valid[i,t+1]: seg[t]=(hist_t0[t+1,:2]-hist_t0[t,:2])/float(frame_dt_s)
        feat.extend((seg/20.).reshape(-1).tolist()); arr=np.asarray(feat,dtype=np.float32)
        if arr.shape!=(FEATURE_DIM,): raise AssertionError(f"feature shape {arr.shape} != {(FEATURE_DIM,)}")
        out[i]=arr
    return out

def frame_motion_features_from_flat(features: torch.Tensor) -> torch.Tensor:
    if features.ndim!=2 or features.shape[-1]!=FEATURE_DIM: raise ValueError(f"features must be [N,{FEATURE_DIM}]")
    n=features.shape[0]; out=features.new_zeros((n,HISTORY_FRAMES,FRAME_MOTION_DIM)); seg=features.new_zeros((n,HISTORY_FRAMES-1,2))
    for t in range(HISTORY_FRAMES-1):
        seg[:,t,0]=features[:,_NAME_TO_INDEX[f"hist_vel_{t}_x"]]; seg[:,t,1]=features[:,_NAME_TO_INDEX[f"hist_vel_{t}_y"]]
    for t in range(HISTORY_FRAMES):
        out[:,t,0]=features[:,_NAME_TO_INDEX[f"hist_offset_{t}_x"]]; out[:,t,1]=features[:,_NAME_TO_INDEX[f"hist_offset_{t}_y"]]; out[:,t,4]=features[:,_NAME_TO_INDEX[f"hist_valid_{t}"]]
        vel=seg[:,0] if t==0 else seg[:,-1] if t==HISTORY_FRAMES-1 else .5*(seg[:,t-1]+seg[:,t]); out[:,t,2:4]=vel
    out[:,:,:4]*=out[:,:,4:5]; return out

def target_source_mask_from_tube(local_semantic_tube,source_class_id,track_valid,features,*,patch_resolution_m=.8,margin_cells=1.):
    tube=local_semantic_tube
    if tube.ndim!=4 or tube.shape[1]!=HISTORY_FRAMES: raise ValueError("local_semantic_tube must be [N,6,H,W]")
    n,t,h,w=tube.shape
    if source_class_id.shape!=(n,) or track_valid.shape!=(n,t) or features.shape!=(n,FEATURE_DIM): raise ValueError("source feature shapes mismatch")
    extent_x=features[:,_NAME_TO_INDEX["extent_x_norm"]].float()*10.; extent_y=features[:,_NAME_TO_INDEX["extent_y_norm"]].float()*10.
    half_x=torch.clamp(extent_x/(2.*patch_resolution_m)+margin_cells,min=1.); half_y=torch.clamp(extent_y/(2.*patch_resolution_m)+margin_cells,min=1.)
    gx=torch.arange(h,device=tube.device,dtype=torch.float32)-(h-1)/2.; gy=torch.arange(w,device=tube.device,dtype=torch.float32)-(w-1)/2.
    gate_x=gx[None,None,:,None].abs()<=half_x[:,None,None,None]; gate_y=gy[None,None,None,:].abs()<=half_y[:,None,None,None]
    same=tube.long()==source_class_id.long()[:,None,None,None]; valid=track_valid.bool()[:,:,None,None]
    return (same&gate_x&gate_y&valid).to(torch.uint8)
