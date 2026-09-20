from __future__ import annotations
from typing import Sequence
import numpy as np
from ..geometry.grid import OccupancyGrid, relative_transform, warp_semantic_grid
from ..protocol import DYNAMIC_CLASS_IDS, HISTORY_FRAMES, SEMANTIC_CLASSES
from .features import FEATURE_DIM, FEATURE_NAMES

DEFAULT_PATCH_SIZE_M = 16.0
DEFAULT_PATCH_RESOLUTION_M = 0.8

def top_surface_semantic(semantics: np.ndarray, *, grid: OccupancyGrid = OccupancyGrid(), free_label: int = 17) -> np.ndarray:
    sem=np.asarray(semantics)
    if tuple(sem.shape)!=tuple(grid.shape_xyz): raise ValueError("semantic grid shape mismatch")
    occupied=sem!=int(free_label); has=occupied.any(axis=2); z=sem.shape[2]-1-np.argmax(occupied[:,:,::-1],axis=2); out=np.full(sem.shape[:2],int(free_label),dtype=np.uint8)
    if bool(has.any()):
        ix,iy=np.nonzero(has); out[ix,iy]=sem[ix,iy,z[ix,iy]].astype(np.uint8)
    return out

def _priority_lut(free_label: int) -> np.ndarray:
    lut=np.full(256,240,dtype=np.int16); dyn=[int(c) for c in DYNAMIC_CLASS_IDS]; ds=set(dyn)
    for rank,c in enumerate(dyn): lut[c]=rank
    base=len(dyn)+8
    for c in range(SEMANTIC_CLASSES):
        if c!=int(free_label) and c not in ds: lut[c]=base+c
    lut[int(free_label)]=255; return lut

def priority_pool2x2(labels: np.ndarray, *, free_label: int = 17) -> np.ndarray:
    x=np.asarray(labels,dtype=np.uint8)
    if x.ndim!=2 or x.shape[0]%2 or x.shape[1]%2: raise ValueError("labels must be an even [X,Y] array")
    X,Y=x.shape; cells=x.reshape(X//2,2,Y//2,2).transpose(0,2,1,3).reshape(X//2,Y//2,4); ranks=_priority_lut(int(free_label))[cells]; pick=np.argmin(ranks,axis=2)
    return np.take_along_axis(cells,pick[...,None],axis=2)[...,0].astype(np.uint8)

def extract_bev_patch(bev,center_xy_m,*,grid=OccupancyGrid(),patch_voxels=40,free_label=17):
    arr=np.asarray(bev,dtype=np.uint8); xy=np.asarray(center_xy_m,dtype=np.float64); vx,vy=float(grid.voxel_size[0]),float(grid.voxel_size[1]); cx=int(np.floor((xy[0]-grid.x_min)/vx));cy=int(np.floor((xy[1]-grid.y_min)/vy));half=patch_voxels//2;x0,y0=cx-half,cy-half;x1,y1=x0+patch_voxels,y0+patch_voxels;out=np.full((patch_voxels,patch_voxels),int(free_label),dtype=np.uint8);sx0,sy0=max(x0,0),max(y0,0);sx1,sy1=min(x1,arr.shape[0]),min(y1,arr.shape[1])
    if sx0<sx1 and sy0<sy1: out[sx0-x0:sx1-x0,sy0-y0:sy1-y0]=arr[sx0:sx1,sy0:sy1]
    return out

def history_offsets_from_features(features) -> np.ndarray:
    x=features.detach().cpu().numpy() if hasattr(features,"detach") else np.asarray(features)
    if x.ndim!=2 or x.shape[1]!=FEATURE_DIM: raise ValueError(f"features must be [N,{FEATURE_DIM}]")
    out=np.zeros((x.shape[0],HISTORY_FRAMES,2),dtype=np.float32); ni={name:i for i,name in enumerate(FEATURE_NAMES)}
    for t in range(HISTORY_FRAMES):
        out[:,t,0]=x[:,ni[f"hist_offset_{t}_x"]]*20.;out[:,t,1]=x[:,ni[f"hist_offset_{t}_y"]]*20.
    return out

def build_local_semantic_tubes(history_semantics:Sequence[np.ndarray],history_poses:Sequence[np.ndarray],source_xy_t0_m,history_offsets_xy_t0_m,track_valid,*,grid=OccupancyGrid(),free_label=17,patch_size_m=DEFAULT_PATCH_SIZE_M,patch_resolution_m=DEFAULT_PATCH_RESOLUTION_M):
    if len(history_semantics)!=HISTORY_FRAMES or len(history_poses)!=HISTORY_FRAMES: raise ValueError("expected six history semantics and poses")
    src=np.asarray(source_xy_t0_m,dtype=np.float32);offs=np.asarray(history_offsets_xy_t0_m,dtype=np.float32);valid=np.asarray(track_valid,dtype=bool);native=float(grid.voxel_size[0]);raw_vox=int(round(float(patch_size_m)/native));pool=int(round(float(patch_resolution_m)/native));out_hw=raw_vox//pool;t0_pose=np.asarray(history_poses[-1],dtype=np.float64);bevs=[]
    for t,(sem,pose) in enumerate(zip(history_semantics,history_poses)):
        aligned=np.asarray(sem) if t==HISTORY_FRAMES-1 else warp_semantic_grid(np.asarray(sem),relative_transform(np.asarray(pose,dtype=np.float64),t0_pose),grid=grid,free_label=int(free_label));bevs.append(top_surface_semantic(aligned,grid=grid,free_label=int(free_label)))
    tubes=np.full((src.shape[0],HISTORY_FRAMES,out_hw,out_hw),int(free_label),dtype=np.uint8)
    for i in range(src.shape[0]):
        for t in range(HISTORY_FRAMES):
            center=src[i]+offs[i,t] if valid[i,t] else src[i];patch=extract_bev_patch(bevs[t],center,grid=grid,patch_voxels=raw_vox,free_label=int(free_label));tubes[i,t]=priority_pool2x2(patch,free_label=int(free_label))
    return tubes
