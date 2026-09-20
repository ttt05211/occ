from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence
import numpy as np
from scipy.ndimage import label, generate_binary_structure, uniform_filter
from ..geometry.grid import OccupancyGrid, relative_transform, transform_points, indices_to_xyz
from ..protocol import DYNAMIC_CLASS_IDS

@dataclass(frozen=True)
class StrongKTAConfig:
    free_label:int=17; min_component_voxels:int=6; max_match_speed_mps:float=25.0; connectivity:int=2; fill_kernel:tuple[int,int,int]=(5,5,1); fill_min_fraction:float=0.3

@lru_cache(maxsize=16)
def _centers(shape,origin,step):
    X,Y,Z=shape; xs=origin[0]+(np.arange(X)+.5)*step[0]; ys=origin[1]+(np.arange(Y)+.5)*step[1]; zs=origin[2]+(np.arange(Z)+.5)*step[2]; a,b,c=np.meshgrid(xs,ys,zs,indexing='ij'); return np.stack([a.ravel(),b.ravel(),c.ravel()],1)
def _metric_to_voxel(pts,grid):
    origin=np.asarray([grid.x_min,grid.y_min,grid.z_min]); step=np.asarray(grid.voxel_size); return np.floor((np.asarray(pts)-origin)/step).astype(np.int64)
def _in_grid(idx,grid):
    sh=np.asarray(grid.shape_xyz); return ((idx>=0)&(idx<sh[None])).all(1)
def inverse_warp(semantics,src_to_dst,grid,free_label):
    sem=np.asarray(semantics); pts=_centers(tuple(grid.shape_xyz),(grid.x_min,grid.y_min,grid.z_min),tuple(grid.voxel_size)); src=transform_points(np.linalg.inv(np.asarray(src_to_dst,dtype=np.float64)),pts); idx=_metric_to_voxel(src,grid); known=_in_grid(idx,grid); flat=np.full(len(pts),int(free_label),dtype=sem.dtype); q=idx[known]; flat[known]=sem[q[:,0],q[:,1],q[:,2]]; return flat.reshape(grid.shape_xyz),known.reshape(grid.shape_xyz)
def majority_fill(semantics,unknown_mask,*,kernel=(5,5,1),min_fraction=.3):
    sem=np.asarray(semantics); unknown=np.asarray(unknown_mask,dtype=bool)
    if not unknown.any(): return sem.copy()
    out=sem.copy(); known=~unknown; denom=np.maximum(uniform_filter(known.astype(np.float32),size=kernel,mode='constant'),1e-6); best=np.zeros_like(denom,dtype=np.float32); lab=np.zeros_like(sem)
    for cls in np.unique(sem[known]):
        score=uniform_filter(((sem==cls)&known).astype(np.float32),size=kernel,mode='constant')/denom; upd=score>best; best[upd]=score[upd]; lab[upd]=cls
    fill=unknown&(best>=float(min_fraction)); out[fill]=lab[fill]; return out
def extract_instances(semantics,ego_to_world,*,grid=OccupancyGrid(),cfg=StrongKTAConfig()):
    sem=np.asarray(semantics); structure=generate_binary_structure(3,int(cfg.connectivity)); out=[]
    for cls in DYNAMIC_CLASS_IDS:
        cmap,n=label(sem==int(cls),structure=structure)
        for cid in range(1,int(n)+1):
            idx=np.argwhere(cmap==cid)
            if len(idx)<int(cfg.min_component_voxels): continue
            centroid=transform_points(ego_to_world,indices_to_xyz(idx,grid).mean(0,keepdims=True))[0]; out.append({'class_id':int(cls),'voxel_indices':idx.astype(np.int64),'centroid_world':centroid,'voxel_count':int(len(idx))})
    return out
def match_instances(previous,current,dt_s,*,max_speed_mps=25.0):
    dt=max(float(dt_s),1e-3); gate=float(max_speed_mps)*dt; velocities={}; by={}
    for j,r in enumerate(current): by.setdefault(int(r['class_id']),[]).append(j)
    for cls,cur_ids in by.items():
        prev_ids=[i for i,r in enumerate(previous) if int(r['class_id'])==cls]
        if not prev_ids: continue
        P=np.stack([np.asarray(previous[i]['centroid_world']) for i in prev_ids]); C=np.stack([np.asarray(current[j]['centroid_world']) for j in cur_ids]); dist=np.linalg.norm(C[:,None]-P[None],axis=-1); npv=dist.argmin(1); ncu=dist.argmin(0)
        for lc,cur_id in enumerate(cur_ids):
            p=int(npv[lc])
            if int(ncu[p])!=lc or float(dist[lc,p])>gate: continue
            v=(C[lc]-P[p])/dt; v=np.asarray(v,dtype=np.float64); v[2]=0.; velocities[int(cur_id)]=v
    return velocities
def strong_kta_sequence(history_semantics,history_ego_to_world,future_ego_to_world,*,frame_dt_s=.5,grid=OccupancyGrid(),cfg=StrongKTAConfig()):
    hist=np.asarray(history_semantics); cur_pose=np.asarray(history_ego_to_world[-1]); prev_pose=np.asarray(history_ego_to_world[-2]); sem0=hist[-1]; prev=extract_instances(hist[-2],prev_pose,grid=grid,cfg=cfg); cur=extract_instances(sem0,cur_pose,grid=grid,cfg=cfg); vel=match_instances(prev,cur,frame_dt_s,max_speed_mps=cfg.max_match_speed_mps); dyn=np.isin(sem0,np.asarray(DYNAMIC_CLASS_IDS,dtype=sem0.dtype)); static=sem0.copy(); static[dyn]=cfg.free_label; covered=np.zeros_like(dyn)
    for comp in cur:
        q=comp['voxel_indices']; covered[q[:,0],q[:,1],q[:,2]]=True
    rest=np.argwhere(dyn&~covered); rest_world=transform_points(cur_pose,indices_to_xyz(rest,grid)) if len(rest) else np.zeros((0,3)); rest_labels=sem0[rest[:,0],rest[:,1],rest[:,2]] if len(rest) else np.zeros((0,),dtype=sem0.dtype); outputs=[]
    for hi,fpose in enumerate(future_ego_to_world):
        dst,known=inverse_warp(static,relative_transform(cur_pose,fpose),grid,cfg.free_label); out=majority_fill(dst,~known,kernel=cfg.fill_kernel,min_fraction=cfg.fill_min_fraction); parts=[]; labs=[]
        for j,comp in enumerate(cur):
            pts=transform_points(cur_pose,indices_to_xyz(comp['voxel_indices'],grid)); v=np.asarray(vel.get(j,np.zeros(3))); parts.append(pts+v[None]*(hi+1)*frame_dt_s); labs.append(np.full(len(pts),int(comp['class_id']),dtype=sem0.dtype))
        if len(rest_world): parts.append(rest_world); labs.append(rest_labels)
        if parts:
            pts=transform_points(np.linalg.inv(np.asarray(fpose)),np.concatenate(parts)); idx=_metric_to_voxel(pts,grid); valid=_in_grid(idx,grid); q=idx[valid]; la=np.concatenate(labs)[valid]; out[q[:,0],q[:,1],q[:,2]]=la
        outputs.append(out)
    return np.stack(outputs),cur,vel
