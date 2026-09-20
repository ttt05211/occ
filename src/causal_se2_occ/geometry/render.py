from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence
import math
import numpy as np
from .grid import OccupancyGrid, indices_to_xyz, transform_points, xyz_to_indices

@dataclass(frozen=True)
class RasterizedRigidComponent:
    class_id:int; voxel_indices:np.ndarray; source_voxel_count:int

def _dedup(idx,grid):
    idx=np.asarray(idx,dtype=np.int64)
    if not len(idx): return idx.reshape(0,3)
    _,Y,Z=grid.shape_xyz; flat=(idx[:,0]*Y+idx[:,1])*Z+idx[:,2]; _,first=np.unique(flat,return_index=True); return idx[np.sort(first)]

def rasterize_rigid_component(voxel_indices,class_id,current_ego_to_world,future_ego_to_world,*,source_center_world,target_center_world,yaw_delta_rad=0.0,grid=OccupancyGrid()):
    src=np.asarray(voxel_indices,dtype=np.int64)
    if not len(src): return RasterizedRigidComponent(int(class_id),np.zeros((0,3),dtype=np.int64),0)
    sc=np.asarray(source_center_world,dtype=np.float64); tc=np.asarray(target_center_world,dtype=np.float64)
    pts=transform_points(current_ego_to_world,indices_to_xyz(src,grid)); th=float(yaw_delta_rad); c,s=math.cos(th),math.sin(th); rel=pts[:,:2]-sc[None,:2]
    moved=pts.copy(); moved[:,0]=tc[0]+c*rel[:,0]-s*rel[:,1]; moved[:,1]=tc[1]+s*rel[:,0]+c*rel[:,1]
    fut=transform_points(np.linalg.inv(np.asarray(future_ego_to_world,dtype=np.float64)),moved); idx,valid=xyz_to_indices(fut,grid); return RasterizedRigidComponent(int(class_id),_dedup(idx[valid],grid),int(len(src)))

def compose_hard_a1(anchor_occ,baseline_components,replacement_components,*,dynamic_class_ids,free_label=17,grid=OccupancyGrid()):
    out=np.asarray(anchor_occ).copy(); clear=np.zeros(grid.shape_xyz,dtype=bool)
    for comp in baseline_components:
        idx=np.asarray(comp.voxel_indices,dtype=np.int64)
        if len(idx): clear[idx[:,0],idx[:,1],idx[:,2]]=True
    dyn=np.asarray(tuple(int(x) for x in dynamic_class_ids)); out[clear & np.isin(out,dyn)]=int(free_label)
    # Frozen A1 WRITE rule: original Strong source/input order, no sorting.
    for comp in replacement_components:
        idx=np.asarray(comp.voxel_indices,dtype=np.int64)
        if len(idx): out[idx[:,0],idx[:,1],idx[:,2]]=int(comp.class_id)
    return out
