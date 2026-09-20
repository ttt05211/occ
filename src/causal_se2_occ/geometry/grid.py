from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np

@dataclass(frozen=True)
class OccupancyGrid:
    x_min: float=-40.0; y_min: float=-40.0; z_min: float=-1.0
    voxel_size: tuple[float,float,float]=(0.4,0.4,0.4)
    shape_xyz: tuple[int,int,int]=(200,200,16)
    @property
    def shape_hwd(self): return self.shape_xyz  # legacy-compatible alias
    @property
    def x_max(self): return self.x_min+self.shape_xyz[0]*self.voxel_size[0]
    @property
    def y_max(self): return self.y_min+self.shape_xyz[1]*self.voxel_size[1]
    @property
    def z_max(self): return self.z_min+self.shape_xyz[2]*self.voxel_size[2]

def quaternion_wxyz_to_matrix(q: Sequence[float])->np.ndarray:
    q=np.asarray(q,dtype=np.float64)
    if q.shape!=(4,): raise ValueError("quaternion must be [w,x,y,z]")
    n=float(np.dot(q,q))
    if n<1e-16: raise ValueError("zero quaternion")
    q=q/math.sqrt(n); w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],dtype=np.float64)

def quaternion_yaw(q):
    r=quaternion_wxyz_to_matrix(q); return float(math.atan2(r[1,0],r[0,0]))

def pose_matrix(translation_xyz, rotation_wxyz):
    out=np.eye(4,dtype=np.float64); out[:3,:3]=quaternion_wxyz_to_matrix(rotation_wxyz); out[:3,3]=np.asarray(translation_xyz,dtype=np.float64); return out

def relative_transform(src_ego_to_world,dst_ego_to_world):
    return np.linalg.inv(np.asarray(dst_ego_to_world,dtype=np.float64))@np.asarray(src_ego_to_world,dtype=np.float64)

def transform_points(T, pts):
    pts=np.asarray(pts,dtype=np.float64); mat=np.asarray(T,dtype=np.float64)
    return pts@mat[:3,:3].T+mat[:3,3]

def indices_to_xyz(indices, grid=OccupancyGrid()):
    idx=np.asarray(indices,dtype=np.int64); origin=np.asarray([grid.x_min,grid.y_min,grid.z_min]); step=np.asarray(grid.voxel_size)
    return origin[None]+(idx.astype(np.float64)+0.5)*step[None]

def xyz_to_indices(points, grid=OccupancyGrid()):
    pts=np.asarray(points,dtype=np.float64); origin=np.asarray([grid.x_min,grid.y_min,grid.z_min]); step=np.asarray(grid.voxel_size)
    idx=np.floor((pts-origin[None])/step[None]).astype(np.int64); shape=np.asarray(grid.shape_xyz,dtype=np.int64); valid=((idx>=0)&(idx<shape[None])).all(1); return idx,valid

def warp_semantic_grid(semantics, src_to_dst, grid=OccupancyGrid(), free_label=17):
    sem=np.asarray(semantics)
    if tuple(sem.shape)!=tuple(grid.shape_xyz): raise ValueError("semantic grid shape mismatch")
    occ=np.argwhere(sem!=int(free_label)); out=np.full_like(sem,int(free_label))
    if not len(occ): return out
    xyz=indices_to_xyz(occ,grid); dst=transform_points(src_to_dst,xyz); idx,valid=xyz_to_indices(dst,grid)
    if not valid.any(): return out
    src=occ[valid]; vals=sem[tuple(src.T)]; idx=idx[valid]; dst=dst[valid]
    centers=indices_to_xyz(idx,grid); dist2=((dst-centers)**2).sum(1); X,Y,Z=grid.shape_xyz; flat=(idx[:,0]*Y+idx[:,1])*Z+idx[:,2]
    order=np.lexsort((dist2,flat)); fs=flat[order]; first=np.ones(len(order),dtype=bool); first[1:]=fs[1:]!=fs[:-1]; chosen=order[first]
    q=idx[chosen]; out[q[:,0],q[:,1],q[:,2]]=vals[chosen]; return out
