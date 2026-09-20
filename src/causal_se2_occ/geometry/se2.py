from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch

@dataclass(frozen=True)
class SE2Target:
    source_displacement_xy_m: np.ndarray
    yaw_rad: float

def wrap_angle_np(angle): return float((float(angle)+math.pi)%(2*math.pi)-math.pi)
def wrap_angle_tensor(angle): return torch.atan2(torch.sin(angle),torch.cos(angle))
def rot2(angle):
    c,s=math.cos(float(angle)),math.sin(float(angle)); return np.asarray([[c,-s],[s,c]],dtype=np.float64)

def source_center_se2_target(source_center_xy,gt_box_center_t0_xy,gt_box_center_future_xy,relative_yaw_rad):
    cs=np.asarray(source_center_xy,dtype=np.float64); a0=np.asarray(gt_box_center_t0_xy,dtype=np.float64); ah=np.asarray(gt_box_center_future_xy,dtype=np.float64)
    if cs.shape!=(2,) or a0.shape!=(2,) or ah.shape!=(2,): raise ValueError("centers must be XY")
    yaw=wrap_angle_np(relative_yaw_rad); R=rot2(yaw); off=cs-a0; d=(ah-a0)+(R@off-off); return SE2Target(d.astype(np.float32),yaw)

def apply_box_centered_rigid_xy(points_xy,a0,ah,yaw):
    pts=np.asarray(points_xy,dtype=np.float64); R=rot2(yaw); return (pts-np.asarray(a0)[None])@R.T+np.asarray(ah)[None]
def apply_source_centered_rigid_xy(points_xy,cs,d,yaw):
    pts=np.asarray(points_xy,dtype=np.float64); R=rot2(yaw); return (pts-np.asarray(cs)[None])@R.T+np.asarray(cs)[None]+np.asarray(d)[None]
def heading_in_t0_from_world_yaw(yaw_world,t0_ego_to_world):
    v=np.asarray([math.cos(float(yaw_world)),math.sin(float(yaw_world)),0.0]); R=np.asarray(t0_ego_to_world,dtype=np.float64)[:3,:3]; q=v@R
    if np.linalg.norm(q[:2])<1e-8: raise ValueError("degenerate heading")
    return math.atan2(float(q[1]),float(q[0]))
def relative_yaw_in_t0(yaw0_world,yawh_world,t0_ego_to_world):
    return wrap_angle_np(heading_in_t0_from_world_yaw(yawh_world,t0_ego_to_world)-heading_in_t0_from_world_yaw(yaw0_world,t0_ego_to_world))
