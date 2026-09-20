from __future__ import annotations
import numpy as np
import torch
from .data.prepare import prepare_causal_arrays
from .data.features import world_points_to_t0
from .geometry.grid import OccupancyGrid,transform_points
from .geometry.render import rasterize_rigid_component,compose_hard_a1
from .priors.strong_kta import strong_kta_sequence
from .protocol import DYNAMIC_CLASS_IDS,YAW_ENABLED_CLASS_IDS

def t0_xy_to_world_preserve_source_z(xy_t0,source_center_world,t0_ego_to_world):
    # XY is defined in frozen t0 ego. Transform a t0 ground-plane point to world, then preserve source z.
    p=np.asarray([float(xy_t0[0]),float(xy_t0[1]),0.0],dtype=np.float64); w=transform_points(t0_ego_to_world,p[None])[0]; w[2]=float(np.asarray(source_center_world)[2]); return w

def forecast_window(model,history_occ,history_poses,future_poses,*,device='cpu',amp_bf16=True,grid=OccupancyGrid(),frame_dt_s=.5,free_label=17):
    anchor,current,vel=strong_kta_sequence(history_occ,history_poses,future_poses,frame_dt_s=frame_dt_s,grid=grid); causal=prepare_causal_arrays(history_occ,history_poses,grid=grid,frame_dt_s=frame_dt_s,free_label=free_label); fields=['features','local_semantic_tube','kta_displacement_xy_m','frame_motion_features','target_source_mask_tube']; x={k:causal[k].to(device) for k in fields}; model=model.to(device).eval()
    with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=(str(device).startswith('cuda') and amp_bf16)):
        o=model(x['features'].float(),x['local_semantic_tube'],x['kta_displacement_xy_m'].float(),x['frame_motion_features'].float(),x['target_source_mask_tube'])
    res=o['residual_xy_m'].float().cpu().numpy(); py=o['yaw_delta_rad'].float().cpu().numpy(); out=[]; t0=np.asarray(history_poses[-1])
    for h,fpose in enumerate(future_poses):
        base=[]; repl=[]; dt=(h+1)*frame_dt_s
        for i,comp in enumerate(current):
            sc=np.asarray(comp['centroid_world'],dtype=np.float64); v=np.asarray(vel.get(i,np.zeros(3)),dtype=np.float64); base.append(rasterize_rigid_component(comp['voxel_indices'],comp['class_id'],t0,fpose,source_center_world=sc,target_center_world=sc+v*dt,yaw_delta_rad=0.,grid=grid)); xy=causal['anchors_xy_t0_m'][i,h].numpy()+res[i,h]; tc=t0_xy_to_world_preserve_source_z(xy,sc,t0); yaw=float(py[i,h]) if int(comp['class_id']) in YAW_ENABLED_CLASS_IDS else 0.; repl.append(rasterize_rigid_component(comp['voxel_indices'],comp['class_id'],t0,fpose,source_center_world=sc,target_center_world=tc,yaw_delta_rad=yaw,grid=grid))
        out.append(compose_hard_a1(anchor[h],base,repl,dynamic_class_ids=DYNAMIC_CLASS_IDS,free_label=free_label,grid=grid))
    return np.stack(out),{'residual_xy_m':res,'yaw_delta_rad':py,'existence_logits':o['existence_logits'].float().cpu().numpy(),'note':'existence logits are not used as deployment gates'}
