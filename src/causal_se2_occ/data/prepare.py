from __future__ import annotations
from typing import Mapping, Sequence
import numpy as np
import torch
from ..geometry.grid import OccupancyGrid, quaternion_yaw
from ..geometry.se2 import relative_yaw_in_t0, source_center_se2_target
from ..priors.strong_kta import StrongKTAConfig, extract_instances, match_instances
from ..protocol import FUTURE_FRAMES, HISTORY_FRAMES, YAW_ENABLED_CLASS_IDS, category_to_dynamic_class
from .features import backward_component_tracks,build_source_features,frame_motion_features_from_flat,target_source_mask_from_tube,world_points_to_t0,world_vec_to_t0
from .tubes import DEFAULT_PATCH_RESOLUTION_M,DEFAULT_PATCH_SIZE_M,build_local_semantic_tubes,history_offsets_from_features

def _annotation_rows(nusc,sample_token):
    sample=nusc.get("sample",str(sample_token));rows=[]
    for ann_token in sample["anns"]:
        ann=nusc.get("sample_annotation",ann_token);cid=category_to_dynamic_class(ann["category_name"])
        if cid is not None: rows.append({"instance_token":str(ann["instance_token"]),"class_id":int(cid),"center_world":np.asarray(ann["translation"],dtype=np.float64),"yaw_world":float(quaternion_yaw(ann["rotation"]))})
    rows.sort(key=lambda r:(int(r["class_id"]),str(r["instance_token"])));return rows
def _annotation_map(nusc,sample_token): return {str(r["instance_token"]):r for r in _annotation_rows(nusc,sample_token)}
def _match_sources_to_annotations(components,annotations,*,max_distance_m=4.):
    pairs=[]
    for ci,comp in enumerate(components):
        cc=np.asarray(comp["centroid_world"],dtype=np.float64)
        for ai,ann in enumerate(annotations):
            if int(comp["class_id"])==int(ann["class_id"]):
                d=float(np.linalg.norm(cc[:2]-np.asarray(ann["center_world"],dtype=np.float64)[:2]))
                if d<=float(max_distance_m):pairs.append((d,ci,ai,str(ann["instance_token"])))
    pairs.sort(key=lambda x:(x[0],x[1],x[2],x[3]));uc=set();ua=set();tokens=[None]*len(components)
    for _,ci,ai,tok in pairs:
        if ci not in uc and ai not in ua: uc.add(ci);ua.add(ai);tokens[ci]=tok
    return tokens

def prepare_causal_arrays(history_occ,history_poses,*,grid=OccupancyGrid(),frame_dt_s=.5,free_label=17,patch_size_m=DEFAULT_PATCH_SIZE_M,patch_resolution_m=DEFAULT_PATCH_RESOLUTION_M):
    if len(history_occ)!=HISTORY_FRAMES or len(history_poses)!=HISTORY_FRAMES:raise ValueError("frozen causal preparation requires six history frames")
    cfg=StrongKTAConfig(free_label=int(free_label));components_by_frame=[extract_instances(np.asarray(s),np.asarray(p),grid=grid,cfg=cfg) for s,p in zip(history_occ,history_poses)];current=components_by_frame[-1];previous=components_by_frame[-2];velocities=match_instances(previous,current,float(frame_dt_s),max_speed_mps=cfg.max_match_speed_mps);tracks,track_valid=backward_component_tracks(components_by_frame,frame_dt_s=float(frame_dt_s),max_speed_mps=cfg.max_match_speed_mps);t0_pose=np.asarray(history_poses[-1],dtype=np.float64);features_np=build_source_features(current,velocities,tracks,track_valid,t0_pose,frame_dt_s=float(frame_dt_s),grid=grid);features=torch.from_numpy(features_np);n=len(current);source_xy=np.zeros((n,2),dtype=np.float32);kta=np.zeros((n,FUTURE_FRAMES,2),dtype=np.float32);anchors=np.zeros_like(kta)
    for i,comp in enumerate(current):
        cw=np.asarray(comp["centroid_world"],dtype=np.float64);ct=world_points_to_t0(cw[None],t0_pose)[0,:2];source_xy[i]=ct.astype(np.float32);vw=np.asarray(velocities.get(i,np.zeros(3)),dtype=np.float64);vt=world_vec_to_t0(vw,t0_pose)[:2]
        for h in range(FUTURE_FRAMES):kd=vt*((h+1)*float(frame_dt_s));kta[i,h]=kd.astype(np.float32);anchors[i,h]=(ct+kd).astype(np.float32)
    offsets=history_offsets_from_features(features_np);tube=torch.from_numpy(build_local_semantic_tubes(history_occ,history_poses,source_xy,offsets,track_valid,grid=grid,free_label=int(free_label),patch_size_m=float(patch_size_m),patch_resolution_m=float(patch_resolution_m)));class_id=torch.as_tensor([int(c["class_id"]) for c in current],dtype=torch.long);track_t=torch.from_numpy(track_valid);fm=frame_motion_features_from_flat(features);sm=target_source_mask_from_tube(tube,class_id,track_t,features,patch_resolution_m=float(patch_resolution_m))
    return {"features":features.float(),"source_class_id":class_id,"source_voxel_count":torch.as_tensor([int(c["voxel_count"]) for c in current],dtype=torch.long),"source_voxel_indices":tuple(torch.from_numpy(np.asarray(c["voxel_indices"],dtype=np.int64)) for c in current),"source_centroid_world":torch.as_tensor(np.stack([np.asarray(c["centroid_world"]) for c in current]) if current else np.zeros((0,3)),dtype=torch.float64),"source_centroid_xy_t0_m":torch.from_numpy(source_xy),"track_valid":track_t,"kta_displacement_xy_m":torch.from_numpy(kta),"anchors_xy_t0_m":torch.from_numpy(anchors),"local_semantic_tube":tube.to(torch.uint8),"frame_motion_features":fm.float(),"target_source_mask_tube":sm.to(torch.uint8),"_current_components":current,"_velocities_world":velocities}

def prepare_training_record(source,window,*,grid=OccupancyGrid(),frame_dt_s=.5,free_label=17,match_max_distance_m=4.,patch_size_m=DEFAULT_PATCH_SIZE_M,patch_resolution_m=DEFAULT_PATCH_RESOLUTION_M):
    history_occ=[source.semantics(window.scene_name,t) for t in window.history_tokens];history_poses=[np.asarray(source.pose(t),dtype=np.float64) for t in window.history_tokens];causal=prepare_causal_arrays(history_occ,history_poses,grid=grid,frame_dt_s=frame_dt_s,free_label=free_label,patch_size_m=patch_size_m,patch_resolution_m=patch_resolution_m);current=causal.pop("_current_components");causal.pop("_velocities_world");anns0=_annotation_rows(source.nusc,window.t0_token);tokens=_match_sources_to_annotations(current,anns0,max_distance_m=match_max_distance_m);a0map={r["instance_token"]:r for r in anns0};fmaps=[_annotation_map(source.nusc,t) for t in window.future_tokens];sx=causal["source_centroid_xy_t0_m"].numpy().astype(np.float64);kta=causal["kta_displacement_xy_m"].numpy().astype(np.float64);cid=causal["source_class_id"].numpy();n=len(current);td=np.zeros((n,6,2),np.float32);tr=np.zeros_like(td);ty=np.zeros((n,6),np.float32);exist=np.zeros((n,6),np.float32);yv=np.zeros((n,6),bool);sv=np.zeros((n,6),bool);sup=np.asarray([t is not None for t in tokens],bool);ye=np.isin(cid,np.asarray(YAW_ENABLED_CLASS_IDS));t0=np.asarray(history_poses[-1])
    for i,tok in enumerate(tokens):
        if tok is None:continue
        ann0=a0map.get(str(tok))
        if ann0 is None:continue
        a0=world_points_to_t0(np.asarray(ann0["center_world"])[None],t0)[0,:2];yaw0=float(ann0["yaw_world"])
        for h in range(6):
            ahrow=fmaps[h].get(str(tok))
            if ahrow is None:continue
            ah=world_points_to_t0(np.asarray(ahrow["center_world"])[None],t0)[0,:2];raw=relative_yaw_in_t0(yaw0,float(ahrow["yaw_world"]),t0);eff=raw if bool(ye[i]) else 0.;target=source_center_se2_target(sx[i],a0,ah,eff);td[i,h]=target.source_displacement_xy_m;tr[i,h]=(target.source_displacement_xy_m.astype(np.float64)-kta[i,h]).astype(np.float32);ty[i,h]=np.float32(raw);exist[i,h]=1.;yv[i,h]=True;sv[i,h]=True
    return {"sample_id":str(window.t0_token),"scene_name":str(window.scene_name),"history_tokens":tuple(window.history_tokens),"t0_token":str(window.t0_token),"future_tokens":tuple(window.future_tokens),**causal,"source_instance_token":tuple(tokens),"supervised_source":torch.from_numpy(sup),"existence":torch.from_numpy(exist),"target_source_displacement_xy_m":torch.from_numpy(td),"target_source_residual_xy_m":torch.from_numpy(tr),"target_yaw_rad":torch.from_numpy(ty),"yaw_label_valid":torch.from_numpy(yv),"se2_target_valid":torch.from_numpy(sv),"yaw_enabled":torch.from_numpy(ye)}
