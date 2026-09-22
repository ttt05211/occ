#!/usr/bin/env python3
"""Frozen-boundary runtime benchmark with real-window exactness gates."""
from __future__ import annotations
import argparse,json,math,platform,time
from pathlib import Path
import numpy as np
import torch
from causal_se2_occ.checkpoint import load_model_checkpoint
from causal_se2_occ.data.cache import load_cache
from causal_se2_occ.data.nuscenes import NuScenesOcc3D,WindowTokens
from causal_se2_occ.geometry.grid import OccupancyGrid,indices_to_xyz,transform_points,relative_transform,xyz_to_indices
from causal_se2_occ.geometry.render import RasterizedRigidComponent,rasterize_rigid_component,compose_hard_a1
from causal_se2_occ.inference import t0_xy_to_world_preserve_source_z
from causal_se2_occ.priors.strong_kta import StrongKTAConfig,extract_instances,match_instances,strong_kta_sequence,inverse_warp
from causal_se2_occ.protocol import DYNAMIC_CLASS_IDS,YAW_ENABLED_CLASS_IDS
from causal_se2_occ.runtime.fastpath import extract_instances_cropped_exact,component_lists_equal,majority_fill_sparse_5x5x1,majority_fill_cuda_exact,inverse_warp_sequence_cuda_exact,compose_hard_a1_fast_exact,baseline_clear_flat_indices
from causal_se2_occ.io import prepare_output_file
GRID=OccupancyGrid();CFG=StrongKTAConfig()
def sync(d):
 if d.type=='cuda':torch.cuda.synchronize(d)
def dedup(idx):
 idx=np.asarray(idx,dtype=np.int64)
 if not len(idx):return idx.reshape(0,3)
 _,Y,Z=GRID.shape_xyz;flat=(idx[:,0]*Y+idx[:,1])*Z+idx[:,2];_,first=np.unique(flat,return_index=True);return idx[np.sort(first)]
def precompute_world(cur,pose):return [transform_points(pose,indices_to_xyz(c['voxel_indices'],GRID)) for c in cur]
def strong_fast(s,d):
 sem0=s['cur_sem'];cur=s['current'];vel=s['vel'];pose=s['cur_pose'];fut=s['future_poses'];world=s['world'];dyn=np.isin(sem0,np.asarray(DYNAMIC_CLASS_IDS,dtype=sem0.dtype));static=sem0.copy();static[dyn]=17;covered=np.zeros_like(dyn)
 for c in cur:q=c['voxel_indices'];covered[q[:,0],q[:,1],q[:,2]]=True
 rest=np.argwhere(dyn&~covered);rw=transform_points(pose,indices_to_xyz(rest,GRID)) if len(rest) else np.zeros((0,3));rl=sem0[tuple(rest.T)] if len(rest) else np.zeros((0,),dtype=sem0.dtype);seq=[relative_transform(pose,p) for p in fut];warped=inverse_warp_sequence_cuda_exact(static,seq,grid=GRID,free_label=17,device=d) if d.type=='cuda' else [inverse_warp(static,T,GRID,17) for T in seq];outputs=[];bases=[]
 for hi,p in enumerate(fut):
  dst,known=warped[hi];out=majority_fill_cuda_exact(dst,~known,device=d) if d.type=='cuda' else majority_fill_sparse_5x5x1(dst,~known);comps=[];parts=[];labs=[];slices=[];cursor=0
  for j,c in enumerate(cur):
   pts=world[j];v=np.asarray(vel.get(j,np.zeros(3)));m=pts+v[None]*(hi+1)*.5;parts.append(m);labs.append(np.full(len(m),int(c['class_id']),dtype=sem0.dtype));slices.append(slice(cursor,cursor+len(m)));cursor+=len(m)
  if len(rw):parts.append(rw);labs.append(rl)
  if parts:
   pts=transform_points(np.linalg.inv(p),np.concatenate(parts));idx,ok=xyz_to_indices(pts,GRID);q=idx[ok];out[q[:,0],q[:,1],q[:,2]]=np.concatenate(labs)[ok]
   for c,sl in zip(cur,slices):comps.append(RasterizedRigidComponent(int(c['class_id']),idx[sl][ok[sl]],len(c['voxel_indices'])))
  outputs.append(out.astype(np.uint8,copy=False));bases.append(comps)
 return outputs,bases
def model_forward(m,x,d):
 with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=d.type=='cuda'):return m(x['features'],x['tube'],x['kta'],x['frame_motion'],x['source_mask'])
def render_fast(s,prior,base,o):
 res=o['residual_xy_m'].float().cpu().numpy();yaw=o['yaw_delta_rad'].float().cpu().numpy();preds=[]
 for hi,p in enumerate(s['future_poses']):
  repl=[]
  for i,c in enumerate(s['current']):
   sc=np.asarray(c['centroid_world'],dtype=np.float64);xy=s['rec']['anchors_xy_t0_m'][i,hi].numpy()+res[i,hi];tc=t0_xy_to_world_preserve_source_z(xy,sc,s['cur_pose']);th=float(yaw[i,hi]) if int(c['class_id']) in YAW_ENABLED_CLASS_IDS else 0.;pts=s['world'][i];co,si=math.cos(th),math.sin(th);rel=pts[:,:2]-sc[None,:2];m=pts.copy();m[:,0]=tc[0]+co*rel[:,0]-si*rel[:,1];m[:,1]=tc[1]+si*rel[:,0]+co*rel[:,1];fp=transform_points(np.linalg.inv(p),m);idx,ok=xyz_to_indices(fp,GRID);repl.append(RasterizedRigidComponent(int(c['class_id']),dedup(idx[ok]),len(c['voxel_indices'])))
  preds.append(compose_hard_a1_fast_exact(prior[hi],base[hi],repl,dynamic_class_ids=DYNAMIC_CLASS_IDS,free_label=17,grid=GRID,precomputed_clear_flat_indices=baseline_clear_flat_indices(base[hi],grid=GRID)))
 return np.stack(preds)
def prepare_state(rec,src,d):
 w=WindowTokens(str(rec['scene_name']),tuple(rec['history_tokens']),str(rec['t0_token']),tuple(rec['future_tokens']));cs=src.semantics(w.scene_name,w.t0_token);ps=src.semantics(w.scene_name,w.history_tokens[-2]);cp=np.asarray(src.pose(w.t0_token));pp=np.asarray(src.pose(w.history_tokens[-2]));fp=[np.asarray(src.pose(t)) for t in w.future_tokens];cur=extract_instances_cropped_exact(cs,cp,grid=GRID,cfg=CFG);prev=extract_instances_cropped_exact(ps,pp,grid=GRID,cfg=CFG);vel=match_instances(prev,cur,.5,max_speed_mps=CFG.max_match_speed_mps)
 if [int(c['class_id']) for c in cur]!=[int(x) for x in rec['source_class_id'].tolist()]:raise RuntimeError('source order mismatch')
 x={'features':rec['features'].float().to(d),'tube':rec['local_semantic_tube'].to(d),'kta':rec['kta_displacement_xy_m'].float().to(d),'frame_motion':rec['frame_motion_features'].float().to(d),'source_mask':rec['target_source_mask_tube'].to(d)}
 return {'rec':rec,'cur_sem':cs,'prev_sem':ps,'cur_pose':cp,'prev_pose':pp,'future_poses':fp,'current':cur,'previous':prev,'vel':vel,'world':precompute_world(cur,cp),'x':x}
def exactness(model,s,d):
 rc=extract_instances(s['cur_sem'],s['cur_pose'],grid=GRID,cfg=CFG);rp=extract_instances(s['prev_sem'],s['prev_pose'],grid=GRID,cfg=CFG)
 if not component_lists_equal(rc,s['current']) or not component_lists_equal(rp,s['previous']):raise RuntimeError('fast component extraction mismatch')
 ref,_,_=strong_kta_sequence([s['prev_sem'],s['cur_sem']],[s['prev_pose'],s['cur_pose']],s['future_poses'],grid=GRID,cfg=CFG);fast,base=strong_fast(s,d)
 if not np.array_equal(ref,np.stack(fast)):raise RuntimeError('fast Strong/KTA mismatch')
 o=model_forward(model,s['x'],d);fp=render_fast(s,fast,base,o);res=o['residual_xy_m'].float().cpu().numpy();yaw=o['yaw_delta_rad'].float().cpu().numpy();rp=[]
 for hi,p in enumerate(s['future_poses']):
  rb=[];rr=[]
  for i,c in enumerate(s['current']):
   sc=np.asarray(c['centroid_world']);v=np.asarray(s['vel'].get(i,np.zeros(3)));rb.append(rasterize_rigid_component(c['voxel_indices'],c['class_id'],s['cur_pose'],p,source_center_world=sc,target_center_world=sc+v*(hi+1)*.5,yaw_delta_rad=0.,grid=GRID));xy=s['rec']['anchors_xy_t0_m'][i,hi].numpy()+res[i,hi];tc=t0_xy_to_world_preserve_source_z(xy,sc,s['cur_pose']);th=float(yaw[i,hi]) if int(c['class_id']) in YAW_ENABLED_CLASS_IDS else 0.;rr.append(rasterize_rigid_component(c['voxel_indices'],c['class_id'],s['cur_pose'],p,source_center_world=sc,target_center_world=tc,yaw_delta_rad=th,grid=GRID))
  rp.append(compose_hard_a1(ref[hi],rb,rr,dynamic_class_ids=DYNAMIC_CLASS_IDS,grid=GRID))
 if not np.array_equal(fp,np.stack(rp)):raise RuntimeError('fast SE2 raster/A1 mismatch')
def timed(fn,d):sync(d);t=time.perf_counter();v=fn();sync(d);return (time.perf_counter()-t)*1000,v
def summary(x):
 a=np.asarray(x,dtype=float);return {'n':len(a),'mean_ms':float(a.mean()),'std_ms':float(a.std(ddof=1)) if len(a)>1 else 0.,'median_ms':float(np.median(a)),'p95_ms':float(np.quantile(a,.95))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataroot',required=True);p.add_argument('--val-cache',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cuda');p.add_argument('--warmup-windows',type=int,default=20);p.add_argument('--measure-windows',type=int,default=200);p.add_argument('--exactness-windows',type=int,default=8);p.add_argument('--seed',type=int,default=20260918);p.add_argument('--output',required=True);a=p.parse_args();prepare_output_file(a.output);d=torch.device(a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu');_,records=load_cache(a.val_cache);rng=np.random.default_rng(a.seed);n=min(len(records),a.warmup_windows+a.measure_windows+a.exactness_windows);selected=[records[int(i)] for i in np.sort(rng.choice(len(records),size=n,replace=False))];src=NuScenesOcc3D(a.dataroot,split='val');ck,model=load_model_checkpoint(a.checkpoint,d);states=[prepare_state(r,src,d) for r in selected]
 for i,s in enumerate(states[:a.exactness_windows]):exactness(model,s,d);print(f'exactness {i+1}/{a.exactness_windows}: PASS')
 start=a.exactness_windows
 for s in states[start:start+a.warmup_windows]:pr,ba=strong_fast(s,d);o=model_forward(model,s['x'],d);render_fast(s,pr,ba,o)
 measured=states[start+a.warmup_windows:start+a.warmup_windows+a.measure_windows];rows={k:[] for k in ['learned_forward','strong_kta_prior','se2_render_a1','full_six_frame_generation','source_extract_match']}
 for s in measured:
  ms,o=timed(lambda:model_forward(model,s['x'],d),d);rows['learned_forward'].append(ms);ms,(pr,ba)=timed(lambda:strong_fast(s,d),d);rows['strong_kta_prior'].append(ms);o=model_forward(model,s['x'],d);ms,_=timed(lambda:render_fast(s,pr,ba,o),d);rows['se2_render_a1'].append(ms);ms,_=timed(lambda:(lambda z:render_fast(s,*z,model_forward(model,s['x'],d)))(strong_fast(s,d)),d);rows['full_six_frame_generation'].append(ms);t=time.perf_counter();c=extract_instances_cropped_exact(s['cur_sem'],s['cur_pose'],grid=GRID,cfg=CFG);p0=extract_instances_cropped_exact(s['prev_sem'],s['prev_pose'],grid=GRID,cfg=CFG);match_instances(p0,c,.5,max_speed_mps=CFG.max_match_speed_mps);rows['source_extract_match'].append((time.perf_counter()-t)*1000)
 sm={k:summary(v) for k,v in rows.items()};full=sm['full_six_frame_generation']['mean_ms'];result={'protocol':'standalone_frozen_boundary_runtime_v1','checkpoint_epoch':ck.get('epoch'),'global_step':ck.get('global_step'),'hardware':torch.cuda.get_device_name(d) if d.type=='cuda' else (platform.processor() or 'CPU'),'precision':'BF16 autocast' if d.type=='cuda' else 'FP32','seed':a.seed,'warmup_windows':a.warmup_windows,'measure_windows':len(measured),'exactness_windows':a.exactness_windows,'timings':sm,'windows_per_s':1000/full,'future_frame_amortized_fps':6000/full,'excluded':'disk I/O, GT labels/support, metrics, cache construction'};Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
