#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
from causal_se2_occ.data.nuscenes import NuScenesOcc3D
from causal_se2_occ.data.prepare import prepare_training_record
from causal_se2_occ.data.cache import save_cache
EXPECTED={'train':(700,20430),'val':(150,4369)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataroot',required=True);p.add_argument('--split',choices=['train','val'],required=True);p.add_argument('--output',required=True);p.add_argument('--max-windows',type=int,default=0);p.add_argument('--no-count-check',action='store_true');a=p.parse_args();src=NuScenesOcc3D(a.dataroot,split=a.split);wins=list(src.iter_windows(max_windows=(a.max_windows or None)));scenes=sorted({w.scene_name for w in wins});formal=(a.max_windows==0)
 if formal and not a.no_count_check:
  es,ew=EXPECTED[a.split]
  if (len(scenes),len(wins))!=(es,ew):raise RuntimeError(f'{a.split} split expected {es} scenes/{ew} windows; got {len(scenes)}/{len(wins)}')
 records=[]
 for i,w in enumerate(wins,1):
  records.append(prepare_training_record(src,w))
  if i==1 or i%100==0 or i==len(wins):print(f'prepare {i}/{len(wins)} {w.scene_name} {w.t0_token}',flush=True)
 scene_json=Path(a.dataroot).resolve()/'v1.0-trainval'/'scene.json';h=hashlib.sha256(scene_json.read_bytes()).hexdigest() if scene_json.exists() else None
 meta={'split':a.split,'scene_count':len(scenes),'scene_names':scenes,'eligible_windows':len(wins),'history_frames':6,'future_frames':6,'frame_dt_s':0.5,'grid_xyz':[200,200,16],'voxel_size_m':[0.4,0.4,0.4],'range_m':[-40,-40,-1,40,40,5.4],'class_count':18,'free_label':17,'patch_size_m':16.0,'patch_resolution_m':0.8,'source_gt_match_max_distance_m':4.0,'gt_usage':'future semantics/instances/yaw are labels/evaluation only; causal tensors use history occupancy only','future_ego_pose_protocol':'future ego poses are required inference inputs for KTA/ego transport','formal_count_check':formal and not a.no_count_check,'source_dataset':'nuScenes v1.0-trainval + Occ3D labels','source_scene_json_sha256':h};save_cache(a.output,records,meta);print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
