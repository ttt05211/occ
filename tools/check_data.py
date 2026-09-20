#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from causal_se2_occ.data.nuscenes import NuScenesOcc3D
EXPECTED={'train':(700,20430),'val':(150,4369)}
def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description='Validate nuScenes + Occ3D layout and frozen 6+6 eligible-window counts.');p.add_argument('--dataroot',required=True);p.add_argument('--check-labels',action=argparse.BooleanOptionalAction,default=True);a=p.parse_args();root=Path(a.dataroot).resolve();report={'dataroot':str(root),'splits':{}};scene_json=root/'v1.0-trainval'/'scene.json';report['scene_json_sha256']=sha256(scene_json) if scene_json.exists() else None;split_sets={}
    for split in ('train','val'):
        src=NuScenesOcc3D(root,split=split);wins=list(src.iter_windows());scenes=sorted({w.scene_name for w in wins});split_sets[split]=set(scenes);missing=[]
        if a.check_labels:
            for w in wins:
                for tok in (*w.history_tokens,*w.future_tokens):
                    q=src.label_path(w.scene_name,tok)
                    if not q.exists():missing.append(str(q))
                    if len(missing)>=20:break
                if len(missing)>=20:break
        es,ew=EXPECTED[split];report['splits'][split]={'scene_count':len(scenes),'eligible_windows':len(wins),'expected_scene_count':es,'expected_eligible_windows':ew,'counts_match':(len(scenes),len(wins))==(es,ew),'first_missing_occ3d_paths':missing}
    overlap=sorted(split_sets['train']&split_sets['val']);report['train_val_scene_overlap']=overlap;report['pass']=all(x['counts_match'] and not x['first_missing_occ3d_paths'] for x in report['splits'].values()) and not overlap;print(json.dumps(report,indent=2))
    if not report['pass']:raise SystemExit(2)
if __name__=='__main__':main()
