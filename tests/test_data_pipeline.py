from pathlib import Path
import numpy as np
import torch
from causal_se2_occ.data.cache import SourceDataset,flatten_supervised,load_cache,save_cache
from causal_se2_occ.data.features import FEATURE_DIM
from causal_se2_occ.data.prepare import prepare_causal_arrays
from causal_se2_occ.geometry.grid import OccupancyGrid
from causal_se2_occ.protocol import CACHE_SCHEMA
def _history():
 g=OccupancyGrid(x_min=-8.,y_min=-8.,z_min=-1.,voxel_size=(.4,.4,.4),shape_xyz=(40,40,8));h=[]
 for t in range(6):
  s=np.full(g.shape_xyz,17,dtype=np.uint8);x=14+t;s[x:x+2,19:21,2:4]=4;h.append(s)
 return h,[np.eye(4) for _ in range(6)],g
def test_direct_causal_preparation():
 h,p,g=_history();r=prepare_causal_arrays(h,p,grid=g);assert r['features'].shape==(1,FEATURE_DIM);assert r['local_semantic_tube'].shape==(1,6,20,20);assert r['frame_motion_features'].shape==(1,6,5);assert r['target_source_mask_tube'].shape==(1,6,20,20);assert int(r['source_class_id'][0])==4;assert not any(k.startswith('future_') for k in r)
def test_cache_roundtrip(tmp_path:Path):
 h,p,g=_history();c=prepare_causal_arrays(h,p,grid=g);c.pop('_current_components');c.pop('_velocities_world');v=torch.ones(1,6,dtype=torch.bool);rec={'sample_id':'s','scene_name':'scene','history_tokens':tuple('abcdef'),'t0_token':'f','future_tokens':tuple(f'f{i}' for i in range(6)),**c,'supervised_source':torch.ones(1,dtype=torch.bool),'existence':torch.ones(1,6),'target_source_displacement_xy_m':torch.zeros(1,6,2),'target_source_residual_xy_m':torch.zeros(1,6,2),'target_yaw_rad':torch.zeros(1,6),'yaw_label_valid':v,'se2_target_valid':v,'yaw_enabled':torch.ones(1,dtype=torch.bool)};path=tmp_path/'c.pt';save_cache(path,[rec],{'split':'smoke'});meta,records=load_cache(path);assert meta['cache_schema']==CACHE_SCHEMA;flat=flatten_supervised(records);assert len(SourceDataset(flat))==1
