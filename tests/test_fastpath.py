import numpy as np
from causal_se2_occ.geometry.grid import OccupancyGrid
from causal_se2_occ.geometry.render import RasterizedRigidComponent,compose_hard_a1
from causal_se2_occ.priors.strong_kta import StrongKTAConfig,majority_fill,extract_instances
from causal_se2_occ.runtime.fastpath import majority_fill_sparse_5x5x1,extract_instances_cropped_exact,component_lists_equal,compose_hard_a1_fast_exact,baseline_clear_flat_indices
from causal_se2_occ.protocol import DYNAMIC_CLASS_IDS
def test_sparse_majority_exact():
 rng=np.random.default_rng(7)
 for _ in range(20):
  sem=rng.integers(0,18,size=(9,10,3),dtype=np.uint8);unknown=rng.random(sem.shape)<.18;assert np.array_equal(majority_fill(sem,unknown),majority_fill_sparse_5x5x1(sem,unknown))
def test_cropped_components_exact():
 g=OccupancyGrid(x_min=0,y_min=0,z_min=0,voxel_size=(1.,1.,1.),shape_xyz=(20,20,4));s=np.full(g.shape_xyz,17,dtype=np.uint8);s[2:5,3:6,1]=4;s[10:13,11:13,0:2]=7;cfg=StrongKTAConfig();assert component_lists_equal(extract_instances(s,np.eye(4),grid=g,cfg=cfg),extract_instances_cropped_exact(s,np.eye(4),grid=g,cfg=cfg))
def test_fast_a1_exact():
 g=OccupancyGrid(x_min=0,y_min=0,z_min=0,voxel_size=(1.,1.,1.),shape_xyz=(6,6,2));a=np.full(g.shape_xyz,17,dtype=np.uint8);a[1,1,0]=4;a[2,2,0]=7;b=[RasterizedRigidComponent(4,np.array([[1,1,0]]),1),RasterizedRigidComponent(7,np.array([[2,2,0]]),1)];r=[RasterizedRigidComponent(4,np.array([[3,3,0]]),1),RasterizedRigidComponent(10,np.array([[3,3,0],[4,4,0]]),2)];ref=compose_hard_a1(a,b,r,dynamic_class_ids=DYNAMIC_CLASS_IDS,grid=g);fast=compose_hard_a1_fast_exact(a,b,r,dynamic_class_ids=DYNAMIC_CLASS_IDS,grid=g,precomputed_clear_flat_indices=baseline_clear_flat_indices(b,grid=g));assert np.array_equal(ref,fast)
