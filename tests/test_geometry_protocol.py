import numpy as np
from causal_se2_occ.geometry.se2 import source_center_se2_target,apply_box_centered_rigid_xy,apply_source_centered_rigid_xy
from causal_se2_occ.geometry.grid import OccupancyGrid
from causal_se2_occ.geometry.render import RasterizedRigidComponent,rasterize_rigid_component,compose_hard_a1
from causal_se2_occ.protocol import DYNAMIC_CLASS_IDS,YAW_ENABLED_CLASS_IDS
def test_source_center_equivalent():
 cs=np.array([3.2,-1.7]);a0=np.array([2.,-2.]);ah=np.array([3.5,.3]);y=.4;t=source_center_se2_target(cs,a0,ah,y);pts=np.array([[2.1,-2.2],[3.2,-1.7],[4.,-1.]]);np.testing.assert_allclose(apply_box_centered_rigid_xy(pts,a0,ah,y),apply_source_centered_rigid_xy(pts,cs,t.source_displacement_xy_m,t.yaw_rad),atol=2e-7,rtol=0)
def test_pedestrian_rule():assert 7 in DYNAMIC_CLASS_IDS and 7 not in YAW_ENABLED_CLASS_IDS
def test_renderer_and_a1():
 g=OccupancyGrid(x_min=0,y_min=0,z_min=0,voxel_size=(1.,1.,1.),shape_xyz=(8,8,4));I=np.eye(4);src=np.array([[2,2,0],[3,2,0],[2,2,1]]);o=rasterize_rigid_component(src,4,I,I,source_center_world=np.array([3.,2.5,.5]),target_center_world=np.array([4.,3.5,.5]),grid=g);assert o.source_voxel_count==3
 g2=OccupancyGrid(x_min=0,y_min=0,z_min=0,voxel_size=(1.,1.,1.),shape_xyz=(5,5,1));a=np.full(g2.shape_xyz,17,dtype=np.uint8);a[1,1,0]=4;a[2,2,0]=11;b=[RasterizedRigidComponent(4,np.array([[1,1,0],[2,2,0]]),2)];r=[RasterizedRigidComponent(4,np.array([[3,3,0]]),1),RasterizedRigidComponent(10,np.array([[3,3,0]]),1)];x=compose_hard_a1(a,b,r,dynamic_class_ids=DYNAMIC_CLASS_IDS,grid=g2);assert x[1,1,0]==17 and x[2,2,0]==11 and x[3,3,0]==10
