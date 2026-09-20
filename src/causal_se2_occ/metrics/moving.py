from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from ..geometry.grid import OccupancyGrid,quaternion_yaw
from ..protocol import DYNAMIC_CLASS_IDS,MOVING_SPEED_THRESHOLD_MPS,MOVING_BOX_MARGIN_M,category_to_dynamic_class
@dataclass(frozen=True)
class Box3D:
    center_xyz:tuple; size_lwh:tuple; yaw:float
def raster_box(box,grid=OccupancyGrid(),margin=.5):
    X,Y,Z=grid.shape_xyz; vx,vy,vz=grid.voxel_size; xs=grid.x_min+(np.arange(X)+.5)*vx; ys=grid.y_min+(np.arange(Y)+.5)*vy; zs=grid.z_min+(np.arange(Z)+.5)*vz; cx,cy,cz=box.center_xyz; l,w,h=box.size_lwh; radius=.5*math.hypot(l+2*margin,w+2*margin); xi=np.where((xs>=cx-radius)&(xs<=cx+radius))[0]; yi=np.where((ys>=cy-radius)&(ys<=cy+radius))[0]; zi=np.where((zs>=cz-h/2-margin)&(zs<=cz+h/2+margin))[0]; out=np.zeros(grid.shape_xyz,dtype=bool)
    if not len(xi) or not len(yi) or not len(zi):return out
    xx,yy=np.meshgrid(xs[xi],ys[yi],indexing='xy'); dx,dy=xx-cx,yy-cy;c,s=math.cos(box.yaw),math.sin(box.yaw); inside=(np.abs(c*dx+s*dy)<=l/2+margin)&(np.abs(-s*dx+c*dy)<=w/2+margin); ry,cxidx=np.where(inside)
    for z in zi:out[xi[cxidx],yi[ry],z]=True
    return out
def moving_support_for_nuscenes(nusc,t0_token,th_token,dt_s,*,grid=OccupancyGrid(),threshold=.5,margin=.5):
    def amap(tok):
        s=nusc.get('sample',str(tok)); o={}
        for at in s['anns']:
            a=nusc.get('sample_annotation',at); cid=category_to_dynamic_class(a['category_name']);
            if cid is not None:o[str(a['instance_token'])]=(a,int(cid))
        return o
    def pose(tok):
        from ..geometry.grid import pose_matrix
        s=nusc.get('sample',str(tok)); sd=nusc.get('sample_data',s['data']['LIDAR_TOP']); e=nusc.get('ego_pose',sd['ego_pose_token']); return pose_matrix(e['translation'],e['rotation'])
    a0,ah=amap(t0_token),amap(th_token); T=pose(th_token); inv=np.linalg.inv(T); ey=math.atan2(T[1,0],T[0,0]); support=np.zeros(grid.shape_xyz,dtype=bool)
    for tok in sorted(set(a0)&set(ah)):
        r0,cid0=a0[tok]; rh,cid=ah[tok]; c0=np.asarray(r0['translation']); ch=np.asarray(rh['translation']);
        if np.linalg.norm(ch[:2]-c0[:2])/float(dt_s)<threshold:continue
        def box(r):
            cw=np.asarray(r['translation']); ce=(inv@np.r_[cw,1.])[:3]; w,l,h=r['size']; y=(quaternion_yaw(r['rotation'])-ey+math.pi)%(2*math.pi)-math.pi; return Box3D(tuple(ce), (float(l),float(w),float(h)),float(y))
        support|=raster_box(box(r0),grid,margin)|raster_box(box(rh),grid,margin)
    return support
