"""Bit-exact-oriented runtime fast paths retained from the frozen runtime work.

These are engineering replacements for slower reference functions. They do not
change the method contract and must be gated against the reference functions
before timing results are accepted.

Provenance: adapted from ``real_motion/runtime_fastpath.py`` on
``ttt05211/swfm@283ff822171884e21352ca6f81ee27b8a76485c5``.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import generate_binary_structure, label, uniform_filter
from ..protocol import DYNAMIC_CLASS_IDS
from ..priors.strong_kta import StrongKTAConfig, transform_points

def majority_fill_sparse_5x5x1(semantics, unknown_mask, *, kernel=(5,5,1), min_fraction=.3):
    if tuple(int(x) for x in kernel)!=(5,5,1): raise ValueError('frozen kernel must be (5,5,1)')
    if abs(float(min_fraction)-.3)>1e-12: raise ValueError('frozen min_fraction must be 0.3')
    sem=np.asarray(semantics); unknown=np.asarray(unknown_mask,dtype=bool)
    if sem.shape!=unknown.shape or sem.ndim!=3: raise ValueError('shape mismatch')
    if not unknown.any(): return sem.copy()
    coords=np.argwhere(unknown); m=len(coords); X,Y,Z=sem.shape
    offsets=np.asarray([(dx,dy) for dx in range(-2,3) for dy in range(-2,3)],dtype=np.int64)
    nx=coords[:,0:1]+offsets[None,:,0]; ny=coords[:,1:2]+offsets[None,:,1]; z=coords[:,2:3]
    inb=(nx>=0)&(nx<X)&(ny>=0)&(ny<Y); gx=np.clip(nx,0,X-1); gy=np.clip(ny,0,Y-1); gz=np.broadcast_to(z,gx.shape)
    known=~unknown; valid_known=inb&known[gx,gy,gz]; neigh=sem[gx,gy,gz].astype(np.int64,copy=False)
    if sem.size and int(sem.min())<0: raise ValueError('negative semantic label unsupported')
    nlabels=int(sem.max())+1 if sem.size else 1; rows=np.broadcast_to(np.arange(m,dtype=np.int64)[:,None],gx.shape)
    rr=rows[valid_known]; ll=neigh[valid_known]; counts=np.bincount(rr*nlabels+ll,minlength=m*nlabels).reshape(m,nlabels); denom=valid_known.sum(1).astype(np.int64)
    mx=counts.max(1); best=counts.argmax(1).astype(sem.dtype,copy=False); ties=(counts==mx[:,None]).sum(1); lhs=10*mx; rhs=3*denom
    fill=(lhs>rhs)&(ties==1); ambiguous=((denom>0)&(lhs==rhs))|((lhs>rhs)&(ties>1))
    if ambiguous.any():
        ai=np.flatnonzero(ambiguous); pk=valid_known[ai].reshape(len(ai),5,5); pl=neigh[ai].reshape(len(ai),5,5)
        den=uniform_filter(pk.astype(np.float32),size=(1,5,5),mode='constant')[:,2,2]; den=np.maximum(den,np.float32(1e-6)); classes=np.unique(sem[known]).astype(np.int64,copy=False)
        masks=((pl[:,None,:,:]==classes[None,:,None,None])&pk[:,None,:,:]).astype(np.float32); scores=uniform_filter(masks,size=(1,1,5,5),mode='constant')[:,:,2,2]/den[:,None]
        bi=np.argmax(scores,1); bs=scores[np.arange(len(ai)),bi]; best[ai]=classes[bi].astype(sem.dtype,copy=False); fill[ai]=bs>=float(min_fraction)
    out=sem.copy(); fc=coords[fill]; out[fc[:,0],fc[:,1],fc[:,2]]=best[fill]; return out

def _box_sum_5x5_torch_int32(x):
    if x.ndim!=3: raise ValueError('input must be [B,X,Y]')
    p=F.pad(x,(2,2,2,2),mode='constant',value=0); s=torch.cumsum(p,1,dtype=torch.int32); s=torch.cumsum(s,2,dtype=torch.int32); s=F.pad(s,(1,0,1,0),mode='constant',value=0)
    return s[:,5:,5:]-s[:,:-5,5:]-s[:,5:,:-5]+s[:,:-5,:-5]

def majority_fill_cuda_exact(semantics,unknown_mask,*,kernel=(5,5,1),min_fraction=.3,device='cuda'):
    dev=torch.device(device)
    if dev.type!='cuda': return majority_fill_sparse_5x5x1(semantics,unknown_mask,kernel=kernel,min_fraction=min_fraction)
    if tuple(kernel)!=(5,5,1) or abs(float(min_fraction)-.3)>1e-12: raise ValueError('frozen fill settings required')
    sem=np.asarray(semantics); unknown=np.asarray(unknown_mask,dtype=bool)
    if sem.shape!=unknown.shape or sem.ndim!=3: raise ValueError('shape mismatch')
    if not unknown.any(): return sem.copy()
    known=~unknown; classes=np.unique(sem[known]).astype(np.int64,copy=False)
    if len(classes)==0:return sem.copy()
    sem_t=torch.from_numpy(np.ascontiguousarray(np.transpose(sem,(2,0,1))).astype(np.int64,copy=False)).to(dev); known_t=torch.from_numpy(np.ascontiguousarray(np.transpose(known,(2,0,1)))).to(dev); unknown_t=torch.from_numpy(np.ascontiguousarray(np.transpose(unknown,(2,0,1)))).to(dev); cls_t=torch.from_numpy(classes).to(dev)
    denom=_box_sum_5x5_torch_int32(known_t.to(torch.int32)); masks=(sem_t[:,None]==cls_t[None,:,None,None])&known_t[:,None]; Z,C,X,Y=masks.shape; counts=_box_sum_5x5_torch_int32(masks.reshape(Z*C,X,Y).to(torch.int32)).reshape(Z,C,X,Y)
    mx,bi=counts.max(1); ties=(counts==mx[:,None]).sum(1); lhs=10*mx; rhs=3*denom; fill_t=unknown_t&(lhs>rhs)&(ties==1); amb_t=unknown_t&(((denom>0)&(lhs==rhs))|((lhs>rhs)&(ties>1))); best_t=cls_t[bi]
    fill=np.transpose(fill_t.cpu().numpy(),(1,2,0)); best=np.transpose(best_t.cpu().numpy(),(1,2,0)).astype(sem.dtype,copy=False); amb=np.transpose(amb_t.cpu().numpy(),(1,2,0)); out=sem.copy(); out[fill]=best[fill]
    coords=np.argwhere(amb)
    if len(coords):
        Xs,Ys,_=sem.shape; offsets=np.asarray([(dx,dy) for dx in range(-2,3) for dy in range(-2,3)],dtype=np.int64); nx=coords[:,0:1]+offsets[None,:,0]; ny=coords[:,1:2]+offsets[None,:,1]; z=coords[:,2:3]; inb=(nx>=0)&(nx<Xs)&(ny>=0)&(ny<Ys); gx=np.clip(nx,0,Xs-1); gy=np.clip(ny,0,Ys-1); gz=np.broadcast_to(z,gx.shape); pk=(inb&known[gx,gy,gz]).reshape(len(coords),5,5); pl=sem[gx,gy,gz].astype(np.int64,copy=False).reshape(len(coords),5,5); den=uniform_filter(pk.astype(np.float32),size=(1,5,5),mode='constant')[:,2,2]; den=np.maximum(den,np.float32(1e-6)); sm=((pl[:,None]==classes[None,:,None,None])&pk[:,None]).astype(np.float32); scores=uniform_filter(sm,size=(1,1,5,5),mode='constant')[:,:,2,2]/den[:,None]; rb=np.argmax(scores,1); rf=scores[np.arange(len(coords)),rb]>=float(min_fraction)
        if rf.any(): fc=coords[rf]; out[fc[:,0],fc[:,1],fc[:,2]]=classes[rb[rf]].astype(sem.dtype,copy=False)
    return out

def inverse_warp_sequence_cuda_exact(semantics, src_to_dst_seq, *, grid, free_label=17, device='cuda', boundary_tol_vox=5e-3):
    dev=torch.device(device)
    if dev.type!='cuda': raise ValueError('CUDA device required')
    sem=np.asarray(semantics)
    if tuple(sem.shape)!=tuple(grid.shape_xyz):raise ValueError('semantic grid shape mismatch')
    shape=tuple(int(x) for x in grid.shape_xyz); origin=np.asarray([grid.x_min,grid.y_min,grid.z_min],dtype=np.float64); step=np.asarray(grid.voxel_size,dtype=np.float64)
    X,Y,Z=shape; xs=origin[0]+(np.arange(X)+.5)*step[0];ys=origin[1]+(np.arange(Y)+.5)*step[1];zs=origin[2]+(np.arange(Z)+.5)*step[2];aa,bb,cc=np.meshgrid(xs,ys,zs,indexing='ij');pts64=np.stack([aa.ravel(),bb.ravel(),cc.ravel()],1);pts32=torch.from_numpy(pts64.astype(np.float32,copy=False)).to(dev);origin32=torch.tensor(origin,dtype=torch.float32,device=dev);step32=torch.tensor(step,dtype=torch.float32,device=dev);sem_t=torch.from_numpy(sem.reshape(-1)).to(dev);out=[];tol=float(boundary_tol_vox)
    for T in src_to_dst_seq:
        inv=np.linalg.inv(np.asarray(T,dtype=np.float64)); R=torch.from_numpy(inv[:3,:3].astype(np.float32)).to(dev); t=torch.from_numpy(inv[:3,3].astype(np.float32)).to(dev); q32=(pts32@R.T+t-origin32)/step32;idx=torch.floor(q32).to(torch.int64); frac=torch.abs(q32-torch.round(q32));unc=torch.any(frac<=tol,1)
        if bool(unc.any()):
            pos_t=torch.nonzero(unc,as_tuple=False).flatten();pos=pos_t.cpu().numpy();ref=pts64[pos]@inv[:3,:3].T+inv[:3,3];ridx=np.floor((ref-origin[None])/step[None]).astype(np.int64);idx[pos_t]=torch.from_numpy(ridx).to(dev)
        known=(idx[:,0]>=0)&(idx[:,0]<X)&(idx[:,1]>=0)&(idx[:,1]<Y)&(idx[:,2]>=0)&(idx[:,2]<Z); flat=torch.full((len(idx),),int(free_label),dtype=sem_t.dtype,device=dev)
        if bool(known.any()):
            q=idx[known];fi=(q[:,0]*Y+q[:,1])*Z+q[:,2];flat[known]=sem_t[fi]
        out.append((flat.reshape(shape).cpu().numpy(),known.reshape(shape).cpu().numpy()))
    return out

def extract_instances_cropped_exact(semantics, ego_to_world, *, grid, cfg=StrongKTAConfig()):
    sem=np.asarray(semantics)
    if tuple(sem.shape)!=tuple(grid.shape_xyz): raise ValueError('semantic grid shape mismatch')
    structure=generate_binary_structure(3,int(cfg.connectivity)); origin=np.asarray([grid.x_min,grid.y_min,grid.z_min],dtype=np.float64); step=np.asarray(grid.voxel_size,dtype=np.float64); out=[]; dyn=np.isin(sem,np.asarray(DYNAMIC_CLASS_IDS,dtype=sem.dtype)); all_idx=np.argwhere(dyn)
    if not len(all_idx):return out
    all_labels=sem[tuple(all_idx.T)]
    for cls in DYNAMIC_CLASS_IDS:
        pts=all_idx[all_labels==int(cls)]
        if len(pts)<int(cfg.min_component_voxels):continue
        lo=pts.min(0); hi=pts.max(0)+1; sl=tuple(slice(int(lo[d]),int(hi[d])) for d in range(3)); cmap,n=label(sem[sl]==int(cls),structure=structure)
        for comp_id in range(1,int(n)+1):
            local=np.argwhere(cmap==comp_id)
            if len(local)<int(cfg.min_component_voxels):continue
            idx=local.astype(np.int64,copy=False)+lo[None].astype(np.int64); pe=origin+(idx.astype(np.float64)+.5)*step; cw=transform_points(ego_to_world,pe.mean(0,keepdims=True))[0]; out.append({'class_id':int(cls),'voxel_indices':idx,'centroid_world':cw,'voxel_count':int(len(idx))})
    return out

def component_lists_equal(a,b):
    if len(a)!=len(b):return False
    for x,y in zip(a,b):
        if int(x['class_id'])!=int(y['class_id']) or int(x['voxel_count'])!=int(y['voxel_count']):return False
        if not np.array_equal(np.asarray(x['voxel_indices'],dtype=np.int64),np.asarray(y['voxel_indices'],dtype=np.int64)):return False
        if not np.array_equal(np.asarray(x['centroid_world'],dtype=np.float64),np.asarray(y['centroid_world'],dtype=np.float64)):return False
    return True

def baseline_clear_mask(baseline_components,*,grid):
    clear=np.zeros(grid.shape_xyz,dtype=bool)
    for comp in baseline_components:
        idx=np.asarray(comp.voxel_indices,dtype=np.int64)
        if len(idx):clear[idx[:,0],idx[:,1],idx[:,2]]=True
    return clear

def baseline_clear_flat_indices(baseline_components,*,grid):
    _,Y,Z=grid.shape_xyz; rows=[]
    for comp in baseline_components:
        idx=np.asarray(comp.voxel_indices,dtype=np.int64)
        if len(idx):rows.append((idx[:,0]*Y+idx[:,1])*Z+idx[:,2])
    return np.unique(np.concatenate(rows)).astype(np.int64,copy=False) if rows else np.zeros((0,),dtype=np.int64)

def compose_hard_a1_fast_exact(anchor_occ,baseline_components,replacement_components,*,dynamic_class_ids=DYNAMIC_CLASS_IDS,free_label=17,grid,precomputed_clear_flat_indices=None):
    out=np.asarray(anchor_occ).copy(); dyn=np.asarray(tuple(int(x) for x in dynamic_class_ids),dtype=np.int64)
    if tuple(out.shape)!=tuple(grid.shape_xyz):raise ValueError('grid mismatch')
    if precomputed_clear_flat_indices is None: clear_flat=np.flatnonzero(baseline_clear_mask(baseline_components,grid=grid).reshape(-1))
    else:clear_flat=np.asarray(precomputed_clear_flat_indices,dtype=np.int64)
    f=out.reshape(-1)
    if len(clear_flat):
        v=f[clear_flat]; m=np.isin(v,dyn); f[clear_flat[m]]=int(free_label)
    for comp in replacement_components:
        idx=np.asarray(comp.voxel_indices,dtype=np.int64)
        if len(idx):out[idx[:,0],idx[:,1],idx[:,2]]=int(comp.class_id)
    return out
