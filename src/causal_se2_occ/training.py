from __future__ import annotations
import math,random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data.cache import SourceDataset
from .losses.objective import clean_objective

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def cosine_scale(step,total):
    frac=min(max(float(step)/max(int(total),1),0.),1.);return .1+.9*.5*(1+math.cos(math.pi*frac))
def move_batch(b,device):return {k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in b.items()}
def train_epoch(model,loader,optimizer,device,*,amp,yaw_weight,shape_weight,patch_resolution,global_step,base_lr=None,total_steps=None,fixed_lr=None,grad_clip=5.):
    model.train(); sums={}; n=0
    for batch in loader:
        b=move_batch(batch,device);
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=(amp and device.type=='cuda')):out=model(b['features'],b['local_semantic_tube'],b['kta_displacement_xy_m'],b['frame_motion_features'],b['target_source_mask_tube']);loss,stats=clean_objective(out,b,yaw_weight=yaw_weight,shape_weight=shape_weight,patch_resolution_m=patch_resolution)
        if not torch.isfinite(loss):raise RuntimeError('non-finite loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip);optimizer.step();global_step+=1
        lr=fixed_lr if fixed_lr is not None else base_lr*cosine_scale(global_step,total_steps)
        for g in optimizer.param_groups:g['lr']=float(lr)
        for k,v in stats.items():sums[k]=sums.get(k,0.)+float(v)
        n+=1
    return global_step,{k:v/max(n,1) for k,v in sums.items()}
def make_loader(flat,batch_size,workers,seed,shuffle,device):
    gen=torch.Generator().manual_seed(int(seed)) if shuffle else None;return DataLoader(SourceDataset(flat),batch_size=batch_size,shuffle=shuffle,generator=gen,num_workers=workers,pin_memory=(device.type=='cuda'),drop_last=False)
