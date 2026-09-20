from __future__ import annotations
import torch
from .losses.objective import clean_objective
from .training import move_batch


def evaluate_objective(model, loader, device, *, amp, yaw_weight, shape_weight, patch_resolution):
    model.eval(); sums={}; n=0
    with torch.no_grad():
        for batch in loader:
            b=move_batch(batch,device)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=(amp and device.type=='cuda')):
                out=model(b['features'],b['local_semantic_tube'],b['kta_displacement_xy_m'],b['frame_motion_features'],b['target_source_mask_tube'])
                _,stats=clean_objective(out,b,yaw_weight=yaw_weight,shape_weight=shape_weight,patch_resolution_m=patch_resolution)
            for k,v in stats.items():sums[k]=sums.get(k,0.)+float(v)
            n+=1
    return {k:v/max(n,1) for k,v in sums.items()}
