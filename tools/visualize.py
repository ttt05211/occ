#!/usr/bin/env python3
import argparse,numpy as np
import matplotlib.pyplot as plt
from causal_se2_occ.io import prepare_output_file
p=argparse.ArgumentParser();p.add_argument('npz');p.add_argument('--horizon-index',type=int,default=5);p.add_argument('--output',required=True);a=p.parse_args();prepare_output_file(a.output);d=np.load(a.npz);x=d['prediction'][a.horizon_index];bev=np.full(x.shape[:2],17,dtype=np.uint8);occ=x!=17;has=occ.any(2);z=x.shape[2]-1-np.argmax(occ[:,:,::-1],2);ii,jj=np.nonzero(has);bev[ii,jj]=x[ii,jj,z[ii,jj]];plt.figure(figsize=(6,6));plt.imshow(bev.T,origin='lower',interpolation='nearest');plt.title(f'Predicted top surface, horizon index {a.horizon_index}');plt.xlabel('X voxel');plt.ylabel('Y voxel');plt.tight_layout();plt.savefig(a.output,dpi=180);print(a.output)
