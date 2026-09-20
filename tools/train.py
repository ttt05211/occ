#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from dataclasses import asdict
from pathlib import Path
import yaml,torch
from causal_se2_occ.data.cache import load_cache,flatten_supervised
from causal_se2_occ.models.stwm import ModelConfig,SourceCenteredSE2Predictor
from causal_se2_occ.protocol import CHECKPOINT_PROTOCOL,MODEL_PROTOCOL,SE2_TARGET_CONTRACT,SE2_SHAPE_CONTRACT
from causal_se2_occ.training import seed_all,make_loader,train_epoch,move_batch
from causal_se2_occ.evaluation import evaluate_objective

def save(path,model,opt,cfg,tcfg,epoch,step,train_meta,val_meta,mode,val_report,tail_source=None):
    torch.save({'protocol':CHECKPOINT_PROTOCOL,'model_protocol':MODEL_PROTOCOL,'arm':'Y','training_mode':mode,'epoch':epoch,'global_step':step,'continuation_step':step,'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'model_config':asdict(cfg),'variant':'CLEAN-SE2','use_representation':True,'overlap_weight':tcfg['shape_weight'],'yaw_weight':tcfg['yaw_weight'],'shape_weight':tcfg['shape_weight'],'safe_weight':None,'safe_margin':None,'se2_target_contract':SE2_TARGET_CONTRACT,'se2_shape_contract':SE2_SHAPE_CONTRACT,'seed':tcfg['seed'],'steps_per_epoch':tcfg['steps_per_epoch'],'schedule_total_steps':tcfg['cosine_epochs']*tcfg['steps_per_epoch'],'tail_lr':tcfg['tail_lr'] if epoch>tcfg['cosine_epochs'] else None,'tail_source_checkpoint':str(tail_source) if tail_source else None,'args':tcfg,'train_cache_metadata':train_meta,'val_cache_metadata':val_meta,'val_report':val_report},path)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--train-cache',required=True);p.add_argument('--val-cache',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--device',default='cuda');p.add_argument('--resume',default='');a=p.parse_args();conf=yaml.safe_load(Path(a.config).read_text());mconf=ModelConfig(**conf['model']);tconf=dict(conf['training']);tm,tr=load_cache(a.train_cache);vm,vr=load_cache(a.val_cache);train=flatten_supervised(tr);val=flatten_supervised(vr)
 overlap=set(train['scene_ids'])&set(val['scene_ids'])
 if overlap:raise RuntimeError(f'train/val scene overlap: {sorted(overlap)[:5]}')
 if bool(tconf.get('expect_clean_e14_step',False)):
  for name,meta,records,es,ew in [('train',tm,tr,700,20430),('val',vm,vr,150,4369)]:
   if str(meta.get('split'))!=name or int(meta.get('scene_count',-1))!=es or int(meta.get('eligible_windows',-1))!=ew or len(records)!=ew:raise RuntimeError(f'formal Clean-E14 requires {name} cache={es} scenes/{ew} windows')
 device=torch.device(a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu');amp=(device.type=='cuda' and tconf.get('amp','bf16')=='bf16');out=Path(a.output_dir)
 if out.exists() and any(out.iterdir()) and not a.resume:raise FileExistsError(f'refusing non-empty output dir: {out}')
 out.mkdir(parents=True,exist_ok=True);seed=int(tconf['seed']);seed_all(seed);model=SourceCenteredSE2Predictor(mconf).to(device);opt=torch.optim.AdamW(model.parameters(),lr=float(tconf['lr']),weight_decay=float(tconf['weight_decay']));bs=int(tconf['batch_size']);workers=int(tconf['num_workers']);cos_epochs=int(tconf['cosine_epochs']);final_epoch=int(tconf['final_epoch']);patch=float(tm.get('patch_resolution_m',.8));train_loader=make_loader(train,bs,workers,seed,True,device);val_loader=make_loader(val,bs,workers,seed,False,device);spe=len(train_loader);tconf['steps_per_epoch']=spe
 if tconf.get('expected_steps_per_epoch') and int(tconf['expected_steps_per_epoch'])!=spe:raise RuntimeError(f'formal steps/epoch mismatch: expected {tconf["expected_steps_per_epoch"]}, got {spe}')
 step=0;start=0
 if a.resume:
  ck=torch.load(a.resume,map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict'],strict=True);opt.load_state_dict(ck['optimizer']);step=int(ck['global_step']);start=int(ck['epoch'])
 else:
  first=next(iter(val_loader));b0=move_batch(first,device);model.eval()
  with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=amp):init=model(b0['features'],b0['local_semantic_tube'],b0['kta_displacement_xy_m'],b0['frame_motion_features'],b0['target_source_mask_tube'])
  if float(init['residual_xy_m'].abs().max())>1e-7 or float(init['yaw_delta_rad'].abs().max())>1e-7:raise RuntimeError('fresh model violates zero-residual/zero-yaw initialization')
 total=cos_epochs*spe;tail_source=out/f'epoch_{cos_epochs:04d}.pt'
 for epoch in range(start+1,final_epoch+1):
  if epoch==cos_epochs+1:
   seed_all(seed+cos_epochs);train_loader=make_loader(train,bs,workers,seed+cos_epochs,True,device)
   for g in opt.param_groups:g['lr']=float(tconf['tail_lr'])
  fixed=float(tconf['tail_lr']) if epoch>cos_epochs else None;step,stats=train_epoch(model,train_loader,opt,device,amp=amp,yaw_weight=float(tconf['yaw_weight']),shape_weight=float(tconf['shape_weight']),patch_resolution=patch,global_step=step,base_lr=float(tconf['lr']),total_steps=total,fixed_lr=fixed,grad_clip=float(tconf['grad_clip']));val_report=evaluate_objective(model,val_loader,device,amp=amp,yaw_weight=float(tconf['yaw_weight']),shape_weight=float(tconf['shape_weight']),patch_resolution=patch);mode='clean_one_stage_from_scratch_v1_tail_continuation' if epoch>cos_epochs else 'clean_one_stage_from_scratch_v1';path=out/f'epoch_{epoch:04d}.pt';save(path,model,opt,mconf,tconf,epoch,step,tm,vm,mode,val_report,tail_source if epoch>cos_epochs else None);save(out/'latest.pt',model,opt,mconf,tconf,epoch,step,tm,vm,mode,val_report,tail_source if epoch>cos_epochs else None);print(json.dumps({'epoch':epoch,'global_step':step,'lr':opt.param_groups[0]['lr'],**{f'train_{k}':v for k,v in stats.items()},**{f'val_{k}':v for k,v in val_report.items()}}),flush=True)
 if bool(tconf.get('expect_clean_e14_step',False)) and step!=18410:raise RuntimeError(f'Clean-E14 provenance expects global_step=18410; got {step}')
if __name__=='__main__':main()
