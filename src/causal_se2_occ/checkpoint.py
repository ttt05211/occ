from __future__ import annotations
from pathlib import Path
import torch
from .models.stwm import SourceCenteredSE2Predictor,config_from_mapping
from .protocol import CHECKPOINT_PROTOCOL

def _validate_legacy_header(ck):
    if str(ck.get("protocol")) != CHECKPOINT_PROTOCOL:
        raise RuntimeError(f"checkpoint protocol mismatch: {ck.get('protocol')!r}")
    if str(ck.get("arm")) != "Y":
        raise RuntimeError(f"Clean checkpoint expects arm='Y', got {ck.get('arm')!r}")
    mode=str(ck.get("training_mode") or "")
    if "balanced" in mode.lower() or "balanced" in str(ck.get("variant") or "").lower():
        raise RuntimeError("balanced/exploratory checkpoint is not the frozen Clean main model")
    if mode not in {"", "clean_one_stage_from_scratch_v1", "clean_one_stage_from_scratch_v1_tail_continuation"}:
        raise RuntimeError(f"unexpected Clean training_mode={mode!r}")

def load_model_checkpoint(path,device='cpu'):
    ck=torch.load(path,map_location='cpu',weights_only=False); _validate_legacy_header(ck)
    cfg=config_from_mapping(ck.get('model_config'));m=SourceCenteredSE2Predictor(cfg).to(device);m.load_state_dict(ck['state_dict'],strict=True);m.eval();return ck,m

def migrate_legacy_checkpoint(src,dst):
    ck=torch.load(src,map_location='cpu',weights_only=False); _validate_legacy_header(ck)
    cfg=config_from_mapping(ck.get('model_config')); model=SourceCenteredSE2Predictor(cfg); model.load_state_dict(ck['state_dict'],strict=True)
    out=dict(ck); out['public_migration']={'source_format':'ttt05211/swfm Clean V18','weight_key_mapping':'identity_strict','note':'state_dict keys and tensor shapes are preserved; no numerical conversion is performed'}
    Path(dst).parent.mkdir(parents=True,exist_ok=True);torch.save(out,dst);return out
