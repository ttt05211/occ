from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import Dataset

from ..protocol import CACHE_SCHEMA, SE2_TARGET_CONTRACT

TRAIN_TENSOR_KEYS = (
    "features",
    "local_semantic_tube",
    "kta_displacement_xy_m",
    "existence",
    "supervised_source",
    "source_class_id",
    "frame_motion_features",
    "target_source_mask_tube",
    "target_source_displacement_xy_m",
    "target_source_residual_xy_m",
    "target_yaw_rad",
    "yaw_label_valid",
    "se2_target_valid",
    "yaw_enabled",
)

def save_cache(path, records: Sequence[Mapping], metadata: Mapping) -> None:
    meta = dict(metadata)
    meta.setdefault("cache_schema", CACHE_SCHEMA)
    meta.setdefault("se2_target_contract", SE2_TARGET_CONTRACT)
    payload = {"version": CACHE_SCHEMA, "metadata": meta, "records": list(records)}
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)

def load_cache(path) -> tuple[dict, list[dict]]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != CACHE_SCHEMA:
        raise RuntimeError(f"cache version {obj.get('version')!r} != {CACHE_SCHEMA!r}")
    meta = dict(obj.get("metadata") or {})
    if meta.get("se2_target_contract") not in {None, SE2_TARGET_CONTRACT}:
        raise RuntimeError("SE(2) target contract mismatch")
    records = list(obj.get("records") or [])
    if not records:
        raise RuntimeError("cache contains no records")
    return meta, records

def flatten_supervised(records: Sequence[Mapping]) -> dict:
    chunks = {k: [] for k in TRAIN_TENSOR_KEYS}
    scene_ids: list[str] = []
    for rec in records:
        missing = [k for k in TRAIN_TENSOR_KEYS if k not in rec]
        if missing:
            raise RuntimeError(f"{rec.get('sample_id','?')}: cache record missing {missing}")
        sup = rec["supervised_source"].bool()
        if not bool(sup.any()):
            continue
        count = int(sup.sum().item())
        for key in TRAIN_TENSOR_KEYS:
            x = rec[key]
            if key == "supervised_source":
                chunks[key].append(torch.ones(count, dtype=torch.bool))
            elif key in {"local_semantic_tube", "target_source_mask_tube"}:
                chunks[key].append(x[sup].to(torch.uint8))
            elif key == "source_class_id":
                chunks[key].append(x[sup].long())
            elif key in {"yaw_label_valid", "se2_target_valid", "yaw_enabled"}:
                chunks[key].append(x[sup].bool())
            else:
                chunks[key].append(x[sup].float())
        scene_ids.extend([str(rec["scene_name"])] * count)
    if not chunks["features"]:
        raise RuntimeError("cache has no supervised sources")
    flat = {key: torch.cat(parts, dim=0) for key, parts in chunks.items()}
    flat["scene_ids"] = scene_ids
    return flat

class SourceDataset(Dataset):
    def __init__(self, flat: Mapping):
        self.flat = dict(flat)
        n = int(self.flat["features"].shape[0])
        for key, value in self.flat.items():
            if key == "scene_ids":
                continue
            if not torch.is_tensor(value) or int(value.shape[0]) != n:
                raise ValueError(f"flat field {key!r} does not align with features")
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict:
        return {
            key: value[index]
            for key, value in self.flat.items()
            if key != "scene_ids"
        }
