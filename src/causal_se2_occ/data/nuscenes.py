from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np

from ..geometry.grid import pose_matrix
from ..protocol import FUTURE_FRAMES, HISTORY_FRAMES

@dataclass(frozen=True)
class WindowTokens:
    scene_name: str
    history_tokens: tuple[str, ...]
    t0_token: str
    future_tokens: tuple[str, ...]

class NuScenesOcc3D:
    """Thin standalone nuScenes + Occ3D adapter for frozen 6+6 keyframe windows."""

    def __init__(self,dataroot,*,split: str,version: str = "v1.0-trainval",verbose: bool = False):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        try:
            from nuscenes.nuscenes import NuScenes
            from nuscenes.utils.splits import create_splits_scenes
        except ImportError as exc:
            raise ImportError(
                "nuScenes raw-data support requires `pip install 'causal-se2-occ[nuscenes]'`"
            ) from exc
        self.dataroot = str(Path(dataroot).resolve())
        self.split = split
        self.version = version
        self.nusc = NuScenes(version=version,dataroot=self.dataroot,verbose=verbose)
        self.allowed_scenes = set(create_splits_scenes()[split])

    def scene_tokens(self, scene: dict) -> list[str]:
        token = str(scene["first_sample_token"]); out=[]
        while token:
            out.append(token)
            token = str(self.nusc.get("sample", token).get("next", ""))
        return out

    def iter_windows(self,*,history: int = HISTORY_FRAMES,future: int = FUTURE_FRAMES,stride: int = 1,max_windows: int | None = None) -> Iterator[WindowTokens]:
        if history != HISTORY_FRAMES or future != FUTURE_FRAMES:
            raise ValueError("the frozen method requires six history and six future frames")
        if stride <= 0:
            raise ValueError("stride must be positive")
        emitted=0
        for scene in self.nusc.scene:
            name=str(scene["name"])
            if name not in self.allowed_scenes:
                continue
            tokens=self.scene_tokens(scene)
            for i in range(history-1,len(tokens)-future,stride):
                yield WindowTokens(name,tuple(tokens[i-history+1:i+1]),str(tokens[i]),tuple(tokens[i+1:i+future+1]))
                emitted+=1
                if max_windows is not None and emitted >= int(max_windows):
                    return

    def label_path(self, scene_name: str, token: str) -> Path:
        return Path(self.dataroot) / "gts" / str(scene_name) / str(token) / "labels.npz"

    def load_occ3d(self, scene_name: str, token: str) -> tuple[np.ndarray, np.ndarray | None]:
        path=self.label_path(scene_name,token)
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path) as data:
            if "semantics" not in data:
                raise KeyError(f"{path} lacks semantics")
            sem=np.asarray(data["semantics"],dtype=np.uint8)
            mask=np.asarray(data["mask_lidar"],dtype=bool) if "mask_lidar" in data else None
        if sem.shape != (200,200,16):
            raise ValueError(f"unexpected Occ3D semantics shape {sem.shape} in {path}")
        if mask is not None and mask.shape != sem.shape:
            raise ValueError(f"mask_lidar shape {mask.shape} != semantics {sem.shape} in {path}")
        return sem,mask

    @lru_cache(maxsize=192)
    def semantics(self, scene_name: str, token: str) -> np.ndarray:
        return self.load_occ3d(scene_name,token)[0]

    @lru_cache(maxsize=768)
    def pose(self, token: str) -> np.ndarray:
        sample=self.nusc.get("sample",str(token))
        lidar_sd=self.nusc.get("sample_data",sample["data"]["LIDAR_TOP"])
        ego=self.nusc.get("ego_pose",lidar_sd["ego_pose_token"])
        return pose_matrix(ego["translation"],ego["rotation"])
