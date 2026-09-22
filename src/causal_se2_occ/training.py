from __future__ import annotations

import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.cache import SourceDataset
from .losses.objective import clean_objective


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_scale(step: int, total: int) -> float:
    frac = min(max(float(step) / max(int(total), 1), 0.0), 1.0)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac))


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def forward_model(model, batch: dict, *, amp: bool, device: torch.device) -> dict:
    """Frozen AMP boundary: autocast covers only the network forward."""
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=(amp and device.type == "cuda"),
    ):
        return model(
            batch["features"],
            batch["local_semantic_tube"],
            batch["kta_displacement_xy_m"],
            batch["frame_motion_features"],
            batch["target_source_mask_tube"],
        )


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    *,
    amp,
    yaw_weight,
    shape_weight,
    patch_resolution,
    global_step,
    base_lr=None,
    total_steps=None,
    fixed_lr=None,
    grad_clip=5.0,
):
    model.train()
    sums = {}
    batches = 0

    for raw in loader:
        batch = move_batch(raw, device)
        output = forward_model(model, batch, amp=amp, device=device)
        # Intentionally outside autocast: this matches the frozen Clean trainer.
        loss, stats = clean_objective(
            output,
            batch,
            yaw_weight=yaw_weight,
            shape_weight=shape_weight,
            patch_resolution_m=patch_resolution,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        global_step += 1

        lr = (
            fixed_lr
            if fixed_lr is not None
            else base_lr * cosine_scale(global_step, total_steps)
        )
        for group in optimizer.param_groups:
            group["lr"] = float(lr)

        for key, value in stats.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        batches += 1

    return global_step, {
        key: value / max(batches, 1)
        for key, value in sums.items()
    }


def make_loader(flat, batch_size, workers, seed, shuffle, device):
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return DataLoader(
        SourceDataset(flat),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
