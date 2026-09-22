from __future__ import annotations

import torch

from .losses.objective import clean_objective
from .training import forward_model, move_batch


def evaluate_objective(
    model,
    loader,
    device,
    *,
    amp,
    yaw_weight,
    shape_weight,
    patch_resolution,
):
    model.eval()
    sums = {}
    batches = 0

    with torch.no_grad():
        for raw in loader:
            batch = move_batch(raw, device)
            output = forward_model(model, batch, amp=amp, device=device)
            # Frozen trainer/evaluator computes the objective outside autocast.
            _, stats = clean_objective(
                output,
                batch,
                yaw_weight=yaw_weight,
                shape_weight=shape_weight,
                patch_resolution_m=patch_resolution,
            )
            for key, value in stats.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            batches += 1

    return {
        key: value / max(batches, 1)
        for key, value in sums.items()
    }
