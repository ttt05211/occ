from __future__ import annotations

import numpy as np
import torch

from .data.features import world_points_to_t0
from .data.prepare import prepare_causal_arrays
from .geometry.grid import OccupancyGrid
from .geometry.render import compose_hard_a1, rasterize_rigid_component
from .priors.strong_kta import strong_kta_sequence
from .protocol import DYNAMIC_CLASS_IDS, YAW_ENABLED_CLASS_IDS


def t0_xy_to_world_preserve_source_z(
    xy_t0: np.ndarray,
    source_center_world: np.ndarray,
    t0_ego_to_world: np.ndarray,
) -> np.ndarray:
    """Frozen conversion from predicted t0 XY to a world-space source centre.

    The historical implementation preserves the source centre's *t0-frame Z*
    while replacing only t0 X/Y, then applies the full t0 pose. This matters
    whenever ego roll/pitch is non-zero.
    """
    pose = np.asarray(t0_ego_to_world, dtype=np.float64)
    src_t0 = world_points_to_t0(
        np.asarray(source_center_world, dtype=np.float64)[None],
        pose,
    )[0]
    p = np.asarray(
        [float(xy_t0[0]), float(xy_t0[1]), float(src_t0[2]), 1.0],
        dtype=np.float64,
    )
    return (pose @ p)[:3]


def forecast_window(
    model,
    history_occ,
    history_poses,
    future_poses,
    *,
    device="cpu",
    amp_bf16=True,
    grid=OccupancyGrid(),
    frame_dt_s=0.5,
    free_label=17,
):
    anchor, current, vel = strong_kta_sequence(
        history_occ,
        history_poses,
        future_poses,
        frame_dt_s=frame_dt_s,
        grid=grid,
    )
    causal = prepare_causal_arrays(
        history_occ,
        history_poses,
        grid=grid,
        frame_dt_s=frame_dt_s,
        free_label=free_label,
    )
    fields = [
        "features",
        "local_semantic_tube",
        "kta_displacement_xy_m",
        "frame_motion_features",
        "target_source_mask_tube",
    ]
    x = {k: causal[k].to(device) for k in fields}
    model = model.to(device).eval()

    use_amp = str(device).startswith("cuda") and amp_bf16
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=use_amp,
    ):
        output = model(
            x["features"].float(),
            x["local_semantic_tube"],
            x["kta_displacement_xy_m"].float(),
            x["frame_motion_features"].float(),
            x["target_source_mask_tube"],
        )

    residual = output["residual_xy_m"].float().cpu().numpy()
    pred_yaw = output["yaw_delta_rad"].float().cpu().numpy()
    predictions = []
    t0_pose = np.asarray(history_poses[-1], dtype=np.float64)

    for horizon_index, future_pose in enumerate(future_poses):
        baseline = []
        replacements = []
        dt = (horizon_index + 1) * frame_dt_s

        for source_index, component in enumerate(current):
            source_center = np.asarray(component["centroid_world"], dtype=np.float64)
            velocity = np.asarray(vel.get(source_index, np.zeros(3)), dtype=np.float64)

            baseline.append(
                rasterize_rigid_component(
                    component["voxel_indices"],
                    component["class_id"],
                    t0_pose,
                    future_pose,
                    source_center_world=source_center,
                    target_center_world=source_center + velocity * dt,
                    yaw_delta_rad=0.0,
                    grid=grid,
                )
            )

            xy_t0 = (
                causal["anchors_xy_t0_m"][source_index, horizon_index].numpy()
                + residual[source_index, horizon_index]
            )
            target_center = t0_xy_to_world_preserve_source_z(
                xy_t0,
                source_center,
                t0_pose,
            )
            yaw = (
                float(pred_yaw[source_index, horizon_index])
                if int(component["class_id"]) in YAW_ENABLED_CLASS_IDS
                else 0.0
            )
            replacements.append(
                rasterize_rigid_component(
                    component["voxel_indices"],
                    component["class_id"],
                    t0_pose,
                    future_pose,
                    source_center_world=source_center,
                    target_center_world=target_center,
                    yaw_delta_rad=yaw,
                    grid=grid,
                )
            )

        predictions.append(
            compose_hard_a1(
                anchor[horizon_index],
                baseline,
                replacements,
                dynamic_class_ids=DYNAMIC_CLASS_IDS,
                free_label=free_label,
                grid=grid,
            )
        )

    return np.stack(predictions), {
        "residual_xy_m": residual,
        "yaw_delta_rad": pred_yaw,
        "existence_logits": output["existence_logits"].float().cpu().numpy(),
        "note": "existence logits are not used as deployment gates",
    }
