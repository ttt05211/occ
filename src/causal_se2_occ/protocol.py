from __future__ import annotations

HISTORY_FRAMES = 6
FUTURE_FRAMES = 6
FRAME_DT_S = 0.5
FREE_LABEL = 17
SEMANTIC_CLASSES = 18
DYNAMIC_CLASS_IDS = (2, 3, 4, 5, 6, 7, 9, 10)
YAW_ENABLED_CLASS_IDS = (2, 3, 4, 5, 6, 9, 10)
REPORT_HORIZONS_S = (1.0, 2.0, 3.0)
REPORT_HORIZON_INDICES = {1.0: 1, 2.0: 3, 3.0: 5}
MOVING_SPEED_THRESHOLD_MPS = 0.5
MOVING_BOX_MARGIN_M = 0.5
CACHE_SCHEMA = "causal_se2_occ_cache_v1"
CHECKPOINT_PROTOCOL = "p0_f9_v18_se2_clean_train_v1"
MODEL_PROTOCOL = "p0_f9_v18_se2_local_spatial_temporal_world_model_v1"
SE2_TARGET_CONTRACT = "source_center_se2_gt_rigid_equivalent_v1"
SE2_SHAPE_CONTRACT = "soft_iou_same_source_footprint_pred_vs_gt_se2_v1"
A1_CONTRACT = "legacy_clear_plus_original_strong_source_write_order_v1"

NUSCENES_LABELS = (
    "others", "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade",
    "vegetation", "free",
)
CATEGORY_PREFIX_TO_CLASS = (
    ("vehicle.bicycle", 2), ("vehicle.bus", 3), ("vehicle.car", 4),
    ("vehicle.construction", 5), ("vehicle.motorcycle", 6),
    ("human.pedestrian", 7), ("vehicle.trailer", 9), ("vehicle.truck", 10),
)

def category_to_dynamic_class(name: str):
    for prefix, cid in CATEGORY_PREFIX_TO_CLASS:
        if name.startswith(prefix):
            return cid
    return None
