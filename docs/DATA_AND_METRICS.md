# Data, coordinates and metrics

Dataset: legally obtained nuScenes v1.0-trainval plus Occ3D semantic labels. Formal windows contain 6 history and 6 future keyframes at 0.5 s spacing.

Formal split/counts: train 700 scenes / 20,430 eligible stride-1 windows; val 150 scenes / 4,369 eligible stride-1 windows. `val128` is not a formal evaluation set.

## Occupancy grid

Axes are X/Y/Z in ego frame; shape `200×200×16`; voxel size `0.4 m`; ranges X/Y `[-40,40)`, Z `[-1,5.4)`; labels `0..17`, free=`17`. The local semantic tube is 16 m × 16 m at 0.8 m resolution (`20×20`).

## Cache

`causal_se2_occ_cache_v1` stores one record per eligible window: causal source features/tubes/masks/KTA tensors, source metadata/observed voxel indices, and training supervision. Cache construction starts from raw user-provided data and has no dependency on old private cache chains. Train/val scene overlap is rejected.

## Metrics

Reporting horizons are 1/2/3 s (future indices 1/3/5).

- Occupancy IoU: sum raw occupied/free intersection/union dataset-wide inside each horizon, then divide.
- Semantic mIoU: sum class raw counts for classes 0..16 inside each horizon; empty class union is skipped; arithmetic-mean the three horizon mIoUs.
- True-moving support: dynamic classes `(2,3,4,5,6,7,9,10)`; common t0/future instances with endpoint interval XY speed ≥0.5 m/s; support is union of t0 and future oriented boxes in future ego frame with 0.5 m margin.
- Macro Moving-mIoU: class IoUs inside support, mean classes per horizon, then mean horizons.
- Horizon-first Micro Moving-IoU: sum dynamic-class intersection/union per horizon, divide, then arithmetic-mean the three horizons.
- Formal evaluation uses the complete occupancy grid; it does not add a `mask_lidar` validity mask.

## Paired scene bootstrap

Resample **scenes** with replacement. For every sampled scene occurrence, include all overlapping windows, sum raw intersections/unions, recompute metrics, then compute paired deltas. Never average scene IoUs or treat overlapping windows as independent samples.
