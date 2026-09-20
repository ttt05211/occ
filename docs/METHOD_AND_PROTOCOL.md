# Method and frozen protocol

Prediction path:

1. Read six historical semantic occupancy grids.
2. Extract same-class 3D connected components and causally match history to obtain Strong/KTA velocity.
3. Build source-centred **2D semantic tubes**, causal source masks, and per-frame motion features.
4. STWM encodes the six-frame tube and six future queries.
5. Heads output KTA-relative XY residual, relative yaw, and existence logits.
6. Transport the raw t0 3D source voxels into each future ego frame.
7. Hard A1 clears baseline dynamic footprints, then writes predictions in original Strong/source order.

## Inputs/outputs

Inputs per source: `features [B,46]`, `local_semantic_tube [B,6,20,20]`, `target_source_mask_tube [B,6,20,20]`, `frame_motion_features [B,6,5]`, `kta_displacement_xy_m [B,6,2]`.

Outputs: `residual_xy_m [B,6,2]`, `yaw_delta_rad [B,6]`, `existence_logits [B,6]`. Existence is trained but is **not** a deployment gate.

## Source-centred SE(2)

Let `c_s` be observed Strong source centroid, `a_0,a_h` matched GT annotation centres, and `R` relative yaw. The source-centre displacement is:

`d_s = (a_h-a_0) + (R-I)(c_s-a_0)`.

This is rigidly equivalent to box-centred motion: `p' = R(p-c_s)+c_s+d_s`. The XY head predicts `r_h=d_s,h-d_KTA,h`; inference uses `d_pred=d_KTA+r_pred`.

Yaw-enabled IDs are `(2,3,4,5,6,9,10)`; pedestrian `7` stays translation-only.

## Original 3D geometry and hard A1

The network sees a 2D tube; the renderer moves the original t0 source `voxel_indices`. It rotates in XY around the observed source centroid, preserves world Z, transforms to future ego, rasterizes/deduplicates.

Hard A1: union all baseline source footprints; clear only dynamic anchor labels inside that union; then write learned replacement components in original source order. Later sources overwrite earlier ones on collision. No confidence or existence gating.

## Training objective

`L = L_trans + L_exist + 19.0 L_yaw + 0.25 L_shape_SE2`.

`L_trans`: Smooth-L1 beta=1 on valid KTA-relative source-centre XY residual. `L_exist`: BCE-with-logits. `L_yaw`: periodic `1-cos(wrap(pred-target))` on yaw-enabled valid labels. `L_shape_SE2`: `1-soft IoU` between the **same observed t0 source footprint** under predicted and GT source-centred SE(2), using bilinear grid_sample and 0.8 m resolution. It is not complete-future-scene supervision.

Prediction may use historical occupancy and historical/future ego poses required by deterministic transport. Future semantic occupancy, future identities/centres/yaw are supervision/evaluation only.
