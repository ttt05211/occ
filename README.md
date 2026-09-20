# Causal SE(2) Occupancy Forecasting

A standalone engineering release of the **frozen Clean-E14 source-centred SE(2) occupancy forecasting method** extracted from [`ttt05211/swfm`](https://github.com/ttt05211/swfm).

This repository is intentionally narrow. It contains the final causal method, the Strong/KTA baseline, training/inference/evaluation/runtime tooling, checkpoint migration, tests, and a small visualization utility. It does **not** carry the historical latent-flow, router/selector/MSP, STPN/backtrace, Moving-Safe, long-tail reweighting, two-wheel special rules, or temporal/FVD research branches.

> Paper title / venue / DOI / BibTeX: **TBD**. No publication status is claimed here.
>
> License: the source repository did not contain a root license. See [LICENSE_STATUS.md](LICENSE_STATUS.md) before redistributing or reusing the source.

## Method at a glance

```mermaid
flowchart LR
    H[6 historical semantic occupancy grids] --> S[Causal source extraction + matching]
    S --> K[Strong / KTA prior]
    S --> T[Source-centred 2D semantic tube\nsource mask + motion features]
    K --> M[STWM encoder + future queries]
    T --> M
    M --> R[6-horizon XY residual\nrelative yaw + existence logits]
    R --> G[Transport observed t0 raw 3D source geometry by SE(2)]
    K --> A[Hard A1 composition]
    G --> A
    A --> O[6 future 18-class occupancy grids]
```

The learned network consumes a **2D source-centred local semantic tube**, not a 3D crop. The renderer transports the **original observed 3D source voxels**. `existence_logits` are trained, but the frozen deployment path does **not** gate predictions with existence. Future ego poses are part of the input protocol for deterministic ego-motion/KTA transport and rendering; future semantics, instances, and yaw are never prediction inputs.

## Frozen method contract

The main model predicts KTA-relative XY residuals and relative yaw at six future offsets (0.5–3.0 s). For an observed source centre `c_s`, matched GT object centres `a_0, a_h`, and GT relative yaw `R`, the source-centred target is

`d_s = (a_h - a_0) + (R - I)(c_s - a_0)`.

The observed 3D source points are then transported by

`p' = R(p - c_s) + c_s + d_s`.

Yaw is enabled for bicycle, bus, car, construction vehicle, motorcycle, trailer, and truck (`2,3,4,5,6,9,10`). Pedestrians (`7`) remain translation-only. The frozen objective is

`L = L_trans + L_exist + 19.0 L_yaw + 0.25 L_shape_SE2`.

`L_shape_SE2` compares the **same observed source footprint** under the predicted and GT source-centred SE(2) transforms. It is not complete-future-scene supervision. See [docs/METHOD_AND_PROTOCOL.md](docs/METHOD_AND_PROTOCOL.md).

## Installation

Python 3.10+ is supported. Core training/inference does not depend on OccFM or a VAE.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[nuscenes,viz,dev]"
```

Core dependencies: NumPy, SciPy, PyTorch, PyYAML. `nuscenes-devkit` is optional until raw-data preparation/evaluation; Matplotlib is optional for visualization. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Data

Expected legal user-provided layout:

```text
<DATA_ROOT>/
├── v1.0-trainval/             # official nuScenes metadata
├── samples/ ...               # normal nuScenes files
└── gts/
    └── scene-xxxx/
        └── <sample-token>/
            └── labels.npz     # Occ3D semantics (+ mask_lidar if present)
```

Frozen protocol: 6 history + 6 future keyframes, 0.5 s interval, grid `200×200×16`, voxel size `0.4 m`, range `x,y∈[-40,40)`, `z∈[-1,5.4)`, labels `0..17` with free=`17`. Official train/val scene split is used. Formal eligible-window counts are **700 scenes / 20,430 train windows** and **150 scenes / 4,369 val windows**.

Check the dataset before building caches:

```bash
python tools/check_data.py --dataroot /path/to/nuscenes
```

Build the standalone caches directly from raw data (no historical private cache chain):

```bash
python tools/prepare_data.py --dataroot /path/to/nuscenes --split train --output data/train_clean_se2.pt
python tools/prepare_data.py --dataroot /path/to/nuscenes --split val   --output data/val_clean_se2.pt
```

The cache stores causal history-derived tensors plus training labels. GT future annotations are used only to build supervision; GT future semantics are read only by evaluation. See [docs/DATA_AND_METRICS.md](docs/DATA_AND_METRICS.md).

## Train Clean-E14 semantics

```bash
python tools/train.py \
  --config configs/final.yaml \
  --train-cache data/train_clean_se2.pt \
  --val-cache data/val_clean_se2.pt \
  --output-dir outputs/clean_e14 \
  --device cuda
```

The command preserves the historical two-stage schedule semantics: epochs 1–10 use the original cosine schedule; then the **same AdamW state** continues at fixed `5e-5` through epoch 14, while Python/NumPy/Torch/DataLoader shuffle RNG are reset to `seed + 10`. Formal E14 is expected at global step **18,410**. It is **not** “14 epochs of cosine”. No epoch is selected on the formal validation set.

For a short plumbing check, use `configs/smoke.yaml` with a small cache made using `--max-windows`.

## Migrate the historical checkpoint

The public model deliberately preserves frozen parameter names and shapes, so the legacy mapping is identity + strict load:

```bash
python tools/migrate_checkpoint.py \
  /path/to/swfm/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt \
  checkpoints/clean_e14.pt
```

The migration performs no numerical conversion. See [docs/CHECKPOINT_MIGRATION.md](docs/CHECKPOINT_MIGRATION.md).

## Inference

```bash
python tools/infer.py \
  --dataroot /path/to/nuscenes \
  --checkpoint checkpoints/clean_e14.pt \
  --split val --window-index 0 \
  --output outputs/prediction.npz \
  --device cuda
```

## Formal evaluation

```bash
python tools/evaluate.py \
  --dataroot /path/to/nuscenes \
  --val-cache data/val_clean_se2.pt \
  --checkpoint checkpoints/clean_e14.pt \
  --output outputs/val_clean_e14.json \
  --bootstrap-samples 2000 \
  --device cuda
```

Strong/KTA baseline only:

```bash
python tools/evaluate_strong_kta.py \
  --dataroot /path/to/nuscenes \
  --output outputs/val_strong_kta.json
```

The formal evaluator reports occupancy IoU, semantic mIoU, original macro Moving-mIoU, horizon-first micro Moving-IoU, and per-class/per-horizon values at 1/2/3 s. It scores the complete frozen `200×200×16` grid; it does **not** silently add a LiDAR-validity mask. Scene bootstrap resamples scenes and re-sums all selected overlapping-window raw intersections/unions before recomputing metrics.

## Historical frozen results

These are **historical values recorded in the source repository**, not a claim that this extracted repository has re-run the full dataset yet:

| Method | IoU | mIoU | Macro Moving | Micro Moving |
|---|---:|---:|---:|---:|
| Strong/KTA | 53.0900 | 40.0515 | 25.8641 | 27.8619 |
| Y600-pred (result reference only) | 53.5783 | 42.9892 | 26.8855 | 30.0794 |
| Clean-E14 | 53.6142 | 43.1178 | 27.2753 | 30.9144 |

Y600 is intentionally not shipped as a historical training path.

## Runtime

The repository retains the frozen bit-exact-oriented fast primitives under `src/causal_se2_occ/runtime/fastpath.py` and unit-tests them against the reference implementations. A standalone frozen-boundary benchmark with real-window exactness gates is available:

```bash
python tools/benchmark_runtime.py \
  --dataroot /path/to/nuscenes \
  --val-cache data/val_clean_se2.pt \
  --checkpoint checkpoints/clean_e14.pt \
  --output outputs/runtime.json \
  --warmup-windows 20 --measure-windows 200 --exactness-windows 8 --device cuda
```

The source repository's frozen L40S/BF16 runtime record was: learned forward `7.4003 ms`; Strong/KTA six-frame prior `86.3618 ms`; complete six-frame generation `100.7462 ms`; `9.9259 windows/s`; future-frame-amortized `59.5556 FPS`; source extraction + matching `31.7073 ms`. The complete-generation timer included prior reconstruction + network forward + predicted SE(2) rendering + hard A1 composition. I/O, GT construction, and metrics were excluded. These numbers remain **historical until the server acceptance command revalidates the extracted fast path**. See [docs/RUNTIME.md](docs/RUNTIME.md).

## Visualization

```bash
python tools/visualize.py outputs/prediction.npz --horizon-index 5 --output outputs/pred_3s.png
```

## Verification status

Local CPU/synthetic tests cover direct history-only data preparation and cache round-trip, frozen parameter count, zero-head initialization, strict identity checkpoint mapping, source-centred SE(2), yaw class rule, original 3D geometry transport, hard A1 write order, raw metrics, shape loss/gradients, optimizer save/restore, and fast-path equivalence. See [docs/LOCAL_VALIDATION.md](docs/LOCAL_VALIDATION.md) for the executed environment/commands. Real-data, legacy-checkpoint, and CUDA acceptance items are explicitly listed in [docs/REPRODUCTION.md](docs/REPRODUCTION.md); missing data/GPU checks are never reported as passes.

```bash
pytest -q
```

## Provenance and scope

See [docs/PROVENANCE.md](docs/PROVENANCE.md) for exact old-repository commits and the keep/remove/rename map. The old repository remains the experiment archive and is not modified by this extraction.

## Limitations

The current method requires future ego poses as part of the evaluation/deployment protocol. It transports geometry observed at `t0`; it does not generate new unseen object shape. Source extraction is occupancy-based and inherits Strong/KTA matching assumptions. Existence is supervised but is not a deployment gate. Formal performance/runtime should only be claimed after the real-data/CUDA acceptance checks pass on the intended environment.
