# Reproduction and acceptance

## Frozen training

- seed `20260915`;
- AdamW lr `5e-4`, weight decay `1e-4`;
- batch size 256, BF16 CUDA, global grad clip 5;
- epochs 1–10 cosine scale 1.0→0.1, LR updated after every optimizer step;
- formal steps/epoch 1315;
- tail entry: retain same AdamW state, reset Python/NumPy/Torch/CUDA + DataLoader RNG to `seed+10 = 20260925`, fixed LR `5e-5`;
- epochs 11–14 unchanged objective/data/architecture;
- E14 global step 18,410;
- checkpoint every epoch/latest; no formal best.pt selection.

## Commands

```bash
python tools/check_data.py --dataroot /data/nuscenes
python tools/prepare_data.py --dataroot /data/nuscenes --split train --output data/train_clean_se2.pt
python tools/prepare_data.py --dataroot /data/nuscenes --split val --output data/val_clean_se2.pt
python tools/train.py --config configs/final.yaml --train-cache data/train_clean_se2.pt --val-cache data/val_clean_se2.pt --output-dir outputs/clean_e14 --device cuda
python tools/migrate_checkpoint.py /legacy/epoch_0014.pt checkpoints/clean_e14.pt
python tools/evaluate.py --dataroot /data/nuscenes --val-cache data/val_clean_se2.pt --checkpoint checkpoints/clean_e14.pt --output outputs/formal_eval.json --bootstrap-samples 2000 --device cuda
python tools/evaluate_strong_kta.py --dataroot /data/nuscenes --output outputs/strong_kta.json
python tools/benchmark_runtime.py --dataroot /data/nuscenes --val-cache data/val_clean_se2.pt --checkpoint checkpoints/clean_e14.pt --output outputs/runtime.json --warmup-windows 20 --measure-windows 200 --exactness-windows 8 --device cuda
```

## Local PASS

Install/import in provided host environment; 2,073,220 parameter count; zero residual/yaw init; synthetic identity strict migration; source-centred SE(2); pedestrian translation-only; raw 3D transport/A1; hand-computable metrics/bootstrap; shape loss/gradients; optimizer save/restore; sparse majority/cropped components/fast A1 equivalence; direct causal data preparation/cache round-trip.

## Server NOT RUN

Actual E14 strict load/hash and old/new output+gradient parity; representative real-window Strong/KTA/raster/A1 parity; exact evaluator raw-count parity; CUDA training resume; full 150-scene/4,369-window validation; real CUDA runtime exactness and timing. Missing data/GPU checks must remain `NOT RUN` until executed.
