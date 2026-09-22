#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from causal_se2_occ.data.cache import flatten_supervised, load_cache
from causal_se2_occ.evaluation import evaluate_objective
from causal_se2_occ.models.stwm import ModelConfig, SourceCenteredSE2Predictor
from causal_se2_occ.protocol import (
    CHECKPOINT_PROTOCOL,
    MODEL_PROTOCOL,
    SE2_SHAPE_CONTRACT,
    SE2_TARGET_CONTRACT,
)
from causal_se2_occ.training import (
    capture_rng_state,
    make_loader,
    make_loader_with_state,
    move_batch,
    restore_rng_state,
    seed_all,
    train_epoch,
)


def save_checkpoint(
    path,
    *,
    model,
    optimizer,
    model_config,
    train_config,
    epoch,
    global_step,
    train_meta,
    val_meta,
    mode,
    val_report,
    train_generator,
    tail_source=None,
):
    payload = {
        "protocol": CHECKPOINT_PROTOCOL,
        "model_protocol": MODEL_PROTOCOL,
        "arm": "Y",
        "training_mode": mode,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "continuation_step": int(global_step),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "variant": "CLEAN-SE2",
        "use_representation": True,
        "overlap_weight": train_config["shape_weight"],
        "yaw_weight": train_config["yaw_weight"],
        "shape_weight": train_config["shape_weight"],
        "safe_weight": None,
        "safe_margin": None,
        "se2_target_contract": SE2_TARGET_CONTRACT,
        "se2_shape_contract": SE2_SHAPE_CONTRACT,
        "seed": train_config["seed"],
        "steps_per_epoch": train_config["steps_per_epoch"],
        "schedule_total_steps": (
            train_config["cosine_epochs"] * train_config["steps_per_epoch"]
        ),
        "tail_lr": (
            train_config["tail_lr"]
            if epoch > train_config["cosine_epochs"]
            else None
        ),
        "tail_source_checkpoint": str(tail_source) if tail_source else None,
        "args": train_config,
        "train_cache_metadata": train_meta,
        "val_cache_metadata": val_meta,
        "val_report": val_report,
        # Extraction-only resume state. The historical E10 -> E11 transition
        # still performs its explicit seed+10 reset before epoch 11.
        "rng_state": capture_rng_state(),
        "train_generator_state": (
            train_generator.get_state() if train_generator is not None else None
        ),
    }
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    model_config = ModelConfig(**config["model"])
    train_config = dict(config["training"])

    train_meta, train_records = load_cache(args.train_cache)
    val_meta, val_records = load_cache(args.val_cache)
    train = flatten_supervised(train_records)
    val = flatten_supervised(val_records)

    overlap = set(train["scene_ids"]) & set(val["scene_ids"])
    if overlap:
        raise RuntimeError(f"train/val scene overlap: {sorted(overlap)[:5]}")

    if bool(train_config.get("expect_clean_e14_step", False)):
        for name, meta, records, expected_scenes, expected_windows in [
            ("train", train_meta, train_records, 700, 20430),
            ("val", val_meta, val_records, 150, 4369),
        ]:
            if (
                str(meta.get("split")) != name
                or int(meta.get("scene_count", -1)) != expected_scenes
                or int(meta.get("eligible_windows", -1)) != expected_windows
                or len(records) != expected_windows
            ):
                raise RuntimeError(
                    f"formal Clean-E14 requires {name} cache="
                    f"{expected_scenes} scenes/{expected_windows} windows"
                )

    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    amp = device.type == "cuda" and train_config.get("amp", "bf16") == "bf16"

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing non-empty output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(train_config["seed"])
    seed_all(seed)

    model = SourceCenteredSE2Predictor(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"]),
    )

    batch_size = int(train_config["batch_size"])
    workers = int(train_config["num_workers"])
    cosine_epochs = int(train_config["cosine_epochs"])
    final_epoch = int(train_config["final_epoch"])
    patch_resolution = float(train_meta.get("patch_resolution_m", 0.8))

    val_loader = make_loader(
        val,
        batch_size,
        workers,
        seed,
        False,
        device,
    )

    global_step = 0
    start_epoch = 0
    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_ckpt["state_dict"], strict=True)
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        global_step = int(resume_ckpt["global_step"])
        start_epoch = int(resume_ckpt["epoch"])

    # Build train loader only after the resume checkpoint is known.
    # Historical E10 -> E11 deliberately ignores previous RNG state and resets
    # everything to seed+10. Other extracted checkpoints restore exact state.
    if start_epoch == cosine_epochs:
        train_loader, train_generator = make_loader_with_state(
            train,
            batch_size,
            workers,
            seed + cosine_epochs,
            True,
            device,
        )
    elif start_epoch > 0:
        if start_epoch < final_epoch and (
            resume_ckpt.get("rng_state") is None
            or resume_ckpt.get("train_generator_state") is None
        ):
            raise RuntimeError(
                "deterministic resume requires rng_state and "
                "train_generator_state; historical E10 is the only supported "
                "legacy boundary because epoch 11 intentionally reseeds"
            )
        if resume_ckpt.get("rng_state") is not None:
            restore_rng_state(resume_ckpt["rng_state"])
        train_loader, train_generator = make_loader_with_state(
            train,
            batch_size,
            workers,
            seed,
            True,
            device,
            generator_state=resume_ckpt.get("train_generator_state"),
        )
    else:
        train_loader, train_generator = make_loader_with_state(
            train,
            batch_size,
            workers,
            seed,
            True,
            device,
        )

    steps_per_epoch = len(train_loader)
    train_config["steps_per_epoch"] = steps_per_epoch
    expected_spe = train_config.get("expected_steps_per_epoch")
    if expected_spe and int(expected_spe) != steps_per_epoch:
        raise RuntimeError(
            f"formal steps/epoch mismatch: expected {expected_spe}, "
            f"got {steps_per_epoch}"
        )

    if not args.resume:
        first = next(iter(val_loader))
        batch0 = move_batch(first, device)
        model.eval()
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=amp,
        ):
            initial = model(
                batch0["features"],
                batch0["local_semantic_tube"],
                batch0["kta_displacement_xy_m"],
                batch0["frame_motion_features"],
                batch0["target_source_mask_tube"],
            )
        if (
            float(initial["residual_xy_m"].abs().max()) > 1e-7
            or float(initial["yaw_delta_rad"].abs().max()) > 1e-7
        ):
            raise RuntimeError(
                "fresh model violates zero-residual/zero-yaw initialization"
            )

    total_steps = cosine_epochs * steps_per_epoch
    tail_source = (
        Path(args.resume)
        if start_epoch == cosine_epochs and args.resume
        else output_dir / f"epoch_{cosine_epochs:04d}.pt"
    )

    for epoch in range(start_epoch + 1, final_epoch + 1):
        if epoch == cosine_epochs + 1:
            # Frozen historical tail boundary.
            seed_all(seed + cosine_epochs)
            train_loader, train_generator = make_loader_with_state(
                train,
                batch_size,
                workers,
                seed + cosine_epochs,
                True,
                device,
            )
            for group in optimizer.param_groups:
                group["lr"] = float(train_config["tail_lr"])

        fixed_lr = (
            float(train_config["tail_lr"])
            if epoch > cosine_epochs
            else None
        )

        global_step, train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            amp=amp,
            yaw_weight=float(train_config["yaw_weight"]),
            shape_weight=float(train_config["shape_weight"]),
            patch_resolution=patch_resolution,
            global_step=global_step,
            base_lr=float(train_config["lr"]),
            total_steps=total_steps,
            fixed_lr=fixed_lr,
            grad_clip=float(train_config["grad_clip"]),
        )

        val_report = evaluate_objective(
            model,
            val_loader,
            device,
            amp=amp,
            yaw_weight=float(train_config["yaw_weight"]),
            shape_weight=float(train_config["shape_weight"]),
            patch_resolution=patch_resolution,
        )

        mode = (
            "clean_one_stage_from_scratch_v1_tail_continuation"
            if epoch > cosine_epochs
            else "clean_one_stage_from_scratch_v1"
        )

        for checkpoint_path in (
            output_dir / f"epoch_{epoch:04d}.pt",
            output_dir / "latest.pt",
        ):
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                train_config=train_config,
                epoch=epoch,
                global_step=global_step,
                train_meta=train_meta,
                val_meta=val_meta,
                mode=mode,
                val_report=val_report,
                train_generator=train_generator,
                tail_source=tail_source if epoch > cosine_epochs else None,
            )

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_report.items()},
        }
        print(json.dumps(row), flush=True)

    if (
        bool(train_config.get("expect_clean_e14_step", False))
        and global_step != 18410
    ):
        raise RuntimeError(
            f"Clean-E14 provenance expects global_step=18410; got {global_step}"
        )


if __name__ == "__main__":
    main()
