# Checkpoint migration

Historical formal checkpoint: `outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt` in the original experiment environment.

The extracted model deliberately preserves the legacy module/parameter names and tensor shapes. Migration therefore performs an **identity mapping** and `strict=True` state-dict load; it does not interpolate, rename, cast, or numerically rewrite weights.

```bash
python tools/migrate_checkpoint.py OLD_E14.pt checkpoints/clean_e14.pt
```

Expected historical metadata: protocol `p0_f9_v18_se2_clean_train_v1`; arm `Y`; epoch `14`; global step `18,410`; final mode `clean_one_stage_from_scratch_v1_tail_continuation`.

The default model has exactly **2,073,220 parameters**. Local tests verify strict identity migration on a synthetic checkpoint. The actual historical E14 binary was not available in this execution environment, so a real strict-load run remains a server acceptance item.
