# Provenance and migration map

Source repository: `ttt05211/swfm`.

| Role | Commit / branch |
|---|---|
| Method implementation baseline | `f379739cd074496a747851d683dbc691ce3be697` |
| Final freeze/checkpoint contract | `ccf7d77e65e9773f441b35083d625b06791bfeaa` on `freeze/v18-main-final-20260918` |
| Full-validation freeze | `2b3841ecc376af07c1af0246709db2e230c97a7f`, strict import fix `f379739c...` |
| Runtime engineering | around `30ede52191846af741b67be79bc4f364c4fdc463` and later exact refinements |
| Paper runtime freeze | `b0ca8e3c58667a65190f554a3fc26fc2dcd75e2d`, clarified by `c08c005e6aef8c020ac62e8aaef1b49723eac12c` |
| Later feature tip inspected | `283ff822171884e21352ca6f81ee27b8a76485c5` |

Freeze documents: `docs/p0_f9_v18_main_experiment_freeze_20260917.md`, `docs/p0_f9_v18_paper_main_table_freeze_20260918.md`, `docs/p0_f9_v18_runtime_inception_protocol_20260918.md` in the old repository.

Kept: Clean source-centred SE(2), Strong/KTA, causal source representation, KTA-relative XY/yaw/existence heads, raw 3D source transport, hard A1, frozen objective/schedule, formal metrics/bootstrap, verified runtime primitives, minimal tools.

Removed: latent flow/VAE routes; STPN/backtrace; selector/MSP/Need-Score/router; motion-density resampling; two-wheel special rules; scene CE/Moving-Safe/anti-regret; unvalidated temporal/FVD losses; historical repair scripts/private cache chains.

Historical formal checkpoint: `outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt`. Data/caches/checkpoints are not committed here.
