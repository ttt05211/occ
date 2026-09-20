# Paper-result provenance

The following values are frozen historical records from `ttt05211/swfm`; they have not yet been re-run by this extracted repository in the current environment.

| Method | IoU | mIoU | Macro Moving-mIoU | Micro Moving-IoU |
|---|---:|---:|---:|---:|
| Strong/KTA | 53.0900 | 40.0515 | 25.8641 | 27.8619 |
| Y600-pred | 53.5783 | 42.9892 | 26.8855 | 30.0794 |
| Clean-E14 | 53.6142 | 43.1178 | 27.2753 | 30.9144 |

Formal main configuration: `configs/final.yaml`; formal checkpoint: historical `outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt`; formal population: 150 val scenes / 4,369 eligible overlapping windows; reporting horizons: 1/2/3 s.
