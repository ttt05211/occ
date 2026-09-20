# Handoff

This repository is the standalone extraction of the frozen Clean-E14 source-centred SE(2) method. The old `ttt05211/swfm` repository remains the experiment archive and should not be rewritten or replaced.

For paper numbers, use `configs/final.yaml`, the 150-scene / 4,369-window validation protocol, and the historical E14 checkpoint only. Do not substitute val128, Y600, Balanced, two-wheel-special, anti-regret, router/MSP, latent-flow, or FVD branches.

Before claiming a new-repository reproduction, complete the server items in `docs/REPRODUCTION.md`, especially real E14 strict loading, old/new real-window parity, exact raw metric-count parity, full validation, and CUDA runtime exactness. Historical values in the README are provenance records until those checks pass.

Checkpoint migration is identity + strict state-dict loading. The default model has 2,073,220 parameters. Future ego poses remain part of the frozen input protocol. Existence logits remain ungated in deployment.
