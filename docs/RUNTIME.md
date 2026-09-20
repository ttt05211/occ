# Runtime protocol

Report separately: learned forward; Strong/KTA six-frame prior reconstruction; predicted SE(2) rendering + hard A1; full six-frame generation; source extraction + matching. I/O, GT construction, and metrics are outside the frozen paper timer. FLOPs, if reported, cover learned forward only.

Historical L40S/BF16 record (20 warmup, 200 measured, 8 exactness, seed 20260918): 2.073220 M params; 28.570391 GFLOPs learned forward; 7.4003 ms learned forward; 86.3618 ms Strong/KTA prior; 100.7462 ms six-frame generation; 9.9259 windows/s; 59.5556 future-frame-amortized FPS; 31.7073 ms source extraction/matching; 45.2989 FPS when additionally charging source extraction; peak 213,963,776 bytes.

These are historical measurements until the server re-runs exactness and timing.

The retained fast path includes sparse exact 5×5×1 majority fill, cropped connected components, sparse A1 CLEAR indices, and the CUDA-oriented exactness path. Runtime optimizations must pass reference equivalence before paper timing is accepted.
