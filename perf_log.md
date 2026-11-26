# VIctory Performance Optimization Log (Target: >= 30.0 Avg FPS)

Hardware: NVIDIA GeForce RTX 4080 (16 GB VRAM, ReBAR GPU Upload supported), Windows 11, Xbox Series S profile (`--console series-s --execute`).
Harness: `scripts/bench_fps.py` (`cargo build --release` → launch → auto-`Enter` → 120s scene load wait → 20s real `Present()` measurement window → force teardown).

| Iter | Commit | Change | Avg FPS | 1% Low FPS | Frames (20s) | Avg ms | 1% Low ms | Status |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 0 | baseline | Phase 1 harness + Present frametime CSV logger (`VICTORY_FRAMETIMES`) | **14.31** | **10.94** | 286 | 69.89 | 91.45 | Baseline |

## Baseline Bottleneck Breakdown (Iter 0 — 14.31 FPS / 69.89 ms/frame)
- **Primary wall — single `behind` submitting thread: 67.5 ms/frame** (`the card idle behind us: 28.2 ms/frame`, GPU only ~59% busy, ~41 ms GPU time, title threads waiting 28.5–29.0 ms at presents).
- Breakdown of submitting thread + parallel CPU/GPU work:
  - **Host buffer copies / refresh (`copies`)**: **22.6 ms/frame** (108.7 MB refreshed; 56.4 MB sent up in 9,360 ranges; 9,190 changed runs under shader writes).
  - **Descriptor heap byte-sync (`descriptor::sync`)**: **23.8 ms total / 9.9 ms waited for on submitting thread** (214 changed heap pages, 230k slots scanned, 12.8k mirrored, 16.4k raw twins made/frame; 3.6–5.8 ms asking, 2.8 ms reading/comparing, 8.8 ms making).
  - **Heap buffer pass (`heap buffers`)**: **5.6 ms/frame**.
  - **Card steps (`VICTORY_CARD_STEPS=2`)**: 3,339 fills / 1,278 KB per frame.
  - **GPU card passes**: 290 hash passes + 467 compare passes per frame; 906 pieces fetched (48.6 MB) with 11.5 waits (7.1 ms on relay thread); 17.0 ms copying back 28.8 MB.
  - **Draws**: 5,388 draws/frame (268 dropped without host PSO, 921 indirect).
