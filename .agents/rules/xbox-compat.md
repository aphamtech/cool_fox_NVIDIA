---
trigger: glob
globs: "**/*.rs, **/*.hlsl, Cargo.toml"
description: "Hardware translation invariants and Rust safety rules for the Xbox Series X|S to NVIDIA RTX compatibility layer."
---
# Xbox Series X|S -> NVIDIA RTX Translation Invariants

1. **No Hallucinated Proprietary Specs:** Never invent bit-level decoding logic for Microsoft's proprietary BCPack format. Define a pluggable `DecompressionBackend` trait, fully implement standard paths (GDeflate/LZ4/Zlib via PC DirectStorage or `VK_NV_memory_decompression`), and leave BCPack as a strictly typed, documented stub behind that trait.
2. **Accurate RDNA 2 -> NVIDIA Mapping:**
   - RDNA 2 defaults to **Wave32** (matching NVIDIA's 32-thread warps). Only emit Wave64-to-Wave32 emulation when shader metadata explicitly specifies `WaveSize(64)`.
   - Rely on native DX12 Tier 3 Tiled Resources for NULL-tile `(0,0,0,0)` reads; focus safety guards on preventing unmapped GPU virtual address (VA) faults on aliased `PlacedResource` heaps.
3. **Zero Render-Thread Stalls:** Never block command-recording threads on synchronous CPU staging copies or CPU decompression. Use lock-free ring buffers (`crossbeam` or `ringbuf`) and async Copy/Compute queues synchronized via `ID3D12Fence` / timeline semaphores.
4. **Strict FFI Isolation:** Encapsulate all `unsafe` COM/DX12/Vulkan FFI calls inside thin, audited wrapper modules exposing safe RAII handles.
5. **Real Verification Only:** Every change must pass `cargo check`, `cargo test`, and `cargo clippy` in the terminal sandbox. Never mock away failing assertions.