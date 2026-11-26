//! Pure-Rust NVIDIA `cuda-oxide` SIMT kernel definitions for Victory.
//!
//! This crate is the canonical Rust source twin of `../kernels/victory_kernels.cu` using
//! `cuda-oxide`'s `#[cuda_module]`, `#[kernel]`, `#[launch_bounds]`, `#[launch_contract]`,
//! and `DisjointSlice<T>` abstractions for zero-UB SIMT execution on Ada Lovelace (`sm_89`).
//!
//! Build into PTX once `cargo-oxide` + Rust `1.99.1` (`rustc-dev`, `llvm-tools`, `rust-src`) +
//! LLVM 21+ + CUDA 13.0+ headers are installed:
//!
//! ```text
//! cd crates/cuda_accel/kernels_oxide
//! cargo oxide build --release --features cuda-device
//! ```
//!
//! `crates/cuda_accel/build.rs` automatically picks up the emitted PTX via
//! `VICTORY_CUDA_OXIDE_PTX` or `target/nvptx64-nvidia-cuda/release/victory_kernels_oxide.ptx`.

#![no_std]

/// Per-pattern offsets into the flat GFX10 RB+ swizzle LUT array.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LutMeta {
    pub x_off: u32,
    pub y_off: u32,
    pub z_off: u32,
    pub z_len: u32,
}

/// C-ABI descriptor matching `victory_gpu::graphics::contracts::DetileDispatchDesc`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DetileDispatchDesc {
    pub src_base_offset: u64,
    pub dst_base_offset: u64,
    pub width_elements: u32,
    pub height_elements: u32,
    pub depth_slices: u32,
    pub element_bytes: u32,
    pub block_dims: [u32; 3],
    pub block_log2: u32,
    pub mip_pitch_elements: u32,
    pub dst_row_pitch_bytes: u32,
    pub dst_depth_pitch_bytes: u32,
    pub tail_x: u32,
    pub tail_y: u32,
    pub pattern_id: u32,
    pub thick: u32,
    pub slice_size: u64,
    pub first_slice: u32,
    pub _pad: u32,
}

/// C-ABI descriptor matching `victory_gpu::graphics::contracts::DecompressBlockDesc`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DecompressBlockDesc {
    pub src_offset: u64,
    pub src_len: u32,
    pub dst_len: u32,
    pub dst_offset: u64,
}

/// C-ABI descriptor matching `victory_gpu::graphics::contracts::ReadbackPieceDesc`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReadbackPieceDesc {
    pub now_offset: u64,
    pub held_offset: u64,
    pub dword_count: u32,
    pub piece_id: u32,
}

/// C-ABI descriptor matching `victory_gpu::graphics::contracts::ReadbackDeltaPair`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReadbackDeltaPair {
    pub piece_id: u32,
    pub dword_offset: u32,
    pub value: u32,
}

/// Pure-Rust RDNA2 (GFX10.3 RB+) element source byte offset calculation.
///
/// Shared between `#[kernel] rdna2_detile_batch` and host validation so the
/// addressing formula stays 100% single-sourced with `victory_formats::texture::Layout::offset`.
#[inline(always)]
pub fn rdna2_element_src_offset(
    d: &DetileDispatchDesc,
    x: u32,
    y: u32,
    z: u32,
    lut_data: &[u32],
    lut_meta: &[LutMeta],
) -> u64 {
    let z_abs = d.first_slice + z;
    let (slice_idx, inner_z) = if d.thick != 0 {
        let bd = d.block_dims[2];
        (z_abs / bd, z_abs % bd)
    } else {
        (z_abs, 0)
    };
    let base = d.src_base_offset + (slice_idx as u64) * d.slice_size;
    if d.pattern_id == u32::MAX || d.block_log2 == 0 {
        let row = (y as u64) * (d.mip_pitch_elements as u64) + (x as u64);
        return base + row * (d.element_bytes as u64);
    }
    let bw = d.block_dims[0];
    let bh = d.block_dims[1];
    let tx = x + d.tail_x;
    let ty = y + d.tail_y;
    let row_blocks = ((ty / bh) as u64) * ((d.mip_pitch_elements / bw) as u64);
    let col_blocks = (tx / bw) as u64;
    let m = lut_meta[d.pattern_id as usize];
    let lut_x = lut_data[(m.x_off + (tx % bw)) as usize];
    let lut_y = lut_data[(m.y_off + (ty % bh)) as usize];
    let lut_z = if m.z_len > 0 {
        lut_data[(m.z_off + inner_z) as usize]
    } else {
        0
    };
    base + ((row_blocks + col_blocks) << d.block_log2) + (((lut_y | lut_z) + lut_x) as u64)
}

/// Pure-Rust `cuda-oxide` device kernel module (compiled under `cargo oxide --features cuda-device`).
///
/// Uses `cuda-oxide`'s `#[cuda_module]`, `#[kernel]`, `#[launch_bounds]`, `#[launch_contract]`,
/// and `DisjointSlice<T>` to emit SM 8.9 PTX matching `kernels/victory_kernels.cu`.
#[cfg(feature = "cuda-device")]
#[cuda_oxide::cuda_module]
pub mod device_kernels {
    use super::*;
    use cuda_std::slice::DisjointSlice;
    use cuda_std::thread::{block_idx_x, thread_idx_x};
    use cuda_std::warp;

    /// Kernel 1: Xbox RDNA2 micro-tile texture detiling directly into DX12-shared VRAM.
    #[cuda_oxide::kernel]
    #[cuda_oxide::launch_bounds(256)]
    #[cuda_oxide::launch_contract(block_dim = [256, 1, 1])]
    pub unsafe fn rdna2_detile_batch(
        tiled_src: &[u8],
        mut linear_dst: DisjointSlice<'_, u8>,
        dispatches: &[DetileDispatchDesc],
        dispatch_block_prefix: &[u32],
        num_dispatches: u32,
        lut_data: &[u32],
        lut_meta: &[LutMeta],
    ) {
        let blk = block_idx_x();
        let mut lo = 0u32;
        let mut hi = num_dispatches;
        while lo + 1 < hi {
            let mid = (lo + hi) >> 1;
            if dispatch_block_prefix[mid as usize] <= blk {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        if lo >= num_dispatches {
            return;
        }
        let d = dispatches[lo as usize];
        let block_start = dispatch_block_prefix[lo as usize];
        let plane = (d.width_elements as u64) * (d.height_elements as u64);
        let total = plane * (d.depth_slices as u64);
        let elem_idx = ((blk - block_start) as u64) * 256 + (thread_idx_x() as u64);
        if elem_idx >= total {
            return;
        }
        let z = (elem_idx / plane) as u32;
        let rem = (elem_idx - (z as u64) * plane) as u32;
        let y = rem / d.width_elements;
        let x = rem - y * d.width_elements;

        let src_off = rdna2_element_src_offset(&d, x, y, z, lut_data, lut_meta) as usize;
        let dst_off = (d.dst_base_offset
            + (z as u64) * (d.dst_depth_pitch_bytes as u64)
            + (y as u64) * (d.dst_row_pitch_bytes as u64)
            + (x as u64) * (d.element_bytes as u64)) as usize;
        let bpe = d.element_bytes as usize;
        for i in 0..bpe {
            linear_dst.write(dst_off + i, tiled_src[src_off + i]);
        }
    }

    /// Kernel 3: Warp-level sparse GPU-to-CPU readback delta compaction using `ballot_sync`.
    #[cuda_oxide::kernel]
    #[cuda_oxide::launch_bounds(256)]
    #[cuda_oxide::launch_contract(block_dim = [256, 1, 1])]
    pub unsafe fn compact_readback_deltas(
        now_buf: &[u32],
        held_buf: &[u32],
        pieces: &[ReadbackPieceDesc],
        num_pieces: u32,
        out_pairs: *mut ReadbackDeltaPair,
        max_pairs: u32,
        global_counter: *mut u32,
        overflow_flag: *mut u32,
    ) {
        let piece_idx = block_idx_x();
        if piece_idx >= num_pieces {
            return;
        }
        let p = pieces[piece_idx as usize];
        let now_base = (p.now_offset >> 2) as usize;
        let held_base = (p.held_offset >> 2) as usize;
        let tid = thread_idx_x();
        let lane = tid & 31;
        let lane_mask_lt = if lane == 0 { 0 } else { (1u32 << lane) - 1 };

        let mut base = 0u32;
        while base < p.dword_count {
            let idx = base + tid;
            let valid = idx < p.dword_count;
            let v_now = if valid { now_buf[now_base + idx as usize] } else { 0 };
            let v_held = if valid { held_buf[held_base + idx as usize] } else { 0 };
            let changed = valid && v_now != v_held;
            let mask = warp::ballot_sync(0xFFFF_FFFF, changed);
            if mask != 0 {
                let warp_count = mask.count_ones();
                let mut warp_base = 0u32;
                if lane == 0 {
                    warp_base = core::intrinsics::atomic_xadd_relaxed(global_counter, warp_count);
                }
                warp_base = warp::shfl_sync(0xFFFF_FFFF, warp_base, 0);
                if warp_base + warp_count > max_pairs {
                    if lane == 0 {
                        core::intrinsics::atomic_or_relaxed(overflow_flag, 1);
                    }
                } else if changed {
                    let out_idx = warp_base + (mask & lane_mask_lt).count_ones();
                    out_pairs.add(out_idx as usize).write(ReadbackDeltaPair {
                        piece_id: p.piece_id,
                        dword_offset: idx,
                        value: v_now,
                    });
                }
            }
            base += 256;
        }
    }
}
