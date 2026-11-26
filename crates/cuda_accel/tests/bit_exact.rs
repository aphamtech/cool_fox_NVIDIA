//! Bit-exact GPU-vs-CPU parity tests for all `cuda-accel` SM 8.9 PTX kernels.
//!
//! Runs directly on the host NVIDIA GPU (`nvcuda.dll` + RTX 4080) and verifies
//! byte-for-byte identity against `victory_formats::texture::Layout::detile`,
//! `lz4_flex::block::decompress`, `zlib_rs` raw DEFLATE, and CPU dword diffing.

use cuda_accel::{
    CudaAccelContext, CudaDecompressCodec, DecompressBlockDesc, DetileDispatchDesc,
    ReadbackDeltaPair, ReadbackPieceDesc, detile_desc_from_layout,
};
use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};

fn deterministic_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut out = vec![0u8; len];
    for b in &mut out {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *b = (state & 0xFF) as u8;
    }
    out
}

fn require_cuda() -> CudaAccelContext {
    CudaAccelContext::new(None).expect("NVIDIA CUDA SM 8.9 context must initialize on RTX GPU")
}

#[test]
fn detile_all_rdna2_swizzle_modes_mips_tails_3d_bit_exact() {
    let ctx = require_cuda();

    let cases: &[(&str, Desc)] = &[
        (
            "linear_2d_rgba8_mips",
            Desc {
                dimension: Dimension::Tex2d,
                width: 137,
                height: 91,
                depth: 1,
                mips: 5,
                element_bytes: 4,
                block: 1,
                samples: 1,
                swizzle: Swizzle::LINEAR,
            },
        ),
        (
            "sw_256b_s_r8",
            Desc {
                dimension: Dimension::Tex2d,
                width: 129,
                height: 65,
                depth: 1,
                mips: 4,
                element_bytes: 1,
                block: 1,
                samples: 1,
                swizzle: Swizzle::SW_256B_S,
            },
        ),
        (
            "sw_256b_d_rg8",
            Desc {
                dimension: Dimension::Tex2d,
                width: 96,
                height: 80,
                depth: 1,
                mips: 4,
                element_bytes: 2,
                block: 1,
                samples: 1,
                swizzle: Swizzle::SW_256B_D,
            },
        ),
        (
            "sw_4kb_s_bc1_array_with_tail",
            Desc {
                dimension: Dimension::Tex2d,
                width: 256,
                height: 128,
                depth: 3,
                mips: 9,
                element_bytes: 8,
                block: 4,
                samples: 1,
                swizzle: Swizzle::SW_4KB_S,
            },
        ),
        (
            "sw_4kb_d_rgba8_odd_with_tail",
            Desc {
                dimension: Dimension::Tex2d,
                width: 313,
                height: 197,
                depth: 2,
                mips: 9,
                element_bytes: 4,
                block: 1,
                samples: 1,
                swizzle: Swizzle::SW_4KB_D,
            },
        ),
        (
            "sw_64kb_s_bc7_full_mipchain",
            Desc {
                dimension: Dimension::Tex2d,
                width: 512,
                height: 512,
                depth: 1,
                mips: 10,
                element_bytes: 16,
                block: 4,
                samples: 1,
                swizzle: Swizzle::SW_64KB_S,
            },
        ),
        (
            "sw_64kb_d_rgba16f_array_with_tail",
            Desc {
                dimension: Dimension::Tex2d,
                width: 256,
                height: 256,
                depth: 2,
                mips: 9,
                element_bytes: 8,
                block: 1,
                samples: 1,
                swizzle: Swizzle::SW_64KB_D,
            },
        ),
        (
            "sw_4kb_s_3d_thick_rgba8_with_tail",
            Desc {
                dimension: Dimension::Tex3d,
                width: 32,
                height: 24,
                depth: 16,
                mips: 5,
                element_bytes: 4,
                block: 1,
                samples: 1,
                swizzle: Swizzle::SW_4KB_S,
            },
        ),
        (
            "sw_64kb_s_3d_thick_bc7_odd",
            Desc {
                dimension: Dimension::Tex3d,
                width: 64,
                height: 48,
                depth: 11,
                mips: 4,
                element_bytes: 16,
                block: 4,
                samples: 1,
                swizzle: Swizzle::SW_64KB_S,
            },
        ),
    ];

    for (idx, (label, desc)) in cases.iter().enumerate() {
        let layout = Layout::new(desc).unwrap_or_else(|e| panic!("{label}: layout failed: {e:?}"));
        let tiled = deterministic_bytes(layout.size as usize, 0x1000 + idx as u64);

        let array_slices = if desc.dimension == Dimension::Tex3d {
            1
        } else {
            desc.depth
        };

        let mut dispatches = Vec::new();
        let mut expected = Vec::new();

        for slice in 0..array_slices {
            for mip in 0..desc.mips {
                let m = &layout.mips[mip as usize];
                let row_bytes = (m.width * desc.element_bytes) as usize;
                // D3D12 256-byte row pitch alignment.
                let row_pitch = row_bytes.next_multiple_of(256);
                let depth_pitch = row_pitch * m.height as usize;
                let sub_len = depth_pitch * m.depth as usize;

                // D3D12 512-byte subresource placement alignment.
                let dst_off = expected.len().next_multiple_of(512);
                expected.resize(dst_off + sub_len, 0);

                layout
                    .detile(
                        &tiled,
                        mip,
                        slice,
                        &mut expected[dst_off..dst_off + sub_len],
                        row_pitch,
                        depth_pitch,
                    )
                    .unwrap_or_else(|e| panic!("{label} mip={mip} slice={slice}: {e:?}"));

                let d = detile_desc_from_layout(
                    &layout,
                    mip,
                    slice,
                    0,
                    dst_off as u64,
                    row_pitch as u32,
                    depth_pitch as u32,
                )
                .unwrap();
                dispatches.push(d);
            }
        }

        let gpu_out = ctx
            .detile_host_sync(&tiled, expected.len(), &dispatches)
            .unwrap_or_else(|e| panic!("{label}: GPU detile failed: {e}"));

        assert_eq!(
            gpu_out, expected,
            "GPU detile mismatch on case `{label}` (dispatches={})",
            dispatches.len()
        );
    }
}

#[test]
fn detile_large_heterogeneous_batch_single_launch() {
    let ctx = require_cuda();

    let mut tiled_batch = Vec::new();
    let mut expected_batch = Vec::new();
    let mut dispatches: Vec<DetileDispatchDesc> = Vec::new();

    for i in 0..32u32 {
        let swizzle = match i % 4 {
            0 => Swizzle::SW_64KB_S,
            1 => Swizzle::SW_64KB_D,
            2 => Swizzle::SW_4KB_S,
            _ => Swizzle::SW_4KB_D,
        };
        let (element_bytes, block) = match i % 3 {
            0 => (16, 4), // BC7
            1 => (8, 4),  // BC1
            _ => (4, 1),  // RGBA8
        };
        let desc = Desc {
            dimension: Dimension::Tex2d,
            width: 128 + (i % 4) * 32,
            height: 128,
            depth: 1,
            mips: 6,
            element_bytes,
            block,
            samples: 1,
            swizzle,
        };
        let layout = Layout::new(&desc).unwrap();
        let src_base = tiled_batch.len().next_multiple_of(256);
        let tiled = deterministic_bytes(layout.size as usize, 0xCAFE_0000 + u64::from(i));
        tiled_batch.resize(src_base, 0);
        tiled_batch.extend_from_slice(&tiled);

        for mip in 0..desc.mips {
            let m = &layout.mips[mip as usize];
            let row_pitch = ((m.width * element_bytes) as usize).next_multiple_of(256);
            let depth_pitch = row_pitch * m.height as usize;
            let dst_off = expected_batch.len().next_multiple_of(512);
            expected_batch.resize(dst_off + depth_pitch, 0);

            layout
                .detile(
                    &tiled,
                    mip,
                    0,
                    &mut expected_batch[dst_off..dst_off + depth_pitch],
                    row_pitch,
                    depth_pitch,
                )
                .unwrap();

            dispatches.push(
                detile_desc_from_layout(
                    &layout,
                    mip,
                    0,
                    src_base as u64,
                    dst_off as u64,
                    row_pitch as u32,
                    depth_pitch as u32,
                )
                .unwrap(),
            );
        }
    }

    let gpu = ctx
        .detile_host_sync(&tiled_batch, expected_batch.len(), &dispatches)
        .expect("32-texture heterogeneous batch detile");
    assert_eq!(gpu, expected_batch);
}

#[test]
fn lz4_batch_decompression_bit_exact() {
    let ctx = require_cuda();

    let mut inputs: Vec<Vec<u8>> = Vec::new();
    // 1. Highly compressible RLE (tests overlapping offset == 1 and long match_len >= 270).
    inputs.push(vec![0x7Au8; 65536]);
    // 2. Repeating 3-byte pattern (tests overlapping offset == 3).
    inputs.push((0..65536).map(|i| [0xAA, 0x55, 0xC3][i % 3]).collect());
    // 3. Incompressible pseudo-random payload (tests lit_len >= 270).
    inputs.push(deterministic_bytes(65536, 0x1234_5678));
    // 4. Structured mixed game-asset blocks of varied lengths (1 KiB .. 64 KiB).
    for k in 0..29usize {
        let len = 1024 + k * 2173;
        let mut buf = vec![0u8; len];
        for (i, b) in buf.iter_mut().enumerate() {
            *b = if (i / 64) % 3 == 0 {
                ((i + k) & 0x0F) as u8
            } else {
                ((i.wrapping_mul(31) ^ (k * 17)) & 0xFF) as u8
            };
        }
        inputs.push(buf);
    }

    let mut packed_src = Vec::new();
    let mut blocks = Vec::new();
    let mut expected = Vec::new();

    for raw in &inputs {
        let compressed = lz4_flex::block::compress(raw);
        let src_off = packed_src.len() as u64;
        packed_src.extend_from_slice(&compressed);

        let dst_off = expected.len() as u64;
        expected.extend_from_slice(raw);

        blocks.push(DecompressBlockDesc {
            src_offset: src_off,
            src_len: compressed.len() as u32,
            dst_len: raw.len() as u32,
            dst_offset: dst_off,
        });
    }

    let (gpu_out, written) = ctx
        .decompress_host_sync(
            CudaDecompressCodec::Lz4RawBlock,
            &packed_src,
            expected.len(),
            &blocks,
        )
        .expect("GPU LZ4 batch decompression");

    for (i, (raw, &w)) in inputs.iter().zip(&written).enumerate() {
        assert_eq!(w as usize, raw.len(), "LZ4 block {i} written length mismatch");
    }
    assert_eq!(gpu_out, expected, "GPU LZ4 output must match lz4_flex bit-for-bit");
}

fn deflate_raw(input: &[u8], level: i32) -> Vec<u8> {
    let config = zlib_rs::DeflateConfig {
        level,
        window_bits: -15,
        ..zlib_rs::DeflateConfig::default()
    };
    let mut out = vec![0u8; input.len() + (input.len() / 8) + 128];
    let (compressed, ret) = zlib_rs::compress_slice(&mut out, input, config);
    assert_eq!(ret, zlib_rs::ReturnCode::Ok);
    compressed.to_vec()
}

#[test]
fn gdeflate_v1_multi_tile_stored_fixed_dynamic_bit_exact() {
    let ctx = require_cuda();

    let mut tiles_plain: Vec<Vec<u8>> = Vec::new();
    let mut tiles_comp: Vec<Vec<u8>> = Vec::new();

    // Tile 0: stored block (BTYPE = 0, level 0) — 64 KiB incompressible.
    let t0 = deterministic_bytes(65536, 0xA001);
    tiles_comp.push(deflate_raw(&t0, 0));
    tiles_plain.push(t0);

    // Tile 1: tiny fixed-Huffman block (BTYPE = 1 typical on small text).
    let t1 = b"Victory GDeflate v1 fixed Huffman verification tile!".to_vec();
    tiles_comp.push(deflate_raw(&t1, 1));
    tiles_plain.push(t1);

    // Tile 2: RLE-heavy dynamic Huffman (tests long overlapping matches & dist=1).
    let t2 = vec![0x42u8; 65536];
    tiles_comp.push(deflate_raw(&t2, 6));
    tiles_plain.push(t2);

    // Tiles 3..34: 32 x 64 KiB mixed texture/geometry-like tiles at compression levels 1, 6, 9.
    for k in 0..32usize {
        let len = if k == 31 { 17321 } else { 65536 };
        let mut tile = vec![0u8; len];
        for (i, b) in tile.iter_mut().enumerate() {
            *b = match (i / 128 + k) % 4 {
                0 => ((i & 0x1F) as u8).wrapping_mul(3),
                1 => (i.wrapping_mul(13) ^ (i >> 5) ^ k) as u8,
                2 => ((i / 4) as u8).wrapping_add(k as u8),
                _ => ((i ^ (i >> 3) ^ (k * 29)) & 0x7F) as u8,
            };
        }
        let level = match k % 3 {
            0 => 1,
            1 => 6,
            _ => 9,
        };
        tiles_comp.push(deflate_raw(&tile, level));
        tiles_plain.push(tile);
    }

    let mut packed_src = Vec::new();
    let mut descs = Vec::new();
    let mut expected = Vec::new();

    for (plain, comp) in tiles_plain.iter().zip(&tiles_comp) {
        let src_off = packed_src.len() as u64;
        packed_src.extend_from_slice(comp);
        let dst_off = expected.len() as u64;
        expected.extend_from_slice(plain);
        descs.push(DecompressBlockDesc {
            src_offset: src_off,
            src_len: comp.len() as u32,
            dst_len: plain.len() as u32,
            dst_offset: dst_off,
        });
    }

    let (gpu_out, written) = ctx
        .decompress_host_sync(
            CudaDecompressCodec::GDeflateV1RawDeflateTile,
            &packed_src,
            expected.len(),
            &descs,
        )
        .expect("GPU GDeflate v1 multi-tile decompression");

    for (i, (plain, &w)) in tiles_plain.iter().zip(&written).enumerate() {
        assert_eq!(w as usize, plain.len(), "GDeflate tile {i} length mismatch");
    }
    assert_eq!(
        gpu_out, expected,
        "GPU GDeflate v1 tiles must match zlib-rs raw DEFLATE bit-for-bit"
    );
}

#[test]
fn sparse_readback_delta_compaction_and_overflow_fallback() {
    let ctx = require_cuda();

    let piece_bytes = 65536usize;
    let num_pieces = 64usize; // 4 MiB across 64 x 64 KiB pieces
    let total_bytes = piece_bytes * num_pieces;

    let held = deterministic_bytes(total_bytes, 0x5555_AAAA);
    let pieces: Vec<ReadbackPieceDesc> = (0..num_pieces)
        .map(|i| ReadbackPieceDesc {
            now_offset: (i * piece_bytes) as u64,
            held_offset: (i * piece_bytes) as u64,
            dword_count: (piece_bytes / 4) as u32,
            piece_id: i as u32,
        })
        .collect();

    // Case 1: 0% dirty -> 0 deltas.
    let mut out_pairs = vec![ReadbackDeltaPair::default(); 4096];
    let count0 = ctx
        .compact_deltas_host_sync(&held, &held, &pieces, &mut out_pairs)
        .unwrap()
        .expect("zero dirty should not overflow");
    assert_eq!(count0, 0);

    // Case 2: Sparse ~2% dirty including exact warp/chunk boundaries (0, 31, 32, 255, 256, 16383).
    let mut now = held.clone();
    let mut expected = Vec::new();
    for p in 0..num_pieces {
        let dwords = if p % 2 == 0 {
            vec![0usize, 31, 32, 255, 256, 1024, 8191, 16383]
        } else {
            (0..200).map(|k| k * 79 + 13).filter(|&d| d < 16384).collect()
        };
        for dw in dwords {
            let off = p * piece_bytes + dw * 4;
            let orig = u32::from_le_bytes(held[off..off + 4].try_into().unwrap());
            let mutated = orig ^ 0xA5A5_5A5A;
            now[off..off + 4].copy_from_slice(&mutated.to_le_bytes());
            expected.push((p as u32, dw as u32, mutated));
        }
    }

    let mut out_pairs = vec![ReadbackDeltaPair::default(); expected.len() + 256];
    let count = ctx
        .compact_deltas_host_sync(&now, &held, &pieces, &mut out_pairs)
        .unwrap()
        .expect("sparse deltas should fit");
    assert_eq!(count, expected.len());

    let mut got: Vec<(u32, u32, u32)> = out_pairs[..count]
        .iter()
        .map(|p| (p.piece_id, p.dword_offset, p.value))
        .collect();
    got.sort_unstable();
    expected.sort_unstable();
    assert_eq!(got, expected);

    // Case 3: Overflow fallback -> returns Ok(None) cleanly when changes exceed out_pairs capacity.
    let mut tiny = vec![ReadbackDeltaPair::default(); 64];
    let overflow = ctx
        .compact_deltas_host_sync(&now, &held, &pieces, &mut tiny)
        .unwrap();
    assert!(
        overflow.is_none(),
        "delta compaction must return None when changes exceed output buffer"
    );
}
