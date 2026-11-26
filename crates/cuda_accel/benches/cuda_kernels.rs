//! Criterion micro-benchmarks comparing `cuda-accel` SM 8.9 GPU kernels against CPU baselines.
//!
//! Run with:
//! `cargo bench --manifest-path crates/cuda_accel/Cargo.toml`

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use cuda_accel::{
    CudaAccelContext, CudaDecompressCodec, DecompressBlockDesc, DetileDispatchDesc,
    ReadbackDeltaPair, ReadbackPieceDesc, detile_desc_from_layout,
};
use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};

fn deterministic_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut s = seed ^ 0xD1B5_4A32_D192_ED03;
    let mut out = vec![0u8; len];
    for b in &mut out {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *b = (s >> 32) as u8;
    }
    out
}

fn bench_detile_batch(c: &mut Criterion) {
    let Ok(ctx) = CudaAccelContext::new(None) else {
        eprintln!("skipping CUDA detile benchmark (no CUDA device)");
        return;
    };

    // 32 x 512x512 BC7 (all 10 mips, SW_64KB_S) ~= 11.4 MiB tiled -> realistic texture streaming batch.
    let desc = Desc {
        dimension: Dimension::Tex2d,
        width: 512,
        height: 512,
        depth: 1,
        mips: 10,
        element_bytes: 16,
        block: 4,
        samples: 1,
        swizzle: Swizzle::SW_64KB_S,
    };
    let layout = Layout::new(&desc).unwrap();
    let num_textures = 32usize;
    let mut tiled_batch = Vec::new();
    let mut dispatches: Vec<DetileDispatchDesc> = Vec::new();
    let mut dst_len = 0usize;

    for i in 0..num_textures {
        let src_base = tiled_batch.len().next_multiple_of(256);
        tiled_batch.resize(src_base, 0);
        tiled_batch.extend_from_slice(&deterministic_bytes(layout.size as usize, 0x42 + i as u64));
        for mip in 0..desc.mips {
            let m = &layout.mips[mip as usize];
            let row_pitch = ((m.width * desc.element_bytes) as usize).next_multiple_of(256);
            let depth_pitch = row_pitch * m.height as usize;
            let dst_off = dst_len.next_multiple_of(512);
            dst_len = dst_off + depth_pitch;
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

    let mut group = c.benchmark_group("detile_32x_512_bc7_10mips");
    group.sample_size(20);
    group.throughput(Throughput::Bytes(dst_len as u64));

    let mut cpu_out = vec![0u8; dst_len];
    group.bench_function("cpu_layout_detile", |b| {
        b.iter(|| {
            let mut cursor = 0usize;
            for i in 0..num_textures {
                let src_base = i * (layout.size as usize).next_multiple_of(256);
                let tiled = &tiled_batch[src_base..src_base + layout.size as usize];
                for mip in 0..desc.mips {
                    let m = &layout.mips[mip as usize];
                    let row_pitch = ((m.width * desc.element_bytes) as usize).next_multiple_of(256);
                    let depth_pitch = row_pitch * m.height as usize;
                    let dst_off = cursor.next_multiple_of(512);
                    layout
                        .detile(
                            tiled,
                            mip,
                            0,
                            &mut cpu_out[dst_off..dst_off + depth_pitch],
                            row_pitch,
                            depth_pitch,
                        )
                        .unwrap();
                    cursor = dst_off + depth_pitch;
                }
            }
            black_box(cpu_out[0])
        });
    });

    // Warm up pinned ring + device scratch buffers once.
    let _ = ctx.detile_host_sync(&tiled_batch, dst_len, &dispatches).unwrap();

    group.bench_function("cuda_sm89_detile_zero_copy_vram", |b| {
        b.iter(|| {
            ctx.detile_to_vram_sync(black_box(&tiled_batch), dst_len, black_box(&dispatches))
                .unwrap();
        });
    });

    group.bench_function("cuda_sm89_detile_htod_kernel_dtoh", |b| {
        b.iter(|| {
            let out = ctx
                .detile_host_sync(black_box(&tiled_batch), dst_len, black_box(&dispatches))
                .unwrap();
            black_box(out[0])
        });
    });

    group.finish();
}

fn bench_gdeflate_and_lz4(c: &mut Criterion) {
    let Ok(ctx) = CudaAccelContext::new(None) else {
        return;
    };

    // 64 x 64 KiB tiles = 4 MiB asset stream.
    let num_tiles = 64usize;
    let tile_len = 65536usize;
    let total_len = num_tiles * tile_len;
    let mut raw_tiles = Vec::with_capacity(num_tiles);
    for k in 0..num_tiles {
        let mut t = vec![0u8; tile_len];
        for (i, b) in t.iter_mut().enumerate() {
            *b = if (i / 64) % 2 == 0 {
                ((i + k) & 0x1F) as u8
            } else {
                ((i.wrapping_mul(17) ^ (i >> 4) ^ k) & 0x7F) as u8
            };
        }
        raw_tiles.push(t);
    }

    // GDeflate v1 (64 independent raw DEFLATE tiles).
    let mut gd_src = Vec::new();
    let mut gd_blocks = Vec::new();
    let def_cfg = zlib_rs::DeflateConfig {
        level: 6,
        window_bits: -15,
        ..zlib_rs::DeflateConfig::default()
    };
    for (i, tile) in raw_tiles.iter().enumerate() {
        let mut tmp = vec![0u8; tile_len + 8192];
        let (comp, ret) = zlib_rs::compress_slice(&mut tmp, tile, def_cfg);
        assert_eq!(ret, zlib_rs::ReturnCode::Ok);
        let src_off = gd_src.len() as u64;
        gd_src.extend_from_slice(comp);
        gd_blocks.push(DecompressBlockDesc {
            src_offset: src_off,
            src_len: comp.len() as u32,
            dst_len: tile_len as u32,
            dst_offset: (i * tile_len) as u64,
        });
    }

    let mut group = c.benchmark_group("gdeflate_64x64kib_4mib");
    group.sample_size(20);
    group.throughput(Throughput::Bytes(total_len as u64));

    group.bench_function("cpu_zlib_rs_sequential", |b| {
        let mut infl = zlib_rs::Inflate::new(false, 15);
        let mut out = vec![0u8; tile_len];
        b.iter(|| {
            let mut sum = 0usize;
            for blk in &gd_blocks {
                infl.reset(false);
                let src = &gd_src[blk.src_offset as usize..(blk.src_offset as usize + blk.src_len as usize)];
                let _ = infl.decompress(src, &mut out, zlib_rs::InflateFlush::Finish);
                sum += infl.total_out() as usize;
            }
            black_box(sum)
        });
    });

    let _ = ctx
        .decompress_host_sync(
            CudaDecompressCodec::GDeflateV1RawDeflateTile,
            &gd_src,
            total_len,
            &gd_blocks,
        )
        .unwrap();

    group.bench_function("cuda_sm89_gdeflate_zero_copy_vram", |b| {
        b.iter(|| {
            ctx.decompress_to_vram_sync(
                CudaDecompressCodec::GDeflateV1RawDeflateTile,
                black_box(&gd_src),
                total_len,
                black_box(&gd_blocks),
            )
            .unwrap();
        });
    });

    group.bench_function("cuda_sm89_gdeflate_64warps", |b| {
        b.iter(|| {
            let (out, _) = ctx
                .decompress_host_sync(
                    CudaDecompressCodec::GDeflateV1RawDeflateTile,
                    black_box(&gd_src),
                    total_len,
                    black_box(&gd_blocks),
                )
                .unwrap();
            black_box(out[0])
        });
    });
    group.finish();

    // LZ4 raw block batch (64 x 64 KiB = 4 MiB).
    let mut lz4_src = Vec::new();
    let mut lz4_blocks = Vec::new();
    for (i, tile) in raw_tiles.iter().enumerate() {
        let comp = lz4_flex::block::compress(tile);
        let src_off = lz4_src.len() as u64;
        lz4_src.extend_from_slice(&comp);
        lz4_blocks.push(DecompressBlockDesc {
            src_offset: src_off,
            src_len: comp.len() as u32,
            dst_len: tile_len as u32,
            dst_offset: (i * tile_len) as u64,
        });
    }

    let mut lgroup = c.benchmark_group("lz4_batch_64x64kib_4mib");
    lgroup.sample_size(20);
    lgroup.throughput(Throughput::Bytes(total_len as u64));

    lgroup.bench_function("cpu_lz4_flex_sequential", |b| {
        let mut out = vec![0u8; tile_len];
        b.iter(|| {
            let mut sum = 0usize;
            for blk in &lz4_blocks {
                let src = &lz4_src[blk.src_offset as usize..(blk.src_offset as usize + blk.src_len as usize)];
                sum += lz4_flex::block::decompress_into(src, &mut out).unwrap();
            }
            black_box(sum)
        });
    });

    let _ = ctx
        .decompress_host_sync(CudaDecompressCodec::Lz4RawBlock, &lz4_src, total_len, &lz4_blocks)
        .unwrap();

    lgroup.bench_function("cuda_sm89_lz4_zero_copy_vram", |b| {
        b.iter(|| {
            ctx.decompress_to_vram_sync(
                CudaDecompressCodec::Lz4RawBlock,
                black_box(&lz4_src),
                total_len,
                black_box(&lz4_blocks),
            )
            .unwrap();
        });
    });

    lgroup.bench_function("cuda_sm89_lz4_batch", |b| {
        b.iter(|| {
            let (out, _) = ctx
                .decompress_host_sync(
                    CudaDecompressCodec::Lz4RawBlock,
                    black_box(&lz4_src),
                    total_len,
                    black_box(&lz4_blocks),
                )
                .unwrap();
            black_box(out[0])
        });
    });
    lgroup.finish();
}

fn bench_sparse_readback_deltas(c: &mut Criterion) {
    let Ok(ctx) = CudaAccelContext::new(None) else {
        return;
    };

    // 256 x 64 KiB = 16 MiB relay footprint with 2% sparse changes.
    let piece_bytes = 65536usize;
    let num_pieces = 256usize;
    let total_bytes = piece_bytes * num_pieces;
    let held = deterministic_bytes(total_bytes, 0x9988);
    let mut now = held.clone();
    for p in 0..num_pieces {
        // ~2% changed dwords (~328 dwords per piece).
        for dw in (0..16384).step_by(50) {
            let off = p * piece_bytes + dw * 4;
            now[off] ^= 0xFF;
        }
    }
    let pieces: Vec<ReadbackPieceDesc> = (0..num_pieces)
        .map(|i| ReadbackPieceDesc {
            now_offset: (i * piece_bytes) as u64,
            held_offset: (i * piece_bytes) as u64,
            dword_count: (piece_bytes / 4) as u32,
            piece_id: i as u32,
        })
        .collect();

    let mut out_pairs = vec![ReadbackDeltaPair::default(); 200_000];
    let mut group = c.benchmark_group("readback_deltas_16mib_2pct_dirty");
    group.sample_size(20);
    group.throughput(Throughput::Bytes(total_bytes as u64));

    group.bench_function("cpu_dword_scan", |b| {
        b.iter(|| {
            let mut count = 0usize;
            for p in &pieces {
                let n = &now[p.now_offset as usize..p.now_offset as usize + piece_bytes];
                let h = &held[p.held_offset as usize..p.held_offset as usize + piece_bytes];
                for dw in 0..(piece_bytes / 4) {
                    let o = dw * 4;
                    if n[o..o + 4] != h[o..o + 4] {
                        count += 1;
                    }
                }
            }
            black_box(count)
        });
    });

    let dev_now = ctx.alloc_and_upload_vram(&now).unwrap();
    let dev_held = ctx.alloc_and_upload_vram(&held).unwrap();

    // Production path: `now` and `held` are already resident in DX12-shared VRAM (zero HtoD; only dense deltas cross PCIe).
    group.bench_function("cuda_sm89_vram_resident_warp_ballot", |b| {
        b.iter(|| {
            let count = unsafe {
                ctx.compact_deltas_vram_sync(
                    dev_now,
                    dev_held,
                    black_box(&pieces),
                    &mut out_pairs,
                )
            }
            .unwrap();
            black_box(count)
        });
    });

    unsafe {
        ctx.free_device_buffer(dev_held);
        ctx.free_device_buffer(dev_now);
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_detile_batch,
    bench_gdeflate_and_lz4,
    bench_sparse_readback_deltas
);
criterion_main!(benches);
