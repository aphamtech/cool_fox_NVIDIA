//! Criterion micro-benchmarks for Subsystem B (`velocity_io`).
//! Owned by `@velocity-io`.
//!
//! Benchmarks:
//! 1. Lock-free `VelocityQueue` enqueue & batch `submit()` throughput (zero render-thread stalls).
//! 2. Multi-threaded `PinnedUploadRing` lease + write + fence-gated reclaim throughput.
//! 3. Async I/O ring batched read throughput (`InMemoryAsyncIo` + `PinnedUploadRing`).
//! 4. `DecompressionBackend` dispatch throughput via `GpuDecompressQueue` for Zlib,
//!    LZ4 (block + frame), and GDeflate v1 at 64 KiB and 4 MiB payloads.

use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

use victory_gpu::graphics::contracts::{
    AsyncIoBackend, AsyncReadCompletion, AsyncReadOp, ComputeQueueBackend, DStorageDestination,
    DStorageRequestDesc, DStorageSource, DecompressionCodec, DecompressionJob, GpuFenceToken,
    VelocityIoError,
};
use victory_gpu::graphics::velocity_io::dstorage_thunk::{
    InMemoryAsyncIo, PinnedUploadRing, VelocityFactory, VelocityQueueConfig,
};
use victory_gpu::graphics::velocity_io::gpu_decompress_queue::{
    GpuDecompressQueue, MockComputeQueueBackend, encode_gdeflate_v1,
};

fn make_benchmark_payload(len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let byte = if (i % 64) < 28 {
            ((i / 64) & 0xFF) as u8
        } else {
            ((i * 17 + (i >> 6)) & 0xFF) as u8
        };
        out.push(byte);
    }
    out
}

fn zlib_compress(input: &[u8]) -> Vec<u8> {
    let config = zlib_rs::DeflateConfig {
        level: 6,
        window_bits: 15,
        ..zlib_rs::DeflateConfig::default()
    };
    let mut out = vec![0u8; input.len() + (input.len() / 8) + 64];
    let (compressed, ret) = zlib_rs::compress_slice(&mut out, input, config);
    assert_eq!(ret, zlib_rs::ReturnCode::Ok);
    compressed.to_vec()
}

fn bench_lockfree_enqueue_submit(c: &mut Criterion) {
    let mut group = c.benchmark_group("velocity_io/lockfree_enqueue_submit");

    let io = Arc::new(InMemoryAsyncIo::new());
    let file_id = io.register_open_file("bench.pak", vec![0xAB; 64 * 1024]);
    let compute = Arc::new(MockComputeQueueBackend::for_bench(1));
    let decompress = Arc::new(GpuDecompressQueue::new(compute));
    let factory = VelocityFactory::new(io, decompress);

    for &batch_size in &[64usize, 512usize] {
        let queue = factory
            .create_queue(VelocityQueueConfig {
                capacity: batch_size * 2,
                staging_ring_bytes: 2 * 1024 * 1024,
                decompress_workers: 2,
                name: format!("bench-eq-{batch_size}"),
                ..VelocityQueueConfig::default()
            })
            .expect("create bench queue");

        let mut sink = vec![0u8; 64];
        let dst_addr = sink.as_mut_ptr() as usize;

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &count| {
                b.iter(|| {
                    for i in 0..count {
                        queue
                            .enqueue_request(DStorageRequestDesc {
                                codec: DecompressionCodec::Uncompressed,
                                secondary_codec: None,
                                intermediate_size: 0,
                                source: DStorageSource::File {
                                    file: file_id,
                                    offset: ((i * 64) % 4096) as u64,
                                    size: 64,
                                },
                                destination: DStorageDestination::Memory {
                                    address: dst_addr,
                                    size: 64,
                                },
                                cancellation_tag: i as u64,
                            })
                            .expect("enqueue");
                    }
                    let sealed = queue.submit().expect("submit");
                    black_box(sealed);
                    assert!(queue.wait_idle(Duration::from_secs(5)));
                });
            },
        );
        queue.close();
    }

    group.finish();
}

fn bench_pinned_upload_ring_multithreaded(c: &mut Criterion) {
    let mut group = c.benchmark_group("velocity_io/pinned_ring_mt");
    const THREADS: usize = 4;
    const OPS_PER_THREAD: usize = 64;
    const CHUNK_BYTES: usize = 64 * 1024;
    const TOTAL_BYTES: u64 = (THREADS * OPS_PER_THREAD * CHUNK_BYTES) as u64;

    group.throughput(Throughput::Bytes(TOTAL_BYTES));
    group.bench_function("4t_x_64_x_64KiB", |b| {
        let ring = Arc::new(
            PinnedUploadRing::new_standalone(8 * 1024 * 1024).expect("8 MiB pinned ring"),
        );
        let fence = Arc::new(AtomicU64::new(1));
        let completed = Arc::new(AtomicU64::new(0));

        b.iter(|| {
            thread::scope(|s| {
                for tid in 0..THREADS {
                    let ring = Arc::clone(&ring);
                    let fence = Arc::clone(&fence);
                    let completed = Arc::clone(&completed);
                    s.spawn(move || {
                        for _ in 0..OPS_PER_THREAD {
                            let mut lease = loop {
                                match ring.try_lease(CHUNK_BYTES) {
                                    Ok(l) => break l,
                                    Err(VelocityIoError::StagingRingFull { .. }) => {
                                        ring.reclaim_completed(completed.load(Ordering::Acquire));
                                        thread::yield_now();
                                    }
                                    Err(e) => panic!("lease failed: {e}"),
                                }
                            };
                            lease.as_mut_slice()[0] = tid as u8;
                            let tok = fence.fetch_add(1, Ordering::AcqRel);
                            ring.retire_lease(
                                lease,
                                GpuFenceToken {
                                    queue_id: 1,
                                    value: tok,
                                },
                            );
                            completed.fetch_max(tok, Ordering::Release);
                        }
                    });
                }
            });
            ring.reclaim_completed(fence.load(Ordering::Acquire));
        });
    });

    group.finish();
}

fn bench_async_io_ring_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("velocity_io/async_io_ring");

    let io = InMemoryAsyncIo::new();
    let total_file = make_benchmark_payload(4 * 1024 * 1024);
    let file_id = io.register_open_file("stream.pak", total_file);
    let ring = PinnedUploadRing::new_standalone(8 * 1024 * 1024).expect("8 MiB ring");

    for &(label, chunk_size, chunks) in &[
        ("64KiB_x_32", 64 * 1024usize, 32usize),
        ("256KiB_x_16", 256 * 1024usize, 16usize),
    ] {
        group.throughput(Throughput::Bytes((chunk_size * chunks) as u64));
        group.bench_function(label, |b| {
            let mut completions: Vec<AsyncReadCompletion> = Vec::with_capacity(chunks);
            let mut fence_val = 1u64;
            b.iter(|| {
                let mut ops = Vec::with_capacity(chunks);
                let mut leases = Vec::with_capacity(chunks);
                for i in 0..chunks {
                    let lease = ring.try_lease(chunk_size).expect("lease");
                    ops.push(AsyncReadOp {
                        op_id: i as u64 + 1,
                        file: file_id,
                        file_offset: (i * chunk_size) as u64,
                        len: chunk_size,
                        dst_ptr: lease.raw_ptr(),
                    });
                    leases.push(lease);
                }
                let submitted = io.submit_read_batch(&ops).expect("submit");
                black_box(submitted);
                completions.clear();
                while completions.len() < chunks {
                    io.poll_completions(&mut completions, Duration::from_millis(10))
                        .expect("poll");
                }
                for lease in leases {
                    ring.retire_lease(
                        lease,
                        GpuFenceToken {
                            queue_id: 1,
                            value: fence_val,
                        },
                    );
                }
                ring.reclaim_completed(fence_val);
                fence_val += 1;
            });
        });
    }

    group.finish();
}

fn bench_decompression_backends(c: &mut Criterion) {
    let mut group = c.benchmark_group("velocity_io/decompression_backends");
    let compute: Arc<dyn ComputeQueueBackend> = Arc::new(MockComputeQueueBackend::for_bench(1));
    let queue = GpuDecompressQueue::new(compute);

    for &(label, size) in &[("64KiB", 64 * 1024usize), ("4MiB", 4 * 1024 * 1024usize)] {
        let payload = make_benchmark_payload(size);
        let zlib_data = zlib_compress(&payload);
        let lz4_block = lz4_flex::block::compress(&payload);
        let mut lz4_frame_enc = lz4_flex::frame::FrameEncoder::new(Vec::new());
        lz4_frame_enc.write_all(&payload).expect("lz4 frame write");
        let lz4_frame = lz4_frame_enc.finish().expect("lz4 frame finish");
        let gdeflate_data = encode_gdeflate_v1(&payload).expect("encode gdeflate");

        group.throughput(Throughput::Bytes(size as u64));

        for (codec_name, codec, compressed) in [
            ("zlib", DecompressionCodec::Zlib, zlib_data.as_slice()),
            ("lz4_block", DecompressionCodec::Lz4, lz4_block.as_slice()),
            ("lz4_frame", DecompressionCodec::Lz4, lz4_frame.as_slice()),
            (
                "gdeflate_v1",
                DecompressionCodec::GDeflate,
                gdeflate_data.as_slice(),
            ),
        ] {
            group.bench_with_input(
                BenchmarkId::new(codec_name, label),
                compressed,
                |b, src| {
                    b.iter(|| {
                        let out = queue
                            .dispatch(&DecompressionJob {
                                job_id: 1,
                                codec,
                                src_bytes: black_box(src),
                                src_gpu_va: None,
                                src_gpu_resource: None,
                                expected_output_len: size,
                                dst_gpu_resource: None,
                                wait_copy_fence: None,
                            })
                            .expect("decompress");
                        black_box(out.bytes_written);
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_lockfree_enqueue_submit,
    bench_pinned_upload_ring_multithreaded,
    bench_async_io_ring_throughput,
    bench_decompression_backends
);
criterion_main!(benches);
