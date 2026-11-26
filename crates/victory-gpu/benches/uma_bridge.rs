//! Criterion micro-benchmarks for Subsystem A (`uma_bridge`).
//!
//! Benchmarks:
//! 1. ReBAR slab sub-allocation + retirement + coalescing reclamation churn.
//! 2. Non-ReBAR `StagingCopyRing` allocation + deferred copy scheduling + flush + reclaim.
//! 3. `IntervalAliasingTracker` overlap queries at 1,000 and 10,000 placed resources.
//! 4. `CoalescingBarrierBatcher` processing 1,000 mixed Xbox frame barriers.

use std::sync::Arc;

use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

use victory_gpu::graphics::contracts::{
    AliasingBarrierDesc, AliasingTracker, BarrierBatcher, GpuFenceToken, HeapId, HeapRange,
    ResourceId, ResourceStateFlags, SubresourceIndex, TransitionBarrierDesc, UavBarrierDesc,
    UploadAllocator,
};
use victory_gpu::graphics::uma_bridge::{
    CoalescingBarrierBatcher, IntervalAliasingTracker, MockCopyQueueBackend, MockHeapBackend,
    RebarAllocatorConfig, RebarUploadAllocator, RecordingBarrierSink,
};

fn bench_rebar_alloc_retire_churn(c: &mut Criterion) {
    let heap = MockHeapBackend::rtx_rebar(64 * 1024 * 1024);
    let copy = MockCopyQueueBackend::new();
    let alloc = RebarUploadAllocator::new(
        heap,
        copy,
        RebarAllocatorConfig {
            rebar_slab_size: 64 * 1024 * 1024,
            staging_ring_capacity: 4 * 1024 * 1024,
            min_alignment: 1,
        },
    )
    .expect("allocator init");

    let mut fence = 1u64;
    c.bench_function("uma_bridge/rebar_alloc_retire_churn_256", |b| {
        b.iter(|| {
            let mut slices = Vec::with_capacity(256);
            for i in 0..256u64 {
                let size = 256 + (i % 8) * 128;
                let align = 1u64 << (4 + (i % 5));
                let slice = alloc.allocate(black_box(size), black_box(align)).expect("alloc");
                slices.push(slice);
            }
            for slice in slices {
                alloc.retire_slice(
                    slice,
                    GpuFenceToken {
                        queue_id: 0,
                        value: fence,
                    },
                );
            }
            let reclaimed = alloc.reclaim_completed(fence);
            fence += 1;
            black_box(reclaimed);
        });
    });
}

fn bench_staging_ring_copy_batch(c: &mut Criterion) {
    let heap = Arc::new(MockHeapBackend::discrete_no_rebar());
    let copy = MockCopyQueueBackend::with_heap(Arc::clone(&heap));
    let dst = heap
        .create_device_local_buffer(4 * 1024 * 1024)
        .expect("dst");
    let alloc = RebarUploadAllocator::new(
        Arc::clone(&heap),
        copy,
        RebarAllocatorConfig {
            rebar_slab_size: 4 * 1024 * 1024,
            staging_ring_capacity: 4 * 1024 * 1024,
            min_alignment: 256,
        },
    )
    .expect("allocator init");

    let payload = [0xCDu8; 512];
    c.bench_function("uma_bridge/staging_ring_batch_128x512b", |b| {
        b.iter(|| {
            for i in 0..128u64 {
                let mut slice = alloc.allocate(512, 256).expect("ring alloc");
                slice.write_sequential(0, black_box(&payload)).expect("write");
                alloc
                    .schedule_copy_to_vram(slice, dst, i * 512)
                    .expect("sched");
            }
            let token = alloc.flush_copies().expect("flush").expect("token");
            let reclaimed = alloc.reclaim_completed(token.value);
            black_box(reclaimed);
        });
    });
}

fn bench_aliasing_tracker_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("uma_bridge/aliasing_interval_query");
    for &count in &[1_000u64, 10_000u64] {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &n| {
            b.iter_batched(
                || {
                    let mut tracker = IntervalAliasingTracker::new();
                    let heap = HeapId(1);
                    // Register `n` non-overlapping base resources + `n/4` aliased overlays.
                    for i in 0..n {
                        let res = ResourceId(i + 1);
                        tracker
                            .register_placed(
                                res,
                                HeapRange {
                                    heap_id: heap,
                                    offset: i * 4096,
                                    size: 4096,
                                },
                            )
                            .expect("register base");
                        let _ = tracker.on_resource_access(res);
                    }
                    let overlay_base = n + 1;
                    for i in 0..(n / 4) {
                        let res = ResourceId(overlay_base + i);
                        tracker
                            .register_placed(
                                res,
                                HeapRange {
                                    heap_id: heap,
                                    offset: i * 4 * 4096 + 2048,
                                    size: 8192,
                                },
                            )
                            .expect("register overlay");
                    }
                    (tracker, overlay_base, n / 4)
                },
                |(mut tracker, overlay_base, overlays)| {
                    let mut emitted = 0usize;
                    for i in 0..overlays {
                        let res = ResourceId(overlay_base + i);
                        emitted += tracker.on_resource_access(black_box(res)).len();
                    }
                    black_box(emitted);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_barrier_batcher_1k_frame(c: &mut Criterion) {
    let mut batcher = CoalescingBarrierBatcher::with_capacity(1024);
    let mut sink = RecordingBarrierSink::new();

    c.bench_function("uma_bridge/barrier_batcher_1k_frame", |b| {
        b.iter(|| {
            batcher.reset();
            sink.calls.clear();
            for i in 0..250u64 {
                let res = ResourceId(i + 1);
                // 1) Foldable transition pair (with Xbox upper bits).
                batcher.push_xbox_transition(
                    res,
                    SubresourceIndex::ALL,
                    ResourceStateFlags::COPY_DEST.0 | 0x0400_0000,
                    ResourceStateFlags::PIXEL_SHADER_RESOURCE.0,
                );
                batcher.push_transition(TransitionBarrierDesc {
                    resource: res,
                    subresource: SubresourceIndex::ALL,
                    before: ResourceStateFlags::PIXEL_SHADER_RESOURCE,
                    after: ResourceStateFlags::NON_PIXEL_SHADER_RESOURCE,
                });
                // 2) Redundant read-to-read transition (stripped).
                batcher.push_transition(TransitionBarrierDesc {
                    resource: res,
                    subresource: SubresourceIndex::ALL,
                    before: ResourceStateFlags::GENERIC_READ,
                    after: ResourceStateFlags::PIXEL_SHADER_RESOURCE,
                });
                // 3) Periodic UAV / aliasing wave.
                if i.is_multiple_of(16) {
                    batcher.push_uav(UavBarrierDesc {
                        resource: Some(res),
                    });
                    batcher.push_uav(UavBarrierDesc { resource: None });
                } else if i.is_multiple_of(31) {
                    batcher.push_aliasing(AliasingBarrierDesc {
                        before: Some(res),
                        after: Some(ResourceId(i + 2)),
                    });
                } else {
                    batcher.push_transition(TransitionBarrierDesc {
                        resource: res,
                        subresource: SubresourceIndex::ALL,
                        before: ResourceStateFlags::RENDER_TARGET,
                        after: ResourceStateFlags::RENDER_TARGET,
                    });
                }
            }
            let flushed = batcher.flush(&mut sink);
            black_box(flushed);
        });
    });
}

criterion_group!(
    benches,
    bench_rebar_alloc_retire_churn,
    bench_staging_ring_copy_batch,
    bench_aliasing_tracker_queries,
    bench_barrier_batcher_1k_frame
);
criterion_main!(benches);
