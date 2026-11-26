//! Comprehensive acceptance test suite for Subsystem A (`uma_bridge`).
//!
//! Covers:
//! - ReBAR sub-allocator alignment, coalescing, sequential WC writes, and fence retirement.
//! - Non-ReBAR `StagingCopyRing` fallback, async Copy Queue batching, byte-exact copies,
//!   fence-gated reclamation, wrap-around, and non-blocking `RingExhausted`.
//! - ReBAR budget spillover into `StagingCopyRing` and recovery after reclamation.
//! - Multi-threaded concurrent allocation/retirement disjointness stress test.
//! - `IntervalAliasingTracker` disjoint, partial-overlap, multi-owner fan-in,
//!   fragment deduplication, unregister (Rule 2 unmapped VA guard), and cross-heap isolation.
//! - `CoalescingBarrierBatcher` single-submit coalescing, Xbox bit stripping, no-op &
//!   read-to-read stripping, folding, write-hazard preservation, UAV dedup & ordering.
//! - Windows D3D12/WARP hardware integration tests (byte-exact Copy Queue roundtrip
//!   and `D3D12CommandListBarrierSink` execution on a live D3D12 command list).

use std::sync::Arc;
use std::thread;

use victory_gpu::graphics::contracts::{
    AliasingBarrierDesc, AliasingTracker, BarrierBatcher, GpuFenceToken, HeapId, HeapRange,
    MemoryTier, ResourceId, ResourceStateFlags, SubresourceIndex, TransitionBarrierDesc,
    TranslatedBarrier, UavBarrierDesc, UmaBridgeError, UploadAllocator,
};
use victory_gpu::graphics::uma_bridge::{
    CoalescingBarrierBatcher, IntervalAliasingTracker, MockCopyQueueBackend, MockHeapBackend,
    RebarAllocatorConfig, RebarUploadAllocator, RecordingBarrierSink,
};

// ---------------------------------------------------------------------------
// 1. ReBAR Sub-Allocator Acceptance Tests
// ---------------------------------------------------------------------------

#[test]
fn rebar_suballoc_alignment_and_sequential_writes() {
    let heap = Arc::new(MockHeapBackend::rtx_rebar(64 * 1024));
    let copy = MockCopyQueueBackend::with_heap(Arc::clone(&heap));
    let alloc = RebarUploadAllocator::new(
        Arc::clone(&heap),
        copy,
        RebarAllocatorConfig {
            rebar_slab_size: 64 * 1024,
            staging_ring_capacity: 16 * 1024,
            min_alignment: 1,
        },
    )
    .expect("allocator init");

    // Reject invalid layouts cleanly.
    assert!(matches!(
        alloc.allocate(0, 256),
        Err(UmaBridgeError::InvalidLayout { size: 0, .. })
    ));
    assert!(matches!(
        alloc.allocate(128, 0),
        Err(UmaBridgeError::InvalidLayout { alignment: 0, .. })
    ));
    assert!(matches!(
        alloc.allocate(128, 48),
        Err(UmaBridgeError::InvalidLayout { alignment: 48, .. })
    ));

    // Verify power-of-two alignments (16, 64, 256, 4096) and sequential writes.
    let alignments = [16u64, 64, 256, 4096];
    let mut slices = Vec::new();
    for (idx, &align) in alignments.iter().enumerate() {
        let mut slice = alloc.allocate(37 + idx as u64 * 13, align).expect("alloc");
        assert_eq!(slice.tier, MemoryTier::RebarWriteCombine);
        assert!(slice.heap_offset.is_multiple_of(align));
        assert!(slice.gpu_va.0.is_multiple_of(align));

        let payload: Vec<u8> = (0..slice.size as usize)
            .map(|b| ((idx * 31 + b * 7) & 0xff) as u8)
            .collect();
        slice.write_sequential(0, &payload).expect("write");

        let readback = heap
            .read_bytes(slice.resource_id, slice.heap_offset, payload.len())
            .expect("readback");
        assert_eq!(readback, payload);

        // Out-of-bounds write must fail without memory corruption.
        assert!(matches!(
            slice.write_sequential(payload.len(), &[0xAA]),
            Err(UmaBridgeError::InvalidLayout { .. })
        ));
        slices.push(slice);
    }

    // Retire all slices and verify full slab coalescing (including alignment padding).
    for slice in slices {
        alloc.retire_slice(slice, GpuFenceToken { queue_id: 0, value: 10 });
    }
    assert_eq!(alloc.reclaim_completed(9), 0);
    assert_eq!(alloc.reclaim_completed(10), alignments.len());
    assert_eq!(alloc.rebar_free_bytes(), 64 * 1024);
    assert_eq!(alloc.rebar_free_fragment_count(), 1);
}

#[test]
fn rebar_budget_spillover_to_staging_ring_and_recovery() {
    let heap = Arc::new(MockHeapBackend::rtx_rebar(4096));
    let copy = MockCopyQueueBackend::with_heap(Arc::clone(&heap));
    let alloc = RebarUploadAllocator::new(
        Arc::clone(&heap),
        copy,
        RebarAllocatorConfig {
            rebar_slab_size: 4096,
            staging_ring_capacity: 8192,
            min_alignment: 1,
        },
    )
    .expect("allocator init");

    // Exhaust the entire 4096 B ReBAR budget.
    let r1 = alloc.allocate(2048, 256).expect("rebar 1");
    let r2 = alloc.allocate(2048, 256).expect("rebar 2");
    assert_eq!(r1.tier, MemoryTier::RebarWriteCombine);
    assert_eq!(r2.tier, MemoryTier::RebarWriteCombine);

    // Next allocation must seamlessly spill over to StagingCopyRing.
    let mut spill = alloc.allocate(1024, 256).expect("spillover");
    assert_eq!(spill.tier, MemoryTier::StagingCopyRing);
    let data = vec![0x5Au8; 1024];
    spill.write_sequential(0, &data).expect("write spill");

    let dst = heap.create_device_local_buffer(4096).expect("dst");
    alloc
        .schedule_copy_to_vram(spill, dst, 512)
        .expect("schedule");
    let token = alloc.flush_copies().expect("flush").expect("fence token");
    assert_eq!(
        heap.read_bytes(dst, 512, 1024).expect("dst bytes"),
        data
    );

    // Reclaim the spillover copy fence and retire ReBAR slices.
    alloc.copy_queue_backend().complete_up_to(token.value);
    assert_eq!(alloc.reclaim_from_copy_queue(), 1);
    assert_eq!(alloc.staging_ring_used_bytes(), 0);

    alloc.retire_slice(r1, GpuFenceToken { queue_id: 0, value: 5 });
    alloc.retire_slice(r2, GpuFenceToken { queue_id: 0, value: 5 });
    assert_eq!(alloc.reclaim_completed(5), 2);

    // Subsequent allocation returns to RebarWriteCombine hot path.
    let r3 = alloc.allocate(4096, 256).expect("rebar recovered");
    assert_eq!(r3.tier, MemoryTier::RebarWriteCombine);
    assert_eq!(r3.heap_offset, 0);
}

// ---------------------------------------------------------------------------
// 2. Non-ReBAR StagingCopyRing Fallback & Async Copy Queue Acceptance Tests
// ---------------------------------------------------------------------------

#[test]
fn non_rebar_ring_fallback_async_copy_fence_reclamation_and_exhaustion() {
    let heap = Arc::new(MockHeapBackend::discrete_no_rebar());
    let copy = MockCopyQueueBackend::with_heap(Arc::clone(&heap));
    let alloc = RebarUploadAllocator::new(
        Arc::clone(&heap),
        copy,
        RebarAllocatorConfig {
            rebar_slab_size: 4096,
            staging_ring_capacity: 2048,
            min_alignment: 1,
        },
    )
    .expect("allocator init");

    // No ReBAR slabs should have been created on a discrete non-ReBAR adapter.
    let (rebar_slabs, staging_slabs, _, _) = heap.slab_counts();
    assert_eq!(rebar_slabs, 0);
    assert_eq!(staging_slabs, 1);

    let dst = heap.create_device_local_buffer(4096).expect("dst");

    // Allocate two 1024 B slices filling the 2048 B ring completely.
    let mut s1 = alloc.allocate(1024, 256).expect("s1");
    let mut s2 = alloc.allocate(1024, 256).expect("s2");
    assert_eq!(s1.tier, MemoryTier::StagingCopyRing);
    assert_eq!(s2.tier, MemoryTier::StagingCopyRing);
    assert_eq!(s1.heap_offset, 0);
    assert_eq!(s2.heap_offset, 1024);

    let p1: Vec<u8> = (0..1024).map(|i| (i & 0xff) as u8).collect();
    let p2: Vec<u8> = (0..1024).map(|i| ((i * 3 + 17) & 0xff) as u8).collect();
    s1.write_sequential(0, &p1).expect("w1");
    s2.write_sequential(0, &p2).expect("w2");

    // Ring is full: must fail immediately with RingExhausted (never block).
    assert!(matches!(
        alloc.allocate(64, 16),
        Err(UmaBridgeError::RingExhausted {
            requested: 64,
            available: 0
        })
    ));

    // Schedule `s2` BEFORE `s1` in separate batches to test out-of-order fence completion.
    alloc.schedule_copy_to_vram(s2, dst, 1024).expect("sched s2");
    let tok_s2 = alloc.flush_copies().expect("flush s2").expect("tok s2");

    alloc.schedule_copy_to_vram(s1, dst, 0).expect("sched s1");
    let tok_s1 = alloc.flush_copies().expect("flush s1").expect("tok s1");
    assert!(tok_s1.value > tok_s2.value);

    // Flushing with nothing queued returns Ok(None).
    assert_eq!(alloc.flush_copies().expect("empty flush"), None);

    // Byte-exact copy verification in destination VRAM buffer.
    assert_eq!(heap.read_bytes(dst, 0, 1024).expect("dst 0"), p1);
    assert_eq!(heap.read_bytes(dst, 1024, 1024).expect("dst 1024"), p2);

    // Completing only `tok_s2` MUST NOT advance ring tail past still-in-flight `s1`!
    assert_eq!(alloc.reclaim_completed(tok_s2.value), 0);
    assert_eq!(alloc.staging_ring_used_bytes(), 2048);
    assert!(matches!(
        alloc.allocate(64, 16),
        Err(UmaBridgeError::RingExhausted { .. })
    ));

    // Completing `tok_s1` reclaims BOTH `s1` and `s2` contiguously!
    assert_eq!(alloc.reclaim_completed(tok_s1.value), 2);
    assert_eq!(alloc.staging_ring_used_bytes(), 0);

    // New allocation succeeds immediately at offset 0.
    let s3 = alloc.allocate(512, 256).expect("s3 after reclaim");
    assert_eq!(s3.heap_offset, 0);
}

#[test]
fn concurrent_multithreaded_alloc_retire_disjoint_ranges() {
    let heap = Arc::new(MockHeapBackend::discrete_no_rebar());
    let copy = MockCopyQueueBackend::with_heap(Arc::clone(&heap));
    let alloc = Arc::new(
        RebarUploadAllocator::new(
            Arc::clone(&heap),
            copy,
            RebarAllocatorConfig {
                rebar_slab_size: 4096,
                staging_ring_capacity: 256 * 1024,
                min_alignment: 1,
            },
        )
        .expect("init"),
    );

    let threads: Vec<_> = (0..4)
        .map(|tid| {
            let alloc = Arc::clone(&alloc);
            thread::spawn(move || {
                let mut local = Vec::new();
                for iter in 0..64 {
                    let mut slice = alloc.allocate(256, 64).expect("ring alloc");
                    let marker = ((tid * 64 + iter) & 0xff) as u8;
                    slice.write_sequential(0, &[marker; 256]).expect("write");
                    local.push((slice, marker));
                }
                local
            })
        })
        .collect();

    let mut all_ranges = Vec::new();
    for handle in threads {
        for (slice, marker) in handle.join().expect("thread join") {
            let bytes = heap
                .read_bytes(slice.resource_id, slice.heap_offset, 256)
                .expect("read");
            assert!(bytes.iter().all(|&b| b == marker));
            all_ranges.push((slice.heap_offset, slice.heap_offset + slice.size));
            alloc.retire_slice(slice, GpuFenceToken { queue_id: 1, value: 1 });
        }
    }

    // Verify all 256 concurrently live slices occupied strictly disjoint ring offsets.
    all_ranges.sort_unstable();
    for pair in all_ranges.windows(2) {
        assert!(
            pair[0].1 <= pair[1].0,
            "overlapping ring slices detected: {:?} vs {:?}",
            pair[0],
            pair[1]
        );
    }

    assert_eq!(alloc.reclaim_completed(1), 256);
    assert_eq!(alloc.staging_ring_used_bytes(), 0);
}

// ---------------------------------------------------------------------------
// 3. IntervalAliasingTracker Acceptance Tests
// ---------------------------------------------------------------------------

#[test]
fn aliasing_disjoint_partial_overlap_fan_in_unregister_and_cross_heap() {
    let mut tracker = IntervalAliasingTracker::new();
    let h1 = HeapId(1);
    let h2 = HeapId(2);

    let a = ResourceId(10);
    let b = ResourceId(20);
    let c = ResourceId(30);
    let d = ResourceId(40);
    let other_heap = ResourceId(50);

    tracker
        .register_placed(
            a,
            HeapRange {
                heap_id: h1,
                offset: 0,
                size: 1024,
            },
        )
        .unwrap();
    tracker
        .register_placed(
            b,
            HeapRange {
                heap_id: h1,
                offset: 1024,
                size: 1024,
            },
        )
        .unwrap();
    tracker
        .register_placed(
            c,
            HeapRange {
                heap_id: h1,
                offset: 512,
                size: 1024,
            },
        )
        .unwrap();
    tracker
        .register_placed(
            d,
            HeapRange {
                heap_id: h1,
                offset: 512,
                size: 1024,
            },
        )
        .unwrap();
    tracker
        .register_placed(
            other_heap,
            HeapRange {
                heap_id: h2,
                offset: 0,
                size: 2048,
            },
        )
        .unwrap();

    // Disjoint accesses emit 0 barriers.
    assert!(tracker.on_resource_access(a).is_empty());
    assert!(tracker.on_resource_access(b).is_empty());

    // Repeated access by current active owner emits 0 barriers.
    assert!(tracker.on_resource_access(a).is_empty());
    assert!(tracker.on_resource_access(b).is_empty());

    // Same offsets on a different heap emit 0 barriers (cross-heap isolation).
    assert!(tracker.on_resource_access(other_heap).is_empty());
    assert_eq!(tracker.active_owner_at(h2, 100), Some(other_heap));

    // Multi-owner fan-in: C [512, 1536) overlaps A [0, 1024) and B [1024, 2048).
    let fan_in = tracker.on_resource_access(c);
    assert_eq!(
        fan_in.as_slice(),
        &[
            AliasingBarrierDesc {
                before: Some(a),
                after: Some(c),
            },
            AliasingBarrierDesc {
                before: Some(b),
                after: Some(c),
            },
        ]
    );
    assert_eq!(tracker.active_owner_at(h1, 0), Some(a));
    assert_eq!(tracker.active_owner_at(h1, 511), Some(a));
    assert_eq!(tracker.active_owner_at(h1, 512), Some(c));
    assert_eq!(tracker.active_owner_at(h1, 1535), Some(c));
    assert_eq!(tracker.active_owner_at(h1, 1536), Some(b));

    // Unregistering C clears its ownership segments so D never receives a
    // dangling ResourceId (Rule 2 unmapped VA protection).
    tracker.unregister_placed(c);
    assert_eq!(tracker.active_owner_at(h1, 700), None);
    assert!(tracker.on_resource_access(d).is_empty());
    assert_eq!(tracker.active_owner_at(h1, 700), Some(d));

    // Re-accessing A [0, 1024) only evicts D on [512, 1024) (not A's own [0, 512)).
    let back_to_a = tracker.on_resource_access(a);
    assert_eq!(
        back_to_a.as_slice(),
        &[AliasingBarrierDesc {
            before: Some(d),
            after: Some(a),
        }]
    );

    // Clearing active ownership on h1 does not affect h2.
    tracker.clear_active_ownership(h1);
    assert_eq!(tracker.active_owner_at(h1, 0), None);
    assert_eq!(tracker.active_owner_at(h2, 100), Some(other_heap));
}

// ---------------------------------------------------------------------------
// 4. CoalescingBarrierBatcher Acceptance Tests
// ---------------------------------------------------------------------------

#[test]
fn barrier_batcher_coalescing_folding_write_hazards_uav_and_xbox_bits() {
    let mut batcher = CoalescingBarrierBatcher::new();
    let mut sink = RecordingBarrierSink::new();
    let r1 = ResourceId(1);
    let r2 = ResourceId(2);
    let r3 = ResourceId(3);

    // 1. Xbox upper compression/DCC bits (bits 25..=31) stripped; identical PC states stripped.
    batcher.push_xbox_transition(
        r1,
        SubresourceIndex::ALL,
        ResourceStateFlags::RENDER_TARGET.0 | 0xFE00_0000,
        ResourceStateFlags::RENDER_TARGET.0 | 0x8000_0000,
    );
    assert_eq!(batcher.pending_len(), 0);

    // 2. Redundant read-to-read where before already contains after -> stripped.
    batcher.push_transition(TransitionBarrierDesc {
        resource: r1,
        subresource: SubresourceIndex::ALL,
        before: ResourceStateFlags::GENERIC_READ,
        after: ResourceStateFlags::PIXEL_SHADER_RESOURCE,
    });
    assert_eq!(batcher.pending_len(), 0);

    // 3. Fold A -> B -> C where B is non-write => A -> C (with Xbox bits stripped).
    batcher.push_xbox_transition(
        r1,
        SubresourceIndex::ALL,
        ResourceStateFlags::COPY_DEST.0 | 0x0200_0000,
        ResourceStateFlags::PIXEL_SHADER_RESOURCE.0 | 0x0400_0000,
    );
    batcher.push_transition(TransitionBarrierDesc {
        resource: r1,
        subresource: SubresourceIndex::ALL,
        before: ResourceStateFlags::PIXEL_SHADER_RESOURCE,
        after: ResourceStateFlags::NON_PIXEL_SHADER_RESOURCE,
    });

    // 4. Cancel A -> B -> A when B is non-write.
    batcher.push_transition(TransitionBarrierDesc {
        resource: r2,
        subresource: SubresourceIndex::ALL,
        before: ResourceStateFlags::RENDER_TARGET,
        after: ResourceStateFlags::PIXEL_SHADER_RESOURCE,
    });
    batcher.push_transition(TransitionBarrierDesc {
        resource: r2,
        subresource: SubresourceIndex::ALL,
        before: ResourceStateFlags::PIXEL_SHADER_RESOURCE,
        after: ResourceStateFlags::RENDER_TARGET,
    });

    // 5. PRESERVE write hazard: PSR -> RENDER_TARGET -> PSR MUST NOT fold or cancel!
    batcher.push_transition(TransitionBarrierDesc {
        resource: r3,
        subresource: SubresourceIndex::ALL,
        before: ResourceStateFlags::PIXEL_SHADER_RESOURCE,
        after: ResourceStateFlags::RENDER_TARGET,
    });
    batcher.push_transition(TransitionBarrierDesc {
        resource: r3,
        subresource: SubresourceIndex::ALL,
        before: ResourceStateFlags::RENDER_TARGET,
        after: ResourceStateFlags::PIXEL_SHADER_RESOURCE,
    });

    // 6. UAV deduplication: specific UAVs subsumed by global Uav(None) in same wave.
    batcher.push_uav(UavBarrierDesc { resource: Some(r1) });
    batcher.push_uav(UavBarrierDesc { resource: Some(r1) });
    batcher.push_uav(UavBarrierDesc { resource: Some(r2) });
    batcher.push_uav(UavBarrierDesc { resource: None });
    batcher.push_uav(UavBarrierDesc { resource: Some(r3) });

    // 7. Aliasing barrier preserves wave ordering; subsequent UAV is in a new wave.
    batcher.push_aliasing(AliasingBarrierDesc {
        before: Some(r1),
        after: Some(r2),
    });
    batcher.push_uav(UavBarrierDesc { resource: Some(r2) });

    // Single coalesced submit call on flush!
    let emitted = batcher.flush(&mut sink);
    assert_eq!(sink.call_count(), 1);
    assert_eq!(emitted, 6);

    let expected = vec![
        TranslatedBarrier::Transition(TransitionBarrierDesc {
            resource: r1,
            subresource: SubresourceIndex::ALL,
            before: ResourceStateFlags::COPY_DEST,
            after: ResourceStateFlags::NON_PIXEL_SHADER_RESOURCE,
        }),
        TranslatedBarrier::Transition(TransitionBarrierDesc {
            resource: r3,
            subresource: SubresourceIndex::ALL,
            before: ResourceStateFlags::PIXEL_SHADER_RESOURCE,
            after: ResourceStateFlags::RENDER_TARGET,
        }),
        TranslatedBarrier::Transition(TransitionBarrierDesc {
            resource: r3,
            subresource: SubresourceIndex::ALL,
            before: ResourceStateFlags::RENDER_TARGET,
            after: ResourceStateFlags::PIXEL_SHADER_RESOURCE,
        }),
        TranslatedBarrier::Uav(UavBarrierDesc { resource: None }),
        TranslatedBarrier::Aliasing(AliasingBarrierDesc {
            before: Some(r1),
            after: Some(r2),
        }),
        TranslatedBarrier::Uav(UavBarrierDesc { resource: Some(r2) }),
    ];
    assert_eq!(sink.flattened(), expected);

    let stats = batcher.stats();
    assert_eq!(stats.pushed_transitions, 8);
    assert_eq!(stats.stripped_noop_or_read_to_read, 3);
    assert_eq!(stats.folded_transitions, 2);
    assert_eq!(stats.pushed_uav, 6);
    assert_eq!(stats.deduplicated_uav, 4);
    assert_eq!(stats.pushed_aliasing, 1);
    assert_eq!(stats.flushed_calls, 1);
    assert_eq!(stats.flushed_barriers, 6);
    assert!(batcher.reduction_ratio() > 0.55);

    // Flushing when empty does not invoke sink.
    assert_eq!(batcher.flush(&mut sink), 0);
    assert_eq!(sink.call_count(), 1);
}

// ---------------------------------------------------------------------------
// 5. Windows D3D12 / WARP Hardware Integration Tests
// ---------------------------------------------------------------------------

#[cfg(windows)]
#[test]
fn d3d12_copy_queue_roundtrip_byte_exact() {
    use std::time::Duration;
    use victory_gpu::graphics::contracts::{BufferCopyOp, CopyQueueBackend, GpuHeapBackend, GpuTimelineFence};
    use victory_gpu::graphics::uma_bridge::d3d12_backend::{D3D12CopyQueueBackend, D3D12HeapBackend};

    let Ok((device, adapter, is_warp)) = D3D12HeapBackend::try_create_device_or_warp() else {
        eprintln!("skipping D3D12 hardware test: no D3D12 hardware or WARP adapter available");
        return;
    };

    let heap = Arc::new(D3D12HeapBackend::new(device, adapter.as_ref()));
    let copy_queue = D3D12CopyQueueBackend::new(Arc::clone(&heap), 1).expect("copy queue");
    let fence = Arc::clone(copy_queue.fence());

    let alloc = RebarUploadAllocator::new(
        Arc::clone(&heap),
        copy_queue,
        RebarAllocatorConfig {
            // Force StagingCopyRing path cleanly if ReBAR is absent, or exercise whichever tier is active.
            rebar_slab_size: 0,
            staging_ring_capacity: 64 * 1024,
            min_alignment: 256,
        },
    )
    .expect("D3D12 allocator init");

    let payload_len = 4096usize;
    let payload: Vec<u8> = (0..payload_len)
        .map(|i| ((i.wrapping_mul(131) ^ (i >> 3) ^ 0xA5) & 0xff) as u8)
        .collect();

    let (vram_id, _, _) = heap
        .create_buffer_slab(payload_len as u64, MemoryTier::DeviceLocalDefault)
        .expect("default VRAM buffer");
    let (readback_id, readback_ptr) = heap
        .create_readback_buffer(payload_len as u64)
        .expect("readback buffer");

    // Step 1: Write payload into StagingCopyRing and copy to DeviceLocalDefault on Copy Queue.
    let mut slice = alloc
        .allocate(payload_len as u64, 256)
        .expect("staging slice");
    assert_eq!(slice.tier, MemoryTier::StagingCopyRing);
    slice.write_sequential(0, &payload).expect("wc write");
    alloc
        .schedule_copy_to_vram(slice, vram_id, 0)
        .expect("schedule copy");
    let upload_token = alloc
        .flush_copies()
        .expect("flush upload")
        .expect("upload fence");
    assert!(
        fence
            .wait_cpu(upload_token, Duration::from_secs(5))
            .expect("wait upload"),
        "upload copy queue timed out (is_warp={is_warp})"
    );
    assert_eq!(alloc.reclaim_from_copy_queue(), 1);
    assert_eq!(alloc.staging_ring_used_bytes(), 0);

    // Step 2: Copy DeviceLocalDefault -> Readback on the Copy Queue and verify byte-exact match.
    let readback_token = alloc
        .copy_queue_backend()
        .submit_buffer_copies(&[BufferCopyOp {
            src_resource: vram_id,
            src_offset: 0,
            dst_resource: readback_id,
            dst_offset: 0,
            size: payload_len as u64,
        }])
        .expect("submit readback");
    assert!(
        fence
            .wait_cpu(readback_token, Duration::from_secs(5))
            .expect("wait readback"),
        "readback copy queue timed out (is_warp={is_warp})"
    );

    // SAFETY: GPU copy to `readback_id` completed (fence waited above) and `readback_ptr`
    // is mapped for `payload_len` bytes until `destroy_buffer_slab`.
    let gpu_bytes = unsafe { std::slice::from_raw_parts(readback_ptr.as_ptr(), payload_len) };
    assert_eq!(gpu_bytes, payload.as_slice());

    heap.destroy_buffer_slab(readback_id);
    heap.destroy_buffer_slab(vram_id);
}

#[cfg(windows)]
#[test]
fn d3d12_command_list_barrier_sink_executes_coalesced_batch() {
    use std::collections::HashMap;
    use std::time::Duration;
    use victory_gpu::graphics::contracts::{GpuHeapBackend, GpuTimelineFence};
    use victory_gpu::graphics::uma_bridge::D3D12CommandListBarrierSink;
    use victory_gpu::graphics::uma_bridge::d3d12_backend::{D3D12HeapBackend, D3D12TimelineFence};
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE,
        ID3D12CommandAllocator, ID3D12CommandList, ID3D12CommandQueue, ID3D12GraphicsCommandList,
    };
    use windows::core::Interface;

    let Ok((device, adapter, _is_warp)) = D3D12HeapBackend::try_create_device_or_warp() else {
        eprintln!("skipping D3D12 barrier test: no D3D12 hardware or WARP adapter available");
        return;
    };

    let heap = D3D12HeapBackend::new(device.clone(), adapter.as_ref());
    let (buf_a, _, _) = heap
        .create_buffer_slab(4096, MemoryTier::DeviceLocalDefault)
        .expect("buf_a");
    let (buf_b, _, _) = heap
        .create_buffer_slab(4096, MemoryTier::DeviceLocalDefault)
        .expect("buf_b");

    let mut resource_map = HashMap::new();
    resource_map.insert(buf_a, heap.get_resource(buf_a).expect("res a"));
    resource_map.insert(buf_b, heap.get_resource(buf_b).expect("res b"));

    let queue_desc = D3D12_COMMAND_QUEUE_DESC {
        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
        Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
        ..Default::default()
    };
    // SAFETY: Standard D3D12 direct command queue, allocator, and command list creation.
    let (queue, allocator, list): (
        ID3D12CommandQueue,
        ID3D12CommandAllocator,
        ID3D12GraphicsCommandList,
    ) = unsafe {
        let q = device.CreateCommandQueue(&queue_desc).expect("direct queue");
        let a = device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .expect("allocator");
        let l = device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &a, None)
            .expect("cmd list");
        (q, a, l)
    };
    let fence = D3D12TimelineFence::new(&device, 0).expect("fence");

    let mut batcher = CoalescingBarrierBatcher::new();
    // Valid transitions starting from COMMON (including Xbox upper bits and folding).
    batcher.push_xbox_transition(
        buf_a,
        SubresourceIndex::ALL,
        ResourceStateFlags::COMMON.0 | 0x0400_0000,
        ResourceStateFlags::COPY_SOURCE.0,
    );
    batcher.push_transition(TransitionBarrierDesc {
        resource: buf_a,
        subresource: SubresourceIndex::ALL,
        before: ResourceStateFlags::COPY_SOURCE,
        after: ResourceStateFlags::COPY_DEST,
    });
    // No-op on buf_b (stripped).
    batcher.push_transition(TransitionBarrierDesc {
        resource: buf_b,
        subresource: SubresourceIndex::ALL,
        before: ResourceStateFlags::COMMON,
        after: ResourceStateFlags::COMMON,
    });
    // Consecutive UAV wave deduplicated to 1 global UAV.
    batcher.push_uav(UavBarrierDesc {
        resource: Some(buf_a),
    });
    batcher.push_uav(UavBarrierDesc { resource: None });

    let mut sink = D3D12CommandListBarrierSink::new(&list, &resource_map);
    let emitted = batcher.flush(&mut sink);
    assert_eq!(emitted, 2);
    assert_eq!(sink.submitted_calls(), 1);
    assert_eq!(sink.submitted_barriers(), 2);
    assert_eq!(sink.skipped_unmapped(), 0);

    // SAFETY: Closing and executing our recorded direct command list and signalling fence 1.
    unsafe {
        list.Close().expect("close list");
        let base: ID3D12CommandList = list.cast().expect("cast list");
        queue.ExecuteCommandLists(&[Some(base)]);
        queue.Signal(fence.raw_fence(), 1).expect("signal");
    }
    assert!(
        fence
            .wait_cpu(GpuFenceToken { queue_id: 0, value: 1 }, Duration::from_secs(5))
            .expect("wait direct")
    );

    drop(allocator);
    heap.destroy_buffer_slab(buf_a);
    heap.destroy_buffer_slab(buf_b);
}
