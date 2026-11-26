//! High-performance rendering subsystems for running Xbox Series X|S builds on
//! discrete NVIDIA RTX GPUs:
//!
//! - [`contracts`]: Shared value types, error enums, and Rust trait contracts.
//! - [`uma_bridge`]: Subsystem A — UMA-to-Discrete Memory & Barrier Translator.
//! - [`velocity_io`]: Subsystem B — Velocity I/O (`DStorageX`) Async GPU Pipeline.

use std::sync::OnceLock;

pub mod contracts;
pub mod uma_bridge;
pub mod velocity_io;

pub use contracts::{
    AliasingBarrierDesc, AliasingTracker, AsyncIoBackend, AsyncReadCompletion, AsyncReadOp,
    BarrierBatchStats, BarrierBatcher, BarrierSink, BcPackFormat, BufferCopyOp,
    ComputeQueueBackend, CopyQueueBackend, DStorageDestination, DStoragePlatformHooks,
    DStoragePriority, DStorageRequestDesc, DStorageSource, DecompressionBackend,
    DecompressionCodec, DecompressionExecutionMode, DecompressionJob, DecompressionOutput,
    FileHandleId, GpuFenceToken, GpuHeapBackend, GpuTimelineFence, GpuVirtualAddress, HeapId,
    HeapRange, MemoryTier, RebarCapabilities, ResourceId, ResourceStateFlags, SubresourceIndex,
    TransitionBarrierDesc, TranslatedBarrier, UavBarrierDesc, UmaBridgeError, UploadAllocator,
    UploadSlice, VelocityIoError,
};
#[cfg(windows)]
pub use uma_bridge::D3D12CommandListBarrierSink;
pub use uma_bridge::{
    CoalescingBarrierBatcher, IntervalAliasingTracker, RebarAllocatorConfig, RebarUploadAllocator,
    RecordingBarrierSink,
};
#[cfg(target_os = "linux")]
pub use velocity_io::IoUringBackend;
#[cfg(windows)]
pub use velocity_io::Win32OverlappedIo;
pub use velocity_io::{
    BcPackStubBackend, DStorageQueueDescX, DStorageRequestX, DecompressionRegistry,
    DefaultPlatformHooks, GDeflateDecompressionBackend, GpuDecompressQueue,
    Lz4DecompressionBackend, PinnedLease, PinnedUploadRing, UncompressedCopyBackend,
    VelocityFactory, VelocityFile, VelocityQueue, VelocityQueueConfig, VelocityStatusArray,
    ZlibDecompressionBackend,
};

/// Returns whether Subsystem A (`uma_bridge` barrier batching and placed-resource
/// aliasing translation) is enabled via `VICTORY_UMA_BRIDGE=1` (off by default).
#[inline]
#[must_use]
pub fn uma_bridge_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("VICTORY_UMA_BRIDGE").is_some_and(|v| v == "1"))
}

/// Returns whether Subsystem B (`velocity_io` async DirectStorage pipeline) is
/// enabled via `VICTORY_VELOCITY_IO=1` (off by default).
#[inline]
#[must_use]
pub fn velocity_io_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("VICTORY_VELOCITY_IO").is_some_and(|v| v == "1"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{Iid, S_OK, This, slot};
    use crate::resource::{
        HeapDesc, ResourceDesc, create_at_address, create_heap, create_placed,
        record_placed_access, uma_aliasing_tracker,
    };

    fn com_release(this: This) -> u32 {
        let release_fn: extern "system" fn(This) -> u32 = slot(this, 0x10);
        release_fn(this)
    }

    const TEST_HEAP_IID: Iid = crate::object::guid(
        0x6b3b2502,
        0x6e51,
        0x45b3,
        [0x90, 0xee, 0x98, 0x84, 0x26, 0x5e, 0x8d, 0xf3],
    );
    const TEST_RESOURCE_IID: Iid = crate::object::guid(
        0x696442be,
        0xa72e,
        0x4059,
        [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
    );

    #[test]
    fn create_placed_registers_aliasing_and_unregisters_on_drop() {
        let heap_desc = HeapDesc::for_test(256 * 1024);
        let mut heap: This = std::ptr::null();
        assert_eq!(
            create_heap(std::ptr::null(), &heap_desc, &TEST_HEAP_IID, &mut heap),
            S_OK
        );
        assert!(!heap.is_null());

        let buf_desc = ResourceDesc::buffer_for_test(128 * 1024);
        let mut res_a: This = std::ptr::null();
        let mut res_b: This = std::ptr::null();
        assert_eq!(
            create_placed(
                std::ptr::null(),
                heap,
                0,
                &buf_desc,
                0,
                std::ptr::null(),
                &TEST_RESOURCE_IID,
                &mut res_a,
            ),
            S_OK
        );
        // Overlapping range [64 KiB, 192 KiB) in the same heap.
        assert_eq!(
            create_placed(
                std::ptr::null(),
                heap,
                64 * 1024,
                &buf_desc,
                0,
                std::ptr::null(),
                &TEST_RESOURCE_IID,
                &mut res_b,
            ),
            S_OK
        );

        let id_a = ResourceId(res_a as usize as u64);
        let id_b = ResourceId(res_b as usize as u64);

        // First access to A claims [0, 128 KiB) with 0 aliasing barriers.
        assert!(record_placed_access(res_a).is_empty());
        // Repeated access to A stays 0 barriers.
        assert!(record_placed_access(res_a).is_empty());

        // Access to overlapping B emits exactly one aliasing barrier (A -> B).
        let b_barriers = record_placed_access(res_b);
        assert_eq!(b_barriers.len(), 1);
        assert_eq!(
            b_barriers[0],
            AliasingBarrierDesc {
                before: Some(id_a),
                after: Some(id_b),
            }
        );

        // Feed the emitted aliasing barrier plus transitions into CoalescingBarrierBatcher
        // and verify coalescing & flush ordering.
        let mut batcher = CoalescingBarrierBatcher::new();
        batcher.push_transition(TransitionBarrierDesc {
            resource: id_a,
            subresource: SubresourceIndex::ALL,
            before: ResourceStateFlags::COPY_DEST,
            after: ResourceStateFlags::PIXEL_SHADER_RESOURCE,
        });
        batcher.push_transition(TransitionBarrierDesc {
            resource: id_a,
            subresource: SubresourceIndex::ALL,
            before: ResourceStateFlags::PIXEL_SHADER_RESOURCE,
            after: ResourceStateFlags::NON_PIXEL_SHADER_RESOURCE,
        });
        for alias in b_barriers {
            batcher.push_aliasing(alias);
        }
        batcher.push_uav(UavBarrierDesc {
            resource: Some(id_b),
        });
        batcher.push_uav(UavBarrierDesc { resource: None });

        let mut sink = RecordingBarrierSink::new();
        let flushed = batcher.flush(&mut sink);
        assert_eq!(sink.call_count(), 1);
        assert_eq!(flushed, 3);
        assert_eq!(
            sink.flattened(),
            vec![
                TranslatedBarrier::Transition(TransitionBarrierDesc {
                    resource: id_a,
                    subresource: SubresourceIndex::ALL,
                    before: ResourceStateFlags::COPY_DEST,
                    after: ResourceStateFlags::NON_PIXEL_SHADER_RESOURCE,
                }),
                TranslatedBarrier::Aliasing(AliasingBarrierDesc {
                    before: Some(id_a),
                    after: Some(id_b),
                }),
                TranslatedBarrier::Uav(UavBarrierDesc { resource: None }),
            ]
        );

        // Rule 2: dropping B must unregister it so later access to A never
        // produces an aliasing barrier referencing destroyed B.
        assert_eq!(com_release(res_b), 0);
        assert!(
            uma_aliasing_tracker()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .range_of(id_b)
                .is_none()
        );
        assert!(record_placed_access(res_a).is_empty());

        assert_eq!(com_release(res_a), 0);
        assert!(
            uma_aliasing_tracker()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .range_of(id_a)
                .is_none()
        );
        assert_eq!(com_release(heap), 0);
    }

    #[test]
    fn create_at_address_tracks_virtual_address_aliasing_and_cleans_up_on_drop() {
        let mut backing = vec![0u8; 128 * 1024];
        let base_va = backing.as_mut_ptr() as u64;
        let buf_desc = ResourceDesc::buffer_for_test(64 * 1024);

        let mut res_a: This = std::ptr::null();
        let mut res_b: This = std::ptr::null();
        assert_eq!(
            create_at_address(
                std::ptr::null(),
                base_va,
                &buf_desc,
                0,
                std::ptr::null(),
                &TEST_RESOURCE_IID,
                &mut res_a,
            ),
            S_OK
        );
        assert_eq!(
            create_at_address(
                std::ptr::null(),
                base_va + 32 * 1024,
                &buf_desc,
                0,
                std::ptr::null(),
                &TEST_RESOURCE_IID,
                &mut res_b,
            ),
            S_OK
        );

        let id_a = ResourceId(res_a as usize as u64);
        let id_b = ResourceId(res_b as usize as u64);

        assert!(record_placed_access(res_a).is_empty());
        let barriers = record_placed_access(res_b);
        assert_eq!(barriers.len(), 1);
        assert_eq!(
            barriers[0],
            AliasingBarrierDesc {
                before: Some(id_a),
                after: Some(id_b),
            }
        );

        assert_eq!(com_release(res_b), 0);
        assert_eq!(com_release(res_a), 0);
        let tracker = uma_aliasing_tracker()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        assert!(tracker.range_of(id_a).is_none());
        assert!(tracker.range_of(id_b).is_none());
    }
}
