//! Frozen shared value types, error enums, and Rust trait contracts bridging
//! Subsystem A (`uma_bridge`) and Subsystem B (`velocity_io`).
//!
//! All COM/D3D12/OS async I/O `unsafe` FFI calls remain isolated behind the
//! narrow backend traits defined here (`GpuHeapBackend`, `CopyQueueBackend`,
//! `BarrierSink`, `AsyncIoBackend`, `ComputeQueueBackend`).

use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::time::Duration;

use smallvec::SmallVec;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Shared identifiers, fence tokens, intervals, and resource states
// ---------------------------------------------------------------------------

/// 64-bit host or Xbox GPU virtual address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct GpuVirtualAddress(pub u64);

/// Opaque identifier for a D3D12 heap tracked by the UMA bridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct HeapId(pub u64);

/// Opaque identifier for a host or placed D3D12 resource (never dereferenced outside FFI).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ResourceId(pub u64);

/// Subresource selector (`SubresourceIndex::ALL == 0xFFFF_FFFF`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct SubresourceIndex(pub u32);

impl SubresourceIndex {
    /// Matches `D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES`.
    pub const ALL: Self = Self(u32::MAX);
}

/// Monotonic timeline fence signal token identifying work submitted on a GPU queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct GpuFenceToken {
    pub queue_id: u32,
    pub value: u64,
}

/// Half-open byte interval `[offset, offset + size)` inside a single D3D12 heap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HeapRange {
    pub heap_id: HeapId,
    pub offset: u64,
    pub size: u64,
}

impl HeapRange {
    /// Returns the exclusive end offset, or `None` on `u64` overflow.
    #[inline]
    #[must_use]
    pub const fn end(&self) -> Option<u64> {
        self.offset.checked_add(self.size)
    }

    /// Returns `true` iff both ranges reside in the same heap, are non-empty,
    /// and intersect on at least one byte.
    #[inline]
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        if self.heap_id != other.heap_id || self.size == 0 || other.size == 0 {
            return false;
        }
        match (self.end(), other.end()) {
            (Some(a_end), Some(b_end)) => self.offset < b_end && other.offset < a_end,
            _ => false,
        }
    }
}

/// Host PC `D3D12_RESOURCE_STATES` bitmask with Xbox upper compression bits stripped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct ResourceStateFlags(pub u32);

impl ResourceStateFlags {
    pub const COMMON: Self = Self(0);
    pub const VERTEX_AND_CONSTANT_BUFFER: Self = Self(0x1);
    pub const INDEX_BUFFER: Self = Self(0x2);
    pub const RENDER_TARGET: Self = Self(0x4);
    pub const UNORDERED_ACCESS: Self = Self(0x8);
    pub const DEPTH_WRITE: Self = Self(0x10);
    pub const DEPTH_READ: Self = Self(0x20);
    pub const NON_PIXEL_SHADER_RESOURCE: Self = Self(0x40);
    pub const PIXEL_SHADER_RESOURCE: Self = Self(0x80);
    pub const STREAM_OUT: Self = Self(0x100);
    pub const INDIRECT_ARGUMENT: Self = Self(0x200);
    pub const COPY_DEST: Self = Self(0x400);
    pub const COPY_SOURCE: Self = Self(0x800);
    pub const RESOLVE_DEST: Self = Self(0x1000);
    pub const RESOLVE_SOURCE: Self = Self(0x2000);
    pub const GENERIC_READ: Self = Self(0x1 | 0x2 | 0x40 | 0x80 | 0x200 | 0x800);
    pub const ALL_READ_STATES: Self =
        Self(Self::GENERIC_READ.0 | Self::DEPTH_READ.0 | Self::RESOLVE_SOURCE.0);
    pub const ALL_WRITE_STATES: Self = Self(
        Self::RENDER_TARGET.0
            | Self::UNORDERED_ACCESS.0
            | Self::DEPTH_WRITE.0
            | Self::STREAM_OUT.0
            | Self::COPY_DEST.0
            | Self::RESOLVE_DEST.0,
    );

    /// Strips Xbox-only compression/DCC hint bits (`bits 25..=31`), matching `host::pc_state`.
    #[inline]
    #[must_use]
    pub const fn from_xbox_raw(raw: u32) -> Self {
        Self(raw & 0x01ff_ffff)
    }

    /// Returns `true` when non-zero and every set bit is a D3D12 read-only state.
    #[inline]
    #[must_use]
    pub const fn is_read_only(self) -> bool {
        self.0 != 0 && (self.0 & !Self::ALL_READ_STATES.0) == 0
    }

    /// Returns `true` if any write/mutating D3D12 state bit is set.
    #[inline]
    #[must_use]
    pub const fn has_write(self) -> bool {
        (self.0 & Self::ALL_WRITE_STATES.0) != 0
    }

    /// Returns `true` when `self` already contains every bit set in `other`.
    #[inline]
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Bitwise union of two resource state masks.
    #[inline]
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

// ---------------------------------------------------------------------------
// Error enums
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum UmaBridgeError {
    #[error("invalid allocation size ({size}) or non-power-of-two alignment ({alignment})")]
    InvalidLayout { size: u64, alignment: u64 },
    #[error("heap range overflow for heap {heap_id:?} at offset {offset:#x} + {size:#x}")]
    RangeOverflow {
        heap_id: HeapId,
        offset: u64,
        size: u64,
    },
    #[error("upload ring exhausted (requested {requested} B, available {available} B)")]
    RingExhausted { requested: u64, available: u64 },
    #[error("allocator out of memory (requested {requested} B in tier {tier:?})")]
    OutOfMemory { requested: u64, tier: MemoryTier },
    #[error("host D3D12 error ({hresult:#010x}): {context}")]
    HostD3d12 { hresult: i32, context: &'static str },
}

#[derive(Debug, Error)]
pub enum VelocityIoError {
    #[error("invalid DirectStorage request: {0}")]
    InvalidRequest(&'static str),
    #[error("async file I/O failed (os={os_code:?}): {message}")]
    IoFailure {
        os_code: Option<i32>,
        message: String,
    },
    #[error("unexpected end of file at offset {offset:#x} (needed {needed} B, got {got} B)")]
    EndOfFile {
        offset: u64,
        needed: usize,
        got: usize,
    },
    #[error("pinned upload ring full (requested {requested} B)")]
    StagingRingFull { requested: usize },
    #[error("decompression failed for {codec:?}: {reason}")]
    DecompressionFailed {
        codec: DecompressionCodec,
        reason: String,
    },
    #[error("proprietary codec {codec:?} is a documented stub: {hint}")]
    ProprietaryCodecStub {
        codec: DecompressionCodec,
        hint: &'static str,
    },
    #[error("async compute queue error ({hresult:#010x}): {context}")]
    ComputeQueue { hresult: i32, context: &'static str },
    #[error("queue closed")]
    QueueClosed,
}

// ---------------------------------------------------------------------------
// Subsystem A (`uma_bridge`) shared descriptors & traits
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    /// Direct CPU-mapped Write-Combine VRAM (ReBAR / `D3D12_HEAP_TYPE_GPU_UPLOAD`,
    /// or `D3D12_HEAP_TYPE_CUSTOM` WC+L1 on UMA devices).
    RebarWriteCombine,
    /// Host-visible Write-Combine L0 (`D3D12_HEAP_TYPE_CUSTOM`, `WRITE_COMBINE` + `L0`)
    /// drained asynchronously via a `D3D12_COMMAND_LIST_TYPE_COPY` queue into L1.
    StagingCopyRing,
    /// Pure device-local VRAM (`D3D12_HEAP_TYPE_DEFAULT`).
    DeviceLocalDefault,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RebarCapabilities {
    pub gpu_upload_heap_supported: bool,
    pub is_uma: bool,
    pub dedicated_vram_bytes: u64,
    pub rebar_budget_bytes: u64,
}

/// Safe handle for a uniquely owned sub-allocated CPU-visible upload slice.
///
/// Writers should use [`UploadSlice::write_sequential`] to preserve write-combine
/// burst efficiency and avoid slow CPU read-modify-write traffic across PCIe.
#[derive(Debug)]
pub struct UploadSlice {
    pub allocation_id: NonZeroU64,
    pub tier: MemoryTier,
    pub resource_id: ResourceId,
    pub heap_offset: u64,
    pub size: u64,
    pub gpu_va: GpuVirtualAddress,
    /// Optional target VRAM resource when `tier == MemoryTier::StagingCopyRing`.
    pub target_resource_id: Option<ResourceId>,
    pub target_offset: u64,
    cpu_ptr: NonNull<u8>,
}

// SAFETY: Ownership of the disjoint sub-allocated byte range `[heap_offset, heap_offset + size)`
// is uniquely held by this `UploadSlice` until returned via `retire_slice` / `schedule_copy_to_vram`.
unsafe impl Send for UploadSlice {}

impl UploadSlice {
    /// Constructs a new `UploadSlice` from an audited heap backend mapping.
    ///
    /// # Safety
    /// `cpu_ptr` must be valid for unique write access to `size` contiguous bytes
    /// until this slice is retired after GPU fence completion.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub unsafe fn new(
        allocation_id: NonZeroU64,
        tier: MemoryTier,
        resource_id: ResourceId,
        heap_offset: u64,
        size: u64,
        gpu_va: GpuVirtualAddress,
        target_resource_id: Option<ResourceId>,
        target_offset: u64,
        cpu_ptr: NonNull<u8>,
    ) -> Self {
        Self {
            allocation_id,
            tier,
            resource_id,
            heap_offset,
            size,
            gpu_va,
            target_resource_id,
            target_offset,
            cpu_ptr,
        }
    }

    /// Writes `src` sequentially at `offset` into the mapped Write-Combine slice.
    pub fn write_sequential(&mut self, offset: usize, src: &[u8]) -> Result<(), UmaBridgeError> {
        let end = offset
            .checked_add(src.len())
            .ok_or(UmaBridgeError::InvalidLayout {
                size: src.len() as u64,
                alignment: 1,
            })?;
        if end as u64 > self.size {
            return Err(UmaBridgeError::InvalidLayout {
                size: end as u64,
                alignment: 1,
            });
        }
        // SAFETY: `self` uniquely owns `[cpu_ptr, cpu_ptr + size)` for writes and
        // `offset + src.len() <= self.size` was checked above.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.cpu_ptr.as_ptr().add(offset), src.len());
        }
        Ok(())
    }

    /// Returns the mapped base pointer (for DMA / async I/O write destinations).
    #[inline]
    #[must_use]
    pub fn raw_cpu_ptr(&self) -> NonNull<u8> {
        self.cpu_ptr
    }
}

/// Cross-subsystem timeline fence contract (implemented over `ID3D12Fence` / timeline semaphore).
pub trait GpuTimelineFence: Send + Sync {
    fn queue_id(&self) -> u32;
    fn completed_value(&self) -> u64;
    fn is_completed(&self, token: GpuFenceToken) -> bool;
    fn wait_cpu(&self, token: GpuFenceToken, timeout: Duration) -> Result<bool, UmaBridgeError>;
}

/// Thin audited FFI seam isolating `CreateHeap` / `CreateCommittedResource` / `Map` calls.
pub trait GpuHeapBackend: Send + Sync {
    fn probe_capabilities(&self) -> RebarCapabilities;
    fn create_buffer_slab(
        &self,
        size: u64,
        tier: MemoryTier,
    ) -> Result<(ResourceId, GpuVirtualAddress, Option<NonNull<u8>>), UmaBridgeError>;
    fn destroy_buffer_slab(&self, resource_id: ResourceId);
}

/// Single buffer copy operation queued onto the async Copy Queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferCopyOp {
    pub src_resource: ResourceId,
    pub src_offset: u64,
    pub dst_resource: ResourceId,
    pub dst_offset: u64,
    pub size: u64,
}

/// Thin audited FFI seam isolating `D3D12_COMMAND_LIST_TYPE_COPY` submissions.
pub trait CopyQueueBackend: Send + Sync {
    fn submit_buffer_copies(&self, copies: &[BufferCopyOp]) -> Result<GpuFenceToken, UmaBridgeError>;
    fn completed_fence_value(&self) -> u64;
    /// Blocks until all submitted copy queue work has completed on the GPU (called on shutdown/drop).
    fn drain_gpu(&self) {}
}

/// Subsystem A.1 — ReBAR-aware sub-allocator & non-ReBAR async Copy Queue ring fallback.
///
/// Consumed directly by Subsystem A command recording and by Subsystem B (`dstorage_thunk.rs`)
/// for pinned upload ring staging.
pub trait UploadAllocator: Send + Sync {
    fn capabilities(&self) -> RebarCapabilities;
    /// Never blocks the render thread; allocates from ReBAR slab or WC+L0 staging ring.
    fn allocate(&self, size: u64, alignment: u64) -> Result<UploadSlice, UmaBridgeError>;
    /// Records a deferred Copy Queue transfer for a `StagingCopyRing` slice into `dst_resource`.
    fn schedule_copy_to_vram(
        &self,
        slice: UploadSlice,
        dst_resource: ResourceId,
        dst_offset: u64,
    ) -> Result<(), UmaBridgeError>;
    /// Submits queued async Copy Queue transfers in one batch and returns its fence token.
    fn flush_copies(&self) -> Result<Option<GpuFenceToken>, UmaBridgeError>;
    /// Reclaims slabs/ring ranges whose fence token has completed on the GPU.
    fn reclaim_completed(&self, completed_fence: u64) -> usize;
    /// Retires a `RebarWriteCombine` or unused `StagingCopyRing` allocation once `retire_after` completes.
    fn retire_slice(&self, slice: UploadSlice, retire_after: GpuFenceToken);
}

/// Barriers emitted by [`AliasingTracker`] and [`BarrierBatcher`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransitionBarrierDesc {
    pub resource: ResourceId,
    pub subresource: SubresourceIndex,
    pub before: ResourceStateFlags,
    pub after: ResourceStateFlags,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AliasingBarrierDesc {
    pub before: Option<ResourceId>,
    pub after: Option<ResourceId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UavBarrierDesc {
    pub resource: Option<ResourceId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranslatedBarrier {
    Transition(TransitionBarrierDesc),
    Aliasing(AliasingBarrierDesc),
    Uav(UavBarrierDesc),
}

/// Subsystem A.2 — Interval tracker for overlapping `CreatePlacedResourceX` heap ranges.
pub trait AliasingTracker: Send {
    fn register_placed(&mut self, resource: ResourceId, range: HeapRange) -> Result<(), UmaBridgeError>;
    fn unregister_placed(&mut self, resource: ResourceId);
    /// Marks `resource` as the active owner of its `HeapRange`, returning the minimal
    /// `D3D12_RESOURCE_BARRIER_TYPE_ALIASING` transitions only for evicted overlapping owners.
    fn on_resource_access(&mut self, resource: ResourceId) -> SmallVec<[AliasingBarrierDesc; 4]>;
    fn active_owner_at(&self, heap_id: HeapId, offset: u64) -> Option<ResourceId>;
    fn clear_active_ownership(&mut self, heap_id: HeapId);
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BarrierBatchStats {
    pub pushed_transitions: u64,
    pub stripped_noop_or_read_to_read: u64,
    pub folded_transitions: u64,
    pub pushed_uav: u64,
    pub deduplicated_uav: u64,
    pub pushed_aliasing: u64,
    pub flushed_calls: u64,
    pub flushed_barriers: u64,
}

/// Thin FFI sink translating `&[TranslatedBarrier]` into a single `ResourceBarrier` call.
pub trait BarrierSink {
    fn submit_barriers(&mut self, barriers: &[TranslatedBarrier]);
}

/// Subsystem A.3 — Coalescing resource transition, aliasing, and UAV barrier batcher.
pub trait BarrierBatcher: Send {
    fn push_transition(&mut self, desc: TransitionBarrierDesc);
    fn push_aliasing(&mut self, desc: AliasingBarrierDesc);
    fn push_uav(&mut self, desc: UavBarrierDesc);
    fn flush(&mut self, sink: &mut dyn BarrierSink) -> usize;
    fn pending_len(&self) -> usize;
    fn stats(&self) -> BarrierBatchStats;
    fn reset(&mut self);
}

// ---------------------------------------------------------------------------
// Subsystem B (`velocity_io`) shared descriptors & traits
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BcPackFormat {
    Bc1 = 1,
    Bc3 = 3,
    Bc4 = 4,
    Bc5 = 5,
    Bc6h = 6,
    Bc7 = 7,
}

impl BcPackFormat {
    #[inline]
    #[must_use]
    pub const fn from_xbox_bits(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Bc1),
            3 => Some(Self::Bc3),
            4 => Some(Self::Bc4),
            5 => Some(Self::Bc5),
            6 => Some(Self::Bc6h),
            7 => Some(Self::Bc7),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecompressionCodec {
    Uncompressed,
    Zlib,
    Lz4,
    GDeflate,
    /// Strictly typed, documented stub behind [`DecompressionBackend`]
    /// per `.agents/rules/xbox-compat.md` Rule 1.
    BcPack(BcPackFormat),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecompressionExecutionMode {
    GpuComputeShader,
    GpuHardwareNv,
    CpuWorkerFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DStoragePriority {
    Low = -1,
    Normal = 0,
    High = 1,
    Realtime = 2,
}

impl DStoragePriority {
    #[inline]
    #[must_use]
    pub const fn from_i8(v: i8) -> Self {
        match v {
            ..=-1 => Self::Low,
            0 => Self::Normal,
            1 => Self::High,
            2.. => Self::Realtime,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct FileHandleId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DStorageSource {
    File {
        file: FileHandleId,
        offset: u64,
        size: u32,
    },
    Memory {
        address: usize,
        size: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DStorageDestination {
    /// Linear CPU/UMA virtual address.
    Memory { address: usize, size: u32 },
    /// Xbox 64 KiB physical-page list (`PAGE_DESTINATION`).
    PhysicalPages {
        page_list_addr: usize,
        page_offset: u16,
        size: u32,
    },
    /// Direct GPU buffer destination (`ID3D12Resource` + byte offset).
    GpuBuffer {
        resource: ResourceId,
        offset: u64,
        size: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DStorageRequestDesc {
    pub codec: DecompressionCodec,
    /// Secondary stage when Xbox chains `ZLIB` then `BCPACK`.
    pub secondary_codec: Option<DecompressionCodec>,
    pub intermediate_size: u32,
    pub source: DStorageSource,
    pub destination: DStorageDestination,
    pub cancellation_tag: u64,
}

/// Async read operation targeting a leased slot inside a pinned upload ring buffer.
#[derive(Debug, Clone, Copy)]
pub struct AsyncReadOp {
    pub op_id: u64,
    pub file: FileHandleId,
    pub file_offset: u64,
    pub len: usize,
    /// Destination pointer inside the pinned upload ring buffer (valid until completion).
    pub dst_ptr: NonNull<u8>,
}

// SAFETY: `dst_ptr` points to a uniquely leased slot in the pinned upload ring until `op_id` completes.
unsafe impl Send for AsyncReadOp {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsyncReadCompletion {
    pub op_id: u64,
    pub bytes_read: usize,
    pub os_error: Option<i32>,
}

/// Platform-isolated async file I/O backend (`OVERLAPPED` + IOCP on Windows, `io_uring` on Linux).
pub trait AsyncIoBackend: Send + Sync {
    fn open_read_only(&self, path: &Path) -> Result<FileHandleId, VelocityIoError>;
    fn close_file(&self, file: FileHandleId);
    /// Submits an async read batch without blocking the calling render/submission thread.
    fn submit_read_batch(&self, ops: &[AsyncReadOp]) -> Result<usize, VelocityIoError>;
    /// Polls/waits for OS async I/O completions on the dedicated I/O completion thread.
    fn poll_completions(
        &self,
        out: &mut Vec<AsyncReadCompletion>,
        timeout: Duration,
    ) -> Result<usize, VelocityIoError>;
}

/// Dependency-inversion hook implemented by `victory-hle` in Phase 2 (`files.rs` + `xmem.rs` + `watch.rs`).
pub trait DStoragePlatformHooks: Send + Sync {
    fn rewrite_virtual_path(&self, raw_utf16_path: &[u16]) -> PathBuf;
    fn write_physical_pages(
        &self,
        page_list_addr: usize,
        page_offset: u16,
        data: &[u8],
    ) -> Result<(), VelocityIoError>;
    fn notify_memory_written(&self, address: usize, len: usize);
}

/// Decompression job handed from `dstorage_thunk` to `gpu_decompress_queue`.
pub struct DecompressionJob<'a> {
    pub job_id: u64,
    pub codec: DecompressionCodec,
    pub src_bytes: &'a [u8],
    pub src_gpu_va: Option<GpuVirtualAddress>,
    pub src_gpu_resource: Option<(ResourceId, u64)>,
    pub expected_output_len: usize,
    pub dst_gpu_resource: Option<(ResourceId, u64)>,
    pub wait_copy_fence: Option<GpuFenceToken>,
}

pub struct DecompressionOutput {
    pub bytes_written: usize,
    pub completion_fence: Option<GpuFenceToken>,
    /// Populated when CPU worker fallback produced bytes that must land in title memory/pages.
    pub cpu_bytes: Option<Vec<u8>>,
}

/// Thin audited FFI seam isolating `D3D12_COMMAND_LIST_TYPE_COMPUTE` queue & fence operations.
pub trait ComputeQueueBackend: Send + Sync {
    fn wait_fence(&self, token: GpuFenceToken) -> Result<(), VelocityIoError>;
    fn dispatch_passthrough_copy(
        &self,
        src: ResourceId,
        src_offset: u64,
        dst: ResourceId,
        dst_offset: u64,
        byte_len: u64,
    ) -> Result<GpuFenceToken, VelocityIoError>;
    fn dispatch_compute_shader(
        &self,
        pipeline_key: &'static str,
        src_va: GpuVirtualAddress,
        dst_va: GpuVirtualAddress,
        params: &[u32],
        thread_groups: [u32; 3],
    ) -> Result<GpuFenceToken, VelocityIoError>;
    fn completed_fence_value(&self) -> u64;
}

/// Subsystem B.2 — Pluggable codec contract (GDeflate / LZ4 / Zlib + documented BCPack stub).
pub trait DecompressionBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn supported_codecs(&self) -> &'static [DecompressionCodec];
    fn execution_mode(&self) -> DecompressionExecutionMode;
    fn decompress(
        &self,
        job: &DecompressionJob<'_>,
        compute: &dyn ComputeQueueBackend,
    ) -> Result<DecompressionOutput, VelocityIoError>;
}
