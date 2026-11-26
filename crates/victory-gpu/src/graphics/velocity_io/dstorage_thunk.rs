//! Xbox Series X|S `DStorageX` ABI thunks, lock-free pinned upload ring, OS async
//! file I/O backends, and zero-stall async queue controllers.
//!
//! Implements Subsystem B.1:
//! - Exact `#[repr(C)]` Xbox ABI layouts ([`DStorageRequestX`] = `0x48` B,
//!   [`DStorageQueueDescX`] = `0x20` B) verified with compile-time layout assertions.
//! - Lock-free [`PinnedUploadRing`] using atomic monotonic head/tail cursors plus
//!   fence-gated [`crossbeam_queue::SegQueue`] retirement (supports both standalone
//!   4 KiB-aligned pinned buffers and Subsystem A's [`UploadAllocator`]).
//! - Platform-isolated [`AsyncIoBackend`] implementations:
//!   - `#[cfg(windows)]` [`Win32OverlappedIo`] over `FILE_FLAG_OVERLAPPED` + IOCP
//!     (`CreateIoCompletionPort` / `ReadFile` / `GetQueuedCompletionStatusEx`).
//!   - `#[cfg(target_os = "linux")]` [`IoUringBackend`] over `io_uring::IoUring`.
//!   - Cross-platform deterministic [`InMemoryAsyncIo`] for tests & benchmarks.
//! - Zero-stall [`VelocityFactory`], [`VelocityQueue`], and [`VelocityStatusArray`]
//!   controllers preserving strict Xbox batch-submit safety (never auto-submitting
//!   before explicit `submit()`) and strict in-order first-failure `HRESULT` semantics.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ffi::{c_char, c_void};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_queue::{ArrayQueue, SegQueue};

use crate::graphics::contracts::{
    AsyncIoBackend, AsyncReadCompletion, AsyncReadOp, BcPackFormat, DStorageDestination,
    DStoragePlatformHooks, DStoragePriority, DStorageRequestDesc, DStorageSource,
    DecompressionCodec, DecompressionJob, FileHandleId, GpuFenceToken, GpuVirtualAddress,
    ResourceId, UploadAllocator, UploadSlice, VelocityIoError,
};
use crate::graphics::velocity_io::gpu_decompress_queue::GpuDecompressQueue;

// ---------------------------------------------------------------------------
// Standard Xbox / Win32 HRESULT constants
// ---------------------------------------------------------------------------

pub const S_OK: i32 = 0;
pub const E_PENDING: i32 = 0x8000_000Au32 as i32;
pub const E_NOTIMPL: i32 = 0x8000_4001u32 as i32;
pub const E_FAIL: i32 = 0x8000_4005u32 as i32;
pub const E_OUTOFMEMORY: i32 = 0x8007_000Eu32 as i32;
pub const E_INVALIDARG: i32 = 0x8007_0057u32 as i32;
pub const HRESULT_INVALID_DATA: i32 = 0x8007_000Du32 as i32;
pub const HRESULT_HANDLE_EOF: i32 = 0x8007_0026u32 as i32;

/// Maps a [`VelocityIoError`] to the corresponding Win32 / Xbox DirectStorage `HRESULT`.
#[must_use]
pub fn velocity_error_to_hresult(err: &VelocityIoError) -> i32 {
    match err {
        VelocityIoError::InvalidRequest(_) => E_INVALIDARG,
        VelocityIoError::EndOfFile { .. } => HRESULT_HANDLE_EOF,
        VelocityIoError::StagingRingFull { .. } => E_OUTOFMEMORY,
        VelocityIoError::DecompressionFailed { .. } => HRESULT_INVALID_DATA,
        VelocityIoError::ProprietaryCodecStub { .. } => E_NOTIMPL,
        VelocityIoError::ComputeQueue { hresult, .. } => *hresult,
        VelocityIoError::IoFailure { os_code, .. } => match os_code {
            Some(code) if *code < 0 => *code,
            Some(code) => (0x8007_0000u32 | ((*code as u32) & 0xFFFF)) as i32,
            None => E_FAIL,
        },
        VelocityIoError::QueueClosed => E_FAIL,
    }
}

// ---------------------------------------------------------------------------
// Xbox Series X|S DStorageX Option Flags & ABI Structs
// ---------------------------------------------------------------------------

/// Xbox `DStorageRequestX::options` bit flags.
pub mod request_options {
    /// Bit 0: Inflate RFC 1950/1951 zlib stream.
    pub const ZLIB: u64 = 1 << 0;
    /// Bits 1..=4: Decode proprietary BCPack format code (`1..=7`, `0` = none).
    pub const BCPACK_MASK: u64 = 0xF << 1;
    /// Bit shift for the 4-bit BCPack format field inside `options`.
    pub const BCPACK_SHIFT: u32 = 1;
    /// Bit 5: Velocity I/O extension bit for standard LZ4 block/frame decompression.
    pub const LZ4: u64 = 1 << 5;
    /// Bit 6: Velocity I/O extension bit for open-spec Microsoft GDeflate v1 decompression.
    pub const GDEFLATE: u64 = 1 << 6;
    /// Bit 21: Destination pointer is an Xbox 64 KiB physical-page number list.
    pub const PAGE_DESTINATION: u64 = 1 << 21;
    /// Bit 22: Source pointer is a linear memory address rather than a file handle.
    pub const MEMORY_SOURCE: u64 = 1 << 22;

    /// Mask of all option bits understood by [`super::DStorageRequestX::to_desc`].
    pub const SUPPORTED_MASK: u64 =
        ZLIB | BCPACK_MASK | LZ4 | GDEFLATE | PAGE_DESTINATION | MEMORY_SOURCE;
}

pub use request_options::{BCPACK_MASK, GDEFLATE, LZ4, MEMORY_SOURCE, PAGE_DESTINATION, ZLIB};

/// Exact Xbox Series X|S DirectStorage request layout (`0x48` bytes on 64-bit),
/// matching `crates/victory-hle/src/modules/dstorage.rs`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DStorageRequestX {
    pub options: u64,
    /// Linear memory address, or (with [`PAGE_DESTINATION`]) an array of 64 KiB page numbers.
    pub destination: *mut c_void,
    /// With [`PAGE_DESTINATION`], low 16 bits give the byte offset inside the first page.
    pub page_offset: u64,
    pub destination_size: u32,
    /// File object pointer, or (with [`MEMORY_SOURCE`]) a linear memory source address.
    pub source: *const c_void,
    pub offset: u64,
    pub source_size: u32,
    /// Intermediate buffer size when chaining two decompression stages (e.g., Zlib -> BCPack).
    pub intermediate_size: u32,
    /// Cancellation tag (typically 0).
    pub tag: u64,
    /// Optional NUL-terminated debug name.
    pub name: *const c_char,
}

/// Exact Xbox Series X|S DirectStorage queue descriptor (`0x20` bytes on 64-bit),
/// matching `crates/victory-hle/src/modules/dstorage.rs`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DStorageQueueDescX {
    pub source_type: u64,
    pub capacity: u16,
    pub priority: i8,
    pub name: *const c_char,
    pub unknown: u64,
}

#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(std::mem::size_of::<DStorageRequestX>() == 0x48);
    assert!(std::mem::offset_of!(DStorageRequestX, options) == 0x00);
    assert!(std::mem::offset_of!(DStorageRequestX, destination) == 0x08);
    assert!(std::mem::offset_of!(DStorageRequestX, page_offset) == 0x10);
    assert!(std::mem::offset_of!(DStorageRequestX, destination_size) == 0x18);
    assert!(std::mem::offset_of!(DStorageRequestX, source) == 0x20);
    assert!(std::mem::offset_of!(DStorageRequestX, offset) == 0x28);
    assert!(std::mem::offset_of!(DStorageRequestX, source_size) == 0x30);
    assert!(std::mem::offset_of!(DStorageRequestX, intermediate_size) == 0x34);
    assert!(std::mem::offset_of!(DStorageRequestX, tag) == 0x38);
    assert!(std::mem::offset_of!(DStorageRequestX, name) == 0x40);

    assert!(std::mem::size_of::<DStorageQueueDescX>() == 0x20);
    assert!(std::mem::offset_of!(DStorageQueueDescX, source_type) == 0x00);
    assert!(std::mem::offset_of!(DStorageQueueDescX, capacity) == 0x08);
    assert!(std::mem::offset_of!(DStorageQueueDescX, priority) == 0x0A);
    assert!(std::mem::offset_of!(DStorageQueueDescX, name) == 0x10);
    assert!(std::mem::offset_of!(DStorageQueueDescX, unknown) == 0x18);
};

impl DStorageRequestX {
    /// Decodes a raw Xbox ABI request into a validated [`DStorageRequestDesc`].
    ///
    /// `resolve_file` translates the opaque `source` COM pointer into a host
    /// [`FileHandleId`] whenever [`MEMORY_SOURCE`] is not set.
    pub fn to_desc(
        &self,
        resolve_file: impl FnOnce(*const c_void) -> Result<FileHandleId, VelocityIoError>,
    ) -> Result<DStorageRequestDesc, VelocityIoError> {
        if self.options & !request_options::SUPPORTED_MASK != 0 {
            return Err(VelocityIoError::InvalidRequest("unsupported request option flags"));
        }
        if self.source.is_null() {
            return Err(VelocityIoError::InvalidRequest("source pointer is null"));
        }
        if self.destination.is_null() {
            return Err(VelocityIoError::InvalidRequest("destination pointer is null"));
        }

        let zlib = (self.options & ZLIB) != 0;
        let lz4 = (self.options & LZ4) != 0;
        let gdeflate = (self.options & GDEFLATE) != 0;
        let bc_bits = ((self.options & BCPACK_MASK) >> request_options::BCPACK_SHIFT) as u8;

        let bc_codec = if bc_bits != 0 {
            let fmt = BcPackFormat::from_xbox_bits(bc_bits).ok_or(
                VelocityIoError::InvalidRequest("unknown BCPack format code"),
            )?;
            Some(DecompressionCodec::BcPack(fmt))
        } else {
            None
        };

        let (codec, secondary_codec) = match (zlib, bc_codec, lz4, gdeflate) {
            (false, None, false, false) => {
                if self.source_size > self.destination_size {
                    return Err(VelocityIoError::InvalidRequest(
                        "uncompressed source_size exceeds destination_size",
                    ));
                }
                (DecompressionCodec::Uncompressed, None)
            }
            (true, None, false, false) => (DecompressionCodec::Zlib, None),
            (false, Some(bc), false, false) => (bc, None),
            // Xbox chains Zlib first, then BCPack into the final destination.
            (true, Some(bc), false, false) => (DecompressionCodec::Zlib, Some(bc)),
            (false, None, true, false) => (DecompressionCodec::Lz4, None),
            (false, None, false, true) => (DecompressionCodec::GDeflate, None),
            _ => {
                return Err(VelocityIoError::InvalidRequest(
                    "conflicting compression codec flags",
                ));
            }
        };

        let source = if (self.options & MEMORY_SOURCE) != 0 {
            DStorageSource::Memory {
                address: self.source as usize,
                size: self.source_size,
            }
        } else {
            DStorageSource::File {
                file: resolve_file(self.source)?,
                offset: self.offset,
                size: self.source_size,
            }
        };

        let destination = if (self.options & PAGE_DESTINATION) != 0 {
            DStorageDestination::PhysicalPages {
                page_list_addr: self.destination as usize,
                page_offset: (self.page_offset & 0xFFFF) as u16,
                size: self.destination_size,
            }
        } else {
            DStorageDestination::Memory {
                address: self.destination as usize,
                size: self.destination_size,
            }
        };

        Ok(DStorageRequestDesc {
            codec,
            secondary_codec,
            intermediate_size: self.intermediate_size,
            source,
            destination,
            cancellation_tag: self.tag,
        })
    }
}

impl DStorageQueueDescX {
    /// Extracts the normalized [`DStoragePriority`] from the raw `i8` priority field.
    #[inline]
    #[must_use]
    pub const fn priority_enum(&self) -> DStoragePriority {
        DStoragePriority::from_i8(self.priority)
    }
}

// ---------------------------------------------------------------------------
// DefaultPlatformHooks
// ---------------------------------------------------------------------------

/// Default [`DStoragePlatformHooks`] used until `victory-hle` injects title-specific
/// virtual path rewriting (`files.rs`) and physical-page writes (`xmem.rs`) in Phase 2.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultPlatformHooks;

impl DStoragePlatformHooks for DefaultPlatformHooks {
    fn rewrite_virtual_path(&self, raw_utf16_path: &[u16]) -> PathBuf {
        let end = raw_utf16_path
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(raw_utf16_path.len());
        PathBuf::from(String::from_utf16_lossy(&raw_utf16_path[..end]))
    }

    fn write_physical_pages(
        &self,
        _page_list_addr: usize,
        _page_offset: u16,
        _data: &[u8],
    ) -> Result<(), VelocityIoError> {
        Err(VelocityIoError::InvalidRequest(
            "physical page writes require HLE platform hooks",
        ))
    }

    fn notify_memory_written(&self, _address: usize, _len: usize) {}
}

// ---------------------------------------------------------------------------
// Lock-Free Fence-Gated PinnedUploadRing
// ---------------------------------------------------------------------------

/// Alignment for standalone pinned host ring buffers (4 KiB OS page / NVMe sector alignment).
pub const PINNED_RING_PAGE_ALIGN: usize = 4096;
/// Per-lease alignment inside standalone pinned ring buffers (64-byte cacheline aligned).
const LEASE_ALIGN: usize = 64;

struct RetiredRange {
    monotonic_start: u64,
    monotonic_end: u64,
    retire_after_fence: u64,
}

struct StandaloneRingStorage {
    base: NonNull<u8>,
    layout: Layout,
    capacity: usize,
    mask: usize,
    head: AtomicU64,
    tail: AtomicU64,
    retired_queue: SegQueue<RetiredRange>,
    /// Coalesces out-of-order retired ranges strictly at `reclaim_completed` time;
    /// never touched by the lock-free `try_lease` or `retire_lease` hot paths.
    pending_reclaim: Mutex<BTreeMap<u64, (u64, u64)>>,
}

// SAFETY: `StandaloneRingStorage` owns a unique heap allocation and hands out strictly
// disjoint byte ranges via atomic CAS on `head` / fence-gated reclamation on `tail`.
unsafe impl Send for StandaloneRingStorage {}
// SAFETY: All hot-path fields are atomic or lock-free (`SegQueue`), and `pending_reclaim`
// is protected by a `Mutex`.
unsafe impl Sync for StandaloneRingStorage {}

impl Drop for StandaloneRingStorage {
    fn drop(&mut self) {
        // SAFETY: `self.base` was allocated with `alloc_zeroed(self.layout)` in `new_standalone`.
        unsafe {
            dealloc(self.base.as_ptr(), self.layout);
        }
    }
}

enum RingBacking {
    Standalone(Box<StandaloneRingStorage>),
    UploadAllocator {
        allocator: Arc<dyn UploadAllocator>,
        alignment: u64,
    },
}

/// Lock-free pinned upload ring buffer with fence-gated reclamation.
///
/// Supports either:
/// - A standalone 4 KiB-aligned pinned host ring ([`PinnedUploadRing::new_standalone`]), or
/// - Delegation to Subsystem A's [`UploadAllocator`] ([`PinnedUploadRing::with_upload_allocator`]).
///
/// Both [`PinnedUploadRing::try_lease`] and [`PinnedUploadRing::retire_lease`] are
/// non-blocking and safe to call concurrently across multiple threads.
pub struct PinnedUploadRing {
    backing: RingBacking,
}

impl std::fmt::Debug for PinnedUploadRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.backing {
            RingBacking::Standalone(s) => f
                .debug_struct("PinnedUploadRing::Standalone")
                .field("capacity", &s.capacity)
                .field("head", &s.head.load(Ordering::Relaxed))
                .field("tail", &s.tail.load(Ordering::Relaxed))
                .finish(),
            RingBacking::UploadAllocator { alignment, .. } => f
                .debug_struct("PinnedUploadRing::UploadAllocator")
                .field("alignment", alignment)
                .finish(),
        }
    }
}

/// Uniquely owned leased slice inside a [`PinnedUploadRing`].
///
/// Must be returned via [`PinnedUploadRing::retire_lease`] once async I/O and GPU
/// consumption have completed (or with a completion [`GpuFenceToken`]).
#[derive(Debug)]
pub struct PinnedLease {
    ptr: NonNull<u8>,
    len: usize,
    ring_offset: usize,
    monotonic_start: u64,
    monotonic_end: u64,
    upload_slice: Option<UploadSlice>,
    /// One-shot heap buffer for rare oversized reads exceeding the staging ring's capacity.
    _overflow_buf: Option<Box<[u8]>>,
}

// SAFETY: Each `PinnedLease` uniquely owns its disjoint `[ptr, ptr + len)` byte slice
// until returned to its originating `PinnedUploadRing`.
unsafe impl Send for PinnedLease {}

impl PinnedLease {
    /// Allocates a one-shot heap-backed lease for reads larger than the staging ring capacity.
    fn new_overflow(len: usize) -> Self {
        let mut buf = vec![0u8; len.max(1)].into_boxed_slice();
        // SAFETY: `buf` has non-zero length and remains pinned on the heap inside `_overflow_buf`
        // for the full lifetime of this `PinnedLease`.
        let ptr = unsafe { NonNull::new_unchecked(buf.as_mut_ptr()) };
        Self {
            ptr,
            len,
            ring_offset: 0,
            monotonic_start: 0,
            monotonic_end: 0,
            upload_slice: None,
            _overflow_buf: Some(buf),
        }
    }
    /// Returns the raw base pointer for OS async I/O (`ReadFile` / `io_uring`) destinations.
    #[inline]
    #[must_use]
    pub fn raw_ptr(&self) -> NonNull<u8> {
        self.ptr
    }

    /// Returns the leased byte length.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if this lease is zero-length.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the byte offset inside the underlying ring/slab.
    #[inline]
    #[must_use]
    pub fn ring_offset(&self) -> usize {
        self.ring_offset
    }

    /// Returns the GPU virtual address if backed by an [`UploadAllocator`].
    #[inline]
    #[must_use]
    pub fn gpu_va(&self) -> Option<GpuVirtualAddress> {
        self.upload_slice.as_ref().map(|s| s.gpu_va)
    }

    /// Returns `(ResourceId, heap_offset)` if backed by an [`UploadAllocator`].
    #[inline]
    #[must_use]
    pub fn gpu_resource(&self) -> Option<(ResourceId, u64)> {
        self.upload_slice
            .as_ref()
            .map(|s| (s.resource_id, s.heap_offset))
    }

    /// Borrows the leased bytes as an immutable slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `self` uniquely owns `[self.ptr, self.ptr + self.len)` initialized bytes.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Borrows the leased bytes as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: `&mut self` grants exclusive access to `[self.ptr, self.ptr + self.len)`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl PinnedUploadRing {
    /// Creates a standalone 4 KiB-aligned pinned ring buffer.
    ///
    /// `capacity_bytes` is rounded up to the next power of two (minimum 4 KiB).
    pub fn new_standalone(capacity_bytes: usize) -> Result<Self, VelocityIoError> {
        let capacity = capacity_bytes
            .max(PINNED_RING_PAGE_ALIGN)
            .checked_next_power_of_two()
            .ok_or(VelocityIoError::InvalidRequest(
                "pinned upload ring capacity overflow",
            ))?;
        let layout = Layout::from_size_align(capacity, PINNED_RING_PAGE_ALIGN).map_err(|_| {
            VelocityIoError::InvalidRequest("invalid pinned upload ring memory layout")
        })?;
        // SAFETY: `layout` has non-zero power-of-two size and 4 KiB alignment.
        let raw = unsafe { alloc_zeroed(layout) };
        let base = NonNull::new(raw).ok_or(VelocityIoError::StagingRingFull {
            requested: capacity,
        })?;

        Ok(Self {
            backing: RingBacking::Standalone(Box::new(StandaloneRingStorage {
                base,
                layout,
                capacity,
                mask: capacity - 1,
                head: AtomicU64::new(0),
                tail: AtomicU64::new(0),
                retired_queue: SegQueue::new(),
                pending_reclaim: Mutex::new(BTreeMap::new()),
            })),
        })
    }

    /// Creates a ring backed by Subsystem A's [`UploadAllocator`].
    #[must_use]
    pub fn with_upload_allocator(allocator: Arc<dyn UploadAllocator>, alignment: u64) -> Self {
        let align = if alignment.is_power_of_two() && alignment >= 64 {
            alignment
        } else {
            256
        };
        Self {
            backing: RingBacking::UploadAllocator {
                allocator,
                alignment: align,
            },
        }
    }

    /// Returns the total byte capacity for standalone rings, or `None` when delegated
    /// to an external [`UploadAllocator`].
    #[must_use]
    pub fn capacity_bytes(&self) -> Option<usize> {
        match &self.backing {
            RingBacking::Standalone(s) => Some(s.capacity),
            RingBacking::UploadAllocator { .. } => None,
        }
    }

    /// Returns currently occupied bytes in a standalone ring (approximate under contention).
    #[must_use]
    pub fn used_bytes(&self) -> usize {
        match &self.backing {
            RingBacking::Standalone(s) => {
                let h = s.head.load(Ordering::Acquire);
                let t = s.tail.load(Ordering::Acquire);
                h.saturating_sub(t) as usize
            }
            RingBacking::UploadAllocator { .. } => 0,
        }
    }

    /// Attempts to lease a contiguous `len`-byte pinned slice without blocking.
    ///
    /// Returns [`VelocityIoError::StagingRingFull`] immediately if insufficient free
    /// space is available before fence reclamation.
    pub fn try_lease(&self, len: usize) -> Result<PinnedLease, VelocityIoError> {
        match &self.backing {
            RingBacking::Standalone(s) => {
                if len == 0 {
                    return Ok(PinnedLease {
                        ptr: s.base,
                        len: 0,
                        ring_offset: 0,
                        monotonic_start: 0,
                        monotonic_end: 0,
                        upload_slice: None,
                        _overflow_buf: None,
                    });
                }
                let aligned_len = len
                    .checked_add(LEASE_ALIGN - 1)
                    .map(|v| v & !(LEASE_ALIGN - 1))
                    .ok_or(VelocityIoError::StagingRingFull { requested: len })?;
                if aligned_len > s.capacity {
                    return Err(VelocityIoError::StagingRingFull { requested: len });
                }

                let mut cur_head = s.head.load(Ordering::Acquire);
                loop {
                    let cur_tail = s.tail.load(Ordering::Acquire);
                    let physical_off = (cur_head as usize) & s.mask;
                    let wrap_padding = if physical_off + aligned_len > s.capacity {
                        s.capacity - physical_off
                    } else {
                        0
                    };
                    let total_advance = (wrap_padding + aligned_len) as u64;
                    let new_head = cur_head.wrapping_add(total_advance);
                    let in_flight = new_head.saturating_sub(cur_tail);
                    if in_flight > s.capacity as u64 {
                        // Avoid empty-ring wrap livelock: when `cur_head == cur_tail` past the
                        // midpoint and `aligned_len > capacity - physical_off`, advance `head`
                        // across `wrap_padding` and immediately retire/reclaim that empty tail
                        // padding so the allocation can succeed from ring offset 0.
                        if wrap_padding > 0 && cur_head == cur_tail {
                            let pad_end = cur_head.wrapping_add(wrap_padding as u64);
                            match s.head.compare_exchange_weak(
                                cur_head,
                                pad_end,
                                Ordering::AcqRel,
                                Ordering::Acquire,
                            ) {
                                Ok(_) => {
                                    s.retired_queue.push(RetiredRange {
                                        monotonic_start: cur_head,
                                        monotonic_end: pad_end,
                                        retire_after_fence: 0,
                                    });
                                    let _ = self.reclaim_completed(0);
                                    cur_head = s.head.load(Ordering::Acquire);
                                    continue;
                                }
                                Err(observed) => {
                                    cur_head = observed;
                                    continue;
                                }
                            }
                        }
                        return Err(VelocityIoError::StagingRingFull { requested: len });
                    }

                    match s.head.compare_exchange_weak(
                        cur_head,
                        new_head,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => {
                            let slice_off = ((cur_head + wrap_padding as u64) as usize) & s.mask;
                            // SAFETY: `slice_off + len <= slice_off + aligned_len <= s.capacity`
                            // and this CAS uniquely reserves `[cur_head, new_head)`.
                            let ptr = unsafe {
                                NonNull::new_unchecked(s.base.as_ptr().add(slice_off))
                            };
                            return Ok(PinnedLease {
                                ptr,
                                len,
                                ring_offset: slice_off,
                                monotonic_start: cur_head,
                                monotonic_end: new_head,
                                upload_slice: None,
                                _overflow_buf: None,
                            });
                        }
                        Err(observed) => cur_head = observed,
                    }
                }
            }
            RingBacking::UploadAllocator {
                allocator,
                alignment,
            } => {
                let alloc_size = (len as u64).max(1);
                let slice = allocator.allocate(alloc_size, *alignment).map_err(|_| {
                    VelocityIoError::StagingRingFull { requested: len }
                })?;
                let ptr = slice.raw_cpu_ptr();
                let ring_offset = slice.heap_offset as usize;
                Ok(PinnedLease {
                    ptr,
                    len,
                    ring_offset,
                    monotonic_start: 0,
                    monotonic_end: 0,
                    upload_slice: Some(slice),
                    _overflow_buf: None,
                })
            }
        }
    }

    /// Lock-free retirement of `lease` gated on `retire_after`.
    ///
    /// The underlying memory will not be reused until [`Self::reclaim_completed`]
    /// observes a completed fence value `>= retire_after.value` (and all earlier
    /// leases in ring order have also retired).
    pub fn retire_lease(&self, lease: PinnedLease, retire_after: GpuFenceToken) {
        match &self.backing {
            RingBacking::Standalone(s) => {
                if lease.monotonic_end > lease.monotonic_start {
                    s.retired_queue.push(RetiredRange {
                        monotonic_start: lease.monotonic_start,
                        monotonic_end: lease.monotonic_end,
                        retire_after_fence: retire_after.value,
                    });
                }
            }
            RingBacking::UploadAllocator { allocator, .. } => {
                if let Some(slice) = lease.upload_slice {
                    allocator.retire_slice(slice, retire_after);
                }
            }
        }
    }

    /// Advances the ring tail past all contiguous retired leases whose fence token
    /// is `<= completed_fence`, returning the number of reclaimed leases.
    pub fn reclaim_completed(&self, completed_fence: u64) -> usize {
        match &self.backing {
            RingBacking::Standalone(s) => {
                let mut pending = s
                    .pending_reclaim
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                while let Some(item) = s.retired_queue.pop() {
                    pending.insert(
                        item.monotonic_start,
                        (item.monotonic_end, item.retire_after_fence),
                    );
                }

                let mut cur_tail = s.tail.load(Ordering::Acquire);
                let mut reclaimed = 0usize;
                while let Some((&start, &(end, fence_val))) = pending.first_key_value() {
                    if start != cur_tail || fence_val > completed_fence {
                        break;
                    }
                    pending.remove(&start);
                    cur_tail = end;
                    reclaimed += 1;
                }
                if reclaimed > 0 {
                    s.tail.store(cur_tail, Ordering::Release);
                }
                reclaimed
            }
            RingBacking::UploadAllocator { allocator, .. } => {
                allocator.reclaim_completed(completed_fence)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// OS Async I/O Backends (`Win32OverlappedIo`, `IoUringBackend`, `InMemoryAsyncIo`)
// ---------------------------------------------------------------------------

/// Thin audited Windows `OVERLAPPED` + I/O Completion Port (`IOCP`) async file I/O backend.
///
/// Opens files with `FILE_FLAG_OVERLAPPED`, associates handles with an IOCP via
/// `CreateIoCompletionPort`, submits non-blocking `ReadFile` calls directly into
/// leased [`PinnedUploadRing`] pointers, and batch-reaps completions with
/// `GetQueuedCompletionStatusEx`.
#[cfg(windows)]
pub struct Win32OverlappedIo {
    iocp: windows_sys::Win32::Foundation::HANDLE,
    next_file_id: AtomicU64,
    files: RwLock<HashMap<FileHandleId, Arc<Win32FileHandle>>>,
    immediate_completions: SegQueue<AsyncReadCompletion>,
}

#[cfg(windows)]
struct Win32FileHandle {
    raw: windows_sys::Win32::Foundation::HANDLE,
}

#[cfg(windows)]
// SAFETY: Win32 file handles opened with `FILE_FLAG_OVERLAPPED` support concurrent
// overlapped reads from multiple threads.
unsafe impl Send for Win32FileHandle {}
#[cfg(windows)]
// SAFETY: See `Send` invariant above.
unsafe impl Sync for Win32FileHandle {}

#[cfg(windows)]
impl Drop for Win32FileHandle {
    fn drop(&mut self) {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        if !self.raw.is_null() && self.raw != INVALID_HANDLE_VALUE {
            // SAFETY: `self.raw` was created by `CreateFileW` and is owned exclusively by this `Arc`.
            unsafe {
                CloseHandle(self.raw);
            }
        }
    }
}

#[cfg(windows)]
#[repr(C)]
struct OverlappedReadContext {
    /// MUST remain the first field at offset 0 so `OVERLAPPED_ENTRY::lpOverlapped`
    /// casts directly back to `*mut OverlappedReadContext`.
    overlapped: windows_sys::Win32::System::IO::OVERLAPPED,
    op_id: u64,
    /// Keeps the OS file handle alive for the entire duration of the in-flight async read.
    _file: Arc<Win32FileHandle>,
}

#[cfg(windows)]
// SAFETY: IOCP handles are kernel objects explicitly designed for multi-threaded
// submission (`ReadFile` / `PostQueuedCompletionStatus`) and reaping (`GetQueuedCompletionStatusEx`).
unsafe impl Send for Win32OverlappedIo {}
#[cfg(windows)]
// SAFETY: See `Send` invariant above.
unsafe impl Sync for Win32OverlappedIo {}

#[cfg(windows)]
impl Win32OverlappedIo {
    /// Sentinel completion key used to wake a thread parked in `GetQueuedCompletionStatusEx`.
    const WAKE_KEY: usize = usize::MAX;
    /// NTSTATUS `STATUS_END_OF_FILE` (`0xC0000011`) stored in `OVERLAPPED.Internal`.
    const STATUS_END_OF_FILE: u32 = 0xC000_0011;

    /// Creates a new Win32 IOCP-backed async file I/O engine.
    pub fn new() -> Result<Self, VelocityIoError> {
        use windows_sys::Win32::Foundation::{GetLastError, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::IO::CreateIoCompletionPort;

        // SAFETY: Creating a fresh standalone I/O Completion Port with default concurrency.
        let iocp = unsafe { CreateIoCompletionPort(INVALID_HANDLE_VALUE, std::ptr::null_mut(), 0, 0) };
        if iocp.is_null() {
            // SAFETY: Thread-local Win32 error retrieval immediately after failed `CreateIoCompletionPort`.
            let err = unsafe { GetLastError() } as i32;
            return Err(VelocityIoError::IoFailure {
                os_code: Some(err),
                message: format!("CreateIoCompletionPort failed (os={err})"),
            });
        }
        Ok(Self {
            iocp,
            next_file_id: AtomicU64::new(1),
            files: RwLock::new(HashMap::new()),
            immediate_completions: SegQueue::new(),
        })
    }

    fn wake_iocp(&self) {
        use windows_sys::Win32::System::IO::PostQueuedCompletionStatus;
        // SAFETY: `self.iocp` is a valid IOCP handle; null OVERLAPPED with `WAKE_KEY` is ignored by poller.
        unsafe {
            let _ = PostQueuedCompletionStatus(self.iocp, 0, Self::WAKE_KEY, std::ptr::null());
        }
    }
}

#[cfg(windows)]
impl Drop for Win32OverlappedIo {
    fn drop(&mut self) {
        use windows_sys::Win32::Foundation::CloseHandle;
        if !self.iocp.is_null() {
            // SAFETY: `self.iocp` was created by `CreateIoCompletionPort` and owned by `self`.
            unsafe {
                CloseHandle(self.iocp);
            }
        }
    }
}

#[cfg(windows)]
impl AsyncIoBackend for Win32OverlappedIo {
    fn open_read_only(&self, path: &Path) -> Result<FileHandleId, VelocityIoError> {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Foundation::{
            CloseHandle, GENERIC_READ, GetLastError, INVALID_HANDLE_VALUE,
        };
        use windows_sys::Win32::Storage::FileSystem::{
            CreateFileW, FILE_ATTRIBUTE_NORMAL, FILE_FLAG_OVERLAPPED, FILE_SHARE_DELETE,
            FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
        };
        use windows_sys::Win32::System::IO::CreateIoCompletionPort;

        let wide: Vec<u16> = path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();

        // SAFETY: `wide` is a valid NUL-terminated UTF-16 path string.
        let raw = unsafe {
            CreateFileW(
                wide.as_ptr(),
                GENERIC_READ,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                std::ptr::null(),
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
                std::ptr::null_mut(),
            )
        };
        if raw == INVALID_HANDLE_VALUE {
            // SAFETY: Thread-local Win32 error retrieval immediately following failed `CreateFileW`.
            let err = unsafe { GetLastError() } as i32;
            return Err(VelocityIoError::IoFailure {
                os_code: Some(err),
                message: format!("CreateFileW failed for '{}' (os={err})", path.display()),
            });
        }

        let file_id = FileHandleId(self.next_file_id.fetch_add(1, Ordering::Relaxed));
        // SAFETY: Associating freshly opened overlapped file handle `raw` with `self.iocp`.
        let associated = unsafe { CreateIoCompletionPort(raw, self.iocp, file_id.0 as usize, 0) };
        if associated.is_null() {
            // SAFETY: Retrieving error and closing `raw` on association failure to avoid handle leak.
            let err = unsafe {
                let code = GetLastError() as i32;
                CloseHandle(raw);
                code
            };
            return Err(VelocityIoError::IoFailure {
                os_code: Some(err),
                message: format!(
                    "CreateIoCompletionPort association failed for '{}' (os={err})",
                    path.display()
                ),
            });
        }

        self.files
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(file_id, Arc::new(Win32FileHandle { raw }));
        Ok(file_id)
    }

    fn close_file(&self, file: FileHandleId) {
        self.files
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&file);
    }

    fn submit_read_batch(&self, ops: &[AsyncReadOp]) -> Result<usize, VelocityIoError> {
        use windows_sys::Win32::Foundation::{ERROR_HANDLE_EOF, ERROR_IO_PENDING, GetLastError};
        use windows_sys::Win32::Storage::FileSystem::ReadFile;
        use windows_sys::Win32::System::IO::OVERLAPPED;

        if ops.is_empty() {
            return Ok(0);
        }

        // Validate handles & lengths up front before issuing any kernel calls.
        let resolved: Vec<Arc<Win32FileHandle>> = {
            let guard = self.files.read().unwrap_or_else(|e| e.into_inner());
            let mut handles = Vec::with_capacity(ops.len());
            for op in ops {
                if op.len > i32::MAX as usize {
                    return Err(VelocityIoError::InvalidRequest(
                        "async read length exceeds Win32 ReadFile limit",
                    ));
                }
                let handle = guard
                    .get(&op.file)
                    .cloned()
                    .ok_or(VelocityIoError::InvalidRequest("unknown file handle"))?;
                handles.push(handle);
            }
            handles
        };

        let mut submitted = 0usize;
        let mut pushed_immediate = false;

        for (op, file) in ops.iter().zip(resolved) {
            if op.len == 0 {
                self.immediate_completions.push(AsyncReadCompletion {
                    op_id: op.op_id,
                    bytes_read: 0,
                    os_error: None,
                });
                pushed_immediate = true;
                submitted += 1;
                continue;
            }

            // SAFETY: `OVERLAPPED` is a plain C POD struct initialized to all zeroes.
            let mut overlapped: OVERLAPPED = unsafe { std::mem::zeroed() };
            overlapped.Anonymous.Anonymous.Offset = op.file_offset as u32;
            overlapped.Anonymous.Anonymous.OffsetHigh = (op.file_offset >> 32) as u32;

            let ctx = Box::new(OverlappedReadContext {
                overlapped,
                op_id: op.op_id,
                _file: Arc::clone(&file),
            });
            let ctx_ptr = Box::into_raw(ctx);

            // SAFETY:
            // - `file.raw` is a valid overlapped handle associated with `self.iocp`.
            // - `op.dst_ptr` points to a uniquely leased pinned ring buffer of at least `op.len`
            //   bytes valid until the corresponding completion is reaped.
            // - `ctx_ptr` is heap-pinned and remains alive until reclaimed in `poll_completions`
            //   (or immediately below if `ReadFile` fails synchronously without queueing a packet).
            let ok = unsafe {
                ReadFile(
                    file.raw,
                    op.dst_ptr.as_ptr(),
                    op.len as u32,
                    std::ptr::null_mut(),
                    std::ptr::addr_of_mut!((*ctx_ptr).overlapped),
                )
            };

            if ok != 0 {
                // Synchronous completion on an IOCP-bound handle STILL posts an IOCP packet.
                submitted += 1;
                continue;
            }

            // SAFETY: Thread-local Win32 error retrieval immediately following `ReadFile == 0`.
            let err = unsafe { GetLastError() };
            if err == ERROR_IO_PENDING {
                submitted += 1;
                continue;
            }

            // Immediate error: Windows does NOT post an IOCP packet, so reclaim `ctx_ptr` now.
            // SAFETY: `ctx_ptr` was created by `Box::into_raw` above and was not queued to IOCP.
            let reclaimed = unsafe { Box::from_raw(ctx_ptr) };
            let os_error = if err == ERROR_HANDLE_EOF {
                None
            } else {
                Some(err as i32)
            };
            self.immediate_completions.push(AsyncReadCompletion {
                op_id: reclaimed.op_id,
                bytes_read: 0,
                os_error,
            });
            pushed_immediate = true;
            submitted += 1;
        }

        if pushed_immediate {
            self.wake_iocp();
        }

        Ok(submitted)
    }

    fn poll_completions(
        &self,
        out: &mut Vec<AsyncReadCompletion>,
        timeout: Duration,
    ) -> Result<usize, VelocityIoError> {
        use windows_sys::Win32::Foundation::{GetLastError, WAIT_TIMEOUT};
        use windows_sys::Win32::System::IO::{GetQueuedCompletionStatusEx, OVERLAPPED_ENTRY};

        let start_len = out.len();
        while let Some(c) = self.immediate_completions.pop() {
            out.push(c);
        }

        // If we already drained immediate completions, do a non-blocking IOCP harvest.
        let timeout_ms = if out.len() > start_len {
            0u32
        } else {
            timeout.as_millis().min(u32::MAX as u128 - 1) as u32
        };

        // SAFETY: `OVERLAPPED_ENTRY` is a plain C POD struct.
        let mut entries: [OVERLAPPED_ENTRY; 64] = unsafe { std::mem::zeroed() };
        let mut removed: u32 = 0;

        // SAFETY: `self.iocp` is a valid IOCP handle; `entries` and `removed` are valid stack targets.
        let ok = unsafe {
            GetQueuedCompletionStatusEx(
                self.iocp,
                entries.as_mut_ptr(),
                entries.len() as u32,
                &mut removed,
                timeout_ms,
                0,
            )
        };

        if ok == 0 {
            // SAFETY: Thread-local Win32 error retrieval immediately following failed wait.
            let err = unsafe { GetLastError() };
            if err == WAIT_TIMEOUT {
                return Ok(out.len() - start_len);
            }
            return Err(VelocityIoError::IoFailure {
                os_code: Some(err as i32),
                message: format!("GetQueuedCompletionStatusEx failed (os={err})"),
            });
        }

        for entry in &entries[..removed as usize] {
            if entry.lpOverlapped.is_null() || entry.lpCompletionKey == Self::WAKE_KEY {
                continue;
            }
            // SAFETY: Every non-null `lpOverlapped` submitted to `self.iocp` is the first field
            // of a heap-allocated `OverlappedReadContext` transferred via `Box::into_raw`.
            let ctx = unsafe { Box::from_raw(entry.lpOverlapped.cast::<OverlappedReadContext>()) };
            let status = ctx.overlapped.Internal as u32;
            let (bytes_read, os_error) = if status == 0 {
                (entry.dwNumberOfBytesTransferred as usize, None)
            } else if status == Self::STATUS_END_OF_FILE {
                (0, None)
            } else {
                (0, Some(status as i32))
            };
            out.push(AsyncReadCompletion {
                op_id: ctx.op_id,
                bytes_read,
                os_error,
            });
        }

        Ok(out.len() - start_len)
    }
}

/// Thin audited Linux `io_uring` async file I/O backend implementing [`AsyncIoBackend`].
#[cfg(target_os = "linux")]
pub struct IoUringBackend {
    ring: Mutex<io_uring::IoUring>,
    next_file_id: AtomicU64,
    files: RwLock<HashMap<FileHandleId, Arc<std::fs::File>>>,
    inflight_files: Mutex<HashMap<u64, Arc<std::fs::File>>>,
}

#[cfg(target_os = "linux")]
impl IoUringBackend {
    /// Creates a new `io_uring` async file I/O backend with `entries` SQ/CQ slots.
    pub fn new(entries: u32) -> Result<Self, VelocityIoError> {
        let ring = io_uring::IoUring::new(entries.max(64)).map_err(|e| VelocityIoError::IoFailure {
            os_code: e.raw_os_error(),
            message: format!("io_uring initialization failed: {e}"),
        })?;
        Ok(Self {
            ring: Mutex::new(ring),
            next_file_id: AtomicU64::new(1),
            files: RwLock::new(HashMap::new()),
            inflight_files: Mutex::new(HashMap::new()),
        })
    }
}

#[cfg(target_os = "linux")]
impl AsyncIoBackend for IoUringBackend {
    fn open_read_only(&self, path: &Path) -> Result<FileHandleId, VelocityIoError> {
        let file = std::fs::File::open(path).map_err(|e| VelocityIoError::IoFailure {
            os_code: e.raw_os_error(),
            message: format!("failed to open '{}': {e}", path.display()),
        })?;
        let id = FileHandleId(self.next_file_id.fetch_add(1, Ordering::Relaxed));
        self.files
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, Arc::new(file));
        Ok(id)
    }

    fn close_file(&self, file: FileHandleId) {
        self.files
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&file);
    }

    fn submit_read_batch(&self, ops: &[AsyncReadOp]) -> Result<usize, VelocityIoError> {
        use std::os::fd::AsRawFd;

        if ops.is_empty() {
            return Ok(0);
        }

        let resolved: Vec<Arc<std::fs::File>> = {
            let guard = self.files.read().unwrap_or_else(|e| e.into_inner());
            let mut handles = Vec::with_capacity(ops.len());
            for op in ops {
                let f = guard
                    .get(&op.file)
                    .cloned()
                    .ok_or(VelocityIoError::InvalidRequest("unknown file handle"))?;
                handles.push(f);
            }
            handles
        };

        let mut ring = self.ring.lock().unwrap_or_else(|e| e.into_inner());
        let mut inflight = self
            .inflight_files
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut submitted = 0usize;

        for (op, file) in ops.iter().zip(resolved) {
            let sqe = io_uring::opcode::Read::new(
                io_uring::types::Fd(file.as_raw_fd()),
                op.dst_ptr.as_ptr(),
                op.len as u32,
            )
            .offset(op.file_offset)
            .build()
            .user_data(op.op_id);

            inflight.insert(op.op_id, file);
            // SAFETY: `op.dst_ptr` points to a uniquely leased pinned buffer valid until CQE
            // reaping, and `inflight` keeps the file descriptor alive until the CQE is reaped.
            let push_res = unsafe { ring.submission().push(&sqe) };
            if push_res.is_err() {
                inflight.remove(&op.op_id);
                break;
            }
            submitted += 1;
        }

        if submitted > 0 {
            ring.submit().map_err(|e| VelocityIoError::IoFailure {
                os_code: e.raw_os_error(),
                message: format!("io_uring submit failed: {e}"),
            })?;
        }
        Ok(submitted)
    }

    fn poll_completions(
        &self,
        out: &mut Vec<AsyncReadCompletion>,
        timeout: Duration,
    ) -> Result<usize, VelocityIoError> {
        let mut ring = self.ring.lock().unwrap_or_else(|e| e.into_inner());
        let start_len = out.len();

        if ring.completion().is_empty() && !timeout.is_zero() {
            let _ = ring.submit_and_wait(1);
        }

        let mut inflight = self
            .inflight_files
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let cq = ring.completion();
        for cqe in cq {
            let op_id = cqe.user_data();
            inflight.remove(&op_id);
            let res = cqe.result();
            if res >= 0 {
                out.push(AsyncReadCompletion {
                    op_id,
                    bytes_read: res as usize,
                    os_error: None,
                });
            } else {
                out.push(AsyncReadCompletion {
                    op_id,
                    bytes_read: 0,
                    os_error: Some(-res),
                });
            }
        }
        Ok(out.len() - start_len)
    }
}

/// Deterministic cross-platform in-memory [`AsyncIoBackend`] for unit tests,
/// acceptance suites, and Criterion micro-benchmarks.
pub struct InMemoryAsyncIo {
    next_file_id: AtomicU64,
    reverse_completions: AtomicBool,
    paths: RwLock<HashMap<PathBuf, Arc<Vec<u8>>>>,
    open_files: RwLock<HashMap<FileHandleId, Arc<Vec<u8>>>>,
    injected_errors: RwLock<HashMap<FileHandleId, i32>>,
    completions: Mutex<VecDeque<AsyncReadCompletion>>,
    ready_cv: Condvar,
}

impl std::fmt::Debug for InMemoryAsyncIo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemoryAsyncIo")
            .field(
                "reverse_completions",
                &self.reverse_completions.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl Default for InMemoryAsyncIo {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryAsyncIo {
    /// Creates an empty in-memory async I/O backend.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_file_id: AtomicU64::new(1),
            reverse_completions: AtomicBool::new(false),
            paths: RwLock::new(HashMap::new()),
            open_files: RwLock::new(HashMap::new()),
            injected_errors: RwLock::new(HashMap::new()),
            completions: Mutex::new(VecDeque::new()),
            ready_cv: Condvar::new(),
        }
    }

    /// Registers an in-memory file at `path` so [`AsyncIoBackend::open_read_only`] can open it.
    pub fn add_file(&self, path: impl Into<PathBuf>, bytes: Vec<u8>) {
        self.paths
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(path.into(), Arc::new(bytes));
    }

    /// Registers and immediately opens an in-memory file, returning its [`FileHandleId`].
    pub fn register_open_file(&self, path: impl Into<PathBuf>, bytes: Vec<u8>) -> FileHandleId {
        let data = Arc::new(bytes);
        self.paths
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(path.into(), Arc::clone(&data));
        let id = FileHandleId(self.next_file_id.fetch_add(1, Ordering::Relaxed));
        self.open_files
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, data);
        id
    }

    /// When enabled, reverses completion order within each submitted batch to deterministically
    /// stress out-of-order completion handling.
    pub fn set_reverse_completion_order(&self, reverse: bool) {
        self.reverse_completions.store(reverse, Ordering::Release);
    }

    /// Injects an OS error code for every read issued against `file`.
    pub fn inject_file_error(&self, file: FileHandleId, os_error: i32) {
        self.injected_errors
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(file, os_error);
    }
}

impl AsyncIoBackend for InMemoryAsyncIo {
    fn open_read_only(&self, path: &Path) -> Result<FileHandleId, VelocityIoError> {
        let data = self
            .paths
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(path)
            .cloned()
            .ok_or_else(|| VelocityIoError::IoFailure {
                os_code: Some(2),
                message: format!("in-memory file not found: '{}'", path.display()),
            })?;
        let id = FileHandleId(self.next_file_id.fetch_add(1, Ordering::Relaxed));
        self.open_files
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, data);
        Ok(id)
    }

    fn close_file(&self, file: FileHandleId) {
        self.open_files
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&file);
    }

    fn submit_read_batch(&self, ops: &[AsyncReadOp]) -> Result<usize, VelocityIoError> {
        if ops.is_empty() {
            return Ok(0);
        }

        let files = self.open_files.read().unwrap_or_else(|e| e.into_inner());
        let errors = self.injected_errors.read().unwrap_or_else(|e| e.into_inner());
        let mut batch_completions = Vec::with_capacity(ops.len());

        for op in ops {
            if let Some(&err_code) = errors.get(&op.file) {
                batch_completions.push(AsyncReadCompletion {
                    op_id: op.op_id,
                    bytes_read: 0,
                    os_error: Some(err_code),
                });
                continue;
            }
            let data = files
                .get(&op.file)
                .ok_or(VelocityIoError::InvalidRequest("unknown file handle"))?;
            let start = (op.file_offset as usize).min(data.len());
            let avail = data.len() - start;
            let to_copy = op.len.min(avail);
            if to_copy > 0 {
                // SAFETY: `op.dst_ptr` points to a uniquely leased pinned buffer of at least `op.len` bytes.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr().add(start),
                        op.dst_ptr.as_ptr(),
                        to_copy,
                    );
                }
            }
            batch_completions.push(AsyncReadCompletion {
                op_id: op.op_id,
                bytes_read: to_copy,
                os_error: None,
            });
        }
        drop(files);
        drop(errors);

        if self.reverse_completions.load(Ordering::Acquire) {
            batch_completions.reverse();
        }

        let count = batch_completions.len();
        {
            let mut q = self.completions.lock().unwrap_or_else(|e| e.into_inner());
            q.extend(batch_completions);
        }
        self.ready_cv.notify_all();
        Ok(count)
    }

    fn poll_completions(
        &self,
        out: &mut Vec<AsyncReadCompletion>,
        timeout: Duration,
    ) -> Result<usize, VelocityIoError> {
        let mut q = self.completions.lock().unwrap_or_else(|e| e.into_inner());
        if q.is_empty() && !timeout.is_zero() {
            let (guard, _) = self
                .ready_cv
                .wait_timeout(q, timeout)
                .unwrap_or_else(|e| e.into_inner());
            q = guard;
        }
        let count = q.len();
        out.extend(q.drain(..));
        Ok(count)
    }
}

// ---------------------------------------------------------------------------
// Safe Controllers: VelocityStatusArray, VelocityFile, VelocityQueue, VelocityFactory
// ---------------------------------------------------------------------------

/// Safe Rust controller backing `IDStorageStatusArrayX`.
///
/// Entries start at `S_OK`, transition to `E_PENDING` on `enqueue_status`, and
/// atomically transition to `S_OK` or the first `HRESULT` error in enqueue order
/// strictly after all prior enqueued requests complete.
pub struct VelocityStatusArray {
    entries: Box<[AtomicI32]>,
}

impl std::fmt::Debug for VelocityStatusArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VelocityStatusArray")
            .field("capacity", &self.entries.len())
            .finish()
    }
}

impl VelocityStatusArray {
    /// Creates a new status array with `capacity` entries initialized to `S_OK`.
    #[must_use]
    pub fn new(capacity: u32) -> Arc<Self> {
        let entries = (0..capacity)
            .map(|_| AtomicI32::new(S_OK))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Arc::new(Self { entries })
    }

    /// Returns the number of entries in this status array.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> u32 {
        self.entries.len() as u32
    }

    /// Returns the `HRESULT` at `index` (`E_PENDING` while in flight, `S_OK` on success,
    /// or the first failure `HRESULT`; `E_INVALIDARG` if `index` is out of range).
    #[inline]
    #[must_use]
    pub fn hresult(&self, index: u32) -> i32 {
        self.entries
            .get(index as usize)
            .map_or(E_INVALIDARG, |entry| entry.load(Ordering::Acquire))
    }

    /// Returns `true` iff entry `index` is no longer `E_PENDING`.
    #[inline]
    #[must_use]
    pub fn is_complete(&self, index: u32) -> bool {
        self.hresult(index) != E_PENDING
    }

    #[inline]
    fn mark_pending(&self, index: u32) -> Result<(), VelocityIoError> {
        let entry = self
            .entries
            .get(index as usize)
            .ok_or(VelocityIoError::InvalidRequest("status array index out of range"))?;
        entry.store(E_PENDING, Ordering::Release);
        Ok(())
    }

    #[inline]
    fn complete(&self, index: u32, hresult: i32) {
        if let Some(entry) = self.entries.get(index as usize) {
            entry.store(hresult, Ordering::Release);
        }
    }
}

/// RAII file handle opened through [`VelocityFactory`], closing the underlying OS
/// handle in [`AsyncIoBackend`] when dropped.
pub struct VelocityFile {
    id: FileHandleId,
    io: Arc<dyn AsyncIoBackend>,
}

impl std::fmt::Debug for VelocityFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VelocityFile").field("id", &self.id).finish()
    }
}

impl VelocityFile {
    /// Returns the opaque [`FileHandleId`] used inside [`DStorageSource::File`].
    #[inline]
    #[must_use]
    pub fn handle_id(&self) -> FileHandleId {
        self.id
    }
}

impl Drop for VelocityFile {
    fn drop(&mut self) {
        self.io.close_file(self.id);
    }
}

/// Configuration for creating a [`VelocityQueue`].
#[derive(Debug, Clone)]
pub struct VelocityQueueConfig {
    pub capacity: usize,
    pub priority: DStoragePriority,
    pub staging_ring_bytes: usize,
    pub decompress_workers: usize,
    pub name: String,
}

impl Default for VelocityQueueConfig {
    fn default() -> Self {
        Self {
            capacity: 4096,
            priority: DStoragePriority::Normal,
            staging_ring_bytes: 32 * 1024 * 1024,
            decompress_workers: 2,
            name: String::from("velocity-queue"),
        }
    }
}

enum QueuedCommand {
    Request(DStorageRequestDesc),
    Status {
        array: Arc<VelocityStatusArray>,
        index: u32,
    },
}

struct SealedBatch {
    base_seq: u64,
    commands: Vec<QueuedCommand>,
}

struct InFlightRead {
    seq: u64,
    desc: DStorageRequestDesc,
    lease: PinnedLease,
}

enum DecompressPayload {
    Pinned {
        lease: PinnedLease,
        valid_len: usize,
    },
    MemorySlice(Vec<u8>),
}

struct DecompressWorkItem {
    seq: u64,
    desc: DStorageRequestDesc,
    payload: DecompressPayload,
}

/// Tracks out-of-order request completions and commits [`VelocityStatusArray`]
/// updates strictly in enqueue sequence order with first-failure `HRESULT` semantics.
struct InOrderCommitter {
    next_commit_seq: u64,
    first_error_since_status: Option<i32>,
    completed_requests: BTreeMap<u64, i32>,
    pending_statuses: BTreeMap<u64, (Arc<VelocityStatusArray>, u32)>,
}

impl InOrderCommitter {
    fn new() -> Self {
        Self {
            next_commit_seq: 0,
            first_error_since_status: None,
            completed_requests: BTreeMap::new(),
            pending_statuses: BTreeMap::new(),
        }
    }

    fn register_status(&mut self, seq: u64, array: Arc<VelocityStatusArray>, index: u32) {
        self.pending_statuses.insert(seq, (array, index));
        self.advance();
    }

    fn complete_request(&mut self, seq: u64, hresult: i32) {
        self.completed_requests.insert(seq, hresult);
        self.advance();
    }

    fn advance(&mut self) {
        loop {
            if let Some(hr) = self.completed_requests.remove(&self.next_commit_seq) {
                if hr != S_OK && self.first_error_since_status.is_none() {
                    self.first_error_since_status = Some(hr);
                }
                self.next_commit_seq += 1;
                continue;
            }
            if let Some((array, index)) = self.pending_statuses.remove(&self.next_commit_seq) {
                let status_hr = self.first_error_since_status.take().unwrap_or(S_OK);
                array.complete(index, status_hr);
                self.next_commit_seq += 1;
                continue;
            }
            break;
        }
    }

    fn next_seq(&self) -> u64 {
        self.next_commit_seq
    }
}

/// Per-queue completion inbox & wakeup primitive used when multiple [`VelocityQueue`]s
/// share a single [`VelocityFactory`] (and therefore a single OS IOCP / `io_uring` backend).
///
/// Holds no back-reference to [`IoCompletionDispatcher`] so `Arc` cycles cannot occur.
struct QueueInbox {
    completions: SegQueue<AsyncReadCompletion>,
    wake_lock: Mutex<bool>,
    wake_cv: Condvar,
}

impl QueueInbox {
    fn new() -> Self {
        Self {
            completions: SegQueue::new(),
            wake_lock: Mutex::new(false),
            wake_cv: Condvar::new(),
        }
    }

    fn push(&self, completion: AsyncReadCompletion) {
        self.completions.push(completion);
        self.wake();
    }

    fn wake(&self) {
        let mut flag = self.wake_lock.lock().unwrap_or_else(|e| e.into_inner());
        *flag = true;
        self.wake_cv.notify_one();
    }
}

struct DispatcherShared {
    io_backend: Arc<dyn AsyncIoBackend>,
    next_op_id: AtomicU64,
    shutdown: AtomicBool,
    routes: Mutex<HashMap<u64, Arc<QueueInbox>>>,
    active_cv: Condvar,
}

/// Factory-wide single-reaper async I/O completion router.
///
/// Prevents completion stealing and `op_id` collisions when a title creates many
/// [`VelocityQueue`] instances (e.g., 17 `fiDeviceDirectStorage` queues) sharing one
/// `Win32OverlappedIo` IOCP handle or Linux `io_uring`.
struct IoCompletionDispatcher {
    shared: Arc<DispatcherShared>,
    reaper: Mutex<Option<JoinHandle<()>>>,
}

impl IoCompletionDispatcher {
    fn new(io_backend: Arc<dyn AsyncIoBackend>) -> Result<Arc<Self>, VelocityIoError> {
        let shared = Arc::new(DispatcherShared {
            io_backend,
            next_op_id: AtomicU64::new(1),
            shutdown: AtomicBool::new(false),
            routes: Mutex::new(HashMap::new()),
            active_cv: Condvar::new(),
        });
        let thread_shared = Arc::clone(&shared);
        let handle = thread::Builder::new()
            .name(String::from("velocity-io-reaper"))
            .spawn(move || dispatcher_reaper_loop(thread_shared))
            .map_err(|e| VelocityIoError::IoFailure {
                os_code: e.raw_os_error(),
                message: format!("failed to spawn velocity-io reaper thread: {e}"),
            })?;
        Ok(Arc::new(Self {
            shared,
            reaper: Mutex::new(Some(handle)),
        }))
    }

    #[inline]
    fn alloc_op_id(&self) -> u64 {
        self.shared.next_op_id.fetch_add(1, Ordering::Relaxed)
    }

    fn submit_routed_batch(
        &self,
        inbox: &Arc<QueueInbox>,
        ops: &[AsyncReadOp],
    ) -> Result<usize, VelocityIoError> {
        if ops.is_empty() {
            return Ok(0);
        }
        // Register routes BEFORE issuing OS async I/O so an immediate completion on
        // another core always finds its owning queue's inbox.
        {
            let mut routes = self.shared.routes.lock().unwrap_or_else(|e| e.into_inner());
            for op in ops {
                routes.insert(op.op_id, Arc::clone(inbox));
            }
            self.shared.active_cv.notify_one();
        }
        match self.shared.io_backend.submit_read_batch(ops) {
            Ok(submitted) => {
                if submitted < ops.len() {
                    let mut routes = self.shared.routes.lock().unwrap_or_else(|e| e.into_inner());
                    for op in &ops[submitted..] {
                        routes.remove(&op.op_id);
                    }
                }
                Ok(submitted)
            }
            Err(err) => {
                let mut routes = self.shared.routes.lock().unwrap_or_else(|e| e.into_inner());
                for op in ops {
                    routes.remove(&op.op_id);
                }
                Err(err)
            }
        }
    }
}

impl Drop for IoCompletionDispatcher {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        {
            let _routes = self.shared.routes.lock().unwrap_or_else(|e| e.into_inner());
            self.shared.active_cv.notify_all();
        }
        if let Some(handle) = self
            .reaper
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            let _ = handle.join();
        }
    }
}

fn dispatcher_reaper_loop(shared: Arc<DispatcherShared>) {
    let mut batch: Vec<AsyncReadCompletion> = Vec::with_capacity(64);
    loop {
        {
            let mut routes = shared.routes.lock().unwrap_or_else(|e| e.into_inner());
            while routes.is_empty() && !shared.shutdown.load(Ordering::Acquire) {
                routes = shared
                    .active_cv
                    .wait(routes)
                    .unwrap_or_else(|e| e.into_inner());
            }
            if routes.is_empty() && shared.shutdown.load(Ordering::Acquire) {
                break;
            }
        }

        batch.clear();
        match shared
            .io_backend
            .poll_completions(&mut batch, Duration::from_millis(2))
        {
            Ok(n) if n > 0 => {
                let mut routes = shared.routes.lock().unwrap_or_else(|e| e.into_inner());
                for completion in batch.drain(..) {
                    if let Some(inbox) = routes.remove(&completion.op_id) {
                        inbox.push(completion);
                    }
                }
            }
            Ok(_) => {}
            Err(err) => {
                tracing::warn!(%err, "velocity_io: async I/O reaper poll_completions failed");
                thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

struct QueueShared {
    priority: DStoragePriority,
    capacity: usize,
    closed: AtomicBool,
    io_pump_done: AtomicBool,
    unsubmitted: ArrayQueue<QueuedCommand>,
    submit_seal_lock: Mutex<()>,
    next_submit_seq: AtomicU64,
    submitted_batches: SegQueue<SealedBatch>,
    inbox: Arc<QueueInbox>,
    dispatcher: Option<Arc<IoCompletionDispatcher>>,
    decompress_queue: SegQueue<DecompressWorkItem>,
    decompress_wake_lock: Mutex<usize>,
    decompress_wake_cv: Condvar,
    committer: Mutex<InOrderCommitter>,
    idle_cv: Condvar,
    io_backend: Arc<dyn AsyncIoBackend>,
    gpu_decompress: Arc<GpuDecompressQueue>,
    hooks: Arc<dyn DStoragePlatformHooks>,
    ring: Arc<PinnedUploadRing>,
    next_op_id: AtomicU64,
}

impl QueueShared {
    fn wake_io_pump(&self) {
        self.inbox.wake();
    }

    fn push_decompress_work(&self, item: DecompressWorkItem) {
        self.decompress_queue.push(item);
        let mut count = self
            .decompress_wake_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *count += 1;
        self.decompress_wake_cv.notify_one();
    }

    fn report_request_done(&self, seq: u64, hresult: i32) {
        let mut committer = self.committer.lock().unwrap_or_else(|e| e.into_inner());
        committer.complete_request(seq, hresult);
        let committed = committer.next_seq();
        let submitted = self.next_submit_seq.load(Ordering::Acquire);
        if committed >= submitted {
            self.idle_cv.notify_all();
        }
    }

    fn register_status_seq(&self, seq: u64, array: Arc<VelocityStatusArray>, index: u32) {
        let mut committer = self.committer.lock().unwrap_or_else(|e| e.into_inner());
        committer.register_status(seq, array, index);
        let committed = committer.next_seq();
        let submitted = self.next_submit_seq.load(Ordering::Acquire);
        if committed >= submitted {
            self.idle_cv.notify_all();
        }
    }
}

/// Safe, zero-stall controller backing `IDStorageQueueX`.
///
/// - [`Self::enqueue_request`] and [`Self::enqueue_status`] push into a lock-free
///   bounded [`ArrayQueue`] and **never** perform disk I/O, decompression, or
///   premature auto-submission before explicit [`Self::submit`].
/// - [`Self::submit`] atomically seals the staged batch and hands it to dedicated
///   background async I/O and decompression worker threads.
pub struct VelocityQueue {
    shared: Arc<QueueShared>,
    io_pump: Mutex<Option<JoinHandle<()>>>,
    decompress_workers: Mutex<Vec<JoinHandle<()>>>,
}

impl std::fmt::Debug for VelocityQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VelocityQueue")
            .field("priority", &self.shared.priority)
            .field("capacity", &self.shared.capacity)
            .field("unsubmitted", &self.shared.unsubmitted.len())
            .field("closed", &self.shared.closed.load(Ordering::Relaxed))
            .finish()
    }
}

impl VelocityQueue {
    /// Creates and starts a new [`VelocityQueue`] with dedicated async I/O pump and
    /// decompression worker threads.
    pub fn new(
        config: VelocityQueueConfig,
        io_backend: Arc<dyn AsyncIoBackend>,
        gpu_decompress: Arc<GpuDecompressQueue>,
        hooks: Arc<dyn DStoragePlatformHooks>,
        ring: Arc<PinnedUploadRing>,
    ) -> Result<Arc<Self>, VelocityIoError> {
        Self::new_internal(config, io_backend, None, gpu_decompress, hooks, ring)
    }

    fn new_internal(
        config: VelocityQueueConfig,
        io_backend: Arc<dyn AsyncIoBackend>,
        dispatcher: Option<Arc<IoCompletionDispatcher>>,
        gpu_decompress: Arc<GpuDecompressQueue>,
        hooks: Arc<dyn DStoragePlatformHooks>,
        ring: Arc<PinnedUploadRing>,
    ) -> Result<Arc<Self>, VelocityIoError> {
        let capacity = config.capacity.max(2);
        let worker_count = config.decompress_workers.clamp(1, 16);

        let shared = Arc::new(QueueShared {
            priority: config.priority,
            capacity,
            closed: AtomicBool::new(false),
            io_pump_done: AtomicBool::new(false),
            unsubmitted: ArrayQueue::new(capacity),
            submit_seal_lock: Mutex::new(()),
            next_submit_seq: AtomicU64::new(0),
            submitted_batches: SegQueue::new(),
            inbox: Arc::new(QueueInbox::new()),
            dispatcher,
            decompress_queue: SegQueue::new(),
            decompress_wake_lock: Mutex::new(0),
            decompress_wake_cv: Condvar::new(),
            committer: Mutex::new(InOrderCommitter::new()),
            idle_cv: Condvar::new(),
            io_backend,
            gpu_decompress,
            hooks,
            ring,
            next_op_id: AtomicU64::new(1),
        });

        let mut decompress_handles = Vec::with_capacity(worker_count);

        let pump_shared = Arc::clone(&shared);
        let pump_handle = thread::Builder::new()
            .name(format!("{}-io", config.name))
            .spawn(move || io_pump_loop(pump_shared))
            .map_err(|e| VelocityIoError::IoFailure {
                os_code: e.raw_os_error(),
                message: format!("failed to spawn velocity-io pump thread: {e}"),
            })?;

        for idx in 0..worker_count {
            let worker_shared = Arc::clone(&shared);
            let handle = thread::Builder::new()
                .name(format!("{}-decomp-{idx}", config.name))
                .spawn(move || decompress_worker_loop(worker_shared))
                .map_err(|e| VelocityIoError::IoFailure {
                    os_code: e.raw_os_error(),
                    message: format!("failed to spawn velocity-io decompress worker: {e}"),
                })?;
            decompress_handles.push(handle);
        }

        Ok(Arc::new(Self {
            shared,
            io_pump: Mutex::new(Some(pump_handle)),
            decompress_workers: Mutex::new(decompress_handles),
        }))
    }

    /// Returns this queue's priority level.
    #[inline]
    #[must_use]
    pub fn priority(&self) -> DStoragePriority {
        self.shared.priority
    }

    /// Returns the number of currently staged (unsubmitted) commands.
    #[inline]
    #[must_use]
    pub fn unsubmitted_len(&self) -> usize {
        self.shared.unsubmitted.len()
    }

    /// Returns the underlying pinned upload ring.
    #[inline]
    #[must_use]
    pub fn upload_ring(&self) -> &Arc<PinnedUploadRing> {
        &self.shared.ring
    }

    /// Stages a DirectStorage request into the lock-free unsubmitted buffer.
    ///
    /// Never blocks on I/O or decompression, and **never** submits before [`Self::submit`].
    pub fn enqueue_request(&self, desc: DStorageRequestDesc) -> Result<(), VelocityIoError> {
        if self.shared.closed.load(Ordering::Acquire) {
            return Err(VelocityIoError::QueueClosed);
        }
        self.shared
            .unsubmitted
            .push(QueuedCommand::Request(desc))
            .map_err(|_| VelocityIoError::StagingRingFull { requested: 1 })
    }

    /// Stages a status array barrier into the lock-free unsubmitted buffer, immediately
    /// marking `status_array[index]` as `E_PENDING`.
    ///
    /// Never blocks on I/O or decompression, and **never** submits before [`Self::submit`].
    pub fn enqueue_status(
        &self,
        status_array: Arc<VelocityStatusArray>,
        index: u32,
    ) -> Result<(), VelocityIoError> {
        if self.shared.closed.load(Ordering::Acquire) {
            return Err(VelocityIoError::QueueClosed);
        }
        status_array.mark_pending(index)?;
        self.shared
            .unsubmitted
            .push(QueuedCommand::Status {
                array: status_array,
                index,
            })
            .map_err(|_| VelocityIoError::StagingRingFull { requested: 1 })
    }

    /// Atomically seals all currently staged commands into a submitted batch and hands
    /// them off to the async I/O pump without blocking on I/O or decompression.
    ///
    /// Returns the number of commands sealed in this batch.
    pub fn submit(&self) -> Result<usize, VelocityIoError> {
        if self.shared.closed.load(Ordering::Acquire) {
            return Err(VelocityIoError::QueueClosed);
        }

        let _seal_guard = self
            .shared
            .submit_seal_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut commands = Vec::with_capacity(self.shared.unsubmitted.len());
        while let Some(cmd) = self.shared.unsubmitted.pop() {
            commands.push(cmd);
        }
        let count = commands.len();
        if count == 0 {
            return Ok(0);
        }

        let base_seq = self
            .shared
            .next_submit_seq
            .fetch_add(count as u64, Ordering::AcqRel);
        self.shared
            .submitted_batches
            .push(SealedBatch { base_seq, commands });
        drop(_seal_guard);

        self.shared.wake_io_pump();
        Ok(count)
    }

    /// Waits until all previously submitted commands have completed or `timeout` elapses.
    ///
    /// Intended for tests, shutdown draining, and synchronization barriers (never called
    /// from the render-thread enqueue path).
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        let target = self.shared.next_submit_seq.load(Ordering::Acquire);
        let deadline = Instant::now() + timeout;
        let mut committer = self
            .shared
            .committer
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        while committer.next_seq() < target {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            let (guard, wait_res) = self
                .shared
                .idle_cv
                .wait_timeout(committer, remaining)
                .unwrap_or_else(|e| e.into_inner());
            committer = guard;
            if wait_res.timed_out() && committer.next_seq() < target {
                return false;
            }
        }
        true
    }

    /// Closes the queue, discards unsubmitted commands, wakes all background workers,
    /// and executes two-phase shutdown (joining the I/O pump before decompression workers
    /// so in-flight pinned ring leases always drain without deadlock).
    pub fn close(&self) {
        if self.shared.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        while self.shared.unsubmitted.pop().is_some() {}
        self.shared.wake_io_pump();

        // Phase 1: wait for the I/O pump to reap all in-flight OS async reads while
        // decompression workers are still alive to retire ring leases.
        let pump = self
            .io_pump
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(handle) = pump {
            let _ = handle.join();
        }

        // Phase 2: signal `io_pump_done`, wake all decompression workers, and join them.
        self.shared.io_pump_done.store(true, Ordering::Release);
        {
            let mut count = self
                .shared
                .decompress_wake_lock
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            *count += 1;
            self.shared.decompress_wake_cv.notify_all();
        }
        let handles: Vec<JoinHandle<()>> = {
            let mut guard = self
                .decompress_workers
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *guard)
        };
        for handle in handles {
            let _ = handle.join();
        }
    }
}

impl Drop for VelocityQueue {
    fn drop(&mut self) {
        self.close();
    }
}

/// Dedicated background I/O pump thread: leases pinned upload ring slices, submits
/// batched async OS reads, and hands completed reads to decompression workers.
fn io_pump_loop(shared: Arc<QueueShared>) {
    struct PumpExitGuard(Arc<QueueShared>);
    impl Drop for PumpExitGuard {
        fn drop(&mut self) {
            self.0.io_pump_done.store(true, Ordering::Release);
            let mut count = self
                .0
                .decompress_wake_lock
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            *count += 1;
            self.0.decompress_wake_cv.notify_all();
        }
    }
    let _exit_guard = PumpExitGuard(Arc::clone(&shared));

    let mut deferred_reads: VecDeque<(u64, DStorageRequestDesc)> = VecDeque::new();
    let mut inflight: HashMap<u64, InFlightRead> = HashMap::new();
    let mut completions: Vec<AsyncReadCompletion> = Vec::with_capacity(64);
    let mut submit_ops: Vec<AsyncReadOp> = Vec::with_capacity(64);

    loop {
        let is_closed = shared.closed.load(Ordering::Acquire);

        // 1. Drain newly submitted sealed batches (or discard unstarted work if closed).
        if is_closed {
            while shared.submitted_batches.pop().is_some() {}
            deferred_reads.clear();
        } else {
            while let Some(batch) = shared.submitted_batches.pop() {
                for (idx, cmd) in batch.commands.into_iter().enumerate() {
                    let seq = batch.base_seq + idx as u64;
                    match cmd {
                        QueuedCommand::Status { array, index } => {
                            shared.register_status_seq(seq, array, index);
                        }
                        QueuedCommand::Request(desc) => match desc.source {
                            DStorageSource::Memory { address, size } => {
                                let bytes = if size == 0 {
                                    Vec::new()
                                } else {
                                    // SAFETY: Caller guarantees linear memory source `[address, address + size)`
                                    // remains valid and readable while the submitted request is in flight.
                                    unsafe {
                                        std::slice::from_raw_parts(
                                            address as *const u8,
                                            size as usize,
                                        )
                                        .to_vec()
                                    }
                                };
                                shared.push_decompress_work(DecompressWorkItem {
                                    seq,
                                    desc,
                                    payload: DecompressPayload::MemorySlice(bytes),
                                });
                            }
                            DStorageSource::File { .. } => {
                                deferred_reads.push_back((seq, desc));
                            }
                        },
                    }
                }
            }
        }

        // 2. Reclaim completed pinned ring slices before leasing new slots.
        shared
            .ring
            .reclaim_completed(shared.gpu_decompress.completed_fence_value());

        // 3. Lease ring slices & build a batch of AsyncReadOps without blocking.
        submit_ops.clear();
        let mut staged_meta: Vec<(u64, InFlightRead)> = Vec::new();
        while let Some((seq, desc)) = deferred_reads.pop_front() {
            let DStorageSource::File { file, offset, size } = desc.source else {
                continue;
            };
            let len = size as usize;
            let lease_result = if shared
                .ring
                .capacity_bytes()
                .is_some_and(|cap| len > cap)
            {
                Ok(PinnedLease::new_overflow(len))
            } else {
                shared.ring.try_lease(len)
            };
            match lease_result {
                Ok(lease) => {
                    let op_id = match shared.dispatcher {
                        Some(ref disp) => disp.alloc_op_id(),
                        None => shared.next_op_id.fetch_add(1, Ordering::Relaxed),
                    };
                    submit_ops.push(AsyncReadOp {
                        op_id,
                        file,
                        file_offset: offset,
                        len,
                        dst_ptr: lease.raw_ptr(),
                    });
                    staged_meta.push((op_id, InFlightRead { seq, desc, lease }));
                    if submit_ops.len() >= 64 {
                        break;
                    }
                }
                Err(VelocityIoError::StagingRingFull { .. }) => {
                    deferred_reads.push_front((seq, desc));
                    break;
                }
                Err(err) => {
                    shared.report_request_done(seq, velocity_error_to_hresult(&err));
                }
            }
        }

        // 4. Submit batch to OS async I/O backend (via shared dispatcher when factory-owned).
        if !submit_ops.is_empty() {
            let submit_res = match shared.dispatcher {
                Some(ref disp) => disp.submit_routed_batch(&shared.inbox, &submit_ops),
                None => shared.io_backend.submit_read_batch(&submit_ops),
            };
            match submit_res {
                Ok(submitted) => {
                    let mut iter = staged_meta.into_iter();
                    for _ in 0..submitted {
                        if let Some((op_id, entry)) = iter.next() {
                            inflight.insert(op_id, entry);
                        }
                    }
                    // Any unsubmitted tail ops go back to the front of deferred_reads.
                    let leftover: Vec<_> = iter.collect();
                    for (_, entry) in leftover.into_iter().rev() {
                        shared
                            .ring
                            .retire_lease(entry.lease, GpuFenceToken::default());
                        deferred_reads.push_front((entry.seq, entry.desc));
                    }
                }
                Err(err) => {
                    let hr = velocity_error_to_hresult(&err);
                    for (_, entry) in staged_meta {
                        shared
                            .ring
                            .retire_lease(entry.lease, GpuFenceToken::default());
                        shared.report_request_done(entry.seq, hr);
                    }
                }
            }
        }

        // 5. Reap OS async I/O completions.
        if !inflight.is_empty() {
            completions.clear();
            let poll_timeout = if !deferred_reads.is_empty() {
                Duration::from_millis(1)
            } else {
                Duration::from_millis(2)
            };
            let got = if shared.dispatcher.is_some() {
                while let Some(c) = shared.inbox.completions.pop() {
                    completions.push(c);
                }
                if completions.is_empty()
                    && shared.submitted_batches.is_empty()
                    && !shared.closed.load(Ordering::Acquire)
                {
                    let mut wake = shared
                        .inbox
                        .wake_lock
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    if !*wake
                        && shared.inbox.completions.is_empty()
                        && shared.submitted_batches.is_empty()
                        && !shared.closed.load(Ordering::Acquire)
                    {
                        let (guard, _) = shared
                            .inbox
                            .wake_cv
                            .wait_timeout(wake, poll_timeout)
                            .unwrap_or_else(|e| e.into_inner());
                        wake = guard;
                    }
                    *wake = false;
                    drop(wake);
                    while let Some(c) = shared.inbox.completions.pop() {
                        completions.push(c);
                    }
                }
                completions.len()
            } else {
                shared
                    .io_backend
                    .poll_completions(&mut completions, poll_timeout)
                    .unwrap_or(0)
            };
            if got > 0 {
                for completion in completions.drain(..) {
                    let Some(entry) = inflight.remove(&completion.op_id) else {
                        continue;
                    };
                    let expected_size = match entry.desc.source {
                        DStorageSource::File { size, .. } => size as usize,
                        DStorageSource::Memory { size, .. } => size as usize,
                    };
                    if let Some(code) = completion.os_error {
                        let err = VelocityIoError::IoFailure {
                            os_code: Some(code),
                            message: format!("async read op {} failed", completion.op_id),
                        };
                        tracing::warn!(%err, seq = entry.seq, "velocity_io: async file read failed");
                        shared
                            .ring
                            .retire_lease(entry.lease, GpuFenceToken::default());
                        shared.report_request_done(entry.seq, velocity_error_to_hresult(&err));
                        continue;
                    }
                    if completion.bytes_read < expected_size {
                        let offset = match entry.desc.source {
                            DStorageSource::File { offset, .. } => offset,
                            DStorageSource::Memory { .. } => 0,
                        };
                        let err = VelocityIoError::EndOfFile {
                            offset,
                            needed: expected_size,
                            got: completion.bytes_read,
                        };
                        tracing::warn!(%err, seq = entry.seq, "velocity_io: unexpected EOF");
                        shared
                            .ring
                            .retire_lease(entry.lease, GpuFenceToken::default());
                        shared.report_request_done(entry.seq, velocity_error_to_hresult(&err));
                        continue;
                    }

                    shared.push_decompress_work(DecompressWorkItem {
                        seq: entry.seq,
                        desc: entry.desc,
                        payload: DecompressPayload::Pinned {
                            lease: entry.lease,
                            valid_len: completion.bytes_read,
                        },
                    });
                }
            }
            continue;
        }

        // 6. If deferred reads are waiting for decompression workers to retire ring leases,
        // briefly yield and retry reclamation without spinning hot.
        if !deferred_reads.is_empty() {
            let reclaimed = shared
                .ring
                .reclaim_completed(shared.gpu_decompress.completed_fence_value());
            if reclaimed == 0 {
                thread::sleep(Duration::from_micros(100));
            }
            continue;
        }

        // 7. Exit cleanly once closed and all in-flight OS reads have been reaped.
        if shared.closed.load(Ordering::Acquire) {
            break;
        }

        // 8. Park until a new batch is submitted or the queue is closed.
        let mut wake = shared
            .inbox
            .wake_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if !*wake
            && shared.submitted_batches.is_empty()
            && shared.inbox.completions.is_empty()
            && !shared.closed.load(Ordering::Acquire)
        {
            let (guard, _) = shared
                .inbox
                .wake_cv
                .wait_timeout(wake, Duration::from_millis(5))
                .unwrap_or_else(|e| e.into_inner());
            wake = guard;
        }
        *wake = false;
    }
}

/// Background decompression worker executing jobs through [`GpuDecompressQueue`],
/// delivering bytes to title destinations, retiring pinned ring leases, and reporting
/// completions to [`InOrderCommitter`].
fn decompress_worker_loop(shared: Arc<QueueShared>) {
    loop {
        if let Some(item) = shared.decompress_queue.pop() {
            let (result, retire_fence) = execute_decompress_item(&shared, item);
            let hr = match result {
                Ok(()) => S_OK,
                Err(ref err) => {
                    tracing::warn!(seq = retire_fence.0, %err, "velocity_io: DirectStorage request failed");
                    velocity_error_to_hresult(err)
                }
            };
            shared.report_request_done(retire_fence.0, hr);
            shared
                .ring
                .reclaim_completed(shared.gpu_decompress.completed_fence_value());
            continue;
        }

        if shared.io_pump_done.load(Ordering::Acquire) && shared.decompress_queue.is_empty() {
            break;
        }

        let mut wake = shared
            .decompress_wake_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if *wake == 0
            && shared.decompress_queue.is_empty()
            && !shared.io_pump_done.load(Ordering::Acquire)
        {
            let (guard, _) = shared
                .decompress_wake_cv
                .wait_timeout(wake, Duration::from_millis(5))
                .unwrap_or_else(|e| e.into_inner());
            wake = guard;
        }
        *wake = 0;
    }
}

fn execute_decompress_item(
    shared: &QueueShared,
    item: DecompressWorkItem,
) -> (Result<(), VelocityIoError>, (u64, GpuFenceToken)) {
    let seq = item.seq;
    let desc = item.desc;
    let mut pinned_lease: Option<PinnedLease> = None;

    let (src_slice_holder, src_gpu_va, src_gpu_resource): (
        Vec<u8>,
        Option<GpuVirtualAddress>,
        Option<(ResourceId, u64)>,
    ) = match item.payload {
        DecompressPayload::Pinned { lease, valid_len } => {
            let gpu_va = lease.gpu_va();
            let gpu_res = lease.gpu_resource();
            let data = lease.as_slice()[..valid_len].to_vec();
            pinned_lease = Some(lease);
            (data, gpu_va, gpu_res)
        }
        DecompressPayload::MemorySlice(bytes) => (bytes, None, None),
    };

    let dst_gpu_resource = match desc.destination {
        DStorageDestination::GpuBuffer {
            resource, offset, ..
        } => Some((resource, offset)),
        _ => None,
    };

    let final_expected_len = match desc.destination {
        DStorageDestination::Memory { size, .. }
        | DStorageDestination::PhysicalPages { size, .. }
        | DStorageDestination::GpuBuffer { size, .. } => size as usize,
    };

    let stage1_expected_len = if desc.secondary_codec.is_some() {
        desc.intermediate_size as usize
    } else {
        final_expected_len
    };

    let stage1_job = DecompressionJob {
        job_id: seq,
        codec: desc.codec,
        src_bytes: &src_slice_holder,
        src_gpu_va,
        src_gpu_resource: if desc.secondary_codec.is_none() {
            src_gpu_resource
        } else {
            None
        },
        expected_output_len: stage1_expected_len,
        dst_gpu_resource: if desc.secondary_codec.is_none() {
            dst_gpu_resource
        } else {
            None
        },
        wait_copy_fence: None,
    };

    let mut completion_fence = GpuFenceToken::default();
    let result = (|| -> Result<(), VelocityIoError> {
        let stage1_out = shared.gpu_decompress.dispatch(&stage1_job)?;
        if let Some(tok) = stage1_out.completion_fence {
            completion_fence = tok;
        }

        let final_out = if let Some(second_codec) = desc.secondary_codec {
            let intermediate = stage1_out.cpu_bytes.ok_or_else(|| {
                VelocityIoError::DecompressionFailed {
                    codec: desc.codec,
                    reason: "chained stage 1 did not produce CPU intermediate bytes".to_owned(),
                }
            })?;
            let stage2_job = DecompressionJob {
                job_id: seq,
                codec: second_codec,
                src_bytes: &intermediate,
                src_gpu_va: None,
                src_gpu_resource: None,
                expected_output_len: final_expected_len,
                dst_gpu_resource,
                wait_copy_fence: stage1_out.completion_fence,
            };
            let out2 = shared.gpu_decompress.dispatch(&stage2_job)?;
            if let Some(tok) = out2.completion_fence {
                completion_fence = tok;
            }
            out2
        } else {
            stage1_out
        };

        deliver_destination(shared.hooks.as_ref(), desc.destination, final_out.cpu_bytes)
    })();

    if let Some(lease) = pinned_lease {
        shared.ring.retire_lease(lease, completion_fence);
    }

    (result, (seq, completion_fence))
}

fn deliver_destination(
    hooks: &dyn DStoragePlatformHooks,
    destination: DStorageDestination,
    cpu_bytes: Option<Vec<u8>>,
) -> Result<(), VelocityIoError> {
    match destination {
        DStorageDestination::GpuBuffer { .. } => {
            // GPU buffer destinations are populated directly via ComputeQueueBackend.
            Ok(())
        }
        DStorageDestination::Memory { address, size } => {
            let bytes = cpu_bytes.unwrap_or_default();
            if bytes.len() > size as usize {
                return Err(VelocityIoError::InvalidRequest(
                    "decompressed payload exceeds destination memory size",
                ));
            }
            if !bytes.is_empty() {
                if address == 0 {
                    return Err(VelocityIoError::InvalidRequest(
                        "destination memory address is null",
                    ));
                }
                // Open any GPU-side read-only write-watch guards covering `[address, address + len)`
                // before writing, matching the legacy DirectStorage path in `dstorage.rs`.
                #[cfg(windows)]
                let writing_guard = crate::watch::kernel_writing(address, bytes.len());
                // SAFETY: Caller guarantees destination `[address, address + size)` is mapped
                // and exclusively writable for the submitted DirectStorage request until status completion.
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), address as *mut u8, bytes.len());
                }
                #[cfg(windows)]
                drop(writing_guard);
                hooks.notify_memory_written(address, bytes.len());
            }
            Ok(())
        }
        DStorageDestination::PhysicalPages {
            page_list_addr,
            page_offset,
            size,
        } => {
            let bytes = cpu_bytes.unwrap_or_default();
            if bytes.len() > size as usize {
                return Err(VelocityIoError::InvalidRequest(
                    "decompressed payload exceeds physical page destination size",
                ));
            }
            hooks.write_physical_pages(page_list_addr, page_offset, &bytes)
        }
    }
}

/// Safe factory controller backing `IDStorageFactoryX`.
pub struct VelocityFactory {
    io_backend: Arc<dyn AsyncIoBackend>,
    dispatcher: Mutex<Option<Arc<IoCompletionDispatcher>>>,
    shared_ring: Mutex<Option<Arc<PinnedUploadRing>>>,
    gpu_decompress: Arc<GpuDecompressQueue>,
    hooks: Arc<dyn DStoragePlatformHooks>,
    upload_allocator: Option<Arc<dyn UploadAllocator>>,
    staging_buffer_size: AtomicU64,
}

impl std::fmt::Debug for VelocityFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VelocityFactory")
            .field(
                "staging_buffer_size",
                &self.staging_buffer_size.load(Ordering::Relaxed),
            )
            .field("has_upload_allocator", &self.upload_allocator.is_some())
            .finish()
    }
}

impl VelocityFactory {
    /// Default Xbox Series X|S DirectStorage staging buffer size (32 MiB).
    pub const DEFAULT_STAGING_BUFFER_BYTES: u64 = 32 * 1024 * 1024;

    /// Creates a new factory with [`DefaultPlatformHooks`] and standalone pinned upload rings.
    #[must_use]
    pub fn new(
        io_backend: Arc<dyn AsyncIoBackend>,
        gpu_decompress: Arc<GpuDecompressQueue>,
    ) -> Self {
        Self {
            io_backend,
            dispatcher: Mutex::new(None),
            shared_ring: Mutex::new(None),
            gpu_decompress,
            hooks: Arc::new(DefaultPlatformHooks),
            upload_allocator: None,
            staging_buffer_size: AtomicU64::new(Self::DEFAULT_STAGING_BUFFER_BYTES),
        }
    }

    fn get_or_start_dispatcher(&self) -> Result<Arc<IoCompletionDispatcher>, VelocityIoError> {
        let mut guard = self.dispatcher.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(ref disp) = *guard {
            return Ok(Arc::clone(disp));
        }
        let disp = IoCompletionDispatcher::new(Arc::clone(&self.io_backend))?;
        *guard = Some(Arc::clone(&disp));
        Ok(disp)
    }

    fn get_or_create_shared_ring(&self) -> Result<Arc<PinnedUploadRing>, VelocityIoError> {
        let mut guard = self.shared_ring.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(ref ring) = *guard {
            return Ok(Arc::clone(ring));
        }
        // Size at least 64 MiB (or `staging_buffer_size`) so all queues created by the title
        // share one cached host ring without duplicating 32 MiB per queue.
        let bytes = (self.staging_buffer_size() as usize).max(64 * 1024 * 1024);
        let ring = Arc::new(PinnedUploadRing::new_standalone(bytes)?);
        *guard = Some(Arc::clone(&ring));
        Ok(ring)
    }

    /// Overrides the [`DStoragePlatformHooks`] implementation (e.g. `victory-hle` hooks).
    #[must_use]
    pub fn with_platform_hooks(mut self, hooks: Arc<dyn DStoragePlatformHooks>) -> Self {
        self.hooks = hooks;
        self
    }

    /// Configures queues created by this factory to lease upload slices through
    /// Subsystem A's [`UploadAllocator`].
    #[must_use]
    pub fn with_upload_allocator(mut self, allocator: Arc<dyn UploadAllocator>) -> Self {
        self.upload_allocator = Some(allocator);
        self
    }

    /// Configures the default staging buffer size (`IDStorageFactoryX::SetStagingBufferSize`).
    pub fn set_staging_buffer_size(&self, size: u32) {
        self.staging_buffer_size
            .store(u64::from(size).max(PINNED_RING_PAGE_ALIGN as u64), Ordering::Relaxed);
    }

    /// Returns the configured staging buffer size in bytes.
    #[inline]
    #[must_use]
    pub fn staging_buffer_size(&self) -> u64 {
        self.staging_buffer_size.load(Ordering::Relaxed)
    }

    /// Opens a read-only file directly via [`AsyncIoBackend`].
    pub fn open_file(&self, path: &Path) -> Result<VelocityFile, VelocityIoError> {
        let id = self.io_backend.open_read_only(path)?;
        Ok(VelocityFile {
            id,
            io: Arc::clone(&self.io_backend),
        })
    }

    /// Opens a read-only file from an Xbox UTF-16 virtual path using
    /// [`DStoragePlatformHooks::rewrite_virtual_path`].
    pub fn open_file_utf16(&self, raw_utf16_path: &[u16]) -> Result<VelocityFile, VelocityIoError> {
        let rewritten = self.hooks.rewrite_virtual_path(raw_utf16_path);
        self.open_file(&rewritten)
    }

    /// Creates a new [`VelocityStatusArray`] (`IDStorageFactoryX::CreateStatusArray`).
    #[must_use]
    pub fn create_status_array(&self, capacity: u32) -> Arc<VelocityStatusArray> {
        VelocityStatusArray::new(capacity)
    }

    /// Creates a new [`VelocityQueue`] from high-level configuration.
    pub fn create_queue(
        &self,
        config: VelocityQueueConfig,
    ) -> Result<Arc<VelocityQueue>, VelocityIoError> {
        let ring = if let Some(ref allocator) = self.upload_allocator {
            Arc::new(PinnedUploadRing::with_upload_allocator(
                Arc::clone(allocator),
                256,
            ))
        } else if config.staging_ring_bytes > 0 {
            Arc::new(PinnedUploadRing::new_standalone(config.staging_ring_bytes)?)
        } else {
            self.get_or_create_shared_ring()?
        };

        let dispatcher = self.get_or_start_dispatcher()?;
        VelocityQueue::new_internal(
            config,
            Arc::clone(&self.io_backend),
            Some(dispatcher),
            Arc::clone(&self.gpu_decompress),
            Arc::clone(&self.hooks),
            ring,
        )
    }

    /// Creates a [`VelocityQueue`] directly from an Xbox [`DStorageQueueDescX`] ABI struct.
    pub fn create_queue_from_abi(
        &self,
        desc: &DStorageQueueDescX,
    ) -> Result<Arc<VelocityQueue>, VelocityIoError> {
        let name = if desc.name.is_null() {
            String::from("dstorage-x")
        } else {
            // SAFETY: Non-null NUL-terminated C string passed in `DStorageQueueDescX`.
            unsafe { std::ffi::CStr::from_ptr(desc.name) }
                .to_string_lossy()
                .into_owned()
        };
        let config = VelocityQueueConfig {
            capacity: usize::from(desc.capacity).max(2),
            priority: desc.priority_enum(),
            // `0` selects the factory-wide shared cached staging ring so titles creating
            // many queues (e.g., 17 queues) do not duplicate 32 MiB per queue.
            staging_ring_bytes: 0,
            decompress_workers: 2,
            name,
        };
        self.create_queue(config)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_abi_decode_chained_zlib_bcpack() {
        let mut dst_buf = [0u8; 64];
        let req = DStorageRequestX {
            options: ZLIB | (7 << request_options::BCPACK_SHIFT),
            destination: dst_buf.as_mut_ptr().cast(),
            page_offset: 0,
            destination_size: 64,
            source: std::ptr::dangling::<c_void>(),
            offset: 0x2000,
            source_size: 32,
            intermediate_size: 48,
            tag: 0xDEAD_BEEF,
            name: std::ptr::null(),
        };
        let desc = req
            .to_desc(|_| Ok(FileHandleId(99)))
            .expect("decode chained request");
        assert_eq!(desc.codec, DecompressionCodec::Zlib);
        assert_eq!(
            desc.secondary_codec,
            Some(DecompressionCodec::BcPack(BcPackFormat::Bc7))
        );
        assert_eq!(desc.intermediate_size, 48);
        assert_eq!(desc.cancellation_tag, 0xDEAD_BEEF);
    }

    #[test]
    fn in_order_committer_records_first_failure_even_when_completed_out_of_order() {
        let status = VelocityStatusArray::new(2);
        status.mark_pending(0).unwrap();
        status.mark_pending(1).unwrap();

        let mut committer = InOrderCommitter::new();
        committer.register_status(2, Arc::clone(&status), 0);
        committer.register_status(4, Arc::clone(&status), 1);

        // Request 1 finishes first with E_NOTIMPL; Request 0 still pending -> status 0 must stay E_PENDING.
        committer.complete_request(1, E_NOTIMPL);
        assert!(!status.is_complete(0));
        assert_eq!(status.hresult(0), E_PENDING);

        // Request 0 finishes with E_INVALIDARG -> status 0 must receive E_INVALIDARG (first in enqueue order).
        committer.complete_request(0, E_INVALIDARG);
        assert!(status.is_complete(0));
        assert_eq!(status.hresult(0), E_INVALIDARG);

        // Next group succeeds -> status 1 receives S_OK.
        committer.complete_request(3, S_OK);
        assert!(status.is_complete(1));
        assert_eq!(status.hresult(1), S_OK);
    }
}
