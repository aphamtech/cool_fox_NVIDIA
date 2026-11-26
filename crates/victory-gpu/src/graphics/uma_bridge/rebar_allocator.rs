//! Subsystem A.1 — ReBAR-aware sub-allocator & non-ReBAR async Copy Queue ring fallback.
//!
//! Translates Xbox Series X|S CPU-visible unified memory writes into two
//! discrete-GPU-safe tiers without blocking command-recording threads:
//!
//! 1. **Hot Path ([`MemoryTier::RebarWriteCombine`])**: Sub-allocates from
//!    pre-allocated slabs (`D3D12_HEAP_TYPE_GPU_UPLOAD` on discrete ReBAR GPUs,
//!    or CPU-visible Write-Combine unified memory on UMA adapters) using an
//!    address-ordered coalescing free-list and lock-free [`SegQueue`]
//!    fence-gated retirement.
//! 2. **Fallback & Spillover ([`MemoryTier::StagingCopyRing`])**: Persistent
//!    power-of-two pinned Write-Combine L0 ring buffer (`D3D12_HEAP_TYPE_CUSTOM`,
//!    `WRITE_COMBINE` + `L0`) with atomic head/tail tracking, fence-gated
//!    reclamation, non-blocking exhaustion (`UmaBridgeError::RingExhausted`),
//!    and lock-free deferred batching onto an async `D3D12_COMMAND_LIST_TYPE_COPY`
//!    queue.

use std::collections::{BTreeMap, HashMap};
use std::num::NonZeroU64;
use std::ptr::NonNull;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crossbeam_queue::SegQueue;

use crate::graphics::contracts::{
    BufferCopyOp, CopyQueueBackend, GpuFenceToken, GpuHeapBackend, GpuVirtualAddress, MemoryTier,
    RebarCapabilities, ResourceId, UmaBridgeError, UploadAllocator, UploadSlice,
};

/// Configuration knobs for [`RebarUploadAllocator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RebarAllocatorConfig {
    /// Target size in bytes of each ReBAR slab allocation (clamped to remaining
    /// ReBAR budget when the adapter budget is smaller).
    pub rebar_slab_size: u64,
    /// Persistent power-of-two byte capacity of the fallback `StagingCopyRing`.
    pub staging_ring_capacity: u64,
    /// Minimum power-of-two alignment applied to every allocation (default `1`).
    pub min_alignment: u64,
}

impl Default for RebarAllocatorConfig {
    fn default() -> Self {
        Self {
            rebar_slab_size: 64 * 1024 * 1024,
            staging_ring_capacity: 32 * 1024 * 1024,
            min_alignment: 1,
        }
    }
}

#[inline]
fn align_up(value: u64, alignment: u64) -> Option<u64> {
    let mask = alignment.checked_sub(1)?;
    value.checked_add(mask).map(|v| v & !mask)
}

// ---------------------------------------------------------------------------
// Hot-path ReBAR slab sub-allocator with address-ordered coalescing free-list
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FreeRange {
    offset: u64,
    size: u64,
}

struct RebarSlab {
    resource_id: ResourceId,
    base_gpu_va: GpuVirtualAddress,
    base_cpu_ptr: NonNull<u8>,
    capacity: u64,
    /// Address-ordered, strictly non-overlapping and non-adjacent free blocks.
    free_ranges: Vec<FreeRange>,
}

// SAFETY: `base_cpu_ptr` points to a persistently mapped slab managed exclusively
// by the allocator; disjoint sub-ranges are handed out uniquely via `UploadSlice`.
unsafe impl Send for RebarSlab {}

impl RebarSlab {
    fn new(
        resource_id: ResourceId,
        base_gpu_va: GpuVirtualAddress,
        base_cpu_ptr: NonNull<u8>,
        capacity: u64,
    ) -> Self {
        Self {
            resource_id,
            base_gpu_va,
            base_cpu_ptr,
            capacity,
            free_ranges: vec![FreeRange {
                offset: 0,
                size: capacity,
            }],
        }
    }

    /// First-fit address-ordered sub-allocation preserving alignment padding in
    /// the free-list so later frees coalesce cleanly without permanent holes.
    fn try_allocate(&mut self, size: u64, alignment: u64) -> Option<(u64, NonNull<u8>)> {
        for idx in 0..self.free_ranges.len() {
            let range = self.free_ranges[idx];
            let range_end = range.offset.checked_add(range.size)?;
            let aligned = align_up(range.offset, alignment)?;
            let alloc_end = aligned.checked_add(size)?;

            if alloc_end <= range_end {
                let prefix_len = aligned - range.offset;
                let suffix_len = range_end - alloc_end;

                match (prefix_len > 0, suffix_len > 0) {
                    (false, false) => {
                        self.free_ranges.remove(idx);
                    }
                    (true, false) => {
                        self.free_ranges[idx] = FreeRange {
                            offset: range.offset,
                            size: prefix_len,
                        };
                    }
                    (false, true) => {
                        self.free_ranges[idx] = FreeRange {
                            offset: alloc_end,
                            size: suffix_len,
                        };
                    }
                    (true, true) => {
                        self.free_ranges[idx] = FreeRange {
                            offset: range.offset,
                            size: prefix_len,
                        };
                        self.free_ranges.insert(
                            idx + 1,
                            FreeRange {
                                offset: alloc_end,
                                size: suffix_len,
                            },
                        );
                    }
                }

                // SAFETY: `aligned + size <= range_end <= self.capacity` and
                // `base_cpu_ptr` is mapped for `self.capacity` contiguous bytes.
                let ptr = unsafe {
                    NonNull::new_unchecked(self.base_cpu_ptr.as_ptr().add(aligned as usize))
                };
                return Some((aligned, ptr));
            }
        }
        None
    }

    /// Returns `[offset, offset + size)` to the free-list, coalescing with
    /// adjacent left/right free blocks.
    fn free_and_coalesce(&mut self, offset: u64, size: u64) {
        if size == 0 || offset.checked_add(size).is_none_or(|e| e > self.capacity) {
            return;
        }
        let end = offset + size;
        let pos = self.free_ranges.partition_point(|r| r.offset < offset);

        let merge_left = pos > 0 && {
            let left = self.free_ranges[pos - 1];
            left.offset + left.size == offset
        };
        let merge_right = pos < self.free_ranges.len() && self.free_ranges[pos].offset == end;

        match (merge_left, merge_right) {
            (true, true) => {
                let right = self.free_ranges.remove(pos);
                self.free_ranges[pos - 1].size += size + right.size;
            }
            (true, false) => {
                self.free_ranges[pos - 1].size += size;
            }
            (false, true) => {
                self.free_ranges[pos].offset = offset;
                self.free_ranges[pos].size += size;
            }
            (false, false) => {
                self.free_ranges.insert(pos, FreeRange { offset, size });
            }
        }
    }

    fn free_bytes(&self) -> u64 {
        self.free_ranges.iter().map(|r| r.size).sum()
    }
}

#[derive(Debug, Clone, Copy)]
struct RetiredRebarSlice {
    resource_id: ResourceId,
    heap_offset: u64,
    size: u64,
    retire_after: u64,
}

struct RebarPoolState {
    slabs: Vec<RebarSlab>,
    total_allocated_bytes: u64,
    pending_retired: Vec<RetiredRebarSlice>,
}

// ---------------------------------------------------------------------------
// Fallback persistent power-of-two StagingCopyRing (WC + L0)
// ---------------------------------------------------------------------------

struct StagingRingBuffer {
    resource_id: ResourceId,
    base_gpu_va: GpuVirtualAddress,
    base_cpu_ptr: NonNull<u8>,
    capacity: u64,
    mask: u64,
    /// Monotonic producer offset (never wraps modulo `capacity`).
    head: AtomicU64,
    /// Monotonic consumer offset advanced only after fence completion.
    tail: AtomicU64,
}

// SAFETY: `base_cpu_ptr` points to a persistently mapped WC+L0 buffer whose
// disjoint sub-ranges are uniquely leased via atomic CAS on `head` and only
// reclaimed after `tail` advances past completed GPU fences.
unsafe impl Send for StagingRingBuffer {}
// SAFETY: Concurrent allocations synchronize via atomic CAS on `head` and
// acquire/release ordering on `tail`; no two live slices overlap.
unsafe impl Sync for StagingRingBuffer {}

#[derive(Debug, Clone, Copy)]
struct RingLease {
    allocation_id: NonZeroU64,
    monotonic_start: u64,
    monotonic_end: u64,
}

#[derive(Debug, Clone, Copy)]
struct RingRetirement {
    allocation_id: NonZeroU64,
    retire_after: u64,
}

#[derive(Debug, Clone, Copy)]
struct RingSlot {
    monotonic_end: u64,
    allocation_id: NonZeroU64,
    retire_fence: Option<u64>,
}

#[derive(Default)]
struct RingReclaimState {
    /// Active (not yet reclaimed) monotonic ring ranges ordered by `monotonic_start`.
    active_by_start: BTreeMap<u64, RingSlot>,
    id_to_start: HashMap<NonZeroU64, u64>,
    early_retirements: HashMap<NonZeroU64, u64>,
}

#[derive(Debug, Clone, Copy)]
struct PendingCopy {
    op: BufferCopyOp,
    allocation_id: NonZeroU64,
    tier: MemoryTier,
}

// ---------------------------------------------------------------------------
// RebarUploadAllocator
// ---------------------------------------------------------------------------

/// ReBAR-aware upload sub-allocator with automatic non-ReBAR / budget-spillover
/// pinned WC+L0 staging ring and async Copy Queue batching.
pub struct RebarUploadAllocator<H: GpuHeapBackend, C: CopyQueueBackend> {
    heap: H,
    copy_queue: C,
    caps: RebarCapabilities,
    config: RebarAllocatorConfig,
    rebar_enabled: bool,
    next_alloc_id: AtomicU64,
    rebar_state: Mutex<RebarPoolState>,
    retired_rebar: SegQueue<RetiredRebarSlice>,
    ring: StagingRingBuffer,
    ring_leases: SegQueue<RingLease>,
    retired_ring: SegQueue<RingRetirement>,
    ring_reclaim: Mutex<RingReclaimState>,
    pending_copies: SegQueue<PendingCopy>,
}

impl<H: GpuHeapBackend, C: CopyQueueBackend> RebarUploadAllocator<H, C> {
    /// Constructs a new [`RebarUploadAllocator`], probing adapter ReBAR/UMA
    /// capabilities and allocating the persistent `StagingCopyRing` (plus an
    /// initial ReBAR slab when ReBAR/UMA write-combine memory is available).
    pub fn new(
        heap: H,
        copy_queue: C,
        config: RebarAllocatorConfig,
    ) -> Result<Self, UmaBridgeError> {
        if config.staging_ring_capacity == 0 || !config.staging_ring_capacity.is_power_of_two() {
            return Err(UmaBridgeError::InvalidLayout {
                size: config.staging_ring_capacity,
                alignment: config.staging_ring_capacity,
            });
        }
        if config.min_alignment == 0 || !config.min_alignment.is_power_of_two() {
            return Err(UmaBridgeError::InvalidLayout {
                size: 0,
                alignment: config.min_alignment,
            });
        }

        let caps = heap.probe_capabilities();
        let rebar_enabled = (caps.gpu_upload_heap_supported || caps.is_uma)
            && caps.rebar_budget_bytes > 0
            && config.rebar_slab_size > 0;

        // Always allocate the persistent StagingCopyRing (WC + L0) so fallback
        // and ReBAR budget spillover are immediately ready with zero render-thread
        // creation stalls.
        let (ring_res_id, ring_gpu_va, ring_cpu_ptr) =
            heap.create_buffer_slab(config.staging_ring_capacity, MemoryTier::StagingCopyRing)?;
        let Some(ring_cpu_ptr) = ring_cpu_ptr else {
            heap.destroy_buffer_slab(ring_res_id);
            return Err(UmaBridgeError::HostD3d12 {
                hresult: -2147467259, // E_FAIL
                context: "StagingCopyRing slab returned unmapped CPU pointer",
            });
        };

        let mut slabs = Vec::new();
        let mut total_allocated_bytes = 0u64;

        if rebar_enabled {
            let initial_slab_size = config.rebar_slab_size.min(caps.rebar_budget_bytes);
            if initial_slab_size > 0
                && let Ok((res_id, gpu_va, cpu_ptr)) =
                    heap.create_buffer_slab(initial_slab_size, MemoryTier::RebarWriteCombine)
            {
                if let Some(cpu_ptr) = cpu_ptr {
                    slabs.push(RebarSlab::new(res_id, gpu_va, cpu_ptr, initial_slab_size));
                    total_allocated_bytes = initial_slab_size;
                } else {
                    heap.destroy_buffer_slab(res_id);
                }
            }
        }

        Ok(Self {
            heap,
            copy_queue,
            caps,
            config,
            rebar_enabled,
            next_alloc_id: AtomicU64::new(1),
            rebar_state: Mutex::new(RebarPoolState {
                slabs,
                total_allocated_bytes,
                pending_retired: Vec::new(),
            }),
            retired_rebar: SegQueue::new(),
            ring: StagingRingBuffer {
                resource_id: ring_res_id,
                base_gpu_va: ring_gpu_va,
                base_cpu_ptr: ring_cpu_ptr,
                capacity: config.staging_ring_capacity,
                mask: config.staging_ring_capacity - 1,
                head: AtomicU64::new(0),
                tail: AtomicU64::new(0),
            },
            ring_leases: SegQueue::new(),
            retired_ring: SegQueue::new(),
            ring_reclaim: Mutex::new(RingReclaimState::default()),
            pending_copies: SegQueue::new(),
        })
    }

    /// Borrows the underlying heap backend.
    #[must_use]
    pub fn heap_backend(&self) -> &H {
        &self.heap
    }

    /// Borrows the underlying copy queue backend.
    #[must_use]
    pub fn copy_queue_backend(&self) -> &C {
        &self.copy_queue
    }

    /// Returns the currently used byte count in the fallback staging ring.
    #[must_use]
    pub fn staging_ring_used_bytes(&self) -> u64 {
        let head = self.ring.head.load(Ordering::Acquire);
        let tail = self.ring.tail.load(Ordering::Acquire);
        head.saturating_sub(tail)
    }

    /// Returns the total free bytes across all currently allocated ReBAR slabs.
    #[must_use]
    pub fn rebar_free_bytes(&self) -> u64 {
        let state = self
            .rebar_state
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        state.slabs.iter().map(RebarSlab::free_bytes).sum()
    }

    /// Returns the number of live free-list fragments across all ReBAR slabs.
    #[must_use]
    pub fn rebar_free_fragment_count(&self) -> usize {
        let state = self
            .rebar_state
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        state.slabs.iter().map(|s| s.free_ranges.len()).sum()
    }

    /// Convenience helper polling `copy_queue.completed_fence_value()` and
    /// reclaiming all completed slices.
    pub fn reclaim_from_copy_queue(&self) -> usize {
        self.reclaim_completed(self.copy_queue.completed_fence_value())
    }

    fn next_allocation_id(&self) -> NonZeroU64 {
        let raw = self.next_alloc_id.fetch_add(1, Ordering::Relaxed);
        // `next_alloc_id` starts at 1; guard against theoretical u64 wrap.
        NonZeroU64::new(raw).unwrap_or(NonZeroU64::MIN)
    }

    fn try_allocate_rebar(&self, size: u64, alignment: u64) -> Option<UploadSlice> {
        if !self.rebar_enabled {
            return None;
        }

        let mut state = self
            .rebar_state
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        for slab in &mut state.slabs {
            if let Some((offset, cpu_ptr)) = slab.try_allocate(size, alignment) {
                let alloc_id = self.next_allocation_id();
                // SAFETY: `cpu_ptr` is uniquely sub-allocated from `slab` for
                // `size` bytes until returned via `retire_slice` and fence completion.
                let slice = unsafe {
                    UploadSlice::new(
                        alloc_id,
                        MemoryTier::RebarWriteCombine,
                        slab.resource_id,
                        offset,
                        size,
                        GpuVirtualAddress(slab.base_gpu_va.0 + offset),
                        None,
                        0,
                        cpu_ptr,
                    )
                };
                return Some(slice);
            }
        }

        // Grow a new ReBAR slab if adapter ReBAR budget allows.
        let remaining_budget = self
            .caps
            .rebar_budget_bytes
            .saturating_sub(state.total_allocated_bytes);
        if size <= remaining_budget {
            let slab_size = self.config.rebar_slab_size.max(size).min(remaining_budget);
            if let Ok((res_id, gpu_va, base_ptr)) = self
                .heap
                .create_buffer_slab(slab_size, MemoryTier::RebarWriteCombine)
            {
                let Some(base_ptr) = base_ptr else {
                    self.heap.destroy_buffer_slab(res_id);
                    return None;
                };
                let mut slab = RebarSlab::new(res_id, gpu_va, base_ptr, slab_size);
                let allocated = slab.try_allocate(size, alignment);
                state.total_allocated_bytes += slab_size;
                state.slabs.push(slab);
                if let Some((offset, cpu_ptr)) = allocated {
                    let alloc_id = self.next_allocation_id();
                    // SAFETY: `cpu_ptr` is uniquely sub-allocated from the new
                    // slab for `size` contiguous bytes.
                    let slice = unsafe {
                        UploadSlice::new(
                            alloc_id,
                            MemoryTier::RebarWriteCombine,
                            res_id,
                            offset,
                            size,
                            GpuVirtualAddress(gpu_va.0 + offset),
                            None,
                            0,
                            cpu_ptr,
                        )
                    };
                    return Some(slice);
                }
            }
        }

        None
    }

    fn allocate_from_ring(&self, size: u64, alignment: u64) -> Result<UploadSlice, UmaBridgeError> {
        let cap = self.ring.capacity;
        let mask = self.ring.mask;

        if size > cap || alignment > cap {
            let head = self.ring.head.load(Ordering::Acquire);
            let tail = self.ring.tail.load(Ordering::Acquire);
            let available = cap.saturating_sub(head.saturating_sub(tail));
            return Err(UmaBridgeError::RingExhausted {
                requested: size,
                available,
            });
        }

        loop {
            let cur_head = self.ring.head.load(Ordering::Acquire);
            let cur_tail = self.ring.tail.load(Ordering::Acquire);
            let used = cur_head.saturating_sub(cur_tail);
            let available = cap.saturating_sub(used);

            let Some(aligned_head) = align_up(cur_head, alignment) else {
                return Err(UmaBridgeError::InvalidLayout { size, alignment });
            };
            let phys = aligned_head & mask;
            let to_end = cap - phys;

            let (alloc_start, alloc_end) = if size <= to_end {
                let Some(end) = aligned_head.checked_add(size) else {
                    return Err(UmaBridgeError::InvalidLayout { size, alignment });
                };
                (aligned_head, end)
            } else {
                // Wrap to physical offset 0 so the slice remains contiguous.
                let Some(wrap_start) = aligned_head.checked_add(to_end) else {
                    return Err(UmaBridgeError::InvalidLayout { size, alignment });
                };
                let Some(end) = wrap_start.checked_add(size) else {
                    return Err(UmaBridgeError::InvalidLayout { size, alignment });
                };
                (wrap_start, end)
            };

            if alloc_end.saturating_sub(cur_tail) > cap {
                return Err(UmaBridgeError::RingExhausted {
                    requested: size,
                    available,
                });
            }

            if self
                .ring
                .head
                .compare_exchange_weak(cur_head, alloc_end, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                let alloc_id = self.next_allocation_id();
                self.ring_leases.push(RingLease {
                    allocation_id: alloc_id,
                    monotonic_start: cur_head,
                    monotonic_end: alloc_end,
                });

                let phys_offset = alloc_start & mask;
                // SAFETY: `phys_offset + size <= cap` (enforced by the wrap check
                // above), and the atomic CAS uniquely reserved `[cur_head, alloc_end)`
                // until retired and reclaimed after GPU fence completion.
                let cpu_ptr = unsafe {
                    NonNull::new_unchecked(self.ring.base_cpu_ptr.as_ptr().add(phys_offset as usize))
                };
                // SAFETY: Unique ownership of `[phys_offset, phys_offset + size)` in
                // the persistently mapped staging ring buffer.
                let slice = unsafe {
                    UploadSlice::new(
                        alloc_id,
                        MemoryTier::StagingCopyRing,
                        self.ring.resource_id,
                        phys_offset,
                        size,
                        GpuVirtualAddress(self.ring.base_gpu_va.0 + phys_offset),
                        None,
                        0,
                        cpu_ptr,
                    )
                };
                return Ok(slice);
            }
        }
    }
}

impl<H: GpuHeapBackend, C: CopyQueueBackend> Drop for RebarUploadAllocator<H, C> {
    fn drop(&mut self) {
        // Wait for all submitted async copy work to complete before unmapping/destroying
        // slabs and the staging ring buffer.
        self.copy_queue.drain_gpu();
        if let Ok(mut state) = self.rebar_state.lock() {
            for slab in state.slabs.drain(..) {
                self.heap.destroy_buffer_slab(slab.resource_id);
            }
        }
        self.heap.destroy_buffer_slab(self.ring.resource_id);
    }
}

impl<H: GpuHeapBackend, C: CopyQueueBackend> UploadAllocator for RebarUploadAllocator<H, C> {
    fn capabilities(&self) -> RebarCapabilities {
        self.caps
    }

    fn allocate(&self, size: u64, alignment: u64) -> Result<UploadSlice, UmaBridgeError> {
        if size == 0 || alignment == 0 || !alignment.is_power_of_two() {
            return Err(UmaBridgeError::InvalidLayout { size, alignment });
        }
        let eff_alignment = alignment.max(self.config.min_alignment);

        if let Some(slice) = self.try_allocate_rebar(size, eff_alignment) {
            return Ok(slice);
        }

        self.allocate_from_ring(size, eff_alignment)
    }

    fn schedule_copy_to_vram(
        &self,
        slice: UploadSlice,
        dst_resource: ResourceId,
        dst_offset: u64,
    ) -> Result<(), UmaBridgeError> {
        if dst_offset.checked_add(slice.size).is_none() {
            return Err(UmaBridgeError::InvalidLayout {
                size: slice.size,
                alignment: 1,
            });
        }

        self.pending_copies.push(PendingCopy {
            op: BufferCopyOp {
                src_resource: slice.resource_id,
                src_offset: slice.heap_offset,
                dst_resource,
                dst_offset,
                size: slice.size,
            },
            allocation_id: slice.allocation_id,
            tier: slice.tier,
        });
        Ok(())
    }

    fn flush_copies(&self) -> Result<Option<GpuFenceToken>, UmaBridgeError> {
        let mut items = Vec::new();
        while let Some(item) = self.pending_copies.pop() {
            items.push(item);
        }
        if items.is_empty() {
            return Ok(None);
        }

        let ops: Vec<BufferCopyOp> = items.iter().map(|i| i.op).collect();
        match self.copy_queue.submit_buffer_copies(&ops) {
            Ok(token) => {
                for item in items {
                    match item.tier {
                        MemoryTier::StagingCopyRing => {
                            self.retired_ring.push(RingRetirement {
                                allocation_id: item.allocation_id,
                                retire_after: token.value,
                            });
                        }
                        MemoryTier::RebarWriteCombine => {
                            self.retired_rebar.push(RetiredRebarSlice {
                                resource_id: item.op.src_resource,
                                heap_offset: item.op.src_offset,
                                size: item.op.size,
                                retire_after: token.value,
                            });
                        }
                        MemoryTier::DeviceLocalDefault => {}
                    }
                }
                Ok(Some(token))
            }
            Err(err) => {
                // Re-queue failed copies so callers can retry without losing ownership.
                for item in items {
                    self.pending_copies.push(item);
                }
                Err(err)
            }
        }
    }

    fn reclaim_completed(&self, completed_fence: u64) -> usize {
        let mut reclaimed = 0usize;

        // 1. Reclaim completed ReBAR slices and coalesce into address-ordered free-lists.
        {
            let mut state = self
                .rebar_state
                .lock()
                .unwrap_or_else(|e| e.into_inner());

            while let Some(retired) = self.retired_rebar.pop() {
                state.pending_retired.push(retired);
            }

            let mut ready = Vec::new();
            state.pending_retired.retain(|item| {
                if item.retire_after <= completed_fence {
                    ready.push(*item);
                    false
                } else {
                    true
                }
            });

            for item in ready {
                if let Some(slab) = state
                    .slabs
                    .iter_mut()
                    .find(|s| s.resource_id == item.resource_id)
                {
                    slab.free_and_coalesce(item.heap_offset, item.size);
                    reclaimed += 1;
                }
            }
        }

        // 2. Reclaim contiguous completed prefix of the StagingCopyRing and advance `tail`.
        {
            let mut ring_state = self
                .ring_reclaim
                .lock()
                .unwrap_or_else(|e| e.into_inner());

            while let Some(lease) = self.ring_leases.pop() {
                let retire_fence = ring_state.early_retirements.remove(&lease.allocation_id);
                ring_state
                    .id_to_start
                    .insert(lease.allocation_id, lease.monotonic_start);
                ring_state.active_by_start.insert(
                    lease.monotonic_start,
                    RingSlot {
                        monotonic_end: lease.monotonic_end,
                        allocation_id: lease.allocation_id,
                        retire_fence,
                    },
                );
            }

            while let Some(ret) = self.retired_ring.pop() {
                if let Some(&start) = ring_state.id_to_start.get(&ret.allocation_id) {
                    if let Some(slot) = ring_state.active_by_start.get_mut(&start) {
                        slot.retire_fence = Some(ret.retire_after);
                    }
                } else {
                    ring_state
                        .early_retirements
                        .insert(ret.allocation_id, ret.retire_after);
                }
            }

            let mut expected_tail = self.ring.tail.load(Ordering::Acquire);
            let mut advanced = false;
            loop {
                let front_ready = ring_state
                    .active_by_start
                    .first_key_value()
                    .and_then(|(&start, slot)| {
                        (start == expected_tail)
                            .then_some(slot)
                            .and_then(|s| s.retire_fence.filter(|&fence| fence <= completed_fence))
                            .map(|_| (start, slot.monotonic_end, slot.allocation_id))
                    });

                let Some((start, end, id)) = front_ready else {
                    break;
                };
                ring_state.active_by_start.remove(&start);
                ring_state.id_to_start.remove(&id);
                expected_tail = end;
                advanced = true;
                reclaimed += 1;
            }

            if advanced {
                self.ring.tail.store(expected_tail, Ordering::Release);
            }
        }

        reclaimed
    }

    fn retire_slice(&self, slice: UploadSlice, retire_after: GpuFenceToken) {
        match slice.tier {
            MemoryTier::RebarWriteCombine => {
                self.retired_rebar.push(RetiredRebarSlice {
                    resource_id: slice.resource_id,
                    heap_offset: slice.heap_offset,
                    size: slice.size,
                    retire_after: retire_after.value,
                });
            }
            MemoryTier::StagingCopyRing => {
                self.retired_ring.push(RingRetirement {
                    allocation_id: slice.allocation_id,
                    retire_after: retire_after.value,
                });
            }
            MemoryTier::DeviceLocalDefault => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic in-memory Mock Backends for headless tests & benchmarks
// ---------------------------------------------------------------------------

struct MockBufferEntry {
    tier: MemoryTier,
    gpu_va: GpuVirtualAddress,
    /// Heap-allocated pinned byte storage so `NonNull<u8>` remains stable until
    /// `destroy_buffer_slab` removes the entry.
    storage: Box<[u8]>,
}

// SAFETY: `MockBufferEntry` owns its `Box<[u8]>` exclusively behind a `Mutex`.
unsafe impl Send for MockBufferEntry {}

#[derive(Default)]
struct MockHeapState {
    buffers: HashMap<ResourceId, MockBufferEntry>,
    created_rebar_slabs: usize,
    created_staging_slabs: usize,
    created_default_slabs: usize,
    destroyed_slabs: usize,
}

/// Deterministic headless [`GpuHeapBackend`] backed by stable heap memory.
pub struct MockHeapBackend {
    caps: RebarCapabilities,
    next_id: AtomicU64,
    next_va: AtomicU64,
    state: Mutex<MockHeapState>,
}

impl MockHeapBackend {
    /// Creates a mock heap backend with explicit capabilities.
    #[must_use]
    pub fn new(caps: RebarCapabilities) -> Self {
        Self {
            caps,
            next_id: AtomicU64::new(1),
            next_va: AtomicU64::new(0x0001_0000_0000),
            state: Mutex::new(MockHeapState::default()),
        }
    }

    /// Preset simulating an NVIDIA RTX discrete GPU with Resizable BAR enabled.
    #[must_use]
    pub fn rtx_rebar(rebar_budget_bytes: u64) -> Self {
        Self::new(RebarCapabilities {
            gpu_upload_heap_supported: true,
            is_uma: false,
            dedicated_vram_bytes: 16 * 1024 * 1024 * 1024,
            rebar_budget_bytes,
        })
    }

    /// Preset simulating a discrete GPU without Resizable BAR support.
    #[must_use]
    pub fn discrete_no_rebar() -> Self {
        Self::new(RebarCapabilities {
            gpu_upload_heap_supported: false,
            is_uma: false,
            dedicated_vram_bytes: 8 * 1024 * 1024 * 1024,
            rebar_budget_bytes: 0,
        })
    }

    /// Preset simulating a UMA adapter.
    #[must_use]
    pub fn uma(budget_bytes: u64) -> Self {
        Self::new(RebarCapabilities {
            gpu_upload_heap_supported: false,
            is_uma: true,
            dedicated_vram_bytes: 0,
            rebar_budget_bytes: budget_bytes,
        })
    }

    /// Allocates a device-local buffer for verifying Copy Queue transfers.
    pub fn create_device_local_buffer(&self, size: u64) -> Result<ResourceId, UmaBridgeError> {
        let (id, _, _) = self.create_buffer_slab(size, MemoryTier::DeviceLocalDefault)?;
        Ok(id)
    }

    /// Reads back bytes from a mock buffer for test assertions.
    #[must_use]
    pub fn read_bytes(&self, resource_id: ResourceId, offset: u64, len: usize) -> Option<Vec<u8>> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let entry = state.buffers.get(&resource_id)?;
        let start = usize::try_from(offset).ok()?;
        let end = start.checked_add(len)?;
        Some(entry.storage.get(start..end)?.to_vec())
    }

    /// Performs an in-memory buffer copy between two mock resources.
    pub fn execute_copy(&self, op: BufferCopyOp) -> Result<(), UmaBridgeError> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let src_start = usize::try_from(op.src_offset).map_err(|_| UmaBridgeError::InvalidLayout {
            size: op.size,
            alignment: 1,
        })?;
        let dst_start = usize::try_from(op.dst_offset).map_err(|_| UmaBridgeError::InvalidLayout {
            size: op.size,
            alignment: 1,
        })?;
        let len = usize::try_from(op.size).map_err(|_| UmaBridgeError::InvalidLayout {
            size: op.size,
            alignment: 1,
        })?;

        let bytes = {
            let src = state
                .buffers
                .get(&op.src_resource)
                .ok_or(UmaBridgeError::HostD3d12 {
                    hresult: -2147024809, // E_INVALIDARG
                    context: "mock copy source resource not found",
                })?;
            let src_end = src_start
                .checked_add(len)
                .filter(|&e| e <= src.storage.len())
                .ok_or(UmaBridgeError::InvalidLayout {
                    size: op.size,
                    alignment: 1,
                })?;
            src.storage[src_start..src_end].to_vec()
        };

        let dst = state
            .buffers
            .get_mut(&op.dst_resource)
            .ok_or(UmaBridgeError::HostD3d12 {
                hresult: -2147024809, // E_INVALIDARG
                context: "mock copy destination resource not found",
            })?;
        let dst_end = dst_start
            .checked_add(len)
            .filter(|&e| e <= dst.storage.len())
            .ok_or(UmaBridgeError::InvalidLayout {
                size: op.size,
                alignment: 1,
            })?;
        dst.storage[dst_start..dst_end].copy_from_slice(&bytes);
        Ok(())
    }

    /// Returns `(rebar_created, staging_created, default_created, destroyed)`.
    #[must_use]
    pub fn slab_counts(&self) -> (usize, usize, usize, usize) {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        (
            state.created_rebar_slabs,
            state.created_staging_slabs,
            state.created_default_slabs,
            state.destroyed_slabs,
        )
    }
}

impl GpuHeapBackend for MockHeapBackend {
    fn probe_capabilities(&self) -> RebarCapabilities {
        self.caps
    }

    fn create_buffer_slab(
        &self,
        size: u64,
        tier: MemoryTier,
    ) -> Result<(ResourceId, GpuVirtualAddress, Option<NonNull<u8>>), UmaBridgeError> {
        if size == 0 {
            return Err(UmaBridgeError::InvalidLayout {
                size: 0,
                alignment: 1,
            });
        }
        // Mirror hardware rule: discrete non-ReBAR adapters cannot create ReBAR slabs.
        if tier == MemoryTier::RebarWriteCombine
            && !self.caps.gpu_upload_heap_supported
            && !self.caps.is_uma
        {
            return Err(UmaBridgeError::OutOfMemory {
                requested: size,
                tier,
            });
        }

        let len = usize::try_from(size).map_err(|_| UmaBridgeError::OutOfMemory {
            requested: size,
            tier,
        })?;
        let mut storage = vec![0u8; len].into_boxed_slice();
        let cpu_ptr = match tier {
            MemoryTier::RebarWriteCombine | MemoryTier::StagingCopyRing => {
                NonNull::new(storage.as_mut_ptr())
            }
            MemoryTier::DeviceLocalDefault => None,
        };

        let id = ResourceId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let va_step = align_up(size, 65536).unwrap_or(size);
        let gpu_va = GpuVirtualAddress(self.next_va.fetch_add(va_step, Ordering::Relaxed));

        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match tier {
            MemoryTier::RebarWriteCombine => state.created_rebar_slabs += 1,
            MemoryTier::StagingCopyRing => state.created_staging_slabs += 1,
            MemoryTier::DeviceLocalDefault => state.created_default_slabs += 1,
        }
        state.buffers.insert(
            id,
            MockBufferEntry {
                tier,
                gpu_va,
                storage,
            },
        );
        let _ = state.buffers.get(&id).map(|e| (e.tier, e.gpu_va));

        Ok((id, gpu_va, cpu_ptr))
    }

    fn destroy_buffer_slab(&self, resource_id: ResourceId) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.buffers.remove(&resource_id).is_some() {
            state.destroyed_slabs += 1;
        }
    }
}

impl GpuHeapBackend for std::sync::Arc<MockHeapBackend> {
    fn probe_capabilities(&self) -> RebarCapabilities {
        (**self).probe_capabilities()
    }

    fn create_buffer_slab(
        &self,
        size: u64,
        tier: MemoryTier,
    ) -> Result<(ResourceId, GpuVirtualAddress, Option<NonNull<u8>>), UmaBridgeError> {
        (**self).create_buffer_slab(size, tier)
    }

    fn destroy_buffer_slab(&self, resource_id: ResourceId) {
        (**self).destroy_buffer_slab(resource_id);
    }
}

/// Deterministic headless [`CopyQueueBackend`] recording submitted batches and
/// optionally executing byte copies against a linked [`MockHeapBackend`].
pub struct MockCopyQueueBackend {
    queue_id: u32,
    heap: Option<std::sync::Arc<MockHeapBackend>>,
    next_fence: AtomicU64,
    completed_fence: AtomicU64,
    submitted_batches: Mutex<Vec<Vec<BufferCopyOp>>>,
}

impl MockCopyQueueBackend {
    /// Creates a standalone mock copy queue (records ops without moving bytes).
    #[must_use]
    pub fn new() -> Self {
        Self {
            queue_id: 1,
            heap: None,
            next_fence: AtomicU64::new(1),
            completed_fence: AtomicU64::new(0),
            submitted_batches: Mutex::new(Vec::new()),
        }
    }

    /// Creates a mock copy queue linked to `heap` so submitted copies execute
    /// byte-exact memory transfers into mock destination resources.
    #[must_use]
    pub fn with_heap(heap: std::sync::Arc<MockHeapBackend>) -> Self {
        Self {
            queue_id: 1,
            heap: Some(heap),
            next_fence: AtomicU64::new(1),
            completed_fence: AtomicU64::new(0),
            submitted_batches: Mutex::new(Vec::new()),
        }
    }

    /// Advances the simulated GPU completed fence value monotonically.
    pub fn complete_up_to(&self, value: u64) {
        self.completed_fence.fetch_max(value, Ordering::Release);
    }

    /// Returns a snapshot of all submitted copy batches.
    #[must_use]
    pub fn submitted_batches(&self) -> Vec<Vec<BufferCopyOp>> {
        self.submitted_batches
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

impl Default for MockCopyQueueBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CopyQueueBackend for MockCopyQueueBackend {
    fn submit_buffer_copies(&self, copies: &[BufferCopyOp]) -> Result<GpuFenceToken, UmaBridgeError> {
        if let Some(heap) = &self.heap {
            for &op in copies {
                heap.execute_copy(op)?;
            }
        }
        let fence_val = self.next_fence.fetch_add(1, Ordering::AcqRel);
        self.submitted_batches
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(copies.to_vec());
        Ok(GpuFenceToken {
            queue_id: self.queue_id,
            value: fence_val,
        })
    }

    fn completed_fence_value(&self) -> u64 {
        self.completed_fence.load(Ordering::Acquire)
    }
}

// ---------------------------------------------------------------------------
// Thin audited Windows D3D12 COM backend (`d3d12_backend`)
// ---------------------------------------------------------------------------

#[cfg(windows)]
pub mod d3d12_backend {
    //! Thin audited D3D12 COM wrappers isolating all `unsafe` FFI calls for
    //! heap slab creation, async Copy Queue submissions, and timeline fences.

    use std::collections::HashMap;
    use std::ptr::NonNull;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0, WAIT_TIMEOUT};
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_COMMAND_LIST_TYPE_COPY, D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE,
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE,
        D3D12_FEATURE_ARCHITECTURE1, D3D12_FEATURE_D3D12_OPTIONS16,
        D3D12_FEATURE_DATA_ARCHITECTURE1, D3D12_FEATURE_DATA_D3D12_OPTIONS16,
        D3D12_FENCE_FLAG_NONE, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES,
        D3D12_HEAP_TYPE_CUSTOM, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_TYPE_GPU_UPLOAD,
        D3D12_HEAP_TYPE_READBACK, D3D12_MEMORY_POOL_L0, D3D12_MEMORY_POOL_UNKNOWN, D3D12_RANGE,
        D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12CreateDevice, ID3D12CommandAllocator,
        ID3D12CommandList, ID3D12CommandQueue, ID3D12Device, ID3D12Fence,
        ID3D12GraphicsCommandList, ID3D12Resource,
    };
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO,
        IDXGIAdapter1, IDXGIAdapter3, IDXGIFactory4,
    };
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
    use windows::core::Interface;

    use crate::graphics::contracts::{
        BufferCopyOp, CopyQueueBackend, GpuFenceToken, GpuHeapBackend, GpuTimelineFence,
        GpuVirtualAddress, MemoryTier, RebarCapabilities, ResourceId, UmaBridgeError,
    };

    #[inline]
    fn d3d12_err(err: windows::core::Error, context: &'static str) -> UmaBridgeError {
        UmaBridgeError::HostD3d12 {
            hresult: err.code().0,
            context,
        }
    }

    /// RAII wrapper for a Win32 auto-reset event handle.
    struct OwnedEvent(HANDLE);

    // SAFETY: The Win32 `HANDLE` is only waited on while holding the owning mutex.
    unsafe impl Send for OwnedEvent {}

    impl OwnedEvent {
        fn new() -> Result<Self, UmaBridgeError> {
            // SAFETY: Creating an unnamed auto-reset Win32 event with default security.
            let handle = unsafe { CreateEventW(None, false, false, None) }
                .map_err(|e| d3d12_err(e, "CreateEventW"))?;
            Ok(Self(handle))
        }
    }

    impl Drop for OwnedEvent {
        fn drop(&mut self) {
            if !self.0.is_invalid() {
                // SAFETY: `self.0` was created by `CreateEventW` and is closed exactly once.
                let _ = unsafe { CloseHandle(self.0) };
            }
        }
    }

    /// Safe RAII wrapper around `ID3D12Fence` implementing [`GpuTimelineFence`].
    pub struct D3D12TimelineFence {
        queue_id: u32,
        fence: ID3D12Fence,
        wait_event: Mutex<OwnedEvent>,
    }

    // SAFETY: `ID3D12Fence` is free-threaded and `wait_event` is mutex-protected.
    unsafe impl Send for D3D12TimelineFence {}
    // SAFETY: `ID3D12Fence` is free-threaded and `wait_event` is mutex-protected.
    unsafe impl Sync for D3D12TimelineFence {}

    impl D3D12TimelineFence {
        /// Creates a new timeline fence initialized to `0`.
        pub fn new(device: &ID3D12Device, queue_id: u32) -> Result<Self, UmaBridgeError> {
            // SAFETY: Valid D3D12 device creating an unshared fence starting at 0.
            let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
                .map_err(|e| d3d12_err(e, "ID3D12Device::CreateFence"))?;
            let wait_event = OwnedEvent::new()?;
            Ok(Self {
                queue_id,
                fence,
                wait_event: Mutex::new(wait_event),
            })
        }

        /// Returns a reference to the underlying `ID3D12Fence`.
        #[must_use]
        pub fn raw_fence(&self) -> &ID3D12Fence {
            &self.fence
        }
    }

    impl GpuTimelineFence for D3D12TimelineFence {
        fn queue_id(&self) -> u32 {
            self.queue_id
        }

        fn completed_value(&self) -> u64 {
            // SAFETY: `ID3D12Fence::GetCompletedValue` is thread-safe on a live fence.
            unsafe { self.fence.GetCompletedValue() }
        }

        fn is_completed(&self, token: GpuFenceToken) -> bool {
            token.queue_id == self.queue_id && self.completed_value() >= token.value
        }

        fn wait_cpu(
            &self,
            token: GpuFenceToken,
            timeout: Duration,
        ) -> Result<bool, UmaBridgeError> {
            if self.completed_value() >= token.value {
                return Ok(true);
            }

            let deadline = std::time::Instant::now().checked_add(timeout);
            let event = self
                .wait_event
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            loop {
                if self.completed_value() >= token.value {
                    return Ok(true);
                }

                // SAFETY: `event.0` is a valid Win32 event handle kept alive under lock.
                unsafe {
                    self.fence
                        .SetEventOnCompletion(token.value, event.0)
                        .map_err(|e| d3d12_err(e, "ID3D12Fence::SetEventOnCompletion"))?;
                }

                if self.completed_value() >= token.value {
                    return Ok(true);
                }

                let timeout_ms = match deadline {
                    Some(dl) => {
                        let Some(rem) = dl.checked_duration_since(std::time::Instant::now()) else {
                            return Ok(self.completed_value() >= token.value);
                        };
                        rem.as_millis().min((u32::MAX - 1) as u128) as u32
                    }
                    None => u32::MAX - 1,
                };
                // SAFETY: `event.0` is valid and owned by `self`.
                let wait_res = unsafe { WaitForSingleObject(event.0, timeout_ms) };
                if self.completed_value() >= token.value {
                    return Ok(true);
                }
                if wait_res == WAIT_TIMEOUT {
                    return Ok(false);
                }
                if wait_res != WAIT_OBJECT_0 {
                    return Err(UmaBridgeError::HostD3d12 {
                        hresult: -2147467259, // E_FAIL
                        context: "WaitForSingleObject failed on D3D12 fence event",
                    });
                }
                // `WAIT_OBJECT_0` fired early from a stale signal left by a prior timed-out wait;
                // loop and re-arm `SetEventOnCompletion` until `completed_value() >= token.value`.
            }
        }
    }

    struct TrackedD3D12Buffer {
        resource: ID3D12Resource,
        is_mapped: bool,
    }

    // SAFETY: `ID3D12Resource` COM handle is free-threaded and unmapped on drop.
    unsafe impl Send for TrackedD3D12Buffer {}

    impl Drop for TrackedD3D12Buffer {
        fn drop(&mut self) {
            if self.is_mapped {
                // SAFETY: Subresource 0 was mapped at creation and is unmapped once on drop.
                unsafe {
                    self.resource.Unmap(0, None);
                }
            }
        }
    }

    /// Safe RAII [`GpuHeapBackend`] backed by a real `ID3D12Device`.
    pub struct D3D12HeapBackend {
        device: ID3D12Device,
        caps: RebarCapabilities,
        next_id: AtomicU64,
        resources: Mutex<HashMap<ResourceId, TrackedD3D12Buffer>>,
    }

    // SAFETY: `ID3D12Device` is free-threaded and `resources` is mutex-protected.
    unsafe impl Send for D3D12HeapBackend {}
    // SAFETY: `ID3D12Device` is free-threaded and `resources` is mutex-protected.
    unsafe impl Sync for D3D12HeapBackend {}

    impl D3D12HeapBackend {
        /// Probes D3D12 capabilities (`OPTIONS16.GPUUploadHeapSupported`,
        /// `ARCHITECTURE1.UMA`, and optional DXGI VRAM budget) and wraps `device`.
        #[must_use]
        pub fn new(device: ID3D12Device, adapter: Option<&IDXGIAdapter1>) -> Self {
            let caps = probe_device_capabilities(&device, adapter);
            Self {
                device,
                caps,
                next_id: AtomicU64::new(1),
                resources: Mutex::new(HashMap::new()),
            }
        }

        /// Attempts to create a D3D12 device on the default hardware adapter,
        /// falling back to the Microsoft WARP software adapter if needed.
        pub fn try_create_device_or_warp()
        -> Result<(ID3D12Device, Option<IDXGIAdapter1>, bool), UmaBridgeError> {
            let mut hw_device: Option<ID3D12Device> = None;
            // SAFETY: Standard D3D12 factory/device creation with feature level 12_0.
            let factory: Option<IDXGIFactory4> = unsafe { CreateDXGIFactory1().ok() };
            if let Some(ref fac) = factory {
                // SAFETY: Enumerating adapter 0 via DXGI factory.
                if let Ok(adapter) = unsafe { fac.EnumAdapters1(0) } {
                    // SAFETY: Creating a D3D12 device on adapter 0.
                    if unsafe {
                        D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut hw_device)
                    }
                    .is_ok()
                        && let Some(dev) = hw_device
                    {
                        return Ok((dev, Some(adapter), false));
                    }
                }
                // Fall back to WARP software adapter when hardware is unavailable.
                // SAFETY: Querying DXGI WARP adapter.
                if let Ok(warp) = unsafe { fac.EnumWarpAdapter::<IDXGIAdapter1>() } {
                    let mut warp_dev: Option<ID3D12Device> = None;
                    // SAFETY: Creating a D3D12 device on the WARP adapter.
                    unsafe {
                        D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_12_0, &mut warp_dev)
                            .map_err(|e| d3d12_err(e, "D3D12CreateDevice(WARP)"))?;
                    }
                    if let Some(dev) = warp_dev {
                        return Ok((dev, Some(warp), true));
                    }
                }
            }

            let mut default_dev: Option<ID3D12Device> = None;
            // SAFETY: Creating default D3D12 device with `None` adapter.
            unsafe {
                D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut default_dev)
                    .map_err(|e| d3d12_err(e, "D3D12CreateDevice(default)"))?;
            }
            let dev = default_dev.ok_or(UmaBridgeError::HostD3d12 {
                hresult: -2147467259,
                context: "D3D12CreateDevice returned null device",
            })?;
            Ok((dev, None, false))
        }

        /// Returns a reference to the wrapped `ID3D12Device`.
        #[must_use]
        pub fn device(&self) -> &ID3D12Device {
            &self.device
        }

        /// Clones the live `ID3D12Resource` handle registered under `id`.
        #[must_use]
        pub fn get_resource(&self, id: ResourceId) -> Option<ID3D12Resource> {
            let guard = self.resources.lock().unwrap_or_else(|e| e.into_inner());
            guard.get(&id).map(|entry| entry.resource.clone())
        }

        /// Creates a CPU-readable `D3D12_HEAP_TYPE_READBACK` buffer (used by
        /// integration tests to verify byte-exact GPU Copy Queue roundtrips).
        pub fn create_readback_buffer(
            &self,
            size: u64,
        ) -> Result<(ResourceId, NonNull<u8>), UmaBridgeError> {
            if size == 0 {
                return Err(UmaBridgeError::InvalidLayout {
                    size: 0,
                    alignment: 1,
                });
            }
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                ..Default::default()
            };
            let desc = buffer_desc(size);
            let mut resource: Option<ID3D12Resource> = None;
            // SAFETY: Creating a committed readback buffer in COPY_DEST state.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut resource,
                    )
                    .map_err(|e| d3d12_err(e, "CreateCommittedResource(READBACK)"))?;
            }
            let res = resource.ok_or(UmaBridgeError::HostD3d12 {
                hresult: -2147467259,
                context: "CreateCommittedResource(READBACK) returned None",
            })?;
            let mut mapped_ptr = std::ptr::null_mut();
            // SAFETY: Mapping subresource 0 of a newly created READBACK buffer.
            unsafe {
                res.Map(0, None, Some(&mut mapped_ptr))
                    .map_err(|e| d3d12_err(e, "ID3D12Resource::Map(READBACK)"))?;
            }
            let ptr = NonNull::new(mapped_ptr.cast::<u8>()).ok_or(UmaBridgeError::HostD3d12 {
                hresult: -2147467259,
                context: "Map(READBACK) returned null pointer",
            })?;

            let id = ResourceId(self.next_id.fetch_add(1, Ordering::Relaxed));
            self.resources
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(
                    id,
                    TrackedD3D12Buffer {
                        resource: res,
                        is_mapped: true,
                    },
                );
            Ok((id, ptr))
        }
    }

    fn buffer_desc(size: u64) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
            ..Default::default()
        }
    }

    fn probe_device_capabilities(
        device: &ID3D12Device,
        adapter: Option<&IDXGIAdapter1>,
    ) -> RebarCapabilities {
        let mut opts16 = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
        // SAFETY: `CheckFeatureSupport` writes into `opts16` of exact byte size.
        let opts16_ok = unsafe {
            device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS16,
                    (&raw mut opts16).cast(),
                    size_of_val(&opts16) as u32,
                )
                .is_ok()
        };
        let gpu_upload_heap_supported = opts16_ok && opts16.GPUUploadHeapSupported.as_bool();

        let mut arch1 = D3D12_FEATURE_DATA_ARCHITECTURE1::default();
        // SAFETY: `CheckFeatureSupport` writes into `arch1` of exact byte size.
        let arch_ok = unsafe {
            device
                .CheckFeatureSupport(
                    D3D12_FEATURE_ARCHITECTURE1,
                    (&raw mut arch1).cast(),
                    size_of_val(&arch1) as u32,
                )
                .is_ok()
        };
        let is_uma = arch_ok && arch1.UMA.as_bool();

        let mut dedicated_vram_bytes = 0u64;
        let mut local_budget = 0u64;
        if let Some(adp) = adapter {
            // SAFETY: Plain DXGI adapter description query.
            if let Ok(desc) = unsafe { adp.GetDesc1() } {
                dedicated_vram_bytes = desc.DedicatedVideoMemory as u64;
            }
            if let Ok(adp3) = adp.cast::<IDXGIAdapter3>() {
                let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                // SAFETY: Querying local segment memory budget from IDXGIAdapter3.
                if unsafe {
                    adp3.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info)
                }
                .is_ok()
                {
                    local_budget = info.Budget;
                }
            }
        }

        let rebar_budget_bytes = if gpu_upload_heap_supported || is_uma {
            let base = if local_budget > 0 {
                local_budget / 4
            } else if dedicated_vram_bytes > 0 {
                dedicated_vram_bytes / 4
            } else {
                256 * 1024 * 1024
            };
            base.clamp(64 * 1024 * 1024, 512 * 1024 * 1024)
        } else {
            0
        };

        RebarCapabilities {
            gpu_upload_heap_supported,
            is_uma,
            dedicated_vram_bytes,
            rebar_budget_bytes,
        }
    }

    impl GpuHeapBackend for D3D12HeapBackend {
        fn probe_capabilities(&self) -> RebarCapabilities {
            self.caps
        }

        fn create_buffer_slab(
            &self,
            size: u64,
            tier: MemoryTier,
        ) -> Result<(ResourceId, GpuVirtualAddress, Option<NonNull<u8>>), UmaBridgeError> {
            if size == 0 {
                return Err(UmaBridgeError::InvalidLayout {
                    size: 0,
                    alignment: 1,
                });
            }

            let (heap_props, should_map) = match tier {
                MemoryTier::RebarWriteCombine => {
                    if self.caps.gpu_upload_heap_supported {
                        // Discrete ReBAR (or UMA with Agility SDK GPU Upload Heap):
                        // D3D12_HEAP_TYPE_GPU_UPLOAD maps directly into device VRAM.
                        (
                            D3D12_HEAP_PROPERTIES {
                                Type: D3D12_HEAP_TYPE_GPU_UPLOAD,
                                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                                MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
                                ..Default::default()
                            },
                            true,
                        )
                    } else if self.caps.is_uma {
                        // On UMA adapters (including WARP), L0 is the unified memory pool
                        // (L1 is unavailable on UMA per the D3D12 specification).
                        // We NEVER combine CUSTOM + L1 + WRITE_COMBINE on discrete
                        // (`UMA == false`) adapters.
                        (
                            D3D12_HEAP_PROPERTIES {
                                Type: D3D12_HEAP_TYPE_CUSTOM,
                                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE,
                                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                                ..Default::default()
                            },
                            true,
                        )
                    } else {
                        return Err(UmaBridgeError::OutOfMemory {
                            requested: size,
                            tier,
                        });
                    }
                }
                MemoryTier::StagingCopyRing => (
                    D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_CUSTOM,
                        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE,
                        MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                        ..Default::default()
                    },
                    true,
                ),
                MemoryTier::DeviceLocalDefault => (
                    D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    false,
                ),
            };

            let desc = buffer_desc(size);
            let mut resource: Option<ID3D12Resource> = None;
            // SAFETY: Valid heap properties and buffer descriptor for a committed resource.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap_props,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut resource,
                    )
                    .map_err(|e| d3d12_err(e, "ID3D12Device::CreateCommittedResource"))?;
            }
            let res = resource.ok_or(UmaBridgeError::HostD3d12 {
                hresult: -2147467259,
                context: "CreateCommittedResource returned None",
            })?;

            // SAFETY: `GetGPUVirtualAddress` is valid on committed buffer resources.
            let gpu_va = GpuVirtualAddress(unsafe { res.GetGPUVirtualAddress() });

            let cpu_ptr = if should_map {
                let mut raw_ptr = std::ptr::null_mut();
                // Empty read range signals write-only CPU access for Write-Combine buffers.
                let write_only_range = D3D12_RANGE { Begin: 0, End: 0 };
                // SAFETY: Subresource 0 of a CPU-visible committed buffer is valid to map.
                unsafe {
                    res.Map(0, Some(&write_only_range), Some(&mut raw_ptr))
                        .map_err(|e| d3d12_err(e, "ID3D12Resource::Map"))?;
                }
                let non_null =
                    NonNull::new(raw_ptr.cast::<u8>()).ok_or(UmaBridgeError::HostD3d12 {
                        hresult: -2147467259,
                        context: "ID3D12Resource::Map returned null pointer",
                    })?;
                Some(non_null)
            } else {
                None
            };

            let id = ResourceId(self.next_id.fetch_add(1, Ordering::Relaxed));
            self.resources
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(
                    id,
                    TrackedD3D12Buffer {
                        resource: res,
                        is_mapped: should_map,
                    },
                );

            Ok((id, gpu_va, cpu_ptr))
        }

        fn destroy_buffer_slab(&self, resource_id: ResourceId) {
            self.resources
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&resource_id);
        }
    }

    impl GpuHeapBackend for Arc<D3D12HeapBackend> {
        fn probe_capabilities(&self) -> RebarCapabilities {
            (**self).probe_capabilities()
        }

        fn create_buffer_slab(
            &self,
            size: u64,
            tier: MemoryTier,
        ) -> Result<(ResourceId, GpuVirtualAddress, Option<NonNull<u8>>), UmaBridgeError> {
            (**self).create_buffer_slab(size, tier)
        }

        fn destroy_buffer_slab(&self, resource_id: ResourceId) {
            (**self).destroy_buffer_slab(resource_id);
        }
    }

    struct CopyRecorderSlot {
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        last_fence: u64,
    }

    // SAFETY: D3D12 command allocators/lists are protected by `recorders` mutex
    // and only reset once `fence.completed_value() >= last_fence`.
    unsafe impl Send for CopyRecorderSlot {}

    /// Safe RAII [`CopyQueueBackend`] submitting batched `CopyBufferRegion`
    /// commands over a dedicated `D3D12_COMMAND_LIST_TYPE_COPY` queue.
    pub struct D3D12CopyQueueBackend {
        heap: Arc<D3D12HeapBackend>,
        queue: ID3D12CommandQueue,
        fence: Arc<D3D12TimelineFence>,
        next_fence: AtomicU64,
        recorders: Mutex<Vec<CopyRecorderSlot>>,
    }

    // SAFETY: D3D12 queues and fences are free-threaded; command recorders are
    // synchronized by `self.recorders`.
    unsafe impl Send for D3D12CopyQueueBackend {}
    // SAFETY: As above.
    unsafe impl Sync for D3D12CopyQueueBackend {}

    impl D3D12CopyQueueBackend {
        /// Creates a dedicated `D3D12_COMMAND_LIST_TYPE_COPY` queue and timeline
        /// fence on `heap.device()`.
        pub fn new(heap: Arc<D3D12HeapBackend>, queue_id: u32) -> Result<Self, UmaBridgeError> {
            let desc = D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_COPY,
                Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
                ..Default::default()
            };
            // SAFETY: Creating a dedicated copy command queue on a live D3D12 device.
            let queue: ID3D12CommandQueue = unsafe { heap.device().CreateCommandQueue(&desc) }
                .map_err(|e| d3d12_err(e, "ID3D12Device::CreateCommandQueue(COPY)"))?;
            let fence = Arc::new(D3D12TimelineFence::new(heap.device(), queue_id)?);

            Ok(Self {
                heap,
                queue,
                fence,
                next_fence: AtomicU64::new(1),
                recorders: Mutex::new(Vec::new()),
            })
        }

        /// Returns the timeline fence signalled after each copy batch.
        #[must_use]
        pub fn fence(&self) -> &Arc<D3D12TimelineFence> {
            &self.fence
        }

        fn create_recorder_slot(&self) -> Result<CopyRecorderSlot, UmaBridgeError> {
            // SAFETY: Creating a copy command allocator and closed copy command list.
            let allocator: ID3D12CommandAllocator = unsafe {
                self.heap
                    .device()
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)
            }
            .map_err(|e| d3d12_err(e, "CreateCommandAllocator(COPY)"))?;

            // SAFETY: Creating a copy command list bound to `allocator`.
            let list: ID3D12GraphicsCommandList = unsafe {
                self.heap.device().CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_COPY,
                    &allocator,
                    None,
                )
            }
            .map_err(|e| d3d12_err(e, "CreateCommandList(COPY)"))?;

            // SAFETY: Closing freshly opened command list so all slots share the closed invariant.
            unsafe {
                list.Close()
                    .map_err(|e| d3d12_err(e, "ID3D12GraphicsCommandList::Close(init)"))?;
            }

            Ok(CopyRecorderSlot {
                allocator,
                list,
                last_fence: 0,
            })
        }
    }

    impl CopyQueueBackend for D3D12CopyQueueBackend {
        fn submit_buffer_copies(
            &self,
            copies: &[BufferCopyOp],
        ) -> Result<GpuFenceToken, UmaBridgeError> {
            // Resolve COM resource handles before touching command list state.
            let mut resolved = Vec::with_capacity(copies.len());
            for &op in copies {
                let src = self
                    .heap
                    .get_resource(op.src_resource)
                    .ok_or(UmaBridgeError::HostD3d12 {
                        hresult: -2147024809,
                        context: "copy source ResourceId not found in D3D12HeapBackend",
                    })?;
                let dst = self
                    .heap
                    .get_resource(op.dst_resource)
                    .ok_or(UmaBridgeError::HostD3d12 {
                        hresult: -2147024809,
                        context: "copy destination ResourceId not found in D3D12HeapBackend",
                    })?;
                resolved.push((src, dst, op));
            }

            let mut recorders = self
                .recorders
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let completed = self.fence.completed_value();

            let slot_idx = match recorders.iter().position(|s| s.last_fence <= completed) {
                Some(idx) => idx,
                None => {
                    let slot = self.create_recorder_slot()?;
                    recorders.push(slot);
                    recorders.len() - 1
                }
            };

            let slot = &mut recorders[slot_idx];
            // SAFETY: `slot.last_fence <= completed`, so the GPU is no longer
            // executing commands from `slot.allocator` or `slot.list`.
            unsafe {
                slot.allocator
                    .Reset()
                    .map_err(|e| d3d12_err(e, "ID3D12CommandAllocator::Reset"))?;
                slot.list
                    .Reset(&slot.allocator, None)
                    .map_err(|e| d3d12_err(e, "ID3D12GraphicsCommandList::Reset"))?;

                for (src, dst, op) in &resolved {
                    slot.list
                        .CopyBufferRegion(dst, op.dst_offset, src, op.src_offset, op.size);
                }

                slot.list
                    .Close()
                    .map_err(|e| d3d12_err(e, "ID3D12GraphicsCommandList::Close"))?;
            }

            let cmd_list: ID3D12CommandList = slot
                .list
                .cast()
                .map_err(|e| d3d12_err(e, "ID3D12GraphicsCommandList -> ID3D12CommandList"))?;
            let fence_val = self.next_fence.fetch_add(1, Ordering::AcqRel);

            // SAFETY: `cmd_list` is closed and valid; `self.fence` is a live fence
            // signalled monotonically on `self.queue`.
            unsafe {
                self.queue.ExecuteCommandLists(&[Some(cmd_list)]);
                self.queue
                    .Signal(self.fence.raw_fence(), fence_val)
                    .map_err(|e| d3d12_err(e, "ID3D12CommandQueue::Signal"))?;
            }

            slot.last_fence = fence_val;

            Ok(GpuFenceToken {
                queue_id: self.fence.queue_id(),
                value: fence_val,
            })
        }

        fn completed_fence_value(&self) -> u64 {
            self.fence.completed_value()
        }

        fn drain_gpu(&self) {
            let last = self.next_fence.load(Ordering::Acquire).saturating_sub(1);
            if last > 0 {
                let _ = self.fence.wait_cpu(
                    GpuFenceToken {
                        queue_id: self.fence.queue_id(),
                        value: last,
                    },
                    Duration::from_secs(30),
                );
            }
        }
    }

    impl Drop for D3D12CopyQueueBackend {
        fn drop(&mut self) {
            self.drain_gpu();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebar_slab_coalesces_left_and_right_neighbors() {
        let heap = MockHeapBackend::rtx_rebar(4096);
        let copy = MockCopyQueueBackend::new();
        let alloc = RebarUploadAllocator::new(
            heap,
            copy,
            RebarAllocatorConfig {
                rebar_slab_size: 4096,
                staging_ring_capacity: 4096,
                min_alignment: 1,
            },
        )
        .unwrap();

        let s1 = alloc.allocate(1024, 256).unwrap();
        let s2 = alloc.allocate(1024, 256).unwrap();
        let s3 = alloc.allocate(1024, 256).unwrap();
        assert_eq!(s1.tier, MemoryTier::RebarWriteCombine);
        assert_eq!(s1.heap_offset, 0);
        assert_eq!(s2.heap_offset, 1024);
        assert_eq!(s3.heap_offset, 2048);

        // Retire outer slices first, then middle slice -> should coalesce into 1 single 4096 B range.
        alloc.retire_slice(s1, GpuFenceToken { queue_id: 0, value: 1 });
        alloc.retire_slice(s3, GpuFenceToken { queue_id: 0, value: 1 });
        assert_eq!(alloc.reclaim_completed(1), 2);
        assert_eq!(alloc.rebar_free_fragment_count(), 2);

        alloc.retire_slice(s2, GpuFenceToken { queue_id: 0, value: 2 });
        assert_eq!(alloc.reclaim_completed(2), 1);
        assert_eq!(alloc.rebar_free_fragment_count(), 1);
        assert_eq!(alloc.rebar_free_bytes(), 4096);

        // A full 4096 B allocation now succeeds in the coalesced slab.
        let full = alloc.allocate(4096, 256).unwrap();
        assert_eq!(full.tier, MemoryTier::RebarWriteCombine);
        assert_eq!(full.heap_offset, 0);
    }

    #[test]
    fn ring_wrap_and_non_blocking_exhaustion() {
        let heap = MockHeapBackend::discrete_no_rebar();
        let copy = MockCopyQueueBackend::new();
        let alloc = RebarUploadAllocator::new(
            heap,
            copy,
            RebarAllocatorConfig {
                rebar_slab_size: 4096,
                staging_ring_capacity: 1024,
                min_alignment: 1,
            },
        )
        .unwrap();

        let a = alloc.allocate(600, 16).unwrap();
        assert_eq!(a.tier, MemoryTier::StagingCopyRing);
        assert_eq!(a.heap_offset, 0);

        // Only 424 bytes left; requesting 512 must return RingExhausted without blocking.
        assert!(matches!(
            alloc.allocate(512, 16),
            Err(UmaBridgeError::RingExhausted { requested: 512, .. })
        ));

        // Retire and reclaim `a`, then allocate 512 — must wrap cleanly to offset 0.
        alloc.retire_slice(a, GpuFenceToken { queue_id: 1, value: 1 });
        assert_eq!(alloc.reclaim_completed(0), 0);
        assert_eq!(alloc.reclaim_completed(1), 1);

        let b = alloc.allocate(512, 64).unwrap();
        assert_eq!(b.heap_offset, 0);
    }
}
