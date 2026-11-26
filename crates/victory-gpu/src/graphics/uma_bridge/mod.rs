//! Subsystem A: UMA-to-Discrete Memory & Barrier Translator.
//!
//! Bridges Xbox Series X|S unified memory (UMA) and permissive barrier patterns
//! onto discrete PC GPUs (NVIDIA RTX / D3D12):
//!
//! - [`rebar_allocator`]: ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD` / UMA WC) slab
//!   sub-allocator with automatic non-ReBAR / budget-spillover pinned WC+L0
//!   ring buffer drained asynchronously over a `D3D12_COMMAND_LIST_TYPE_COPY` queue.
//! - [`aliasing_tracker`]: `O(log N + K)` interval tracker for overlapping
//!   `CreatePlacedResource` / `CreatePlacedResourceX` ranges emitting minimal
//!   `D3D12_RESOURCE_BARRIER_TYPE_ALIASING` transitions.
//! - [`barrier_batcher`]: Coalescing transition, aliasing, and UAV barrier
//!   batcher stripping Xbox compression bits, no-ops, redundant read-to-read
//!   transitions, and duplicate UAV barriers.

pub mod aliasing_tracker;
pub mod barrier_batcher;
pub mod rebar_allocator;

pub use aliasing_tracker::IntervalAliasingTracker;
#[cfg(windows)]
pub use barrier_batcher::D3D12CommandListBarrierSink;
pub use barrier_batcher::{CoalescingBarrierBatcher, RecordingBarrierSink};
#[cfg(windows)]
pub use rebar_allocator::d3d12_backend;
pub use rebar_allocator::{
    MockCopyQueueBackend, MockHeapBackend, RebarAllocatorConfig, RebarUploadAllocator,
};
