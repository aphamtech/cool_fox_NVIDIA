//! Subsystem B: Velocity I/O (`DStorageX`) Async GPU Pipeline.
//!
//! Owned by `@velocity-io`.
//!
//! Provides:
//! - [`dstorage_thunk`]: Exact Xbox Series X|S `DStorageRequestX` / `DStorageQueueDescX`
//!   ABI decoding, zero-stall lock-free `VelocityFactory` / `VelocityQueue` /
//!   `VelocityStatusArray` controllers, fence-gated `PinnedUploadRing`, and
//!   platform-isolated OS async file I/O (`Win32OverlappedIo` over IOCP on Windows,
//!   `IoUringBackend` on Linux, and deterministic `InMemoryAsyncIo`).
//! - [`gpu_decompress_queue`]: Dedicated Async Compute Queue (`D3D12_COMMAND_LIST_TYPE_COMPUTE`)
//!   dispatcher synchronized via timeline fences, pluggable [`DecompressionRegistry`],
//!   standard `Uncompressed` / `Zlib` / `LZ4` / `GDeflate` v1 backends, and a strictly
//!   typed, documented `BcPackStubBackend` stub (per `.agents/rules/xbox-compat.md` Rule 1).

pub mod dstorage_thunk;
pub mod gpu_decompress_queue;

pub use dstorage_thunk::{
    DStorageQueueDescX, DStorageRequestX, DefaultPlatformHooks, InMemoryAsyncIo, PinnedLease,
    PinnedUploadRing, VelocityFactory, VelocityFile, VelocityQueue, VelocityQueueConfig,
    VelocityStatusArray,
};
#[cfg(target_os = "linux")]
pub use dstorage_thunk::IoUringBackend;
#[cfg(windows)]
pub use dstorage_thunk::Win32OverlappedIo;
pub use gpu_decompress_queue::{
    BcPackStubBackend, DecompressionRegistry, GDeflateDecompressionBackend, GpuDecompressQueue,
    Lz4DecompressionBackend, MockComputeQueueBackend, UncompressedCopyBackend,
    ZlibDecompressionBackend, encode_gdeflate_v1,
};
#[cfg(windows)]
pub use gpu_decompress_queue::d3d12_compute;
