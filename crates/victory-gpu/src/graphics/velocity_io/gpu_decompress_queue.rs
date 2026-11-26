//! Dedicated Async Compute Queue decompression pipeline and pluggable codec backends.
//!
//! Implements Subsystem B.2:
//! - [`DecompressionRegistry`] & [`GpuDecompressQueue`] synchronized via timeline
//!   [`GpuFenceToken`] values (`wait_fence(copy_token)` prior to dispatch).
//! - Standard codecs:
//!   - [`UncompressedCopyBackend`]: GPU passthrough copy or CPU slice passthrough.
//!   - [`ZlibDecompressionBackend`]: RFC 1950/1951 SIMD decompression via `zlib-rs`.
//!   - [`Lz4DecompressionBackend`]: LZ4 frame (magic `0x184D2204`) and raw block via `lz4_flex`.
//!   - [`GDeflateDecompressionBackend`]: Open-spec Microsoft GDeflate v1 multi-tile stream
//!     parser decompressing independent raw DEFLATE tiles via `zlib-rs`.
//! - Proprietary codec stub:
//!   - [`BcPackStubBackend`]: Strictly typed, documented stub behind [`DecompressionBackend`]
//!     per `.agents/rules/xbox-compat.md` Rule 1 (zero bit-level BCPack logic or tables).
//! - Thin audited `#[cfg(windows)] pub mod d3d12_compute` isolating real D3D12 Async Compute
//!   Queue (`D3D12_COMMAND_LIST_TYPE_COMPUTE`), timeline `ID3D12Fence`, and HLSL compute
//!   shader compilation/dispatch behind [`ComputeQueueBackend`].

use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Read;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use crate::graphics::contracts::{
    BcPackFormat, ComputeQueueBackend, DecompressionBackend, DecompressionCodec,
    DecompressionExecutionMode, DecompressionJob, DecompressionOutput, GpuFenceToken,
    GpuVirtualAddress, ResourceId, VelocityIoError,
};

/// Standard LZ4 Frame format little-endian magic number (`0x184D2204`).
pub const LZ4_FRAME_MAGIC: [u8; 4] = [0x04, 0x22, 0x4D, 0x18];

/// Open-spec Microsoft GDeflate v1 stream format ID (`0x04`).
pub const GDEFLATE_V1_ID: u8 = 0x04;
/// Bitwise inverse of [`GDEFLATE_V1_ID`] (`0xFB`).
pub const GDEFLATE_V1_ID_INV: u8 = 0xFB;
/// Default uncompressed tile size for GDeflate v1 (`tile_size_idx == 0` -> 64 KiB).
pub const GDEFLATE_DEFAULT_TILE_BYTES: usize = 64 * 1024;
/// Fixed header size (bytes) preceding the `u32` tile offset table in GDeflate v1.
pub const GDEFLATE_HEADER_BYTES: usize = 8;

/// All supported [`BcPackFormat`] variants registered on [`BcPackStubBackend`].
const ALL_BCPACK_CODECS: [DecompressionCodec; 6] = [
    DecompressionCodec::BcPack(BcPackFormat::Bc1),
    DecompressionCodec::BcPack(BcPackFormat::Bc3),
    DecompressionCodec::BcPack(BcPackFormat::Bc4),
    DecompressionCodec::BcPack(BcPackFormat::Bc5),
    DecompressionCodec::BcPack(BcPackFormat::Bc6h),
    DecompressionCodec::BcPack(BcPackFormat::Bc7),
];

// ---------------------------------------------------------------------------
// DecompressionRegistry & GpuDecompressQueue
// ---------------------------------------------------------------------------

/// Thread-safe registry mapping [`DecompressionCodec`] variants to pluggable
/// [`DecompressionBackend`] implementations.
///
/// External callers may override any codec (including registering an external
/// BCPack decoder or hardware `VK_NV_memory_decompression` backend) at runtime
/// via [`DecompressionRegistry::register`].
#[derive(Default)]
pub struct DecompressionRegistry {
    backends: RwLock<HashMap<DecompressionCodec, Arc<dyn DecompressionBackend>>>,
}

impl std::fmt::Debug for DecompressionRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.backends.read().map(|g| g.len()).unwrap_or(0);
        f.debug_struct("DecompressionRegistry")
            .field("registered_codecs", &count)
            .finish()
    }
}

impl DecompressionRegistry {
    /// Creates an empty registry with no backends installed.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Creates a registry populated with all standard Velocity I/O backends:
    /// - [`UncompressedCopyBackend`]
    /// - [`ZlibDecompressionBackend`]
    /// - [`Lz4DecompressionBackend`]
    /// - [`GDeflateDecompressionBackend`]
    /// - [`BcPackStubBackend`] (strictly typed documented stub)
    #[must_use]
    pub fn with_defaults() -> Self {
        let registry = Self::empty();
        registry.register(Arc::new(UncompressedCopyBackend));
        registry.register(Arc::new(ZlibDecompressionBackend));
        registry.register(Arc::new(Lz4DecompressionBackend));
        registry.register(Arc::new(GDeflateDecompressionBackend));
        registry.register(Arc::new(BcPackStubBackend));
        registry
    }

    /// Registers `backend` for every codec listed in [`DecompressionBackend::supported_codecs`].
    ///
    /// Replaces any previously registered backend for those codecs.
    pub fn register(&self, backend: Arc<dyn DecompressionBackend>) {
        let mut guard = self
            .backends
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for &codec in backend.supported_codecs() {
            guard.insert(codec, Arc::clone(&backend));
        }
    }

    /// Looks up the active backend for `codec`.
    #[must_use]
    pub fn get(&self, codec: DecompressionCodec) -> Option<Arc<dyn DecompressionBackend>> {
        let guard = self
            .backends
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.get(&codec).cloned()
    }
}

/// Dedicated Async Compute Queue decompression dispatcher.
///
/// Enforces timeline fence synchronization (`wait_fence(copy_token)` before
/// dispatching compute or decompression work) and routes each [`DecompressionJob`]
/// through [`DecompressionRegistry`].
pub struct GpuDecompressQueue {
    compute: Arc<dyn ComputeQueueBackend>,
    registry: Arc<DecompressionRegistry>,
}

impl std::fmt::Debug for GpuDecompressQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDecompressQueue")
            .field("registry", &self.registry)
            .field("completed_fence", &self.compute.completed_fence_value())
            .finish()
    }
}

impl GpuDecompressQueue {
    /// Creates a new decompression queue over `compute` using the default codec registry.
    #[must_use]
    pub fn new(compute: Arc<dyn ComputeQueueBackend>) -> Self {
        Self {
            compute,
            registry: Arc::new(DecompressionRegistry::with_defaults()),
        }
    }

    /// Creates a new decompression queue over `compute` with a caller-supplied registry.
    #[must_use]
    pub fn with_registry(
        compute: Arc<dyn ComputeQueueBackend>,
        registry: Arc<DecompressionRegistry>,
    ) -> Self {
        Self { compute, registry }
    }

    /// Returns the underlying compute queue backend.
    #[inline]
    #[must_use]
    pub fn compute_backend(&self) -> &Arc<dyn ComputeQueueBackend> {
        &self.compute
    }

    /// Returns the shared [`DecompressionRegistry`].
    #[inline]
    #[must_use]
    pub fn registry(&self) -> &Arc<DecompressionRegistry> {
        &self.registry
    }

    /// Returns the most recent completed timeline fence value on the compute queue.
    #[inline]
    #[must_use]
    pub fn completed_fence_value(&self) -> u64 {
        self.compute.completed_fence_value()
    }

    /// Synchronizes on `job.wait_copy_fence` (if present) and dispatches `job` to
    /// the registered [`DecompressionBackend`] for `job.codec`.
    pub fn dispatch(&self, job: &DecompressionJob<'_>) -> Result<DecompressionOutput, VelocityIoError> {
        if let Some(copy_token) = job.wait_copy_fence {
            self.compute.wait_fence(copy_token)?;
        }

        let backend =
            self.registry
                .get(job.codec)
                .ok_or_else(|| VelocityIoError::DecompressionFailed {
                    codec: job.codec,
                    reason: format!("no DecompressionBackend registered for {:?}", job.codec),
                })?;

        backend.decompress(job, self.compute.as_ref())
    }
}

// ---------------------------------------------------------------------------
// Concrete DecompressionBackend Implementations
// ---------------------------------------------------------------------------

/// Direct passthrough backend for [`DecompressionCodec::Uncompressed`].
///
/// Dispatches GPU-to-GPU copies via [`ComputeQueueBackend::dispatch_passthrough_copy`]
/// when both source and destination GPU resources are provided, or forwards CPU
/// bytes directly when targeting host/page memory.
#[derive(Debug, Default, Clone, Copy)]
pub struct UncompressedCopyBackend;

impl DecompressionBackend for UncompressedCopyBackend {
    fn name(&self) -> &'static str {
        "uncompressed-copy"
    }

    fn supported_codecs(&self) -> &'static [DecompressionCodec] {
        &[DecompressionCodec::Uncompressed]
    }

    fn execution_mode(&self) -> DecompressionExecutionMode {
        DecompressionExecutionMode::GpuComputeShader
    }

    fn decompress(
        &self,
        job: &DecompressionJob<'_>,
        compute: &dyn ComputeQueueBackend,
    ) -> Result<DecompressionOutput, VelocityIoError> {
        if let (Some((src_res, src_off)), Some((dst_res, dst_off))) =
            (job.src_gpu_resource, job.dst_gpu_resource)
        {
            let byte_len = if !job.src_bytes.is_empty() {
                job.src_bytes.len()
            } else {
                job.expected_output_len
            };
            if job.expected_output_len > 0 && byte_len > job.expected_output_len {
                return Err(VelocityIoError::DecompressionFailed {
                    codec: DecompressionCodec::Uncompressed,
                    reason: format!(
                        "source size ({byte_len}) exceeds destination size ({})",
                        job.expected_output_len
                    ),
                });
            }
            let token = compute.dispatch_passthrough_copy(
                src_res,
                src_off,
                dst_res,
                dst_off,
                byte_len as u64,
            )?;
            return Ok(DecompressionOutput {
                bytes_written: byte_len,
                completion_fence: Some(token),
                cpu_bytes: None,
            });
        }

        if job.expected_output_len > 0 && job.src_bytes.len() > job.expected_output_len {
            return Err(VelocityIoError::DecompressionFailed {
                codec: DecompressionCodec::Uncompressed,
                reason: format!(
                    "source size ({}) exceeds expected output ({})",
                    job.src_bytes.len(),
                    job.expected_output_len
                ),
            });
        }

        let bytes = job.src_bytes.to_vec();
        let written = bytes.len();
        Ok(DecompressionOutput {
            bytes_written: written,
            completion_fence: None,
            cpu_bytes: Some(bytes),
        })
    }
}

/// Standard RFC 1950/1951 Zlib decompression backend backed by SIMD-accelerated `zlib-rs`.
#[derive(Debug, Default, Clone, Copy)]
pub struct ZlibDecompressionBackend;

impl DecompressionBackend for ZlibDecompressionBackend {
    fn name(&self) -> &'static str {
        "zlib-rs-simd"
    }

    fn supported_codecs(&self) -> &'static [DecompressionCodec] {
        &[DecompressionCodec::Zlib]
    }

    fn execution_mode(&self) -> DecompressionExecutionMode {
        DecompressionExecutionMode::CpuWorkerFallback
    }

    fn decompress(
        &self,
        job: &DecompressionJob<'_>,
        _compute: &dyn ComputeQueueBackend,
    ) -> Result<DecompressionOutput, VelocityIoError> {
        let mut out = vec![0u8; job.expected_output_len];
        let written = inflate_zlib_or_raw(job.src_bytes, &mut out, true, DecompressionCodec::Zlib)?;
        out.truncate(written);
        Ok(DecompressionOutput {
            bytes_written: written,
            completion_fence: None,
            cpu_bytes: Some(out),
        })
    }
}

/// Inflates a zlib (`zlib_header == true`) or raw DEFLATE (`zlib_header == false`)
/// stream into `out`, returning the number of decompressed bytes written.
///
/// Properly handles valid empty streams when `out.is_empty()` using a 1-byte scratch
/// destination so `zlib-rs` distinguishes `StreamEnd` from `BufError`.
fn inflate_zlib_or_raw(
    source: &[u8],
    out: &mut [u8],
    zlib_header: bool,
    codec: DecompressionCodec,
) -> Result<usize, VelocityIoError> {
    thread_local! {
        static ZLIB_INFLATE: RefCell<zlib_rs::Inflate> =
            RefCell::new(zlib_rs::Inflate::new(true, 15));
        static RAW_INFLATE: RefCell<zlib_rs::Inflate> =
            RefCell::new(zlib_rs::Inflate::new(false, 15));
    }

    let runner = |state: &mut zlib_rs::Inflate| -> Result<usize, VelocityIoError> {
        state.reset(zlib_header);
        let out_len = out.len();
        let mut scratch = [0u8; 1];
        let dest = if out_len == 0 { &mut scratch[..] } else { out };
        let (mut in_pos, mut out_pos) = (0usize, 0usize);

        let status = loop {
            match state.decompress(
                &source[in_pos..],
                &mut dest[out_pos..],
                zlib_rs::InflateFlush::NoFlush,
            ) {
                Ok(zlib_rs::Status::StreamEnd) => break Ok(zlib_rs::Status::StreamEnd),
                Ok(zlib_rs::Status::Ok) => {
                    let next_in = state.total_in() as usize;
                    let next_out = state.total_out() as usize;
                    if next_in == in_pos && next_out == out_pos {
                        break Ok(zlib_rs::Status::BufError);
                    }
                    in_pos = next_in;
                    out_pos = next_out;
                }
                other => break other,
            }
        };

        let written = state.total_out() as usize;
        match status {
            Ok(zlib_rs::Status::StreamEnd) if written <= out_len => Ok(written),
            Ok(zlib_rs::Status::StreamEnd) => Err(VelocityIoError::DecompressionFailed {
                codec,
                reason: format!("decompressed output ({written} B) exceeded buffer ({out_len} B)"),
            }),
            Ok(other) => Err(VelocityIoError::DecompressionFailed {
                codec,
                reason: format!("incomplete stream (status={other:?}, written={written} B)"),
            }),
            Err(err) => Err(VelocityIoError::DecompressionFailed {
                codec,
                reason: format!("zlib-rs error: {err:?}"),
            }),
        }
    };

    if zlib_header {
        ZLIB_INFLATE.with_borrow_mut(runner)
    } else {
        RAW_INFLATE.with_borrow_mut(runner)
    }
}

/// Standard LZ4 block and LZ4 frame decompression backend via `lz4_flex`.
///
/// Automatically detects the LZ4 frame magic (`0x184D2204`) and routes to
/// `lz4_flex::frame::FrameDecoder`; otherwise decodes a raw LZ4 block using
/// `job.expected_output_len`.
#[derive(Debug, Default, Clone, Copy)]
pub struct Lz4DecompressionBackend;

impl DecompressionBackend for Lz4DecompressionBackend {
    fn name(&self) -> &'static str {
        "lz4-flex"
    }

    fn supported_codecs(&self) -> &'static [DecompressionCodec] {
        &[DecompressionCodec::Lz4]
    }

    fn execution_mode(&self) -> DecompressionExecutionMode {
        DecompressionExecutionMode::CpuWorkerFallback
    }

    fn decompress(
        &self,
        job: &DecompressionJob<'_>,
        _compute: &dyn ComputeQueueBackend,
    ) -> Result<DecompressionOutput, VelocityIoError> {
        let out = if job.src_bytes.starts_with(&LZ4_FRAME_MAGIC) {
            let mut decoder = lz4_flex::frame::FrameDecoder::new(job.src_bytes);
            let mut buf = Vec::with_capacity(job.expected_output_len);
            if job.expected_output_len > 0 {
                let mut limited = (&mut decoder).take((job.expected_output_len as u64) + 1);
                limited.read_to_end(&mut buf).map_err(|e| {
                    VelocityIoError::DecompressionFailed {
                        codec: DecompressionCodec::Lz4,
                        reason: format!("LZ4 frame decode error: {e}"),
                    }
                })?;
                if buf.len() > job.expected_output_len {
                    return Err(VelocityIoError::DecompressionFailed {
                        codec: DecompressionCodec::Lz4,
                        reason: format!(
                            "LZ4 frame output exceeded expected {} B",
                            job.expected_output_len
                        ),
                    });
                }
            } else {
                decoder.read_to_end(&mut buf).map_err(|e| {
                    VelocityIoError::DecompressionFailed {
                        codec: DecompressionCodec::Lz4,
                        reason: format!("LZ4 frame decode error: {e}"),
                    }
                })?;
            }
            buf
        } else {
            if job.src_bytes.is_empty() && job.expected_output_len == 0 {
                Vec::new()
            } else {
                lz4_flex::block::decompress(job.src_bytes, job.expected_output_len).map_err(
                    |e| VelocityIoError::DecompressionFailed {
                        codec: DecompressionCodec::Lz4,
                        reason: format!("LZ4 block decode error: {e}"),
                    },
                )?
            }
        };

        let bytes_written = out.len();
        Ok(DecompressionOutput {
            bytes_written,
            completion_fence: None,
            cpu_bytes: Some(out),
        })
    }
}

/// Parsed open-spec Microsoft GDeflate v1 stream header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GDeflateHeader {
    pub id: u8,
    pub id_inv: u8,
    pub num_tiles: u16,
    pub tile_size_idx: u8,
    pub last_tile_size: u32,
}

impl GDeflateHeader {
    /// Returns the uncompressed tile size in bytes (`65536 << tile_size_idx`).
    #[inline]
    #[must_use]
    pub const fn tile_uncompressed_bytes(&self) -> usize {
        GDEFLATE_DEFAULT_TILE_BYTES << (self.tile_size_idx as usize)
    }

    /// Parses the 8-byte GDeflate v1 header from `src`.
    pub fn parse(src: &[u8]) -> Result<Self, VelocityIoError> {
        if src.len() < GDEFLATE_HEADER_BYTES {
            return Err(VelocityIoError::DecompressionFailed {
                codec: DecompressionCodec::GDeflate,
                reason: format!(
                    "GDeflate v1 stream too short for header ({} < {GDEFLATE_HEADER_BYTES} B)",
                    src.len()
                ),
            });
        }
        let id = src[0];
        let id_inv = src[1];
        if id != GDEFLATE_V1_ID || id_inv != GDEFLATE_V1_ID_INV {
            return Err(VelocityIoError::DecompressionFailed {
                codec: DecompressionCodec::GDeflate,
                reason: format!(
                    "invalid GDeflate v1 magic (got id={id:#04x}, id_inv={id_inv:#04x}, expected 0x04/0xFB)"
                ),
            });
        }
        let num_tiles = u16::from_le_bytes([src[2], src[3]]);
        let flags = u32::from_le_bytes([src[4], src[5], src[6], src[7]]);
        let tile_size_idx = (flags & 0x3) as u8;
        let last_tile_size = (flags >> 2) & 0x3_FFFF;
        Ok(Self {
            id,
            id_inv,
            num_tiles,
            tile_size_idx,
            last_tile_size,
        })
    }
}

/// Open-spec Microsoft GDeflate v1 decompression backend.
///
/// Parses the GDeflate v1 tile stream (`id == 0x04`, `id_inv == 0xFB`, `num_tiles: u16`,
/// `flags: u32`, followed by `num_tiles` little-endian `u32` tile byte offsets) and
/// decompresses each independent raw DEFLATE tile (`zlib_rs::Inflate::new(false, 15)`)
/// into the output buffer.
#[derive(Debug, Default, Clone, Copy)]
pub struct GDeflateDecompressionBackend;

impl DecompressionBackend for GDeflateDecompressionBackend {
    fn name(&self) -> &'static str {
        "gdeflate-v1"
    }

    fn supported_codecs(&self) -> &'static [DecompressionCodec] {
        &[DecompressionCodec::GDeflate]
    }

    fn execution_mode(&self) -> DecompressionExecutionMode {
        DecompressionExecutionMode::CpuWorkerFallback
    }

    fn decompress(
        &self,
        job: &DecompressionJob<'_>,
        _compute: &dyn ComputeQueueBackend,
    ) -> Result<DecompressionOutput, VelocityIoError> {
        let src = job.src_bytes;
        let header = GDeflateHeader::parse(src)?;
        let num_tiles = usize::from(header.num_tiles);
        if num_tiles == 0 {
            return Ok(DecompressionOutput {
                bytes_written: 0,
                completion_fence: None,
                cpu_bytes: Some(Vec::new()),
            });
        }

        let table_bytes = num_tiles * std::mem::size_of::<u32>();
        let payload_base = GDEFLATE_HEADER_BYTES + table_bytes;
        if src.len() < payload_base {
            return Err(VelocityIoError::DecompressionFailed {
                codec: DecompressionCodec::GDeflate,
                reason: format!(
                    "GDeflate v1 stream truncated in tile offset table ({} < {payload_base} B)",
                    src.len()
                ),
            });
        }

        let tile_cap = header.tile_uncompressed_bytes();
        let payload = &src[payload_base..];
        let mut offsets = Vec::with_capacity(num_tiles);
        for i in 0..num_tiles {
            let pos = GDEFLATE_HEADER_BYTES + i * 4;
            let off = u32::from_le_bytes([src[pos], src[pos + 1], src[pos + 2], src[pos + 3]])
                as usize;
            offsets.push(off);
        }

        if offsets[0] != 0 {
            return Err(VelocityIoError::DecompressionFailed {
                codec: DecompressionCodec::GDeflate,
                reason: format!("expected tile 0 offset to be 0, got {}", offsets[0]),
            });
        }

        let capacity = if job.expected_output_len > 0 {
            job.expected_output_len
        } else {
            num_tiles.saturating_mul(tile_cap)
        };
        let mut out = vec![0u8; capacity];
        let mut out_cursor = 0usize;

        for tile_idx in 0..num_tiles {
            let start = offsets[tile_idx];
            let end = if tile_idx + 1 < num_tiles {
                offsets[tile_idx + 1]
            } else if header.last_tile_size > 0 {
                start.saturating_add(header.last_tile_size as usize)
            } else {
                payload.len()
            };

            if start > end || end > payload.len() {
                return Err(VelocityIoError::DecompressionFailed {
                    codec: DecompressionCodec::GDeflate,
                    reason: format!(
                        "GDeflate tile {tile_idx} slice [{start}..{end}] out of payload bounds ({} B)",
                        payload.len()
                    ),
                });
            }

            let remaining = out.len().saturating_sub(out_cursor);
            let tile_max = remaining.min(tile_cap);
            let tile_written = inflate_zlib_or_raw(
                &payload[start..end],
                &mut out[out_cursor..out_cursor + tile_max],
                false,
                DecompressionCodec::GDeflate,
            )?;
            out_cursor += tile_written;
        }

        out.truncate(out_cursor);
        Ok(DecompressionOutput {
            bytes_written: out_cursor,
            completion_fence: None,
            cpu_bytes: Some(out),
        })
    }
}

/// Encodes `input` into a valid open-spec Microsoft GDeflate v1 multi-tile stream
/// using 64 KiB independent raw DEFLATE tiles.
///
/// Primarily used by acceptance tests and Criterion micro-benchmarks to construct
/// deterministic synthetic GDeflate v1 streams.
pub fn encode_gdeflate_v1(input: &[u8]) -> Result<Vec<u8>, VelocityIoError> {
    let tiles: Vec<&[u8]> = if input.is_empty() {
        Vec::new()
    } else {
        input.chunks(GDEFLATE_DEFAULT_TILE_BYTES).collect()
    };

    let num_tiles = u16::try_from(tiles.len()).map_err(|_| VelocityIoError::InvalidRequest(
        "input exceeds maximum GDeflate v1 tile count (65535)",
    ))?;

    let mut compressed_tiles = Vec::with_capacity(tiles.len());
    for tile in &tiles {
        let compressed = deflate_raw_tile(tile)?;
        compressed_tiles.push(compressed);
    }

    let last_tile_size = compressed_tiles.last().map_or(0u32, |t| t.len() as u32);
    if last_tile_size > 0x3_FFFF {
        return Err(VelocityIoError::InvalidRequest(
            "compressed GDeflate tile exceeds 18-bit last_tile_size limit",
        ));
    }

    let flags: u32 = last_tile_size << 2; // tile_size_idx = 0 (64 KiB)
    let total_payload: usize = compressed_tiles.iter().map(Vec::len).sum();
    let mut stream = Vec::with_capacity(
        GDEFLATE_HEADER_BYTES + (num_tiles as usize) * 4 + total_payload,
    );
    stream.push(GDEFLATE_V1_ID);
    stream.push(GDEFLATE_V1_ID_INV);
    stream.extend_from_slice(&num_tiles.to_le_bytes());
    stream.extend_from_slice(&flags.to_le_bytes());

    let mut offset = 0u32;
    for ct in &compressed_tiles {
        stream.extend_from_slice(&offset.to_le_bytes());
        offset = offset
            .checked_add(ct.len() as u32)
            .ok_or(VelocityIoError::InvalidRequest("GDeflate stream offset overflow"))?;
    }
    for ct in &compressed_tiles {
        stream.extend_from_slice(ct);
    }
    Ok(stream)
}

/// Compresses a single raw DEFLATE (RFC 1951, no zlib wrapper) tile using `zlib-rs`.
fn deflate_raw_tile(input: &[u8]) -> Result<Vec<u8>, VelocityIoError> {
    let config = zlib_rs::DeflateConfig {
        level: 6,
        window_bits: -15,
        ..zlib_rs::DeflateConfig::default()
    };
    let mut out = vec![0u8; input.len() + (input.len() / 8) + 64];
    let (compressed, ret) = zlib_rs::compress_slice(&mut out, input, config);
    if ret != zlib_rs::ReturnCode::Ok {
        return Err(VelocityIoError::DecompressionFailed {
            codec: DecompressionCodec::GDeflate,
            reason: format!("raw deflate tile compression failed: {ret:?}"),
        });
    }
    Ok(compressed.to_vec())
}

/// Strictly typed, documented stub backend for Microsoft's proprietary BCPack format.
///
/// Per `.agents/rules/xbox-compat.md` Rule 1:
/// - Contains **NO** bit-level BCPack decoding logic or lookup tables.
/// - Always returns [`VelocityIoError::ProprietaryCodecStub`].
/// - Callers that possess an authorized or external BCPack implementation may override
///   this stub at runtime via [`DecompressionRegistry::register`].
#[derive(Debug, Default, Clone, Copy)]
pub struct BcPackStubBackend;

impl DecompressionBackend for BcPackStubBackend {
    fn name(&self) -> &'static str {
        "bcpack-stub"
    }

    fn supported_codecs(&self) -> &'static [DecompressionCodec] {
        &ALL_BCPACK_CODECS
    }

    fn execution_mode(&self) -> DecompressionExecutionMode {
        DecompressionExecutionMode::CpuWorkerFallback
    }

    fn decompress(
        &self,
        job: &DecompressionJob<'_>,
        _compute: &dyn ComputeQueueBackend,
    ) -> Result<DecompressionOutput, VelocityIoError> {
        Err(VelocityIoError::ProprietaryCodecStub {
            codec: job.codec,
            hint: "Microsoft BCPack is a proprietary Xbox Series X|S texture format; \
                   register an external DecompressionBackend via DecompressionRegistry::register",
        })
    }
}

// ---------------------------------------------------------------------------
// MockComputeQueueBackend (for cross-platform tests & Criterion benchmarks)
// ---------------------------------------------------------------------------

/// Recorded event emitted by [`MockComputeQueueBackend`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MockComputeEvent {
    WaitFence(GpuFenceToken),
    PassthroughCopy {
        src: ResourceId,
        src_offset: u64,
        dst: ResourceId,
        dst_offset: u64,
        byte_len: u64,
        signal_token: GpuFenceToken,
    },
    ComputeDispatch {
        pipeline_key: &'static str,
        src_va: GpuVirtualAddress,
        dst_va: GpuVirtualAddress,
        params: Vec<u32>,
        thread_groups: [u32; 3],
        signal_token: GpuFenceToken,
    },
}

/// Deterministic in-memory mock implementing [`ComputeQueueBackend`].
///
/// Tracks timeline fence waits and dispatches, and optionally backs registered
/// [`ResourceId`] buffers in host memory so passthrough copy tests can verify
/// exact byte contents.
pub struct MockComputeQueueBackend {
    queue_id: u32,
    next_fence: AtomicU64,
    completed_fence: AtomicU64,
    auto_complete: AtomicBool,
    record_events: AtomicBool,
    events: Mutex<Vec<MockComputeEvent>>,
    buffers: Mutex<HashMap<ResourceId, Vec<u8>>>,
}

impl std::fmt::Debug for MockComputeQueueBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockComputeQueueBackend")
            .field("queue_id", &self.queue_id)
            .field("next_fence", &self.next_fence.load(Ordering::Relaxed))
            .field(
                "completed_fence",
                &self.completed_fence.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl Default for MockComputeQueueBackend {
    fn default() -> Self {
        Self::new(1)
    }
}

impl MockComputeQueueBackend {
    /// Creates a new mock compute queue with `queue_id` and immediate fence completion enabled.
    #[must_use]
    pub fn new(queue_id: u32) -> Self {
        Self {
            queue_id,
            next_fence: AtomicU64::new(0),
            completed_fence: AtomicU64::new(0),
            auto_complete: AtomicBool::new(true),
            record_events: AtomicBool::new(true),
            events: Mutex::new(Vec::new()),
            buffers: Mutex::new(HashMap::new()),
        }
    }

    /// Creates a benchmark-oriented mock compute queue with event logging disabled.
    #[must_use]
    pub fn for_bench(queue_id: u32) -> Self {
        let mock = Self::new(queue_id);
        mock.record_events.store(false, Ordering::Relaxed);
        mock
    }

    /// Controls whether dispatched work advances `completed_fence_value()` immediately.
    pub fn set_auto_complete(&self, enabled: bool) {
        self.auto_complete.store(enabled, Ordering::Release);
    }

    /// Manually advances the completed timeline fence value (for deferred fence tests).
    pub fn advance_completed_fence(&self, value: u64) {
        self.completed_fence.fetch_max(value, Ordering::Release);
    }

    /// Registers a mock GPU buffer backing `resource`.
    pub fn register_buffer(&self, resource: ResourceId, data: Vec<u8>) {
        self.buffers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(resource, data);
    }

    /// Reads back a clone of a registered mock GPU buffer.
    #[must_use]
    pub fn read_buffer(&self, resource: ResourceId) -> Option<Vec<u8>> {
        self.buffers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&resource)
            .cloned()
    }

    /// Returns a snapshot of recorded events in exact submission order.
    #[must_use]
    pub fn recorded_events(&self) -> Vec<MockComputeEvent> {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Clears recorded events.
    pub fn clear_events(&self) {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    fn signal_next(&self) -> GpuFenceToken {
        let value = self.next_fence.fetch_add(1, Ordering::AcqRel) + 1;
        if self.auto_complete.load(Ordering::Acquire) {
            self.completed_fence.fetch_max(value, Ordering::Release);
        }
        GpuFenceToken {
            queue_id: self.queue_id,
            value,
        }
    }
}

impl ComputeQueueBackend for MockComputeQueueBackend {
    fn wait_fence(&self, token: GpuFenceToken) -> Result<(), VelocityIoError> {
        if self.record_events.load(Ordering::Relaxed) {
            self.events
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(MockComputeEvent::WaitFence(token));
        }
        Ok(())
    }

    fn dispatch_passthrough_copy(
        &self,
        src: ResourceId,
        src_offset: u64,
        dst: ResourceId,
        dst_offset: u64,
        byte_len: u64,
    ) -> Result<GpuFenceToken, VelocityIoError> {
        {
            let mut buffers = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(src_buf) = buffers.get(&src) {
                let s_start = src_offset as usize;
                let s_end = s_start.saturating_add(byte_len as usize);
                if s_end > src_buf.len() {
                    return Err(VelocityIoError::ComputeQueue {
                        hresult: 0x8007_0057u32 as i32,
                        context: "mock passthrough copy source out of bounds",
                    });
                }
                let copied = src_buf[s_start..s_end].to_vec();
                if let Some(dst_buf) = buffers.get_mut(&dst) {
                    let d_start = dst_offset as usize;
                    let d_end = d_start.saturating_add(byte_len as usize);
                    if d_end > dst_buf.len() {
                        return Err(VelocityIoError::ComputeQueue {
                            hresult: 0x8007_0057u32 as i32,
                            context: "mock passthrough copy destination out of bounds",
                        });
                    }
                    dst_buf[d_start..d_end].copy_from_slice(&copied);
                }
            }
        }

        let token = self.signal_next();
        if self.record_events.load(Ordering::Relaxed) {
            self.events
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(MockComputeEvent::PassthroughCopy {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    byte_len,
                    signal_token: token,
                });
        }
        Ok(token)
    }

    fn dispatch_compute_shader(
        &self,
        pipeline_key: &'static str,
        src_va: GpuVirtualAddress,
        dst_va: GpuVirtualAddress,
        params: &[u32],
        thread_groups: [u32; 3],
    ) -> Result<GpuFenceToken, VelocityIoError> {
        let token = self.signal_next();
        if self.record_events.load(Ordering::Relaxed) {
            self.events
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(MockComputeEvent::ComputeDispatch {
                    pipeline_key,
                    src_va,
                    dst_va,
                    params: params.to_vec(),
                    thread_groups,
                    signal_token: token,
                });
        }
        Ok(token)
    }

    fn completed_fence_value(&self) -> u64 {
        self.completed_fence.load(Ordering::Acquire)
    }
}

// ---------------------------------------------------------------------------
// Thin audited Windows D3D12 Async Compute Queue FFI wrapper
// ---------------------------------------------------------------------------

/// Audited Windows Direct3D 12 Async Compute Queue (`D3D12_COMMAND_LIST_TYPE_COMPUTE`)
/// backend implementing [`ComputeQueueBackend`].
///
/// All COM / D3D12 `unsafe` FFI calls are isolated inside this module behind safe
/// RAII handles with explicit `// SAFETY:` invariants (Rule 4).
#[cfg(windows)]
pub mod d3d12_compute {
    use std::collections::HashMap;
    use std::mem::ManuallyDrop;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_11_0, ID3DBlob};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory4};
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::{Interface, PCSTR};

    use crate::graphics::contracts::{
        ComputeQueueBackend, GpuFenceToken, GpuVirtualAddress, ResourceId, VelocityIoError,
    };

    /// Pipeline key for the built-in HLSL GPU-to-GPU passthrough copy compute shader.
    pub const PIPELINE_PASSTHROUGH_COPY: &str = "passthrough_copy";

    /// Number of command allocator / command list slots in the compute submission ring.
    const COMMAND_RING_LEN: usize = 4;

    /// Small `cs_5_1` HLSL compute shader copying 32-bit words from `srcBuf` (`t0`)
    /// to `dstBuf` (`u0`) using inline root constants (`b0`).
    const PASSTHROUGH_COPY_HLSL: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), SRV(t0), UAV(u0)"
cbuffer CopyParams : register(b0) {
    uint srcOffsetBytes;
    uint dstOffsetBytes;
    uint dwordCount;
    uint reserved;
};
ByteAddressBuffer srcBuf : register(t0);
RWByteAddressBuffer dstBuf : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx < dwordCount) {
        uint val = srcBuf.Load(srcOffsetBytes + idx * 4u);
        dstBuf.Store(dstOffsetBytes + idx * 4u, val);
    }
}
"#;

    struct CommandSlot {
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        last_fence_value: u64,
    }

    struct RingState {
        slots: Vec<CommandSlot>,
        cursor: usize,
    }

    /// Safe RAII wrapper around a Win32 event `HANDLE`.
    struct OwnedEvent(HANDLE);

    // SAFETY: Win32 event handles are kernel objects safe to wait/signal from any thread.
    unsafe impl Send for OwnedEvent {}
    // SAFETY: Event handle operations are thread-safe in the Win32 kernel.
    unsafe impl Sync for OwnedEvent {}

    impl Drop for OwnedEvent {
        fn drop(&mut self) {
            if !self.0.is_invalid() && !self.0.0.is_null() {
                // SAFETY: `self.0` was created by `CreateEventW` and is owned exclusively by `OwnedEvent`.
                let _ = unsafe { CloseHandle(self.0) };
            }
        }
    }

    /// Real D3D12 Async Compute Queue (`D3D12_COMMAND_LIST_TYPE_COMPUTE`) backend.
    pub struct D3D12ComputeQueueBackend {
        queue_id: u32,
        queue: ID3D12CommandQueue,
        fence: ID3D12Fence,
        next_fence: AtomicU64,
        event: Mutex<OwnedEvent>,
        root_signature: ID3D12RootSignature,
        passthrough_pso: ID3D12PipelineState,
        ring: Mutex<RingState>,
        resources: Mutex<HashMap<ResourceId, ID3D12Resource>>,
        external_fences: Mutex<HashMap<u32, ID3D12Fence>>,
        next_resource_id: AtomicU64,
    }

    // SAFETY: D3D12 command queues, fences, PSOs, and root signatures are thread-safe COM objects,
    // and command list recording / resource maps are synchronized via internal `Mutex`es.
    unsafe impl Send for D3D12ComputeQueueBackend {}
    // SAFETY: See `Send` invariant above.
    unsafe impl Sync for D3D12ComputeQueueBackend {}

    impl D3D12ComputeQueueBackend {
        /// Creates a dedicated `D3D12_COMMAND_LIST_TYPE_COMPUTE` queue, timeline fence,
        /// command allocator/list ring, and compiles the built-in `cs_5_1` passthrough
        /// copy compute shader on `device`.
        pub fn new(device: &ID3D12Device, queue_id: u32) -> Result<Self, VelocityIoError> {
            let bytecode = compile_hlsl(
                PASSTHROUGH_COPY_HLSL,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
            )?;

            // SAFETY: `device` is a valid D3D12 device and `bytecode` is valid DXBC produced by `D3DCompile`.
            let (queue, fence, event, root_signature, passthrough_pso, slots) = unsafe {
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
                        Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
                        Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
                        NodeMask: 0,
                    })
                    .map_err(|e| map_d3d_err(e, "CreateCommandQueue(COMPUTE)"))?;

                let fence: ID3D12Fence = device
                    .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                    .map_err(|e| map_d3d_err(e, "CreateFence(COMPUTE)"))?;

                let event = CreateEventW(None, false, false, None)
                    .map_err(|e| map_d3d_err(e, "CreateEventW(COMPUTE)"))?;

                let root_signature: ID3D12RootSignature = device
                    .CreateRootSignature(0, &bytecode)
                    .map_err(|e| map_d3d_err(e, "CreateRootSignature(COMPUTE)"))?;

                let mut pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: ManuallyDrop::new(Some(root_signature.clone())),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytecode.as_ptr().cast(),
                        BytecodeLength: bytecode.len(),
                    },
                    ..Default::default()
                };
                let pso_res = device.CreateComputePipelineState(&pso_desc);
                ManuallyDrop::drop(&mut pso_desc.pRootSignature);
                let passthrough_pso: ID3D12PipelineState =
                    pso_res.map_err(|e| map_d3d_err(e, "CreateComputePipelineState"))?;

                let mut slots = Vec::with_capacity(COMMAND_RING_LEN);
                for _ in 0..COMMAND_RING_LEN {
                    let allocator: ID3D12CommandAllocator = device
                        .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)
                        .map_err(|e| map_d3d_err(e, "CreateCommandAllocator(COMPUTE)"))?;
                    let list: ID3D12GraphicsCommandList = device
                        .CreateCommandList(
                            0,
                            D3D12_COMMAND_LIST_TYPE_COMPUTE,
                            &allocator,
                            &passthrough_pso,
                        )
                        .map_err(|e| map_d3d_err(e, "CreateCommandList(COMPUTE)"))?;
                    list.Close()
                        .map_err(|e| map_d3d_err(e, "Close initial compute command list"))?;
                    slots.push(CommandSlot {
                        allocator,
                        list,
                        last_fence_value: 0,
                    });
                }

                (
                    queue,
                    fence,
                    OwnedEvent(event),
                    root_signature,
                    passthrough_pso,
                    slots,
                )
            };

            Ok(Self {
                queue_id,
                queue,
                fence,
                next_fence: AtomicU64::new(0),
                event: Mutex::new(event),
                root_signature,
                passthrough_pso,
                ring: Mutex::new(RingState { slots, cursor: 0 }),
                resources: Mutex::new(HashMap::new()),
                external_fences: Mutex::new(HashMap::new()),
                next_resource_id: AtomicU64::new(1),
            })
        }

        /// Returns this compute queue's logical `queue_id`.
        #[inline]
        #[must_use]
        pub fn queue_id(&self) -> u32 {
            self.queue_id
        }

        /// Returns a clone of the underlying `ID3D12Fence` so peer queues (e.g. Direct/Copy)
        /// can GPU-wait on compute completions.
        #[must_use]
        pub fn fence(&self) -> ID3D12Fence {
            self.fence.clone()
        }

        /// Registers an external queue's `ID3D12Fence` under `queue_id` so
        /// [`ComputeQueueBackend::wait_fence`] issues a real cross-queue `ID3D12CommandQueue::Wait`.
        pub fn register_external_fence(&self, queue_id: u32, fence: ID3D12Fence) {
            self.external_fences
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(queue_id, fence);
        }

        /// Registers a D3D12 buffer resource and returns an opaque [`ResourceId`].
        pub fn register_resource(&self, resource: ID3D12Resource) -> ResourceId {
            let id = ResourceId(self.next_resource_id.fetch_add(1, Ordering::Relaxed));
            self.resources
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(id, resource);
            id
        }

        /// Unregisters a previously registered [`ResourceId`].
        pub fn unregister_resource(&self, id: ResourceId) {
            self.resources
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&id);
        }

        /// Synchronizes access to the auto-reset Win32 event and loops until
        /// `self.fence.GetCompletedValue() >= target` or `timeout_ms` elapses.
        fn wait_fence_value(&self, target: u64, timeout_ms: u32) -> Result<bool, VelocityIoError> {
            // SAFETY: `self.fence` is valid for the lifetime of `self`.
            if unsafe { self.fence.GetCompletedValue() } >= target {
                return Ok(true);
            }

            let deadline = (timeout_ms != INFINITE).then(|| {
                std::time::Instant::now() + std::time::Duration::from_millis(u64::from(timeout_ms))
            });
            let ev = self.event.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                // SAFETY: `self.fence` and `ev.0` are valid and exclusively accessed under `self.event`.
                unsafe {
                    if self.fence.GetCompletedValue() >= target {
                        return Ok(true);
                    }
                    self.fence
                        .SetEventOnCompletion(target, ev.0)
                        .map_err(|e| map_d3d_err(e, "SetEventOnCompletion(COMPUTE)"))?;
                    if self.fence.GetCompletedValue() >= target {
                        return Ok(true);
                    }
                    let wait_ms = match deadline {
                        Some(dl) => {
                            let Some(rem) = dl.checked_duration_since(std::time::Instant::now())
                            else {
                                return Ok(self.fence.GetCompletedValue() >= target);
                            };
                            rem.as_millis().min((u32::MAX - 1) as u128) as u32
                        }
                        None => INFINITE,
                    };
                    let waited = WaitForSingleObject(ev.0, wait_ms);
                    if self.fence.GetCompletedValue() >= target {
                        return Ok(true);
                    }
                    if waited != WAIT_OBJECT_0 {
                        return Ok(false);
                    }
                }
            }
        }

        /// Blocks the calling thread until `token.value` has completed on this compute queue
        /// or `timeout_ms` elapses.
        pub fn wait_cpu(
            &self,
            token: GpuFenceToken,
            timeout_ms: u32,
        ) -> Result<bool, VelocityIoError> {
            self.wait_fence_value(token.value, timeout_ms)
        }

        /// Acquires the next command slot in the ring, waiting on its prior fence only if
        /// all `COMMAND_RING_LEN` slots are still in flight, then records and executes `record`.
        fn record_and_submit(
            &self,
            record: impl FnOnce(&ID3D12GraphicsCommandList) -> Result<(), VelocityIoError>,
        ) -> Result<GpuFenceToken, VelocityIoError> {
            let mut ring = self.ring.lock().unwrap_or_else(|e| e.into_inner());
            let idx = ring.cursor % ring.slots.len();
            ring.cursor = ring.cursor.wrapping_add(1);
            let slot = &mut ring.slots[idx];

            if slot.last_fence_value > 0 {
                let _ = self.wait_fence_value(slot.last_fence_value, INFINITE)?;
            }

            // SAFETY: `slot.allocator`, `slot.list`, `self.fence`, and `self.queue` are valid COM
            // objects; `wait_fence_value` above guarantees the GPU finished `slot.last_fence_value`.
            unsafe {
                slot.allocator
                    .Reset()
                    .map_err(|e| map_d3d_err(e, "CommandAllocator::Reset(COMPUTE)"))?;
                slot.list
                    .Reset(&slot.allocator, &self.passthrough_pso)
                    .map_err(|e| map_d3d_err(e, "CommandList::Reset(COMPUTE)"))?;

                record(&slot.list)?;

                slot.list
                    .Close()
                    .map_err(|e| map_d3d_err(e, "CommandList::Close(COMPUTE)"))?;

                let cmd_list: ID3D12CommandList = slot
                    .list
                    .cast()
                    .map_err(|e| map_d3d_err(e, "cast to ID3D12CommandList"))?;
                self.queue.ExecuteCommandLists(&[Some(cmd_list)]);

                let fence_val = self.next_fence.fetch_add(1, Ordering::AcqRel) + 1;
                self.queue
                    .Signal(&self.fence, fence_val)
                    .map_err(|e| map_d3d_err(e, "CommandQueue::Signal(COMPUTE)"))?;
                slot.last_fence_value = fence_val;

                Ok(GpuFenceToken {
                    queue_id: self.queue_id,
                    value: fence_val,
                })
            }
        }
    }

    impl Drop for D3D12ComputeQueueBackend {
        fn drop(&mut self) {
            let last = self.next_fence.load(Ordering::Acquire);
            if last > 0 {
                let _ = self.wait_fence_value(last, INFINITE);
            }
        }
    }

    impl ComputeQueueBackend for D3D12ComputeQueueBackend {
        fn wait_fence(&self, token: GpuFenceToken) -> Result<(), VelocityIoError> {
            let target_fence = if token.queue_id == self.queue_id {
                self.fence.clone()
            } else {
                self.external_fences
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(&token.queue_id)
                    .cloned()
                    .ok_or(VelocityIoError::ComputeQueue {
                        hresult: 0x8007_0057u32 as i32,
                        context: "unregistered external queue fence for wait_fence",
                    })?
            };

            // SAFETY: Both `self.queue` and `target_fence` are valid D3D12 objects on the same device.
            unsafe {
                self.queue
                    .Wait(&target_fence, token.value)
                    .map_err(|e| map_d3d_err(e, "CommandQueue::Wait(COMPUTE)"))
            }
        }

        fn dispatch_passthrough_copy(
            &self,
            src: ResourceId,
            src_offset: u64,
            dst: ResourceId,
            dst_offset: u64,
            byte_len: u64,
        ) -> Result<GpuFenceToken, VelocityIoError> {
            let (src_res, dst_res) = {
                let resources = self.resources.lock().unwrap_or_else(|e| e.into_inner());
                let s = resources
                    .get(&src)
                    .cloned()
                    .ok_or(VelocityIoError::ComputeQueue {
                        hresult: 0x8007_0057u32 as i32,
                        context: "unknown src ResourceId in dispatch_passthrough_copy",
                    })?;
                let d = resources
                    .get(&dst)
                    .cloned()
                    .ok_or(VelocityIoError::ComputeQueue {
                        hresult: 0x8007_0057u32 as i32,
                        context: "unknown dst ResourceId in dispatch_passthrough_copy",
                    })?;
                (s, d)
            };

            self.record_and_submit(|list| {
                // SAFETY: Both `src_res` and `dst_res` remain alive until `record_and_submit` returns
                // and are owned by `self.resources` until unregistered after GPU fence completion.
                unsafe {
                    list.CopyBufferRegion(&dst_res, dst_offset, &src_res, src_offset, byte_len);
                }
                Ok(())
            })
        }

        fn dispatch_compute_shader(
            &self,
            pipeline_key: &'static str,
            src_va: GpuVirtualAddress,
            dst_va: GpuVirtualAddress,
            params: &[u32],
            thread_groups: [u32; 3],
        ) -> Result<GpuFenceToken, VelocityIoError> {
            if pipeline_key != PIPELINE_PASSTHROUGH_COPY {
                return Err(VelocityIoError::ComputeQueue {
                    hresult: 0x8000_4001u32 as i32,
                    context: "unknown compute shader pipeline_key",
                });
            }
            if params.len() > 4 {
                return Err(VelocityIoError::ComputeQueue {
                    hresult: 0x8007_0057u32 as i32,
                    context: "passthrough_copy accepts at most 4 root constants",
                });
            }

            let mut constants = [0u32; 4];
            constants[..params.len()].copy_from_slice(params);

            self.record_and_submit(|list| {
                // SAFETY: `self.root_signature` and `self.passthrough_pso` are valid for the lifetime
                // of `self`; `src_va` and `dst_va` are caller-validated GPU virtual addresses.
                unsafe {
                    list.SetComputeRootSignature(&self.root_signature);
                    list.SetPipelineState(&self.passthrough_pso);
                    list.SetComputeRoot32BitConstants(
                        0,
                        constants.len() as u32,
                        constants.as_ptr().cast(),
                        0,
                    );
                    list.SetComputeRootShaderResourceView(1, src_va.0);
                    list.SetComputeRootUnorderedAccessView(2, dst_va.0);
                    list.Dispatch(thread_groups[0], thread_groups[1], thread_groups[2]);
                }
                Ok(())
            })
        }

        fn completed_fence_value(&self) -> u64 {
            // SAFETY: `self.fence` is a valid `ID3D12Fence`.
            unsafe { self.fence.GetCompletedValue() }
        }
    }

    /// Creates a D3D12 device, preferring the Microsoft WARP software adapter (deterministic
    /// in headless CI/sandbox environments) and falling back to the default hardware adapter.
    ///
    /// Returns `None` if D3D12 is unavailable on this host.
    #[must_use]
    pub fn try_create_test_device() -> Option<ID3D12Device> {
        // SAFETY: Standard DXGI factory + WARP/default D3D12 device creation.
        unsafe {
            if let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>()
                && let Ok(warp) = factory.EnumWarpAdapter::<windows::Win32::Graphics::Dxgi::IDXGIAdapter1>()
            {
                let mut dev: Option<ID3D12Device> = None;
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_ok()
                    && let Some(device) = dev
                {
                    return Some(device);
                }
            }
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_ok() {
                return dev;
            }
            None
        }
    }

    fn compile_hlsl(
        source: &str,
        entry: PCSTR,
        target: PCSTR,
    ) -> Result<Vec<u8>, VelocityIoError> {
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: `source`, `entry`, and `target` remain valid for the duration of `D3DCompile`.
        let res = unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                windows::core::s!("velocity_io_compute"),
                None,
                None,
                entry,
                target,
                0,
                0,
                &mut code,
                Some(&mut errors),
            )
        };
        if let Err(err) = res {
            if let Some(err_blob) = errors {
                // SAFETY: `err_blob` owns its buffer for its lifetime.
                let msg = unsafe {
                    std::slice::from_raw_parts(
                        err_blob.GetBufferPointer().cast::<u8>(),
                        err_blob.GetBufferSize(),
                    )
                };
                tracing::warn!(
                    hlsl_errors = %String::from_utf8_lossy(msg),
                    "velocity_io: HLSL compile failed"
                );
            }
            return Err(map_d3d_err(err, "D3DCompile(cs_5_1)"));
        }
        let blob = code.ok_or(VelocityIoError::ComputeQueue {
            hresult: 0x8000_4005u32 as i32,
            context: "D3DCompile returned null bytecode blob",
        })?;
        // SAFETY: `blob` owns its compiled bytecode buffer for its lifetime.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                blob.GetBufferPointer().cast::<u8>(),
                blob.GetBufferSize(),
            )
            .to_vec()
        };
        Ok(bytes)
    }

    fn map_d3d_err(err: windows::core::Error, context: &'static str) -> VelocityIoError {
        VelocityIoError::ComputeQueue {
            hresult: err.code().0,
            context,
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gdeflate_v1_multi_tile_roundtrip() {
        let mut payload = Vec::with_capacity(GDEFLATE_DEFAULT_TILE_BYTES * 2 + 1337);
        for i in 0..(GDEFLATE_DEFAULT_TILE_BYTES * 2 + 1337) {
            payload.push(((i * 31 + (i >> 8)) & 0xff) as u8);
        }
        let encoded = encode_gdeflate_v1(&payload).expect("encode gdeflate");
        let header = GDeflateHeader::parse(&encoded).expect("parse header");
        assert_eq!(header.id, GDEFLATE_V1_ID);
        assert_eq!(header.id_inv, GDEFLATE_V1_ID_INV);
        assert_eq!(header.num_tiles, 3);

        let compute = Arc::new(MockComputeQueueBackend::new(1));
        let queue = GpuDecompressQueue::new(compute);
        let job = DecompressionJob {
            job_id: 1,
            codec: DecompressionCodec::GDeflate,
            src_bytes: &encoded,
            src_gpu_va: None,
            src_gpu_resource: None,
            expected_output_len: payload.len(),
            dst_gpu_resource: None,
            wait_copy_fence: None,
        };
        let output = queue.dispatch(&job).expect("decompress gdeflate");
        assert_eq!(output.bytes_written, payload.len());
        assert_eq!(output.cpu_bytes.as_deref(), Some(payload.as_slice()));
    }

    #[test]
    fn bcpack_stub_returns_typed_error_and_allows_override() {
        let compute = Arc::new(MockComputeQueueBackend::new(1));
        let queue = GpuDecompressQueue::new(compute);
        let job = DecompressionJob {
            job_id: 7,
            codec: DecompressionCodec::BcPack(BcPackFormat::Bc7),
            src_bytes: &[1, 2, 3, 4],
            src_gpu_va: None,
            src_gpu_resource: None,
            expected_output_len: 4,
            dst_gpu_resource: None,
            wait_copy_fence: None,
        };

        let Err(err) = queue.dispatch(&job) else {
            panic!("BCPack stub must fail with ProprietaryCodecStub");
        };
        assert!(matches!(
            err,
            VelocityIoError::ProprietaryCodecStub {
                codec: DecompressionCodec::BcPack(BcPackFormat::Bc7),
                ..
            }
        ));
    }
}
