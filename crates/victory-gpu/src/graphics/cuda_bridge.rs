//! Zero-copy DX12 ↔ CUDA (`cuda-oxide`) interop bridge.
//!
//! Exports DX12 shared VRAM buffers (`D3D12_HEAP_FLAG_SHARED`) and shared timeline fences
//! (`D3D12_FENCE_FLAG_SHARED`) via Win32 NT handles (`ID3D12Device::CreateSharedHandle`)
//! and imports them into `cuda-accel` (`cuImportExternalMemory` / `cuImportExternalSemaphore`)
//! with zero PCIe host round-trips between CUDA kernels and DX12 queues.
//!
//! When compiled without `--features cuda-oxide`, or when `VICTORY_CUDA_OXIDE=0` is set,
//! or on non-NVIDIA hosts, this module cleanly provides [`NativeFallbackComputeBackend`]
//! with zero CUDA runtime dependency.

use std::sync::{Arc, OnceLock};

use crate::graphics::contracts::{
    DecompressBlockDesc, DecompressionCodec, DetileDispatchDesc, GpuComputeBackend,
    ReadbackDeltaPair, ReadbackPieceDesc, SharedBufferHandle, SharedFenceHandle, VelocityIoError,
};

/// Returns whether the optional `cuda-oxide` GPU acceleration path is compiled in and enabled.
///
/// Set `VICTORY_CUDA_OXIDE=0` at runtime to disable CUDA offload and force native CPU/DX12 fallback
/// for A/B testing without recompiling.
#[inline]
pub fn is_cuda_oxide_enabled() -> bool {
    #[cfg(feature = "cuda-oxide")]
    {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var_os("VICTORY_CUDA_OXIDE").is_none_or(|v| v != "0"))
    }
    #[cfg(not(feature = "cuda-oxide"))]
    {
        false
    }
}

/// Native fallback backend used when `--features cuda-oxide` is disabled or no CUDA GPU is active.
#[derive(Debug, Default, Clone, Copy)]
pub struct NativeFallbackComputeBackend;

impl GpuComputeBackend for NativeFallbackComputeBackend {
    fn backend_name(&self) -> &'static str {
        "native-dx12-cpu-fallback"
    }

    fn import_shared_buffer(
        &self,
        _nt_handle: usize,
        _byte_len: u64,
        _dedicated: bool,
    ) -> Result<SharedBufferHandle, VelocityIoError> {
        Err(VelocityIoError::ComputeDispatchFailed(
            "cuda-oxide feature is disabled; using native fallback".into(),
        ))
    }

    fn release_shared_buffer(&self, _handle: SharedBufferHandle) {}

    fn import_shared_fence(&self, _nt_handle: usize) -> Result<SharedFenceHandle, VelocityIoError> {
        Err(VelocityIoError::ComputeDispatchFailed(
            "cuda-oxide feature is disabled; using native fallback".into(),
        ))
    }

    fn release_shared_fence(&self, _handle: SharedFenceHandle) {}

    fn enqueue_detile_batch(
        &self,
        _host_tiled_bytes: &[u8],
        _dst_linear_vram: SharedBufferHandle,
        _dispatches: &[DetileDispatchDesc],
        _wait: Option<(SharedFenceHandle, u64)>,
        _signal: Option<(SharedFenceHandle, u64)>,
    ) -> Result<(), VelocityIoError> {
        Err(VelocityIoError::ComputeDispatchFailed(
            "cuda-oxide feature is disabled; using native CPU detile".into(),
        ))
    }

    fn enqueue_decompress_batch(
        &self,
        codec: DecompressionCodec,
        _host_compressed_bytes: &[u8],
        _dst_uncompressed_vram: SharedBufferHandle,
        _blocks: &[DecompressBlockDesc],
        _wait: Option<(SharedFenceHandle, u64)>,
        _signal: Option<(SharedFenceHandle, u64)>,
    ) -> Result<(), VelocityIoError> {
        Err(VelocityIoError::DecompressionFailed {
            codec,
            reason: "cuda-oxide feature is disabled; using native CPU decompression".into(),
        })
    }

    fn compact_readback_deltas(
        &self,
        _now_vram: SharedBufferHandle,
        _held_vram: SharedBufferHandle,
        _pieces: &[ReadbackPieceDesc],
        _out_pairs: &mut [ReadbackDeltaPair],
        _wait: Option<(SharedFenceHandle, u64)>,
    ) -> Result<Option<usize>, VelocityIoError> {
        Ok(None)
    }

    fn synchronize(&self) -> Result<(), VelocityIoError> {
        Ok(())
    }
}

#[cfg(feature = "cuda-oxide")]
mod imp {
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, OnceLock};

    use cuda_accel::{
        CudaAccelContext, CudaDecompressCodec, ImportedBufferId, ImportedFenceId,
    };
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
    use windows::core::{Interface, PCWSTR};

    use super::is_cuda_oxide_enabled;
    use crate::graphics::contracts::{
        DecompressBlockDesc, DecompressionCodec, DetileDispatchDesc, GpuComputeBackend,
        ReadbackDeltaPair, ReadbackPieceDesc, SharedBufferHandle, SharedFenceHandle,
        VelocityIoError,
    };

    const GENERIC_ALL_ACCESS: u32 = 0x1000_0000;

    /// Returns the 8-byte Windows LUID of a D3D12 device's physical adapter.
    pub fn d3d12_adapter_luid(device: &ID3D12Device) -> [u8; 8] {
        // SAFETY: `GetAdapterLuid` is a plain query on a live `ID3D12Device`.
        let luid = unsafe { device.GetAdapterLuid() };
        let lo = luid.LowPart.to_le_bytes();
        let hi = luid.HighPart.to_le_bytes();
        [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]
    }

    fn close_nt_handle(handle: HANDLE) {
        if !handle.is_invalid() && !handle.0.is_null() {
            // SAFETY: closes the temporary Win32 NT handle after CUDA imported its reference.
            unsafe {
                let _ = windows_sys::Win32::Foundation::CloseHandle(handle.0);
            }
        }
    }

    fn map_detile_desc(d: &DetileDispatchDesc) -> cuda_accel::DetileDispatchDesc {
        cuda_accel::DetileDispatchDesc {
            src_base_offset: d.src_base_offset,
            dst_base_offset: d.dst_base_offset,
            width_elements: d.width_elements,
            height_elements: d.height_elements,
            depth_slices: d.depth_slices,
            element_bytes: d.element_bytes,
            block_dims: d.block_dims,
            block_log2: d.block_log2,
            mip_pitch_elements: d.mip_pitch_elements,
            dst_row_pitch_bytes: d.dst_row_pitch_bytes,
            dst_depth_pitch_bytes: d.dst_depth_pitch_bytes,
            tail_x: d.tail_x,
            tail_y: d.tail_y,
            pattern_id: d.pattern_id,
            thick: d.thick,
            slice_size: d.slice_size,
            first_slice: d.first_slice,
            _pad: d._pad,
        }
    }

    fn map_block_desc(b: &DecompressBlockDesc) -> cuda_accel::DecompressBlockDesc {
        cuda_accel::DecompressBlockDesc {
            src_offset: b.src_offset,
            src_len: b.src_len,
            dst_len: b.dst_len,
            dst_offset: b.dst_offset,
        }
    }

    fn map_piece_desc(p: &ReadbackPieceDesc) -> cuda_accel::ReadbackPieceDesc {
        cuda_accel::ReadbackPieceDesc {
            now_offset: p.now_offset,
            held_offset: p.held_offset,
            dword_count: p.dword_count,
            piece_id: p.piece_id,
        }
    }

    /// Active zero-copy DX12 ↔ CUDA compute bridge backed by `cuda-accel`.
    pub struct CudaOxideBridge {
        ctx: Arc<CudaAccelContext>,
    }

    impl CudaOxideBridge {
        /// Initializes the bridge on the CUDA device matching `device`'s DXGI adapter LUID.
        pub fn for_d3d12_device(device: &ID3D12Device) -> Result<Arc<Self>, VelocityIoError> {
            let luid = d3d12_adapter_luid(device);
            let ctx = CudaAccelContext::new(Some(luid)).map_err(|e| {
                VelocityIoError::ComputeDispatchFailed(format!("cuda-accel init failed: {e}"))
            })?;
            Ok(Arc::new(Self { ctx: Arc::new(ctx) }))
        }

        /// Initializes the bridge on CUDA device 0 (used for CPU-triggered decompression when
        /// no D3D12 device is bound yet).
        pub fn default_device() -> Result<Arc<Self>, VelocityIoError> {
            let ctx = CudaAccelContext::new(None).map_err(|e| {
                VelocityIoError::ComputeDispatchFailed(format!("cuda-accel init failed: {e}"))
            })?;
            Ok(Arc::new(Self { ctx: Arc::new(ctx) }))
        }

        #[inline]
        pub fn context(&self) -> &CudaAccelContext {
            &self.ctx
        }

        /// Allocates a DX12 `DEFAULT` VRAM buffer with `D3D12_HEAP_FLAG_SHARED` and imports it
        /// zero-copy into CUDA via Win32 NT handle.
        pub fn create_shared_vram_buffer(
            self: &Arc<Self>,
            device: &ID3D12Device,
            byte_len: u64,
        ) -> Result<SharedVramBuffer, VelocityIoError> {
            let aligned_len = byte_len.max(65536).next_multiple_of(65536);
            let heap_props = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Alignment: 0,
                Width: aligned_len,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            };

            let mut resource: Option<ID3D12Resource> = None;
            // SAFETY: creates a committed shared default-heap buffer on a live D3D12 device.
            unsafe {
                device
                    .CreateCommittedResource(
                        &heap_props,
                        D3D12_HEAP_FLAG_SHARED,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut resource,
                    )
                    .map_err(|e| {
                        VelocityIoError::ComputeDispatchFailed(format!(
                            "CreateCommittedResource(SHARED, {aligned_len} B) failed: {e}"
                        ))
                    })?;
            }
            let resource = resource.ok_or_else(|| {
                VelocityIoError::ComputeDispatchFailed(
                    "CreateCommittedResource returned null resource".into(),
                )
            })?;

            let child: ID3D12DeviceChild = resource.cast().map_err(|e| {
                VelocityIoError::ComputeDispatchFailed(format!("cast ID3D12DeviceChild: {e}"))
            })?;

            // SAFETY: exports a Win32 NT handle for the newly created shared resource.
            let nt_handle = unsafe {
                device
                    .CreateSharedHandle(&child, None, GENERIC_ALL_ACCESS, PCWSTR::null())
                    .map_err(|e| {
                        VelocityIoError::ComputeDispatchFailed(format!(
                            "CreateSharedHandle(buffer) failed: {e}"
                        ))
                    })?
            };

            let import_res = self.import_shared_buffer(nt_handle.0 as usize, aligned_len, true);
            // Per CUDA Driver API docs, CUDA duplicates the kernel reference and does not own the handle.
            close_nt_handle(nt_handle);
            let cuda_handle = import_res?;

            Ok(SharedVramBuffer {
                bridge: Arc::clone(self),
                resource,
                cuda_handle,
                byte_len: aligned_len,
            })
        }

        /// Allocates a DX12 timeline fence with `D3D12_FENCE_FLAG_SHARED` and imports it
        /// zero-copy into CUDA as an external semaphore.
        pub fn create_shared_timeline_fence(
            self: &Arc<Self>,
            device: &ID3D12Device,
        ) -> Result<SharedTimelineFence, VelocityIoError> {
            // SAFETY: creates a shared timeline fence starting at value 0.
            let fence: ID3D12Fence = unsafe {
                device.CreateFence(0, D3D12_FENCE_FLAG_SHARED).map_err(|e| {
                    VelocityIoError::ComputeDispatchFailed(format!(
                        "CreateFence(SHARED) failed: {e}"
                    ))
                })?
            };
            let child: ID3D12DeviceChild = fence.cast().map_err(|e| {
                VelocityIoError::ComputeDispatchFailed(format!("cast fence ID3D12DeviceChild: {e}"))
            })?;

            // SAFETY: exports a Win32 NT handle for the shared timeline fence.
            let nt_handle = unsafe {
                device
                    .CreateSharedHandle(&child, None, GENERIC_ALL_ACCESS, PCWSTR::null())
                    .map_err(|e| {
                        VelocityIoError::ComputeDispatchFailed(format!(
                            "CreateSharedHandle(fence) failed: {e}"
                        ))
                    })?
            };

            let import_res = self.import_shared_fence(nt_handle.0 as usize);
            close_nt_handle(nt_handle);
            let cuda_handle = import_res?;

            Ok(SharedTimelineFence {
                bridge: Arc::clone(self),
                fence,
                cuda_handle,
                next_value: AtomicU64::new(1),
            })
        }
    }

    impl GpuComputeBackend for CudaOxideBridge {
        fn backend_name(&self) -> &'static str {
            "cuda-oxide-sm89-dx12-interop"
        }

        fn import_shared_buffer(
            &self,
            nt_handle: usize,
            byte_len: u64,
            dedicated: bool,
        ) -> Result<SharedBufferHandle, VelocityIoError> {
            // SAFETY: caller passes a live Win32 NT handle exported from D3D12.
            let id = unsafe {
                self.ctx
                    .import_shared_buffer(nt_handle as *mut c_void, byte_len, dedicated)
                    .map_err(|e| {
                        VelocityIoError::ComputeDispatchFailed(format!(
                            "cuImportExternalMemory failed: {e}"
                        ))
                    })?
            };
            Ok(SharedBufferHandle {
                id: id.0,
                byte_len,
            })
        }

        fn release_shared_buffer(&self, handle: SharedBufferHandle) {
            self.ctx.release_shared_buffer(ImportedBufferId(handle.id));
        }

        fn import_shared_fence(
            &self,
            nt_handle: usize,
        ) -> Result<SharedFenceHandle, VelocityIoError> {
            // SAFETY: caller passes a live Win32 NT handle exported from a D3D12 fence.
            let id = unsafe {
                self.ctx
                    .import_shared_fence(nt_handle as *mut c_void)
                    .map_err(|e| {
                        VelocityIoError::ComputeDispatchFailed(format!(
                            "cuImportExternalSemaphore failed: {e}"
                        ))
                    })?
            };
            Ok(SharedFenceHandle { id: id.0 })
        }

        fn release_shared_fence(&self, handle: SharedFenceHandle) {
            self.ctx.release_shared_fence(ImportedFenceId(handle.id));
        }

        fn enqueue_detile_batch(
            &self,
            host_tiled_bytes: &[u8],
            dst_linear_vram: SharedBufferHandle,
            dispatches: &[DetileDispatchDesc],
            wait: Option<(SharedFenceHandle, u64)>,
            signal: Option<(SharedFenceHandle, u64)>,
        ) -> Result<(), VelocityIoError> {
            let mapped: Vec<cuda_accel::DetileDispatchDesc> =
                dispatches.iter().map(map_detile_desc).collect();
            self.ctx
                .enqueue_detile_batch(
                    host_tiled_bytes,
                    ImportedBufferId(dst_linear_vram.id),
                    &mapped,
                    wait.map(|(h, v)| (ImportedFenceId(h.id), v)),
                    signal.map(|(h, v)| (ImportedFenceId(h.id), v)),
                )
                .map_err(|e| {
                    VelocityIoError::ComputeDispatchFailed(format!(
                        "CUDA detile batch dispatch failed: {e}"
                    ))
                })
        }

        fn enqueue_decompress_batch(
            &self,
            codec: DecompressionCodec,
            host_compressed_bytes: &[u8],
            dst_uncompressed_vram: SharedBufferHandle,
            blocks: &[DecompressBlockDesc],
            wait: Option<(SharedFenceHandle, u64)>,
            signal: Option<(SharedFenceHandle, u64)>,
        ) -> Result<(), VelocityIoError> {
            let cuda_codec = match codec {
                DecompressionCodec::Lz4 => CudaDecompressCodec::Lz4RawBlock,
                DecompressionCodec::GDeflate => CudaDecompressCodec::GDeflateV1RawDeflateTile,
                other => {
                    return Err(VelocityIoError::DecompressionFailed {
                        codec: other,
                        reason: format!("codec {other:?} is not routed to cuda-oxide"),
                    });
                }
            };
            let mapped: Vec<cuda_accel::DecompressBlockDesc> =
                blocks.iter().map(map_block_desc).collect();
            self.ctx
                .enqueue_decompress_batch(
                    cuda_codec,
                    host_compressed_bytes,
                    ImportedBufferId(dst_uncompressed_vram.id),
                    &mapped,
                    wait.map(|(h, v)| (ImportedFenceId(h.id), v)),
                    signal.map(|(h, v)| (ImportedFenceId(h.id), v)),
                )
                .map_err(|e| VelocityIoError::DecompressionFailed {
                    codec,
                    reason: format!("CUDA decompression dispatch failed: {e}"),
                })
        }

        fn compact_readback_deltas(
            &self,
            now_vram: SharedBufferHandle,
            held_vram: SharedBufferHandle,
            pieces: &[ReadbackPieceDesc],
            out_pairs: &mut [ReadbackDeltaPair],
            wait: Option<(SharedFenceHandle, u64)>,
        ) -> Result<Option<usize>, VelocityIoError> {
            let mapped_pieces: Vec<cuda_accel::ReadbackPieceDesc> =
                pieces.iter().map(map_piece_desc).collect();
            let mut tmp_pairs = vec![cuda_accel::ReadbackDeltaPair::default(); out_pairs.len()];
            let res = self
                .ctx
                .compact_readback_deltas(
                    ImportedBufferId(now_vram.id),
                    ImportedBufferId(held_vram.id),
                    &mapped_pieces,
                    &mut tmp_pairs,
                    wait.map(|(h, v)| (ImportedFenceId(h.id), v)),
                    false,
                )
                .map_err(|e| {
                    VelocityIoError::ComputeDispatchFailed(format!(
                        "CUDA readback delta compaction failed: {e}"
                    ))
                })?;
            if let Some(count) = res {
                for (dst, src) in out_pairs[..count].iter_mut().zip(&tmp_pairs[..count]) {
                    *dst = ReadbackDeltaPair {
                        piece_id: src.piece_id,
                        dword_offset: src.dword_offset,
                        value: src.value,
                    };
                }
            }
            Ok(res)
        }

        fn synchronize(&self) -> Result<(), VelocityIoError> {
            self.ctx.synchronize().map_err(|e| {
                VelocityIoError::ComputeDispatchFailed(format!("CUDA synchronize failed: {e}"))
            })
        }
    }

    /// RAII zero-copy DX12 VRAM buffer simultaneously mapped in CUDA.
    pub struct SharedVramBuffer {
        bridge: Arc<CudaOxideBridge>,
        resource: ID3D12Resource,
        cuda_handle: SharedBufferHandle,
        byte_len: u64,
    }

    // SAFETY: D3D12 resources and CUDA external memory handles are thread-safe under caller sync.
    unsafe impl Send for SharedVramBuffer {}
    unsafe impl Sync for SharedVramBuffer {}

    impl SharedVramBuffer {
        #[inline]
        pub fn d3d12_resource(&self) -> &ID3D12Resource {
            &self.resource
        }

        #[inline]
        pub fn cuda_handle(&self) -> SharedBufferHandle {
            self.cuda_handle
        }

        #[inline]
        pub fn byte_len(&self) -> u64 {
            self.byte_len
        }
    }

    impl Drop for SharedVramBuffer {
        fn drop(&mut self) {
            // Release CUDA external mapping before dropping the owning DX12 resource.
            self.bridge.release_shared_buffer(self.cuda_handle);
        }
    }

    /// RAII zero-copy DX12 timeline fence simultaneously mapped as a CUDA external semaphore.
    pub struct SharedTimelineFence {
        bridge: Arc<CudaOxideBridge>,
        fence: ID3D12Fence,
        cuda_handle: SharedFenceHandle,
        next_value: AtomicU64,
    }

    // SAFETY: D3D12 fences and CUDA external semaphores are free-threaded.
    unsafe impl Send for SharedTimelineFence {}
    unsafe impl Sync for SharedTimelineFence {}

    impl SharedTimelineFence {
        #[inline]
        pub fn d3d12_fence(&self) -> &ID3D12Fence {
            &self.fence
        }

        #[inline]
        pub fn cuda_handle(&self) -> SharedFenceHandle {
            self.cuda_handle
        }

        #[inline]
        pub fn next_signal_value(&self) -> u64 {
            self.next_value.fetch_add(1, Ordering::Relaxed)
        }
    }

    impl Drop for SharedTimelineFence {
        fn drop(&mut self) {
            // Release CUDA external semaphore before dropping the owning DX12 fence.
            self.bridge.release_shared_fence(self.cuda_handle);
        }
    }

    /// Process-wide lazily initialized [`CudaOxideBridge`] matched to the host D3D12 adapter.
    pub fn global_bridge() -> Option<Arc<CudaOxideBridge>> {
        if !is_cuda_oxide_enabled() {
            return None;
        }
        static BRIDGE: OnceLock<Option<Arc<CudaOxideBridge>>> = OnceLock::new();
        BRIDGE
            .get_or_init(|| {
                let res = if let Some(host) = crate::host::host() {
                    if !host.is_nvidia() {
                        return None;
                    }
                    CudaOxideBridge::for_d3d12_device(&host.device)
                } else {
                    CudaOxideBridge::default_device()
                };
                match res {
                    Ok(bridge) => Some(bridge),
                    Err(err) => {
                        tracing::warn!(
                            error = %err,
                            "cuda-oxide initialization unavailable; falling back to native CPU/DX12 path"
                        );
                        None
                    }
                }
            })
            .clone()
    }
}

#[cfg(feature = "cuda-oxide")]
pub use imp::{CudaOxideBridge, SharedTimelineFence, SharedVramBuffer, d3d12_adapter_luid, global_bridge};

/// Returns the active [`GpuComputeBackend`] (`CudaOxideBridge` when `--features cuda-oxide` is
/// enabled on an NVIDIA adapter, otherwise [`NativeFallbackComputeBackend`]).
pub fn active_compute_backend() -> Arc<dyn GpuComputeBackend> {
    #[cfg(feature = "cuda-oxide")]
    if let Some(bridge) = imp::global_bridge() {
        return bridge;
    }
    static FALLBACK: OnceLock<Arc<dyn GpuComputeBackend>> = OnceLock::new();
    Arc::clone(FALLBACK.get_or_init(|| Arc::new(NativeFallbackComputeBackend)))
}

#[cfg(all(test, feature = "cuda-oxide"))]
mod tests {
    use super::*;
    use crate::graphics::contracts::{
        DetileDispatchDesc, GpuComputeBackend, ReadbackDeltaPair, ReadbackPieceDesc,
    };
    use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::core::Interface;

    /// End-to-end zero-copy DX12 ↔ CUDA verification on the live RTX 4080:
    /// 1. Creates a DX12 `D3D12_HEAP_FLAG_SHARED` VRAM buffer + `D3D12_FENCE_FLAG_SHARED` timeline fence.
    /// 2. Imports both into CUDA via Win32 NT handles (`CreateSharedHandle` -> `cuImportExternal*`).
    /// 3. Detiles a 256×256 BC7 9-mip Xbox RDNA2 texture directly into the shared VRAM buffer on CUDA
    ///    and signals the shared fence (`value = 1`).
    /// 4. Waits on that exact timeline fence from a DX12 Copy Queue (`queue.Wait(&fence, 1)`), copies
    ///    from the shared VRAM buffer into a DX12 readback buffer, and verifies 100% bit-exact parity
    ///    against CPU `Layout::detile`!
    #[test]
    fn zero_copy_dx12_cuda_detile_and_timeline_fence_parity() {
        let Some(host) = crate::host::host() else {
            return;
        };
        if !host.is_nvidia() {
            return;
        }
        let bridge = CudaOxideBridge::for_d3d12_device(&host.device)
            .expect("CudaOxideBridge must initialize on NVIDIA RTX GPU");

        let desc = Desc {
            dimension: Dimension::Tex2d,
            width: 256,
            height: 256,
            depth: 1,
            mips: 9,
            element_bytes: 16,
            block: 4,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S,
        };
        let layout = Layout::new(&desc).unwrap();
        let mut tiled = vec![0u8; layout.size as usize];
        for (i, b) in tiled.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(37) ^ (i >> 7)) & 0xFF) as u8;
        }

        // Compute linear layout for mip 0 (64x64 BC7 blocks = 64 * 1024 = 65536 B).
        let row_pitch = 1024usize;
        let rows = 64usize;
        let linear_len = row_pitch * rows;
        let mut expected = vec![0u8; linear_len];
        layout
            .detile(&tiled, 0, 0, &mut expected, row_pitch, linear_len)
            .unwrap();

        let shared_buf = bridge
            .create_shared_vram_buffer(&host.device, linear_len as u64)
            .expect("create_shared_vram_buffer");
        let shared_fence = bridge
            .create_shared_timeline_fence(&host.device)
            .expect("create_shared_timeline_fence");

        let d = cuda_accel::detile_desc_from_layout(
            &layout,
            0,
            0,
            0,
            0,
            row_pitch as u32,
            linear_len as u32,
        )
        .unwrap();
        let dispatch = DetileDispatchDesc {
            src_base_offset: d.src_base_offset,
            dst_base_offset: d.dst_base_offset,
            width_elements: d.width_elements,
            height_elements: d.height_elements,
            depth_slices: d.depth_slices,
            element_bytes: d.element_bytes,
            block_dims: d.block_dims,
            block_log2: d.block_log2,
            mip_pitch_elements: d.mip_pitch_elements,
            dst_row_pitch_bytes: d.dst_row_pitch_bytes,
            dst_depth_pitch_bytes: d.dst_depth_pitch_bytes,
            tail_x: d.tail_x,
            tail_y: d.tail_y,
            pattern_id: d.pattern_id,
            thick: d.thick,
            slice_size: d.slice_size,
            first_slice: d.first_slice,
            _pad: d._pad,
        };

        let fence_val = shared_fence.next_signal_value();
        bridge
            .enqueue_detile_batch(
                &tiled,
                shared_buf.cuda_handle(),
                &[dispatch],
                None,
                Some((shared_fence.cuda_handle(), fence_val)),
            )
            .expect("enqueue_detile_batch");

        // DX12 Copy Queue waits on CUDA's shared timeline fence (zero CPU stall) and copies to readback.
        let readback = host.read_back_buffer(linear_len as u64).unwrap();
        unsafe {
            let queue: ID3D12CommandQueue = host
                .device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_COPY,
                    ..Default::default()
                })
                .unwrap();
            let alloc: ID3D12CommandAllocator = host
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)
                .unwrap();
            let list: ID3D12GraphicsCommandList = host
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &alloc, None)
                .unwrap();
            list.CopyBufferRegion(&readback, 0, shared_buf.d3d12_resource(), 0, linear_len as u64);
            list.Close().unwrap();

            // GPU-side wait for CUDA signal!
            queue.Wait(shared_fence.d3d12_fence(), fence_val).unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);

            let done_fence: ID3D12Fence = host
                .device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .unwrap();
            queue.Signal(&done_fence, 1).unwrap();
            done_fence
                .SetEventOnCompletion(1, windows::Win32::Foundation::HANDLE::default())
                .unwrap();

            let mut mapped = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut mapped)).unwrap();
            let got = std::slice::from_raw_parts(mapped.cast::<u8>(), linear_len);
            assert_eq!(
                got,
                expected.as_slice(),
                "zero-copy DX12 <-> CUDA detiled VRAM must match CPU Layout::detile bit-for-bit"
            );
            readback.Unmap(0, None);
        }
    }

    /// Verifies Kernel 3 (`compact_readback_deltas`) directly on two DX12-shared VRAM buffers.
    #[test]
    fn zero_copy_dx12_cuda_readback_delta_compaction() {
        let Some(host) = crate::host::host() else {
            return;
        };
        if !host.is_nvidia() {
            return;
        }
        let bridge = CudaOxideBridge::for_d3d12_device(&host.device)
            .expect("CudaOxideBridge must initialize on NVIDIA RTX GPU");

        let piece_bytes = 65536usize;
        let num_pieces = 4usize;
        let total_bytes = piece_bytes * num_pieces;

        let mut held_host = vec![0xA5u8; total_bytes];
        let mut now_host = held_host.clone();
        // Mutate 12 specific dwords across pieces 0..4.
        let mut expected = Vec::new();
        for piece in 0..num_pieces {
            for k in [0usize, 17, 4095, 16383] {
                let val = 0xDEAD_0000u32 | ((piece as u32) << 12) | (k as u32 & 0x0FFF);
                let byte_off = piece * piece_bytes + k * 4;
                now_host[byte_off..byte_off + 4].copy_from_slice(&val.to_le_bytes());
                expected.push((piece as u32, k as u32, val));
            }
        }
        // Keep held_host immutable so borrow is clean.
        let _ = &mut held_host;

        let now_buf = bridge
            .create_shared_vram_buffer(&host.device, total_bytes as u64)
            .unwrap();
        let held_buf = bridge
            .create_shared_vram_buffer(&host.device, total_bytes as u64)
            .unwrap();

        // Populate both DX12 shared VRAM buffers via DX12 upload buffer + CopyBufferRegion.
        let (upload_now, now_ptr) = host.upload_buffer(total_bytes as u64).unwrap();
        let (upload_held, held_ptr) = host.upload_buffer(total_bytes as u64).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(now_host.as_ptr(), now_ptr, total_bytes);
            std::ptr::copy_nonoverlapping(held_host.as_ptr(), held_ptr, total_bytes);

            let list = host.new_list().unwrap();
            list.close();
            list.reset();
            list.list
                .CopyBufferRegion(now_buf.d3d12_resource(), 0, &upload_now, 0, total_bytes as u64);
            list.list
                .CopyBufferRegion(held_buf.d3d12_resource(), 0, &upload_held, 0, total_bytes as u64);
            list.close();
            host.execute(&[&list]);
        }

        let pieces: Vec<ReadbackPieceDesc> = (0..num_pieces)
            .map(|i| ReadbackPieceDesc {
                now_offset: (i * piece_bytes) as u64,
                held_offset: (i * piece_bytes) as u64,
                dword_count: (piece_bytes / 4) as u32,
                piece_id: i as u32,
            })
            .collect();

        let mut out_pairs = vec![ReadbackDeltaPair::default(); 256];
        let count = bridge
            .compact_readback_deltas(
                now_buf.cuda_handle(),
                held_buf.cuda_handle(),
                &pieces,
                &mut out_pairs,
                None,
            )
            .unwrap()
            .expect("sparse deltas should fit in output buffer");

        let mut got: Vec<(u32, u32, u32)> = out_pairs[..count]
            .iter()
            .map(|p| (p.piece_id, p.dword_offset, p.value))
            .collect();
        got.sort_unstable();
        expected.sort_unstable();
        assert_eq!(got, expected);
    }
}
