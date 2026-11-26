//! `cuda-accel` — Optional NVIDIA CUDA / `cuda-oxide` GPU acceleration runtime for Victory.
//!
//! Provides zero-copy DX12 ↔ CUDA Win32 NT shared-handle interop (`ExternalMemory` +
//! `ExternalSemaphore`) and SM 8.9 PTX dispatch for:
//! 1. **Xbox RDNA2 (GFX10.3 RB+) micro-tile texture detiling** (`rdna2_detile_batch`)
//! 2. **Open-spec Microsoft GDeflate v1 multi-tile & LZ4 raw block decompression**
//!    (`gdeflate_v1_tile_decompress`, `lz4_decompress_batch`)
//! 3. **Warp-level sparse GPU-to-CPU readback delta compaction** (`compact_readback_deltas`)
//!
//! Zero link-time dependency on `cuda.lib`: dynamically loads `nvcuda.dll` at runtime and
//! embeds the build-time compiled PTX (`$OUT_DIR/victory_kernels.ptx`).

pub mod driver;
pub mod interop;
pub mod luts;

use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use driver::{
    CU_EVENT_DISABLE_TIMING, CU_MEMHOSTALLOC_PORTABLE, CU_STREAM_NON_BLOCKING, CUcontext,
    CUdevice, CUdeviceptr, CUevent, CUfunction, CUmodule, CUstream, CudaApi, CudaError, CudaResult,
};
use interop::{CudaDeviceInfo, ExternalMemory, ExternalSemaphore, find_device};
pub use luts::{
    DecompressBlockDesc, DetileDispatchDesc, LutMeta, ReadbackDeltaPair, ReadbackPieceDesc,
    detile_desc_from_layout, pattern_id_for_layout,
};

/// Embedded PTX compiled into `$OUT_DIR/victory_kernels.ptx` by `build.rs`.
#[cfg(victory_cuda_ptx_ready)]
const KERNELS_PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/victory_kernels.ptx"));

/// Number of ring slots for non-blocking async HtoD staging + kernel parameter upload.
const RING_SLOTS: usize = 4;

/// Supported codecs for `CudaAccelContext::enqueue_decompress_batch` / `decompress_host_sync`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaDecompressCodec {
    Lz4RawBlock,
    GDeflateV1RawDeflateTile,
}

/// Opaque handle to an imported DX12 shared VRAM buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImportedBufferId(pub u64);

/// Opaque handle to an imported DX12 shared timeline fence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImportedFenceId(pub u64);

thread_local! {
    static BOUND_CTX: Cell<usize> = const { Cell::new(0) };
}

#[inline]
fn ensure_current(api: &'static CudaApi, ctx: CUcontext) -> CudaResult<()> {
    let raw = ctx as usize;
    if BOUND_CTX.with(|c| c.get()) == raw {
        return Ok(());
    }
    // SAFETY: `ctx` is a retained primary context for the lifetime of `CudaAccelContext`.
    api.check("cuCtxSetCurrent", unsafe { (api.cuCtxSetCurrent)(ctx) })?;
    BOUND_CTX.with(|c| c.set(raw));
    Ok(())
}

/// Growable pinned host + device buffer pair for one ring slot.
struct StagingSlot {
    pinned_in: *mut u8,
    pinned_in_cap: usize,
    dev_in: CUdeviceptr,
    dev_in_cap: usize,

    pinned_meta: *mut u8,
    pinned_meta_cap: usize,
    dev_meta: CUdeviceptr,
    dev_meta_cap: usize,

    dev_out_meta: CUdeviceptr,
    dev_out_meta_cap: usize,

    done_event: CUevent,
    in_flight: bool,
}

// SAFETY: `StagingSlot` is only accessed while holding `CudaAccelContext::stream_state`'s Mutex.
unsafe impl Send for StagingSlot {}

impl StagingSlot {
    fn new(api: &'static CudaApi) -> CudaResult<Self> {
        let mut done_event: CUevent = std::ptr::null_mut();
        // SAFETY: creates a timing-disabled CUDA event on the active context.
        api.check("cuEventCreate", unsafe {
            (api.cuEventCreate)(&mut done_event, CU_EVENT_DISABLE_TIMING)
        })?;
        Ok(Self {
            pinned_in: std::ptr::null_mut(),
            pinned_in_cap: 0,
            dev_in: 0,
            dev_in_cap: 0,
            pinned_meta: std::ptr::null_mut(),
            pinned_meta_cap: 0,
            dev_meta: 0,
            dev_meta_cap: 0,
            dev_out_meta: 0,
            dev_out_meta_cap: 0,
            done_event,
            in_flight: false,
        })
    }

    fn wait_ready(&mut self, api: &'static CudaApi) -> CudaResult<()> {
        if self.in_flight {
            // SAFETY: `done_event` was recorded on our stream before setting `in_flight`.
            api.check("cuEventSynchronize", unsafe {
                (api.cuEventSynchronize)(self.done_event)
            })?;
            self.in_flight = false;
        }
        Ok(())
    }

    fn ensure_input(&mut self, api: &'static CudaApi, bytes: usize) -> CudaResult<()> {
        let need = bytes.max(4096).next_power_of_two();
        if need > self.pinned_in_cap {
            if !self.pinned_in.is_null() {
                // SAFETY: `self.pinned_in` was allocated by `cuMemHostAlloc` and slot is idle.
                unsafe { let _ = (api.cuMemFreeHost)(self.pinned_in.cast()); }
                self.pinned_in = std::ptr::null_mut();
                self.pinned_in_cap = 0;
            }
            let mut ptr: *mut c_void = std::ptr::null_mut();
            // SAFETY: allocates `need` bytes of portable pinned host RAM.
            api.check("cuMemHostAlloc(in)", unsafe {
                (api.cuMemHostAlloc)(&mut ptr, need, CU_MEMHOSTALLOC_PORTABLE)
            })?;
            self.pinned_in = ptr.cast();
            self.pinned_in_cap = need;
        }
        if need > self.dev_in_cap {
            if self.dev_in != 0 {
                // SAFETY: `self.dev_in` was allocated by `cuMemAlloc` and slot is idle.
                unsafe { let _ = (api.cuMemFree)(self.dev_in); }
                self.dev_in = 0;
                self.dev_in_cap = 0;
            }
            let mut dptr: CUdeviceptr = 0;
            // SAFETY: allocates `need` bytes of VRAM.
            api.check("cuMemAlloc(in)", unsafe { (api.cuMemAlloc)(&mut dptr, need) })?;
            self.dev_in = dptr;
            self.dev_in_cap = need;
        }
        Ok(())
    }

    fn ensure_meta(&mut self, api: &'static CudaApi, bytes: usize) -> CudaResult<()> {
        let need = bytes.max(4096).next_power_of_two();
        if need > self.pinned_meta_cap {
            if !self.pinned_meta.is_null() {
                // SAFETY: `self.pinned_meta` was allocated by `cuMemHostAlloc` and slot is idle.
                unsafe { let _ = (api.cuMemFreeHost)(self.pinned_meta.cast()); }
                self.pinned_meta = std::ptr::null_mut();
                self.pinned_meta_cap = 0;
            }
            let mut ptr: *mut c_void = std::ptr::null_mut();
            // SAFETY: allocates `need` bytes of portable pinned host RAM.
            api.check("cuMemHostAlloc(meta)", unsafe {
                (api.cuMemHostAlloc)(&mut ptr, need, CU_MEMHOSTALLOC_PORTABLE)
            })?;
            self.pinned_meta = ptr.cast();
            self.pinned_meta_cap = need;
        }
        if need > self.dev_meta_cap {
            if self.dev_meta != 0 {
                // SAFETY: `self.dev_meta` was allocated by `cuMemAlloc` and slot is idle.
                unsafe { let _ = (api.cuMemFree)(self.dev_meta); }
                self.dev_meta = 0;
                self.dev_meta_cap = 0;
            }
            let mut dptr: CUdeviceptr = 0;
            // SAFETY: allocates `need` bytes of VRAM.
            api.check("cuMemAlloc(meta)", unsafe { (api.cuMemAlloc)(&mut dptr, need) })?;
            self.dev_meta = dptr;
            self.dev_meta_cap = need;
        }
        Ok(())
    }

    fn ensure_out_meta(&mut self, api: &'static CudaApi, bytes: usize) -> CudaResult<()> {
        let need = bytes.max(1024).next_power_of_two();
        if need > self.dev_out_meta_cap {
            if self.dev_out_meta != 0 {
                // SAFETY: `self.dev_out_meta` was allocated by `cuMemAlloc` and slot is idle.
                unsafe { let _ = (api.cuMemFree)(self.dev_out_meta); }
                self.dev_out_meta = 0;
                self.dev_out_meta_cap = 0;
            }
            let mut dptr: CUdeviceptr = 0;
            // SAFETY: allocates `need` bytes of VRAM.
            api.check("cuMemAlloc(out_meta)", unsafe { (api.cuMemAlloc)(&mut dptr, need) })?;
            self.dev_out_meta = dptr;
            self.dev_out_meta_cap = need;
        }
        Ok(())
    }

    fn destroy(&mut self, api: &'static CudaApi) {
        // SAFETY: called during `CudaAccelContext::drop` after synchronizing the stream.
        unsafe {
            if !self.pinned_in.is_null() {
                let _ = (api.cuMemFreeHost)(self.pinned_in.cast());
            }
            if self.dev_in != 0 {
                let _ = (api.cuMemFree)(self.dev_in);
            }
            if !self.pinned_meta.is_null() {
                let _ = (api.cuMemFreeHost)(self.pinned_meta.cast());
            }
            if self.dev_meta != 0 {
                let _ = (api.cuMemFree)(self.dev_meta);
            }
            if self.dev_out_meta != 0 {
                let _ = (api.cuMemFree)(self.dev_out_meta);
            }
            if !self.done_event.is_null() {
                let _ = (api.cuEventDestroy)(self.done_event);
            }
        }
    }
}

struct StreamState {
    stream: CUstream,
    slots: Vec<StagingSlot>,
    next_slot: usize,
    /// Scratch buffer for host-sync APIs and delta compaction output.
    dev_scratch_out: CUdeviceptr,
    dev_scratch_out_cap: usize,
}

// SAFETY: accessed exclusively under `CudaAccelContext::stream_state` Mutex.
unsafe impl Send for StreamState {}

impl StreamState {
    fn ensure_scratch_out(&mut self, api: &'static CudaApi, bytes: usize) -> CudaResult<CUdeviceptr> {
        let need = bytes.max(4096).next_power_of_two();
        if need > self.dev_scratch_out_cap {
            if self.dev_scratch_out != 0 {
                // SAFETY: stream synchronized or caller owns exclusive mutex before grow.
                api.check("cuStreamSynchronize(grow)", unsafe {
                    (api.cuStreamSynchronize)(self.stream)
                })?;
                unsafe { let _ = (api.cuMemFree)(self.dev_scratch_out); }
                self.dev_scratch_out = 0;
                self.dev_scratch_out_cap = 0;
            }
            let mut dptr: CUdeviceptr = 0;
            // SAFETY: allocates `need` bytes of VRAM.
            api.check("cuMemAlloc(scratch_out)", unsafe {
                (api.cuMemAlloc)(&mut dptr, need)
            })?;
            self.dev_scratch_out = dptr;
            self.dev_scratch_out_cap = need;
        }
        Ok(self.dev_scratch_out)
    }
}

/// Primary CUDA accelerator context owning the SM 8.9 PTX module, GFX10 swizzle LUTs,
/// async staging ring, and imported DX12 external memory/semaphore tables.
pub struct CudaAccelContext {
    api: &'static CudaApi,
    info: CudaDeviceInfo,
    device: CUdevice,
    ctx: CUcontext,
    module: CUmodule,
    fn_detile: CUfunction,
    fn_lz4: CUfunction,
    fn_gdeflate: CUfunction,
    fn_compact_deltas: CUfunction,
    dev_lut_data: CUdeviceptr,
    dev_lut_meta: CUdeviceptr,
    stream_state: Mutex<StreamState>,
    ext_buffers: Mutex<HashMap<u64, ExternalMemory>>,
    ext_fences: Mutex<HashMap<u64, ExternalSemaphore>>,
    next_id: AtomicU64,
}

// SAFETY: all mutable CUDA state is guarded by `Mutex` and `ensure_current()` binds
// the retained primary context on whichever OS thread invokes an operation.
unsafe impl Send for CudaAccelContext {}
unsafe impl Sync for CudaAccelContext {}

impl CudaAccelContext {
    /// Initializes `CudaAccelContext` on the CUDA device matching `adapter_luid` (or device 0).
    pub fn new(adapter_luid: Option<[u8; 8]>) -> CudaResult<Self> {
        #[cfg(not(victory_cuda_ptx_ready))]
        {
            let _ = adapter_luid;
            return Err(CudaError::PtxUnavailable);
        }

        #[cfg(victory_cuda_ptx_ready)]
        {
            let api = CudaApi::load()?;
            let info = find_device(api, adapter_luid)?;
            let device = info.device;

            let mut ctx: CUcontext = std::ptr::null_mut();
            // SAFETY: retains the primary context for `device`.
            api.check("cuDevicePrimaryCtxRetain", unsafe {
                (api.cuDevicePrimaryCtxRetain)(&mut ctx, device)
            })?;
            ensure_current(api, ctx)?;

            // Ensure null-terminated PTX image for `cuModuleLoadData`.
            let mut ptx_z = Vec::with_capacity(KERNELS_PTX.len() + 1);
            ptx_z.extend_from_slice(KERNELS_PTX);
            ptx_z.push(0);

            let mut module: CUmodule = std::ptr::null_mut();
            // SAFETY: `ptx_z` is NUL-terminated PTX source.
            if let Err(err) = api.check("cuModuleLoadData", unsafe {
                (api.cuModuleLoadData)(&mut module, ptx_z.as_ptr().cast())
            }) {
                // SAFETY: release retained primary context on load failure.
                unsafe { let _ = (api.cuDevicePrimaryCtxRelease)(device); }
                return Err(err);
            }

            let get_fn = |name: &str| -> CudaResult<CUfunction> {
                let cname = CString::new(name).expect("kernel name has no NUL");
                let mut f: CUfunction = std::ptr::null_mut();
                // SAFETY: `module` is loaded and `cname` is NUL-terminated.
                api.check("cuModuleGetFunction", unsafe {
                    (api.cuModuleGetFunction)(&mut f, module, cname.as_ptr())
                })?;
                Ok(f)
            };

            let fn_detile = get_fn("rdna2_detile_batch")?;
            let fn_lz4 = get_fn("lz4_decompress_batch")?;
            let fn_gdeflate = get_fn("gdeflate_v1_tile_decompress")?;
            let fn_compact_deltas = get_fn("compact_readback_deltas")?;

            // Upload all 45 GFX10.3 RB+ swizzle LUTs once to immutable device memory.
            let packed = luts::pack_all_swizzle_luts();
            let lut_data_bytes = std::mem::size_of_val(packed.data.as_slice());
            let lut_meta_bytes = std::mem::size_of_val(packed.meta.as_slice());

            let mut dev_lut_data: CUdeviceptr = 0;
            let mut dev_lut_meta: CUdeviceptr = 0;
            // SAFETY: allocates and initializes device LUT buffers.
            api.check("cuMemAlloc(lut_data)", unsafe {
                (api.cuMemAlloc)(&mut dev_lut_data, lut_data_bytes)
            })?;
            api.check("cuMemcpyHtoD(lut_data)", unsafe {
                (api.cuMemcpyHtoD)(dev_lut_data, packed.data.as_ptr().cast(), lut_data_bytes)
            })?;
            api.check("cuMemAlloc(lut_meta)", unsafe {
                (api.cuMemAlloc)(&mut dev_lut_meta, lut_meta_bytes)
            })?;
            api.check("cuMemcpyHtoD(lut_meta)", unsafe {
                (api.cuMemcpyHtoD)(dev_lut_meta, packed.meta.as_ptr().cast(), lut_meta_bytes)
            })?;

            let mut stream: CUstream = std::ptr::null_mut();
            // SAFETY: creates a non-blocking CUDA compute stream.
            api.check("cuStreamCreate", unsafe {
                (api.cuStreamCreate)(&mut stream, CU_STREAM_NON_BLOCKING)
            })?;

            let mut slots = Vec::with_capacity(RING_SLOTS);
            for _ in 0..RING_SLOTS {
                slots.push(StagingSlot::new(api)?);
            }

            tracing::info!(
                device = %info.name,
                sm = format_args!("{}.{}", info.sm_major, info.sm_minor),
                sms = info.sm_count,
                lut_bytes = lut_data_bytes + lut_meta_bytes,
                "cuda-accel SM 8.9 runtime initialized (zero-copy DX12 NT interop ready)"
            );

            Ok(Self {
                api,
                info,
                device,
                ctx,
                module,
                fn_detile,
                fn_lz4,
                fn_gdeflate,
                fn_compact_deltas,
                dev_lut_data,
                dev_lut_meta,
                stream_state: Mutex::new(StreamState {
                    stream,
                    slots,
                    next_slot: 0,
                    dev_scratch_out: 0,
                    dev_scratch_out_cap: 0,
                }),
                ext_buffers: Mutex::new(HashMap::new()),
                ext_fences: Mutex::new(HashMap::new()),
                next_id: AtomicU64::new(1),
            })
        }
    }

    /// Returns hardware metadata for the bound NVIDIA GPU.
    #[inline]
    pub fn device_info(&self) -> &CudaDeviceInfo {
        &self.info
    }

    /// Imports a DX12-shared VRAM resource (`D3D12_HEAP_FLAG_SHARED`) via its Win32 NT handle.
    ///
    /// # Safety
    /// `nt_handle` must be a valid Win32 NT handle exported via `ID3D12Device::CreateSharedHandle`
    /// for a D3D12 buffer/heap of at least `byte_len` bytes on the same adapter.
    pub unsafe fn import_shared_buffer(
        &self,
        nt_handle: *mut c_void,
        byte_len: u64,
        dedicated: bool,
    ) -> CudaResult<ImportedBufferId> {
        ensure_current(self.api, self.ctx)?;
        // SAFETY: caller guarantees valid DX12 NT handle.
        let mem = unsafe {
            ExternalMemory::import_d3d12_nt_handle(
                self.api, self.ctx, nt_handle, byte_len, dedicated,
            )?
        };
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.ext_buffers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, mem);
        Ok(ImportedBufferId(id))
    }

    /// Releases an imported DX12 shared buffer mapping.
    pub fn release_shared_buffer(&self, id: ImportedBufferId) {
        self.ext_buffers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&id.0);
    }

    /// Imports a DX12-shared timeline fence (`D3D12_FENCE_FLAG_SHARED`) via its Win32 NT handle.
    ///
    /// # Safety
    /// `nt_handle` must be a valid Win32 NT handle exported via `ID3D12Device::CreateSharedHandle`
    /// for an `ID3D12Fence` with `D3D12_FENCE_FLAG_SHARED`.
    pub unsafe fn import_shared_fence(&self, nt_handle: *mut c_void) -> CudaResult<ImportedFenceId> {
        ensure_current(self.api, self.ctx)?;
        // SAFETY: caller guarantees valid DX12 fence NT handle.
        let sem = unsafe { ExternalSemaphore::import_d3d12_fence_nt_handle(self.api, self.ctx, nt_handle)? };
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.ext_fences
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, sem);
        Ok(ImportedFenceId(id))
    }

    /// Releases an imported DX12 shared fence semaphore.
    pub fn release_shared_fence(&self, id: ImportedFenceId) {
        self.ext_fences
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&id.0);
    }

    fn resolve_buffer(&self, id: ImportedBufferId) -> CudaResult<(CUdeviceptr, u64)> {
        let map = self.ext_buffers.lock().unwrap_or_else(|e| e.into_inner());
        let mem = map
            .get(&id.0)
            .ok_or_else(|| CudaError::InvalidArgument(format!("unknown shared buffer id {}", id.0)))?;
        Ok((mem.device_ptr(), mem.byte_len()))
    }

    fn wait_fence_on_stream(&self, stream: CUstream, wait: Option<(ImportedFenceId, u64)>) -> CudaResult<()> {
        if let Some((fid, val)) = wait {
            let map = self.ext_fences.lock().unwrap_or_else(|e| e.into_inner());
            let sem = map
                .get(&fid.0)
                .ok_or_else(|| CudaError::InvalidArgument(format!("unknown shared fence id {}", fid.0)))?;
            sem.wait_async(stream, val)?;
        }
        Ok(())
    }

    fn signal_fence_on_stream(&self, stream: CUstream, signal: Option<(ImportedFenceId, u64)>) -> CudaResult<()> {
        if let Some((fid, val)) = signal {
            let map = self.ext_fences.lock().unwrap_or_else(|e| e.into_inner());
            let sem = map
                .get(&fid.0)
                .ok_or_else(|| CudaError::InvalidArgument(format!("unknown shared fence id {}", fid.0)))?;
            sem.signal_async(stream, val)?;
        }
        Ok(())
    }

    /// Asynchronously uploads `host_tiled_bytes` and detiles all `dispatches` directly into
    /// a DX12-shared VRAM buffer (`dst_vram`), with optional DX12 timeline semaphore wait/signal.
    ///
    /// Non-blocking: returns immediately once the HtoD copy, `rdna2_detile_batch` launch, and
    /// semaphore signal are queued on the CUDA stream.
    pub fn enqueue_detile_batch(
        &self,
        host_tiled_bytes: &[u8],
        dst_vram: ImportedBufferId,
        dispatches: &[DetileDispatchDesc],
        wait: Option<(ImportedFenceId, u64)>,
        signal: Option<(ImportedFenceId, u64)>,
    ) -> CudaResult<()> {
        let (dst_ptr, _dst_len) = self.resolve_buffer(dst_vram)?;
        self.enqueue_detile_into_dev_ptr(host_tiled_bytes, dst_ptr, dispatches, wait, signal)
    }

    fn enqueue_detile_into_dev_ptr(
        &self,
        host_tiled_bytes: &[u8],
        dst_ptr: CUdeviceptr,
        dispatches: &[DetileDispatchDesc],
        wait: Option<(ImportedFenceId, u64)>,
        signal: Option<(ImportedFenceId, u64)>,
    ) -> CudaResult<()> {
        ensure_current(self.api, self.ctx)?;
        let mut st = self.stream_state.lock().unwrap_or_else(|e| e.into_inner());
        let stream = st.stream;

        if dispatches.is_empty() {
            self.wait_fence_on_stream(stream, wait)?;
            self.signal_fence_on_stream(stream, signal)?;
            return Ok(());
        }

        // Build exclusive block-count prefix sum so one 1D launch covers the entire batch with zero waste.
        let mut prefix = Vec::with_capacity(dispatches.len() + 1);
        let mut total_blocks = 0u32;
        prefix.push(0u32);
        for d in dispatches {
            let elems = u64::from(d.width_elements)
                * u64::from(d.height_elements)
                * u64::from(d.depth_slices);
            let blks = elems.div_ceil(256) as u32;
            total_blocks = total_blocks
                .checked_add(blks)
                .ok_or_else(|| CudaError::InvalidArgument("detile batch exceeds u32 grid blocks".into()))?;
            prefix.push(total_blocks);
        }

        if total_blocks == 0 {
            self.wait_fence_on_stream(stream, wait)?;
            self.signal_fence_on_stream(stream, signal)?;
            return Ok(());
        }

        let slot_idx = st.next_slot % RING_SLOTS;
        st.next_slot = st.next_slot.wrapping_add(1);
        let slot = &mut st.slots[slot_idx];
        slot.wait_ready(self.api)?;

        let desc_bytes = std::mem::size_of_val(dispatches);
        let prefix_offset = desc_bytes.next_multiple_of(16);
        let prefix_bytes = std::mem::size_of_val(prefix.as_slice());
        let total_meta_bytes = prefix_offset + prefix_bytes;

        slot.ensure_input(self.api, host_tiled_bytes.len())?;
        slot.ensure_meta(self.api, total_meta_bytes)?;

        // Copy host tiled bytes and metadata into pinned staging memory for full-speed PCIe DMA.
        // SAFETY: `slot.pinned_in` and `slot.pinned_meta` are sized >= requested lengths and idle.
        unsafe {
            if !host_tiled_bytes.is_empty() {
                std::ptr::copy_nonoverlapping(
                    host_tiled_bytes.as_ptr(),
                    slot.pinned_in,
                    host_tiled_bytes.len(),
                );
            }
            std::ptr::copy_nonoverlapping(
                dispatches.as_ptr().cast::<u8>(),
                slot.pinned_meta,
                desc_bytes,
            );
            std::ptr::copy_nonoverlapping(
                prefix.as_ptr().cast::<u8>(),
                slot.pinned_meta.add(prefix_offset),
                prefix_bytes,
            );
        }

        self.wait_fence_on_stream(stream, wait)?;

        if !host_tiled_bytes.is_empty() {
            // SAFETY: async HtoD from pinned host staging to device input buffer.
            self.api.check("cuMemcpyHtoDAsync(detile_in)", unsafe {
                (self.api.cuMemcpyHtoDAsync)(
                    slot.dev_in,
                    slot.pinned_in.cast(),
                    host_tiled_bytes.len(),
                    stream,
                )
            })?;
        }
        // SAFETY: async HtoD for dispatch descriptors + block prefix table.
        self.api.check("cuMemcpyHtoDAsync(detile_meta)", unsafe {
            (self.api.cuMemcpyHtoDAsync)(
                slot.dev_meta,
                slot.pinned_meta.cast(),
                total_meta_bytes,
                stream,
            )
        })?;

        let mut arg_src = slot.dev_in;
        let mut arg_dst = dst_ptr;
        let mut arg_descs = slot.dev_meta;
        let mut arg_prefix = slot.dev_meta + prefix_offset as u64;
        let mut arg_num = dispatches.len() as u32;
        let mut arg_lut_data = self.dev_lut_data;
        let mut arg_lut_meta = self.dev_lut_meta;
        let mut params: [*mut c_void; 7] = [
            (&mut arg_src as *mut CUdeviceptr).cast(),
            (&mut arg_dst as *mut CUdeviceptr).cast(),
            (&mut arg_descs as *mut CUdeviceptr).cast(),
            (&mut arg_prefix as *mut CUdeviceptr).cast(),
            (&mut arg_num as *mut u32).cast(),
            (&mut arg_lut_data as *mut CUdeviceptr).cast(),
            (&mut arg_lut_meta as *mut CUdeviceptr).cast(),
        ];

        // SAFETY: kernel signature matches `rdna2_detile_batch` in `victory_kernels.cu`.
        self.api.check("cuLaunchKernel(rdna2_detile_batch)", unsafe {
            (self.api.cuLaunchKernel)(
                self.fn_detile,
                total_blocks,
                1,
                1,
                256,
                1,
                1,
                0,
                stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })?;

        // SAFETY: records completion event so this ring slot isn't overwritten while in flight.
        self.api.check("cuEventRecord(detile)", unsafe {
            (self.api.cuEventRecord)(slot.done_event, stream)
        })?;
        slot.in_flight = true;

        self.signal_fence_on_stream(stream, signal)?;
        Ok(())
    }

    /// Synchronous host-to-GPU-to-host detile helper for bit-exact unit tests and benchmarks.
    pub fn detile_host_sync(
        &self,
        host_tiled_bytes: &[u8],
        dst_len: usize,
        dispatches: &[DetileDispatchDesc],
    ) -> CudaResult<Vec<u8>> {
        ensure_current(self.api, self.ctx)?;
        let dst_ptr = {
            let mut st = self.stream_state.lock().unwrap_or_else(|e| e.into_inner());
            let ptr = st.ensure_scratch_out(self.api, dst_len)?;
            if dst_len > 0 {
                // Zero scratch padding bytes so row-pitch/subresource gaps are deterministic.
                // SAFETY: `ptr` has at least `dst_len` bytes on the active CUDA context.
                self.api.check("cuMemsetD8Async(detile_out)", unsafe {
                    (self.api.cuMemsetD8Async)(ptr, 0, dst_len, st.stream)
                })?;
            }
            ptr
        };
        self.enqueue_detile_into_dev_ptr(host_tiled_bytes, dst_ptr, dispatches, None, None)?;
        let mut out = vec![0u8; dst_len];
        let st = self.stream_state.lock().unwrap_or_else(|e| e.into_inner());
        if dst_len > 0 {
            // SAFETY: copies detiled result from device scratch buffer back to host and syncs stream.
            self.api.check("cuMemcpyDtoHAsync(detile_out)", unsafe {
                (self.api.cuMemcpyDtoHAsync)(out.as_mut_ptr().cast(), dst_ptr, dst_len, st.stream)
            })?;
        }
        self.api.check("cuStreamSynchronize(detile_host_sync)", unsafe {
            (self.api.cuStreamSynchronize)(st.stream)
        })?;
        Ok(out)
    }

    /// Asynchronously decompresses a batch of LZ4 raw blocks or GDeflate v1 raw DEFLATE tiles
    /// directly into a DX12-shared VRAM buffer (`dst_vram`).
    pub fn enqueue_decompress_batch(
        &self,
        codec: CudaDecompressCodec,
        host_compressed_bytes: &[u8],
        dst_vram: ImportedBufferId,
        blocks: &[DecompressBlockDesc],
        wait: Option<(ImportedFenceId, u64)>,
        signal: Option<(ImportedFenceId, u64)>,
    ) -> CudaResult<()> {
        let (dst_ptr, _dst_len) = self.resolve_buffer(dst_vram)?;
        self.enqueue_decompress_into_dev_ptr(
            codec,
            host_compressed_bytes,
            dst_ptr,
            blocks,
            wait,
            signal,
            false,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_decompress_into_dev_ptr(
        &self,
        codec: CudaDecompressCodec,
        host_compressed_bytes: &[u8],
        dst_ptr: CUdeviceptr,
        blocks: &[DecompressBlockDesc],
        wait: Option<(ImportedFenceId, u64)>,
        signal: Option<(ImportedFenceId, u64)>,
        readback_status: bool,
    ) -> CudaResult<Vec<u32>> {
        ensure_current(self.api, self.ctx)?;
        let mut st = self.stream_state.lock().unwrap_or_else(|e| e.into_inner());
        let stream = st.stream;

        if blocks.is_empty() {
            self.wait_fence_on_stream(stream, wait)?;
            self.signal_fence_on_stream(stream, signal)?;
            return Ok(Vec::new());
        }

        let slot_idx = st.next_slot % RING_SLOTS;
        st.next_slot = st.next_slot.wrapping_add(1);
        let slot = &mut st.slots[slot_idx];
        slot.wait_ready(self.api)?;

        let desc_bytes = std::mem::size_of_val(blocks);
        let status_bytes = blocks.len() * std::mem::size_of::<u32>() * 2;

        slot.ensure_input(self.api, host_compressed_bytes.len())?;
        slot.ensure_meta(self.api, desc_bytes)?;
        slot.ensure_out_meta(self.api, status_bytes)?;

        // SAFETY: pinned staging buffers are sized >= payload and slot is idle.
        unsafe {
            if !host_compressed_bytes.is_empty() {
                std::ptr::copy_nonoverlapping(
                    host_compressed_bytes.as_ptr(),
                    slot.pinned_in,
                    host_compressed_bytes.len(),
                );
            }
            std::ptr::copy_nonoverlapping(blocks.as_ptr().cast::<u8>(), slot.pinned_meta, desc_bytes);
        }

        self.wait_fence_on_stream(stream, wait)?;

        if !host_compressed_bytes.is_empty() {
            // SAFETY: async HtoD compressed payload upload.
            self.api.check("cuMemcpyHtoDAsync(decomp_in)", unsafe {
                (self.api.cuMemcpyHtoDAsync)(
                    slot.dev_in,
                    slot.pinned_in.cast(),
                    host_compressed_bytes.len(),
                    stream,
                )
            })?;
        }
        // SAFETY: async HtoD descriptor upload.
        self.api.check("cuMemcpyHtoDAsync(decomp_meta)", unsafe {
            (self.api.cuMemcpyHtoDAsync)(slot.dev_meta, slot.pinned_meta.cast(), desc_bytes, stream)
        })?;

        let written_ptr = slot.dev_out_meta;
        let status_ptr = slot.dev_out_meta + (blocks.len() * std::mem::size_of::<u32>()) as u64;

        let mut arg_src = slot.dev_in;
        let mut arg_dst = dst_ptr;
        let mut arg_blocks = slot.dev_meta;
        let mut arg_num = blocks.len() as u32;
        let mut arg_written = written_ptr;
        let mut arg_status = status_ptr;
        let mut params: [*mut c_void; 6] = [
            (&mut arg_src as *mut CUdeviceptr).cast(),
            (&mut arg_dst as *mut CUdeviceptr).cast(),
            (&mut arg_blocks as *mut CUdeviceptr).cast(),
            (&mut arg_num as *mut u32).cast(),
            (&mut arg_written as *mut CUdeviceptr).cast(),
            (&mut arg_status as *mut CUdeviceptr).cast(),
        ];

        let (func, grid_x, block_x) = match codec {
            CudaDecompressCodec::Lz4RawBlock => {
                // 1 CTA per LZ4 block spreads across all 76 SMs with zero warp divergence.
                (self.fn_lz4, blocks.len() as u32, 1u32)
            }
            CudaDecompressCodec::GDeflateV1RawDeflateTile => {
                // 1 warp (32 threads) per 64 KiB independent raw DEFLATE tile.
                (self.fn_gdeflate, blocks.len() as u32, 32u32)
            }
        };

        // SAFETY: parameters match `lz4_decompress_batch` and `gdeflate_v1_tile_decompress`.
        self.api.check("cuLaunchKernel(decompress)", unsafe {
            (self.api.cuLaunchKernel)(
                func,
                grid_x,
                1,
                1,
                block_x,
                1,
                1,
                0,
                stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })?;

        self.api.check("cuEventRecord(decompress)", unsafe {
            (self.api.cuEventRecord)(slot.done_event, stream)
        })?;
        slot.in_flight = true;

        self.signal_fence_on_stream(stream, signal)?;

        if !readback_status {
            return Ok(Vec::new());
        }

        let mut written = vec![0u32; blocks.len()];
        let mut statuses = vec![0u32; blocks.len()];
        // SAFETY: reads back per-block output sizes and error status codes on stream.
        self.api.check("cuMemcpyDtoHAsync(written)", unsafe {
            (self.api.cuMemcpyDtoHAsync)(
                written.as_mut_ptr().cast(),
                written_ptr,
                blocks.len() * std::mem::size_of::<u32>(),
                stream,
            )
        })?;
        self.api.check("cuMemcpyDtoHAsync(status)", unsafe {
            (self.api.cuMemcpyDtoHAsync)(
                statuses.as_mut_ptr().cast(),
                status_ptr,
                blocks.len() * std::mem::size_of::<u32>(),
                stream,
            )
        })?;
        self.api.check("cuStreamSynchronize(decompress)", unsafe {
            (self.api.cuStreamSynchronize)(stream)
        })?;
        slot.in_flight = false;

        for (index, &st_code) in statuses.iter().enumerate() {
            if st_code != 0 {
                return Err(CudaError::DecompressKernelFailed {
                    index,
                    status: st_code,
                });
            }
        }
        Ok(written)
    }

    /// Synchronous host-to-GPU-to-host decompression helper returning `(bytes, per_block_written)`.
    pub fn decompress_host_sync(
        &self,
        codec: CudaDecompressCodec,
        host_compressed_bytes: &[u8],
        dst_total_len: usize,
        blocks: &[DecompressBlockDesc],
    ) -> CudaResult<(Vec<u8>, Vec<u32>)> {
        ensure_current(self.api, self.ctx)?;
        let dst_ptr = {
            let mut st = self.stream_state.lock().unwrap_or_else(|e| e.into_inner());
            st.ensure_scratch_out(self.api, dst_total_len)?
        };
        let written = self.enqueue_decompress_into_dev_ptr(
            codec,
            host_compressed_bytes,
            dst_ptr,
            blocks,
            None,
            None,
            true,
        )?;
        let mut out = vec![0u8; dst_total_len];
        if dst_total_len > 0 {
            let st = self.stream_state.lock().unwrap_or_else(|e| e.into_inner());
            // SAFETY: stream already synchronized after kernel, copy decompressed bytes to host.
            self.api.check("cuMemcpyDtoH(decomp_out)", unsafe {
                (self.api.cuMemcpyDtoH)(out.as_mut_ptr().cast(), dst_ptr, dst_total_len)
            })?;
            drop(st);
        }
        Ok((out, written))
    }

    /// Compacts changed 32-bit dwords across `pieces` comparing `now_vram` against `held_vram`.
    ///
    /// Returns `Ok(Some(count))` with dense `(piece_id, dword_offset, value)` entries written
    /// into `out_pairs[..count]`, or `Ok(None)` if changed dwords exceeded `out_pairs.len()`
    /// (signaling caller to fall back to a full contiguous copy).
    pub fn compact_readback_deltas(
        &self,
        now_vram: ImportedBufferId,
        held_vram: ImportedBufferId,
        pieces: &[ReadbackPieceDesc],
        out_pairs: &mut [ReadbackDeltaPair],
        wait: Option<(ImportedFenceId, u64)>,
        update_held: bool,
    ) -> CudaResult<Option<usize>> {
        let (now_ptr, _) = self.resolve_buffer(now_vram)?;
        let (held_ptr, _) = self.resolve_buffer(held_vram)?;
        self.compact_deltas_dev_ptrs(now_ptr, held_ptr, pieces, out_pairs, wait, update_held)
    }

    fn compact_deltas_dev_ptrs(
        &self,
        now_ptr: CUdeviceptr,
        held_ptr: CUdeviceptr,
        pieces: &[ReadbackPieceDesc],
        out_pairs: &mut [ReadbackDeltaPair],
        wait: Option<(ImportedFenceId, u64)>,
        update_held: bool,
    ) -> CudaResult<Option<usize>> {
        ensure_current(self.api, self.ctx)?;
        if pieces.is_empty() {
            return Ok(Some(0));
        }

        let mut st = self.stream_state.lock().unwrap_or_else(|e| e.into_inner());
        let stream = st.stream;

        let max_pairs = out_pairs.len() as u32;
        let out_bytes = std::mem::size_of_val(out_pairs).max(16);
        let dev_out_pairs = st.ensure_scratch_out(self.api, out_bytes)?;

        let slot_idx = st.next_slot % RING_SLOTS;
        st.next_slot = st.next_slot.wrapping_add(1);
        let slot = &mut st.slots[slot_idx];
        slot.wait_ready(self.api)?;

        let pieces_bytes = std::mem::size_of_val(pieces);
        slot.ensure_meta(self.api, pieces_bytes)?;
        slot.ensure_out_meta(self.api, 16)?; // [global_counter: u32, overflow_flag: u32]

        // SAFETY: `slot.pinned_meta` is idle and large enough for `pieces`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                pieces.as_ptr().cast::<u8>(),
                slot.pinned_meta,
                pieces_bytes,
            );
        }

        self.wait_fence_on_stream(stream, wait)?;

        // SAFETY: upload piece descriptors and zero the 2 atomic counters on stream.
        self.api.check("cuMemcpyHtoDAsync(pieces)", unsafe {
            (self.api.cuMemcpyHtoDAsync)(slot.dev_meta, slot.pinned_meta.cast(), pieces_bytes, stream)
        })?;
        self.api.check("cuMemsetD32Async(counters)", unsafe {
            (self.api.cuMemsetD32Async)(slot.dev_out_meta, 0, 2, stream)
        })?;

        let counter_ptr = slot.dev_out_meta;
        let overflow_ptr = slot.dev_out_meta + 4;

        let mut arg_now = now_ptr;
        let mut arg_held = held_ptr;
        let mut arg_pieces = slot.dev_meta;
        let mut arg_num_pieces = pieces.len() as u32;
        let mut arg_out_pairs = dev_out_pairs;
        let mut arg_max_pairs = max_pairs;
        let mut arg_counter = counter_ptr;
        let mut arg_overflow = overflow_ptr;
        let mut arg_update_held = u32::from(update_held);
        let mut params: [*mut c_void; 9] = [
            (&mut arg_now as *mut CUdeviceptr).cast(),
            (&mut arg_held as *mut CUdeviceptr).cast(),
            (&mut arg_pieces as *mut CUdeviceptr).cast(),
            (&mut arg_num_pieces as *mut u32).cast(),
            (&mut arg_out_pairs as *mut CUdeviceptr).cast(),
            (&mut arg_max_pairs as *mut u32).cast(),
            (&mut arg_counter as *mut CUdeviceptr).cast(),
            (&mut arg_overflow as *mut CUdeviceptr).cast(),
            (&mut arg_update_held as *mut u32).cast(),
        ];

        // SAFETY: signature matches `compact_readback_deltas` in `victory_kernels.cu`.
        self.api.check("cuLaunchKernel(compact_readback_deltas)", unsafe {
            (self.api.cuLaunchKernel)(
                self.fn_compact_deltas,
                pieces.len() as u32,
                1,
                1,
                256,
                1,
                1,
                0,
                stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })?;

        let mut header = [0u32; 2];
        // SAFETY: reads back `[count, overflow]` (8 bytes) over PCIe.
        self.api.check("cuMemcpyDtoHAsync(delta_hdr)", unsafe {
            (self.api.cuMemcpyDtoHAsync)(header.as_mut_ptr().cast(), counter_ptr, 8, stream)
        })?;
        self.api.check("cuStreamSynchronize(delta_hdr)", unsafe {
            (self.api.cuStreamSynchronize)(stream)
        })?;

        let count = header[0] as usize;
        let overflow = header[1] != 0;
        if overflow || count > out_pairs.len() {
            return Ok(None);
        }
        if count > 0 {
            let copy_bytes = count * std::mem::size_of::<ReadbackDeltaPair>();
            // SAFETY: only copies the exact dense delta payload `count * 12` bytes over PCIe.
            self.api.check("cuMemcpyDtoH(delta_pairs)", unsafe {
                (self.api.cuMemcpyDtoH)(out_pairs.as_mut_ptr().cast(), dev_out_pairs, copy_bytes)
            })?;
        }
        Ok(Some(count))
    }

    /// Synchronous host-side delta compaction helper for unit tests and Criterion benchmarks.
    pub fn compact_deltas_host_sync(
        &self,
        now_bytes: &[u8],
        held_bytes: &[u8],
        pieces: &[ReadbackPieceDesc],
        out_pairs: &mut [ReadbackDeltaPair],
    ) -> CudaResult<Option<usize>> {
        ensure_current(self.api, self.ctx)?;
        let mut dev_now: CUdeviceptr = 0;
        let mut dev_held: CUdeviceptr = 0;
        let now_len = now_bytes.len().max(16);
        let held_len = held_bytes.len().max(16);

        // SAFETY: temporary device buffers for host-sync test/benchmark invocation.
        self.api.check("cuMemAlloc(now)", unsafe {
            (self.api.cuMemAlloc)(&mut dev_now, now_len)
        })?;
        if let Err(err) = self.api.check("cuMemAlloc(held)", unsafe {
            (self.api.cuMemAlloc)(&mut dev_held, held_len)
        }) {
            unsafe { let _ = (self.api.cuMemFree)(dev_now); }
            return Err(err);
        }

        let res = (|| -> CudaResult<Option<usize>> {
            if !now_bytes.is_empty() {
                self.api.check("cuMemcpyHtoD(now)", unsafe {
                    (self.api.cuMemcpyHtoD)(dev_now, now_bytes.as_ptr().cast(), now_bytes.len())
                })?;
            }
            if !held_bytes.is_empty() {
                self.api.check("cuMemcpyHtoD(held)", unsafe {
                    (self.api.cuMemcpyHtoD)(dev_held, held_bytes.as_ptr().cast(), held_bytes.len())
                })?;
            }
            self.compact_deltas_dev_ptrs(dev_now, dev_held, pieces, out_pairs, None, false)
        })();

        // SAFETY: always release temporary device allocations.
        unsafe {
            let _ = (self.api.cuMemFree)(dev_held);
            let _ = (self.api.cuMemFree)(dev_now);
        }
        res
    }

    /// Benchmark/test helper: runs GDeflate v1 or LZ4 decompression directly into VRAM without
    /// DtoH readback (measures production zero-copy DX12-shared VRAM destination latency).
    pub fn decompress_to_vram_sync(
        &self,
        codec: CudaDecompressCodec,
        host_compressed_bytes: &[u8],
        dst_total_len: usize,
        blocks: &[DecompressBlockDesc],
    ) -> CudaResult<()> {
        ensure_current(self.api, self.ctx)?;
        let dst_ptr = {
            let mut st = self.stream_state.lock().unwrap_or_else(|e| e.into_inner());
            st.ensure_scratch_out(self.api, dst_total_len)?
        };
        let _ = self.enqueue_decompress_into_dev_ptr(
            codec,
            host_compressed_bytes,
            dst_ptr,
            blocks,
            None,
            None,
            false,
        )?;
        self.synchronize()
    }

    /// Benchmark/test helper: runs detiling into internal device scratch VRAM and synchronizes
    /// without DtoH readback (measures production zero-copy HtoD + SM 8.9 kernel latency).
    pub fn detile_to_vram_sync(
        &self,
        host_tiled_bytes: &[u8],
        dst_len: usize,
        dispatches: &[DetileDispatchDesc],
    ) -> CudaResult<()> {
        ensure_current(self.api, self.ctx)?;
        let dst_ptr = {
            let mut st = self.stream_state.lock().unwrap_or_else(|e| e.into_inner());
            st.ensure_scratch_out(self.api, dst_len)?
        };
        self.enqueue_detile_into_dev_ptr(host_tiled_bytes, dst_ptr, dispatches, None, None)?;
        self.synchronize()
    }

    /// Benchmark/test helper: allocates a device buffer initialized from `bytes` (freed via
    /// [`Self::free_device_buffer`]) for VRAM-resident delta-compaction benchmarking.
    pub fn alloc_and_upload_vram(&self, bytes: &[u8]) -> CudaResult<u64> {
        ensure_current(self.api, self.ctx)?;
        let mut dptr: CUdeviceptr = 0;
        let len = bytes.len().max(16);
        // SAFETY: allocates `len` bytes in VRAM and uploads `bytes`.
        self.api.check("cuMemAlloc(bench)", unsafe { (self.api.cuMemAlloc)(&mut dptr, len) })?;
        if !bytes.is_empty() {
            if let Err(err) = self.api.check("cuMemcpyHtoD(bench)", unsafe {
                (self.api.cuMemcpyHtoD)(dptr, bytes.as_ptr().cast(), bytes.len())
            }) {
                unsafe { let _ = (self.api.cuMemFree)(dptr); }
                return Err(err);
            }
        }
        Ok(dptr)
    }

    /// Frees a device buffer created by [`Self::alloc_and_upload_vram`].
    ///
    /// # Safety
    /// `dev_ptr` must have been returned by [`Self::alloc_and_upload_vram`] on this context
    /// and must no longer be referenced by in-flight GPU work.
    pub unsafe fn free_device_buffer(&self, dev_ptr: u64) {
        let _ = ensure_current(self.api, self.ctx);
        if dev_ptr != 0 {
            // SAFETY: caller's contract.
            unsafe { let _ = (self.api.cuMemFree)(dev_ptr); }
        }
    }

    /// Runs `compact_readback_deltas` directly on two resident VRAM device pointers (zero HtoD).
    ///
    /// # Safety
    /// `now_dev_ptr` and `held_dev_ptr` must be valid device buffers covering all `pieces`.
    pub unsafe fn compact_deltas_vram_sync(
        &self,
        now_dev_ptr: u64,
        held_dev_ptr: u64,
        pieces: &[ReadbackPieceDesc],
        out_pairs: &mut [ReadbackDeltaPair],
    ) -> CudaResult<Option<usize>> {
        self.compact_deltas_dev_ptrs(now_dev_ptr, held_dev_ptr, pieces, out_pairs, None, false)
    }

    /// Waits for all pending work on the CUDA stream (worker threads & tests only).
    pub fn synchronize(&self) -> CudaResult<()> {
        ensure_current(self.api, self.ctx)?;
        let mut st = self.stream_state.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: synchronizes our dedicated non-blocking stream.
        self.api.check("cuStreamSynchronize", unsafe {
            (self.api.cuStreamSynchronize)(st.stream)
        })?;
        for slot in &mut st.slots {
            slot.in_flight = false;
        }
        Ok(())
    }
}

impl Drop for CudaAccelContext {
    fn drop(&mut self) {
        let _ = ensure_current(self.api, self.ctx);
        // Clear imported external handles before unloading module / primary context.
        self.ext_buffers.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.ext_fences.lock().unwrap_or_else(|e| e.into_inner()).clear();

        if let Ok(mut st) = self.stream_state.lock() {
            // SAFETY: synchronize and free stream + ring slots + scratch memory.
            unsafe {
                let _ = (self.api.cuStreamSynchronize)(st.stream);
                for slot in &mut st.slots {
                    slot.destroy(self.api);
                }
                if st.dev_scratch_out != 0 {
                    let _ = (self.api.cuMemFree)(st.dev_scratch_out);
                }
                if !st.stream.is_null() {
                    let _ = (self.api.cuStreamDestroy)(st.stream);
                }
            }
        }
        // SAFETY: release device LUTs, module, and primary context reference.
        unsafe {
            if self.dev_lut_data != 0 {
                let _ = (self.api.cuMemFree)(self.dev_lut_data);
            }
            if self.dev_lut_meta != 0 {
                let _ = (self.api.cuMemFree)(self.dev_lut_meta);
            }
            if !self.module.is_null() {
                let _ = (self.api.cuModuleUnload)(self.module);
            }
            let _ = (self.api.cuDevicePrimaryCtxRelease)(self.device);
        }
    }
}
