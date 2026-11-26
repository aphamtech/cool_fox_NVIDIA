//! Zero-copy DX12 ↔ CUDA Win32 NT shared-handle external memory and timeline semaphore interop.
//!
//! Per CUDA Driver API specification (`cuda.h`), `cuImportExternalMemory` and
//! `cuImportExternalSemaphore` duplicate/reference the underlying NT kernel object
//! and do **not** take ownership of the Win32 `HANDLE` — the caller retains responsibility
//! for closing the NT handle after import returns.

use std::ffi::c_void;

use crate::driver::{
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE, CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE,
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC, CUDA_EXTERNAL_MEMORY_DEDICATED,
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC, CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC,
    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS, CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS, CUcontext,
    CUdevice, CUdeviceptr, CUexternalMemory, CUexternalSemaphore, CUstream, CudaApi, CudaError,
    CudaResult, ExternalHandleUnion, ExternalSemaphoreFenceParams,
    ExternalSemaphoreKeyedMutexSignalParams, ExternalSemaphoreKeyedMutexWaitParams,
    ExternalSemaphoreSignalParamsInner, ExternalSemaphoreWaitParamsInner, Win32ExternalHandle,
};

/// Metadata describing a selected CUDA device.
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub ordinal: i32,
    pub device: CUdevice,
    pub name: String,
    pub luid: [u8; 8],
    pub node_mask: u32,
    pub sm_major: i32,
    pub sm_minor: i32,
    pub sm_count: i32,
}

/// Finds the CUDA device matching an 8-byte Windows DXGI adapter LUID (or device 0 if `None`).
pub fn find_device(api: &'static CudaApi, target_luid: Option<[u8; 8]>) -> CudaResult<CudaDeviceInfo> {
    let mut count = 0i32;
    // SAFETY: `count` is a valid out pointer.
    api.check("cuDeviceGetCount", unsafe { (api.cuDeviceGetCount)(&mut count) })?;
    if count <= 0 {
        return Err(CudaError::DriverUnavailable);
    }

    for ordinal in 0..count {
        let mut dev: CUdevice = 0;
        // SAFETY: valid out pointer and ordinal < count.
        api.check("cuDeviceGet", unsafe { (api.cuDeviceGet)(&mut dev, ordinal) })?;

        let mut luid = [0u8; 8];
        let mut node_mask = 0u32;
        // SAFETY: `luid` holds 8 bytes and `node_mask` is a valid out pointer.
        let rc = unsafe { (api.cuDeviceGetLuid)(luid.as_mut_ptr().cast(), &mut node_mask, dev) };
        if rc != 0 {
            continue;
        }

        if let Some(want) = target_luid {
            if luid != want {
                continue;
            }
        }

        let mut name_buf = [0u8; 128];
        // SAFETY: `name_buf` is 128 bytes.
        api.check("cuDeviceGetName", unsafe {
            (api.cuDeviceGetName)(name_buf.as_mut_ptr().cast(), name_buf.len() as i32, dev)
        })?;
        let nul = name_buf.iter().position(|&b| b == 0).unwrap_or(name_buf.len());
        let name = String::from_utf8_lossy(&name_buf[..nul]).into_owned();

        let mut sm_major = 0;
        let mut sm_minor = 0;
        let mut sm_count = 0;
        // SAFETY: valid out pointers for device attributes.
        unsafe {
            let _ = (api.cuDeviceGetAttribute)(
                &mut sm_major,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                dev,
            );
            let _ = (api.cuDeviceGetAttribute)(
                &mut sm_minor,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                dev,
            );
            let _ = (api.cuDeviceGetAttribute)(
                &mut sm_count,
                CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                dev,
            );
        }

        return Ok(CudaDeviceInfo {
            ordinal,
            device: dev,
            name,
            luid,
            node_mask,
            sm_major,
            sm_minor,
            sm_count,
        });
    }

    match target_luid {
        Some(luid) => Err(CudaError::AdapterMismatch(luid)),
        None => Err(CudaError::DriverUnavailable),
    }
}

/// RAII wrapper for a DX12 VRAM buffer imported into CUDA via Win32 NT handle (`cuImportExternalMemory`).
pub struct ExternalMemory {
    api: &'static CudaApi,
    ctx: CUcontext,
    ext_mem: CUexternalMemory,
    dev_ptr: CUdeviceptr,
    byte_len: u64,
}

// SAFETY: `CUexternalMemory` and its mapped `CUdeviceptr` belong to the primary context
// and are accessed under stream synchronization.
unsafe impl Send for ExternalMemory {}
unsafe impl Sync for ExternalMemory {}

impl ExternalMemory {
    /// Imports a DX12 committed resource (`dedicated == true`) or placed heap (`dedicated == false`)
    /// from its Win32 NT shared handle and maps the entire range to a `CUdeviceptr`.
    ///
    /// # Safety
    /// `nt_handle` must be a valid Win32 NT handle exported via `ID3D12Device::CreateSharedHandle`
    /// for a D3D12 resource/heap of at least `byte_len` bytes on the same physical GPU adapter.
    pub unsafe fn import_d3d12_nt_handle(
        api: &'static CudaApi,
        ctx: CUcontext,
        nt_handle: *mut c_void,
        byte_len: u64,
        dedicated: bool,
    ) -> CudaResult<Self> {
        if nt_handle.is_null() || byte_len == 0 {
            return Err(CudaError::InvalidArgument(
                "external memory import requires non-null NT handle and non-zero size".into(),
            ));
        }

        // SAFETY: binds the primary context on the calling thread.
        api.check("cuCtxSetCurrent", unsafe { (api.cuCtxSetCurrent)(ctx) })?;

        let desc = CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
            handle_type: if dedicated {
                CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE
            } else {
                CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP
            },
            handle: ExternalHandleUnion {
                win32: Win32ExternalHandle {
                    handle: nt_handle,
                    name: std::ptr::null(),
                },
            },
            size: byte_len,
            flags: if dedicated { CUDA_EXTERNAL_MEMORY_DEDICATED } else { 0 },
            reserved: [0; 16],
        };

        let mut ext_mem: CUexternalMemory = std::ptr::null_mut();
        // SAFETY: `desc` matches CUDA Driver API ABI and `nt_handle` is a valid D3D12 NT handle.
        api.check("cuImportExternalMemory", unsafe {
            (api.cuImportExternalMemory)(&mut ext_mem, &desc)
        })?;

        let buf_desc = CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
            offset: 0,
            size: byte_len,
            flags: 0,
            reserved: [0; 16],
        };
        let mut dev_ptr: CUdeviceptr = 0;
        // SAFETY: `ext_mem` was just imported and `buf_desc` spans `[0, byte_len)`.
        let map_rc = unsafe {
            (api.cuExternalMemoryGetMappedBuffer)(&mut dev_ptr, ext_mem, &buf_desc)
        };
        if let Err(err) = api.check("cuExternalMemoryGetMappedBuffer", map_rc) {
            // SAFETY: clean up `ext_mem` on mapping error.
            unsafe { (api.cuDestroyExternalMemory)(ext_mem) };
            return Err(err);
        }

        Ok(Self {
            api,
            ctx,
            ext_mem,
            dev_ptr,
            byte_len,
        })
    }

    #[inline]
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.dev_ptr
    }

    #[inline]
    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

impl Drop for ExternalMemory {
    fn drop(&mut self) {
        // SAFETY: bind context and release device pointer + external memory object.
        unsafe {
            let _ = (self.api.cuCtxSetCurrent)(self.ctx);
            if self.dev_ptr != 0 {
                let _ = (self.api.cuMemFree)(self.dev_ptr);
                self.dev_ptr = 0;
            }
            if !self.ext_mem.is_null() {
                let _ = (self.api.cuDestroyExternalMemory)(self.ext_mem);
                self.ext_mem = std::ptr::null_mut();
            }
        }
    }
}

/// RAII wrapper for a DX12 shared timeline fence (`D3D12_FENCE_FLAG_SHARED`) imported into CUDA.
pub struct ExternalSemaphore {
    api: &'static CudaApi,
    ctx: CUcontext,
    ext_sem: CUexternalSemaphore,
}

// SAFETY: `CUexternalSemaphore` handles are thread-safe to wait/signal on CUDA streams.
unsafe impl Send for ExternalSemaphore {}
unsafe impl Sync for ExternalSemaphore {}

impl ExternalSemaphore {
    /// Imports a DX12 timeline fence from its Win32 NT shared handle.
    ///
    /// # Safety
    /// `nt_handle` must be a valid Win32 NT handle exported via `ID3D12Device::CreateSharedHandle`
    /// for an `ID3D12Fence` created with `D3D12_FENCE_FLAG_SHARED`.
    pub unsafe fn import_d3d12_fence_nt_handle(
        api: &'static CudaApi,
        ctx: CUcontext,
        nt_handle: *mut c_void,
    ) -> CudaResult<Self> {
        if nt_handle.is_null() {
            return Err(CudaError::InvalidArgument(
                "external semaphore import requires non-null NT handle".into(),
            ));
        }
        // SAFETY: binds primary context on calling thread.
        api.check("cuCtxSetCurrent", unsafe { (api.cuCtxSetCurrent)(ctx) })?;

        let desc = CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC {
            handle_type: CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE,
            handle: ExternalHandleUnion {
                win32: Win32ExternalHandle {
                    handle: nt_handle,
                    name: std::ptr::null(),
                },
            },
            flags: 0,
            reserved: [0; 16],
        };

        let mut ext_sem: CUexternalSemaphore = std::ptr::null_mut();
        // SAFETY: `desc` matches CUDA Driver API ABI and `nt_handle` is a valid D3D12 fence NT handle.
        api.check("cuImportExternalSemaphore", unsafe {
            (api.cuImportExternalSemaphore)(&mut ext_sem, &desc)
        })?;

        Ok(Self { api, ctx, ext_sem })
    }

    /// Enqueues a GPU-side timeline wait on `stream` until the D3D12 fence reaches `value`.
    pub fn wait_async(&self, stream: CUstream, value: u64) -> CudaResult<()> {
        let params = CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS {
            params: ExternalSemaphoreWaitParamsInner {
                fence: ExternalSemaphoreFenceParams { value },
                nv_sci_sync: 0,
                keyed_mutex: ExternalSemaphoreKeyedMutexWaitParams {
                    key: 0,
                    timeout_ms: 0,
                },
                reserved: [0; 10],
            },
            flags: 0,
            reserved: [0; 16],
        };
        // SAFETY: `ext_sem` and `stream` are live handles on the active CUDA context.
        self.api.check("cuWaitExternalSemaphoresAsync", unsafe {
            (self.api.cuWaitExternalSemaphoresAsync)(&self.ext_sem, &params, 1, stream)
        })
    }

    /// Enqueues a GPU-side timeline signal on `stream` setting the D3D12 fence to `value`.
    pub fn signal_async(&self, stream: CUstream, value: u64) -> CudaResult<()> {
        let params = CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS {
            params: ExternalSemaphoreSignalParamsInner {
                fence: ExternalSemaphoreFenceParams { value },
                nv_sci_sync: 0,
                keyed_mutex: ExternalSemaphoreKeyedMutexSignalParams { key: 0 },
                reserved: [0; 12],
            },
            flags: 0,
            reserved: [0; 16],
        };
        // SAFETY: `ext_sem` and `stream` are live handles on the active CUDA context.
        self.api.check("cuSignalExternalSemaphoresAsync", unsafe {
            (self.api.cuSignalExternalSemaphoresAsync)(&self.ext_sem, &params, 1, stream)
        })
    }
}

impl Drop for ExternalSemaphore {
    fn drop(&mut self) {
        // SAFETY: bind context and release external semaphore object.
        unsafe {
            let _ = (self.api.cuCtxSetCurrent)(self.ctx);
            if !self.ext_sem.is_null() {
                let _ = (self.api.cuDestroyExternalSemaphore)(self.ext_sem);
                self.ext_sem = std::ptr::null_mut();
            }
        }
    }
}
