//! Dynamic loader and exact C-ABI bindings for the NVIDIA CUDA Driver API (`nvcuda.dll`).
//!
//! Zero link-time dependency on `cuda.lib` or `bindgen`/`libclang`: loads `nvcuda.dll`
//! at runtime via `LoadLibraryA` / `GetProcAddress` on Windows. All ABI structs match
//! CUDA Toolkit 12.8+ `cuda.h` byte-for-byte with compile-time layout assertions.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

use thiserror::Error;

pub type CUresult = c_uint;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUexternalMemory = *mut c_void;
pub type CUexternalSemaphore = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;

/// Non-blocking CUDA stream (does not implicitly synchronize with the legacy default stream).
pub const CU_STREAM_NON_BLOCKING: c_uint = 0x1;
/// Disable timing on CUDA events for minimal recording overhead.
pub const CU_EVENT_DISABLE_TIMING: c_uint = 0x2;
/// Portable pinned host memory usable across all contexts.
pub const CU_MEMHOSTALLOC_PORTABLE: c_uint = 0x1;

/// `CUexternalMemoryHandleType` values from `cuda.h`.
pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP: c_uint = 4;
pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE: c_uint = 5;
/// Flag required for D3D12 committed resources (`CreateCommittedResource`).
pub const CUDA_EXTERNAL_MEMORY_DEDICATED: c_uint = 0x1;

/// `CUexternalSemaphoreHandleType` values from `cuda.h`.
pub const CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE: c_uint = 4;

/// `CUdevice_attribute` constants from `cuda.h`.
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_uint = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_uint = 76;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_uint = 16;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Win32ExternalHandle {
    pub handle: *mut c_void,
    pub name: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union ExternalHandleUnion {
    pub fd: c_int,
    pub win32: Win32ExternalHandle,
    pub nv_sci_obj: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
    pub handle_type: c_uint,
    pub handle: ExternalHandleUnion,
    pub size: u64,
    pub flags: c_uint,
    pub reserved: [c_uint; 16],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
    pub offset: u64,
    pub size: u64,
    pub flags: c_uint,
    pub reserved: [c_uint; 16],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC {
    pub handle_type: c_uint,
    pub handle: ExternalHandleUnion,
    pub flags: c_uint,
    pub reserved: [c_uint; 16],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExternalSemaphoreFenceParams {
    pub value: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExternalSemaphoreKeyedMutexSignalParams {
    pub key: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExternalSemaphoreKeyedMutexWaitParams {
    pub key: u64,
    pub timeout_ms: c_uint,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExternalSemaphoreSignalParamsInner {
    pub fence: ExternalSemaphoreFenceParams,
    pub nv_sci_sync: u64,
    pub keyed_mutex: ExternalSemaphoreKeyedMutexSignalParams,
    pub reserved: [c_uint; 12],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS {
    pub params: ExternalSemaphoreSignalParamsInner,
    pub flags: c_uint,
    pub reserved: [c_uint; 16],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExternalSemaphoreWaitParamsInner {
    pub fence: ExternalSemaphoreFenceParams,
    pub nv_sci_sync: u64,
    pub keyed_mutex: ExternalSemaphoreKeyedMutexWaitParams,
    pub reserved: [c_uint; 10],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS {
    pub params: ExternalSemaphoreWaitParamsInner,
    pub flags: c_uint,
    pub reserved: [c_uint; 16],
}

// Compile-time verification against CUDA 12.8 x86_64 MSVC `cuda.h` struct sizes.
const _: () = {
    assert!(std::mem::size_of::<CUDA_EXTERNAL_MEMORY_HANDLE_DESC>() == 104);
    assert!(std::mem::size_of::<CUDA_EXTERNAL_MEMORY_BUFFER_DESC>() == 88);
    assert!(std::mem::size_of::<CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC>() == 96);
    assert!(std::mem::size_of::<CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS>() == 144);
    assert!(std::mem::size_of::<CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS>() == 144);
};

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("nvcuda.dll is not available on this host")]
    DriverUnavailable,
    #[error("missing CUDA driver export `{0}`")]
    MissingSymbol(&'static str),
    #[error("embedded SM 8.9 PTX module was not built (install CUDA Toolkit or set VICTORY_CUDA_OXIDE_PTX)")]
    PtxUnavailable,
    #[error("no CUDA device matches adapter LUID {0:02x?}")]
    AdapterMismatch([u8; 8]),
    #[error("CUDA driver call `{op}` failed with code {code} ({name})")]
    Driver {
        op: &'static str,
        code: CUresult,
        name: String,
    },
    #[error("invalid CUDA accelerator argument: {0}")]
    InvalidArgument(String),
    #[error("CUDA decompression kernel failed on tile/block {index} (status={status})")]
    DecompressKernelFailed { index: usize, status: u32 },
}

pub type CudaResult<T> = Result<T, CudaError>;

type FnCuInit = unsafe extern "system" fn(c_uint) -> CUresult;
type FnCuGetErrorName = unsafe extern "system" fn(CUresult, *mut *const c_char) -> CUresult;
type FnCuDeviceGetCount = unsafe extern "system" fn(*mut c_int) -> CUresult;
type FnCuDeviceGet = unsafe extern "system" fn(*mut CUdevice, c_int) -> CUresult;
type FnCuDeviceGetName = unsafe extern "system" fn(*mut c_char, c_int, CUdevice) -> CUresult;
type FnCuDeviceGetLuid = unsafe extern "system" fn(*mut c_char, *mut c_uint, CUdevice) -> CUresult;
type FnCuDeviceGetAttribute = unsafe extern "system" fn(*mut c_int, c_uint, CUdevice) -> CUresult;
type FnCuDevicePrimaryCtxRetain = unsafe extern "system" fn(*mut CUcontext, CUdevice) -> CUresult;
type FnCuDevicePrimaryCtxRelease = unsafe extern "system" fn(CUdevice) -> CUresult;
type FnCuCtxSetCurrent = unsafe extern "system" fn(CUcontext) -> CUresult;
type FnCuModuleLoadData = unsafe extern "system" fn(*mut CUmodule, *const c_void) -> CUresult;
type FnCuModuleUnload = unsafe extern "system" fn(CUmodule) -> CUresult;
type FnCuModuleGetFunction =
    unsafe extern "system" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
type FnCuMemAlloc = unsafe extern "system" fn(*mut CUdeviceptr, usize) -> CUresult;
type FnCuMemFree = unsafe extern "system" fn(CUdeviceptr) -> CUresult;
type FnCuMemHostAlloc = unsafe extern "system" fn(*mut *mut c_void, usize, c_uint) -> CUresult;
type FnCuMemFreeHost = unsafe extern "system" fn(*mut c_void) -> CUresult;
type FnCuMemcpyHtoD = unsafe extern "system" fn(CUdeviceptr, *const c_void, usize) -> CUresult;
type FnCuMemcpyDtoH = unsafe extern "system" fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
type FnCuMemcpyHtoDAsync =
    unsafe extern "system" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult;
type FnCuMemcpyDtoHAsync =
    unsafe extern "system" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult;
type FnCuMemsetD8Async =
    unsafe extern "system" fn(CUdeviceptr, u8, usize, CUstream) -> CUresult;
type FnCuMemsetD32Async =
    unsafe extern "system" fn(CUdeviceptr, c_uint, usize, CUstream) -> CUresult;
type FnCuStreamCreate = unsafe extern "system" fn(*mut CUstream, c_uint) -> CUresult;
type FnCuStreamDestroy = unsafe extern "system" fn(CUstream) -> CUresult;
type FnCuStreamSynchronize = unsafe extern "system" fn(CUstream) -> CUresult;
type FnCuEventCreate = unsafe extern "system" fn(*mut CUevent, c_uint) -> CUresult;
type FnCuEventDestroy = unsafe extern "system" fn(CUevent) -> CUresult;
type FnCuEventRecord = unsafe extern "system" fn(CUevent, CUstream) -> CUresult;
type FnCuEventSynchronize = unsafe extern "system" fn(CUevent) -> CUresult;
type FnCuLaunchKernel = unsafe extern "system" fn(
    CUfunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    CUstream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CUresult;
type FnCuImportExternalMemory = unsafe extern "system" fn(
    *mut CUexternalMemory,
    *const CUDA_EXTERNAL_MEMORY_HANDLE_DESC,
) -> CUresult;
type FnCuExternalMemoryGetMappedBuffer = unsafe extern "system" fn(
    *mut CUdeviceptr,
    CUexternalMemory,
    *const CUDA_EXTERNAL_MEMORY_BUFFER_DESC,
) -> CUresult;
type FnCuDestroyExternalMemory = unsafe extern "system" fn(CUexternalMemory) -> CUresult;
type FnCuImportExternalSemaphore = unsafe extern "system" fn(
    *mut CUexternalSemaphore,
    *const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC,
) -> CUresult;
type FnCuSignalExternalSemaphoresAsync = unsafe extern "system" fn(
    *const CUexternalSemaphore,
    *const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS,
    c_uint,
    CUstream,
) -> CUresult;
type FnCuWaitExternalSemaphoresAsync = unsafe extern "system" fn(
    *const CUexternalSemaphore,
    *const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS,
    c_uint,
    CUstream,
) -> CUresult;
type FnCuDestroyExternalSemaphore = unsafe extern "system" fn(CUexternalSemaphore) -> CUresult;

/// Loaded function table for the NVIDIA CUDA Driver API (`nvcuda.dll`).
pub struct CudaApi {
    pub cuInit: FnCuInit,
    pub cuGetErrorName: FnCuGetErrorName,
    pub cuDeviceGetCount: FnCuDeviceGetCount,
    pub cuDeviceGet: FnCuDeviceGet,
    pub cuDeviceGetName: FnCuDeviceGetName,
    pub cuDeviceGetLuid: FnCuDeviceGetLuid,
    pub cuDeviceGetAttribute: FnCuDeviceGetAttribute,
    pub cuDevicePrimaryCtxRetain: FnCuDevicePrimaryCtxRetain,
    pub cuDevicePrimaryCtxRelease: FnCuDevicePrimaryCtxRelease,
    pub cuCtxSetCurrent: FnCuCtxSetCurrent,
    pub cuModuleLoadData: FnCuModuleLoadData,
    pub cuModuleUnload: FnCuModuleUnload,
    pub cuModuleGetFunction: FnCuModuleGetFunction,
    pub cuMemAlloc: FnCuMemAlloc,
    pub cuMemFree: FnCuMemFree,
    pub cuMemHostAlloc: FnCuMemHostAlloc,
    pub cuMemFreeHost: FnCuMemFreeHost,
    pub cuMemcpyHtoD: FnCuMemcpyHtoD,
    pub cuMemcpyDtoH: FnCuMemcpyDtoH,
    pub cuMemcpyHtoDAsync: FnCuMemcpyHtoDAsync,
    pub cuMemcpyDtoHAsync: FnCuMemcpyDtoHAsync,
    pub cuMemsetD8Async: FnCuMemsetD8Async,
    pub cuMemsetD32Async: FnCuMemsetD32Async,
    pub cuStreamCreate: FnCuStreamCreate,
    pub cuStreamDestroy: FnCuStreamDestroy,
    pub cuStreamSynchronize: FnCuStreamSynchronize,
    pub cuEventCreate: FnCuEventCreate,
    pub cuEventDestroy: FnCuEventDestroy,
    pub cuEventRecord: FnCuEventRecord,
    pub cuEventSynchronize: FnCuEventSynchronize,
    pub cuLaunchKernel: FnCuLaunchKernel,
    pub cuImportExternalMemory: FnCuImportExternalMemory,
    pub cuExternalMemoryGetMappedBuffer: FnCuExternalMemoryGetMappedBuffer,
    pub cuDestroyExternalMemory: FnCuDestroyExternalMemory,
    pub cuImportExternalSemaphore: FnCuImportExternalSemaphore,
    pub cuSignalExternalSemaphoresAsync: FnCuSignalExternalSemaphoresAsync,
    pub cuWaitExternalSemaphoresAsync: FnCuWaitExternalSemaphoresAsync,
    pub cuDestroyExternalSemaphore: FnCuDestroyExternalSemaphore,
}

// SAFETY: CUDA driver function pointers loaded from the process-wide `nvcuda.dll` module
// remain valid for the lifetime of the process and are thread-safe to invoke.
unsafe impl Send for CudaApi {}
unsafe impl Sync for CudaApi {}

impl CudaApi {
    /// Returns the process-wide loaded CUDA Driver API table, initializing the driver once.
    pub fn load() -> CudaResult<&'static CudaApi> {
        static API: OnceLock<Result<CudaApi, String>> = OnceLock::new();
        match API.get_or_init(|| Self::load_inner().map_err(|e| e.to_string())) {
            Ok(api) => Ok(api),
            Err(msg) => {
                if msg.contains("nvcuda.dll") {
                    Err(CudaError::DriverUnavailable)
                } else {
                    Err(CudaError::InvalidArgument(msg.clone()))
                }
            }
        }
    }

    pub fn check(&self, op: &'static str, code: CUresult) -> CudaResult<()> {
        if code == CUDA_SUCCESS {
            return Ok(());
        }
        let mut ptr: *const c_char = std::ptr::null();
        // SAFETY: `cuGetErrorName` writes a static string pointer into `ptr`.
        let name = unsafe {
            if (self.cuGetErrorName)(code, &mut ptr) == CUDA_SUCCESS && !ptr.is_null() {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            } else {
                format!("UNKNOWN_{code}")
            }
        };
        Err(CudaError::Driver { op, code, name })
    }

    #[cfg(windows)]
    fn load_inner() -> CudaResult<CudaApi> {
        use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};

        // SAFETY: null-terminated ASCII DLL name.
        let module = unsafe { LoadLibraryA(c"nvcuda.dll".as_ptr().cast()) };
        if module.is_null() {
            return Err(CudaError::DriverUnavailable);
        }

        macro_rules! sym {
            ($name:literal, $ty:ty) => {{
                let cname = concat!($name, "\0");
                // SAFETY: `module` is a live HMODULE from `LoadLibraryA` and `cname` is NUL-terminated.
                let addr = unsafe { GetProcAddress(module, cname.as_ptr()) };
                match addr {
                    Some(f) => unsafe { std::mem::transmute::<_, $ty>(f) },
                    None => return Err(CudaError::MissingSymbol($name)),
                }
            }};
        }

        let api = CudaApi {
            cuInit: sym!("cuInit", FnCuInit),
            cuGetErrorName: sym!("cuGetErrorName", FnCuGetErrorName),
            cuDeviceGetCount: sym!("cuDeviceGetCount", FnCuDeviceGetCount),
            cuDeviceGet: sym!("cuDeviceGet", FnCuDeviceGet),
            cuDeviceGetName: sym!("cuDeviceGetName", FnCuDeviceGetName),
            cuDeviceGetLuid: sym!("cuDeviceGetLuid", FnCuDeviceGetLuid),
            cuDeviceGetAttribute: sym!("cuDeviceGetAttribute", FnCuDeviceGetAttribute),
            cuDevicePrimaryCtxRetain: sym!("cuDevicePrimaryCtxRetain", FnCuDevicePrimaryCtxRetain),
            cuDevicePrimaryCtxRelease: sym!("cuDevicePrimaryCtxRelease_v2", FnCuDevicePrimaryCtxRelease),
            cuCtxSetCurrent: sym!("cuCtxSetCurrent", FnCuCtxSetCurrent),
            cuModuleLoadData: sym!("cuModuleLoadData", FnCuModuleLoadData),
            cuModuleUnload: sym!("cuModuleUnload", FnCuModuleUnload),
            cuModuleGetFunction: sym!("cuModuleGetFunction", FnCuModuleGetFunction),
            cuMemAlloc: sym!("cuMemAlloc_v2", FnCuMemAlloc),
            cuMemFree: sym!("cuMemFree_v2", FnCuMemFree),
            cuMemHostAlloc: sym!("cuMemHostAlloc", FnCuMemHostAlloc),
            cuMemFreeHost: sym!("cuMemFreeHost", FnCuMemFreeHost),
            cuMemcpyHtoD: sym!("cuMemcpyHtoD_v2", FnCuMemcpyHtoD),
            cuMemcpyDtoH: sym!("cuMemcpyDtoH_v2", FnCuMemcpyDtoH),
            cuMemcpyHtoDAsync: sym!("cuMemcpyHtoDAsync_v2", FnCuMemcpyHtoDAsync),
            cuMemcpyDtoHAsync: sym!("cuMemcpyDtoHAsync_v2", FnCuMemcpyDtoHAsync),
            cuMemsetD8Async: sym!("cuMemsetD8Async", FnCuMemsetD8Async),
            cuMemsetD32Async: sym!("cuMemsetD32Async", FnCuMemsetD32Async),
            cuStreamCreate: sym!("cuStreamCreate", FnCuStreamCreate),
            cuStreamDestroy: sym!("cuStreamDestroy_v2", FnCuStreamDestroy),
            cuStreamSynchronize: sym!("cuStreamSynchronize", FnCuStreamSynchronize),
            cuEventCreate: sym!("cuEventCreate", FnCuEventCreate),
            cuEventDestroy: sym!("cuEventDestroy_v2", FnCuEventDestroy),
            cuEventRecord: sym!("cuEventRecord", FnCuEventRecord),
            cuEventSynchronize: sym!("cuEventSynchronize", FnCuEventSynchronize),
            cuLaunchKernel: sym!("cuLaunchKernel", FnCuLaunchKernel),
            cuImportExternalMemory: sym!("cuImportExternalMemory", FnCuImportExternalMemory),
            cuExternalMemoryGetMappedBuffer: sym!(
                "cuExternalMemoryGetMappedBuffer",
                FnCuExternalMemoryGetMappedBuffer
            ),
            cuDestroyExternalMemory: sym!("cuDestroyExternalMemory", FnCuDestroyExternalMemory),
            cuImportExternalSemaphore: sym!("cuImportExternalSemaphore", FnCuImportExternalSemaphore),
            cuSignalExternalSemaphoresAsync: sym!(
                "cuSignalExternalSemaphoresAsync",
                FnCuSignalExternalSemaphoresAsync
            ),
            cuWaitExternalSemaphoresAsync: sym!(
                "cuWaitExternalSemaphoresAsync",
                FnCuWaitExternalSemaphoresAsync
            ),
            cuDestroyExternalSemaphore: sym!(
                "cuDestroyExternalSemaphore",
                FnCuDestroyExternalSemaphore
            ),
        };

        // SAFETY: `cuInit(0)` is safe to invoke once during process initialization.
        let rc = unsafe { (api.cuInit)(0) };
        api.check("cuInit", rc)?;
        Ok(api)
    }

    #[cfg(not(windows))]
    fn load_inner() -> CudaResult<CudaApi> {
        Err(CudaError::DriverUnavailable)
    }
}
