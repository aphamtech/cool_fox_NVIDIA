//! Optional NVIDIA DLSS Super Resolution / DLAA integration via the NVIDIA NGX
//! D3D12 C ABI (`_nvngx.dll` + user-supplied `nvngx_dlss.dll`).
//!
//! When running on a hardware NVIDIA RTX GPU and `nvngx_dlss.dll` is found
//! (in `VICTORY_DLSS_DIR`, `VICTORY_GAME_DIR`, the executable directory, or the
//! current working directory), this module initializes NGX on the host D3D12
//! device and evaluates DLSS on the presented scene plane prior to swap-chain
//! composition / display.
//!
//! Configuration environment variables:
//! - `VICTORY_DLSS`: `off`/`0`, `auto` (default: DLAA when `nvngx_dlss.dll` is
//!   present on NVIDIA), `dlaa`, `quality`, `balanced`, `performance`, or
//!   `ultra_performance`.
//! - `VICTORY_DLSS_DIR`: optional explicit directory containing `nvngx_dlss.dll`.
//! - `VICTORY_DLSS_WIDTH` / `VICTORY_DLSS_HEIGHT`: optional explicit output
//!   dimensions (overrides the mode's default scale factor).
//! - `VICTORY_DLSS_RESET`: `1` (default, safe single-frame reconstruction when
//!   game motion vectors / camera jitter are not exported) or `0` (temporal).
//!
//! Note: Because VIctory translates an Xbox Series binary without engine source
//! hooks for per-frame camera jitter or G-buffer motion vectors, DLSS runs with
//! zeroed motion vectors, constant depth, and `Reset=1` by default so it never
//! introduces temporal ghosting while still leveraging NVIDIA Tensor Cores for
//! reconstruction/upscaling before the crisp UI overlay is composited on top.

use std::ffi::{CString, OsStr, c_char, c_void};
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;
use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows_sys::Win32::System::Registry::{
    HKEY, HKEY_LOCAL_MACHINE, KEY_READ, REG_EXPAND_SZ, REG_SZ, RegCloseKey, RegOpenKeyExW,
    RegQueryValueExW,
};

use crate::host::{Host, transition};

const NVSDK_NGX_RESULT_SUCCESS: u32 = 0x1;
const NVSDK_NGX_VERSION_API: u32 = 0x0000_0015;
const NVSDK_NGX_FEATURE_SUPER_SAMPLING: u32 = 1;
const NVSDK_NGX_DLSS_FEATURE_FLAG_IS_HDR: i32 = 1 << 0;
/// Generic application ID for unofficial/custom NGX integrations when no
/// NVIDIA developer portal ProjectID is assigned.
const VICTORY_NGX_APP_ID: u64 = 0x5649_4354_4F52_5901;

#[repr(C)]
struct NgxPathListInfo {
    path: *const *const u16,
    length: u32,
}

#[repr(C)]
struct NgxLoggingInfo {
    logging_callback: *const c_void,
    minimum_logging_level: u32,
    disable_other_logging_sinks: bool,
}

#[repr(C)]
struct NgxFeatureCommonInfo {
    path_list_info: NgxPathListInfo,
    internal_data: *mut c_void,
    logging_info: NgxLoggingInfo,
}

type FnD3D12InitExt = unsafe extern "C" fn(
    app_id: u64,
    app_data_path: *const u16,
    device: *mut c_void,
    feature_info: *const NgxFeatureCommonInfo,
    sdk_version: u32,
) -> u32;
type FnD3D12Shutdown1 = unsafe extern "C" fn(device: *mut c_void) -> u32;
type FnD3D12GetCapabilityParameters = unsafe extern "C" fn(out_params: *mut *mut c_void) -> u32;
type FnD3D12AllocateParameters = unsafe extern "C" fn(out_params: *mut *mut c_void) -> u32;
type FnD3D12DestroyParameters = unsafe extern "C" fn(params: *mut c_void) -> u32;
type FnD3D12CreateFeature = unsafe extern "C" fn(
    cmd_list: *mut c_void,
    feature_id: u32,
    params: *mut c_void,
    out_handle: *mut *mut c_void,
) -> u32;
type FnD3D12EvaluateFeature = unsafe extern "C" fn(
    cmd_list: *mut c_void,
    handle: *mut c_void,
    params: *mut c_void,
    callback: *const c_void,
) -> u32;
type FnD3D12ReleaseFeature = unsafe extern "C" fn(handle: *mut c_void) -> u32;

type FnParamSetUI = unsafe extern "C" fn(params: *mut c_void, name: *const c_char, val: u32);
type FnParamSetI = unsafe extern "C" fn(params: *mut c_void, name: *const c_char, val: i32);
type FnParamSetF = unsafe extern "C" fn(params: *mut c_void, name: *const c_char, val: f32);
type FnParamSetD3D12Resource =
    unsafe extern "C" fn(params: *mut c_void, name: *const c_char, val: *mut c_void);
type FnParamGetI =
    unsafe extern "C" fn(params: *mut c_void, name: *const c_char, out: *mut i32) -> u32;

/// User-selected DLSS quality / reconstruction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DlssMode {
    Off,
    Dlaa,
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
}

impl DlssMode {
    pub(crate) fn from_env() -> Self {
        static MODE: OnceLock<DlssMode> = OnceLock::new();
        *MODE.get_or_init(|| {
            parse_dlss_mode(std::env::var("VICTORY_DLSS").ok().as_deref())
        })
    }

    /// Maps the live F10 menu `RenderQuality` preset onto the matching DLSS mode.
    pub(crate) fn from_render_quality(q: crate::fsr::RenderQuality) -> Self {
        match q {
            crate::fsr::RenderQuality::Native => DlssMode::Dlaa,
            crate::fsr::RenderQuality::Quality => DlssMode::Quality,
            crate::fsr::RenderQuality::Performance => DlssMode::Performance,
            crate::fsr::RenderQuality::UltraPerformance => DlssMode::UltraPerformance,
        }
    }

    fn perf_quality_value(self) -> i32 {
        match self {
            DlssMode::Performance => 0,
            DlssMode::Balanced => 1,
            DlssMode::Quality => 2,
            DlssMode::UltraPerformance => 3,
            DlssMode::Dlaa => 5,
            DlssMode::Off => 5,
        }
    }

    /// Computes `(out_width, out_height)` for a given input scene size.
    pub(crate) fn output_dimensions(self, in_w: u32, in_h: u32) -> (u32, u32) {
        let env_u32 = |name: &str| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.trim().parse::<u32>().ok())
                .filter(|&v| v >= 64 && v <= 7680)
        };
        if let (Some(w), Some(h)) = (env_u32("VICTORY_DLSS_WIDTH"), env_u32("VICTORY_DLSS_HEIGHT")) {
            return (w.max(in_w), h.max(in_h));
        }
        scale_dimensions(self, in_w, in_h)
    }
}

fn parse_dlss_mode(raw: Option<&str>) -> DlssMode {
    match raw.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("0" | "off" | "false" | "no" | "disabled") => DlssMode::Off,
        Some("1" | "auto" | "dlaa" | "native") | None => DlssMode::Dlaa,
        Some("quality" | "maxquality") => DlssMode::Quality,
        Some("balanced") => DlssMode::Balanced,
        Some("performance" | "perf" | "maxperf") => DlssMode::Performance,
        Some("ultra" | "ultraperformance" | "ultra_performance") => DlssMode::UltraPerformance,
        _ => DlssMode::Dlaa,
    }
}

fn scale_dimensions(mode: DlssMode, in_w: u32, in_h: u32) -> (u32, u32) {
    let (num, den) = match mode {
        DlssMode::Off | DlssMode::Dlaa => (1, 1),
        DlssMode::Quality => (3, 2),           // 1.5x
        DlssMode::Balanced => (17, 10),        // 1.7x
        DlssMode::Performance => (2, 1),       // 2.0x
        DlssMode::UltraPerformance => (3, 1),  // 3.0x
    };
    if num == den {
        return (in_w, in_h);
    }
    let out_w = ((u64::from(in_w) * num) / den).clamp(u64::from(in_w), 3840) as u32;
    let out_h = ((u64::from(in_h) * num) / den).clamp(u64::from(in_h), 2160) as u32;
    // Align to even pixels for swap-chain compatibility.
    (out_w & !1, out_h & !1)
}

fn dlss_reset_flag() -> i32 {
    static RESET: OnceLock<i32> = OnceLock::new();
    *RESET.get_or_init(|| {
        if std::env::var_os("VICTORY_DLSS_RESET").is_some_and(|v| v == "0") {
            0
        } else {
            1
        }
    })
}

fn to_wide(s: &OsStr) -> Vec<u16> {
    s.encode_wide().chain(std::iter::once(0)).collect()
}

/// Discovers directories containing `nvngx_dlss.dll`.
fn find_dlss_search_dirs() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(dir) = std::env::var("VICTORY_DLSS_DIR")
        && !dir.trim().is_empty()
    {
        candidates.push(PathBuf::from(dir));
    }
    if let Ok(dir) = std::env::var("VICTORY_GAME_DIR")
        && !dir.trim().is_empty()
    {
        candidates.push(PathBuf::from(dir));
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
    {
        candidates.push(parent.to_path_buf());
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd);
    }

    let mut found = Vec::new();
    for dir in candidates {
        if dir.join("nvngx_dlss.dll").is_file() && !found.contains(&dir) {
            found.push(dir);
        }
    }
    found
}

/// Queries `HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\NGXCore\NGXPath`
/// for the NVIDIA driver's `_nvngx.dll` directory.
fn registry_ngx_core_path() -> Option<PathBuf> {
    let subkey = to_wide(OsStr::new(
        r"SYSTEM\CurrentControlSet\Services\nvlddmkm\NGXCore",
    ));
    let value_name = to_wide(OsStr::new("NGXPath"));
    let mut hkey: HKEY = std::ptr::null_mut();
    // SAFETY: standard Win32 registry read into local buffers.
    unsafe {
        if RegOpenKeyExW(
            HKEY_LOCAL_MACHINE,
            subkey.as_ptr(),
            0,
            KEY_READ,
            &mut hkey,
        ) != 0
        {
            return None;
        }
        let mut reg_type = 0u32;
        let mut bytes = 0u32;
        let rc = RegQueryValueExW(
            hkey,
            value_name.as_ptr(),
            std::ptr::null(),
            &mut reg_type,
            std::ptr::null_mut(),
            &mut bytes,
        );
        if rc != 0 || bytes < 2 || (reg_type != REG_SZ && reg_type != REG_EXPAND_SZ) {
            RegCloseKey(hkey);
            return None;
        }
        let mut buf = vec![0u16; (bytes as usize + 1) / 2];
        let rc = RegQueryValueExW(
            hkey,
            value_name.as_ptr(),
            std::ptr::null(),
            &mut reg_type,
            buf.as_mut_ptr().cast::<u8>(),
            &mut bytes,
        );
        RegCloseKey(hkey);
        if rc != 0 {
            return None;
        }
        let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
        let path = PathBuf::from(String::from_utf16_lossy(&buf[..len]));
        let dll = path.join("_nvngx.dll");
        dll.is_file().then_some(dll)
    }
}

struct NgxApi {
    module: HMODULE,
    init_ext: FnD3D12InitExt,
    shutdown1: FnD3D12Shutdown1,
    get_cap_params: FnD3D12GetCapabilityParameters,
    alloc_params: FnD3D12AllocateParameters,
    destroy_params: FnD3D12DestroyParameters,
    create_feature: FnD3D12CreateFeature,
    evaluate_feature: FnD3D12EvaluateFeature,
    release_feature: FnD3D12ReleaseFeature,
    set_ui: FnParamSetUI,
    set_i: FnParamSetI,
    set_f: FnParamSetF,
    set_d3d12_res: FnParamSetD3D12Resource,
    get_i: FnParamGetI,
}

// SAFETY: loaded DLL module and exported function pointers are process-wide and immutable.
unsafe impl Send for NgxApi {}
unsafe impl Sync for NgxApi {}

impl NgxApi {
    fn load() -> Option<Self> {
        let module = unsafe {
            let from_reg = registry_ngx_core_path().and_then(|p| {
                let w = to_wide(p.as_os_str());
                let h = LoadLibraryW(w.as_ptr());
                (!h.is_null()).then_some(h)
            });
            match from_reg {
                Some(h) => h,
                None => {
                    let w = to_wide(OsStr::new("_nvngx.dll"));
                    let h = LoadLibraryW(w.as_ptr());
                    if h.is_null() {
                        return None;
                    }
                    h
                }
            }
        };

        unsafe fn sym<T: Copy>(module: HMODULE, name: &[u8]) -> Option<T> {
            // SAFETY: caller passes null-terminated ASCII export name and valid HMODULE.
            let p = unsafe { GetProcAddress(module, name.as_ptr()) }?;
            Some(unsafe { std::mem::transmute_copy(&p) })
        }

        // SAFETY: all symbol names are null-terminated and match NGX C exports.
        let loaded = unsafe {
            Some(NgxApi {
                module,
                init_ext: sym(module, b"NVSDK_NGX_D3D12_Init_Ext\0")?,
                shutdown1: sym(module, b"NVSDK_NGX_D3D12_Shutdown1\0")?,
                get_cap_params: sym(module, b"NVSDK_NGX_D3D12_GetCapabilityParameters\0")?,
                alloc_params: sym(module, b"NVSDK_NGX_D3D12_AllocateParameters\0")?,
                destroy_params: sym(module, b"NVSDK_NGX_D3D12_DestroyParameters\0")?,
                create_feature: sym(module, b"NVSDK_NGX_D3D12_CreateFeature\0")?,
                evaluate_feature: sym(module, b"NVSDK_NGX_D3D12_EvaluateFeature\0")?,
                release_feature: sym(module, b"NVSDK_NGX_D3D12_ReleaseFeature\0")?,
                set_ui: sym(module, b"NVSDK_NGX_Parameter_SetUI\0")?,
                set_i: sym(module, b"NVSDK_NGX_Parameter_SetI\0")?,
                set_f: sym(module, b"NVSDK_NGX_Parameter_SetF\0")?,
                set_d3d12_res: sym(module, b"NVSDK_NGX_Parameter_SetD3d12Resource\0")?,
                get_i: sym(module, b"NVSDK_NGX_Parameter_GetI\0")?,
            })
        };
        if loaded.is_none() {
            // SAFETY: module was loaded by LoadLibraryW above.
            unsafe { FreeLibrary(module) };
        }
        loaded
    }
}

impl Drop for NgxApi {
    fn drop(&mut self) {
        if !self.module.is_null() {
            // SAFETY: valid HMODULE owned by NgxApi.
            unsafe { FreeLibrary(self.module) };
        }
    }
}

struct DlssInstance {
    handle: *mut c_void,
    mode: DlssMode,
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
    format: DXGI_FORMAT,
    output: ID3D12Resource,
    depth: ID3D12Resource,
    mvs: ID3D12Resource,
}

/// Presenter-owned NVIDIA DLSS state.
pub(crate) struct Dlss {
    api: NgxApi,
    device: ID3D12Device,
    params: *mut c_void,
    mode: DlssMode,
    instance: Option<DlssInstance>,
}

// SAFETY: protected by Presenter's mutex; D3D12 & NGX handles are only used
// while holding that lock.
unsafe impl Send for Dlss {}

impl Dlss {
    /// Attempts to initialize NVIDIA NGX + `nvngx_dlss.dll` on `host`.
    /// Returns `None` cleanly (with a log message) if disabled, non-NVIDIA,
    /// `nvngx_dlss.dll` is not found, or NGX reports DLSS unavailable.
    pub(crate) fn try_new(host: &Host) -> Option<Self> {
        let mode = DlssMode::from_render_quality(crate::fsr::render_quality());
        if DlssMode::from_env() == DlssMode::Off || !host.is_nvidia() {
            crate::fsr::set_dlss_available(false);
            return None;
        }
        Self::init_with_mode(host, mode)
    }

    /// Attempts to initialize NVIDIA DLSS on demand (e.g. when switched to
    /// DLSS from the in-game F10 menu).
    pub(crate) fn try_new_live(host: &Host) -> Option<Self> {
        if !host.is_nvidia() {
            crate::fsr::set_dlss_available(false);
            return None;
        }
        let mode = DlssMode::from_render_quality(crate::fsr::render_quality());
        Self::init_with_mode(host, mode)
    }

    fn init_with_mode(host: &Host, mode: DlssMode) -> Option<Self> {
        let search_dirs = find_dlss_search_dirs();
        if search_dirs.is_empty() {
            crate::fsr::set_dlss_available(false);
            tracing::info!(
                "NVIDIA DLSS: nvngx_dlss.dll not found in VICTORY_GAME_DIR / executable directory (set VICTORY_DLSS_DIR to override)"
            );
            return None;
        }
        let Some(api) = NgxApi::load() else {
            crate::fsr::set_dlss_available(false);
            tracing::warn!(
                "NVIDIA DLSS: nvngx_dlss.dll found at {}, but NVIDIA NGX driver runtime (_nvngx.dll) could not be loaded",
                search_dirs[0].display()
            );
            return None;
        };

        let wide_dirs: Vec<Vec<u16>> = search_dirs
            .iter()
            .map(|p| to_wide(p.as_os_str()))
            .collect();
        let wide_ptrs: Vec<*const u16> = wide_dirs.iter().map(|v| v.as_ptr()).collect();
        let app_data = std::env::temp_dir().join("victory-ngx");
        let _ = std::fs::create_dir_all(&app_data);
        let app_data_wide = to_wide(app_data.as_os_str());

        let common = NgxFeatureCommonInfo {
            path_list_info: NgxPathListInfo {
                path: wide_ptrs.as_ptr(),
                length: wide_ptrs.len() as u32,
            },
            internal_data: std::ptr::null_mut(),
            logging_info: NgxLoggingInfo {
                logging_callback: std::ptr::null(),
                minimum_logging_level: 0,
                disable_other_logging_sinks: false,
            },
        };

        // SAFETY: all pointers live through the Init_Ext call; `host.device` is a live ID3D12Device.
        let init_rc = unsafe {
            (api.init_ext)(
                VICTORY_NGX_APP_ID,
                app_data_wide.as_ptr(),
                host.device.as_raw(),
                &common,
                NVSDK_NGX_VERSION_API,
            )
        };
        if init_rc != NVSDK_NGX_RESULT_SUCCESS {
            tracing::warn!(
                result = format_args!("{init_rc:#010x}"),
                dll_dir = %search_dirs[0].display(),
                "NVIDIA DLSS: NVSDK_NGX_D3D12_Init_Ext failed; falling back to standard presentation"
            );
            return None;
        }

        let mut cap_params: *mut c_void = std::ptr::null_mut();
        // SAFETY: NGX is initialized on `host.device`.
        let cap_rc = unsafe { (api.get_cap_params)(&mut cap_params) };
        if cap_rc != NVSDK_NGX_RESULT_SUCCESS || cap_params.is_null() {
            tracing::warn!(
                result = format_args!("{cap_rc:#010x}"),
                "NVIDIA DLSS: GetCapabilityParameters failed"
            );
            unsafe { (api.shutdown1)(host.device.as_raw()) };
            return None;
        }

        let mut available = 0i32;
        let key_avail = CString::new("SuperSampling.Available").ok()?;
        // SAFETY: valid NGX parameter block and null-terminated ASCII key.
        let avail_rc = unsafe { (api.get_i)(cap_params, key_avail.as_ptr(), &mut available) };
        if avail_rc != NVSDK_NGX_RESULT_SUCCESS || available == 0 {
            tracing::warn!(
                result = format_args!("{avail_rc:#010x}"),
                available,
                dll_dir = %search_dirs[0].display(),
                "NVIDIA DLSS: SuperSampling not available on this GPU/driver/DLL"
            );
            unsafe {
                (api.destroy_params)(cap_params);
                (api.shutdown1)(host.device.as_raw());
            }
            return None;
        }

        let mut params: *mut c_void = std::ptr::null_mut();
        let alloc_rc = unsafe { (api.alloc_params)(&mut params) };
        unsafe { (api.destroy_params)(cap_params) };
        if alloc_rc != NVSDK_NGX_RESULT_SUCCESS || params.is_null() {
            tracing::warn!(
                result = format_args!("{alloc_rc:#010x}"),
                "NVIDIA DLSS: AllocateParameters failed"
            );
            unsafe { (api.shutdown1)(host.device.as_raw()) };
            return None;
        }

        crate::fsr::set_dlss_available(true);
        tracing::info!(
            mode = ?mode,
            dll_dir = %search_dirs[0].display(),
            reset = dlss_reset_flag(),
            "NVIDIA DLSS initialized on host D3D12 device"
        );

        Some(Dlss {
            api,
            device: host.device.clone(),
            params,
            mode,
            instance: None,
        })
    }

    pub(crate) fn output_dimensions(&self, in_w: u32, in_h: u32) -> (u32, u32) {
        DlssMode::from_render_quality(crate::fsr::render_quality()).output_dimensions(in_w, in_h)
    }

    /// Evaluates DLSS on `color` (which is currently in `color_state` and
    /// returned to `color_state` afterwards), returning a reference to the
    /// upscaled/reconstructed VRAM output texture left in
    /// `D3D12_RESOURCE_STATE_UNORDERED_ACCESS`.
    pub(crate) fn evaluate(
        &mut self,
        host: &Host,
        list: &ID3D12GraphicsCommandList,
        color: &ID3D12Resource,
        color_state: D3D12_RESOURCE_STATES,
        in_w: u32,
        in_h: u32,
        format: DXGI_FORMAT,
    ) -> Option<&ID3D12Resource> {
        let live_mode = DlssMode::from_render_quality(crate::fsr::render_quality());
        self.mode = live_mode;
        let (out_w, out_h) = live_mode.output_dimensions(in_w, in_h);
        let recreate = self.instance.as_ref().is_none_or(|inst| {
            (inst.mode, inst.in_w, inst.in_h, inst.out_w, inst.out_h, inst.format)
                != (live_mode, in_w, in_h, out_w, out_h, format)
        });
        if recreate && !self.recreate(host, in_w, in_h, out_w, out_h, format) {
            crate::fsr::set_dlss_active(false);
            return None;
        }
        let inst = self.instance.as_ref()?;

        let dlss_in_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        // SAFETY: `list` is open; `color` and `inst` textures are live VRAM resources.
        unsafe {
            if color_state != dlss_in_state {
                list.ResourceBarrier(&[transition(color, color_state, dlss_in_state)]);
            }

            let set_res = |name: &[u8], res: &ID3D12Resource| {
                (self.api.set_d3d12_res)(self.params, name.as_ptr().cast(), res.as_raw());
            };
            let set_ui = |name: &[u8], v: u32| {
                (self.api.set_ui)(self.params, name.as_ptr().cast(), v);
            };
            let set_i = |name: &[u8], v: i32| {
                (self.api.set_i)(self.params, name.as_ptr().cast(), v);
            };
            let set_f = |name: &[u8], v: f32| {
                (self.api.set_f)(self.params, name.as_ptr().cast(), v);
            };

            set_res(b"Color\0", color);
            set_res(b"Output\0", &inst.output);
            set_res(b"Depth\0", &inst.depth);
            set_res(b"MotionVectors\0", &inst.mvs);
            set_f(b"Jitter.Offset.X\0", 0.0);
            set_f(b"Jitter.Offset.Y\0", 0.0);
            set_i(b"Reset\0", dlss_reset_flag());
            set_f(b"MV.Scale.X\0", 1.0);
            set_f(b"MV.Scale.Y\0", 1.0);
            set_f(b"Sharpness\0", 0.0);
            set_ui(b"DLSS.Render.Subrect.Dimensions.Width\0", in_w);
            set_ui(b"DLSS.Render.Subrect.Dimensions.Height\0", in_h);
            set_ui(b"DLSS.Input.Color.Subrect.Base.X\0", 0);
            set_ui(b"DLSS.Input.Color.Subrect.Base.Y\0", 0);
            set_ui(b"DLSS.Output.Subrect.Base.X\0", 0);
            set_ui(b"DLSS.Output.Subrect.Base.Y\0", 0);
            set_ui(b"DLSS.Input.Depth.Subrect.Base.X\0", 0);
            set_ui(b"DLSS.Input.Depth.Subrect.Base.Y\0", 0);
            set_ui(b"DLSS.Input.MV.Subrect.Base.X\0", 0);
            set_ui(b"DLSS.Input.MV.Subrect.Base.Y\0", 0);

            let rc = (self.api.evaluate_feature)(
                list.as_raw(),
                inst.handle,
                self.params,
                std::ptr::null(),
            );
            if color_state != dlss_in_state {
                list.ResourceBarrier(&[transition(color, dlss_in_state, color_state)]);
            }
            if rc != NVSDK_NGX_RESULT_SUCCESS {
                static WARNED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    tracing::warn!(
                        result = format_args!("{rc:#010x}"),
                        "NVIDIA DLSS: EvaluateFeature failed; falling back to direct presentation"
                    );
                }
                return None;
            }
        }
        Some(&inst.output)
    }

    fn recreate(
        &mut self,
        host: &Host,
        in_w: u32,
        in_h: u32,
        out_w: u32,
        out_h: u32,
        format: DXGI_FORMAT,
    ) -> bool {
        if let Some(old) = self.instance.take() {
            // SAFETY: valid NGX handle created on this device.
            unsafe { (self.api.release_feature)(old.handle) };
        }
        let make_tex = |w: u32,
                        h: u32,
                        fmt: DXGI_FORMAT,
                        flags: D3D12_RESOURCE_FLAGS,
                        state: D3D12_RESOURCE_STATES|
         -> Option<ID3D12Resource> {
            let mut res: Option<ID3D12Resource> = None;
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(w),
                Height: h,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: fmt,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: flags,
                ..Default::default()
            };
            // SAFETY: valid VRAM texture creation on live device.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_DEFAULT,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        state,
                        None,
                        &mut res,
                    )
                    .ok()?;
            }
            res
        };

        let Some(output) = make_tex(
            out_w,
            out_h,
            format,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        ) else {
            return false;
        };
        let Some(depth) = make_tex(
            in_w,
            in_h,
            DXGI_FORMAT_R32_FLOAT,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        ) else {
            return false;
        };
        let Some(mvs) = make_tex(
            in_w,
            in_h,
            DXGI_FORMAT_R16G16_FLOAT,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        ) else {
            return false;
        };

        let Ok(init_list) = host.new_list() else {
            return false;
        };
        let is_hdr = matches!(format, DXGI_FORMAT_R16G16B16A16_FLOAT);
        let mut handle: *mut c_void = std::ptr::null_mut();
        // SAFETY: valid open command list and NGX parameter block.
        let rc = unsafe {
            (self.api.set_ui)(self.params, b"CreationNodeMask\0".as_ptr().cast(), 1);
            (self.api.set_ui)(self.params, b"VisibilityNodeMask\0".as_ptr().cast(), 1);
            (self.api.set_ui)(self.params, b"Width\0".as_ptr().cast(), in_w);
            (self.api.set_ui)(self.params, b"Height\0".as_ptr().cast(), in_h);
            (self.api.set_ui)(self.params, b"OutWidth\0".as_ptr().cast(), out_w);
            (self.api.set_ui)(self.params, b"OutHeight\0".as_ptr().cast(), out_h);
            (self.api.set_i)(
                self.params,
                b"PerfQualityValue\0".as_ptr().cast(),
                self.mode.perf_quality_value(),
            );
            (self.api.set_i)(
                self.params,
                b"DLSS.Feature.Create.Flags\0".as_ptr().cast(),
                if is_hdr { NVSDK_NGX_DLSS_FEATURE_FLAG_IS_HDR } else { 0 },
            );
            (self.api.create_feature)(
                init_list.list.as_raw(),
                NVSDK_NGX_FEATURE_SUPER_SAMPLING,
                self.params,
                &mut handle,
            )
        };
        init_list.close();
        if rc != NVSDK_NGX_RESULT_SUCCESS || handle.is_null() {
            tracing::warn!(
                result = format_args!("{rc:#010x}"),
                in_w,
                in_h,
                out_w,
                out_h,
                format = format.0,
                "NVIDIA DLSS: CreateFeature failed"
            );
            return false;
        }
        host.execute(&[&init_list]);
        tracing::info!(
            mode = ?self.mode,
            in_w,
            in_h,
            out_w,
            out_h,
            format = format.0,
            "NVIDIA DLSS Super Resolution feature ready"
        );
        self.instance = Some(DlssInstance {
            handle,
            mode: self.mode,
            in_w,
            in_h,
            out_w,
            out_h,
            format,
            output,
            depth,
            mvs,
        });
        true
    }
}

impl Drop for Dlss {
    fn drop(&mut self) {
        // SAFETY: releasing owned NGX handles and shutting down NGX on our device.
        unsafe {
            if let Some(inst) = self.instance.take() {
                (self.api.release_feature)(inst.handle);
            }
            if !self.params.is_null() {
                (self.api.destroy_params)(self.params);
                self.params = std::ptr::null_mut();
            }
            (self.api.shutdown1)(self.device.as_raw());
        }
    }
}

/// Checks whether `dir` contains `nvngx_dlss.dll`.
#[allow(dead_code)]
pub(crate) fn has_dlss_dll(dir: &Path) -> bool {
    dir.join("nvngx_dlss.dll").is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_dlss_modes() {
        assert_eq!(parse_dlss_mode(Some("0")), DlssMode::Off);
        assert_eq!(parse_dlss_mode(Some("off")), DlssMode::Off);
        assert_eq!(parse_dlss_mode(None), DlssMode::Dlaa);
        assert_eq!(parse_dlss_mode(Some("auto")), DlssMode::Dlaa);
        assert_eq!(parse_dlss_mode(Some("dlaa")), DlssMode::Dlaa);
        assert_eq!(parse_dlss_mode(Some("quality")), DlssMode::Quality);
        assert_eq!(parse_dlss_mode(Some("balanced")), DlssMode::Balanced);
        assert_eq!(parse_dlss_mode(Some("performance")), DlssMode::Performance);
        assert_eq!(
            parse_dlss_mode(Some("ultra_performance")),
            DlssMode::UltraPerformance
        );
    }

    #[test]
    fn scales_dimensions_cleanly() {
        assert_eq!(scale_dimensions(DlssMode::Dlaa, 1920, 1080), (1920, 1080));
        assert_eq!(scale_dimensions(DlssMode::Quality, 1280, 720), (1920, 1080));
        assert_eq!(
            scale_dimensions(DlssMode::Performance, 1920, 1080),
            (3840, 2160)
        );
        assert_eq!(
            scale_dimensions(DlssMode::UltraPerformance, 1920, 1080),
            (3840, 2160)
        );
    }
}
