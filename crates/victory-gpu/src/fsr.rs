//! AMD FidelityFX Super Resolution (FSR) detection, interception state, and
//! shared telemetry for Frame Generation and diagnostics.
//!
//! When GTA VI loads `amd_fidelityfx_upscaler_xbox.dll` from the game directory,
//! `victory-hle` wraps its exported FidelityFX API entry points (`ffxCreateContext`,
//! `ffxDestroyContext`, `ffxDispatch`, `ffxConfigure`, `ffxQuery`) and forwards
//! every call straight through to the real Xbox DLL while notifying this module.
//!
//! On each `ffxDispatch` call we safely validate any title `ID3D12Resource*`
//! handles inside the descriptor (via `resource::is_resource`) and capture the
//! live host motion-vector and depth textures so our GPU Frame Generation pass
//! (`crate::framegen`) can perform real motion-vector-guided interpolation.

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

#[cfg(windows)]
use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

/// Standard FidelityFX API 2.0 / FSR 3.1 descriptor header (`ffxApiHeader`).
pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE: u64 = 0x0001_0000;
pub const FFX_API_DISPATCH_DESC_TYPE_UPSCALE: u64 = 0x0001_0001;
pub const FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE: u64 = 0x0001_0002;
pub const FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE: u64 = 0x0001_0003;
pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGEN: u64 = 0x0002_0000;
pub const FFX_API_DISPATCH_DESC_TYPE_FRAMEGEN: u64 = 0x0002_0001;

/// Filename of the Xbox FidelityFX upscaler DLL shipped in the game directory.
pub const FSR_XBOX_DLL_NAME: &str = "amd_fidelityfx_upscaler_xbox.dll";

/// Live user-selectable upscaler backend (default: FSR).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum UpscalerChoice {
    Fsr = 0,
    Dlss = 1,
}

impl UpscalerChoice {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Dlss,
            _ => Self::Fsr,
        }
    }

    pub fn toggle(self) -> Self {
        match self {
            Self::Fsr => Self::Dlss,
            Self::Dlss => Self::Fsr,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Fsr => "FSR",
            Self::Dlss => "DLSS",
        }
    }
}

/// Live user-selectable render quality preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RenderQuality {
    Native = 0,
    Quality = 1,
    Performance = 2,
    UltraPerformance = 3,
}

impl RenderQuality {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Quality,
            2 => Self::Performance,
            3 => Self::UltraPerformance,
            _ => Self::Native,
        }
    }

    pub fn next(self) -> Self {
        Self::from_u8((self as u8 + 1) % 4)
    }

    pub fn prev(self) -> Self {
        Self::from_u8((self as u8 + 3) % 4)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Native => "NATIVE",
            Self::Quality => "QUALITY",
            Self::Performance => "PERFORMANCE",
            Self::UltraPerformance => "ULTRA-PERFORMANCE",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::Native => "NAT",
            Self::Quality => "QUAL",
            Self::Performance => "PERF",
            Self::UltraPerformance => "UPRF",
        }
    }

    /// Corresponding `FfxApiUpscaleQualityMode` enum value for `ffxQuery`.
    pub fn ffx_quality_mode(self) -> u32 {
        match self {
            Self::Native => 0,           // FFX_UPSCALE_QUALITY_MODE_NATIVEAA (1.0x)
            Self::Quality => 1,          // FFX_UPSCALE_QUALITY_MODE_QUALITY (1.5x)
            Self::Performance => 3,      // FFX_UPSCALE_QUALITY_MODE_PERFORMANCE (2.0x)
            Self::UltraPerformance => 4, // FFX_UPSCALE_QUALITY_MODE_ULTRA_PERFORMANCE (3.0x)
        }
    }
}

use std::sync::atomic::AtomicU8;

static LIVE_SETTINGS_INIT: OnceLock<()> = OnceLock::new();
/// Selected upscaler (`0 = FSR` default, `1 = DLSS`).
static LIVE_UPSCALER: AtomicU8 = AtomicU8::new(0);
/// Selected render quality (`0 = Native`, `1 = Quality`, `2 = Performance`, `3 = UltraPerformance`).
static LIVE_QUALITY: AtomicU8 = AtomicU8::new(0);
/// Live FrameGen override (`0 = unset/env default`, `1 = OFF`, `2 = ON`).
static LIVE_FRAMEGEN: AtomicU8 = AtomicU8::new(0);
static DLSS_AVAILABLE: AtomicBool = AtomicBool::new(false);
static DLSS_ACTIVE: AtomicBool = AtomicBool::new(false);

fn init_live_settings() {
    LIVE_SETTINGS_INIT.get_or_init(|| {
        // Default upscaler is FSR unless VICTORY_UPSCALER=dlss is explicitly set.
        if let Ok(v) = std::env::var("VICTORY_UPSCALER")
            && v.trim().eq_ignore_ascii_case("dlss")
        {
            LIVE_UPSCALER.store(UpscalerChoice::Dlss as u8, Ordering::Relaxed);
        }
        let q = std::env::var("VICTORY_RENDER_QUALITY")
            .ok()
            .or_else(|| std::env::var("VICTORY_DLSS").ok());
        if let Some(raw) = q {
            let parsed = match raw.trim().to_ascii_lowercase().as_str() {
                "quality" | "maxquality" => RenderQuality::Quality,
                "performance" | "perf" | "maxperf" => RenderQuality::Performance,
                "ultra" | "ultra-performance" | "ultra_performance" | "ultraperformance" => {
                    RenderQuality::UltraPerformance
                }
                _ => RenderQuality::Native,
            };
            LIVE_QUALITY.store(parsed as u8, Ordering::Relaxed);
        }
    });
}

pub fn upscaler_choice() -> UpscalerChoice {
    init_live_settings();
    UpscalerChoice::from_u8(LIVE_UPSCALER.load(Ordering::Relaxed))
}

pub fn set_upscaler_choice(choice: UpscalerChoice) {
    init_live_settings();
    let prev = LIVE_UPSCALER.swap(choice as u8, Ordering::AcqRel);
    if prev != choice as u8 {
        tracing::info!(upscaler = choice.label(), "in-game menu: upscaler changed");
    }
}

pub fn render_quality() -> RenderQuality {
    init_live_settings();
    RenderQuality::from_u8(LIVE_QUALITY.load(Ordering::Relaxed))
}

pub fn set_render_quality(quality: RenderQuality) {
    init_live_settings();
    let prev = LIVE_QUALITY.swap(quality as u8, Ordering::AcqRel);
    if prev != quality as u8 {
        tracing::info!(quality = quality.label(), "in-game menu: render quality changed");
    }
}

/// Live FrameGen enabled flag, combining F10 menu toggles with startup config.
pub fn framegen_enabled_live() -> bool {
    match LIVE_FRAMEGEN.load(Ordering::Acquire) {
        1 => false,
        2 => true,
        _ => {
            #[cfg(windows)]
            {
                crate::framegen::wanted()
            }
            #[cfg(not(windows))]
            {
                false
            }
        }
    }
}

pub fn set_framegen_enabled_live(on: bool) {
    let val = if on { 2 } else { 1 };
    let prev = LIVE_FRAMEGEN.swap(val, Ordering::AcqRel);
    if prev != val {
        tracing::info!(framegen = on, "in-game menu: FrameGen toggled");
    }
}

pub fn set_dlss_available(available: bool) {
    DLSS_AVAILABLE.store(available, Ordering::Relaxed);
}

pub fn dlss_available() -> bool {
    DLSS_AVAILABLE.load(Ordering::Relaxed)
}

pub fn set_dlss_active(active: bool) {
    DLSS_ACTIVE.store(active, Ordering::Relaxed);
}

pub fn dlss_active() -> bool {
    DLSS_ACTIVE.load(Ordering::Relaxed)
}

/// Overrides the `qualityMode` field on FidelityFX API `ffxQuery` calls
/// (`GETRENDERRESOLUTIONFROMQUALITYMODE` / `GETUPSCALERATIOFROMQUALITYMODE`)
/// so the live F10 render quality selection feeds directly into
/// `amd_fidelityfx_upscaler_xbox.dll`.
#[cfg(windows)]
pub fn apply_ffx_query_override(desc: *mut c_void) {
    if desc.is_null() {
        return;
    }
    let mut hdr = [0u8; 16];
    if !crate::host::read_into(desc.cast::<u8>().cast_const(), &mut hdr) {
        return;
    }
    let desc_type = u64::from_le_bytes(hdr[0..8].try_into().unwrap());
    let mode = render_quality().ffx_quality_mode();
    match desc_type {
        // ffxQueryDescUpscaleGetUpscaleRatioFromQualityMode: header(16) + qualityMode(u32 @ 16)
        FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE => {
            let mut probe = [0u8; 24];
            if crate::host::read_into(desc.cast::<u8>().cast_const(), &mut probe) {
                // SAFETY: verified mapped readable/writable query struct owned by caller.
                unsafe { desc.cast::<u8>().add(16).cast::<u32>().write_unaligned(mode) };
            }
        }
        // ffxQueryDescUpscaleGetRenderResolutionFromQualityMode: header(16) + displayW(16) + displayH(20) + qualityMode(u32 @ 24)
        FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE => {
            let mut probe = [0u8; 32];
            if crate::host::read_into(desc.cast::<u8>().cast_const(), &mut probe) {
                // SAFETY: verified mapped query struct owned by caller.
                unsafe { desc.cast::<u8>().add(24).cast::<u32>().write_unaligned(mode) };
            }
        }
        _ => {}
    }
}

#[cfg(not(windows))]
pub fn apply_ffx_query_override(_desc: *mut c_void) {}

static DLL_PRESENT: AtomicBool = AtomicBool::new(false);
static DLL_LOADED: AtomicBool = AtomicBool::new(false);
static ACTIVE_CONTEXTS: AtomicU32 = AtomicU32::new(0);
static DISPATCH_COUNT: AtomicU64 = AtomicU64::new(0);
static LAST_DISPATCH_FRAME: AtomicU64 = AtomicU64::new(0);
static RENDER_W: AtomicU32 = AtomicU32::new(0);
static RENDER_H: AtomicU32 = AtomicU32::new(0);
static UPSCALE_W: AtomicU32 = AtomicU32::new(0);
static UPSCALE_H: AtomicU32 = AtomicU32::new(0);
static MV_CAPTURED_FRAME: AtomicU64 = AtomicU64::new(0);
static NATIVE_FG_CONTEXTS: AtomicU32 = AtomicU32::new(0);

static DLL_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Host GPU resources captured from the title's live `ffxDispatch` call into
/// `amd_fidelityfx_upscaler_xbox.dll`.
#[cfg(windows)]
#[derive(Clone)]
#[allow(dead_code)]
pub(crate) struct CapturedFrameInputs {
    pub(crate) frame: u64,
    pub(crate) motion_vectors: Option<ID3D12Resource>,
    pub(crate) depth: Option<ID3D12Resource>,
    pub(crate) mv_scale: [f32; 2],
    pub(crate) jitter: [f32; 2],
    pub(crate) reset: bool,
}

#[cfg(windows)]
// SAFETY: ID3D12Resource COM wrappers are free-threaded.
unsafe impl Send for CapturedFrameInputs {}

#[cfg(windows)]
static CAPTURED: Mutex<Option<CapturedFrameInputs>> = Mutex::new(None);

/// Snapshot of FSR, Frame Generation, and Shadow Quality status for diagnostics
/// (console startup, HUD panel, window title, and periodic log lines).
#[derive(Debug, Clone)]
pub struct DiagnosticStatus {
    pub upscaler: UpscalerChoice,
    pub render_quality: RenderQuality,
    pub dlss_available: bool,
    pub dlss_active: bool,
    pub fsr_dll_present: bool,
    pub fsr_dll_loaded: bool,
    pub fsr_dll_path: Option<PathBuf>,
    pub fsr_contexts: u32,
    pub fsr_dispatches: u64,
    pub fsr_active: bool,
    pub fsr_render_size: (u32, u32),
    pub fsr_upscale_size: (u32, u32),
    pub fsr_has_motion_vectors: bool,
    pub framegen_enabled: bool,
    pub framegen_active: bool,
    pub framegen_mode: &'static str,
    pub framegen_generated: u64,
    pub shadow_scale: u32,
    pub shadow_maps_scaled: u64,
}

/// Probes `dir` (the game directory root) for `amd_fidelityfx_upscaler_xbox.dll`
/// and records its availability for Frame Generation and diagnostics.
pub fn probe_game_dir(dir: &Path) -> Option<PathBuf> {
    let candidate = dir.join(FSR_XBOX_DLL_NAME);
    if candidate.is_file() {
        DLL_PRESENT.store(true, Ordering::Release);
        *DLL_PATH.lock().unwrap_or_else(|e| e.into_inner()) = Some(candidate.clone());
        return Some(candidate);
    }
    None
}

/// Ensures we have probed `VICTORY_GAME_DIR` / exe dir / cwd at least once.
pub fn ensure_probed() {
    static PROBED: OnceLock<()> = OnceLock::new();
    PROBED.get_or_init(|| {
        if DLL_PRESENT.load(Ordering::Acquire) {
            return;
        }
        let mut dirs = Vec::new();
        if let Ok(d) = std::env::var("VICTORY_GAME_DIR")
            && !d.trim().is_empty()
        {
            dirs.push(PathBuf::from(d));
        }
        if let Ok(exe) = std::env::current_exe()
            && let Some(p) = exe.parent()
        {
            dirs.push(p.to_path_buf());
        }
        if let Ok(cwd) = std::env::current_dir() {
            dirs.push(cwd);
        }
        for dir in dirs {
            if probe_game_dir(&dir).is_some() {
                break;
            }
        }
    });
}

/// True if `amd_fidelityfx_upscaler_xbox.dll` was found in the game directory
/// or loaded at runtime by the title.
pub fn dll_present() -> bool {
    ensure_probed();
    DLL_PRESENT.load(Ordering::Acquire) || DLL_LOADED.load(Ordering::Acquire)
}

/// True if the title loaded `amd_fidelityfx_upscaler_xbox.dll` into memory.
pub fn dll_loaded() -> bool {
    DLL_LOADED.load(Ordering::Acquire)
}

/// Records that the title loaded `amd_fidelityfx_upscaler_xbox.dll`.
pub fn note_dll_loaded(path: Option<&Path>) {
    DLL_PRESENT.store(true, Ordering::Release);
    let first = !DLL_LOADED.swap(true, Ordering::AcqRel);
    if let Some(p) = path {
        *DLL_PATH.lock().unwrap_or_else(|e| e.into_inner()) = Some(p.to_path_buf());
    }
    if first {
        tracing::info!(
            path = ?path.map(|p| p.display().to_string()),
            "FSR: amd_fidelityfx_upscaler_xbox.dll loaded by title"
        );
    }
}

/// Records a successful `ffxCreateContext` call inside `amd_fidelityfx_upscaler_xbox.dll`.
pub fn note_context_created(desc_type: u64, max_render: (u32, u32), max_upscale: (u32, u32)) {
    DLL_LOADED.store(true, Ordering::Release);
    if desc_type == FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGEN {
        NATIVE_FG_CONTEXTS.fetch_add(1, Ordering::Relaxed);
        tracing::info!("FSR: native Frame Generation context created via FidelityFX DLL");
    } else {
        ACTIVE_CONTEXTS.fetch_add(1, Ordering::Relaxed);
        if max_render.0 > 0 && max_render.1 > 0 {
            RENDER_W.store(max_render.0, Ordering::Relaxed);
            RENDER_H.store(max_render.1, Ordering::Relaxed);
        }
        if max_upscale.0 > 0 && max_upscale.1 > 0 {
            UPSCALE_W.store(max_upscale.0, Ordering::Relaxed);
            UPSCALE_H.store(max_upscale.1, Ordering::Relaxed);
        }
        tracing::info!(
            desc_type = format_args!("{desc_type:#x}"),
            render_w = max_render.0,
            render_h = max_render.1,
            upscale_w = max_upscale.0,
            upscale_h = max_upscale.1,
            "FSR: upscaler context created via amd_fidelityfx_upscaler_xbox.dll"
        );
    }
}

/// Records an `ffxDestroyContext` call.
pub fn note_context_destroyed() {
    let _ = ACTIVE_CONTEXTS.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
        Some(v.saturating_sub(1))
    });
}

/// Called from `victory-hle`'s `ffxDispatch` wrapper for
/// `amd_fidelityfx_upscaler_xbox.dll`.
///
/// Safely probes `desc` (without ever dereferencing unmapped memory or assuming
/// brittle field offsets) to record dispatch telemetry and capture live host
/// motion vectors and depth buffers for GPU Frame Generation.
#[cfg(windows)]
pub fn note_dispatch_raw(desc: *const c_void) {
    DLL_LOADED.store(true, Ordering::Release);
    let count = DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    let frame = crate::frame::current();
    LAST_DISPATCH_FRAME.store(frame, Ordering::Relaxed);

    if desc.is_null() {
        return;
    }

    // Read up to 512 bytes safely via host::read_into so we can inspect header
    // and embedded title ID3D12Resource* handles without any crash risk.
    let mut raw = [0u8; 512];
    if !crate::host::read_into(desc.cast::<u8>(), &mut raw) {
        // Fallback: try reading at least the 16-byte header.
        let mut hdr = [0u8; 16];
        if !crate::host::read_into(desc.cast::<u8>(), &mut hdr) {
            return;
        }
        let desc_type = u64::from_le_bytes(hdr[0..8].try_into().unwrap());
        if count == 1 {
            tracing::info!(
                desc_type = format_args!("{desc_type:#x}"),
                "FSR: first ffxDispatch observed"
            );
        }
        return;
    }

    let desc_type = u64::from_le_bytes(raw[0..8].try_into().unwrap());
    let mut motion_vectors: Option<ID3D12Resource> = None;
    let mut depth: Option<ID3D12Resource> = None;
    let mut mv_dims = (0u32, 0u32);
    let mut out_dims = (0u32, 0u32);

    // Scan 8-byte aligned slots past the 16-byte header + 8-byte commandList for
    // valid title ID3D12Resource* pointers (FfxApiResource.resource).
    for offset in (24..raw.len()).step_by(8) {
        let ptr = usize::from_le_bytes(raw[offset..offset + 8].try_into().unwrap());
        if ptr < 0x1_0000 || ptr >= 0x7fff_ffff_0000 {
            continue;
        }
        let res = ptr as crate::object::This;
        if !crate::resource::is_resource(res) {
            continue;
        }
        let Some(host_tex) = crate::resource::host_resource(res) else {
            continue;
        };
        // SAFETY: live D3D12 resource owned by resource table.
        let d = unsafe { host_tex.GetDesc() };
        if d.Dimension != windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DIMENSION_TEXTURE2D {
            continue;
        }
        let fmt = d.Format.0 as u32;
        let flags = d.Flags.0 as u32;
        let is_depth = (flags & 2) != 0
            || matches!(fmt, 19 | 20 | 39 | 40 | 44 | 45 | 53 | 55);
        // R32G32_FLOAT (16), R16G16_FLOAT (34), R16G16_UNORM (35), R16G16_SNORM (37)
        let is_mv = matches!(fmt, 16 | 34 | 35 | 37);
        if is_depth && depth.is_none() {
            depth = Some(host_tex.clone());
        } else if is_mv && motion_vectors.is_none() {
            motion_vectors = Some(host_tex.clone());
            mv_dims = (d.Width as u32, d.Height);
        } else if !is_depth && !is_mv {
            // Color/output textures: track largest as upscale size, smaller as render size.
            let dims = (d.Width as u32, d.Height);
            if dims.0 >= out_dims.0 && dims.1 >= out_dims.1 {
                out_dims = dims;
            }
        }
    }

    if mv_dims.0 > 0 && mv_dims.1 > 0 {
        RENDER_W.store(mv_dims.0, Ordering::Relaxed);
        RENDER_H.store(mv_dims.1, Ordering::Relaxed);
    }
    if out_dims.0 > 0 && out_dims.1 > 0 {
        UPSCALE_W.store(out_dims.0, Ordering::Relaxed);
        UPSCALE_H.store(out_dims.1, Ordering::Relaxed);
    }

    let has_mv = motion_vectors.is_some();
    if has_mv {
        MV_CAPTURED_FRAME.store(frame, Ordering::Relaxed);
    }

    // Default normalized MV scale (pixel-space vs UV-space detected cleanly in shader).
    let mv_scale = if mv_dims.0 > 0 && mv_dims.1 > 0 {
        [1.0 / mv_dims.0 as f32, 1.0 / mv_dims.1 as f32]
    } else {
        [1.0, 1.0]
    };

    *CAPTURED.lock().unwrap_or_else(|e| e.into_inner()) = Some(CapturedFrameInputs {
        frame,
        motion_vectors,
        depth,
        mv_scale,
        jitter: [0.0, 0.0],
        reset: false,
    });

    if count == 1 {
        tracing::info!(
            desc_type = format_args!("{desc_type:#x}"),
            has_mv,
            render_w = RENDER_W.load(Ordering::Relaxed),
            render_h = RENDER_H.load(Ordering::Relaxed),
            upscale_w = UPSCALE_W.load(Ordering::Relaxed),
            upscale_h = UPSCALE_H.load(Ordering::Relaxed),
            "FSR: first ffxDispatch via amd_fidelityfx_upscaler_xbox.dll"
        );
    }
}

/// Non-windows stub for `note_dispatch_raw`.
#[cfg(not(windows))]
pub fn note_dispatch_raw(_desc: *const c_void) {
    DLL_LOADED.store(true, Ordering::Release);
    DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Returns captured FSR motion vectors & depth if dispatched within the last
/// 2 frames (so stale textures from menus/cutscenes are never used).
#[cfg(windows)]
pub(crate) fn fresh_inputs(current_frame: u64) -> Option<CapturedFrameInputs> {
    let guard = CAPTURED.lock().unwrap_or_else(|e| e.into_inner());
    let inputs = guard.as_ref()?;
    if current_frame.saturating_sub(inputs.frame) <= 2 && inputs.motion_vectors.is_some() {
        Some(inputs.clone())
    } else {
        None
    }
}

/// True when FSR has active contexts or dispatched within the last 120 frames.
pub fn fsr_active() -> bool {
    let dispatches = DISPATCH_COUNT.load(Ordering::Relaxed);
    let last = LAST_DISPATCH_FRAME.load(Ordering::Relaxed);
    let now = crate::frame::current();
    ACTIVE_CONTEXTS.load(Ordering::Relaxed) > 0
        || (dispatches > 0 && now.saturating_sub(last) <= 120)
}

/// True if `amd_fidelityfx_upscaler_xbox.dll` recently supplied motion vectors.
pub fn has_recent_motion_vectors() -> bool {
    let mv_frame = MV_CAPTURED_FRAME.load(Ordering::Relaxed);
    mv_frame > 0 && crate::frame::current().saturating_sub(mv_frame) <= 120
}

/// Full snapshot of FSR, FrameGen, and Shadow Quality for diagnostics & HUD.
pub fn status() -> DiagnosticStatus {
    ensure_probed();
    let dll_path = DLL_PATH.lock().unwrap_or_else(|e| e.into_inner()).clone();
    #[cfg(windows)]
    let (fg_enabled, fg_active, fg_mode, fg_generated) = crate::framegen::status();
    #[cfg(not(windows))]
    let (fg_enabled, fg_active, fg_mode, fg_generated) = (false, false, "OFF", 0u64);

    #[cfg(windows)]
    let (shadow_scale, shadow_maps_scaled) = (
        crate::resource::shadow_quality_scale(),
        crate::resource::shadow_maps_scaled_count(),
    );
    #[cfg(not(windows))]
    let (shadow_scale, shadow_maps_scaled) = (1u32, 0u64);

    let native_fg = NATIVE_FG_CONTEXTS.load(Ordering::Relaxed) > 0;

    DiagnosticStatus {
        upscaler: upscaler_choice(),
        render_quality: render_quality(),
        dlss_available: dlss_available(),
        dlss_active: dlss_active(),
        fsr_dll_present: DLL_PRESENT.load(Ordering::Acquire),
        fsr_dll_loaded: DLL_LOADED.load(Ordering::Acquire),
        fsr_dll_path: dll_path,
        fsr_contexts: ACTIVE_CONTEXTS.load(Ordering::Relaxed),
        fsr_dispatches: DISPATCH_COUNT.load(Ordering::Relaxed),
        fsr_active: fsr_active(),
        fsr_render_size: (
            RENDER_W.load(Ordering::Relaxed),
            RENDER_H.load(Ordering::Relaxed),
        ),
        fsr_upscale_size: (
            UPSCALE_W.load(Ordering::Relaxed),
            UPSCALE_H.load(Ordering::Relaxed),
        ),
        fsr_has_motion_vectors: has_recent_motion_vectors(),
        framegen_enabled: fg_enabled || native_fg,
        framegen_active: fg_active || native_fg,
        framegen_mode: if native_fg { "FFX-FG" } else { fg_mode },
        framegen_generated: fg_generated,
        shadow_scale,
        shadow_maps_scaled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probes_missing_and_present_upscaler_dll() {
        let dir = tempfile::tempdir().unwrap();
        assert!(probe_game_dir(dir.path()).is_none());
        let dll = dir.path().join(FSR_XBOX_DLL_NAME);
        std::fs::write(&dll, b"MZ").unwrap();
        assert_eq!(probe_game_dir(dir.path()), Some(dll));
    }
}
