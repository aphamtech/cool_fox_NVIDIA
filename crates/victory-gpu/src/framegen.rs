//! GPU Frame Generation (2x interpolated output) integrated with
//! `amd_fidelityfx_upscaler_xbox.dll`.
//!
//! When `amd_fidelityfx_upscaler_xbox.dll` is present in the game directory
//! (or `VICTORY_FRAMEGEN=1` is set), this module generates one motion-compensated
//! intermediate frame ($t = 0.5$) between the previous and current base 3D scene
//! frames directly in dedicated VRAM before the crisp UI overlay is composited:
//!
//! - **FSR-MV mode**: when `amd_fidelityfx_upscaler_xbox.dll` dispatched FSR on
//!   the current frame and `crate::fsr` captured the game's real motion vectors,
//!   uses those game motion vectors directly for bidirectional reprojection.
//! - **OPTFLOW mode**: when real FSR motion vectors are not available on a given
//!   frame (e.g. menus, cutscenes, Series X profile, or before the first
//!   `ffxDispatch`), estimates a quarter-resolution dense optical-flow velocity
//!   field on the GPU and interpolates with occlusion & scene-cut rejection.
//!
//! Environment switches:
//! - `VICTORY_FRAMEGEN`: `auto` (default: enabled when
//!   `amd_fidelityfx_upscaler_xbox.dll` is present), `1`/`on`/`fsr` (always on),
//!   or `0`/`off` (disabled).
//! - `VICTORY_FRAMEGEN_SYNC`: `1` (default: sync-interval 1 on the generated
//!   frame so DWM displays it instead of discarding it) or `0` (uncapped).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{PCSTR, s};

use crate::host::{Host, transition};

const MODE_OFF: u8 = 0;
const MODE_OPTFLOW: u8 = 1;
const MODE_FSR_MV: u8 = 2;

static FG_ENABLED: AtomicBool = AtomicBool::new(false);
static FG_ACTIVE: AtomicBool = AtomicBool::new(false);
static FG_MODE: AtomicU8 = AtomicU8::new(MODE_OFF);
static FG_GENERATED: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrameGenConfig {
    Off,
    Auto,
    ForceOn,
}

fn parse_framegen_config(raw: Option<&str>) -> FrameGenConfig {
    match raw.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("0" | "off" | "false" | "no" | "disabled") => FrameGenConfig::Off,
        Some("1" | "on" | "true" | "yes" | "fsr" | "force") => FrameGenConfig::ForceOn,
        Some("auto") | None => FrameGenConfig::Auto,
        _ => FrameGenConfig::Auto,
    }
}

pub(crate) fn config() -> FrameGenConfig {
    static CFG: OnceLock<FrameGenConfig> = OnceLock::new();
    *CFG.get_or_init(|| parse_framegen_config(std::env::var("VICTORY_FRAMEGEN").ok().as_deref()))
}

/// Whether Frame Generation should be enabled given current config and whether
/// `amd_fidelityfx_upscaler_xbox.dll` is present in the game directory.
pub(crate) fn wanted() -> bool {
    match config() {
        FrameGenConfig::Off => false,
        FrameGenConfig::ForceOn => true,
        FrameGenConfig::Auto => crate::fsr::dll_present(),
    }
}

/// Sync interval for the generated frame present call (`1` by default so DWM
/// presents the interpolated frame on screen rather than dropping it).
pub(crate) fn sync_interval() -> u32 {
    static SYNC: OnceLock<u32> = OnceLock::new();
    *SYNC.get_or_init(|| {
        if std::env::var_os("VICTORY_FRAMEGEN_SYNC").is_some_and(|v| v == "0") {
            0
        } else {
            1
        }
    })
}

/// Diagnostics snapshot `(enabled, active, mode_label, generated_frames)`.
pub(crate) fn status() -> (bool, bool, &'static str, u64) {
    let enabled = crate::fsr::framegen_enabled_live();
    let active = enabled && FG_ACTIVE.load(Ordering::Relaxed);
    let mode = if !enabled {
        "OFF"
    } else {
        match FG_MODE.load(Ordering::Relaxed) {
            MODE_FSR_MV => "FSR-MV",
            MODE_OPTFLOW => "OPTFLOW",
            _ => "READY",
        }
    };
    (enabled, active, mode, FG_GENERATED.load(Ordering::Relaxed))
}

/// Records that one generated frame was presented.
pub(crate) fn note_generated_present() {
    FG_ACTIVE.store(true, Ordering::Relaxed);
    FG_GENERATED.fetch_add(1, Ordering::Relaxed);
}

const FLOW_SHADER: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), DescriptorTable(SRV(t0, numDescriptors=2), UAV(u0, numDescriptors=1)), StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP)"
cbuffer Params : register(b0) {
    uint flowW;
    uint flowH;
    uint fullW;
    uint fullH;
};
Texture2D<float4> prevTex : register(t0);
Texture2D<float4> currTex : register(t1);
RWTexture2D<float2> flowOut : register(u0);
SamplerState lin : register(s0);

float luma(float3 c) { return dot(c, float3(0.299, 0.587, 0.114)); }

float patchCost(float2 uv, float2 delta, float2 texel) {
    float cost = 0.0;
    [unroll] for (int dy = -1; dy <= 1; dy++) {
        [unroll] for (int dx = -1; dx <= 1; dx++) {
            float2 off = float2(dx, dy) * texel * 2.0;
            float lp = luma(prevTex.SampleLevel(lin, saturate(uv + off), 0).rgb);
            float lc = luma(currTex.SampleLevel(lin, saturate(uv + off + delta), 0).rgb);
            cost += abs(lp - lc);
        }
    }
    return cost;
}

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void cs_flow(uint3 id : SV_DispatchThreadID) {
    if (id.x >= flowW || id.y >= flowH) return;
    float2 uv = (float2(id.xy) + 0.5) / float2(flowW, flowH);
    float2 texel = 1.0 / float2(max(fullW, 1u), max(fullH, 1u));
    float2 best = float2(0.0, 0.0);
    float bestCost = patchCost(uv, best, texel) - 0.02;

    // Coarse search (+/- 24 pixels at 8px step)
    [loop] for (int y = -3; y <= 3; y++) {
        [loop] for (int x = -3; x <= 3; x++) {
            if (x == 0 && y == 0) continue;
            float2 d = float2(x, y) * texel * 8.0;
            float c = patchCost(uv, d, texel);
            if (c < bestCost) {
                bestCost = c;
                best = d;
            }
        }
    }
    // Fine refinement (+/- 6 pixels at 2px step around coarse winner)
    float2 base = best;
    [loop] for (int y2 = -2; y2 <= 2; y2++) {
        [loop] for (int x2 = -2; x2 <= 2; x2++) {
            if (x2 == 0 && y2 == 0) continue;
            float2 d = base + float2(x2, y2) * texel * 2.0;
            float c = patchCost(uv, d, texel);
            if (c < bestCost) {
                bestCost = c;
                best = d;
            }
        }
    }
    // Reject scene cuts / noisy matches
    if (bestCost > 1.8) {
        best = float2(0.0, 0.0);
    }
    flowOut[id.xy] = best;
}
"#;

const INTERP_SHADER: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), DescriptorTable(SRV(t0, numDescriptors=3), UAV(u0, numDescriptors=1)), StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP)"
cbuffer Params : register(b0) {
    uint outW;
    uint outH;
    uint mvMode;   // 0 = optflow (UV delta), 1 = FSR game motion vectors
    uint pad0;
    float mvScaleX;
    float mvScaleY;
};
Texture2D<float4> prevTex : register(t0);
Texture2D<float4> currTex : register(t1);
Texture2D<float2> velTex  : register(t2);
RWTexture2D<float4> outTex : register(u0);
SamplerState lin : register(s0);

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void cs_interp(uint3 id : SV_DispatchThreadID) {
    if (id.x >= outW || id.y >= outH) return;
    float2 uv = (float2(id.xy) + 0.5) / float2(outW, outH);
    float2 rawVel = velTex.SampleLevel(lin, uv, 0);
    float2 vel = rawVel;
    if (mvMode == 1u) {
        // Normalize pixel-space MVs if magnitude exceeds UV space.
        float mag = max(abs(rawVel.x), abs(rawVel.y));
        if (mag > 1.5) {
            vel = rawVel * float2(mvScaleX, mvScaleY);
        }
    }
    // Clamp extreme displacements to avoid edge stretching artifacts.
    vel = clamp(vel, float2(-0.15, -0.15), float2(0.15, 0.15));

    float4 cPrevWarp = prevTex.SampleLevel(lin, saturate(uv - 0.5 * vel), 0);
    float4 cCurrWarp = currTex.SampleLevel(lin, saturate(uv + 0.5 * vel), 0);
    float4 cCurr = currTex.SampleLevel(lin, uv, 0);
    float4 cPrev = prevTex.SampleLevel(lin, uv, 0);

    float warpDiff = length(cPrevWarp.rgb - cCurrWarp.rgb);
    float staticDiff = length(cPrev.rgb - cCurr.rgb);

    // Bidirectional blend when consistent; fall back cleanly on disocclusions / cuts.
    float4 blended = 0.5 * (cPrevWarp + cCurrWarp);
    float disocc = saturate((warpDiff - 0.22) * 3.0);
    if (staticDiff > 0.55 && warpDiff > 0.35) {
        // Hard scene cut: show current frame cleanly without ghosting.
        blended = cCurr;
    } else {
        blended = lerp(blended, cCurr, disocc * 0.85);
    }
    outTex[id.xy] = float4(saturate(blended.rgb), 1.0);
}
"#;

fn compile_cs(source: &str, entry: PCSTR) -> windows::core::Result<ID3DBlob> {
    let mut code: Option<ID3DBlob> = None;
    let mut err: Option<ID3DBlob> = None;
    // SAFETY: source & entry string live through D3DCompile.
    let res = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("framegen"),
            None,
            None,
            entry,
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut err),
        )
    };
    if let Err(e) = res {
        if let Some(blob) = err {
            // SAFETY: valid error blob bytes.
            let msg = unsafe {
                let s = std::slice::from_raw_parts(
                    blob.GetBufferPointer().cast::<u8>(),
                    blob.GetBufferSize(),
                );
                String::from_utf8_lossy(s).into_owned()
            };
            tracing::warn!(%msg, "framegen shader compile error");
        }
        return Err(e);
    }
    code.ok_or_else(windows::core::Error::empty)
}

struct Targets {
    width: u32,
    height: u32,
    raw_fmt: DXGI_FORMAT,
    srv_fmt: DXGI_FORMAT,
    out_fmt: DXGI_FORMAT,
    prev_scene: ID3D12Resource,
    interp_scene: ID3D12Resource,
    flow_tex: ID3D12Resource,
    flow_w: u32,
    flow_h: u32,
    has_prev: bool,
}

/// Presenter-owned GPU Frame Generator.
pub(crate) struct FrameGen {
    flow_root: ID3D12RootSignature,
    flow_pso: ID3D12PipelineState,
    interp_root: ID3D12RootSignature,
    interp_pso: ID3D12PipelineState,
    heaps: Vec<ID3D12DescriptorHeap>,
    step: u32,
    targets: Option<Targets>,
}

// SAFETY: D3D12 interfaces are free-threaded; FrameGen is protected by Presenter's mutex.
unsafe impl Send for FrameGen {}

impl FrameGen {
    /// Initializes GPU Frame Generation when enabled (`VICTORY_FRAMEGEN` /
    /// `amd_fidelityfx_upscaler_xbox.dll` present in game directory).
    pub(crate) fn try_new(host: &Host) -> Option<Self> {
        let on = crate::fsr::framegen_enabled_live();
        FG_ENABLED.store(on, Ordering::Relaxed);
        if !on {
            FG_MODE.store(MODE_OFF, Ordering::Relaxed);
            return None;
        }
        Self::new_inner(host)
    }

    /// Creates GPU Frame Generation on demand when toggled ON from the F10 menu.
    pub(crate) fn try_new_live(host: &Host) -> Option<Self> {
        FG_ENABLED.store(true, Ordering::Relaxed);
        Self::new_inner(host)
    }

    /// Clears temporal frame history when FrameGen is toggled OFF from the F10 menu.
    pub(crate) fn reset_history(&mut self) {
        if let Some(t) = self.targets.as_mut() {
            t.has_prev = false;
        }
        FG_ACTIVE.store(false, Ordering::Relaxed);
        FG_MODE.store(MODE_OFF, Ordering::Relaxed);
    }

    fn new_inner(host: &Host) -> Option<Self> {
        let flow_blob = compile_cs(FLOW_SHADER, s!("cs_flow")).ok()?;
        let interp_blob = compile_cs(INTERP_SHADER, s!("cs_interp")).ok()?;
        // SAFETY: compiled shader blobs live through root signature and PSO creation.
        let made = unsafe {
            let bytes = |b: &ID3DBlob| {
                std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize())
            };
            let code = |b: &ID3DBlob| D3D12_SHADER_BYTECODE {
                pShaderBytecode: b.GetBufferPointer(),
                BytecodeLength: b.GetBufferSize(),
            };
            let flow_root: ID3D12RootSignature =
                host.device.CreateRootSignature(0, bytes(&flow_blob)).ok()?;
            let flow_pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&flow_root),
                    CS: code(&flow_blob),
                    ..Default::default()
                })
                .ok()?;
            let interp_root: ID3D12RootSignature =
                host.device.CreateRootSignature(0, bytes(&interp_blob)).ok()?;
            let interp_pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&interp_root),
                    CS: code(&interp_blob),
                    ..Default::default()
                })
                .ok()?;
            let step = host
                .device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            FrameGen {
                flow_root,
                flow_pso,
                interp_root,
                interp_pso,
                heaps: Vec::new(),
                step,
                targets: None,
            }
        };
        tracing::info!(
            fsr_dll_present = crate::fsr::dll_present(),
            sync = sync_interval(),
            "FSR Frame Generation GPU pipeline ready (2x interpolated output)"
        );
        Some(made)
    }

    /// Picks a UAV-capable VRAM format for `interp_scene` that matches `srv_fmt`
    /// (avoiding `B8G8R8A8_UNORM` where UAV writes aren't universally supported).
    fn uav_format(srv_fmt: DXGI_FORMAT) -> DXGI_FORMAT {
        match srv_fmt {
            DXGI_FORMAT_R16G16B16A16_FLOAT => DXGI_FORMAT_R16G16B16A16_FLOAT,
            DXGI_FORMAT_R10G10B10A2_UNORM => DXGI_FORMAT_R10G10B10A2_UNORM,
            _ => DXGI_FORMAT_R8G8B8A8_UNORM,
        }
    }

    fn ensure_targets(
        &mut self,
        host: &Host,
        width: u32,
        height: u32,
        raw_fmt: DXGI_FORMAT,
        srv_fmt: DXGI_FORMAT,
    ) -> bool {
        if self.targets.as_ref().is_some_and(|t| {
            (t.width, t.height, t.raw_fmt, t.srv_fmt) == (width, height, raw_fmt, srv_fmt)
        }) {
            return true;
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
            // SAFETY: valid default-heap VRAM texture creation.
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
        let flow_w = width.div_ceil(4).max(1);
        let flow_h = height.div_ceil(4).max(1);
        let Some(prev_scene) = make_tex(
            width,
            height,
            raw_fmt,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        ) else {
            return false;
        };
        // Prefer matching `srv_fmt` directly for single-plane CopyResource
        // compatibility (e.g. BGRA8 on NVIDIA), falling back to a universally
        // UAV-capable format otherwise.
        let fallback_fmt = Self::uav_format(srv_fmt);
        let (interp_scene, out_fmt) = if let Some(t) = make_tex(
            width,
            height,
            srv_fmt,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        ) {
            (t, srv_fmt)
        } else if let Some(t) = make_tex(
            width,
            height,
            fallback_fmt,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        ) {
            (t, fallback_fmt)
        } else {
            return false;
        };
        let Some(flow_tex) = make_tex(
            flow_w,
            flow_h,
            DXGI_FORMAT_R16G16_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        ) else {
            return false;
        };
        crate::host::name(&prev_scene, || format!("framegen prev {width}x{height}"));
        crate::host::name(&interp_scene, || format!("framegen interp {width}x{height}"));
        crate::host::name(&flow_tex, || format!("framegen flow {flow_w}x{flow_h}"));
        self.targets = Some(Targets {
            width,
            height,
            raw_fmt,
            srv_fmt,
            out_fmt,
            prev_scene,
            interp_scene,
            flow_tex,
            flow_w,
            flow_h,
            has_prev: false,
        });
        true
    }

    /// Generates an intermediate frame ($t = 0.5$) between the stored previous
    /// scene and `curr_scene` (`curr_state` -> returned to `curr_state`),
    /// returning `(interp_texture, interp_format)` left in
    /// `D3D12_RESOURCE_STATE_UNORDERED_ACCESS`.
    ///
    /// On the very first frame (no previous frame yet), seeds `prev_scene` and
    /// returns `None`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate(
        &mut self,
        host: &Host,
        slot: usize,
        list: &ID3D12GraphicsCommandList,
        curr_scene: &ID3D12Resource,
        curr_state: D3D12_RESOURCE_STATES,
        width: u32,
        height: u32,
        raw_fmt: DXGI_FORMAT,
        srv_fmt: DXGI_FORMAT,
        frame: u64,
    ) -> Option<(ID3D12Resource, DXGI_FORMAT)> {
        if !self.ensure_targets(host, width, height, raw_fmt, srv_fmt) {
            return None;
        }
        // Descriptor layout per slot (7 descriptors):
        //   0: flow SRV t0 (prev_scene)
        //   1: flow SRV t1 (curr_scene)
        //   2: flow UAV u0 (flow_tex)
        //   3: interp SRV t0 (prev_scene)
        //   4: interp SRV t1 (curr_scene)
        //   5: interp SRV t2 (FSR motionVectors OR flow_tex)
        //   6: interp UAV u0 (interp_scene)
        while self.heaps.len() <= slot {
            // SAFETY: valid descriptor heap creation.
            let heap: ID3D12DescriptorHeap = unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: 7,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
                    .ok()?
            };
            self.heaps.push(heap);
        }
        let targets = self.targets.as_mut()?;
        if !targets.has_prev {
            // Seed prev_scene on first frame.
            // SAFETY: open list and live textures of identical raw_fmt & size.
            unsafe {
                let mut pre = Vec::with_capacity(2);
                if curr_state != D3D12_RESOURCE_STATE_COPY_SOURCE {
                    pre.push(transition(curr_scene, curr_state, D3D12_RESOURCE_STATE_COPY_SOURCE));
                }
                pre.push(transition(
                    &targets.prev_scene,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ));
                list.ResourceBarrier(&pre);
                list.CopyResource(&targets.prev_scene, curr_scene);
                let mut post = Vec::with_capacity(2);
                if curr_state != D3D12_RESOURCE_STATE_COPY_SOURCE {
                    post.push(transition(curr_scene, D3D12_RESOURCE_STATE_COPY_SOURCE, curr_state));
                }
                post.push(transition(
                    &targets.prev_scene,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                ));
                list.ResourceBarrier(&post);
            }
            targets.has_prev = true;
            return None;
        }

        let fsr_inputs = crate::fsr::fresh_inputs(frame);
        let fsr_mv = fsr_inputs
            .as_ref()
            .and_then(|i| i.motion_vectors.as_ref())
            .and_then(|mv| {
                // Verify MV format can be viewed as float2 SRV cleanly.
                // SAFETY: live D3D12 resource.
                let d = unsafe { mv.GetDesc() };
                let fmt = match d.Format.0 {
                    16 => DXGI_FORMAT_R32G32_FLOAT,
                    34 => DXGI_FORMAT_R16G16_FLOAT,
                    35 => DXGI_FORMAT_R16G16_UNORM,
                    37 => DXGI_FORMAT_R16G16_SNORM,
                    _ => return None,
                };
                Some((mv.clone(), fmt))
            });

        let use_fsr_mv = fsr_mv.is_some();
        FG_MODE.store(
            if use_fsr_mv { MODE_FSR_MV } else { MODE_OPTFLOW },
            Ordering::Relaxed,
        );

        let heap = &self.heaps[slot];
        let step = self.step as usize;
        // SAFETY: open list, dedicated per-slot heap, live VRAM textures.
        unsafe {
            let cpu = heap.GetCPUDescriptorHandleForHeapStart();
            let gpu = heap.GetGPUDescriptorHandleForHeapStart();
            let cpu_at = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu.ptr + i * step };
            let gpu_at = |i: usize| D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu.ptr + (i * step) as u64,
            };
            let srv_2d = |fmt: DXGI_FORMAT| D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: fmt,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MipLevels: 1,
                        ..Default::default()
                    },
                },
            };
            let uav_2d = |fmt: DXGI_FORMAT| D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: fmt,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                ..Default::default()
            };

            let scene_srv = srv_2d(targets.srv_fmt);
            host.device
                .CreateShaderResourceView(Some(&targets.prev_scene), Some(&scene_srv), cpu_at(0));
            host.device
                .CreateShaderResourceView(Some(curr_scene), Some(&scene_srv), cpu_at(1));
            host.device.CreateUnorderedAccessView(
                Some(&targets.flow_tex),
                None,
                Some(&uav_2d(DXGI_FORMAT_R16G16_FLOAT)),
                cpu_at(2),
            );
            host.device
                .CreateShaderResourceView(Some(&targets.prev_scene), Some(&scene_srv), cpu_at(3));
            host.device
                .CreateShaderResourceView(Some(curr_scene), Some(&scene_srv), cpu_at(4));
            if let Some((ref mv, mv_fmt)) = fsr_mv {
                host.device
                    .CreateShaderResourceView(Some(mv), Some(&srv_2d(mv_fmt)), cpu_at(5));
            } else {
                host.device.CreateShaderResourceView(
                    Some(&targets.flow_tex),
                    Some(&srv_2d(DXGI_FORMAT_R16G16_FLOAT)),
                    cpu_at(5),
                );
            }
            host.device.CreateUnorderedAccessView(
                Some(&targets.interp_scene),
                None,
                Some(&uav_2d(targets.out_fmt)),
                cpu_at(6),
            );

            // Transition curr_scene for compute SRV read.
            if curr_state != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE {
                list.ResourceBarrier(&[transition(
                    curr_scene,
                    curr_state,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )]);
            }

            list.SetDescriptorHeaps(&[Some(heap.clone())]);

            // Pass 1 (when no fresh FSR MVs): GPU optical flow estimation.
            if !use_fsr_mv {
                list.ResourceBarrier(&[transition(
                    &targets.flow_tex,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
                list.SetComputeRootSignature(&self.flow_root);
                list.SetPipelineState(&self.flow_pso);
                let constants = [targets.flow_w, targets.flow_h, width, height];
                list.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                list.SetComputeRootDescriptorTable(1, gpu_at(0));
                list.Dispatch(targets.flow_w.div_ceil(8), targets.flow_h.div_ceil(8), 1);
                list.ResourceBarrier(&[transition(
                    &targets.flow_tex,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )]);
            }

            // Pass 2: bidirectional motion-compensated interpolation into interp_scene.
            list.SetComputeRootSignature(&self.interp_root);
            list.SetPipelineState(&self.interp_pso);
            let mv_scale = fsr_inputs.as_ref().map_or([1.0, 1.0], |i| i.mv_scale);
            let constants = [
                width,
                height,
                u32::from(use_fsr_mv),
                0u32,
                mv_scale[0].to_bits(),
                mv_scale[1].to_bits(),
            ];
            list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            list.SetComputeRootDescriptorTable(1, gpu_at(3));
            list.Dispatch(width.div_ceil(8), height.div_ceil(8), 1);

            // Copy curr_scene -> prev_scene for the next frame and restore curr_scene state.
            list.ResourceBarrier(&[
                transition(
                    curr_scene,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                ),
                transition(
                    &targets.prev_scene,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ),
            ]);
            list.CopyResource(&targets.prev_scene, curr_scene);
            let mut restore = Vec::with_capacity(2);
            restore.push(transition(
                &targets.prev_scene,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            ));
            if curr_state != D3D12_RESOURCE_STATE_COPY_SOURCE {
                restore.push(transition(
                    curr_scene,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    curr_state,
                ));
            }
            list.ResourceBarrier(&restore);
        }

        Some((targets.interp_scene.clone(), targets.out_fmt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_framegen_env_modes() {
        assert_eq!(parse_framegen_config(None), FrameGenConfig::Auto);
        assert_eq!(parse_framegen_config(Some("auto")), FrameGenConfig::Auto);
        assert_eq!(parse_framegen_config(Some("0")), FrameGenConfig::Off);
        assert_eq!(parse_framegen_config(Some("off")), FrameGenConfig::Off);
        assert_eq!(parse_framegen_config(Some("1")), FrameGenConfig::ForceOn);
        assert_eq!(parse_framegen_config(Some("fsr")), FrameGenConfig::ForceOn);
    }

    #[test]
    fn framegen_shaders_compile() {
        assert!(compile_cs(FLOW_SHADER, s!("cs_flow")).is_ok());
        assert!(compile_cs(INTERP_SHADER, s!("cs_interp")).is_ok());
    }
}
