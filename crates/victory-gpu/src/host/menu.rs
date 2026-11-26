//! In-game FSR & Upscaler configuration overlay toggled with `F10`.
//!
//! Available in both `release` and `diag` builds. Renders a self-contained,
//! multi-color pixel-crisp menu directly onto the swap-chain back buffer via
//! `CopyTextureRegion` (zero shader/PSO/descriptor-heap disruption), positioned
//! top-center so it never collides with the top-left `F3` diagnostics panel.
//!
//! Controls (active while the game window has focus):
//! - `F10`: toggle menu open / closed
//! - `Up` / `Down`: select option (Upscaler, Render Quality, FrameGen)
//! - `Left` / `Right` / `Enter`: adjust selected setting live
//! - `Esc`: close menu

use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, VK_DOWN, VK_ESCAPE, VK_F10, VK_LEFT, VK_RETURN, VK_RIGHT, VK_UP,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId};

use super::{Host, HostList, transition};
use crate::fsr::{self, RenderQuality, UpscalerChoice};

const GLYPH: (u32, u32) = (5, 7);
const CELL: (u32, u32) = (6, 10);
const COLUMNS: u32 = 46;
const LINES: u32 = 9;
const RIM: u32 = 6;
const BORDER: u32 = 2;
const UPLOADS: usize = 4;
const STATUS_REFRESH: Duration = Duration::from_millis(250);

const ITEM_COUNT: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Color {
    Bg,
    Border,
    HeaderBg,
    SelectedBg,
    Text,
    Muted,
    Accent,
    Warn,
}

impl Color {
    const fn rgba(self) -> [u8; 4] {
        match self {
            Color::Bg => [14, 20, 32, 255],
            Color::Border => [78, 134, 216, 255],
            Color::HeaderBg => [27, 56, 100, 255],
            Color::SelectedBg => [37, 82, 143, 255],
            Color::Text => [244, 248, 255, 255],
            Color::Muted => [156, 178, 208, 255],
            Color::Accent => [92, 232, 152, 255],
            Color::Warn => [245, 184, 76, 255],
        }
    }
}

/// Encodes an RGBA8 color into the swap chain's exact texel bytes.
fn encode_pixel(format: DXGI_FORMAT, rgba: [u8; 4], out: &mut [u8]) -> Option<usize> {
    let [r, g, b, a] = rgba;
    match format {
        DXGI_FORMAT_R8G8B8A8_UNORM => {
            out[..4].copy_from_slice(&[r, g, b, a]);
            Some(4)
        }
        DXGI_FORMAT_B8G8R8A8_UNORM => {
            out[..4].copy_from_slice(&[b, g, r, a]);
            Some(4)
        }
        DXGI_FORMAT_R10G10B10A2_UNORM => {
            let r10 = (u32::from(r) * 1023 + 127) / 255;
            let g10 = (u32::from(g) * 1023 + 127) / 255;
            let b10 = (u32::from(b) * 1023 + 127) / 255;
            let a2 = u32::from(a >> 6);
            let packed = r10 | (g10 << 10) | (b10 << 20) | (a2 << 30);
            out[..4].copy_from_slice(&packed.to_le_bytes());
            Some(4)
        }
        DXGI_FORMAT_R16G16B16A16_FLOAT => {
            let f16 = |c: u8| f32_to_f16(f32::from(c) / 255.0).to_le_bytes();
            out[0..2].copy_from_slice(&f16(r));
            out[2..4].copy_from_slice(&f16(g));
            out[4..6].copy_from_slice(&f16(b));
            out[6..8].copy_from_slice(&f16(a));
            Some(8)
        }
        _ => None,
    }
}

fn pixel_bytes(format: DXGI_FORMAT) -> Option<usize> {
    match format {
        DXGI_FORMAT_R8G8B8A8_UNORM
        | DXGI_FORMAT_B8G8R8A8_UNORM
        | DXGI_FORMAT_R10G10B10A2_UNORM => Some(4),
        DXGI_FORMAT_R16G16B16A16_FLOAT => Some(8),
        _ => None,
    }
}

/// Minimal IEEE-754 binary32 -> binary16 encoder for normalized [0, 1] UI colors.
fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (bits & 0x007f_ffff) >> 13;
    if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7c00
    } else {
        sign | ((exp as u16) << 10) | (mant as u16)
    }
}

/// 5x7 uppercase bitmap glyphs covering all menu characters.
#[rustfmt::skip]
const fn glyph(c: u8) -> [u8; 7] {
    match c {
        b'0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        b'1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        b'2' => [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        b'3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        b'4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        b'5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        b'6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        b'7' => [0b11111, 0b00001, 0b00010, 0b00010, 0b00100, 0b00100, 0b00100],
        b'8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        b'9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        b'A' => [0b00100, 0b01010, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001],
        b'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        b'C' => [0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111],
        b'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        b'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        b'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        b'G' => [0b01111, 0b10000, 0b10000, 0b10011, 0b10001, 0b10001, 0b01111],
        b'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        b'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        b'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
        b'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        b'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        b'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        b'N' => [0b10001, 0b11001, 0b11001, 0b10101, 0b10011, 0b10011, 0b10001],
        b'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        b'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        b'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        b'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        b'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        b'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        b'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        b'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        b'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        b'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        b'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        b'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        b'.' => [0, 0, 0, 0, 0, 0b01100, 0b01100],
        b'/' => [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
        b'-' => [0, 0, 0, 0b11111, 0, 0, 0],
        b':' => [0, 0b00100, 0b00100, 0, 0b00100, 0b00100, 0],
        b'<' => [0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010],
        b'>' => [0b01000, 0b00100, 0b00010, 0b00001, 0b00010, 0b00100, 0b01000],
        b'[' => [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110],
        b']' => [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110],
        b'(' => [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
        b')' => [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
        b'|' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        b' ' => [0; 7],
        _ => [0b11111, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11111],
    }
}

const fn box_size(scale: u32) -> (u32, u32) {
    (
        (COLUMNS * CELL.0 + 2 * RIM) * scale,
        (LINES * CELL.1 + 2 * RIM) * scale,
    )
}

fn scale_for(height: u32) -> u32 {
    (height / 540).max(1)
}

struct StyledLine {
    text: String,
    bg: Color,
    fg: Color,
}

fn build_lines(selected: usize) -> Vec<StyledLine> {
    let st = fsr::status();
    let upscaler = st.upscaler;
    let quality = st.render_quality;
    let fg_on = st.framegen_enabled;

    let upscaler_val = match upscaler {
        UpscalerChoice::Fsr => "FSR",
        UpscalerChoice::Dlss => {
            if st.dlss_active || st.dlss_available {
                "DLSS"
            } else {
                "DLSS (N/A)"
            }
        }
    };
    let fg_val = if fg_on { "ON" } else { "OFF" };

    let row = |idx: usize, label: &str, value: &str, value_accent: Color| -> StyledLine {
        let is_sel = selected == idx;
        let mark = if is_sel { '>' } else { ' ' };
        StyledLine {
            text: format!("{mark} {label:<18} < {value} >"),
            bg: if is_sel { Color::SelectedBg } else { Color::Bg },
            fg: if is_sel { value_accent } else { Color::Text },
        }
    };

    let fsr_state = if st.fsr_active {
        "ACTIVE"
    } else if st.fsr_dll_loaded {
        "LOADED"
    } else if st.fsr_dll_present {
        "READY"
    } else {
        "NO-DLL"
    };
    let fg_state = if st.framegen_active {
        st.framegen_mode
    } else if st.framegen_enabled {
        "ARMED"
    } else {
        "OFF"
    };

    vec![
        StyledLine {
            text: "  VICTORY FSR / UPSCALER MENU      [F10]".to_owned(),
            bg: Color::HeaderBg,
            fg: Color::Text,
        },
        StyledLine {
            text: String::new(),
            bg: Color::Bg,
            fg: Color::Muted,
        },
        row(
            0,
            "1. UPSCALER",
            upscaler_val,
            if upscaler == UpscalerChoice::Dlss && !st.dlss_available && !st.dlss_active {
                Color::Warn
            } else {
                Color::Accent
            },
        ),
        row(1, "2. RENDER QUALITY", quality.label(), Color::Accent),
        row(
            2,
            "3. FRAMEGEN",
            fg_val,
            if fg_on { Color::Accent } else { Color::Warn },
        ),
        StyledLine {
            text: String::new(),
            bg: Color::Bg,
            fg: Color::Muted,
        },
        StyledLine {
            text: format!(
                "  STATUS: FSR {fsr_state} | FG {fg_state} | SHD {}X",
                st.shadow_scale * st.shadow_scale
            ),
            bg: Color::Bg,
            fg: Color::Muted,
        },
        StyledLine {
            text: "  UP/DOWN: SELECT   LEFT/RIGHT: CHANGE".to_owned(),
            bg: Color::Bg,
            fg: Color::Muted,
        },
        StyledLine {
            text: "  ENTER: CYCLE      F10 / ESC: CLOSE".to_owned(),
            bg: Color::Bg,
            fg: Color::Muted,
        },
    ]
}

fn pixel_color(lines: &[StyledLine], x: u32, y: u32, w: u32, h: u32) -> Color {
    if x < BORDER || y < BORDER || x >= w - BORDER || y >= h - BORDER {
        return Color::Border;
    }
    let Some(iy) = y.checked_sub(RIM) else {
        return Color::HeaderBg;
    };
    let row_idx = (iy / CELL.1) as usize;
    let Some(line) = lines.get(row_idx) else {
        return Color::Bg;
    };
    let Some(ix) = x.checked_sub(RIM) else {
        return line.bg;
    };
    let col = ix / CELL.0;
    let gx = ix % CELL.0;
    let gy = iy % CELL.1;
    // Vertically center the 7px glyph inside the 10px cell (1px top pad).
    if col < COLUMNS
        && gx < GLYPH.0
        && gy >= 1
        && gy < 1 + GLYPH.1
        && let Some(&ch) = line.text.as_bytes().get(col as usize)
    {
        let bits = glyph(ch.to_ascii_uppercase())[(gy - 1) as usize];
        if (bits >> (GLYPH.0 - 1 - gx)) & 1 != 0 {
            return line.fg;
        }
    }
    line.bg
}

fn paint(lines: &[StyledLine], scale: u32, format: DXGI_FORMAT, pitch: usize) -> Option<Vec<u8>> {
    let bpp = pixel_bytes(format)?;
    let (width, height) = box_size(scale);
    let logical_w = width / scale;
    let logical_h = height / scale;
    let mut out = vec![0u8; pitch * height as usize];
    let mut enc = [0u8; 8];
    for ly in 0..logical_h {
        let top = pitch * (ly * scale) as usize;
        for lx in 0..logical_w {
            let color = pixel_color(lines, lx, ly, logical_w, logical_h);
            let n = encode_pixel(format, color.rgba(), &mut enc)?;
            debug_assert_eq!(n, bpp);
            for sx in 0..scale {
                let at = top + (lx * scale + sx) as usize * bpp;
                out[at..at + bpp].copy_from_slice(&enc[..bpp]);
            }
        }
        for sy in 1..scale as usize {
            out.copy_within(top..top + pitch, top + sy * pitch);
        }
    }
    Some(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MenuAction {
    ToggleOpen,
    Close,
    Up,
    Down,
    Left,
    Right,
}

#[derive(Default)]
struct KeyState {
    f10: bool,
    up: bool,
    down: bool,
    left: bool,
    right: bool,
    enter: bool,
    esc: bool,
}

impl KeyState {
    fn edge(slot: &mut bool, vk: u16) -> bool {
        // SAFETY: plain Win32 keyboard query.
        let down = unsafe { GetAsyncKeyState(i32::from(vk)) } < 0;
        let rising = down && !*slot;
        *slot = down;
        rising
    }

    fn foreground_is_us() -> bool {
        let mut pid = 0u32;
        // SAFETY: plain Win32 query into local u32.
        unsafe { GetWindowThreadProcessId(GetForegroundWindow(), &mut pid) };
        pid == std::process::id()
    }

    fn poll(&mut self, menu_open: bool) -> Vec<MenuAction> {
        let mut actions = Vec::new();
        if !Self::foreground_is_us() {
            *self = KeyState::default();
            return actions;
        }
        if Self::edge(&mut self.f10, VK_F10) {
            actions.push(MenuAction::ToggleOpen);
        }
        if !menu_open {
            // Keep edge trackers synchronized so keys held before opening
            // do not immediately fire on the frame the menu opens.
            let _ = Self::edge(&mut self.up, VK_UP);
            let _ = Self::edge(&mut self.down, VK_DOWN);
            let _ = Self::edge(&mut self.left, VK_LEFT);
            let _ = Self::edge(&mut self.right, VK_RIGHT);
            let _ = Self::edge(&mut self.enter, VK_RETURN);
            let _ = Self::edge(&mut self.esc, VK_ESCAPE);
            return actions;
        }
        if Self::edge(&mut self.esc, VK_ESCAPE) {
            actions.push(MenuAction::Close);
        }
        if Self::edge(&mut self.up, VK_UP) {
            actions.push(MenuAction::Up);
        }
        if Self::edge(&mut self.down, VK_DOWN) {
            actions.push(MenuAction::Down);
        }
        if Self::edge(&mut self.left, VK_LEFT) {
            actions.push(MenuAction::Left);
        }
        if Self::edge(&mut self.right, VK_RIGHT) || Self::edge(&mut self.enter, VK_RETURN) {
            actions.push(MenuAction::Right);
        }
        actions
    }
}

struct Buffers {
    format: DXGI_FORMAT,
    scale: u32,
    pitch: u32,
    uploads: Vec<(ID3D12Resource, *mut u8)>,
    card: ID3D12Resource,
    turn: usize,
    fresh: bool,
}

// SAFETY: D3D12 resources are free-threaded; CPU upload writes happen under Presenter lock.
unsafe impl Send for Buffers {}

impl Buffers {
    fn new(host: &Host, format: DXGI_FORMAT, scale: u32) -> Option<Self> {
        let bpp = pixel_bytes(format)? as u32;
        let (width, height) = box_size(scale);
        let pitch = (width * bpp).next_multiple_of(256);
        let bytes = u64::from(pitch) * u64::from(height);
        let uploads = (0..UPLOADS)
            .map(|_| host.upload_buffer(bytes))
            .collect::<Option<Vec<_>>>()?;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut card: Option<ID3D12Resource> = None;
        // SAFETY: valid default-heap buffer creation.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut card,
                )
                .ok()?;
        }
        Some(Buffers {
            format,
            scale,
            pitch,
            uploads,
            card: card?,
            turn: 0,
            fresh: false,
        })
    }

    fn write(&mut self, lines: &[StyledLine]) {
        let Some(picture) = paint(lines, self.scale, self.format, self.pitch as usize) else {
            return;
        };
        self.turn = (self.turn + 1) % UPLOADS;
        // SAFETY: upload slot is sized for `pitch * height` bytes and not read by active GPU lists.
        unsafe {
            std::ptr::copy_nonoverlapping(
                picture.as_ptr(),
                self.uploads[self.turn].1,
                picture.len(),
            );
        }
        self.fresh = true;
    }
}

/// Presenter-owned in-game `F10` menu state and GPU overlay.
pub(super) struct Menu {
    open: bool,
    selected: usize,
    dirty: bool,
    last_refresh: Instant,
    keys: KeyState,
    buffers: Option<Buffers>,
    failed: bool,
}

impl Menu {
    pub(super) fn new() -> Self {
        Menu {
            open: false,
            selected: 0,
            dirty: true,
            last_refresh: Instant::now(),
            keys: KeyState::default(),
            buffers: None,
            failed: false,
        }
    }

    pub(super) fn poll(&mut self) {
        let actions = self.keys.poll(self.open);
        for action in actions {
            self.apply_action(action);
        }
        if self.open && self.last_refresh.elapsed() >= STATUS_REFRESH {
            self.dirty = true;
            self.last_refresh = Instant::now();
        }
    }

    fn apply_action(&mut self, action: MenuAction) {
        match action {
            MenuAction::ToggleOpen => {
                self.open = !self.open;
                self.dirty = true;
                self.last_refresh = Instant::now();
                tracing::info!(open = self.open, "in-game FSR menu (F10)");
            }
            MenuAction::Close => {
                if self.open {
                    self.open = false;
                    tracing::info!("in-game FSR menu closed");
                }
            }
            MenuAction::Up => {
                self.selected = (self.selected + ITEM_COUNT - 1) % ITEM_COUNT;
                self.dirty = true;
            }
            MenuAction::Down => {
                self.selected = (self.selected + 1) % ITEM_COUNT;
                self.dirty = true;
            }
            MenuAction::Left => {
                self.adjust_selected(false);
                self.dirty = true;
            }
            MenuAction::Right => {
                self.adjust_selected(true);
                self.dirty = true;
            }
        }
    }

    fn adjust_selected(&self, forward: bool) {
        match self.selected {
            0 => {
                let next = fsr::upscaler_choice().toggle();
                fsr::set_upscaler_choice(next);
            }
            1 => {
                let cur = fsr::render_quality();
                let next = if forward { cur.next() } else { cur.prev() };
                fsr::set_render_quality(next);
            }
            2 => {
                let next = !fsr::framegen_enabled_live();
                fsr::set_framegen_enabled_live(next);
            }
            _ => {}
        }
    }

    /// Copies the F10 menu onto `back` (which must be in `COPY_DEST` state)
    /// when open.
    pub(super) fn draw(
        &mut self,
        host: &Host,
        list: &HostList,
        back: &ID3D12Resource,
        (width, height): (u32, u32),
        format: DXGI_FORMAT,
    ) {
        if !self.open || self.failed {
            return;
        }
        #[cfg(feature = "diagnostics")]
        let _ours = crate::replay::record::unrecorded();

        let scale = scale_for(height);
        if self
            .buffers
            .as_ref()
            .is_none_or(|b| (b.format, b.scale) != (format, scale))
        {
            std::mem::forget(self.buffers.take());
            self.buffers = Buffers::new(host, format, scale);
            self.failed = self.buffers.is_none();
            self.dirty = true;
        }
        let Some(buffers) = &mut self.buffers else {
            return;
        };
        if std::mem::take(&mut self.dirty) {
            let lines = build_lines(self.selected);
            buffers.write(&lines);
        }
        let (w, h) = box_size(scale);
        let copy_w = w.min(width);
        let copy_h = h.min(height);
        let dst_x = width.saturating_sub(copy_w) / 2;
        let dst_y = (height / 12).min(height.saturating_sub(copy_h));
        let cmd = &list.list;
        // SAFETY: open command list, live swap-chain back buffer in COPY_DEST, valid box bounds.
        unsafe {
            if std::mem::take(&mut buffers.fresh) {
                cmd.CopyResource(&buffers.card, &buffers.uploads[buffers.turn].0);
                cmd.ResourceBarrier(&[transition(
                    &buffers.card,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
            }
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&buffers.card),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: format,
                            Width: w,
                            Height: h,
                            Depth: 1,
                            RowPitch: buffers.pitch,
                        },
                    },
                },
            };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(back),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            let part = D3D12_BOX {
                left: 0,
                top: 0,
                front: 0,
                right: copy_w,
                bottom: copy_h,
                back: 1,
            };
            cmd.CopyTextureRegion(&to, dst_x, dst_y, 0, &from, Some(&part));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn menu_glyphs_are_valid_and_distinct() {
        let used: Vec<u8> = (b'0'..=b'9')
            .chain(b'A'..=b'Z')
            .chain(*b"./-:<>[]()| ")
            .collect();
        for (i, a) in used.iter().enumerate() {
            assert_ne!(glyph(*a), glyph(b'#'), "{} missing glyph", *a as char);
            assert!(glyph(*a).iter().all(|row| row >> GLYPH.0 == 0));
            for b in &used[i + 1..] {
                assert_ne!(glyph(*a), glyph(*b), "{} == {}", *a as char, *b as char);
            }
        }
    }

    #[test]
    fn menu_navigation_updates_live_settings() {
        fsr::set_upscaler_choice(UpscalerChoice::Fsr);
        fsr::set_render_quality(RenderQuality::Native);
        fsr::set_framegen_enabled_live(true);

        let mut menu = Menu::new();
        assert!(!menu.open);
        menu.apply_action(MenuAction::ToggleOpen);
        assert!(menu.open);
        assert_eq!(menu.selected, 0);

        // Row 0: Upscaler FSR <-> DLSS
        menu.apply_action(MenuAction::Right);
        assert_eq!(fsr::upscaler_choice(), UpscalerChoice::Dlss);
        menu.apply_action(MenuAction::Left);
        assert_eq!(fsr::upscaler_choice(), UpscalerChoice::Fsr);

        // Row 1: Render Quality Native -> Quality -> Performance -> UltraPerformance -> Native
        menu.apply_action(MenuAction::Down);
        assert_eq!(menu.selected, 1);
        menu.apply_action(MenuAction::Right);
        assert_eq!(fsr::render_quality(), RenderQuality::Quality);
        menu.apply_action(MenuAction::Right);
        assert_eq!(fsr::render_quality(), RenderQuality::Performance);
        menu.apply_action(MenuAction::Right);
        assert_eq!(fsr::render_quality(), RenderQuality::UltraPerformance);
        menu.apply_action(MenuAction::Right);
        assert_eq!(fsr::render_quality(), RenderQuality::Native);

        // Row 2: FrameGen ON <-> OFF
        menu.apply_action(MenuAction::Down);
        assert_eq!(menu.selected, 2);
        menu.apply_action(MenuAction::Right);
        assert!(!fsr::framegen_enabled_live());
        menu.apply_action(MenuAction::Left);
        assert!(fsr::framegen_enabled_live());

        menu.apply_action(MenuAction::Close);
        assert!(!menu.open);
    }

    #[test]
    fn menu_paints_for_all_swapchain_formats() {
        let lines = build_lines(0);
        for fmt in [
            DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_FORMAT_B8G8R8A8_UNORM,
            DXGI_FORMAT_R10G10B10A2_UNORM,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
        ] {
            let (w, h) = box_size(1);
            let bpp = pixel_bytes(fmt).unwrap();
            let pitch = (w as usize * bpp).next_multiple_of(256);
            let buf = paint(&lines, 1, fmt, pitch).unwrap();
            assert_eq!(buf.len(), pitch * h as usize);
        }
    }
}
