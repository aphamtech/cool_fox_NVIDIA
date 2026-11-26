//! Native Win32 progress window for shader precompilation (`precompile`).
//!
//! Win32 windows belong to the thread that created them, so the window runs
//! on its own lightweight UI thread. Worker threads only bump lock-free
//! atomic counters (and occasionally update the status label), while a
//! 60 ms timer double-buffers and repaints the window without flicker.
//!
//! Closing the window via `[X]` / `Alt+F4` sets a cancellation flag so
//! `victory run --execute --precompile` can stop scanning immediately and
//! launch the title right away (remaining shaders translate on demand).
//! Without a desktop session (or on non-Windows targets), no window is
//! created and precompilation proceeds normally in headless mode.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

pub(crate) struct State {
    pub(crate) containers: AtomicU64,
    pub(crate) translated: AtomicU64,
    pub(crate) cached: AtomicU64,
    pub(crate) failed: AtomicU64,
    pub(crate) files_done: AtomicU64,
    pub(crate) files_total: AtomicU64,
    pub(crate) bytes_done: AtomicU64,
    pub(crate) bytes_total: AtomicU64,
    pub(crate) cancelled: AtomicBool,
    pub(crate) allow_skip: bool,
    pub(crate) status: Mutex<String>,
}

impl State {
    fn new(allow_skip: bool) -> Self {
        Self {
            containers: AtomicU64::new(0),
            translated: AtomicU64::new(0),
            cached: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            files_done: AtomicU64::new(0),
            files_total: AtomicU64::new(0),
            bytes_done: AtomicU64::new(0),
            bytes_total: AtomicU64::new(0),
            cancelled: AtomicBool::new(false),
            allow_skip,
            status: Mutex::new("Scanning game files...".to_owned()),
        }
    }

    pub(crate) fn fraction(&self) -> Option<f64> {
        let total = self.bytes_total.load(Ordering::Relaxed);
        if total == 0 {
            return None;
        }
        let done = self.bytes_done.load(Ordering::Relaxed).min(total);
        Some(done as f64 / total as f64)
    }

    pub(crate) fn status_text(&self) -> String {
        self.status
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

/// A progress window displaying shader precompilation counts, alive until dropped.
pub struct Progress {
    state: Arc<State>,
    /// Held for the window's lifetime: its `Drop` closes the window and joins its thread.
    #[cfg(windows)]
    _window: Option<imp::Window>,
}

/// Lock-free handle to a [`Progress`] window's counters for worker threads.
#[derive(Clone)]
pub struct Counter(Arc<State>);

impl Counter {
    /// Records one shader container:
    /// - `Some(true)`: served from the on-disk shader cache.
    /// - `Some(false)`: newly translated and stored in the cache.
    /// - `None`: failed to translate.
    pub fn container(&self, outcome: Option<bool>) {
        self.0.containers.fetch_add(1, Ordering::Relaxed);
        let counter = match outcome {
            Some(true) => &self.0.cached,
            Some(false) => &self.0.translated,
            None => &self.0.failed,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Advances the determinate progress bar by `bytes` processed.
    pub fn add_bytes(&self, bytes: u64) {
        if bytes > 0 {
            self.0.bytes_done.fetch_add(bytes, Ordering::Relaxed);
        }
    }

    /// Returns `true` if the user closed the progress window to cancel/skip.
    pub fn is_cancelled(&self) -> bool {
        self.0.cancelled.load(Ordering::Relaxed)
    }
}

impl Progress {
    /// Opens the progress window; if no desktop session is available, runs headless.
    pub fn new(title: &str, allow_skip: bool) -> Self {
        let state = Arc::new(State::new(allow_skip));
        #[cfg(windows)]
        let window = imp::Window::new(title, state.clone(), std::time::Instant::now());
        #[cfg(not(windows))]
        let _ = title;
        Self {
            state,
            #[cfg(windows)]
            _window: window,
        }
    }

    /// Returns a cloneable counter handle for worker threads.
    pub fn counter(&self) -> Counter {
        Counter(self.state.clone())
    }

    /// Sets the discovered file count and total byte size for determinate progress.
    pub fn set_totals(&self, files: u64, bytes: u64) {
        self.state.files_total.store(files, Ordering::Relaxed);
        self.state.bytes_total.store(bytes, Ordering::Relaxed);
    }

    /// Updates the current phase/archive status line shown in the window header.
    pub fn set_status(&self, text: impl Into<String>) {
        *self
            .state
            .status
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = text.into();
    }

    /// Advances the determinate byte counter (for skipped/empty files or remainders).
    pub fn add_bytes(&self, bytes: u64) {
        if bytes > 0 {
            self.state.bytes_done.fetch_add(bytes, Ordering::Relaxed);
        }
    }

    /// Marks one candidate file finished.
    pub fn finish_file(&self) {
        self.state.files_done.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns `true` if the user closed the progress window.
    pub fn is_cancelled(&self) -> bool {
        self.state.cancelled.load(Ordering::Relaxed)
    }
}

pub(crate) fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(' ');
        }
        out.push(c);
    }
    out
}

pub(crate) fn clock(seconds: f64) -> String {
    let s = seconds.max(0.0) as u64;
    if s >= 3600 {
        format!("{} h {} min", s / 3600, s % 3600 / 60)
    } else if s >= 60 {
        format!("{} min {} s", s / 60, s % 60)
    } else {
        format!("{s} s")
    }
}

pub(crate) fn format_lines(state: &State, elapsed: f64) -> [String; 3] {
    let load = |n: &AtomicU64| n.load(Ordering::Relaxed);
    let containers = load(&state.containers);
    let translated = load(&state.translated);
    let cached = load(&state.cached);
    let failed = load(&state.failed);
    let status = state.status_text();

    let header = match state.fraction() {
        Some(frac) => format!("{status}  ({:.0}%)", (frac * 100.0).clamp(0.0, 100.0)),
        None => status,
    };

    let stats = format!(
        "{} shaders   •   {} compiled   •   {} cached   •   {} failed",
        thousands(containers),
        thousands(translated),
        thousands(cached),
        thousands(failed),
    );

    let rate = if elapsed > 0.25 {
        containers as f64 / elapsed
    } else {
        0.0
    };
    let footer = if state.allow_skip {
        format!(
            "{} / s   •   {}   •   Close window to skip & launch",
            thousands(rate as u64),
            clock(elapsed),
        )
    } else {
        format!("{} / s   •   {}", thousands(rate as u64), clock(elapsed))
    };

    [header, stats, footer]
}

#[cfg(windows)]
mod imp {
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
    use windows_sys::Win32::Graphics::Gdi::{
        BeginPaint, BitBlt, CLEARTYPE_QUALITY, CLIP_DEFAULT_PRECIS, CreateCompatibleBitmap,
        CreateCompatibleDC, CreateFontW, CreateSolidBrush, DEFAULT_CHARSET, DEFAULT_GUI_FONT,
        DEFAULT_PITCH, DT_END_ELLIPSIS, DT_NOPREFIX, DT_SINGLELINE, DT_VCENTER, DeleteDC,
        DeleteObject, DrawTextW, EndPaint, FF_SWISS, FW_NORMAL, FW_SEMIBOLD, FillRect,
        GetStockObject, HFONT, InvalidateRect, OUT_DEFAULT_PRECIS, PAINTSTRUCT, SRCCOPY,
        SelectObject, SetBkMode, SetTextColor, TRANSPARENT, UpdateWindow,
    };
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        CW_USEDEFAULT, CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW,
        GetClientRect, GetMessageW, GetSystemMetrics, IDC_ARROW, KillTimer, LoadCursorW, MSG,
        PostMessageW, PostQuitMessage, RegisterClassW, SM_CXSCREEN, SM_CYSCREEN, SW_SHOWNORMAL,
        SetTimer, ShowWindow, TranslateMessage, WM_APP, WM_CLOSE, WM_DESTROY, WM_ERASEBKGND,
        WM_PAINT, WM_TIMER, WNDCLASSW, WS_CAPTION, WS_MINIMIZEBOX, WS_OVERLAPPED, WS_SYSMENU,
        WS_VISIBLE,
    };

    use super::{State, format_lines};

    const TIMER: usize = 1;
    const TIMER_MS: u32 = 60;
    /// Internal message sent by [`Window::drop`] so programmatic teardown
    /// does not set the user-cancellation flag.
    const WM_APP_CLOSE: u32 = WM_APP + 1;

    const WIDTH: i32 = 560;
    const HEIGHT: i32 = 168;

    const BACKGROUND: u32 = rgb(24, 26, 31);
    const HEADER_TEXT: u32 = rgb(238, 241, 246);
    const BODY_TEXT: u32 = rgb(203, 210, 222);
    const DIM_TEXT: u32 = rgb(139, 148, 165);
    const TRACK: u32 = rgb(44, 48, 59);
    const TRACK_BORDER: u32 = rgb(62, 68, 82);
    const FILL: u32 = rgb(90, 170, 250);
    const FILL_HIGHLIGHT: u32 = rgb(134, 198, 255);

    const fn rgb(r: u8, g: u8, b: u8) -> u32 {
        (r as u32) | ((g as u32) << 8) | ((b as u32) << 16)
    }

    /// The window handle, passed back only to Win32 `PostMessageW`.
    #[derive(Clone, Copy)]
    struct Handle(HWND);

    // SAFETY: `HWND` is an opaque OS handle; only `PostMessageW` is called
    // from another thread, which Win32 explicitly guarantees is thread-safe.
    unsafe impl Send for Handle {}
    // SAFETY: No Rust state behind the handle is accessed across threads.
    unsafe impl Sync for Handle {}

    struct Showing {
        state: Arc<State>,
        started: Instant,
        closed: Arc<AtomicBool>,
    }

    /// Active progress state painted by `wndproc`. Replaced cleanly on each
    /// new window so sequential windows (e.g. in tests) never show stale data.
    static SHOWING: Mutex<Option<Showing>> = Mutex::new(None);

    struct Fonts {
        heading: HFONT,
        body: HFONT,
        small: HFONT,
        owned: bool,
    }

    impl Fonts {
        fn create() -> Self {
            let face = wide("Segoe UI");
            let make = |height: i32, weight: u32| -> HFONT {
                // SAFETY: `CreateFontW` takes a valid NUL-terminated UTF-16
                // face name and returns either a valid `HFONT` or null.
                unsafe {
                    CreateFontW(
                        height,
                        0,
                        0,
                        0,
                        weight as i32,
                        0,
                        0,
                        0,
                        u32::from(DEFAULT_CHARSET),
                        u32::from(OUT_DEFAULT_PRECIS),
                        u32::from(CLIP_DEFAULT_PRECIS),
                        u32::from(CLEARTYPE_QUALITY),
                        u32::from(DEFAULT_PITCH | FF_SWISS),
                        face.as_ptr(),
                    )
                }
            };
            let heading = make(-15, FW_SEMIBOLD);
            let body = make(-13, FW_NORMAL);
            let small = make(-12, FW_NORMAL);
            if !heading.is_null() && !body.is_null() && !small.is_null() {
                Self {
                    heading,
                    body,
                    small,
                    owned: true,
                }
            } else {
                for f in [heading, body, small] {
                    if !f.is_null() {
                        // SAFETY: Each non-null handle was returned by `CreateFontW` above.
                        unsafe { DeleteObject(f) };
                    }
                }
                // SAFETY: `GetStockObject(DEFAULT_GUI_FONT)` returns a process-owned stock object.
                let fallback = unsafe { GetStockObject(DEFAULT_GUI_FONT) } as HFONT;
                Self {
                    heading: fallback,
                    body: fallback,
                    small: fallback,
                    owned: false,
                }
            }
        }
    }

    impl Drop for Fonts {
        fn drop(&mut self) {
            if self.owned {
                for f in [self.heading, self.body, self.small] {
                    if !f.is_null() {
                        // SAFETY: Each font was created with `CreateFontW` and is
                        // no longer selected into any DC after the message loop ends.
                        unsafe { DeleteObject(f) };
                    }
                }
            }
        }
    }

    thread_local! {
        static UI_FONTS: RefCell<Option<Fonts>> = const { RefCell::new(None) };
    }

    /// The progress window's thread and handle, kept by [`super::Progress`].
    pub(super) struct Window {
        hwnd: Handle,
        closed: Arc<AtomicBool>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl Window {
        pub(super) fn new(title: &str, state: Arc<State>, started: Instant) -> Option<Self> {
            let closed = Arc::new(AtomicBool::new(false));
            *SHOWING.lock().unwrap_or_else(|e| e.into_inner()) = Some(Showing {
                state,
                started,
                closed: closed.clone(),
            });
            let (wanted, made) = std::sync::mpsc::channel::<Handle>();
            let title = wide(title);
            let thread = std::thread::Builder::new()
                .name("victory-progress".into())
                .spawn(move || run(&title, &wanted))
                .ok()?;
            let hwnd = made.recv().ok()?;
            if hwnd.0.is_null() {
                let _ = thread.join();
                *SHOWING.lock().unwrap_or_else(|e| e.into_inner()) = None;
                return None;
            }
            Some(Window {
                hwnd,
                closed,
                thread: Some(thread),
            })
        }
    }

    impl Drop for Window {
        fn drop(&mut self) {
            if !self.closed.load(Ordering::Acquire) {
                // SAFETY: `self.hwnd.0` is our window and `closed` is still false;
                // even if it races with `WM_DESTROY`, Win32 safely rejects a stale
                // HWND in `PostMessageW` for a custom `WM_APP_CLOSE` message.
                unsafe { PostMessageW(self.hwnd.0, WM_APP_CLOSE, 0, 0) };
            }
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
            *SHOWING.lock().unwrap_or_else(|e| e.into_inner()) = None;
        }
    }

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain([0]).collect()
    }

    fn centered_origin() -> (i32, i32) {
        // SAFETY: `GetSystemMetrics` is a pure Win32 query with constant indices.
        let (sw, sh) = unsafe { (GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN)) };
        if sw > WIDTH && sh > HEIGHT {
            ((sw - WIDTH) / 2, (sh - HEIGHT) / 2)
        } else {
            (CW_USEDEFAULT, CW_USEDEFAULT)
        }
    }

    fn run(title: &[u16], made: &std::sync::mpsc::Sender<Handle>) {
        // SAFETY: Registers our window class, creates the progress window on
        // this thread, pumps its message loop, and tears down thread-local GDI
        // fonts after `GetMessageW` exits.
        unsafe {
            let instance = GetModuleHandleW(std::ptr::null());
            let class = wide("VictoryProgress");
            let wc = WNDCLASSW {
                style: 0,
                lpfnWndProc: Some(wndproc),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: instance,
                hIcon: std::ptr::null_mut(),
                hCursor: LoadCursorW(std::ptr::null_mut(), IDC_ARROW),
                hbrBackground: std::ptr::null_mut(),
                lpszMenuName: std::ptr::null(),
                lpszClassName: class.as_ptr(),
            };
            RegisterClassW(&wc);

            let (x, y) = centered_origin();
            UI_FONTS.with(|slot| *slot.borrow_mut() = Some(Fonts::create()));

            let hwnd = CreateWindowExW(
                0,
                class.as_ptr(),
                title.as_ptr(),
                WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_VISIBLE,
                x,
                y,
                WIDTH,
                HEIGHT,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                instance,
                std::ptr::null_mut(),
            );
            if hwnd.is_null() {
                UI_FONTS.with(|slot| *slot.borrow_mut() = None);
                let _ = made.send(Handle(hwnd));
                return;
            }
            let _ = made.send(Handle(hwnd));
            ShowWindow(hwnd, SW_SHOWNORMAL);
            UpdateWindow(hwnd);
            SetTimer(hwnd, TIMER, TIMER_MS, None);
            let mut msg = MSG::default();
            while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            UI_FONTS.with(|slot| *slot.borrow_mut() = None);
        }
    }

    unsafe extern "system" fn wndproc(
        hwnd: HWND,
        msg: u32,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        match msg {
            WM_ERASEBKGND => 1,
            WM_TIMER => {
                // SAFETY: Repaints our own window without erasing background.
                unsafe { InvalidateRect(hwnd, std::ptr::null(), 0) };
                0
            }
            WM_PAINT => {
                // SAFETY: Called from `WM_PAINT` on the window thread.
                unsafe { paint(hwnd) };
                0
            }
            WM_CLOSE => {
                if let Some(showing) = SHOWING.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
                    showing.state.cancelled.store(true, Ordering::Relaxed);
                }
                // SAFETY: Destroys our own window on the window thread.
                unsafe { DestroyWindow(hwnd) };
                0
            }
            WM_APP_CLOSE => {
                // SAFETY: Programmatic close from `Window::drop`; destroys our own window.
                unsafe { DestroyWindow(hwnd) };
                0
            }
            WM_DESTROY => {
                if let Some(showing) = SHOWING.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
                    showing.closed.store(true, Ordering::Release);
                }
                // SAFETY: Kills our timer and exits this thread's message loop.
                unsafe {
                    KillTimer(hwnd, TIMER);
                    PostQuitMessage(0);
                }
                0
            }
            // SAFETY: Standard fallback for unhandled window messages.
            _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
        }
    }

    fn snapshot() -> ([String; 3], Option<f64>, f64) {
        let guard = SHOWING.lock().unwrap_or_else(|e| e.into_inner());
        let Some(showing) = guard.as_ref() else {
            return (
                [
                    "Precompiling shaders...".to_owned(),
                    String::new(),
                    String::new(),
                ],
                None,
                0.0,
            );
        };
        let elapsed = showing.started.elapsed().as_secs_f64();
        let lines = format_lines(&showing.state, elapsed);
        let fraction = showing.state.fraction();
        (lines, fraction, elapsed)
    }

    /// Paints the window double-buffered into a memory DC to eliminate flicker.
    ///
    /// # Safety
    /// Must only be called from `WM_PAINT` on the window's owning thread.
    unsafe fn paint(hwnd: HWND) {
        // SAFETY: All GDI objects (`hdc`, `mem_dc`, `bitmap`, brushes) are
        // created on this thread, paired with restore/delete calls, and
        // bracketed by `BeginPaint`/`EndPaint`.
        unsafe {
            let mut ps = PAINTSTRUCT::default();
            let hdc = BeginPaint(hwnd, &mut ps);
            let mut client = RECT::default();
            GetClientRect(hwnd, &mut client);
            let w = (client.right - client.left).max(1);
            let h = (client.bottom - client.top).max(1);

            let mem_dc = CreateCompatibleDC(hdc);
            let bitmap = CreateCompatibleBitmap(hdc, w, h);
            let use_mem = !mem_dc.is_null() && !bitmap.is_null();
            let (target_dc, old_bmp) = if use_mem {
                (mem_dc, SelectObject(mem_dc, bitmap))
            } else {
                (hdc, std::ptr::null_mut())
            };

            let bg = CreateSolidBrush(BACKGROUND);
            FillRect(target_dc, &client, bg);
            DeleteObject(bg);

            SetBkMode(target_dc, TRANSPARENT as i32);
            let (lines, fraction, elapsed) = snapshot();

            UI_FONTS.with(|slot| {
                let fonts = slot.borrow();
                let fallback = GetStockObject(DEFAULT_GUI_FONT);
                let (f0, f1, f2) = match fonts.as_ref() {
                    Some(f) => (f.heading, f.body, f.small),
                    None => (fallback as HFONT, fallback as HFONT, fallback as HFONT),
                };
                let styles = [
                    (f0, HEADER_TEXT, 14, 24),
                    (f1, BODY_TEXT, 44, 22),
                    (f2, DIM_TEXT, 70, 20),
                ];
                let old_font = SelectObject(target_dc, f0);
                for (line, (font, color, top, height)) in lines.iter().zip(styles) {
                    SelectObject(target_dc, font);
                    SetTextColor(target_dc, color);
                    let mut rect = RECT {
                        left: 18,
                        top,
                        right: client.right - 18,
                        bottom: top + height,
                    };
                    let words: Vec<u16> = line.encode_utf16().collect();
                    DrawTextW(
                        target_dc,
                        words.as_ptr(),
                        words.len() as i32,
                        &mut rect,
                        DT_SINGLELINE | DT_VCENTER | DT_END_ELLIPSIS | DT_NOPREFIX,
                    );
                }
                SelectObject(target_dc, old_font);
            });

            // Progress bar track with subtle 1 px border.
            let bar_top = client.bottom - 30;
            let bar_bottom = bar_top + 12;
            let left = 18;
            let right = (client.right - 18).max(left + 4);

            let border_brush = CreateSolidBrush(TRACK_BORDER);
            FillRect(
                target_dc,
                &RECT {
                    left: left - 1,
                    top: bar_top - 1,
                    right: right + 1,
                    bottom: bar_bottom + 1,
                },
                border_brush,
            );
            DeleteObject(border_brush);

            let track = CreateSolidBrush(TRACK);
            FillRect(
                target_dc,
                &RECT {
                    left,
                    top: bar_top,
                    right,
                    bottom: bar_bottom,
                },
                track,
            );
            DeleteObject(track);

            let span = (right - left) as f64;
            let (fill_left, fill_right) = match fraction {
                Some(frac) => {
                    let end = left + (span * frac.clamp(0.0, 1.0)).round() as i32;
                    (left, end.clamp(left, right))
                }
                None => {
                    let phase = (elapsed * 1.1).fract();
                    let start = left + (span * (phase - 0.25).max(0.0)) as i32;
                    let end = left + (span * phase.min(1.0)) as i32;
                    (start.clamp(left, right), end.clamp(left, right))
                }
            };

            if fill_right > fill_left {
                let fill = CreateSolidBrush(FILL);
                FillRect(
                    target_dc,
                    &RECT {
                        left: fill_left,
                        top: bar_top,
                        right: fill_right,
                        bottom: bar_bottom,
                    },
                    fill,
                );
                DeleteObject(fill);

                let highlight = CreateSolidBrush(FILL_HIGHLIGHT);
                FillRect(
                    target_dc,
                    &RECT {
                        left: fill_left,
                        top: bar_top,
                        right: fill_right,
                        bottom: (bar_top + 2).min(bar_bottom),
                    },
                    highlight,
                );
                DeleteObject(highlight);
            }

            if use_mem {
                BitBlt(hdc, 0, 0, w, h, mem_dc, 0, 0, SRCCOPY);
                SelectObject(mem_dc, old_bmp);
            }
            if !bitmap.is_null() {
                DeleteObject(bitmap);
            }
            if !mem_dc.is_null() {
                DeleteDC(mem_dc);
            }
            EndPaint(hwnd, &ps);
        }
    }
}

#[cfg(not(windows))]
mod imp {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thousands_formats_with_thin_ascii_groups() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1_000), "1 000");
        assert_eq!(thousands(12_345), "12 345");
        assert_eq!(thousands(1_234_567), "1 234 567");
    }

    #[test]
    fn clock_formats_seconds_minutes_and_hours() {
        assert_eq!(clock(0.0), "0 s");
        assert_eq!(clock(42.9), "42 s");
        assert_eq!(clock(65.0), "1 min 5 s");
        assert_eq!(clock(3661.0), "1 h 1 min");
    }

    #[test]
    fn counter_tallies_cached_translated_failed_and_bytes() {
        let state = Arc::new(State::new(true));
        let counter = Counter(state.clone());
        counter.container(Some(true));
        counter.container(Some(true));
        counter.container(Some(false));
        counter.container(None);
        counter.add_bytes(500);
        state.bytes_total.store(1000, Ordering::Relaxed);

        assert_eq!(state.containers.load(Ordering::Relaxed), 4);
        assert_eq!(state.cached.load(Ordering::Relaxed), 2);
        assert_eq!(state.translated.load(Ordering::Relaxed), 1);
        assert_eq!(state.failed.load(Ordering::Relaxed), 1);
        assert_eq!(state.fraction(), Some(0.5));
        assert!(!counter.is_cancelled());

        state.cancelled.store(true, Ordering::Relaxed);
        assert!(counter.is_cancelled());

        let [header, stats, footer] = format_lines(&state, 2.0);
        assert!(header.contains("(50%)"), "header={header}");
        assert!(stats.contains("4 shaders"), "stats={stats}");
        assert!(stats.contains("1 compiled"), "stats={stats}");
        assert!(stats.contains("2 cached"), "stats={stats}");
        assert!(stats.contains("1 failed"), "stats={stats}");
        assert!(footer.contains("Close window to skip"), "footer={footer}");
    }
}
