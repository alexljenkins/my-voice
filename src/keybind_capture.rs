//! Keybind-capture popup: a small floating window that records a push-to-talk
//! hotkey by letting the user press it. Pure-Rust, no GTK — a winit window with a
//! softbuffer CPU surface and an ab_glyph runtime system-font rasterizer.
//!
//! Public API: [`capture`] opens the popup on the current thread, runs its own
//! winit event loop, and returns the chosen hotkey string (or `None` on cancel).
//! The caller is expected to run this in a dedicated subprocess so owning the
//! thread/event loop here is fine.
//!
//! Hotkey string format (must match the listener parser): `[Mod+]*MainKey`,
//! modifiers in canonical order `Ctrl`, `Shift`, `Alt`, `Super`, joined by `+`,
//! followed by exactly one non-modifier main key. Examples: `CapsLock`, `F12`,
//! `Ctrl+Period`, `Ctrl+Shift+K`, `Alt+Slash`.
#![cfg(target_os = "linux")]

use std::num::NonZeroU32;
use std::rc::Rc;

use ab_glyph::{Font, FontVec, PxScale, ScaleFont};
use anyhow::{anyhow, Context};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, KeyCode, ModifiersState, NamedKey, PhysicalKey};
use winit::window::{Window, WindowId, WindowLevel};

const WIN_W: u32 = 380;
const WIN_H: u32 = 150;

// Colours (0x00RRGGBB in softbuffer's native format).
const BG: u32 = 0x0021_2733;
const FG: u32 = 0x00E6_EDF3;
const DIM: u32 = 0x0090_9DAB;

/// Candidate system fonts, tried in order. First existing file wins. Runtime
/// load keeps the binary lean (no embedded font).
const FONT_CANDIDATES: &[&str] = &[
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/TTF/Vera.ttf",
];

/// Open the capture popup. Returns Some(hotkey_string) on commit, None on cancel.
/// Creates a winit event loop on the CURRENT thread (caller runs this in a
/// dedicated subprocess, so owning the thread/event loop is fine).
pub fn capture() -> anyhow::Result<Option<String>> {
    let event_loop = EventLoop::new().context("create winit event loop")?;
    event_loop.set_control_flow(ControlFlow::Wait);

    let font = load_font(); // None => text is skipped, capture still works.
    let mut app = CaptureApp::new(font);
    event_loop.run_app(&mut app).context("run winit event loop")?;

    if let Some(err) = app.error.take() {
        return Err(err);
    }
    Ok(app.result.take())
}

/// Try each candidate path; return the first font that loads. `None` is fine —
/// the window still renders (blank body) and key capture is unaffected.
fn load_font() -> Option<FontVec> {
    for path in FONT_CANDIDATES {
        if let Ok(bytes) = std::fs::read(path) {
            if let Ok(font) = FontVec::try_from_vec(bytes) {
                return Some(font);
            }
        }
    }
    None
}

struct CaptureApp {
    font: Option<FontVec>,
    window: Option<Rc<Window>>,
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,
    mods: ModifiersState,
    /// Set once on commit; read after the loop exits.
    result: Option<String>,
    /// Surfaced to the caller if a fatal init error occurs mid-loop.
    error: Option<anyhow::Error>,
    /// True after we asked the loop to exit, so late events are ignored.
    done: bool,
}

impl CaptureApp {
    fn new(font: Option<FontVec>) -> Self {
        Self {
            font,
            window: None,
            surface: None,
            mods: ModifiersState::empty(),
            result: None,
            error: None,
            done: false,
        }
    }

    fn finish(&mut self, event_loop: &ActiveEventLoop, result: Option<String>) {
        self.result = result;
        self.done = true;
        event_loop.exit();
    }

    fn fail(&mut self, event_loop: &ActiveEventLoop, err: anyhow::Error) {
        self.error = Some(err);
        self.done = true;
        event_loop.exit();
    }
}

impl ApplicationHandler for CaptureApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_title("Set my-voice hotkey")
            .with_inner_size(LogicalSize::new(WIN_W, WIN_H))
            .with_min_inner_size(LogicalSize::new(WIN_W, WIN_H))
            .with_resizable(false)
            .with_window_level(WindowLevel::AlwaysOnTop)
            .with_active(true);

        let window = match event_loop.create_window(attrs) {
            Ok(w) => Rc::new(w),
            Err(e) => return self.fail(event_loop, anyhow!(e).context("create window")),
        };

        // Best-effort centering (X11/some Wayland compositors honour this; many
        // Wayland compositors ignore outer-position requests, which is fine).
        if let Some(mon) = window.current_monitor() {
            let msize = mon.size();
            let wsize: PhysicalSize<u32> = window.outer_size();
            if msize.width > wsize.width && msize.height > wsize.height {
                let x = mon.position().x + ((msize.width - wsize.width) / 2) as i32;
                let y = mon.position().y + ((msize.height - wsize.height) / 2) as i32;
                window.set_outer_position(winit::dpi::PhysicalPosition::new(x, y));
            }
        }

        // Try to pull keyboard focus to us (no-op on compositors that refuse).
        window.focus_window();

        let context = match softbuffer::Context::new(window.clone()) {
            Ok(c) => c,
            Err(e) => {
                return self.fail(event_loop, anyhow!(e.to_string()).context("softbuffer context"))
            }
        };
        let surface = match softbuffer::Surface::new(&context, window.clone()) {
            Ok(s) => s,
            Err(e) => {
                return self.fail(event_loop, anyhow!(e.to_string()).context("softbuffer surface"))
            }
        };

        self.window = Some(window);
        self.surface = Some(surface);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        if self.done {
            return;
        }
        match event {
            WindowEvent::CloseRequested => {
                // Window X / compositor close => cancel.
                self.finish(event_loop, None);
            }
            WindowEvent::ModifiersChanged(m) => {
                self.mods = m.state();
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                self.on_key(event_loop, event);
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.draw() {
                    // A draw failure shouldn't abort capture; just log to stderr.
                    eprintln!("keybind_capture: draw error: {e:#}");
                }
            }
            _ => {}
        }
    }
}

impl CaptureApp {
    fn on_key(&mut self, event_loop: &ActiveEventLoop, ev: KeyEvent) {
        // Only act on press; ignore key-up.
        if ev.state != ElementState::Pressed {
            return;
        }

        // Escape cancels — but only as a bare Escape (no modifiers). With a
        // modifier held, Escape is a legitimate main key (e.g. Ctrl+Escape).
        if matches!(ev.logical_key, Key::Named(NamedKey::Escape)) && self.mods.is_empty() {
            self.finish(event_loop, None);
            return;
        }

        // Ignore pure modifier presses — keep waiting for a real main key. The
        // live modifier line is driven by ModifiersChanged, not here.
        if is_modifier(&ev.physical_key) {
            return;
        }

        // Map to a main-key token. Unmappable key => ignore, keep waiting.
        let Some(main) = main_key_token(&ev) else {
            return;
        };

        let hotkey = build_hotkey(self.mods, &main);
        self.finish(event_loop, Some(hotkey));
    }

    fn draw(&mut self) -> anyhow::Result<()> {
        let (Some(window), Some(surface)) = (&self.window, &mut self.surface) else {
            return Ok(());
        };
        let size = window.inner_size();
        let (w, h) = (size.width.max(1), size.height.max(1));
        surface
            .resize(
                NonZeroU32::new(w).unwrap(),
                NonZeroU32::new(h).unwrap(),
            )
            .map_err(|e| anyhow!(e.to_string()))?;

        let mut buf = surface.buffer_mut().map_err(|e| anyhow!(e.to_string()))?;
        for px in buf.iter_mut() {
            *px = BG;
        }

        if let Some(font) = &self.font {
            let scaled = font.as_scaled(PxScale::from(22.0));
            let line_h = scaled.height();
            // Prompt line.
            draw_text(&mut buf, w, h, font, 24.0, 48.0, 22.0, FG, "Press a key…");
            // Live modifier line.
            let mods_line = modifier_preview(self.mods);
            draw_text(
                &mut buf,
                w,
                h,
                font,
                24.0,
                48.0 + line_h + 12.0,
                22.0,
                if self.mods.is_empty() { DIM } else { FG },
                &mods_line,
            );
            // Hint line.
            draw_text(
                &mut buf,
                w,
                h,
                font,
                24.0,
                (h as f32) - 22.0,
                14.0,
                DIM,
                "Esc to cancel",
            );
        }

        buf.present().map_err(|e| anyhow!(e.to_string()))?;
        Ok(())
    }
}

/// Render `text` into the softbuffer at the given top-left baseline-ish origin.
/// Simple left-to-right layout with kerning; alpha-blends glyph coverage over BG.
fn draw_text(
    buf: &mut [u32],
    buf_w: u32,
    buf_h: u32,
    font: &FontVec,
    x: f32,
    baseline: f32,
    px: f32,
    color: u32,
    text: &str,
) {
    let scaled = font.as_scaled(PxScale::from(px));
    let mut caret = x;
    let mut prev: Option<ab_glyph::GlyphId> = None;
    let (cr, cg, cb) = (
        ((color >> 16) & 0xFF) as f32,
        ((color >> 8) & 0xFF) as f32,
        (color & 0xFF) as f32,
    );

    for ch in text.chars() {
        let gid = scaled.glyph_id(ch);
        if let Some(p) = prev {
            caret += scaled.kern(p, gid);
        }
        let glyph = gid.with_scale_and_position(px, ab_glyph::point(caret, baseline));
        if let Some(outline) = font.outline_glyph(glyph) {
            let bounds = outline.px_bounds();
            outline.draw(|gx, gy, cov| {
                if cov <= 0.0 {
                    return;
                }
                let dx = bounds.min.x as i32 + gx as i32;
                let dy = bounds.min.y as i32 + gy as i32;
                if dx < 0 || dy < 0 || dx as u32 >= buf_w || dy as u32 >= buf_h {
                    return;
                }
                let idx = dy as usize * buf_w as usize + dx as usize;
                let bg = buf[idx];
                let (br, bg_, bb) = (
                    ((bg >> 16) & 0xFF) as f32,
                    ((bg >> 8) & 0xFF) as f32,
                    (bg & 0xFF) as f32,
                );
                let a = cov.clamp(0.0, 1.0);
                let r = (cr * a + br * (1.0 - a)) as u32;
                let g = (cg * a + bg_ * (1.0 - a)) as u32;
                let b = (cb * a + bb * (1.0 - a)) as u32;
                buf[idx] = (r << 16) | (g << 8) | b;
            });
        }
        caret += scaled.h_advance(gid);
        prev = Some(gid);
    }
}

/// Human-readable live modifier preview, e.g. "Ctrl + Shift + …".
fn modifier_preview(mods: ModifiersState) -> String {
    let active = active_modifier_tokens(mods);
    if active.is_empty() {
        "(no modifiers)".to_string()
    } else {
        format!("{} + …", active.join(" + "))
    }
}

/// Modifier tokens currently held, in canonical order: Ctrl, Shift, Alt, Super.
fn active_modifier_tokens(mods: ModifiersState) -> Vec<&'static str> {
    let mut v = Vec::new();
    if mods.control_key() {
        v.push("Ctrl");
    }
    if mods.shift_key() {
        v.push("Shift");
    }
    if mods.alt_key() {
        v.push("Alt");
    }
    if mods.super_key() {
        v.push("Super");
    }
    v
}

/// Assemble the final `[Mod+]*MainKey` string.
fn build_hotkey(mods: ModifiersState, main: &str) -> String {
    let mut parts = active_modifier_tokens(mods);
    parts.push(main);
    parts.join("+")
}

/// Is this physical key one of the left/right modifier keys? Used to skip pure
/// modifier presses while waiting for a real main key.
fn is_modifier(phys: &PhysicalKey) -> bool {
    matches!(
        phys,
        PhysicalKey::Code(
            KeyCode::ControlLeft
                | KeyCode::ControlRight
                | KeyCode::ShiftLeft
                | KeyCode::ShiftRight
                | KeyCode::AltLeft
                | KeyCode::AltRight
                | KeyCode::SuperLeft
                | KeyCode::SuperRight
        )
    )
}

/// Map a winit key event to the listener's main-key token, or `None` if it's a
/// key we don't emit (caller then keeps waiting). Driven off `physical_key`
/// (KeyCode) so it's layout-stable and case-insensitive.
fn main_key_token(ev: &KeyEvent) -> Option<String> {
    let PhysicalKey::Code(code) = ev.physical_key else {
        return None;
    };
    let tok = match code {
        // Letters -> single uppercase A..Z.
        KeyCode::KeyA => "A",
        KeyCode::KeyB => "B",
        KeyCode::KeyC => "C",
        KeyCode::KeyD => "D",
        KeyCode::KeyE => "E",
        KeyCode::KeyF => "F",
        KeyCode::KeyG => "G",
        KeyCode::KeyH => "H",
        KeyCode::KeyI => "I",
        KeyCode::KeyJ => "J",
        KeyCode::KeyK => "K",
        KeyCode::KeyL => "L",
        KeyCode::KeyM => "M",
        KeyCode::KeyN => "N",
        KeyCode::KeyO => "O",
        KeyCode::KeyP => "P",
        KeyCode::KeyQ => "Q",
        KeyCode::KeyR => "R",
        KeyCode::KeyS => "S",
        KeyCode::KeyT => "T",
        KeyCode::KeyU => "U",
        KeyCode::KeyV => "V",
        KeyCode::KeyW => "W",
        KeyCode::KeyX => "X",
        KeyCode::KeyY => "Y",
        KeyCode::KeyZ => "Z",

        // Digit row -> Digit0..Digit9.
        KeyCode::Digit0 => "Digit0",
        KeyCode::Digit1 => "Digit1",
        KeyCode::Digit2 => "Digit2",
        KeyCode::Digit3 => "Digit3",
        KeyCode::Digit4 => "Digit4",
        KeyCode::Digit5 => "Digit5",
        KeyCode::Digit6 => "Digit6",
        KeyCode::Digit7 => "Digit7",
        KeyCode::Digit8 => "Digit8",
        KeyCode::Digit9 => "Digit9",

        // Function keys F1..F24.
        KeyCode::F1 => "F1",
        KeyCode::F2 => "F2",
        KeyCode::F3 => "F3",
        KeyCode::F4 => "F4",
        KeyCode::F5 => "F5",
        KeyCode::F6 => "F6",
        KeyCode::F7 => "F7",
        KeyCode::F8 => "F8",
        KeyCode::F9 => "F9",
        KeyCode::F10 => "F10",
        KeyCode::F11 => "F11",
        KeyCode::F12 => "F12",
        KeyCode::F13 => "F13",
        KeyCode::F14 => "F14",
        KeyCode::F15 => "F15",
        KeyCode::F16 => "F16",
        KeyCode::F17 => "F17",
        KeyCode::F18 => "F18",
        KeyCode::F19 => "F19",
        KeyCode::F20 => "F20",
        KeyCode::F21 => "F21",
        KeyCode::F22 => "F22",
        KeyCode::F23 => "F23",
        KeyCode::F24 => "F24",

        // Named keys.
        KeyCode::CapsLock => "CapsLock",
        KeyCode::ScrollLock => "ScrollLock",
        KeyCode::NumLock => "NumLock",
        KeyCode::Space => "Space",
        KeyCode::Tab => "Tab",
        KeyCode::Enter | KeyCode::NumpadEnter => "Enter",
        KeyCode::Backspace => "Backspace",
        KeyCode::Insert => "Insert",
        KeyCode::Delete => "Delete",
        KeyCode::Home => "Home",
        KeyCode::End => "End",
        KeyCode::PageUp => "PageUp",
        KeyCode::PageDown => "PageDown",
        KeyCode::ArrowUp => "Up",
        KeyCode::ArrowDown => "Down",
        KeyCode::ArrowLeft => "Left",
        KeyCode::ArrowRight => "Right",

        // Punctuation.
        KeyCode::Period => "Period",
        KeyCode::Comma => "Comma",
        KeyCode::Slash => "Slash",
        KeyCode::Backslash => "Backslash",
        KeyCode::Semicolon => "Semicolon",
        KeyCode::Quote => "Apostrophe",
        KeyCode::BracketLeft => "LeftBracket",
        KeyCode::BracketRight => "RightBracket",
        KeyCode::Minus => "Minus",
        KeyCode::Equal => "Equal",
        KeyCode::Backquote => "Grave",

        // Anything else: not in the listener's vocabulary — ignore, keep waiting.
        _ => return None,
    };
    Some(tok.to_string())
}
