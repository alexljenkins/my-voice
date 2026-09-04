//! Native X11/XWayland listening orb.
//!
//! The cpal callback publishes one lossy atomic loudness value. The overlay
//! reads it on its own thread, so drawing cannot block or consume audio.

use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tracing::{info, warn};
use x11rb::connection::Connection;
use x11rb::protocol::randr::ConnectionExt as _;
use x11rb::protocol::shape::{ConnectionExt as _, SK, SO};
use x11rb::protocol::xproto::{
    ClipOrdering, ColormapAlloc, ConnectionExt as _, CreateGCAux, CreateWindowAux, ImageFormat,
    StackMode, VisualClass, WindowClass,
};

const SIZE: usize = 180;
const POINTS: usize = 96;
const BOTTOM_MARGIN: i32 = 24;
const STYLES: [IndicatorStyle; 10] = [
    IndicatorStyle::Calm,
    IndicatorStyle::Agreeable,
    IndicatorStyle::Thoughtful,
    IndicatorStyle::Neutral,
    IndicatorStyle::Cold,
    IndicatorStyle::Defensive,
    IndicatorStyle::Anxious,
    IndicatorStyle::Frustrated,
    IndicatorStyle::Angry,
    IndicatorStyle::Random,
];
const CONCRETE_STYLES: [IndicatorStyle; 9] = [
    IndicatorStyle::Calm,
    IndicatorStyle::Agreeable,
    IndicatorStyle::Thoughtful,
    IndicatorStyle::Neutral,
    IndicatorStyle::Cold,
    IndicatorStyle::Defensive,
    IndicatorStyle::Anxious,
    IndicatorStyle::Frustrated,
    IndicatorStyle::Angry,
];

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IndicatorStyle {
    Calm,
    Agreeable,
    Thoughtful,
    #[default]
    Neutral,
    Cold,
    Defensive,
    Anxious,
    Frustrated,
    Angry,
    Random,
}

impl IndicatorStyle {
    pub const ALL: &'static [Self] = &STYLES;
}

impl fmt::Display for IndicatorStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", format!("{self:?}").to_lowercase())
    }
}

/// Choose once at successful keydown. `hold_id` makes each choice independent
/// without adding a random-number dependency.
pub fn choose_style(configured: IndicatorStyle, hold_id: u64) -> IndicatorStyle {
    if configured != IndicatorStyle::Random {
        return configured;
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;
    choose_random(hold_id ^ nanos.rotate_left(17))
}

fn choose_random(mut seed: u64) -> IndicatorStyle {
    seed ^= seed >> 12;
    seed ^= seed << 25;
    seed ^= seed >> 27;
    CONCRETE_STYLES[(seed.wrapping_mul(0x2545_f491_4f6c_dd1d) % 9) as usize]
}

#[derive(Clone)]
pub struct IndicatorHandle {
    tx: Option<mpsc::Sender<Command>>,
}

impl IndicatorHandle {
    pub fn show(&self, style: IndicatorStyle) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(Command::Show(style));
        }
    }

    pub fn hide(&self) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(Command::Hide);
        }
    }
}

#[derive(Clone, Copy)]
enum Command {
    Show(IndicatorStyle),
    Hide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OverlayState {
    visible: bool,
    style: IndicatorStyle,
}

impl Default for OverlayState {
    fn default() -> Self {
        Self {
            visible: false,
            style: IndicatorStyle::Neutral,
        }
    }
}

fn transition(state: OverlayState, command: Command) -> OverlayState {
    match command {
        Command::Show(style) => OverlayState {
            visible: true,
            style,
        },
        Command::Hide => OverlayState {
            visible: false,
            ..state
        },
    }
}

pub fn spawn(signal: Arc<AtomicU32>) -> IndicatorHandle {
    if std::env::var_os("DISPLAY").is_none() {
        warn!("DISPLAY is unavailable; listening orb disabled");
        return IndicatorHandle { tx: None };
    }
    let (tx, rx) = mpsc::channel();
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let result = Overlay::new().and_then(|mut overlay| {
            let _ = ready_tx.send(Ok(()));
            overlay.run(rx, signal)
        });
        if let Err(e) = result {
            let _ = ready_tx.send(Err(e.to_string()));
            warn!("listening orb stopped: {e}");
        }
    });
    match ready_rx.recv_timeout(Duration::from_secs(2)) {
        Ok(Ok(())) => {
            info!("listening orb ready on X11/XWayland");
            IndicatorHandle { tx: Some(tx) }
        }
        Ok(Err(e)) => {
            warn!("listening orb unavailable: {e}");
            IndicatorHandle { tx: None }
        }
        Err(_) => {
            warn!("listening orb startup timed out");
            IndicatorHandle { tx: None }
        }
    }
}

struct Overlay {
    conn: x11rb::rust_connection::RustConnection,
    window: u32,
    gc: u32,
    renderer: Renderer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DisplayGeometry {
    x: i16,
    y: i16,
    width: u16,
    height: u16,
}

fn window_position(display: DisplayGeometry) -> (i16, i16) {
    let x = i32::from(display.x) + (i32::from(display.width) - SIZE as i32) / 2;
    let y = i32::from(display.y) + i32::from(display.height) - SIZE as i32 - BOTTOM_MARGIN;
    (
        x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
        y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
    )
}

fn primary_display<C: Connection>(
    conn: &C,
    root: u32,
    fallback: DisplayGeometry,
) -> DisplayGeometry {
    let resolved = (|| -> Result<DisplayGeometry, x11rb::errors::ReplyOrIdError> {
        let primary = conn.randr_get_output_primary(root)?.reply()?.output;
        if primary == 0 {
            return Ok(fallback);
        }
        let resources = conn.randr_get_screen_resources_current(root)?.reply()?;
        let output = conn
            .randr_get_output_info(primary, resources.config_timestamp)?
            .reply()?;
        if output.crtc == 0 {
            return Ok(fallback);
        }
        let crtc = conn
            .randr_get_crtc_info(output.crtc, resources.config_timestamp)?
            .reply()?;
        if crtc.width == 0 || crtc.height == 0 {
            return Ok(fallback);
        }
        Ok(DisplayGeometry {
            x: crtc.x,
            y: crtc.y,
            width: crtc.width,
            height: crtc.height,
        })
    })();
    match resolved {
        Ok(display) => display,
        Err(e) => {
            warn!("cannot resolve the primary display through RandR: {e}; using the X root");
            fallback
        }
    }
}

impl Overlay {
    fn new() -> anyhow::Result<Self> {
        let (conn, screen_num) = x11rb::connect(None)?;
        let screen = &conn.setup().roots[screen_num];
        let visual = screen
            .allowed_depths
            .iter()
            .find(|depth| depth.depth == 32)
            .and_then(|depth| {
                depth
                    .visuals
                    .iter()
                    .find(|visual| visual.class == VisualClass::TRUE_COLOR)
            })
            .ok_or_else(|| anyhow::anyhow!("display has no 32-bit ARGB visual"))?;
        let root = screen.root;
        let display = primary_display(
            &conn,
            root,
            DisplayGeometry {
                x: 0,
                y: 0,
                width: screen.width_in_pixels,
                height: screen.height_in_pixels,
            },
        );
        let window = conn.generate_id()?;
        let colormap = conn.generate_id()?;
        let gc = conn.generate_id()?;
        conn.create_colormap(ColormapAlloc::NONE, colormap, root, visual.visual_id)?;
        let (x, y) = window_position(display);
        conn.create_window(
            32,
            window,
            root,
            x,
            y,
            SIZE as u16,
            SIZE as u16,
            0,
            WindowClass::INPUT_OUTPUT,
            visual.visual_id,
            &CreateWindowAux::new()
                .background_pixel(0)
                .border_pixel(0)
                .colormap(colormap)
                .override_redirect(1),
        )?;
        conn.create_gc(gc, window, &CreateGCAux::new())?;
        conn.shape_rectangles(
            SO::SET,
            SK::INPUT,
            ClipOrdering::UNSORTED,
            window,
            0,
            0,
            &[],
        )?;
        conn.flush()?;
        Ok(Self {
            conn,
            window,
            gc,
            renderer: Renderer::new(),
        })
    }

    fn run(&mut self, rx: mpsc::Receiver<Command>, signal: Arc<AtomicU32>) -> anyhow::Result<()> {
        let mut state = OverlayState::default();
        let mut last = Instant::now();
        loop {
            let command = if state.visible {
                match rx.recv_timeout(Duration::from_millis(16)) {
                    Ok(command) => Some(command),
                    Err(mpsc::RecvTimeoutError::Timeout) => None,
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            } else {
                match rx.recv() {
                    Ok(command) => Some(command),
                    Err(_) => break,
                }
            };
            match command {
                Some(Command::Show(style)) => {
                    state = transition(state, Command::Show(style));
                    self.renderer.reset(style);
                    self.conn.map_window(self.window)?;
                    self.conn.configure_window(
                        self.window,
                        &x11rb::protocol::xproto::ConfigureWindowAux::new()
                            .stack_mode(StackMode::ABOVE),
                    )?;
                    self.conn.flush()?;
                    last = Instant::now();
                }
                Some(Command::Hide) => {
                    state = transition(state, Command::Hide);
                    self.conn.unmap_window(self.window)?;
                    self.conn.flush()?;
                }
                None if state.visible => {
                    let now = Instant::now();
                    let delta = now.duration_since(last).as_secs_f32().clamp(0.001, 0.05);
                    last = now;
                    let level = f32::from_bits(signal.load(Ordering::Relaxed));
                    let pixels = self.renderer.frame(level, delta);
                    self.conn.put_image(
                        ImageFormat::Z_PIXMAP,
                        self.window,
                        self.gc,
                        SIZE as u16,
                        SIZE as u16,
                        0,
                        0,
                        0,
                        32,
                        pixels,
                    )?;
                    self.conn.flush()?;
                }
                None => {}
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Profile {
    hue_from: f32,
    hue_to: f32,
    saturation: f32,
    lightness: f32,
    sharpness: f32,
    spike_gain: f32,
    convex_rounding: f32,
    concave_rounding: f32,
    clusters: f32,
    smoothing: f32,
    stiffness: f32,
    damping: f32,
    breath_speed: f32,
}

fn profile(style: IndicatorStyle) -> Profile {
    let values = match style {
        IndicatorStyle::Calm => (
            208., 239., 0.77, 0.66, 1., 0.30, 1., 1., 9., 0.20, 100., 1.4, 0.20,
        ),
        IndicatorStyle::Agreeable => (
            223., 257., 0.79, 0.64, 1.3, 0.33, 0.79, 0.85, 10., 0.24, 130., 1.28, 0.22,
        ),
        IndicatorStyle::Thoughtful => (
            238., 274., 0.82, 0.63, 1.6, 0.36, 0.59, 0.70, 11., 0.28, 150., 1.35, 0.16,
        ),
        IndicatorStyle::Neutral | IndicatorStyle::Random => (
            248., 289., 0.84, 0.62, 2., 0.40, 0.31, 0.50, 13., 0.33, 200., 1., 0.26,
        ),
        IndicatorStyle::Cold => (
            209., 243., 0.81, 0.90, 3.4, 0.35, 0.48, 0.62, 21., 0.49, 90., 1.5, 0.12,
        ),
        IndicatorStyle::Defensive => (
            292., 332., 0.91, 0.58, 4.6, 0.58, 0.61, 0.72, 28., 0.63, 260., 0.8, 0.70,
        ),
        IndicatorStyle::Anxious => (
            276., 326., 0.89, 0.59, 2., 0.60, 1., 1., 30., 0.26, 300., 0.5, 1.,
        ),
        IndicatorStyle::Frustrated => (
            320., 361., 0.96, 0.55, 6.9, 0.73, 0.88, 0.91, 42., 0.88, 280., 0.67, 1.28,
        ),
        IndicatorStyle::Angry => (
            342., 381., 0.99, 0.54, 8., 0.80, 1., 1., 48., 1., 300., 0.6, 1.5,
        ),
    };
    Profile {
        hue_from: values.0,
        hue_to: values.1,
        saturation: values.2,
        lightness: values.3,
        sharpness: values.4,
        spike_gain: values.5,
        convex_rounding: values.6,
        concave_rounding: values.7,
        clusters: values.8,
        smoothing: values.9,
        stiffness: values.10,
        damping: values.11,
        breath_speed: values.12,
    }
}

struct Renderer {
    profile: Profile,
    radii: [f32; POINTS],
    velocity: [f32; POINTS],
    targets: [f32; POINTS],
    rounded: [f32; POINTS],
    phase: f32,
    activity: f32,
    pixels: Vec<u8>,
}

impl Renderer {
    fn new() -> Self {
        Self {
            profile: profile(IndicatorStyle::Neutral),
            radii: [50.; POINTS],
            velocity: [0.; POINTS],
            targets: [0.; POINTS],
            rounded: [0.; POINTS],
            phase: 0.,
            activity: 0.,
            pixels: vec![0; SIZE * SIZE * 4],
        }
    }

    fn reset(&mut self, style: IndicatorStyle) {
        self.profile = profile(style);
        self.velocity.fill(0.);
        self.activity = 0.;
        self.phase = 0.;
    }

    fn frame(&mut self, level: f32, delta: f32) -> &[u8] {
        let easing = if level > self.activity { 0.38 } else { 0.035 };
        self.activity += (level.clamp(0., 1.) - self.activity) * easing;
        self.phase += delta;
        self.update_contour(delta);
        rasterize(&mut self.pixels, &self.radii, self.profile);
        &self.pixels
    }

    fn update_contour(&mut self, delta: f32) {
        let p = self.profile;
        let clusters = p.clusters.round();
        for index in 0..POINTS {
            let angle = index as f32 / POINTS as f32 * std::f32::consts::TAU;
            let energy = ((angle * clusters + self.phase * p.breath_speed * 2.1).sin() * 0.5 + 0.5)
                .powf(p.sharpness / (1.0 + p.smoothing * 0.35))
                * self.activity;
            let idle = (angle * 3. + self.phase * p.breath_speed * 2.1).sin() * 0.15;
            self.targets[index] =
                48. * (1. + idle * (1. - self.activity * 0.6) + energy * p.spike_gain);
        }
        round_corners(
            &self.targets,
            &mut self.rounded,
            p.convex_rounding,
            p.concave_rounding,
        );
        let step = delta.min(1. / 30.);
        for index in 0..POINTS {
            let offset = self.rounded[index] - self.radii[index];
            self.velocity[index] += offset * p.stiffness * step;
            self.velocity[index] *= (-2. * p.stiffness.sqrt() * p.damping * step).exp();
            self.radii[index] += self.velocity[index] * step;
        }
    }
}

fn round_corners(source: &[f32], target: &mut [f32], convex: f32, concave: f32) {
    for index in 0..source.len() {
        let average = (source[(index + source.len() - 1) % source.len()]
            + source[(index + 1) % source.len()])
            / 2.;
        let strength = if source[index] >= average {
            convex
        } else {
            concave
        }
        .clamp(0., 1.);
        target[index] = source[index] + (average - source[index]) * strength;
    }
}

fn rasterize(pixels: &mut [u8], radii: &[f32; POINTS], profile: Profile) {
    let center = (SIZE as f32 - 1.) / 2.;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            let distance = dx.hypot(dy);
            let angle = dy.atan2(dx).rem_euclid(std::f32::consts::TAU);
            let at = angle / std::f32::consts::TAU * POINTS as f32;
            let index = at.floor() as usize % POINTS;
            let next = (index + 1) % POINTS;
            let radius = radii[index] + (radii[next] - radii[index]) * at.fract();
            let inside = radius - distance;
            let alpha = if inside < -1. {
                0.
            } else if inside < 1. {
                (inside + 1.) * 0.5
            } else {
                (inside / 18.).clamp(0., 1.) * 0.72
            };
            let rim = (1. - inside.abs()).clamp(0., 1.);
            let hue = profile.hue_from
                + (profile.hue_to - profile.hue_from) * angle / std::f32::consts::TAU;
            let (mut r, mut g, mut b) = hsl_to_rgb(hue, profile.saturation, profile.lightness);
            let white = rim * 0.6;
            r += (1. - r) * white;
            g += (1. - g) * white;
            b += (1. - b) * white;
            let alpha = (alpha + rim * 0.28).clamp(0., 1.);
            let offset = (y * SIZE + x) * 4;
            pixels[offset] = (b * alpha * 255.) as u8;
            pixels[offset + 1] = (g * alpha * 255.) as u8;
            pixels[offset + 2] = (r * alpha * 255.) as u8;
            pixels[offset + 3] = (alpha * 255.) as u8;
        }
    }
}

fn hsl_to_rgb(hue: f32, saturation: f32, lightness: f32) -> (f32, f32, f32) {
    let h = hue.rem_euclid(360.) / 60.;
    let chroma = (1. - (2. * lightness - 1.).abs()) * saturation;
    let secondary = chroma * (1. - (h.rem_euclid(2.) - 1.).abs());
    let (r, g, b) = match h.floor() as u8 {
        0 => (chroma, secondary, 0.),
        1 => (secondary, chroma, 0.),
        2 => (0., chroma, secondary),
        3 => (0., secondary, chroma),
        4 => (secondary, 0., chroma),
        _ => (chroma, 0., secondary),
    };
    let offset = lightness - chroma / 2.;
    (r + offset, g + offset, b + offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_never_selects_random() {
        for seed in 0..500 {
            assert_ne!(choose_random(seed), IndicatorStyle::Random);
        }
    }

    #[test]
    fn random_selection_can_reach_every_named_style() {
        let selected: std::collections::HashSet<_> = (0..500).map(choose_random).collect();
        assert_eq!(selected.len(), CONCRETE_STYLES.len());
    }

    #[test]
    fn fixed_style_does_not_change() {
        assert_eq!(choose_style(IndicatorStyle::Cold, 44), IndicatorStyle::Cold);
    }

    #[test]
    fn show_and_hide_transitions_keep_the_recording_style() {
        let shown = transition(
            OverlayState::default(),
            Command::Show(IndicatorStyle::Angry),
        );
        assert!(shown.visible);
        assert_eq!(shown.style, IndicatorStyle::Angry);
        let hidden = transition(shown, Command::Hide);
        assert!(!hidden.visible);
        assert_eq!(hidden.style, IndicatorStyle::Angry);
    }

    #[test]
    fn window_centers_above_the_selected_displays_bottom_edge() {
        assert_eq!(
            window_position(DisplayGeometry {
                x: 0,
                y: 0,
                width: 3840,
                height: 2160,
            }),
            (1830, 1956)
        );
        assert_eq!(
            window_position(DisplayGeometry {
                x: 3840,
                y: 200,
                width: 1920,
                height: 1080,
            }),
            (4710, 1076)
        );
    }

    #[test]
    fn corner_rounding_moves_peaks_toward_neighbours() {
        let source = [1., 3., 1.];
        let mut target = [0.; 3];
        round_corners(&source, &mut target, 1., 1.);
        assert_eq!(target[1], 1.);
    }

    #[test]
    fn loudness_changes_rendered_contour() {
        let mut renderer = Renderer::new();
        let quiet = renderer.frame(0., 0.016).to_vec();
        for _ in 0..12 {
            renderer.frame(1., 0.016);
        }
        assert_ne!(quiet, renderer.frame(1., 0.016));
    }

    #[test]
    #[ignore = "requires a live X11 or XWayland display"]
    fn live_overlay_is_viewable_and_has_no_input_region() -> anyhow::Result<()> {
        let mut overlay = Overlay::new()?;
        overlay.renderer.reset(IndicatorStyle::Neutral);
        let pixels = overlay.renderer.frame(0.7, 0.016);
        overlay.conn.map_window(overlay.window)?;
        overlay.conn.put_image(
            ImageFormat::Z_PIXMAP,
            overlay.window,
            overlay.gc,
            SIZE as u16,
            SIZE as u16,
            0,
            0,
            0,
            32,
            pixels,
        )?;
        overlay.conn.flush()?;
        let geometry = overlay.conn.get_geometry(overlay.window)?.reply()?;
        println!(
            "overlay geometry: {}x{}+{}+{}",
            geometry.width, geometry.height, geometry.x, geometry.y
        );
        assert_eq!((geometry.x, geometry.y), (1830, 1956));
        assert_eq!(
            overlay
                .conn
                .get_window_attributes(overlay.window)?
                .reply()?
                .map_state,
            x11rb::protocol::xproto::MapState::VIEWABLE
        );
        assert!(overlay
            .conn
            .shape_get_rectangles(overlay.window, SK::INPUT)?
            .reply()?
            .rectangles
            .is_empty());
        thread::sleep(Duration::from_millis(400));
        overlay.conn.unmap_window(overlay.window)?;
        overlay.conn.flush()?;
        Ok(())
    }
}
