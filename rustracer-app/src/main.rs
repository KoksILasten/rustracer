//! Rustracer — GPU-accelerated photon mapping path tracer with egui UI.

use anyhow::Context;
use glam::Vec3;
use rustracer_core::camera::Camera;
use rustracer_core::scene::Light;
use rustracer_renderer::{RenderConfig, Renderer};
use std::path::PathBuf;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let scene_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "scene.gltf".to_string());

    tracing::info!("Loading scene: {}", scene_path);

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = RustracerApp::new(scene_path.into());
    event_loop.run_app(&mut app)?;

    Ok(())
}

struct RustracerApp {
    scene_path: PathBuf,
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    #[allow(dead_code)]
    instance: wgpu::Instance,
    renderer: Renderer,
    config: wgpu::SurfaceConfiguration,
    camera: Camera,
    mouse_pressed: bool,
    last_mouse: Option<(f64, f64)>,
    shift_held: bool,

    // egui
    egui_ctx: egui::Context,
    egui_winit: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,

    // Editable params
    spp: u32,
    light_emission: [f32; 3],
    point_light_emission: [f32; 3],
    exposure: f32,
    gamma: f32,
    ambient: f32,
    accumulate_frames: u32,
    enable_photon_map: bool,
    enable_denoise: bool,
    photon_debug: bool,
    photon_scale: f32,
    // moving light
    lights: Vec<Light>,
    moving_light: bool,
    moving_light_index: usize,
    /// Radius (world units) of the glowing ball drawn where the moving light
    /// is, so a light with no geometry can actually be seen.
    light_size: f32,
    /// How bright the gizmo ball glows.
    light_glow: f32,
    /// Multiplier on the orbit rate; 0 parks the light.
    light_speed: f32,
    /// Orbit angle in radians. Advanced by wall-clock time, not by frame count:
    /// at 200 fps a per-frame step spun the light 4x faster than at 50 fps.
    light_phase: f32,
    last_frame: std::time::Instant,
    // fps counter
    fps: f32,
    frame_count: u64,
    fps_timer: std::time::Instant,
}

impl RustracerApp {
    fn new(scene_path: PathBuf) -> Self {
        Self {
            scene_path,
            state: None,
        }
    }
}

impl ApplicationHandler for RustracerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window_attrs = Window::default_attributes()
            .with_title("Rustracer - Photon Mapper")
            .with_inner_size(winit::dpi::PhysicalSize::new(800, 600));
        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let state = pollster::block_on(init_app(window, surface, instance, &self.scene_path));

        match state {
            Ok(s) => self.state = Some(s),
            Err(e) => {
                tracing::error!("Failed to initialize: {:?}", e);
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else { return };

        // Let egui handle the event first
        let egui_handled = state
            .egui_winit
            .on_window_event(&state.window, &event)
            .consumed;
        if egui_handled {
            if state.mouse_pressed {
                state.mouse_pressed = false;
            }
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.config.width = size.width.max(1);
                state.config.height = size.height.max(1);
                state
                    .surface
                    .configure(&state.renderer.device, &state.config);
                state.renderer.reset_accumulation();
            }
            WindowEvent::RedrawRequested => {
                render_frame(state, window_id);
                state.window.request_redraw();
            }
            WindowEvent::MouseInput {
                state: bs, button, ..
            } => {
                if !egui_handled && button == MouseButton::Left {
                    state.mouse_pressed = bs.is_pressed();
                    if state.mouse_pressed {
                        state.last_mouse = None;
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if state.mouse_pressed && !egui_handled {
                    if let Some((lx, ly)) = state.last_mouse {
                        let dx = (position.x - lx) as f32;
                        let dy = (position.y - ly) as f32;
                        if state.shift_held {
                            state.camera.pan(-dx * 0.01, dy * 0.01);
                        } else {
                            state.camera.orbit(-dx * 0.005, -dy * 0.005);
                        }
                        state.renderer.update_camera(&state.camera);
                        state.renderer.reset_accumulation();
                    }
                    state.last_mouse = Some((position.x, position.y));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if !egui_handled {
                    let scroll = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                        winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                    };
                    state.camera.zoom(scroll);
                    state.renderer.update_camera(&state.camera);
                    state.renderer.reset_accumulation();
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state: ks,
                        ..
                    },
                ..
            } => {
                if !egui_handled {
                    let pressed = ks.is_pressed();
                    match key {
                        KeyCode::ShiftLeft | KeyCode::ShiftRight => state.shift_held = pressed,
                        KeyCode::Escape if pressed => event_loop.exit(),
                        KeyCode::KeyS if pressed => save_output(state),
                        KeyCode::KeyR if pressed => state.renderer.reset_accumulation(),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
}

async fn init_app(
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    instance: wgpu::Instance,
    scene_path: &PathBuf,
) -> anyhow::Result<AppState> {
    let adapter = rustracer_renderer::device::select_adapter(&instance, Some(&surface))
        .await
        .context("selecting a GPU adapter")?;

    let size = window.inner_size();
    let width = size.width.max(1);
    let height = size.height.max(1);

    let spp = 4u32;
    let light_emission = [1.0f32, 1.0, 1.0];
    let point_light_emission = [1.0f32, 1.0, 1.0];
    let exposure = 1.0f32;
    let gamma = 2.2f32;
    let ambient = 0.03f32;
    // 0 would overwrite the history every frame (one frame of samples, forever);
    // a non-zero value is what lets highlights actually converge.
    let accumulate_frames = 32u32;
    let enable_photon_map = false;
    let enable_denoise = true;
    let photon_debug = false;
    let photon_scale = 0.02f32;

    let render_config = RenderConfig {
        width,
        height,
        photon_count: 50_000,
        spp,
        exposure,
        gamma,
        light_emission,
        point_light_emission,
        accumulate_frames,
        enable_photon_map,
        enable_denoise,
        photon_debug,
        debug_view: 0,
        photon_scale,
        photon_radius: 0.5,
        ambient,
    };

    let mut scene = if scene_path.exists() {
        rustracer_core::loader::load_gltf(scene_path)
            .with_context(|| format!("loading scene '{}'", scene_path.display()))?
    } else if scene_path.to_str() == Some("dark-room") {
        rustracer_core::scene::Scene::dark_room()
    } else {
        tracing::warn!(
            "Scene '{}' not found, using test Cornell box (or pass 'dark-room')",
            scene_path.display()
        );
        rustracer_core::scene::Scene::cornell_box()
    };

    // Optional scene.toml next to the glTF overrides materials and lights.
    let descriptor_path = scene_path.with_file_name("scene.toml");
    if descriptor_path.exists() {
        let toml_str = std::fs::read_to_string(&descriptor_path)
            .with_context(|| format!("reading '{}'", descriptor_path.display()))?;
        let descriptor = rustracer_core::config::parse_descriptor(&toml_str)?;
        rustracer_core::config::apply_descriptor(&mut scene, &descriptor);
        tracing::info!("Applied {}", descriptor_path.display());
    }

    // A moving point light (orbit around the box center), toggled in the UI.
    // Pushed last so its index is `scene.lights.len()`.
    let moving_light_index = scene.lights.len();
    scene.lights.push(Light::Point {
        position: Vec3::new(0.8, 1.2, 0.0),
        color: Vec3::new(1.0, 0.5, 0.15),
        intensity: 40.0,
    });

    let camera = Camera::frame_scene(&scene);

    let mut renderer = Renderer::new(&adapter, render_config, &scene)
        .await
        .context("initializing the renderer")?;
    renderer.update_camera(&camera);
    // Build the photon grid once at startup (trace-once)
    if renderer.config.enable_photon_map {
        renderer.rebuild_photons().ok();
    }

    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
        format: wgpu::TextureFormat::Bgra8Unorm,
        width,
        height,
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&renderer.device, &surface_config);

    // egui setup
    let egui_ctx = egui::Context::default();
    let egui_winit = egui_winit::State::new(
        egui_ctx.clone(),
        egui::viewport::ViewportId::ROOT,
        &window,
        None,
        None,
        None,
    );
    let egui_renderer =
        egui_wgpu::Renderer::new(&renderer.device, surface_config.format, None, 1, false);

    // Scene-relative light gizmo: a fixed world radius would be invisible on a
    // large model and cartoonishly big on a small one.
    let scene_diag = scene
        .bounds()
        .map(|(lo, hi)| (hi - lo).length())
        .unwrap_or(1.0)
        .max(1e-3);

    Ok(AppState {
        window,
        surface,
        instance,
        renderer,
        config: surface_config,
        camera,
        mouse_pressed: false,
        last_mouse: None,
        shift_held: false,
        egui_ctx,
        egui_winit,
        egui_renderer,
        spp,
        light_emission,
        point_light_emission,
        exposure,
        gamma,
        ambient,
        accumulate_frames,
        enable_photon_map,
        enable_denoise,
        photon_debug,
        photon_scale,
        lights: scene.lights.clone(),
        moving_light: true,
        moving_light_index,
        light_size: scene_diag * 0.045,
        light_glow: 1.0,
        light_speed: 1.0,
        light_phase: 0.0,
        last_frame: std::time::Instant::now(),
        frame_count: 0,
        fps: 0.0,
        fps_timer: std::time::Instant::now(),
    })
}

/// EMA length used while the moving light is in flight. The user's Accum.
/// Frames budget applies to a parked light (where the history can grow and the
/// image converges); a moving light needs a short history, or the running mean
/// smears illumination from where the light used to be. 24 frames at 4 spp is
/// ~96 samples: enough to settle the highlights, while over that window the
/// light travels roughly a third of its own diameter, so nothing visibly trails.
const MOVING_HISTORY: u32 = 24;

fn render_frame(state: &mut AppState, _window_id: winit::window::WindowId) {
    // FPS counter
    state.frame_count += 1;
    if state.fps_timer.elapsed().as_secs() >= 2 {
        state.fps = state.frame_count as f32 / state.fps_timer.elapsed().as_secs_f32();
        tracing::info!("FPS: {:.1}", state.fps);
        state.frame_count = 0;
        state.fps_timer = std::time::Instant::now();
    }

    // Animate the moving point light (orbit around the box center). Two things
    // matter here:
    //
    // * The light moves smoothly, every frame. Quantising its motion into steps
    //   (or holding it still between steps) reads as jumping on screen, which is
    //   worse than a little noise.
    // * The accumulation history is an EMA whose length is capped while the
    //   light moves. A long history of *different* light positions ghosts, so the
    //   moving case forgets fast (MOVING_HISTORY frames -- a fraction of the
    //   light's own diameter travelled, so no visible smear) and still averages
    //   enough samples to keep highlights quiet. A parked light keeps the user's
    //   full budget and converges properly.
    let now = std::time::Instant::now();
    let dt = now
        .duration_since(state.last_frame)
        .as_secs_f32()
        .clamp(0.0, 0.1);
    state.last_frame = now;
    const ORBIT_RATE: f32 = 0.7; // radians per second at speed 1.0
    if state.moving_light {
        state.light_phase = (state.light_phase + dt * state.light_speed * ORBIT_RATE) % std::f32::consts::TAU;
        let t = state.light_phase;
        let target = Vec3::new(0.8 * t.cos(), 1.2, 0.8 * t.sin());
        let parked = state.light_speed.abs() < 1e-3;
        if let Some(Light::Point { position, .. }) = state.lights.get_mut(state.moving_light_index)
        {
            if !parked {
                *position = target;
            }
        }
        // Swap the history budget when the light starts or stops moving. The
        // reset is what stops the parked image from inheriting the moving EMA.
        let budget = if parked {
            state.accumulate_frames
        } else {
            MOVING_HISTORY.min(state.accumulate_frames.max(1))
        };
        if state.renderer.config.accumulate_frames != budget {
            state.renderer.config.accumulate_frames = budget;
            state.renderer.reset_accumulation_soft();
        }
        if !parked {
            state.renderer.update_lights(&state.lights);
        }
        let marker = match state.lights.get(state.moving_light_index) {
            Some(Light::Point { position, .. }) => Some((*position, state.light_size)),
            _ => None,
        };
        state.renderer.light_marker = marker;
        state.renderer.light_glow = state.light_glow;
    } else {
        state.renderer.light_marker = None;
        // Hand the accumulation budget back to the user's knob: the cap above
        // only exists because a *moving* light cannot afford a long history.
        if state.renderer.config.accumulate_frames != state.accumulate_frames {
            state.renderer.config.accumulate_frames = state.accumulate_frames;
            state.renderer.reset_accumulation_soft();
        }
    }

    let output = match state.surface.get_current_texture() {
        Ok(t) => t,
        Err(wgpu::SurfaceError::Lost) => {
            state
                .surface
                .configure(&state.renderer.device, &state.config);
            return;
        }
        Err(e) => {
            tracing::warn!("Surface error: {:?}", e);
            return;
        }
    };

    let view = output
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    // Build egui UI
    let raw_input = state.egui_winit.take_egui_input(&state.window);
    let full_output = state.egui_ctx.run(raw_input, |ctx| {
        egui::Window::new("Render Settings")
            .default_pos([10.0, 10.0])
            .show(ctx, |ui| {
                ui.label(format!(
                    "Frame: {} | FPS: {:.0}",
                    state.renderer.sample_count(),
                    state.fps
                ));
                ui.separator();

                ui.horizontal(|ui| {
                    ui.label("SPP:");
                    let changed = ui.add(egui::Slider::new(&mut state.spp, 1..=16)).changed();
                    ui.add(egui::DragValue::new(&mut state.spp).range(1..=16).speed(1));
                    if changed {
                        state.renderer.config.spp = state.spp;
                        state.renderer.reset_accumulation();
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Exposure:");
                    let changed = ui
                        .add(egui::Slider::new(&mut state.exposure, 0.1..=5.0))
                        .changed();
                    ui.add(
                        egui::DragValue::new(&mut state.exposure)
                            .range(0.01..=10.0)
                            .speed(0.1),
                    );
                    if changed {
                        state.renderer.config.exposure = state.exposure;
                        state.renderer.reset_accumulation();
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Gamma:");
                    let changed = ui
                        .add(egui::Slider::new(&mut state.gamma, 0.5..=4.0))
                        .changed();
                    ui.add(
                        egui::DragValue::new(&mut state.gamma)
                            .range(0.1..=10.0)
                            .speed(0.1),
                    );
                    if changed {
                        state.renderer.config.gamma = state.gamma;
                        state.renderer.reset_accumulation();
                    }
                });

                ui.separator();
                if ui
                    .checkbox(&mut state.enable_photon_map, "Photon Map (WIP)")
                    .changed()
                {
                    state.renderer.config.enable_photon_map = state.enable_photon_map;
                    if state.enable_photon_map {
                        state.renderer.rebuild_photons().ok();
                    }
                    state.renderer.reset_accumulation();
                }
                if ui.button("Rebuild Photons").clicked() {
                    state.renderer.rebuild_photons().ok();
                    state.renderer.reset_accumulation();
                }
                ui.horizontal(|ui| {
                    ui.label("Photon Scale:");
                    if ui
                        .add(
                            egui::Slider::new(&mut state.photon_scale, 0.0..=2.0).logarithmic(true),
                        )
                        .changed()
                    {
                        state.renderer.config.photon_scale = state.photon_scale;
                        state.renderer.reset_accumulation();
                    }
                });
                if ui
                    .checkbox(&mut state.photon_debug, "Photon Debug (density)")
                    .changed()
                {
                    state.renderer.config.photon_debug = state.photon_debug;
                    state.renderer.reset_accumulation();
                }
                if ui.checkbox(&mut state.enable_denoise, "Denoise").changed() {
                    state.renderer.config.enable_denoise = state.enable_denoise;
                }

                ui.horizontal(|ui| {
                    ui.label("Accum. Frames:");
                    let changed = ui
                        .add(egui::Slider::new(&mut state.accumulate_frames, 0u32..=60))
                        .changed();
                    ui.add(
                        egui::DragValue::new(&mut state.accumulate_frames)
                            .range(0..=120)
                            .speed(1),
                    );
                    if changed {
                        state.renderer.config.accumulate_frames = state.accumulate_frames;
                    }
                });
                if state.moving_light && state.light_speed.abs() > 1e-3 {
                    ui.label(
                        egui::RichText::new(format!(
                            "history is {} frames while the light is moving",
                            MOVING_HISTORY.min(state.accumulate_frames.max(1))
                        ))
                        .small()
                        .weak(),
                    );
                }
                ui.horizontal(|ui| {
                    ui.label("Ambient:");
                    if ui
                        .add(egui::Slider::new(&mut state.ambient, 0.0..=2.0))
                        .changed()
                    {
                        state.renderer.config.ambient = state.ambient;
                        state.renderer.reset_accumulation();
                    }
                });

                ui.separator();
                ui.label("Area Light (RGB):");
                let mut changed = false;
                changed |= ui
                    .add(egui::Slider::new(&mut state.light_emission[0], 0.0..=5.0).text("R"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut state.light_emission[1], 0.0..=5.0).text("G"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut state.light_emission[2], 0.0..=5.0).text("B"))
                    .changed();
                if changed {
                    state.renderer.config.light_emission = state.light_emission;
                    state.renderer.reset_accumulation();
                }
                ui.label("Point Light (RGB):");
                let mut changed = false;
                changed |= ui
                    .add(
                        egui::Slider::new(&mut state.point_light_emission[0], 0.0..=5.0)
                            .text("R"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut state.point_light_emission[1], 0.0..=5.0)
                            .text("G"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut state.point_light_emission[2], 0.0..=5.0)
                            .text("B"),
                    )
                    .changed();
                if changed {
                    state.renderer.config.point_light_emission = state.point_light_emission;
                    state.renderer.reset_accumulation();
                }
                if ui
                    .checkbox(&mut state.moving_light, "Moving light")
                    .changed()
                {
                    state.renderer.reset_accumulation();
                }
                if state.moving_light {
                    ui.horizontal(|ui| {
                        ui.label("Light size:");
                        ui.add(
                            egui::Slider::new(&mut state.light_size, 0.01..=0.50)
                                .fixed_decimals(2),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Glow:");
                        ui.add(egui::Slider::new(&mut state.light_glow, 0.1..=6.0).fixed_decimals(2));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Spin speed:");
                        ui.add(egui::Slider::new(&mut state.light_speed, 0.0..=8.0).fixed_decimals(2));
                    });
                }

                ui.separator();
                if ui.button("Reset to Defaults").clicked() {
                    state.spp = 4;
                    state.renderer.config.spp = 4;
                    state.exposure = 1.0;
                    state.renderer.config.exposure = 1.0;
                    state.gamma = 2.2;
                    state.renderer.config.gamma = 2.2;
                    state.light_emission = [1.0, 1.0, 1.0];
                    state.renderer.config.light_emission = [1.0, 1.0, 1.0];
                    state.point_light_emission = [1.0, 1.0, 1.0];
                    state.renderer.config.point_light_emission = [1.0, 1.0, 1.0];
                    state.ambient = 0.03;
                    state.renderer.config.ambient = 0.03;
                    state.light_phase = 0.0;
                    state.accumulate_frames = 32;
                    state.renderer.config.accumulate_frames = 32;
                    state.enable_photon_map = false;
                    state.renderer.config.enable_photon_map = false;
                    state.enable_denoise = true;
                    state.renderer.config.enable_denoise = true;
                    state.photon_debug = false;
                    state.renderer.config.photon_debug = false;
                    state.photon_scale = 0.02;
                    state.renderer.config.photon_scale = 0.02;
                    state.renderer.reset_accumulation();
                }
            });
    });

    // Render the scene
    {
        let mut encoder =
            state
                .renderer
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("scene_encoder"),
                });
        state.renderer.record_frame(&mut encoder, &view);
        state.renderer.queue.submit(Some(encoder.finish()));
    }

    // Render egui on top
    {
        let mut encoder =
            state
                .renderer
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("egui_encoder"),
                });
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [state.config.width, state.config.height],
            pixels_per_point: state.window.scale_factor() as f32,
        };
        let paint_jobs = state
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        for (id, delta) in &full_output.textures_delta.set {
            state.egui_renderer.update_texture(
                &state.renderer.device,
                &state.renderer.queue,
                *id,
                delta,
            );
        }
        state.egui_renderer.update_buffers(
            &state.renderer.device,
            &state.renderer.queue,
            &mut encoder,
            &paint_jobs,
            &screen_descriptor,
        );
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            #[allow(unsafe_code)]
            let rp_static = unsafe {
                std::mem::transmute::<&mut wgpu::RenderPass<'_>, &mut wgpu::RenderPass<'static>>(
                    &mut rp,
                )
            };
            state
                .egui_renderer
                .render(rp_static, &paint_jobs, &screen_descriptor);
        }
        for id in &full_output.textures_delta.free {
            state.egui_renderer.free_texture(id);
        }
        state.renderer.queue.submit(Some(encoder.finish()));
    }

    output.present();
    state
        .egui_winit
        .handle_platform_output(&state.window, full_output.platform_output);
}

fn save_output(state: &AppState) {
    let acc_buffer = &state.renderer.buffers.accumulation;
    let size = (state.config.width * state.config.height) as u64 * 16;

    let staging = state
        .renderer
        .device
        .create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_save"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
    let mut encoder =
        state
            .renderer
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("save_encoder"),
            });
    encoder.copy_buffer_to_buffer(acc_buffer.buffer(), 0, &staging, 0, size);
    state.renderer.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).ok();
    });
    state.renderer.device.poll(wgpu::Maintain::Wait);

    if rx.recv().unwrap().is_ok() {
        let data = slice.get_mapped_range();
        let float_data: &[f32] = bytemuck::cast_slice(&data);
        let filename = format!("rustracer_frame{}.png", state.renderer.sample_count());
        // Match the shader tonemap (exposure → Reinhard → gamma) using the
        // live UI values so the saved image equals what is on screen.
        let exposure = state.renderer.config.exposure;
        let gamma = state.renderer.config.gamma;

        let pixels: Vec<u8> = float_data
            .chunks_exact(4)
            .flat_map(|c| {
                let map = |v: f32| {
                    let e = v * exposure;
                    (e / (1.0 + e)).clamp(0.0, 1.0).powf(1.0 / gamma) * 255.0
                };
                [map(c[0]) as u8, map(c[1]) as u8, map(c[2]) as u8, 255u8]
            })
            .collect();
        image::save_buffer(
            &filename,
            &pixels,
            state.config.width,
            state.config.height,
            image::ColorType::Rgba8,
        )
        .ok();
        tracing::info!("Saved output to {}", filename);
    }
}

