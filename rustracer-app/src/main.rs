//! Rustracer — GPU-accelerated photon mapping path tracer with egui UI.

use rustracer_core::camera::Camera;
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
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let scene_path = args.get(1).cloned().unwrap_or_else(|| "scene.gltf".to_string());

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
    // fps counter
    fps: f32,
    frame_count: u64,
    fps_timer: std::time::Instant,
}

impl RustracerApp {
    fn new(scene_path: PathBuf) -> Self {
        Self { scene_path, state: None }
    }
}

impl ApplicationHandler for RustracerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window_attrs = Window::default_attributes()
            .with_title("Rustracer — Photon Mapper")
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
        let egui_handled = state.egui_winit.on_window_event(&state.window, &event).consumed;
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
                state.surface.configure(&state.renderer.device, &state.config);
                state.renderer.reset_accumulation();
            }
            WindowEvent::RedrawRequested => {
                render_frame(state, window_id);
                state.window.request_redraw();
            }
            WindowEvent::MouseInput { state: bs, button, .. } => {
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
                event: KeyEvent { physical_key: PhysicalKey::Code(key), state: ks, .. },
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
    let adapter = rustracer_renderer::device::select_adapter(&instance, Some(&surface)).await?;

    let size = window.inner_size();
    let width = size.width.max(1);
    let height = size.height.max(1);

    let spp = 4u32;
    let light_emission = [25.0f32, 25.0, 25.0];
    let point_light_emission = [25.0f32, 25.0, 25.0];
    let exposure = 1.0f32;
    let gamma = 2.2f32;
    let ambient = 0.03f32;
    let accumulate_frames = 0u32;
    let enable_photon_map = false;
    let enable_denoise = true;
    let photon_debug = false;
    let photon_scale = 0.02f32;

    let render_config = RenderConfig {
        width, height,
        photon_count: 50_000,
        max_bounces: 6,
        spp, exposure, gamma,
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

    let scene = if scene_path.exists() {
        rustracer_core::loader::load_gltf(scene_path)?
    } else {
        tracing::warn!("Scene '{}' not found, using test Cornell box", scene_path.display());
        create_test_scene()
    };

    let camera = frame_camera(&scene);

    let mut renderer = Renderer::new(&adapter, render_config, &scene).await?;
    renderer.update_camera(&camera);
    // Build the photon grid once at startup (trace-once)
    if renderer.config.enable_photon_map {
        renderer.rebuild_photons().ok();
    }

    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
        format: wgpu::TextureFormat::Bgra8Unorm,
        width, height,
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&renderer.device, &surface_config);

    // egui setup
    let egui_ctx = egui::Context::default();
    let egui_winit = egui_winit::State::new(
        egui_ctx.clone(), egui::viewport::ViewportId::ROOT, &window, None, None, None,
    );
    let egui_renderer = egui_wgpu::Renderer::new(&renderer.device, surface_config.format, None, 1, false);

    Ok(AppState {
        window, surface, instance, renderer,
        config: surface_config, camera,
        mouse_pressed: false, last_mouse: None, shift_held: false,
        egui_ctx, egui_winit, egui_renderer,
        spp, light_emission, point_light_emission, exposure, gamma,
        ambient, accumulate_frames, enable_photon_map, enable_denoise, photon_debug, photon_scale,
        frame_count: 0, fps: 0.0, fps_timer: std::time::Instant::now(),
    })
}

fn render_frame(state: &mut AppState, _window_id: winit::window::WindowId) {
    // FPS counter
    state.frame_count += 1;
    if state.fps_timer.elapsed().as_secs() >= 2 {
        state.fps = state.frame_count as f32 / state.fps_timer.elapsed().as_secs_f32();
        tracing::info!("FPS: {:.1}", state.fps);
        state.frame_count = 0;
        state.fps_timer = std::time::Instant::now();
    }

    let output = match state.surface.get_current_texture() {
        Ok(t) => t,
        Err(wgpu::SurfaceError::Lost) => {
            state.surface.configure(&state.renderer.device, &state.config);
            return;
        }
        Err(e) => {
            tracing::warn!("Surface error: {:?}", e);
            return;
        }
    };

    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Build egui UI
    let raw_input = state.egui_winit.take_egui_input(&state.window);
    let full_output = state.egui_ctx.run(raw_input, |ctx| {
        egui::Window::new("Render Settings")
            .default_pos([10.0, 10.0])
            .show(ctx, |ui| {
                ui.label(format!("Frame: {} | FPS: {:.0}", state.renderer.sample_count(), state.fps));
                ui.separator();

                ui.horizontal(|ui| {
                    ui.label("SPP:");
                    let changed = ui.add(egui::Slider::new(&mut state.spp, 1..=16)).changed();
                    ui.add(egui::DragValue::new(&mut state.spp).range(1..=16).speed(1));
                    if changed { state.renderer.config.spp = state.spp; state.renderer.reset_accumulation(); }
                });
                ui.horizontal(|ui| {
                    ui.label("Exposure:");
                    let changed = ui.add(egui::Slider::new(&mut state.exposure, 0.1..=5.0)).changed();
                    ui.add(egui::DragValue::new(&mut state.exposure).range(0.01..=10.0).speed(0.1));
                    if changed { state.renderer.config.exposure = state.exposure; state.renderer.reset_accumulation(); }
                });
                ui.horizontal(|ui| {
                    ui.label("Gamma:");
                    let changed = ui.add(egui::Slider::new(&mut state.gamma, 0.5..=4.0)).changed();
                    ui.add(egui::DragValue::new(&mut state.gamma).range(0.1..=10.0).speed(0.1));
                    if changed { state.renderer.config.gamma = state.gamma; state.renderer.reset_accumulation(); }
                });

                ui.separator();
                if ui.checkbox(&mut state.enable_photon_map, "Photon Map (WIP)").changed() {
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
                    if ui.add(egui::Slider::new(&mut state.photon_scale, 0.0..=2.0).logarithmic(true)).changed() {
                        state.renderer.config.photon_scale = state.photon_scale;
                        state.renderer.reset_accumulation();
                    }
                });
                if ui.checkbox(&mut state.photon_debug, "Photon Debug (density)").changed() {
                    state.renderer.config.photon_debug = state.photon_debug;
                    state.renderer.reset_accumulation();
                }
                if ui.checkbox(&mut state.enable_denoise, "Denoise").changed() {
                    state.renderer.config.enable_denoise = state.enable_denoise;
                }

                ui.horizontal(|ui| {
                    ui.label("Accum. Frames:");
                    let changed = ui.add(egui::Slider::new(&mut state.accumulate_frames, 0u32..=60)).changed();
                    ui.add(egui::DragValue::new(&mut state.accumulate_frames).range(0..=120).speed(1));
                    if changed { state.renderer.config.accumulate_frames = state.accumulate_frames; }
                });
                ui.horizontal(|ui| {
                    ui.label("Ambient:");
                    if ui.add(egui::Slider::new(&mut state.ambient, 0.0..=0.5)).changed() {
                        state.renderer.config.ambient = state.ambient;
                    }
                });

                ui.separator();
                ui.label("Area Light (RGB):");
                let mut changed = false;
                changed |= ui.add(egui::Slider::new(&mut state.light_emission[0], 0.0..=200.0).text("R")).changed();
                changed |= ui.add(egui::Slider::new(&mut state.light_emission[1], 0.0..=200.0).text("G")).changed();
                changed |= ui.add(egui::Slider::new(&mut state.light_emission[2], 0.0..=200.0).text("B")).changed();
                if changed {
                    state.renderer.config.light_emission = state.light_emission;
                    state.renderer.reset_accumulation();
                }
                ui.label("Point Light (RGB):");
                let mut changed = false;
                changed |= ui.add(egui::Slider::new(&mut state.point_light_emission[0], 0.0..=200.0).text("R")).changed();
                changed |= ui.add(egui::Slider::new(&mut state.point_light_emission[1], 0.0..=200.0).text("G")).changed();
                changed |= ui.add(egui::Slider::new(&mut state.point_light_emission[2], 0.0..=200.0).text("B")).changed();
                if changed {
                    state.renderer.config.point_light_emission = state.point_light_emission;
                    state.renderer.reset_accumulation();
                }

                ui.separator();
                if ui.button("Reset to Defaults").clicked() {
                    state.spp = 4; state.renderer.config.spp = 4;
                    state.exposure = 1.0; state.renderer.config.exposure = 1.0;
                    state.gamma = 2.2; state.renderer.config.gamma = 2.2;
                    state.light_emission = [25.0, 25.0, 25.0]; state.renderer.config.light_emission = [25.0, 25.0, 25.0];
                    state.point_light_emission = [25.0, 25.0, 25.0]; state.renderer.config.point_light_emission = [25.0, 25.0, 25.0];
                    state.ambient = 0.03; state.renderer.config.ambient = 0.03;
                    state.accumulate_frames = 0; state.renderer.config.accumulate_frames = 0;
                    state.enable_photon_map = false; state.renderer.config.enable_photon_map = false;
                    state.enable_denoise = true; state.renderer.config.enable_denoise = true;
                    state.photon_debug = false; state.renderer.config.photon_debug = false;
                    state.photon_scale = 0.02; state.renderer.config.photon_scale = 0.02;
                    state.renderer.reset_accumulation();
                }
            });
    });

    // Render the scene
    {
        let mut encoder = state.renderer.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("scene_encoder") },
        );
        state.renderer.record_frame(&mut encoder, &view);
        state.renderer.queue.submit(Some(encoder.finish()));
    }

    // Render egui on top
    {
        let mut encoder = state.renderer.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("egui_encoder") },
        );
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [state.config.width, state.config.height],
            pixels_per_point: state.window.scale_factor() as f32,
        };
        let paint_jobs = state.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        for (id, delta) in &full_output.textures_delta.set {
            state.egui_renderer.update_texture(&state.renderer.device, &state.renderer.queue, *id, delta);
        }
        state.egui_renderer.update_buffers(&state.renderer.device, &state.renderer.queue, &mut encoder, &paint_jobs, &screen_descriptor);
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            #[allow(unsafe_code)]
            let rp_static = unsafe {
                std::mem::transmute::<&mut wgpu::RenderPass<'_>, &mut wgpu::RenderPass<'static>>(&mut rp)
            };
            state.egui_renderer.render(rp_static, &paint_jobs, &screen_descriptor);
        }
        for id in &full_output.textures_delta.free {
            state.egui_renderer.free_texture(id);
        }
        state.renderer.queue.submit(Some(encoder.finish()));
    }

    output.present();
    state.egui_winit.handle_platform_output(&state.window, full_output.platform_output);
}

fn save_output(state: &AppState) {
    let acc_buffer = &state.renderer.buffers.accumulation;
    let size = (state.config.width * state.config.height) as u64 * 16;

    let staging = state.renderer.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_save"), size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.renderer.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("save_encoder") },
    );
    encoder.copy_buffer_to_buffer(acc_buffer.buffer(), 0, &staging, 0, size);
    state.renderer.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| { tx.send(result).ok(); });
    state.renderer.device.poll(wgpu::Maintain::Wait);

    if rx.recv().unwrap().is_ok() {
        let data = slice.get_mapped_range();
        let float_data: &[f32] = bytemuck::cast_slice(&data);
        let filename = format!("rustracer_{}spp.png", state.renderer.sample_count());

        let pixels: Vec<u8> = float_data
            .chunks_exact(4)
            .flat_map(|c| {
                let tr = (c[0] / (1.0 + c[0])).clamp(0.0, 1.0);
                let tg = (c[1] / (1.0 + c[1])).clamp(0.0, 1.0);
                let tb = (c[2] / (1.0 + c[2])).clamp(0.0, 1.0);
                [
                    (tr.powf(1.0 / 2.2) * 255.0) as u8,
                    (tg.powf(1.0 / 2.2) * 255.0) as u8,
                    (tb.powf(1.0 / 2.2) * 255.0) as u8,
                    255u8,
                ]
            })
            .collect();
        image::save_buffer(&filename, &pixels, state.config.width, state.config.height, image::ColorType::Rgba8).ok();
        tracing::info!("Saved output to {}", filename);
    }
}

// ── Test scene: Cornell box with objects ──

/// Frame the whole scene in view, whatever its size or position: place the
/// camera on a diagonal at a distance that fits the bounds' bounding sphere.
fn frame_camera(scene: &rustracer_core::scene::Scene) -> Camera {
    if let Some((bmin, bmax)) = scene.bounds() {
        let center = (bmin + bmax) * 0.5;
        let diag = (bmax - bmin).length().max(1e-3);
        let fov = 70f32.to_radians();
        let dist = (diag * 0.5 / (fov * 0.5).tan()) * 1.3;
        let dir = glam::Vec3::new(0.55, 0.35, 0.75).normalize();
        return Camera {
            position: center + dir * dist,
            look_at: center,
            up: glam::Vec3::Y,
            fov_degrees: 70.0,
            ..Default::default()
        };
    }
    Camera::default()
}

fn create_test_scene() -> rustracer_core::scene::Scene {
    use glam::Vec3;
    use rustracer_core::material::Material;
    use rustracer_core::scene::{Light, Mesh, Scene};

    let mut scene = Scene::default();

    // Materials
    scene.materials.push(Material::lambertian(Vec3::new(0.73, 0.73, 0.73))); // 0: white
    scene.materials.push(Material::lambertian(Vec3::new(0.65, 0.05, 0.05))); // 1: red
    scene.materials.push(Material::lambertian(Vec3::new(0.12, 0.45, 0.15))); // 2: green
    scene.materials.push(Material::lambertian(Vec3::new(0.73, 0.73, 0.73))); // 3: white (back/ceiling)
    scene.materials.push(Material::emissive(Vec3::new(25.0, 20.0, 14.0), 1.0)); // 4: warm light
    scene.materials.push(Material::metal(Vec3::new(0.95, 0.64, 0.54), 0.1)); // 5: gold metal
    scene.materials.push(Material::dielectric(1.5, Vec3::ONE)); // 6: glass
    scene.materials.push(Material::lambertian(Vec3::new(0.1, 0.1, 0.7))); // 7: blue

    // Cornell box walls
    let floor   = create_quad(Vec3::new(-1.0,-1.0, 1.0),Vec3::new( 1.0,-1.0, 1.0),Vec3::new( 1.0,-1.0,-1.0),Vec3::new(-1.0,-1.0,-1.0));
    let ceiling = create_quad(Vec3::new(-1.0, 2.0,-1.0),Vec3::new( 1.0, 2.0,-1.0),Vec3::new( 1.0, 2.0, 1.0),Vec3::new(-1.0, 2.0, 1.0));
    let back_w  = create_quad(Vec3::new(-1.0,-1.0,-1.0),Vec3::new( 1.0,-1.0,-1.0),Vec3::new( 1.0, 2.0,-1.0),Vec3::new(-1.0, 2.0,-1.0));
    let left_w  = create_quad(Vec3::new(-1.0,-1.0, 1.0),Vec3::new(-1.0,-1.0,-1.0),Vec3::new(-1.0, 2.0,-1.0),Vec3::new(-1.0, 2.0, 1.0));
    let right_w = create_quad(Vec3::new( 1.0,-1.0,-1.0),Vec3::new( 1.0,-1.0, 1.0),Vec3::new( 1.0, 2.0, 1.0),Vec3::new( 1.0, 2.0,-1.0));
    let light_q = create_quad(Vec3::new(-0.3, 1.99,-0.3),Vec3::new( 0.3, 1.99,-0.3),Vec3::new( 0.3, 1.99, 0.3),Vec3::new(-0.3, 1.99, 0.3));

    // Objects: tall box (right), short box (left)
    let tall_box  = create_cube(Vec3::new( 0.3, -1.0, -0.4), Vec3::new(1.0, -0.4, 0.0), 0);  // white
    let short_box = create_cube(Vec3::new(-0.7, -1.0,  0.1), Vec3::new(-0.4, -0.8, 0.8), 0);  // white
    let gold_cube = create_cube(Vec3::new(-0.2, -1.0, -0.7), Vec3::new(-0.05, -0.65, -0.55), 5); // gold metal
    let glass_cube = create_cube(Vec3::new( 0.5, -0.8, -0.7), Vec3::new(0.6, -0.5, -0.6), 6); // glass

    // Push all quads as meshes
    let walls = [floor, ceiling, back_w, left_w, right_w, light_q];
    let wall_mats = [0, 0, 3, 1, 2, 4];
    for (i, quad) in walls.iter().enumerate() {
        scene.meshes.push(std::sync::Arc::new(Mesh::new(
            quad.0.clone(), quad.1.clone(), quad.2.clone(), quad.3.clone(), wall_mats[i],
        )));
    }

    // Push object quads
    for (positions, normals, texcoords, indices, mat_id) in [tall_box, short_box, gold_cube, glass_cube].iter().flatten() {
        scene.meshes.push(std::sync::Arc::new(Mesh::new(
            positions.clone(), normals.clone(), texcoords.clone(), indices.clone(), *mat_id,
        )));
    }

    scene.lights.push(Light::Area { triangle_index: 10, color: Vec3::new(25.0, 20.0, 14.0), intensity: 1.0 });
    scene.lights.push(Light::Area { triangle_index: 11, color: Vec3::new(25.0, 20.0, 14.0), intensity: 1.0 });

    scene
}

fn create_quad(v0: glam::Vec3, v1: glam::Vec3, v2: glam::Vec3, v3: glam::Vec3) -> (Vec<glam::Vec3>, Vec<glam::Vec3>, Vec<glam::Vec2>, Vec<u32>) {
    let positions = vec![v0, v1, v2, v3];
    let normal = (v1 - v0).cross(v2 - v0).normalize();
    let normals = vec![normal; 4];
    let texcoords = vec![glam::Vec2::new(0.0,0.0), glam::Vec2::new(1.0,0.0), glam::Vec2::new(1.0,1.0), glam::Vec2::new(0.0,1.0)];
    let indices = vec![0, 1, 2, 0, 2, 3];
    (positions, normals, texcoords, indices)
}

/// Create a box from min to max. Returns 6 quads with the given material id.
fn create_cube(min: glam::Vec3, max: glam::Vec3, material_id: usize) -> Vec<(Vec<glam::Vec3>, Vec<glam::Vec3>, Vec<glam::Vec2>, Vec<u32>, usize)> {
    let lo = min;
    let hi = max;
    let faces: [(glam::Vec3, glam::Vec3, glam::Vec3, glam::Vec3); 6] = [
        (glam::Vec3::new(lo.x,lo.y,lo.z), glam::Vec3::new(lo.x,hi.y,lo.z), glam::Vec3::new(lo.x,hi.y,hi.z), glam::Vec3::new(lo.x,lo.y,hi.z)), // -X
        (glam::Vec3::new(hi.x,lo.y,hi.z), glam::Vec3::new(hi.x,hi.y,hi.z), glam::Vec3::new(hi.x,hi.y,lo.z), glam::Vec3::new(hi.x,lo.y,lo.z)), // +X
        (glam::Vec3::new(lo.x,lo.y,hi.z), glam::Vec3::new(hi.x,lo.y,hi.z), glam::Vec3::new(hi.x,lo.y,lo.z), glam::Vec3::new(lo.x,lo.y,lo.z)), // -Y
        (glam::Vec3::new(lo.x,hi.y,lo.z), glam::Vec3::new(hi.x,hi.y,lo.z), glam::Vec3::new(hi.x,hi.y,hi.z), glam::Vec3::new(lo.x,hi.y,hi.z)), // +Y
        (glam::Vec3::new(hi.x,lo.y,lo.z), glam::Vec3::new(hi.x,hi.y,lo.z), glam::Vec3::new(lo.x,hi.y,lo.z), glam::Vec3::new(lo.x,lo.y,lo.z)), // -Z
        (glam::Vec3::new(lo.x,lo.y,hi.z), glam::Vec3::new(lo.x,hi.y,hi.z), glam::Vec3::new(hi.x,hi.y,hi.z), glam::Vec3::new(hi.x,lo.y,hi.z)), // +Z
    ];
    faces.into_iter().map(|(v0, v1, v2, v3)| {
        let (p, n, t, i) = create_quad(v0, v1, v2, v3);
        (p, n, t, i, material_id)
    }).collect()
}
