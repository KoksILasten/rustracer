//! Headless render: load a glTF scene, render N accumulating frames,
//! save the tonemapped result as PNG. Used to reproduce visual bugs
//! without a window.
//!
//! cargo run -p rustracer-app --example render_headless -- <scene.gltf> \
//!     [frames] [out.png] [denoise] [app_like] [photons] [spp] [accum] \
//!     [w] [h] [debug_view] [zoom] [yaw] [pitch] [raw_save]
//!
//! Pass `1` for the boolean slots. `app_like` mirrors the app's defaults
//! (spp 4, 32-frame accumulation). `raw_save` writes the linear accumulation
//! instead of the tonemapped image.

mod common;

use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: render_headless <scene.gltf> [frames]");
    let frames: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let out = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "render_headless.png".to_string());
    let denoise = std::env::args().nth(4).map(|s| s == "1").unwrap_or(false);
    let app_like = std::env::args().nth(5).map(|s| s == "1").unwrap_or(false);
    let photons = std::env::args().nth(6).map(|s| s == "1").unwrap_or(false);
    let spp_arg: u32 = std::env::args()
        .nth(7)
        .and_then(|s| s.parse().ok())
        .unwrap_or(if app_like { 4 } else { 1 });
    let accum_arg: u32 = std::env::args()
        .nth(8)
        .and_then(|s| s.parse().ok())
        // 32: the app's own default (main.rs). Keeping 0 here made "app_like"
        // overwrite history every frame while the app accumulated — the two
        // disagreed, which is exactly the bug this flag exists to reproduce.
        .unwrap_or(if app_like { 32 } else { frames });
    let w_arg: u32 = std::env::args()
        .nth(9)
        .and_then(|s| s.parse().ok())
        .unwrap_or(800);
    let h_arg: u32 = std::env::args()
        .nth(10)
        .and_then(|s| s.parse().ok())
        .unwrap_or(600);
    let debug_view: u32 = std::env::args()
        .nth(11)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let zoom_arg: f32 = std::env::args()
        .nth(12)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let orbit_yaw: f32 = std::env::args()
        .nth(13)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let orbit_pitch: f32 = std::env::args()
        .nth(14)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let raw_save = std::env::args().nth(15).map(|s| s == "1").unwrap_or(false);

    let mut scene = rustracer_core::loader::load_gltf(PathBuf::from(&path))?;
    // Honour the sibling scene.toml exactly like the app does, so headless
    // renders can A/B environment and light overrides.
    let desc_path = PathBuf::from(&path).with_file_name("scene.toml");
    if desc_path.exists() {
        let desc = rustracer_core::config::parse_descriptor(&std::fs::read_to_string(&desc_path)?)?;
        rustracer_core::config::apply_descriptor(&mut scene, &desc);
        println!("applied {}", desc_path.display());
    }
    println!(
        "scene: {} meshes, {} mats, {} tris, roles a/mr/e/n = {}/{}/{}/{}",
        scene.meshes.len(),
        scene.materials.len(),
        scene.triangle_count(),
        scene.tex_albedo.len(),
        scene.tex_mr.len(),
        scene.tex_emissive.len(),
        scene.tex_normal.len()
    );

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(rustracer_renderer::device::select_adapter(&instance, None))?;
    let (width, height) = (w_arg, h_arg);
    let config = rustracer_renderer::RenderConfig {
        width,
        height,
        spp: spp_arg,
        accumulate_frames: accum_arg,
        enable_denoise: denoise || app_like,
        enable_photon_map: photons,
        photon_count: 50_000,
        debug_view,
        ..Default::default()
    };
    let mut renderer =
        pollster::block_on(rustracer_renderer::Renderer::new(&adapter, config, &scene))?;
    if photons {
        renderer.rebuild_photons()?;
    }
    let mut camera = rustracer_core::camera::Camera::frame_scene(&scene);
    if zoom_arg != 0.0 {
        camera.zoom(zoom_arg);
    }
    if orbit_yaw != 0.0 || orbit_pitch != 0.0 {
        camera.orbit(orbit_yaw, orbit_pitch);
    }
    renderer.update_camera(&camera);

    let target = renderer.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless_target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    for _ in 0..frames {
        let mut encoder = renderer.device.create_command_encoder(&Default::default());
        renderer.record_frame(&mut encoder, &view);
        renderer.queue.submit(Some(encoder.finish()));
    }
    renderer.device.poll(wgpu::Maintain::Wait);

    // Read the accumulation buffer back and save (raw or tonemapped PNG).
    let floats = common::read_accumulation(&renderer)?;
    if raw_save {
        // Raw f32 RGBA dump for precision-critical comparisons.
        let mut path = out.clone();
        path.push_str(".bin");
        std::fs::write(&path, bytemuck::cast_slice::<f32, u8>(&floats))?;
        println!("saved raw {path}");
        return Ok(());
    }
    common::save_png(&floats, width, height, &out)?;
    println!("saved {out}");
    Ok(())
}
