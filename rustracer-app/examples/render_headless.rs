//! Headless render: load a glTF scene, render N accumulating frames,
//! save the tonemapped result as PNG. Used to reproduce visual bugs
//! without a window.
//!
//! cargo run -p rustracer-app --example render_headless -- <scene.gltf> [frames]

use std::path::PathBuf;

fn frame_camera(scene: &rustracer_core::scene::Scene) -> rustracer_core::camera::Camera {
    use rustracer_core::camera::Camera;
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

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: render_headless <scene.gltf> [frames]");
    let frames: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(128);
    let out = std::env::args().nth(3).unwrap_or_else(|| "render_headless.png".to_string());
    let denoise = std::env::args().nth(4).map(|s| s == "1").unwrap_or(false);
    let app_like = std::env::args().nth(5).map(|s| s == "1").unwrap_or(false);
    let photons = std::env::args().nth(6).map(|s| s == "1").unwrap_or(false);
    let spp_arg: u32 = std::env::args().nth(7).and_then(|s| s.parse().ok()).unwrap_or(if app_like { 4 } else { 1 });
    let accum_arg: u32 = std::env::args().nth(8).and_then(|s| s.parse().ok()).unwrap_or(if app_like { 0 } else { frames });
    let w_arg: u32 = std::env::args().nth(9).and_then(|s| s.parse().ok()).unwrap_or(800);
    let h_arg: u32 = std::env::args().nth(10).and_then(|s| s.parse().ok()).unwrap_or(600);
    let debug_view: u32 = std::env::args().nth(11).and_then(|s| s.parse().ok()).unwrap_or(0);
    let zoom_arg: f32 = std::env::args().nth(12).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let orbit_yaw: f32 = std::env::args().nth(13).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let orbit_pitch: f32 = std::env::args().nth(14).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let raw_save = std::env::args().nth(15).map(|s| s == "1").unwrap_or(false);

    let scene = rustracer_core::loader::load_gltf(PathBuf::from(&path))?;
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
        max_bounces: 6,
        debug_view,
        ..Default::default()
    };
    let mut renderer = pollster::block_on(rustracer_renderer::Renderer::new(&adapter, config, &scene))?;
    if photons {
        renderer.rebuild_photons()?;
    }
    let mut camera = frame_camera(&scene);
    if zoom_arg != 0.0 {
        camera.zoom(zoom_arg);
    }
    if orbit_yaw != 0.0 || orbit_pitch != 0.0 {
        camera.orbit(orbit_yaw, orbit_pitch);
    }
    renderer.update_camera(&camera);

    let target = renderer.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless_target"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
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

    // Read the accumulation buffer back.
    let acc = &renderer.buffers.accumulation;
    let size = (width * height) as u64 * 16;
    let staging = renderer.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = renderer.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(acc.buffer(), 0, &staging, 0, size);
    renderer.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    renderer.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let floats: &[f32] = bytemuck::cast_slice(&data);
    if raw_save {
        // Raw f32 RGBA dump for precision-critical comparisons.
        let mut path = out.clone();
        path.push_str(".bin");
        std::fs::write(&path, bytemuck::cast_slice::<f32, u8>(floats))?;
        println!("saved raw {path}");
        return Ok(());
    }
    let pixels: Vec<u8> = floats
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
    image::save_buffer(&out, &pixels, width, height, image::ColorType::Rgba8)?;
    println!("saved {out}");
    Ok(())
}
