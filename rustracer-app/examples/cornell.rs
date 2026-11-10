//! Render the built-in Cornell box (the app's test scene) to a PNG, with
//! enough accumulation to show color bleed cleanly: the floor picks up red
//! near the red wall and green near the green wall.
//!
//! cargo run -p rustracer-app --example cornell --release

mod common;

use glam::Vec3;

fn main() -> anyhow::Result<()> {
    let scene = rustracer_core::scene::Scene::cornell_box();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter =
        pollster::block_on(rustracer_renderer::device::select_adapter(&instance, None))?;

    let (width, height) = (640u32, 480u32);
    let frames = 512u32;
    let config = rustracer_renderer::RenderConfig {
        width,
        height,
        spp: 4,
        accumulate_frames: frames,
        enable_denoise: false,
        ..Default::default()
    };
    let mut renderer =
        pollster::block_on(rustracer_renderer::Renderer::new(&adapter, config, &scene))?;
    // Straight-on view into the open front so the floor near both colored
    // walls is visible — frame_scene's diagonal hides the bleed contact.
    renderer.update_camera(&rustracer_core::camera::Camera {
        position: Vec3::new(0.0, 0.5, 3.0),
        look_at: Vec3::new(0.0, 0.0, 0.0),
        up: Vec3::Y,
        fov_degrees: 50.0,
        ..Default::default()
    });

    let target = renderer.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cornell_target"),
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

    let floats = common::read_accumulation(&renderer)?;
    common::save_png(&floats, width, height, "cornell.png")?;
    println!("saved cornell.png ({frames} frames x 4 spp, denoise off)");
    Ok(())
}
