//! Studio product render: drop the helmet on a glossy floor surrounded by
//! emissive softbox panels, so the metal and visor have real geometry to
//! reflect instead of a featureless gradient sky.
//!
//! cargo run -p rustracer-app --example studio -- <scene.gltf> <out.png> [frames] [w] [h] [yaw] [pitch]

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use rustracer_core::camera::Camera;
use rustracer_core::material::Material;
use rustracer_core::math::{Vec2, Vec3};
use rustracer_core::scene::{Light, Mesh, Scene};

/// A planar quad facing +normal, wound so its geometric normal matches.
fn add_quad(
    scene: &mut Scene,
    center: Vec3,
    normal: Vec3,
    w: f32,
    h: f32,
    material_id: usize,
) {
    let n = normal.normalize();
    let up = if n.dot(Vec3::Y).abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let t = up.cross(n).normalize();
    let b = n.cross(t);

    let hw = t * (w * 0.5);
    let hh = b * (h * 0.5);
    let positions = vec![
        center - hw - hh,
        center + hw - hh,
        center + hw + hh,
        center - hw + hh,
    ];
    let normals = vec![n; 4];
    let texcoords = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 1.0),
    ];
    let indices = vec![0, 1, 2, 0, 2, 3];
    scene.meshes.push(Arc::new(Mesh::new(
        positions,
        normals,
        texcoords,
        indices,
        material_id,
    )));
}

/// Emissive panel: adds the quad plus one area light per triangle.
fn add_softbox(
    scene: &mut Scene,
    center: Vec3,
    normal: Vec3,
    w: f32,
    h: f32,
    color: Vec3,
    intensity: f32,
) {
    let mat_id = scene.materials.len();
    scene
        .materials
        .push(Material::emissive(color, intensity));
    let first_tri = scene.triangle_count() as u32;
    add_quad(scene, center, normal, w, h, mat_id);
    for tri in [first_tri, first_tri + 1] {
        scene.lights.push(Light::Area {
            triangle_index: tri,
            color: color * intensity,
            intensity: 1.0,
        });
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().expect("usage: studio <scene.gltf> [out.png]");
    let out = args.get(2).cloned().unwrap_or_else(|| "studio.png".to_string());
    let frames: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);
    let width: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1280);
    let height: u32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(900);
    let yaw: f32 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let pitch: f32 = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let env_intensity: f32 = args.get(8).and_then(|s| s.parse().ok()).unwrap_or(0.10);

    let mut scene = rustracer_core::loader::load_gltf(PathBuf::from(&path))?;
    let desc_path = PathBuf::from(&path).with_file_name("scene.toml");
    if desc_path.exists() {
        let desc = rustracer_core::config::parse_descriptor(&std::fs::read_to_string(&desc_path)?)?;
        rustracer_core::config::apply_descriptor(&mut scene, &desc);
        println!("applied {}", desc_path.display());
    }

    // Dim cool dome: just enough to keep the shadow side alive. A bright sky
    // would reflect in the floor as a grey horizon and flatten every highlight.
    for light in scene.lights.iter_mut() {
        if let Light::Environment {
            color_zenith,
            color_horizon,
            intensity,
        } = light
        {
            *color_zenith = Vec3::new(0.62, 0.66, 0.80);
            *color_horizon = Vec3::new(0.012, 0.012, 0.015);
            *intensity = env_intensity;
        }
    }

    // Frame the subject before the studio geometry joins the bounds.
    let (bmin, bmax) = scene.bounds().expect("empty scene");
    let center = (bmin + bmax) * 0.5;
    let diag = (bmax - bmin).length();

    // Glossy dark floor the helmet sits on — mirrors the panels, fades to black.
    let floor_mat = scene.materials.len();
    scene
        .materials
        .push(Material::metal(Vec3::new(0.42, 0.42, 0.45), 0.32));
    let ground_y = bmin.y;
    add_quad(
        &mut scene,
        Vec3::new(center.x, ground_y, center.z),
        Vec3::Y,
        diag * 60.0,
        diag * 60.0,
        floor_mat,
    );

    // Softboxes: deliberately placed beside or behind the camera so they
    // light and show up in the reflections without ever entering the frame.
    let key_pos = center + Vec3::new(-1.45, 0.85, 1.55) * diag;
    let fill_pos = center + Vec3::new(1.55, 0.45, 1.35) * diag;
    let top_pos = center + Vec3::new(0.0, 2.0, 0.35) * diag;
    let rim_pos = center + Vec3::new(-0.35, 1.25, -1.9) * diag;
    let side_pos = center + Vec3::new(-1.9, 0.35, 0.15) * diag;

    add_softbox(
        &mut scene,
        key_pos,
        center - key_pos,
        diag * 1.4,
        diag * 1.0,
        Vec3::new(1.0, 0.97, 0.93),
        11.0,
    );
    add_softbox(
        &mut scene,
        fill_pos,
        center - fill_pos,
        diag * 1.0,
        diag * 0.9,
        Vec3::new(0.78, 0.86, 1.0),
        8.0,
    );
    add_softbox(
        &mut scene,
        top_pos,
        center - top_pos,
        diag * 2.2,
        diag * 1.8,
        Vec3::new(1.0, 1.0, 1.0),
        14.0,
    );
    add_softbox(
        &mut scene,
        rim_pos,
        center - rim_pos,
        diag * 1.4,
        diag * 0.55,
        Vec3::new(0.85, 0.92, 1.0),
        26.0,
    );
    add_softbox(
        &mut scene,
        side_pos,
        center - side_pos,
        diag * 1.2,
        diag * 1.0,
        Vec3::new(0.95, 0.97, 1.0),
        9.0,
    );

    let right_pos = center + Vec3::new(2.0, 0.55, -0.2) * diag;
    add_softbox(
        &mut scene,
        right_pos,
        center - right_pos,
        diag * 1.3,
        diag * 1.1,
        Vec3::new(0.88, 0.94, 1.0),
        12.0,
    );

    // Second kicker opposite the first, so both silhouette edges peel off
    // the black background and the shell stops reading as a flat cut-out.
    let kicker_r = center + Vec3::new(1.55, 1.05, -1.75) * diag;
    add_softbox(
        &mut scene,
        kicker_r,
        center - kicker_r,
        diag * 1.1,
        diag * 0.5,
        Vec3::new(0.90, 0.95, 1.0),
        22.0,
    );

    println!(
        "studio: {} meshes, {} mats, {} tris, {} lights",
        scene.meshes.len(),
        scene.materials.len(),
        scene.triangle_count(),
        scene.lights.len()
    );

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(rustracer_renderer::device::select_adapter(&instance, None))?;
    let config = rustracer_renderer::RenderConfig {
        width,
        height,
        spp: 8,
        accumulate_frames: frames,
        enable_denoise: true,
        enable_photon_map: false,
        photon_count: 50_000,
        exposure: 2.3,
        ..Default::default()
    };
    let mut renderer =
        pollster::block_on(rustracer_renderer::Renderer::new(&adapter, config, &scene))?;

    // Product framing: tight-ish 35 mm look from three-quarter front.
    let mut camera = Camera {
        position: center + Vec3::new(0.55, 0.16, 0.85).normalize() * diag * 1.55,
        look_at: center,
        up: Vec3::Y,
        fov_degrees: 32.0,
        ..Default::default()
    };
    if yaw != 0.0 || pitch != 0.0 {
        camera.orbit(yaw.to_radians(), pitch.to_radians());
    }
    renderer.update_camera(&camera);

    let target = renderer.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("studio_target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    // Optional light-gizmo smoke test: RS_LIGHT_MARKER=x,y,z,radius draws the
    // tonemap pass's light ball at a known world position.
    if let Ok(spec) = std::env::var("RS_LIGHT_MARKER") {
        let v: Vec<f32> = spec.split(',').filter_map(|t| t.trim().parse().ok()).collect();
        if v.len() == 4 || v.len() == 5 {
            renderer.light_marker = Some((Vec3::new(v[0], v[1], v[2]), v[3]));
            if v.len() == 5 {
                renderer.light_glow = v[4];
            }
            println!("marker test: {:?} radius {}", [v[0], v[1], v[2]], v[3]);
        }
    }

    for _ in 0..frames {
        let mut encoder = renderer.device.create_command_encoder(&Default::default());
        renderer.record_frame(&mut encoder, &view);
        renderer.queue.submit(Some(encoder.finish()));
    }
    renderer.device.poll(wgpu::Maintain::Wait);

    // Save the frame the app would present — tonemap, exposure and denoise
    // included — instead of the raw accumulation buffer.
    let pixels = common::read_target(&renderer, &target, width, height)?;
    common::save_png_bgra(&pixels, width, height, &out)?;
    println!("saved {out}");
    Ok(())
}
