//! Regression tests for CPU↔GPU layout agreement and full-pipeline dispatch.
//!
//! The CPU struct sizes must match the WGSL struct spans exactly — a
//! mismatch is exactly what produced "Buffer is bound with size 48 where the
//! shader expects 64" validation failures at dispatch time.

use std::mem::size_of;

#[test]
fn shared_struct_sizes_match_wgsl() {
    let module = naga::front::wgsl::parse_str(include_str!("../shaders/common.wgsl")).unwrap();
    let expected = [
        ("Light", size_of::<rustracer_core::scene::LightGpu>()),
        ("Material", size_of::<rustracer_core::material::MaterialGpu>()),
        ("Triangle", size_of::<rustracer_core::scene::TriangleGpu>()),
        ("BVHNode", size_of::<rustracer_core::bvh::BVHNode>()),
        ("Camera", size_of::<rustracer_core::camera::CameraUniform>()),
        ("Photon", size_of::<rustracer_renderer::PhotonGpu>()),
    ];
    for (name, rust_size) in expected {
        let ty = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("WGSL struct {name} not found"))
            .1;
        let naga::TypeInner::Struct { span, .. } = &ty.inner else {
            panic!("{name} is not a struct");
        };
        assert_eq!(
            rust_size, *span as usize,
            "{name}: Rust size {rust_size} != WGSL span {span} (bytes)"
        );
    }
    // Sanity: Light used to be 48 on the Rust side and 64 in WGSL (vec3 pad
    // alignment). Keep the regression sharp.
    assert_eq!(size_of::<rustracer_core::scene::LightGpu>(), 64);
    assert_eq!(size_of::<rustracer_core::material::MaterialGpu>(), 64);
}

/// Headless full-pipeline dispatch with GPU validation: gathers + photon
/// rebuild (trace/count/scatter) + denoise + tonemap must not produce a
/// single validation error, for both a procedural scene and, when present,
/// the bundled ToyCar glTF (the scene that crashed).
#[test]
#[ignore = "requires a GPU adapter; run with --ignored --nocapture"]
fn full_pipeline_dispatches_without_validation_errors() {
    pollster::block_on(async {
        use glam::{Vec2, Vec3};
        use rustracer_core::material::Material;
        use rustracer_core::scene::{Light, Mesh, Scene};
        use rustracer_renderer::{RenderConfig, Renderer};

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = rustracer_renderer::device::select_adapter(&instance, None)
            .await
            .unwrap();
        println!("GPU: {:?}", adapter.get_info());

        let mut scenes: Vec<(&str, Scene)> = Vec::new();

        // 1) Procedural scene: emissive quad + one triangle + environment.
        let mut scene = Scene::default();
        scene.materials.push(Material::lambertian(Vec3::splat(0.5)));
        scene.materials.push(Material::emissive(Vec3::new(10.0, 10.0, 10.0), 1.0));
        scene.meshes.push(std::sync::Arc::new(Mesh::new(
            vec![
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![Vec3::Z; 3],
            vec![Vec2::ZERO; 3],
            vec![0, 1, 2],
            0,
        )));
        scene.meshes.push(std::sync::Arc::new(Mesh::new(
            vec![
                Vec3::new(-1.0, 0.0, -1.0),
                Vec3::new(1.0, 0.0, -1.0),
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(-1.0, 0.0, 1.0),
            ],
            vec![Vec3::Y; 4],
            vec![Vec2::ZERO; 4],
            vec![0, 1, 2, 0, 2, 3],
            1, // emissive floor quad → 2 area lights
        )));
        scene.lights.push(Light::Environment { texture: None, intensity: 0.2 });
        scenes.push(("procedural", scene));

        // 2) Bundled ToyCar glTF if it exists (this is the scene that
        //    crashed with the Light size mismatch).
        let toycar = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("assets/scenes/ToyCar/ToyCar.gltf");
        if toycar.exists() {
            let gltf_scene = rustracer_core::loader::load_gltf(&toycar).unwrap();
            println!(
                "ToyCar: {} meshes, {} materials, {} tri(s), {} light(s), tex roles a/mr/e = {}/{}/{}",
                gltf_scene.meshes.len(),
                gltf_scene.materials.len(),
                gltf_scene.triangle_count(),
                gltf_scene.lights.len(),
                gltf_scene.tex_albedo.len(),
                gltf_scene.tex_mr.len(),
                gltf_scene.tex_emissive.len(),
            );
            scenes.push(("ToyCar", gltf_scene));
        } else {
            println!("ToyCar scene not downloaded; skipping");
        }

        for (name, scene) in scenes {
            println!("── testing {name} ──");
            let config = RenderConfig {
                width: 32,
                height: 32,
                photon_count: 512,
                max_bounces: 6,
                enable_photon_map: true,
                spp: 2,
                ..Default::default()
            };
            let mut renderer = Renderer::new(&adapter, config, &scene).await.unwrap();
            renderer.device.push_error_scope(wgpu::ErrorFilter::Validation);

            renderer.rebuild_photons().unwrap();

            let target = renderer.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("headless_test_target"),
                size: wgpu::Extent3d { width: 32, height: 32, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Bgra8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let mut encoder = renderer.device.create_command_encoder(&Default::default());
            renderer.record_frame(&mut encoder, &target.create_view(&Default::default()));
            renderer.queue.submit(Some(encoder.finish()));
            renderer.device.poll(wgpu::Maintain::Wait);

            let err = renderer.device.pop_error_scope().await;
            assert!(err.is_none(), "{name}: GPU validation error: {err:?}");
            println!("{name}: OK (no validation errors)");
        }
    });
}
