//! Headless scene-loading check: prints stats for a glTF scene without
//! opening a window. Run with:
//! cargo run -p rustracer-app --example loadcheck -- <scene.gltf>

use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: loadcheck <scene.gltf>");
    let scene = rustracer_core::loader::load_gltf(PathBuf::from(path.clone()))?;
    println!("OK {}", path);
    println!("  meshes: {}", scene.meshes.len());
    println!("  materials: {}", scene.materials.len());
    println!("  textures: {}", scene.textures.len());
    println!(
        "  texture roles (albedo/mr/emissive/normal): {}/{}/{}/{}",
        scene.tex_albedo.len(),
        scene.tex_mr.len(),
        scene.tex_emissive.len(),
        scene.tex_normal.len()
    );
    let tris: usize = scene.meshes.iter().map(|m| m.triangle_count()).sum();
    println!("  triangles: {}", tris);
    let area = scene.lights.iter().filter(|l| matches!(l, rustracer_core::scene::Light::Area { .. })).count();
    let env = scene.lights.len() - area;
    println!("  lights: {} total ({} area from emissives, {} environment)", scene.lights.len(), area, env);
    Ok(())
}
