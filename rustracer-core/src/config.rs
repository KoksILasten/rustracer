//! Scene configuration via TOML descriptor files.
//!
//! Optional companion to glTF loading — overrides material parameters
//! and adds lights beyond what glTF specifies.

use crate::material::{Material, MaterialKind};
use crate::scene::{Light, Scene};
use glam::Vec3;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct SceneDescriptor {
    pub environment: Option<EnvironmentConfig>,
    #[serde(default)]
    pub lights: Vec<LightConfig>,
    #[serde(default)]
    pub materials: Vec<MaterialOverride>,
}

#[derive(Debug, Deserialize)]
pub struct EnvironmentConfig {
    pub texture: Option<String>,
    #[serde(default = "default_intensity")]
    pub intensity: f32,
    pub sky_zenith: Option<[f32; 3]>,
    pub sky_horizon: Option<[f32; 3]>,
}

fn default_intensity() -> f32 { 1.0 }

#[derive(Debug, Deserialize)]
pub struct LightConfig {
    #[serde(rename = "type")]
    pub kind: String,
    pub position: Option<[f32; 3]>,
    pub direction: Option<[f32; 3]>,
    pub color: Option<[f32; 3]>,
    pub intensity: f32,
    pub triangle_index: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct MaterialOverride {
    pub index: Option<usize>,
    pub albedo: Option<[f32; 3]>,
    pub roughness: Option<f32>,
    pub metallic: Option<f32>,
    pub emissive: Option<[f32; 3]>,
    pub ior: Option<f32>,
    pub kind: Option<String>,
}

/// Apply a scene descriptor to an existing scene.
pub fn apply_descriptor(scene: &mut Scene, descriptor: &SceneDescriptor) -> anyhow::Result<()> {
    if let Some(env) = &descriptor.environment {
        let tex_id = if let Some(path) = &env.texture {
            let tex = crate::texture::Texture::from_file(path)?;
            let id = scene.textures.len();
            scene.textures.push(tex);
            Some(id)
        } else if env.sky_zenith.is_some() || env.sky_horizon.is_some() {
            let zenith = env.sky_zenith.unwrap_or([0.3, 0.5, 1.0]);
            let horizon = env.sky_horizon.unwrap_or([1.0, 1.0, 1.0]);
            let tex = crate::texture::Texture::gradient_sky(256, 128, zenith, horizon);
            let id = scene.textures.len();
            scene.textures.push(tex);
            Some(id)
        } else {
            None
        };
        scene.environment_intensity = env.intensity;
        scene.lights.push(Light::Environment { texture: tex_id, intensity: env.intensity });
    }

    for lc in &descriptor.lights {
        let color = Vec3::from_array(lc.color.unwrap_or([1.0; 3]));
        let light = match lc.kind.as_str() {
            "point" => Light::Point {
                position: Vec3::from_array(lc.position.unwrap_or([0.0; 3])),
                color, intensity: lc.intensity,
            },
            "directional" => Light::Directional {
                direction: Vec3::from_array(lc.direction.unwrap_or([0.0, -1.0, 0.0])).normalize(),
                color, intensity: lc.intensity,
            },
            "area" => Light::Area {
                triangle_index: lc.triangle_index.unwrap_or(0),
                color, intensity: lc.intensity,
            },
            _ => { tracing::warn!("Unknown light kind '{}'", lc.kind); continue; }
        };
        scene.lights.push(light);
    }

    for ov in &descriptor.materials {
        if let Some(idx) = ov.index {
            if let Some(mat) = scene.materials.get_mut(idx) { apply_override(mat, ov); }
        } else {
            for mat in &mut scene.materials { apply_override(mat, ov); }
        }
    }

    Ok(())
}

fn apply_override(mat: &mut Material, ov: &MaterialOverride) {
    if let Some(a) = ov.albedo { mat.albedo = Vec3::from_array(a); }
    if let Some(r) = ov.roughness { mat.roughness = r; }
    if let Some(m) = ov.metallic { mat.metallic = m; }
    if let Some(e) = ov.emissive { mat.emissive = Vec3::from_array(e); }
    if let Some(ior) = ov.ior { mat.ior = ior; }
    if let Some(kind) = &ov.kind {
        mat.kind = match kind.as_str() {
            "lambertian" => MaterialKind::Lambertian,
            "metal" => MaterialKind::Metal,
            "dielectric" => MaterialKind::Dielectric,
            "emissive" => MaterialKind::Emissive,
            _ => return,
        };
        match mat.kind {
            MaterialKind::Metal => { mat.metallic = 1.0; }
            MaterialKind::Dielectric => { mat.metallic = 0.0; mat.roughness = 0.0; if ov.ior.is_none() { mat.ior = 1.5; } }
            MaterialKind::Emissive => { mat.emissive = mat.albedo; }
            _ => {}
        }
    }
}

pub fn parse_descriptor(toml_str: &str) -> anyhow::Result<SceneDescriptor> {
    Ok(toml::from_str(toml_str)?)
}
