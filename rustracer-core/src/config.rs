//! Scene configuration via TOML descriptor files (`scene.toml`).
//!
//! Optional companion to glTF loading — overrides material parameters and
//! adds lights beyond what glTF specifies. The app loads a `scene.toml` from
//! the glTF's directory and applies it after `load_gltf`.

use crate::material::{Material, MaterialKind};
use crate::scene::{Light, Scene};
use glam::Vec3;
use serde::Deserialize;

/// Default procedural-sky colors (match the WGSL gradient in gather.wgsl).
const SKY_ZENITH: [f32; 3] = [0.55, 0.6, 0.75];
const SKY_HORIZON: [f32; 3] = [0.95, 0.93, 0.9];

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
    #[serde(default = "default_intensity")]
    pub intensity: f32,
    pub sky_zenith: Option<[f32; 3]>,
    pub sky_horizon: Option<[f32; 3]>,
}

fn default_intensity() -> f32 {
    1.0
}

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

/// Apply a scene descriptor to an already-loaded scene.
pub fn apply_descriptor(scene: &mut Scene, descriptor: &SceneDescriptor) {
    if let Some(env) = &descriptor.environment {
        let zenith = Vec3::from_array(env.sky_zenith.unwrap_or(SKY_ZENITH));
        let horizon = Vec3::from_array(env.sky_horizon.unwrap_or(SKY_HORIZON));
        // Replace any existing environment light (e.g. the loader's default)
        // so the override actually takes effect instead of leaving two.
        scene
            .lights
            .retain(|l| !matches!(l, Light::Environment { .. }));
        scene.lights.push(Light::Environment {
            color_zenith: zenith,
            color_horizon: horizon,
            intensity: env.intensity,
        });
    }

    for lc in &descriptor.lights {
        let color = Vec3::from_array(lc.color.unwrap_or([1.0; 3]));
        let light = match lc.kind.as_str() {
            "point" => Light::Point {
                position: Vec3::from_array(lc.position.unwrap_or([0.0; 3])),
                color,
                intensity: lc.intensity,
            },
            "directional" => Light::Directional {
                direction: Vec3::from_array(lc.direction.unwrap_or([0.0, -1.0, 0.0])).normalize(),
                color,
                intensity: lc.intensity,
            },
            "area" => Light::Area {
                triangle_index: lc.triangle_index.unwrap_or(0),
                color,
                intensity: lc.intensity,
            },
            _ => {
                tracing::warn!("Unknown light kind '{}'", lc.kind);
                continue;
            }
        };
        scene.lights.push(light);
    }

    for ov in &descriptor.materials {
        if let Some(idx) = ov.index {
            if idx < scene.materials.len() {
                apply_override(&mut scene.materials[idx], ov);
            } else {
                tracing::warn!(
                    "Material override index {idx} out of range ({} materials)",
                    scene.materials.len()
                );
            }
        } else {
            for mat in &mut scene.materials {
                apply_override(mat, ov);
            }
        }
    }
}

fn apply_override(mat: &mut Material, ov: &MaterialOverride) {
    // The kind sets defaults first; explicit field overrides win afterwards.
    if let Some(kind) = &ov.kind {
        match kind.as_str() {
            "lambertian" => mat.kind = MaterialKind::Lambertian,
            "metal" => mat.kind = MaterialKind::Metal,
            "dielectric" => mat.kind = MaterialKind::Dielectric,
            "emissive" => mat.kind = MaterialKind::Emissive,
            other => tracing::warn!("Unknown material kind '{other}'"),
        }
    }
    match mat.kind {
        MaterialKind::Metal if ov.metallic.is_none() => mat.metallic = 1.0,
        MaterialKind::Dielectric => {
            if ov.metallic.is_none() {
                mat.metallic = 0.0;
            }
            if ov.roughness.is_none() {
                mat.roughness = 0.0;
            }
            if ov.ior.is_none() {
                mat.ior = 1.5;
            }
        }
        _ => {}
    }
    if let Some(a) = ov.albedo {
        mat.albedo = Vec3::from_array(a);
    }
    if let Some(r) = ov.roughness {
        mat.roughness = r;
    }
    if let Some(m) = ov.metallic {
        mat.metallic = m;
    }
    if let Some(e) = ov.emissive {
        mat.emissive = Vec3::from_array(e);
    }
    if let Some(ior) = ov.ior {
        mat.ior = ior;
    }
    if mat.kind == MaterialKind::Emissive && ov.emissive.is_none() {
        mat.emissive = mat.albedo;
    }
}

pub fn parse_descriptor(toml_str: &str) -> anyhow::Result<SceneDescriptor> {
    Ok(toml::from_str(toml_str)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn environment_override_replaces_the_default_light() {
        let mut scene = Scene::default();
        scene.lights.push(Light::Environment {
            color_zenith: Vec3::ZERO,
            color_horizon: Vec3::ZERO,
            intensity: 1.0,
        });
        let desc: SceneDescriptor = parse_descriptor(
            r#"
            [environment]
            intensity = 2.0
            sky_zenith = [0.1, 0.2, 0.3]
            sky_horizon = [0.7, 0.8, 0.9]
            "#,
        )
        .unwrap();
        apply_descriptor(&mut scene, &desc);
        assert_eq!(scene.lights.len(), 1, "override must replace, not append");
        match &scene.lights[0] {
            Light::Environment {
                color_zenith,
                color_horizon,
                intensity,
            } => {
                assert_eq!(*intensity, 2.0);
                assert_eq!(color_zenith, &Vec3::new(0.1, 0.2, 0.3));
                assert_eq!(color_horizon, &Vec3::new(0.7, 0.8, 0.9));
            }
            _ => panic!("expected an environment light"),
        }
    }

    #[test]
    fn material_kind_sets_defaults_and_explicit_values_win() {
        let mut scene = Scene::default();
        scene
            .materials
            .push(Material::lambertian(Vec3::new(0.5, 0.5, 0.5)));
        let desc: SceneDescriptor = parse_descriptor(
            r#"
            [[materials]]
            index = 0
            kind = "dielectric"
            ior = 1.3
            "#,
        )
        .unwrap();
        apply_descriptor(&mut scene, &desc);
        let m = &scene.materials[0];
        assert_eq!(m.kind, MaterialKind::Dielectric);
        assert_eq!(m.ior, 1.3, "explicit ior wins over the dielectric default");
        assert_eq!(m.metallic, 0.0);
        assert_eq!(m.roughness, 0.0);
    }
}
