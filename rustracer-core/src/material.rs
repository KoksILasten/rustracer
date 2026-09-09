//! Material system.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// Material type tag for the GPU.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialKind {
    Lambertian = 0,
    Metal = 1,
    Dielectric = 2,
    Emissive = 3,
}

/// GPU-compatible material data (packed struct).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MaterialGpu {
    pub albedo: [f32; 4],       // rgba for base color, + alpha
    pub emissive: [f32; 3],     // emission color
    pub roughness: f32,         // GGX roughness (0 = mirror, 1 = diffuse)
    pub metallic: f32,          // metalness (0 = dielectric, 1 = metal)
    pub ior: f32,               // index of refraction for dielectrics
    pub kind: u32,              // MaterialKind discriminant
    pub _pad: f32,              // pad to 48 bytes (WGSL array stride)
}

/// Host-side material definition.
#[derive(Debug, Clone)]
pub struct Material {
    pub kind: MaterialKind,
    pub albedo: Vec3,
    pub roughness: f32,
    pub metallic: f32,
    pub ior: f32,
    pub emissive: Vec3,
    /// Optional albedo texture index.
    pub albedo_texture: Option<usize>,
    /// Optional roughness texture index (also used for metallic if
    /// metallic_texture is unset).
    pub roughness_texture: Option<usize>,
    pub metallic_texture: Option<usize>,
    pub normal_texture: Option<usize>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            kind: MaterialKind::Lambertian,
            albedo: Vec3::new(0.8, 0.8, 0.8),
            roughness: 0.5,
            metallic: 0.0,
            ior: 1.5,
            emissive: Vec3::ZERO,
            albedo_texture: None,
            roughness_texture: None,
            metallic_texture: None,
            normal_texture: None,
        }
    }
}

impl Material {
    /// Lambertian (diffuse) material.
    pub fn lambertian(albedo: Vec3) -> Self {
        Self {
            kind: MaterialKind::Lambertian,
            albedo,
            roughness: 1.0,
            ..Default::default()
        }
    }

    /// GGX microfacet conductor (metal).
    pub fn metal(albedo: Vec3, roughness: f32) -> Self {
        Self {
            kind: MaterialKind::Metal,
            albedo,
            roughness: roughness.clamp(0.001, 1.0),
            metallic: 1.0,
            ..Default::default()
        }
    }

    /// Smooth dielectric (glass).
    pub fn dielectric(ior: f32, albedo: Vec3) -> Self {
        Self {
            kind: MaterialKind::Dielectric,
            albedo,
            roughness: 0.0,
            ior,
            ..Default::default()
        }
    }

    /// Emissive (area light).
    pub fn emissive(color: Vec3, intensity: f32) -> Self {
        Self {
            kind: MaterialKind::Emissive,
            albedo: color,
            emissive: color * intensity,
            ..Default::default()
        }
    }

    /// Convert to GPU-compatible packed struct.
    pub fn to_gpu(&self) -> MaterialGpu {
        MaterialGpu {
            albedo: [self.albedo.x, self.albedo.y, self.albedo.z, 1.0],
            emissive: self.emissive.to_array(),
            roughness: self.roughness,
            metallic: self.metallic,
            ior: self.ior,
            kind: self.kind as u32,
            _pad: 0.0,
        }
    }

    /// Return the base reflectance at normal incidence (F0).
    pub fn f0(&self) -> Vec3 {
        match self.kind {
            MaterialKind::Lambertian => self.albedo * std::f32::consts::FRAC_1_PI,
            MaterialKind::Metal => self.albedo,
            MaterialKind::Dielectric => {
                let r = ((self.ior - 1.0) / (self.ior + 1.0)).powi(2);
                Vec3::splat(r)
            }
            MaterialKind::Emissive => Vec3::ZERO,
        }
    }
}
