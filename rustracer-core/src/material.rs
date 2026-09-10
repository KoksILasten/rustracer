//! Material system.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// Sentinel meaning "no texture" in a `MaterialGpu` texture slot.
pub const TEX_NONE: u32 = u32::MAX;

/// Material flag bits (kept in sync with `common.wgsl`).
pub const MAT_FLAG_ALPHA_CUTOUT: u32 = 1 << 0;

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
///
/// Layout (64 bytes, matches `common.wgsl` exactly):
/// ```text
/// offset  size  field
/// 0       16    albedo   (vec4)
/// 16      12    emissive (vec3)
/// 28      4     roughness
/// 32      4     metallic
/// 36      4     ior
/// 40      4     kind     (MaterialKind discriminant)
/// 44      4     flags    (MAT_FLAG_*)
/// 48      4     tex_albedo   (layer in the albedo texture array, TEX_NONE if absent)
/// 52      4     tex_mr       (metallic-roughness array layer,   TEX_NONE if absent)
/// 56      4     tex_emissive (emissive array layer,             TEX_NONE if absent)
/// 60      4     tex_normal   (normal-map array layer,           TEX_NONE if absent)
/// ```
///
/// Textures are uploaded as one `texture_2d_array` per *role*. The loader
/// compacts each role's textures so that every array has a uniform size and
/// assigns role-local layer indices; `TEX_NONE` means "no texture for this
/// role" (shaders then fall back to the plain factor values).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MaterialGpu {
    pub albedo: [f32; 4],       // base color, linear RGB + alpha
    pub emissive: [f32; 3],     // emission color (factor * texture average)
    pub roughness: f32,         // GGX roughness (0 = mirror, 1 = diffuse)
    pub metallic: f32,          // metalness (0 = dielectric, 1 = metal)
    pub ior: f32,               // index of refraction for dielectrics
    pub kind: u32,              // MaterialKind discriminant
    pub flags: u32,             // MAT_FLAG_* bits
    pub tex_albedo: u32,        // albedo texture-array layer or TEX_NONE
    pub tex_mr: u32,            // metallic-roughness (R=occlusion G=rough B=metal) layer
    pub tex_emissive: u32,      // emissive texture-array layer
    pub tex_normal: u32,        // normal-map texture-array layer
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
    /// Layer index into `Scene::tex_albedo`.
    pub albedo_texture: Option<usize>,
    /// Layer index into `Scene::tex_mr` (metallic-roughness, G/B channels).
    pub roughness_texture: Option<usize>,
    /// Kept for compatibility; unused (metallic shares the MR texture).
    pub metallic_texture: Option<usize>,
    /// Layer index into `Scene::tex_emissive`.
    pub emissive_texture: Option<usize>,
    /// Layer index into `Scene::tex_normal` (tangent-space normal map).
    pub normal_texture: Option<usize>,
    /// Kill fragments below alpha 0.5 (glTF `alphaMode: MASK`).
    pub alpha_cutout: bool,
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
            emissive_texture: None,
            normal_texture: None,
            alpha_cutout: false,
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
        let mut flags = 0u32;
        if self.alpha_cutout {
            flags |= MAT_FLAG_ALPHA_CUTOUT;
        }
        MaterialGpu {
            albedo: [self.albedo.x, self.albedo.y, self.albedo.z, 1.0],
            emissive: self.emissive.to_array(),
            roughness: self.roughness,
            metallic: self.metallic,
            ior: self.ior,
            kind: self.kind as u32,
            flags,
            tex_albedo: self.albedo_texture.map(|i| i as u32).unwrap_or(TEX_NONE),
            tex_mr: self.roughness_texture.map(|i| i as u32).unwrap_or(TEX_NONE),
            tex_emissive: self.emissive_texture.map(|i| i as u32).unwrap_or(TEX_NONE),
            tex_normal: self.normal_texture.map(|i| i as u32).unwrap_or(TEX_NONE),
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

    /// Does this material actually emit light (used to build area lights)?
    pub fn emits_light(&self) -> bool {
        self.emissive_texture.is_some() || self.emissive.length() > 1e-4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_material_layout_is_64_bytes() {
        // Must match the WGSL `Material` struct in common.wgsl.
        assert_eq!(std::mem::size_of::<MaterialGpu>(), 64);
        assert_eq!(std::mem::align_of::<MaterialGpu>(), 4.max(4));
    }

    #[test]
    fn gpu_material_offsets_match_wgsl() {
        let m = MaterialGpu {
            albedo: [1.0; 4],
            emissive: [2.0; 3],
            roughness: 3.0,
            metallic: 4.0,
            ior: 5.0,
            kind: 6,
            flags: 7,
            tex_albedo: 8,
            tex_mr: 9,
            tex_emissive: 10,
            tex_normal: 11,
        };
        let bytes = bytemuck::bytes_of(&m);
        assert_eq!(f32::from_le_bytes(bytes[28..32].try_into().unwrap()), 3.0); // roughness
        assert_eq!(f32::from_le_bytes(bytes[32..36].try_into().unwrap()), 4.0); // metallic
        assert_eq!(u32::from_le_bytes(bytes[40..44].try_into().unwrap()), 6);   // kind
        assert_eq!(u32::from_le_bytes(bytes[44..48].try_into().unwrap()), 7);   // flags
        assert_eq!(u32::from_le_bytes(bytes[48..52].try_into().unwrap()), 8);   // tex_albedo
        assert_eq!(u32::from_le_bytes(bytes[56..60].try_into().unwrap()), 10);  // tex_emissive
        assert_eq!(u32::from_le_bytes(bytes[60..64].try_into().unwrap()), 11);  // tex_normal
    }

    #[test]
    fn texture_none_is_max_u32() {
        assert_eq!(TEX_NONE, u32::MAX);
    }
}
