//! Math types for Rustracer.
//!
//! Re-exports `glam` vector/matrix types with `bytemuck` support for
//! zero-copy GPU transfers. All types are `#[repr(C)]` and compatible
//! with WGSL layouts (e.g., `vec3<f32>` = 3 x f32, 12 bytes / 16 bytes
//! with padding).

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Re-exports — prefer these in Rustracer code so switching math backends
// is a single-file change.
// ---------------------------------------------------------------------------
pub use glam::{
    BVec3, BVec3A, BVec4, BVec4A, DMat2, DMat3, DMat4, DVec2, DVec3, DVec4, EulerRot, I16Vec2,
    I16Vec3, I16Vec4, I64Vec2, I64Vec3, I64Vec4, IVec2, IVec3, IVec4, Mat2, Mat3, Mat3A, Mat4,
    Quat, U16Vec2, U16Vec3, U16Vec4, U64Vec2, U64Vec3, U64Vec4, UVec2, UVec3, UVec4, Vec2, Vec3,
    Vec3A, Vec4,
};

// ---------------------------------------------------------------------------
// Bytemuck trait impls — these are already implemented by glam when the
// "bytemuck" feature is enabled, so we just re-assert.
// ---------------------------------------------------------------------------

// Safety: glam types with the "bytemuck" feature are Pod + Zeroable.
// We assert this at compile time with these static checks.
const _: fn() = || {
    // If these don't compile, the glam "bytemuck" feature may be disabled.
    fn assert_pod<T: Pod>() {}
    fn assert_zeroable<T: Zeroable>() {}
    assert_pod::<Vec3>();
    assert_zeroable::<Vec3>();
};

// ---------------------------------------------------------------------------
// GPU-compatible packed types
// ---------------------------------------------------------------------------

/// A 3D vector with explicit padding to 16 bytes (WGSL `vec3<f32>` alignment).
/// Use this in GPU-bound structs where vec3 alignment matters.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vec3Padded {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub _pad: f32,
}

impl From<Vec3> for Vec3Padded {
    fn from(v: Vec3) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
            _pad: 0.0,
        }
    }
}

impl From<Vec3Padded> for Vec3 {
    fn from(v: Vec3Padded) -> Self {
        Vec3::new(v.x, v.y, v.z)
    }
}

/// A 3x3 column-major matrix padded for WGSL alignment.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Mat3Padded {
    pub col0: Vec3Padded,
    pub col1: Vec3Padded,
    pub col2: Vec3Padded,
}

impl From<Mat3> for Mat3Padded {
    fn from(m: Mat3) -> Self {
        Self {
            col0: m.x_axis.into(),
            col1: m.y_axis.into(),
            col2: m.z_axis.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Math utilities
// ---------------------------------------------------------------------------

/// Reflect a vector around a normal.
#[inline]
pub fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    v - 2.0 * v.dot(n) * n
}

/// Refract a vector through a surface with the given eta ratio.
/// Returns `None` for total internal reflection.
#[inline]
pub fn refract(v: Vec3, n: Vec3, eta: f32) -> Option<Vec3> {
    let cos_theta = (-v).dot(n).min(1.0);
    let sin2_theta = 1.0 - cos_theta * cos_theta;
    let r_out_perp = eta * (v + cos_theta * n);
    let r_out_parallel = -(1.0 - r_out_perp.length_squared()).abs().sqrt() * n;
    if sin2_theta > 1.0 {
        None // TIR
    } else {
        Some(r_out_perp + r_out_parallel)
    }
}

/// Fresnel reflectance (Schlick approximation).
#[inline]
pub fn fresnel_schlick(cos_theta: f32, f0: Vec3) -> Vec3 {
    f0 + (Vec3::ONE - f0) * (1.0 - cos_theta).clamp(0.0, 1.0).powi(5)
}

/// Cosine-weighted hemisphere sample given two uniform random numbers.
#[inline]
pub fn sample_cosine_hemisphere(u1: f32, u2: f32) -> Vec3 {
    let phi = 2.0 * std::f32::consts::PI * u1;
    let cos_theta = u2.sqrt();
    let sin_theta = (1.0 - u2).sqrt();
    Vec3::new(phi.cos() * sin_theta, cos_theta, phi.sin() * sin_theta)
}

/// Transform a local hemisphere direction to world space given a normal.
#[inline]
pub fn to_world(local_dir: Vec3, normal: Vec3) -> Vec3 {
    let up = if normal.y.abs() < 0.999 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let tangent = normal.cross(up).normalize();
    let bitangent = normal.cross(tangent);
    tangent * local_dir.x + normal * local_dir.y + bitangent * local_dir.z
}
