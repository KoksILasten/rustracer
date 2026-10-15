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
