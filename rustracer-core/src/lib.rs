//! Rustracer core crate — shared types independent of the GPU backend.
//!
//! This crate defines the scene representation, geometric primitives,
//! acceleration structures, materials, textures, and lights. All GPU-bound
//! types use `#[repr(C)]` layouts compatible with WGSL structs.

pub mod bvh;
pub mod camera;
pub mod config;
pub mod loader;
pub mod material;
pub mod math;
pub mod scene;
pub mod texture;
