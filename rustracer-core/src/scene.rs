//! Scene graph: geometry, lights, and the full scene assembly.

use crate::bvh::{AABB, BVHBuilder};
use crate::material::{Material, MaterialGpu};
use crate::texture::Texture;
use glam::{Vec2, Vec3};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Triangle mesh
// ---------------------------------------------------------------------------

/// A triangle mesh with indexed vertices.
#[derive(Debug, Clone)]
pub struct Mesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub texcoords: Vec<Vec2>,
    pub indices: Vec<u32>, // groups of 3 = one triangle
    pub material_id: usize,
}

impl Mesh {
    pub fn new(
        positions: Vec<Vec3>,
        normals: Vec<Vec3>,
        texcoords: Vec<Vec2>,
        indices: Vec<u32>,
        material_id: usize,
    ) -> Self {
        Self {
            positions,
            normals,
            texcoords,
            indices,
            material_id,
        }
    }

    /// Number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Get the three vertex indices for a triangle.
    pub fn triangle_indices(&self, tri: usize) -> [u32; 3] {
        let i = tri * 3;
        [self.indices[i], self.indices[i + 1], self.indices[i + 2]]
    }

    /// Get the three vertex positions for a triangle.
    pub fn triangle_positions(&self, tri: usize) -> [Vec3; 3] {
        let [i0, i1, i2] = self.triangle_indices(tri);
        [
            self.positions[i0 as usize],
            self.positions[i1 as usize],
            self.positions[i2 as usize],
        ]
    }

    /// Compute the AABB of this triangle.
    pub fn triangle_aabb(&self, tri: usize) -> AABB {
        AABB::from_points(self.triangle_positions(tri).iter())
    }

    /// Compute the centroid of this triangle.
    pub fn triangle_centroid(&self, tri: usize) -> Vec3 {
        let p = self.triangle_positions(tri);
        (p[0] + p[1] + p[2]) / 3.0
    }
}

/// GPU-compatible triangle representation (Möller-Trumbore ready).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TriangleGpu {
    pub v0: [f32; 3],
    pub _pad0: f32,
    pub v1: [f32; 3],
    pub _pad1: f32,
    pub v2: [f32; 3],
    pub _pad2: f32,
    pub n0: [f32; 3],
    pub _pad3: f32,
    pub n1: [f32; 3],
    pub _pad4: f32,
    pub n2: [f32; 3],
    pub _pad5: f32,
    pub uv0: [f32; 2],
    pub uv1: [f32; 2],
    pub uv2: [f32; 2],
    pub material_id: u32,
    pub _pad6: f32,
}

// ---------------------------------------------------------------------------
// Lights
// ---------------------------------------------------------------------------

/// Light types in the scene.
#[derive(Debug, Clone)]
pub enum Light {
    Point {
        position: Vec3,
        color: Vec3,
        intensity: f32,
    },
    Directional {
        direction: Vec3,
        color: Vec3,
        intensity: f32,
    },
    Area {
        /// Index into a triangle-primitive list for emissive geometry.
        triangle_index: u32,
        color: Vec3,
        intensity: f32,
    },
    Environment {
        texture: Option<usize>, // HDR environment map index, or None for procedural sky
        intensity: f32,
    },
}

/// GPU-compatible light data for photon emission.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightGpu {
    pub data: [f32; 4],  // meaning depends on kind
    pub color: [f32; 3], // emission color
    pub intensity: f32,
    pub kind: u32, // 0=point, 1=directional, 2=area, 3=environment
    pub _pad: [f32; 3],
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

/// The full scene assembly.
#[derive(Debug, Clone)]
pub struct Scene {
    pub meshes: Vec<Arc<Mesh>>,
    pub materials: Vec<Material>,
    pub textures: Vec<Texture>,
    pub lights: Vec<Light>,
    pub environment_intensity: f32,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            meshes: Vec::new(),
            materials: Vec::new(),
            textures: Vec::new(),
            lights: Vec::new(),
            environment_intensity: 1.0,
        }
    }
}

impl Scene {
    /// Total triangle count across all meshes.
    pub fn triangle_count(&self) -> usize {
        self.meshes.iter().map(|m| m.triangle_count()).sum()
    }

    /// Flatten all triangles into a GPU-ready slice.
    pub fn flatten_triangles(&self) -> Vec<TriangleGpu> {
        let mut tris = Vec::with_capacity(self.triangle_count());
        for mesh in &self.meshes {
            for t in 0..mesh.triangle_count() {
                let [i0, i1, i2] = mesh.triangle_indices(t);
                let v0 = mesh.positions.get(i0 as usize).copied().unwrap_or(Vec3::ZERO);
                let v1 = mesh.positions.get(i1 as usize).copied().unwrap_or(Vec3::ZERO);
                let v2 = mesh.positions.get(i2 as usize).copied().unwrap_or(Vec3::ZERO);
                let n0 = mesh.normals.get(i0 as usize).copied().unwrap_or(Vec3::Y);
                let n1 = mesh.normals.get(i1 as usize).copied().unwrap_or(Vec3::Y);
                let n2 = mesh.normals.get(i2 as usize).copied().unwrap_or(Vec3::Y);
                let uv0 = mesh.texcoords.get(i0 as usize).copied().unwrap_or(Vec2::ZERO);
                let uv1 = mesh.texcoords.get(i1 as usize).copied().unwrap_or(Vec2::ZERO);
                let uv2 = mesh.texcoords.get(i2 as usize).copied().unwrap_or(Vec2::ZERO);
                tris.push(TriangleGpu {
                    v0: v0.to_array(),
                    _pad0: 0.0,
                    v1: v1.to_array(),
                    _pad1: 0.0,
                    v2: v2.to_array(),
                    _pad2: 0.0,
                    n0: n0.to_array(),
                    _pad3: 0.0,
                    n1: n1.to_array(),
                    _pad4: 0.0,
                    n2: n2.to_array(),
                    _pad5: 0.0,
                    uv0: uv0.to_array(),
                    uv1: uv1.to_array(),
                    uv2: uv2.to_array(),
                    material_id: mesh.material_id as u32,
                    _pad6: 0.0,
                });
            }
        }
        tris
    }

    /// Build a BVH over all triangles.
    pub fn build_bvh(&self) -> crate::bvh::BVH {
        let mut centroids = Vec::new();
        let mut bboxes = Vec::new();
        for mesh in &self.meshes {
            for t in 0..mesh.triangle_count() {
                centroids.push(mesh.triangle_centroid(t));
                bboxes.push(mesh.triangle_aabb(t));
            }
        }
        BVHBuilder::new(centroids, bboxes).build()
    }

    /// Flatten materials to GPU format.
    pub fn flatten_materials(&self) -> Vec<MaterialGpu> {
        self.materials.iter().map(|m| m.to_gpu()).collect()
    }
}
