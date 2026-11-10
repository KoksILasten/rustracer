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
        /// Procedural sky gradient color straight up (zenith).
        color_zenith: Vec3,
        /// Procedural sky gradient color at the horizon.
        color_horizon: Vec3,
        intensity: f32,
    },
}

/// GPU-compatible light data for photon emission.
///
/// MUST match the WGSL `Light` struct in `common.wgsl` (64 bytes). WGSL
/// pads `kind: u32` then aligns the trailing `_pad: vec3` to 16 bytes, so
/// the Rust struct needs explicit padding to land the tail at offset 48:
/// ```text
/// 0   data: vec4       16
/// 16  color: vec3      28
/// 28  intensity: f32   32
/// 32  kind: u32        36
/// 36  (alignment gap)  48
/// 48  _pad: vec3       60 → struct size 64
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightGpu {
    pub data: [f32; 4],  // meaning depends on kind
    pub color: [f32; 3], // emission color
    pub intensity: f32,
    pub kind: u32, // 0=point, 1=directional, 2=area, 3=environment
    pub _pad_a: [f32; 3],
    pub _pad_b: [f32; 4],
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

/// The full scene assembly.
#[derive(Debug, Clone)]
pub struct Scene {
    pub meshes: Vec<Arc<Mesh>>,
    pub materials: Vec<Material>,
    /// Every glTF texture, kept in glTF texture order (linear, normalized
    /// [0..1] floats). Normal maps and other non-sampled textures live here
    /// too, but GPU upload only happens for the three role arrays below.
    pub textures: Vec<Texture>,
    /// Albedo textures, uniform size, role-compacted layer indices.
    pub tex_albedo: Vec<Texture>,
    /// Metallic-roughness textures (G = roughness, B = metallic).
    pub tex_mr: Vec<Texture>,
    /// Emissive textures.
    pub tex_emissive: Vec<Texture>,
    /// Tangent-space normal maps.
    pub tex_normal: Vec<Texture>,
    pub lights: Vec<Light>,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            meshes: Vec::new(),
            materials: Vec::new(),
            textures: Vec::new(),
            tex_albedo: Vec::new(),
            tex_mr: Vec::new(),
            tex_emissive: Vec::new(),
            tex_normal: Vec::new(),
            lights: Vec::new(),
        }
    }
}

impl Scene {
    /// Total triangle count across all meshes.
    pub fn triangle_count(&self) -> usize {
        self.meshes.iter().map(|m| m.triangle_count()).sum()
    }

    /// World-space bounds of all geometry, if any exists.
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        let mut bb = crate::bvh::AABB::EMPTY;
        for mesh in &self.meshes {
            for p in &mesh.positions {
                bb.extend(*p);
            }
        }
        if bb.min.x > bb.max.x {
            None
        } else {
            Some((bb.min, bb.max))
        }
    }

    /// Flatten all triangles into a GPU-ready slice.
    pub fn flatten_triangles(&self) -> Vec<TriangleGpu> {
        let mut tris = Vec::with_capacity(self.triangle_count());
        for mesh in &self.meshes {
            for t in 0..mesh.triangle_count() {
                let [i0, i1, i2] = mesh.triangle_indices(t);
                let v0 = mesh
                    .positions
                    .get(i0 as usize)
                    .copied()
                    .unwrap_or(Vec3::ZERO);
                let v1 = mesh
                    .positions
                    .get(i1 as usize)
                    .copied()
                    .unwrap_or(Vec3::ZERO);
                let v2 = mesh
                    .positions
                    .get(i2 as usize)
                    .copied()
                    .unwrap_or(Vec3::ZERO);
                let n0 = mesh.normals.get(i0 as usize).copied().unwrap_or(Vec3::Y);
                let n1 = mesh.normals.get(i1 as usize).copied().unwrap_or(Vec3::Y);
                let n2 = mesh.normals.get(i2 as usize).copied().unwrap_or(Vec3::Y);
                let uv0 = mesh
                    .texcoords
                    .get(i0 as usize)
                    .copied()
                    .unwrap_or(Vec2::ZERO);
                let uv1 = mesh
                    .texcoords
                    .get(i1 as usize)
                    .copied()
                    .unwrap_or(Vec2::ZERO);
                let uv2 = mesh
                    .texcoords
                    .get(i2 as usize)
                    .copied()
                    .unwrap_or(Vec2::ZERO);
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

    /// The classic Cornell box: colored walls, a warm ceiling light, and a
    /// few objects (gold metal, glass). Used by the app when no glTF is
    /// given, and by the `cornell` example to showcase color bleed.
    pub fn cornell_box() -> Scene {
        let mut scene = Scene::default();

        // Materials
        scene.materials.push(Material::lambertian(Vec3::new(0.73, 0.73, 0.73))); // 0: white
        scene.materials.push(Material::lambertian(Vec3::new(0.65, 0.05, 0.05))); // 1: red
        scene.materials.push(Material::lambertian(Vec3::new(0.12, 0.45, 0.15))); // 2: green
        scene.materials.push(Material::lambertian(Vec3::new(0.73, 0.73, 0.73))); // 3: white (back/ceiling)
        scene.materials.push(Material::emissive(Vec3::new(25.0, 20.0, 14.0), 1.0)); // 4: warm light
        scene.materials.push(Material::metal(Vec3::new(0.95, 0.64, 0.54), 0.1)); // 5: gold metal
        scene.materials.push(Material::dielectric(1.5, Vec3::ONE)); // 6: glass
        scene.materials.push(Material::lambertian(Vec3::new(0.1, 0.1, 0.7))); // 7: blue

        // Cornell box walls
        let floor = create_quad(
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
        );
        let ceiling = create_quad(
            Vec3::new(-1.0, 2.0, -1.0),
            Vec3::new(1.0, 2.0, -1.0),
            Vec3::new(1.0, 2.0, 1.0),
            Vec3::new(-1.0, 2.0, 1.0),
        );
        let back_w = create_quad(
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, 2.0, -1.0),
            Vec3::new(-1.0, 2.0, -1.0),
        );
        let left_w = create_quad(
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(-1.0, 2.0, -1.0),
            Vec3::new(-1.0, 2.0, 1.0),
        );
        let right_w = create_quad(
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 2.0, 1.0),
            Vec3::new(1.0, 2.0, -1.0),
        );
        let light_q = create_quad(
            Vec3::new(-0.3, 1.99, -0.3),
            Vec3::new(0.3, 1.99, -0.3),
            Vec3::new(0.3, 1.99, 0.3),
            Vec3::new(-0.3, 1.99, 0.3),
        );

        // Objects
        let tall_box = create_cube(Vec3::new(0.3, -1.0, -0.4), Vec3::new(1.0, -0.4, 0.0), 0);
        let short_box = create_cube(Vec3::new(-0.7, -1.0, 0.1), Vec3::new(-0.4, -0.8, 0.8), 0);
        let gold_cube = create_cube(Vec3::new(-0.2, -1.0, -0.7), Vec3::new(-0.05, -0.65, -0.55), 5);
        let glass_cube = create_cube(Vec3::new(0.5, -0.8, -0.7), Vec3::new(0.6, -0.5, -0.6), 6);

        let walls = [floor, ceiling, back_w, left_w, right_w, light_q];
        let wall_mats = [0, 0, 3, 1, 2, 4];
        for (i, quad) in walls.iter().enumerate() {
            scene.meshes.push(Arc::new(Mesh::new(
                quad.0.clone(),
                quad.1.clone(),
                quad.2.clone(),
                quad.3.clone(),
                wall_mats[i],
            )));
        }
        for (positions, normals, texcoords, indices, mat_id) in
            [tall_box, short_box, gold_cube, glass_cube].iter().flatten()
        {
            scene.meshes.push(Arc::new(Mesh::new(
                positions.clone(),
                normals.clone(),
                texcoords.clone(),
                indices.clone(),
                *mat_id,
            )));
        }

        // Ceiling light quad = walls[5], its two triangles are 10 and 11.
        scene.lights.push(Light::Area { triangle_index: 10, color: Vec3::new(25.0, 20.0, 14.0), intensity: 1.0 });
        scene.lights.push(Light::Area { triangle_index: 11, color: Vec3::new(25.0, 20.0, 14.0), intensity: 1.0 });

        scene
    }

    /// A dark room with a glossy tiled floor that mirrors a thin ceiling
    /// light strip — a specular-reflection stress test. Front wall is left
    /// open so the camera can look in.
    pub fn dark_room() -> Scene {
        let mut scene = Scene::default();

        // Materials
        scene.materials.push(Material::metal(Vec3::new(0.08, 0.08, 0.10), 0.07)); // 0: dark tile A (mirror)
        scene.materials.push(Material::metal(Vec3::new(0.16, 0.16, 0.20), 0.20)); // 1: dark tile B (softer)
        scene.materials.push(Material::lambertian(Vec3::new(0.03, 0.03, 0.04))); // 2: dark wall
        scene.materials.push(Material::lambertian(Vec3::new(0.04, 0.04, 0.05))); // 3: ceiling
        scene.materials.push(Material::emissive(Vec3::new(40.0, 36.0, 28.0), 1.0)); // 4: light strip

        let half = 2.0; // floor is 4 x 4
        let tile = 0.5;
        let n = 8;

        // Glossy tiled floor: an 8x8 checkerboard of two dark reflective metals.
        for iz in 0..n {
            for ix in 0..n {
                let x0 = -half + ix as f32 * tile;
                let z0 = -half + iz as f32 * tile;
                let mat = if (ix + iz) % 2 == 0 { 0 } else { 1 };
                let quad = create_quad(
                    Vec3::new(x0, 0.0, z0 + tile),
                    Vec3::new(x0 + tile, 0.0, z0 + tile),
                    Vec3::new(x0 + tile, 0.0, z0),
                    Vec3::new(x0, 0.0, z0),
                );
                scene.meshes.push(Arc::new(Mesh::new(quad.0, quad.1, quad.2, quad.3, mat)));
            }
        }

        // Walls and ceiling (front left open for the camera).
        let left = create_quad(
            Vec3::new(-half, 0.0, half),
            Vec3::new(-half, 0.0, -half),
            Vec3::new(-half, 3.0, -half),
            Vec3::new(-half, 3.0, half),
        );
        let right = create_quad(
            Vec3::new(half, 0.0, -half),
            Vec3::new(half, 0.0, half),
            Vec3::new(half, 3.0, half),
            Vec3::new(half, 3.0, -half),
        );
        let back = create_quad(
            Vec3::new(-half, 0.0, -half),
            Vec3::new(half, 0.0, -half),
            Vec3::new(half, 3.0, -half),
            Vec3::new(-half, 3.0, -half),
        );
        let ceiling = create_quad(
            Vec3::new(-half, 3.0, -half),
            Vec3::new(half, 3.0, -half),
            Vec3::new(half, 3.0, half),
            Vec3::new(-half, 3.0, half),
        );
        for (q, mat) in [(left, 2), (right, 2), (back, 2), (ceiling, 3)] {
            scene.meshes.push(Arc::new(Mesh::new(q.0, q.1, q.2, q.3, mat)));
        }

        // Thin emissive strip on the ceiling (the only light source).
        let light_tri = scene.triangle_count() as u32;
        let strip = create_quad(
            Vec3::new(-1.5, 2.99, 0.15),
            Vec3::new(1.5, 2.99, 0.15),
            Vec3::new(1.5, 2.99, -0.15),
            Vec3::new(-1.5, 2.99, -0.15),
        );
        scene.meshes.push(Arc::new(Mesh::new(strip.0, strip.1, strip.2, strip.3, 4)));

        let light_color = Vec3::new(40.0, 36.0, 28.0);
        scene.lights.push(Light::Area { triangle_index: light_tri, color: light_color, intensity: 1.0 });
        scene.lights.push(Light::Area { triangle_index: light_tri + 1, color: light_color, intensity: 1.0 });

        scene
    }
}

fn create_quad(
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    v3: Vec3,
) -> (Vec<Vec3>, Vec<Vec3>, Vec<Vec2>, Vec<u32>) {
    let positions = vec![v0, v1, v2, v3];
    let normal = (v1 - v0).cross(v2 - v0).normalize();
    let normals = vec![normal; 4];
    let texcoords = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 1.0),
    ];
    let indices = vec![0, 1, 2, 0, 2, 3];
    (positions, normals, texcoords, indices)
}

/// Create a box from min to max. Returns 6 quads with the given material id.
fn create_cube(
    min: Vec3,
    max: Vec3,
    material_id: usize,
) -> Vec<(Vec<Vec3>, Vec<Vec3>, Vec<Vec2>, Vec<u32>, usize)> {
    let lo = min;
    let hi = max;
    let faces: [(Vec3, Vec3, Vec3, Vec3); 6] = [
        (Vec3::new(lo.x, lo.y, lo.z), Vec3::new(lo.x, hi.y, lo.z), Vec3::new(lo.x, hi.y, hi.z), Vec3::new(lo.x, lo.y, hi.z)), // -X
        (Vec3::new(hi.x, lo.y, hi.z), Vec3::new(hi.x, hi.y, hi.z), Vec3::new(hi.x, hi.y, lo.z), Vec3::new(hi.x, lo.y, lo.z)), // +X
        (Vec3::new(lo.x, lo.y, hi.z), Vec3::new(hi.x, lo.y, hi.z), Vec3::new(hi.x, lo.y, lo.z), Vec3::new(lo.x, lo.y, lo.z)), // -Y
        (Vec3::new(lo.x, hi.y, lo.z), Vec3::new(hi.x, hi.y, lo.z), Vec3::new(hi.x, hi.y, hi.z), Vec3::new(lo.x, hi.y, hi.z)), // +Y
        (Vec3::new(hi.x, lo.y, lo.z), Vec3::new(hi.x, hi.y, lo.z), Vec3::new(lo.x, hi.y, lo.z), Vec3::new(lo.x, lo.y, lo.z)), // -Z
        (Vec3::new(lo.x, lo.y, hi.z), Vec3::new(lo.x, hi.y, hi.z), Vec3::new(hi.x, hi.y, hi.z), Vec3::new(hi.x, lo.y, hi.z)), // +Z
    ];
    faces
        .into_iter()
        .map(|(v0, v1, v2, v3)| {
            let (p, n, t, i) = create_quad(v0, v1, v2, v3);
            (p, n, t, i, material_id)
        })
        .collect()
}
