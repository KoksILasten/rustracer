//! glTF 2.0 scene loader.
//!
//! Loads `.gltf`/`.glb` files and populates a `Scene` with meshes,
//! materials, and light extraction from emissive materials.

use crate::material::Material;
use crate::scene::{Light, Mesh, Scene};
use crate::texture::Texture;
use glam::{Vec2, Vec3};
use std::path::Path;

/// Load a glTF 2.0 scene from a file path.
pub fn load_gltf(path: impl AsRef<Path>) -> anyhow::Result<Scene> {
    let (document, buffers, images) = gltf::import(path)?;
    let mut scene = Scene::default();

    // ── Textures ──
    for img in images {
        let width = img.width;
        let height = img.height;
        let pixels = img.pixels;
        let data: Vec<[f32; 4]> = match img.format {
            gltf::image::Format::R8G8B8 => pixels
                .chunks_exact(3)
                .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0, 1.0])
                .collect(),
            gltf::image::Format::R8G8B8A8 => pixels
                .chunks_exact(4)
                .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0, c[3] as f32 / 255.0])
                .collect(),
            gltf::image::Format::R16G16B16 | gltf::image::Format::R16G16B16A16 => {
                let bpp = match img.format {
                    gltf::image::Format::R16G16B16 => 6,
                    _ => 8,
                };
                pixels.chunks_exact(bpp).map(|c| {
                    [
                        u16::from_le_bytes([c[0], c[1]]) as f32 / 65535.0,
                        u16::from_le_bytes([c[2], c[3]]) as f32 / 65535.0,
                        u16::from_le_bytes([c[4], c[5]]) as f32 / 65535.0,
                        if bpp >= 8 { u16::from_le_bytes([c[6], c[7]]) as f32 / 65535.0 } else { 1.0 },
                    ]
                }).collect()
            }
            _ => pixels.chunks(4).map(|c| {
                [c.first().copied().unwrap_or(255) as f32 / 255.0,
                 c.get(1).copied().unwrap_or(255) as f32 / 255.0,
                 c.get(2).copied().unwrap_or(255) as f32 / 255.0,
                 c.get(3).copied().unwrap_or(255) as f32 / 255.0]
            }).collect(),
        };
        scene.textures.push(Texture { width, height, data });
    }

    // ── Materials ──
    for mat in document.materials() {
        let pbr = mat.pbr_metallic_roughness();
        let albedo = Vec3::from_array(pbr.base_color_factor()[..3].try_into().unwrap());
        let roughness = pbr.roughness_factor();
        let metallic = pbr.metallic_factor();
        let emissive = Vec3::from_array(mat.emissive_factor());

        let albedo_tex: Option<usize> = pbr.base_color_texture()
            .map(|t: gltf::texture::Info<'_>| t.texture().source().index());
        let roughness_tex: Option<usize> = pbr.metallic_roughness_texture()
            .map(|t: gltf::texture::Info<'_>| t.texture().source().index());
        let normal_tex: Option<usize> = mat.normal_texture()
            .map(|t: gltf::material::NormalTexture<'_>| t.texture().source().index());

        let kind = if emissive.length() > 0.01 && mat.emissive_texture().is_none() {
            // Factor-based emissives become area lights. Textured emissives are
            // skipped: the renderer cannot sample the emissive texture yet, and
            // classifying the whole material as emissive would turn the entire
            // mesh into a glowing area light.
            crate::material::MaterialKind::Emissive
        } else if metallic > 0.5 {
            crate::material::MaterialKind::Metal
        } else {
            crate::material::MaterialKind::Lambertian
        };

        scene.materials.push(Material {
            kind, albedo, roughness, metallic, ior: 1.5, emissive,
            albedo_texture: albedo_tex, roughness_texture: roughness_tex,
            metallic_texture: None, normal_texture: normal_tex,
        });
    }

    if scene.materials.is_empty() {
        scene.materials.push(Material::lambertian(Vec3::new(0.8, 0.8, 0.8)));
    }

    // ── Meshes ──
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buf: gltf::Buffer<'_>| Some(&buffers[buf.index()]));

            let positions: Vec<Vec3> = reader.read_positions()
                .map(|iter: gltf::mesh::util::ReadPositions<'_>| iter.map(Vec3::from_array).collect())
                .unwrap_or_default();
            if positions.is_empty() { continue; }

            let normals: Vec<Vec3> = reader.read_normals()
                .map(|iter: gltf::mesh::util::ReadNormals<'_>| iter.map(Vec3::from_array).collect())
                .unwrap_or_else(|| vec![Vec3::Y; positions.len()]);

            let texcoords: Vec<Vec2> = reader.read_tex_coords(0)
                .map(|iter: gltf::mesh::util::ReadTexCoords<'_>| iter.into_f32().map(Vec2::from_array).collect())
                .unwrap_or_else(|| vec![Vec2::ZERO; positions.len()]);

            let indices: Vec<u32> = reader.read_indices()
                .map(|iter: gltf::mesh::util::ReadIndices<'_>| iter.into_u32().collect())
                .unwrap_or_else(|| (0..positions.len() as u32).collect());

            let material_id = primitive.material().index()
                .unwrap_or(scene.materials.len().saturating_sub(1));

            scene.meshes.push(std::sync::Arc::new(Mesh::new(
                positions, normals, texcoords, indices, material_id,
            )));
        }
    }

    if scene.lights.is_empty() {
        scene.lights.push(Light::Environment { texture: None, intensity: 1.0 });
    }

    // Area lights from emissive materials
    let mut tri_offset = 0u32;
    for mesh in &scene.meshes {
        if let Some(mat) = scene.materials.get(mesh.material_id) {
            if mat.kind == crate::material::MaterialKind::Emissive {
                for t in 0..mesh.triangle_count() {
                    scene.lights.push(Light::Area {
                        triangle_index: tri_offset + t as u32,
                        color: mat.emissive, intensity: 1.0,
                    });
                }
            }
        }
        tri_offset += mesh.triangle_count() as u32;
    }

    Ok(scene)
}
