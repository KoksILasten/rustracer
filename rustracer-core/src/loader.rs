//! glTF 2.0 scene loader.
//!
//! Loads `.gltf`/`.glb` files and populates a `Scene` with:
//! - world-space meshes baked from the default scene's node hierarchy
//!   (transforms, mirrored winding, strip/fan triangulation),
//! - materials with metallic-roughness + base-color/emissive textures
//!   (sRGB decoded on the CPU; role-compacted into uniform-size arrays),
//! - area lights extracted from emissive materials,
//! - an environment light when nothing else emits.
//!
//! Unsupported features (skins, morph targets, line/point primitives,
//! alpha BLEND) are reported loudly instead of being silently mis-rendered.

use crate::material::{Material, MaterialKind};
use crate::scene::{Light, Mesh, Scene};
use crate::texture::Texture;
use glam::{Mat4, Vec2, Vec3};
use std::path::Path;

/// Maximum layers in a role texture array (device `max_texture_array_layers`
/// is at least 256 on all real adapters; stay clear of it).
const MAX_ROLE_LAYERS: usize = 240;

// ---------------------------------------------------------------------------
// Texture decoding
// ---------------------------------------------------------------------------

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn decode_to_rgba(img: &gltf::image::Data) -> Option<Vec<[f32; 4]>> {
    use gltf::image::Format as F;
    let px = &img.pixels;
    let mut out = Vec::with_capacity((img.width * img.height) as usize);
    match img.format {
        F::R8 => {
            for &b in px {
                let v = b as f32 / 255.0;
                out.push([v, v, v, 1.0]);
            }
        }
        F::R8G8B8 => {
            for c in px.chunks_exact(3) {
                out.push([c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0, 1.0]);
            }
        }
        F::R8G8B8A8 => {
            for c in px.chunks_exact(4) {
                out.push([
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                    c[3] as f32 / 255.0,
                ]);
            }
        }
        F::R16G16B16 => {
            for c in px.chunks_exact(6) {
                let r = u16::from_le_bytes([c[0], c[1]]) as f32 / 65535.0;
                let g = u16::from_le_bytes([c[2], c[3]]) as f32 / 65535.0;
                let b = u16::from_le_bytes([c[4], c[5]]) as f32 / 65535.0;
                out.push([r, g, b, 1.0]);
            }
        }
        F::R16G16B16A16 => {
            for c in px.chunks_exact(8) {
                out.push([
                    u16::from_le_bytes([c[0], c[1]]) as f32 / 65535.0,
                    u16::from_le_bytes([c[2], c[3]]) as f32 / 65535.0,
                    u16::from_le_bytes([c[4], c[5]]) as f32 / 65535.0,
                    u16::from_le_bytes([c[6], c[7]]) as f32 / 65535.0,
                ]);
            }
        }
        F::R32G32B32FLOAT => {
            for c in px.chunks_exact(12) {
                out.push([
                    f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                    f32::from_le_bytes([c[4], c[5], c[6], c[7]]),
                    f32::from_le_bytes([c[8], c[9], c[10], c[11]]),
                    1.0,
                ]);
            }
        }
        F::R32G32B32A32FLOAT => {
            for c in px.chunks_exact(16) {
                out.push([
                    f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                    f32::from_le_bytes([c[4], c[5], c[6], c[7]]),
                    f32::from_le_bytes([c[8], c[9], c[10], c[11]]),
                    f32::from_le_bytes([c[12], c[13], c[14], c[15]]),
                ]);
            }
        }
        other => {
            tracing::error!("Unsupported glTF image format {:?}", other);
            return None;
        }
    }
    if out.len() == (img.width * img.height) as usize {
        Some(out)
    } else {
        tracing::error!("glTF image byte size does not match its dimensions");
        None
    }
}

fn decode_srgb(tex: &mut Texture) {
    for p in &mut tex.data {
        p[0] = srgb_to_linear(p[0]);
        p[1] = srgb_to_linear(p[1]);
        p[2] = srgb_to_linear(p[2]);
    }
}

// ---------------------------------------------------------------------------
// Role compaction
// ---------------------------------------------------------------------------

/// Copy every used master texture into a uniform-size role array.
/// Returns the role array and, for every master index, the role layer (or
/// None when the texture is unused or was dropped).
fn build_role(master: &[Texture], used: &[Option<usize>], srgb: bool) -> (Vec<Texture>, Vec<Option<usize>>) {
    let mut map: Vec<Option<usize>> = vec![None; master.len()];
    if master.is_empty() {
        return (Vec::new(), map);
    }
    // Reference dimensions: first texture that is actually used.
    let first = used.iter().flatten().next().copied().unwrap_or(0);
    let (ref_w, ref_h) = (master[first].width, master[first].height);

    let mut role = Vec::new();
    for m in used.iter().flatten() {
        if map[*m].is_some() {
            continue;
        }
        let tex = &master[*m];
        if tex.width != ref_w || tex.height != ref_h {
            tracing::warn!(
                "Dropping {}x{} texture from an array of {}x{} textures (mixed sizes are not supported yet)",
                tex.width, tex.height, ref_w, ref_h
            );
            continue;
        }
        if role.len() >= MAX_ROLE_LAYERS {
            tracing::warn!("Too many textures for one role array (max {MAX_ROLE_LAYERS}); dropping the rest");
            continue;
        }
        let mut copy = tex.clone();
        if srgb {
            decode_srgb(&mut copy);
        }
        map[*m] = Some(role.len());
        role.push(copy);
    }
    (role, map)
}

/// Like `build_role`, but scales the X/Y channels of every texture by
/// `scales[master_index]` (used to bake glTF normalTexture.scale).
fn build_role_scaled(
    master: &[Texture],
    used: &[Option<usize>],
    scales: &[f32],
) -> (Vec<Texture>, Vec<Option<usize>>) {
    let mut map: Vec<Option<usize>> = vec![None; master.len()];
    if master.is_empty() {
        return (Vec::new(), map);
    }
    let first = used.iter().flatten().next().copied().unwrap_or(0);
    let (ref_w, ref_h) = (master[first].width, master[first].height);

    let mut role = Vec::new();
    for m in used.iter().flatten() {
        if map[*m].is_some() {
            continue;
        }
        let tex = &master[*m];
        if tex.width != ref_w || tex.height != ref_h {
            tracing::warn!(
                "Dropping {}x{} texture from an array of {}x{} textures (mixed sizes are not supported yet)",
                tex.width, tex.height, ref_w, ref_h
            );
            continue;
        }
        if role.len() >= MAX_ROLE_LAYERS {
            tracing::warn!(
                "Too many textures for one role array (max {MAX_ROLE_LAYERS}); dropping the rest"
            );
            continue;
        }
        let scale = scales.get(*m).copied().unwrap_or(1.0);
        let mut copy = tex.clone();
        if (scale - 1.0).abs() > 1e-6 {
            for p in &mut copy.data {
                p[0] *= scale;
                p[1] *= scale;
            }
        }
        map[*m] = Some(role.len());
        role.push(copy);
    }
    (role, map)
}

// ---------------------------------------------------------------------------
// Main loader
// ---------------------------------------------------------------------------

/// Load a glTF 2.0 scene from a file path.
pub fn load_gltf(path: impl AsRef<Path>) -> anyhow::Result<Scene> {
    let (document, buffers, images) = gltf::import(path)?;
    let mut scene = Scene::default();

    // ── Textures (master list, in glTF texture order) ──
    for tex in document.textures() {
        let img = &images[tex.source().index()];
        match decode_to_rgba(img) {
            Some(data) => scene.textures.push(Texture { width: img.width, height: img.height, data }),
            None => scene.textures.push(Texture::solid([0.5, 0.5, 0.5, 1.0])),
        }
    }

    // ── Materials ──
    // Each material first records raw glTF texture indices; the role arrays
    // are built afterwards and the indices remapped to role layers.
    struct Pending {
        material: Material,
        albedo: Option<usize>,
        mr: Option<usize>,
        emissive: Option<usize>,
        normal: Option<usize>,
        normal_scale: f32,
    }

    let mut pending: Vec<Pending> = Vec::new();
    for mat in document.materials() {
        let pbr = mat.pbr_metallic_roughness();
        let mut m = Material {
            kind: MaterialKind::Lambertian,
            albedo: Vec3::from_array(pbr.base_color_factor()[..3].try_into().unwrap()),
            roughness: pbr.roughness_factor(),
            metallic: pbr.metallic_factor(),
            ior: 1.5,
            emissive: Vec3::from_array(mat.emissive_factor()),
            ..Default::default()
        };

        // glTF defaults that the old code got wrong:
        m.roughness = if m.roughness.is_nan() || m.roughness < 0.0 { 1.0 } else { m.roughness };
        m.metallic = if m.metallic.is_nan() || m.metallic < 0.0 { 0.0 } else { m.metallic };

        // alphaMode MASK → alpha-cutout. BLEND (true transparency) is not
        // supported by the integrator yet: refuse to silently mis-render.
        match mat.alpha_mode() {
            gltf::material::AlphaMode::Opaque => {}
            gltf::material::AlphaMode::Mask => m.alpha_cutout = true,
            gltf::material::AlphaMode::Blend => {
                tracing::warn!(
                    "Material {:?} uses alphaMode BLEND (transparency), which is not supported; rendering it opaque",
                    mat.name()
                );
            }
        }

        let p = Pending {
            albedo: pbr.base_color_texture().map(|t| t.texture().index()),
            mr: pbr.metallic_roughness_texture().map(|t| t.texture().index()),
            emissive: mat.emissive_texture().map(|t| t.texture().index()),
            normal: mat.normal_texture().map(|t| {
                if t.tex_coord() != 0 {
                    tracing::warn!(
                        "Normal texture on material {:?} uses TEXCOORD_{} (only set 0 is supported)",
                        mat.name(),
                        t.tex_coord()
                    );
                }
                t.texture().index()
            }),
            normal_scale: mat.normal_texture().map(|t| t.scale()).unwrap_or(1.0),
            material: m,
        };
        pending.push(p);
    }

    // Default material when the file defines none.
    if pending.is_empty() {
        pending.push(Pending {
            material: Material::lambertian(Vec3::new(0.8, 0.8, 0.8)),
            albedo: None,
            mr: None,
            emissive: None,
            normal: None,
            normal_scale: 1.0,
        });
    }

    // Textured emissives with a zero factor should still glow with the
    // texture's own values (factor [1,1,1] acts as "use texture as-is").
    for p in &mut pending {
        if p.emissive.is_some() && p.material.emissive.length() < 1e-6 {
            p.material.emissive = Vec3::ONE;
        }
    }

    let raw_used_albedo: Vec<Option<usize>> = pending.iter().map(|p| p.albedo).collect();
    let raw_used_mr: Vec<Option<usize>> = pending.iter().map(|p| p.mr).collect();
    let raw_used_emissive: Vec<Option<usize>> = pending.iter().map(|p| p.emissive).collect();
    let raw_used_normal: Vec<Option<usize>> = pending.iter().map(|p| p.normal).collect();

    let (role_albedo, map_albedo) = build_role(&scene.textures, &raw_used_albedo, true);
    let (role_mr, map_mr) = build_role(&scene.textures, &raw_used_mr, false);
    let (role_emissive, map_emissive) = build_role(&scene.textures, &raw_used_emissive, true);

    // Normal maps get the material's `scale` baked into the X/Y channels.
    // Multiple materials sharing one normal texture with different scales
    // get the first scale (rare; warn).
    let mut normal_scales: Vec<f32> = vec![1.0; scene.textures.len()];
    for p in &pending {
        if let Some(id) = p.normal {
            if (normal_scales[id] - p.normal_scale).abs() > 1e-4 && normal_scales[id] != 1.0 {
                tracing::warn!("Normal texture {id} reused with different scales; using the first");
            } else {
                normal_scales[id] = p.normal_scale;
            }
        }
    }
    let (role_normal, map_normal) = build_role_scaled(&scene.textures, &raw_used_normal, &normal_scales);

    // Remap each material's raw glTF texture index through the role map
    // (maps are indexed by *master texture id*, which is what `raw` holds).
    for p in &mut pending {
        p.material.albedo_texture = p.albedo.and_then(|i| map_albedo.get(i).copied().flatten());
        p.material.roughness_texture = p.mr.and_then(|i| map_mr.get(i).copied().flatten());
        p.material.emissive_texture = p.emissive.and_then(|i| map_emissive.get(i).copied().flatten());
        p.material.normal_texture = p.normal.and_then(|i| map_normal.get(i).copied().flatten());
    }

    scene.tex_albedo = role_albedo;
    scene.tex_mr = role_mr;
    scene.tex_emissive = role_emissive;
    scene.tex_normal = role_normal;
    scene.materials = pending.into_iter().map(|p| p.material).collect();

    // ── Mesh traversal: bake the default scene graph into world space ──
    let scene_gltf = document
        .default_scene()
        .or_else(|| document.scenes().next());

    let mut node_stack: Vec<(gltf::Node<'_>, Mat4)> = Vec::new();
    if let Some(root) = scene_gltf {
        // Push roots reversed so the LIFO stack visits them in document order.
        let roots: Vec<gltf::Node<'_>> = root.nodes().collect();
        for node in roots.into_iter().rev() {
            node_stack.push((node, Mat4::IDENTITY));
        }
    }
    if node_stack.is_empty() {
        tracing::warn!("glTF file has no scene nodes; nothing to render");
    }

    while let Some((node, parent)) = node_stack.pop() {
        let local = Mat4::from_cols_array_2d(&node.transform().matrix());
        let world = parent * local;

        if let Some(mesh) = node.mesh() {
            if node.skin().is_some() {
                tracing::warn!(
                    "Node {:?} is skinned; skinning is not supported, rendering the bind pose",
                    node.name()
                );
            }
            for primitive in mesh.primitives() {
                if primitive.morph_targets().next().is_some() {
                    tracing::warn!(
                        "Primitive of mesh {:?} has morph targets; morphs are not supported, rendering the base shape",
                        mesh.name()
                    );
                }
                if let Some(m) = bake_primitive(&primitive, &buffers, &world, &scene.materials) {
                    scene.meshes.push(std::sync::Arc::new(m));
                }
            }
        }
        // Push children reversed so the LIFO stack visits them in
        // document order.
        let children: Vec<gltf::Node<'_>> = node.children().collect();
        for child in children.into_iter().rev() {
            node_stack.push((child, world));
        }
    }

    // ── Area lights from emissive materials (world-space triangles) ──
    let mut tri_offset = 0u32;
    for mesh in &scene.meshes {
        if let Some(mat) = scene.materials.get(mesh.material_id) {
            if mat.emits_light() {
                // Effective radiance: emissive factor * (average texture color or 1).
                let tex_avg = mat.emissive_texture.and_then(|i| scene.tex_emissive.get(i));
                let color = if let Some(tex) = tex_avg {
                    let mut avg = Vec3::ZERO;
                    for p in &tex.data {
                        avg += Vec3::new(p[0], p[1], p[2]);
                    }
                    (avg / tex.data.len() as f32) * mat.emissive
                } else {
                    mat.emissive
                };
                if color.length() > 0.05 {
                    for t in 0..mesh.triangle_count() {
                        scene.lights.push(Light::Area {
                            triangle_index: tri_offset + t as u32,
                            color,
                            intensity: 1.0,
                        });
                    }
                }
            }
        }
        tri_offset += mesh.triangle_count() as u32;
    }

    if scene.lights.is_empty() {
        scene.lights.push(Light::Environment { texture: None, intensity: 1.0 });
    }

    Ok(scene)
}

// ---------------------------------------------------------------------------
// Primitive baking
// ---------------------------------------------------------------------------

fn bake_primitive(
    primitive: &gltf::Primitive<'_>,
    buffers: &[gltf::buffer::Data],
    world: &Mat4,
    materials: &[Material],
) -> Option<Mesh> {
    let reader = primitive.reader(|buf| Some(&buffers[buf.index()]));

    let positions: Vec<Vec3> = reader
        .read_positions()?
        .map(Vec3::from_array)
        .collect();
    if positions.is_empty() {
        return None;
    }
    let n_verts = positions.len();

    // Triangulate into an index list, in primitive-local vertex space.
    // Converts strips/fans and generates implicit indices when absent.
    let indices_raw: Vec<u32> = match reader.read_indices() {
        Some(iter) => iter.into_u32().collect(),
        None => (0..n_verts as u32).collect(),
    };
    let triangulate = |mode: gltf::mesh::Mode, indices: &[u32]| -> Vec<[u32; 3]> {
        let mut out = Vec::new();
        match mode {
            gltf::mesh::Mode::Triangles => {
                for t in indices.chunks_exact(3) {
                    out.push([t[0], t[1], t[2]]);
                }
            }
            gltf::mesh::Mode::TriangleStrip => {
                // Triangle strips wind alternately.
                for i in 0..indices.len().saturating_sub(2) {
                    let (a, b, c) = (indices[i], indices[i + 1], indices[i + 2]);
                    out.push(if i % 2 == 0 { [a, b, c] } else { [a, c, b] });
                }
            }
            gltf::mesh::Mode::TriangleFan => {
                for i in 1..indices.len().saturating_sub(1) {
                    out.push([indices[0], indices[i], indices[i + 1]]);
                }
            }
            // Points/lines cannot be represented in a triangle renderer.
            _ => {}
        }
        out
    };
    let mode = primitive.mode();
    match mode {
        gltf::mesh::Mode::Triangles
        | gltf::mesh::Mode::TriangleStrip
        | gltf::mesh::Mode::TriangleFan => {}
        other => {
            tracing::warn!("Primitive mode {other:?} is not supported; skipping");
            return None;
        }
    }
    let mut tris = triangulate(mode, &indices_raw);
    if tris.is_empty() {
        return None;
    }

    // Normals: use authored ones or compute smooth vertex normals.
    let mut normals: Vec<Vec3> = match reader.read_normals() {
        Some(iter) => iter.map(Vec3::from_array).collect(),
        None => {
            let mut acc = vec![Vec3::ZERO; n_verts];
            for tri in &tris {
                let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
                let n = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
                acc[a] += n;
                acc[b] += n;
                acc[c] += n;
            }
            acc.into_iter()
                .map(|n| if n.length_squared() > 1e-12 { n.normalize() } else { Vec3::Y })
                .collect()
        }
    };
    if normals.len() != n_verts {
        normals = vec![Vec3::Y; n_verts];
    }

    let texcoords: Vec<Vec2> = reader
        .read_tex_coords(0)
        .map(|iter| iter.into_f32().map(Vec2::from_array).collect())
        .unwrap_or_else(|| vec![Vec2::ZERO; n_verts]);

    // World transform: positions, normals (inverse-transpose), winding.
    let normal_mat = world.inverse().transpose();
    let mirrored = world.determinant() < 0.0;

    let transformed_pos: Vec<Vec3> = positions
        .iter()
        .map(|p| world.transform_point3(*p))
        .collect();
    let transformed_n: Vec<Vec3> = normals
        .iter()
        .map(|n| {
            let t = normal_mat.transform_vector3(*n);
            if t.length_squared() > 1e-12 { t.normalize() } else { *n }
        })
        .collect();

    let mut indices: Vec<u32> = Vec::with_capacity(tris.len() * 3);
    for tri in &mut tris {
        if mirrored {
            // Keep the geometric (winding) normal aligned with the
            // transformed shading normal: cross(Ma, Mb) = det(M) M⁻ᵀ(a×b).
            indices.extend_from_slice(&[tri[0], tri[2], tri[1]]);
        } else {
            indices.extend_from_slice(tri);
        }
    }

    let material_id = primitive
        .material()
        .index()
        .map_or(materials.len().saturating_sub(1), |i| i.min(materials.len().saturating_sub(1)));

    Some(Mesh::new(transformed_pos, transformed_n, texcoords, indices, material_id))
}

