// Photon tracing pass.
// Emits photons from light sources, traces through the BVH, applies
// Russian roulette at each bounce, and stores surviving photons.
// BVH traversal is inlined to avoid WGSL array-parameter restrictions.

#include common.wgsl

@group(0) @binding(0) var<storage, read_write> photons: array<Photon>;
@group(0) @binding(1) var<storage, read> bvh_nodes: array<BVHNode>;
@group(0) @binding(2) var<storage, read> bvh_prims: array<u32>;
@group(0) @binding(3) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(4) var<storage, read> lights: array<Light>;
@group(0) @binding(5) var<storage, read> materials: array<Material>;
@group(0) @binding(6) var base_tex: texture_2d_array<f32>;
@group(0) @binding(7) var tex_sampler: sampler;
@group(0) @binding(8) var normal_tex: texture_2d_array<f32>;
@group(0) @binding(9) var<uniform> gp: GridParams;

struct GridParams {
    room_min_x: f32, room_min_y: f32, room_min_z: f32,
    cell_size: f32,
    grid_x: u32, grid_y: u32, grid_z: u32,
    num_cells: u32,
    photon_slots: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
}

// Same gradient as gather's sky_color (kept in sync).
fn sky_radiance(dir: vec3<f32>) -> vec3<f32> {
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(vec3(0.55, 0.6, 0.75), vec3(0.95, 0.93, 0.9), pow(t, 0.5));
}

// ── Inline BVH traversal ──
// Returns HitResult; hit.t == INF means miss.

fn trace_bvh(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> HitResult {
    var hit: HitResult;
    hit.t = INF;
    hit.prim_id = 0u;

    let inv_dir = 1.0 / ray_dir;
    var node_idx: u32 = 0u;
    var stack: array<u32, 64>;
    var stack_ptr: u32 = 0u;

    loop {
        let node = bvh_nodes[node_idx];

        let t0 = (node.bbox_min - ray_origin) * inv_dir;
        let t1 = (node.bbox_max - ray_origin) * inv_dir;
        let tmin = min(t0, t1);
        let tmax = max(t0, t1);
        let t_enter = max(max(tmin.x, tmin.y), tmin.z);
        let t_exit  = min(min(tmax.x, tmax.y), tmax.z);

        if t_enter <= t_exit && t_exit > 0.0 && t_enter < hit.t {
            if node.prim_count > 0u {
                for (var i = 0u; i < node.prim_count; i++) {
                    let prim_idx = bvh_prims[node.left_first + i];
                    let tri = triangles[prim_idx];
                    let t = intersect_triangle(ray_origin, ray_dir, tri.v0, tri.v1, tri.v2);
                    if t < hit.t {
                        hit.t = t;
                        hit.prim_id = prim_idx;
                    }
                }
                if stack_ptr == 0u { break; }
                stack_ptr -= 1u;
                node_idx = stack[stack_ptr];
            } else {
                let left  = node.left_first;
                let right = node.right_child;
                if stack_ptr < 63u {
                    stack[stack_ptr] = right;
                    stack_ptr += 1u;
                }
                node_idx = left;
            }
        } else {
            if stack_ptr == 0u { break; }
            stack_ptr -= 1u;
            node_idx = stack[stack_ptr];
        }
    }
    return hit;
}

fn surface_albedo(mat: Material, uv: vec2<f32>) -> vec3<f32> {
    if mat.tex_albedo != TEX_NONE {
        return textureSampleLevel(base_tex, tex_sampler, uv, i32(mat.tex_albedo), 0.0).rgb;
    }
    return mat.albedo.rgb;
}

// Soft reflection for rough metals (kept in sync with gather.wgsl).
fn bounce_dir(rd: vec3<f32>, n: vec3<f32>, rough: f32, rng: ptr<function, Rng>) -> vec3<f32> {
    let mirror = reflect(rd, n);
    let blur = smoothstep(0.0, 1.0, rough);
    if blur <= 0.001 { return mirror; }
    let diff = sample_cosine_hemisphere(rng);
    return normalize(mix(mirror, to_world(diff, n), blur));
}

// ── Main ──

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let photon_index = global_id.x;
    let num_lights = arrayLength(&lights);
    if num_lights == 0u { return; }

    var rng = rng_init(photon_index * 1103515245u + 12345u);

    // Pick a light uniformly. Environment lights participate too: they
    // emit photons from a dome around the scene, which is what gives
    // env-lit scenes (ToyCar, DamagedHelmet) indirect light and color
    // bleed in the photon map.
    let li = u32(rng_f32(&rng) * f32(num_lights));
    let light = lights[li];

    // ── Emit photon from light ──
    var ray: Ray;
    var power: vec3<f32>;
    ray.origin = light.data.xyz;

    if light.kind == 0u {
        // Point light: uniform sphere
        var u1: f32;
        var u2: f32;
        loop {
            u1 = rng_f32(&rng) * 2.0 - 1.0;
            u2 = rng_f32(&rng) * 2.0 - 1.0;
            if u1 * u1 + u2 * u2 < 1.0 { break; }
        }
        let z = 1.0 - 2.0 * (u1 * u1 + u2 * u2);
        let r = sqrt(1.0 - z * z);
        ray.dir = normalize(vec3(r * 2.0 * u1, r * 2.0 * u2, z));
        power = light.color * light.intensity;
    } else if light.kind == 1u {
        // Directional: disk far away
        let disk_radius = 10.0;
        let disk_u = (rng_f32(&rng) - 0.5) * disk_radius;
        let disk_v = (rng_f32(&rng) - 0.5) * disk_radius;
        let up = select(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), abs(light.data.y) < 0.999);
        let tangent = normalize(cross(up, light.data.xyz));
        let bitangent = cross(light.data.xyz, tangent);
        ray.origin = disk_u * tangent + disk_v * bitangent - light.data.xyz * 100.0;
        ray.dir = light.data.xyz;
        power = light.color * light.intensity;
    } else if light.kind == 2u {
        // Area light (emissive triangle)
        let tri_idx = u32(light.data.x);
        let tri = triangles[tri_idx];
        let u1 = rng_f32(&rng);
        let u2 = rng_f32(&rng);
        let su = sqrt(u1);
        let u = 1.0 - su;
        let v = u2 * su;
        let w = 1.0 - u - v;
        ray.origin = tri.v0 * u + tri.v1 * v + tri.v2 * w;
        let n = get_triangle_normal(tri);
        let local_dir = sample_cosine_hemisphere(&rng);
        ray.dir = to_world(local_dir, n);
        power = light.color * light.intensity;
    } else if light.kind == 3u {
        // Environment dome: emit from a sphere around the scene bounds.
        let local_dir = sample_cosine_hemisphere(&rng);
        let dir = to_world(local_dir, vec3(0.0, 1.0, 0.0));
        let span = vec3(f32(gp.grid_x), f32(gp.grid_y), f32(gp.grid_z)) * gp.cell_size;
        let center = vec3(gp.room_min_x, gp.room_min_y, gp.room_min_z) + span * 0.5;
        let R = max(max(span.x, span.y), span.z) * 1.2 + 0.05;
        ray.origin = center + dir * R;
        ray.dir = -dir;
        power = sky_radiance(dir) * light.intensity;
    } else {
        return;
    }

    // ── Trace photon through the scene ──
    for (var bounce = 0u; bounce < 6u; bounce++) {
        let hit = trace_bvh(ray.origin, ray.dir);
        if hit.t >= INF { return; } // miss → discard photon

        let tri = triangles[hit.prim_id];
        let mat = materials[tri.material_id];
        let hit_pos = ray.origin + ray.dir * hit.t;
        let bb = barycentric(tri, ray.origin, ray.dir, hit.t);
        let uv_hit = bb.uv;
        let normal = get_triangle_normal(tri);
        let front_face = dot(ray.dir, normal) < 0.0;
        let sn_geom = smooth_normal(tri, bb, normal);
        let face_normal = select(-sn_geom, sn_geom, front_face);

        // Perturbed shading normal from the tangent-space normal map.
        var shn = face_normal;
        if mat.tex_normal != TEX_NONE {
            let nm = textureSampleLevel(normal_tex, tex_sampler, uv_hit, i32(mat.tex_normal), 0.0).rgb * 2.0 - 1.0;
            let frame = tangent_frame(tri, face_normal);
            shn = normalize(frame.t * nm.x + frame.b * nm.y + face_normal * nm.z);
        }

        // Alpha-cutout: pass through transparent texels.
        if (mat.flags & MAT_FLAG_ALPHA_CUTOUT) != 0u && mat.tex_albedo != TEX_NONE {
            let alpha = textureSampleLevel(base_tex, tex_sampler, uv_hit, i32(mat.tex_albedo), 0.0).a;
            if alpha < 0.5 {
                ray.origin = hit_pos + ray.dir * 0.002;
                continue;
            }
        }

        // Hit a light source: absorb (its emission is handled directly).
        if mat.tex_emissive != TEX_NONE || length(mat.emissive) > 1e-4 || mat.kind == 3u {
            return;
        }

        if mat.kind == 2u {
            // Dielectric — store nothing, refract or reflect.
            let f0 = vec3(pow((mat.ior - 1.0) / (mat.ior + 1.0), 2.0));
            let cos_theta = abs(dot(ray.dir, face_normal));
            let fr = fresnel_schlick(cos_theta, f0).r;
            if rng_f32(&rng) < fr {
                ray.dir = reflect(ray.dir, face_normal);
                ray.origin = hit_pos + face_normal * 0.001;
            } else {
                let eta = select(mat.ior, 1.0 / mat.ior, front_face);
                ray.dir = refract(ray.dir, face_normal, eta);
                ray.origin = hit_pos - face_normal * 0.001;
            }
            continue;
        }

        // PBR surface — store the photon, then scatter.
        let store_idx = photon_index * 6u + bounce;
        photons[store_idx].position = hit_pos;
        photons[store_idx].power = power;
        photons[store_idx].incident = -ray.dir;
        photons[store_idx].normal = shn;

        let albedo = surface_albedo(mat, uv_hit);
        let f0 = mix(vec3(0.04), albedo, vec3(mat.metallic));
        let ct = abs(dot(ray.dir, shn));
        let fr = fresnel_schlick(ct, f0);
        let diffuse_weight = (1.0 - mat.metallic) * albedo;
        let spec_prob = clamp(max(max(fr.r, fr.g), fr.b), 0.0, 0.98);
        let do_spec = rng_f32(&rng) < spec_prob;

        if do_spec {
            ray.dir = bounce_dir(ray.dir, shn, mat.roughness, &rng);
            power = power * f0;
        } else {
            let local_dir = sample_cosine_hemisphere(&rng);
            ray.dir = to_world(local_dir, shn);
            power = power * diffuse_weight;
        }
        ray.origin = hit_pos + shn * 0.001;

        // Russian roulette
        let p = max(max(power.r, power.g), power.b);
        if p <= 0.0 || rng_f32(&rng) > p { return; }
        power = power / p;
    }
}
