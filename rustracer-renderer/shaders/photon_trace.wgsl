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

// ── Main ──

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let photon_index = global_id.x;

    let light = lights[0u];
    var rng = rng_init(photon_index * 1103515245u + 12345u);

    if light.kind == 3u { return; } // skip environment lights for now

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
    }

    // ── Trace photon through the scene ──
    for (var bounce = 0u; bounce < 6u; bounce++) {
        let hit = trace_bvh(ray.origin, ray.dir);

        if hit.t >= INF { return; } // miss → discard photon

        let tri = triangles[hit.prim_id];
        let mat = materials[tri.material_id];
        let hit_pos = ray.origin + ray.dir * hit.t;
        let normal = get_triangle_normal(tri);
        let front_face = dot(ray.dir, normal) < 0.0;
        let face_normal = select(-normal, normal, front_face);

        if mat.kind == 0u || mat.kind == 1u {
            // Diffuse or metal — store photon
            let store_idx = photon_index * 6u + bounce;
            photons[store_idx].position = hit_pos;
            photons[store_idx].power = power;
            photons[store_idx].incident = -ray.dir;
            photons[store_idx].normal = face_normal;

            // Russian roulette
            let p = max(max(power.r, power.g), power.b);
            if rng_f32(&rng) > p { return; }
            power = power / p;

            let albedo = mat.albedo.rgb;
            if mat.kind == 0u {
                let local_dir = sample_cosine_hemisphere(&rng);
                ray.dir = to_world(local_dir, face_normal);
                power = power * albedo;
            } else {
                ray.dir = reflect(ray.dir, face_normal);
                power = power * albedo;
            }
            ray.origin = hit_pos + face_normal * 0.001;

        } else if mat.kind == 2u {
            // Dielectric
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
            // power unchanged

        } else if mat.kind == 3u {
            return; // emissive — absorb
        }
    }
}
