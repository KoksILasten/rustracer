// Multi-bounce path tracer with area-light NEE + optional photon map overlay.

#include common.wgsl

@group(0) @binding(0) var<storage, read> sorted_photons: array<Photon>;
@group(0) @binding(1) var<storage, read_write> accumulation: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> camera: Camera;
@group(0) @binding(3) var<storage, read> bvh_nodes: array<BVHNode>;
@group(0) @binding(4) var<storage, read> bvh_prims: array<u32>;
@group(0) @binding(5) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(6) var<storage, read> materials: array<Material>;
@group(0) @binding(7) var<uniform> img_params: ImageParams;
@group(0) @binding(8) var<storage, read_write> gbuffer: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> grid_meta: array<vec2<u32>>;
@group(0) @binding(10) var<uniform> gp: GridParams;

struct GridParams {
    room_min_x: f32, room_min_y: f32, room_min_z: f32,
    cell_size: f32,
    grid_x: u32, grid_y: u32, grid_z: u32,
    num_cells: u32,
    photon_slots: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
}

fn photon_cell(pos: vec3<f32>) -> vec3<i32> {
    let gx = i32((pos.x - gp.room_min_x) / gp.cell_size);
    let gy = i32((pos.y - gp.room_min_y) / gp.cell_size);
    let gz = i32((pos.z - gp.room_min_z) / gp.cell_size);
    return vec3(gx, gy, gz);
}

fn cell_index(c: vec3<i32>) -> u32 {
    return u32(c.x) + u32(c.y) * gp.grid_x + u32(c.z) * gp.grid_x * gp.grid_y;
}

struct ImageParams { width: u32, height: u32, spp: u32, frame: u32, light_emission: vec3<f32>, ambient: f32, accumulate_alpha: f32, use_photons: u32, photon_count: u32, photon_radius: f32, photon_scale: f32, photon_debug: u32, pl_r: f32, pl_g: f32, pl_b: f32, _pad0: u32, _pad1: u32, _pad2: u32, }

fn trace_bvh(ro: vec3<f32>, rd: vec3<f32>) -> HitResult {
    var hit: HitResult; hit.t = INF; hit.prim_id = 0u;
    let inv = 1.0 / rd; var idx: u32 = 0u;
    var stk: array<u32, 64>; var sp: u32 = 0u;
    loop {
        let n = bvh_nodes[idx];
        let t0 = (n.bbox_min - ro) * inv; let t1 = (n.bbox_max - ro) * inv;
        let tm = min(t0, t1); let tx = max(t0, t1);
        let te = max(max(tm.x, tm.y), tm.z); let ta = min(min(tx.x, tx.y), tx.z);
        if te <= ta && ta > 0.0 && te < hit.t {
            if n.prim_count > 0u {
                for (var i = 0u; i < n.prim_count; i++) {
                    let pi = bvh_prims[n.left_first + i];
                    let tr = triangles[pi];
                    let t = intersect_triangle(ro, rd, tr.v0, tr.v1, tr.v2);
                    if t < hit.t { hit.t = t; hit.prim_id = pi; }
                }
                if sp == 0u { break; } sp -= 1u; idx = stk[sp];
            } else { if sp < 63u { stk[sp] = n.right_child; sp += 1u; } idx = n.left_first; }
        } else { if sp == 0u { break; } sp -= 1u; idx = stk[sp]; }
    }
    return hit;
}

fn shadowed(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32) -> bool {
    let sh = trace_bvh(origin, dir);
    if sh.t >= max_dist - 0.001 { return false; }
    let bt = triangles[sh.prim_id];
    return materials[bt.material_id].kind != 3u;
}

fn point_light_contrib(hp: vec3<f32>, sn: vec3<f32>, light_pos: vec3<f32>) -> vec3<f32> {
    let to = light_pos - hp; let r2 = dot(to, to); let r = sqrt(r2);
    let ld = to / r; let ndl = dot(sn, ld);
    if ndl <= 0.0 || shadowed(hp + sn * 0.01, ld, r - 0.001) { return vec3(0.0); }
    return vec3(img_params.pl_r, img_params.pl_g, img_params.pl_b) * ndl / (r2 + 1.0);
}

fn nee(hp: vec3<f32>, sn: vec3<f32>) -> vec3<f32> {
    let lpt = vec3(-0.3, 1.99, -0.3) + vec3(0.6, 0.0, 0.6) * vec3(0.5, 0.0, 0.5);
    let to = lpt - hp; let r2 = dot(to, to); let r = sqrt(r2);
    let ld = to / r; let ndl = dot(sn, ld);
    if ndl <= 0.0 || shadowed(hp + sn * 0.01, ld, r - 0.001) { return vec3(0.0); }
    let cl = ld.y;
    return img_params.light_emission * ndl * cl * 0.36 / (r2 + 1.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = global_id.xy;
    if pixel.x >= img_params.width || pixel.y >= img_params.height { return; }
    let base_seed = pixel.y * img_params.width + pixel.x + img_params.frame * 2654435761u;
    var rng = rng_init(base_seed * 1664525u + 1013904223u);
    var col = vec3(0.0);
    var first_normal = vec3(0.0, 1.0, 0.0);
    var first_depth = 1e30;
    let spp = max(1u, img_params.spp);

    for (var s = 0u; s < spp; s++) {
        let jx = rng_f32(&rng) - 0.5; let jy = rng_f32(&rng) - 0.5;
        let cam_inv = camera.inv_half_res;
        let uv = vec2<f32>((f32(pixel.x) + jx) * cam_inv.x - 1.0, 1.0 - (f32(pixel.y) + jy) * cam_inv.y);
        var ro = camera.position;
        var rd = normalize(camera.forward + camera.right * uv.x + camera.up * uv.y);
        var thr = vec3(1.0);

        for (var b = 0u; b < 6u; b++) {
            let hit = trace_bvh(ro, rd);
            if hit.t >= INF { break; }
            let tr = triangles[hit.prim_id];
            let mat = materials[tr.material_id];
            let hp = ro + rd * hit.t;
            let gn = get_triangle_normal(tr);
            let ff = dot(rd, gn) < 0.0;
            let sn = select(-gn, gn, ff);

            // Record first-hit GBuffer data (from first sample)
            if b == 0u && hit.t < first_depth {
                first_normal = sn;
                first_depth = hit.t;
            }

            col += thr * mat.emissive;

            if mat.kind == 0u {
                let albedo = mat.albedo.rgb;
                // Direct lighting (always)
                col += thr * albedo * nee(hp, sn) / PI;
                let t = f32(img_params.frame) * 0.005;
                let pl = vec3(cos(t) * 0.6, 0.2 + sin(t * 1.3) * 0.5, sin(t) * 0.5);
                col += thr * albedo * point_light_contrib(hp, sn, pl) / PI;
                col += thr * albedo * img_params.ambient;

                if img_params.use_photons == 1u {
                    // Photon indirect: grid lookup, budget-capped for bounded cost.
                    // Photons REPLACE the noisy multi-bounce → smooth at low SPP.
                    let r2 = img_params.photon_radius * img_params.photon_radius;
                    var ph_sum = vec3(0.0);
                    var ph_hits = 0u;
                    var budget = 96u;
                    var done = false;
                    let base = photon_cell(hp);
                    let gxmax = i32(gp.grid_x) - 1;
                    let gymax = i32(gp.grid_y) - 1;
                    let gzmax = i32(gp.grid_z) - 1;
                    for (var dz = -1; dz <= 1 && !done; dz++) {
                        for (var dy = -1; dy <= 1 && !done; dy++) {
                            for (var dx = -1; dx <= 1 && !done; dx++) {
                                let cx = clamp(base.x + dx, 0, gxmax);
                                let cy = clamp(base.y + dy, 0, gymax);
                                let cz = clamp(base.z + dz, 0, gzmax);
                                let cell = cell_index(vec3(cx, cy, cz));
                                let start = grid_meta[cell].x;
                                let count = grid_meta[cell].y;
                                for (var i = 0u; i < count && !done; i++) {
                                    if budget == 0u { done = true; break; }
                                    budget -= 1u;
                                    let ph = sorted_photons[start + i];
                                    let pp = ph.position;
                                    let d2 = dot(pp - hp, pp - hp);
                                    if d2 < r2 {
                                        ph_sum += ph.power * max(dot(sn, -ph.incident), 0.0);
                                        ph_hits += 1u;
                                        if ph_hits >= 24u { done = true; break; }
                                    }
                                }
                            }
                        }
                    }
                    if ph_hits > 0u {
                        col += thr * albedo * ph_sum * img_params.photon_scale / (PI * f32(ph_hits) * img_params.photon_radius * img_params.photon_radius);
                    }
                    // Photon debug: heatmap of photon density (green = covered, red = none)
                    if img_params.photon_debug == 1u {
                        let density = min(f32(ph_hits) / 24.0, 1.0);
                        col = vec3(1.0 - density, density, 0.0) * 2.0;
                    }
                    break; // photons handle indirect — no bounce
                } else {
                    // Bounce for multi-bounce PT (color bleeding)
                    let local = sample_cosine_hemisphere(&rng);
                    rd = to_world(local, sn);
                    ro = hp + sn * 0.001;
                    thr = thr * albedo;
                    let p = max(max(thr.r, thr.g), thr.b);
                    if rng_f32(&rng) > p { break; }
                    thr = thr / p;
                }

            } else if mat.kind == 1u {
                rd = reflect(rd, sn);
                ro = hp + sn * 0.001;
                thr = thr * mat.albedo.rgb;
            } else if mat.kind == 2u {
                let f0 = pow((mat.ior - 1.0) / (mat.ior + 1.0), 2.0);
                let ct = abs(dot(rd, sn));
                let fr = fresnel_schlick(ct, vec3(f0)).r;
                if rng_f32(&rng) < fr {
                    rd = reflect(rd, sn);
                    ro = hp + sn * 0.001;
                } else {
                    let eta = select(mat.ior, 1.0 / mat.ior, ff);
                    rd = refract(rd, sn, eta);
                    ro = hp - sn * 0.001;
                    thr = thr * mat.albedo.rgb;
                }
            } else if mat.kind == 3u { break; }
        }
    }
    col = col / f32(spp);
    let idx = pixel.y * img_params.width + pixel.x;
    let prev = accumulation[idx];
    accumulation[idx] = mix(prev, vec4(col, 1.0), img_params.accumulate_alpha);
    // Write GBuffer (denoiser guidance): normal.xyz + depth.w
    gbuffer[idx] = vec4(first_normal, first_depth);
}
