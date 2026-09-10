// Multi-bounce path tracer: general light NEE + optional photon overlay.
//
// Materials use the metallic-roughness model: an albedo (optionally
// textured), roughness/metallic from factors or the MR texture's G/B
// channels, and an emissive term (factor * emissive texture). Surfaces
// sample one light (area / point / directional / environment) per bounce.

#include common.wgsl

@group(0) @binding(0) var<storage, read> sorted_photons: array<Photon>;
@group(0) @binding(1) var<storage, read_write> frame: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> camera: Camera;
@group(0) @binding(3) var<storage, read> bvh_nodes: array<BVHNode>;
@group(0) @binding(4) var<storage, read> bvh_prims: array<u32>;
@group(0) @binding(5) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(6) var<storage, read> materials: array<Material>;
@group(0) @binding(7) var<uniform> img_params: ImageParams;
@group(0) @binding(8) var<storage, read_write> gbuffer: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> grid_meta: array<vec2<u32>>;
@group(0) @binding(10) var<uniform> gp: GridParams;
@group(0) @binding(11) var<storage, read> lights: array<Light>;
@group(0) @binding(12) var base_tex: texture_2d_array<f32>;
@group(0) @binding(13) var mr_tex: texture_2d_array<f32>;
@group(0) @binding(14) var emissive_tex: texture_2d_array<f32>;
@group(0) @binding(15) var tex_sampler: sampler;
@group(0) @binding(16) var normal_tex: texture_2d_array<f32>;
// Per-frame luminance moments (E[l], E[l²]) over this frame's spp samples.
@group(0) @binding(17) var<storage, read_write> frame_moments: array<vec4<f32>>;

// Luminance of a radiance sample (same weights as the denoiser).
fn luma_of(c: vec3<f32>) -> f32 { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

// Accumulate a radiance contribution into the pixel accumulator AND the
// per-frame luminance moments used to estimate sampling variance.
fn add_radiance(col: ptr<function, vec3<f32>>, lum_sum: ptr<function, f32>, lum2_sum: ptr<function, f32>, c: vec3<f32>) {
    *col += c;
    let l = luma_of(c);
    *lum_sum += l;
    *lum2_sum += l * l;
}

struct GridParams {
    room_min_x: f32, room_min_y: f32, room_min_z: f32,
    cell_size: f32,
    grid_x: u32, grid_y: u32, grid_z: u32,
    num_cells: u32,
    photon_slots: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
}

struct ImageParams { width: u32, height: u32, spp: u32, frame: u32, light_emission: vec3<f32>, ambient: f32, accumulate_alpha: f32, use_photons: u32, photon_count: u32, photon_radius: f32, photon_scale: f32, photon_debug: u32, pl_r: f32, pl_g: f32, pl_b: f32, _pad0: u32, _pad1: u32, _pad2: u32, }

fn photon_cell(pos: vec3<f32>) -> vec3<i32> {
    let gx = i32((pos.x - gp.room_min_x) / gp.cell_size);
    let gy = i32((pos.y - gp.room_min_y) / gp.cell_size);
    let gz = i32((pos.z - gp.room_min_z) / gp.cell_size);
    return vec3(gx, gy, gz);
}

fn cell_index(c: vec3<i32>) -> u32 {
    return u32(c.x) + u32(c.y) * gp.grid_x + u32(c.z) * gp.grid_x * gp.grid_y;
}

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

// Is something opaque between `origin` and `origin + dir * max_dist`?
fn shadowed(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32) -> bool {
    let sh = trace_bvh(origin, dir);
    if sh.t >= max_dist - 0.001 { return false; }
    let bt = triangles[sh.prim_id];
    let bm = materials[bt.material_id];
    // Alpha-cutout surfaces may let shadow rays through.
    if (bm.flags & MAT_FLAG_ALPHA_CUTOUT) != 0u && bm.tex_albedo != TEX_NONE {
        let uv = barycentric(bt, origin, dir, sh.t).uv;
        let alpha = textureSampleLevel(base_tex, tex_sampler, uv, i32(bm.tex_albedo), 0.0).a;
        if alpha < 0.5 { return false; }
    }
    return true;
}

// Procedural sky: cool zenith, warm horizon — glTF scenes without their own
// environment get a neutral studio-ish dome scaled by the env light.
fn sky_color(dir: vec3<f32>, intensity: f32) -> vec3<f32> {
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    let grad = mix(vec3(0.55, 0.6, 0.75), vec3(0.95, 0.93, 0.9), pow(t, 0.5));
    return grad * intensity;
}

struct MatParams {
    albedo: vec3<f32>,
    rough: f32,
    metal: f32,
}

// Sample the material parameters at a hit (textures or plain factors).
fn eval_material(mat: Material, uv: vec2<f32>) -> MatParams {
    var albedo = mat.albedo.rgb;
    var rough = mat.roughness;
    var metal = mat.metallic;
    if mat.tex_albedo != TEX_NONE {
        albedo = textureSampleLevel(base_tex, tex_sampler, uv, i32(mat.tex_albedo), 0.0).rgb;
    }
    if mat.tex_mr != TEX_NONE {
        let mr = textureSampleLevel(mr_tex, tex_sampler, uv, i32(mat.tex_mr), 0.0);
        rough = mr.g;   // metallic-roughness texture: G = roughness
        metal = mr.b;   // B = metallic
    }
    return MatParams(albedo, rough, metal);
}

fn emissive_at(mat: Material, uv: vec2<f32>) -> vec3<f32> {
    if mat.tex_emissive != TEX_NONE {
        return mat.emissive * textureSampleLevel(emissive_tex, tex_sampler, uv, i32(mat.tex_emissive), 0.0).rgb;
    }
    return mat.emissive;
}

// One-sample direct lighting over ALL lights (uniform pick). Returns the
// incident irradiance (before the albedo/pi BRDF factor).
fn direct_light(hp: vec3<f32>, sn: vec3<f32>, rng: ptr<function, Rng>) -> vec3<f32> {
    let n = arrayLength(&lights);
    if n == 0u { return vec3(0.0); }
    let li = u32(rng_f32(rng) * f32(n));
    let l = lights[li];

    if l.kind == 0u { // point
        let to = l.data.xyz - hp;
        let r2 = dot(to, to);
        let r = sqrt(r2);
        let ld = to / r;
        if dot(sn, ld) <= 0.0 { return vec3(0.0); }
        if shadowed(hp + sn * 0.01, ld, r - 0.001) { return vec3(0.0); }
        return l.color * l.intensity * max(dot(sn, ld), 0.0) / (r2 + 0.01) * f32(n);
    } else if l.kind == 1u { // directional
        let ld = normalize(l.data.xyz);
        let ndl = dot(sn, ld);
        if ndl <= 0.0 { return vec3(0.0); }
        if shadowed(hp + sn * 0.01, ld, 1e30) { return vec3(0.0); }
        return l.color * l.intensity * ndl * f32(n);
    } else if l.kind == 2u { // area (emissive triangle)
        let tri = triangles[u32(l.data.x)];
        // Uniform sample the triangle.
        let u1 = rng_f32(rng); let u2 = rng_f32(rng);
        let su = sqrt(u1);
        let u = 1.0 - su; let v = u2 * su; let w = 1.0 - u - v;
        let lp = tri.v0 * w + tri.v1 * u + tri.v2 * v;
        let to = lp - hp;
        let r2 = dot(to, to);
        let r = sqrt(r2);
        let ld = to / r;
        let ndl = dot(sn, ld);
        if ndl <= 0.0 { return vec3(0.0); }
        // Geometric normal of the light triangle (front face only).
        let ln = get_triangle_normal(tri);
        let ndl_l = dot(ln, -ld);
        if ndl_l <= 0.0 { return vec3(0.0); }
        if shadowed(hp + sn * 0.01, ld, r - 0.001) { return vec3(0.0); }
        let area = 0.5 * length(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
        return l.color * l.intensity * ndl * ndl_l * area * f32(n) / (r2 + 0.01);
    }
    return vec3(0.0); // environment handled separately
}

// Soft reflection for rough metals: blur the mirror direction by roughness.
fn bounce_dir(rd: vec3<f32>, n: vec3<f32>, rough: f32, rng: ptr<function, Rng>) -> vec3<f32> {
    let mirror = reflect(rd, n);
    let blur = smoothstep(0.0, 1.0, rough);
    if blur <= 0.001 { return mirror; }
    let diff = sample_cosine_hemisphere(rng);
    return normalize(mix(mirror, to_world(diff, n), blur));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = global_id.xy;
    if pixel.x >= img_params.width || pixel.y >= img_params.height { return; }
    let base_seed = pixel.y * img_params.width + pixel.x + img_params.frame * 2654435761u;
    var rng = rng_init(base_seed * 1664525u + 1013904223u);
    var col = vec3(0.0);
    var lum_sum = 0.0;
    var lum2_sum = 0.0;
    var first_normal = vec3(0.0, 1.0, 0.0);
    var first_depth = 1e30;
    let spp = max(1u, img_params.spp);

    // Environment light info (intensity * color), if present.
    var env_intensity = 0.0;
    var env_color = vec3(1.0);
    {
        let nl = arrayLength(&lights);
        for (var i = 0u; i < nl; i++) {
            if lights[i].kind == 3u {
                env_intensity = lights[i].intensity;
                env_color = lights[i].color;
                break;
            }
        }
    }

    for (var s = 0u; s < spp; s++) {
        let jx = rng_f32(&rng) - 0.5; let jy = rng_f32(&rng) - 0.5;
        let cam_inv = camera.inv_half_res;
        let uv = vec2<f32>((f32(pixel.x) + jx) * cam_inv.x - 1.0, 1.0 - (f32(pixel.y) + jy) * cam_inv.y);
        var ro = camera.position;
        var rd = normalize(camera.forward + camera.right * uv.x + camera.up * uv.y);
        var thr = vec3(1.0);

        for (var b = 0u; b < 6u; b++) {
            let hit = trace_bvh(ro, rd);
            if hit.t >= INF {
                // Background: environment dome.
                add_radiance(&col, &lum_sum, &lum2_sum, thr * sky_color(rd, env_intensity) * env_color);
                break;
            }
            let tr = triangles[hit.prim_id];
            let mat = materials[tr.material_id];
            let hp = ro + rd * hit.t;
            let bb = barycentric(tr, ro, rd, hit.t);
            let uv_hit = bb.uv;
            let gn = get_triangle_normal(tr);
            let ff = dot(rd, gn) < 0.0;
            // Smooth interpolated geometric normal: used for the G-buffer
            // (denoiser gate) and as the TBN base for normal mapping. Flat
            // per-triangle normals make the denoiser average color inside
            // each triangle but not across, producing wireframe patches.
            let sn_geom = smooth_normal(tr, bb, gn);
            let sn = select(-sn_geom, sn_geom, ff);

            // Perturbed shading normal from the tangent-space normal map.
            var shn = sn;
            if mat.tex_normal != TEX_NONE {
                let nm = textureSampleLevel(normal_tex, tex_sampler, uv_hit, i32(mat.tex_normal), 0.0).rgb * 2.0 - 1.0;
                let frame = tangent_frame(tr, sn);
                shn = normalize(frame.t * nm.x + frame.b * nm.y + sn * nm.z);
            }

            // Record first-hit GBuffer data (from first sample). The
            // denoiser gates on GEOMETRIC normals — normal-map detail is
            // texture information and must not block smoothing (it made the
            // A-trous weights collapse into per-triangle patches).
            if b == 0u && hit.t < first_depth {
                first_normal = sn;
                first_depth = hit.t;
            }

            // Material evaluation (textures or factors).
            let params = eval_material(mat, uv_hit);
            let albedo = params.albedo;
            let rough = params.rough;
            let metal = params.metal;
            var alpha_val = 1.0;
            if mat.tex_albedo != TEX_NONE {
                alpha_val = textureSampleLevel(base_tex, tex_sampler, uv_hit, i32(mat.tex_albedo), 0.0).a;
            }

            // Alpha-cutout: skip this surface entirely.
            if (mat.flags & MAT_FLAG_ALPHA_CUTOUT) != 0u && alpha_val < 0.5 {
                ro = hp + rd * 0.002;
                continue;
            }

            add_radiance(&col, &lum_sum, &lum2_sum, thr * emissive_at(mat, uv_hit));

            // Debug views: dump the raw value and stop bouncing.
            // 1 = UVs, 2 = smooth normal, 3 = perturbed normal, 4 = flat normal.
            if img_params._pad0 == 1u { col = vec3(uv_hit, 0.0); break; }
            if img_params._pad0 == 2u { col = sn * 0.5 + 0.5; break; }
            if img_params._pad0 == 3u { col = shn * 0.5 + 0.5; break; }
            if img_params._pad0 == 4u { col = gn * 0.5 + 0.5; break; }
            if img_params._pad0 == 5u { col = emissive_at(mat, uv_hit); break; }

            // Screen-space direct texture samples (bypass uv_hit entirely):
            // 8 = emissive layer 0, 9 = albedo layer 0.
            if img_params._pad0 == 8u {
                let puv = vec2(f32(pixel.x), f32(pixel.y)) / vec2(f32(img_params.width), f32(img_params.height));
                col = textureSampleLevel(emissive_tex, tex_sampler, puv, 0, 0.0).rgb;
                break;
            }
            if img_params._pad0 == 9u {
                let puv = vec2(f32(pixel.x), f32(pixel.y)) / vec2(f32(img_params.width), f32(img_params.height));
                col = textureSampleLevel(base_tex, tex_sampler, puv, 0, 0.0).rgb;
                break;
            }
            // 11 = direct emissive sample at uv_hit (no layer/factor),
            // 12 = the emissive factor alone, 13 = tex_emissive layer value.
            if img_params._pad0 == 11u {
                col = textureSampleLevel(emissive_tex, tex_sampler, uv_hit, 0, 0.0).rgb;
                break;
            }
            if img_params._pad0 == 12u {
                col = mat.emissive;
                break;
            }
            if img_params._pad0 == 13u {
                col = vec3(f32(mat.tex_emissive) / 65535.0, 0.0, 0.0);
                break;
            }
            // 17 = uv_hit with v wrapped into [0,1) first (precision-safe
            // for the 8-bit save; raw v in [1,2) would collapse to ~25
            // byte levels after tonemap).
            if img_params._pad0 == 17u {
                col = vec3(fract(uv_hit), 0.0);
                break;
            }
            // 18 = albedo at uv_hit, 19 = tangent t, 20 = bitangent b,
            // 21 = world hit position.
            if img_params._pad0 == 18u {
                col = textureSampleLevel(base_tex, tex_sampler, uv_hit, i32(mat.tex_albedo), 0.0).rgb;
                break;
            }
            if img_params._pad0 == 19u {
                let frame = tangent_frame(tr, sn);
                col = frame.t * 0.5 + 0.5;
                break;
            }
            if img_params._pad0 == 20u {
                let frame = tangent_frame(tr, sn);
                col = frame.b * 0.5 + 0.5;
                break;
            }
            if img_params._pad0 == 21u {
                col = hp;
                break;
            }

            if mat.kind == 2u {
                // Dielectric (glass): specular + transmission.
                let f0 = vec3(pow((mat.ior - 1.0) / (mat.ior + 1.0), 2.0));
                let ct = abs(dot(rd, shn));
                let fr = fresnel_schlick(ct, f0).r;
                if rng_f32(&rng) < fr {
                    rd = reflect(rd, shn);
                    ro = hp + shn * 0.001;
                } else {
                    let eta = select(mat.ior, 1.0 / mat.ior, ff);
                    rd = refract(rd, shn, eta);
                    ro = hp - shn * 0.001;
                    thr = thr * mat.albedo.rgb;
                }
                continue;
            }

            // PBR conductor/dielectric: F0 = mix(0.04, albedo, metallic).
            let f0 = mix(vec3(0.04), albedo, vec3(metal));
            let ct = abs(dot(rd, shn));
            let fr = fresnel_schlick(ct, f0);
            let diffuse_weight = (1.0 - metal) * albedo;
            let spec_prob = clamp(max(max(fr.r, fr.g), fr.b), 0.0, 0.98);
            let do_spec = rng_f32(&rng) < spec_prob;

            // Direct lighting on the diffuse path (irradiance * BRDF).
            if !do_spec {
                var direct = direct_light(hp, shn, &rng);
                // Environment hemisphere contribution (approximate irradiance).
                if env_intensity > 0.0 && !do_spec {
                    add_radiance(&col, &lum_sum, &lum2_sum, thr * diffuse_weight * sky_color(shn, env_intensity) * env_color);
                }
                add_radiance(&col, &lum_sum, &lum2_sum, thr * diffuse_weight * direct / PI);
                // Simple ambient term as a last-resort fill.
                add_radiance(&col, &lum_sum, &lum2_sum, thr * albedo * img_params.ambient);
            }

            // Photon-map indirect (only on the diffuse path).
            if img_params.use_photons == 1u && !do_spec {
                let r2 = img_params.photon_radius * img_params.photon_radius;
                var ph_sum = vec3(0.0);
                var ph_hits = 0u;
                var budget = 384u;
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
                                    ph_sum += ph.power * max(dot(shn, -ph.incident), 0.0);
                                    ph_hits += 1u;
                                    if ph_hits >= 64u { done = true; break; }
                                }
                            }
                        }
                    }
                }
                if ph_hits > 0u {
                    add_radiance(&col, &lum_sum, &lum2_sum, thr * diffuse_weight * ph_sum * img_params.photon_scale / (PI * f32(ph_hits) * img_params.photon_radius * img_params.photon_radius));
                }
                if img_params.photon_debug == 1u {
                    let density = min(f32(ph_hits) / 64.0, 1.0);
                    col = vec3(1.0 - density, density, 0.0) * 2.0;
                }
            }

            // Next bounce direction.
            if do_spec {
                rd = bounce_dir(rd, shn, rough, &rng);
                ro = hp + shn * 0.001;
                // Specular throughput: reflection * fresnel (approximate).
                thr = thr * f0;
            } else {
                let local = sample_cosine_hemisphere(&rng);
                rd = to_world(local, shn);
                ro = hp + shn * 0.001;
                thr = thr * diffuse_weight;
                // Photons replace the noisy multi-bounce.
                if img_params.use_photons == 1u { break; }
            }

            // Russian roulette.
            let p = max(max(thr.r, thr.g), thr.b);
            if p <= 0.0 || rng_f32(&rng) > p { break; }
            thr = thr / p;
        }
    }
    // Clamp fireflies before accumulation: a single huge specular/light
    // sample would otherwise persist as a white speckle (the denoiser's
    // edge-stopping deliberately keeps high-contrast pixels).
    col = min(col / f32(spp), vec3(10.0));
    let idx = pixel.y * img_params.width + pixel.x;
    // The raw sample goes to the frame buffer; the composite pass blends it
    // into the EMA accumulation history (denoise runs on `frame`, never on
    // the history buffer itself — that would re-filter the running average).
    frame[idx] = vec4(col, 1.0);
    // Per-frame luminance moments over the spp samples. The denoiser scales
    // this variance by the EMA attenuation factor alpha/(2-alpha) to get
    // the residual noise of the accumulated history.
    frame_moments[idx] = vec4(min(lum_sum / f32(spp), 10.0), min(lum2_sum / f32(spp), 100.0), 0.0, 0.0);
    // Write GBuffer (denoiser guidance): normal.xyz + depth.w
    gbuffer[idx] = vec4(first_normal, first_depth);
}
