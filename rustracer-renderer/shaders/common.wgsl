// Rustracer — Common types and helpers shared across shaders.
// NOTE: Functions that take runtime-sized arrays (storage buffers) are
// NOT placed here — WGSL forbids passing array<> as function arguments.
// Each shader inlines the BVH traversal, accessing its own bindings.

// ── Math helpers ──

const PI: f32 = 3.141592653589793;
const INF: f32 = 1e38;
const EPS: f32 = 1e-6;

fn reflect(v: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return v - 2.0 * dot(v, n) * n;
}

fn refract(v: vec3<f32>, n: vec3<f32>, eta: f32) -> vec3<f32> {
    let cos_theta = min(dot(-v, n), 1.0);
    let r_out_perp = eta * (v + cos_theta * n);
    let r_out_parallel = -sqrt(abs(1.0 - dot(r_out_perp, r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// ── PCG random number generator (GPU-friendly) ──

struct Rng {
    state: u32,
    inc: u32,
}

fn rng_init(seed: u32) -> Rng {
    var r: Rng;
    r.state = seed;
    r.inc = 1u;
    r.state = r.state * 747796405u + 2891336453u;
    r.state = ((r.state >> ((r.state >> 28u) + 4u)) ^ r.state) * 277803737u;
    return r;
}

fn rng_next(r: ptr<function, Rng>) -> u32 {
    let old_state = (*r).state;
    (*r).state = old_state * 747796405u + (*r).inc;
    let word = ((old_state >> ((old_state >> 28u) + 4u)) ^ old_state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rng_f32(r: ptr<function, Rng>) -> f32 {
    return f32(rng_next(r)) / 4294967296.0;
}

fn rng_vec3(r: ptr<function, Rng>) -> vec3<f32> {
    return vec3(rng_f32(r), rng_f32(r), rng_f32(r));
}

// ── Hemisphere sampling ──

fn sample_cosine_hemisphere(r: ptr<function, Rng>) -> vec3<f32> {
    let u1 = rng_f32(r);
    let u2 = rng_f32(r);
    let phi = 2.0 * PI * u1;
    let cos_theta = sqrt(u2);
    let sin_theta = sqrt(1.0 - u2);
    return vec3(cos(phi) * sin_theta, cos_theta, sin(phi) * sin_theta);
}

fn to_world(local_dir: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let up = select(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), abs(normal.y) < 0.999);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return tangent * local_dir.x + normal * local_dir.y + bitangent * local_dir.z;
}

// ── Möller-Trumbore ray-triangle intersection ──

fn intersect_triangle(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
) -> f32 {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(ray_dir, edge2);
    let a = dot(edge1, h);
    if abs(a) < 1e-7 { return INF; }
    let f = 1.0 / a;
    let s = ray_origin - v0;
    let u = f * dot(s, h);
    if u < 0.0 || u > 1.0 { return INF; }
    let q = cross(s, edge1);
    let v = f * dot(ray_dir, q);
    if v < 0.0 || u + v > 1.0 { return INF; }
    let t = f * dot(edge2, q);
    if t > EPS { return t; }
    return INF;
}

// ── GPU structs (match rustracer-core repr(C)) ──

struct Ray {
    origin: vec3<f32>,
    dir: vec3<f32>,
}

struct BVHNode {
    bbox_min: vec3<f32>,
    left_first: u32,
    bbox_max: vec3<f32>,
    right_child: u32,
    prim_count: u32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct Triangle {
    v0: vec3<f32>, _pad0: f32,
    v1: vec3<f32>, _pad1: f32,
    v2: vec3<f32>, _pad2: f32,
    n0: vec3<f32>, _pad3: f32,
    n1: vec3<f32>, _pad4: f32,
    n2: vec3<f32>, _pad5: f32,
    uv0: vec2<f32>, uv1: vec2<f32>, uv2: vec2<f32>,
    material_id: u32, _pad6: f32,
}

struct Photon {
    position: vec3<f32>, _pad0: f32,
    power: vec3<f32>, _pad1: f32,
    incident: vec3<f32>, _pad2: f32,
    normal: vec3<f32>, _pad3: f32,
}

struct Camera {
    position: vec3<f32>, _pad0: f32,
    forward: vec3<f32>, _pad1: f32,
    right: vec3<f32>, _pad2: f32,
    up: vec3<f32>, _pad3: f32,
    inv_half_res: vec2<f32>,
    near: f32,
    fov_scale: f32,
    aperture: f32,
    focus_dist: f32,
}

// ── Material constants (mirror rustracer-core) ──

const TEX_NONE: u32 = 0xFFFFFFFFu;
const MAT_FLAG_ALPHA_CUTOUT: u32 = 1u;

struct Material {
    albedo: vec4<f32>,      // base color (linear) + alpha
    emissive: vec3<f32>,    // emission factor color
    roughness: f32,         // GGX roughness (0 = mirror, 1 = diffuse)
    metallic: f32,          // metalness (0 = dielectric, 1 = metal)
    ior: f32,               // index of refraction for dielectrics
    kind: u32,              // MaterialKind discriminant
    flags: u32,             // MAT_FLAG_* bits
    tex_albedo: u32,        // layer in the albedo texture array
    tex_mr: u32,            // layer in the metallic-roughness array
    tex_emissive: u32,      // layer in the emissive array
    tex_normal: u32,        // layer in the normal-map array
}

// ── Tangent frame from triangle UV gradients (no stored tangents) ──

struct Bary {
    uv: vec2<f32>,
    u: f32, // weight of v1
    v: f32, // weight of v2
}

// Barycentric coordinates + interpolated UV of a hit point on a triangle.
fn barycentric(tri: Triangle, ro: vec3<f32>, rd: vec3<f32>, t: f32) -> Bary {
    let e1 = tri.v1 - tri.v0;
    let e2 = tri.v2 - tri.v0;
    let p = ro + rd * t - tri.v0;
    let d00 = dot(e1, e1); let d01 = dot(e1, e2); let d11 = dot(e2, e2);
    let d20 = dot(p, e1); let d21 = dot(p, e2);
    let denom = d00 * d11 - d01 * d01;
    var u: f32 = 0.0; var v: f32 = 0.0;
    if denom != 0.0 {
        u = (d11 * d20 - d01 * d21) / denom;
        v = (d00 * d21 - d01 * d20) / denom;
    }
    let w = 1.0 - u - v;
    return Bary(tri.uv0 * w + tri.uv1 * u + tri.uv2 * v, u, v);
}

// Smooth interpolated vertex normal at a hit point (falls back to the flat
// geometric normal for degenerate normals).
fn smooth_normal(tri: Triangle, b: Bary, geometric: vec3<f32>) -> vec3<f32> {
    let w = 1.0 - b.u - b.v;
    var n = tri.n0 * w + tri.n1 * b.u + tri.n2 * b.v;
    if dot(n, n) < 1e-8 { return geometric; }
    return normalize(n);
}

struct TanFrame {
    t: vec3<f32>,
    b: vec3<f32>,
}

// dp/du, dp/dv from the triangle edges; Gram-Schmidt against the shading
// normal and fix handedness. Fall back to an arbitrary perpendicular when
// the UVs are degenerate.
fn tangent_frame(tri: Triangle, n: vec3<f32>) -> TanFrame {
    let e1 = tri.v1 - tri.v0;
    let e2 = tri.v2 - tri.v0;
    let d1 = tri.uv1 - tri.uv0;
    let d2 = tri.uv2 - tri.uv0;
    let det = d1.x * d2.y - d2.x * d1.y;
    let f = select(1.0 / det, 0.0, abs(det) < 1e-12);
    let t = (e1 * d2.y - e2 * d1.y) * f;
    let b = (e2 * d1.x - e1 * d2.x) * f;
    var tt = t - n * dot(n, t);
    if dot(tt, tt) < 1e-10 {
        tt = cross(n, vec3(1.0, 0.0, 0.0));
        if dot(tt, tt) < 1e-6 {
            tt = cross(n, vec3(0.0, 1.0, 0.0));
        }
    }
    tt = normalize(tt);
    let bb = cross(n, tt) * select(-1.0, 1.0, dot(cross(n, tt), b) > 0.0);
    return TanFrame(tt, bb);
}

struct Light {
    data: vec4<f32>,
    color: vec3<f32>,
    intensity: f32,
    kind: u32,
    _pad: vec3<f32>,
}

struct HitResult {
    t: f32,
    prim_id: u32,
}

// ── Helper for triangle normals ──

fn get_triangle_normal(tri: Triangle) -> vec3<f32> {
    let edge1 = tri.v1 - tri.v0;
    let edge2 = tri.v2 - tri.v0;
    return normalize(cross(edge1, edge2));
}
