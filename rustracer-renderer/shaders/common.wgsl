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

struct Material {
    albedo: vec4<f32>,
    emissive: vec3<f32>,
    roughness: f32,
    metallic: f32,
    ior: f32,
    kind: u32,
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
