// Multi-iteration A-Trous edge-avoiding wavelet denoiser (SVGF-style).
// Edge stopping uses normal angle, depth difference, AND color variance.
// Called once per iteration with increasing step sizes (1,2,4,8).

@group(0) @binding(0) var<storage, read_write> frame: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> gbuffer: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: DenoiseParams;

struct DenoiseParams {
    width: u32,
    height: u32,
    step_size: u32,
    phi_color: f32,
    phi_normal: f32,
    phi_depth: f32,
    _pad: vec2<u32>,
}

fn spatial_weight(dx: i32, dy: i32, step: i32) -> f32 {
    let d2 = f32(dx*dx + dy*dy);
    return exp(-d2 / (2.0 * f32(step*step)));
}

fn normal_weight(nc: vec3<f32>, nn: vec3<f32>) -> f32 {
    return pow(max(dot(nc, nn), 0.0), params.phi_normal);
}

fn depth_weight(dc: f32, dn: f32) -> f32 {
    // Relative depth difference: scale the threshold by the distance of the
    // center pixel so the same settings work for any scene scale.
    return exp(-abs(dc - dn) / (params.phi_depth * max(dc, 1e-4)));
}

fn color_weight(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let lum_w = vec3(0.2126, 0.7152, 0.0722);
    let lum_a = dot(a, lum_w);
    let lum_b = dot(b, lum_w);
    let lum_diff = abs(lum_a - lum_b);
    // Luma-relative threshold (SVGF-style) with an absolute floor and a
    // Gaussian falloff. Bright-vs-dark edges (emissive HUD decal on glass)
    // get a huge ratio → weight ≈ 0 → hard stop, while flat-region noise
    // stays under the 0.15 floor → still smooths even in dark areas.
    let scale = max(params.phi_color * max(lum_a, lum_b), 0.15);
    let r = lum_diff / scale;
    return exp(-r * r);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = global_id.xy;
    if pixel.x >= params.width || pixel.y >= params.height { return; }

    let step = i32(params.step_size);
    let idx = pixel.y * params.width + pixel.x;
    let center = frame[idx].rgb;
    let n_center = gbuffer[idx].xyz;
    let d_center = gbuffer[idx].w;

    var sum = center;
    var total_weight = 1.0;

    for (var dy = -2 * step; dy <= 2 * step; dy += step) {
        for (var dx = -2 * step; dx <= 2 * step; dx += step) {
            if dx == 0 && dy == 0 { continue; }
            let nx = i32(pixel.x) + dx;
            let ny = i32(pixel.y) + dy;
            if nx < 0 || ny < 0 || nx >= i32(params.width) || ny >= i32(params.height) { continue; }

            let nidx = u32(ny) * params.width + u32(nx);
            let neighbor = frame[nidx].rgb;
            let n_neighbor = gbuffer[nidx].xyz;
            let d_neighbor = gbuffer[nidx].w;

            let sw = spatial_weight(dx, dy, step);
            let nw = normal_weight(n_center, n_neighbor);
            let dw = depth_weight(d_center, d_neighbor);
            let cw = color_weight(center, neighbor);
            let w = sw * nw * dw * cw;

            sum += neighbor * w;
            total_weight += w;
        }
    }

    let filtered = sum / total_weight;
    frame[idx] = vec4(filtered, frame[idx].a);
}
