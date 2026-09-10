// Multi-iteration A-Trous edge-avoiding wavelet denoiser (SVGF-style).
// Uses separate input/output buffers (ping-pong) to avoid read-write race conditions
// across compute workgroups.
// Standard B3-spline wavelet kernel [1/16, 4/16, 6/16, 4/16, 1/16] expanded by step size 2^i.

@group(0) @binding(0) var<storage, read> input_frame: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output_frame: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> gbuffer: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: DenoiseParams;

struct DenoiseParams {
    width: u32,
    height: u32,
    step_size: u32,
    phi_color: f32,
    phi_normal: f32,
    phi_depth: f32,
    _pad: vec2<u32>,
}

// 1D B3-spline wavelet scaling kernel: [1/16, 4/16, 6/16, 4/16, 1/16]
// Offset:      -2      -1       0      +1      +2
// Weight:    0.0625   0.25   0.375    0.25   0.0625
const kernel_1d = array<f32, 3>(0.375, 0.25, 0.0625);

fn spatial_weight(dx_step: i32, dy_step: i32) -> f32 {
    let kx = kernel_1d[abs(dx_step)];
    let ky = kernel_1d[abs(dy_step)];
    return kx * ky;
}

fn normal_weight(nc: vec3<f32>, nn: vec3<f32>) -> f32 {
    return pow(max(dot(nc, nn), 0.0), params.phi_normal);
}

fn depth_weight(dc: f32, dn: f32) -> f32 {
    return exp(-abs(dc - dn) / (params.phi_depth * max(dc, 1e-4)));
}

fn color_weight(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let diff = a - b;
    let dist2 = dot(diff, diff);
    let max_c = max(max(a.r, max(a.g, a.b)), max(b.r, max(b.g, b.b)));
    let scale = max(params.phi_color * max_c, 0.15);
    return exp(-dist2 / (scale * scale));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = global_id.xy;
    if pixel.x >= params.width || pixel.y >= params.height { return; }

    let step = i32(params.step_size);
    let idx = pixel.y * params.width + pixel.x;
    let center_val = input_frame[idx];
    let center = center_val.rgb;
    let n_center = gbuffer[idx].xyz;
    let d_center = gbuffer[idx].w;

    var sum = vec3<f32>(0.0);
    var total_weight = 0.0;

    for (var dy_step = -2; dy_step <= 2; dy_step++) {
        for (var dx_step = -2; dx_step <= 2; dx_step++) {
            let nx = i32(pixel.x) + dx_step * step;
            let ny = i32(pixel.y) + dy_step * step;
            if nx < 0 || ny < 0 || nx >= i32(params.width) || ny >= i32(params.height) { continue; }

            let nidx = u32(ny) * params.width + u32(nx);
            let neighbor = input_frame[nidx].rgb;
            let n_neighbor = gbuffer[nidx].xyz;
            let d_neighbor = gbuffer[nidx].w;

            let sw = spatial_weight(dx_step, dy_step);
            let nw = normal_weight(n_center, n_neighbor);
            let dw = depth_weight(d_center, d_neighbor);
            let cw = color_weight(center, neighbor);
            let w = sw * nw * dw * cw;

            sum += neighbor * w;
            total_weight += w;
        }
    }

    if total_weight > 1e-6 {
        output_frame[idx] = vec4(sum / total_weight, center_val.a);
    } else {
        output_frame[idx] = center_val;
    }
}
