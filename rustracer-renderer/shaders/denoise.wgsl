// Multi-iteration A-Trous edge-avoiding wavelet denoiser (SVGF-style).
// Separate input/output buffers (ping-pong) avoid read-write races across
// compute workgroups.
//
// The color edge-stop is VARIANCE-GUIDED: it uses the accumulated luminance
// moments (E[l], E[l²]) of the EMA history, so it only smooths pixels whose
// sampling variance is still high (noise). Converged pixels — including
// fine texture detail that has accumulated over frames — have low variance
// and are left alone, so the denoiser does NOT blur textures away. As the
// accumulation converges, the denoiser automatically backs off everywhere.
//
// Spatial kernel: standard B3-spline wavelet scaling kernel
// [1/16, 4/16, 6/16, 4/16, 1/16], expanded by step 2^i per iteration.

@group(0) @binding(0) var<storage, read> input_frame: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output_frame: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> gbuffer: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> frame_moments: array<vec4<f32>>; // per-frame (E[l], E[l²])
@group(0) @binding(4) var<uniform> params: DenoiseParams;

struct DenoiseParams {
    width: u32,
    height: u32,
    step_size: u32,
    lambda: f32,   // SVGF lambda: variance multiplier for the color stop
    phi_normal: f32,
    phi_depth: f32,
    alpha: f32,    // EMA blend factor of the accumulation history
    _pad: u32,
}

// 1D B3-spline wavelet scaling kernel: [1/16, 4/16, 6/16, 4/16, 1/16]
const kernel_1d = array<f32, 3>(0.375, 0.25, 0.0625);

fn spatial_weight(dx_step: i32, dy_step: i32) -> f32 {
    return kernel_1d[abs(dx_step)] * kernel_1d[abs(dy_step)];
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

fn normal_weight(nc: vec3<f32>, nn: vec3<f32>) -> f32 {
    return pow(max(dot(nc, nn), 0.0), params.phi_normal);
}

fn depth_weight(dc: f32, dn: f32) -> f32 {
    return exp(-abs(dc - dn) / (params.phi_depth * max(dc, 1e-4)));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = global_id.xy;
    if pixel.x >= params.width || pixel.y >= params.height { return; }

    let step = i32(params.step_size);
    let idx = pixel.y * params.width + pixel.x;
    let center = input_frame[idx].rgb;
    let n_center = gbuffer[idx].xyz;
    let d_center = gbuffer[idx].w;
    let lum_center = luminance(center);

    // Spatial variance estimate: average the raw per-pixel variance
    // (E[l²] - E[l]²) over a 3x3 neighborhood to tame the noise of a
    // low-sample variance estimate. Scale by the EMA attenuation factor
    // alpha/(2-alpha): for an EMA accumulation this is exactly the residual
    // variance of the history (shrinks toward zero as the history
    // converges), and for accumulate_frames == 0 (alpha = 1) the factor is
    // 1, leaving the fresh-frame variance.
    var var_acc = 0.0;
    var var_cnt = 0.0;
    for (var qy = -1; qy <= 1; qy++) {
        for (var qx = -1; qx <= 1; qx++) {
            let nx = i32(pixel.x) + qx;
            let ny = i32(pixel.y) + qy;
            if nx < 0 || ny < 0 || nx >= i32(params.width) || ny >= i32(params.height) { continue; }
            let qidx = u32(ny) * params.width + u32(nx);
            let m = frame_moments[qidx];
            var_acc += max(m.y - m.x * m.x, 0.0);
            var_cnt += 1.0;
        }
    }
    let var_frame = var_acc / max(var_cnt, 1.0);
    let ema_scale = params.alpha / max(2.0 - params.alpha, 1e-4);
    let var_c = var_frame * ema_scale;

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

            // Variance-guided color stop: only differences well beyond the
            // estimated sampling noise are treated as edges; when the pixel
            // has converged (var_c ≈ 0) ANY luminance difference is signal
            // (texture) and the weight collapses, preserving detail.
            let dl = luminance(neighbor) - lum_center;
            let cw = exp(-(dl * dl) / (var_c * params.lambda + 1e-6));

            let w = sw * nw * dw * cw;
            sum += neighbor * w;
            total_weight += w;
        }
    }

    if total_weight > 1e-6 {
        output_frame[idx] = vec4(sum / total_weight, 1.0);
    } else {
        // Either converged (nothing to smooth) or no similar neighbors.
        output_frame[idx] = vec4(center, 1.0);
    }
}
