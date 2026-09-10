// Composite pass: blend the fresh frame into the EMA accumulation history.
// The denoiser runs AFTER this pass, on the accumulated image, and gates on
// the per-frame luminance variance (from gather) scaled by the EMA
// attenuation factor alpha/(2-alpha) — the residual variance of the
// history. As the history converges the denoiser's weights collapse, so it
// stops filtering: noise keeps getting removed, texture detail is left
// alone, and no compounding can occur.

@group(0) @binding(0) var<storage, read> frame: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> accumulation: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: CompositeParams;

struct CompositeParams {
    width: u32,
    height: u32,
    alpha: f32,
    _pad: u32,
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = global_id.xy;
    if pixel.x >= params.width || pixel.y >= params.height { return; }
    let idx = pixel.y * params.width + pixel.x;
    let fresh = frame[idx];
    let prev = accumulation[idx];
    // alpha == 1.0 → overwrite (accumulate_frames == 0 or first frame
    // after reset); otherwise exponential moving average.
    accumulation[idx] = mix(prev, fresh, vec4(params.alpha));
}
