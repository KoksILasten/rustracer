// Composite pass: blend the fresh (denoised) frame into the EMA accumulation
// history. The denoiser runs BEFORE this pass, on the fresh frame only —
// never on the history buffer, which would re-filter the running average.

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
