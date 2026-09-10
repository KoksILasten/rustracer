// Composite pass: blend the fresh (optionally denoised) frame into the
// EMA accumulation history. The denoiser must NEVER run on the accumulation
// buffer itself — with accumulate_frames > 0 that re-filters the running
// average every frame and ghosts the image. The gather pass writes the raw
// frame here, the denoiser smooths it in place, and this pass does the
// history blend last.

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
