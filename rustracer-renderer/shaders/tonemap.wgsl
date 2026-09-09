// Tonemap: accumulation → display. No averaging (accumulation is overwritten each frame).

@group(0) @binding(0) var<storage, read> accumulation: array<vec4<f32>>;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: ToneMapParams;

struct ToneMapParams {
    exposure: f32,
    gamma: f32,
    sample_count: u32,
    width: u32,
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = global_id.xy;
    if pixel.x >= params.width { return; }

    let hdr = accumulation[pixel.y * params.width + pixel.x];
    // Apply exposure, then Reinhard, then gamma
    let exposed = hdr.rgb * params.exposure;
    let mapped = exposed / (exposed + 1.0);
    let gamma_corrected = pow(mapped, vec3(1.0 / params.gamma));
    textureStore(output, pixel, vec4(gamma_corrected, 1.0));
}
