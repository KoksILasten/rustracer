// Tonemap: accumulated frame → display (exposure → Reinhard → gamma).

@group(0) @binding(0) var<storage, read> accumulation: array<vec4<f32>>;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: ToneMapParams;
@group(0) @binding(3) var<storage, read> gbuffer: array<vec4<f32>>;

struct ToneMapParams {
    exposure: f32,
    gamma: f32,
    width: u32,
    // Gizmo brightness multiplier.
    glow: f32,
    // xy = gizmo centre in pixels, z = radius in pixels, w = camera distance
    // of the light (0 = hidden). Used as a per-pixel occlusion test.
    marker: vec4<f32>,
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = global_id.xy;
    if pixel.x >= params.width { return; }

    let hdr = accumulation[pixel.y * params.width + pixel.x];
    // Apply exposure, then Reinhard, then gamma
    let exposed = hdr.rgb * params.exposure;
    let mapped = exposed / (exposed + 1.0);
    var color = pow(mapped, vec3(1.0 / params.gamma));

    // Light gizmo: a small warm ball with a soft halo, drawn in display
    // space so it stays crisp no matter how many samples have accumulated.
    if params.marker.w > 0.0 {
        let p = vec2<f32>(f32(pixel.x) + 0.5, f32(pixel.y) + 0.5);
        let d = distance(p, params.marker.xy);
        let r = max(params.marker.z, 1.5);
        let hit_dist = gbuffer[pixel.y * params.width + pixel.x].w;
        // The marker's distance from camera is params.marker.w. If a surface was hit closer, occlude it.
        if hit_dist > params.marker.w - 0.2 {
            let ball = 1.0 - smoothstep(r * 0.75, r, d);
            let halo = 1.0 - smoothstep(r, r * 3.5, d);
            let core = vec3<f32>(1.0, 0.94, 0.80) * pow(ball, 1.4) * 0.85;
            let ring = vec3<f32>(1.0, 0.46, 0.14) * halo * halo * 0.8;
            // Soft-saturating add: raising glow gets brighter without flattening the
            // whole disc to a clipped white circle.
            color = color + (core + ring) * params.glow * (1.0 - 0.55 * color);
        }
    }

    textureStore(output, pixel, vec4(color, 1.0));
}
