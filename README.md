# Rustracer - WIP

<img width="796" height="626" alt="image" src="https://github.com/user-attachments/assets/4533adf7-41a7-4e18-aad2-acea41f39d2c" />


GPU-accelerated path tracer with photon mapping, written in Rust on wgpu.

An interactive renderer: the image accumulates progressively on the GPU while you navigate the scene, and an egui panel gives live control over samples, exposure, tonemapping, the photon map and the denoiser.

## Features

- Progressive path tracing in WGSL on wgpu (WebGPU), with continuous accumulation while the camera moves
- Optional photon mapping pass for indirect light, with up to a million photons stored in a uniform grid for fast lookup
- Two-pass edge-aware à-trous denoise on the G-buffer (normals, depth and per-pixel variance), enabled by default. It stands down as accumulation progresses, so a converged frame keeps its texture detail
- Physically based material model: Lambertian diffuse, GGX metal with roughness, dielectric glass with index of refraction, emissive area lights
- glTF scene loading with albedo, roughness, metallic and normal textures
- `scene.toml` override file for environment intensity, sky gradient, extra point lights and material overrides
- BVH acceleration structure over triangles
- Accumulation budget (`Accum. Frames`): the running mean warms up as `1/n`, then settles to a fixed history. The budget applies to a still frame; an animated light keeps a short history so its own motion cannot smear
- Orbiting point light with a visible glowing ball: size, brightness and spin speed in the panel. The ball is depth-tested against the G-buffer, so scene geometry occludes it
- egui control panel: samples per pixel, accumulation budget, exposure, gamma, ambient, light emission, photon count and radius, photon debug view, denoise toggle, moving-light size/glow/speed
- Save the current frame as PNG with `S`

## Architecture

A Cargo workspace with three crates:

| Crate | Role |
|---|---|
| `rustracer-core` | Host-side scene representation: math, materials, textures, lights, BVH, glTF loader. GPU-bound types use `#[repr(C)]` layouts compatible with WGSL structs |
| `rustracer-renderer` | wgpu device and buffer management, accumulation, G-buffer, photon map, denoise compute pipeline, tonemapping |
| `rustracer-app` | winit window and egui shell; camera navigation and keyboard shortcuts |

## Build and run

```bash
cargo run --release -- path/to/scene.gltf
```

The scene path defaults to `scene.gltf` in the working directory. A `scene.toml` next to it is optional and overrides materials and lights beyond what the glTF file provides.

Requires a GPU with Vulkan or DirectX 12 support (wgpu picks the backend automatically).

Development commands are in the `justfile` (`just check`, `just test`, `just validate-shaders`).

## Headless renders

Three examples render without a window, which is how visual regressions get checked and how the portfolio stills are produced.

```bash
# Reproduce what the app shows, without a window: spp 4, 32-frame accumulation, denoise on
cargo run --release -p rustracer-app --example render_headless -- \
  assets/scenes/DamagedHelmet/DamagedHelmet.gltf 128 out.png 1 1 0 4 32

# Advertising still: the scene is dropped into a built-in studio (glossy floor, emissive
# softbox panels) instead of an empty room. args: <gltf> [out] [frames] [w] [h]
cargo run --release -p rustracer-app --example studio -- \
  assets/scenes/DamagedHelmet/DamagedHelmet.gltf studio.png 200 900 1200

# Cornell box, for checking colour bleed and the denoiser
cargo run --release -p rustracer-app --example cornell
```

`render_headless` takes every knob the app has, as positional arguments; the full list is in the file header. `studio` takes the same plus yaw, pitch and environment intensity, and `RS_LIGHT_MARKER="x,y,z,radius[,glow]"` places the glowing ball at an explicit point — that is how the occlusion tests drive it.

## Status

In development! Personal portfolio project. I implemented the renderer from the BVH and material system to the photon map and denoise compute shaders.

## Sample scenes

Three glTF scenes from the Khronos glTF-Sample-Assets collection are set up for local testing. They are not committed to the repo; fetch them with:

```bash
python assets/scripts/download_sample_scenes.py
```

This downloads ToyCar (CC0, 109k triangles), DamagedHelmet (CC-BY 4.0, attribution in its LICENSE.md) and Lantern (CC0) into `assets/scenes/`.

```bash
cargo run --release -- assets/scenes/ToyCar/ToyCar.gltf
cargo run --release -- assets/scenes/DamagedHelmet/DamagedHelmet.gltf
cargo run --release -- assets/scenes/Lantern/Lantern.gltf
```

## License

MIT OR Apache-2.0. See [LICENSE](LICENSE). Sample scene licenses live next to each scene under `assets/scenes/`.
