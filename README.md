# Rustracer

<img width="800" height="630" alt="image" src="https://github.com/user-attachments/assets/ed0c0d82-06b3-46c1-b06a-72c018122755" />

GPU-accelerated path tracer with photon mapping, written in Rust on wgpu.

An interactive renderer: the image accumulates progressively on the GPU while you navigate the scene, and an egui panel gives live control over samples, exposure, tonemapping, the photon map and the denoiser.

## Features

- Progressive path tracing in WGSL on wgpu (WebGPU), with continuous accumulation while the camera moves
- Optional photon mapping pass for indirect light, with up to a million photons stored in a uniform grid for fast lookup
- Edge-aware bilateral denoise pass on the G-buffer (normals and depth), enabled by default
- Physically based material model: Lambertian diffuse, GGX metal with roughness, dielectric glass with index of refraction, emissive area lights
- glTF scene loading with albedo, roughness, metallic and normal textures
- `scene.toml` override file for environment intensity, sky gradient, extra point lights and material overrides
- BVH acceleration structure over triangles
- egui control panel: samples per pixel, exposure, gamma, ambient, light emission, photon count and radius, photon debug view, denoise toggle
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

## Status

In development. Personal portfolio project; the renderer is my own implementation, from the BVH and material system to the photon map and denoise compute shaders.

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

Headless loading check (no window):

```bash
cargo run -p rustracer-app --example loadcheck -- assets/scenes/ToyCar/ToyCar.gltf
```

Note on lighting: glTF punctual light extensions are not imported yet. Scenes without emissive materials render under the sky environment, and `scene.toml` point lights can be added for more direction.

## License

MIT OR Apache-2.0. See [LICENSE](LICENSE). Sample scene licenses live next to each scene under `assets/scenes/`.
