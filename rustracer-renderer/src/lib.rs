//! Rustracer renderer — wgpu-based GPU path tracing and photon mapping.

pub mod buffer;
pub mod device;
pub mod pipeline;
pub mod shader;

use rustracer_core::camera::CameraUniform;
use rustracer_core::scene::Scene;

// Photon grid over the scene volume. The grid is sized from the scene's
// world-space bounds at renderer creation (cell size auto-scaled so the
// longest axis fits GRID_MAX_DIM cells); buffers are allocated for the
// worst case so per-scene dims only live in the GridParams uniform.
const GRID_MAX_DIM: u32 = 26;
const GRID_MAX_CELLS: u32 = 26 * 26 * 26;

#[derive(Debug, Clone)]
pub struct RenderConfig {
    pub width: u32, pub height: u32,
    pub photon_count: u32, pub max_bounces: u32,
    pub spp: u32,
    pub exposure: f32,
    pub gamma: f32,
    pub light_emission: [f32; 3],
    pub point_light_emission: [f32; 3],
    pub accumulate_frames: u32, // 0=overwrite, N=EMA over N frames
    pub enable_photon_map: bool,
    pub enable_denoise: bool,
    pub photon_debug: bool,
    /// Debug view (0 = normal render; 1 = UVs, 2 = smooth normal,
    /// 3 = perturbed normal, 4 = flat normal).
    pub debug_view: u32,
    pub photon_scale: f32,
    /// Photon lookup radius; auto-scaled from the scene's grid cell size
    /// at renderer creation.
    pub photon_radius: f32,
    pub ambient: f32,
}
impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            width: 800, height: 600,
            photon_count: 1_000_000, max_bounces: 8,
            spp: 1, exposure: 1.0, gamma: 2.2,
            light_emission: [20.0, 20.0, 20.0],
            point_light_emission: [20.0, 20.0, 20.0],
            accumulate_frames: 0,
            enable_photon_map: false,
            enable_denoise: true,
            photon_debug: false,
            debug_view: 0,
            photon_scale: 0.02,
            photon_radius: 0.5,
            ambient: 0.03,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ImageParams { pub width: u32, pub height: u32, pub spp: u32, pub frame: u32, pub light_emission: [f32; 3], pub ambient: f32, pub accumulate_alpha: f32, pub use_photons: u32, pub photon_count: u32, pub photon_radius: f32, pub photon_scale: f32, pub photon_debug: u32, pub point_light_emission: [f32; 3], pub _pad: [u32; 3] }

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PhotonGpu {
    pub position: [f32; 3], pub _pad0: f32, pub power: [f32; 3], pub _pad1: f32,
    pub incident: [f32; 3], pub _pad2: f32, pub normal: [f32; 3], pub _pad3: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SampleCountUniform { pub count: u32, pub _pad: [u32; 3] }

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ToneMapUniform { pub exposure: f32, pub gamma: f32, pub sample_count: u32, pub width: u32 }

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DenoiseParamsUniform { pub width: u32, pub height: u32, pub step_size: u32, pub phi_color: f32, pub phi_normal: f32, pub phi_depth: f32, pub _pad: [u32; 2] }

/// Uniform grid over the scene for photon lookup.
/// All scalar fields (4-byte aligned) to match WGSL exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GridParamsUniform {
    pub room_min_x: f32, pub room_min_y: f32, pub room_min_z: f32,
    pub cell_size: f32,
    pub grid_x: u32, pub grid_y: u32, pub grid_z: u32,
    pub num_cells: u32,
    pub photon_slots: u32,
    pub _pad: [u32; 3],
}

pub struct GpuBuffers {
    pub triangles:    buffer::TypedBuffer<rustracer_core::scene::TriangleGpu>,
    pub bvh_nodes:    buffer::TypedBuffer<rustracer_core::bvh::BVHNode>,
    pub bvh_prims:    buffer::TypedBuffer<u32>,
    pub materials:    buffer::TypedBuffer<rustracer_core::material::MaterialGpu>,
    pub lights:       buffer::TypedBuffer<rustracer_core::scene::LightGpu>,
    pub camera:       buffer::UniformBuffer<CameraUniform>,
    pub photons:      buffer::TypedBuffer<PhotonGpu>,
    pub accumulation: buffer::TypedBuffer<[f32; 4]>,
    pub gbuffer:      buffer::TypedBuffer<[f32; 4]>, // normal.xyz + depth.w
    pub sample_count: buffer::UniformBuffer<SampleCountUniform>,
    pub tonemap_params: buffer::UniformBuffer<ToneMapUniform>,
    pub img_params:   buffer::UniformBuffer<ImageParams>,
    pub denoise_params: buffer::UniformBuffer<DenoiseParamsUniform>,
    pub grid_params:   buffer::UniformBuffer<GridParamsUniform>,
    pub grid_counts:   buffer::TypedBuffer<u32>,
    pub grid_meta:     buffer::TypedBuffer<[u32; 2]>, // [start, count] per cell
    pub grid_cursor:   buffer::TypedBuffer<u32>,
    pub sorted_photons: buffer::TypedBuffer<PhotonGpu>,
    // Role texture arrays (uniform sizes; see rustracer-core::loader).
    pub tex_albedo: wgpu::Texture,
    pub tex_albedo_view: wgpu::TextureView,
    pub tex_mr: wgpu::Texture,
    pub tex_mr_view: wgpu::TextureView,
    pub tex_emissive: wgpu::Texture,
    pub tex_emissive_view: wgpu::TextureView,
    pub tex_normal: wgpu::Texture,
    pub tex_normal_view: wgpu::TextureView,
    pub tex_sampler: wgpu::Sampler,
    // Output is a storage texture (written by tonemap pass)
    pub output_texture: wgpu::Texture,
    pub output_view: wgpu::TextureView,
}

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: RenderConfig,
    pub buffers: GpuBuffers,
    sample_counter: u32,
    first_frame_after_reset: bool,
    photons_dirty: bool,
    photon_trace_pipeline: wgpu::ComputePipeline,
    photon_count_pipeline: wgpu::ComputePipeline,
    photon_scatter_pipeline: wgpu::ComputePipeline,
    gather_pipeline: wgpu::ComputePipeline,
    tonemap_pipeline: wgpu::ComputePipeline,
    quad_pipeline: wgpu::RenderPipeline,
    photon_trace_bgl: wgpu::BindGroupLayout,
    photon_count_bgl: wgpu::BindGroupLayout,
    photon_scatter_bgl: wgpu::BindGroupLayout,
    gather_bgl: wgpu::BindGroupLayout,
    tonemap_bgl: wgpu::BindGroupLayout,
    denoise_pipeline: wgpu::ComputePipeline,
    denoise_bgl: wgpu::BindGroupLayout,
    quad_bgl: wgpu::BindGroupLayout,
    quad_sampler: wgpu::Sampler,
}

impl Renderer {
    pub async fn new(adapter: &wgpu::Adapter, mut config: RenderConfig, scene: &Scene) -> anyhow::Result<Self> {
        let (device, queue) = device::create_device(adapter).await?;

        let triangles = buffer::TypedBuffer::from_slice(&device, "triangles", &scene.flatten_triangles(), wgpu::BufferUsages::STORAGE);
        let bvh_data = scene.build_bvh();
        let bvh_nodes = buffer::TypedBuffer::from_slice(&device, "bvh_nodes", &bvh_data.nodes, wgpu::BufferUsages::STORAGE);
        let bvh_prims = buffer::TypedBuffer::from_slice(&device, "bvh_prims", &bvh_data.prim_indices, wgpu::BufferUsages::STORAGE);
        let materials = buffer::TypedBuffer::from_slice(&device, "materials", &scene.flatten_materials(), wgpu::BufferUsages::STORAGE);
        let lights_raw: Vec<rustracer_core::scene::LightGpu> = scene.lights.iter().map(light_to_gpu).collect();
        let lights = buffer::TypedBuffer::from_slice(&device, "lights", &lights_raw, wgpu::BufferUsages::STORAGE);
        let camera = buffer::UniformBuffer::new(&device, "camera", &rustracer_core::camera::Camera::default().to_uniform(config.width, config.height));

        let num_pixels = (config.width * config.height) as u64;
        let photons = buffer::TypedBuffer::new_zeroed(&device, "photons", config.photon_count as u64 * config.max_bounces as u64, wgpu::BufferUsages::STORAGE);
        let accumulation = buffer::TypedBuffer::new_zeroed(&device, "accumulation", num_pixels, wgpu::BufferUsages::STORAGE);
        let gbuffer = buffer::TypedBuffer::new_zeroed(&device, "gbuffer", num_pixels, wgpu::BufferUsages::STORAGE);
        let sample_count = buffer::UniformBuffer::new(&device, "sample_count", &SampleCountUniform { count: 0, _pad: [0; 3] });
        let tonemap_params = buffer::UniformBuffer::new(&device, "tonemap_params", &ToneMapUniform { exposure: 1.0, gamma: 2.2, sample_count: 0, width: config.width });
        let img_params = buffer::UniformBuffer::new(&device, "img_params", &ImageParams { width: config.width, height: config.height, spp: config.spp, frame: 0, light_emission: config.light_emission, ambient: config.ambient, accumulate_alpha: 0.0, use_photons: if config.enable_photon_map { 1 } else { 0 }, photon_count: config.photon_count, photon_radius: config.photon_radius, photon_scale: config.photon_scale, photon_debug: if config.photon_debug { 1 } else { 0 }, point_light_emission: config.point_light_emission, _pad: [config.debug_view, 0, 0] });
        let denoise_params = buffer::UniformBuffer::new(&device, "denoise_params", &DenoiseParamsUniform { width: config.width, height: config.height, step_size: 1, phi_color: 0.35, phi_normal: 12.0, phi_depth: 0.05, _pad: [0; 2] });
        // Per-scene photon grid: cover the scene bounds with a cell size
        // that fits the longest axis into GRID_MAX_DIM cells.
        let (bmin, bmax) = scene.bounds().unwrap_or((glam::Vec3::splat(-1.05), glam::Vec3::splat(1.05)));
        let ext = bmax - bmin;
        let max_ext = ext.max_element().max(1e-3);
        let cell_size = max_ext / (GRID_MAX_DIM as f32 - 2.0);
        let dim = |e: f32| ((e / cell_size).ceil() as u32).clamp(1, GRID_MAX_DIM);
        let grid_x = dim(ext.x);
        let grid_y = dim(ext.y);
        let grid_z = dim(ext.z);
        let num_cells = grid_x * grid_y * grid_z;
        let room_min = bmin - glam::Vec3::splat(cell_size * 0.5);
        // Photon lookup radius scales with the scene: a few cells wide.
        config.photon_radius = (cell_size * 4.0).clamp(0.01, 2.0);
        tracing::info!(
            "Photon grid {}x{}x{} = {} cells, cell {:.4}, radius {:.4}, origin {:?}",
            grid_x, grid_y, grid_z, num_cells, cell_size, config.photon_radius, room_min
        );
        let grid_params = buffer::UniformBuffer::new(&device, "grid_params", &GridParamsUniform {
            room_min_x: room_min.x, room_min_y: room_min.y, room_min_z: room_min.z,
            cell_size,
            grid_x, grid_y, grid_z,
            num_cells,
            photon_slots: config.photon_count * config.max_bounces,
            _pad: [0; 3],
        });
        let max_stored = (config.photon_count * config.max_bounces) as u64;
        let grid_counts = buffer::TypedBuffer::new_zeroed(&device, "grid_counts", GRID_MAX_CELLS as u64, wgpu::BufferUsages::STORAGE);
        let grid_meta = buffer::TypedBuffer::new_zeroed(&device, "grid_meta", GRID_MAX_CELLS as u64, wgpu::BufferUsages::STORAGE);
        let grid_cursor = buffer::TypedBuffer::new_zeroed(&device, "grid_cursor", GRID_MAX_CELLS as u64, wgpu::BufferUsages::STORAGE);
        let sorted_photons = buffer::TypedBuffer::new_zeroed(&device, "sorted_photons", max_stored, wgpu::BufferUsages::STORAGE);

        // Storage texture for tonemap output
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("output_texture"),
            size: wgpu::Extent3d { width: config.width, height: config.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Role texture arrays + shared sampler for material sampling.
        let (tex_albedo, tex_albedo_view) = create_role_array(&device, &queue, "tex_albedo", &scene.tex_albedo);
        let (tex_mr, tex_mr_view) = create_role_array(&device, &queue, "tex_mr", &scene.tex_mr);
        let (tex_emissive, tex_emissive_view) = create_role_array(&device, &queue, "tex_emissive", &scene.tex_emissive);
        let (tex_normal, tex_normal_view) = create_role_array(&device, &queue, "tex_normal", &scene.tex_normal);
        let tex_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("material_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let buffers = GpuBuffers { triangles, bvh_nodes, bvh_prims, materials, lights, camera, photons, accumulation, gbuffer, sample_count, tonemap_params, img_params, denoise_params, grid_params, grid_counts, grid_meta, grid_cursor, sorted_photons, tex_albedo, tex_albedo_view, tex_mr, tex_mr_view, tex_emissive, tex_emissive_view, tex_normal, tex_normal_view, tex_sampler, output_texture, output_view };

        // Sampler for fullscreen quad
        let quad_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let shaders = shader::ShaderBundle::load(&device)?;
        let (photon_trace_pipeline, photon_trace_bgl) = pipeline::create_photon_trace_pipeline(&device, &shaders, &buffers);
        let (photon_count_pipeline, photon_count_bgl) = pipeline::create_photon_count_pipeline(&device, &shaders);
        let (photon_scatter_pipeline, photon_scatter_bgl) = pipeline::create_photon_scatter_pipeline(&device, &shaders);
        let (gather_pipeline, gather_bgl) = pipeline::create_gather_pipeline(&device, &shaders, &buffers);
        let (tonemap_pipeline, tonemap_bgl) = pipeline::create_tonemap_pipeline(&device, &shaders, &buffers);
        let (denoise_pipeline, denoise_bgl) = pipeline::create_denoise_pipeline(&device, &shaders);
        let (quad_pipeline, quad_bgl) = pipeline::create_quad_pipeline(&device, &shaders);

        Ok(Self { device, queue, config, buffers, sample_counter: 0, first_frame_after_reset: true, photons_dirty: true, photon_trace_pipeline, photon_count_pipeline, photon_scatter_pipeline, gather_pipeline, tonemap_pipeline, denoise_pipeline, quad_pipeline, photon_trace_bgl, photon_count_bgl, photon_scatter_bgl, gather_bgl, tonemap_bgl, denoise_bgl, quad_bgl, quad_sampler })
    }

    /// Record all rendering commands into the encoder. No queue.write_buffer calls.
    pub fn record_frame(&mut self, encoder: &mut wgpu::CommandEncoder, surface_view: &wgpu::TextureView) {
        // Increment frame counter for temporal noise
        self.sample_counter = self.sample_counter.wrapping_add(1);
        // Force overwrite on first frame after reset to avoid flicker
        let alpha = if self.first_frame_after_reset { self.first_frame_after_reset = false; 1.0 }
                    else if self.config.accumulate_frames > 0 { 1.0 / self.config.accumulate_frames as f32 }
                    else { 1.0 };
        self.buffers.img_params.write(&self.queue, &ImageParams {
            width: self.config.width, height: self.config.height,
            spp: self.config.spp,
            frame: self.sample_counter,
            light_emission: self.config.light_emission,
            ambient: self.config.ambient,
            accumulate_alpha: alpha,
            use_photons: if self.config.enable_photon_map { 1 } else { 0 },
            photon_count: self.config.photon_count,
            photon_radius: self.config.photon_radius,
            photon_scale: self.config.photon_scale,
            photon_debug: if self.config.photon_debug { 1 } else { 0 },
            point_light_emission: self.config.point_light_emission,
            _pad: [self.config.debug_view, 0, 0],
        });
        self.buffers.tonemap_params.write(&self.queue, &ToneMapUniform {
            exposure: self.config.exposure,
            gamma: self.config.gamma,
            sample_count: self.sample_counter,
            width: self.config.width,
        });

        // Pass 0: Photon trace runs only during rebuild_photons() (trace-once).

        // Pass 1: Gather (camera rays → accumulation buffer)
        {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("gather_bg"), layout: &self.gather_bgl, entries: &self.buffers.gather_entries() });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("gather"), timestamp_writes: None });
            pass.set_pipeline(&self.gather_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((self.config.width + 7) / 8, (self.config.height + 7) / 8, 1);
        }

        // Pass 2: Denoise (5 A-Trous iterations with increasing step sizes)
        if self.config.enable_denoise {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("denoise_bg"), layout: &self.denoise_bgl, entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buffers.accumulation.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buffers.gbuffer.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buffers.denoise_params.buffer().as_entire_binding() },
            ]});
            for iter in 0..5u32 {
                let step = 1u32 << iter;
                self.buffers.denoise_params.write(&self.queue, &DenoiseParamsUniform {
                    width: self.config.width, height: self.config.height,
                    step_size: step, phi_color: 0.35, phi_normal: 12.0, phi_depth: 0.05, _pad: [0; 2],
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("denoise"), timestamp_writes: None });
                pass.set_pipeline(&self.denoise_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups((self.config.width + 7) / 8, (self.config.height + 7) / 8, 1);
            }
        }

        // Pass 3: Tonemap → storage texture
        {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("tonemap_bg"), layout: &self.tonemap_bgl, entries: &self.buffers.tonemap_entries() });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("tonemap"), timestamp_writes: None });
            pass.set_pipeline(&self.tonemap_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((self.config.width + 7) / 8, (self.config.height + 7) / 8, 1);
        }

        // Pass 4: Fullscreen quad → swapchain
        {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("quad_bg"),
                layout: &self.quad_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.buffers.output_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.quad_sampler) },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("quad_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.quad_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    pub fn update_camera(&mut self, camera: &rustracer_core::camera::Camera) {
        let uniform = camera.to_uniform(self.config.width, self.config.height);
        self.buffers.camera.write(&self.queue, &uniform);
    }

    pub fn reset_accumulation(&mut self) {
        let num_pixels = (self.config.width * self.config.height) as u64;
        self.buffers.accumulation = buffer::TypedBuffer::new_zeroed(&self.device, "accumulation", num_pixels, wgpu::BufferUsages::STORAGE);
        self.first_frame_after_reset = true;
    }

    /// Trace photons once and build the spatial hash grid.
    /// Called when photons are first needed or the scene/lighting changes.
    pub fn rebuild_photons(&mut self) -> anyhow::Result<()> {
        let num_photons = (self.config.photon_count as u64).max(1);
        let trace_wgs = (num_photons as u32 / 256).max(1);
        let slots_wgs = ((num_photons as u32 * self.config.max_bounces as u32) / 256).max(1);

        // Pass A: trace photons + count per cell
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("photon_rebuild_a") });
            let tbg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("trace_bg"), layout: &self.photon_trace_bgl, entries: &self.buffers.photon_trace_entries() });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("photon_trace"), timestamp_writes: None });
                pass.set_pipeline(&self.photon_trace_pipeline);
                pass.set_bind_group(0, &tbg, &[]);
                pass.dispatch_workgroups(trace_wgs, 1, 1);
            }
            let cbg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("count_bg"), layout: &self.photon_count_bgl, entries: &self.buffers.photon_count_entries() });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("photon_count"), timestamp_writes: None });
                pass.set_pipeline(&self.photon_count_pipeline);
                pass.set_bind_group(0, &cbg, &[]);
                pass.dispatch_workgroups(slots_wgs, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }

        // Read back grid_counts, compute prefix sums (CPU, rare operation)
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid_counts_staging"),
            size: GRID_MAX_CELLS as u64 * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("photon_readback") });
        encoder.copy_buffer_to_buffer(self.buffers.grid_counts.buffer(), 0, &staging, 0, GRID_MAX_CELLS as u64 * 4);
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().map_err(|_| anyhow::anyhow!("grid readback failed"))??;

        let mut starts = vec![0u32; GRID_MAX_CELLS as usize];
        let mut counts_vec = vec![0u32; GRID_MAX_CELLS as usize];
        {
            let data = slice.get_mapped_range();
            let counts: &[u32] = bytemuck::cast_slice(&data);
            let mut acc = 0u32;
            for i in 0..GRID_MAX_CELLS as usize {
                starts[i] = acc;
                counts_vec[i] = counts[i];
                acc += counts[i];
            }
        }

        // Write grid_meta as [start, count] pairs, zero cursor, then scatter
        let meta: Vec<[u32; 2]> = (0..GRID_MAX_CELLS as usize)
            .map(|i| [starts[i], counts_vec[i]])
            .collect();
        self.buffers.grid_meta.write(&self.queue, &meta);
        self.buffers.grid_cursor.write(&self.queue, &vec![0u32; GRID_MAX_CELLS as usize]);
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("photon_rebuild_b") });
            let sbg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("scatter_bg"), layout: &self.photon_scatter_bgl, entries: &self.buffers.photon_scatter_entries() });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("photon_scatter"), timestamp_writes: None });
                pass.set_pipeline(&self.photon_scatter_pipeline);
                pass.set_bind_group(0, &sbg, &[]);
                pass.dispatch_workgroups(slots_wgs, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }

        self.photons_dirty = false;
        tracing::info!("Photon map rebuilt: {} photons, {} cells", num_photons, GRID_MAX_CELLS);
        Ok(())
    }

    pub fn sample_count(&self) -> u32 { self.sample_counter }
}

impl GpuBuffers {
    fn photon_trace_entries(&self) -> [wgpu::BindGroupEntry<'_>; 10] {
        [
            wgpu::BindGroupEntry { binding: 0, resource: self.photons.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: self.bvh_nodes.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: self.bvh_prims.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: self.triangles.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: self.lights.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: self.materials.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.tex_albedo_view) },
            wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&self.tex_sampler) },
            wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&self.tex_normal_view) },
            wgpu::BindGroupEntry { binding: 9, resource: self.grid_params.buffer().as_entire_binding() },
        ]
    }

    fn gather_entries(&self) -> [wgpu::BindGroupEntry<'_>; 17] {
        [
            wgpu::BindGroupEntry { binding: 0, resource: self.sorted_photons.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: self.accumulation.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: self.camera.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: self.bvh_nodes.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: self.bvh_prims.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: self.triangles.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: self.materials.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: self.img_params.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: self.gbuffer.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: self.grid_meta.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: self.grid_params.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: self.lights.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(&self.tex_albedo_view) },
            wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(&self.tex_mr_view) },
            wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(&self.tex_emissive_view) },
            wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::Sampler(&self.tex_sampler) },
            wgpu::BindGroupEntry { binding: 16, resource: wgpu::BindingResource::TextureView(&self.tex_normal_view) },
        ]
    }

    fn photon_count_entries(&self) -> [wgpu::BindGroupEntry<'_>; 3] {
        [
            wgpu::BindGroupEntry { binding: 0, resource: self.photons.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: self.grid_counts.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: self.grid_params.buffer().as_entire_binding() },
        ]
    }

    fn photon_scatter_entries(&self) -> [wgpu::BindGroupEntry<'_>; 5] {
        [
            wgpu::BindGroupEntry { binding: 0, resource: self.photons.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: self.grid_meta.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: self.grid_cursor.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: self.sorted_photons.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: self.grid_params.buffer().as_entire_binding() },
        ]
    }

    fn tonemap_entries(&self) -> [wgpu::BindGroupEntry<'_>; 3] {
        [
            wgpu::BindGroupEntry { binding: 0, resource: self.accumulation.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.output_view) },
            wgpu::BindGroupEntry { binding: 2, resource: self.tonemap_params.buffer().as_entire_binding() },
        ]
    }
}

fn light_to_gpu(light: &rustracer_core::scene::Light) -> rustracer_core::scene::LightGpu {
    match light {
        rustracer_core::scene::Light::Point { position, color, intensity } =>
            rustracer_core::scene::LightGpu { data: [position.x, position.y, position.z, 0.0], color: color.to_array(), intensity: *intensity, kind: 0, _pad_a: [0.0; 3], _pad_b: [0.0; 4] },
        rustracer_core::scene::Light::Directional { direction, color, intensity } =>
            rustracer_core::scene::LightGpu { data: [direction.x, direction.y, direction.z, 0.0], color: color.to_array(), intensity: *intensity, kind: 1, _pad_a: [0.0; 3], _pad_b: [0.0; 4] },
        rustracer_core::scene::Light::Area { triangle_index, color, intensity } =>
            rustracer_core::scene::LightGpu { data: [*triangle_index as f32, 0.0, 0.0, 0.0], color: color.to_array(), intensity: *intensity, kind: 2, _pad_a: [0.0; 3], _pad_b: [0.0; 4] },
        rustracer_core::scene::Light::Environment { intensity, .. } =>
            rustracer_core::scene::LightGpu { data: [0.0; 4], color: [1.0, 1.0, 1.0], intensity: *intensity, kind: 3, _pad_a: [0.0; 3], _pad_b: [0.0; 4] },
    }
}

/// Upload a role's textures as one `texture_2d_array` (layers = texture
/// count; empty roles get a 1x1x1 placeholder so bindings stay valid).
fn create_role_array(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    textures: &[rustracer_core::texture::Texture],
) -> (wgpu::Texture, wgpu::TextureView) {
    let layers = textures.len().max(1) as u32;
    let (width, height) = textures
        .first()
        .map(|t| (t.width, t.height))
        .unwrap_or((1, 1));

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: layers },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    for (i, tex) in textures.iter().enumerate() {
        // Quantize the linear floats to unorm bytes (values are [0,1]).
        let bytes: Vec<u8> = tex
            .data
            .iter()
            .flat_map(|c| {
                [
                    (c[0].clamp(0.0, 1.0) * 255.0) as u8,
                    (c[1].clamp(0.0, 1.0) * 255.0) as u8,
                    (c[2].clamp(0.0, 1.0) * 255.0) as u8,
                    (c[3].clamp(0.0, 1.0) * 255.0) as u8,
                ]
            })
            .collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: i as u32 },
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(width * 4), rows_per_image: Some(height) },
            wgpu::Extent3d { width: tex.width.min(width), height: tex.height.min(height), depth_or_array_layers: 1 },
        );
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some(&format!("{label}_view")),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    (texture, view)
}
