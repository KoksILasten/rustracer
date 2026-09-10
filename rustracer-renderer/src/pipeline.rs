//! Compute + render pipeline creation.

use crate::shader::ShaderBundle;
use crate::GpuBuffers;

pub fn create_photon_trace_pipeline(
    device: &wgpu::Device, shaders: &ShaderBundle, _buffers: &GpuBuffers,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("photon_trace_bgl"),
        entries: &[
            storage_buffer_binding(0, true), storage_buffer_binding(1, false),
            storage_buffer_binding(2, false), storage_buffer_binding(3, false),
            storage_buffer_binding(4, false), storage_buffer_binding(5, false),
            texture_array_binding(6), sampler_binding(7), texture_array_binding(8),
            uniform_buffer_binding(9),
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("photon_trace_layout"), bind_group_layouts: &[&bgl], push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("photon_trace"), layout: Some(&layout), module: &shaders.photon_trace,
        entry_point: Some("main"), compilation_options: Default::default(), cache: None,
    });
    (pipeline, bgl)
}

pub fn create_photon_count_pipeline(
    device: &wgpu::Device, shaders: &ShaderBundle,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("photon_count_bgl"),
        entries: &[storage_buffer_binding(0, false), storage_buffer_binding(1, true), uniform_buffer_binding(2)],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("photon_count_layout"), bind_group_layouts: &[&bgl], push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("photon_count"), layout: Some(&layout), module: &shaders.photon_count,
        entry_point: Some("main"), compilation_options: Default::default(), cache: None,
    });
    (pipeline, bgl)
}

pub fn create_photon_scatter_pipeline(
    device: &wgpu::Device, shaders: &ShaderBundle,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("photon_scatter_bgl"),
        entries: &[
            storage_buffer_binding(0, false), storage_buffer_binding(1, false),
            storage_buffer_binding(2, true), storage_buffer_binding(3, true),
            uniform_buffer_binding(4),
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("photon_scatter_layout"), bind_group_layouts: &[&bgl], push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("photon_scatter"), layout: Some(&layout), module: &shaders.photon_scatter,
        entry_point: Some("main"), compilation_options: Default::default(), cache: None,
    });
    (pipeline, bgl)
}

pub fn create_gather_pipeline(
    device: &wgpu::Device, shaders: &ShaderBundle, _buffers: &GpuBuffers,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gather_bgl"),
        entries: &[
            storage_buffer_binding(0, false), storage_buffer_binding(1, true),
            uniform_buffer_binding(2), storage_buffer_binding(3, false),
            storage_buffer_binding(4, false), storage_buffer_binding(5, false),
            storage_buffer_binding(6, false), uniform_buffer_binding(7),
            storage_buffer_binding(8, true), storage_buffer_binding(9, false),
            uniform_buffer_binding(10), storage_buffer_binding(11, false),
            texture_array_binding(12), texture_array_binding(13), texture_array_binding(14),
            sampler_binding(15), texture_array_binding(16),
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gather_layout"), bind_group_layouts: &[&bgl], push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gather"), layout: Some(&layout), module: &shaders.gather,
        entry_point: Some("main"), compilation_options: Default::default(), cache: None,
    });
    (pipeline, bgl)
}

pub fn create_tonemap_pipeline(
    device: &wgpu::Device, shaders: &ShaderBundle, _buffers: &GpuBuffers,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tonemap_bgl"),
        entries: &[
            storage_buffer_binding(0, false),
            storage_texture_binding(1, wgpu::StorageTextureAccess::WriteOnly, wgpu::TextureFormat::Rgba8Unorm),
            uniform_buffer_binding(2),
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("tonemap_layout"), bind_group_layouts: &[&bgl], push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("tonemap"), layout: Some(&layout), module: &shaders.tonemap,
        entry_point: Some("main"), compilation_options: Default::default(), cache: None,
    });
    (pipeline, bgl)
}

pub fn create_denoise_pipeline(
    device: &wgpu::Device, shaders: &ShaderBundle,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("denoise_bgl"),
        entries: &[storage_buffer_binding(0, true), storage_buffer_binding(1, false), uniform_buffer_binding(2)],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("denoise_layout"), bind_group_layouts: &[&bgl], push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("denoise"), layout: Some(&layout), module: &shaders.denoise,
        entry_point: Some("main"), compilation_options: Default::default(), cache: None,
    });
    (pipeline, bgl)
}

pub fn create_quad_pipeline(
    device: &wgpu::Device, shaders: &ShaderBundle,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("quad_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("quad_layout"), bind_group_layouts: &[&bgl], push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("quad"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shaders.fullscreen,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shaders.fullscreen,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Bgra8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        depth_stencil: None,
        cache: None,
    });
    (pipeline, bgl)
}

#[inline]
fn storage_buffer_binding(binding: u32, read_write: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: !read_write },
            has_dynamic_offset: false, min_binding_size: None,
        },
        count: None,
    }
}

#[inline]
fn uniform_buffer_binding(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false, min_binding_size: None,
        },
        count: None,
    }
}

#[inline]
fn texture_array_binding(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    }
}

#[inline]
fn sampler_binding(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

#[inline]
fn storage_texture_binding(binding: u32, access: wgpu::StorageTextureAccess, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}
