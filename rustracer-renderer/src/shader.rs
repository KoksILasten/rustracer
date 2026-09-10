//! Shader loading and preprocessing.
//!
//! Loads `.wgsl` files from the `shaders/` directory, resolves
//! `#include` directives by inlining referenced files.

use wgpu::ShaderModule;

/// Pre-loaded shader modules for each rendering pass.
pub struct ShaderBundle {
    pub photon_trace: ShaderModule,
    pub photon_count: ShaderModule,
    pub photon_scatter: ShaderModule,
    pub gather: ShaderModule,
    pub tonemap: ShaderModule,
    pub denoise: ShaderModule,
    pub composite: ShaderModule,
    pub fullscreen: ShaderModule,
}

impl ShaderBundle {
    /// Load and compile all shaders from the shaders directory.
    pub fn load(device: &wgpu::Device) -> anyhow::Result<Self> {
        // At compile time, the shaders are embedded via include_str!.
        // This avoids runtime file I/O issues.
        Ok(Self {
            photon_trace: Self::compile(device, "photon_trace", &preprocess_shader(
                include_str!("../shaders/photon_trace.wgsl"),
                &[("common.wgsl", include_str!("../shaders/common.wgsl"))],
            )),
            photon_count: Self::compile(device, "photon_count", &preprocess_shader(
                include_str!("../shaders/photon_count.wgsl"),
                &[("common.wgsl", include_str!("../shaders/common.wgsl"))],
            )),
            photon_scatter: Self::compile(device, "photon_scatter", &preprocess_shader(
                include_str!("../shaders/photon_scatter.wgsl"),
                &[("common.wgsl", include_str!("../shaders/common.wgsl"))],
            )),
            gather: Self::compile(device, "gather", &preprocess_shader(
                include_str!("../shaders/gather.wgsl"),
                &[("common.wgsl", include_str!("../shaders/common.wgsl"))],
            )),
            tonemap: Self::compile(device, "tonemap", include_str!("../shaders/tonemap.wgsl")),
            denoise: Self::compile(device, "denoise", include_str!("../shaders/denoise.wgsl")),
            composite: Self::compile(device, "composite", include_str!("../shaders/composite.wgsl")),
            fullscreen: Self::compile(device, "fullscreen", include_str!("../shaders/fullscreen.wgsl")),
        })
    }

    fn compile(device: &wgpu::Device, name: &str, source: &str) -> ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)),
        })
    }
}

/// Very simple `#include <filename>` preprocessor.
/// Scans for `#include <name>` lines and replaces them with the file contents.
fn preprocess_shader(source: &str, includes: &[(&str, &str)]) -> String {
    let mut result = String::with_capacity(source.len());
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("#include ") {
            let name = trimmed["#include ".len()..].trim();
            if let Some((_, content)) = includes.iter().find(|(n, _)| n == &name) {
                result.push_str(content);
                result.push('\n');
            } else {
                tracing::warn!("Shader include '{}' not found", name);
                result.push_str(line);
                result.push('\n');
            }
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }
    result
}
