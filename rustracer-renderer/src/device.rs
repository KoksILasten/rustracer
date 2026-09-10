//! WGPU device creation and adapter selection.

/// Create a wgpu device and queue from the given adapter.
pub async fn create_device(
    adapter: &wgpu::Adapter,
) -> anyhow::Result<(wgpu::Device, wgpu::Queue)> {
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Rustracer GPU"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: 256 * 1024 * 1024,
                    max_buffer_size: 512 * 1024 * 1024,
                    max_storage_buffers_per_shader_stage: 16,
                    ..wgpu::Limits::downlevel_defaults()
                },
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await?;

    Ok((device, queue))
}

/// Select the best available adapter, preferring discrete GPUs.
pub async fn select_adapter(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'_>>,
) -> anyhow::Result<wgpu::Adapter> {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: surface,
            force_fallback_adapter: false,
        })
        .await
        .ok_or_else(|| anyhow::anyhow!("No suitable GPU adapter found"))?;

    let info = adapter.get_info();
    tracing::info!(
        "Selected GPU: {} ({:?}), backend: {:?}",
        info.name,
        info.device_type,
        info.backend,
    );

    Ok(adapter)
}
