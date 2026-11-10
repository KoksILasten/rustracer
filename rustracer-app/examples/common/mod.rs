//! Shared helpers for the headless render examples.

use rustracer_renderer::Renderer;

/// Read the accumulation buffer back as RGBA f32 (row-major, `w*h*4`).
pub fn read_accumulation(renderer: &Renderer) -> anyhow::Result<Vec<f32>> {
    let width = renderer.config.width;
    let height = renderer.config.height;
    let acc = &renderer.buffers.accumulation;
    let size = (width * height) as u64 * 16;
    let staging = renderer.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = renderer.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(acc.buffer(), 0, &staging, 0, size);
    renderer.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    renderer.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let floats: &[f32] = bytemuck::cast_slice(&data);
    Ok(floats.to_vec())
}

/// Save a tonemapped PNG (Reinhard → gamma 2.2) from a raw RGBA f32 buffer.
///
/// Note this is *not* what the app shows: it bypasses the renderer's own
/// tonemap and denoise passes. Prefer [`read_target`] + [`save_png_bgra`] for
/// anything meant to look like the on-screen result.
pub fn save_png(floats: &[f32], width: u32, height: u32, out: &str) -> anyhow::Result<()> {
    let pixels: Vec<u8> = floats
        .chunks_exact(4)
        .flat_map(|c| {
            let map = |v: f32| (v / (1.0 + v)).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0;
            [map(c[0]) as u8, map(c[1]) as u8, map(c[2]) as u8, 255u8]
        })
        .collect();
    image::save_buffer(out, &pixels, width, height, image::ColorType::Rgba8)?;
    Ok(())
}

/// Read a render target back as serialized BGRA8 bytes, undoing wgpu's
/// 256-byte row padding.
pub fn read_target(
    renderer: &Renderer,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> anyhow::Result<Vec<u8>> {
    let unpadded = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;
    let size = padded as u64 * height as u64;
    let staging = renderer.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_target"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = renderer.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &staging,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    renderer.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    renderer.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let mut out = Vec::with_capacity((unpadded * height) as usize);
    for row in 0..height as usize {
        let start = row * padded as usize;
        out.extend_from_slice(&data[start..start + unpadded as usize]);
    }
    Ok(out)
}

/// Save serialized BGRA8 pixels as an RGBA PNG.
pub fn save_png_bgra(bgra: &[u8], width: u32, height: u32, out: &str) -> anyhow::Result<()> {
    let pixels: Vec<u8> = bgra
        .chunks_exact(4)
        .flat_map(|c| [c[2], c[1], c[0], 255u8])
        .collect();
    image::save_buffer(out, &pixels, width, height, image::ColorType::Rgba8)?;
    Ok(())
}
