//! Typed GPU buffer management.

use std::marker::PhantomData;
use wgpu::util::DeviceExt;

/// A typed GPU buffer holding elements of type `T`.
pub struct TypedBuffer<T: bytemuck::Pod> {
    buffer: wgpu::Buffer,
    capacity: u64,
    _marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> TypedBuffer<T> {
    pub fn from_slice(
        device: &wgpu::Device,
        label: &str,
        data: &[T],
        usage: wgpu::BufferUsages,
    ) -> Self {
        // wgpu forbids zero-size buffers; keep at least one element and
        // write real data only when there is any.
        let capacity = data.len().max(1) as u64;
        let buffer = if !data.is_empty() {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            })
        } else {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<T>() as u64,
                usage: usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        Self {
            capacity,
            buffer,
            _marker: PhantomData,
        }
    }

    pub fn new_zeroed(
        device: &wgpu::Device,
        label: &str,
        count: u64,
        usage: wgpu::BufferUsages,
    ) -> Self {
        let size = (count as u64).max(1) * std::mem::size_of::<T>() as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self {
            capacity: count,
            buffer,
            _marker: PhantomData,
        }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn write(&self, queue: &wgpu::Queue, data: &[T]) {
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
    }

    pub fn copy_to_staging(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
    ) -> wgpu::Buffer {
        let size = self.capacity * std::mem::size_of::<T>() as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_output"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, size);
        staging
    }
}

/// A uniform buffer for bind group uniform bindings.
pub struct UniformBuffer<T: bytemuck::Pod> {
    buffer: wgpu::Buffer,
    _marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> UniformBuffer<T> {
    pub fn new(device: &wgpu::Device, label: &str, data: &T) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            buffer,
            _marker: PhantomData,
        }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn write(&self, queue: &wgpu::Queue, data: &T) {
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(data));
    }

    pub fn read(&self, _device: &wgpu::Device) -> T {
        unsafe { std::mem::zeroed() }
    }
}
