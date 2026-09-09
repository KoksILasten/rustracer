//! Texture types and loading.

use image::GenericImageView;
use std::path::Path;

/// A 2D texture, either loaded from an image or procedurally generated.
#[derive(Debug, Clone)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub data: Vec<[f32; 4]>, // RGBA, linear float
}

impl Texture {
    /// Load a texture from an image file (supports PNG, JPEG, HDR, EXR, etc.).
    pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let img = image::open(path)?;
        let (w, h) = img.dimensions();
        let data: Vec<[f32; 4]> = img
            .to_rgba32f()
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();
        Ok(Self {
            width: w,
            height: h,
            data,
        })
    }

    /// Create a solid-color 1x1 texture (useful as a default).
    pub fn solid(color: [f32; 4]) -> Self {
        Self {
            width: 1,
            height: 1,
            data: vec![color],
        }
    }

    /// Create a procedural gradient sky texture.
    pub fn gradient_sky(width: u32, height: u32, zenith: [f32; 3], horizon: [f32; 3]) -> Self {
        let mut data = vec![[0.0; 4]; (width * height) as usize];
        for y in 0..height {
            let t = y as f32 / height as f32;
            let color = [
                zenith[0] + (horizon[0] - zenith[0]) * t,
                zenith[1] + (horizon[1] - zenith[1]) * t,
                zenith[2] + (horizon[2] - zenith[2]) * t,
                1.0,
            ];
            for x in 0..width {
                data[(y * width + x) as usize] = color;
            }
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Sample the texture with bilinear filtering.
    pub fn sample_bilinear(&self, u: f32, v: f32) -> [f32; 4] {
        let u = u.fract().rem_euclid(1.0);
        let v = v.fract().rem_euclid(1.0);
        let x = u * self.width as f32 - 0.5;
        let y = v * self.height as f32 - 0.5;
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        let idx = |ix: i32, iy: i32| {
            let ix = ix.rem_euclid(self.width as i32) as u32;
            let iy = iy.rem_euclid(self.height as i32) as u32;
            self.data[(iy * self.width + ix) as usize]
        };

        let c00 = idx(x0, y0);
        let c10 = idx(x0 + 1, y0);
        let c01 = idx(x0, y0 + 1);
        let c11 = idx(x0 + 1, y0 + 1);

        let lerp = |a: [f32; 4], b: [f32; 4], t: f32| {
            [
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
                a[3] + (b[3] - a[3]) * t,
            ]
        };

        lerp(lerp(c00, c10, fx), lerp(c01, c11, fx), fy)
    }
}
