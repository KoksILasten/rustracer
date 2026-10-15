//! Texture types.

/// A 2D texture: RGBA, linear float, `data` in row-major order.
#[derive(Debug, Clone)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub data: Vec<[f32; 4]>, // RGBA, linear float
}

impl Texture {
    /// Create a solid-color 1x1 texture (useful as a default).
    pub fn solid(color: [f32; 4]) -> Self {
        Self {
            width: 1,
            height: 1,
            data: vec![color],
        }
    }
}
