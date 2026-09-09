//! Camera models for ray generation.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// A perspective pinhole camera.
#[derive(Debug, Clone)]
pub struct Camera {
    pub position: Vec3,
    pub look_at: Vec3,
    pub up: Vec3,
    pub fov_degrees: f32,
    pub near: f32,
    pub aperture: f32,  // depth-of-field (0 = pinhole)
    pub focus_dist: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 5.0),
            look_at: Vec3::ZERO,
            up: Vec3::Y,
            fov_degrees: 60.0,
            near: 0.1,
            aperture: 0.0,
            focus_dist: 5.0,
        }
    }
}

/// GPU-compatible camera uniforms.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub forward: [f32; 3],
    pub _pad1: f32,
    pub right: [f32; 3],
    pub _pad2: f32,
    pub up: [f32; 3],
    pub _pad3: f32,
    pub inv_half_res: [f32; 2],
    pub near: f32,
    pub fov_scale: f32,
    pub aperture: f32,
    pub focus_dist: f32,
    pub _pad_end: [f32; 2], // pad to 96 bytes (WGSL struct alignment with vec3)
}

impl Camera {
    /// Build the GPU uniform data for this camera given the output resolution.
    pub fn to_uniform(&self, width: u32, height: u32) -> CameraUniform {
        let forward = (self.look_at - self.position).normalize();
        let right = forward.cross(self.up).normalize();
        let up = right.cross(forward);
        let aspect = width as f32 / height as f32;
        let fov_scale = (self.fov_degrees.to_radians() * 0.5).tan();

        CameraUniform {
            position: self.position.to_array(),
            _pad0: 0.0,
            forward: forward.to_array(),
            _pad1: 0.0,
            right: (right * aspect * fov_scale).to_array(),
            _pad2: 0.0,
            up: (up * fov_scale).to_array(),
            _pad3: 0.0,
            inv_half_res: [2.0 / width as f32, 2.0 / height as f32],
            near: self.near,
            fov_scale,
            aperture: self.aperture,
            focus_dist: self.focus_dist,
            _pad_end: [0.0; 2],
        }
    }

    /// Orbit the camera around the look-at point.
    pub fn orbit(&mut self, delta_yaw: f32, delta_pitch: f32) {
        let forward = (self.look_at - self.position).normalize();
        let _right = forward.cross(self.up).normalize();
        let dist = (self.look_at - self.position).length();

        // Rotate around world up for yaw, local right for pitch
        let rot_yaw = glam::Quat::from_axis_angle(self.up, delta_yaw);
        let new_forward = rot_yaw * forward;
        let right = new_forward.cross(self.up).normalize();
        let rot_pitch = glam::Quat::from_axis_angle(right, delta_pitch);
        let final_forward = rot_pitch * new_forward;

        self.position = self.look_at - final_forward * dist;
    }

    /// Zoom by moving towards/away from look-at.
    pub fn zoom(&mut self, delta: f32) {
        let forward = (self.look_at - self.position).normalize();
        let dist = (self.look_at - self.position).length();
        let new_dist = (dist - delta).max(0.01);
        self.position = self.look_at - forward * new_dist;
    }

    /// Pan by moving position and look-at in screen space.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let forward = (self.look_at - self.position).normalize();
        let right = forward.cross(self.up).normalize();
        let up = right.cross(forward);
        let offset = right * dx + up * dy;
        self.position += offset;
        self.look_at += offset;
    }
}
