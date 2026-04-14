use std::f32::consts::FRAC_PI_2;

use cgmath::InnerSpace;
use winit::keyboard::KeyCode;

use crate::camera::Camera;

pub const CAMERA_EYE: cgmath::Point3<f32> = cgmath::Point3 { x: 0.0, y: 5.0, z: -10.0 };
pub const CAMERA_TARGET: cgmath::Point3<f32> = cgmath::Point3 { x: 0.0, y: 0.0, z: 0.0 };

const SAFE_PITCH: f32 = FRAC_PI_2 - 0.001;

pub struct CameraController {
    pub speed: f32,
    sensitivity: f32,
    yaw: f32,
    pitch: f32,
    mouse_dx: f32,
    mouse_dy: f32,
    is_up_pressed: bool,
    is_down_pressed: bool,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
}

impl CameraController {
    pub fn new(speed: f32) -> Self {
        // Derive initial yaw/pitch from the default eye→target direction.
        let dir = (CAMERA_TARGET - CAMERA_EYE).normalize();
        let yaw = dir.z.atan2(dir.x);
        let pitch = dir.y.asin().clamp(-SAFE_PITCH, SAFE_PITCH);
        Self {
            speed,
            sensitivity: 0.002,
            yaw,
            pitch,
            mouse_dx: 0.0,
            mouse_dy: 0.0,
            is_up_pressed: false,
            is_down_pressed: false,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        }
    }

    pub fn handle_key(&mut self, key: KeyCode, is_pressed: bool) {
        match key {
            KeyCode::Space => self.is_up_pressed = is_pressed,
            KeyCode::ShiftLeft => self.is_down_pressed = is_pressed,
            KeyCode::KeyW | KeyCode::ArrowUp => self.is_forward_pressed = is_pressed,
            KeyCode::KeyA | KeyCode::ArrowLeft => self.is_left_pressed = is_pressed,
            KeyCode::KeyS | KeyCode::ArrowDown => self.is_backward_pressed = is_pressed,
            KeyCode::KeyD | KeyCode::ArrowRight => self.is_right_pressed = is_pressed,
            _ => {}
        }
    }

    pub fn handle_mouse_motion(&mut self, dx: f64, dy: f64) {
        self.mouse_dx += dx as f32;
        self.mouse_dy += dy as f32;
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt: f32) {
        // Apply accumulated mouse delta.
        self.yaw -= self.mouse_dx * self.sensitivity;
        self.pitch -= self.mouse_dy * self.sensitivity; // dy down → look down → pitch decreases
        self.pitch = self.pitch.clamp(-SAFE_PITCH, SAFE_PITCH);
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;

        // Build basis vectors from yaw + pitch.
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();

        let forward = cgmath::Vector3::new(
            cos_pitch * sin_yaw,
            sin_pitch,
            cos_pitch * cos_yaw,
        );
        // Right vector stays flat so strafing doesn't drift vertically.
        let right = cgmath::Vector3::new(-cos_yaw, 0.0, sin_yaw);
        let up = cgmath::Vector3::unit_y();

        let v = self.speed * dt;
        if self.is_forward_pressed  { camera.eye += forward * v; }
        if self.is_backward_pressed { camera.eye -= forward * v; }
        if self.is_right_pressed    { camera.eye += right   * v; }
        if self.is_left_pressed     { camera.eye -= right   * v; }
        if self.is_up_pressed       { camera.eye += up      * v; }
        if self.is_down_pressed     { camera.eye -= up      * v; }

        camera.target = camera.eye + forward;
    }
}
