/// Which model file this entity renders as.
#[derive(Debug, Clone)]
pub struct ModelTag(pub &'static str);

#[derive(Debug, Copy, Clone)]
pub struct Position(pub cgmath::Vector3<f32>);

#[derive(Debug, Copy, Clone)]
pub struct Rotation(pub cgmath::Quaternion<f32>);

#[derive(Debug, Copy, Clone)]
pub struct PointLight {
    pub color:     cgmath::Vector3<f32>,
    pub intensity: f32,
}

/// Point light attached to the camera, toggled with F.
#[derive(Debug, Copy, Clone)]
pub struct Flashlight {
    pub enabled: bool,
}

/// Sun-like infinite light — direction only, no attenuation.
/// `direction` points *toward* the light source so dot(normal, dir) > 0 on lit faces.
#[derive(Debug, Copy, Clone)]
pub struct DirectionalLight {
    pub direction: cgmath::Vector3<f32>,
    pub color:     cgmath::Vector3<f32>,
    pub intensity: f32,
}
