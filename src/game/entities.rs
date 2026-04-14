use cgmath::prelude::*;

use super::components::{ModelTag, Position, Rotation};

pub type Teapot = (Position, Rotation, ModelTag);

pub fn teapot(position: cgmath::Vector3<f32>) -> Teapot {
    (
        Position(position),
        Rotation(cgmath::Quaternion::one()),
        ModelTag(r"dust/Untitled.gltf"),
    )
}
