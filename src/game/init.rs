use cgmath::InnerSpace;
use hecs::World;

use super::camera::{CameraController, CAMERA_EYE};
use super::components::{DirectionalLight, Flashlight, PointLight, Position};
use super::entities::teapot;

pub fn init(world: &mut World) {
    // Camera entity — carries the flashlight (F to toggle).
    world.spawn((
        CameraController::new(1_000.0),
        Position(cgmath::vec3(CAMERA_EYE.x, CAMERA_EYE.y, CAMERA_EYE.z)),
        PointLight {
            color:     cgmath::vec3(1.0, 0.92, 0.8), // warm white
            intensity: 200.0,
        },
        Flashlight { enabled: false },
    ));

    world.spawn(teapot(cgmath::vec3(0.0, 0.0, 0.0)));

    // Sun — comes from upper-left-front, angled ~45° downward.
    // direction points *toward* the light source so dot(normal, dir) is positive on lit faces.
    world.spawn((DirectionalLight {
        direction: cgmath::vec3(0.4, 1.0, 0.3).normalize(),
        color:     cgmath::vec3(1.0, 0.95, 0.85), // slightly warm white
        intensity: 1.5,
    },));
}
