use hecs::World;

use crate::camera::Camera;

use super::camera::CameraController;
use super::components::Position;

pub fn update(world: &mut World, camera: &mut Camera, dt: f32) {
    // Step 1: update camera from controller input.
    for (_, controller) in world.query_mut::<&mut CameraController>() {
        controller.update_camera(camera, dt);
    }

    // Step 2: sync camera world position back into the Position component
    // so that lights (and anything else) attached to the camera entity stay in sync.
    for (_, (_, pos)) in world.query_mut::<(&CameraController, &mut Position)>() {
        pos.0 = cgmath::vec3(camera.eye.x, camera.eye.y, camera.eye.z);
    }
}
