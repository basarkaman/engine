use hecs::World;
use winit::keyboard::KeyCode;

use super::camera::CameraController;
use super::components::Flashlight;

pub fn handle_key(key: KeyCode, pressed: bool, world: &mut World) {
    // F — toggle flashlight on key-down only (not on repeat/release).
    if key == KeyCode::KeyF && pressed {
        for (_, flashlight) in world.query_mut::<&mut Flashlight>() {
            flashlight.enabled = !flashlight.enabled;
        }
    }

    for (_, controller) in world.query_mut::<&mut CameraController>() {
        controller.handle_key(key, pressed);
    }
}

pub fn handle_mouse_motion(dx: f64, dy: f64, world: &mut World) {
    for (_, controller) in world.query_mut::<&mut CameraController>() {
        controller.handle_mouse_motion(dx, dy);
    }
}
