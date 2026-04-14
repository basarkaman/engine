use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::PhysicalKey,
    window::Window,
};

use crate::game::{CAMERA_EYE, CAMERA_TARGET};
use crate::renderer::State;

pub struct Config {
    pub backends: wgpu::Backends,
    pub power_preference: wgpu::PowerPreference,
    pub force_software: bool,
}

impl Config {
    fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();

        let backends = if args.iter().any(|a| a == "-vulkan") {
            wgpu::Backends::VULKAN
        } else if args.iter().any(|a| a == "-dx12") {
            wgpu::Backends::DX12
        } else if args.iter().any(|a| a == "-opengl") {
            wgpu::Backends::GL
        } else if args.iter().any(|a| a == "-metal") {
            wgpu::Backends::METAL
        } else {
            wgpu::Backends::PRIMARY
        };

        let force_software = args.iter().any(|a| a == "-software");

        let power_preference = if force_software {
            wgpu::PowerPreference::None
        } else if args.iter().any(|a| a == "-discrete") {
            wgpu::PowerPreference::HighPerformance
        } else if args.iter().any(|a| a == "-integrated") {
            wgpu::PowerPreference::LowPower
        } else {
            wgpu::PowerPreference::default()
        };

        Self { backends, power_preference, force_software }
    }
}

pub struct App {
    state: Option<State>,
    config: Config,
}

impl App {
    pub fn new(config: Config) -> Self {
        Self { state: None, config }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(event_loop.create_window(Window::default_attributes()).unwrap());
        // Lock cursor for mouse-look.
        let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(winit::window::CursorGrabMode::Confined));
        window.set_cursor_visible(false);
        self.state = Some(
            pollster::block_on(State::new(window, CAMERA_EYE, CAMERA_TARGET, &self.config))
                .unwrap(),
        );
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let winit::event::DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if let Some(state) = &mut self.state {
                if !state.ui_mode() {
                    state.handle_mouse_motion(dx, dy);
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window().request_redraw();
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: ()) {}

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        // Always forward to egui first.
        let egui_consumed = state.egui_on_window_event(&event);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    Err(e) => {
                        log::error!("{e}");
                        event_loop.exit();
                    }
                }
            }
            // Let egui consume mouse clicks when UI mode is active.
            WindowEvent::MouseInput { .. } if egui_consumed => {}
            WindowEvent::MouseInput { .. } => {}
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => {
                // Escape and Tab always reach the game handler (for exit / UI toggle).
                use winit::keyboard::KeyCode;
                let always_pass = matches!(code, KeyCode::Escape | KeyCode::Tab);
                if always_pass || !egui_consumed {
                    state.handle_key(event_loop, code, key_state.is_pressed());
                }
            }
            _ => {}
        }
    }
}

pub fn run() -> anyhow::Result<()> {
    env_logger::init();

    let config = Config::from_args();
    let event_loop = EventLoop::new()?;
    // Poll: event loop never sleeps, renders continuously.
    // Default (Wait) sleeps until an event arrives — causes stuttering.
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::new(config);
    event_loop.run_app(&mut app)?;

    Ok(())
}
