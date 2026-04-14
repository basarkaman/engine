# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
cargo build          # Build (also copies res/ via build.rs)
cargo run            # Run the engine
cargo run -- -vulkan # Force Vulkan backend
cargo run -- -dx12   # Force DX12 backend
cargo run -- -discrete   # Prefer high-performance GPU
cargo run -- -integrated # Prefer integrated GPU
```

No tests or lints are configured. `build.rs` copies `res/` into `OUT_DIR/res/` at compile time — all asset paths go through `resources::res_path()` which resolves against `OUT_DIR`, not the working directory.

## Architecture

Rust/wgpu forward-rendering engine with a **hecs** ECS game layer, glTF model loading, normal mapping, and Blinn-Phong lighting. Detailed reference: `ENGINE.md`.

### Module map

```
src/
├── app.rs       — winit ApplicationHandler, CLI → Config, run() entry point
├── camera.rs    — Camera, CameraUniform, OPENGL_TO_WGPU_MATRIX
├── frustum.rs   — Frustum (6 planes), per-mesh sphere cull test
├── instance.rs  — InstanceRaw (model matrix + normal matrix), build_instance_data_for()
├── model.rs     — ModelVertex, Mesh, Material, Model, DrawModel trait
├── texture.rs   — Texture wrapper; from_bytes (sRGB) vs from_image_linear (normal maps)
├── resources.rs — res_path(), load_model() → load_gltf / load_obj
├── renderer.rs  — State struct, update(), render()
└── game/
    ├── components.rs — ModelTag, Position, Rotation, PointLight, Flashlight, DirectionalLight
    ├── init.rs       — world setup: camera entity + flashlight, Sponza, sun
    ├── entities.rs   — teapot() helper (spawns Sponza.gltf)
    ├── camera.rs     — CameraController (FPS: WASD + mouse yaw/pitch)
    ├── input.rs      — keyboard/mouse dispatch; F key toggles Flashlight
    └── systems.rs    — update(): run camera controller, sync Position ← camera.eye
```

### Data flow

1. `app.rs` creates the window and calls `pollster::block_on(State::new(...))`.
2. `State` owns the ECS world, all GPU resources, loaded models, and the camera.
3. Each frame: `update()` → camera + lights → instance buffers → `render()`.
4. `render()` does frustum culling per mesh, then a single forward pass.

### Bind groups (fixed layout, all draw calls)

| Group | Contents |
|-------|----------|
| 0 | diffuse texture + sampler, normal texture + sampler (per material) |
| 1 | `CameraUniform` { view_position, view_proj } |
| 2 | point light { position, intensity, color } — flashlight |
| 3 | directional light { direction, intensity, color } — sun |

### Lighting

- **Point light**: Blinn-Phong, attenuation = `intensity / (dist² + intensity)`. Lives on the camera entity; F key toggles `Flashlight.enabled` (renderer zeroes intensity when disabled).
- **Directional light**: Blinn-Phong, no attenuation. `direction` points **toward** the light source.
- **Normal mapping**: diffuse → `Rgba8UnormSrgb`, normal maps → `Rgba8Unorm` (must not be sRGB-decoded). Flat fallback = 1×1 pixel `(128,128,255)`.

### ECS conventions

- **hecs** is a plain data store — no system scheduler. Queries run manually from `update()`, `render()`, and input handlers.
- Entity archetypes: *(CameraController, Position, PointLight, Flashlight)*, *(ModelTag, Position, Rotation)*, *(DirectionalLight,)*.
- `ModelTag` string must match the key in `State.models` HashMap (e.g. `r"sponza/Sponza.gltf"`).
- Instance buffers are fixed-size (allocated at startup); adding entities at runtime requires recreating them.

### glTF loading rules

- glTF is Y-up right-handed — **no coordinate transform** is applied to positions/normals/tangents.
- Tangent `w` component = bitangent sign; shader reconstructs bitangent as `w * cross(N, T)`.
- Materials without a normal texture get `flat_normal_texture()` (no perturbation).
- OBJ meshes are merged by material to reduce draw calls; default tangent = `[1,0,0,1]`.

### Surface reconfiguration

On `Suboptimal` from `get_current_texture()`: **drop** the surface texture first, then reconfigure, then return early. Skipping the drop causes a wgpu validation error.
