# Engine Specification

Rust/wgpu forward renderer with hecs ECS, glTF/OBJ model loading, normal mapping, Blinn-Phong lighting, shadow mapping, and an egui debug overlay.

---

## Table of Contents

1. [Project Layout](#project-layout)
2. [Dependencies](#dependencies)
3. [Build System](#build-system)
4. [Entry Point & Event Loop](#entry-point--event-loop)
5. [GPU State](#gpu-state)
6. [Bind Group Layouts](#bind-group-layouts)
7. [Render Pipelines](#render-pipelines)
8. [Shadow Mapping](#shadow-mapping)
9. [Shaders](#shaders)
10. [Camera](#camera)
11. [Frustum Culling](#frustum-culling)
12. [Instance System](#instance-system)
13. [Model & Material](#model--material)
14. [Texture Handling](#texture-handling)
15. [Resource Loading](#resource-loading)
16. [ECS Game Layer](#ecs-game-layer)
17. [egui Debug Overlay](#egui-debug-overlay)
18. [Frame Loop](#frame-loop)
19. [Key Design Decisions](#key-design-decisions)

---

## Project Layout

```
engine/
├── Cargo.toml
├── build.rs                  # copies res/ to OUT_DIR at compile time
├── res/
│   └── sponza/
│       ├── Sponza.gltf
│       ├── Sponza.bin
│       └── *.jpg / *.png     # textures
└── src/
    ├── main.rs               # calls engine::run()
    ├── lib.rs                # re-exports run, declares modules
    ├── app.rs                # winit ApplicationHandler, CLI config, egui dispatch
    ├── camera.rs             # Camera, CameraUniform, OPENGL_TO_WGPU_MATRIX
    ├── frustum.rs            # Frustum (6 planes), sphere cull test
    ├── instance.rs           # InstanceRaw, build_instance_data_for()
    ├── model.rs              # ModelVertex, Mesh, Material, Model, DrawModel trait
    ├── texture.rs            # Texture wrapper (sRGB + linear variants)
    ├── resources.rs          # res_path, load_model, glTF/OBJ loaders
    ├── renderer.rs           # State struct, update(), render()
    ├── shader.wgsl           # main forward pass vertex + fragment shader
    ├── shadow.wgsl           # shadow pass vertex shader (depth-only)
    ├── depth_debug.wgsl      # fullscreen depth buffer visualization
    └── game/
        ├── mod.rs
        ├── camera.rs         # CameraController (FPS), CAMERA_EYE/TARGET
        ├── components.rs     # ECS component types
        ├── entities.rs       # teapot() spawn helper (loads Sponza.gltf)
        ├── init.rs           # world setup: camera entity, model, sun
        ├── input.rs          # keyboard/mouse dispatch
        └── systems.rs        # update(): move camera, sync Position component
```

---

## Dependencies

```toml
[dependencies]
wgpu        = "29.0"
winit       = "0.30"
cgmath      = "0.18"
hecs        = "0.10"
gltf        = "1.4"
tobj        = "3.2"
image       = "0.24"
bytemuck    = { version = "1.24", features = ["derive"] }
anyhow      = "1"
log         = "0.4"
env_logger  = "0.10"
pollster    = "0.3"
egui        = "0.34"
egui-wgpu   = "0.34"
egui-winit  = "0.34"

[build-dependencies]
anyhow      = "1"
fs_extra    = "1.2"
glob        = "0.3"
```

---

## Build System

`build.rs` runs at compile time:

1. Reads `OUT_DIR` from the environment.
2. Copies the entire `res/` directory tree into `{OUT_DIR}/res/` using `fs_extra::dir::copy`.
3. All runtime asset paths are resolved relative to `{OUT_DIR}/res/` via `res_path()` in `resources.rs`.

Model and texture loading does **not** depend on the working directory at runtime.

---

## Entry Point & Event Loop

### `src/main.rs`

```rust
fn main() { engine::run(); }
```

### `src/lib.rs`

Declares all modules and re-exports `run` from `app`.

### `src/app.rs`

**`Config` struct** — parsed from CLI args:

```rust
pub struct Config {
    pub backends:          wgpu::Backends,       // default: PRIMARY; flags: -vulkan, -dx12, -opengl, -metal
    pub power_preference:  wgpu::PowerPreference, // default: default; -discrete / -integrated
    pub force_software:    bool,                 // -software flag
}
```

CLI flags:

| Flag | Effect |
|------|--------|
| `-vulkan` | `Backends::VULKAN` |
| `-dx12` | `Backends::DX12` |
| `-opengl` | `Backends::GL` |
| `-metal` | `Backends::METAL` |
| `-software` | Force CPU rasterizer (`force_fallback_adapter = true`) |
| `-discrete` | `PowerPreference::HighPerformance` |
| `-integrated` | `PowerPreference::LowPower` |

**`App` struct** — implements `winit::application::ApplicationHandler`:

- Owns `Option<State>` (created lazily on `resumed()`).
- `resumed()`: creates the window, locks the cursor, then calls `pollster::block_on(State::new(window, CAMERA_EYE, CAMERA_TARGET, &config))`.
- `device_event()`: routes `MouseMotion` to `State::handle_mouse_motion` unless UI mode is active.
- `window_event()`:
  - All events go through `State::egui_on_window_event()` first. If egui consumes the event, the game does not see it (except Escape and Tab, which always pass through).
  - `KeyboardInput` → `State::handle_key()` (handles Escape=quit, Tab=toggle UI mode, then forwards to `game::handle_key`).
  - `MouseInput` → consumed by egui when in UI mode.
  - `Resized` → `State::resize()`.
  - `RedrawRequested` → `State::update()` then `State::render()`.
  - `CloseRequested` → exit event loop.
- `about_to_wait()`: requests a redraw every frame.

**`run()`** — public entry point:

1. Parses `std::env::args()` into `Config`.
2. Creates `EventLoop`.
3. Sets control flow to `Poll` (never sleeps — continuous rendering without stuttering).
4. Runs `App { state: None, config }`.

---

## GPU State

### `State` struct fields

| Field | Type | Purpose |
|---|---|---|
| `surface` | `wgpu::Surface<'static>` | Swap-chain surface |
| `device` | `wgpu::Device` | Logical GPU device |
| `queue` | `wgpu::Queue` | Command submission |
| `config` | `wgpu::SurfaceConfiguration` | Surface format, size, present mode |
| `is_surface_configured` | `bool` | Guard: don't render before first resize |
| `render_pipeline` | `wgpu::RenderPipeline` | Forward rendering pipeline (filled triangles) |
| `wireframe_pipeline` | `wgpu::RenderPipeline` | Same shader, `PolygonMode::Line` |
| `wireframe` | `bool` | Toggle between filled and wireframe rendering |
| `models` | `HashMap<String, ModelEntry>` | Loaded models keyed by tag string |
| `camera` | `Camera` | View/projection parameters |
| `camera_uniform` | `CameraUniform` | GPU-side camera data |
| `camera_buffer` | `wgpu::Buffer` | Uniform buffer for camera (group 1) |
| `camera_bind_group` | `wgpu::BindGroup` | Group 1 |
| `world` | `hecs::World` | ECS world |
| `depth_texture` | `texture::Texture` | Main depth buffer, recreated on resize |
| `window` | `Arc<Window>` | Shared window handle |
| `light_buffer` | `wgpu::Buffer` | Uniform buffer for point light (group 2) |
| `light_bind_group` | `wgpu::BindGroup` | Group 2 |
| `dir_light_buffer` | `wgpu::Buffer` | Uniform buffer for directional light (group 3) |
| `dir_light_bind_group` | `wgpu::BindGroup` | Group 3 |
| `light_space_buffer` | `wgpu::Buffer` | Light-space matrix uniform (shared by both shadow passes) |
| `shadow_render_view` | `wgpu::TextureView` | Shadow map depth attachment |
| `shadow_pipeline` | `wgpu::RenderPipeline` | Depth-only shadow rendering pipeline |
| `shadow_pass_bg` | `wgpu::BindGroup` | Shadow pass group 0 (light-space matrix only) |
| `shadow_main_bg` | `wgpu::BindGroup` | Main pass group 4 (light-space matrix + shadow map + comparison sampler) |
| `fps_frame_count` | `u32` | FPS counter accumulator |
| `fps_timer` | `Instant` | FPS counter window start |
| `last_frame` | `Instant` | For `dt` calculation |
| `last_fps` | `f32` | Most recent FPS measurement |
| `frustum` | `Frustum` | Current frame's view frustum |
| `target_frame_time` | `Duration` | 1 / monitor Hz |
| `frame_deadline` | `Instant` | When the current frame must end |
| `timestamp_query_set` | `Option<wgpu::QuerySet>` | 4-slot timestamp query set; `None` if GPU doesn't support `TIMESTAMP_QUERY` |
| `timestamp_resolve_buf` | `Option<wgpu::Buffer>` | Resolve target for timestamp queries (`QUERY_RESOLVE \| COPY_SRC`, 32 bytes) |
| `timestamp_read_buf` | `Option<wgpu::Buffer>` | CPU-readable copy of resolved timestamps (`COPY_DST \| MAP_READ`, 32 bytes) |
| `timestamp_period` | `f32` | Nanoseconds per GPU timestamp tick, from `queue.get_timestamp_period()` |
| `last_gpu_ms` | `f32` | GPU time of the most recent shadow + main passes, in milliseconds |
| `egui_ctx` | `egui::Context` | egui context |
| `egui_winit` | `egui_winit::State` | egui-winit bridge (input, platform) |
| `egui_renderer` | `egui_wgpu::Renderer` | egui wgpu backend |
| `ui_mode` | `bool` | When true: cursor free, egui interactive; Tab toggles |
| `lighting_enabled` | `bool` | When false: shader outputs raw diffuse (fullbright) |
| `show_depth` | `bool` | When true: renders depth buffer as grayscale overlay |
| `depth_debug_pipeline` | `wgpu::RenderPipeline` | Fullscreen triangle pipeline for depth visualization |
| `depth_debug_bind_group_layout` | `wgpu::BindGroupLayout` | Layout for depth debug bind group |
| `depth_debug_bind_group` | `wgpu::BindGroup` | Binds the main depth texture for visualization |

### `ModelEntry`

```rust
struct ModelEntry {
    model:          model::Model,
    instance_buffer: wgpu::Buffer,
    instance_count: u32,
}
```

### `State::new()` (async)

1. Create `wgpu::Instance` with configured backends.
2. Create surface from window.
3. Enumerate and print all available adapters.
4. Request adapter (power preference from Config, `force_fallback_adapter` if `-software`).
5. Print selected adapter info and warn if requested type was unavailable.
6. Request device — requires `Features::POLYGON_MODE_LINE` for wireframe, `limits.max_bind_groups = 5` (groups 0–4). `Features::TIMESTAMP_QUERY` is requested if the adapter supports it (checked via `adapter.features()` before device creation); if not supported, GPU frametime is unavailable.
7. Configure surface: prefer sRGB format, `Immediate` present mode (fallback to `Fifo`), `desired_maximum_frame_latency = 2`.
8. Create all bind group layouts (groups 0–4 + shadow pass group 0 + depth debug group).
9. Create render pipeline (filled) and wireframe pipeline (`PolygonMode::Line`).
10. Create shadow map texture (2048×2048 `Depth32Float`), render view, sample view, comparison sampler.
11. Compute initial light-space matrix from ECS `DirectionalLight`.
12. Create shadow pipeline (depth-only, no fragment shader).
13. Create depth debug pipeline (fullscreen triangle, reads depth texture).
14. Load all models referenced by ECS entities.
15. Create camera, lights, depth texture.
16. Create per-model instance buffers.
17. Initialize egui context, winit bridge, and wgpu renderer.
18. Derive `target_frame_time` from monitor refresh rate (fallback 60 Hz).
19. Create GPU timestamp query resources if `TIMESTAMP_QUERY` is supported: a 4-slot `QuerySet`, a resolve buffer, and a CPU-readable map buffer (each 32 bytes).

---

## Bind Group Layouts

Five bind group layouts for the main render pipeline, one for the shadow pass, and one for the depth debug overlay.

### Main Pass — Group 0: Material textures

```
binding 0: texture_2d<f32>  (TEXTURE | COPY_DST)  — diffuse/albedo
binding 1: sampler (filtering)                       — diffuse sampler
binding 2: texture_2d<f32>  (TEXTURE | COPY_DST)  — normal map
binding 3: sampler (filtering)                       — normal sampler
```
Visibility: `FRAGMENT`

### Main Pass — Group 1: Camera

```
binding 0: uniform buffer — CameraUniform { view_position: [f32;4], view_proj: [[f32;4];4] }
```
Visibility: `VERTEX | FRAGMENT`

### Main Pass — Group 2: Point light (flashlight)

```
binding 0: uniform buffer — LightUniform { position: [f32;3], intensity: f32, color: [f32;3], lighting_on: f32 }
```
Visibility: `VERTEX | FRAGMENT`

`lighting_on` is `1.0` for normal lighting and `0.0` for fullbright (raw diffuse output).

### Main Pass — Group 3: Directional light (sun)

```
binding 0: uniform buffer — DirLightUniform { direction: [f32;3], intensity: f32, color: [f32;3], _pad: f32 }
```
Visibility: `FRAGMENT`

### Main Pass — Group 4: Shadow map

```
binding 0: uniform buffer — LightSpaceUniform { matrix: [[f32;4];4] }
binding 1: texture_depth_2d                          — shadow map
binding 2: sampler_comparison (LessEqual)             — shadow comparison sampler
```
Visibility: `FRAGMENT`

### Shadow Pass — Group 0: Light-space matrix

```
binding 0: uniform buffer — LightSpaceUniform { matrix: [[f32;4];4] }
```
Visibility: `VERTEX`

### Depth Debug — Group 0: Depth texture

```
binding 0: texture_depth_2d — main depth buffer
```
Visibility: `FRAGMENT`

---

## Render Pipelines

Three pipelines share the same vertex format but differ in configuration.

### Forward pipeline (filled)

- **Vertex shader**: `vs_main` in `shader.wgsl`
- **Fragment shader**: `fs_main` in `shader.wgsl`
- **Primitive**: TriangleList, CCW front face, back-face culling, `PolygonMode::Fill`
- **Depth stencil**: Depth32Float, `Less` compare, write enabled
- **Color target**: surface format, no blending (opaque replacement)
- **Bind group layouts**: groups 0–4 (material, camera, point light, dir light, shadow)
- **Vertex buffers**: two — `ModelVertex` (step `Vertex`, stride 48) + `InstanceRaw` (step `Instance`, stride 100)

### Wireframe pipeline

Same as the forward pipeline except `PolygonMode::Line`.

### Shadow pipeline

- **Vertex shader**: `vs_shadow` in `shadow.wgsl`
- **Fragment shader**: none (depth-only pass)
- **Primitive**: TriangleList, CCW front face, back-face culling
- **Depth stencil**: Depth32Float, `Less` compare, write enabled
- **Bind group layouts**: group 0 only (light-space matrix)
- **Vertex buffers**: same two buffers as the forward pipeline

### Depth debug pipeline

- **Vertex shader**: `vs_main` in `depth_debug.wgsl` (fullscreen triangle from `vertex_index`)
- **Fragment shader**: `fs_main` in `depth_debug.wgsl` (reads depth buffer, outputs log-scaled grayscale)
- **Primitive**: TriangleList (3 vertices, no vertex buffer)
- **No depth stencil**
- **Bind group layouts**: group 0 (depth texture)
- **Blending**: none (opaque replacement)

### Vertex buffer attribute layout

**Buffer 0 — `ModelVertex`** (stride = 48 bytes, step = `Vertex`):

| Location | Format | Offset | Field |
|----------|--------|--------|-------|
| 0 | float32x3 | 0 | position |
| 1 | float32x2 | 12 | tex_coords |
| 2 | float32x3 | 20 | normal |
| 3 | float32x4 | 32 | tangent (xyz + bitangent sign w) |

**Buffer 1 — `InstanceRaw`** (stride = 100 bytes, step = `Instance`):

| Location | Format | Offset | Field |
|----------|--------|--------|-------|
| 5 | float32x4 | 0 | model matrix col 0 |
| 6 | float32x4 | 16 | model matrix col 1 |
| 7 | float32x4 | 32 | model matrix col 2 |
| 8 | float32x4 | 48 | model matrix col 3 |
| 9 | float32x3 | 64 | normal matrix row 0 |
| 10 | float32x3 | 76 | normal matrix row 1 |
| 11 | float32x3 | 88 | normal matrix row 2 |

---

## Shadow Mapping

The engine uses a single directional light shadow map. The sun's orthographic projection covers a configurable volume centered on the world origin.

### Constants

```rust
const SHADOW_MAP_SIZE:  u32  = 2048;     // shadow map resolution (square)
const SHADOW_ORTHO_SIZE: f32 = 2000.0;    // half-extent of the ortho frustum (world units)
const SHADOW_LIGHT_DIST: f32 = 5000.0;   // how far back the shadow camera sits
const SHADOW_ZNEAR: f32     = 3000.0;    // near clip of the shadow camera
const SHADOW_ZFAR: f32      = 8000.0;    // far clip of the shadow camera
```

### Light-space matrix computation

```rust
fn compute_light_space_matrix(direction: Vector3<f32>) -> Matrix4<f32>
```

1. Normalize `direction` (points toward the light source).
2. Place the shadow camera at `direction * SHADOW_LIGHT_DIST`.
3. Look at the origin with `look_at_rh`.
4. If the light is nearly vertical (Y component > 0.99), use Z-up to avoid a degenerate up vector; otherwise use Y-up.
5. Apply orthographic projection: `ortho(-ORTHO_SIZE, ORTHO_SIZE, -ORTHO_SIZE, ORTHO_SIZE, ZNEAR, ZFAR)`.
6. Multiply by `OPENGL_TO_WGPU_MATRIX` to remap Z from [-1,1] to [0,1].

### Shadow map texture

- Format: `Depth32Float`
- Size: `SHADOW_MAP_SIZE × SHADOW_MAP_SIZE`
- Usage: `RENDER_ATTACHMENT | TEXTURE_BINDING`
- Two views:
  - Default view: used as the depth attachment in the shadow pass.
  - `DepthOnly` aspect view: used for shadow sampling in the fragment shader.

### Shadow comparison sampler

```rust
address_mode:  ClampToEdge (all axes)
mag_filter:    Linear
min_filter:    Linear
compare:       LessEqual
```

### Render flow

Each frame consists of up to four render passes submitted in a single command buffer:

1. **Shadow pass** — renders scene from light's perspective into the shadow map (depth only, no color attachment).
2. **Main pass** — renders scene with Blinn-Phong lighting, shadow factor applied to the sun contribution.
3. **Depth debug pass** (optional) — overlays the main depth buffer as log-scaled grayscale using a fullscreen triangle.
4. **egui pass** — renders the debug UI window on top.

---

## Shaders

### `shader.wgsl` — Main forward pass

#### Uniforms

```wgsl
struct Camera {
    view_pos:  vec4<f32>,
    view_proj: mat4x4<f32>,
}
@group(1) @binding(0) var<uniform> camera: Camera;

struct Light {
    position:   vec3<f32>,
    intensity:  f32,
    color:      vec3<f32>,
    lighting_on: f32,     // 1.0 = normal lighting, 0.0 = fullbright
}
@group(2) @binding(0) var<uniform> light: Light;

struct DirLight {
    direction: vec3<f32>,
    intensity: f32,
    color:     vec3<f32>,
    _pad:      f32,
}
@group(3) @binding(0) var<uniform> dir_light: DirLight;

struct LightSpace {
    matrix: mat4x4<f32>,
}
@group(4) @binding(0) var<uniform> light_space: LightSpace;
@group(4) @binding(1) var t_shadow: texture_depth_2d;
@group(4) @binding(2) var s_shadow: sampler_comparison;
```

#### Textures (group 0)

```wgsl
@group(0) @binding(0) var t_diffuse: texture_2d<f32>;
@group(0) @binding(1) var s_diffuse: sampler;
@group(0) @binding(2) var t_normal:  texture_2d<f32>;
@group(0) @binding(3) var s_normal:  sampler;
```

#### Vertex stage

Inputs: `VertexInput` (locations 0–3) + `InstanceInput` (locations 5–11)

```wgsl
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords:     vec2<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) world_position: vec3<f32>,
    @location(3) world_tangent:  vec4<f32>,   // xyz=tangent, w=bitangent sign
}
```

Operations:
1. Reconstruct `model_matrix` (mat4) and `normal_matrix` (mat3) from instance inputs.
2. `world_pos = model_matrix * vec4(position, 1.0)`
3. `world_normal = normalize(normal_matrix * normal)`
4. `world_tangent.xyz = normalize(normal_matrix * tangent.xyz)`, preserve `w`
5. `clip_position = camera.view_proj * world_pos`

#### Fragment stage

1. **Diffuse**: `textureSample(t_diffuse, s_diffuse, tex_coords)`

2. **Normal mapping**:
   ```wgsl
   let n_sample  = textureSample(t_normal, s_normal, tex_coords).xyz;
   let n_tangent = n_sample * 2.0 - 1.0;
   let N = normalize(world_normal);
   let T = normalize(world_tangent.xyz);
   let B = world_tangent.w * cross(N, T);
   let normal = normalize(mat3x3(T, B, N) * n_tangent);
   ```

3. **Point light** (Blinn-Phong):
   ```wgsl
   let light_dir    = normalize(light.position - world_position);
   let half_dir     = normalize(view_dir + light_dir);
   let ambient      = light.color * 0.05;
   let diffuse      = light.color * max(dot(normal, light_dir), 0.0);
   let specular     = light.color * pow(max(dot(normal, half_dir), 0.0), 32.0);
   let dist         = length(light.position - world_position);
   let attenuation  = light.intensity / (dist * dist + light.intensity);
   let point_contrib = ambient + (diffuse + specular) * attenuation;
   ```

4. **Directional light** (Blinn-Phong, no attenuation):
   ```wgsl
   let sun_dir      = normalize(dir_light.direction);
   let sun_diffuse  = dir_light.color * max(dot(normal, sun_dir), 0.0);
   let sun_half     = normalize(view_dir + sun_dir);
   let sun_specular = dir_light.color * pow(max(dot(normal, sun_half), 0.0), 32.0);
   let sun_contrib  = (sun_diffuse + sun_specular) * dir_light.intensity;
   ```

5. **Fullbright check**: if `light.lighting_on < 0.5`, return `object_color` immediately (no lighting).

6. **Shadow factor** (5×5 PCF):
   ```wgsl
   fn shadow_factor(world_pos: vec3<f32>) -> f32 {
       let ls   = light_space.matrix * vec4<f32>(world_pos, 1.0);
       let proj = ls.xyz / ls.w;
       let uv   = proj.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
       if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z > 1.0 {
           return 1.0;   // outside light frustum — consider lit
       }
       let bias  = proj.z - 0.005;
       let texel = 1.0 / 2048.0;
       var sum   = 0.0;
       for (var x = -2; x <= 2; x++) {
           for (var y = -2; y <= 2; y++) {
               let offset = vec2<f32>(f32(x), f32(y)) * texel;
               sum += textureSampleCompare(t_shadow, s_shadow, uv + offset, bias);
           }
       }
       return sum / 25.0;
   }
   ```
   - NDC to UV: flip Y because NDC +Y is up but UV +Y is down.
   - Bias of 0.005 to reduce shadow acne.
   - Fragments outside the light frustum are considered lit (factor = 1.0).
   - 5×5 kernel (25 samples), averaged — produces soft shadow edges.
   - `textureSampleCompare` returns 1.0 when lit and 0.0 when in shadow.

7. **Output**: `vec4((point_contrib + sun_contrib * shadow) * object_color.xyz, object_color.a)`
   - Point light is **not** attenuated by the shadow map.
   - Sun contribution is multiplied by the shadow factor.

### `shadow.wgsl` — Shadow pass (depth only)

```wgsl
struct LightSpace {
    matrix: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> light_space: LightSpace;

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
}

@vertex
fn vs_shadow(model: VertexInput, instance: InstanceInput) -> @builtin(position) vec4<f32> {
    let m = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    return light_space.matrix * m * vec4<f32>(model.position, 1.0);
}
```

No fragment shader — the shadow pass writes depth only.

### `depth_debug.wgsl` — Depth buffer visualization

Renders a fullscreen triangle (no vertex buffer, uses `vertex_index`):

```wgsl
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pos[idx], 0.0, 1.0);
}

@group(0) @binding(0) var t_depth: texture_depth_2d;

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    // Read depth, reconstruct linear eye-space Z, apply log scale.
    // Near objects → white, far objects / empty → black.
}
```

Linear Z reconstruction:

```
z_eye = znear * zfar / (zfar - depth * (zfar - znear))
log_d = log2(z_eye / znear) / log2(zfar / znear)
output = 1.0 - clamp(log_d, 0.0, 1.0)
```

---

## Camera

### `src/camera.rs`

```rust
pub struct Camera {
    pub eye:    cgmath::Point3<f32>,
    pub target: cgmath::Point3<f32>,
    pub up:     cgmath::Vector3<f32>,
    pub aspect: f32,
    pub fovy:   f32,
    pub znear:  f32,    // 0.1
    pub zfar:   f32,    // 1,000,000
}
```

**`build_view_projection_matrix()`**:

```rust
let view = Matrix4::look_at_rh(self.eye, self.target, self.up);
let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
proj * view
```

Note: this returns the raw view-projection without the `OPENGL_TO_WGPU_MATRIX` correction. That matrix is applied separately in `update()` when writing the camera uniform.

**`OPENGL_TO_WGPU_MATRIX`**: Converts from OpenGL clip space (Z in [-1,1]) to wgpu/Vulkan clip space (Z in [0,1]):

```rust
#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::from_cols(
    Vector4::new(1.0, 0.0, 0.0, 0.0),
    Vector4::new(0.0, 1.0, 0.0, 0.0),
    Vector4::new(0.0, 0.0, 0.5, 0.0),
    Vector4::new(0.0, 0.0, 0.5, 1.0),
);
```

Note: stored column-major via `from_cols`, which is the correct layout for `cgmath::Matrix4`.

**`CameraUniform`** (bytemuck Pod/Zeroable):

```rust
pub struct CameraUniform {
    pub view_position: [f32; 4],      // eye position, w=1
    pub view_proj:     [[f32; 4]; 4],
}
```

### `src/game/camera.rs` — FPS Controller

```rust
pub const CAMERA_EYE:    Point3<f32> = Point3 { x: 0.0,  y: 5.0,  z: -10.0 };
pub const CAMERA_TARGET: Point3<f32> = Point3 { x: 0.0,  y: 0.0,  z: 0.0 };
```

**`CameraController`**:

```rust
pub struct CameraController {
    pub speed:   f32,     // default: 1000.0; mutable from egui debug panel
    sensitivity: f32,     // default: 0.002
    yaw:         f32,     // radians, horizontal rotation
    pitch:       f32,     // radians, vertical rotation, clamped ±(π/2 - 0.001)
    // key/mouse state booleans...
}
```

Initial `yaw` and `pitch` are derived from `CAMERA_EYE → CAMERA_TARGET` direction so the camera starts looking at the origin.

**Direction calculation** (spherical, Y-up):

```rust
let forward = vec3(
    cos_pitch * sin_yaw,
    sin_pitch,
    cos_pitch * cos_yaw,
);
let right = vec3(-cos_yaw, 0.0, sin_yaw);  // no vertical drift
let up = vec3(0.0, 1.0, 0.0);
```

Keys: W/Up=forward, S/Down=back, A/Left=strafe left, D/Right=strafe right, Space=up, ShiftLeft=down.
Mouse: drag → yaw/pitch delta × sensitivity.

---

## Frustum Culling

### `src/frustum.rs`

```rust
pub struct Frustum {
    planes: [cgmath::Vector4<f32>; 6],
}
```

**`from_vp(vp: &Matrix4<f32>) -> Frustum`** — extracts planes from the view-projection matrix using the Gribb/Hartmann row-combination method, adapted for Z in [0,1]:

```
planes[0] = row3 + row0   (left)
planes[1] = row3 - row0   (right)
planes[2] = row3 + row1   (bottom)
planes[3] = row3 - row1   (top)
planes[4] = row2         (near, Z >= 0)
planes[5] = row3 - row2   (far, Z <= 1)
```

Each row is extracted as `Vector4::new(vp[col0][row], vp[col1][row], vp[col2][row], vp[col3][row])` (column-major `cgmath` convention). Planes are normalized by their xyz magnitude.

**`cull_sphere(center: Point3<f32>, radius: f32) -> bool`** — returns `true` if sphere is **entirely outside** the frustum (safe to skip draw call):

```rust
for p in &self.planes {
    let signed_dist = p.x * center.x + p.y * center.y + p.z * center.z + p.w;
    if signed_dist < -radius { return true; }
}
false
```

**`Default`** — returns an identity frustum where nothing is culled (all planes oriented inward).

Used in `render()`: each mesh has a `bounding_sphere: (Point3<f32>, f32)`, skip draw if culled. Draw and cull counts are logged at debug level each frame.

---

## Instance System

### `src/instance.rs`

```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    pub model:  [[f32; 4]; 4],   // 64 bytes — model matrix
    pub normal: [[f32; 3]; 3],   // 36 bytes — normal matrix rows
}
// stride = 64 + 36 = 100 bytes (with trailing padding to 16-byte alignment implied by GPU)
```

The normal matrix is stored as three `[f32; 3]` rows at locations 9, 10, 11. The GPU reads them as three `vec3` vertex attributes, then the shader reconstructs `mat3x3(row0, row1, row2)`.

**`build_instance_data_for(world: &World, tag: &str) -> Vec<InstanceRaw>`**:

1. Query ECS for all entities with `(&Position, &Rotation, &ModelTag)`.
2. Filter where `ModelTag.0 == tag`.
3. For each match: `model_matrix = translation(position) * from(rotation)`.
4. `normal_matrix = Matrix3::from(rotation)` — uses the rotation quaternion directly, which is equivalent to the inverse-transpose for pure rotation transforms.
5. Return `Vec<InstanceRaw>`.

Instance buffer is a `wgpu::Buffer` with `VERTEX | COPY_DST` usage, rewritten each frame via `queue.write_buffer`.

---

## Model & Material

### `src/model.rs`

**`Vertex` trait** — provides `desc()` for `VertexBufferLayout`.

**`ModelVertex`** (bytemuck Pod/Zeroable):

```rust
#[repr(C)]
pub struct ModelVertex {
    pub position:   [f32; 3],   // location 0
    pub tex_coords: [f32; 2],   // location 1
    pub normal:     [f32; 3],   // location 2
    pub tangent:    [f32; 4],   // location 3 — xyz=tangent, w=±1 bitangent sign
}
// stride = 48 bytes
```

**`Material`**:

```rust
pub struct Material {
    pub name:             String,
    pub diffuse_texture:  texture::Texture,
    pub normal_texture:   texture::Texture,
    pub bind_group:       wgpu::BindGroup,   // group 0
}
```

**`Mesh`**:

```rust
pub struct Mesh {
    pub name:             String,
    pub vertex_buffer:    wgpu::Buffer,
    pub index_buffer:     wgpu::Buffer,
    pub num_elements:     u32,
    pub material:         usize,             // index into Model.materials
    pub bounding_sphere:  (cgmath::Point3<f32>, f32),
}
```

**`Model`**:

```rust
pub struct Model {
    pub meshes:    Vec<Mesh>,
    pub materials: Vec<Material>,
}
```

**`DrawModel` trait** on `wgpu::RenderPass`:

```rust
fn draw_mesh_instanced(&mut self, mesh: &Mesh, material: &Material,
    instances: Range<u32>, camera_bind_group: &BindGroup);

fn draw_model_instanced(&mut self, model: &Model,
    instances: Range<u32>, camera_bind_group: &BindGroup);
```

`draw_mesh_instanced` sets:
- `set_bind_group(0, material.bind_group)` — textures
- `set_bind_group(1, camera_bind_group)` — camera (passed in)
- Groups 2, 3, 4 are set per-frame by the render loop, not per-mesh.
- `set_vertex_buffer(0, mesh.vertex_buffer)`
- `set_index_buffer(mesh.index_buffer, Uint32)`
- `draw_indexed(0..num_elements, 0, instances)`

---

## Texture Handling

### `src/texture.rs`

```rust
pub struct Texture {
    pub texture: wgpu::Texture,
    pub view:    wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}
```

**`DEPTH_FORMAT`**: `Depth32Float`

**`create_depth_texture(device, config, label)`**:
- Format: `Depth32Float`
- Usage: `RENDER_ATTACHMENT | TEXTURE_BINDING`
- View aspect: `DepthOnly`
- Sampler: `CompareFunction::LessEqual` (for shadow sampling)

**`from_bytes(device, queue, bytes, label)`** — for sRGB color textures:
1. Decode PNG/JPEG bytes via `image::load_from_memory`.
2. Call `from_image_impl(..., Rgba8UnormSrgb)`.

**`from_image(device, queue, img, label)`** — for sRGB color textures:
- Calls `from_image_impl(..., Rgba8UnormSrgb)`.

**`from_image_linear(device, queue, img, label)`** — for linear textures (normal maps):
- Calls `from_image_impl(..., Rgba8Unorm)`.
- **Critical**: normal map directions must NOT be gamma-corrected by the GPU.

**`from_image_impl(device, queue, rgba_image, label, format)`**:
- Creates `wgpu::Texture` with `TEXTURE_BINDING | COPY_DST`, `mip_level_count = 1`.
- Writes pixels via `queue.write_texture`.
- Sampler: `mag_filter = Linear`, `min_filter = Nearest`, `mipmap_filter = Nearest`, `address_mode = ClampToEdge` (all axes).
- Returns `Texture { texture, view, sampler }`.

---

## Resource Loading

### `src/resources.rs`

**`res_path(file_name: &str) -> PathBuf`**:

```rust
let mut path = PathBuf::from(env!("OUT_DIR"));
path.push("res");
for part in file_name.split(['/', '\\']) {
    if !part.is_empty() { path.push(part); }
}
```

Handles both `/` and `\` separators so `ModelTag` strings work on Windows.

**`load_model(file_name, device, queue, layout) -> Result<Model>`**:
- If `file_name` ends in `.gltf` or `.glb` → `load_gltf`
- Otherwise → `load_obj`

**`create_material_bind_group(device, layout, diffuse, normal, label) -> BindGroup`**:
Creates group 0 bind group with 4 entries: diffuse view, diffuse sampler, normal view, normal sampler.

**`flat_normal_texture(device, queue, label) -> Texture`**:
Creates a 1×1 `Rgba8Unorm` texture with pixel `(128, 128, 255, 255)`. This encodes the tangent-space direction `(0, 0, 1)` — no normal perturbation, surface normals pass through unchanged. Used as fallback when a material has no normal map.

**`solid_color_texture(device, queue, rgba: [u8;4], label) -> Texture`**:
Creates a 1×1 `Rgba8UnormSrgb` texture. Used as fallback diffuse.

**`gltf_image_to_texture(data, label, device, queue, linear: bool) -> Texture`**:
Handles all glTF image pixel formats:
- `R8`, `R8G8`, `R8G8B8`, `R8G8B8A8` — direct conversion to RGBA8
- `R16`, `R16G16`, `R16G16B16`, `R16G16B16A16` — shift high byte down to 8-bit
- `R32G32B32FLOAT`, `R32G32B32A32FLOAT` — clamp to [0,1], then scale to 8-bit
- `Jpeg`, `Png` — decode via `image` crate
- If `linear=true` → `from_image_linear`, else → `from_image`

### glTF Loader (`load_gltf`)

```
gltf::import(path) → (document, buffers, images)
```

**Per material**:
1. `pbr.base_color_texture()` → `gltf_image_to_texture(..., linear=false)` (sRGB diffuse)
   - Fallback: `solid_color_texture([r*255, g*255, b*255, a*255])` from `pbr.base_color_factor()`
2. `mat.normal_texture()` → `gltf_image_to_texture(..., linear=true)` (linear normal map)
   - Fallback: `flat_normal_texture()`
3. Apply glTF sampler wrap modes: `Repeat`, `ClampToEdge`, `MirrorRepeat` (both U and V axes; W defaults to `Repeat`). Filter modes: Linear for mag/min/mipmap.
4. `create_material_bind_group(...)` → `Material`

**Fallback**: if the glTF file has no materials at all, creates a single default material with white diffuse and flat normal.

**Per primitive**:
1. Read `POSITION` (vec3) — **no coordinate transform** (glTF is already Y-up, right-handed)
2. Read `NORMAL` (vec3) — defaults to `[0,0,0]` if absent
3. Read `TEXCOORD_0` (vec2) — defaults to `[0,0]` if absent
4. Read `TANGENT` (vec4, w = bitangent sign) — defaults to `[1,0,0,1]` if absent
5. Read indices as `Vec<u32>` — defaults to sequential `[0, 1, 2, ...]` if absent
6. Build `Vec<ModelVertex>`, compute `bounding_sphere`
7. Create vertex buffer (`VERTEX`) and index buffer (`INDEX`)

### OBJ Loader (`load_obj`)

Uses `tobj::load_obj_buf_async` with triangulate=true.

- Resolves MTL files relative to the OBJ directory.
- Rewrites `mtllib` lines that contain absolute paths to use just the filename, so cross-platform OBJ files load correctly.
- Merges all sub-meshes sharing the same material into one mesh (reduces draw calls).
- Default tangent for OBJ meshes: `[1.0, 0.0, 0.0, 1.0]` (flat normal map works fine since no tangent data).
- Applies `flat_normal_texture()` for all materials.
- Flips V tex-coords: `tex_coords = [u, 1.0 - v]`.
- Sorts meshes by material ID for deterministic rendering order.

---

## ECS Game Layer

Uses **hecs** (archetypal ECS). No systems framework — queries are run manually.

### Components (`src/game/components.rs`)

```rust
pub struct ModelTag(pub &'static str);            // which model file to render
pub struct Position(pub cgmath::Vector3<f32>);     // world position
pub struct Rotation(pub cgmath::Quaternion<f32>);   // world rotation
pub struct PointLight {
    pub color:     cgmath::Vector3<f32>,
    pub intensity: f32,
}
pub struct Flashlight {
    pub enabled: bool,     // false = light zeroed out in renderer
}
pub struct DirectionalLight {
    pub direction: cgmath::Vector3<f32>,  // points TOWARD light source
    pub color:     cgmath::Vector3<f32>,
    pub intensity: f32,
}
```

### World setup (`src/game/init.rs`)

```rust
// Camera entity: drives the view, carries the flashlight
world.spawn((
    CameraController::new(1_000.0),
    Position(cgmath::vec3(CAMERA_EYE.x, CAMERA_EYE.y, CAMERA_EYE.z)),
    PointLight { color: vec3(1.0, 0.92, 0.8), intensity: 200.0 },
    Flashlight { enabled: false },
));

// Sponza model
world.spawn(teapot(vec3(0.0, 0.0, 0.0)));

// Sun (directional light only, no position)
world.spawn((DirectionalLight {
    direction: vec3(0.4, 1.0, 0.3).normalize(),
    color:     vec3(1.0, 0.95, 0.85),
    intensity: 1.5,
},));
```

`teapot()` in `entities.rs` returns:
```rust
(ModelTag(r"sponza/Sponza.gltf"), Position(origin), Rotation::one())
```

### Systems (`src/game/systems.rs`)

`update(world, camera, dt)`:
1. Find entity with `CameraController`, call `controller.update_camera(camera, dt)`.
2. Sync `Position` component on that entity to `camera.eye` — so the `PointLight` position is always at the camera.

### Input (`src/game/input.rs`)

`handle_key(key, pressed, world)`:
- `KeyF` + pressed → toggle all `Flashlight.enabled` in world.
- All keys → forward to all `CameraController.handle_key()`.

`handle_mouse_motion(dx, dy, world)`:
- Forward to all `CameraController.handle_mouse_motion()`.

---

## egui Debug Overlay

The engine uses `egui` with `egui-wgpu` and `egui-winit` for an in-game debug panel.

### UI mode toggle

- **Tab** toggles `ui_mode`. When entering UI mode, the cursor is released (`CursorGrabMode::None`) and made visible. When exiting, the cursor is re-locked and hidden.
- **Escape** always exits the application (bypasses egui).
- When `ui_mode` is true, egui consumes mouse clicks and the camera does not receive mouse motion.

### Debug window

The `egui::Window` labeled "Debug" is divided into sections with `ui.heading()` separators.

**Header**:
- **FPS**: current frames per second.
- **GPU**: frame GPU time in milliseconds (shadow + main passes). Only shown when `TIMESTAMP_QUERY` is supported. Measured via `RenderPassTimestampWrites` on the shadow and main passes; the result is read back each frame via blocking `device.poll(PollType::Wait)` after submit.
- **Camera position**: `(x, y, z)` of the camera eye.

**Render section**:
- **Lighting toggle**: "Lighting: ON/OFF". When off, `LightUniform.lighting_on = 0.0` and the shader returns raw diffuse (fullbright).
- **Depth map toggle**: "Depth Map: ON/OFF". Enables the log-scaled grayscale depth overlay pass.
- **Wireframe toggle**: "Wireframe: ON/OFF". Switches between `render_pipeline` and `wireframe_pipeline`.

**Camera section**:
- **FOV slider**: `egui::Slider` 10°–120°, modifies `Camera.fovy` directly; takes effect next frame since `view_proj` is rebuilt each `update()`.
- **Speed slider**: logarithmic slider 10–10 000, modifies `CameraController.speed` (field is `pub`).

**Flashlight section**:
- **Enabled checkbox**: toggles `Flashlight.enabled`.
- **Intensity slider**: logarithmic, 0–2 000, modifies `PointLight.intensity`.

**Sun section**:
- **Dir X / Y / Z sliders**: −1.0 to 1.0, modify `DirectionalLight.direction` components. Direction is re-uploaded to the GPU and the light-space matrix is recomputed every frame, so changes are immediate. If all three components are zero the shader will `normalize(0,0,0)` (NaN); avoid this.
- **Intensity slider**: 0.0–5.0, modifies `DirectionalLight.intensity`.
- **Color picker**: `egui::color_picker::color_edit_button_rgb`, modifies `DirectionalLight.color`.

**Footer**:
- **Help text**: "Tab — open UI" or "Tab — return to game".

### egui render integration

1. `State::egui_on_window_event()` forwards each `WindowEvent` to `egui_winit::State::on_window_event()`. Returns whether egui consumed the event.
2. In `render()`, after the scene pass:
   - `egui_ctx.begin_pass(raw_input)` — starts a frame.
   - Build UI widgets.
   - `egui_ctx.end_pass()` — produces `FullOutput`.
   - Write back changed values: `lighting_enabled`, `show_depth`, `wireframe`, `camera.fovy`, `Flashlight.enabled`, `PointLight.intensity`, `DirectionalLight.{direction, intensity, color}`, `CameraController.speed`.
   - `handle_platform_output()` for cursor/clipboard.
   - `tessellate()` produces triangles.
   - `update_texture()` / `update_buffers()` upload egui geometry.
   - Begin a new render pass with `LoadOp::Load` on the color attachment (composites over the scene).
   - `forget_lifetime()` on the render pass to satisfy egui's `'static` lifetime requirement.
   - `egui_renderer.render()` draws the UI.
   - Free released textures.
   - Submit all command buffers (egui's extra + the main encoder).

---

## Frame Loop

### `update()` in `renderer.rs`

1. **Frame pacing**:
   - `target_frame_time` = 1 / monitor refresh rate (queried via winit, fallback to 60 Hz).
   - Compute `remaining = frame_deadline - now - spin_margin`.
   - `std::thread::sleep(remaining)` for the bulk, then spin-wait the last ~200 µs for precision.
   - Update `frame_deadline += target_frame_time`.
   - If the frame is overdue (`now > frame_deadline`), reset the deadline to `now + target_frame_time` to avoid spiral of death.

2. **Delta time**: `dt = now.duration_since(last_frame).as_secs_f32(); last_frame = now;`

3. **Camera**: `game::update(&mut world, &mut camera, dt)`

4. **Camera uniform**: recompute `view_proj`, write to `camera_buffer`.

5. **Frustum**: `frustum = Frustum::from_vp(&vp)` where `vp = OPENGL_TO_WGPU_MATRIX * camera.build_view_projection_matrix()`.

6. **Point light**: query `(&Position, &PointLight, Option<&Flashlight>)` from world. If `Flashlight.enabled` is false, zero out `intensity`. Set `lighting_on` based on `State::lighting_enabled`. Write to `light_buffer`.

7. **Directional light**: query `&DirectionalLight`, build `DirLightUniform`, write to `dir_light_buffer` every frame. This is required for egui sun direction/intensity/color changes to take effect immediately (the buffer is not static).

8. **Light-space matrix**: recompute from `DirectionalLight.direction` each frame and write to `light_space_buffer`.

9. **Instance data**: for each loaded model, call `build_instance_data_for(world, tag)`, write to instance buffer.

10. **FPS counter**: every second, compute `fps = frame_count / elapsed`, update window title to `"engine | {fps:.0} FPS"`, and reset the counter.

### `render()` in `renderer.rs`

1. `surface.get_current_texture()` — handle all cases:
   - `Success` → continue rendering.
   - `Suboptimal` → drop the surface texture, reconfigure surface, return early.
   - `Timeout` / `Occluded` / `Validation` → return `Ok(())` (skip frame).
   - `Outdated` → reconfigure surface, return early.
   - `Lost` → bail with error.

2. Create `TextureView` from surface texture.

3. Create `CommandEncoder`.

4. **Pass 1: Shadow map**:
   - If `TIMESTAMP_QUERY` is supported, attach `RenderPassTimestampWrites` with slots 0 (begin) and 1 (end).
   - Begin render pass with no color attachments and `shadow_render_view` as depth attachment, clear to 1.0.
   - Set `shadow_pipeline`.
   - Set bind group 0 (`shadow_pass_bg` — light-space matrix).
   - For each model: set instance buffer as slot 1, then for each mesh set vertex/index buffers and `draw_indexed`.
   - End pass.

5. **Pass 2: Main scene**:
   - If `TIMESTAMP_QUERY` is supported, attach `RenderPassTimestampWrites` with slots 2 (begin) and 3 (end).
   - Begin render pass with color attachment (clear to `{r:0.1, g:0.2, b:0.3, a:1.0}`) and depth attachment (clear to 1.0).
   - Set `render_pipeline` or `wireframe_pipeline` based on `State::wireframe`.
   - Set bind groups 2 (point light), 3 (dir light), 4 (shadow).
   - For each model:
     - Set instance buffer as slot 1.
     - For each mesh: check `frustum.cull_sphere(mesh.bounding_sphere)`, skip if culled.
     - Call `draw_mesh_instanced(mesh, material, 0..instance_count, camera_bind_group)`.
   - End pass.

6. **Pass 3: Depth debug** (optional, only if `show_depth` is true):
   - Begin render pass with color attachment (clear to black, no depth stencil).
   - Set `depth_debug_pipeline`.
   - Set bind group 0 (`depth_debug_bind_group`).
   - Draw 3 vertices (fullscreen triangle).

7. **egui pass**:
   - Take raw input, `begin_pass`, build UI, `end_pass`.
   - Write back any changed values (lighting, flashlight, wireframe, depth debug, intensity).
   - Handle platform output (cursor changes, clipboard).
   - Tessellate shapes.
   - Update egui textures and buffers.
   - Begin render pass with `LoadOp::Load` (composites over scene).
   - `forget_lifetime()` to satisfy borrow checker.
   - `egui_renderer.render()`.
   - Free released textures.

8. **Timestamp resolve** (if supported): `encoder.resolve_query_set(query_set, 0..4, resolve_buf, 0)` followed by `encoder.copy_buffer_to_buffer(resolve_buf, read_buf, 32)`. Both commands are recorded into the encoder before `finish()`.

9. `queue.submit(extra_cmds + encoder.finish())`.

10. `surface_texture.present()`.

11. **GPU frametime readback** (if supported): `read_buf.slice(..).map_async(Read, ...)` then `device.poll(PollType::Wait)` to block until the GPU has written the timestamps. Read `[u64; 4]` via `bytemuck::cast_slice`, compute `(ts[3] - ts[0]) * timestamp_period / 1_000_000.0` milliseconds (timestamp slots: 0=shadow begin, 1=shadow end, 2=main begin, 3=main end). Unmap the buffer. The blocking poll causes a brief CPU stall equal to the GPU frame time, which is acceptable for a debug measurement.

---

## Key Design Decisions

### Coordinate system

- **Y-up, right-handed** throughout. Camera uses `look_at_rh`. glTF models load without transformation (glTF is also Y-up right-handed).
- `OPENGL_TO_WGPU_MATRIX` remaps Z from `[-1,1]` to `[0,1]` for wgpu's NDC. This matrix is stored column-major via `from_cols`.

### Normal map texture format

- Diffuse textures: `Rgba8UnormSrgb` — GPU applies inverse gamma on read, shader works in linear space.
- Normal maps: `Rgba8Unorm` — **must not** be sRGB. Direction vectors must not be gamma-corrected.

### Bitangent sign (`tangent.w`)

- glTF stores tangent as `vec4` where `w = ±1` indicates the bitangent handedness.
- Shader reconstructs: `B = tangent.w * cross(N, T)`. Do not store bitangent explicitly.
- OBJ meshes use default tangent `[1, 0, 0, 1]`.

### Light direction convention

- `DirectionalLight.direction` points **toward** the light source (the direction light rays travel from).
- `dot(normal, sun_dir)` is positive on surfaces facing the sun. This is the "surface-to-light" convention.
- The shader does `normalize(dir_light.direction)` — the vector must be non-zero at upload time.

### Point light attenuation

```
attenuation = intensity / (dist² + intensity)
```

- When `dist=0`: attenuation = 1.0 (full intensity).
- As `dist → ∞`: attenuation → 0.
- `intensity` acts as a softening constant — higher values keep brightness elevated at longer range.

### Flashlight implementation

- The camera entity carries `PointLight` + `Flashlight` + `Position` components.
- `systems.rs` syncs `camera.eye → Position` each frame.
- Renderer reads position from `Position` component, not directly from camera.
- When `Flashlight.enabled` is false, renderer uploads `{position, intensity: 0, color}` — position is preserved (not zeroed) to avoid artifacts.
- `lighting_on` field is separate from flashlight: it controls fullbright mode globally (when `0.0`, shader outputs raw diffuse color).

### Shadow mapping

- Orthographic projection from the directional light's perspective.
- Light-space matrix is recomputed every frame from the ECS `DirectionalLight.direction`.
- Shadow map resolution: 2048×2048 `Depth32Float`.
- The shadow pass renders all meshes with the same instance buffers as the main pass.
- Shadow bias: 0.005 (constant, applied in the shader).
- Fragments outside the light frustum return `shadow_factor = 1.0` (lit, no shadow).
- Only the **sun** contribution is attenuated by shadows. The point light is not.
- Soft shadows via **5×5 PCF** (Percentage Closer Filtering): 25 `textureSampleCompare` samples on a uniform grid with a texel-sized step, averaged. Result is `sum / 25.0`.

### GPU timestamp queries

- `TIMESTAMP_QUERY` is requested conditionally — checked against `adapter.features()` before device creation, so the engine still runs on GPUs that don't support it.
- Four timestamp slots: shadow pass begin (0), shadow pass end (1), main pass begin (2), main pass end (3). The delta `ts[3] - ts[0]` covers the entire GPU frame excluding egui and depth debug (both negligible).
- Timestamps are read back synchronously via `device.poll(PollType::Wait)` in the same frame they are written. This stalls the CPU for roughly one GPU frame time but keeps the implementation simple.
- `timestamp_period` (nanoseconds per tick) is obtained from `queue.get_timestamp_period()` at startup and used to convert tick differences to milliseconds.

### Fullbright mode

- When `light.lighting_on` is `0.0`, the fragment shader skips all lighting calculations and returns `object_color` directly.
- This allows toggling lighting on/off from the egui debug panel without changing the light structure layout.
- Controlled by the `State::lighting_enabled` boolean, which is toggled via the debug UI.

### Depth debug visualization

- A separate pipeline renders a fullscreen triangle that reads the main depth buffer and outputs log-scaled grayscale.
- Near objects appear white, far objects and empty sky appear black.
- Linear Z is reconstructed from the depth buffer using `z_eye = znear * zfar / (zfar - depth * (zfar - znear))`, then `log_d = log2(z_eye / znear) / log2(zfar / znear)`.
- The depth debug pass clears the color attachment to black and renders over the scene (it does not blend).

### Wireframe mode

- A second pipeline uses the same shader and layout but with `PolygonMode::Line` instead of `Fill`.
- Requires `Features::POLYGON_MODE_LINE` from the device.
- Toggled via the egui debug panel.

### Frustum culling

- Per-mesh sphere test (not per-model). Sponza has many meshes; per-mesh culling gives significant savings.
- Bounding sphere is computed from vertex positions during load: centroid + max radius.
- `cull_sphere` returns `true` = skip draw call.
- Frustum extraction uses row-based Gribb/Hartmann method adapted for wgpu's Z in [0,1].

### Instance buffer updates

- Instance buffers are rewritten every frame via `queue.write_buffer`.
- Instance count is updated dynamically each frame (not fixed at startup).
- Buffer is created at startup with a size based on the initial instance count. Adding many more entities at runtime requires recreating buffers.

### Surface reconfiguration

- On `Suboptimal`: drop the `SurfaceTexture` first, then reconfigure, then skip the rest of render for that frame.
- On `Outdated`: reconfigure surface, return early.
- On `Resized`: reconfigure surface, recreate depth texture, recreate depth debug bind group.
- `is_surface_configured` flag prevents rendering before the first resize event.

### Frame pacing

- Target frame time is derived from the monitor's refresh rate (queried via winit, fallback to 60 Hz).
- The engine uses a hybrid sleep + spin-wait strategy: `thread::sleep` for the bulk of the frame time, then a spin loop for the last ~200 µs for precision.
- If a frame runs over its deadline, the deadline is reset to `now + target_frame_time` to prevent accumulating lag.

### egui integration

- egui is initialized alongside wgpu and winit during `State::new()`.
- Input events flow through `State::egui_on_window_event()` before being dispatched to the game.
- When `ui_mode` is active, egui consumes mouse clicks and the camera does not receive mouse delta.
- Escape and Tab always reach the game handler regardless of egui consumption.
- The egui render pass uses `LoadOp::Load` to composite over the scene.
- `forget_lifetime()` is used on the wgpu render pass to satisfy egui's `'static` lifetime requirement.
- ECS components modified from the debug panel (`DirectionalLight`, `PointLight`, `CameraController.speed`) are written to GPU buffers the same frame, so changes are immediately visible.

### Event loop

- Control flow is set to `Poll` (never sleeps between events). This prevents stuttering caused by the default `Wait` mode waking up too late.
- FPS is displayed in the window title bar, updated every second.

### Backend selection

Controlled by CLI flags:

| Flag | Backend / Preference |
|------|---------------------|
| (default) | `Backends::PRIMARY`, default power preference |
| `-vulkan` | `Backends::VULKAN` |
| `-dx12` | `Backends::DX12` |
| `-opengl` | `Backends::GL` |
| `-metal` | `Backends::METAL` |
| `-software` | Force CPU rasterizer (`force_fallback_adapter = true`) |
| `-discrete` | `PowerPreference::HighPerformance` |
| `-integrated` | `PowerPreference::LowPower` |

The engine enumerates and prints all available adapters at startup, then prints the selected adapter's name, backend, device type, and driver info. Warnings are emitted if the requested preference could not be satisfied.

### ECS usage

- **hecs** is used as a plain data store. There is no system scheduler.
- Queries are run explicitly from `update()`, `render()`, input handlers, and the egui closure.
- The world is owned by `State` in `renderer.rs` and passed by mutable reference to game functions.