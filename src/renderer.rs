use std::{collections::HashMap, iter, sync::Arc, time::{Duration, Instant}};

use egui_wgpu::ScreenDescriptor;

use hecs::World;
use wgpu::util::DeviceExt;
use winit::{event_loop::ActiveEventLoop, keyboard::KeyCode, window::Window};

use crate::app::Config;
use crate::camera::{Camera, CameraUniform, OPENGL_TO_WGPU_MATRIX};
use crate::frustum::Frustum;
use crate::game::{DirectionalLight, Flashlight, ModelTag, PointLight, Position};
use crate::instance::{InstanceRaw, build_instance_data_for};
use crate::model::{DrawModel, Vertex};
use crate::resources;
use crate::texture;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform {
    position:    [f32; 3],
    intensity:   f32,
    color:       [f32; 3],
    lighting_on: f32,   // 1.0 = lit, 0.0 = fullbright
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DirLightUniform {
    direction: [f32; 3],
    intensity: f32,
    color:     [f32; 3],
    _pad:      f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightSpaceUniform {
    matrix: [[f32; 4]; 4],
}

/// Minimum number of instance slots allocated per model buffer.
/// Actual capacity is always rounded up to the next power of two.
const MIN_CAPACITY: u32 = 64;

// Shadow map configuration — tweak these to cover your scene.
const SHADOW_MAP_SIZE:  u32  = 2048;
const SHADOW_ORTHO_SIZE: f32 = 2000.0; // half-extent of the ortho frustum (world units)
const SHADOW_LIGHT_DIST: f32 = 5000.0; // how far back the shadow camera sits
const SHADOW_ZNEAR: f32      = 3000.0; // near clip of the shadow camera
const SHADOW_ZFAR:  f32      = 8000.0; // far  clip of the shadow camera

/// Per-model GPU resources and current instance count.
struct ModelEntry {
    model: crate::model::Model,
    instance_buffer: wgpu::Buffer,
    /// Number of instances currently written into the buffer.
    instance_count: u32,
    /// Allocated capacity of `instance_buffer` in instances.
    /// The buffer is recreated (doubled) when `instance_count` exceeds this.
    instance_capacity: u32,
}

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    render_pipeline: wgpu::RenderPipeline,
    wireframe_pipeline: wgpu::RenderPipeline,
    wireframe: bool,
    /// One entry per unique ModelTag found in the world after game::init.
    models: HashMap<String, ModelEntry>,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    world: World,
    depth_texture: texture::Texture,
    window: Arc<Window>,
    light_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    dir_light_buffer: wgpu::Buffer,
    dir_light_bind_group: wgpu::BindGroup,
    // ── Shadow mapping ─────────────────────────────────────────────────────────
    light_space_buffer: wgpu::Buffer,
    /// View used as the depth attachment in the shadow pass.
    shadow_render_view: wgpu::TextureView,
    shadow_pipeline: wgpu::RenderPipeline,
    /// Bind group 0 for the shadow pass  (light_space_matrix).
    shadow_pass_bg: wgpu::BindGroup,
    /// Bind group 4 for the main pass   (light_space_matrix + shadow map + sampler).
    shadow_main_bg: wgpu::BindGroup,
    fps_frame_count: u32,
    fps_timer: Instant,
    last_frame: Instant,
    last_fps: f32,
    frustum: Frustum,
    /// Target time between frames (derived from monitor refresh rate).
    target_frame_time: Duration,
    /// Absolute timestamp of the next desired frame start.
    frame_deadline: Instant,
    // ── GPU timestamp queries ──────────────────────────────────────────────────
    /// None when the GPU/driver doesn't support TIMESTAMP_QUERY.
    timestamp_query_set:   Option<wgpu::QuerySet>,
    timestamp_resolve_buf: Option<wgpu::Buffer>, // QUERY_RESOLVE | COPY_SRC
    timestamp_read_buf:    Option<wgpu::Buffer>,  // COPY_DST | MAP_READ
    /// Nanoseconds per GPU timestamp tick.
    timestamp_period: f32,
    /// GPU time of the previous frame's shadow + main passes, in milliseconds.
    last_gpu_ms: f32,
    // ── egui ──────────────────────────────────────────────────────────────────
    egui_ctx: egui::Context,
    egui_winit: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    /// When true: cursor is free, egui accepts input. Tab toggles this.
    ui_mode: bool,
    /// When false the shader outputs raw diffuse (fullbright).
    lighting_enabled: bool,
    /// When true, renders the depth buffer as a grayscale overlay.
    show_depth: bool,
    depth_debug_pipeline: wgpu::RenderPipeline,
    depth_debug_bind_group_layout: wgpu::BindGroupLayout,
    depth_debug_bind_group: wgpu::BindGroup,
}

impl State {
    pub async fn new(
        window: Arc<Window>,
        camera_eye: cgmath::Point3<f32>,
        camera_target: cgmath::Point3<f32>,
        config: &Config,
    ) -> anyhow::Result<State> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: config.backends,
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });

        // List all available adapters
        let all_adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
        println!("Available GPUs:");
        for a in &all_adapters {
            let info = a.get_info();
            println!("  [{:?}] {} ({:?})", info.backend, info.name, info.device_type);
        }

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: config.power_preference,
                compatible_surface: Some(&surface),
                force_fallback_adapter: config.force_software,
            })
            .await
            .unwrap();

        let info = adapter.get_info();
        println!(
            "Selected GPU: {} | Backend: {:?} | Type: {:?} | Driver: {}",
            info.name, info.backend, info.device_type, info.driver_info,
        );

        // Warn if the selected adapter doesn't match what was requested
        if config.power_preference == wgpu::PowerPreference::HighPerformance
            && info.device_type != wgpu::DeviceType::DiscreteGpu
        {
            println!("Warning: -discrete requested but no discrete GPU found, using {:?} instead", info.device_type);
        }
        if config.power_preference == wgpu::PowerPreference::LowPower
            && info.device_type != wgpu::DeviceType::IntegratedGpu
        {
            println!("Warning: -integrated requested but no integrated GPU found, using {:?} instead", info.device_type);
        }
        if config.force_software && info.device_type != wgpu::DeviceType::Cpu {
            println!("Warning: -software requested but no software renderer found, using {:?} instead", info.device_type);
        }

        let adapter_features = adapter.features();
        let mut required_features = wgpu::Features::POLYGON_MODE_LINE;
        if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        } else {
            println!("TIMESTAMP_QUERY not supported — GPU frametime unavailable");
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits: wgpu::Limits {
                    max_bind_groups: 5,
                    ..wgpu::Limits::default()
                },
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: if surface_caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
                wgpu::PresentMode::Immediate
            } else {
                wgpu::PresentMode::Fifo
            },
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // Diffuse
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Normal map
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let camera = Camera {
            eye: camera_eye,
            target: camera_target,
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 1_000_000.0,
        };

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let mut world = World::new();
        crate::game::init(&mut world);

        // Discover all unique ModelTags spawned by game::init and load each model once.
        let tags: std::collections::HashSet<String> = world
            .query::<&ModelTag>()
            .iter()
            .map(|(_, t)| t.0.to_string())
            .collect();

        println!("Loading models: {:?}", tags);
        let mut models = HashMap::new();
        for tag in &tags {
            let model = resources::load_model(tag, &device, &queue, &texture_bind_group_layout)
                .await
                .unwrap();
            let instance_data = build_instance_data_for(&world, tag);
            let instance_count = instance_data.len() as u32;
            // Allocate at least MIN_CAPACITY slots, rounded to the next power of two,
            // so the buffer can absorb runtime spawns without immediate reallocation.
            let instance_capacity = instance_count.max(MIN_CAPACITY).next_power_of_two();
            let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(tag.as_str()),
                size: (instance_capacity as usize * std::mem::size_of::<InstanceRaw>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if !instance_data.is_empty() {
                queue.write_buffer(&instance_buffer, 0, bytemuck::cast_slice(&instance_data));
            }
            models.insert(tag.clone(), ModelEntry { model, instance_buffer, instance_count, instance_capacity });
        }

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("light_bind_group_layout"),
            });

        let light_uniform = LightUniform {
            position:    [0.0, 0.0, 0.0],
            intensity:   1.0,
            color:       [1.0, 1.0, 1.0],
            lighting_on: 1.0,
        };
        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Buffer"),
            contents: bytemuck::cast_slice(&[light_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
            label: Some("light_bind_group"),
        });

        let dir_light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("dir_light_bind_group_layout"),
            });

        let dir_light_data = world
            .query::<&DirectionalLight>()
            .iter()
            .next()
            .map(|(_, dl)| DirLightUniform {
                direction: [dl.direction.x, dl.direction.y, dl.direction.z],
                intensity: dl.intensity,
                color:     [dl.color.x, dl.color.y, dl.color.z],
                _pad:      0.0,
            })
            .unwrap_or(DirLightUniform {
                direction: [0.0, 1.0, 0.0],
                intensity: 0.0,
                color:     [1.0, 1.0, 1.0],
                _pad:      0.0,
            });
        let dir_light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dir Light Buffer"),
            contents: bytemuck::cast_slice(&[dir_light_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let dir_light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &dir_light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: dir_light_buffer.as_entire_binding(),
            }],
            label: Some("dir_light_bind_group"),
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        // Group 4 layout for the main pass: light-space matrix + shadow map + comparison sampler.
        // Must be created before render_pipeline_layout so it can be listed there.
        let shadow_main_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_main_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&texture_bind_group_layout),
                    Some(&camera_bind_group_layout),
                    Some(&light_bind_group_layout),
                    Some(&dir_light_bind_group_layout),
                    Some(&shadow_main_bgl),          // group 4: shadow map
                ],
                immediate_size: 0,
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[crate::model::ModelVertex::desc(), InstanceRaw::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        let wireframe_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Wireframe Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[crate::model::ModelVertex::desc(), InstanceRaw::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Line,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        // ── Shadow mapping resources ───────────────────────────────────────────
        let shadow_map = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow_map"),
            size: wgpu::Extent3d {
                width:  SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture::Texture::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[texture::Texture::DEPTH_FORMAT],
        });
        // Default view — used as the depth attachment during the shadow pass.
        let shadow_render_view = shadow_map.create_view(&wgpu::TextureViewDescriptor::default());
        // DepthOnly view — used for shadow sampling in the fragment shader.
        let shadow_sample_view = shadow_map.create_view(&wgpu::TextureViewDescriptor {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });

        // Compute the initial light-space matrix from the DirectionalLight in the ECS.
        let init_dir = world
            .query::<&DirectionalLight>()
            .iter()
            .next()
            .map(|(_, dl)| dl.direction)
            .unwrap_or_else(|| cgmath::vec3(0.4, 1.0, 0.3));
        let init_ls = LightSpaceUniform {
            matrix: Self::compute_light_space_matrix(init_dir).into(),
        };
        let light_space_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("light_space_buffer"),
            contents: bytemuck::cast_slice(&[init_ls]),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Comparison sampler for textureSampleCompare in the fragment shader.
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // Group 0 layout for the shadow pipeline: just the light-space matrix.
        let shadow_pass_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_pass_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let shadow_pass_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:  Some("shadow_pass_bg"),
            layout: &shadow_pass_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: light_space_buffer.as_entire_binding(),
            }],
        });

        let shadow_main_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:  Some("shadow_main_bg"),
            layout: &shadow_main_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: light_space_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: wgpu::BindingResource::TextureView(&shadow_sample_view),
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
            ],
        });

        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("shadow.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shadow.wgsl").into()),
        });
        let shadow_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label:              Some("shadow_pipeline_layout"),
                bind_group_layouts: &[Some(&shadow_pass_bgl)],
                immediate_size:     0,
            });
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("shadow_pipeline"),
            layout: Some(&shadow_pipeline_layout),
            vertex: wgpu::VertexState {
                module:               &shadow_shader,
                entry_point:          Some("vs_shadow"),
                buffers:              &[crate::model::ModelVertex::desc(), InstanceRaw::desc()],
                compilation_options:  Default::default(),
            },
            fragment: None,   // depth-only; no colour attachment
            primitive: wgpu::PrimitiveState {
                topology:   wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode:  Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:               texture::Texture::DEPTH_FORMAT,
                depth_write_enabled:  Some(true),
                depth_compare:        Some(wgpu::CompareFunction::Less),
                stencil:              wgpu::StencilState::default(),
                bias:                 wgpu::DepthBiasState::default(),
            }),
            multisample:   wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:          None,
        });

        // ── Depth debug pipeline ───────────────────────────────────────────────
        let depth_debug_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("depth_debug_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        let depth_debug_bind_group = Self::make_depth_debug_bind_group(
            &device,
            &depth_debug_bind_group_layout,
            &depth_texture,
        );

        let depth_debug_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("depth_debug.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("depth_debug.wgsl").into()),
        });
        let depth_debug_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("depth_debug_layout"),
                bind_group_layouts: &[Some(&depth_debug_bind_group_layout)],
                immediate_size: 0,
            });
        let depth_debug_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("depth_debug_pipeline"),
                layout: Some(&depth_debug_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &depth_debug_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &depth_debug_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        // ── egui ──────────────────────────────────────────────────────────────
        let egui_ctx = egui::Context::default();
        let egui_winit = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            None,
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            surface_format,
            egui_wgpu::RendererOptions {
                depth_stencil_format: None,
                msaa_samples: 1,
                ..Default::default()
            },
        );

        // Derive target frame time from the monitor's refresh rate.
        let refresh_mhz = window
            .current_monitor()
            .and_then(|m| m.refresh_rate_millihertz())
            .unwrap_or(60_000);
        let target_frame_time = Duration::from_nanos(1_000_000_000_000 / refresh_mhz as u64);
        println!(
            "Monitor refresh: {:.1} Hz  →  target frame time: {:.3} ms",
            refresh_mhz as f64 / 1000.0,
            target_frame_time.as_secs_f64() * 1000.0,
        );

        // ── GPU timestamp query resources ──────────────────────────────────────
        let timestamp_period = queue.get_timestamp_period();
        let (timestamp_query_set, timestamp_resolve_buf, timestamp_read_buf) =
            if device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
                let qs = device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("timestamp_qs"),
                    ty:    wgpu::QueryType::Timestamp,
                    count: 4, // 0=shadow_begin 1=shadow_end 2=main_begin 3=main_end
                });
                let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label:              Some("timestamp_resolve"),
                    size:               4 * 8, // 4 × u64
                    usage:              wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label:              Some("timestamp_read"),
                    size:               4 * 8,
                    usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                (Some(qs), Some(resolve_buf), Some(read_buf))
            } else {
                (None, None, None)
            };

        let now = Instant::now();
        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            render_pipeline,
            wireframe_pipeline,
            wireframe: false,
            models,
            camera,
            camera_buffer,
            camera_bind_group,
            camera_uniform,
            world,
            depth_texture,
            window,
            light_buffer,
            light_bind_group,
            dir_light_buffer,
            dir_light_bind_group,
            light_space_buffer,
            shadow_render_view,
            shadow_pipeline,
            shadow_pass_bg,
            shadow_main_bg,
            fps_frame_count: 0,
            fps_timer: now,
            last_frame: now,
            last_fps: 0.0,
            frustum: Frustum::default(),
            target_frame_time,
            frame_deadline: now + target_frame_time,
            timestamp_query_set,
            timestamp_resolve_buf,
            timestamp_read_buf,
            timestamp_period,
            last_gpu_ms: 0.0,
            egui_ctx,
            egui_winit,
            egui_renderer,
            ui_mode: false,
            lighting_enabled: true,
            show_depth: false,
            depth_debug_pipeline,
            depth_debug_bind_group_layout,
            depth_debug_bind_group,
        })
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    /// Mutable access to the ECS world so game code can spawn or despawn entities at runtime.
    /// After calling this, the instance buffers will be updated automatically on the next frame.
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.is_surface_configured = true;
            self.camera.aspect = self.config.width as f32 / self.config.height as f32;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture =
                texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
            self.depth_debug_bind_group = Self::make_depth_debug_bind_group(
                &self.device,
                &self.depth_debug_bind_group_layout,
                &self.depth_texture,
            );
        }
    }

    pub fn handle_mouse_motion(&mut self, dx: f64, dy: f64) {
        crate::game::handle_mouse_motion(dx, dy, &mut self.world);
    }

    fn make_depth_debug_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        depth_texture: &texture::Texture,
    ) -> wgpu::BindGroup {
        let depth_only_view = depth_texture.texture.create_view(
            &wgpu::TextureViewDescriptor {
                aspect: wgpu::TextureAspect::DepthOnly,
                ..Default::default()
            },
        );
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("depth_debug_bind_group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&depth_only_view),
            }],
        })
    }

    pub fn ui_mode(&self) -> bool {
        self.ui_mode
    }

    /// Forward a winit WindowEvent to egui. Returns true if egui consumed it.
    pub fn egui_on_window_event(&mut self, event: &winit::event::WindowEvent) -> bool {
        let response = self.egui_winit.on_window_event(&self.window, event);
        response.consumed
    }

    /// Toggle between UI mode (cursor free, egui interactive) and game mode (cursor locked).
    pub fn toggle_ui_mode(&mut self) {
        self.ui_mode = !self.ui_mode;
        if self.ui_mode {
            let _ = self.window.set_cursor_grab(winit::window::CursorGrabMode::None);
            self.window.set_cursor_visible(true);
        } else {
            let _ = self.window.set_cursor_grab(winit::window::CursorGrabMode::Locked)
                .or_else(|_| self.window.set_cursor_grab(winit::window::CursorGrabMode::Confined));
            self.window.set_cursor_visible(false);
        }
    }

    pub fn handle_key(&mut self, event_loop: &ActiveEventLoop, key: KeyCode, pressed: bool) {
        if key == KeyCode::Escape && pressed {
            event_loop.exit();
        } else if key == KeyCode::Tab && pressed {
            self.toggle_ui_mode();
        } else {
            crate::game::handle_key(key, pressed, &mut self.world);
        }
    }

    /// Builds the orthographic light-space matrix used by both render passes.
    /// `direction` points **toward** the light source (same convention as `DirectionalLight`).
    fn compute_light_space_matrix(direction: cgmath::Vector3<f32>) -> cgmath::Matrix4<f32> {
        use cgmath::InnerSpace;
        let dir = direction.normalize();
        // Place the shadow camera behind the scene along the light direction.
        let p = dir * SHADOW_LIGHT_DIST;
        let eye = cgmath::Point3::new(p.x, p.y, p.z);
        let target = cgmath::Point3::new(0.0, 0.0, 0.0);
        // Avoid a degenerate up vector when the light is almost straight down.
        let up = if dir.y.abs() > 0.99 {
            cgmath::Vector3::unit_z()
        } else {
            cgmath::Vector3::unit_y()
        };
        let view = cgmath::Matrix4::look_at_rh(eye, target, up);
        let proj = cgmath::ortho(
            -SHADOW_ORTHO_SIZE,  SHADOW_ORTHO_SIZE,
            -SHADOW_ORTHO_SIZE,  SHADOW_ORTHO_SIZE,
             SHADOW_ZNEAR,       SHADOW_ZFAR,
        );
        // OPENGL_TO_WGPU_MATRIX remaps Z from [-1,1] → [0,1] for wgpu.
        OPENGL_TO_WGPU_MATRIX * proj * view
    }

    pub fn update(&mut self) {
        // ── Frame pacing ───────────────────────────────────────────────────────
        let now = Instant::now();
        if self.frame_deadline > now {
            let remaining = self.frame_deadline - now;
            const SPIN_MARGIN: Duration = Duration::from_micros(200);
            if remaining > SPIN_MARGIN {
                std::thread::sleep(remaining - SPIN_MARGIN);
            }
            while Instant::now() < self.frame_deadline {}
        }
        self.frame_deadline += self.target_frame_time;
        let now = Instant::now();
        if now > self.frame_deadline {
            self.frame_deadline = now + self.target_frame_time;
        }
        // ── Delta time ─────────────────────────────────────────────────────────
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;

        self.fps_frame_count += 1;
        let elapsed = self.fps_timer.elapsed();
        if elapsed.as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frame_count as f32 / elapsed.as_secs_f32();
            self.window.set_title(&format!("engine | {:.0} FPS", self.last_fps));
            self.fps_frame_count = 0;
            self.fps_timer = Instant::now();
        }

        crate::game::update(&mut self.world, &mut self.camera, dt);
        log::info!("{:?}", self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
        let vp = OPENGL_TO_WGPU_MATRIX * self.camera.build_view_projection_matrix();
        self.frustum = Frustum::from_vp(&vp);

        // Collect the first PointLight from ECS and upload to GPU.
        // If a Flashlight component is present and disabled, zero out intensity.
        let light_uniform = self.world
            .query::<(&Position, &PointLight, Option<&Flashlight>)>()
            .iter()
            .next()
            .map(|(_, (pos, light, flashlight))| {
                let on = flashlight.map(|f| f.enabled).unwrap_or(true);
                LightUniform {
                    position:    [pos.0.x, pos.0.y, pos.0.z],
                    intensity:   if on { light.intensity } else { 0.0 },
                    color:       [light.color.x, light.color.y, light.color.z],
                    lighting_on: if self.lighting_enabled { 1.0 } else { 0.0 },
                }
            })
            .unwrap_or(LightUniform {
                position:    [0.0, 0.0, 0.0],
                intensity:   0.0,
                color:       [1.0, 1.0, 1.0],
                lighting_on: if self.lighting_enabled { 1.0 } else { 0.0 },
            });
        self.queue.write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&[light_uniform]));

        log::info!("{:?}", self.camera_uniform);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // Recompute and upload directional light + light-space matrix each frame
        // so that egui slider changes take effect immediately.
        let (dir, dir_uniform) = self.world
            .query::<&DirectionalLight>()
            .iter()
            .next()
            .map(|(_, dl)| {
                let uniform = DirLightUniform {
                    direction: [dl.direction.x, dl.direction.y, dl.direction.z],
                    intensity: dl.intensity,
                    color:     [dl.color.x, dl.color.y, dl.color.z],
                    _pad:      0.0,
                };
                (dl.direction, uniform)
            })
            .unwrap_or_else(|| {
                let d = cgmath::vec3(0.4, 1.0, 0.3);
                let u = DirLightUniform {
                    direction: [d.x, d.y, d.z],
                    intensity: 0.0,
                    color:     [1.0, 1.0, 1.0],
                    _pad:      0.0,
                };
                (d, u)
            });
        self.queue.write_buffer(&self.dir_light_buffer, 0, bytemuck::cast_slice(&[dir_uniform]));

        let ls = LightSpaceUniform {
            matrix: Self::compute_light_space_matrix(dir).into(),
        };
        self.queue.write_buffer(
            &self.light_space_buffer, 0,
            bytemuck::cast_slice(&[ls]),
        );

        for (tag, entry) in &mut self.models {
            let instance_data = build_instance_data_for(&self.world, tag);
            let count = instance_data.len() as u32;

            // Grow the buffer when the entity count exceeds current capacity.
            if count > entry.instance_capacity {
                let new_cap = count.next_power_of_two();
                entry.instance_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(tag.as_str()),
                    size: (new_cap as usize * std::mem::size_of::<InstanceRaw>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                entry.instance_capacity = new_cap;
            }

            entry.instance_count = count;
            if !instance_data.is_empty() {
                self.queue.write_buffer(
                    &entry.instance_buffer,
                    0,
                    bytemuck::cast_slice(&instance_data),
                );
            }
        }
    }

    pub fn render(&mut self) -> anyhow::Result<()> {
        if !self.is_surface_configured {
            return Ok(());
        }

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                drop(surface_texture);
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                anyhow::bail!("Lost device");
            }
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // ── Pass 1: shadow map ─────────────────────────────────────────────────
        {
            let ts = self.timestamp_query_set.as_ref().map(|qs| wgpu::RenderPassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index:       Some(1),
            });
            let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow_pass"),
                color_attachments: &[],   // depth only
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_render_view,
                    depth_ops: Some(wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes:    ts,
                multiview_mask:      None,
            });
            shadow_pass.set_pipeline(&self.shadow_pipeline);
            shadow_pass.set_bind_group(0, &self.shadow_pass_bg, &[]);

            for entry in self.models.values() {
                shadow_pass.set_vertex_buffer(1, entry.instance_buffer.slice(..));
                let instances = 0..entry.instance_count;
                for mesh in &entry.model.meshes {
                    shadow_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    shadow_pass.set_index_buffer(
                        mesh.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    shadow_pass.draw_indexed(0..mesh.num_elements, 0, instances.clone());
                }
            }
        }

        // ── Pass 2: main scene ─────────────────────────────────────────────────
        {
            let ts = self.timestamp_query_set.as_ref().map(|qs| wgpu::RenderPassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(2),
                end_of_pass_write_index:       Some(3),
            });
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: ts,
                multiview_mask: None,
            });

            let pipeline = if self.wireframe { &self.wireframe_pipeline } else { &self.render_pipeline };
            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(2, &self.light_bind_group, &[]);
            render_pass.set_bind_group(3, &self.dir_light_bind_group, &[]);
            render_pass.set_bind_group(4, &self.shadow_main_bg, &[]);

            let frustum = &self.frustum;
            let mut drawn = 0u32;
            let mut culled = 0u32;

            for entry in self.models.values() {
                render_pass.set_vertex_buffer(1, entry.instance_buffer.slice(..));
                let instances = 0..entry.instance_count;

                for mesh in &entry.model.meshes {
                    let (center, radius) = mesh.bounding_sphere;
                    if frustum.cull_sphere(center, radius) {
                        culled += 1;
                        continue;
                    }
                    if let Some(material) = entry.model.materials.get(mesh.material) {
                        render_pass.draw_mesh_instanced(
                            mesh,
                            material,
                            instances.clone(),
                            &self.camera_bind_group,
                        );
                        drawn += 1;
                    }
                }
            }
            log::debug!("draw calls: {drawn} drawn, {culled} culled");
        }

        // ── Depth debug pass (optional) ────────────────────────────────────────
        if self.show_depth {
            let mut depth_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("depth_debug_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            depth_pass.set_pipeline(&self.depth_debug_pipeline);
            depth_pass.set_bind_group(0, &self.depth_debug_bind_group, &[]);
            depth_pass.draw(0..3, 0..1);
        }

        // ── egui pass ─────────────────────────────────────────────────────────
        let raw_input = self.egui_winit.take_egui_input(&self.window);

        // Read mutable ECS state before the egui closure.
        let fps = self.last_fps;
        let gpu_ms = self.last_gpu_ms;
        let eye = self.camera.eye;
        let ui_mode = self.ui_mode;
        let mut lighting_enabled = self.lighting_enabled;
        let mut show_depth = self.show_depth;
        let mut wireframe = self.wireframe;
        let mut fovy = self.camera.fovy;

        let mut flashlight_on = self.world
            .query::<&crate::game::Flashlight>()
            .iter()
            .next()
            .map(|(_, f)| f.enabled)
            .unwrap_or(false);
        let mut flashlight_intensity = self.world
            .query::<&crate::game::PointLight>()
            .iter()
            .next()
            .map(|(_, l)| l.intensity)
            .unwrap_or(200.0);

        let (mut sun_dir, mut sun_intensity, mut sun_color) = self.world
            .query::<&DirectionalLight>()
            .iter()
            .next()
            .map(|(_, dl)| {
                let d = dl.direction;
                let c = dl.color;
                (
                    [d.x, d.y, d.z],
                    dl.intensity,
                    [c.x, c.y, c.z],
                )
            })
            .unwrap_or(([0.4, 1.0, 0.3], 1.5, [1.0, 0.95, 0.85]));

        let mut cam_speed = self.world
            .query::<&mut crate::game::CameraController>()
            .iter()
            .next()
            .map(|(_, c)| c.speed)
            .unwrap_or(1000.0);

        // Build UI using begin_pass / end_pass so we can use &Context directly.
        self.egui_ctx.begin_pass(raw_input);
        egui::Window::new("Debug")
            .default_pos([10.0, 10.0])
            .resizable(true)
            .show(&self.egui_ctx, |ui| {
                ui.label(format!("FPS: {fps:.0}"));
                if gpu_ms > 0.0 {
                    ui.label(format!("GPU: {gpu_ms:.2} ms"));
                }
                ui.label(format!("Camera: ({:.1}, {:.1}, {:.1})", eye.x, eye.y, eye.z));

                ui.separator();
                ui.heading("Render");
                let lighting_label = if lighting_enabled { "Lighting: ON" } else { "Lighting: OFF" };
                if ui.button(lighting_label).clicked() { lighting_enabled = !lighting_enabled; }
                let depth_label = if show_depth { "Depth Map: ON" } else { "Depth Map: OFF" };
                if ui.button(depth_label).clicked() { show_depth = !show_depth; }
                let wire_label = if wireframe { "Wireframe: ON" } else { "Wireframe: OFF" };
                if ui.button(wire_label).clicked() { wireframe = !wireframe; }

                ui.separator();
                ui.heading("Camera");
                ui.add(egui::Slider::new(&mut fovy, 10.0..=120.0).text("FOV"));
                ui.add(egui::Slider::new(&mut cam_speed, 10.0..=10000.0).text("Speed").logarithmic(true));

                ui.separator();
                ui.heading("Flashlight");
                ui.checkbox(&mut flashlight_on, "Enabled");
                ui.add(egui::Slider::new(&mut flashlight_intensity, 0.0..=2000.0).text("Intensity").logarithmic(true));

                ui.separator();
                ui.heading("Sun");
                ui.add(egui::Slider::new(&mut sun_dir[0], -1.0..=1.0).text("Dir X"));
                ui.add(egui::Slider::new(&mut sun_dir[1], -1.0..=1.0).text("Dir Y"));
                ui.add(egui::Slider::new(&mut sun_dir[2], -1.0..=1.0).text("Dir Z"));
                ui.add(egui::Slider::new(&mut sun_intensity, 0.0..=5.0).text("Intensity"));
                ui.horizontal(|ui| {
                    ui.label("Color");
                    egui::color_picker::color_edit_button_rgb(ui, &mut sun_color);
                });

                ui.separator();
                ui.label(if ui_mode { "Tab — return to game" } else { "Tab — open UI" });
            });
        let full_output = self.egui_ctx.end_pass();

        // Write back changed values.
        self.lighting_enabled = lighting_enabled;
        self.show_depth = show_depth;
        self.wireframe = wireframe;
        self.camera.fovy = fovy;

        for (_, f) in self.world.query_mut::<&mut crate::game::Flashlight>() {
            f.enabled = flashlight_on;
        }
        for (_, l) in self.world.query_mut::<&mut crate::game::PointLight>() {
            l.intensity = flashlight_intensity;
        }
        for (_, dl) in self.world.query_mut::<&mut DirectionalLight>() {
            dl.direction = cgmath::vec3(sun_dir[0], sun_dir[1], sun_dir[2]);
            dl.intensity = sun_intensity;
            dl.color     = cgmath::vec3(sun_color[0], sun_color[1], sun_color[2]);
        }
        for (_, c) in self.world.query_mut::<&mut crate::game::CameraController>() {
            c.speed = cam_speed;
        }

        self.egui_winit
            .handle_platform_output(&self.window, full_output.platform_output);

        let tris = self.egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }

        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: full_output.pixels_per_point,
        };
        // update_buffers may produce extra command buffers for texture uploads.
        let extra_cmds = self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &tris,
            &screen_descriptor,
        );

        {
            // forget_lifetime() converts RenderPass<'enc> → RenderPass<'static>
            // so egui_renderer.render() (which requires 'static) can accept it.
            let mut egui_pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            self.egui_renderer.render(&mut egui_pass, &tris, &screen_descriptor);
        }

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        // ─────────────────────────────────────────────────────────────────────

        // Resolve timestamp queries into the resolve buffer, then copy to the
        // CPU-readable buffer — both must happen inside the encoder before finish().
        if let (Some(qs), Some(resolve_buf), Some(read_buf)) = (
            &self.timestamp_query_set,
            &self.timestamp_resolve_buf,
            &self.timestamp_read_buf,
        ) {
            encoder.resolve_query_set(qs, 0..4, resolve_buf, 0);
            encoder.copy_buffer_to_buffer(resolve_buf, 0, read_buf, 0, 4 * 8);
        }

        self.queue.submit(extra_cmds.into_iter().chain(iter::once(encoder.finish())));
        output.present();

        // Map the read buffer, block until the GPU has written the timestamps,
        // compute the frame's GPU time, then immediately unmap.
        if let Some(ref read_buf) = self.timestamp_read_buf {
            let slice = read_buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            {
                let view = slice.get_mapped_range();
                let ts: &[u64] = bytemuck::cast_slice(&*view);
                // ts[0]=shadow_begin ts[1]=shadow_end ts[2]=main_begin ts[3]=main_end
                if ts.len() >= 4 && ts[3] >= ts[0] {
                    let diff_ns = (ts[3] - ts[0]) as f64 * self.timestamp_period as f64;
                    self.last_gpu_ms = (diff_ns / 1_000_000.0) as f32;
                }
            }
            read_buf.unmap();
        }

        Ok(())
    }
}
