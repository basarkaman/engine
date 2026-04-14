// ─── Uniforms ────────────────────────────────────────────────────────────────

struct Camera {
    view_pos:  vec4<f32>,
    view_proj: mat4x4<f32>,
}
@group(1) @binding(0)
var<uniform> camera: Camera;

struct Light {
    position:   vec3<f32>,
    intensity:  f32,
    color:      vec3<f32>,
    lighting_on: f32,   // 1.0 = normal lighting, 0.0 = fullbright
}
@group(2) @binding(0)
var<uniform> light: Light;

struct DirLight {
    direction: vec3<f32>,
    intensity: f32,
    color:     vec3<f32>,
    _pad:      f32,
}
@group(3) @binding(0)
var<uniform> dir_light: DirLight;

// ─── Shadow map (group 4) ─────────────────────────────────────────────────────

struct LightSpace {
    matrix: mat4x4<f32>,
}
@group(4) @binding(0) var<uniform> light_space: LightSpace;
@group(4) @binding(1) var t_shadow: texture_depth_2d;
@group(4) @binding(2) var s_shadow: sampler_comparison;

// ─── Vertex stage ─────────────────────────────────────────────────────────────

struct VertexInput {
    @location(0) position:   vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal:     vec3<f32>,
    @location(3) tangent:    vec4<f32>,  // xyz = tangent, w = bitangent sign
}

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    @location(9)  normal_matrix_0: vec3<f32>,
    @location(10) normal_matrix_1: vec3<f32>,
    @location(11) normal_matrix_2: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords:     vec2<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) world_position: vec3<f32>,
    @location(3) world_tangent:  vec4<f32>,  // xyz = tangent, w = bitangent sign
}

@vertex
fn vs_main(model: VertexInput, instance: InstanceInput) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    let normal_matrix = mat3x3<f32>(
        instance.normal_matrix_0,
        instance.normal_matrix_1,
        instance.normal_matrix_2,
    );

    let world_pos = model_matrix * vec4<f32>(model.position, 1.0);

    var out: VertexOutput;
    out.tex_coords     = model.tex_coords;
    out.world_normal   = normalize(normal_matrix * model.normal);
    out.world_tangent  = vec4<f32>(normalize(normal_matrix * model.tangent.xyz), model.tangent.w);
    out.world_position = world_pos.xyz;
    out.clip_position  = camera.view_proj * world_pos;
    return out;
}

// ─── Fragment stage ───────────────────────────────────────────────────────────

@group(0) @binding(0) var t_diffuse: texture_2d<f32>;
@group(0) @binding(1) var s_diffuse: sampler;
@group(0) @binding(2) var t_normal:  texture_2d<f32>;
@group(0) @binding(3) var s_normal:  sampler;

// Returns 1.0 when the fragment is fully lit, 0.0 when fully in shadow.
// Uses a 3×3 PCF kernel to soften shadow edges.
fn shadow_factor(world_pos: vec3<f32>) -> f32 {
    let ls   = light_space.matrix * vec4<f32>(world_pos, 1.0);
    let proj = ls.xyz / ls.w;

    let uv = proj.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);

    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z > 1.0 {
        return 1.0;
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let object_color = textureSample(t_diffuse, s_diffuse, in.tex_coords);

    // ── Normal mapping ────────────────────────────────────────────────────────
    let n_sample  = textureSample(t_normal, s_normal, in.tex_coords).xyz;
    let n_tangent = n_sample * 2.0 - 1.0;

    let N = normalize(in.world_normal);
    let T = normalize(in.world_tangent.xyz);
    let B = in.world_tangent.w * cross(N, T);
    let normal = normalize(mat3x3<f32>(T, B, N) * n_tangent);

    let view_dir  = normalize(camera.view_pos.xyz - in.world_position);

    // ── Point light ───────────────────────────────────────────────────────────
    let light_dir = normalize(light.position - in.world_position);
    let half_dir  = normalize(view_dir + light_dir);

    let ambient  = light.color * 0.05;
    let diffuse  = light.color * max(dot(normal, light_dir), 0.0);
    let specular = light.color * pow(max(dot(normal, half_dir), 0.0), 32.0);

    let dist        = length(light.position - in.world_position);
    let attenuation = light.intensity / (dist * dist + light.intensity);

    let point_contrib = ambient + (diffuse + specular) * attenuation;

    // ── Directional light (sun) ───────────────────────────────────────────────
    let sun_dir      = normalize(dir_light.direction);
    let sun_diffuse  = dir_light.color * max(dot(normal, sun_dir), 0.0);
    let sun_half     = normalize(view_dir + sun_dir);
    let sun_specular = dir_light.color * pow(max(dot(normal, sun_half), 0.0), 32.0);
    let sun_contrib  = (sun_diffuse + sun_specular) * dir_light.intensity;

    // Fullbright mode: skip all lighting, return raw diffuse.
    if light.lighting_on < 0.5 {
        return object_color;
    }

    // Sun contribution is attenuated by the shadow map; point light is not.
    let shadow = shadow_factor(in.world_position);
    let result = (point_contrib + sun_contrib * shadow) * object_color.xyz;
    return vec4<f32>(result, object_color.a);
}
