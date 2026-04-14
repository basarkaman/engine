// ─── Shadow pass — depth only ─────────────────────────────────────────────────
// Renders the scene from the directional light's point of view to fill
// the shadow map.  No colour attachment; only depth is written.

struct LightSpace {
    matrix: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> light_space: LightSpace;

// Only the attributes this shader actually reads need to be declared.
// The pipeline still supplies the full ModelVertex + InstanceRaw buffers.
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
