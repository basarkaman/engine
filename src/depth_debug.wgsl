// Fullscreen triangle — no vertex buffer needed.
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    // Single triangle that covers the entire NDC clip space.
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
    let px    = vec2<i32>(i32(frag_pos.x), i32(frag_pos.y));
    let depth = textureLoad(t_depth, px, 0);

    // Reconstruct linear eye-space Z from NDC depth (znear=0.1, zfar=1e6).
    let znear = 0.1;
    let zfar  = 1000000.0;
    let z_eye = znear * zfar / (zfar - depth * (zfar - znear));

    // Log scale: spreads the [znear, zfar] range perceptually.
    // Result: near objects → white, far objects / empty sky → black.
    let log_d = log2(z_eye / znear) / log2(zfar / znear);
    let d = 1.0 - clamp(log_d, 0.0, 1.0);

    return vec4<f32>(d, d, d, 1.0);
}
