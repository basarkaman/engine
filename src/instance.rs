use std::mem;

use hecs::World;

use crate::game::{ModelTag, Position, Rotation};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    pub model:  [[f32; 4]; 4],  // 64 bytes — model transform
    pub normal: [[f32; 3]; 3],  // 36 bytes — rotation-only matrix for normals
}

impl InstanceRaw {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // model matrix — 4 × vec4 at locations 5-8
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // normal matrix — 3 × vec3 at locations 9-11
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 19]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 22]>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// Builds instance transforms for all entities with the given ModelTag.
pub fn build_instance_data_for(world: &World, tag: &str) -> Vec<InstanceRaw> {
    world
        .query::<(&Position, &Rotation, &ModelTag)>()
        .iter()
        .filter(|(_, (_, _, t))| t.0 == tag)
        .map(|(_, (pos, rot, _))| {
            let model  = cgmath::Matrix4::from_translation(pos.0) * cgmath::Matrix4::from(rot.0);
            let normal = cgmath::Matrix3::from(rot.0);
            InstanceRaw {
                model:  model.into(),
                normal: normal.into(),
            }
        })
        .collect()
}
