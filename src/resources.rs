use std::io::{BufReader, Cursor};

use wgpu::util::DeviceExt;

use crate::{model, texture};

// ── Module-level helpers ──────────────────────────────────────────────────────

/// Compute a bounding sphere (center, radius) from a slice of positions.
/// Uses two passes: one to find the centroid, one to find the max distance.
fn bounding_sphere(positions: &[[f32; 3]]) -> (cgmath::Point3<f32>, f32) {
    if positions.is_empty() {
        return (cgmath::Point3::new(0.0, 0.0, 0.0), 0.0);
    }
    let n = positions.len() as f32;
    // Single fold to sum all three axes at once.
    let (sx, sy, sz) = positions
        .iter()
        .fold((0.0f32, 0.0f32, 0.0f32), |(sx, sy, sz), p| {
            (sx + p[0], sy + p[1], sz + p[2])
        });
    let (cx, cy, cz) = (sx / n, sy / n, sz / n);
    let center = cgmath::Point3::new(cx, cy, cz);
    let radius = positions
        .iter()
        .map(|p| {
            let d = [p[0] - cx, p[1] - cy, p[2] - cz];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        })
        .fold(0.0f32, f32::max);
    (center, radius)
}

/// Build the path to a resource file under `res/` next to the executable.
/// This works both during `cargo run` (res/ is copied to target/{profile}/)
/// and when the binary is distributed to another machine.
fn res_path(file_name: &str) -> std::path::PathBuf {
    let mut path = std::env::current_exe()
        .expect("cannot determine executable path")
        .parent()
        .expect("executable has no parent directory")
        .to_path_buf();
    path.push("res");
    // Split on both / and \ so ModelTag works with either separator.
    for part in file_name.split(['/', '\\']) {
        if !part.is_empty() {
            path.push(part);
        }
    }
    println!("[res] {}", path.display());
    path
}

/// Create a material bind group with diffuse (bindings 0–1) and normal map (bindings 2–3).
fn create_material_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    diffuse: &texture::Texture,
    normal: &texture::Texture,
    label: Option<&str>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&diffuse.view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&diffuse.sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&normal.view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&normal.sampler) },
        ],
        label,
    })
}

/// Create a 1×1 solid-colour texture (sRGB).
fn solid_color_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rgba: [u8; 4],
    label: &str,
) -> anyhow::Result<texture::Texture> {
    let img = image::DynamicImage::ImageRgba8(
        image::RgbaImage::from_pixel(1, 1, image::Rgba(rgba)),
    );
    texture::Texture::from_image(device, queue, &img, Some(label))
}

/// Create a 1×1 flat normal map (linear RGB = 128, 128, 255).
/// Decodes to tangent-space (0, 0, 1) — no surface perturbation.
fn flat_normal_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
) -> anyhow::Result<texture::Texture> {
    let img = image::DynamicImage::ImageRgba8(
        image::RgbaImage::from_pixel(1, 1, image::Rgba([128, 128, 255, 255])),
    );
    texture::Texture::from_image_linear(device, queue, &img, Some(label))
}

/// Convert a GLTF image buffer to a wgpu Texture, normalising all formats to RGBA8.
/// Pass `linear = true` for normal maps (Rgba8Unorm); `false` for colour data (Rgba8UnormSrgb).
fn gltf_image_to_texture(
    data: &gltf::image::Data,
    label: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    linear: bool,
) -> anyhow::Result<texture::Texture> {
    use gltf::image::Format;

    let rgba: Vec<u8> = match data.format {
        Format::R8       => data.pixels.iter().flat_map(|&r| [r, r, r, 255]).collect(),
        Format::R8G8     => data.pixels.chunks(2).flat_map(|c| [c[0], c[1], 0, 255]).collect(),
        Format::R8G8B8   => data.pixels.chunks(3).flat_map(|c| [c[0], c[1], c[2], 255]).collect(),
        Format::R8G8B8A8 => data.pixels.clone(),
        // 16-bit: shift the high byte down to get an 8-bit approximation.
        Format::R16 => data.pixels.chunks(2).flat_map(|c| {
            let v = (u16::from_le_bytes([c[0], c[1]]) >> 8) as u8;
            [v, v, v, 255]
        }).collect(),
        Format::R16G16 => data.pixels.chunks(4).flat_map(|c| {
            let r = (u16::from_le_bytes([c[0], c[1]]) >> 8) as u8;
            let g = (u16::from_le_bytes([c[2], c[3]]) >> 8) as u8;
            [r, g, 0, 255]
        }).collect(),
        Format::R16G16B16 => data.pixels.chunks(6).flat_map(|c| {
            let r = (u16::from_le_bytes([c[0], c[1]]) >> 8) as u8;
            let g = (u16::from_le_bytes([c[2], c[3]]) >> 8) as u8;
            let b = (u16::from_le_bytes([c[4], c[5]]) >> 8) as u8;
            [r, g, b, 255]
        }).collect(),
        Format::R16G16B16A16 => data.pixels.chunks(8).flat_map(|c| {
            let r = (u16::from_le_bytes([c[0], c[1]]) >> 8) as u8;
            let g = (u16::from_le_bytes([c[2], c[3]]) >> 8) as u8;
            let b = (u16::from_le_bytes([c[4], c[5]]) >> 8) as u8;
            let a = (u16::from_le_bytes([c[6], c[7]]) >> 8) as u8;
            [r, g, b, a]
        }).collect(),
        // 32-bit float: clamp to [0,1] then scale to 8-bit.
        Format::R32G32B32FLOAT => data.pixels.chunks(12).flat_map(|c| {
            let r = (f32::from_le_bytes([c[0],c[1],c[2],c[3]]).clamp(0.0,1.0)*255.0) as u8;
            let g = (f32::from_le_bytes([c[4],c[5],c[6],c[7]]).clamp(0.0,1.0)*255.0) as u8;
            let b = (f32::from_le_bytes([c[8],c[9],c[10],c[11]]).clamp(0.0,1.0)*255.0) as u8;
            [r, g, b, 255]
        }).collect(),
        Format::R32G32B32A32FLOAT => data.pixels.chunks(16).flat_map(|c| {
            let r = (f32::from_le_bytes([c[0],c[1],c[2],c[3]]).clamp(0.0,1.0)*255.0) as u8;
            let g = (f32::from_le_bytes([c[4],c[5],c[6],c[7]]).clamp(0.0,1.0)*255.0) as u8;
            let b = (f32::from_le_bytes([c[8],c[9],c[10],c[11]]).clamp(0.0,1.0)*255.0) as u8;
            let a = (f32::from_le_bytes([c[12],c[13],c[14],c[15]]).clamp(0.0,1.0)*255.0) as u8;
            [r, g, b, a]
        }).collect(),
    };

    let img = image::DynamicImage::ImageRgba8(
        image::RgbaImage::from_raw(data.width, data.height, rgba)
            .ok_or_else(|| anyhow::anyhow!("Failed to create image buffer for {}", label))?,
    );
    if linear {
        texture::Texture::from_image_linear(device, queue, &img, Some(label))
    } else {
        texture::Texture::from_image(device, queue, &img, Some(label))
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

pub async fn load_string(file_name: &str) -> anyhow::Result<String> {
    Ok(std::fs::read_to_string(res_path(file_name))?)
}

pub async fn load_binary(file_name: &str) -> anyhow::Result<Vec<u8>> {
    Ok(std::fs::read(res_path(file_name))?)
}

pub async fn load_texture(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<texture::Texture> {
    let data = load_binary(file_name).await?;
    texture::Texture::from_bytes(device, queue, &data, file_name)
}

/// Dispatch to the correct loader based on file extension.
pub async fn load_model(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<model::Model> {
    let ext = std::path::Path::new(file_name)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "gltf" || ext == "glb" {
        return load_gltf(file_name, device, queue, layout).await;
    }
    load_obj(file_name, device, queue, layout).await
}

// ── OBJ loader ───────────────────────────────────────────────────────────────

async fn load_obj(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<model::Model> {
    // Resolve asset paths relative to the OBJ's own directory.
    // e.g. "de_dust2/generic.obj" → base = "de_dust2/"
    let base_dir = std::path::Path::new(file_name)
        .parent()
        .unwrap_or(std::path::Path::new(""))
        .to_string_lossy()
        .to_string();

    // Resolve a name relative to the model's directory.
    // If the name is an absolute path (e.g. from another machine's MTL export),
    // strip it down to just the filename so we can find it in base_dir.
    let in_dir = |name: &str| -> String {
        let bare = std::path::Path::new(name)
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| name.to_string());
        if base_dir.is_empty() { bare } else { format!("{}/{}", base_dir, bare) }
    };

    let obj_text = load_string(file_name).await?;

    // Rewrite "mtllib C:\...\foo.mtl" lines to "mtllib foo.mtl".
    // OBJ files exported from other machines often embed absolute paths that
    // don't exist on the current machine.
    let obj_text = obj_text
        .lines()
        .map(|line| {
            if line.starts_with("mtllib ") {
                let tokens: Vec<&str> = line["mtllib ".len()..].split_whitespace().collect();
                let mtl_file = tokens
                    .iter()
                    .rev()
                    .find(|t| t.to_lowercase().ends_with(".mtl"))
                    .map(|t| {
                        std::path::Path::new(t)
                            .file_name()
                            .map(|f| f.to_string_lossy().to_string())
                            .unwrap_or_else(|| t.to_string())
                    })
                    .unwrap_or_else(|| line["mtllib ".len()..].to_string());
                format!("mtllib {}", mtl_file)
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    let obj_cursor = Cursor::new(obj_text);
    let mut obj_reader = BufReader::new(obj_cursor);

    let (models, obj_materials) = tobj::load_obj_buf_async(
        &mut obj_reader,
        &tobj::LoadOptions { triangulate: true, single_index: true, ..Default::default() },
        |p| {
            let path = in_dir(&p);
            async move {
                match load_string(&path).await {
                    Ok(mat_text) => tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text))),
                    Err(_) => Ok((Vec::new(), Default::default())),
                }
            }
        },
    )
    .await?;

    let mut materials = Vec::new();
    for m in obj_materials? {
        let diffuse_texture = load_texture(&in_dir(&m.diffuse_texture), device, queue).await?;
        let normal_texture  = flat_normal_texture(device, queue, &format!("{}_flat_normal", m.name))?;
        let bind_group = create_material_bind_group(device, layout, &diffuse_texture, &normal_texture, None);
        materials.push(model::Material { name: m.name, diffuse_texture, normal_texture, bind_group });
    }

    // Merge all tobj meshes that share the same material into one GPU buffer.
    // This reduces draw calls from O(groups) to O(unique materials).
    let mut merged: std::collections::HashMap<usize, (Vec<model::ModelVertex>, Vec<u32>)> =
        std::collections::HashMap::new();

    for m in &models {
        let mat_id = m.mesh.material_id.unwrap_or(0);
        let (verts, indices) = merged.entry(mat_id).or_default();
        let base = verts.len() as u32;
        for i in 0..m.mesh.positions.len() / 3 {
            verts.push(model::ModelVertex {
                position:   [
                    m.mesh.positions[i * 3],
                    m.mesh.positions[i * 3 + 1],
                    m.mesh.positions[i * 3 + 2],
                ],
                tex_coords: [m.mesh.texcoords[i * 2], 1.0 - m.mesh.texcoords[i * 2 + 1]],
                normal:     if m.mesh.normals.is_empty() {
                    [0.0, 0.0, 0.0]
                } else {
                    [
                        m.mesh.normals[i * 3],
                        m.mesh.normals[i * 3 + 1],
                        m.mesh.normals[i * 3 + 2],
                    ]
                },
                tangent: [1.0, 0.0, 0.0, 1.0],
            });
        }
        indices.extend(m.mesh.indices.iter().map(|idx| idx + base));
    }

    let mut meshes: Vec<model::Mesh> = merged
        .into_iter()
        .map(|(mat_id, (vertices, indices))| {
            let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some(&format!("{} Vertex Buffer mat{}", file_name, mat_id)),
                contents: bytemuck::cast_slice(&vertices),
                usage:    wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some(&format!("{} Index Buffer mat{}", file_name, mat_id)),
                contents: bytemuck::cast_slice(&indices),
                usage:    wgpu::BufferUsages::INDEX,
            });
            model::Mesh {
                name: file_name.to_string(),
                vertex_buffer,
                index_buffer,
                num_elements: indices.len() as u32,
                material: mat_id,
                bounding_sphere: bounding_sphere(&positions),
            }
        })
        .collect();

    // Stable sort so rendering order is deterministic.
    meshes.sort_by_key(|m| m.material);

    println!("Loaded '{}': {} draw calls ({} source groups)", file_name, meshes.len(), models.len());
    Ok(model::Model { meshes, materials })
}

// ── GLTF/GLB loader ──────────────────────────────────────────────────────────

async fn load_gltf(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<model::Model> {
    let path = res_path(file_name);
    let (document, buffers, images) = gltf::import(&path)?;

    println!("GLTF images in file: {}", images.len());
    for (i, img) in images.iter().enumerate() {
        println!("  image[{}]: {:?} {}x{}", i, img.format, img.width, img.height);
    }

    let wrap = |mode: gltf::texture::WrappingMode| match mode {
        gltf::texture::WrappingMode::ClampToEdge    => wgpu::AddressMode::ClampToEdge,
        gltf::texture::WrappingMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
        gltf::texture::WrappingMode::Repeat         => wgpu::AddressMode::Repeat,
    };

    // Build materials.
    let mut materials: Vec<model::Material> = Vec::new();
    for mat in document.materials() {
        let label = mat.name().unwrap_or("gltf_material").to_string();
        let pbr = mat.pbr_metallic_roughness();

        let mut diffuse_texture = if let Some(tex_info) = pbr.base_color_texture() {
            let src_index = tex_info.texture().source().index();
            let mut tex = gltf_image_to_texture(&images[src_index], &label, device, queue, false)?;
            let s = tex_info.texture().sampler();
            tex.sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wrap(s.wrap_s()),
                address_mode_v: wrap(s.wrap_t()),
                address_mode_w: wgpu::AddressMode::Repeat,
                mag_filter:    wgpu::FilterMode::Linear,
                min_filter:    wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            });
            tex
        } else {
            let [r, g, b, a] = pbr.base_color_factor();
            solid_color_texture(device, queue, [
                (r * 255.0) as u8,
                (g * 255.0) as u8,
                (b * 255.0) as u8,
                (a * 255.0) as u8,
            ], &label)?
        };
        // Ensure diffuse sampler uses Repeat by default (glTF spec).
        let _ = &mut diffuse_texture;

        let normal_texture = if let Some(norm_info) = mat.normal_texture() {
            let src_index = norm_info.texture().source().index();
            let normal_label = format!("{}_normal", label);
            let mut tex = gltf_image_to_texture(&images[src_index], &normal_label, device, queue, true)?;
            let s = norm_info.texture().sampler();
            tex.sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wrap(s.wrap_s()),
                address_mode_v: wrap(s.wrap_t()),
                address_mode_w: wgpu::AddressMode::Repeat,
                mag_filter:    wgpu::FilterMode::Linear,
                min_filter:    wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            });
            tex
        } else {
            flat_normal_texture(device, queue, &format!("{}_flat_normal", label))?
        };

        let bind_group = create_material_bind_group(device, layout, &diffuse_texture, &normal_texture, Some(&label));
        materials.push(model::Material { name: label, diffuse_texture, normal_texture, bind_group });
    }

    // Fallback: if the file has no materials at all, use plain white + flat normal.
    if materials.is_empty() {
        let diffuse_texture = solid_color_texture(device, queue, [255, 255, 255, 255], "gltf_default")?;
        let normal_texture  = flat_normal_texture(device, queue, "gltf_default_normal")?;
        let bind_group = create_material_bind_group(device, layout, &diffuse_texture, &normal_texture, Some("gltf_default"));
        materials.push(model::Material { name: "default".to_string(), diffuse_texture, normal_texture, bind_group });
    }

    // Build meshes from every primitive in every mesh node.
    let mut meshes: Vec<model::Mesh> = Vec::new();
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buf| Some(buffers[buf.index()].0.as_slice()));

            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .map(|it| it.collect())
                .unwrap_or_default();
            if positions.is_empty() {
                continue;
            }

            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|it| it.collect())
                .unwrap_or_else(|| vec![[0.0, 0.0, 0.0]; positions.len()]);

            let tex_coords: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|tc| tc.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

            let tangents: Vec<[f32; 4]> = reader
                .read_tangents()
                .map(|it| it.collect())
                .unwrap_or_else(|| vec![[1.0, 0.0, 0.0, 1.0]; positions.len()]);

            let indices: Vec<u32> = reader
                .read_indices()
                .map(|it| it.into_u32().collect())
                .unwrap_or_else(|| (0..positions.len() as u32).collect());

            // glTF is already Y-up — no axis conversion needed.
            let vertices: Vec<model::ModelVertex> = (0..positions.len())
                .map(|i| {
                    model::ModelVertex {
                        position:   positions[i],
                        tex_coords: tex_coords[i],
                        normal:     normals[i],
                        tangent:    tangents[i],
                    }
                })
                .collect();

            // Compute bounding sphere from the transformed positions directly.
            let transformed_positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
            let bs = bounding_sphere(&transformed_positions);

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some(&format!("{} GLTF Vertex Buffer", file_name)),
                contents: bytemuck::cast_slice(&vertices),
                usage:    wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some(&format!("{} GLTF Index Buffer", file_name)),
                contents: bytemuck::cast_slice(&indices),
                usage:    wgpu::BufferUsages::INDEX,
            });

            let material = primitive.material().index().unwrap_or(0).min(materials.len() - 1);
            meshes.push(model::Mesh {
                name: mesh.name().unwrap_or(file_name).to_string(),
                vertex_buffer,
                index_buffer,
                num_elements: indices.len() as u32,
                material,
                bounding_sphere: bs,
            });
        }
    }

    println!("Loaded GLTF '{}': {} meshes, {} materials", file_name, meshes.len(), materials.len());
    Ok(model::Model { meshes, materials })
}
