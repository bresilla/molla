use crate::draw::{Camera, Triangle};
use eframe::{egui, egui_wgpu, wgpu};
use wgpu::util::DeviceExt;

#[path = "studio_vapor.rs"]
mod vapor;
pub use vapor::paint as paint_vapor;

#[path = "studio_spray.rs"]
mod spray;
pub use spray::paint as paint_spray;

#[allow(dead_code)]
#[path = "studio_plume.rs"]
mod plume;
#[allow(unused_imports)]
pub use plume::{PlumeBillboard, paint as paint_plume, paint_slot as paint_plume_slot};

#[allow(dead_code)]
#[path = "drone_gltf.rs"]
pub(crate) mod model_asset;

#[path = "studio_foliage.rs"]
mod foliage;

#[path = "studio_leaves.rs"]
pub(crate) mod leaves;
pub(crate) use leaves::LeafInstance;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 4],
    uv: [f32; 2],
}

fn copy_to_mapped(destination: &mut wgpu::BufferViewMut, source: &[u8]) {
    assert_eq!(destination.len(), source.len());
    let (blocks, tail) = source.as_chunks::<256>();
    let (mapped, mut remainder) = destination.slice(..).into_chunks::<256>();
    for (block, bytes) in mapped.into_iter().zip(blocks) {
        block.write(*bytes);
    }
    remainder.copy_from_slice(tail);
}

struct Renderer {
    pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    shadow: wgpu::TextureView,
    vertices: wgpu::Buffer,
    uniform: wgpu::Buffer,
    bindings: wgpu::BindGroup,
    shadow_bindings: wgpu::BindGroup,
    capacity: u64,
    uploads: wgpu::util::StagingBelt,
    shadow_sampler: wgpu::Sampler,
    material_sampler: wgpu::Sampler,
    textured: bool,
    nature: Nature,
    leaf_pipeline: wgpu::RenderPipeline,
    leaf_shadow_pipeline: wgpu::RenderPipeline,
    leaf_bindings: wgpu::BindGroup,
    leaf_shadow_bindings: wgpu::BindGroup,
    leaf_mesh: wgpu::Buffer,
    leaf_mesh_vertices: u32,
    leaf_instances: wgpu::Buffer,
    leaf_capacity: u64,
    // Secondary viewport slot (slot = 1) for simultaneous split-screen 3D rendering!
    secondary_shadow: wgpu::TextureView,
    secondary_vertices: wgpu::Buffer,
    secondary_uniform: wgpu::Buffer,
    secondary_bindings: wgpu::BindGroup,
    secondary_shadow_bindings: wgpu::BindGroup,
    secondary_capacity: u64,
}

struct Nature {
    textures: foliage::Textures,
    leaf_sampler: wgpu::Sampler,
    bark_sampler: wgpu::Sampler,
}

impl Nature {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let sampler = |label, address_mode| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                address_mode_u: address_mode,
                address_mode_v: address_mode,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                anisotropy_clamp: 8,
                ..Default::default()
            })
        };
        Self {
            textures: foliage::load(device, queue),
            leaf_sampler: sampler("leaf filtering", wgpu::AddressMode::ClampToEdge),
            bark_sampler: sampler("bark filtering", wgpu::AddressMode::Repeat),
        }
    }
}

fn material_texture(device: &wgpu::Device, queue: &wgpu::Queue, model: bool) -> wgpu::TextureView {
    let images = if model {
        model_asset::textures()
    } else {
        vec![image::RgbaImage::from_pixel(1, 1, image::Rgba([255; 4])); 2]
    };
    let width = images.iter().map(|i| i.width()).max().unwrap();
    let height = images.iter().map(|i| i.height()).max().unwrap();
    let levels = width.max(height).ilog2() + 1;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Mavic albedo mip chain"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: images.len() as u32,
        },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (layer, image) in images.into_iter().enumerate() {
        let mut image =
            image::imageops::resize(&image, width, height, image::imageops::FilterType::Triangle);
        for level in 0..levels {
            let (w, h) = image.dimensions();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                image.as_raw(),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 4),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            image = image::imageops::resize(
                &image,
                (w / 2).max(1),
                (h / 2).max(1),
                image::imageops::FilterType::Triangle,
            );
        }
    }
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}

fn bindings(
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    uniform: &wgpu::Buffer,
    shadow: &wgpu::TextureView,
    shadow_sampler: &wgpu::Sampler,
    texture: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    nature: &Nature,
) -> wgpu::BindGroup {
    let view = |binding, view| wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    };
    let filter = |binding, sampler| wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Sampler(sampler),
    };
    let t = &nature.textures;
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("studio lighting and surfaces"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(shadow),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(shadow_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(texture),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            view(5, &t.leaf_color),
            view(6, &t.leaf_normal),
            view(7, &t.bark_color),
            view(8, &t.bark_normal),
            filter(9, &nature.leaf_sampler),
            filter(10, &nature.bark_sampler),
        ],
    })
}

pub fn initialize(state: &egui_wgpu::RenderState) {
    vapor::initialize(state);
    spray::initialize(state);
    plume::initialize(state);
    let device = &state.device;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Cable Lab studio"),
        source: wgpu::ShaderSource::Wgsl(include_str!("studio.wgsl").into()),
    });
    let attributes =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4, 3 => Float32x2];
    let vertices = [wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &attributes,
    }];
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("studio camera"),
        contents: &[0; 80],
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let shadow = device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("studio shadows"),
            size: wgpu::Extent3d {
                width: 2048,
                height: 2048,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("studio shadow filtering"),
        compare: Some(wgpu::CompareFunction::LessEqual),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("studio shaded surfaces"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vertex"),
            compilation_options: Default::default(),
            buffers: &vertices,
        },
        primitive: wgpu::PrimitiveState {
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24Plus,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 4,
            ..Default::default()
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fragment"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: state.target_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("studio shadow depth"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("shadow_vertex"),
            compilation_options: Default::default(),
            buffers: &vertices,
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: None,
        multiview_mask: None,
        cache: None,
    });
    let shadow_bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("studio shadow camera"),
        layout: &shadow_pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }],
    });
    let material_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Mavic trilinear filtering"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });
    let material = material_texture(device, &state.queue, false);
    let nature = Nature::new(device, &state.queue);
    let bindings = bindings(
        device,
        &pipeline,
        &uniform,
        &shadow,
        &sampler,
        &material,
        &material_sampler,
        &nature,
    );
    let [leaf_pipeline, leaf_shadow_pipeline] =
        leaves::pipelines(device, &shader, state.target_format);
    let leaf_shadow_bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("leaf shadow camera"),
        layout: &leaf_shadow_pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }],
    });
    let leaf_bindings = self::bindings(
        device,
        &leaf_pipeline,
        &uniform,
        &shadow,
        &sampler,
        &material,
        &material_sampler,
        &nature,
    );
    let leaf_shape = leaves::mesh();
    let leaf_mesh = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("canonical leaf"),
        contents: bytemuck::cast_slice(&leaf_shape),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let leaf_capacity = 1 << 20;
    let leaf_instances = leaf_instance_buffer(device, leaf_capacity);
    let capacity = 4 * 1024 * 1024;
    let vertices = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("studio mesh"),
        size: capacity,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let secondary_capacity = 2 * 1024 * 1024;
    let secondary_vertices = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("studio mesh slot 1"),
        size: secondary_capacity,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let secondary_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("studio camera slot 1"),
        contents: &[0; 80],
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let secondary_shadow = device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("studio shadows slot 1"),
            size: wgpu::Extent3d {
                width: 1024,
                height: 1024,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default());
    let secondary_shadow_bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("studio shadow camera slot 1"),
        layout: &shadow_pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: secondary_uniform.as_entire_binding(),
        }],
    });
    let secondary_bindings = self::bindings(
        device,
        &pipeline,
        &secondary_uniform,
        &secondary_shadow,
        &sampler,
        &material,
        &material_sampler,
        &nature,
    );
    state.renderer.write().callback_resources.insert(Renderer {
        pipeline,
        shadow_pipeline,
        shadow,
        vertices,
        uniform,
        bindings,
        shadow_bindings,
        capacity,
        uploads: wgpu::util::StagingBelt::new(device.clone(), capacity),
        shadow_sampler: sampler,
        material_sampler,
        textured: false,
        nature,
        leaf_pipeline,
        leaf_shadow_pipeline,
        leaf_bindings,
        leaf_shadow_bindings,
        leaf_mesh,
        leaf_mesh_vertices: leaf_shape.len() as u32,
        leaf_instances,
        leaf_capacity,
        secondary_shadow,
        secondary_vertices,
        secondary_uniform,
        secondary_bindings,
        secondary_shadow_bindings,
        secondary_capacity,
    });
}

fn leaf_instance_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("leaf instances"),
        size,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

struct Frame {
    slot: usize,
    vertices: Vec<Vertex>,
    shadow_vertices: u32,
    camera: [[f32; 4]; 5],
    leaves: Vec<LeafInstance>,
}

pub fn paint(painter: &egui::Painter, camera: &Camera, rect: egui::Rect, triangles: Vec<Triangle>) {
    paint_leaves(painter, camera, rect, triangles, Vec::new());
}

/// Paint triangles in vacuum/space mode (no ground plane, crisp orbital sun +
/// Earthshine lighting, and optional emissive thruster plume material >= 14).
#[allow(dead_code)]
pub fn paint_space(
    painter: &egui::Painter,
    camera: &Camera,
    rect: egui::Rect,
    triangles: Vec<Triangle>,
) {
    paint_internal(painter, camera, rect, triangles, Vec::new(), true, 0);
}

/// Paint triangles in vacuum/space mode into a specific viewport slot (`0` or `1`)
/// so two simultaneous split-screen 3D viewports render cleanly in the same frame!
#[allow(dead_code)]
pub fn paint_space_slot(
    painter: &egui::Painter,
    camera: &Camera,
    rect: egui::Rect,
    triangles: Vec<Triangle>,
    slot: usize,
) {
    paint_internal(painter, camera, rect, triangles, Vec::new(), true, slot.min(1));
}

/// Paint triangles and instanced leaves in one shaded and shadowed pass.
pub fn paint_leaves(
    painter: &egui::Painter,
    camera: &Camera,
    rect: egui::Rect,
    triangles: Vec<Triangle>,
    leaves: Vec<LeafInstance>,
) {
    paint_internal(painter, camera, rect, triangles, leaves, false, 0);
}

fn paint_internal(
    painter: &egui::Painter,
    camera: &Camera,
    rect: egui::Rect,
    mut triangles: Vec<Triangle>,
    leaves: Vec<LeafInstance>,
    space: bool,
    slot: usize,
) {
    use molla_math::Vec3;
    let shadow_vertices = (triangles.len() * 3) as u32;
    let shadow_extent = triangles
        .iter()
        .filter(|t| t.material < 3.0 || (t.material >= 10.0 && (t.material < 13.8 || t.material > 14.5)))
        .flat_map(|t| t.p)
        .map(|p| (p - if space { Vec3::ZERO } else { Vec3::Y }).length() as f32 + 0.5)
        .fold(3.5_f32, f32::max);
    if !space {
        let ground = [
            Vec3::new(-100.0, -0.001, -100.0),
            Vec3::new(-100.0, -0.001, 100.0),
            Vec3::new(100.0, -0.001, 100.0),
            Vec3::new(100.0, -0.001, -100.0),
        ];
        for ids in [[0, 1, 2], [0, 2, 3]] {
            triangles.push(Triangle {
                p: ids.map(|i| ground[i]),
                color: egui::Color32::from_rgb(65, 75, 90),
                material: 3.0,
                surface: None,
            });
        }
    }
    let mut vertices = Vec::with_capacity(triangles.len() * 3);
    for triangle in triangles {
        let normal = (triangle.p[1] - triangle.p[0])
            .cross(triangle.p[2] - triangle.p[0])
            .normalize_or_zero()
            .to_array()
            .map(|v| v as f32);
        let color = [
            triangle.color.r() as f32 / 255.0,
            triangle.color.g() as f32 / 255.0,
            triangle.color.b() as f32 / 255.0,
            triangle.material,
        ];
        vertices.extend((0..3).map(|i| {
            Vertex {
                position: triangle.p[i].to_array().map(|v| v as f32),
                normal: triangle
                    .surface
                    .map_or(normal, |s| s.normals[i].to_array().map(|v| v as f32)),
                color,
                uv: triangle.surface.map_or([0.0; 2], |s| s.uv[i]),
            }
        }));
    }
    let (eye, ray) = camera.ray(rect.center(), rect);
    let forward = -ray;
    let right = Vec3::Y.cross(forward).normalize();
    let up = forward.cross(right);
    let vector = |v: Vec3| [v.x as f32, v.y as f32, v.z as f32, 0.0];
    painter.add(egui_wgpu::Callback::new_paint_callback(
        rect,
        Frame {
            slot,
            vertices,
            shadow_vertices,
            leaves,
            camera: [
                vector(eye),
                vector(right),
                vector(up),
                vector(forward),
                [
                    2.5 * rect.height() / rect.width(),
                    if space { 1.0 } else { 0.0 },
                    0.0,
                    shadow_extent,
                ],
            ],
        },
    ));
}

impl egui_wgpu::CallbackTrait for Frame {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let renderer = resources.get_mut::<Renderer>().unwrap();
        if self.slot == 1 {
            let bytes = bytemuck::cast_slice(&self.vertices);
            if bytes.len() as u64 > renderer.secondary_capacity {
                renderer.secondary_capacity = (bytes.len() as u64).next_power_of_two();
                renderer.secondary_vertices = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("studio mesh slot 1"),
                    size: renderer.secondary_capacity,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            if !bytes.is_empty() {
                queue.write_buffer(&renderer.secondary_vertices, 0, bytes);
            }
            queue.write_buffer(&renderer.secondary_uniform, 0, bytemuck::cast_slice(&self.camera));
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("studio shadows slot 1"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &renderer.secondary_shadow,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&renderer.shadow_pipeline);
            pass.set_bind_group(0, &renderer.secondary_shadow_bindings, &[]);
            pass.set_vertex_buffer(0, renderer.secondary_vertices.slice(..));
            pass.draw(0..self.shadow_vertices, 0..1);
            return Vec::new();
        }
        if !renderer.textured
            && self
                .vertices
                .iter()
                .any(|v| v.color[3] >= 8.0 && v.color[3] < 10.0)
        {
            let texture = material_texture(device, queue, true);
            renderer.bindings = bindings(
                device,
                &renderer.pipeline,
                &renderer.uniform,
                &renderer.shadow,
                &renderer.shadow_sampler,
                &texture,
                &renderer.material_sampler,
                &renderer.nature,
            );
            renderer.leaf_bindings = bindings(
                device,
                &renderer.leaf_pipeline,
                &renderer.uniform,
                &renderer.shadow,
                &renderer.shadow_sampler,
                &texture,
                &renderer.material_sampler,
                &renderer.nature,
            );
            renderer.textured = true;
        }
        let bytes = bytemuck::cast_slice(&self.vertices);
        if bytes.len() as u64 > renderer.capacity {
            renderer.capacity = (bytes.len() as u64).next_power_of_two();
            renderer.vertices = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("studio mesh"),
                size: renderer.capacity,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !bytes.is_empty() {
            let mut upload_commands = device.create_command_encoder(&Default::default());
            let mut upload = renderer.uploads.write_buffer(
                &mut upload_commands,
                &renderer.vertices,
                0,
                wgpu::BufferSize::new(bytes.len() as u64).unwrap(),
            );
            copy_to_mapped(&mut upload, bytes);
            drop(upload);
            renderer.uploads.finish();
            queue.submit([upload_commands.finish()]);
            renderer.uploads.recall();
        }
        let leaf_bytes: &[u8] = bytemuck::cast_slice(&self.leaves);
        if leaf_bytes.len() as u64 > renderer.leaf_capacity {
            renderer.leaf_capacity = (leaf_bytes.len() as u64).next_power_of_two();
            renderer.leaf_instances = leaf_instance_buffer(device, renderer.leaf_capacity);
        }
        if !leaf_bytes.is_empty() {
            queue.write_buffer(&renderer.leaf_instances, 0, leaf_bytes);
        }
        queue.write_buffer(&renderer.uniform, 0, bytemuck::cast_slice(&self.camera));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("studio shadows"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &renderer.shadow,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&renderer.shadow_pipeline);
        pass.set_bind_group(0, &renderer.shadow_bindings, &[]);
        pass.set_vertex_buffer(0, renderer.vertices.slice(..));
        pass.draw(0..self.shadow_vertices, 0..1);
        if !self.leaves.is_empty() {
            pass.set_pipeline(&renderer.leaf_shadow_pipeline);
            pass.set_bind_group(0, &renderer.leaf_shadow_bindings, &[]);
            pass.set_vertex_buffer(0, renderer.leaf_mesh.slice(..));
            pass.set_vertex_buffer(1, renderer.leaf_instances.slice(..));
            pass.draw(0..renderer.leaf_mesh_vertices, 0..self.leaves.len() as u32);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let renderer = resources.get::<Renderer>().unwrap();
        pass.set_pipeline(&renderer.pipeline);
        if self.slot == 1 {
            pass.set_bind_group(0, &renderer.secondary_bindings, &[]);
            pass.set_vertex_buffer(0, renderer.secondary_vertices.slice(..));
            pass.draw(0..self.vertices.len() as u32, 0..1);
            return;
        }
        pass.set_bind_group(0, &renderer.bindings, &[]);
        pass.set_vertex_buffer(0, renderer.vertices.slice(..));
        pass.draw(0..self.vertices.len() as u32, 0..1);
        if !self.leaves.is_empty() {
            pass.set_pipeline(&renderer.leaf_pipeline);
            pass.set_bind_group(0, &renderer.leaf_bindings, &[]);
            pass.set_vertex_buffer(0, renderer.leaf_mesh.slice(..));
            pass.set_vertex_buffer(1, renderer.leaf_instances.slice(..));
            pass.draw(0..renderer.leaf_mesh_vertices, 0..self.leaves.len() as u32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires Vulkan; renderer upload readback"]
    fn bounded_renderer_upload_preserves_partial_blocks_and_large_meshes() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::default(),
            backend_options: Default::default(),
            memory_budget_thresholds: Default::default(),
            display: None,
        });
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        for len in [4, 128, 256, 260, 65540, 2_000_012, 32_000_012] {
            let expected: Vec<u8> = (0..len).map(|i| (i * 163 + i / 251) as u8).collect();
            let source = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("studio upload test source"),
                size: len as u64,
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            });
            let target = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("studio upload test readback"),
                size: len as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let start = std::time::Instant::now();
            source
                .slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(&expected);
            let bulk_ms = start.elapsed().as_secs_f64() * 1000.0;
            let start = std::time::Instant::now();
            copy_to_mapped(&mut source.slice(..).get_mapped_range_mut(), &expected);
            println!(
                "studio {len} bytes: bulk {bulk_ms:.3} ms, bounded {:.3} ms",
                start.elapsed().as_secs_f64() * 1000.0
            );
            source.unmap();
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&source, 0, &target, 0, len as u64);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            target
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(30)),
                })
                .unwrap();
            rx.recv().unwrap().unwrap();
            assert_eq!(&*target.slice(..).get_mapped_range(), expected.as_slice());
            target.unmap();
        }
    }
}
