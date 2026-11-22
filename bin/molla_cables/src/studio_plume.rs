//! Volumetric velocity-stretched vacuum gas & adiabatic cryo-vent plume renderer.

use crate::draw::Camera;
use eframe::{egui, egui_wgpu, wgpu};
use molla_math::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct PlumeBillboard {
    pub pos: Vec3,
    pub vel: Vec3,
    pub radius: f64,
    pub stretch: f64,
    pub opacity: f32,
    pub age_frac: f32,
    pub seed: f32,
    /// `0.0` = Draco RCS exhaust, `1.0` = Cryo-cooled vestibule air vent.
    pub kind: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    uv: [f32; 2],
    params: [f32; 4],
}

struct Renderer {
    pipeline: wgpu::RenderPipeline,
    bindings: [wgpu::BindGroup; 2],
    uniforms: [wgpu::Buffer; 2],
}

pub fn initialize(state: &egui_wgpu::RenderState) {
    let d = &state.device;
    let shader = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("space vacuum plume"),
        source: wgpu::ShaderSource::Wgsl(include_str!("studio_plume.wgsl").into()),
    });
    let attributes = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x4];
    let pipeline = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("volumetric space plumes"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vertex"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &attributes,
            }],
        },
        primitive: Default::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24Plus,
            depth_write_enabled: Some(false),
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
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    let make_slot = |label| {
        let uniform = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: 80,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let binding = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        (uniform, binding)
    };
    let (u0, b0) = make_slot("plume camera 0");
    let (u1, b1) = make_slot("plume camera 1");
    state.renderer.write().callback_resources.insert(Renderer {
        pipeline,
        bindings: [b0, b1],
        uniforms: [u0, u1],
    });
}

pub fn paint(
    painter: &egui::Painter,
    camera: &Camera,
    rect: egui::Rect,
    puffs: &[PlumeBillboard],
) {
    paint_slot(painter, camera, rect, puffs, 0);
}

#[allow(dead_code)]
pub fn paint_slot(
    painter: &egui::Painter,
    camera: &Camera,
    rect: egui::Rect,
    puffs: &[PlumeBillboard],
    slot: usize,
) {
    if puffs.is_empty() {
        return;
    }
    let (eye, ray) = camera.ray(rect.center(), rect);
    let forward = -ray;
    let right = Vec3::Y.cross(forward).normalize();
    let up = forward.cross(right);
    let mut visible: Vec<&PlumeBillboard> = puffs
        .iter()
        .filter(|p| -(p.pos - eye).dot(forward) > 0.12)
        .collect();
    visible.sort_unstable_by(|a, b| {
        b.pos
            .distance_squared(eye)
            .total_cmp(&a.pos.distance_squared(eye))
    });
    let mut vertices = Vec::with_capacity(visible.len() * 6);
    for p in visible {
        // Velocity-aligned billboard axes: elongate along projected jet velocity in screen plane!
        let to_eye = (eye - p.pos).normalize_or_zero();
        let vel_proj = p.vel - to_eye * p.vel.dot(to_eye);
        let (axis_along, axis_across) = if vel_proj.length_squared() > 1e-4 {
            let along = vel_proj.normalize();
            let across = to_eye.cross(along).normalize_or_zero();
            if across.length_squared() > 0.5 {
                (along, across)
            } else {
                (up, right)
            }
        } else {
            (up, right)
        };
        let rx = p.radius;
        let ry = p.radius * p.stretch;
        let params = [p.opacity, p.age_frac, p.seed, p.kind];
        for (x, y) in [
            (-1.0, -1.0),
            (1.0, -1.0),
            (1.0, 1.0),
            (-1.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
        ] {
            let world = p.pos + axis_across * (x * rx) + axis_along * (y * ry);
            vertices.push(Vertex {
                position: world.to_array().map(|v| v as f32),
                uv: [x as f32, y as f32],
                params,
            });
        }
    }
    if vertices.is_empty() {
        return;
    }
    let v = |p: Vec3| [p.x as f32, p.y as f32, p.z as f32, 0.0];
    painter.add(egui_wgpu::Callback::new_paint_callback(
        rect,
        Frame {
            slot: slot.min(1),
            vertices,
            buffer: std::sync::OnceLock::new(),
            camera: [
                v(eye),
                v(right),
                v(up),
                v(forward),
                [2.5 * rect.height() / rect.width(), 0.0, 0.0, 0.0],
            ],
        },
    ));
}

struct Frame {
    slot: usize,
    vertices: Vec<Vertex>,
    buffer: std::sync::OnceLock<wgpu::Buffer>,
    camera: [[f32; 4]; 5],
}

impl egui_wgpu::CallbackTrait for Frame {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _: &egui_wgpu::ScreenDescriptor,
        _: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let renderer = resources.get::<Renderer>().unwrap();
        queue.write_buffer(&renderer.uniforms[self.slot], 0, bytemuck::cast_slice(&self.camera));
        let bytes = bytemuck::cast_slice(&self.vertices);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("plume billboards"),
            size: bytes.len().max(4) as u64,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: true,
        });
        if !bytes.is_empty() {
            super::copy_to_mapped(&mut buffer.slice(..).get_mapped_range_mut(), bytes);
        }
        buffer.unmap();
        let _ = self.buffer.set(buffer);
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
        pass.set_bind_group(0, &renderer.bindings[self.slot], &[]);
        pass.set_vertex_buffer(0, self.buffer.get().unwrap().slice(..));
        pass.draw(0..self.vertices.len() as u32, 0..1);
    }
}
