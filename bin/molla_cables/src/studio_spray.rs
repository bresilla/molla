use crate::draw::Camera;
use eframe::{egui, egui_wgpu, wgpu};
use molla_math::Vec3;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    uv: [f32; 2],
    opacity: f32,
}
struct Renderer {
    pipeline: wgpu::RenderPipeline,
    binding: wgpu::BindGroup,
    uniform: wgpu::Buffer,
}

pub fn initialize(state: &egui_wgpu::RenderState) {
    let d = &state.device;
    let shader = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("water spray"),
        source: wgpu::ShaderSource::Wgsl(include_str!("studio_spray.wgsl").into()),
    });
    let attributes = wgpu::vertex_attr_array![0=>Float32x3,1=>Float32x2,2=>Float32];
    let pipeline = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("water parcels"),
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
    let uniform = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("spray camera"),
        size: 80,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let binding = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("spray camera"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }],
    });
    state.renderer.write().callback_resources.insert(Renderer {
        pipeline,
        binding,
        uniform,
    });
}

pub fn paint(
    painter: &egui::Painter,
    camera: &Camera,
    rect: egui::Rect,
    points: &[(Vec3, f64, f32)],
) {
    if points.is_empty() {
        return;
    }
    let (eye, ray) = camera.ray(rect.center(), rect);
    let forward = -ray;
    let right = Vec3::Y.cross(forward).normalize();
    let up = forward.cross(right);
    let mut points: Vec<_> = points
        .iter()
        .filter(|(p, _, _)| -(*p - eye).dot(forward) > 0.1)
        .collect();
    points.sort_unstable_by(|a, b| {
        b.0.distance_squared(eye)
            .total_cmp(&a.0.distance_squared(eye))
    });
    let mut vertices = Vec::with_capacity(points.len() * 6);
    for &(p, r, opacity) in points {
        for (x, y) in [
            (-1.0, -1.0),
            (1.0, -1.0),
            (1.0, 1.0),
            (-1.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
        ] {
            vertices.push(Vertex {
                position: (p + (right * x + up * y) * r).to_array().map(|v| v as f32),
                uv: [x as f32, y as f32],
                opacity,
            });
        }
    }
    let v = |p: Vec3| [p.x as f32, p.y as f32, p.z as f32, 0.0];
    painter.add(egui_wgpu::Callback::new_paint_callback(
        rect,
        Frame {
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
        queue.write_buffer(&renderer.uniform, 0, bytemuck::cast_slice(&self.camera));
        let bytes = bytemuck::cast_slice(&self.vertices);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spray billboards"),
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
        pass.set_bind_group(0, &renderer.binding, &[]);
        pass.set_vertex_buffer(0, self.buffer.get().unwrap().slice(..));
        pass.draw(0..self.vertices.len() as u32, 0..1);
    }
}
