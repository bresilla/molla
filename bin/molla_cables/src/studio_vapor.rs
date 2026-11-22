use crate::draw::Camera;
use eframe::{egui, egui_wgpu, wgpu};
use molla_math::Vec3;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    uv: [f32; 2],
    seed: f32,
}

struct Renderer {
    pipeline: wgpu::RenderPipeline,
    bindings: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    vertices: wgpu::Buffer,
    capacity: u64,
}

pub fn initialize(state: &egui_wgpu::RenderState) {
    let device = &state.device;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("orchard vapor"),
        source: wgpu::ShaderSource::Wgsl(include_str!("studio_vapor.wgsl").into()),
    });
    let attributes = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32];
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("depth-tested translucent vapor"),
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
        primitive: wgpu::PrimitiveState::default(),
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
    let uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vapor camera"),
        size: 80,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vapor camera"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }],
    });
    let capacity = 128 * 1024;
    let vertices = vertex_buffer(device, capacity);
    state.renderer.write().callback_resources.insert(Renderer {
        pipeline,
        bindings,
        uniform,
        vertices,
        capacity,
    });
}

fn vertex_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("advected vapor ribbons"),
        size,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn mesh(paths: &[Vec<Vec3>], eye: Vec3, right: Vec3, forward: Vec3) -> Vec<Vertex> {
    let mut quads = Vec::new();
    for (seed, path) in paths.iter().enumerate() {
        if path.len() < 4 || path.iter().any(|p| !p.is_finite()) {
            continue;
        }
        let mut length = vec![0.0];
        for pair in path.windows(2) {
            length.push(length.last().unwrap() + pair[1].distance(pair[0]));
        }
        let total = *length.last().unwrap();
        if total < 0.12 {
            continue;
        }
        let edges: Vec<_> = path
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let tangent = path[(i + 1).min(path.len() - 1)] - path[i.saturating_sub(1)];
                let mut side = tangent.cross(eye - *p).normalize_or_zero();
                if side.length_squared() < 0.5 {
                    side = right;
                }
                let t = length[i] / total;
                let width = 0.11 + 0.16 * (1.0 - t);
                [-1.0, 1.0].map(|s| Vertex {
                    position: (*p + side * width * s).to_array().map(|v| v as f32),
                    uv: [s as f32, t as f32],
                    seed: seed as f32 * 7.31,
                })
            })
            .collect();
        for (i, edge) in edges.windows(2).enumerate() {
            let center = (path[i] + path[i + 1]) * 0.5;
            if -(center - eye).dot(forward) < 0.2 {
                continue;
            }
            let vertices = [
                edge[0][0], edge[0][1], edge[1][0], edge[1][0], edge[0][1], edge[1][1],
            ];
            quads.push((-(center - eye).dot(forward), vertices));
        }
    }
    quads.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    quads
        .into_iter()
        .flat_map(|(_, vertices)| vertices)
        .collect()
}

pub fn paint(
    painter: &egui::Painter,
    camera: &Camera,
    rect: egui::Rect,
    paths: &[Vec<Vec3>],
    time: f64,
) {
    let (eye, ray) = camera.ray(rect.center(), rect);
    let forward = -ray;
    let right = Vec3::Y.cross(forward).normalize();
    let up = forward.cross(right);
    let vertices = mesh(paths, eye, right, forward);
    if vertices.is_empty() {
        return;
    }
    let vector = |p: Vec3| [p.x as f32, p.y as f32, p.z as f32, 0.0];
    painter.add(egui_wgpu::Callback::new_paint_callback(
        rect,
        Frame {
            vertices,
            camera: [
                vector(eye),
                vector(right),
                vector(up),
                vector(forward),
                [2.5 * rect.height() / rect.width(), time as f32, 0.0, 0.0],
            ],
        },
    ));
}

struct Frame {
    vertices: Vec<Vertex>,
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
        let renderer = resources.get_mut::<Renderer>().unwrap();
        let bytes = bytemuck::cast_slice(&self.vertices);
        if bytes.len() as u64 > renderer.capacity {
            renderer.capacity = (bytes.len() as u64).next_power_of_two();
            renderer.vertices = vertex_buffer(device, renderer.capacity);
        }
        queue.write_buffer(&renderer.vertices, 0, bytes);
        queue.write_buffer(&renderer.uniform, 0, bytemuck::cast_slice(&self.camera));
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
        pass.set_bind_group(0, &renderer.bindings, &[]);
        pass.set_vertex_buffer(0, renderer.vertices.slice(..));
        pass.draw(0..self.vertices.len() as u32, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vapor_ribbons_are_finite_bounded_and_sorted_back_to_front() {
        let path = vec![Vec3::ZERO, Vec3::Y * 0.2, Vec3::Y * 0.4, Vec3::Y * 0.6];
        let paths = [path.clone(), path.iter().map(|p| *p - Vec3::Z).collect()];
        let vertices = mesh(&paths, Vec3::Z * 4.0, Vec3::X, Vec3::Z);
        assert_eq!(vertices.len(), 36);
        assert!(
            vertices
                .iter()
                .all(|v| v.position.iter().all(|p| p.is_finite())
                    && v.position[0].abs() <= 0.271
                    && (0.0..=1.0).contains(&v.uv[1]))
        );
        assert_eq!(vertices[0].position[2], -1.0);
        assert!(mesh(&[vec![Vec3::ZERO; 4]], Vec3::Z, Vec3::X, Vec3::Z).is_empty());
        assert!(mesh(&[path], -Vec3::Z, Vec3::X, Vec3::Z).is_empty());
    }
}
