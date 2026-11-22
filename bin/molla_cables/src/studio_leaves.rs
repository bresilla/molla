//! Instanced leaf blades: one canonical mesh posed per leaf in the vertex shader.
use crate::foliage_layout as layout;
use eframe::wgpu;

pub const MIRROR: u32 = 1;
pub const DROOP: u32 = 2;
pub const CAMBER: u32 = 4;
pub const PLATE: u32 = 8;
pub const PETIOLE: u32 = 16;
/// Segments along the canonical blade.
const SEGMENTS: usize = 8;
/// Lateral growth of the canonical blade beyond its outline.
const MARGIN: f64 = 1.06;

/// Pose and look of one leaf, read by `leaf_vertex` and `leaf_shadow_vertex`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LeafInstance {
    /// Blade centroid and length, in metres.
    pub center_length: [f32; 4],
    /// Base-to-tip direction and texture code `(layer + 64 * palette) / 4096`.
    pub direction_code: [f32; 4],
    /// Upper-face normal and shape flags.
    pub normal_flags: [f32; 4],
    /// Petiole anchor and plate camber height, in metres.
    pub anchor_bulge: [f32; 4],
    /// Tint (0.5 neutral) and plate width, in metres.
    pub tint_span: [f32; 4],
}

/// Canonical vertices `(t, x, face, kind)`: both faces of the blade, then the petiole.
pub fn mesh() -> Vec<[f32; 4]> {
    let mut out = Vec::new();
    let mut triangle = |c: [(f64, f64); 3], kind: [f32; 3]| {
        let area = (c[1].0 - c[0].0) * (c[2].1 - c[0].1) - (c[1].1 - c[0].1) * (c[2].0 - c[0].0);
        let upper = if area < 0.0 { [0, 1, 2] } else { [2, 1, 0] };
        for (face, order) in [(1.0, upper), (-1.0, [upper[2], upper[1], upper[0]])] {
            out.extend(order.map(|i| [c[i].0 as f32, c[i].1 as f32, face, kind[i]]));
        }
    };
    for i in 0..SEGMENTS {
        let (a, b) = (i as f64 / SEGMENTS as f64, (i + 1) as f64 / SEGMENTS as f64);
        for sign in [-1.0, 1.0] {
            let c = [
                (a, 0.0),
                (a, layout::half_width(a) * sign * MARGIN),
                (b, layout::half_width(b) * sign * MARGIN),
                (b, 0.0),
            ];
            triangle([c[0], c[1], c[2]], [0.0; 3]);
            triangle([c[0], c[2], c[3]], [0.0; 3]);
        }
    }
    triangle([(0.0, -1.0), (0.0, 1.0), (1.0, 0.0)], [1.0, 2.0, 3.0]);
    out
}

fn buffers() -> [wgpu::VertexBufferLayout<'static>; 2] {
    const SHAPE: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x4];
    const INSTANCE: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        1 => Float32x4, 2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4
    ];
    [
        wgpu::VertexBufferLayout {
            array_stride: 16,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &SHAPE,
        },
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LeafInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &INSTANCE,
        },
    ]
}

/// Shaded and shadow pipelines for instanced leaves.
pub fn pipelines(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
) -> [wgpu::RenderPipeline; 2] {
    let buffers = buffers();
    let shaded = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("instanced leaves"),
        layout: None,
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("leaf_vertex"),
            compilation_options: Default::default(),
            buffers: &buffers,
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
            module: shader,
            entry_point: Some("fragment"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    let shadows = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("instanced leaf shadows"),
        layout: None,
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("leaf_shadow_vertex"),
            compilation_options: Default::default(),
            buffers: &buffers,
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
    [shaded, shadows]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_leaf_has_both_faces_and_a_petiole() {
        let mesh = mesh();
        assert_eq!(mesh.len(), (SEGMENTS * 4 * 2 + 2) * 3);
        let upper = mesh.iter().filter(|v| v[3] == 0.0 && v[2] > 0.0).count();
        let lower = mesh.iter().filter(|v| v[3] == 0.0 && v[2] < 0.0).count();
        assert_eq!(upper, lower);
        for triangle in mesh.chunks(3).filter(|t| t[0][3] == 0.0) {
            let [a, b, c] = [triangle[0], triangle[1], triangle[2]];
            let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            assert!(area * a[2] < 0.0, "upper faces wind with -area, lower with +area");
            assert!(triangle.iter().all(|v| v[1].abs() <= 0.33 && (0.0..=1.0).contains(&v[0])));
        }
        assert_eq!(mesh.iter().filter(|v| v[3] > 0.0).count(), 6);
    }
}
