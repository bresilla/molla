use molla_core::{Error, Result};
use molla_geometry::LbvhBroadPhase;
use molla_math::Vec3;
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    vertices: u32,
    triangles: u32,
    edges: u32,
    springs: u32,
    tets: u32,
    obstacles: u32,
    contacts: u32,
    final_state: u32,
    h: f32,
    strain_limit: f32,
    pad: [u32; 2],
}

pub(super) struct GeometryValidator {
    tree: Option<LbvhBroadPhase>,
    uniforms: wgpu::Buffer,
    flags: wgpu::Buffer,
    bindings: wgpu::BindGroup,
    primitives: wgpu::ComputePipeline,
    intersections: wgpu::ComputePipeline,
    counts: Uniforms,
}

impl GeometryValidator {
    pub(super) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &[[f32; 3]],
        triangles: &[[u32; 3]],
        counts: [u32; 4],
        inputs: [&wgpu::Buffer; 14],
    ) -> Result<Self> {
        let tree = if triangles.is_empty() {
            None
        } else {
            let point = |i: u32| Vec3::from_array(positions[i as usize].map(f64::from));
            let bounds: Vec<_> = triangles
                .iter()
                .map(|t| {
                    let a = point(t[0]);
                    let b = point(t[1]);
                    let c = point(t[2]);
                    (a.min(b).min(c), a.max(b).max(c))
                })
                .collect();
            let mut tree = LbvhBroadPhase::new(
                Arc::new(device.clone()),
                Arc::new(queue.clone()),
                triangles.len().max(2),
                64,
            )?;
            tree.build_bounds(&bounds)?;
            Some(tree)
        };
        let allocate = |label: &'static str, size, usage| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false,
                }
            )
        };
        let uniforms = allocate(
            "bn.geometry.uniforms",
            48,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let flags = allocate(
            "bn.geometry.flags",
            16,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let dummy = tree.is_none().then(|| {
            [
                allocate("bn.geometry.empty_aabb", 32, wgpu::BufferUsages::STORAGE),
                allocate("bn.geometry.empty_nodes", 48, wgpu::BufferUsages::STORAGE),
                allocate("bn.geometry.empty_indices", 16, wgpu::BufferUsages::STORAGE),
            ]
        });
        let tree_buffers = match &tree {
            Some(tree) => [
                tree.primitive_aabb_buffer(),
                tree.nodes_buffer(),
                tree.sorted_idx_buffer(),
                tree.pair_count_buffer(),
            ],
            None => {
                let d = dummy.as_ref().unwrap();
                [&d[0], &d[1], &d[2], &d[2]]
            }
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bn.geometry.layout"),
            entries: &(0..20)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 0 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: binding != 15 && binding != 19,
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let buffers: Vec<_> = std::iter::once(&uniforms)
            .chain(inputs)
            .chain(tree_buffers)
            .chain(std::iter::once(&flags))
            .collect();
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bn.geometry.bindings"),
            layout: &layout,
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(binding, b)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bn.geometry"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/barrier_geometry.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bn.geometry.pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Ok(Self {
            tree,
            uniforms,
            flags,
            bindings,
            primitives: pipeline("validate_primitives"),
            intersections: pipeline("validate_intersections"),
            counts: Uniforms {
                vertices: positions.len() as u32,
                triangles: triangles.len() as u32,
                edges: counts[0],
                springs: counts[1],
                tets: counts[2],
                obstacles: counts[3],
                contacts: 0,
                final_state: 0,
                h: 0.0,
                strain_limit: 0.0,
                pad: [0; 2],
            },
        })
    }

    /// Validate GPU geometry and copy integer status bits into `scalars[12]`.
    pub(super) fn encode(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        h: f32,
        strain_limit: f32,
        final_state: bool,
        contacts: bool,
        scalars: &wgpu::Buffer,
    ) -> Result<()> {
        let u = Uniforms {
            h,
            strain_limit,
            final_state: u32::from(final_state),
            contacts: u32::from(contacts),
            ..self.counts
        };
        molla_core::gpu_write_buffer!(queue, &self.uniforms, 0, bytemuck::bytes_of(&u));
        encoder.clear_buffer(&self.flags, 0, None);
        let dispatch = |encoder: &mut wgpu::CommandEncoder, pipeline, count: u32| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bn.geometry"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.bindings, &[]);
            pass.dispatch_workgroups(count.div_ceil(64).max(1), 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        };
        dispatch(
            encoder,
            &self.primitives,
            u.vertices.max(u.triangles).max(u.springs).max(u.tets),
        );
        if contacts && let Some(tree) = &self.tree {
            tree.encode_refit_bounds(encoder, u.triangles)?;
            dispatch(encoder, &self.intersections, u.vertices + u.edges);
        }
        encoder.copy_buffer_to_buffer(&self.flags, 0, scalars, 48, 4);
        Ok(())
    }

    pub(super) fn check(flags: u32, final_state: bool) -> Result<()> {
        if flags == 0 {
            return Ok(());
        }
        let stage = if final_state { "accepted" } else { "initial" };
        let names: Vec<_> = [
            (1, "nonfinite arithmetic"),
            (2, "obstacle INTERSECTS"),
            (4, "strain limit"),
            (8, "degenerate geometry"),
            (16, "mesh INTERSECTS"),
            (32, "BVH traversal overflow"),
            (64, "inverted tet"),
        ]
        .into_iter()
        .filter_map(|(bit, name)| (flags & bit != 0).then_some(name))
        .collect();
        Err(Error::Solver(format!(
            "barrier_newton: {stage} geometry rejected ({flags:#x}): {}",
            names.join(", ")
        )))
    }
}

#[cfg(all(test, feature = "resource-counters"))]
mod tests {
    use super::*;
    use molla_compute::Engine;
    use wgpu::util::DeviceExt;

    #[test]
    #[ignore = "requires GPU; run via make test-ppf-geometry"]
    fn live_gpu_bounds_detect_moved_intersections_and_clear_rejected_state() {
        let engine = Engine::new_gpu().unwrap();
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let positions = [
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [-0.2, 1.0, -0.2],
            [0.2, 1.0, -0.2],
            [0.0, 1.0, 0.2],
        ];
        let rows: Vec<_> = positions.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
        let init = |bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("geometry test input"),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        let x = init(bytemuck::cast_slice(&rows));
        let previous = init(bytemuck::cast_slice(&rows));
        let zero = init(&[0; 128]);
        let triangles = [[0, 1, 2], [3, 4, 5]];
        let tris = init(bytemuck::cast_slice(&[[0u32, 1, 2, 0], [3, 4, 5, 0]]));
        let edges = init(bytemuck::cast_slice(&[
            [0u32, 1, 0, 0],
            [1, 2, 0, 0],
            [2, 0, 0, 0],
            [3, 4, 0, 0],
            [4, 5, 0, 0],
            [5, 3, 0, 0],
        ]));
        let validator = GeometryValidator::new(
            device,
            queue,
            &positions,
            &triangles,
            [6, 0, 0, 0],
            [
                &x, &previous, &zero, &previous, &zero, &tris, &edges, &zero, &zero, &zero, &zero,
                &zero, &zero, &zero,
            ],
        )
        .unwrap();
        let scalars = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let probe = |rows: &[[f32; 4]]| {
            queue.write_buffer(&x, 0, bytemuck::cast_slice(rows));
            let mut encoder = device.create_command_encoder(&Default::default());
            validator
                .encode(queue, &mut encoder, 1.0 / 240.0, 0.0, true, true, &scalars)
                .unwrap();
            encoder.copy_buffer_to_buffer(&scalars, 0, &stage, 0, 64);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            stage.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).unwrap();
            });
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let mapped = stage.slice(..).get_mapped_range().unwrap();
            let flags = bytemuck::cast_slice::<_, u32>(&mapped)[12];
            drop(mapped);
            stage.unmap();
            flags
        };
        let diagnostics = validator.tree.as_ref().unwrap().diagnostics();
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        assert_eq!(probe(&rows), 0);
        let mut moved = rows.clone();
        for p in &mut moved {
            p[0] += 100.0;
            p[1] = 0.0;
        }
        assert_ne!(probe(&moved) & 16, 0);
        assert_eq!(probe(&rows), 0);
        let mut collapsed = rows.clone();
        collapsed[2] = collapsed[1];
        assert_ne!(probe(&collapsed) & 8, 0);
        let mut nonfinite = rows.clone();
        nonfinite[0][0] = f32::NAN;
        assert_ne!(probe(&nonfinite) & 1, 0);
        assert_eq!(probe(&rows), 0);
        assert_eq!(allocations.finish().observations, 0);
        assert_eq!(validator.tree.as_ref().unwrap().diagnostics(), diagnostics);
    }
}
