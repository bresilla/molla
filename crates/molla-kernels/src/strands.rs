//! Retained color-batched oriented-fiber dynamics and guarded publication.

use molla_core::{Error, Result};
use std::{num::NonZeroU64, sync::Arc};

/// Bytes of one `Krylov` record in `strands_coupled.wgsl`.
pub const KRYLOV_BYTES: u64 = 768;
pub const SHADER: &str = concat!(
    include_str!("../shaders/strands.wgsl"),
    "\n",
    include_str!("../shaders/strands_coupled.wgsl"),
    "\n",
    include_str!("../shaders/strands_contact.wgsl")
);
const ENTRIES: [&str; 19] = [
    "predict",
    "initialize_bonds",
    "project_bonds",
    "project_ground",
    "recover",
    "damage_bonds",
    "damp_bonds",
    "friction",
    "validate",
    "publish",
    "freeze_rows",
    "solve_coupled",
    "apply_coupled",
    "detect_contacts",
    "project_contacts",
    "contact_velocities",
    "color_contacts",
    "project_grab",
    "compact_contacts",
];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FiberRecord {
    pub com_mass: [f32; 4],
    pub inertia: [[f32; 4]; 3],
    pub inverse: [[f32; 4]; 3],
    pub vertices: [u32; 4],
    pub surface: [u32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MotionRecord {
    pub center: [f32; 4],
    pub rotation: [f32; 4],
    pub velocity: [f32; 4],
    pub omega: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BondRecord {
    pub fibers: [u32; 4],
    pub anchor0: [f32; 4],
    pub anchor1: [f32; 4],
    pub rest_offset: [f32; 4],
    pub rest_rotation: [f32; 4],
    pub translation: [f32; 4],
    pub rotation: [f32; 4],
    pub force_limit: [f32; 4],
    pub torque_limit: [f32; 4],
    pub material: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BondStateRecord {
    pub lambda0: [f32; 4],
    pub lambda1: [f32; 4],
    pub damage: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StrandParams {
    pub step: [f32; 4],
    pub gravity: [f32; 4],
    pub counts: [u32; 4],
    pub coupled: [f32; 4],
    pub numerics: [f32; 4],
    pub grab_anchor: [f32; 4],
    pub grab_target: [f32; 4],
    pub grab_velocity: [f32; 4],
    pub islands: [u32; 4],
    /// Tree preconditioner plan offsets in `colors`: parents, island levels, level bounds, order.
    pub tree: [u32; 4],
}

pub struct StrandProgram {
    layout: wgpu::BindGroupLayout,
    pipelines: Vec<wgpu::ComputePipeline>,
}

impl StrandProgram {
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        if device.limits().max_storage_buffers_per_shader_stage < 16
            || device.limits().max_bindings_per_bind_group < 18
        {
            return Err(Error::Gpu(
                "strand kernel needs 16 storage bindings and 18 bindings per group".into(),
            ));
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("strands.layout"),
            entries: &(0..18)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 11 || binding == 12 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: binding < 4 || binding == 15,
                            }
                        },
                        has_dynamic_offset: binding == 12,
                        min_binding_size: if binding == 12 {
                            NonZeroU64::new(16)
                        } else {
                            None
                        },
                    },
                })
                .collect::<Vec<_>>(),
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("strands"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("strands.pipeline"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = ENTRIES
            .iter()
            .map(|entry| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();
        Ok(Self { layout, pipelines })
    }
}

pub struct StrandDispatch {
    program: Arc<StrandProgram>,
    group: wgpu::BindGroup,
    groups: Vec<(u32, u32)>,
    batch_stride: u32,
    counts: [u32; 2],
    islands: u32,
}

impl StrandDispatch {
    pub fn new(
        device: &wgpu::Device,
        program: Arc<StrandProgram>,
        buffers: [&wgpu::Buffer; 18],
        groups: Vec<(u32, u32)>,
        counts: [u32; 2],
        batch_stride: u32,
        islands: u32,
    ) -> Result<Self> {
        let limit = device.limits().max_compute_workgroups_per_dimension;
        if counts[0] == 0
            || islands == 0
            || islands > limit
            || counts.iter().any(|&n| n.div_ceil(64) > limit)
            || groups.iter().any(|&(_, n)| n.div_ceil(64) > limit)
        {
            return Err(Error::Gpu("invalid strand dispatch capacity".into()));
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("strands.bindings"),
            layout: &program.layout,
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(i, &buffer)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: 0,
                        size: if i == 12 { NonZeroU64::new(16) } else { None },
                    }),
                })
                .collect::<Vec<_>>(),
        });
        Ok(Self {
            program,
            group,
            groups,
            batch_stride,
            counts,
            islands,
        })
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        iterations: usize,
        ground: bool,
        coupled: Option<usize>,
        pairs: u32,
        grab: bool,
    ) {
        let dispatch =
            |encoder: &mut wgpu::CommandEncoder, pipeline: usize, count: u32, batch: u32| {
                if count == 0 {
                    return;
                }
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(ENTRIES[pipeline]),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.program.pipelines[pipeline]);
                pass.set_bind_group(0, &self.group, &[batch * self.batch_stride]);
                pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
                molla_core::gpu_metrics::record_dispatch();
            };
        dispatch(encoder, 0, self.counts[0], 0);
        dispatch(encoder, 1, self.counts[1], 0);
        dispatch(encoder, 13, pairs, 0);
        if pairs > 0 {
            dispatch(encoder, 18, 1, 0);
            dispatch(encoder, 16, 1, 0);
        }
        let outer = coupled.unwrap_or(iterations);
        for iteration in 0..outer {
            if coupled.is_some() {
                dispatch(encoder, 10, self.counts[1].max(u32::from(grab)), 0);
                dispatch(encoder, 11, self.islands * 64, 0);
                dispatch(encoder, 12, self.counts[0].max(self.counts[1]), 0);
            } else {
                for (i, &(_, count)) in self.groups.iter().enumerate() {
                    dispatch(encoder, 2, count, (i * 2 + iteration % 2) as u32);
                }
            }
            for _ in 0..iterations.div_ceil(outer) {
                if pairs > 0 {
                    dispatch(encoder, 14, 64, 0);
                }
                if ground {
                    dispatch(encoder, 3, self.counts[0], 0);
                }
                if grab && coupled.is_none() {
                    dispatch(encoder, 17, 1, 0);
                }
            }
        }
        if ground && pairs > 0 {
            for _ in 0..8 {
                dispatch(encoder, 3, self.counts[0], 0);
            }
        }
        dispatch(encoder, 4, self.counts[0], 0);
        dispatch(encoder, 5, self.counts[1], 0);
        for (i, &(_, count)) in self.groups.iter().enumerate() {
            dispatch(encoder, 6, count, (i * 2) as u32);
        }
        if ground {
            dispatch(encoder, 7, self.counts[0], 0);
        }
        if pairs > 0 {
            dispatch(encoder, 15, 64, 0);
        }
        dispatch(encoder, 8, self.counts[0].max(self.counts[1]), 0);
        dispatch(encoder, 9, self.counts[0].max(self.counts[1]), 0);
    }
}
