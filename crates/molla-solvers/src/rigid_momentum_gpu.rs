//! Retained world-momentum transport for unrestricted floating articulations.

use crate::FeatherstoneSolver;
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_math::{GpuMat3, GpuTransform, Real};
use molla_sim::{BodyFlags, JointType, Model, State};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Joint {
    parent: i32,
    child: u32,
    kind: u32,
    dof: u32,
    coordinate: u32,
    columns: u32,
    reversed: u32,
    padding: u32,
    parent_frame: GpuTransform,
    child_frame: GpuTransform,
    axis: [f32; 4],
    mass: [f32; 4],
    inertia: GpuMat3,
    com: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<Joint>() == 192);

#[derive(Clone, Copy)]
struct PairedBuffers<'a> {
    rates: [&'a wgpu::Buffer; 2],
    velocities: [&'a wgpu::Buffer; 2],
    positions: [&'a wgpu::Buffer; 2],
    coordinates: [&'a wgpu::Buffer; 2],
    projected_low: Option<&'a wgpu::Buffer>,
    active: &'a wgpu::Buffer,
    status: &'a wgpu::Buffer,
}

fn paired_source(width: usize) -> String {
    let source = include_str!("../shaders/rigid_momentum.wgsl")
        .split("@compute")
        .next()
        .unwrap();
    let mut source = source.to_owned();
    for name in ["input_rate", "output_rate", "delta", "response_delta"] {
        let from = format!(" {name}: array<f32>");
        assert_eq!(source.matches(&from).count(), 1);
        source = source.replace(&from, &format!(" {name}: array<vec2<f32>>"));
    }
    format!(
        "{source}\n{}",
        include_str!("../shaders/rigid_momentum_paired.wgsl")
    )
    .replace("__ART_WIDTH__", &width.to_string())
}

pub(super) struct MomentumGpu {
    pipeline: wgpu::ComputePipeline,
    groups: [wgpu::BindGroup; 2],
    params: wgpu::Buffer,
    articulations: u32,
    correction: bool,
    workgroup_size: u32,
}

struct MomentumProgram {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

#[cfg(test)]
#[path = "rigid_momentum_paired_tests.rs"]
mod paired_tests;

impl MomentumGpu {
    pub fn new(
        engine: &Engine,
        model: &Model,
        states: &[State; 2],
        projected: Option<&State>,
        delta: &wgpu::Buffer,
        correction: Option<&wgpu::Buffer>,
        solver: &FeatherstoneSolver,
    ) -> Result<Option<Self>> {
        let paired = if let Some(predictor) = solver.gpu_paired_velocity_delta_buffer() {
            if predictor != delta {
                return Err(Error::Gpu(
                    "paired momentum requires the paired predictor delta".into(),
                ));
            }
            Some(PairedBuffers {
                rates: [
                    solver.gpu_paired_rates_buffer(&states[0])?,
                    solver.gpu_paired_rates_buffer(&states[1])?,
                ],
                velocities: [
                    solver.gpu_body_velocity_compensation_buffer(&states[0])?,
                    solver.gpu_body_velocity_compensation_buffer(&states[1])?,
                ],
                positions: [
                    solver.gpu_position_compensation_buffer(&states[0])?,
                    solver.gpu_position_compensation_buffer(&states[1])?,
                ],
                coordinates: [
                    solver.gpu_coordinate_compensation_buffer(&states[0])?,
                    solver.gpu_coordinate_compensation_buffer(&states[1])?,
                ],
                projected_low: projected
                    .map(|s| solver.gpu_position_compensation_buffer(s))
                    .transpose()?,
                active: solver.gpu_active_articulations_buffer()?,
                status: solver.gpu_status_buffer().unwrap(),
            })
        } else {
            None
        };
        Self::from_buffers(
            engine,
            model,
            states,
            projected,
            delta,
            correction,
            Some(solver),
            paired.as_ref(),
        )
    }

    fn from_buffers(
        engine: &Engine,
        model: &Model,
        states: &[State; 2],
        projected: Option<&State>,
        delta: &wgpu::Buffer,
        correction: Option<&wgpu::Buffer>,
        solver: Option<&FeatherstoneSolver>,
        paired: Option<&PairedBuffers<'_>>,
    ) -> Result<Option<Self>> {
        let count = model
            .articulation_count
            .max(usize::from(model.joint_count > 0));
        let starts = model.articulation_start.host()?;
        let children = model.joint_child.host()?;
        let parents = model.joint_parent.host()?;
        let kinds = model.joint_type.host()?;
        let dofs = model.joint_dof_offset.host()?;
        let lower = model.joint_limit_lower.host()?;
        let upper = model.joint_limit_upper.host()?;
        let caps = model.joint_velocity_limit.host()?;
        let flags = model.body_flags.host()?;
        let enabled = model.joint_enabled.host()?;
        let mut records = Vec::with_capacity(model.joint_count);
        let mut arts = Vec::with_capacity(count);
        let mut width = 1;
        for art in 0..count {
            let range = if model.articulation_count == 0 {
                0..model.joint_count
            } else {
                starts[art] as usize..starts[art + 1] as usize
            };
            let root = range.start;
            let eligible = !range.is_empty()
                && [JointType::Free as u32, JointType::D6 as u32].contains(&kinds[root])
                && parents[root] < 0
                && enabled[root] != 0
                && children[range.clone()]
                    .iter()
                    .all(|&body| flags[body as usize] & BodyFlags::IMMOVABLE == 0)
                && (dofs[root] as usize..dofs[root] as usize + 6).all(|d| {
                    !lower[d].is_finite() && !upper[d].is_finite() && !caps[d].is_finite()
                });
            arts.push([root as u32, range.end as u32, u32::from(eligible), 0]);
            width = width.max(range.len());
            for j in range.clone() {
                let child = children[j] as usize;
                let mut columns = 0;
                let first = dofs[j] as usize;
                let angular = if kinds[j] == JointType::Ball as u32 {
                    0
                } else {
                    3
                };
                let partial = (kinds[j] == JointType::D6 as u32
                    || kinds[j] == JointType::Ball as u32)
                    && molla_sim::joint_control::partial_angular_lock(
                        &lower[first + angular..first + angular + 3],
                        &upper[first + angular..first + angular + 3],
                    );
                for d in dofs[j]..dofs[j + 1] {
                    let locked = (kinds[j] <= 2 || kinds[j] == JointType::D6 as u32)
                        && lower[d as usize].is_finite()
                        && upper[d as usize].is_finite()
                        && (upper[d as usize] - lower[d as usize]).abs() < 1e-12
                        && !(partial && d - dofs[j] >= angular as u32);
                    if enabled[j] != 0 && !locked {
                        columns |= 1 << (d - dofs[j]);
                    }
                }
                let axis = model.joint_axis.host()?[j];
                let com = model.body_com.host()?[child];
                records.push(Joint {
                    parent: children[root..j]
                        .iter()
                        .position(|&body| body == parents[j])
                        .map_or(-1, |p| p as i32),
                    child: child as u32,
                    kind: kinds[j],
                    dof: dofs[j],
                    coordinate: model.joint_coord_offset.host()?[j],
                    columns,
                    reversed: model.joint_reversed.host()?[j],
                    padding: 0,
                    parent_frame: model.joint_x_p.host()?[j].into(),
                    child_frame: model.joint_x_c.host()?[j].into(),
                    axis: [axis.x as f32, axis.y as f32, axis.z as f32, 0.0],
                    mass: [model.body_mass.host()?[child] as f32, 0.0, 0.0, 0.0],
                    inertia: model.body_inertia.host()?[child].into(),
                    com: [com.x as f32, com.y as f32, com.z as f32, 0.0],
                });
            }
        }
        if !arts.iter().any(|a| a[2] != 0) {
            return Ok(None);
        }
        if width > 32 {
            return Err(Error::Gpu(
                "floating momentum articulation exceeds 32 joints".into(),
            ));
        }
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        if let Some(p) = paired {
            let check = |b: &wgpu::Buffer, n: u64| {
                b.size() >= n && b.usage().contains(wgpu::BufferUsages::STORAGE)
            };
            let rates = model.joint_dof_total as u64 * 8;
            if device.limits().max_storage_buffers_per_shader_stage < 21
                || device.limits().max_compute_workgroup_storage_size < 4
                || count > device.limits().max_compute_workgroups_per_dimension as usize
                || records.len() as u64 * 192
                    > device.limits().max_storage_buffer_binding_size as u64
                || count as u64 * 16 > device.limits().max_storage_buffer_binding_size as u64
                || !check(delta, rates)
                || correction.is_some_and(|b| !check(b, rates))
                || !check(p.active, count as u64 * 4)
                || !check(p.status, 4)
                || p.rates[0] == p.rates[1]
                || p.velocities[0] == p.velocities[1]
                || projected.is_some() != p.projected_low.is_some()
                || (0..2).any(|i| {
                    !check(p.rates[i], rates)
                        || !check(p.velocities[i], model.body_count as u64 * 32)
                        || !check(p.positions[i], model.body_count as u64 * 16)
                        || !check(p.coordinates[i], model.joint_coord_total as u64 * 4)
                })
            {
                return Err(Error::Gpu("invalid paired momentum buffer contract".into()));
            }
            if p.projected_low
                .is_some_and(|b| !check(b, model.body_count as u64 * 16))
            {
                return Err(Error::Gpu(
                    "short paired momentum reference residues".into(),
                ));
            }
            for side in 0..2 {
                let input = &states[side];
                let output = &states[1 - side];
                let reference = projected.unwrap_or(input);
                let reads = [
                    input.body_q.device_buffer_current()?,
                    input.body_qd.device_buffer_current()?,
                    p.rates[side],
                    p.positions[side],
                    p.velocities[side],
                    reference.body_q.device_buffer_current()?,
                    p.projected_low.unwrap_or(p.positions[side]),
                    delta,
                    correction.unwrap_or(delta),
                    p.active,
                ];
                let writes = [
                    output.body_q.device_buffer_current()?,
                    output.body_qd.device_buffer_current()?,
                    p.rates[1 - side],
                    output.joint_q.device_buffer_current()?,
                    p.positions[1 - side],
                    p.coordinates[1 - side],
                    p.status,
                    p.velocities[1 - side],
                    output.joint_qd.device_buffer_current()?,
                ];
                if writes
                    .iter()
                    .enumerate()
                    .any(|(i, b)| reads.contains(b) || writes[..i].contains(b))
                {
                    return Err(Error::Gpu("aliased paired momentum state".into()));
                }
            }
        }
        let data = |label, bytes: &[u8]| {
            let buffer = molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size: bytes.len() as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }
            );
            molla_core::gpu_write_buffer!(queue, &buffer, 0, bytes);
            buffer
        };
        let arts_buffer = data(
            "floating momentum articulations",
            bytemuck::cast_slice(&arts),
        );
        let joints = data("floating momentum joints", bytemuck::cast_slice(&records));
        let params = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("floating momentum params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let create = |device: &wgpu::Device| -> Result<MomentumProgram> {
            let entries: Vec<_> = (0..if paired.is_some() { 22 } else { 19 })
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 17 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: ![6, 7, 8, 9, 10, 11, 16, 20, 21].contains(&binding),
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                })
                .collect();
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("floating momentum layout"),
                entries: &entries,
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("floating momentum pipeline layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("floating momentum"),
                source: wgpu::ShaderSource::Wgsl(
                    if paired.is_some() {
                        paired_source(width)
                    } else {
                        include_str!("../shaders/rigid_momentum.wgsl")
                            .replace("__ART_WIDTH__", &width.to_string())
                    }
                    .into(),
                ),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("floating momentum"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("transport"),
                compilation_options: Default::default(),
                cache: None,
            });
            Ok(MomentumProgram { pipeline, layout })
        };
        #[cfg(not(target_arch = "wasm32"))]
        let program = engine.cached_gpu_program(
            &format!("width={width};paired={}", paired.is_some()),
            create,
        )?;
        #[cfg(target_arch = "wasm32")]
        let program = create(device)?;
        let layout = &program.layout;
        let pipeline = program.pipeline.clone();
        let group = |i: usize| -> Result<_> {
            let input = &states[i];
            let output = &states[1 - i];
            let reference = projected.unwrap_or(input);
            let position = |side| -> Result<&wgpu::Buffer> {
                match paired {
                    Some(p) => Ok(p.positions[side]),
                    None => solver
                        .unwrap()
                        .gpu_position_compensation_buffer(&states[side]),
                }
            };
            let mut buffers = vec![
                &arts_buffer,
                &joints,
                input.body_q.device_buffer_current()?,
                input.body_qd.device_buffer_current()?,
                match paired {
                    Some(p) => p.rates[i],
                    None => input.joint_qd.device_buffer_current()?,
                },
                position(i)?,
                output.body_q.device_buffer_current()?,
                output.body_qd.device_buffer_current()?,
                match paired {
                    Some(p) => p.rates[1 - i],
                    None => output.joint_qd.device_buffer_current()?,
                },
                output.joint_q.device_buffer_current()?,
                position(1 - i)?,
                match paired {
                    Some(p) => p.coordinates[1 - i],
                    None => solver.unwrap().gpu_coordinate_compensation_buffer(output)?,
                },
                reference.body_q.device_buffer_current()?,
                match paired {
                    Some(p) => p.projected_low.unwrap_or(p.positions[i]),
                    None => solver
                        .unwrap()
                        .gpu_position_compensation_buffer(reference)?,
                },
                delta,
                match paired {
                    Some(p) => p.active,
                    None => solver.unwrap().gpu_active_articulations_buffer()?,
                },
                match paired {
                    Some(p) => p.status,
                    None => solver.unwrap().gpu_status_buffer().unwrap(),
                },
                &params,
                correction.unwrap_or(delta),
            ];
            if let Some(p) = paired {
                buffers.extend([
                    p.velocities[i],
                    p.velocities[1 - i],
                    output.joint_qd.device_buffer_current()?,
                ]);
            }
            Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("floating momentum group"),
                layout: &layout,
                entries: &buffers
                    .iter()
                    .enumerate()
                    .map(|(binding, buffer)| wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            }))
        };
        let groups = [group(0)?, group(1)?];
        Ok(Some(Self {
            pipeline,
            groups,
            params,
            articulations: count as u32,
            correction: correction.is_some(),
            workgroup_size: if paired.is_some() { 1 } else { 32 },
        }))
    }

    pub fn prepare(&self, engine: &Engine, dt: Real) {
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &self.params,
            0,
            bytemuck::cast_slice(&[
                (dt as f32).to_bits(),
                self.articulations,
                u32::from(self.correction),
                0,
            ]),
        );
    }

    pub fn encode_profiled(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: usize,
        timestamps: Option<&(wgpu::QuerySet, [u32; 2])>,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("floating momentum transport"),
            timestamp_writes: timestamps.map(|(query_set, indices)| {
                wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(indices[0]),
                    end_of_pass_write_index: Some(indices[1]),
                }
            }),
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.groups[input], &[]);
        pass.dispatch_workgroups(self.articulations.div_ceil(self.workgroup_size), 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }
}
