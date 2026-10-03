//! Retained multi-DOF inverse dynamics with compiled child-anchor wrench output.

use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_sim::{Model, State};

const WORKGROUP: u32 = 32;

/// Global DOF accelerations/efforts, world COM loads and compiled-joint world wrenches.
pub struct ReactionWrenchBuffers {
    pub acceleration: wgpu::Buffer,
    /// Dense world COM/angular inertial accelerations; nonkinematic entries must be zero.
    pub boundary_acceleration: wgpu::Buffer,
    pub external: wgpu::Buffer,
    pub active: wgpu::Buffer,
    pub efforts: wgpu::Buffer,
    pub joints: wgpu::Buffer,
    /// Bit 0: nonfinite input/output; bit 1: capacity exceeded; bit 2: nonkinematic acceleration.
    pub status: wgpu::Buffer,
    dimensions: [usize; 4],
}

/// Bindings for one immutable model layout, input state and set of reaction buffers.
pub struct ReactionWrenchDispatch {
    pipeline: wgpu::ComputePipeline,
    groups: Option<[wgpu::BindGroup; 3]>,
    status: wgpu::Buffer,
    articulations: u32,
}

impl ReactionWrenchDispatch {
    /// Reset status and encode one evaluation without allocations, uploads or readback.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.status, 0, None);
        if self.articulations == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipeline);
        for (index, group) in self
            .groups
            .as_ref()
            .expect("nonempty reaction dispatch")
            .iter()
            .enumerate()
        {
            pass.set_bind_group(index as u32, group, &[]);
        }
        pass.dispatch_workgroups(self.articulations.div_ceil(WORKGROUP), 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }
}

pub struct ReactionWrenchKernel {
    pipeline: wgpu::ComputePipeline,
    model_layout: wgpu::BindGroupLayout,
    state_layout: wgpu::BindGroupLayout,
    uniform_group: wgpu::BindGroup,
    starts: wgpu::Buffer,
    zero: wgpu::Buffer,
    articulation_offsets: Vec<u32>,
    dof_offsets: Vec<u32>,
    dimensions: [usize; 4],
}

fn layout(model: &Model) -> Result<(Vec<u32>, Vec<u32>, [usize; 4])> {
    let starts = if model.articulation_count == 0 {
        if model.joint_count == 0 {
            vec![0]
        } else {
            vec![0, model.joint_count as u32]
        }
    } else {
        model.articulation_start.host()?.to_vec()
    };
    let dofs = model.joint_dof_offset.host()?.to_vec();
    let kinds = model.joint_type.host()?;
    if starts.first() != Some(&0)
        || starts.last().copied() != Some(model.joint_count as u32)
        || (model.articulation_count != 0 && starts.len() != model.articulation_count + 1)
        || starts
            .windows(2)
            .any(|range| range[0] > range[1] || range[1] - range[0] > 64)
        || dofs.len() != model.joint_count + 1
        || dofs.first() != Some(&0)
        || dofs.last().copied() != Some(model.joint_dof_total as u32)
        || kinds.len() != model.joint_count
    {
        return Err(Error::Gpu(
            "invalid reaction layout or articulation above 64 joints".into(),
        ));
    }
    for (joint, &kind) in kinds.iter().enumerate() {
        let width = [1, 1, 3, 0, 6, 6, 6, 2]
            .get(kind as usize)
            .ok_or_else(|| Error::Gpu("unsupported reaction joint kind".into()))?;
        if dofs[joint + 1].checked_sub(dofs[joint]) != Some(*width) {
            return Err(Error::Gpu("reaction joint width mismatch".into()));
        }
    }
    let dimensions = [
        model.body_count,
        model.joint_count,
        model.joint_dof_total,
        starts.len() - 1,
    ];
    Ok((starts, dofs, dimensions))
}

impl ReactionWrenchKernel {
    pub fn new(engine: &Engine, model: &Model) -> Result<Self> {
        let (articulation_offsets, dof_offsets, dimensions) = layout(model)?;
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("reaction kernel needs a device".into()))?;
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("reaction kernel needs a queue".into()))?;
        if device.limits().max_storage_buffers_per_shader_stage < 28
            || device.limits().max_bind_groups < 3
        {
            return Err(Error::Gpu(
                "reaction kernel needs 28 storage buffers and three bind groups".into(),
            ));
        }
        if dimensions[3].div_ceil(WORKGROUP as usize)
            > device.limits().max_compute_workgroups_per_dimension as usize
        {
            return Err(Error::Gpu(
                "reaction articulation count exceeds device dispatch capacity".into(),
            ));
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::reaction_wrenches"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("../shaders/reaction_wrenches.wgsl"),
                    include_str!("../shaders/articulated_response_common.wgsl")
                )
                .into(),
            ),
        });
        let model_entries: Vec<_> = (0..18).map(|binding| storage(binding, true)).collect();
        let state_entries: Vec<_> = (0..10)
            .map(|binding| storage(binding, binding < 6 || binding == 9))
            .collect();
        let model_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::reaction model"),
            entries: &model_entries,
        });
        let state_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::reaction state"),
            entries: &state_entries,
        });
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::reaction uniforms"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::reaction layout"),
            bind_group_layouts: &[
                Some(&model_layout),
                Some(&state_layout),
                Some(&uniform_layout),
            ],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::reaction pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("reaction_wrenches"),
            compilation_options: Default::default(),
            cache: None,
        });
        let starts = buffer(
            device,
            "molla::reaction articulation offsets",
            articulation_offsets.len(),
            4,
        );
        molla_core::gpu_write_buffer!(
            queue,
            &starts,
            0,
            bytemuck::cast_slice(&articulation_offsets)
        );
        let uniform = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::reaction uniforms"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        molla_core::gpu_write_buffer!(
            queue,
            &uniform,
            0,
            bytemuck::cast_slice(&[dimensions[3] as u32, 0, 0, 0]),
        );
        let uniform_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::reaction uniforms"),
            layout: &uniform_layout,
            entries: &[entry(0, Some(&uniform))?],
        });
        let zero = buffer(device, "molla::reaction empty storage", 1, 32);
        Ok(Self {
            pipeline,
            model_layout,
            state_layout,
            uniform_group,
            starts,
            zero,
            articulation_offsets,
            dof_offsets,
            dimensions,
        })
    }

    pub fn create_buffers(&self, engine: &Engine) -> Result<ReactionWrenchBuffers> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("reaction buffers need a device".into()))?;
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("reaction buffers need a queue".into()))?;
        let [bodies, joints, dofs, articulations] = self.dimensions;
        let active = buffer(device, "molla::reaction active", articulations, 4);
        molla_core::gpu_write_buffer!(
            queue,
            &active,
            0,
            bytemuck::cast_slice(&vec![1u32; articulations.max(1)]),
        );
        Ok(ReactionWrenchBuffers {
            acceleration: buffer(device, "molla::reaction acceleration", dofs, 4),
            boundary_acceleration: buffer(
                device,
                "molla::reaction boundary acceleration",
                bodies,
                32,
            ),
            external: buffer(device, "molla::reaction external", bodies, 32),
            efforts: buffer(device, "molla::reaction efforts", dofs, 4),
            joints: buffer(device, "molla::reaction joints", joints, 32),
            status: buffer(device, "molla::reaction status", 1, 4),
            active,
            dimensions: self.dimensions,
        })
    }

    /// Retain bindings to uploaded buffers; reconstruct after any model/state-buffer replacement.
    pub fn prepare(
        &self,
        engine: &Engine,
        model: &Model,
        state: &State,
        buffers: &ReactionWrenchBuffers,
    ) -> Result<ReactionWrenchDispatch> {
        let (starts, dofs, dimensions) = layout(model)?;
        if starts != self.articulation_offsets
            || dofs != self.dof_offsets
            || dimensions != self.dimensions
            || buffers.dimensions != dimensions
        {
            return Err(Error::Gpu(
                "reaction kernel layout changed; recompile its bindings".into(),
            ));
        }
        let [bodies, joints, dof_count, articulations] = dimensions;
        if state.body_q.len() != bodies
            || state.body_qd.len() != bodies
            || state.joint_qd.len() != dof_count
        {
            return Err(Error::Gpu(
                "reaction input state dimensions do not match model".into(),
            ));
        }
        for (buffer, count, stride) in [
            (&buffers.acceleration, dof_count, 4),
            (&buffers.boundary_acceleration, bodies, 32),
            (&buffers.external, bodies, 32),
            (&buffers.active, articulations, 4),
            (&buffers.efforts, dof_count, 4),
            (&buffers.joints, joints, 32),
            (&buffers.status, 1, 4),
        ] {
            if buffer.size() < count.max(1) as u64 * stride
                || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            {
                return Err(Error::Gpu(
                    "reaction buffer is undersized or lacks storage usage".into(),
                ));
            }
        }
        if !buffers
            .status
            .usage()
            .contains(wgpu::BufferUsages::COPY_DST)
        {
            return Err(Error::Gpu(
                "reaction status requires clear-buffer usage".into(),
            ));
        }
        if articulations == 0 {
            return Ok(ReactionWrenchDispatch {
                pipeline: self.pipeline.clone(),
                groups: None,
                status: buffers.status.clone(),
                articulations: 0,
            });
        }
        let empty_dofs = (dof_count == 0).then_some(&self.zero);
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("reaction bindings need a device".into()))?;
        let model_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::reaction model"),
            layout: &self.model_layout,
            entries: &[
                entry(0, model.joint_type.device_buffer())?,
                entry(1, model.joint_parent.device_buffer())?,
                entry(2, model.joint_child.device_buffer())?,
                entry(3, model.joint_x_p.device_buffer())?,
                entry(4, model.joint_x_c.device_buffer())?,
                entry(5, model.joint_axis.device_buffer())?,
                entry(6, model.joint_dof_offset.device_buffer())?,
                entry(7, model.joint_reversed.device_buffer())?,
                entry(8, model.joint_motion_flags.device_buffer())?,
                entry(9, model.joint_limit_lower.device_buffer().or(empty_dofs))?,
                entry(10, model.joint_limit_upper.device_buffer().or(empty_dofs))?,
                entry(11, model.body_mass.device_buffer())?,
                entry(12, model.body_inertia.device_buffer())?,
                entry(13, model.body_com.device_buffer())?,
                entry(14, model.body_flags.device_buffer())?,
                entry(15, model.body_world.device_buffer())?,
                entry(16, model.gravity.device_buffer())?,
                entry(17, Some(&self.starts))?,
            ],
        });
        let state_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::reaction state"),
            layout: &self.state_layout,
            entries: &[
                entry(0, state.body_q.device_buffer())?,
                entry(1, state.joint_qd.device_buffer().or(empty_dofs))?,
                entry(2, state.body_qd.device_buffer())?,
                entry(3, Some(&buffers.acceleration))?,
                entry(4, Some(&buffers.external))?,
                entry(5, Some(&buffers.active))?,
                entry(6, Some(&buffers.efforts))?,
                entry(7, Some(&buffers.joints))?,
                entry(8, Some(&buffers.status))?,
                entry(9, Some(&buffers.boundary_acceleration))?,
            ],
        });
        Ok(ReactionWrenchDispatch {
            pipeline: self.pipeline.clone(),
            groups: Some([model_group, state_group, self.uniform_group.clone()]),
            status: buffers.status.clone(),
            articulations: dimensions[3] as u32,
        })
    }
}

fn buffer(device: &wgpu::Device, label: &str, count: usize, stride: u64) -> wgpu::Buffer {
    molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size: count.max(1) as u64 * stride,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    )
}

fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn entry(binding: u32, buffer: Option<&wgpu::Buffer>) -> Result<wgpu::BindGroupEntry<'_>> {
    Ok(wgpu::BindGroupEntry {
        binding,
        resource: buffer
            .ok_or_else(|| Error::Gpu(format!("reaction binding {binding} is not uploaded")))?
            .as_entire_binding(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_core::BodyId;
    use molla_math::{Transform, Vec3};
    use molla_sim::{BodyParams, ModelBuilder};

    #[test]
    fn validates_empty_multidof_and_capacity_layouts() {
        let empty = ModelBuilder::new().build().unwrap();
        assert_eq!(layout(&empty).unwrap().2, [0, 0, 0, 0]);
        for count in [1, 64, 65] {
            let mut builder = ModelBuilder::new();
            builder.begin_articulation();
            let mut parent = BodyId::NONE;
            for _ in 0..count {
                let child = builder.add_body(BodyParams::new());
                builder
                    .add_joint_ball(parent, child, Transform::IDENTITY, Transform::IDENTITY)
                    .unwrap();
                parent = child;
            }
            assert_eq!(layout(&builder.build().unwrap()).is_ok(), count <= 64);
        }
        let mut builder = ModelBuilder::new();
        let child = builder.add_body(BodyParams::new());
        builder
            .add_joint_revolute(
                BodyId::NONE,
                child,
                Transform::IDENTITY,
                Transform::IDENTITY,
                Vec3::X,
            )
            .unwrap();
        let mut model = builder.build().unwrap();
        model.joint_dof_offset.host_mut().unwrap()[1] = 0;
        assert!(layout(&model).is_err());
    }
}
