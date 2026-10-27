//! Device substep load assembly and stable-handle reaction sample storage.

use crate::{
    distance_gpu::DistanceGpu,
    joint_reactions::{JointReaction, JointReactionSample, ReactionFrame},
};
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_kernels::{ReactionWrenchBuffers, ReactionWrenchDispatch, ReactionWrenchKernel};
use molla_math::{GpuSpatialVector, GpuTransform, Real, SpatialVector, Vec3};
use molla_sim::{
    State,
    runtime::{JointBreakTrigger, JointHandle, RigidScene},
};
use std::collections::BTreeMap;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct JointRecord {
    parent: i32,
    child: u32,
    compiled: i32,
    reversed: u32,
    parent_frame: GpuTransform,
    child_frame: GpuTransform,
    row_start: u32,
    row_count: u32,
    padding: [u32; 2],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Mean {
    child: GpuSpatialVector,
    parent: GpuSpatialVector,
    peaks: [f32; 2],
    count: u32,
    padding: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Sample {
    child: GpuSpatialVector,
    parent: GpuSpatialVector,
    parent_anchor: [f32; 4],
    child_anchor: [f32; 4],
    tag: [u32; 4],
}

const _: () = {
    assert!(std::mem::size_of::<JointRecord>() == 96);
    assert!(std::mem::size_of::<Mean>() == 80);
    assert!(std::mem::size_of::<Sample>() == 112);
};

pub(crate) struct ReactionGpu {
    evaluate: [ReactionWrenchDispatch; 2],
    buffers: ReactionWrenchBuffers,
    assemble: wgpu::ComputePipeline,
    store: wgpu::ComputePipeline,
    halt: wgpu::ComputePipeline,
    groups: [wgpu::BindGroup; 2],
    params: wgpu::Buffer,
    pub means: wgpu::Buffer,
    pub samples: wgpu::Buffer,
    pub break_header: wgpu::Buffer,
    pub break_events: wgpu::Buffer,
    break_limits: wgpu::Buffer,
    owners: Vec<JointHandle>,
    selected: wgpu::Buffer,
    selection: Vec<u32>,
    pub substeps: usize,
    capture: bool,
    bodies: u32,
    dofs: u32,
    pub bind_groups_created: u64,
}

impl ReactionGpu {
    pub fn new(
        engine: &Engine,
        scene: &RigidScene,
        states: &[State; 2],
        distance: Option<&DistanceGpu>,
        loops: Option<&crate::loop_forces_gpu::LoopGpu>,
        delta: &wgpu::Buffer,
        external_loads: Option<&wgpu::Buffer>,
        status: &wgpu::Buffer,
        substeps: usize,
        capture: bool,
    ) -> Result<Self> {
        if substeps == 0 {
            return Err(Error::Gpu("reaction substep count must be nonzero".into()));
        }
        u32::try_from(substeps)
            .map_err(|_| Error::Gpu("reaction substep count overflow".into()))?;
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let model = scene.model();
        let kernel = ReactionWrenchKernel::new(engine, model)?;
        let mut buffers = kernel.create_buffers(engine)?;
        if model.body_count != 0 {
            buffers.boundary_acceleration = model
                .body_kinematic_acceleration
                .device_buffer_current()?
                .clone();
        }
        let evaluate = [
            kernel.prepare(engine, model, &states[0], &buffers)?,
            kernel.prepare(engine, model, &states[1], &buffers)?,
        ];
        let mut owners = Vec::new();
        let mut records = Vec::new();
        let mut rows = Vec::<[u32; 2]>::new();
        let mut rows_by_owner = BTreeMap::<JointHandle, Vec<[u32; 2]>>::new();
        if let Some(distance) = distance {
            for (index, (&owner, &cone)) in distance
                .owners
                .iter()
                .zip(&distance.angular_rows)
                .enumerate()
            {
                if let Some(owner) = owner {
                    rows_by_owner
                        .entry(owner)
                        .or_default()
                        .push([index as u32, u32::from(cone)]);
                }
            }
        }
        if let Some(loops) = loops {
            for (index, &owner) in loops.owners.iter().enumerate() {
                rows_by_owner
                    .entry(owner)
                    .or_default()
                    .push([index as u32, 2]);
            }
        }
        for (id, joint) in scene.joints() {
            if !crate::distance_constraints::enabled(scene, id, joint) {
                continue;
            }
            let compiled = scene.joint_index(id).map(|index| index.index());
            let start = rows.len();
            if compiled.is_none()
                && let Some(references) = rows_by_owner.get(&id)
            {
                rows.extend_from_slice(references);
            }
            let child = scene.body_index(joint.child).unwrap().index();
            records.push(JointRecord {
                parent: joint
                    .parent
                    .map_or(-1, |body| scene.body_index(body).unwrap().index() as i32),
                child: child as u32,
                compiled: compiled.map_or(-1, |joint| joint as i32),
                reversed: u32::from(compiled.is_some_and(|joint| {
                    model.joint_child.host().unwrap()[joint] as usize != child
                })),
                parent_frame: joint.frame_parent.into(),
                child_frame: joint.frame_child.into(),
                row_start: start as u32,
                row_count: (rows.len() - start) as u32,
                padding: [
                    u32::from(
                        crate::rigid_loop_constraints::axis_kind(joint.kind)
                            && scene.joint_softness(id)?.is_none(),
                    ),
                    0,
                ],
            });
            owners.push(id);
        }
        let sample_count = owners
            .len()
            .checked_mul(if capture { substeps } else { 0 })
            .and_then(|count| u32::try_from(count).ok())
            .ok_or_else(|| Error::Gpu("reaction sample count overflow".into()))?
            as usize;
        let limit = device.limits().max_storage_buffer_binding_size;
        if model
            .body_count
            .max(model.joint_dof_total)
            .max(owners.len())
            .div_ceil(64)
            > device.limits().max_compute_workgroups_per_dimension as usize
        {
            return Err(Error::Gpu(
                "GPU reaction dispatch exceeds device capacity".into(),
            ));
        }
        if sample_count as u64 * 112 > limit
            || owners.len() as u64 * 96 > limit
            || owners.len() as u64 * 112 > limit
            || rows.len() as u64 * 8 > limit
        {
            return Err(Error::Gpu(
                "GPU reaction frame exceeds storage binding capacity".into(),
            ));
        }
        let data = |label, bytes: &[u8], minimum| {
            let result = buffer(device, label, (bytes.len() as u64).max(minimum));
            if !bytes.is_empty() {
                molla_core::gpu_write_buffer!(queue, &result, 0, bytes);
            }
            result
        };
        let record_buffer = data("reaction joint records", bytemuck::cast_slice(&records), 96);
        let row_buffer = data("reaction row references", bytemuck::cast_slice(&rows), 8);
        let mut incident = vec![Vec::new(); model.body_count];
        for (index, record) in records.iter().enumerate() {
            if record.compiled >= 0 || record.row_count == 0 {
                continue;
            }
            incident[record.child as usize].push(index as u32);
            if record.parent >= 0 {
                incident[record.parent as usize].push(index as u32);
            }
        }
        let mut offsets = Vec::with_capacity(model.body_count + 1);
        let mut references = Vec::new();
        offsets.push(0u32);
        for joints in incident {
            references.extend(joints);
            offsets.push(
                u32::try_from(references.len())
                    .map_err(|_| Error::Gpu("reaction incidence count overflow".into()))?,
            );
        }
        let body_offsets = data("reaction body offsets", bytemuck::cast_slice(&offsets), 4);
        let body_joints = data("reaction body joints", bytemuck::cast_slice(&references), 4);
        let zero = buffer(device, "reaction empty input", 32);
        let means = buffer(
            device,
            "reaction frame means",
            (owners.len() as u64 * 80).max(80),
        );
        let samples = buffer(
            device,
            "reaction substep samples",
            (sample_count as u64 * 112).max(112),
        );
        let break_limits = buffer(
            device,
            "reaction break limits",
            (owners.len() as u64 * 8).max(8),
        );
        let break_header = buffer(device, "reaction break header", 8);
        let break_events = buffer(
            device,
            "reaction break events",
            (owners.len() as u64 * 112).max(112),
        );
        let selection = vec![1u32; owners.len()];
        let selected = data(
            "reaction selected joints",
            bytemuck::cast_slice(&selection),
            4,
        );
        let params = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("reaction runtime params"),
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let entries: Vec<_> = (0..24)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 15 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: ![7, 8, 11, 12, 14, 19, 20].contains(&binding),
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reaction runtime layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("reaction runtime pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("runtime reactions"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../molla-kernels/shaders/runtime_reactions.wgsl").into(),
            ),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let assemble = pipeline("assemble");
        let store = pipeline("store");
        let halt = pipeline("halt");
        let groups = std::array::from_fn(|index| {
            let state = &states[index];
            let bindings = [
                &record_buffer,
                &row_buffer,
                distance.map_or(&zero, |d| &d.row_forces),
                state.body_q.device_buffer().unwrap_or(&zero),
                model.body_com.device_buffer().unwrap_or(&zero),
                external_loads.unwrap_or_else(|| state.body_f.device_buffer().unwrap_or(&zero)),
                delta,
                &buffers.acceleration,
                &buffers.external,
                &buffers.joints,
                &selected,
                &means,
                &samples,
                &buffers.status,
                status,
                &params,
                &body_offsets,
                &body_joints,
                &break_limits,
                &break_header,
                &break_events,
                scene
                    .control()
                    .body_external_f
                    .device_buffer()
                    .unwrap_or(&zero),
                loops.map_or(&zero, |l| &l.wrenches),
                loops.map_or(&zero, |l| &l.parent_wrenches),
            ];
            let entries: Vec<_> = bindings
                .iter()
                .enumerate()
                .map(|(binding, value)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: value.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("reaction runtime bindings"),
                layout: &layout,
                entries: &entries,
            })
        });
        Ok(Self {
            evaluate,
            buffers,
            assemble,
            store,
            halt,
            groups,
            params,
            means,
            samples,
            break_header,
            break_events,
            break_limits,
            owners,
            selected,
            selection,
            substeps,
            capture,
            bodies: model.body_count as u32,
            dofs: model.joint_dof_total as u32,
            bind_groups_created: if model.articulation_count == 0 { 3 } else { 7 },
        })
    }

    pub fn sizes(&self) -> [u64; 2] {
        if !self.capture {
            return [0; 2];
        }
        [
            self.owners.len() as u64 * 80,
            self.owners.len() as u64 * self.substeps as u64 * 112,
        ]
    }

    pub(crate) fn activation_layout(&self) -> (&wgpu::Buffer, &wgpu::Buffer, &[JointHandle]) {
        (&self.buffers.active, &self.selected, &self.owners)
    }

    pub fn set_active(
        &mut self,
        queue: &wgpu::Queue,
        scene: &RigidScene,
        selected: &[bool],
        articulations: &[bool],
    ) -> Result<()> {
        let limits: Vec<[f32; 2]> = self
            .owners
            .iter()
            .map(|&owner| {
                let limits = scene.joint_break_limits(owner)?;
                let convert = |value: Option<Real>| -> Result<f32> {
                    let value = value.unwrap_or(Real::INFINITY);
                    let narrowed = value as f32;
                    if value.is_finite()
                        && (!narrowed.is_finite() || (value > 0.0 && narrowed < f32::MIN_POSITIVE))
                    {
                        return Err(Error::Solver(
                            "GPU joint break threshold is outside normal f32 range".into(),
                        ));
                    }
                    Ok(narrowed)
                };
                Ok([convert(limits.force)?, convert(limits.torque)?])
            })
            .collect::<Result<_>>()?;
        if !limits.is_empty() {
            molla_core::gpu_write_buffer!(
                queue,
                &self.break_limits,
                0,
                bytemuck::cast_slice(&limits)
            );
        }
        molla_core::gpu_write_buffer!(queue, &self.break_header, 0, &[0; 8]);
        for (selection, &owner) in self.selection.iter_mut().zip(&self.owners) {
            let joint = scene.joint(owner).unwrap();
            *selection = u32::from(
                selected[scene.body_index(joint.child).unwrap().index()]
                    && joint
                        .parent
                        .is_none_or(|body| selected[scene.body_index(body).unwrap().index()]),
            );
        }
        if !self.selection.is_empty() {
            molla_core::gpu_write_buffer!(
                queue,
                &self.selected,
                0,
                bytemuck::cast_slice(&self.selection)
            );
        }
        if !articulations.is_empty() {
            let mask: Vec<u32> = articulations.iter().copied().map(u32::from).collect();
            molla_core::gpu_write_buffer!(
                queue,
                &self.buffers.active,
                0,
                bytemuck::cast_slice(&mask)
            );
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn dispatch(&self, engine: &Engine, input: usize, dt: Real, substep: usize) {
        self.dispatch_before_halt(engine, input, dt, substep, |_| {});
    }

    pub fn dispatch_before_halt(
        &self,
        engine: &Engine,
        input: usize,
        dt: Real,
        substep: usize,
        before_halt: impl FnOnce(&mut wgpu::CommandEncoder),
    ) {
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        molla_core::gpu_write_buffer!(
            queue,
            &self.params,
            0,
            bytemuck::cast_slice(&[
                (dt as f32).to_bits(),
                (1.0 / self.substeps as f32).to_bits(),
                substep as u32,
                self.owners.len() as u32,
                self.bodies,
                self.dofs,
                u32::from(self.capture),
                0,
            ]),
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        if substep == 0 {
            encoder.clear_buffer(&self.samples, 0, None);
            encoder.clear_buffer(&self.means, 0, None);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.assemble);
            pass.set_bind_group(0, &self.groups[input], &[]);
            pass.dispatch_workgroups(self.bodies.max(self.dofs).div_ceil(64).max(1), 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        self.evaluate[input].encode(&mut encoder);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.store);
            pass.set_bind_group(0, &self.groups[input], &[]);
            pass.dispatch_workgroups((self.owners.len() as u32).div_ceil(64).max(1), 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        before_halt(&mut encoder);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.halt);
            pass.set_bind_group(0, &self.groups[input], &[]);
            pass.dispatch_workgroups(1, 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        queue.submit([encoder.finish()]);
    }

    pub fn decode_breaks(
        &self,
        bytes: &[u8],
        dt: Real,
        boundary: usize,
    ) -> Result<Vec<(JointBreakTrigger, JointReactionSample)>> {
        if !bytes.len().is_multiple_of(112) || bytes.len() / 112 > self.owners.len() {
            return Err(Error::Solver("invalid GPU break event size".into()));
        }
        let mut events = Vec::new();
        for bytes in bytes.chunks_exact(112) {
            let event = bytemuck::pod_read_unaligned::<Sample>(bytes);
            let trigger = match event.tag[1] {
                1 => JointBreakTrigger::Force,
                2 => JointBreakTrigger::Torque,
                3 => JointBreakTrigger::Both,
                _ => return Err(Error::Solver("invalid GPU break trigger".into())),
            };
            if event.tag[0] != 1
                || event.tag[2] as usize >= self.owners.len()
                || event.tag[3] as usize + 1 != boundary
            {
                return Err(Error::Solver("invalid GPU break boundary".into()));
            }
            let child: SpatialVector = event.child.into();
            let parent: SpatialVector = event.parent.into();
            let vector = |v: [f32; 4]| Vec3::new(v[0] as Real, v[1] as Real, v[2] as Real);
            let anchors = [vector(event.parent_anchor), vector(event.child_anchor)];
            if !child.linear.is_finite()
                || !child.angular.is_finite()
                || !parent.linear.is_finite()
                || !parent.angular.is_finite()
                || anchors.iter().any(|v| !v.is_finite())
            {
                return Err(Error::Solver("nonfinite GPU break event".into()));
            }
            events.push((
                trigger,
                JointReactionSample {
                    joint: self.owners[event.tag[2] as usize],
                    child,
                    parent,
                    anchors,
                    substep: boundary - 1,
                    time_offset: boundary as Real * dt,
                    sample_dt: dt,
                },
            ));
        }
        events.sort_by_key(|(_, sample)| sample.joint);
        if events.windows(2).any(|v| v[0].1.joint == v[1].1.joint) {
            return Err(Error::Solver("duplicate GPU break event".into()));
        }
        Ok(events)
    }

    pub fn decode(&self, means: &[u8], samples: &[u8], dt: Real) -> Result<ReactionFrame> {
        let sizes = self.sizes();
        if means.len() as u64 != sizes[0] || samples.len() as u64 != sizes[1] {
            return Err(Error::Solver("invalid GPU reaction frame size".into()));
        }
        let mut result = ReactionFrame::default();
        let sub_dt = dt / self.substeps as Real;
        let vector = |v: [f32; 4]| Vec3::new(v[0] as Real, v[1] as Real, v[2] as Real);
        let finite = |w: SpatialVector| w.linear.is_finite() && w.angular.is_finite();
        for (&owner, bytes) in self.owners.iter().zip(means.chunks_exact(80)) {
            let mean = bytemuck::pod_read_unaligned::<Mean>(bytes);
            if mean.count == 0 {
                continue;
            }
            let child = mean.child.into();
            let parent = mean.parent.into();
            if !finite(child)
                || !finite(parent)
                || mean.peaks.iter().any(|v| !v.is_finite() || *v < 0.0)
                || mean.count as usize > self.substeps
            {
                return Err(Error::Solver("invalid GPU reaction mean".into()));
            }
            result.means.insert(
                owner,
                JointReaction {
                    child,
                    parent,
                    sample_dt: sub_dt * mean.count as Real,
                    peak_force: mean.peaks[0] as Real,
                    peak_torque: mean.peaks[1] as Real,
                },
            );
        }
        for (index, bytes) in samples.chunks_exact(112).enumerate() {
            let sample = bytemuck::pod_read_unaligned::<Sample>(bytes);
            if sample.tag[0] == 0 {
                continue;
            }
            let child = sample.child.into();
            let parent = sample.parent.into();
            let anchors = [vector(sample.parent_anchor), vector(sample.child_anchor)];
            if sample.tag[0] != 1
                || !finite(child)
                || !finite(parent)
                || anchors.iter().any(|v| !v.is_finite())
            {
                return Err(Error::Solver("invalid GPU substep reaction".into()));
            }
            let substep = index / self.owners.len();
            result.samples.push(JointReactionSample {
                joint: self.owners[index % self.owners.len()],
                child,
                parent,
                anchors,
                substep,
                time_offset: (substep + 1) as Real * sub_dt,
                sample_dt: sub_dt,
            });
        }
        Ok(result)
    }
}

fn buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    )
}
