use crate::FeatherstoneSolver;
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_math::{Real, Transform, Vec3};
use molla_sim::runtime::{BodyHandle, JointHandle, RigidScene, SceneStamp};
use molla_sim::{JointType, State};
use std::collections::BTreeMap;

#[path = "rigid_rate_precision_gpu.rs"]
mod rates;

pub(super) fn decode_rates(state: &mut State, pairs: &[u8], body_low: &[u8]) -> Result<()> {
    if pairs.len() != state.joint_qd.len() * 8 || body_low.len() != state.body_qd.len() * 32 {
        return Err(Error::Gpu("invalid runtime paired readback sizes".into()));
    }
    let rates: Vec<Real> = pairs
        .chunks_exact(8)
        .map(|bytes| {
            let parts = bytemuck::pod_read_unaligned::<[f32; 2]>(bytes);
            parts[0] as Real + parts[1] as Real
        })
        .collect();
    let velocities: Vec<_> = state
        .body_qd
        .host()?
        .iter()
        .zip(body_low.chunks_exact(32))
        .map(|(v, bytes)| {
            let low = bytemuck::pod_read_unaligned::<[f32; 8]>(bytes);
            molla_math::SpatialVector::new(
                v.linear + Vec3::new(low[0] as Real, low[1] as Real, low[2] as Real),
                v.angular + Vec3::new(low[4] as Real, low[5] as Real, low[6] as Real),
            )
        })
        .collect();
    if rates.iter().any(|v| !v.is_finite())
        || velocities
            .iter()
            .any(|v| !v.linear.is_finite() || !v.angular.is_finite())
    {
        return Err(Error::Solver("nonfinite runtime paired readback".into()));
    }
    state.joint_qd.host_mut()?.copy_from_slice(&rates);
    state.body_qd.host_mut()?.copy_from_slice(&velocities);
    Ok(())
}

struct Transfers {
    coordinates: Vec<[u32; 4]>,
    positions: Vec<[u32; 4]>,
}

impl Transfers {
    fn rounded(source: &State) -> Result<Self> {
        let entry = |value: Real| {
            let low = (value - value as f32 as Real) as f32;
            if !low.is_finite() {
                return Err(Error::Solver("nonfinite runtime precision input".into()));
            }
            Ok([0, 0, low.to_bits(), 0])
        };
        Ok(Self {
            coordinates: source
                .joint_q
                .host()?
                .iter()
                .map(|&value| entry(value))
                .collect::<Result<_>>()?,
            positions: source
                .body_q
                .host()?
                .iter()
                .flat_map(|pose| pose.position.to_array().into_iter().chain([0.0]))
                .map(entry)
                .collect::<Result<_>>()?,
        })
    }
}

pub(super) struct Layout {
    identities: Vec<SceneStamp>,
    bodies: Vec<BodyHandle>,
    joints: Vec<(
        Option<(JointHandle, SceneStamp)>,
        u32,
        std::ops::Range<usize>,
    )>,
    frames: Vec<(Option<BodyHandle>, Transform, Transform, Vec3, u32)>,
    dofs: Vec<std::ops::Range<usize>>,
}

impl Layout {
    pub fn new(scene: &RigidScene) -> Result<Self> {
        let model = scene.model();
        let mut bodies: Vec<_> = scene.bodies().map(|(id, _)| id).collect();
        bodies.sort_by_key(|&id| scene.body_index(id).unwrap().index());
        let mut owners = vec![None; model.joint_count];
        for (id, _) in scene.joints() {
            if let Some(joint) = scene.joint_index(id) {
                owners[joint.index()] = Some((id, scene.joint_identity_stamp(id).unwrap()));
            }
        }
        let offsets = model.joint_coord_offset.host()?;
        let rate_offsets = model.joint_dof_offset.host()?;
        let mut dofs = vec![0..0; model.body_count];
        let kinds = model.joint_type.host()?;
        let mut joints = vec![(None, 0, 0..0); model.body_count];
        let mut frames = vec![
            (
                None,
                Transform::IDENTITY,
                Transform::IDENTITY,
                Vec3::ZERO,
                0
            );
            model.body_count
        ];
        let parents = model.joint_parent.host()?;
        let xp = model.joint_x_p.host()?;
        let xc = model.joint_x_c.host()?;
        let axes = model.joint_axis.host()?;
        let reversed = model.joint_reversed.host()?;
        for (joint, &child) in model.joint_child.host()?.iter().enumerate() {
            dofs[child as usize] = rate_offsets[joint] as usize..rate_offsets[joint + 1] as usize;
            joints[child as usize] = (
                owners[joint].clone(),
                kinds[joint],
                offsets[joint] as usize..offsets[joint + 1] as usize,
            );
            frames[child as usize] = (
                (parents[joint] >= 0).then(|| bodies[parents[joint] as usize]),
                xp[joint],
                xc[joint],
                axes[joint],
                reversed[joint],
            );
        }
        Ok(Self {
            identities: bodies
                .iter()
                .map(|&body| scene.body_identity_stamp(body).unwrap())
                .collect(),
            bodies,
            joints,
            frames,
            dofs,
        })
    }

    fn remap(
        &self,
        old: &Self,
        source: &State,
        old_q: &[Real],
        old_poses: &[Transform],
    ) -> Result<Transfers> {
        let q = source.joint_q.host()?;
        let poses = source.body_q.host()?;
        if poses.len() != self.bodies.len()
            || q.len()
                != self
                    .joints
                    .iter()
                    .map(|joint| joint.2.end)
                    .max()
                    .unwrap_or(0)
        {
            return Err(Error::Solver(
                "runtime precision state size mismatch".into(),
            ));
        }
        let mut transfers = Transfers::rounded(source)?;
        let indices: BTreeMap<_, _> = old
            .bodies
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();
        for (body, id) in self.bodies.iter().enumerate() {
            let Some(&previous) = indices.get(id) else {
                continue;
            };
            if self.identities[body] != old.identities[previous] {
                continue;
            }
            let same_pose = old_poses.get(previous) == Some(&poses[body]);
            if same_pose {
                for axis in 0..3 {
                    transfers.positions[body * 4 + axis] = [(previous * 4 + axis) as u32, 2, 0, 0];
                }
            }
            let (owner, kind, range) = &self.joints[body];
            let (old_owner, old_kind, old_range) = &old.joints[previous];
            if owner == old_owner
                && kind == old_kind
                && self.frames[body] == old.frames[previous]
                && old_q.get(old_range.clone()) == Some(&q[range.clone()])
            {
                for (to, from) in range.clone().zip(old_range.clone()) {
                    transfers.coordinates[to] = [from as u32, 1, 0, 0];
                }
            }
        }
        Ok(transfers)
    }

    pub fn transfers(
        &self,
        old: &Self,
        source: &State,
        old_q: &[f64],
        has_low: bool,
    ) -> Result<Vec<[u32; 4]>> {
        let invalid = || Error::Solver("incompatible contact detachment precision layout".into());
        if self.identities != old.identities || self.bodies != old.bodies {
            return Err(invalid());
        }
        let coordinates = source.joint_q.host()?;
        let mut transfers = vec![[0; 4]; coordinates.len()];
        for ((owner, kind, range), (old_owner, old_kind, old_range)) in
            self.joints.iter().zip(&old.joints)
        {
            if range.is_empty() {
                continue;
            }
            if owner == old_owner
                && kind == old_kind
                && range.len() == old_range.len()
                && coordinates[range.clone()] == old_q[old_range.clone()]
            {
                for (to, from) in range.clone().zip(old_range.clone()) {
                    transfers[to] = [from as u32, u32::from(has_low), 0, 0];
                }
            } else if owner.is_none() && *kind == JointType::Free as u32 {
                for axis in 0..3 {
                    let to = range.start + axis;
                    let rounding = (coordinates[to] - coordinates[to] as f32 as f64) as f32;
                    if !rounding.is_finite() {
                        return Err(invalid());
                    }
                    transfers[to] = [0, 0, rounding.to_bits(), 0];
                }
            } else {
                return Err(invalid());
            }
        }
        Ok(transfers)
    }
}

pub(super) fn encode(
    engine: &molla_compute::Engine,
    encoder: &mut wgpu::CommandEncoder,
    transfers: &[[u32; 4]],
    previous: Option<[&wgpu::Buffer; 2]>,
    outputs: [&wgpu::Buffer; 2],
) {
    if transfers.is_empty() {
        return;
    }
    let device = engine.device().unwrap();
    let allocate = |label, size| {
        molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        )
    };
    let table = allocate(
        "contact detachment precision map",
        transfers.len() as u64 * 16,
    );
    let empty = allocate("contact detachment empty residue", 4);
    molla_core::gpu_write_buffer!(
        engine.queue().unwrap(),
        &table,
        0,
        bytemuck::cast_slice(transfers)
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("contact detachment precision"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/contact_precision.wgsl").into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("contact detachment precision"),
        layout: None,
        module: &shader,
        entry_point: Some("transfer"),
        compilation_options: Default::default(),
        cache: None,
    });
    for output in outputs {
        let old = previous.unwrap_or([&empty, &empty]);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("contact detachment precision inputs"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[&table, old[0], old[1], output]
                .into_iter()
                .enumerate()
                .map(|(i, buffer)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(transfers.len().div_ceil(64) as u32, 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }
}

fn copy_retained(
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    transfers: &[[u32; 4]],
    previous: [&wgpu::Buffer; 2],
    outputs: [&wgpu::Buffer; 2],
) -> Result<()> {
    if transfers
        .iter()
        .enumerate()
        .any(|(i, m)| m[1] > 2 || (m[1] != 0 && (m[2] != 0 || m[0] as usize != i)))
    {
        return Err(Error::Solver(
            "non-copying same-layout precision transfer".into(),
        ));
    }
    for output in outputs {
        let mut start = 0;
        while start < transfers.len() {
            let mapping = transfers[start];
            if mapping[1] == 0 {
                let mut end = start + 1;
                while end < transfers.len() && transfers[end][1] == 0 {
                    end += 1;
                }
                let constants: Vec<_> = transfers[start..end].iter().map(|m| m[2]).collect();
                molla_core::gpu_write_buffer!(
                    queue,
                    output,
                    start as u64 * 4,
                    bytemuck::cast_slice(&constants)
                );
                start = end;
                continue;
            }
            let mut end = start + 1;
            while end < transfers.len()
                && transfers[end][1] == mapping[1]
                && transfers[end][0] == mapping[0] + (end - start) as u32
            {
                end += 1;
            }
            encoder.copy_buffer_to_buffer(
                previous[mapping[1] as usize - 1],
                mapping[0] as u64 * 4,
                output,
                start as u64 * 4,
                (end - start) as u64 * 4,
            );
            start = end;
        }
    }
    Ok(())
}

pub(super) struct RuntimePrecision {
    layout: Layout,
    checkpoints: Option<[wgpu::Buffer; 2]>,
    published_q: Vec<Real>,
    published_poses: Vec<Transform>,
    last_side: usize,
    prepared: bool,
    rates: Option<rates::RuntimeRates>,
}

impl RuntimePrecision {
    pub(super) fn new(
        engine: &Engine,
        scene: &RigidScene,
        states: &[State; 2],
        solver: &mut FeatherstoneSolver,
        projected: Option<&State>,
    ) -> Result<Option<Self>> {
        let model = scene.model();
        if model.body_count == 0 {
            return Ok(None);
        }
        let mut compensated = states.each_ref().to_vec();
        compensated.extend(projected);
        solver.prepare_gpu_coordinate_compensation(&compensated)?;
        let rates = if solver.gpu_uses_paired_rates() {
            let per_state = (model.joint_dof_total as u64 * 8)
                .checked_add(model.body_count as u64 * 32)
                .ok_or_else(|| Error::Gpu("runtime rate size overflow".into()))?;
            let bytes = per_state
                .checked_mul(compensated.len() as u64)
                .and_then(|n| n.checked_add(model.joint_dof_total as u64 * 8))
                .ok_or_else(|| Error::Gpu("runtime rate size overflow".into()))?;
            solver.prepare_gpu_paired_rates(&compensated, bytes)?;
            Some(rates::RuntimeRates::new(engine, states, solver)?)
        } else {
            None
        };
        let storage = |label, size| {
            molla_core::gpu_create_buffer!(
                engine.device().unwrap(),
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }
            )
        };
        Ok(Some(Self {
            layout: Layout::new(scene)?,
            checkpoints: Some([
                storage(
                    "rigid coordinate checkpoint",
                    model.joint_coord_total.max(1) as u64 * 4,
                ),
                storage("rigid position checkpoint", model.body_count as u64 * 16),
            ]),
            published_q: Vec::new(),
            published_poses: Vec::new(),
            last_side: 0,
            prepared: false,
            rates,
        }))
    }

    pub(super) fn prepare_frame(
        &mut self,
        engine: &Engine,
        solver: &FeatherstoneSolver,
        states: &[State; 2],
        source: &State,
    ) -> Result<()> {
        let rate_transfers = self
            .rates
            .as_ref()
            .map(|rates| rates.plan(&self.layout, source))
            .transpose()?;
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        if let Some(checkpoints) = &self.checkpoints {
            if source.joint_q.host()? != self.published_q
                || source.body_q.host()? != self.published_poses
            {
                if self.published_poses.is_empty() {
                    let transfers = Transfers::rounded(source)?;
                    for (table, outputs) in [
                        (
                            &transfers.coordinates,
                            [
                                solver.gpu_coordinate_compensation_buffer(&states[0])?,
                                solver.gpu_coordinate_compensation_buffer(&states[1])?,
                            ],
                        ),
                        (
                            &transfers.positions,
                            [
                                solver.gpu_position_compensation_buffer(&states[0])?,
                                solver.gpu_position_compensation_buffer(&states[1])?,
                            ],
                        ),
                    ] {
                        copy_retained(
                            engine.queue().unwrap(),
                            &mut encoder,
                            table,
                            [&checkpoints[0], &checkpoints[1]],
                            outputs,
                        )?;
                    }
                } else {
                    let transfers = self.layout.remap(
                        &self.layout,
                        source,
                        &self.published_q,
                        &self.published_poses,
                    )?;
                    for (from, to) in [
                        solver.gpu_coordinate_compensation_buffer(&states[self.last_side])?,
                        solver.gpu_position_compensation_buffer(&states[self.last_side])?,
                    ]
                    .into_iter()
                    .zip(checkpoints)
                    {
                        encoder.copy_buffer_to_buffer(from, 0, to, 0, to.size());
                    }
                    copy_retained(
                        engine.queue().unwrap(),
                        &mut encoder,
                        &transfers.coordinates,
                        [&checkpoints[0], &checkpoints[1]],
                        [
                            solver.gpu_coordinate_compensation_buffer(&states[0])?,
                            solver.gpu_coordinate_compensation_buffer(&states[1])?,
                        ],
                    )?;
                    copy_retained(
                        engine.queue().unwrap(),
                        &mut encoder,
                        &transfers.positions,
                        [&checkpoints[0], &checkpoints[1]],
                        [
                            solver.gpu_position_compensation_buffer(&states[0])?,
                            solver.gpu_position_compensation_buffer(&states[1])?,
                        ],
                    )?;
                }
                self.published_q = source.joint_q.host()?.to_vec();
                self.published_poses = source.body_q.host()?.to_vec();
            } else if self.last_side != 0 {
                for (from, to) in [
                    (
                        solver.gpu_coordinate_compensation_buffer(&states[self.last_side])?,
                        solver.gpu_coordinate_compensation_buffer(&states[0])?,
                    ),
                    (
                        solver.gpu_position_compensation_buffer(&states[self.last_side])?,
                        solver.gpu_position_compensation_buffer(&states[0])?,
                    ),
                ] {
                    encoder.copy_buffer_to_buffer(from, 0, to, 0, from.size());
                }
            }
            for (from, to) in [
                solver.gpu_coordinate_compensation_buffer(&states[0])?,
                solver.gpu_position_compensation_buffer(&states[0])?,
            ]
            .into_iter()
            .zip(checkpoints)
            {
                encoder.copy_buffer_to_buffer(from, 0, to, 0, to.size());
            }
        }
        if let Some(rates) = &mut self.rates {
            rates.prepare_frame(
                engine,
                &mut encoder,
                rate_transfers.flatten(),
                self.last_side,
            )?;
        }
        engine.queue().unwrap().submit([encoder.finish()]);
        self.prepared = true;
        Ok(())
    }

    pub(super) fn rollback(
        &mut self,
        engine: &Engine,
        solver: &FeatherstoneSolver,
        states: &[State; 2],
    ) -> Result<()> {
        if !self.prepared {
            return Ok(());
        }
        if let Some(checkpoints) = &self.checkpoints {
            let mut encoder = engine
                .device()
                .unwrap()
                .create_command_encoder(&Default::default());
            for state in states {
                for (from, to) in checkpoints.iter().zip([
                    solver.gpu_coordinate_compensation_buffer(state)?,
                    solver.gpu_position_compensation_buffer(state)?,
                ]) {
                    encoder.copy_buffer_to_buffer(from, 0, to, 0, from.size());
                }
            }
            if let Some(rates) = &self.rates {
                rates.rollback(&mut encoder);
            }
            engine.queue().unwrap().submit([encoder.finish()]);
        }
        self.last_side = 0;
        self.prepared = false;
        Ok(())
    }

    pub(super) fn committed(&mut self, source: &State, side: usize) -> Result<()> {
        if let Some(rates) = &mut self.rates {
            rates.committed(source)?;
        }
        self.published_q = source.joint_q.host()?.to_vec();
        self.published_poses = source.body_q.host()?.to_vec();
        self.last_side = side;
        self.prepared = false;
        Ok(())
    }

    pub(super) fn restore_frame(
        &mut self,
        engine: &Engine,
        solver: &FeatherstoneSolver,
        states: &[State; 2],
        source: &State,
    ) -> Result<()> {
        self.prepared = true;
        self.rollback(engine, solver, states)?;
        self.committed(source, 0)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn inherit_detachment_precision(
        &mut self,
        engine: &Engine,
        source: &State,
        solver: &FeatherstoneSolver,
        states: &[State; 2],
        previous: &Self,
        old_solver: &FeatherstoneSolver,
        old_states: &[State; 2],
    ) -> Result<()> {
        if self.checkpoints.is_none() {
            return Ok(());
        }
        if source.body_q.host()? != previous.published_poses {
            return Err(Error::Solver(
                "contact detachment changed body poses".into(),
            ));
        }
        let old = &old_states[previous.last_side];
        let old_buffers = if previous.checkpoints.is_some() {
            Some([
                old_solver.gpu_coordinate_compensation_buffer(old)?,
                old_solver.gpu_position_compensation_buffer(old)?,
            ])
        } else {
            None
        };
        let transfers = self.layout.transfers(
            &previous.layout,
            source,
            &previous.published_q,
            old_buffers.is_some(),
        )?;
        let rate_transfers = self
            .rates
            .as_ref()
            .map(|rates| {
                rates.inherit_plan(
                    &self.layout,
                    &previous.layout,
                    source,
                    previous.rates.as_ref(),
                )
            })
            .transpose()?;
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        encode(
            engine,
            &mut encoder,
            &transfers,
            old_buffers,
            [
                solver.gpu_coordinate_compensation_buffer(&states[0])?,
                solver.gpu_coordinate_compensation_buffer(&states[1])?,
            ],
        );
        if let Some([_, positions]) = old_buffers {
            for state in states {
                let to = solver.gpu_position_compensation_buffer(state)?;
                encoder.copy_buffer_to_buffer(positions, 0, to, 0, to.size());
            }
        }
        if let Some(rates) = &self.rates {
            rates.inherit(
                engine,
                &mut encoder,
                rate_transfers.as_ref().unwrap(),
                previous.rates.as_ref(),
                previous.last_side,
            );
        }
        engine.queue().unwrap().submit([encoder.finish()]);
        self.committed(source, 0)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn inherit_rebuild(
        &mut self,
        engine: &Engine,
        source: &State,
        solver: &FeatherstoneSolver,
        states: &[State; 2],
        previous: &Self,
        old_solver: &FeatherstoneSolver,
        old_states: &[State; 2],
    ) -> Result<()> {
        let transfers = self.layout.remap(
            &previous.layout,
            source,
            &previous.published_q,
            &previous.published_poses,
        )?;
        let rate_transfers = self
            .rates
            .as_ref()
            .map(|rates| {
                rates.inherit_plan(
                    &self.layout,
                    &previous.layout,
                    source,
                    previous.rates.as_ref(),
                )
            })
            .transpose()?;
        let old = &old_states[previous.last_side];
        let buffers = Some([
            old_solver.gpu_coordinate_compensation_buffer(old)?,
            old_solver.gpu_position_compensation_buffer(old)?,
        ]);
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        encode(
            engine,
            &mut encoder,
            &transfers.coordinates,
            buffers,
            [
                solver.gpu_coordinate_compensation_buffer(&states[0])?,
                solver.gpu_coordinate_compensation_buffer(&states[1])?,
            ],
        );
        encode(
            engine,
            &mut encoder,
            &transfers.positions,
            buffers,
            [
                solver.gpu_position_compensation_buffer(&states[0])?,
                solver.gpu_position_compensation_buffer(&states[1])?,
            ],
        );
        if let Some(rates) = &self.rates {
            rates.inherit(
                engine,
                &mut encoder,
                rate_transfers.as_ref().unwrap(),
                previous.rates.as_ref(),
                previous.last_side,
            );
        }
        engine.queue().unwrap().submit([encoder.finish()]);
        self.committed(source, 0)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn inherit_precision(
        &mut self,
        engine: &Engine,
        source: &State,
        solver: &FeatherstoneSolver,
        states: &[State; 2],
        previous: &Self,
        old_solver: &FeatherstoneSolver,
        old_states: &[State; 2],
    ) -> Result<()> {
        if self.checkpoints.is_none() || previous.checkpoints.is_none() {
            return Ok(());
        }
        if source.joint_q.host()? != previous.published_q
            || source.body_q.host()? != previous.published_poses
            || self.rates.is_some()
        {
            return self.inherit_rebuild(
                engine, source, solver, states, previous, old_solver, old_states,
            );
        }
        let old = &old_states[previous.last_side];
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        for state in states {
            for (from, to) in [
                (
                    old_solver.gpu_coordinate_compensation_buffer(old)?,
                    solver.gpu_coordinate_compensation_buffer(state)?,
                ),
                (
                    old_solver.gpu_position_compensation_buffer(old)?,
                    solver.gpu_position_compensation_buffer(state)?,
                ),
            ] {
                encoder.copy_buffer_to_buffer(from, 0, to, 0, to.size());
            }
        }
        engine.queue().unwrap().submit([encoder.finish()]);
        self.committed(source, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_math::Mat3;
    use molla_sim::runtime::{BodyDesc, JointDesc};
    use molla_sim::{BodyParams, MassProperties};

    fn body(scene: &mut RigidScene, x: Real) -> BodyHandle {
        scene
            .insert_body(BodyDesc {
                params: BodyParams::new().with_position(Vec3::X * x),
                ..Default::default()
            })
            .unwrap()
    }

    #[test]
    fn initial_and_edited_states_seed_authored_rounding_residues() {
        let mut scene = RigidScene::new().unwrap();
        let keep = body(&mut scene, 4096.0001);
        let edited = body(&mut scene, 4096.0002);
        let old = Layout::new(&scene).unwrap();
        let state = scene.state().clone_host().unwrap();
        let seeded = Transfers::rounded(&state).unwrap();
        for (i, &value) in state.joint_q.host().unwrap().iter().enumerate() {
            let high = value as f32 as Real;
            let low = f32::from_bits(seeded.coordinates[i][2]) as Real;
            assert!((value - high - low).abs() < 2e-11);
        }
        scene
            .set_body_pose(edited, Transform::from_translation(Vec3::X * 4096.0003))
            .unwrap();
        let layout = Layout::new(&scene).unwrap();
        let transfers = layout
            .remap(
                &old,
                scene.state(),
                state.joint_q.host().unwrap(),
                state.body_q.host().unwrap(),
            )
            .unwrap();
        let keep = scene.body_index(keep).unwrap().index();
        assert_eq!(transfers.positions[keep * 4], [(keep * 4) as u32, 2, 0, 0]);
        let edited = scene.body_index(edited).unwrap().index();
        let value = scene.state().body_q.host().unwrap()[edited].position.x;
        let expected = (value - value as f32 as Real) as f32;
        assert_eq!(
            transfers.positions[edited * 4],
            [0, 0, expected.to_bits(), 0]
        );
        assert_eq!(
            transfers.coordinates[layout.joints[edited].2.start],
            [0, 0, expected.to_bits(), 0]
        );
    }

    #[test]
    fn mapping_preserves_survivors_and_resets_new_or_edited_bodies() {
        let mut scene = RigidScene::new().unwrap();
        let retired = body(&mut scene, 0.);
        let keep = body(&mut scene, 1024.);
        let edited = body(&mut scene, 1025.);
        let old = Layout::new(&scene).unwrap();
        let state = scene.state().clone_host().unwrap();
        scene.remove_body(retired).unwrap();
        let new = body(&mut scene, 1024.);
        scene
            .set_body_pose(edited, Transform::from_translation(Vec3::X * 2048.))
            .unwrap();
        let layout = Layout::new(&scene).unwrap();
        let transfers = layout
            .remap(
                &old,
                scene.state(),
                state.joint_q.host().unwrap(),
                state.body_q.host().unwrap(),
            )
            .unwrap();
        let index = scene.body_index(keep).unwrap().index();
        for axis in 0..3 {
            assert_eq!(
                transfers.positions[index * 4 + axis],
                [(4 + axis) as u32, 2, 0, 0]
            );
        }
        for (to, from) in layout.joints[index].2.clone().zip(old.joints[1].2.clone()) {
            assert_eq!(transfers.coordinates[to], [from as u32, 1, 0, 0]);
        }
        for id in [edited, new] {
            let index = scene.body_index(id).unwrap().index();
            assert!(
                transfers.positions[index * 4..index * 4 + 4]
                    .iter()
                    .all(|&v| v == [0; 4])
            );
            assert!(
                transfers.coordinates[layout.joints[index].2.clone()]
                    .iter()
                    .all(|&v| v == [0; 4])
            );
        }
    }

    #[test]
    fn mapping_reseeds_implicit_com_from_published_coordinates() {
        let mut scene = RigidScene::new().unwrap();
        let id = body(&mut scene, 1024.0001);
        let old = Layout::new(&scene).unwrap();
        let state = scene.state().clone_host().unwrap();
        scene
            .set_body_mass_properties(
                id,
                MassProperties {
                    mass: 2.,
                    local_com: Vec3::Y * 0.25,
                    inertia: Mat3::IDENTITY,
                },
            )
            .unwrap();
        let layout = Layout::new(&scene).unwrap();
        let transfers = layout
            .remap(
                &old,
                scene.state(),
                state.joint_q.host().unwrap(),
                state.body_q.host().unwrap(),
            )
            .unwrap();
        for axis in 0..3 {
            let value = scene.state().joint_q.host().unwrap()[axis];
            let low = (value - value as f32 as Real) as f32;
            assert_eq!(transfers.coordinates[axis], [0, 0, low.to_bits(), 0]);
            assert!((value as f32 as Real + low as Real - value).abs() < 2e-11);
        }
        assert!(transfers.coordinates[3..].iter().all(|&v| v == [0; 4]));
    }

    #[test]
    fn detachment_seeds_published_translation_without_adding_old_residue() {
        let mut scene = RigidScene::new().unwrap();
        let id = body(&mut scene, 1025.00017);
        let pose = scene.body_pose(id).unwrap();
        let joint = scene
            .insert_joint(JointDesc {
                parent: None,
                child: id,
                kind: JointType::Fixed,
                axis: Vec3::X,
                frame_parent: pose,
                frame_child: Transform::IDENTITY,
            })
            .unwrap();
        let old = Layout::new(&scene).unwrap();
        let state = scene.state().clone_host().unwrap();
        assert_eq!(scene.model().joint_coord_total, 0);
        scene.break_joints(&[joint]).unwrap();
        assert_eq!(scene.body_pose(id).unwrap(), pose);
        let layout = Layout::new(&scene).unwrap();
        for has_low in [false, true] {
            let transfers = layout
                .transfers(&old, scene.state(), state.joint_q.host().unwrap(), has_low)
                .unwrap();
            for (axis, mapping) in transfers.iter().take(3).enumerate() {
                let value = scene.state().joint_q.host().unwrap()[axis];
                let low = (value - value as f32 as Real) as f32;
                assert_eq!(*mapping, [0, 0, low.to_bits(), 0]);
                assert!((value as f32 as Real + low as Real - value).abs() < 2e-11);
            }
        }
    }

    #[test]
    fn body_provenance_disambiguates_new_scenes_and_divergent_forks() {
        let mut scene = RigidScene::new().unwrap();
        let shared = body(&mut scene, 1024.);
        let mut fork = scene.fork().unwrap();
        let a = body(&mut scene, 1025.);
        let b = body(&mut fork, 1025.);
        assert_eq!(a, b);
        assert_eq!(
            scene.body_identity_stamp(shared),
            fork.body_identity_stamp(shared)
        );
        assert_ne!(scene.body_identity_stamp(a), fork.body_identity_stamp(b));
        let old = Layout::new(&scene).unwrap();
        let q = scene.state().joint_q.host().unwrap();
        let poses = scene.state().body_q.host().unwrap();
        let transfers = Layout::new(&fork)
            .unwrap()
            .remap(&old, fork.state(), q, poses)
            .unwrap();
        assert_eq!(transfers.positions[0], [0, 2, 0, 0]);
        assert!(transfers.positions[4..].iter().all(|&v| v == [0; 4]));
        assert!(transfers.coordinates[7..].iter().all(|&v| v == [0; 4]));
        let mut replacement = RigidScene::new().unwrap();
        assert_eq!(body(&mut replacement, 1024.), shared);
        assert_eq!(body(&mut replacement, 1025.), a);
        let transfers = Layout::new(&replacement)
            .unwrap()
            .remap(&old, replacement.state(), q, poses)
            .unwrap();
        assert!(
            transfers
                .positions
                .iter()
                .chain(&transfers.coordinates)
                .all(|&v| v == [0; 4])
        );
    }

    #[test]
    fn mapping_does_not_copy_residues_into_changed_authored_frames() {
        let mut scene = RigidScene::new().unwrap();
        let child = body(&mut scene, 1.);
        let joint = scene
            .insert_joint(JointDesc {
                parent: None,
                child,
                kind: JointType::Prismatic,
                axis: Vec3::X,
                frame_parent: Transform::IDENTITY,
                frame_child: Transform::IDENTITY,
            })
            .unwrap();
        let old = Layout::new(&scene).unwrap();
        let state = scene.state().clone_host().unwrap();
        scene
            .set_joint_frames(
                joint,
                Transform::from_translation(Vec3::Y),
                Transform::IDENTITY,
            )
            .unwrap();
        let transfers = Layout::new(&scene)
            .unwrap()
            .remap(
                &old,
                scene.state(),
                state.joint_q.host().unwrap(),
                state.body_q.host().unwrap(),
            )
            .unwrap();
        assert_eq!(transfers.coordinates, vec![[0; 4]]);
        assert_eq!(transfers.positions[0], [0, 2, 0, 0]);
    }

    #[test]
    fn independently_inserted_fork_joints_do_not_share_coordinate_residues() {
        let mut scene = RigidScene::new().unwrap();
        let child = body(&mut scene, 1.);
        let mut fork = scene.fork().unwrap();
        let desc = JointDesc {
            parent: None,
            child,
            kind: JointType::Prismatic,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        };
        let a = scene.insert_joint(desc.clone()).unwrap();
        let b = fork.insert_joint(desc).unwrap();
        assert_eq!(a, b);
        assert_ne!(scene.joint_identity_stamp(a), fork.joint_identity_stamp(b));
        let transfers = Layout::new(&fork)
            .unwrap()
            .remap(
                &Layout::new(&scene).unwrap(),
                fork.state(),
                scene.state().joint_q.host().unwrap(),
                scene.state().body_q.host().unwrap(),
            )
            .unwrap();
        assert_eq!(transfers.coordinates, vec![[0; 4]]);
    }

    #[test]
    fn mapping_rejects_a_state_with_incompatible_coordinate_count() {
        let mut scene = RigidScene::new().unwrap();
        body(&mut scene, 1.);
        let layout = Layout::new(&scene).unwrap();
        let empty = RigidScene::new().unwrap();
        assert!(layout.remap(&layout, empty.state(), &[], &[]).is_err());
        let mut state = scene.state().clone_host().unwrap();
        state.joint_q = empty.state().clone_host().unwrap().joint_q;
        assert!(layout.remap(&layout, &state, &[], &[]).is_err());
    }
}
