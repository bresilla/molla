use super::precision::RuntimePrecision;
use crate::FeatherstoneSolver;
use crate::contact_publication_gpu::{ContactPublicationBinding, ResidentContactPublication};
use crate::contact_response_gpu::{
    ContactResponseConfig, ContactResponseState, ImpactPositionConfig, ImpactResponseConfig,
    ResidentContactResponse, ResidentImpactPositionResponse, ResidentImpactResponse,
};
use crate::coulomb_disk_gpu::{CoulombDiskConfig, CoulombDiskSolution};
use crate::distance_gpu::{DistanceGpu, ScalarJointPublication};
use crate::impact_constraints_gpu::{ImpactGpuConfig, ImpactGpuSolution};
use crate::impact_episodes_gpu::{
    ImpactEpisodeConfig, ImpactEpisodeLayout, ResidentImpactEpisodes,
};
use crate::impact_merge_gpu::{ImpactMergeConfig, ResidentImpactMerge};
use crate::normal_contact_gpu::{ContactStateBuffers, NormalContactRow, ResidentNormalContacts};
use crate::penalty_contact::{ContactForce, FrictionRegime};
use molla_compute::Engine;
use molla_core::{BodyId, Error, Result, ShapeId};
use molla_geometry::{
    LbvhBroadPhase, RESIDENT_MANIFOLD_POINTS, ResidentCollisionOptions, ResidentNarrowPhase,
    ResidentShapeBounds,
};
use molla_math::{Real, Vec3};
use molla_sim::State;
use molla_sim::runtime::RigidScene;

/// Retained broad-phase pair and per-island contact-response capacities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuContactCapacity {
    pub pairs: u32,
    pub contacts_per_island: u32,
}

impl Default for GpuContactCapacity {
    fn default() -> Self {
        Self {
            pairs: 64,
            contacts_per_island: 64,
        }
    }
}

pub(super) struct RuntimeContacts {
    broad: LbvhBroadPhase,
    bounds: ResidentShapeBounds,
    narrow: ResidentNarrowPhase,
    normal: ResidentNormalContacts,
    episode_layout: ImpactEpisodeLayout,
    identities: wgpu::Buffer,
    episodes: ResidentImpactEpisodes,
    episode_checkpoint: bool,
    episode_layout_checkpoint: Option<ImpactEpisodeLayout>,
    impact: ResidentImpactResponse,
    position: ResidentImpactPositionResponse,
    merge: ResidentImpactMerge,
    projected: State,
    response: ResidentContactResponse,
    reactions: Option<crate::contact_reaction_gpu::ContactReactionGpu>,
    publication: Option<ResidentContactPublication>,
    scalar_publication: Option<ScalarJointPublication>,
    scalar_count: usize,
    force_bindings: [Option<usize>; 2],
    rate_bindings: [Option<usize>; 2],
    external: wgpu::Buffer,
    dt: wgpu::Buffer,
    readback: wgpu::Buffer,
    offsets: [u64; 9],
    substeps: usize,
    shape_count: u32,
    body_count: usize,
    width: usize,
    pub(super) precision: Option<RuntimePrecision>,
}

fn storage(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    )
}

impl RuntimeContacts {
    pub(super) fn new(
        engine: &Engine,
        scene: &RigidScene,
        states: &[State; 2],
        solver: &mut FeatherstoneSolver,
        capacity: GpuContactCapacity,
        reactions: bool,
        previous: Option<&Self>,
        detachment: bool,
    ) -> Result<Option<(Self, Option<DistanceGpu>)>> {
        let model = scene.model();
        let source = scene.state();
        if model.shape_count == 0 {
            return Ok(None);
        }
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let rows = capacity
            .pairs
            .checked_mul(RESIDENT_MANIFOLD_POINTS)
            .filter(|&v| v != 0)
            .ok_or_else(|| Error::Gpu("invalid GPU contact pair capacity".into()))?;
        let limit = device.limits().max_storage_buffer_binding_size;
        let gram_bytes = u64::from(capacity.contacts_per_island)
            .checked_mul(u64::from(capacity.contacts_per_island))
            .and_then(|n| n.checked_mul(16))
            .and_then(|n| n.checked_mul(model.articulation_count.max(1) as u64));
        if capacity.contacts_per_island == 0
            || u64::from(rows) * 128 > limit
            || gram_bytes.is_none_or(|n| n > limit)
        {
            return Err(Error::Gpu("invalid GPU contact island capacity".into()));
        }
        let mut projected = source.clone_host()?;
        projected.upload_to_device(device, queue)?;
        let precision = RuntimePrecision::new(engine, scene, states, solver, Some(&projected))?;
        let empty = storage(device, "contact empty state", 32);
        let mut broad = LbvhBroadPhase::new(
            device.clone(),
            queue.clone(),
            model.shape_count,
            capacity.pairs as usize,
        )?;
        broad.dispatch_no_readback(&molla_geometry::collect_shape_aabbs(model, source, 0.0))?;
        let poses = states
            .each_ref()
            .map(|s| s.body_q.device_buffer().unwrap_or(&empty));
        let residues = if precision.is_some() {
            states
                .iter()
                .map(|s| solver.gpu_position_compensation_buffer(s).cloned())
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let low: Vec<_> = residues.iter().collect();
        let (count, a, b) = broad.pair_buffers();
        let narrow = ResidentNarrowPhase::with_collision_options(
            device,
            model,
            &poses,
            a,
            b,
            count,
            capacity.pairs,
            ResidentCollisionOptions {
                plane_position_residues: &low,
                ..Default::default()
            },
        )?;
        solver.set_gpu_status_buffer(narrow.status_buffer())?;
        let coordinate_residues = if model.joint_coord_total == 0 {
            None
        } else {
            Some(crate::distance_gpu::DistancePrecision {
                coordinates: [
                    solver.gpu_coordinate_compensation_buffer(&states[0])?,
                    solver.gpu_coordinate_compensation_buffer(&states[1])?,
                ],
                positions: [
                    solver.gpu_position_compensation_buffer(&states[0])?,
                    solver.gpu_position_compensation_buffer(&states[1])?,
                ],
            })
        };
        let distance = DistanceGpu::new_for_coupling(
            engine,
            scene,
            states,
            narrow.status_buffer(),
            solver.gpu_velocity_delta_buffer().unwrap(),
            coordinate_residues,
        )?;
        solver.set_external_scalar_bounds(distance.is_some())?;
        let scalar = distance.as_ref().and_then(|d| d.scalar_buffers());
        let scalar_count = scalar.map_or(0, |s| s.capacity);
        let bounds = ResidentShapeBounds::new(
            device,
            model,
            &poses,
            broad.primitive_aabb_buffer(),
            narrow.status_buffer(),
            0.0,
        )?;
        let external = storage(
            device,
            "runtime external contact wrenches",
            (model.body_count as u64 * 32).max(32),
        );
        let normal_states = states.each_ref().map(|s| ContactStateBuffers {
            poses: s.body_q.device_buffer().unwrap_or(&empty),
            velocities: s.body_qd.device_buffer().unwrap_or(&empty),
            external_forces: &external,
        });
        let normal = ResidentNormalContacts::new(
            device,
            model,
            &normal_states,
            a,
            b,
            count,
            narrow.manifold_count_buffer(),
            narrow.manifold_buffer(),
            narrow.status_buffer(),
            capacity.pairs,
        )?;
        let dt = storage(device, "runtime contact step size", 4);
        let response_states = [0, 1].map(|side| ContactResponseState {
            input: &states[side],
            predicted_rates: states[1 - side].joint_qd.device_buffer().unwrap_or(&empty),
        });
        let islands = u32::try_from(model.articulation_count.max(1))
            .ok()
            .and_then(|n| n.checked_add(scalar_count))
            .ok_or_else(|| Error::Gpu("contact island count overflow".into()))?;
        let width = capacity
            .contacts_per_island
            .checked_add(distance.as_ref().map_or(Ok(0), |distance| {
                distance.contact_scalar_capacity(capacity.contacts_per_island)
            })?)
            .ok_or_else(|| Error::Gpu("joint/contact block count overflow".into()))?;
        let mut previous_layout = previous.map(|p| p.episode_layout.clone());
        if detachment && let Some(layout) = &mut previous_layout {
            layout.preserve_after_detachment(scene);
        }
        let episode_layout = ImpactEpisodeLayout::new(scene, previous_layout.as_ref())?;
        let identities = storage(
            device,
            "runtime impact identities",
            model.shape_count as u64 * 4,
        );
        molla_core::gpu_write_buffer!(
            queue,
            &identities,
            0,
            bytemuck::cast_slice(episode_layout.tokens()),
        );
        let episodes = ResidentImpactEpisodes::new(
            device,
            normal.row_buffer(),
            &identities,
            narrow.status_buffer(),
            ImpactEpisodeConfig {
                row_capacity: rows,
                shape_capacity: model.shape_count as u32,
                table_capacity: capacity
                    .pairs
                    .checked_mul(2)
                    .and_then(u32::checked_next_power_of_two)
                    .ok_or_else(|| Error::Gpu("impact episode capacity overflow".into()))?,
                approach_speed: 1e-4,
                normal_similarity: 0.95,
            },
        )?;
        if let Some(previous) = previous {
            let mut encoder = device.create_command_encoder(&Default::default());
            episodes.encode_inherit(device, &mut encoder, &previous.episodes)?;
            queue.submit([encoder.finish()]);
        }
        let impact = ResidentImpactResponse::with_episode_onsets(
            engine,
            model,
            &response_states,
            episodes.impact_row_buffer(),
            &[],
            scalar,
            Some(solver.gpu_velocity_delta_buffer().unwrap()),
            episodes.onset_buffer(),
            &dt,
            narrow.status_buffer(),
            ImpactResponseConfig {
                row_capacity: rows,
                impacts: ImpactGpuConfig {
                    island_capacity: islands,
                    blocks_per_island: width,
                    max_iterations: 8192,
                    velocity_tolerance: 1e-7,
                },
            },
        )?;
        impact
            .islands()
            .set_contact_limit(queue, capacity.contacts_per_island)?;
        let friction_states = states.each_ref().map(|input| ContactResponseState {
            input,
            predicted_rates: impact.corrected_rates_buffer(),
        });
        let response = ResidentContactResponse::with_excluded_impact_groups(
            engine,
            model,
            &friction_states,
            normal.row_buffer(),
            &[],
            scalar,
            Some(solver.gpu_velocity_delta_buffer().unwrap()),
            impact.islands().articulation_islands_buffer(),
            capacity.contacts_per_island,
            &dt,
            narrow.status_buffer(),
            ContactResponseConfig {
                row_capacity: rows,
                disks: CoulombDiskConfig {
                    island_capacity: islands,
                    contacts_per_island: width,
                    max_iterations: 8192,
                    velocity_tolerance: 1e-7,
                },
            },
        )?;
        let merge = ResidentImpactMerge::new(
            engine,
            &impact,
            &response,
            narrow.status_buffer(),
            ImpactMergeConfig {
                contact_rows: rows,
                scalar_rows: scalar_count,
                dofs: model.joint_dof_total as u32,
            },
        )?;
        let position = ResidentImpactPositionResponse::new(
            engine,
            model,
            &states.each_ref(),
            episodes.impact_row_buffer(),
            &[],
            distance.as_ref().and_then(|d| d.scalar_position_buffers()),
            episodes.onset_buffer(),
            narrow.status_buffer(),
            ImpactPositionConfig {
                row_capacity: rows,
                island_capacity: islands,
                blocks_per_island: width,
                max_iterations: 8192,
                position_tolerance: 1e-7,
            },
        )?;
        position
            .islands()
            .set_contact_limit(queue, capacity.contacts_per_island)?;
        let scalar_publication = distance
            .as_ref()
            .map(|d| {
                d.scalar_publication(
                    engine,
                    merge.scalar_solution_buffer(),
                    narrow.status_buffer(),
                )
            })
            .transpose()?;
        let reactions = reactions.then(|| {
            crate::contact_reaction_gpu::ContactReactionGpu::new(
                engine,
                model,
                states,
                crate::contact_reaction_gpu::ContactReactionInputs {
                    rows: normal.row_buffer(),
                    heads: normal.incidence_buffers().0,
                    links: normal.incidence_buffers().1,
                    world_impulses: merge.world_impulse_buffer(),
                    response_delta: merge.velocity_delta_buffer(),
                },
                &external,
                &dt,
                solver.gpu_velocity_delta_buffer().unwrap(),
                narrow.status_buffer(),
                rows,
            )
        });
        let mut bindings = Vec::new();
        let mut force_bindings = [None; 2];
        let mut rate_bindings = [None; 2];
        for side in 0..2 {
            if model.body_count != 0 {
                force_bindings[side] = Some(bindings.len());
                bindings.push(ContactPublicationBinding {
                    source: normal.force_buffer(),
                    destination: states[side].body_f.device_buffer().unwrap(),
                    bytes: model.body_count as u64 * 32,
                });
            }
            if model.joint_dof_total != 0 {
                rate_bindings[side] = Some(bindings.len());
                bindings.push(ContactPublicationBinding {
                    source: response.corrected_rates_buffer(),
                    destination: states[1 - side].joint_qd.device_buffer().unwrap(),
                    bytes: model.joint_dof_total as u64 * 4,
                });
            }
        }
        let publication = if bindings.is_empty() {
            None
        } else {
            Some(ResidentContactPublication::new(
                device,
                narrow.status_buffer(),
                &bindings,
            )?)
        };
        let slots = u64::from(islands) * u64::from(width);
        let mut offsets = [0; 9];
        for (i, bytes) in [
            u64::from(rows) * 128,
            u64::from(rows) * 16,
            slots * 32,
            slots * 4,
            u64::from(islands) * 4,
            slots * 48,
            slots * 4,
            u64::from(islands) * 4,
        ]
        .into_iter()
        .enumerate()
        {
            offsets[i + 1] = offsets[i] + bytes;
        }
        let substeps = solver.substeps.max(1);
        let readback = Self::allocate_readback(device, offsets[8], substeps)?;
        Ok(Some((
            Self {
                broad,
                bounds,
                narrow,
                normal,
                episode_layout,
                identities,
                episodes,
                episode_checkpoint: false,
                episode_layout_checkpoint: None,
                impact,
                position,
                merge,
                projected,
                response,
                reactions,
                publication,
                scalar_publication,
                scalar_count: scalar_count as usize,
                force_bindings,
                rate_bindings,
                external,
                dt,
                readback,
                offsets,
                substeps,
                shape_count: model.shape_count as u32,
                body_count: model.body_count,
                width: width as usize,
                precision,
            },
            distance,
        )))
    }

    fn allocate_readback(
        device: &wgpu::Device,
        stride: u64,
        substeps: usize,
    ) -> Result<wgpu::Buffer> {
        let size = stride
            .checked_mul(substeps as u64)
            .filter(|&n| n <= device.limits().max_buffer_size)
            .ok_or_else(|| Error::Gpu("GPU contact frame readback capacity exceeded".into()))?;
        Ok(molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("runtime contact frame readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }
        ))
    }

    pub(super) fn prepare_frame(
        &mut self,
        engine: &Engine,
        solver: &FeatherstoneSolver,
        states: &[State; 2],
        scene: &RigidScene,
        substeps: usize,
        dt: Real,
    ) -> Result<()> {
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let source = scene.state();
        let layout = ImpactEpisodeLayout::new(scene, Some(&self.episode_layout))?;
        if layout.tokens() != self.episode_layout.tokens() {
            molla_core::gpu_write_buffer!(
                queue,
                &self.identities,
                0,
                bytemuck::cast_slice(layout.tokens())
            );
        }
        self.episode_layout_checkpoint = Some(std::mem::replace(&mut self.episode_layout, layout));
        if substeps > self.substeps {
            self.readback = Self::allocate_readback(device, self.offsets[8], substeps)?;
            self.substeps = substeps;
        }
        molla_core::gpu_write_buffer!(queue, &self.dt, 0, bytemuck::bytes_of(&(dt as f32)));
        let mut encoder = device.create_command_encoder(&Default::default());
        self.episodes.encode_checkpoint(&mut encoder);
        for diagnostic in [
            self.impact.impacts().diagnostic_buffer(),
            self.response.disks().diagnostic_buffer(),
            self.position.diagnostic_buffer(),
        ] {
            encoder.clear_buffer(diagnostic, 0, None);
        }
        if self.body_count != 0 {
            encoder.copy_buffer_to_buffer(
                states[0].body_f.device_buffer().unwrap(),
                0,
                &self.external,
                0,
                self.body_count as u64 * 32,
            );
        }
        queue.submit([encoder.finish()]);
        self.episode_checkpoint = true;
        if let Some(precision) = &mut self.precision {
            precision.prepare_frame(engine, solver, states, source)?;
        }
        Ok(())
    }

    pub(super) fn activation_buffers<'a>(
        &'a self,
        status: &'a wgpu::Buffer,
        velocities: &'a [&'a wgpu::Buffer],
        accelerations: &'a wgpu::Buffer,
    ) -> crate::rigid_contact_activation_gpu::ActivationBuffers<'a> {
        crate::rigid_contact_activation_gpu::ActivationBuffers {
            rows: self.normal.row_buffer(),
            impacts: self.episodes.impact_row_buffer(),
            onsets: self.episodes.onset_buffer(),
            status,
            velocities,
            accelerations,
            identities: Some(&self.identities),
        }
    }

    pub(super) fn collider_identity(
        &self,
        scene: &RigidScene,
        collider: molla_sim::runtime::ColliderHandle,
    ) -> Option<u32> {
        scene
            .collider_indices(collider)?
            .first()
            .and_then(|shape| self.episode_layout.tokens().get(shape.index()))
            .copied()
    }

    pub(super) fn predict(
        &self,
        engine: &Engine,
        side: usize,
        activation: Option<&super::activation::RuntimeActivation>,
    ) -> Result<()> {
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        self.bounds.encode(&mut encoder, side)?;
        self.broad.encode_refit(&mut encoder, self.shape_count)?;
        self.narrow.encode(&mut encoder, side)?;
        self.normal.encode_rows(&mut encoder, side)?;
        self.episodes.encode(&mut encoder);
        if let Some(activation) = activation {
            activation.encode(&mut encoder, side)?;
        }
        self.normal.encode_forces(&mut encoder, side)?;
        if let Some(binding) = self.force_bindings[side] {
            self.publication
                .as_ref()
                .unwrap()
                .encode(&mut encoder, binding)?;
        }
        engine.queue().unwrap().submit([encoder.finish()]);
        Ok(())
    }

    pub(super) fn correct(&self, engine: &Engine, side: usize, step: usize) -> Result<()> {
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        self.encode_response(&mut encoder, side, true)?;
        self.encode_readout(&mut encoder, side, step)?;
        engine.queue().unwrap().submit([encoder.finish()]);
        Ok(())
    }

    fn encode_response(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        side: usize,
        position: bool,
    ) -> Result<()> {
        self.impact.encode(encoder, side)?;
        self.response.encode(encoder, side)?;
        self.merge.encode(encoder);
        if position {
            self.position.encode(encoder, side)?;
        }
        if let Some(binding) = self.rate_bindings[side] {
            self.publication
                .as_ref()
                .unwrap()
                .encode(encoder, binding)?;
        }
        Ok(())
    }

    fn encode_readout(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        side: usize,
        step: usize,
    ) -> Result<()> {
        if let Some(publication) = &self.scalar_publication {
            publication.encode(encoder);
        }
        if let Some(reactions) = &self.reactions {
            reactions.encode(encoder, side);
        }
        for (i, buffer) in [
            self.normal.row_buffer(),
            self.merge.world_impulse_buffer(),
            self.response.disks().solution_buffer(),
            self.response.islands().contact_indices_buffer(),
            self.response.islands().contact_counts_buffer(),
            self.impact.impacts().solution_buffer(),
            self.impact.islands().contact_indices_buffer(),
            self.impact.islands().contact_counts_buffer(),
        ]
        .into_iter()
        .enumerate()
        {
            encoder.copy_buffer_to_buffer(
                buffer,
                0,
                &self.readback,
                step as u64 * self.offsets[8] + self.offsets[i],
                self.offsets[i + 1] - self.offsets[i],
            );
        }
        Ok(())
    }

    pub(super) fn correct_with_prediction(
        &self,
        engine: &Engine,
        distance: &DistanceGpu,
        side: usize,
        dt: Real,
        substeps: usize,
        step: usize,
    ) -> Result<()> {
        distance.retain_prediction_displacement(engine, None)?;
        for iteration in 0..16 {
            distance.prepare_scalar_rows(engine, side, dt, substeps, step)?;
            let mut encoder = engine
                .device()
                .unwrap()
                .create_command_encoder(&Default::default());
            self.encode_response(&mut encoder, side, iteration == 0)?;
            engine.queue().unwrap().submit([encoder.finish()]);
            if iteration == 0 {
                distance.retain_prediction_displacement(
                    engine,
                    Some(self.position.displacement_buffer()),
                )?;
            }
        }
        distance.verify_contact_prediction(engine, side)?;
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        self.encode_readout(&mut encoder, side, step)?;
        engine.queue().unwrap().submit([encoder.finish()]);
        Ok(())
    }

    pub(super) fn readback_bytes(&self, substeps: usize) -> u64 {
        self.offsets[8] * substeps as u64
    }

    pub(super) fn drift(
        &mut self,
        solver: &mut FeatherstoneSolver,
        scene: &RigidScene,
        input: &State,
        output: &mut State,
        dt: Real,
    ) -> Result<()> {
        if self.body_count == 0 {
            return solver.drift_gpu(scene.model(), input, scene.control(), dt, output);
        }
        solver.apply_position_displacement_gpu(
            scene.model(),
            input,
            scene.control(),
            self.position.displacement_buffer(),
            &mut self.projected,
        )?;
        solver.drift_gpu(scene.model(), &self.projected, scene.control(), dt, output)
    }

    pub(super) fn rollback_episodes(&mut self, engine: &Engine) {
        if self.episode_checkpoint {
            let mut encoder = engine
                .device()
                .unwrap()
                .create_command_encoder(&Default::default());
            self.episodes.encode_rollback(&mut encoder);
            engine.queue().unwrap().submit([encoder.finish()]);
        }
        if let Some(layout) = self.episode_layout_checkpoint.take() {
            molla_core::gpu_write_buffer!(
                engine.queue().unwrap(),
                &self.identities,
                0,
                bytemuck::cast_slice(layout.tokens()),
            );
            self.episode_layout = layout;
        }
        self.episode_checkpoint = false;
    }

    pub(super) fn commit_episodes(&mut self, scene: &RigidScene) {
        self.episode_layout.record_published(scene);
        self.episode_layout_checkpoint = None;
        self.episode_checkpoint = false;
    }

    pub(super) fn fault_diagnostics(&self, engine: &Engine) -> Result<String> {
        let device = engine.device().unwrap();
        let sources = [
            self.impact.impacts().diagnostic_buffer(),
            self.response.disks().diagnostic_buffer(),
            self.position.diagnostic_buffer(),
        ];
        let readback = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("runtime contact failure diagnostics"),
                size: sources.iter().map(|b| b.size()).sum(),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut offset = 0;
        for source in sources {
            encoder.copy_buffer_to_buffer(source, 0, &readback, offset, source.size());
            offset += source.size();
        }
        engine.queue().unwrap().submit([encoder.finish()]);
        super::map_read(device, &readback)?;
        let mapped = molla_core::gpu_readback!(readback.slice(..));
        let mut records = Vec::new();
        let mut offset = 0;
        for (stage, source) in ["impact", "friction", "position"].into_iter().zip(sources) {
            for (island, bytes) in mapped[offset..offset + source.size() as usize]
                .chunks_exact(16)
                .enumerate()
            {
                let record: [u32; 4] = bytemuck::pod_read_unaligned(bytes);
                if record[0] != 0 {
                    records.push(format!(
                        "{stage} island={island} code={} iterations={} residual={} detail={}",
                        record[0],
                        record[1],
                        f32::from_bits(record[2]),
                        record[3]
                    ));
                }
            }
            offset += source.size() as usize;
        }
        drop(mapped);
        readback.unmap();
        Ok(records.join("; "))
    }

    pub(super) fn projection_reference(&self) -> &State {
        &self.projected
    }

    pub(super) fn momentum_correction(&self) -> &wgpu::Buffer {
        self.merge.velocity_delta_buffer()
    }

    pub(super) fn reaction_buffers(&self) -> Option<(&wgpu::Buffer, &wgpu::Buffer)> {
        self.reactions.as_ref().map(|r| (&r.loads, &r.delta))
    }

    pub(super) fn readings(
        &self,
        engine: &Engine,
        substeps: usize,
        dt: Real,
        completion: Option<&super::soil_completion::CompletionDeadline<'_>>,
    ) -> Result<Vec<ContactForce>> {
        super::map_read_with_completion(engine.device().unwrap(), &self.readback, completion)?;
        let mapped = molla_core::gpu_readback!(self.readback.slice(..));
        let result = self.decode(&mapped[..self.readback_bytes(substeps) as usize], dt);
        drop(mapped);
        self.readback.unmap();
        result
    }

    fn decode(&self, bytes: &[u8], dt: Real) -> Result<Vec<ContactForce>> {
        let invalid = || Error::Gpu("invalid GPU contact frame readout".into());
        let vector = |v: [f32; 4]| Vec3::new(v[0] as Real, v[1] as Real, v[2] as Real);
        let mut result = Vec::new();
        for frame in bytes.chunks_exact(self.offsets[8] as usize) {
            let section = |i: usize| &frame[self.offsets[i] as usize..self.offsets[i + 1] as usize];
            let rows: Vec<NormalContactRow> = section(0)
                .chunks_exact(128)
                .map(bytemuck::pod_read_unaligned)
                .collect();
            let impulses: Vec<[f32; 4]> = section(1)
                .chunks_exact(16)
                .map(bytemuck::pod_read_unaligned)
                .collect();
            let solutions: Vec<CoulombDiskSolution> = section(2)
                .chunks_exact(32)
                .map(bytemuck::pod_read_unaligned)
                .collect();
            let indices: Vec<u32> = section(3)
                .chunks_exact(4)
                .map(bytemuck::pod_read_unaligned)
                .collect();
            let mut regimes = vec![(FrictionRegime::Inactive, 0.0); rows.len()];
            for (island, chunk) in section(4).chunks_exact(4).enumerate() {
                let count = bytemuck::pod_read_unaligned::<u32>(chunk) as usize;
                if count > self.width {
                    return Err(invalid());
                }
                for slot in island * self.width..island * self.width + count {
                    let row = indices[slot] as usize;
                    if row >= rows.len()
                        && row < rows.len() + self.scalar_count
                        && matches!(solutions[slot].regime, 4..=6)
                    {
                        continue;
                    }
                    let Some(normal) = rows.get(row) else {
                        return Err(invalid());
                    };
                    let regime = match solutions[slot].regime {
                        2 => FrictionRegime::Sticking,
                        3 => FrictionRegime::Sliding,
                        _ => return Err(invalid()),
                    };
                    let load = normal.normal_force[3] as Real * (dt as f32) as Real;
                    let cap = solutions[slot].applied_cap as Real;
                    let static_gap = (cap - normal.coefficients[0] as Real * load).abs();
                    let dynamic_gap = (cap - normal.coefficients[1] as Real * load).abs();
                    let coefficient =
                        normal.coefficients[usize::from(dynamic_gap < static_gap)] as Real;
                    regimes[row] = (regime, coefficient);
                }
            }
            let impact_solutions: Vec<ImpactGpuSolution> = section(5)
                .chunks_exact(48)
                .map(bytemuck::pod_read_unaligned)
                .collect();
            let impact_indices: Vec<u32> = section(6)
                .chunks_exact(4)
                .map(bytemuck::pod_read_unaligned)
                .collect();
            for (island, chunk) in section(7).chunks_exact(4).enumerate() {
                let count = bytemuck::pod_read_unaligned::<u32>(chunk) as usize;
                if count > self.width {
                    return Err(invalid());
                }
                for slot in island * self.width..island * self.width + count {
                    let row = impact_indices[slot] as usize;
                    if row >= rows.len() && row < rows.len() + self.scalar_count {
                        continue;
                    }
                    let Some(normal) = rows.get(row) else {
                        return Err(invalid());
                    };
                    let solution = impact_solutions[slot];
                    if solution.diagnostics[0] > 1
                        || solution
                            .impulse
                            .iter()
                            .chain(&solution.velocity)
                            .any(|v| !v.is_finite())
                    {
                        return Err(invalid());
                    }
                    let slip = solution.velocity[1].hypot(solution.velocity[2]);
                    let regime = if normal.coefficients[0] == 0. {
                        FrictionRegime::Inactive
                    } else if slip <= 1e-4 {
                        FrictionRegime::Sticking
                    } else {
                        FrictionRegime::Sliding
                    };
                    regimes[row] = (
                        regime,
                        normal.coefficients[solution.diagnostics[0] as usize] as Real,
                    );
                }
            }
            for (i, row) in rows.iter().enumerate() {
                if row.coefficients[3] == 0.0 && row.force_on_b[3] != -1.0 {
                    continue;
                }
                if row.endpoints[..2].iter().any(|&v| v >= self.shape_count)
                    || row.endpoints[2..]
                        .iter()
                        .any(|&v| v != u32::MAX && v as usize >= self.body_count)
                {
                    return Err(invalid());
                }
                let impulse = vector(impulses[i]);
                let normal = vector(row.normal_force);
                let impact_normal = impulse.dot(normal);
                let normal_force = row.normal_force[3] as Real + impact_normal / dt;
                let tangent_impulse = impulse - normal * impact_normal;
                let force = vector(row.force_on_b) + impulse / dt;
                if !force.is_finite()
                    || !vector(row.point_depth).is_finite()
                    || !vector(row.normal_force).is_finite()
                {
                    return Err(invalid());
                }
                result.push(ContactForce {
                    bodies: [row.endpoints[2], row.endpoints[3]]
                        .map(|v| (v != u32::MAX).then_some(BodyId(v))),
                    colliders: [
                        Some(ShapeId(row.endpoints[0])),
                        Some(ShapeId(row.endpoints[1])),
                    ],
                    point: vector(row.point_depth),
                    normal: vector(row.normal_force),
                    penetration: row.point_depth[3] as Real,
                    friction: row.coefficients[1] as Real,
                    static_friction: row.coefficients[0] as Real,
                    friction_regime: regimes[i].0,
                    applied_friction: regimes[i].1,
                    tangent_impulse,
                    normal_force,
                    sample_dt: dt,
                    normal_impulse: normal_force * dt,
                    force_on_b: force,
                    acted: force.length_squared() > 0.0,
                });
            }
        }
        Ok(result)
    }
}
