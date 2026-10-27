//! Explicit MPM grid projection with resident articulated mass inputs.

use super::*;
use crate::soil_contact_projection_gpu::{
    SoilBoundaryScan, SoilContactProjectionGpu, SoilProjectionCapacity, SoilProjectionInputs,
    SoilRatePublicationGpu, SoilRatePublicationInputs,
};

pub const MPM_COUPLING_CAPACITY_FAULT: u32 = 16;
pub const MPM_COUPLING_BOUNDARY_FAULT: u32 = 32;

#[cfg(any(test, feature = "ordered-sticking-nodes"))]
#[path = "soil_node_order_gpu.rs"]
mod node_order;

/// Fixed body order and world-COM Jacobians, matching the prescribed solid buffers.
pub struct MpmMassCouplingInputs<'a> {
    pub mass: &'a wgpu::Buffer,
    pub jacobians: &'a wgpu::Buffer,
    pub free_rates: &'a wgpu::Buffer,
    pub dofs: u32,
    pub nodes_per_body: u32,
    pub max_owned_bytes: u64,
}

/// Experimental sticking-volume projection; not a RigidWorld scheduler.
/// The caller supplies free machine rates and mass/Jacobians at the current
/// solid configuration, checks GPU status before committing, then publishes
/// corrected rates and drifts the machine. The reported wrench is observational:
/// do not apply it again after adopting the projected rates.
/// Fixed-grid inward normals constrain the coupled solve. Tangential boundary
/// damping remains in free-grid prediction; general frictional contact is absent.
pub struct PreparedMpmMassCoupling {
    #[cfg(feature = "ordered-sticking-nodes")]
    node_order: node_order::SoilNodeOrderGpu,
    #[cfg(test)]
    trace_geometry: [wgpu::Buffer; 3],
    engine: Arc<Engine>,
    step_epoch: Arc<()>,
    grid_window_cells: [i32; 3],
    rigid_epoch: Arc<()>,
    _lease: Arc<()>,
    grid: wgpu::Buffer,
    nodes: wgpu::Buffer,
    spans: wgpu::Buffer,
    params: wgpu::Buffer,
    indirect: wgpu::Buffer,
    projection: SoilContactProjectionGpu,
    particle_fault: particle_fault_bridge::ParticleFaultBridge,
    boundary_scan: SoilBoundaryScan,
    pipelines: [wgpu::ComputePipeline; 5],
    group: wgpu::BindGroup,
    bodies: u32,
    capacity: u32,
    total: u32,
    bytes: u64,
}

fn invalid(message: &str) -> Error {
    Error::Solver(format!("MPM mass coupling: {message}"))
}

impl PreparedMpmMassCoupling {
    pub(crate) fn required_bytes(
        soil: &MpmSimulation,
        dofs: u32,
        bodies: u32,
        nodes_per_body: u32,
    ) -> Result<u64> {
        if !(1..=32).contains(&dofs) || !(1..=64).contains(&bodies) {
            return Err(invalid("unsupported projection width or body count"));
        }
        let count = bodies
            .checked_mul(nodes_per_body)
            .filter(|&n| n > 0)
            .ok_or_else(|| invalid("node capacity overflow"))?;
        let n = u64::from(dofs);
        Ok(soil.g_mass.size()
            + u64::from(count) * 32 * (1 + u64::from(cfg!(feature = "ordered-sticking-nodes")))
            + u64::from(bodies) * 168
            + n * n * 4
            + n * 8
            + 96)
    }

    #[cfg(test)]
    pub(crate) fn trace_geometry(&self) -> [wgpu::Buffer; 3] {
        self.trace_geometry.clone()
    }

    #[cfg(test)]
    pub(crate) fn prepare_projection_replay_capture(
        &mut self,
    ) -> crate::soil_contact_projection_gpu::SoilProjectionReplay {
        self.projection.prepare_replay_capture(&self.engine)
    }

    pub fn prepare_machine_publication(
        &self,
        input: SoilRatePublicationInputs<'_>,
    ) -> Result<SoilRatePublicationGpu> {
        self.projection.prepare_publication(&self.engine, input)
    }

    pub fn new(
        engine: Arc<Engine>,
        soil: &mut MpmSimulation,
        input: MpmMassCouplingInputs<'_>,
    ) -> Result<Self> {
        Self::new_with_boundary_scan(engine, soil, input, SoilBoundaryScan::Full)
    }

    pub fn boundary_scan(&self) -> SoilBoundaryScan {
        self.boundary_scan
    }

    /// Prepare immutable boundary enumeration for this coupling's lifetime.
    pub fn new_with_boundary_scan(
        engine: Arc<Engine>,
        soil: &mut MpmSimulation,
        input: MpmMassCouplingInputs<'_>,
        boundary_scan: SoilBoundaryScan,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| invalid("GPU engine required"))?;
        crate::kernels::validate_soil_device(device, &soil.device)?;
        Self::validate_mode(soil)?;
        let bodies = soil.rigid_count;
        let total = u32::try_from(soil.g_mass.size() / 4)
            .map_err(|_| invalid("physical grid size overflow"))?;
        let count = bodies
            .checked_mul(input.nodes_per_body)
            .filter(|&n| n > 0)
            .ok_or_else(|| invalid("node capacity overflow"))?;
        let bytes = Self::required_bytes(soil, input.dofs, bodies, input.nodes_per_body)?;
        #[cfg(feature = "ordered-sticking-nodes")]
        node_order::SoilNodeOrderGpu::validate(device, input.nodes_per_body)?;
        let n = u64::from(input.dofs);
        let projection_bytes = u64::from(bodies) * 160 + n * n * 4 + n * 8 + 20;
        let extra = bytes - projection_bytes;
        if bytes > input.max_owned_bytes
            || u64::from(count) * 32 > device.limits().max_storage_buffer_binding_size
            || u64::from(total) * 4 > device.limits().max_storage_buffer_binding_size
            || bodies > device.limits().max_compute_workgroups_per_dimension
            || device.limits().max_storage_buffers_per_shader_stage < 13
            || (boundary_scan == SoilBoundaryScan::ExperimentalCached
                && device.limits().max_compute_workgroup_storage_size < 2048)
            || total.max(bodies).div_ceil(64) > device.limits().max_compute_workgroups_per_dimension
        {
            return Err(invalid("coupling capacity or budget exceeded"));
        }
        let buffer = |label, size, usage| {
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
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let nodes = buffer("MPM projected contact nodes", u64::from(count) * 32, usage);
        let spans = buffer("MPM projected contact spans", u64::from(bodies) * 8, usage);
        let owners = buffer("MPM projected contact owners", u64::from(total) * 4, usage);
        let params = buffer(
            "MPM mass coupling parameters",
            64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let indirect = buffer(
            "MPM projected particle dispatch",
            12,
            usage | wgpu::BufferUsages::INDIRECT,
        );
        let prepare_projection = match boundary_scan {
            SoilBoundaryScan::Full => SoilContactProjectionGpu::new_with_fixed_boundaries,
            SoilBoundaryScan::ExperimentalCached => {
                SoilContactProjectionGpu::new_with_fixed_boundaries_cached
            }
        };
        let projection = prepare_projection(
            &engine,
            SoilProjectionInputs {
                nodes: &nodes,
                spans: &spans,
                mass: input.mass,
                jacobians: input.jacobians,
                free_rates: input.free_rates,
            },
            SoilProjectionCapacity {
                dofs: input.dofs,
                bodies,
                nodes: count,
                max_owned_bytes: input.max_owned_bytes - extra,
            },
        )?;
        #[cfg(feature = "ordered-sticking-nodes")]
        let node_order = node_order::SoilNodeOrderGpu::new(
            device,
            &nodes,
            &spans,
            projection.status_buffer(),
            input.nodes_per_body,
            bodies,
        )?;
        #[cfg(feature = "ordered-sticking-nodes")]
        let gather_nodes = node_order.input();
        #[cfg(not(feature = "ordered-sticking-nodes"))]
        let gather_nodes = &nodes;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MPM mass coupling"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/mpm_mass_coupling.wgsl").into(),
            ),
        });
        let entries: Vec<_> = (0..13)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 11 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: [1, 2, 6, 7].contains(&binding),
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MPM mass coupling"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MPM mass coupling"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = ["reset", "gather", "apply", "publish", "finish"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[
                        ("ACTIVE_GRID", f64::from(soil.use_active_grid)),
                        ("SPARSE_GRID", f64::from(soil.sparse_grid)),
                        (
                            "METADATA_WORD",
                            (soil.active_tiles.metadata_offset / 4) as f64,
                        ),
                        (
                            "TILE_IDS_WORD",
                            (soil.active_tiles.metadata_offset / 4
                                + 4
                                + u64::from(soil.active_tiles.map_capacity)
                                    * if soil.sparse_grid { 2 } else { 1 })
                                as f64,
                        ),
                    ],
                    ..Default::default()
                },
                cache: None,
            })
        });
        let buffers = [
            &soil.g_velocity,
            &soil.rigid_center,
            &soil.rigid_vel,
            gather_nodes,
            &spans,
            &owners,
            projection.motion_buffer(),
            projection.impulses_buffer(),
            &soil.rigid_wrench,
            projection.status_buffer(),
            &indirect,
            &params,
            &soil.g_kdiag,
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MPM mass coupling"),
            layout: &layout,
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let lease = soil.rigid_bridge.upgrade().unwrap_or_else(|| Arc::new(()));
        soil.rigid_bridge = Arc::downgrade(&lease);
        let bytes = extra + projection.owned_buffer_bytes();
        let particle_fault =
            particle_fault_bridge::ParticleFaultBridge::new(soil, projection.status_buffer());
        Ok(Self {
            #[cfg(feature = "ordered-sticking-nodes")]
            node_order,
            particle_fault,
            #[cfg(test)]
            trace_geometry: [
                soil.rigid_center.clone(),
                soil.rigid_vel.clone(),
                owners.clone(),
            ],
            engine,
            step_epoch: soil.step_epoch.clone(),
            grid_window_cells: soil.grid_window_cells,
            rigid_epoch: soil.rigid_epoch.clone(),
            _lease: lease,
            grid: soil.g_velocity.clone(),
            nodes,
            spans,
            params,
            indirect,
            projection,
            boundary_scan,
            pipelines,
            group,
            bodies,
            capacity: input.nodes_per_body,
            total,
            bytes,
        })
    }

    pub(super) fn validate_mode(soil: &MpmSimulation) -> Result<()> {
        crate::soil_candidates_gpu::validate_node_window(
            soil.cfg.grid_origin,
            soil.cfg.cell_size,
            soil.cfg.grid_size,
            soil.grid_window_cells,
        )?;
        if !soil.prescribed_solids
            || soil.rigid_count == 0
            || !matches!(soil.cfg.time_step, MpmTimeStep::Explicit)
            || soil.cfg.grid_size < 5
            || !soil.cfg.cell_size.is_finite()
            || soil.cfg.cell_size <= 0.0
            || soil.cfg.grid_origin.iter().any(|v| !v.is_finite())
        {
            return Err(invalid("explicit prescribed solids required"));
        }
        if soil.use_particle_order
            && soil
                .particle_order
                .as_ref()
                .is_none_or(|order| order.capacity < soil.particle_count)
        {
            return Err(invalid("particle count exceeds prepared ordering capacity"));
        }
        Ok(())
    }

    /// Queue one projected soil substep, without mapping or waiting.
    /// Capacity/numeric faults suppress G2P/plasticity via GPU indirect dispatch.
    /// Grid scratch may change on a fault and belongs to the caller's rollback.
    pub fn step(&mut self, soil: &mut MpmSimulation, dt: f32) -> Result<()> {
        self.step_with_commands(soil, dt, |_| {}, |_| {})
    }

    pub fn step_profiled(&mut self, soil: &mut MpmSimulation, dt: f32) -> Result<()> {
        self.step_impl(soil, dt, true, |_| {}, |_| {})
    }

    pub(crate) fn step_with_commands(
        &mut self,
        soil: &mut MpmSimulation,
        dt: f32,
        before: impl FnOnce(&mut wgpu::CommandEncoder),
        after: impl FnOnce(&mut wgpu::CommandEncoder),
    ) -> Result<()> {
        self.step_impl(soil, dt, false, before, after)
    }

    fn step_impl(
        &mut self,
        soil: &mut MpmSimulation,
        dt: f32,
        explicit_profile: bool,
        before: impl FnOnce(&mut wgpu::CommandEncoder),
        after: impl FnOnce(&mut wgpu::CommandEncoder),
    ) -> Result<()> {
        self.engine.check_device()?;
        Self::validate_mode(soil)?;
        soil.validate_prescribed_step(dt)?;
        if !Arc::ptr_eq(&self.step_epoch, &soil.step_epoch)
            || self.grid_window_cells != soil.grid_window_cells
            || !Arc::ptr_eq(&self.rigid_epoch, &soil.rigid_epoch)
            || self.grid != soil.g_velocity
            || self.bodies != soil.rigid_count
        {
            return Err(invalid("stale prepared soil configuration"));
        }
        let queue = self.engine.queue().unwrap();
        let words = [
            soil.cfg.grid_origin[0].to_bits(),
            soil.cfg.grid_origin[1].to_bits(),
            soil.cfg.grid_origin[2].to_bits(),
            soil.cfg.cell_size.to_bits(),
            soil.cfg.grid_size,
            self.bodies,
            self.capacity,
            self.total,
            dt.to_bits(),
            soil.particle_count,
            soil.particle_count.div_ceil(WORKGROUP_SIZE),
            0,
            soil.grid_window_cells[0] as u32,
            soil.grid_window_cells[1] as u32,
            soil.grid_window_cells[2] as u32,
            0,
        ];
        molla_core::gpu_write_buffer!(queue, &self.params, 0, bytemuck::cast_slice(&words));
        step_projected(
            &self.engine,
            soil,
            dt,
            explicit_profile,
            &self.indirect,
            before,
            |soil, encoder, _| self.encode(soil, encoder),
            |encoder| self.particle_fault.encode(encoder),
            after,
        )
    }

    fn encode(&self, soil: &MpmSimulation, encoder: &mut wgpu::CommandEncoder) {
        if soil.use_active_grid {
            self.particle_fault.encode(encoder);
        }
        for i in 0..5 {
            if i == 3 && !soil.use_active_grid {
                continue;
            }
            if i == 2 {
                #[cfg(feature = "ordered-sticking-nodes")]
                self.node_order.encode(encoder);
                self.projection.encode(encoder);
            }
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines[i]);
            pass.set_bind_group(0, &self.group, &[]);
            if soil.use_active_grid && (i == 1 || i == 2) {
                pass.dispatch_workgroups_indirect(&soil.active_tiles.indirect, 0);
            } else {
                let groups = if i == 4 {
                    1
                } else if i == 3 || (i == 0 && soil.use_active_grid) {
                    self.bodies.div_ceil(64)
                } else {
                    self.total.max(self.bodies).div_ceil(64)
                };
                pass.dispatch_workgroups(groups, 1, 1);
            }
            molla_core::gpu_metrics::record_dispatch();
        }
    }

    pub fn status_buffer(&self) -> &wgpu::Buffer {
        self.projection.status_buffer()
    }
    pub fn rates_buffer(&self) -> &wgpu::Buffer {
        self.projection.rates_buffer()
    }
    pub fn motion_buffer(&self) -> &wgpu::Buffer {
        self.projection.motion_buffer()
    }
    pub fn impulses_buffer(&self) -> &wgpu::Buffer {
        self.projection.impulses_buffer()
    }
    /// Additional boundary impulse after free-grid prediction, in body order.
    pub fn boundary_impulses_buffer(&self) -> &wgpu::Buffer {
        self.projection.boundary_impulses_buffer()
    }
    pub fn owned_buffer_bytes(&self) -> u64 {
        self.bytes
    }
    pub fn contact_nodes_buffer(&self) -> &wgpu::Buffer {
        &self.nodes
    }
    pub fn body_spans_buffer(&self) -> &wgpu::Buffer {
        &self.spans
    }
}

pub(super) fn step_projected(
    engine: &Engine,
    soil: &mut MpmSimulation,
    dt: f32,
    explicit_profile: bool,
    indirect: &wgpu::Buffer,
    before: impl FnOnce(&mut wgpu::CommandEncoder),
    encode: impl FnOnce(&mut MpmSimulation, &mut wgpu::CommandEncoder, bool),
    finish: impl FnOnce(&mut wgpu::CommandEncoder),
    after: impl FnOnce(&mut wgpu::CommandEncoder),
) -> Result<()> {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    if explicit_profile {
        soil.begin_step_profile(queue, dt, 1)?;
    }
    let profiled = explicit_profile || soil.profile_window_batch(dt, 1);
    if let Some(guard) = &soil.motion_guard {
        guard.upload(queue);
    }
    molla_core::gpu_write_buffer!(
        queue,
        &soil.uniform_buf,
        0,
        bytemuck::bytes_of(&soil.step_uniforms(dt))
    );
    molla_core::gpu_write_buffer!(
        queue,
        &soil.material_table,
        0,
        bytemuck::cast_slice(&soil.step_material_table())
    );
    if let Some(ids) = soil.particle_material_dirty.take() {
        molla_core::gpu_write_buffer!(queue, &soil.p_material, 0, bytemuck::cast_slice(&ids));
    }
    if soil.use_particle_order {
        soil.particle_order.as_ref().unwrap().upload(
            queue,
            &soil.cfg,
            soil.particle_count,
            soil.grid_window_cells,
        );
    }
    if soil.use_active_grid {
        soil.active_tiles.upload(
            queue,
            &soil.cfg,
            soil.particle_count,
            soil.rigid_count,
            soil.grid_window_cells,
        )?;
    }
    let mut encoder = device.create_command_encoder(&Default::default());
    before(&mut encoder);
    #[cfg(test)]
    if soil.active_grid_needs_reset {
        if let Some(transfer) = &soil.compensated_active_transfer {
            transfer.clear(&mut encoder);
        }
    }
    if soil.use_active_grid {
        soil.initialize_active_grid(&mut encoder, profiled);
    } else if soil.active_grid_needs_reset {
        encoder.clear_buffer(
            &soil.active_tiles.list,
            soil.active_tiles.metadata_offset,
            Some(16),
        );
        soil.active_grid_needs_reset = false;
    }
    soil.encode_particle_order(&mut encoder, profiled, 0);
    if soil.use_active_grid {
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            soil.bind_step(&mut pass);
            soil.profile_begin_pass(profiled, &mut pass, MpmProfileStage::GridReset, 0);
            pass.set_pipeline(&soil.active_pipelines[0]);
            pass.dispatch_workgroups_indirect(&soil.active_tiles.indirect, 0);
            molla_core::gpu_metrics::record_dispatch();
            soil.profile_end_pass(profiled, &mut pass);
        }
        soil.profile_begin_encoder(
            profiled,
            &mut encoder,
            MpmProfileStage::ActiveTiles,
            Some(0),
        );
        soil.active_tiles
            .encode_build(&mut encoder, soil.particle_count, soil.rigid_count);
        soil.profile_end_encoder(profiled, &mut encoder);
    }
    for (stage, kind, groups) in [
        (
            MpmProfileStage::GridReset,
            0,
            soil.cfg.grid_size.pow(3).div_ceil(WORKGROUP_SIZE),
        ),
        (
            MpmProfileStage::RigidReset,
            1,
            soil.rigid_count.div_ceil(WORKGROUP_SIZE),
        ),
        (
            MpmProfileStage::ParticleToGrid,
            2,
            soil.particle_count.div_ceil(WORKGROUP_SIZE),
        ),
        (
            MpmProfileStage::GridUpdate,
            3,
            soil.cfg.grid_size.pow(3).div_ceil(WORKGROUP_SIZE),
        ),
    ] {
        if kind == 0 && soil.use_active_grid {
            continue;
        }
        let mut pass = encoder.begin_compute_pass(&Default::default());
        soil.bind_step(&mut pass);
        soil.profile_begin_pass(profiled, &mut pass, stage, 0);
        if kind == 2 {
            soil.encode_p2g(&mut pass);
        } else if kind == 3 && soil.use_active_grid {
            pass.set_pipeline(&soil.active_pipelines[2]);
            pass.dispatch_workgroups_indirect(&soil.active_tiles.indirect, 0);
            molla_core::gpu_metrics::record_dispatch();
        } else {
            pass.set_pipeline(match kind {
                0 => &soil.pipe_reset,
                1 => &soil.pipe_rigid_wrench_reset,
                _ => &soil.pipe_grid,
            });
            pass.dispatch_workgroups(groups, 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        soil.profile_end_pass(profiled, &mut pass);
    }
    let contact_end = soil.profile_begin_scope(
        profiled,
        &mut encoder,
        MpmProfileStage::ContactProjection,
        Some(0),
    );
    encode(soil, &mut encoder, profiled);
    soil.profile_end_scope(&mut encoder, contact_end);
    for (stage, plastic) in soil
        .particle_publication_stages()
        .into_iter()
        .zip([false, true])
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        soil.bind_step(&mut pass);
        soil.profile_begin_pass(profiled, &mut pass, stage, 0);
        soil.encode_projected_particles(&mut pass, indirect, plastic);
        soil.profile_end_pass(profiled, &mut pass);
    }
    finish(&mut encoder);
    if profiled {
        soil.profiler
            .as_mut()
            .unwrap()
            .finish_batch(&mut encoder, 1);
    }
    after(&mut encoder);
    queue.submit([encoder.finish()]);
    Ok(())
}

#[cfg(test)]
#[path = "mpm_mass_coupling_tests.rs"]
mod tests;
