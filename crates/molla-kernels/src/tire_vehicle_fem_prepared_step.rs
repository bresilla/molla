use super::*;
use crate::{BrushTerrainGpu, PreparedTireVehicleChassisStep};
use molla_compute::Engine;

#[path = "tire_vehicle_fem_cavity.rs"]
mod cavity;

fn invalid(message: &str) -> Error {
    Error::Solver(format!("prepared vehicle FEM: {message}"))
}

/// Retained node → chassis/brush → hub assembly → hub integration schedule.
/// Configuration refresh is explicit; stepping performs no host transfers or waits.
/// Output handles are overwritten by later steps. GPU faults are not rolled back.
pub struct PreparedTireVehicleFemStep {
    chassis: PreparedTireVehicleChassisStep,
    source: wgpu::Buffer,
    node_rows: Vec<TireFemGpuBatchAtlasParamRow>,
    hub_rows: Vec<TireFemGpuAtlasHubParamRow>,
    params: Vec<TireFemGpuParams>,
    node_base: wgpu::Buffer,
    hub_base: wgpu::Buffer,
    bridge_uniform: wgpu::Buffer,
    bridge_config: BrushHubUniforms,
    bridge_io: wgpu::BindGroup,
    bridge_params: wgpu::BindGroup,
    pipelines: [wgpu::ComputePipeline; 3],
    node_workgroups: u32,
    cavity: Option<cavity::CavityFrame>,
}

impl PreparedTireVehicleFemStep {
    /// Includes the chassis and parameter snapshots; excludes existing atlas/maps.
    pub fn required_bytes(wheels: u32) -> u64 {
        PreparedTireVehicleChassisStep::required_bytes(wheels) + 176 * wheels as u64 + 32
    }

    pub fn new(
        engine: &Engine,
        fem_kernel: &TireFemGpuKernel,
        vehicle_kernel: &TireVehicleYawGpuKernel,
        brush_kernel: &TireBrushGpuKernel,
        owner: &TireResidentVehicleFemAtlasOwner,
        terrain: Option<&BrushTerrainGpu<'_>>,
        dt: f32,
        params: &[TireFemGpuParams],
        hub_params: &[TireFemHubParams],
        force_scale: f32,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| invalid("GPU engine required"))?;
        crate::soil_gpu_identity::validate_soil_device(device, &fem_kernel.device)?;
        crate::soil_gpu_identity::validate_soil_device(device, &owner.fem.device)?;
        let n = owner.tire_count();
        if n == 0
            || n > 64
            || owner.fem.tire_count() != n
            || owner.vehicle.wheel_count() != n
            || params.len() != n
            || hub_params.len() != n
            || Self::required_bytes(n as u32) > max_owned_bytes
        {
            return Err(invalid("owner/parameter counts or owned byte budget"));
        }
        let contract = tire_fem_gpu_batch_atlas_dispatch_contract(&owner.fem.atlas)?;
        let chassis = PreparedTireVehicleChassisStep::new(
            engine,
            vehicle_kernel,
            brush_kernel,
            &owner.vehicle,
            terrain,
            dt,
            PreparedTireVehicleChassisStep::required_bytes(n as u32),
        )?;
        let allocate = |size, usage| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("prepared vehicle FEM configuration"),
                    size,
                    usage,
                    mapped_at_creation: false,
                }
            )
        };
        let usage = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let node_base = allocate(112 * n as u64, usage | wgpu::BufferUsages::STORAGE);
        let hub_base = allocate(64 * n as u64, usage);
        let bridge_uniform = allocate(
            32,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let node_rows = params
            .iter()
            .zip(&owner.fem.atlas.param_rows)
            .map(|(p, r)| tire_fem_gpu_batch_atlas_param_row_from_parts(*p, r.counts, r.counts2))
            .collect::<Vec<_>>();
        let hub_rows = hub_params
            .iter()
            .copied()
            .map(tire_fem_gpu_atlas_hub_param_row)
            .collect::<Vec<_>>();
        let bridge_config = Self::bridge_config(n, force_scale);
        let queue = engine.queue().unwrap();
        molla_core::gpu_write_buffer!(queue, &node_base, 0, bytemuck::cast_slice(&node_rows));
        molla_core::gpu_write_buffer!(queue, &hub_base, 0, bytemuck::cast_slice(&hub_rows));
        molla_core::gpu_write_buffer!(
            queue,
            &bridge_uniform,
            0,
            bytemuck::bytes_of(&bridge_config)
        );
        let bridge_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prepared vehicle FEM brush hub bridge"),
            layout: &fem_kernel.atlas_hub_params_from_brush_layout_io,
            entries: &[
                entry(45, &chassis.output().buffer),
                entry(46, &owner.fem.atlas_hub_param_rows_buf),
            ],
        });
        let bridge_params = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prepared vehicle FEM bridge parameters"),
            layout: &fem_kernel.layout_uniforms,
            entries: &[entry(0, &bridge_uniform)],
        });
        Ok(Self {
            chassis,
            source: owner.fem.pos_a.clone(),
            node_rows,
            hub_rows,
            params: params.to_vec(),
            node_base,
            hub_base,
            bridge_uniform,
            bridge_config,
            bridge_io,
            bridge_params,
            pipelines: [
                fem_kernel.atlas_contract_pipeline.clone(),
                fem_kernel.atlas_hub_params_from_brush_pipeline.clone(),
                fem_kernel.atlas_hub_pipeline.clone(),
            ],
            node_workgroups: contract.workgroup_count,
            cavity: None,
        })
    }

    fn bridge_config(n: usize, scale: f32) -> BrushHubUniforms {
        BrushHubUniforms {
            counts: [n as u32, 0, 0, 0],
            params: [finite_f32_or(scale, 1.).clamp(-10., 10.), 0., 0., 0.],
        }
    }

    fn validate_source(&self, owner: &TireResidentVehicleFemAtlasOwner) -> Result<()> {
        if owner.fem.pos_a != self.source || owner.tire_count() != self.params.len() {
            return Err(invalid("foreign or replaced FEM atlas"));
        }
        Ok(())
    }

    pub(crate) fn validate(
        &self,
        engine: &Engine,
        owner: &TireResidentVehicleFemAtlasOwner,
    ) -> Result<()> {
        self.validate_source(owner)?;
        self.chassis.validate(engine, &owner.vehicle)
    }

    /// Refreshes changed fixed-count FEM, hub and wheel rows without allocation.
    /// Earlier submitted steps retain queue order. Raw parameter validity follows legacy APIs.
    pub fn update_configuration(
        &mut self,
        engine: &Engine,
        owner: &TireResidentVehicleFemAtlasOwner,
        params: &[TireFemGpuParams],
        hub_params: &[TireFemHubParams],
        force_scale: f32,
    ) -> Result<()> {
        self.validate_source(owner)?;
        let n = self.params.len();
        if params.len() != n || hub_params.len() != n {
            return Err(invalid("changed parameter count"));
        }
        if self.cavity.is_some() {
            cavity::CavityFrame::validate_pressures(params)?;
        }
        self.chassis
            .update_wheel_configuration(engine, &owner.vehicle)?;
        if let Some(cavity) = &mut self.cavity {
            cavity.update_pressures(engine, params)?;
        }
        let mut node_changed = false;
        let mut hub_changed = false;
        for i in 0..n {
            let saved = &mut self.node_rows[i];
            let row = tire_fem_gpu_batch_atlas_param_row_from_parts(
                params[i],
                saved.counts,
                saved.counts2,
            );
            node_changed |= bytemuck::bytes_of(saved) != bytemuck::bytes_of(&row);
            *saved = row;
            let row = tire_fem_gpu_atlas_hub_param_row(hub_params[i]);
            hub_changed |= bytemuck::bytes_of(&self.hub_rows[i]) != bytemuck::bytes_of(&row);
            self.hub_rows[i] = row;
        }
        self.params.copy_from_slice(params);
        let queue = engine.queue().unwrap();
        if node_changed {
            molla_core::gpu_write_buffer!(
                queue,
                &self.node_base,
                0,
                bytemuck::cast_slice(&self.node_rows)
            );
        }
        if hub_changed {
            molla_core::gpu_write_buffer!(
                queue,
                &self.hub_base,
                0,
                bytemuck::cast_slice(&self.hub_rows)
            );
        }
        let config = Self::bridge_config(n, force_scale);
        if bytemuck::bytes_of(&config) != bytemuck::bytes_of(&self.bridge_config) {
            self.bridge_config = config;
            molla_core::gpu_write_buffer!(
                queue,
                &self.bridge_uniform,
                0,
                bytemuck::bytes_of(&config)
            );
        }
        Ok(())
    }

    /// Submits nine ordered dispatches, then advances all four state parities.
    /// Immutable base hub rows are restored on-device each step before force assembly.
    pub fn step(
        &self,
        engine: &Engine,
        owner: &mut TireResidentVehicleFemAtlasOwner,
    ) -> Result<TireBrushResidentOutput> {
        if self.cavity.is_some() {
            return Err(invalid("cavity mode requires step_cavity_frame"));
        }
        self.step_inner(engine, owner, None)
    }

    fn step_inner(
        &self,
        engine: &Engine,
        owner: &mut TireResidentVehicleFemAtlasOwner,
        indirect: Option<&wgpu::Buffer>,
    ) -> Result<TireBrushResidentOutput> {
        self.validate(engine, owner)?;
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let n = self.params.len() as u64;
        if indirect.is_some() {
            let fem = &owner.fem;
            let (pos, next_pos, vel, next_vel) = if fem.parity {
                (&fem.pos_b, &fem.pos_a, &fem.vel_b, &fem.vel_a)
            } else {
                (&fem.pos_a, &fem.pos_b, &fem.vel_a, &fem.vel_b)
            };
            encoder.copy_buffer_to_buffer(pos, 0, next_pos, 0, pos.size());
            encoder.copy_buffer_to_buffer(vel, 0, next_vel, 0, vel.size());
            PreparedTireVehicleChassisStep::preserve_next_states(&mut encoder, &owner.vehicle);
        }
        encoder.copy_buffer_to_buffer(&self.node_base, 0, &owner.fem.param_rows_buf, 0, 112 * n);
        encoder.copy_buffer_to_buffer(
            &self.hub_base,
            0,
            &owner.fem.atlas_hub_param_rows_buf,
            0,
            64 * n,
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines[0]);
            pass.set_bind_group(
                0,
                if owner.fem.parity {
                    &owner.fem.bg_b_to_a
                } else {
                    &owner.fem.bg_a_to_b
                },
                &[],
            );
            molla_core::gpu_metrics::record_dispatch();
            if let Some(buffer) = indirect {
                pass.dispatch_workgroups_indirect(buffer, 16);
            } else {
                pass.dispatch_workgroups(self.node_workgroups, 1, 1);
            }
        }
        let output =
            self.chassis
                .encode_dispatch(engine, &mut encoder, &owner.vehicle, indirect)?;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines[1]);
            pass.set_bind_group(0, &self.bridge_io, &[]);
            pass.set_bind_group(1, &self.bridge_params, &[]);
            molla_core::gpu_metrics::record_dispatch();
            if let Some(buffer) = indirect {
                pass.dispatch_workgroups_indirect(buffer, 28);
            } else {
                pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
            }
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines[2]);
            pass.set_bind_group(
                0,
                if owner.fem.parity {
                    &owner.fem.atlas_hub_bg_a
                } else {
                    &owner.fem.atlas_hub_bg_b
                },
                &[],
            );
            molla_core::gpu_metrics::record_dispatch();
            if let Some(buffer) = indirect {
                pass.dispatch_workgroups_indirect(buffer, 40);
            } else {
                pass.dispatch_workgroups(n as u32, 1, 1);
            }
        }
        engine.queue().unwrap().submit([encoder.finish()]);
        PreparedTireVehicleChassisStep::advance_parities(&mut owner.vehicle);
        owner.fem.parity = !owner.fem.parity;
        owner.fem.atlas.params.copy_from_slice(&self.params);
        owner.fem.atlas.param_rows.copy_from_slice(&self.node_rows);
        Ok(output)
    }

    /// Enables GPU cavity loads; budget covers additional cavity resources only.
    /// Reconstruct this prepared owner after a cavity fault; no in-place reset exists.
    pub fn prepare_cavities(
        &mut self,
        engine: &Engine,
        owner: &TireResidentVehicleFemAtlasOwner,
        surfaces: &[molla_vehicle::TetraTireFemCavitySurface<'_>],
        max_bytes: u64,
    ) -> Result<()> {
        self.validate(engine, owner)?;
        if self.cavity.is_some() {
            return Err(invalid("cavity mode already prepared"));
        }
        cavity::CavityFrame::validate_pressures(&self.params)?;
        let mut prepared = cavity::CavityFrame::new(
            engine,
            owner,
            &self.node_base,
            self.node_workgroups,
            surfaces,
            max_bytes,
        )?;
        prepared.update_pressures(engine, &self.params)?;
        self.cavity = Some(prepared);
        Ok(())
    }

    /// Evaluates pressure once, then submits guarded FEM/chassis substeps.
    /// Submission is not acceptance: inspect cavity_status_buffer before committing
    /// external frame time. Invalid cavity data latches a fault and preserves current
    /// physical state across parity changes. Other GPU/solver faults are not rolled back.
    pub fn step_cavity_frame(
        &mut self,
        engine: &Engine,
        owner: &mut TireResidentVehicleFemAtlasOwner,
        substeps: u32,
    ) -> Result<TireBrushResidentOutput> {
        self.validate(engine, owner)?;
        if !(1..=4096).contains(&substeps) {
            return Err(invalid("cavity frame substep count"));
        }
        let cavity = self
            .cavity
            .as_mut()
            .ok_or_else(|| invalid("prepare cavity mode first"))?;
        cavity.refresh(engine, owner.fem.parity);
        for _ in 0..substeps {
            self.step_inner(engine, owner, Some(&self.cavity.as_ref().unwrap().control))?;
        }
        Ok(self.chassis.output())
    }

    /// First four u32 words: sticky fault, latest refresh accepted, refresh count, reserved.
    pub fn cavity_status_buffer(&self) -> Option<&wgpu::Buffer> {
        self.cavity.as_ref().map(|c| &c.control)
    }

    /// Per-wheel [volume, pressure, bead load, valid] at the last frame boundary.
    pub fn cavity_records_buffer(&self) -> Option<&wgpu::Buffer> {
        self.cavity.as_ref().map(|c| &c.records)
    }
}

#[test]
fn prepared_vehicle_fem_byte_contract() {
    assert_eq!(core::mem::size_of::<TireFemGpuBatchAtlasParamRow>(), 112);
    assert_eq!(core::mem::size_of::<TireFemGpuAtlasHubParamRow>(), 64);
    assert_eq!(core::mem::size_of::<BrushHubUniforms>(), 32);
    assert_eq!(PreparedTireVehicleFemStep::required_bytes(4), 2900);
}
