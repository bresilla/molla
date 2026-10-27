use super::*;
use crate::{BrushTerrainGpu, PreparedTireBrushTerrain};
use molla_compute::{DeviceHealth, Engine};
use std::sync::Arc;

fn invalid(message: &str) -> Error {
    Error::Solver(format!("prepared chassis step: {message}"))
}

fn sources(owner: &TireVehicleResidentChassisBrushOwner) -> [&wgpu::Buffer; 11] {
    [
        &owner.brush.spin.state_buffers[0],
        &owner.brush.spin.state_buffers[1],
        &owner.brush.spin.commands,
        &owner.brush.suspension.geometry,
        &owner.brush.suspension.params,
        &owner.brush.suspension.rows,
        &owner.planar.state_buffers[0],
        &owner.planar.state_buffers[1],
        &owner.planar.wheel_offsets,
        &owner.vertical.state_buffers[0],
        &owner.vertical.state_buffers[1],
    ]
}

fn properties(owner: &TireVehicleResidentChassisBrushOwner) -> [u32; 7] {
    [
        owner.planar.mass,
        owner.planar.yaw_inertia,
        owner.planar.linear_damping,
        owner.planar.yaw_damping,
        owner.vertical.mass,
        owner.vertical.gravity,
        owner.vertical.vertical_damping,
    ]
    .map(f32::to_bits)
}

fn uniforms(owner: &TireVehicleResidentChassisBrushOwner, dt: f32) -> [VehicleUniforms; 5] {
    let neutral = VehicleUniforms {
        counts: [owner.wheel_count() as u32, 0, 0, 0],
        pose: [0.; 4],
        velocity: [0.; 4],
        params0: [1., 1., 0., 0.],
        params1: [0.; 4],
    };
    [
        VehicleUniforms {
            params0: [1., 1., dt, 0.],
            ..neutral
        },
        neutral,
        neutral,
        VehicleUniforms {
            params0: [
                finite_positive_or(owner.planar.mass, 1., 1e-6),
                finite_positive_or(owner.planar.yaw_inertia, 1., 1e-6),
                dt,
                finite_or(owner.planar.linear_damping, 0.).max(0.),
            ],
            params1: [finite_or(owner.planar.yaw_damping, 0.).max(0.), 0., 0., 0.],
            ..neutral
        },
        VehicleUniforms {
            params0: [
                finite_positive_or(owner.vertical.mass, 1., 1e-6),
                finite_or(owner.vertical.gravity, 9.81).max(0.),
                dt,
                finite_or(owner.vertical.vertical_damping, 0.).max(0.),
            ],
            ..neutral
        },
    ]
}

/// Six-stage brush chassis stepping with one submission and retained resources.
/// Sources, terrain metadata and dt are fixed at preparation. Wheel configuration
/// changes require an explicit refresh before stepping.
/// Command/suspension/terrain buffer contents may change in queue order. Returned
/// brush output is overwritten by later steps. Owner resources must belong to the
/// engine; submitted GPU faults/device loss are not transactional frame rollback.
pub struct PreparedTireVehicleChassisStep {
    health: Arc<DeviceHealth>,
    sources: [wgpu::Buffer; 11],
    properties: [u32; 7],
    packed: Vec<[f32; 4]>,
    packed_buffer: wgpu::Buffer,
    rows: wgpu::Buffer,
    pipelines: [wgpu::ComputePipeline; 5],
    spin: [wgpu::BindGroup; 2],
    suspension: [wgpu::BindGroup; 2],
    row_bindings: [wgpu::BindGroup; 8],
    planar: [wgpu::BindGroup; 2],
    vertical: [wgpu::BindGroup; 2],
    uniforms: [wgpu::BindGroup; 5],
    brush: PreparedTireBrushTerrain,
}

impl PreparedTireVehicleChassisStep {
    /// Owned GPU buffer bytes; excludes borrowed owner/maps and driver resources.
    pub fn required_bytes(wheels: u32) -> u64 {
        432 * u64::from(wheels) + 436
    }

    pub fn new(
        engine: &Engine,
        kernel: &TireVehicleYawGpuKernel,
        brush_kernel: &TireBrushGpuKernel,
        owner: &TireVehicleResidentChassisBrushOwner,
        terrain: Option<&BrushTerrainGpu<'_>>,
        dt: f32,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| invalid("GPU engine required"))?;
        crate::soil_gpu_identity::validate_soil_device(device, &kernel.device)?;
        let n = owner.brush.brush_input.validate()?;
        if [
            owner.wheel_count(),
            owner.brush.suspension.wheel_count,
            owner.planar.wheel_count,
            owner.vertical.wheel_count,
        ]
        .into_iter()
        .any(|count| count != n)
            || !dt.is_finite()
            || dt <= 0.
            || Self::required_bytes(n as u32) > max_owned_bytes
        {
            return Err(invalid("owner counts, timestep or owned byte budget"));
        }
        let brush = PreparedTireBrushTerrain::new(
            engine,
            brush_kernel,
            n as u32,
            terrain,
            PreparedTireBrushTerrain::required_bytes(n as u32),
        )?;
        let allocate = |size, usage| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("prepared chassis storage"),
                    size,
                    usage,
                    mapped_at_creation: false
                }
            )
        };
        let packed = owner.brush.brush_input.packed_rows();
        let packed_buffer = allocate(
            96 * n as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let rows = allocate(
            112 * n as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let queue = engine.queue().unwrap();
        molla_core::gpu_write_buffer!(queue, &packed_buffer, 0, bytemuck::cast_slice(&packed));
        let uniforms = uniforms(owner, dt).map(|data| {
            let buffer = allocate(
                80,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );
            molla_core::gpu_write_buffer!(queue, &buffer, 0, bytemuck::bytes_of(&data));
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("prepared chassis parameters"),
                layout: &kernel.layout_uniforms,
                entries: &[entry(0, &buffer)],
            })
        });
        let bind = |layout: &wgpu::BindGroupLayout, buffers: &[&wgpu::Buffer]| {
            let entries: Vec<_> = buffers
                .iter()
                .enumerate()
                .map(|(i, b)| entry(i as u32, b))
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("prepared chassis IO"),
                layout,
                entries: &entries,
            })
        };
        let spin = std::array::from_fn(|s| {
            bind(
                &kernel.layout_io,
                &[
                    &owner.brush.spin.state_buffers[s],
                    &owner.brush.spin.commands,
                    &owner.brush.spin.state_buffers[1 - s],
                ],
            )
        });
        let suspension = std::array::from_fn(|v| {
            bind(
                &kernel.layout_io4,
                &[
                    &owner.brush.suspension.geometry,
                    &owner.brush.suspension.params,
                    &owner.brush.suspension.rows,
                    &owner.vertical.state_buffers[v],
                ],
            )
        });
        let row_bindings = std::array::from_fn(|i| {
            bind(
                &kernel.layout_io6,
                &[
                    &owner.brush.spin.state_buffers[i / 4],
                    &packed_buffer,
                    &rows,
                    &owner.brush.suspension.rows,
                    &owner.planar.state_buffers[i / 2 % 2],
                    &owner.vertical.state_buffers[i % 2],
                ],
            )
        });
        let output = brush.output();
        let planar = std::array::from_fn(|p| {
            bind(
                &kernel.layout_io4,
                &[
                    &owner.planar.wheel_offsets,
                    &output.buffer,
                    &owner.planar.state_buffers[1 - p],
                    &owner.planar.state_buffers[p],
                ],
            )
        });
        let vertical = std::array::from_fn(|v| {
            bind(
                &kernel.layout_io4,
                &[
                    &owner.brush.suspension.rows,
                    &owner.brush.suspension.rows,
                    &owner.vertical.state_buffers[1 - v],
                    &owner.vertical.state_buffers[v],
                ],
            )
        });
        Ok(Self {
            health: engine.device_health().unwrap().clone(),
            sources: sources(owner).map(Clone::clone),
            properties: properties(owner),
            packed,
            packed_buffer,
            rows,
            pipelines: [
                kernel.pipeline_wheel_spin_state.clone(),
                kernel.pipeline_suspension_from_vertical.clone(),
                kernel
                    .pipeline_brush_rows_from_spin_suspension_chassis
                    .clone(),
                kernel.pipeline_planar_state_from_brush.clone(),
                kernel.pipeline_vertical_state_from_suspension.clone(),
            ],
            spin,
            suspension,
            row_bindings,
            planar,
            vertical,
            uniforms,
            brush,
        })
    }

    fn validate_identity(
        &self,
        engine: &Engine,
        owner: &TireVehicleResidentChassisBrushOwner,
    ) -> Result<usize> {
        self.health.check()?;
        if !engine
            .device_health()
            .is_some_and(|h| Arc::ptr_eq(h, &self.health))
            || sources(owner)
                .into_iter()
                .zip(&self.sources)
                .any(|(a, b)| a != b)
            || properties(owner) != self.properties
        {
            return Err(invalid("foreign/stale owner or engine"));
        }
        let input = &owner.brush.brush_input;
        let n = input.validate()?;
        if n != self.brush.wheel_count() {
            return Err(invalid("changed wheel count"));
        }
        Ok(n)
    }

    pub(crate) fn validate(
        &self,
        engine: &Engine,
        owner: &TireVehicleResidentChassisBrushOwner,
    ) -> Result<()> {
        let n = self.validate_identity(engine, owner)?;
        let input = &owner.brush.brush_input;
        for i in 0..n {
            for (j, row) in [
                input.wheel_mounts[i],
                input.wheel_controls[i],
                input.tire_rows[i],
                input.material_rows[i],
                input.extra_rows[i],
                input.suspension_rows[i],
            ]
            .into_iter()
            .enumerate()
            {
                if row.map(f32::to_bits) != self.packed[6 * i + j].map(f32::to_bits) {
                    return Err(invalid("wheel configuration changed; reprepare"));
                }
            }
        }
        Ok(())
    }

    /// Refreshes changed fixed-count wheel rows with one upload and no allocation.
    /// Returns whether an upload occurred. Earlier submitted steps retain queue order.
    pub fn update_wheel_configuration(
        &mut self,
        engine: &Engine,
        owner: &TireVehicleResidentChassisBrushOwner,
    ) -> Result<bool> {
        let n = self.validate_identity(engine, owner)?;
        let input = &owner.brush.brush_input;
        let mut changed = false;
        for i in 0..n {
            for (j, row) in [
                input.wheel_mounts[i],
                input.wheel_controls[i],
                input.tire_rows[i],
                input.material_rows[i],
                input.extra_rows[i],
                input.suspension_rows[i],
            ]
            .into_iter()
            .enumerate()
            {
                let saved = &mut self.packed[6 * i + j];
                changed |= row.map(f32::to_bits) != saved.map(f32::to_bits);
                *saved = row;
            }
        }
        if changed {
            molla_core::gpu_write_buffer!(
                engine.queue().unwrap(),
                &self.packed_buffer,
                0,
                bytemuck::cast_slice(&self.packed)
            );
        }
        Ok(changed)
    }

    fn encode_stage(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        stage: usize,
        bindings: &wgpu::BindGroup,
        indirect: Option<&wgpu::Buffer>,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipelines[stage]);
        pass.set_bind_group(0, bindings, &[]);
        pass.set_bind_group(1, &self.uniforms[stage], &[]);
        molla_core::gpu_metrics::record_dispatch();
        if let Some(buffer) = indirect {
            pass.dispatch_workgroups_indirect(buffer, 28);
        } else {
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    /// Validates before encoding, submits all six stages, then advances owner parities.
    /// No hot buffer/pipeline/binding creation, host transfer or blocking read occurs.
    pub fn step(
        &self,
        engine: &Engine,
        owner: &mut TireVehicleResidentChassisBrushOwner,
    ) -> Result<TireBrushResidentOutput> {
        self.validate(engine, owner)?;
        let mut encoder = engine
            .device()
            .ok_or_else(|| invalid("GPU engine required"))?
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let output = self.encode(engine, &mut encoder, owner)?;
        engine.queue().unwrap().submit([encoder.finish()]);
        Self::advance_parities(owner);
        Ok(output)
    }

    pub(crate) fn output(&self) -> TireBrushResidentOutput {
        self.brush.output()
    }

    pub(crate) fn encode(
        &self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        owner: &TireVehicleResidentChassisBrushOwner,
    ) -> Result<TireBrushResidentOutput> {
        self.encode_dispatch(engine, encoder, owner, None)
    }

    pub(crate) fn preserve_next_states(
        encoder: &mut wgpu::CommandEncoder,
        owner: &TireVehicleResidentChassisBrushOwner,
    ) {
        for (buffers, current) in [
            (
                &owner.brush.spin.state_buffers,
                owner.brush.spin.current_state,
            ),
            (&owner.planar.state_buffers, owner.planar.current_state),
            (&owner.vertical.state_buffers, owner.vertical.current_state),
        ] {
            encoder.copy_buffer_to_buffer(
                &buffers[current],
                0,
                &buffers[1 - current],
                0,
                buffers[current].size(),
            );
        }
    }

    pub(crate) fn encode_dispatch(
        &self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        owner: &TireVehicleResidentChassisBrushOwner,
        indirect: Option<&wgpu::Buffer>,
    ) -> Result<TireBrushResidentOutput> {
        self.validate(engine, owner)?;
        let (s, p, v) = (
            owner.brush.spin.current_state,
            owner.planar.current_state,
            owner.vertical.current_state,
        );
        self.encode_stage(encoder, 0, &self.spin[s], indirect);
        self.encode_stage(encoder, 1, &self.suspension[v], indirect);
        self.encode_stage(
            encoder,
            2,
            &self.row_bindings[(1 - s) * 4 + p * 2 + v],
            indirect,
        );
        self.brush
            .encode_dispatch(engine, encoder, &self.rows, indirect.map(|b| (b, 40)))?;
        self.encode_stage(encoder, 3, &self.planar[p], indirect);
        self.encode_stage(encoder, 4, &self.vertical[v], indirect);
        Ok(self.brush.output())
    }

    pub(crate) fn advance_parities(owner: &mut TireVehicleResidentChassisBrushOwner) {
        owner.brush.spin.current_state ^= 1;
        owner.planar.current_state ^= 1;
        owner.vertical.current_state ^= 1;
    }
}

#[test]
fn prepared_chassis_step_budget_and_uniform_contract() {
    assert_eq!(core::mem::size_of::<VehicleUniforms>(), 80);
    assert_eq!(PreparedTireVehicleChassisStep::required_bytes(4), 2164);
    for n in 1..=64 {
        assert_eq!(
            PreparedTireVehicleChassisStep::required_bytes(n),
            208 * u64::from(n) + 400 + PreparedTireBrushTerrain::required_bytes(n)
        );
    }
}
