//! GPU batch step for the layered tire thermal model.
//!
//! This mirrors the reusable four-node `TireThermalLayers` update shape:
//! surface, tread, carcass, and rim temperatures are advanced with the same
//! heat-source routing and conductance network, but batched for many wheels.

use bytemuck::{Pod, Zeroable};
use molla_core::{Error, Result};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default, PartialEq)]
pub struct TireThermalGpuInput {
    /// `(sliding_power_w, carcass_flex_power_w, rolling_power_w, brake_rim_power_w)`.
    pub powers: [f32; 4],
    /// `(road_c, dt_s, reserved, reserved)`.
    pub environment: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq)]
pub struct TireThermalGpuState {
    /// `(surface_c, tread_c, carcass_c, rim_c)`.
    pub temperatures: [f32; 4],
    /// `(depth_m, max_depth_m, wear_rate_m3_per_j, ref_patch_area_m2)`.
    pub wear: [f32; 4],
    /// `(surface_j_per_k, tread_j_per_k, carcass_j_per_k, rim_j_per_k)`.
    pub capacities: [f32; 4],
    /// `(ambient_c, road_w_per_k, air_w_per_k, surface_tread_w_per_k)`.
    pub coeffs0: [f32; 4],
    /// `(tread_carcass_w_per_k, carcass_rim_w_per_k, reserved, reserved)`.
    pub coeffs1: [f32; 4],
}

impl Default for TireThermalGpuState {
    fn default() -> Self {
        Self::passenger_road()
    }
}

impl TireThermalGpuState {
    /// Match `TireThermalLayers::passenger_road()` using f32 storage.
    pub const fn passenger_road() -> Self {
        Self {
            temperatures: [20.0, 20.0, 20.0, 20.0],
            wear: [0.0, 0.008, 2.0e-13, 0.020],
            capacities: [800.0, 8.0e3, 1.8e4, 3.0e4],
            coeffs0: [20.0, 45.0, 18.0, 120.0],
            coeffs1: [35.0, 22.0, 0.0, 0.0],
        }
    }

    /// Match `TireThermalLayers::fast_response_test()` using f32 storage.
    pub const fn fast_response_test() -> Self {
        Self {
            temperatures: [20.0, 20.0, 20.0, 20.0],
            wear: [0.0, 0.008, 2.0e-13, 0.020],
            capacities: [520.0, 5.5e3, 1.25e4, 2.5e4],
            coeffs0: [20.0, 60.0, 28.0, 150.0],
            coeffs1: [45.0, 25.0, 0.0, 0.0],
        }
    }

    /// Match `TireThermalLayers::heavy_duty_slow_response()` using f32 storage.
    pub const fn heavy_duty_slow_response() -> Self {
        Self {
            temperatures: [20.0, 20.0, 20.0, 20.0],
            wear: [0.0, 0.008, 2.0e-13, 0.020],
            capacities: [1_400.0, 1.4e4, 3.2e4, 5.5e4],
            coeffs0: [20.0, 38.0, 14.0, 105.0],
            coeffs1: [30.0, 20.0, 0.0, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default, PartialEq)]
pub struct TireThermalGpuOutput {
    /// `(surface_c, tread_c, carcass_c, rim_c)`.
    pub temperatures: [f32; 4],
    /// `(depth_m, max_depth_m, wear_ratio, mu_wear_factor)`.
    pub wear: [f32; 4],
    /// `(rubber_peak_c, mu_factor, carcass_pressure_factor, wear_temperature_factor)`.
    pub diagnostics: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default, PartialEq)]
struct TireThermalUniforms {
    count: u32,
    _pad: [u32; 3],
}

pub struct TireThermalGpuKernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

pub struct TireThermalGpuBatch {
    capacity: usize,
    input_buf: wgpu::Buffer,
    state_in_buf: wgpu::Buffer,
    state_out_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    stage_state: wgpu::Buffer,
    stage_out: wgpu::Buffer,
    bg: wgpu::BindGroup,
}

impl TireThermalGpuBatch {
    /// Number of wheel/replay thermal states this persistent allocation can step.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TireThermalRigReplayBatch {
    pub inputs: Vec<TireThermalGpuInput>,
    pub states: Vec<TireThermalGpuState>,
    pub scenario_ids: Vec<String>,
}

impl TireThermalRigReplayBatch {
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }

    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    pub fn capacity_hint(&self) -> usize {
        self.len().max(64)
    }
}

impl TireThermalGpuKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::tire_thermal"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tire_thermal.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_thermal io"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
                uniform_entry(4),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::tire_thermal layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::tire_thermal::thermal_step"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("thermal_step"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { pipeline, layout }
    }

    /// Allocate reusable buffers for up to `capacity` thermal/wear states.
    ///
    /// This is the production path for viewer/rig loops: uploads only the
    /// current wheel commands/state, reuses storage/readback buffers, and still
    /// supports arbitrary `n <= capacity` through the shader's count uniform.
    pub fn create_batch(
        &self,
        device: &wgpu::Device,
        capacity: usize,
    ) -> Result<TireThermalGpuBatch> {
        let capacity = capacity.max(1);
        let in_bytes = (capacity * core::mem::size_of::<TireThermalGpuInput>()) as u64;
        let state_bytes = (capacity * core::mem::size_of::<TireThermalGpuState>()) as u64;
        let out_bytes = (capacity * core::mem::size_of::<TireThermalGpuOutput>()) as u64;
        let input_buf = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("tire_thermal persistent input"),
                size: in_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let state_in_buf = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("tire_thermal persistent state in"),
                size: state_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let state_out_buf = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("tire_thermal persistent state out"),
                size: state_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }
        );
        let out_buf = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("tire_thermal persistent output"),
                size: out_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }
        );
        let uniform_buf = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("tire_thermal persistent uniforms"),
                size: core::mem::size_of::<TireThermalUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let stage_state = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("tire_thermal persistent stage state"),
                size: state_bytes.max(4),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }
        );
        let stage_out = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("tire_thermal persistent stage output"),
                size: out_bytes.max(4),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }
        );
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_thermal persistent bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state_in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: state_out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        Ok(TireThermalGpuBatch {
            capacity,
            input_buf,
            state_in_buf,
            state_out_buf,
            out_buf,
            uniform_buf,
            stage_state,
            stage_out,
            bg,
        })
    }

    pub fn dispatch_persistent_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &TireThermalGpuBatch,
        inputs: &[TireThermalGpuInput],
        states: &[TireThermalGpuState],
    ) -> Result<(Vec<TireThermalGpuState>, Vec<TireThermalGpuOutput>)> {
        validate_thermal_inputs(inputs, states)?;
        let n = inputs.len();
        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        if n > batch.capacity {
            return Err(Error::Solver(format!(
                "tire_thermal: batch capacity {} is smaller than {n} wheels",
                batch.capacity
            )));
        }
        let uniforms = TireThermalUniforms {
            count: n as u32,
            _pad: [0; 3],
        };
        let state_bytes = (n * core::mem::size_of::<TireThermalGpuState>()) as u64;
        let out_bytes = (n * core::mem::size_of::<TireThermalGpuOutput>()) as u64;
        molla_core::gpu_write_buffer!(queue, &batch.input_buf, 0, bytemuck::cast_slice(inputs));
        molla_core::gpu_write_buffer!(queue, &batch.state_in_buf, 0, bytemuck::cast_slice(states));
        molla_core::gpu_write_buffer!(queue, &batch.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_thermal persistent encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_thermal persistent pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &batch.bg, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&batch.state_out_buf, 0, &batch.stage_state, 0, state_bytes);
        encoder.copy_buffer_to_buffer(&batch.out_buf, 0, &batch.stage_out, 0, out_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        Ok((
            read_buffer(device, &batch.stage_state, state_bytes, "persistent_state")?,
            read_buffer(device, &batch.stage_out, out_bytes, "persistent_output")?,
        ))
    }

    pub fn dispatch_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        inputs: &[TireThermalGpuInput],
        states: &[TireThermalGpuState],
    ) -> Result<(Vec<TireThermalGpuState>, Vec<TireThermalGpuOutput>)> {
        validate_thermal_inputs(inputs, states)?;
        let n = inputs.len();
        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let in_bytes = std::mem::size_of_val(inputs) as u64;
        let state_bytes = (n * core::mem::size_of::<TireThermalGpuState>()) as u64;
        let out_bytes = (n * core::mem::size_of::<TireThermalGpuOutput>()) as u64;
        let uniforms = TireThermalUniforms {
            count: n as u32,
            _pad: [0; 3],
        };

        let input_buf = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("tire_thermal input"),
                size: in_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let state_in_buf = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("tire_thermal state in"),
                size: state_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let state_out_buf = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("tire_thermal state out"),
                size: state_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }
        );
        let out_buf = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("tire_thermal output"),
                size: out_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }
        );
        let stage_state = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("tire_thermal stage state"),
                size: state_bytes.max(4),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }
        );
        let stage_out = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("tire_thermal stage output"),
                size: out_bytes.max(4),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }
        );
        let uniform_buf = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("tire_thermal uniforms"),
                size: core::mem::size_of::<TireThermalUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );

        molla_core::gpu_write_buffer!(queue, &input_buf, 0, bytemuck::cast_slice(inputs));
        molla_core::gpu_write_buffer!(queue, &state_in_buf, 0, bytemuck::cast_slice(states));
        molla_core::gpu_write_buffer!(queue, &uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_thermal bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state_in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: state_out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_thermal encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_thermal pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&state_out_buf, 0, &stage_state, 0, state_bytes);
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage_out, 0, out_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        Ok((
            read_buffer(device, &stage_state, state_bytes, "state")?,
            read_buffer(device, &stage_out, out_bytes, "output")?,
        ))
    }
}

pub fn tire_thermal_reference_step(
    input: &TireThermalGpuInput,
    state: &TireThermalGpuState,
) -> (TireThermalGpuState, TireThermalGpuOutput) {
    let mut next = *state;
    let sliding_power_w = finite_nonnegative(input.powers[0]);
    let carcass_flex_power_w = finite_nonnegative(input.powers[1]);
    let rolling_power_w = finite_nonnegative(input.powers[2]);
    let brake_rim_power_w = finite_nonnegative(input.powers[3]);
    let total_dt = finite_nonnegative(input.environment[1]).min(10.0);
    if total_dt <= 0.0 {
        return thermal_output(next);
    }

    let ambient_c = finite_or(state.coeffs0[0], 20.0);
    let road_c = finite_or(input.environment[0], ambient_c);
    let c_surface = finite_positive_or(state.capacities[0], 800.0, 1.0);
    let c_tread = finite_positive_or(state.capacities[1], 8.0e3, 1.0);
    let c_carcass = finite_positive_or(state.capacities[2], 1.8e4, 1.0);
    let c_rim = finite_positive_or(state.capacities[3], 3.0e4, 1.0);
    let g_road = finite_nonnegative(state.coeffs0[1]);
    let g_air = finite_nonnegative(state.coeffs0[2]);
    let g_st = finite_nonnegative(state.coeffs0[3]);
    let g_tc = finite_nonnegative(state.coeffs1[0]);
    let g_cr = finite_nonnegative(state.coeffs1[1]);

    let mut remaining = total_dt;
    while remaining > 1.0e-9 {
        let h = remaining.min(0.05);
        remaining -= h;
        let surface_c = finite_or(next.temperatures[0], ambient_c);
        let tread_c = finite_or(next.temperatures[1], ambient_c);
        let carcass_c = finite_or(next.temperatures[2], ambient_c);
        let rim_c = finite_or(next.temperatures[3], ambient_c);

        let q_surface_in = 0.70 * sliding_power_w + 0.15 * rolling_power_w;
        let q_tread_in =
            0.30 * sliding_power_w + 0.35 * rolling_power_w + 0.20 * carcass_flex_power_w;
        let q_carcass_in =
            0.80 * carcass_flex_power_w + 0.50 * rolling_power_w + 0.15 * brake_rim_power_w;
        let q_rim_in = 0.85 * brake_rim_power_w;

        let q_st = g_st * (surface_c - tread_c);
        let q_tc = g_tc * (tread_c - carcass_c);
        let q_cr = g_cr * (carcass_c - rim_c);
        let q_surface_air = g_air * (surface_c - ambient_c);
        let q_surface_road = g_road * (surface_c - road_c);
        let q_carcass_air = 0.35 * g_air * (carcass_c - ambient_c);
        let q_rim_air = 0.25 * g_air * (rim_c - ambient_c);

        next.temperatures[0] =
            surface_c + (q_surface_in - q_st - q_surface_air - q_surface_road) / c_surface * h;
        next.temperatures[1] = tread_c + (q_tread_in + q_st - q_tc) / c_tread * h;
        next.temperatures[2] =
            carcass_c + (q_carcass_in + q_tc - q_cr - q_carcass_air) / c_carcass * h;
        next.temperatures[3] = rim_c + (q_rim_in + q_cr - q_rim_air) / c_rim * h;
    }

    for t in &mut next.temperatures {
        *t = finite_or(*t, ambient_c).clamp(-80.0, 350.0);
    }
    step_wear(
        &mut next.wear,
        sliding_power_w,
        next.temperatures[0],
        total_dt,
    );
    thermal_output(next)
}

pub fn tire_thermal_input_from_rig_snapshot(
    snapshot: &molla_vehicle::TireRigSnapshot,
    dt_s: f32,
    road_temperature_c: f32,
) -> TireThermalGpuInput {
    let solver_dissipated_w = snapshot
        .solver_budget
        .map(|s| finite_nonnegative(s.dissipated_power_w as f32) * 0.15)
        .unwrap_or(0.0);
    let rolling_power_w = finite_nonnegative(
        (snapshot.telemetry.normal_load_n * snapshot.speed_m_per_s.abs() * 0.0025) as f32,
    );
    TireThermalGpuInput {
        powers: [
            finite_nonnegative(snapshot.telemetry.dissipated_power_w as f32),
            solver_dissipated_w,
            rolling_power_w,
            0.0,
        ],
        environment: [
            finite_or(
                snapshot
                    .surface_temperature_c
                    .map(|v| v as f32)
                    .unwrap_or(road_temperature_c),
                road_temperature_c,
            ),
            finite_nonnegative(dt_s),
            0.0,
            0.0,
        ],
    }
}

pub fn tire_thermal_state_from_rig_snapshot(
    snapshot: &molla_vehicle::TireRigSnapshot,
) -> TireThermalGpuState {
    let mut state = TireThermalGpuState::passenger_road();
    let surface = finite_or(
        snapshot
            .surface_temperature_c
            .map(|v| v as f32)
            .unwrap_or(20.0),
        20.0,
    );
    let tread = finite_or(
        snapshot
            .tread_temperature_c
            .map(|v| v as f32)
            .unwrap_or(surface),
        surface,
    );
    let carcass = finite_or(
        snapshot
            .carcass_temperature_c
            .map(|v| v as f32)
            .unwrap_or(tread),
        tread,
    );
    let depth_m = snapshot
        .tread_depth_m
        .map(|v| finite_nonnegative(v as f32))
        .or_else(|| {
            snapshot
                .wear_ratio
                .map(|r| finite_nonnegative(r as f32).clamp(0.0, 1.0) * state.wear[1])
        })
        .unwrap_or(state.wear[0])
        .clamp(0.0, state.wear[1].max(1.0e-6));
    state.temperatures = [surface, tread, carcass, carcass.min(80.0)];
    state.wear[0] = depth_m;
    state
}

pub fn tire_thermal_build_rig_replay_batch(
    snapshots: &[molla_vehicle::TireRigSnapshot],
    dt_s: f32,
    road_temperature_c: f32,
) -> TireThermalRigReplayBatch {
    let mut batch = TireThermalRigReplayBatch {
        inputs: Vec::with_capacity(snapshots.len()),
        states: Vec::with_capacity(snapshots.len()),
        scenario_ids: Vec::with_capacity(snapshots.len()),
    };
    for snapshot in snapshots {
        batch.inputs.push(tire_thermal_input_from_rig_snapshot(
            snapshot,
            dt_s,
            road_temperature_c,
        ));
        batch
            .states
            .push(tire_thermal_state_from_rig_snapshot(snapshot));
        batch.scenario_ids.push(snapshot.scenario_id.clone());
    }
    batch
}

pub fn tire_thermal_reference_rig_replay_batch(
    replay: &TireThermalRigReplayBatch,
) -> Vec<(TireThermalGpuState, TireThermalGpuOutput)> {
    replay
        .inputs
        .iter()
        .zip(replay.states.iter())
        .map(|(input, state)| tire_thermal_reference_step(input, state))
        .collect()
}

fn thermal_output(state: TireThermalGpuState) -> (TireThermalGpuState, TireThermalGpuOutput) {
    let ambient_c = finite_or(state.coeffs0[0], 20.0);
    let surface_c = finite_or(state.temperatures[0], ambient_c);
    let tread_c = finite_or(state.temperatures[1], ambient_c);
    let carcass_c = finite_or(state.temperatures[2], ambient_c);
    let output = TireThermalGpuOutput {
        temperatures: state.temperatures,
        wear: [
            finite_nonnegative(state.wear[0]),
            state.wear[1].max(1.0e-6),
            wear_ratio(state.wear),
            wear_mu_factor(state.wear),
        ],
        diagnostics: [
            surface_c.max(tread_c).max(carcass_c),
            thermal_mu_factor(state.temperatures, ambient_c),
            carcass_pressure_factor(carcass_c, ambient_c),
            wear_temperature_factor(surface_c),
        ],
    };
    (state, output)
}

fn step_wear(wear: &mut [f32; 4], power_w: f32, surface_c: f32, dt: f32) {
    let depth_m = finite_nonnegative(wear[0]);
    let max_depth_m = finite_positive_or(wear[1], 0.008, 1.0e-6).max(1.0e-6);
    let wear_rate_m3_per_j = finite_nonnegative(wear[2]);
    let ref_patch_area_m2 = finite_positive_or(wear[3], 0.035, 1.0e-6);
    let volume_removed =
        wear_rate_m3_per_j * finite_nonnegative(power_w) * wear_temperature_factor(surface_c) * dt;
    let depth_step = volume_removed / ref_patch_area_m2;
    wear[0] = (depth_m + depth_step).min(max_depth_m);
    wear[1] = max_depth_m;
    wear[2] = wear_rate_m3_per_j;
    wear[3] = ref_patch_area_m2;
}

fn wear_ratio(wear: [f32; 4]) -> f32 {
    (finite_nonnegative(wear[0]) / wear[1].max(1.0e-9)).clamp(0.0, 1.0)
}

fn wear_mu_factor(wear: [f32; 4]) -> f32 {
    1.0 - 0.30 * wear_ratio(wear)
}

fn thermal_mu_factor(temperatures: [f32; 4], ambient_c: f32) -> f32 {
    let surface_c = finite_or(temperatures[0], ambient_c);
    let tread_c = finite_or(temperatures[1], ambient_c);
    let carcass_c = finite_or(temperatures[2], ambient_c);
    let surface_peak = 1.08 - (surface_c - 80.0).abs() * 0.006;
    let bulk_hot_penalty =
        ((0.5 * (tread_c + carcass_c) - 115.0).max(0.0) * 0.0025).clamp(0.0, 0.25);
    (surface_peak - bulk_hot_penalty).clamp(0.45, 1.08)
}

fn carcass_pressure_factor(carcass_c: f32, ambient_c: f32) -> f32 {
    (1.0 + (finite_or(carcass_c, ambient_c) - ambient_c) * 0.0015).clamp(0.85, 1.20)
}

fn wear_temperature_factor(surface_c: f32) -> f32 {
    let t = finite_or(surface_c, 20.0);
    if t < 70.0 {
        (0.65 + (t + 20.0).max(0.0) / 90.0 * 0.35).clamp(0.45, 1.0)
    } else {
        (1.0 + ((t - 70.0) / 45.0).powi(2)).clamp(1.0, 6.0)
    }
}

fn validate_thermal_inputs(
    inputs: &[TireThermalGpuInput],
    states: &[TireThermalGpuState],
) -> Result<()> {
    if inputs.len() != states.len() {
        return Err(Error::Solver(format!(
            "tire_thermal: input/state length mismatch {} != {}",
            inputs.len(),
            states.len()
        )));
    }
    Ok(())
}

fn read_buffer<T: Pod>(
    device: &wgpu::Device,
    stage: &wgpu::Buffer,
    bytes: u64,
    label: &str,
) -> Result<Vec<T>> {
    let slice = stage.slice(0..bytes);
    let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
    let label = label.to_owned();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        sender
            .send(r.map_err(|e| Error::Gpu(format!("tire_thermal {label} readback: {e:?}"))))
            .ok();
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Gpu(format!("tire_thermal poll: {e:?}")))?;
    receiver
        .recv()
        .map_err(|e| Error::Gpu(format!("tire_thermal recv: {e:?}")))??;
    let data = molla_core::gpu_readback!(slice);
    let out = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    stage.unmap();
    Ok(out)
}

fn finite_nonnegative(v: f32) -> f32 {
    if v.is_finite() { v.max(0.0) } else { 0.0 }
}

fn finite_or(v: f32, fallback: f32) -> f32 {
    if v.is_finite() { v } else { fallback }
}

fn finite_positive_or(v: f32, fallback: f32, floor: f32) -> f32 {
    if v.is_finite() && v >= floor {
        v
    } else {
        fallback.max(floor)
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
