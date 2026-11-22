use super::{AirConfig, AirForces, AirState, invalid};
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_math::Vec3;
use std::sync::Arc;

const SAMPLE_CAPACITY: usize = 4096;
/// Fixed-point unit of reactions and boundary impulses deposited on the device (2^-24).
pub(crate) const REACTION_QUANTUM: f64 = 1.0 / 16_777_216.0;
/// Linear and angular boundary impulse accumulators, eight `i32` lanes.
const BOUNDARY_BYTES: u64 = 32;

/// Device buffers through which another kernel samples this air and deposits reactions.
pub struct AirCoupling<'a> {
    pub(crate) device: &'a Arc<wgpu::Device>,
    pub(crate) settings: &'a wgpu::Buffer,
    pub(crate) velocity: &'a wgpu::Buffer,
    pub(crate) reactions: &'a wgpu::Buffer,
    pub(crate) boundary: &'a wgpu::Buffer,
}

fn pack(v: Vec3) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, 0.0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a GPU; kernel timing diagnostic"]
    fn profile_air_kernels() {
        let engine = Engine::new_gpu().unwrap();
        let grid = super::super::Grid::new([64, 32, 48], 0.15, Vec3::new(-4.8, 0.0, -3.6)).unwrap();
        let mut air = GpuAir::new(&engine, &AirState::new(AirConfig::new(grid)).unwrap()).unwrap();
        air.step(&engine, 1.0 / 60.0, &AirForces::new(grid))
            .unwrap();
        air.snapshot(&engine).unwrap();
        for (entry, name) in ["advect", "divergence", "red", "black", "project", "sample"]
            .iter()
            .enumerate()
        {
            let start = std::time::Instant::now();
            let mut commands = air.owner.create_command_encoder(&Default::default());
            {
                let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(name),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&air.program.pipelines[entry]);
                pass.set_bind_group(0, &air.bindings, &[]);
                for _ in 0..20 {
                    match entry {
                        2 | 3 => pass.dispatch_workgroups(4, 8, 12),
                        5 => pass.dispatch_workgroups(32, 1, 1),
                        _ => pass.dispatch_workgroups(9, 9, 13),
                    }
                }
            }
            engine.queue().unwrap().submit([commands.finish()]);
            air.owner
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(30)),
                })
                .unwrap();
            println!(
                "air kernel {name}: {:.3} ms per dispatch",
                start.elapsed().as_secs_f64() * 1000.0 / 20.0
            );
        }
    }
}
fn finite(values: &[f32]) -> Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("GPU float range exceeded"));
    }
    Ok(())
}

struct Program {
    layout: wgpu::BindGroupLayout,
    pipelines: Vec<wgpu::ComputePipeline>,
}

impl Program {
    fn new(device: &wgpu::Device) -> Result<Self> {
        let entries: Vec<_> = (0..9)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding == 5 || binding == 6,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MAC air"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MAC air"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MAC air"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("../../../molla-kernels/shaders/air_field.wgsl"),
                    include_str!("../../../molla-kernels/shaders/airflow.wgsl")
                )
                .into(),
            ),
        });
        let pipelines = [
            "advect",
            "divergence",
            "red",
            "black",
            "project",
            "sample_points",
        ]
        .iter()
        .map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        })
        .collect();
        Ok(Self { layout, pipelines })
    }
}

/// Resident velocity/pressure fields; snapshots explicitly synchronize with the device.
pub struct GpuAir {
    owner: Arc<wgpu::Device>,
    program: Arc<Program>,
    config: AirConfig,
    drive: Vec3,
    buffers: Vec<wgpu::Buffer>,
    bindings: wgpu::BindGroup,
    staging: wgpu::Buffer,
    sample_staging: wgpu::Buffer,
    boundary_staging: wgpu::Buffer,
    uploads: wgpu::util::StagingBelt,
}

impl GpuAir {
    pub fn new(engine: &Engine, state: &AirState) -> Result<Self> {
        engine.check_device()?;
        state.config.validate(1e-6)?;
        let program = engine.cached_gpu_program("mac-air-v7-drive", Program::new)?;
        let n = state.config.grid.len as u64;
        let packed: Vec<_> = state.velocity.iter().copied().map(pack).collect();
        finite(bytemuck::cast_slice(&packed))?;
        engine.checked_gpu_setup(|device| {
            let limits = device.limits();
            if n * 16 > limits.max_storage_buffer_binding_size as u64
                || n * 20 > limits.max_buffer_size
                || n.div_ceil(128) > limits.max_compute_workgroups_per_dimension as u64
            {
                return Err(invalid("air grid exceeds device limits"));
            }
            let storage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST;
            let buffers: Vec<_> = [
                96,
                n * 16,
                n * 16,
                n * 4,
                n * 4,
                n * 16,
                SAMPLE_CAPACITY as u64 * 16,
                SAMPLE_CAPACITY as u64 * 16,
                n * 12,
                BOUNDARY_BYTES,
            ]
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                molla_core::gpu_create_buffer!(
                    device,
                    &wgpu::BufferDescriptor {
                        label: Some("MAC air field"),
                        size,
                        usage: if i == 0 {
                            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
                        } else {
                            storage
                        },
                        mapped_at_creation: false,
                    }
                )
            })
            .collect();
            let entries: Vec<_> = buffers[..9]
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect();
            let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MAC air fields"),
                layout: &program.layout,
                entries: &entries,
            });
            let staging = molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("MAC air snapshot"),
                    size: n * 20,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }
            );
            let boundary_staging = molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("air boundary impulses"),
                    size: BOUNDARY_BYTES,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }
            );
            let sample_staging = molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("air velocity samples"),
                    size: SAMPLE_CAPACITY as u64 * 16,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }
            );
            let g = state.config.grid;
            let dimensions = [
                g.cells[0] as u32,
                g.cells[1] as u32,
                g.cells[2] as u32,
                g.len as u32,
            ];
            let settings = [
                [
                    g.origin.x as f32,
                    g.origin.y as f32,
                    g.origin.z as f32,
                    g.spacing as f32,
                ],
                [
                    state.config.density as f32,
                    state.config.viscosity as f32,
                    1e-6,
                    state.config.relaxation as f32,
                ],
                pack(state.config.ambient),
                [
                    state.config.gust_strength as f32,
                    state.config.time as f32,
                    state.config.exterior_decay_length as f32,
                    state.config.turbulence_strength as f32,
                ],
                pack(state.drive),
            ];
            finite(bytemuck::cast_slice(&settings))?;
            if settings[0][3] <= 0.0 || settings[1][0] <= 0.0 {
                return Err(invalid("GPU settings underflow"));
            }
            molla_core::gpu_write_buffer!(
                engine.queue().unwrap(),
                &buffers[0],
                0,
                bytemuck::cast_slice(&dimensions)
            );
            molla_core::gpu_write_buffer!(
                engine.queue().unwrap(),
                &buffers[0],
                16,
                bytemuck::cast_slice(&settings)
            );
            molla_core::gpu_write_buffer!(
                engine.queue().unwrap(),
                &buffers[1],
                0,
                bytemuck::cast_slice(&packed)
            );
            let pressure: Vec<f32> = state.pressure.iter().map(|p| *p as f32).collect();
            finite(&pressure)?;
            molla_core::gpu_write_buffer!(
                engine.queue().unwrap(),
                &buffers[3],
                0,
                bytemuck::cast_slice(&pressure)
            );
            Ok(Self {
                owner: engine.device().unwrap().clone(),
                program: program.clone(),
                config: state.config,
                drive: state.drive,
                buffers,
                bindings,
                staging,
                sample_staging,
                boundary_staging,
                uploads: wgpu::util::StagingBelt::new(device.clone(), n * 16 + 65536),
            })
        })
    }

    fn check(&self, engine: &Engine) -> Result<()> {
        engine.check_device()?;
        if !Arc::ptr_eq(engine.device().unwrap(), &self.owner) {
            return Err(invalid("air field belongs to another device"));
        }
        Ok(())
    }

    /// Background wind and gust amplitude from the next step on; the next step also moves
    /// the whole air mass by the change in background wind.
    pub fn set_background(&mut self, ambient: Vec3, gust_strength: f64) -> Result<()> {
        if !ambient.is_finite() || !gust_strength.is_finite() || gust_strength < 0.0 {
            return Err(invalid("background wind must be finite with non-negative gusts"));
        }
        self.drive += ambient - self.config.ambient;
        self.config.ambient = ambient;
        self.config.gust_strength = gust_strength;
        Ok(())
    }

    /// Background wind and gust amplitude of the next step.
    pub fn background(&self) -> (Vec3, f64) {
        (self.config.ambient, self.config.gust_strength)
    }

    /// Buffers for kernels that sample this air and deposit reactions into its next step.
    pub fn coupling(&self) -> AirCoupling<'_> {
        AirCoupling {
            device: &self.owner,
            settings: &self.buffers[0],
            velocity: &self.buffers[1],
            reactions: &self.buffers[8],
            boundary: &self.buffers[9],
        }
    }

    /// Read and reset the linear and angular impulses (N·s, N·m·s) that coupled kernels
    /// recorded for reactions outside the depositable interior.
    pub fn take_boundary_impulses(&mut self, engine: &Engine) -> Result<(Vec3, Vec3)> {
        self.check(engine)?;
        let mut commands = self.owner.create_command_encoder(&Default::default());
        commands.copy_buffer_to_buffer(
            &self.buffers[9],
            0,
            &self.boundary_staging,
            0,
            BOUNDARY_BYTES,
        );
        commands.clear_buffer(&self.buffers[9], 0, None);
        engine.queue().unwrap().submit([commands.finish()]);
        let slice = self.boundary_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.owner
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(30)),
            })
            .map_err(|e| Error::Gpu(e.to_string()))?;
        rx.recv_timeout(std::time::Duration::from_secs(30))
            .map_err(|e| Error::Gpu(e.to_string()))?
            .map_err(|e| Error::Gpu(e.to_string()))?;
        let lanes: Vec<i32> = bytemuck::cast_slice(&molla_core::gpu_readback!(slice)).to_vec();
        self.boundary_staging.unmap();
        let vector = |k: usize| {
            Vec3::new(lanes[k] as f64, lanes[k + 1] as f64, lanes[k + 2] as f64)
                * REACTION_QUANTUM
        };
        Ok((vector(0), vector(4)))
    }

    pub fn step(&mut self, engine: &Engine, dt: f64, forces: &AirForces) -> Result<()> {
        self.check(engine)?;
        let c = self.config;
        c.validate(dt)?;
        forces.validate(c.grid)?;
        let g = c.grid;
        let dimensions = [
            g.cells[0] as u32,
            g.cells[1] as u32,
            g.cells[2] as u32,
            g.len as u32,
        ];
        let settings = [
            [
                g.origin.x as f32,
                g.origin.y as f32,
                g.origin.z as f32,
                g.spacing as f32,
            ],
            [
                c.density as f32,
                c.viscosity as f32,
                dt as f32,
                c.relaxation as f32,
            ],
            pack(c.ambient),
            [
                c.gust_strength as f32,
                c.time as f32,
                c.exterior_decay_length as f32,
                c.turbulence_strength as f32,
            ],
            pack(self.drive),
        ];
        finite(bytemuck::cast_slice(&settings))?;
        if settings[0][3] <= 0.0
            || settings[1][0] <= 0.0
            || settings[1][2] <= 0.0
            || settings[1][3] <= 0.0
            || settings[1][3] >= 2.0
        {
            return Err(invalid(
                "GPU settings underflow or relaxation outside range",
            ));
        }
        let mut spans: Vec<std::ops::Range<usize>> = Vec::new();
        for (i, value) in forces.values.iter().enumerate() {
            if *value == Vec3::ZERO {
                continue;
            }
            if let Some(last) = spans.last_mut()
                && i <= last.end + 8
            {
                last.end = i + 1;
            } else {
                spans.push(i..i + 1);
            }
        }
        let dense = spans.iter().map(|r| r.len()).sum::<usize>() > forces.values.len() / 2;
        if dense {
            spans = vec![0..forces.values.len()];
        }
        let packed: Vec<Vec<[f32; 4]>> = spans
            .iter()
            .map(|span| forces.values[span.clone()].iter().copied().map(pack).collect())
            .collect();
        for values in &packed {
            finite(bytemuck::cast_slice(values))?;
        }
        let queue = engine.queue().unwrap();
        molla_core::gpu_write_buffer!(
            queue,
            &self.buffers[0],
            0,
            bytemuck::cast_slice(&dimensions)
        );
        molla_core::gpu_write_buffer!(queue, &self.buffers[0], 16, bytemuck::cast_slice(&settings));
        let mut commands = self.owner.create_command_encoder(&Default::default());
        if !dense {
            commands.clear_buffer(&self.buffers[5], 0, None);
        }
        for (span, values) in spans.into_iter().zip(&packed) {
            let offset = span.start as u64 * 16;
            let bytes = bytemuck::cast_slice(values);
            let mut upload = self.uploads.write_buffer(
                &mut commands,
                &self.buffers[5],
                offset,
                wgpu::BufferSize::new(bytes.len() as u64).unwrap(),
            );
            molla_compute::copy_to_mapped(&mut upload, bytes);
            molla_core::gpu_metrics::record_upload(bytes.len() as u64);
        }
        self.uploads.finish_and_recall_on_submit(&commands);
        {
            let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("air step"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.bindings, &[]);
            let mut dispatch = |p: usize| {
                pass.set_pipeline(&self.program.pipelines[p]);
                if p == 2 || p == 3 {
                    pass.dispatch_workgroups(
                        g.cells[0].div_ceil(16) as u32,
                        g.cells[1].div_ceil(4) as u32,
                        g.cells[2].div_ceil(4) as u32,
                    );
                } else {
                    pass.dispatch_workgroups(
                        (g.cells[0] + 1).div_ceil(8) as u32,
                        (g.cells[1] + 1).div_ceil(4) as u32,
                        (g.cells[2] + 1).div_ceil(4) as u32,
                    );
                }
            };
            dispatch(0);
            dispatch(1);
            for _ in 0..c.pressure_sweeps {
                dispatch(2);
                dispatch(3);
            }
            dispatch(4);
        }
        commands.clear_buffer(&self.buffers[8], 0, None);
        queue.submit([commands.finish()]);
        self.config.time += dt;
        self.drive = Vec3::ZERO;
        molla_core::gpu_write_buffer!(
            queue,
            &self.buffers[0],
            64,
            bytemuck::cast_slice(&[
                self.config.gust_strength as f32,
                self.config.time as f32,
                self.config.exterior_decay_length as f32,
                self.config.turbulence_strength as f32
            ])
        );
        Ok(())
    }

    pub fn snapshot(&self, engine: &Engine) -> Result<AirState> {
        self.check(engine)?;
        let n = self.config.grid.len as u64;
        let mut commands = self.owner.create_command_encoder(&Default::default());
        commands.copy_buffer_to_buffer(&self.buffers[1], 0, &self.staging, 0, n * 16);
        commands.copy_buffer_to_buffer(&self.buffers[3], 0, &self.staging, n * 16, n * 4);
        engine.queue().unwrap().submit([commands.finish()]);
        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.owner
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(30)),
            })
            .map_err(|e| Error::Gpu(e.to_string()))?;
        rx.recv_timeout(std::time::Duration::from_secs(30))
            .map_err(|e| Error::Gpu(e.to_string()))?
            .map_err(|e| Error::Gpu(e.to_string()))?;
        let result = (|| {
            let bytes = molla_core::gpu_readback!(slice);
            let floats: &[f32] = bytemuck::cast_slice(&bytes);
            finite(floats)?;
            let velocity = floats[..n as usize * 4]
                .chunks_exact(4)
                .map(|v| Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64))
                .collect();
            let pressure = floats[n as usize * 4..].iter().map(|v| *v as f64).collect();
            Ok(AirState {
                config: self.config,
                velocity,
                pressure,
                drive: self.drive,
            })
        })();
        self.staging.unmap();
        result
    }

    /// Read local velocities without transferring the full air grid to the host.
    pub fn sample(&mut self, engine: &Engine, points: &[Vec3]) -> Result<Vec<Vec3>> {
        self.check(engine)?;
        let packed: Vec<_> = points.iter().copied().map(pack).collect();
        finite(bytemuck::cast_slice(&packed))?;
        let queue = engine.queue().unwrap();
        let mut result = Vec::with_capacity(points.len());
        for chunk in packed.chunks(SAMPLE_CAPACITY) {
            let mut commands = self.owner.create_command_encoder(&Default::default());
            {
                let bytes = bytemuck::cast_slice(chunk);
                let mut upload = self.uploads.write_buffer(
                    &mut commands,
                    &self.buffers[6],
                    0,
                    wgpu::BufferSize::new(bytes.len() as u64).unwrap(),
                );
                molla_compute::copy_to_mapped(&mut upload, bytes);
                molla_core::gpu_metrics::record_upload(bytes.len() as u64);
            }
            self.uploads.finish_and_recall_on_submit(&commands);
            {
                let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("local air sampling"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.program.pipelines[5]);
                pass.set_bind_group(0, &self.bindings, &[]);
                pass.dispatch_workgroups(chunk.len().div_ceil(128) as u32, 1, 1);
            }
            let size = chunk.len() as u64 * 16;
            commands.copy_buffer_to_buffer(&self.buffers[7], 0, &self.sample_staging, 0, size);
            queue.submit([commands.finish()]);
            let slice = self.sample_staging.slice(..size);
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            self.owner
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(30)),
                })
                .map_err(|e| Error::Gpu(e.to_string()))?;
            rx.recv_timeout(std::time::Duration::from_secs(30))
                .map_err(|e| Error::Gpu(e.to_string()))?
                .map_err(|e| Error::Gpu(e.to_string()))?;
            let valid = {
                let bytes = molla_core::gpu_readback!(slice);
                let values: &[f32] = bytemuck::cast_slice(&bytes);
                let valid = finite(values);
                if valid.is_ok() {
                    result.extend(
                        values
                            .chunks_exact(4)
                            .map(|v| Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64)),
                    );
                }
                valid
            };
            self.sample_staging.unmap();
            valid?;
        }
        Ok(result)
    }
}
