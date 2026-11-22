use super::{Anchor, Attachment, BladeFrame, Config, LeafBody, Particle, invalid, validate};
use crate::airflow::AirCoupling;
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_math::Vec3;
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct State {
    position: [f32; 4],
    velocity: [f32; 4],
    force: [f32; 4],
    air_force: [f32; 4],
    air_torque: [f32; 4],
    direction: [f32; 4],
    normal: [f32; 4],
    angular: [f32; 4],
}

/// Source of the air each particle step uses.
#[derive(Clone, Copy)]
enum Air<'a> {
    Uniform,
    Spatial(&'a [Vec3]),
    Coupled(Option<f64>),
}

fn v(v: Vec3, w: f64) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, w as f32]
}
fn xyz(v: [f32; 4]) -> Vec3 {
    Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64)
}
fn pack(p: Particle) -> State {
    State {
        position: v(p.position, if p.attached { 1.0 } else { 0.0 }),
        velocity: v(p.velocity, p.age),
        force: v(p.force, p.peak_force),
        air_force: v(p.air_force, p.wind_speed),
        air_torque: v(p.air_torque, p.leaf.map_or(0.0, |leaf| leaf.camber)),
        direction: p.leaf.map_or([0.0; 4], |l| v(l.frame.direction, l.length)),
        normal: p.leaf.map_or([0.0; 4], |l| v(l.frame.normal, l.width)),
        angular: p
            .leaf
            .map_or([0.0; 4], |l| v(l.frame.angular_velocity, l.area)),
    }
}
fn finite(data: &[u8]) -> Result<()> {
    if bytemuck::cast_slice::<u8, f32>(data)
        .iter()
        .any(|v| !v.is_finite())
    {
        return Err(invalid());
    }
    Ok(())
}

struct Program {
    pipeline: wgpu::ComputePipeline,
}
impl Program {
    fn new(device: &wgpu::Device) -> Result<Self> {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("breakable attachments"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("../../../molla-kernels/shaders/air_field.wgsl"),
                    include_str!("../../../molla-kernels/shaders/attachments.wgsl")
                )
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("breakable attachments"),
            layout: None,
            module: &module,
            entry_point: Some("advance"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self { pipeline })
    }
}

/// Group 1: air settings, velocity, reaction accumulator and boundary impulses.
fn air_bindings(
    device: &wgpu::Device,
    program: &Program,
    buffers: [&wgpu::Buffer; 4],
) -> wgpu::BindGroup {
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("attachment air coupling"),
        layout: &program.pipeline.get_bind_group_layout(1),
        entries: &entries,
    })
}

pub struct GpuAttachments {
    owner: Arc<wgpu::Device>,
    program: Arc<Program>,
    bindings: wgpu::BindGroup,
    buffers: Vec<wgpu::Buffer>,
    staging: wgpu::Buffer,
    coefficients: Vec<Attachment>,
    count: usize,
    has_leaf_bodies: bool,
    uploads: wgpu::util::StagingBelt,
    air: wgpu::BindGroup,
    coupled: bool,
}

impl GpuAttachments {
    pub fn new(
        engine: &Engine,
        coefficients: Vec<Attachment>,
        particles: &[Particle],
    ) -> Result<Self> {
        engine.check_device()?;
        if coefficients.is_empty() || coefficients.len() != particles.len() {
            return Err(invalid());
        }
        let count = particles.len();
        for (&a, &p) in coefficients.iter().zip(particles) {
            validate(
                a,
                p,
                Anchor {
                    position: p.position,
                    velocity: Vec3::ZERO,
                    valid: true,
                },
                Config::default(),
            )?;
        }
        let program = engine.cached_gpu_program("breakable-attachments-v8-drive", Program::new)?;
        engine.checked_gpu_setup(|device| {
            let storage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST;
            let create = |label: &str, size: u64, usage| -> Result<wgpu::Buffer> {
                if size > device.limits().max_buffer_size
                    || size > device.limits().max_storage_buffer_binding_size
                {
                    return Err(invalid());
                }
                Ok(molla_core::gpu_create_buffer!(
                    device,
                    &wgpu::BufferDescriptor {
                        label: Some(label),
                        size,
                        usage,
                        mapped_at_creation: false,
                    }
                ))
            };
            if count.div_ceil(64) > device.limits().max_compute_workgroups_per_dimension as usize {
                return Err(invalid());
            }
            let buffers = vec![
                create("attachment coefficients", count as u64 * 48, storage)?,
                create("attachment states", count as u64 * 128, storage)?,
                create("attachment anchors", count as u64 * 32, storage)?,
                create(
                    "attachment settings",
                    64,
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                )?,
                create("attachment local wind", count as u64 * 16, storage)?,
                create("attachment blade frames", count as u64 * 48, storage)?,
            ];
            let entries: Vec<_> = buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect();
            let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("attachment bindings"),
                layout: &program.pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
            let staging = create(
                "attachment readback",
                count as u64 * 128,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            )?;
            let packed: Vec<_> = coefficients
                .iter()
                .map(|a| {
                    [
                        [
                            a.mass as f32,
                            a.stiffness as f32,
                            a.damping as f32,
                            a.strength as f32,
                        ],
                        [
                            a.drag as f32,
                            a.radius as f32,
                            a.petiole.map_or(0.0, |p| p.sheltering as f32),
                            0.0,
                        ],
                        a.petiole.map_or([0.0; 4], |p| {
                            [
                                p.length as f32,
                                p.bending as f32,
                                p.torsion as f32,
                                p.damping_ratio as f32,
                            ]
                        }),
                    ]
                })
                .collect();
            let bytes = bytemuck::cast_slice(&packed);
            finite(bytes)?;
            if packed
                .iter()
                .any(|a| a[0][0] <= 0.0 || a[0][1] <= 0.0 || a[0][3] <= 0.0)
            {
                return Err(invalid());
            }
            molla_core::gpu_write_buffer!(engine.queue().unwrap(), &buffers[0], 0, bytes);
            let idle = [
                create("uncoupled air settings", 96, wgpu::BufferUsages::UNIFORM)?,
                create("uncoupled air velocity", 16, storage)?,
                create("uncoupled air reactions", 16, storage)?,
                create("uncoupled air boundary", 32, storage)?,
            ];
            let air = air_bindings(device, &program, [&idle[0], &idle[1], &idle[2], &idle[3]]);
            let mut result = Self {
                owner: engine.device().unwrap().clone(),
                program,
                bindings,
                buffers,
                staging,
                coefficients,
                count,
                has_leaf_bodies: false,
                uploads: wgpu::util::StagingBelt::new(device.clone(), count as u64 * 96 + 256),
                air,
                coupled: false,
            };
            result.restore(engine, particles)?;
            Ok(result)
        })
    }

    fn check(&self, engine: &Engine) -> Result<()> {
        engine.check_device()?;
        if !engine.device().is_some_and(|d| Arc::ptr_eq(d, &self.owner)) {
            return Err(invalid());
        }
        Ok(())
    }

    pub fn restore(&mut self, engine: &Engine, particles: &[Particle]) -> Result<()> {
        self.check(engine)?;
        if particles.len() != self.count {
            return Err(invalid());
        }
        for (&a, &p) in self.coefficients.iter().zip(particles) {
            validate(
                a,
                p,
                Anchor {
                    position: p.position,
                    velocity: Vec3::ZERO,
                    valid: true,
                },
                Config::default(),
            )?;
        }
        let states: Vec<_> = particles.iter().copied().map(pack).collect();
        let bytes = bytemuck::cast_slice(&states);
        finite(bytes)?;
        if states.iter().zip(particles).any(|(s, p)| {
            p.leaf.is_some() && (s.direction[3] <= 0.0 || s.normal[3] <= 0.0 || s.angular[3] <= 0.0)
        }) {
            return Err(invalid());
        }
        molla_core::gpu_write_buffer!(engine.queue().unwrap(), &self.buffers[1], 0, bytes);
        self.has_leaf_bodies = particles.iter().any(|p| p.leaf.is_some());
        Ok(())
    }

    pub fn step(&mut self, engine: &Engine, anchors: &[Anchor], config: Config) -> Result<()> {
        self.advance(engine, anchors, config, Air::Uniform, None)
    }

    /// Advance with a world-space air velocity for each particle, in metres per second.
    pub fn step_spatial(
        &mut self,
        engine: &Engine,
        anchors: &[Anchor],
        config: Config,
        wind: &[Vec3],
    ) -> Result<()> {
        self.advance(engine, anchors, config, Air::Spatial(wind), None)
    }

    /// Attached blade frames follow their owner; detached blades retain integrated rotation.
    pub fn step_spatial_blades(
        &mut self,
        engine: &Engine,
        anchors: &[Anchor],
        config: Config,
        wind: &[Vec3],
        frames: &[BladeFrame],
    ) -> Result<()> {
        self.advance(engine, anchors, config, Air::Spatial(wind), Some(frames))
    }

    /// Bind the air whose velocity coupled steps sample and whose next step receives reactions.
    pub fn couple_air(&mut self, engine: &Engine, air: &AirCoupling<'_>) -> Result<()> {
        self.check(engine)?;
        if !Arc::ptr_eq(air.device, &self.owner) {
            return Err(invalid());
        }
        self.air = air_bindings(
            &self.owner,
            &self.program,
            [air.settings, air.velocity, air.reactions, air.boundary],
        );
        self.coupled = true;
        Ok(())
    }

    /// Like `step_spatial_blades`, sampling the coupled air on the device; with
    /// `reaction_scale` set, each particle deposits its reaction force and torque times that
    /// scale into the air's next step.
    pub fn step_coupled_blades(
        &mut self,
        engine: &Engine,
        anchors: &[Anchor],
        config: Config,
        frames: &[BladeFrame],
        reaction_scale: Option<f64>,
    ) -> Result<()> {
        if !self.coupled || reaction_scale.is_some_and(|s| !s.is_finite() || s < 0.0) {
            return Err(invalid());
        }
        self.advance(engine, anchors, config, Air::Coupled(reaction_scale), Some(frames))
    }

    fn advance(
        &mut self,
        engine: &Engine,
        anchors: &[Anchor],
        config: Config,
        air: Air<'_>,
        frames: Option<&[BladeFrame]>,
    ) -> Result<()> {
        self.check(engine)?;
        let wind = match air {
            Air::Spatial(wind) => Some(wind),
            _ => None,
        };
        if anchors.len() != self.count
            || wind.is_some_and(|w| w.len() != self.count)
            || frames.is_some_and(|f| f.len() != self.count)
            || (self.has_leaf_bodies && config.dt > 0.1)
        {
            return Err(invalid());
        }
        validate(
            self.coefficients[0],
            Particle::attached(Vec3::ZERO),
            Anchor {
                position: Vec3::ZERO,
                velocity: Vec3::ZERO,
                valid: true,
            },
            config,
        )?;
        let local: Vec<_> = wind.unwrap_or(&[]).iter().map(|w| v(*w, 0.0)).collect();
        finite(bytemuck::cast_slice(&local))?;
        let mut blade_frames = Vec::with_capacity(frames.map_or(0, <[_]>::len));
        for frame in frames.unwrap_or(&[]) {
            let (d, n) = (frame.direction, frame.normal);
            if (d.length_squared() - 1.0).abs() > 1e-4
                || (n.length_squared() - 1.0).abs() > 1e-4
                || d.dot(n).abs() > 1e-4
            {
                return Err(invalid());
            }
            blade_frames.push([v(d, 0.0), v(n, 0.0), v(frame.angular_velocity, 0.0)]);
        }
        finite(bytemuck::cast_slice(&blade_frames))?;
        let packed: Vec<_> = anchors
            .iter()
            .map(|a| {
                [
                    v(a.position, if a.valid { 1.0 } else { 0.0 }),
                    v(a.velocity, 0.0),
                ]
            })
            .collect();
        let (sample, deposit, scale) = match air {
            Air::Coupled(scale) => (1.0, scale.map_or(0.0, |_| 1.0), scale.unwrap_or(0.0)),
            _ => (0.0, 0.0, 0.0),
        };
        let settings = [
            v(config.gravity, 0.0),
            v(config.wind, if frames.is_some() { 1.0 } else { 0.0 }),
            [
                config.dt as f32,
                config.ground as f32,
                config.strength_scale as f32,
                if wind.is_some() { 1.0 } else { 0.0 },
            ],
            [sample, deposit, scale as f32, 0.0],
        ];
        finite(bytemuck::cast_slice(&packed))?;
        finite(bytemuck::cast_slice(&settings))?;
        if settings[2][0] <= 0.0 || settings[2][2] <= 0.0 {
            return Err(invalid());
        }
        let queue = engine.queue().unwrap();
        let mut commands = self.owner.create_command_encoder(&Default::default());
        for (buffer, bytes) in [
            (&self.buffers[5], bytemuck::cast_slice::<_, u8>(&blade_frames)),
            (&self.buffers[4], bytemuck::cast_slice(&local)),
            (&self.buffers[2], bytemuck::cast_slice(&packed)),
        ] {
            if bytes.is_empty() {
                continue;
            }
            let mut upload = self.uploads.write_buffer(
                &mut commands,
                buffer,
                0,
                wgpu::BufferSize::new(bytes.len() as u64).unwrap(),
            );
            molla_compute::copy_to_mapped(&mut upload, bytes);
            molla_core::gpu_metrics::record_upload(bytes.len() as u64);
        }
        self.uploads.finish_and_recall_on_submit(&commands);
        molla_core::gpu_write_buffer!(queue, &self.buffers[3], 0, bytemuck::cast_slice(&settings));
        {
            let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("leaf attachments"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.program.pipeline);
            pass.set_bind_group(0, &self.bindings, &[]);
            pass.set_bind_group(1, &self.air, &[]);
            pass.dispatch_workgroups(self.count.div_ceil(64) as u32, 1, 1);
        }
        queue.submit([commands.finish()]);
        Ok(())
    }

    pub fn snapshot(&self, engine: &Engine) -> Result<Vec<Particle>> {
        self.check(engine)?;
        let mut commands = self.owner.create_command_encoder(&Default::default());
        commands.copy_buffer_to_buffer(
            &self.buffers[1],
            0,
            &self.staging,
            0,
            self.count as u64 * 128,
        );
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
            finite(&bytes)?;
            let records: &[State] = bytemuck::cast_slice(&bytes);
            let result: Vec<_> = records
                .iter()
                .map(|s| Particle {
                    position: xyz(s.position),
                    velocity: xyz(s.velocity),
                    attached: s.position[3] > 0.5,
                    age: s.velocity[3] as f64,
                    force: xyz(s.force),
                    peak_force: s.force[3] as f64,
                    air_force: xyz(s.air_force),
                    wind_speed: s.air_force[3] as f64,
                    air_torque: xyz(s.air_torque),
                    leaf: (s.direction[3] > 0.0).then(|| LeafBody {
                        frame: BladeFrame {
                            direction: xyz(s.direction),
                            normal: xyz(s.normal),
                            angular_velocity: xyz(s.angular),
                        },
                        length: s.direction[3] as f64,
                        width: s.normal[3] as f64,
                        area: s.angular[3] as f64,
                        camber: s.air_torque[3] as f64,
                    }),
                })
                .collect();
            for (&a, &p) in self.coefficients.iter().zip(&result) {
                validate(
                    a,
                    p,
                    Anchor {
                        position: p.position,
                        velocity: Vec3::ZERO,
                        valid: true,
                    },
                    Config::default(),
                )?;
            }
            Ok(result)
        })();
        self.staging.unmap();
        result
    }
}
