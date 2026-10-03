use crate::FeatherstoneSolver;
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_math::Real;
use molla_sim::State;
use molla_sim::runtime::{BodyKind, RigidScene};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BodyDamping {
    linear: f32,
    angular: f32,
    articulation: u32,
    enabled: u32,
}

pub(super) struct RuntimeDamping {
    pipeline: wgpu::ComputePipeline,
    groups: [wgpu::BindGroup; 2],
    coefficients: wgpu::Buffer,
    external: wgpu::Buffer,
    bodies: Vec<BodyDamping>,
    enabled: bool,
}

fn storage(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }
    )
}

impl RuntimeDamping {
    pub(super) fn new(
        engine: &Engine,
        scene: &RigidScene,
        states: &[State; 2],
        solver: &FeatherstoneSolver,
        contacts: bool,
    ) -> Result<Option<Self>> {
        let model = scene.model();
        if model.body_count == 0 {
            return Ok(None);
        }
        let device = engine.device().unwrap();
        if model.body_count.div_ceil(64)
            > device.limits().max_compute_workgroups_per_dimension as usize
            || model.body_count as u64 * 32 > device.limits().max_storage_buffer_binding_size
        {
            return Err(Error::Gpu("GPU body damping capacity exceeded".into()));
        }
        let mut bodies = vec![
            BodyDamping {
                linear: 0.,
                angular: 0.,
                articulation: u32::MAX,
                enabled: 0
            };
            model.body_count
        ];
        let children = model.joint_child.host()?;
        for (art, range) in model.articulation_start.host()?.windows(2).enumerate() {
            for &child in &children[range[0] as usize..range[1] as usize] {
                bodies[child as usize].articulation = art as u32;
            }
        }
        if bodies.iter().any(|b| b.articulation == u32::MAX) {
            return Err(Error::Gpu("missing GPU damping articulation".into()));
        }
        let coefficients = storage(
            device,
            "body damping coefficients",
            model.body_count as u64 * 16,
        );
        let external = storage(
            device,
            "body damping frame forces",
            model.body_count as u64 * 32,
        );
        let params = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("body damping parameters"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &params,
            0,
            bytemuck::cast_slice(&[model.body_count as u32, u32::from(contacts), 0, 0]),
        );
        let entries: Vec<_> = [true, true, true, true, true, false, true, true, false, true]
            .into_iter()
            .enumerate()
            .map(|(index, read_only)| wgpu::BindGroupLayoutEntry {
                binding: index as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if index == 9 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("body damping layout"),
            entries: &entries,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("body damping"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/rigid_damping.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("body damping pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("body damping"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("apply"),
            compilation_options: Default::default(),
            cache: None,
        });
        let groups = states.each_ref().map(|state| -> Result<_> {
            let buffers = [
                &coefficients,
                model.body_mass.device_buffer_current()?,
                model.body_inertia.device_buffer_current()?,
                state.body_q.device_buffer_current()?,
                state.body_qd.device_buffer_current()?,
                state.body_f.device_buffer_current()?,
                &external,
                solver.gpu_active_articulations_buffer()?,
                solver.gpu_status_buffer().unwrap(),
                &params,
            ];
            let entries: Vec<_> = buffers
                .into_iter()
                .enumerate()
                .map(|(i, buffer)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("body damping bindings"),
                layout: &layout,
                entries: &entries,
            }))
        });
        let [a, b] = groups;
        Ok(Some(Self {
            pipeline,
            groups: [a?, b?],
            coefficients,
            external,
            bodies,
            enabled: false,
        }))
    }

    pub(super) fn prepare_frame(
        &mut self,
        engine: &Engine,
        scene: &RigidScene,
        state: &State,
        dt: Real,
    ) -> Result<()> {
        let narrow = |damping: Real| -> Result<f32> {
            let value = if damping > 0. {
                1. / (1. / damping + dt)
            } else {
                0.
            };
            let narrowed = value as f32;
            if !narrowed.is_finite() || (value > 0. && narrowed == 0.) {
                return Err(Error::Gpu(
                    "body damping coefficient is outside f32 range".into(),
                ));
            }
            Ok(narrowed)
        };
        self.enabled = false;
        for (body, desc) in scene.bodies() {
            let index = scene.body_index(body).unwrap().index();
            let info = &mut self.bodies[index];
            info.linear = narrow(desc.linear_damping)?;
            info.angular = narrow(desc.angular_damping)?;
            info.enabled = u32::from(desc.enabled && desc.kind == BodyKind::Dynamic);
            self.enabled |= info.enabled != 0 && (info.linear > 0. || info.angular > 0.);
        }
        if !self.enabled {
            return Ok(());
        }
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &self.coefficients,
            0,
            bytemuck::cast_slice(&self.bodies),
        );
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            state.body_f.device_buffer_current()?,
            0,
            &self.external,
            0,
            self.bodies.len() as u64 * 32,
        );
        engine.queue().unwrap().submit([encoder.finish()]);
        Ok(())
    }

    pub(super) fn dispatch(&self, engine: &Engine, side: usize) {
        if !self.enabled {
            return;
        }
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("body damping"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.groups[side], &[]);
            pass.dispatch_workgroups((self.bodies.len() as u32).div_ceil(64), 1, 1);
        }
        engine.queue().unwrap().submit([encoder.finish()]);
    }
}
