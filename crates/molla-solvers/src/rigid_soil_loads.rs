use super::*;

#[path = "rigid_soil_loads_observation.rs"]
mod observation;
pub use observation::MpmSoilFrameLoadObservation;

#[derive(Clone, Debug)]
pub struct MpmSoilBodyImpulse {
    pub body: molla_sim::runtime::BodyHandle,
    /// World linear impulse in N s over all substeps of the frame.
    pub linear: [f32; 3],
    /// Integral of world torque about each substep's instantaneous body COM, in N m s.
    pub angular: [f32; 3],
    /// Integral of world torque about the fixed world origin, in N m s.
    pub world_origin_angular: [f32; 3],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a real GPU adapter"]
    fn frame_impulses_integrate_all_slots_and_keep_committed_sample() {
        let engine = Engine::new_gpu().unwrap();
        assert!(!engine.adapter_info().unwrap().is_software);
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let mut scene = RigidScene::new().unwrap();
        let bodies: Vec<_> = (0..65)
            .map(|_| scene.insert_body(Default::default()).unwrap())
            .collect();
        let reaction = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame load fixture"),
            size: bodies.len() as u64 * 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let centers = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame load centers"),
            size: bodies.len() as u64 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut loads = FrameLoads::new(
            device,
            &reaction,
            &centers,
            bodies.clone(),
            scene.model_stamp(),
        )
        .unwrap();
        assert_eq!(loads.retained_bytes(), 65 * 144 + 16);
        let timeout = std::time::Duration::from_secs(5);
        assert!(loads.read(&engine, &scene.model_stamp(), timeout).is_err());
        let mut expected = vec![[0.0f32; 12]; bodies.len()];
        loads.begin(&engine, 0.125);
        for step in 0..3 {
            let rows: Vec<[f32; 8]> = (0..bodies.len())
                .map(|i| {
                    std::array::from_fn(|k| {
                        if k == 3 || k == 7 {
                            999.0
                        } else {
                            (i as f32 + 1.0) * (k as f32 - 2.0) * (step as f32 - 1.0) + 4.0
                        }
                    })
                })
                .collect();
            queue.write_buffer(&reaction, 0, bytemuck::cast_slice(&rows));
            let positions: Vec<[f32; 4]> = (0..bodies.len())
                .map(|i| {
                    [
                        i as f32 * 0.25 - 4.0,
                        step as f32 * 2.0 - 1.0,
                        (i % 3) as f32 - step as f32,
                        f32::NAN,
                    ]
                })
                .collect();
            queue.write_buffer(&centers, 0, bytemuck::cast_slice(&positions));
            let mut encoder = device.create_command_encoder(&Default::default());
            loads.encode(&mut encoder);
            queue.submit([encoder.finish()]);
            for ((sum, row), center) in expected.iter_mut().zip(rows).zip(positions) {
                for k in [0, 1, 2, 4, 5, 6] {
                    sum[k] += 0.125 * row[k];
                }
                let c = glam::DVec3::new(center[0] as f64, center[1] as f64, center[2] as f64);
                let f = glam::DVec3::new(row[0] as f64, row[1] as f64, row[2] as f64);
                let t = glam::DVec3::new(row[4] as f64, row[5] as f64, row[6] as f64);
                for (k, value) in (8..11).zip((0.125 * (t + c.cross(f))).to_array()) {
                    sum[k] += value as f32;
                }
            }
        }
        loads.commit(17);
        let check = |sample: MpmSoilFrameLoads| {
            assert_eq!(sample.committed_frame, 17);
            assert_eq!(sample.simulated_seconds, 0.375);
            for ((row, expected), body) in sample.bodies.iter().zip(&expected).zip(&bodies) {
                assert_eq!(row.body, *body);
                assert_eq!(row.linear, [expected[0], expected[1], expected[2]]);
                assert_eq!(row.angular, [expected[4], expected[5], expected[6]]);
                assert_eq!(
                    row.world_origin_angular,
                    [expected[8], expected[9], expected[10]]
                );
            }
        };
        check(loads.read(&engine, &scene.model_stamp(), timeout).unwrap());
        for _ in 0..2 {
            loads.begin(&engine, 0.25);
            let mut encoder = device.create_command_encoder(&Default::default());
            loads.encode(&mut encoder);
            queue.submit([encoder.finish()]);
            check(loads.read(&engine, &scene.model_stamp(), timeout).unwrap());
        }
        loads.begin(&engine, 0.25);
        loads.commit(18);
        assert!(loads.read(&engine, &scene.model_stamp(), timeout).is_err());
        scene.remove_body(bodies[0]).unwrap();
        assert!(!loads.matches_stamp(&scene.model_stamp()));
    }

    #[test]
    #[ignore = "requires a real GPU adapter"]
    fn frame_origin_impulses_translate_and_reject_nonfinite_readback() {
        let engine = Engine::new_gpu().unwrap();
        assert!(!engine.adapter_info().unwrap().is_software);
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let mut scene = RigidScene::new().unwrap();
        let body = scene.insert_body(Default::default()).unwrap();
        let buffer = |size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("origin impulse fixture"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let reaction = buffer(32);
        let centers = buffer(16);
        let short_reaction = buffer(16);
        let two_reactions = buffer(64);
        assert!(
            FrameLoads::new(
                device,
                &short_reaction,
                &centers,
                vec![body],
                scene.model_stamp()
            )
            .is_err()
        );
        assert!(
            FrameLoads::new(
                device,
                &reaction,
                &centers,
                vec![body; 2],
                scene.model_stamp()
            )
            .is_err()
        );
        assert!(
            FrameLoads::new(
                device,
                &two_reactions,
                &centers,
                vec![body; 2],
                scene.model_stamp()
            )
            .is_err()
        );
        assert!(FrameLoads::new(device, &reaction, &centers, vec![], scene.model_stamp()).is_err());
        let mut loads =
            FrameLoads::new(device, &reaction, &centers, vec![body], scene.model_stamp()).unwrap();
        let timeout = std::time::Duration::from_secs(5);
        let mut samples = Vec::new();
        for shift in [[0.0f32; 3], [8.0, -4.0, 2.0]] {
            loads.begin(&engine, 0.25);
            for (center, force) in [
                ([1.0, 2.0, 3.0], [2.0, -3.0, 4.0]),
                ([-2.0, 1.0, 4.0], [-1.0, 6.0, 2.0]),
            ] {
                let position = [
                    center[0] + shift[0],
                    center[1] + shift[1],
                    center[2] + shift[2],
                    99.0,
                ];
                let row = [
                    force[0],
                    force[1],
                    force[2],
                    f32::NAN,
                    3.0,
                    -2.0,
                    1.0,
                    f32::NAN,
                ];
                queue.write_buffer(&centers, 0, bytemuck::cast_slice(&position));
                queue.write_buffer(&reaction, 0, bytemuck::cast_slice(&row));
                let mut encoder = device.create_command_encoder(&Default::default());
                loads.encode(&mut encoder);
                queue.submit([encoder.finish()]);
            }
            loads.commit(samples.len() as u64 + 1);
            samples.push(
                loads
                    .read(&engine, &scene.model_stamp(), timeout)
                    .unwrap()
                    .bodies
                    .remove(0),
            );
        }
        let v = |a: [f32; 3]| glam::DVec3::from_array(a.map(f64::from));
        assert_eq!(samples[0].linear, samples[1].linear);
        assert_eq!(samples[0].angular, samples[1].angular);
        assert_eq!(
            v(samples[1].world_origin_angular) - v(samples[0].world_origin_angular),
            glam::DVec3::new(8.0, -4.0, 2.0).cross(v(samples[0].linear))
        );
        let final_center_shortcut =
            v(samples[0].angular) + glam::DVec3::new(-2.0, 1.0, 4.0).cross(v(samples[0].linear));
        assert!((v(samples[0].world_origin_angular) - final_center_shortcut).length() > 1.0);
        let committed = loads.ready.unwrap().0;
        queue.write_buffer(&loads.buffers[committed], 32, bytemuck::bytes_of(&f32::NAN));
        assert!(loads.read(&engine, &scene.model_stamp(), timeout).is_err());
    }
}

#[derive(Clone, Debug)]
pub struct MpmSoilFrameLoads {
    pub committed_frame: u64,
    /// Sum of the f32 substep intervals used by soil, in seconds.
    pub simulated_seconds: f64,
    pub bodies: Vec<MpmSoilBodyImpulse>,
}

pub(super) struct FrameLoads {
    device: wgpu::Device,
    observations: observation::State,
    buffers: [wgpu::Buffer; 2],
    staging: wgpu::Buffer,
    params: wgpu::Buffer,
    groups: [wgpu::BindGroup; 2],
    pipeline: wgpu::ComputePipeline,
    bodies: Vec<molla_sim::runtime::BodyHandle>,
    stamp: SceneStamp,
    working: usize,
    ready: Option<(usize, u64, f64)>,
    duration: f64,
    steps: usize,
}

impl FrameLoads {
    pub(super) fn matches_stamp(&self, stamp: &SceneStamp) -> bool {
        self.stamp == *stamp
    }

    pub(super) fn retained_bytes(&self) -> u64 {
        Self::bytes(self.bodies.len())
    }

    pub(super) fn bytes(count: usize) -> u64 {
        count as u64 * 144 + 16
    }

    pub(super) fn new(
        device: &wgpu::Device,
        reactions: &wgpu::Buffer,
        centers: &wgpu::Buffer,
        bodies: Vec<molla_sim::runtime::BodyHandle>,
        stamp: SceneStamp,
    ) -> Result<Self> {
        let size = bodies.len() as u64 * 48;
        if size == 0
            || bodies.len() > u32::MAX as usize
            || bodies.len() as u64 * 32 > reactions.size()
            || bodies.len() as u64 * 16 > centers.size()
            || size > device.limits().max_storage_buffer_binding_size
            || (bodies.len() as u32).div_ceil(64)
                > device.limits().max_compute_workgroups_per_dimension
        {
            return Err(invalid("invalid frame load capacity"));
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
        let buffers = std::array::from_fn(|_| {
            buffer(
                "soil frame impulse",
                size,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            )
        });
        let staging = buffer(
            "soil frame impulse readback",
            size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let params = buffer(
            "soil frame impulse interval",
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("soil frame loads"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/rigid_soil_loads.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("soil frame loads"),
            layout: None,
            module: &shader,
            entry_point: Some("integrate"),
            compilation_options: Default::default(),
            cache: None,
        });
        let groups = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("soil frame loads"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: reactions.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers[i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: centers.as_entire_binding(),
                    },
                ],
            })
        });
        Ok(Self {
            device: device.clone(),
            observations: observation::State::default(),
            buffers,
            staging,
            params,
            groups,
            pipeline,
            bodies,
            stamp,
            working: 0,
            ready: None,
            duration: 0.0,
            steps: 0,
        })
    }

    pub(super) fn begin(&mut self, engine: &Engine, dt: f32) {
        self.duration = f64::from(dt);
        self.steps = 0;
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.buffers[self.working], 0, None);
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &self.params,
            0,
            bytemuck::cast_slice(&[dt.to_bits(), self.bodies.len() as u32, 0, 0])
        );
        engine.queue().unwrap().submit([encoder.finish()]);
    }

    pub(super) fn encode(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.groups[self.working], &[]);
        pass.dispatch_workgroups((self.bodies.len() as u32).div_ceil(64), 1, 1);
        molla_core::gpu_metrics::record_dispatch();
        self.steps += 1;
    }

    pub(super) fn commit(&mut self, frame: u64) {
        if self.steps != 0 {
            self.ready = Some((self.working, frame, self.duration * self.steps as f64));
            self.working = 1 - self.working;
        } else {
            self.ready = None;
        }
    }

    pub(super) fn read(
        &self,
        engine: &Engine,
        stamp: &SceneStamp,
        timeout: std::time::Duration,
    ) -> Result<MpmSoilFrameLoads> {
        self.check_observation(engine, stamp)?;
        if self.observation_pending() {
            return Err(invalid("asynchronous frame load observation pending"));
        }
        let (index, committed_frame, simulated_seconds) = self
            .ready
            .ok_or_else(|| invalid("no committed frame load sample"))?;
        let device = engine.device().unwrap();
        let completion =
            soil_completion::CompletionDeadline::new(engine.device_health().unwrap(), timeout)?;
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            &self.buffers[index],
            0,
            &self.staging,
            0,
            self.staging.size(),
        );
        engine.queue().unwrap().submit([encoder.finish()]);
        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let result = (|| {
            completion.poll(device)?;
            completion
                .receive(&rx)?
                .map_err(|e| Error::Gpu(e.to_string()))?;
            let data = molla_core::gpu_readback!(slice);
            self.decode(
                committed_frame,
                simulated_seconds,
                bytemuck::cast_slice(&data),
            )
        })();
        self.staging.unmap();
        result
    }

    fn decode(
        &self,
        committed_frame: u64,
        simulated_seconds: f64,
        rows: &[[f32; 12]],
    ) -> Result<MpmSoilFrameLoads> {
        if rows.len() != self.bodies.len()
            || !simulated_seconds.is_finite()
            || simulated_seconds <= 0.0
            || rows.iter().flatten().any(|v| !v.is_finite())
        {
            return Err(invalid("invalid frame impulse payload or duration"));
        }
        Ok(MpmSoilFrameLoads {
            committed_frame,
            simulated_seconds,
            bodies: self
                .bodies
                .iter()
                .zip(rows)
                .map(|(&body, row)| MpmSoilBodyImpulse {
                    body,
                    linear: [row[0], row[1], row[2]],
                    angular: [row[4], row[5], row[6]],
                    world_origin_angular: [row[8], row[9], row[10]],
                })
                .collect(),
        })
    }
}
