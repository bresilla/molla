//! Resident rigid-state input and force-buffer output for MPM coupling.

use super::*;
use molla_sim::State;
use molla_sim::runtime::{BodyHandle, RigidScene, SceneStamp};

#[derive(Clone, Copy, Debug)]
pub struct MpmRigidDeviceBinding {
    pub body: BodyHandle,
    pub shape: MpmSolidShape,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BindingRow {
    index: [u32; 4],
    geometry: [f32; 4],
    com_radius: [f32; 4],
}

/// Couples one soil substep to current resident rigid states without readback.
/// Call `step_soil_and_apply`, execute one GPU machine substep with the same dt,
/// then `restore_forces` before switching input states. Both external-force
/// buffers are restored, so soil reactions never become persistent loads.
/// The caller owns contact-force exclusivity and valid resident state contents.
/// This is not a RigidWorld scheduler, shared failure transaction or CPU fallback.
/// Explicit reactions require dense soil; kinematics-only bridges accept active
/// and compact grids. One live bridge owns each soil's rigid binding lease.
/// Dropping a pending bridge queues force restoration while the device is healthy.
pub struct PreparedMpmRigidDeviceBridge {
    engine: EngineRef,
    stamp: SceneStamp,
    epoch: Arc<()>,
    centers: wgpu::Buffer,
    motion: wgpu::Buffer,
    sources: [[wgpu::Buffer; 3]; 2],
    saved: Option<[wgpu::Buffer; 2]>,
    table: wgpu::Buffer,
    params: wgpu::Buffer,
    groups: [wgpu::BindGroup; 2],
    pipelines: [wgpu::ComputePipeline; 2],
    count: u32,
    pending: bool,
    failed: bool,
    _ownership: Arc<()>,
}

type EngineRef = Arc<Engine>;

fn invalid(message: &str) -> Error {
    Error::Solver(format!("MPM rigid device bridge: {message}"))
}

fn sources(engine: &Engine, state: &State) -> Result<[wgpu::Buffer; 3]> {
    let device = engine
        .device()
        .ok_or_else(|| invalid("GPU engine required"))?;
    Ok([
        state.body_q.device_buffer_for(device)?.clone(),
        state.body_qd.device_buffer_for(device)?.clone(),
        state.body_f.device_buffer_for(device)?.clone(),
    ])
}

impl PreparedMpmRigidDeviceBridge {
    /// Prepare fixed bindings outside stepping. Duplicate, missing or disabled
    /// body handles reject before changing soil ownership. Model edits/removal,
    /// source replacement and host rigid setters require explicit preparation again.
    pub fn new(
        engine: EngineRef,
        soil: &mut MpmSimulation,
        scene: &RigidScene,
        states: &[State; 2],
        bindings: &[MpmRigidDeviceBinding],
        max_bytes: u64,
    ) -> Result<Self> {
        Self::new_internal(engine, soil, scene, states, bindings, max_bytes, false)
    }

    /// Prepare resident pose/COM updates without force backups or explicit reactions.
    /// Active and compact grids require an external projected-coupling scheduler.
    pub fn new_kinematics(
        engine: EngineRef,
        soil: &mut MpmSimulation,
        scene: &RigidScene,
        states: &[State; 2],
        bindings: &[MpmRigidDeviceBinding],
        max_bytes: u64,
    ) -> Result<Self> {
        Self::new_internal(engine, soil, scene, states, bindings, max_bytes, true)
    }

    fn new_internal(
        engine: EngineRef,
        soil: &mut MpmSimulation,
        scene: &RigidScene,
        states: &[State; 2],
        bindings: &[MpmRigidDeviceBinding],
        max_bytes: u64,
        kinematics_only: bool,
    ) -> Result<Self> {
        engine.check_device()?;
        if soil.rigid_bridge.upgrade().is_some() {
            return Err(invalid("soil already has a live device bridge"));
        }
        if !kinematics_only && (soil.use_active_grid || soil.sparse_grid) {
            return Err(invalid(
                "active/compact grid requires shared machine overflow handling",
            ));
        }
        let device = engine
            .device()
            .ok_or_else(|| invalid("GPU engine required"))?;
        let queue = engine.queue().unwrap();
        crate::kernels::validate_soil_device(device, &soil.device)?;
        if bindings.is_empty()
            || bindings.len() > soil.cfg.max_rigid as usize
            || !matches!(soil.cfg.time_step, MpmTimeStep::Explicit)
            || !soil.cfg.gravity_y.is_finite()
        {
            return Err(invalid(
                "nonempty bounded binding list and explicit soil required",
            ));
        }
        let source = [sources(&engine, &states[0])?, sources(&engine, &states[1])?];
        if source[0].iter().any(|a| source[1].contains(a)) {
            return Err(invalid("input and output state buffers must be distinct"));
        }
        let bodies = scene.model().body_count;
        if source.iter().flatten().any(|b| {
            b.size() < bodies as u64 * 32
                || b.size() > device.limits().max_storage_buffer_binding_size
        }) {
            return Err(invalid("body state capacity mismatch"));
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut rows = Vec::with_capacity(bindings.len());
        let mut solids = Vec::with_capacity(bindings.len());
        for binding in bindings {
            if !scene.body(binding.body).is_some_and(|b| b.enabled) {
                return Err(invalid("missing or disabled body"));
            }
            let body = scene
                .body_index(binding.body)
                .ok_or_else(|| invalid("missing or disabled body"))?
                .index();
            if !seen.insert(body) {
                return Err(invalid("duplicate body binding"));
            }
            let pose = scene.body_pose(binding.body)?;
            let twist = scene.body_velocity(binding.body)?;
            let center = scene.body_world_com(binding.body)?;
            let solid = MpmPrescribedSolid {
                shape: binding.shape,
                center: center.as_vec3().to_array(),
                orientation: pose.rotation.as_quat().to_array(),
                linear_velocity: twist.linear.as_vec3().to_array(),
                angular_velocity: twist.angular.as_vec3().to_array(),
            };
            let (bounds, motion) = solid.rows()?;
            let com = scene.model().body_com.host()?[body].as_vec3().to_array();
            if com.iter().any(|v| !v.is_finite()) {
                return Err(invalid("nonfinite local COM"));
            }
            rows.push(BindingRow {
                index: [body as u32, 0, 0, 0],
                geometry: motion[3],
                com_radius: [com[0], com[1], com[2], bounds[3]],
            });
            solids.push(solid);
        }
        let bytes = rows.len() as u64 * 48
            + 16
            + if kinematics_only {
                0
            } else {
                source[0][2].size() + source[1][2].size()
            };
        if bytes > max_bytes
            || rows.len() as u64 * 48 > device.limits().max_storage_buffer_binding_size
            || (bindings.len() as u32).div_ceil(64)
                > device.limits().max_compute_workgroups_per_dimension
        {
            return Err(invalid("budget or dispatch capacity exceeded"));
        }
        let buffer = |label, size, usage| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false
                }
            )
        };
        let table = buffer(
            "MPM rigid body bindings",
            rows.len() as u64 * 48,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let params = buffer(
            "MPM rigid body counts",
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let saved = (!kinematics_only).then(|| {
            std::array::from_fn(|i| {
                buffer(
                    "MPM external force backup",
                    source[i][2].size(),
                    wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                )
            })
        });
        let mut entries = mixed_storage_entries(&[true, true, true, false, false, true, false]);
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 7,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MPM rigid bridge"),
            entries: &entries,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MPM rigid bridge"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/mpm_rigid_device.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MPM rigid bridge"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = ["kinematics", "apply_reactions"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let groups = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MPM rigid bridge state"),
                layout: &layout,
                entries: &[
                    buffer_entry(0, &table),
                    buffer_entry(1, &source[i][0]),
                    buffer_entry(2, &source[i][1]),
                    buffer_entry(3, &soil.rigid_center),
                    buffer_entry(4, &soil.rigid_vel),
                    buffer_entry(5, &soil.rigid_wrench),
                    buffer_entry(6, &source[i][2]),
                    buffer_entry(7, &params),
                ],
            })
        });
        molla_core::gpu_write_buffer!(queue, &table, 0, bytemuck::cast_slice(&rows));
        molla_core::gpu_write_buffer!(
            queue,
            &params,
            0,
            bytemuck::cast_slice(&[bindings.len() as u32, 0, 0, 0])
        );
        soil.set_prescribed_solids(queue, &solids)?;
        let ownership = Arc::new(());
        soil.rigid_bridge = Arc::downgrade(&ownership);
        Ok(Self {
            stamp: scene.model_stamp(),
            epoch: soil.rigid_epoch.clone(),
            centers: soil.rigid_center.clone(),
            motion: soil.rigid_vel.clone(),
            sources: source,
            saved,
            table,
            params,
            groups,
            pipelines,
            count: bindings.len() as u32,
            pending: false,
            failed: false,
            _ownership: ownership,
            engine,
        })
    }

    pub fn buffer_bytes(&self) -> u64 {
        self.table.size()
            + self.params.size()
            + self
                .saved
                .iter()
                .flatten()
                .map(wgpu::Buffer::size)
                .sum::<u64>()
    }

    pub(crate) fn validate(
        &self,
        soil: &MpmSimulation,
        scene: &RigidScene,
        states: &[State; 2],
    ) -> Result<()> {
        self.engine.check_device()?;
        if self.failed
            || (self.saved.is_some() && (soil.use_active_grid || soil.sparse_grid))
            || scene.model_stamp() != self.stamp
            || !Arc::ptr_eq(&self.epoch, &soil.rigid_epoch)
            || self.centers != soil.rigid_center
            || self.motion != soil.rigid_vel
        {
            return Err(invalid("faulted or stale scene/soil binding"));
        }
        self.validate_sources(states)
    }

    fn validate_sources(&self, states: &[State; 2]) -> Result<()> {
        self.engine.check_device()?;
        for (state, expected) in states.iter().zip(&self.sources) {
            if sources(&self.engine, state)? != *expected {
                return Err(invalid("state buffers replaced"));
            }
        }
        Ok(())
    }

    /// Encode current resident solid poses and COM velocities without advancing
    /// soil, backing up forces, or applying reactions. Encoder uses this engine.
    pub fn encode_kinematics(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        soil: &MpmSimulation,
        scene: &RigidScene,
        states: &[State; 2],
        current: usize,
    ) -> Result<()> {
        self.validate(soil, scene, states)?;
        if current >= 2 || self.pending {
            return Err(invalid("invalid state or pending explicit reaction"));
        }
        self.encode(encoder, current, 0);
        Ok(())
    }

    /// Submit resident kinematics, one explicit soil substep, then add that
    /// substep's reaction to the selected machine input. No mapping or wait.
    /// The caller must run its GPU machine substep before restoring forces.
    /// Post-submission errors poison this bridge; reconstruction is not rollback
    /// of advanced soil or machine state. GPU fault completion belongs to the caller.
    pub fn step_soil_and_apply(
        &mut self,
        soil: &mut MpmSimulation,
        scene: &RigidScene,
        states: &mut [State; 2],
        current: usize,
        dt: f32,
    ) -> Result<()> {
        self.step_soil_and_apply_observed(soil, scene, states, current, dt, |_| {})
    }

    pub(crate) fn step_soil_and_apply_observed(
        &mut self,
        soil: &mut MpmSimulation,
        scene: &RigidScene,
        states: &mut [State; 2],
        current: usize,
        dt: f32,
        observe: impl FnOnce(&mut wgpu::CommandEncoder),
    ) -> Result<()> {
        self.validate(soil, scene, states)?;
        soil.validate_prescribed_step(dt)?;
        let saved = self
            .saved
            .as_ref()
            .ok_or_else(|| invalid("kinematics-only bridge cannot apply explicit reactions"))?;
        if current >= 2 || self.pending || soil.particle_count == 0 {
            return Err(invalid("invalid state, empty soil, or unrestored forces"));
        }
        self.failed = true;
        let device = self.engine.device().unwrap();
        let queue = self.engine.queue().unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        for i in 0..2 {
            encoder.copy_buffer_to_buffer(&self.sources[i][2], 0, &saved[i], 0, saved[i].size());
        }
        self.encode(&mut encoder, current, 0);
        queue.submit([encoder.finish()]);
        #[cfg(test)]
        self.capture_boundary(soil, current)?;
        #[cfg(test)]
        self.check_submission("bridge kinematics");
        soil.step_with(device, queue, dt)?;
        #[cfg(test)]
        self.check_submission("MPM step");
        let mut encoder = device.create_command_encoder(&Default::default());
        self.encode(&mut encoder, current, 1);
        observe(&mut encoder);
        queue.submit([encoder.finish()]);
        #[cfg(test)]
        self.check_submission("reaction application");
        for state in states {
            state.body_f.mark_device_dirty();
        }
        self.pending = true;
        self.failed = false;
        Ok(())
    }

    /// Restore both prepared external-force buffers without a host transfer.
    pub fn restore_forces(&mut self, states: &mut [State; 2]) -> Result<()> {
        self.validate_sources(states)?;
        if !self.pending {
            return Err(invalid("no pending force application"));
        }
        let saved = self
            .saved
            .as_ref()
            .ok_or_else(|| invalid("no force backup prepared"))?;
        let mut encoder = self
            .engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        for i in 0..2 {
            encoder.copy_buffer_to_buffer(&saved[i], 0, &self.sources[i][2], 0, saved[i].size());
            states[i].body_f.mark_device_dirty();
        }
        self.engine.queue().unwrap().submit([encoder.finish()]);
        self.pending = false;
        Ok(())
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder, state: usize, stage: usize) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines[stage]);
        pass.set_bind_group(0, &self.groups[state], &[]);
        pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }

    #[cfg(test)]
    fn check_submission(&self, stage: &str) {
        if std::env::var_os("MOLLA_SOIL_RUNTIME_GPU_CHECKPOINTS").is_some() {
            eprintln!("soil bridge checkpoint: {stage} begin");
            if let Err(error) = self.engine.device().unwrap().poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            }) {
                eprintln!("soil bridge checkpoint: {stage} failed: {error}");
                std::process::exit(76);
            }
            eprintln!("soil bridge checkpoint: {stage} complete");
        }
    }

    #[cfg(test)]
    fn capture_boundary(&self, soil: &mut MpmSimulation, current: usize) -> Result<()> {
        let Some(path) = std::env::var_os("MOLLA_SOIL_RUNTIME_CAPTURE") else {
            return Ok(());
        };
        let path = std::path::PathBuf::from(path);
        if path.join("soil.bin").exists() {
            return Ok(());
        }
        std::fs::create_dir_all(&path).map_err(|e| invalid(&e.to_string()))?;
        let device = self.engine.device().unwrap();
        let queue = self.engine.queue().unwrap();
        for (name, buffer) in [
            ("centers", &self.centers),
            ("motion", &self.motion),
            ("body-q", &self.sources[current][0]),
            ("body-qd", &self.sources[current][1]),
        ] {
            let words = super::replay_tests::read_words(device, queue, buffer);
            eprintln!(
                "captured {name}: {:?}",
                words
                    .iter()
                    .copied()
                    .map(f32::from_bits)
                    .collect::<Vec<_>>()
            );
            let bytes: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
            std::fs::write(path.join(format!("{name}.bin")), bytes)
                .map_err(|e| invalid(&e.to_string()))?;
        }
        let checkpoint = soil.checkpoint_soil_with(queue)?;
        std::fs::write(path.join("soil.bin"), checkpoint.to_bytes())
            .map_err(|e| invalid(&e.to_string()))?;
        eprintln!("soil boundary captured before physical advancement");
        Ok(())
    }
}

impl Drop for PreparedMpmRigidDeviceBridge {
    fn drop(&mut self) {
        if !self.pending || self.engine.check_device().is_err() {
            return;
        }
        let Some(saved) = &self.saved else {
            return;
        };
        let mut encoder = self
            .engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        for i in 0..2 {
            encoder.copy_buffer_to_buffer(&saved[i], 0, &self.sources[i][2], 0, saved[i].size());
        }
        self.engine.queue().unwrap().submit([encoder.finish()]);
    }
}

#[cfg(test)]
#[path = "mpm_rigid_device_tests.rs"]
mod tests;
