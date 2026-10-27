//! Frame-transactional soil ownership for GPU rigid worlds.

use super::*;
use crate::mpm::{
    MpmFrameBackup, MpmRigidDeviceBinding, MpmSimulation, PreparedMpmRigidDeviceBridge,
};

#[path = "rigid_soil_profile.rs"]
mod profile;
pub use profile::{MpmSoilMachineTiming, MpmSoilNewtonWork};

#[path = "rigid_soil_loads.rs"]
mod loads;
pub use loads::{MpmSoilBodyImpulse, MpmSoilFrameLoadObservation, MpmSoilFrameLoads};

#[path = "rigid_soil_fault.rs"]
mod fault;

#[path = "rigid_soil_reduction.rs"]
mod reduction;
pub use reduction::MpmSoilReactionInfo;

#[cfg(test)]
#[path = "rigid_soil_contact_trace_tests.rs"]
mod contact_trace;
#[path = "rigid_soil_projection.rs"]
mod projection;
use crate::soil_contact_projection_gpu::SoilBoundaryScan;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MpmSoilTransferInfo {
    pub fixed_order: bool,
    pub shared_chunks: bool,
    pub grid_reuse: bool,
    pub particle_ordering: bool,
    pub parallel_sort: bool,
    pub gather_bytes: u64,
    pub ordering_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MpmSoilMotionInfo {
    /// Conservative effective metres per axis per soil substep; None disables admission.
    pub bounds: Option<[f32; 3]>,
    pub retained_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MpmSoilRuntimeStats {
    pub committed_frames: u64,
    pub rolled_back_frames: u64,
    pub retained_bytes: u64,
    pub live_bindings: usize,
    pub faulted: bool,
    pub dropped_profile_frames: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct MpmSoilBodyWrench {
    pub body: molla_sim::runtime::BodyHandle,
    pub force: [f32; 3],
    /// World torque about the body's COM at the last substep's force evaluation.
    pub torque: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct MpmSoilWrenchFrame {
    pub committed_frame: u64,
    pub substep_dt: f32,
    pub bodies: Vec<MpmSoilBodyWrench>,
}

#[derive(Clone, Debug)]
pub struct MpmSoilSurfaceFrame {
    pub committed_frame: u64,
    pub surface: crate::kernels::SoilSurfaceSnapshot,
}

#[derive(Clone, Debug)]
pub struct MpmSoilBodyContactMotion {
    pub body: molla_sim::runtime::BodyHandle,
    pub motion: crate::soil_contact_motion_gpu::SoilContactMotion,
}

#[derive(Clone, Debug)]
pub struct MpmSoilContactMotionFrame {
    pub committed_frame: u64,
    pub substep_dt: f32,
    pub bodies: Vec<MpmSoilBodyContactMotion>,
}

/// Optional MPM GPU timestamps and host submission/completion durations.
#[derive(Clone, Debug)]
pub struct MpmSoilFrameProfile {
    pub committed_frame: u64,
    pub profile: crate::mpm::MpmStepProfile,
    /// Host wall durations, not GPU kernel times; completion includes the final device wait.
    pub host_stage_ms: [(&'static str, f64); 6],
    pub machine_stages: Vec<MpmSoilMachineTiming>,
    pub newton_work: Vec<MpmSoilNewtonWork>,
    /// Mapped timestamp/work bytes, including any gap between their retained ranges.
    pub machine_profile_readback_bytes: u64,
}

pub(super) enum ProfileHostStage {
    Refresh,
    Soil,
    Dynamics,
    Corrections,
    Restore,
    Completion,
}

const HOST_STAGES: [(&str, f64); 6] = [
    ("refresh-submit", 0.0),
    ("soil-bridge-submit", 0.0),
    ("dynamics-submit", 0.0),
    ("corrections-submit", 0.0),
    ("force-restore-submit", 0.0),
    ("completion", 0.0),
];

pub(super) struct RuntimeSoil {
    soil: MpmSimulation,
    bindings: Vec<MpmRigidDeviceBinding>,
    bridge: Option<PreparedMpmRigidDeviceBridge>,
    projection: Option<projection::ProjectedSoil>,
    machine_fault: Option<crate::mpm::MpmMachineFaultBridge>,
    pub(super) projection_config: Option<projection::Mode>,
    prepared_stamp: Option<SceneStamp>,
    published: Option<(SceneStamp, f32)>,
    substep_dt: f32,
    pub(super) completion_timeout: std::time::Duration,
    backup: MpmFrameBackup,
    max_bytes: u64,
    applied: bool,
    stats: MpmSoilRuntimeStats,
    profile_enabled: bool,
    profiling_current: bool,
    profile_frame: Option<u64>,
    profile_host_ms: [(&'static str, f64); 6],
    profile_completion_start: Option<std::time::Instant>,
    machine_profiler: Option<profile::MachineProfiler>,
    loads: Option<loads::FrameLoads>,
    #[cfg(test)]
    fail_after_steps: Option<usize>,
    #[cfg(test)]
    contact_trace: Option<contact_trace::Probe>,
    #[cfg(test)]
    force_separate_submissions: bool,
}

fn invalid(message: &str) -> Error {
    Error::Solver(format!("GPU rigid soil runtime: {message}"))
}

impl RuntimeSoil {
    pub(super) fn has_prepared_projection(&self) -> bool {
        self.projection.is_some()
    }

    pub(super) fn quarantine(&mut self) {
        self.stats.faulted = true;
        self.published = None;
        self.soil.discard_step_profile();
        self.profile_frame = None;
        self.profiling_current = false;
        if let Some(profiler) = &mut self.machine_profiler {
            profiler.reset();
        }
    }

    pub(super) fn validate(&self, scene: &RigidScene) -> Result<()> {
        if self.stats.faulted {
            return Err(invalid("faulted attachment requires rebuilding the world"));
        }
        if self.projection_config.is_some() {
            projection::ProjectedSoil::validate(scene)?;
        }
        if scene.breakable_joints().next().is_some() || scene.loop_joints().next().is_some() {
            return Err(invalid(
                "soil frame transactions do not support breakable or loop joints",
            ));
        }
        for binding in &self.bindings {
            if scene.body_colliders(binding.body).next().is_some() {
                return Err(invalid("soil-bound bodies cannot also own rigid colliders"));
            }
        }
        Ok(())
    }

    pub(super) fn validate_step(&self, scene: &RigidScene) -> Result<()> {
        self.validate(scene)?;
        if self.soil.active_grid_enabled() && self.projection_config.is_none() {
            return Err(invalid(
                "active soil requires projected coupling before stepping",
            ));
        }
        Ok(())
    }

    pub(super) fn wake(
        &self,
        scene: &RigidScene,
        sleeping: &mut super::super::sleep::SleepingGroups,
    ) {
        for binding in &self.bindings {
            if scene.body(binding.body).is_some_and(|b| b.enabled) {
                sleeping.wake(binding.body);
            }
        }
    }

    pub(super) fn prepare(
        &mut self,
        engine: &Arc<Engine>,
        scene: &RigidScene,
        states: &[State; 2],
    ) -> Result<()> {
        self.validate(scene)?;
        if self.bridge.is_none() && self.prepared_stamp == Some(scene.model_stamp()) {
            return Ok(());
        }
        if self
            .bridge
            .as_ref()
            .is_some_and(|b| b.validate(&self.soil, scene, states).is_ok())
        {
            return Ok(());
        }
        self.projection = None;
        self.bridge = None;
        self.machine_fault = None;
        if self
            .loads
            .as_ref()
            .is_some_and(|loads| !loads.matches_stamp(&scene.model_stamp()))
        {
            self.loads = None;
        }
        self.prepared_stamp = None;
        self.published = None;
        self.bindings.retain(|b| scene.body(b.body).is_some());
        let active: Vec<_> = self
            .bindings
            .iter()
            .copied()
            .filter(|b| scene.body(b.body).is_some_and(|b| b.enabled))
            .collect();
        if active.is_empty() {
            self.soil
                .set_prescribed_solids(engine.queue().unwrap(), &[])?;
        } else {
            let prepare = if self.soil.active_grid_enabled() {
                PreparedMpmRigidDeviceBridge::new_kinematics
            } else {
                PreparedMpmRigidDeviceBridge::new
            };
            self.bridge = Some(prepare(
                engine.clone(),
                &mut self.soil,
                scene,
                states,
                &active,
                self.max_bytes
                    - self.backup.bytes()
                    - self
                        .loads
                        .as_ref()
                        .map_or(0, loads::FrameLoads::retained_bytes),
            )?);
        }
        self.stats.live_bindings = active.len();
        self.prepared_stamp = Some(scene.model_stamp());
        self.stats.retained_bytes =
            self.backup.bytes() + self.bridge.as_ref().map_or(0, |b| b.buffer_bytes());
        self.stats.retained_bytes += self
            .loads
            .as_ref()
            .map_or(0, loads::FrameLoads::retained_bytes);
        Ok(())
    }

    pub(super) fn prepare_projection(
        &mut self,
        engine: &Arc<Engine>,
        scene: &RigidScene,
        states: &[State; 2],
        solver: &FeatherstoneSolver,
        shared_mass: Option<[&wgpu::Buffer; 2]>,
        shared_jacobians: Option<[&wgpu::Buffer; 2]>,
    ) -> Result<()> {
        if self.soil.active_grid_enabled() || self.soil.soil_motion_bound_enabled() {
            let status = solver
                .gpu_status_buffer()
                .ok_or_else(|| invalid("missing machine status"))?;
            if self
                .machine_fault
                .as_ref()
                .is_none_or(|guard| !guard.matches(&self.soil, status))
            {
                self.machine_fault =
                    Some(crate::mpm::MpmMachineFaultBridge::new(&self.soil, status)?);
            }
        }
        if let Some(projection) = &self.projection {
            if self.projection_config != Some(projection.mode())
                || !projection.matches_operators(shared_mass, shared_jacobians)
            {
                return Err(invalid(
                    "prepared boundary scan does not match coupling configuration",
                ));
            }
            return Ok(());
        }
        if self.bridge.is_none() {
            return Ok(());
        }
        let Some(mode) = self.projection_config else {
            return Ok(());
        };
        let active: Vec<_> = self
            .bindings
            .iter()
            .copied()
            .filter(|b| scene.body(b.body).is_some_and(|b| b.enabled))
            .collect();
        let projected = projection::ProjectedSoil::new(
            engine,
            &mut self.soil,
            scene,
            states,
            solver,
            shared_mass,
            shared_jacobians,
            &active,
            mode,
            self.max_bytes.saturating_sub(self.stats.retained_bytes),
        )?;
        self.stats.retained_bytes += projected.bytes();
        self.projection = Some(projected);
        Ok(())
    }

    pub(super) fn fault_detail(
        &self,
        engine: &Engine,
        completion: &soil_completion::CompletionDeadline<'_>,
    ) -> Result<Option<String>> {
        fault::detail(self, engine, completion)
    }

    pub(super) fn begin(&mut self, engine: &Engine, dt: f32) -> Result<()> {
        if let Some(projection) = &self.projection {
            projection.invalidate_newton_history(engine.queue().unwrap());
        }
        if let Some(loads) = &mut self.loads {
            loads.begin(engine, dt);
        }
        self.backup.begin(&self.soil)
    }

    pub(super) fn begin_profile(&mut self, queue: &wgpu::Queue, dt: f32, substeps: usize) {
        self.profiling_current =
            self.profile_enabled && self.soil.begin_frame_profile(queue, dt, substeps);
        if self.profiling_current {
            self.profile_host_ms = HOST_STAGES;
            self.profile_completion_start = None;
            if let Some(profiler) = &mut self.machine_profiler {
                profiler.reset();
            }
        }
        if self.profile_enabled && !self.profiling_current {
            self.stats.dropped_profile_frames = self.stats.dropped_profile_frames.saturating_add(1);
        }
    }

    pub(super) fn profile_host_start(&self) -> Option<std::time::Instant> {
        self.profiling_current.then(std::time::Instant::now)
    }

    pub(super) fn profile_host_end(
        &mut self,
        stage: ProfileHostStage,
        start: Option<std::time::Instant>,
    ) {
        if let Some(start) = start {
            self.profile_host_ms[stage as usize].1 += start.elapsed().as_secs_f64() * 1000.0;
        }
    }

    pub(super) fn profile_submissions_finished(&mut self) {
        self.profile_completion_start = self.profile_host_start();
    }

    pub(super) fn machine_timestamp(
        &mut self,
        stage: &'static str,
        substep: Option<usize>,
    ) -> Option<(wgpu::QuerySet, [u32; 2])> {
        if !self.profiling_current {
            return None;
        }
        self.machine_profiler.as_mut()?.timestamp(stage, substep)
    }

    pub(super) fn resolve_machine_profile(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.profiling_current
            && let Some(profiler) = &self.machine_profiler
        {
            profiler.resolve(encoder);
        }
    }

    pub(super) fn before_step(
        &mut self,
        engine: &Engine,
        scene: &RigidScene,
        states: &mut [State; 2],
        index: usize,
        dt: f32,
    ) -> Result<()> {
        self.validate_step(scene)?;
        self.substep_dt = dt;
        if let Some(projected) = &self.projection {
            let mut encoder = engine
                .device()
                .unwrap()
                .create_command_encoder(&Default::default());
            self.bridge.as_ref().unwrap().encode_kinematics(
                &mut encoder,
                &self.soil,
                scene,
                states,
                index,
            )?;
            projected.prepare_input(&mut encoder, index);
            engine.queue().unwrap().submit([encoder.finish()]);
            return Ok(());
        }
        if self.soil.particle_count() != 0 {
            if let Some(bridge) = &mut self.bridge {
                bridge.step_soil_and_apply_observed(
                    &mut self.soil,
                    scene,
                    states,
                    index,
                    dt,
                    |encoder| {
                        if let Some(loads) = &mut self.loads {
                            loads.encode(encoder);
                        }
                    },
                )?;
                self.applied = true;
            } else {
                self.soil
                    .step_with(engine.device().unwrap(), engine.queue().unwrap(), dt)?;
            }
        }
        #[cfg(test)]
        if let Some(steps) = &mut self.fail_after_steps {
            *steps = steps.saturating_sub(1);
            if *steps == 0 {
                return Err(invalid("injected failure after soil advancement"));
            }
        }
        Ok(())
    }

    pub(super) fn after_prediction(
        &mut self,
        engine: &Engine,
        states: &mut [State; 2],
        index: usize,
        dt: f32,
    ) -> Result<bool> {
        let Some(projected) = &mut self.projection else {
            self.propagate_machine_fault(engine)?;
            return Ok(false);
        };
        let work_source = self
            .profiling_current
            .then(|| projected.newton_work_status(index).cloned())
            .flatten();
        if work_source.is_some() {
            self.machine_profiler
                .as_ref()
                .ok_or_else(|| invalid("missing machine profiler"))?
                .validate_work_slot()?;
        }
        projected.step(
            engine,
            &mut self.soil,
            index,
            dt,
            |encoder| {
                if let Some(loads) = &mut self.loads {
                    loads.encode(encoder);
                }
                if let Some(source) = &work_source {
                    self.machine_profiler
                        .as_mut()
                        .unwrap()
                        .encode_work(encoder, source);
                }
            },
            #[cfg(test)]
            self.force_separate_submissions,
        )?;
        states[1 - index].joint_qd.mark_device_dirty();
        #[cfg(test)]
        if let Some(probe) = &mut self.contact_trace {
            probe.record(engine, index, dt);
        }
        #[cfg(test)]
        if let Some(steps) = &mut self.fail_after_steps {
            *steps = steps.saturating_sub(1);
            if *steps == 0 {
                return Err(invalid("injected failure after projected soil advancement"));
            }
        }
        Ok(true)
    }

    fn propagate_machine_fault(&self, engine: &Engine) -> Result<()> {
        if let Some(guard) = &self.machine_fault {
            engine.check_device()?;
            let mut encoder = engine
                .device()
                .unwrap()
                .create_command_encoder(&Default::default());
            guard.encode(&mut encoder, &self.soil)?;
            engine.queue().unwrap().submit([encoder.finish()]);
        }
        Ok(())
    }

    pub(super) fn after_step(&mut self, states: &mut [State; 2]) -> Result<()> {
        if self.applied {
            self.bridge.as_mut().unwrap().restore_forces(states)?;
            self.applied = false;
        }
        Ok(())
    }

    pub(super) fn rollback(&mut self, states: Option<&mut [State; 2]>) -> Result<()> {
        if self.profiling_current {
            self.soil.discard_step_profile();
            if let Some(profiler) = &mut self.machine_profiler {
                profiler.reset();
            }
            self.profiling_current = false;
            self.profile_frame = None;
        }
        let restored = states.map_or(Ok(()), |states| self.after_step(states));
        self.projection = None;
        self.bridge = None;
        self.machine_fault = None;
        self.prepared_stamp = None;
        self.applied = false;
        let result = restored.and_then(|_| {
            if self.backup.rollback(&mut self.soil)? {
                self.stats.rolled_back_frames += 1;
            }
            Ok(())
        });
        self.stats.faulted |= result.is_err();
        self.stats.live_bindings = 0;
        self.stats.retained_bytes = self.backup.bytes();
        self.stats.retained_bytes += self
            .loads
            .as_ref()
            .map_or(0, loads::FrameLoads::retained_bytes);
        result
    }

    pub(super) fn commit(&mut self) {
        self.backup.commit();
        self.stats.committed_frames += 1;
        if let Some(loads) = &mut self.loads {
            loads.commit(self.stats.committed_frames);
        }
        if self.profiling_current {
            let start = self.profile_completion_start.take();
            self.profile_host_end(ProfileHostStage::Completion, start);
            if self.soil.step_profile_pending() {
                self.profile_frame = Some(self.stats.committed_frames);
            } else {
                self.soil.discard_step_profile();
                self.stats.dropped_profile_frames =
                    self.stats.dropped_profile_frames.saturating_add(1);
            }
            self.profiling_current = false;
        }
        self.published = if self.soil.particle_count() != 0 && self.bridge.is_some() {
            self.prepared_stamp
                .clone()
                .map(|stamp| (stamp, self.substep_dt))
        } else {
            None
        };
    }
}

#[cfg(test)]
#[path = "rigid_soil_gpu_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "rigid_soil_active_runtime_tests.rs"]
mod active_tests;

#[cfg(test)]
#[path = "rigid_soil_window_tests.rs"]
mod window_tests;

#[cfg(test)]
#[path = "rigid_soil_motion_tests.rs"]
mod motion_tests;

#[cfg(test)]
#[path = "rigid_soil_projection_tests.rs"]
mod projected_tests;

impl RigidWorld {
    /// Blocking, compact last-substep moving-soil/slip observation for unilateral contact.
    /// Uses corrected velocities at the contact configuration, before machine drift.
    pub fn read_mpm_soil_contact_motion(&mut self) -> Result<MpmSoilContactMotionFrame> {
        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| invalid("GPU world required"))?;
        gpu.engine.check_device()?;
        let soil = gpu
            .soil
            .as_mut()
            .ok_or_else(|| invalid("no soil attached"))?;
        let (stamp, substep_dt) = soil
            .published
            .clone()
            .ok_or_else(|| invalid("no committed bound-soil sample"))?;
        if soil.stats.faulted || stamp != self.scene.model_stamp() {
            return Err(invalid("faulted or stale body association"));
        }
        let projection = soil
            .projection
            .as_ref()
            .ok_or_else(|| invalid("unilateral projection required"))?;
        let bodies: Vec<_> = soil
            .bindings
            .iter()
            .filter(|binding| {
                self.scene
                    .body(binding.body)
                    .is_some_and(|body| body.enabled)
            })
            .map(|binding| binding.body)
            .collect();
        let samples = projection.read_contact_motion(&gpu.engine, soil.completion_timeout)?;
        if samples.len() != bodies.len() {
            return Err(invalid("contact motion body count mismatch"));
        }
        Ok(MpmSoilContactMotionFrame {
            committed_frame: soil.stats.committed_frames,
            substep_dt,
            bodies: bodies
                .into_iter()
                .zip(samples)
                .map(|(body, motion)| MpmSoilBodyContactMotion { body, motion })
                .collect(),
        })
    }

    /// Attach owned explicit soil with exclusive primitive-body contacts.
    /// Active/compact soil requires enabling projected coupling before stepping.
    /// The budget includes retained frame rollback and rigid bridge buffers.
    /// Tree joints are supported; loop/breakable joints and colliders on bound
    /// bodies reject. Soil advances once per rigid substep with the same dt.
    pub fn attach_mpm_soil(
        &mut self,
        soil: MpmSimulation,
        bindings: Vec<MpmRigidDeviceBinding>,
        max_bytes: u64,
    ) -> Result<()> {
        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| invalid("GPU world required"))?;
        if gpu.soil.is_some() {
            return Err(invalid("detach existing soil before replacement"));
        }
        if bindings.is_empty()
            || bindings
                .iter()
                .any(|b| !self.scene.body(b.body).is_some_and(|b| b.enabled))
        {
            return Err(invalid("nonempty enabled body bindings required"));
        }
        let backup = MpmFrameBackup::new(gpu.engine.clone(), &soil, max_bytes)?;
        let mut runtime = RuntimeSoil {
            soil,
            bindings,
            bridge: None,
            projection: None,
            machine_fault: None,
            projection_config: None,
            prepared_stamp: None,
            published: None,
            substep_dt: 0.0,
            completion_timeout: std::time::Duration::from_secs(5),
            backup,
            max_bytes,
            applied: false,
            stats: MpmSoilRuntimeStats::default(),
            profile_enabled: false,
            profiling_current: false,
            profile_frame: None,
            profile_host_ms: HOST_STAGES,
            profile_completion_start: None,
            machine_profiler: None,
            loads: None,
            #[cfg(test)]
            fail_after_steps: None,
            #[cfg(test)]
            contact_trace: None,
            #[cfg(test)]
            force_separate_submissions: false,
        };
        runtime.validate(&self.scene)?;
        if gpu
            .compiled
            .as_ref()
            .is_none_or(|c| c.stamp != self.scene.model_stamp())
        {
            gpu.compile(
                &mut self.scene,
                &mut self.solver,
                self.reaction_readout,
                false,
            )?;
        }
        runtime.prepare(
            &gpu.engine,
            &self.scene,
            &gpu.compiled.as_ref().unwrap().states,
        )?;
        gpu.soil = Some(runtime);
        Ok(())
    }

    pub fn mpm_soil_stats(&self) -> Option<MpmSoilRuntimeStats> {
        self.gpu.as_ref()?.soil.as_ref().map(|s| s.stats)
    }

    /// Attached motion-bound configuration and logical guard storage, without readback.
    pub fn mpm_soil_motion_info(&self) -> Option<MpmSoilMotionInfo> {
        let (bounds, retained_bytes) = self
            .gpu
            .as_ref()?
            .soil
            .as_ref()?
            .soil
            .soil_motion_guard_info();
        Some(MpmSoilMotionInfo {
            bounds,
            retained_bytes,
        })
    }

    /// Prepares changed soil coupling without advancing a frame or reading back state.
    pub fn prepare_mpm_soil_coupling(&mut self) -> Result<()> {
        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| invalid("GPU world required"))?;
        gpu.engine.check_device()?;
        gpu.soil
            .as_ref()
            .ok_or_else(|| invalid("no soil attached"))?
            .validate(&self.scene)?;
        if gpu
            .compiled
            .as_ref()
            .is_none_or(|compiled| compiled.stamp != self.scene.model_stamp())
        {
            gpu.compile(
                &mut self.scene,
                &mut self.solver,
                self.reaction_readout,
                false,
            )?;
        }
        let compiled = gpu
            .compiled
            .as_ref()
            .ok_or_else(|| invalid("unprepared machine"))?;
        let soil = gpu.soil.as_mut().unwrap();
        soil.prepare(&gpu.engine, &self.scene, &compiled.states)?;
        soil.prepare_projection(
            &gpu.engine,
            &self.scene,
            &compiled.states,
            &self.solver,
            compiled.native_mass(),
            compiled.native_jacobians(),
        )
    }

    /// Sets conservative per-axis displacement bounds in metres per soil substep.
    /// A violation rejects the coupled frame and restores its machine/soil state.
    /// May be set before the first frame; bounds must be finite and nonnegative.
    /// Returns retained guard bytes, budgeted separately from coupling storage:
    /// 16 + 128 * max(max_particles, 1). None disables but retains the allocation.
    pub fn set_mpm_soil_motion_bound(
        &mut self,
        bound: Option<[f64; 3]>,
        max_bytes: u64,
    ) -> Result<u64> {
        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| invalid("GPU world required"))?;
        gpu.engine.check_device()?;
        let soil = gpu
            .soil
            .as_mut()
            .ok_or_else(|| invalid("no soil attached"))?;
        soil.validate(&self.scene)?;
        let bytes = soil.soil.configure_coupled_motion_bound(bound, max_bytes)?;
        soil.projection = None;
        soil.machine_fault = None;
        soil.stats.retained_bytes = soil.backup.bytes()
            + soil
                .bridge
                .as_ref()
                .map_or(0, |bridge| bridge.buffer_bytes())
            + soil
                .loads
                .as_ref()
                .map_or(0, loads::FrameLoads::retained_bytes);
        Ok(bytes)
    }

    /// Selects fused guarded particle proposals while retaining separate publication.
    /// Requires active motion bounds; prepares one pipeline without buffer growth.
    pub fn set_mpm_soil_motion_projection_fused(&mut self, enabled: bool) -> Result<()> {
        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| invalid("GPU world required"))?;
        gpu.engine.check_device()?;
        let soil = gpu
            .soil
            .as_mut()
            .ok_or_else(|| invalid("no soil attached"))?;
        soil.validate(&self.scene)?;
        soil.soil.set_soil_motion_projection_fused(enabled)
    }

    /// Retained proposal-fusion selection, without a GPU readback.
    pub fn mpm_soil_motion_projection_fused(&self) -> Option<bool> {
        Some(
            self.gpu
                .as_ref()?
                .soil
                .as_ref()?
                .soil
                .soil_motion_projection_fused(),
        )
    }

    /// Attached grid layout and physical grid/metadata bytes, without readback.
    pub fn mpm_soil_grid_storage(&self) -> Option<(crate::mpm::MpmGridStorage, u64)> {
        let soil = &self.gpu.as_ref()?.soil.as_ref()?.soil;
        Some((soil.grid_storage(), soil.grid_storage_bytes()))
    }

    /// Live transfer selection and logical retained scratch bytes, without readback.
    pub fn mpm_soil_transfer_info(&self) -> Option<MpmSoilTransferInfo> {
        let soil = &self.gpu.as_ref()?.soil.as_ref()?.soil;
        Some(MpmSoilTransferInfo {
            fixed_order: soil.deterministic_p2g_enabled(),
            shared_chunks: soil.deterministic_p2g_shared_chunks_enabled(),
            grid_reuse: soil.deterministic_p2g_grid_reuse_enabled(),
            particle_ordering: soil.particle_tile_ordering_enabled(),
            parallel_sort: soil.particle_tile_ordering_uses_parallel_sort(),
            gather_bytes: soil.deterministic_p2g_storage_bytes(),
            ordering_bytes: soil.particle_tile_ordering_storage_bytes(),
        })
    }

    /// Enable finite-mass sticking projection for one unrestricted articulation.
    /// Rigid colliders, joint bounds, loop/breakable/distance constraints and
    /// joint reaction readouts are not supported on this experimental path.
    /// Node overflow faults the whole frame; detach to change the capacity.
    pub fn enable_mpm_soil_mass_coupling(&mut self, nodes_per_body: u32) -> Result<()> {
        self.enable_mpm_soil_mass_coupling_with_boundary_scan(
            nodes_per_body,
            SoilBoundaryScan::Full,
        )
    }

    pub fn mpm_soil_boundary_scan(&self) -> Option<SoilBoundaryScan> {
        self.gpu
            .as_ref()?
            .soil
            .as_ref()?
            .projection_config
            .and_then(|mode| match mode {
                projection::Mode::Sticking { boundary_scan, .. } => Some(boundary_scan),
                projection::Mode::Unilateral(_)
                | projection::Mode::Friction { .. }
                | projection::Mode::Admm { .. } => None,
            })
    }
    pub fn mpm_soil_unilateral_config(&self) -> Option<crate::mpm::MpmUnilateralConfig> {
        match self.gpu.as_ref()?.soil.as_ref()?.projection_config? {
            projection::Mode::Unilateral(config) => Some(config),
            _ => None,
        }
    }

    pub fn mpm_soil_friction_config(&self) -> Option<crate::mpm::MpmUnilateralConfig> {
        match self.gpu.as_ref()?.soil.as_ref()?.projection_config? {
            projection::Mode::Friction { config, .. } | projection::Mode::Admm { config, .. } => {
                Some(config)
            }
            _ => None,
        }
    }

    pub fn mpm_soil_friction_fixed_walls(&self) -> Option<bool> {
        match self.gpu.as_ref()?.soil.as_ref()?.projection_config? {
            projection::Mode::Friction { fixed_walls, .. } => Some(fixed_walls),
            projection::Mode::Admm { .. } => Some(true),
            _ => None,
        }
    }

    pub fn mpm_soil_newton_enabled(&self) -> bool {
        self.gpu
            .as_ref()
            .and_then(|gpu| gpu.soil.as_ref())
            .is_some_and(|soil| {
                matches!(
                    soil.projection_config,
                    Some(projection::Mode::Friction { newton: true, .. })
                )
            })
    }

    /// True only for explicitly selected experimental ADMM contact.
    pub fn mpm_soil_admm_enabled(&self) -> bool {
        self.gpu
            .as_ref()
            .and_then(|gpu| gpu.soil.as_ref())
            .is_some_and(|soil| {
                matches!(soil.projection_config, Some(projection::Mode::Admm { .. }))
            })
    }

    /// Prepared experimental ADMM backend shared by both coupling parities.
    pub fn mpm_soil_admm_backend(&self) -> Option<&'static str> {
        let projected = self.gpu.as_ref()?.soil.as_ref()?.projection.as_ref()?;
        matches!(projected.mode(), projection::Mode::Admm { .. })
            .then_some("experimental-admm-guarded")
    }

    /// Preparation backend for an instantiated experimental ADMM coupling.
    pub fn mpm_soil_admm_preparation_backend(&self) -> Option<&'static str> {
        self.mpm_soil_admm_backend().map(|_| {
            crate::soil_admm_geometry_gpu::ExperimentalSoilAdmmGeometryGpu::preparation_backend_label()
        })
    }

    /// Enables frictionless analytic contact with certified candidates and fixed walls.
    /// Capacity, coverage and multi-body-node faults reject and roll back the frame.
    pub fn enable_mpm_soil_unilateral_coupling(
        &mut self,
        config: crate::mpm::MpmUnilateralConfig,
    ) -> Result<()> {
        config.validate()?;
        self.enable_mpm_soil_projection(projection::Mode::Unilateral(config))
    }

    /// Enables experimental finite-mass Coulomb contact without fixed-wall support.
    /// Every attached body requires one coefficient. Boundary, capacity, coverage
    /// and solver faults reject and roll back the frame. Detach to reconfigure.
    pub fn enable_mpm_soil_friction_coupling(
        &mut self,
        config: crate::mpm::MpmUnilateralConfig,
        coefficients: &[(molla_sim::runtime::BodyHandle, f32)],
    ) -> Result<()> {
        self.enable_mpm_soil_friction_mode(config, coefficients, false)
    }

    /// Enables shared finite-mass Coulomb contact with simultaneous fixed-wall response.
    /// Max iterations bounds shared sweeps; conditional Newton solves are bounded by 64.
    /// Capacity, coverage and solver faults roll back the frame. Detach to reconfigure.
    pub fn enable_mpm_soil_wall_friction_coupling(
        &mut self,
        config: crate::mpm::MpmUnilateralConfig,
        coefficients: &[(molla_sim::runtime::BodyHandle, f32)],
    ) -> Result<()> {
        self.enable_mpm_soil_friction_mode(config, coefficients, true)
    }

    fn enable_mpm_soil_friction_mode(
        &mut self,
        config: crate::mpm::MpmUnilateralConfig,
        coefficients: &[(molla_sim::runtime::BodyHandle, f32)],
        fixed_walls: bool,
    ) -> Result<()> {
        self.enable_mpm_soil_friction_solver(config, coefficients, fixed_walls, false)
    }

    /// Enables device-prepared Newton wall/Coulomb coupling with paired machine rates.
    /// Solver and coverage faults reject and roll back the frame.
    pub fn enable_mpm_soil_newton_coupling(
        &mut self,
        config: crate::mpm::MpmUnilateralConfig,
        coefficients: &[(molla_sim::runtime::BodyHandle, f32)],
    ) -> Result<()> {
        self.enable_mpm_soil_friction_solver(config, coefficients, true, true)
    }

    /// Enables experimental guarded ADMM with paired rates and frame rollback on fault.
    pub fn enable_mpm_soil_admm_coupling(
        &mut self,
        config: crate::mpm::MpmUnilateralConfig,
        coefficients: &[(molla_sim::runtime::BodyHandle, f32)],
    ) -> Result<()> {
        config.validate()?;
        let coefficients = self.ordered_mpm_soil_coefficients(coefficients)?;
        self.enable_mpm_soil_projection(projection::Mode::Admm {
            config,
            coefficients,
        })
    }

    fn enable_mpm_soil_friction_solver(
        &mut self,
        config: crate::mpm::MpmUnilateralConfig,
        coefficients: &[(molla_sim::runtime::BodyHandle, f32)],
        fixed_walls: bool,
        newton: bool,
    ) -> Result<()> {
        config.validate()?;
        let values = self.ordered_mpm_soil_coefficients(coefficients)?;
        self.enable_mpm_soil_projection(projection::Mode::Friction {
            config,
            coefficients: values,
            fixed_walls,
            newton,
        })
    }

    fn ordered_mpm_soil_coefficients(
        &self,
        coefficients: &[(molla_sim::runtime::BodyHandle, f32)],
    ) -> Result<[f32; 64]> {
        let soil = self
            .gpu
            .as_ref()
            .and_then(|gpu| gpu.soil.as_ref())
            .ok_or_else(|| invalid("no GPU soil attached"))?;
        if coefficients.len() != soil.bindings.len() || soil.bindings.len() > 64 {
            return Err(invalid(
                "friction coefficients must cover every attached body",
            ));
        }
        let mut values = [0.0; 64];
        let mut seen = [false; 64];
        for &(body, coefficient) in coefficients {
            if !coefficient.is_finite() || coefficient < 0.0 {
                return Err(invalid("finite nonnegative friction coefficient required"));
            }
            let slot = soil
                .bindings
                .iter()
                .position(|binding| binding.body == body)
                .ok_or_else(|| invalid("friction body is not attached"))?;
            if seen[slot] {
                return Err(invalid("duplicate friction body"));
            }
            seen[slot] = true;
            values[slot] = coefficient;
        }
        Ok(values)
    }

    pub fn mpm_soil_friction_coefficients(
        &self,
    ) -> Option<Vec<(molla_sim::runtime::BodyHandle, f32)>> {
        let soil = self.gpu.as_ref()?.soil.as_ref()?;
        let (projection::Mode::Friction { coefficients, .. }
        | projection::Mode::Admm { coefficients, .. }) = soil.projection_config?
        else {
            return None;
        };
        Some(
            soil.bindings
                .iter()
                .enumerate()
                .map(|(slot, binding)| (binding.body, coefficients[slot]))
                .collect(),
        )
    }

    /// Boundary enumeration is immutable while coupled; detach to reconfigure.
    pub fn enable_mpm_soil_mass_coupling_with_boundary_scan(
        &mut self,
        nodes_per_body: u32,
        boundary_scan: SoilBoundaryScan,
    ) -> Result<()> {
        if nodes_per_body == 0 {
            return Err(invalid("nonzero node capacity required"));
        }
        self.enable_mpm_soil_projection(projection::Mode::Sticking {
            nodes_per_body,
            boundary_scan,
        })
    }

    fn enable_mpm_soil_projection(&mut self, mode: projection::Mode) -> Result<()> {
        projection::ProjectedSoil::validate(&self.scene)?;
        if self.reaction_readout {
            return Err(invalid(
                "mass coupling requires nonzero node capacity and no joint reaction readout",
            ));
        }
        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| invalid("GPU world required"))?;
        gpu.engine.check_device()?;
        let soil = gpu
            .soil
            .as_mut()
            .ok_or_else(|| invalid("no soil attached"))?;
        if soil.projection_config.is_some() {
            return Err(invalid(
                "mass coupling already enabled; detach to reconfigure",
            ));
        }
        if soil.soil.compensated_rigid_wrench_enabled() {
            return Err(invalid(
                "disable compensated direct reactions before projected coupling",
            ));
        }
        let compiled = gpu
            .compiled
            .as_ref()
            .ok_or_else(|| invalid("unprepared machine"))?;
        if compiled.contacts.is_some()
            || compiled.distance.is_some()
            || compiled.reactions.is_some()
        {
            return Err(invalid(
                "mass coupling cannot share a separate constraint solver",
            ));
        }
        soil.prepare(&gpu.engine, &self.scene, &compiled.states)?;
        soil.projection_config = Some(mode);
        if let Err(error) = soil.prepare_projection(
            &gpu.engine,
            &self.scene,
            &compiled.states,
            &self.solver,
            compiled.native_mass(),
            compiled.native_jacobians(),
        ) {
            soil.projection_config = None;
            return Err(error);
        }
        soil.published = None;
        Ok(())
    }

    /// Enable bounded GPU accumulation of all substep soil impulses. Reprepare
    /// after model edits; no per-substep host readback is performed.
    pub fn prepare_mpm_soil_frame_loads(&mut self) -> Result<()> {
        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| invalid("GPU world required"))?;
        gpu.engine.check_device()?;
        gpu.soil
            .as_ref()
            .ok_or_else(|| invalid("no soil attached"))?
            .validate(&self.scene)?;
        if gpu
            .compiled
            .as_ref()
            .is_none_or(|c| c.stamp != self.scene.model_stamp())
        {
            gpu.compile(
                &mut self.scene,
                &mut self.solver,
                self.reaction_readout,
                false,
            )?;
        }
        let soil = gpu
            .soil
            .as_mut()
            .ok_or_else(|| invalid("no soil attached"))?;
        soil.validate(&self.scene)?;
        let compiled = gpu
            .compiled
            .as_ref()
            .ok_or_else(|| invalid("unprepared machine"))?;
        soil.prepare(&gpu.engine, &self.scene, &compiled.states)?;
        if soil.loads.is_some() {
            return Ok(());
        }
        let bodies: Vec<_> = soil
            .bindings
            .iter()
            .filter(|b| self.scene.body(b.body).is_some_and(|b| b.enabled))
            .map(|b| b.body)
            .collect();
        let bytes = loads::FrameLoads::bytes(bodies.len());
        let total = soil
            .stats
            .retained_bytes
            .checked_add(bytes)
            .filter(|&n| n <= soil.max_bytes)
            .ok_or_else(|| invalid("frame loads exceed coupling budget"))?;
        soil.loads = Some(loads::FrameLoads::new(
            gpu.engine.device().unwrap(),
            soil.soil.rigid_wrench_buffer(),
            soil.soil.rigid_center_buffer(),
            bodies,
            self.scene.model_stamp(),
        )?);
        soil.stats.retained_bytes = total;
        Ok(())
    }

    /// Observe total frame impulse; dividing by simulated_seconds gives mean force
    /// and mean COM torque. world_origin_angular integrates torque about world zero.
    pub fn read_mpm_soil_frame_loads(&mut self) -> Result<MpmSoilFrameLoads> {
        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| invalid("GPU world required"))?;
        let soil = gpu
            .soil
            .as_ref()
            .ok_or_else(|| invalid("no soil attached"))?;
        if soil.stats.faulted {
            return Err(invalid("faulted soil attachment"));
        }
        soil.loads
            .as_ref()
            .ok_or_else(|| invalid("frame loads not prepared"))?
            .read(
                &gpu.engine,
                &self.scene.model_stamp(),
                soil.completion_timeout,
            )
    }

    fn frame_load_observer(&mut self) -> Result<(&Engine, &mut loads::FrameLoads, SceneStamp)> {
        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| invalid("GPU world required"))?;
        let soil = gpu
            .soil
            .as_mut()
            .ok_or_else(|| invalid("no soil attached"))?;
        if soil.stats.faulted {
            return Err(invalid("faulted soil attachment"));
        }
        let loads = soil
            .loads
            .as_mut()
            .ok_or_else(|| invalid("frame loads not prepared"))?;
        Ok((&gpu.engine, loads, self.scene.model_stamp()))
    }

    /// Copy the latest committed frame without waiting; false means staging is busy.
    /// Revisions must increase within an observation generation.
    pub fn try_request_mpm_soil_frame_loads(&mut self) -> Result<bool> {
        let (engine, loads, stamp) = self.frame_load_observer()?;
        loads.request_observation(engine, &stamp)
    }

    /// Poll a delayed frame load observation without waiting for GPU completion.
    pub fn poll_mpm_soil_frame_loads(&mut self) -> Result<Option<MpmSoilFrameLoadObservation>> {
        let (engine, loads, stamp) = self.frame_load_observer()?;
        loads.poll_observation(engine, &stamp)
    }

    /// Invalidate observations without reusing a pending mapped stage prematurely.
    pub fn reset_mpm_soil_frame_load_observations(&mut self) -> Result<()> {
        let (engine, loads, stamp) = self.frame_load_observer()?;
        loads.reset_observations(engine, &stamp)
    }

    /// Checks owner/generation, model association and known device health, not freshness.
    pub fn is_current_mpm_soil_frame_load_observation(
        &self,
        sample: &MpmSoilFrameLoadObservation,
    ) -> bool {
        self.gpu.as_ref().is_some_and(|gpu| {
            gpu.engine.check_device().is_ok()
                && gpu.soil.as_ref().is_some_and(|soil| {
                    !soil.stats.faulted
                        && soil.loads.as_ref().is_some_and(|loads| {
                            loads.matches_stamp(&self.scene.model_stamp())
                                && loads.is_current_observation(sample)
                        })
                })
        })
    }

    /// Allocate retained MPM and machine timestamp storage for whole coupled frames.
    /// Unread or over-capacity captures are skipped without skipping physics.
    pub fn prepare_mpm_soil_frame_profiling(&mut self, max_substeps: usize) -> Result<()> {
        self.prepare_mpm_soil_frame_profiling_mode(max_substeps, false)
    }

    /// Prepared unilateral candidate index backend; absent for unindexed coupling.
    pub fn mpm_soil_candidate_index_backend(&self) -> Option<&'static str> {
        self.gpu
            .as_ref()?
            .soil
            .as_ref()?
            .projection
            .as_ref()?
            .indexed_sort_backend()
    }

    /// Prepared Newton backend shared by both coupling parities.
    pub fn mpm_soil_newton_backend(&self) -> Option<&'static str> {
        self.gpu
            .as_ref()?
            .soil
            .as_ref()?
            .projection
            .as_ref()?
            .newton_method_name()
    }

    /// Whether the attached projection composes guard, solve and publication submissions.
    pub fn mpm_soil_composes_submissions(&self) -> Option<bool> {
        let runtime = self.gpu.as_ref()?.soil.as_ref()?;
        runtime.projection.as_ref()?;
        Some(projection::composed_submissions(
            #[cfg(test)]
            runtime.force_separate_submissions,
        ))
    }

    /// Native rates are projected before the single final pose drift.
    pub fn mpm_soil_defers_native_prediction_drift(&self) -> bool {
        self.gpu
            .as_ref()
            .is_some_and(|gpu| gpu.native_drift_deferred())
    }

    /// Retain whole-frame timestamps with nested Newton preparation stages.
    pub fn prepare_mpm_soil_frame_profiling_with_preparation(
        &mut self,
        max_substeps: usize,
    ) -> Result<()> {
        self.prepare_mpm_soil_frame_profiling_mode(max_substeps, true)
    }

    fn prepare_mpm_soil_frame_profiling_mode(
        &mut self,
        max_substeps: usize,
        details: bool,
    ) -> Result<()> {
        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| invalid("GPU world required"))?;
        gpu.engine.check_device()?;
        let soil = gpu
            .soil
            .as_mut()
            .ok_or_else(|| invalid("no soil attached"))?;
        if soil.stats.faulted {
            return Err(invalid("faulted attachment"));
        }
        if details {
            soil.soil
                .prepare_step_profiling_with_preparation(max_substeps)?;
        } else {
            soil.soil.prepare_step_profiling(max_substeps)?;
        }
        soil.machine_profiler = Some(profile::MachineProfiler::new(
            gpu.engine.device().unwrap(),
            max_substeps,
        )?);
        soil.profile_enabled = true;
        Ok(())
    }

    /// Read one committed frame's kernel timestamps and host durations outside physical stepping.
    /// The retained capture can precede the latest frame when observation is delayed.
    pub fn read_mpm_soil_frame_profile(&mut self) -> Result<Option<MpmSoilFrameProfile>> {
        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| invalid("GPU world required"))?;
        gpu.engine.check_device()?;
        let soil = gpu
            .soil
            .as_mut()
            .ok_or_else(|| invalid("no soil attached"))?;
        if soil.stats.faulted {
            return Err(invalid("faulted attachment"));
        }
        let Some(committed_frame) = soil.profile_frame.take() else {
            return Ok(None);
        };
        let Some(profile) = soil.soil.read_step_profile()? else {
            return Ok(None);
        };
        let machine_profile = soil
            .machine_profiler
            .as_mut()
            .map(|p| p.read(&gpu.engine, soil.completion_timeout))
            .transpose()?
            .unwrap_or_default();
        Ok(Some(MpmSoilFrameProfile {
            committed_frame,
            profile,
            host_stage_ms: soil.profile_host_ms,
            machine_stages: machine_profile.stages,
            newton_work: machine_profile.work,
            machine_profile_readback_bytes: machine_profile.readback_bytes,
        }))
    }

    /// Set the GPU completion budget for an attached soil frame (default five seconds).
    /// Expiry quarantines the shared logical device and invalidates the attachment;
    /// it does not reset the physical GPU or perform a GPU rollback.
    pub fn set_mpm_soil_completion_timeout(&mut self, timeout: std::time::Duration) -> Result<()> {
        if timeout.is_zero() || std::time::Instant::now().checked_add(timeout).is_none() {
            return Err(invalid("invalid completion timeout"));
        }
        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| invalid("GPU world required"))?;
        gpu.engine.check_device()?;
        let soil = gpu
            .soil
            .as_mut()
            .ok_or_else(|| invalid("no soil attached"))?;
        if soil.stats.faulted {
            return Err(invalid("faulted attachment"));
        }
        soil.completion_timeout = timeout;
        Ok(())
    }

    /// Blocking observation of the last substep of the last committed frame,
    /// not a frame-average load. Model edits invalidate the body association.
    pub fn read_mpm_soil_wrenches(&mut self) -> Result<MpmSoilWrenchFrame> {
        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| invalid("GPU world required"))?;
        let soil = gpu
            .soil
            .as_mut()
            .ok_or_else(|| invalid("no soil attached"))?;
        let (stamp, substep_dt) = soil
            .published
            .clone()
            .ok_or_else(|| invalid("no committed bound-soil sample"))?;
        if soil.stats.faulted || stamp != self.scene.model_stamp() {
            return Err(invalid("faulted or stale body association"));
        }
        let bindings: Vec<_> = soil
            .bindings
            .iter()
            .filter(|b| self.scene.body(b.body).is_some_and(|b| b.enabled))
            .collect();
        let values = soil.soil.read_rigid_wrenches(&gpu.engine)?;
        if values.len() != bindings.len() {
            return Err(invalid("wrench count mismatch"));
        }
        Ok(MpmSoilWrenchFrame {
            committed_frame: soil.stats.committed_frames,
            substep_dt,
            bodies: bindings
                .into_iter()
                .zip(values)
                .map(|(binding, (force, torque))| MpmSoilBodyWrench {
                    body: binding.body,
                    force,
                    torque,
                })
                .collect(),
        })
    }

    /// Explicit blocking observation, outside machine/soil stepping.
    pub fn read_mpm_soil_positions(&mut self) -> Result<Vec<[f32; 3]>> {
        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| invalid("GPU world required"))?;
        let soil = gpu
            .soil
            .as_mut()
            .ok_or_else(|| invalid("no soil attached"))?;
        if soil.stats.faulted {
            return Err(invalid("faulted soil state"));
        }
        soil.soil.read_positions(&gpu.engine)
    }

    /// Blocking lossless particle-state observation at the last committed frame.
    /// Rejects uncommitted, faulted, stale and over-limit state before allocation.
    /// Staging is 160 bytes per particle plus 16 bytes; host encoding is separate.
    /// Excludes machine state, grid scratch, Newton history and world-tile IDs.
    pub fn read_mpm_soil_checkpoint(
        &mut self,
        max_particles: u32,
    ) -> Result<(u64, crate::mpm::MpmSoilCheckpoint)> {
        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| invalid("GPU world required"))?;
        let soil = gpu
            .soil
            .as_mut()
            .ok_or_else(|| invalid("no soil attached"))?;
        let (stamp, _) = soil
            .published
            .as_ref()
            .ok_or_else(|| invalid("no committed bound-soil sample"))?;
        if soil.stats.faulted || *stamp != self.scene.model_stamp() {
            return Err(invalid("faulted or stale body association"));
        }
        if soil.soil.particle_count() > max_particles {
            return Err(invalid("soil checkpoint exceeds particle limit"));
        }
        let queue = gpu
            .engine
            .queue()
            .ok_or_else(|| invalid("GPU queue required"))?;
        let checkpoint = soil.soil.checkpoint_soil_with(queue)?;
        Ok((soil.stats.committed_frames, checkpoint))
    }

    /// Moves physical soil-grid bounds at a committed coupled-frame boundary.
    /// Captures bounded soil state; particles, rigid poses and velocities stay put.
    /// Changed axes require lattice-preserving interior support. Owned soil and
    /// swept guards reject; coordinate rebasing and tile activation are separate.
    /// Invalidates prepared coupling and last-substep observations until stepping.
    pub fn move_mpm_soil_grid_window(
        &mut self,
        origin: [f32; 3],
        max_particles: u32,
    ) -> Result<u64> {
        self.change_mpm_soil_grid_window(max_particles, |soil, checkpoint| {
            soil.move_grid_window_from_checkpoint(checkpoint, origin)
        })
    }

    /// Moves to absolute integer window cells against the unchanged lattice anchor.
    /// Requires a committed frame and bounded interior support on moved axes.
    /// Preserves particle/machine coordinates and invalidates prepared observations.
    pub fn move_mpm_soil_grid_window_cells(
        &mut self,
        cells: [i32; 3],
        max_particles: u32,
    ) -> Result<u64> {
        self.change_mpm_soil_grid_window(max_particles, |soil, checkpoint| {
            soil.move_grid_window_cells_from_checkpoint(checkpoint, cells)
        })
    }

    fn change_mpm_soil_grid_window(
        &mut self,
        max_particles: u32,
        change: impl FnOnce(&mut MpmSimulation, &crate::mpm::MpmSoilCheckpoint) -> Result<()>,
    ) -> Result<u64> {
        let (frame, checkpoint) = self.read_mpm_soil_checkpoint(max_particles)?;
        let gpu = self.gpu.as_mut().unwrap();
        gpu.engine.check_device()?;
        let soil = gpu.soil.as_mut().unwrap();
        change(&mut soil.soil, &checkpoint)?;
        let changed = checkpoint.config().grid_origin.map(f32::to_bits)
            != soil.soil.config().grid_origin.map(f32::to_bits)
            || checkpoint.grid_window_cells() != soil.soil.grid_window_cells();
        if changed {
            soil.projection = None;
            soil.bridge = None;
            soil.machine_fault = None;
            soil.prepared_stamp = None;
            soil.published = None;
            soil.soil.discard_step_profile();
            soil.profile_frame = None;
            if let Some(profiler) = &mut soil.machine_profiler {
                profiler.reset();
            }
            soil.stats.retained_bytes = soil.backup.bytes()
                + soil
                    .loads
                    .as_ref()
                    .map_or(0, loads::FrameLoads::retained_bytes);
        }
        Ok(frame)
    }

    /// Borrows live particle rendering inputs and their committed-frame ordinal.
    /// Rejects faulted worlds; rendering must use the simulation device and queue.
    pub fn mpm_soil_particle_render_view(
        &self,
    ) -> Result<(u64, crate::mpm::MpmParticleRenderView<'_>)> {
        let gpu = self
            .gpu
            .as_ref()
            .ok_or_else(|| invalid("GPU world required"))?;
        let soil = gpu
            .soil
            .as_ref()
            .ok_or_else(|| invalid("no soil attached"))?;
        if soil.stats.faulted {
            return Err(invalid("faulted soil state"));
        }
        Ok((
            soil.stats.committed_frames,
            soil.soil.particle_render_view(&gpu.engine)?,
        ))
    }

    /// Prepare an externally owned observation bridge with a separate explicit byte budget.
    pub fn prepare_mpm_soil_surface(
        &self,
        config: crate::kernels::SoilSurfaceConfig,
        max_owned_bytes: u64,
    ) -> Result<crate::kernels::PreparedSoilSurface> {
        let gpu = self
            .gpu
            .as_ref()
            .ok_or_else(|| invalid("GPU world required"))?;
        let soil = gpu
            .soil
            .as_ref()
            .ok_or_else(|| invalid("no soil attached"))?;
        if soil.stats.faulted {
            return Err(invalid("faulted soil state"));
        }
        soil.soil
            .prepare_geometric_surface(&gpu.engine, config, max_owned_bytes)
    }

    /// Queue optional geometry telemetry tagged with the currently committed frame.
    pub fn try_request_mpm_soil_surface(
        &self,
        surface: &mut crate::kernels::PreparedSoilSurface,
    ) -> Result<bool> {
        let gpu = self
            .gpu
            .as_ref()
            .ok_or_else(|| invalid("GPU world required"))?;
        let soil = gpu
            .soil
            .as_ref()
            .ok_or_else(|| invalid("no soil attached"))?;
        if soil.stats.faulted {
            return Err(invalid("faulted soil state"));
        }
        soil.soil
            .try_request_geometric_surface(&gpu.engine, surface, soil.stats.committed_frames)
    }

    /// Poll delayed geometry without waiting; its revision is the captured committed frame.
    pub fn poll_mpm_soil_surface(
        &self,
        surface: &mut crate::kernels::PreparedSoilSurface,
    ) -> Result<Option<crate::kernels::SoilSurfaceObservation>> {
        let gpu = self
            .gpu
            .as_ref()
            .ok_or_else(|| invalid("GPU world required"))?;
        let soil = gpu
            .soil
            .as_ref()
            .ok_or_else(|| invalid("no soil attached"))?;
        if soil.stats.faulted {
            return Err(invalid("faulted soil state"));
        }
        soil.soil.poll_geometric_surface(&gpu.engine, surface)
    }

    /// Export completed soil geometry with committed-frame provenance and bounded readback.
    pub fn read_mpm_soil_surface(
        &self,
        surface: &mut crate::kernels::PreparedSoilSurface,
    ) -> Result<MpmSoilSurfaceFrame> {
        let gpu = self
            .gpu
            .as_ref()
            .ok_or_else(|| invalid("GPU world required"))?;
        let soil = gpu
            .soil
            .as_ref()
            .ok_or_else(|| invalid("no soil attached"))?;
        if soil.stats.faulted {
            return Err(invalid("faulted soil state"));
        }
        Ok(MpmSoilSurfaceFrame {
            committed_frame: soil.stats.committed_frames,
            surface: soil.soil.read_geometric_surface(
                &gpu.engine,
                surface,
                soil.completion_timeout,
            )?,
        })
    }

    /// Return the owned soil and clear runtime-prescribed solids on a healthy device.
    pub fn detach_mpm_soil(&mut self) -> Result<Option<MpmSimulation>> {
        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| invalid("GPU world required"))?;
        gpu.engine.check_device()?;
        if gpu.soil.as_ref().is_some_and(|s| s.stats.faulted) {
            return Err(invalid("faulted soil cannot be returned as valid state"));
        }
        let Some(mut soil) = gpu.soil.take() else {
            return Ok(None);
        };
        soil.projection = None;
        soil.bridge = None;
        soil.soil
            .set_prescribed_solids(gpu.engine.queue().unwrap(), &[])?;
        Ok(Some(soil.soil))
    }
}
