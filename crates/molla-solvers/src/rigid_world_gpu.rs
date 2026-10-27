//! Device-resident articulated stepping with a frame-boundary runtime readback.

use super::RigidWorld;
use super::{ConstraintReadings, DistanceBound};
use crate::{FeatherstoneSolver, Solver};
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_math::{GpuSpatialVector, GpuTransform, Real};
use molla_sim::runtime::JointHandle;
use molla_sim::runtime::{DynamicsStamp, RigidScene, SceneStamp};
use molla_sim::{Contacts, State};
use molla_storage::{Buffer, BufferElement};
use std::collections::BTreeMap;
use std::sync::Arc;

#[path = "rigid_activation_gpu.rs"]
mod activation;
#[path = "rigid_contacts_gpu.rs"]
mod contacts;
#[path = "rigid_damping_gpu.rs"]
mod damping;
#[path = "rigid_momentum_gpu.rs"]
mod momentum;
#[path = "rigid_soil_gpu.rs"]
mod soil;
#[path = "rigid_soil_completion.rs"]
mod soil_completion;
pub use soil::{
    MpmSoilBodyContactMotion, MpmSoilBodyImpulse, MpmSoilBodyWrench, MpmSoilContactMotionFrame,
    MpmSoilFrameLoadObservation, MpmSoilFrameLoads, MpmSoilFrameProfile, MpmSoilMachineTiming,
    MpmSoilMotionInfo, MpmSoilNewtonWork, MpmSoilReactionInfo, MpmSoilRuntimeStats,
    MpmSoilSurfaceFrame, MpmSoilTransferInfo, MpmSoilWrenchFrame,
};
#[cfg(test)]
#[path = "rigid_dynamics_parity_tests.rs"]
mod dynamics_parity;
#[cfg(test)]
#[path = "rigid_paired_runtime_tests.rs"]
mod paired_runtime_tests;
#[path = "rigid_contact_precision_gpu.rs"]
mod precision;
#[cfg(test)]
#[path = "rigid_distance_scale_gpu_tests.rs"]
mod scale_tests;
#[cfg(test)]
#[path = "rigid_contact_reaction_tests.rs"]
mod tests;
pub use contacts::GpuContactCapacity;

pub(super) struct RuntimeGpu {
    engine: Arc<Engine>,
    compiled: Option<Compiled>,
    upload: Vec<u8>,
    diagnostics: GpuRuntimeDiagnostics,
    render_stamp: Option<DynamicsStamp>,
    contact_capacity: GpuContactCapacity,
    rounding: crate::kernels::FeatherstoneRounding,
    execution: crate::kernels::FeatherstoneExecution,
    paired_rates: bool,
    native_prediction: Option<(Real, u64)>,
    soil: Option<soil::RuntimeSoil>,
    #[cfg(test)]
    dynamics_probe: Option<dynamics_parity::Probe>,
    #[cfg(test)]
    fail_after_substep: Option<usize>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuRuntimeDiagnostics {
    pub model_compilations: u64,
    /// Retained tree/constraint/force/reaction groups, excluding contact stages.
    pub current_model_bind_groups_created: u64,
    pub completed_frames: u64,
    pub last_substeps: usize,
    /// State bytes uploaded for the last completed frame, excluding model and controls.
    pub last_state_upload_bytes: u64,
    /// State, readout, event and status bytes read across the last completed frame.
    pub last_state_readback_bytes: u64,
}

/// Live same-device transforms, encoded as position vec4 then xyzw quaternion vec4.
/// Retained bindings must recheck both buffer identities and the active side before
/// each draw. These are not snapshots; queue draws before the next world mutation.
pub struct GpuBodyRenderView<'a> {
    buffers: [&'a wgpu::Buffer; 2],
    active_side: usize,
    body_count: usize,
    committed_frame: u64,
}

impl<'a> GpuBodyRenderView<'a> {
    pub fn buffers(&self) -> [&'a wgpu::Buffer; 2] {
        self.buffers
    }

    pub fn active_side(&self) -> usize {
        self.active_side
    }

    pub fn body_count(&self) -> usize {
        self.body_count
    }

    pub fn committed_frame(&self) -> u64 {
        self.committed_frame
    }
}

struct Compiled {
    native_prediction: Option<[crate::native_prediction_gpu::PreparedNativePrediction; 2]>,
    stamp: SceneStamp,
    render_side: usize,
    solver_status: wgpu::Buffer,
    states: [State; 2],
    readback: wgpu::Buffer,
    offsets: [u64; 15],
    break_readback: wgpu::Buffer,
    capture: bool,
    distance: Option<crate::distance_gpu::DistanceGpu>,
    loops: Option<crate::loop_forces_gpu::LoopGpu>,
    reactions: Option<crate::reaction_gpu::ReactionGpu>,
    contacts: Option<contacts::RuntimeContacts>,
    activation: Option<activation::RuntimeActivation>,
    damping: Option<damping::RuntimeDamping>,
    motion: Option<precision::RuntimePrecision>,
    momentum: Option<momentum::MomentumGpu>,
}

impl Compiled {
    fn native_mass(&self) -> Option<[&wgpu::Buffer; 2]> {
        self.native_prediction
            .as_ref()
            .map(|p| p.each_ref().map(|p| p.mass_matrix()))
    }
    fn native_jacobians(&self) -> Option<[&wgpu::Buffer; 2]> {
        self.native_prediction
            .as_ref()
            .map(|p| p.each_ref().map(|p| p.com_jacobians()))
    }
    fn failure(&self, engine: &Engine, status: u32) -> Error {
        let detail = self
            .contacts
            .as_ref()
            .and_then(|contacts| contacts.fault_diagnostics(engine).ok())
            .filter(|s| !s.is_empty())
            .map(|s| format!(" ({s})"))
            .unwrap_or_default();
        Error::Solver(format!("GPU rigid runtime{detail} solver fault {status}"))
    }

    fn residues(&self) -> Option<&precision::RuntimePrecision> {
        self.contacts
            .as_ref()
            .and_then(|c| c.precision.as_ref())
            .or(self.motion.as_ref())
    }

    fn precision_parts(&mut self) -> (&[State; 2], Option<&mut precision::RuntimePrecision>) {
        (
            &self.states,
            self.contacts
                .as_mut()
                .and_then(|c| c.precision.as_mut())
                .or(self.motion.as_mut()),
        )
    }
}

const BREAK_STOP: u32 = 1 << 30;

#[cfg(test)]
fn trace_soil_runtime(stage: &str) {
    if std::env::var_os("MOLLA_SOIL_RUNTIME_TRACE").is_some() {
        eprintln!("soil runtime: {stage}");
    }
}

#[cfg(test)]
fn check_soil_submission(engine: &Engine, stage: &str) {
    if std::env::var_os("MOLLA_SOIL_RUNTIME_GPU_CHECKPOINTS").is_some() {
        eprintln!("soil GPU checkpoint: {stage} begin");
        let result = engine.device().unwrap().poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        });
        if let Err(error) = result {
            eprintln!("soil GPU checkpoint: {stage} failed: {error}");
            std::process::exit(76);
        }
        eprintln!("soil GPU checkpoint: {stage} complete");
    }
}

struct Segment {
    readings: ConstraintReadings,
    reactions: Option<crate::joint_reactions::ReactionFrame>,
    breaks: Vec<(
        molla_sim::runtime::JointBreakTrigger,
        crate::joint_reactions::JointReactionSample,
    )>,
    completed: usize,
    contacts: Vec<crate::penalty_contact::ContactForce>,
    awakened: Vec<molla_sim::runtime::BodyHandle>,
}

fn unsupported(what: &str) -> Error {
    Error::Solver(format!("GPU rigid runtime does not yet support {what}"))
}

fn validate(scene: &RigidScene) -> Result<()> {
    crate::kinematic_motion::validate_gpu(scene.model())?;
    if scene.loop_joints().any(|(id, j)| {
        crate::distance_constraints::enabled(scene, id, j)
            && j.kind != molla_sim::JointType::Distance
            && !(matches!(
                j.kind,
                molla_sim::JointType::Ball
                    | molla_sim::JointType::Fixed
                    | molla_sim::JointType::Revolute
                    | molla_sim::JointType::Prismatic
                    | molla_sim::JointType::D6
            ) && scene.joint_softness(id).unwrap().is_none())
            && !(scene.joint_softness(id).unwrap().is_some()
                && matches!(
                    j.kind,
                    molla_sim::JointType::Fixed
                        | molla_sim::JointType::Revolute
                        | molla_sim::JointType::Prismatic
                        | molla_sim::JointType::Ball
                        | molla_sim::JointType::D6
                ))
    }) {
        return Err(unsupported("loop constraints"));
    }
    for (id, joint) in scene.loop_joints() {
        if crate::distance_constraints::enabled(scene, id, joint)
            && !(crate::rigid_loop_constraints::axis_kind(joint.kind)
                || joint.kind == molla_sim::JointType::Ball)
            && molla_sim::joint_control::JointAxis::ALL
                .into_iter()
                .any(|axis| {
                    scene
                        .joint_axis_settings(id, axis)
                        .is_ok_and(|settings| settings.brake != 0.0)
                })
        {
            return Err(unsupported("loop axis brakes"));
        }
    }
    let model = scene.model();
    if model.shape_count != 0 {
        let offsets = model.joint_dof_offset.host()?;
        let types = model.joint_type.host()?;
        let lower = model.joint_limit_lower.host()?;
        let upper = model.joint_limit_upper.host()?;
        let enabled = model.joint_enabled.host()?;
        for j in 0..model.joint_count {
            if enabled[j] == 0 {
                continue;
            }
            let scalar = matches!(types[j], v if v==molla_sim::JointType::Prismatic as u32 || v==molla_sim::JointType::Revolute as u32);
            for d in offsets[j] as usize..offsets[j + 1] as usize {
                let scalar_bound = scalar
                    || types[j] == molla_sim::JointType::D6 as u32
                    || types[j] == molla_sim::JointType::Ball as u32;
                let locked = lower[d].is_finite() && lower[d] == upper[d];
                if !locked && !scalar_bound && (lower[d].is_finite() || upper[d].is_finite()) {
                    return Err(unsupported("coupled contacts and joint axis bounds"));
                }
            }
        }
    }
    if model
        .articulation_start
        .host()?
        .windows(2)
        .any(|r| r[1] - r[0] > 32)
    {
        return Err(unsupported("articulations larger than 32 joints"));
    }
    let control = scene.control();
    if control.joint_motor_model.host()?.iter().any(|&v| v > 1) {
        return Err(unsupported("unknown motor model"));
    }
    Ok(())
}

fn validate_paired(scene: &RigidScene, capture: bool) -> Result<()> {
    let model = scene.model();
    if capture
        || scene.colliders().next().is_some()
        || scene.loop_joints().next().is_some()
        || scene.breakable_joints().next().is_some()
        || model.body_count == 0
        || model
            .body_flags
            .host()?
            .iter()
            .any(|f| f & molla_sim::BodyFlags::IMMOVABLE != 0)
        || model.joint_type.host()?.iter().any(|&k| {
            k == molla_sim::JointType::Distance as u32 || k > molla_sim::JointType::D6 as u32
        })
        || model
            .joint_limit_lower
            .host()?
            .iter()
            .chain(model.joint_limit_upper.host()?)
            .chain(model.joint_velocity_limit.host()?)
            .any(|v| v.is_finite())
    {
        return Err(unsupported(
            "paired rates outside unrestricted dynamic trees without external constraints or reaction capture",
        ));
    }
    Ok(())
}

impl RigidWorld {
    /// Borrow current GPU body transforms; scene edits and failed steps invalidate the view.
    pub fn gpu_body_render_view(&self) -> Result<GpuBodyRenderView<'_>> {
        let gpu = self
            .gpu
            .as_ref()
            .ok_or_else(|| unsupported("CPU render buffers"))?;
        gpu.engine.check_device()?;
        let compiled = gpu
            .compiled
            .as_ref()
            .ok_or_else(|| unsupported("missing GPU state"))?;
        if gpu.render_stamp.as_ref() != Some(&self.scene.dynamics_stamp())
            || compiled.stamp != self.scene.model_stamp()
            || self.mpm_soil_stats().is_some_and(|soil| soil.faulted)
        {
            return Err(Error::Gpu(
                "GPU body render state is not published for this scene".into(),
            ));
        }
        let buffers = [0, 1].map(|side| compiled.states[side].body_q.device_buffer());
        let [Some(first), Some(second)] = buffers else {
            return Err(Error::Gpu("GPU body render buffers are empty".into()));
        };
        Ok(GpuBodyRenderView {
            buffers: [first, second],
            active_side: compiled.render_side,
            body_count: self.scene.model().body_count,
            committed_frame: gpu.diagnostics.completed_frames,
        })
    }

    pub fn gpu_diagnostics(&self) -> Option<GpuRuntimeDiagnostics> {
        self.gpu.as_ref().map(|gpu| gpu.diagnostics)
    }

    /// Reports the compiled world's native control/prediction path.
    pub fn gpu_uses_native_prediction(&self) -> bool {
        self.gpu
            .as_ref()
            .and_then(|gpu| gpu.compiled.as_ref())
            .is_some_and(|compiled| compiled.native_prediction.is_some())
    }

    pub fn gpu_contact_capacity(&self) -> Option<GpuContactCapacity> {
        self.gpu.as_ref().map(|gpu| gpu.contact_capacity)
    }

    /// Construct GPU tree dynamics and supported external constraints; unsupported features return errors.
    pub fn from_scene_gpu(scene: RigidScene, engine: Arc<Engine>) -> Result<Self> {
        Self::from_scene_gpu_with_contact_capacity(scene, engine, GpuContactCapacity::default())
    }

    /// Construct with fixed contact capacities; overflow aborts the frame without truncation.
    pub fn from_scene_gpu_with_contact_capacity(
        scene: RigidScene,
        engine: Arc<Engine>,
        contact_capacity: GpuContactCapacity,
    ) -> Result<Self> {
        Self::from_scene_gpu_with_options(
            scene,
            engine,
            contact_capacity,
            crate::kernels::FeatherstoneRounding::Storage,
        )
    }

    /// Construct with fixed capacities and an explicit immutable rounding backend.
    pub fn from_scene_gpu_with_options(
        scene: RigidScene,
        engine: Arc<Engine>,
        contact_capacity: GpuContactCapacity,
        rounding: crate::kernels::FeatherstoneRounding,
    ) -> Result<Self> {
        Self::from_scene_gpu_with_machine_options(
            scene,
            engine,
            contact_capacity,
            rounding,
            crate::kernels::FeatherstoneExecution::Generic,
        )
    }

    pub fn from_scene_gpu_with_machine_options(
        scene: RigidScene,
        engine: Arc<Engine>,
        contact_capacity: GpuContactCapacity,
        rounding: crate::kernels::FeatherstoneRounding,
        execution: crate::kernels::FeatherstoneExecution,
    ) -> Result<Self> {
        Self::from_scene_gpu_rate_mode(
            scene,
            engine,
            contact_capacity,
            rounding,
            execution,
            false,
            None,
        )
    }

    /// Construct experimental paired-rate dynamics for unrestricted dynamic trees.
    pub fn from_scene_gpu_with_paired_machine_options(
        scene: RigidScene,
        engine: Arc<Engine>,
        contact_capacity: GpuContactCapacity,
        rounding: crate::kernels::FeatherstoneRounding,
        execution: crate::kernels::FeatherstoneExecution,
    ) -> Result<Self> {
        Self::from_scene_gpu_rate_mode(
            scene,
            engine,
            contact_capacity,
            rounding,
            execution,
            true,
            None,
        )
    }

    /// Constructs native force/direct-effort prediction with paired drift at a fixed substep.
    /// Predictor storage covers both state parities; soil coupling borrows their mass matrices.
    pub fn from_scene_gpu_with_native_prediction(
        scene: RigidScene,
        engine: Arc<Engine>,
        substep_dt: Real,
        max_prediction_bytes: u64,
    ) -> Result<Self> {
        if !substep_dt.is_finite()
            || !(substep_dt as f32).is_finite()
            || substep_dt as f32 <= 0.0
            || engine
                .device()
                .is_none_or(|d| !d.features().contains(wgpu::Features::SHADER_F64))
            || max_prediction_bytes
                < 2 * crate::native_prediction_gpu::PreparedNativePrediction::required_bytes(
                    scene.model(),
                )?
        {
            return Err(unsupported(
                "native prediction timestep, storage budget or f64 device",
            ));
        }
        Self::from_scene_gpu_rate_mode(
            scene,
            engine,
            GpuContactCapacity::default(),
            crate::kernels::FeatherstoneRounding::Storage,
            crate::kernels::FeatherstoneExecution::Generic,
            true,
            Some((substep_dt, max_prediction_bytes)),
        )
    }

    fn from_scene_gpu_rate_mode(
        scene: RigidScene,
        engine: Arc<Engine>,
        contact_capacity: GpuContactCapacity,
        rounding: crate::kernels::FeatherstoneRounding,
        execution: crate::kernels::FeatherstoneExecution,
        paired_rates: bool,
        native_prediction: Option<(Real, u64)>,
    ) -> Result<Self> {
        if paired_rates {
            validate_paired(&scene, false)?;
        }
        if contact_capacity.pairs == 0
            || contact_capacity.contacts_per_island == 0
            || contact_capacity
                .pairs
                .checked_mul(molla_geometry::RESIDENT_MANIFOLD_POINTS)
                .is_none()
        {
            return Err(Error::Gpu("invalid GPU contact capacity".into()));
        }
        validate(&scene)?;
        engine.check_device()?;
        let mut world = Self::from_scene(scene)?;
        let mut gpu = RuntimeGpu {
            engine,
            compiled: None,
            upload: Vec::new(),
            diagnostics: GpuRuntimeDiagnostics::default(),
            render_stamp: None,
            contact_capacity,
            rounding,
            execution,
            paired_rates,
            native_prediction,
            soil: None,
            #[cfg(test)]
            dynamics_probe: None,
            #[cfg(test)]
            fail_after_substep: None,
        };
        gpu.compile(&mut world.scene, &mut world.solver, false, false)?;
        gpu.render_stamp = Some(world.scene.dynamics_stamp());
        world.gpu = Some(gpu);
        Ok(world)
    }

    pub(super) fn step_gpu(&mut self, dt: Real) -> Result<()> {
        if !dt.is_finite() || dt <= 0.0 || !(dt as f32).is_finite() {
            return Err(Error::Solver(
                "step dt must be positive and finite on GPU".into(),
            ));
        }
        validate(&self.scene)?;
        if !self.solver.is_gpu() {
            return Err(Error::Solver("GPU runtime requires its GPU solver".into()));
        }
        if !self.devices.is_empty() || !self.wheels.is_empty() || !self.tracks.is_empty() {
            return Err(unsupported("actuator devices, wheels or tracks"));
        }
        if let Some(soil) = &self.gpu.as_ref().unwrap().soil {
            soil.validate_step(&self.scene)?;
        }
        let changed = self.settling.changed_inputs(&self.scene, &self.wheels);
        let mut sleeping = self.sleeping.clone();
        let configured = self.solver.substeps;
        let substeps = configured.max(1);
        let sub_dt = dt / substeps as Real;
        let mut checkpoint = Some(self.scene.checkpoint_step()?);
        let mut original = None;
        let mut original_gpu = None;
        let diagnostics = self.gpu.as_ref().unwrap().diagnostics;
        self.gpu.as_mut().unwrap().render_stamp = None;
        let mut upload_bytes = 0;
        let mut readback_bytes = 0;
        let mut readings = ConstraintReadings::default();
        let mut frame = crate::joint_reactions::ReactionFrame::default();
        let mut events = Vec::new();
        let mut contacts = Vec::new();
        let result = (|| {
            let mut completed = 0;
            while completed < substeps {
                validate(&self.scene)?;
                sleeping.refresh(&self.scene, &self.wheels);
                if let Some(soil) = &self.gpu.as_ref().unwrap().soil {
                    soil.wake(&self.scene, &mut sleeping);
                }
                let graph = self.islands()?;
                let (selected, active) =
                    sleeping.selection(&self.scene, self.scene.state(), &graph)?;
                self.solver.substeps = substeps - completed;
                let segment_dt = sub_dt * self.solver.substeps as Real;
                let gpu = self.gpu.as_mut().unwrap();
                let segment = gpu.step(
                    &mut self.scene,
                    &mut self.solver,
                    segment_dt,
                    activation::FrameSelection {
                        bodies: &selected,
                        articulations: &active,
                        sleeping: &sleeping,
                    },
                    self.reaction_readout,
                )?;
                upload_bytes += gpu.diagnostics.last_state_upload_bytes;
                readback_bytes += gpu.diagnostics.last_state_readback_bytes;
                let weight = segment_dt / dt;
                readings.accumulate(segment.readings, weight, dt);
                contacts.extend(segment.contacts);
                if let Some(reactions) = segment.reactions {
                    for (owner, mean) in reactions.means {
                        let total = frame.means.entry(owner).or_default();
                        total.child = total.child + mean.child.scale(weight);
                        total.parent = total.parent + mean.parent.scale(weight);
                        total.sample_dt += mean.sample_dt;
                        total.peak_force = total.peak_force.max(mean.peak_force);
                        total.peak_torque = total.peak_torque.max(mean.peak_torque);
                    }
                    for mut sample in reactions.samples {
                        sample.substep += completed;
                        sample.time_offset = (sample.substep + 1) as Real * sub_dt;
                        frame.samples.push(sample);
                    }
                }
                let mut broken = Vec::new();
                for (trigger, mut sample) in segment.breaks {
                    sample.substep += completed;
                    sample.time_offset = (sample.substep + 1) as Real * sub_dt;
                    let joint = self.scene.joint(sample.joint).unwrap();
                    events.push(super::JointBreakEvent {
                        frame: self.completed_steps,
                        joint: sample.joint,
                        parent: joint.parent,
                        child: joint.child,
                        trigger,
                        limits: self.scene.joint_break_limits(sample.joint)?,
                        sample,
                    });
                    broken.push(sample.joint);
                }
                completed += segment.completed;
                for body in segment.awakened {
                    sleeping.wake(body);
                }
                sleeping.acknowledge(&self.scene);
                if !broken.is_empty() {
                    if original.is_none() {
                        let working = self.scene.fork()?;
                        self.scene.restore_step(checkpoint.take().unwrap())?;
                        original = Some(std::mem::replace(&mut self.scene, working));
                    }
                    self.scene.break_joints(&broken)?;
                    self.solver.substeps = (substeps - completed).max(1);
                    let previous = self.gpu.as_mut().unwrap().compile(
                        &mut self.scene,
                        &mut self.solver,
                        self.reaction_readout,
                        true,
                    )?;
                    if original_gpu.is_none() {
                        original_gpu = previous;
                    }
                }
            }
            Ok(())
        })();
        self.solver.substeps = configured;
        self.contact_revision = None;
        if let Err(error) = result {
            let detached = original.is_some();
            if let Some(scene) = original {
                self.scene = scene;
            } else if let Some(checkpoint) = checkpoint {
                self.scene.restore_step(checkpoint)?;
            }
            let gpu = self.gpu.as_mut().unwrap();
            let failed_device = gpu.engine.check_device().is_err();
            let soil_rollback = if failed_device {
                if let Some(soil) = &mut gpu.soil {
                    soil.quarantine();
                }
                Ok(())
            } else if let Some(soil) = &mut gpu.soil {
                soil.rollback(gpu.compiled.as_mut().map(|c| &mut c.states))
            } else {
                Ok(())
            };
            if !failed_device {
                if let Some((compiled, solver)) = original_gpu {
                    gpu.compiled = Some(compiled);
                    self.solver = solver;
                    self.solver.substeps = configured;
                }
                if let Some(contacts) = gpu.compiled.as_mut().and_then(|c| c.contacts.as_mut()) {
                    contacts.rollback_episodes(&gpu.engine);
                }
                if let Some(compiled) = &mut gpu.compiled
                    && let (states, Some(precision)) = compiled.precision_parts()
                {
                    if detached {
                        precision.restore_frame(
                            &gpu.engine,
                            &self.solver,
                            states,
                            self.scene.state(),
                        )?;
                    } else {
                        precision.rollback(&gpu.engine, &self.solver, states)?;
                    }
                } else {
                    gpu.compiled = None;
                }
            }
            let compilations = gpu.diagnostics.model_compilations;
            gpu.diagnostics = diagnostics;
            gpu.diagnostics.model_compilations = compilations;
            self.reaction_frame = crate::joint_reactions::ReactionFrame::default();
            self.solver.publish_frame_readouts(Vec::new(), Vec::new());
            return soil_rollback.and(Err(error));
        }
        let gpu = self.gpu.as_mut().unwrap();
        if let Some(soil) = &mut gpu.soil {
            soil.commit();
        }
        if let Some(contacts) = gpu.compiled.as_mut().and_then(|c| c.contacts.as_mut()) {
            contacts.commit_episodes(&self.scene);
        }
        gpu.diagnostics.completed_frames = diagnostics.completed_frames + 1;
        gpu.diagnostics.last_substeps = substeps;
        gpu.diagnostics.last_state_upload_bytes = upload_bytes;
        gpu.diagnostics.last_state_readback_bytes = readback_bytes;
        gpu.render_stamp = Some(self.scene.dynamics_stamp());
        self.distance_readings = readings.distance;
        self.solver.publish_frame_readouts(contacts, Vec::new());
        self.contacts_current();
        self.cone_readings = readings.cones;
        self.distance_stamp = Some(self.scene.dynamics_stamp());
        if self.reaction_readout {
            self.reaction_frame = frame;
            self.reaction_stamp = Some(self.scene.dynamics_stamp());
        }
        self.break_events.extend(events);
        sleeping.acknowledge(&self.scene);
        self.sleeping = sleeping;
        self.finish_settling(dt, &changed);
        Ok(())
    }
}

fn map_read(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<()> {
    map_read_with_completion(device, buffer, None)
}

fn map_read_with_completion(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    completion: Option<&soil_completion::CompletionDeadline<'_>>,
) -> Result<()> {
    let (tx, rx) = std::sync::mpsc::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let mapping = if let Some(completion) = completion {
        completion.poll(device)?;
        completion.receive(&rx)?
    } else {
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(e.to_string()))?;
        rx.recv().map_err(|e| Error::Gpu(e.to_string()))?
    };
    mapping.map_err(|e| Error::Gpu(e.to_string()))
}

fn defer_native_drift(native: bool, projected: bool, probe: bool) -> bool {
    cfg!(feature = "deferred-native-drift") && native && projected && !probe
}

#[test]
fn deferred_native_drift_selection() {
    for native in [false, true] {
        for projected in [false, true] {
            for probe in [false, true] {
                assert_eq!(
                    defer_native_drift(native, projected, probe),
                    cfg!(feature = "deferred-native-drift") && native && projected && !probe
                );
            }
        }
    }
}

impl RuntimeGpu {
    fn has_prediction_probe(&self) -> bool {
        #[cfg(test)]
        {
            self.dynamics_probe.is_some()
        }
        #[cfg(not(test))]
        {
            false
        }
    }

    fn native_drift_deferred(&self) -> bool {
        defer_native_drift(
            self.compiled
                .as_ref()
                .is_some_and(|c| c.native_prediction.is_some()),
            self.soil
                .as_ref()
                .is_some_and(|s| s.has_prepared_projection()),
            self.has_prediction_probe(),
        )
    }
    fn compile(
        &mut self,
        scene: &mut RigidScene,
        solver: &mut FeatherstoneSolver,
        capture: bool,
        detachment: bool,
    ) -> Result<Option<(Compiled, FeatherstoneSolver)>> {
        self.render_stamp = None;
        let device = self
            .engine
            .device()
            .cloned()
            .ok_or_else(|| Error::Gpu("missing GPU device".into()))?;
        let allocation = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let result = self.compile_inner(scene, solver, capture, detachment);
        let internal_error = wait(&device, internal.pop());
        let allocation_error = wait(&device, allocation.pop());
        if let Some(error) = allocation_error? {
            return Err(Error::Gpu(format!("GPU runtime allocation: {error}")));
        }
        if let Some(error) = internal_error? {
            return Err(Error::Gpu(format!("GPU runtime initialization: {error}")));
        }
        let (next, created, bind_groups) = result?;
        let previous = self.compiled.replace(next);
        let old_solver = std::mem::replace(solver, created);
        self.diagnostics.model_compilations += 1;
        self.diagnostics.current_model_bind_groups_created = bind_groups;
        Ok(previous.map(|compiled| (compiled, old_solver)))
    }

    fn compile_inner(
        &self,
        scene: &mut RigidScene,
        solver: &FeatherstoneSolver,
        capture: bool,
        detachment: bool,
    ) -> Result<(Compiled, FeatherstoneSolver, u64)> {
        validate(scene)?;
        if self.paired_rates {
            validate_paired(scene, capture)?;
        }
        let device = self
            .engine
            .device()
            .ok_or_else(|| Error::Gpu("missing GPU device".into()))?;
        let queue = self
            .engine
            .queue()
            .ok_or_else(|| Error::Gpu("missing GPU queue".into()))?;
        let construction_trace =
            std::env::var_os("MOLLA_GPU_CONSTRUCTION_TRACE").map(|_| std::time::Instant::now());
        let trace = |stage: &str| {
            if let Some(start) = construction_trace {
                eprintln!(
                    "GPU construction rigid-world t={:.6}s {stage}",
                    start.elapsed().as_secs_f64()
                );
            }
        };
        trace("uploads begin");
        scene.upload_model_to_device(device, queue)?;
        scene.upload_controls_to_device(device, queue)?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let construct = if self.paired_rates {
            FeatherstoneSolver::new_gpu_with_paired_rates
        } else {
            FeatherstoneSolver::new_gpu_with_options
        };
        trace("featherstone begin");
        let created = if self.native_prediction.is_some() {
            FeatherstoneSolver::new_gpu_for_native_prediction(&self.engine, scene.model())
        } else {
            construct(&self.engine, scene.model(), self.rounding, self.execution)
        };
        trace("featherstone created; validation wait begin");
        if let Some(error) = wait(device, scope.pop())? {
            return Err(Error::Gpu(format!(
                "GPU runtime pipeline validation: {error}"
            )));
        }
        let mut created = created?;
        trace("featherstone validated");
        created.substeps = solver.substeps;
        created.joint_damping = solver.joint_damping;
        created.rigorous_d_solve = self.native_prediction.is_some() || solver.rigorous_d_solve;
        created.rigorous_distance = solver.rigorous_distance;
        created.stable_motor_feedback = solver.stable_motor_feedback;
        let mut states = [scene.state().clone_host()?, scene.state().clone_host()?];
        for state in &mut states {
            state.upload_to_device(device, queue)?;
        }
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        trace("contacts begin");
        let contacts = contacts::RuntimeContacts::new(
            &self.engine,
            scene,
            &states,
            &mut created,
            self.contact_capacity,
            capture || scene.breakable_joints().next().is_some(),
            self.compiled.as_ref().and_then(|c| c.contacts.as_ref()),
            detachment,
        );
        if let Some(error) = wait(device, scope.pop())? {
            return Err(Error::Gpu(format!(
                "GPU runtime contact pipeline validation: {error}"
            )));
        }
        trace("contacts validated");
        let (mut contacts, distance, mut motion) = if let Some((contacts, distance)) = contacts? {
            (Some(contacts), distance, None)
        } else {
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            trace("precision begin");
            let motion =
                precision::RuntimePrecision::new(&self.engine, scene, &states, &mut created, None)?;
            trace("precision ready");
            let residues = motion
                .as_ref()
                .map(|_| {
                    Ok(crate::distance_gpu::DistancePrecision {
                        coordinates: [
                            created.gpu_coordinate_compensation_buffer(&states[0])?,
                            created.gpu_coordinate_compensation_buffer(&states[1])?,
                        ],
                        positions: [
                            created.gpu_position_compensation_buffer(&states[0])?,
                            created.gpu_position_compensation_buffer(&states[1])?,
                        ],
                    })
                })
                .transpose()?;
            trace("distance begin");
            let distance = crate::distance_gpu::DistanceGpu::new(
                &self.engine,
                scene,
                &states,
                created.gpu_status_buffer().unwrap(),
                created.gpu_velocity_delta_buffer().unwrap(),
                residues,
            );
            if let Some(error) = wait(device, scope.pop())? {
                return Err(Error::Gpu(format!(
                    "GPU distance pipeline validation: {error}"
                )));
            }
            let distance = distance?;
            if scene.model().requires_angular_lock_rows()? && distance.is_none() {
                return Err(Error::Gpu(
                    "partial Ball/D6 angular equality rows are missing".into(),
                ));
            }
            created.set_external_scalar_bounds(distance.is_some())?;
            (None, distance, motion)
        };
        trace("distance ready");
        let residues = contacts
            .as_mut()
            .and_then(|c| c.precision.as_mut())
            .or(motion.as_mut());
        if let Some(previous) = &self.compiled
            && let (Some(new), Some(old)) = (residues, previous.residues())
        {
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let inherited = if detachment {
                new.inherit_detachment_precision(
                    &self.engine,
                    scene.state(),
                    &created,
                    &states,
                    old,
                    solver,
                    &previous.states,
                )
            } else if previous.stamp == scene.model_stamp() {
                new.inherit_precision(
                    &self.engine,
                    scene.state(),
                    &created,
                    &states,
                    old,
                    solver,
                    &previous.states,
                )
            } else {
                new.inherit_rebuild(
                    &self.engine,
                    scene.state(),
                    &created,
                    &states,
                    old,
                    solver,
                    &previous.states,
                )
            };
            if let Some(error) = wait(device, scope.pop())? {
                return Err(Error::Gpu(format!(
                    "GPU runtime precision transfer validation: {error}"
                )));
            }
            inherited?;
        }
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let loops = if scene.model().body_count == 0 {
            Ok(None)
        } else {
            crate::loop_forces_gpu::LoopGpu::new(
                &self.engine,
                scene,
                &states,
                created.gpu_status_buffer().unwrap(),
                [
                    created.gpu_position_compensation_buffer(&states[0])?,
                    created.gpu_position_compensation_buffer(&states[1])?,
                ],
                [
                    created.gpu_coordinate_compensation_buffer(&states[0])?,
                    created.gpu_coordinate_compensation_buffer(&states[1])?,
                ],
            )
        };
        if let Some(error) = wait(device, scope.pop())? {
            return Err(Error::Gpu(format!(
                "GPU compliant loop pipeline validation: {error}"
            )));
        }
        let loops = loops?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let reactions = if capture || scene.breakable_joints().next().is_some() {
            let contact = contacts.as_ref().and_then(|c| c.reaction_buffers());
            crate::reaction_gpu::ReactionGpu::new(
                &self.engine,
                scene,
                &states,
                distance.as_ref(),
                loops.as_ref(),
                contact.map_or_else(
                    || created.gpu_velocity_delta_buffer().unwrap(),
                    |(_, delta)| delta,
                ),
                contact.map(|(loads, _)| loads),
                created.gpu_status_buffer().unwrap(),
                solver.substeps.max(1),
                capture,
            )
            .map(Some)
        } else {
            Ok(None)
        };
        if let Some(error) = wait(device, scope.pop())? {
            return Err(Error::Gpu(format!(
                "GPU reaction pipeline validation: {error}"
            )));
        }
        let reactions = reactions?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        trace("momentum begin");
        let momentum = momentum::MomentumGpu::new(
            &self.engine,
            scene.model(),
            &states,
            contacts.as_ref().map(|c| c.projection_reference()),
            created
                .gpu_paired_velocity_delta_buffer()
                .unwrap_or_else(|| created.gpu_velocity_delta_buffer().unwrap()),
            contacts.as_ref().map(|c| c.momentum_correction()),
            &created,
        );
        if let Some(error) = wait(device, scope.pop())? {
            return Err(Error::Gpu(format!(
                "GPU momentum pipeline validation: {error}"
            )));
        }
        let momentum = momentum?;
        trace("momentum ready");
        let reaction_sizes = reactions
            .as_ref()
            .filter(|_| capture)
            .map_or([0; 2], |r| r.sizes());
        let model = scene.model();
        let retained_precision =
            motion.is_some() || contacts.as_ref().is_some_and(|c| c.precision.is_some());
        let paired_rates = created.gpu_paired_velocity_delta_buffer().is_some();
        let mut offsets = [0; 15];
        for (i, bytes) in [
            4,
            model.body_count as u64 * 32,
            model.body_count as u64 * 32,
            model.joint_coord_total as u64 * 4,
            model.joint_dof_total as u64 * 4,
            distance.as_ref().map_or(0, |d| d.owners.len() as u64 * 32),
            reaction_sizes[0],
            reaction_sizes[1],
            reactions.as_ref().map_or(0, |r| r.break_events.size()),
            contacts.as_ref().map_or(0, |_| model.body_count as u64 * 4),
            if retained_precision {
                model.body_count as u64 * 16
            } else {
                0
            },
            if retained_precision {
                model.joint_coord_total as u64 * 4
            } else {
                0
            },
            if paired_rates {
                model.joint_dof_total as u64 * 8
            } else {
                0
            },
            if paired_rates {
                model.body_count as u64 * 32
            } else {
                0
            },
        ]
        .into_iter()
        .enumerate()
        {
            offsets[i + 1] = offsets[i] + bytes;
        }
        let readback = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla rigid frame readback"),
                size: offsets[14],
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }
        );
        let break_readback = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla rigid break notification"),
                size: 12,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }
        );
        let bind_groups = created.gpu_bind_groups_created().unwrap()
            + distance.as_ref().map_or(0, |d| d.bind_groups_created())
            + loops.as_ref().map_or(0, |l| l.bind_groups_created())
            + u64::from(contacts.is_some() && distance.is_some())
            + reactions.as_ref().map_or(0, |r| r.bind_groups_created)
            + u64::from(momentum.is_some()) * 2;
        let native_prediction = if let Some((dt, budget)) = self.native_prediction {
            let configured = created.substeps;
            created.substeps = 1;
            let prepare = |index: usize| {
                crate::native_prediction_gpu::PreparedNativePrediction::new(
                    &self.engine,
                    &created,
                    scene.model(),
                    scene.control(),
                    &states[index],
                    &states[1 - index],
                    dt,
                    budget / 2,
                )
            };
            let prepared = [prepare(0)?, prepare(1)?];
            created.substeps = configured;
            Some(prepared)
        } else {
            None
        };
        let mut next = Compiled {
            native_prediction,
            stamp: scene.model_stamp(),
            render_side: 0,
            solver_status: created.gpu_status_buffer().unwrap().clone(),
            states,
            readback,
            offsets,
            break_readback,
            capture,
            distance,
            loops,
            reactions,
            contacts,
            activation: None,
            damping: None,
            motion,
            momentum,
        };
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        trace("activation begin");
        let activation = activation::RuntimeActivation::new(&self.engine, scene, &created, &next);
        if let Some(error) = wait(device, scope.pop())? {
            return Err(Error::Gpu(format!(
                "GPU contact activation validation: {error}"
            )));
        }
        next.activation = activation?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        trace("damping begin");
        let damping = damping::RuntimeDamping::new(
            &self.engine,
            scene,
            &next.states,
            &created,
            next.contacts.is_some(),
        );
        if let Some(error) = wait(device, scope.pop())? {
            return Err(Error::Gpu(format!("GPU body damping validation: {error}")));
        }
        next.damping = damping?;
        trace("ready");
        let bind_groups = bind_groups + u64::from(next.damping.is_some()) * 2;
        Ok((next, created, bind_groups))
    }

    fn step(
        &mut self,
        scene: &mut RigidScene,
        solver: &mut FeatherstoneSolver,
        dt: Real,
        selection: activation::FrameSelection<'_>,
        capture: bool,
    ) -> Result<Segment> {
        #[cfg(test)]
        trace_soil_runtime("segment begin");
        let selected = selection.bodies;
        let active = selection.articulations;
        self.engine.check_device()?;
        if self
            .compiled
            .as_ref()
            .is_some_and(|c| solver.gpu_status_buffer() != Some(&c.solver_status))
        {
            return Err(Error::Solver("GPU runtime solver was replaced".into()));
        }
        if self.compiled.as_ref().is_none_or(|c| {
            c.stamp != scene.model_stamp()
                || c.capture != capture
                || c.reactions.is_some() != (capture || scene.breakable_joints().next().is_some())
                || c.reactions
                    .as_ref()
                    .is_some_and(|r| r.substeps != solver.substeps.max(1))
        }) {
            self.compile(scene, solver, capture, false)?;
        }
        let device = self.engine.device().unwrap();
        let queue = self.engine.queue().unwrap();
        scene.upload_controls_to_device(device, queue)?;
        let prediction_probe = self.has_prediction_probe();
        let compiled = self.compiled.as_mut().unwrap();
        if let Some(soil) = &mut self.soil {
            soil.prepare(&self.engine, scene, &compiled.states)?;
            if soil.projection_config.is_some()
                && (compiled.contacts.is_some()
                    || compiled.distance.is_some()
                    || compiled.reactions.is_some())
            {
                return Err(Error::Solver(
                    "mass coupling cannot share a separate constraint solver".into(),
                ));
            }
            soil.prepare_projection(
                &self.engine,
                scene,
                &compiled.states,
                solver,
                compiled.native_mass(),
                compiled.native_jacobians(),
            )?;
        }
        let deferred_drift = defer_native_drift(
            compiled.native_prediction.is_some(),
            self.soil
                .as_ref()
                .is_some_and(|s| s.has_prepared_projection()),
            prediction_probe,
        );
        solver.set_gpu_active_articulations(active)?;
        if let Some(distance) = &mut compiled.distance {
            distance.set_active(queue, active)?;
            distance.refresh_feedback(queue, scene)?;
        }
        if let Some(reactions) = &mut compiled.reactions {
            reactions.set_active(queue, scene, selected, active)?;
        }
        let input = &mut compiled.states[0];
        let source = scene.state();
        upload(
            queue,
            &mut input.body_q,
            source.body_q.host()?,
            &mut self.upload,
        )?;
        upload(
            queue,
            &mut input.body_qd,
            source.body_qd.host()?,
            &mut self.upload,
        )?;
        upload(
            queue,
            &mut input.body_f,
            source.body_f.host()?,
            &mut self.upload,
        )?;
        upload(
            queue,
            &mut input.joint_q,
            source.joint_q.host()?,
            &mut self.upload,
        )?;
        upload(
            queue,
            &mut input.joint_qd,
            source.joint_qd.host()?,
            &mut self.upload,
        )?;
        molla_core::gpu_write_buffer!(
            queue,
            solver.gpu_status_buffer().unwrap(),
            0,
            bytemuck::bytes_of(&0u32),
        );
        let substeps = solver.substeps.max(1);
        #[cfg(test)]
        trace_soil_runtime("controls and input uploaded");
        let sub_dt = dt / substeps as Real;
        if self
            .native_prediction
            .is_some_and(|(expected, _)| expected.to_bits() != sub_dt.to_bits())
        {
            return Err(unsupported(
                "native prediction substep changed; rebuild world",
            ));
        }
        if sub_dt as f32 == 0.0 {
            return Err(Error::Solver("GPU substep underflows f32".into()));
        }
        let configured = solver.substeps;
        if let Some(prediction) = &compiled.native_prediction {
            let mut encoder = device.create_command_encoder(&Default::default());
            for p in prediction {
                p.reset_internal(&mut encoder);
            }
            queue.submit([encoder.finish()]);
        }
        if let Some(damping) = &mut compiled.damping {
            damping.prepare_frame(&self.engine, scene, &compiled.states[0], sub_dt)?;
        }
        if let Some(contacts) = &mut compiled.contacts {
            contacts.prepare_frame(
                &self.engine,
                solver,
                &compiled.states,
                scene,
                substeps,
                sub_dt,
            )?;
        } else if let Some(motion) = &mut compiled.motion {
            motion.prepare_frame(&self.engine, solver, &compiled.states, scene.state())?;
        }
        if let Some(loops) = &mut compiled.loops {
            loops.prepare_frame(&self.engine, scene, &compiled.states, active)?;
        }
        if let Some(activation) = &compiled.activation {
            activation.prepare(
                &self.engine,
                scene,
                &selection,
                compiled.contacts.as_ref().unwrap(),
            )?;
        }
        if let Some(soil) = &mut self.soil {
            soil.begin(&self.engine, sub_dt as f32)?;
            soil.begin_profile(queue, sub_dt as f32, substeps);
        }
        solver.substeps = 1;
        if let Some(momentum) = &compiled.momentum {
            momentum.prepare(&self.engine, sub_dt);
        }
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let result = (|| {
            #[cfg(test)]
            trace_soil_runtime("frame prepared; refreshing kinematics");
            #[cfg(test)]
            check_soil_submission(&self.engine, "frame preparation");
            let refresh_start = self
                .soil
                .as_ref()
                .and_then(|soil| soil.profile_host_start());
            if scene.model().joint_count != 0 {
                let [input, scratch] = &mut compiled.states;
                solver.set_gpu_dispatch_timestamps(
                    self.soil
                        .as_mut()
                        .and_then(|s| s.machine_timestamp("refresh", None)),
                );
                solver.refresh_gpu(scene.model(), input, scene.control(), scratch)?;
                let mut encoder = device.create_command_encoder(&Default::default());
                for (from, to) in [
                    (
                        scratch.body_q.device_buffer_current()?,
                        input.body_q.device_buffer_current()?,
                    ),
                    (
                        scratch.body_qd.device_buffer_current()?,
                        input.body_qd.device_buffer_current()?,
                    ),
                    (
                        solver.gpu_position_compensation_buffer(scratch)?,
                        solver.gpu_position_compensation_buffer(input)?,
                    ),
                ] {
                    encoder.copy_buffer_to_buffer(from, 0, to, 0, from.size());
                }
                if solver.gpu_uses_paired_rates() {
                    let from = solver.gpu_body_velocity_compensation_buffer(scratch)?;
                    let to = solver.gpu_body_velocity_compensation_buffer(input)?;
                    encoder.copy_buffer_to_buffer(from, 0, to, 0, from.size());
                }
                queue.submit([encoder.finish()]);
            }
            if let Some(soil) = &mut self.soil {
                soil.profile_host_end(soil::ProfileHostStage::Refresh, refresh_start);
            }
            #[cfg(test)]
            check_soil_submission(&self.engine, "kinematic refresh");
            for i in 0..substeps {
                #[cfg(test)]
                trace_soil_runtime(&format!("substep {i}/{substeps} begin"));
                if let Some(contacts) = &compiled.contacts {
                    contacts.predict(&self.engine, i % 2, compiled.activation.as_ref())?;
                }
                if let Some(damping) = &compiled.damping {
                    damping.dispatch(&self.engine, i % 2);
                }
                if let Some(loops) = &compiled.loops {
                    loops.dispatch(&self.engine, i % 2, sub_dt);
                }
                if let Some(soil) = &mut self.soil {
                    let start = soil.profile_host_start();
                    soil.before_step(
                        &self.engine,
                        scene,
                        &mut compiled.states,
                        i % 2,
                        sub_dt as f32,
                    )?;
                    soil.profile_host_end(soil::ProfileHostStage::Soil, start);
                }
                #[cfg(test)]
                check_soil_submission(&self.engine, "soil coupling");
                let [a, b] = &mut compiled.states;
                let (input, output) = if i % 2 == 0 { (a, b) } else { (b, a) };
                #[cfg(test)]
                trace_soil_runtime("soil submitted; machine dispatch begin");
                let dynamics_start = self
                    .soil
                    .as_ref()
                    .and_then(|soil| soil.profile_host_start());
                let dynamics_timestamps = self
                    .soil
                    .as_mut()
                    .filter(|_| scene.model().joint_count != 0)
                    .and_then(|s| s.machine_timestamp("dynamics", Some(i)));
                if let Some(prediction) = &compiled.native_prediction {
                    let predict = if deferred_drift {
                        crate::native_prediction_gpu::PreparedNativePrediction::rates_profiled
                    } else {
                        crate::native_prediction_gpu::PreparedNativePrediction::step_profiled
                    };
                    predict(
                        &prediction[i % 2],
                        solver,
                        scene.model(),
                        scene.control(),
                        input,
                        output,
                        sub_dt,
                        dynamics_timestamps,
                    )?;
                } else {
                    solver.set_gpu_dispatch_timestamps(dynamics_timestamps);
                    solver.step(
                        scene.model(),
                        input,
                        scene.control(),
                        &Contacts::with_capacity(0),
                        sub_dt,
                        output,
                    )?;
                }
                #[cfg(test)]
                if let Some(probe) = &mut self.dynamics_probe {
                    assert!(compiled.distance.is_none() && active == [true]);
                    probe.check(&self.engine, solver, scene, input, output, sub_dt)?;
                }
                if let Some(soil) = &mut self.soil {
                    soil.profile_host_end(soil::ProfileHostStage::Dynamics, dynamics_start);
                }
                let projected = if let Some(soil) = &mut self.soil {
                    let start = soil.profile_host_start();
                    let projected = soil.after_prediction(
                        &self.engine,
                        &mut compiled.states,
                        i % 2,
                        sub_dt as f32,
                    )?;
                    soil.profile_host_end(soil::ProfileHostStage::Soil, start);
                    projected
                } else {
                    false
                };
                debug_assert!(!deferred_drift || projected);
                let [a, b] = &mut compiled.states;
                let (input, output) = if i % 2 == 0 { (a, b) } else { (b, a) };
                #[cfg(test)]
                trace_soil_runtime("machine dispatched");
                #[cfg(test)]
                check_soil_submission(&self.engine, "machine dynamics");
                let corrections_start = self
                    .soil
                    .as_ref()
                    .and_then(|soil| soil.profile_host_start());
                if projected {
                    #[cfg(test)]
                    if let Some(probe) = &mut self.dynamics_probe {
                        probe.before_drift(&self.engine, solver, scene, input, output, sub_dt)?;
                    }
                    solver.set_gpu_dispatch_timestamps(
                        self.soil
                            .as_mut()
                            .and_then(|s| s.machine_timestamp("soil-corrected-drift", Some(i))),
                    );
                    solver.drift_gpu(scene.model(), input, scene.control(), sub_dt, output)?;
                    if deferred_drift {
                        output.body_f.mark_device_dirty();
                    }
                    #[cfg(test)]
                    if let Some(probe) = &mut self.dynamics_probe {
                        probe.after_drift(&self.engine, solver, output, sub_dt)?;
                    }
                }
                if let Some(loops) = &compiled.loops {
                    loops.restore(&self.engine, [input, output])?;
                }
                if let Some(contacts) = &mut compiled.contacts {
                    if let Some(distance) = &compiled.distance {
                        if distance.needs_contact_prediction() {
                            contacts.correct_with_prediction(
                                &self.engine,
                                distance,
                                i % 2,
                                sub_dt,
                                substeps,
                                i,
                            )?;
                        } else {
                            distance.prepare_scalar_rows(
                                &self.engine,
                                i % 2,
                                sub_dt,
                                substeps,
                                i,
                            )?;
                            contacts.correct(&self.engine, i % 2, i)?;
                        }
                    } else {
                        contacts.correct(&self.engine, i % 2, i)?;
                    }
                    contacts.drift(solver, scene, input, output, sub_dt)?;
                }
                if compiled.contacts.is_none()
                    && let Some(distance) = &compiled.distance
                {
                    distance.dispatch(&self.engine, i % 2, sub_dt, substeps, i)?;
                    solver.drift_gpu(scene.model(), input, scene.control(), sub_dt, output)?;
                }
                let momentum_timestamps = if compiled.momentum.is_some() {
                    self.soil
                        .as_mut()
                        .and_then(|s| s.machine_timestamp("momentum", Some(i)))
                } else {
                    None
                };
                if let Some(reactions) = &compiled.reactions {
                    reactions.dispatch_before_halt(&self.engine, i % 2, sub_dt, i, |encoder| {
                        if let Some(momentum) = &compiled.momentum {
                            momentum.encode_profiled(encoder, i % 2, momentum_timestamps.as_ref());
                        }
                    });
                } else if let Some(momentum) = &compiled.momentum {
                    let mut encoder = device.create_command_encoder(&Default::default());
                    momentum.encode_profiled(&mut encoder, i % 2, momentum_timestamps.as_ref());
                    queue.submit([encoder.finish()]);
                }
                if let Some(soil) = &mut self.soil {
                    soil.profile_host_end(soil::ProfileHostStage::Corrections, corrections_start);
                    #[cfg(test)]
                    check_soil_submission(&self.engine, "momentum transport");
                    let start = soil.profile_host_start();
                    soil.after_step(&mut compiled.states)?;
                    soil.profile_host_end(soil::ProfileHostStage::Restore, start);
                }
                #[cfg(test)]
                if self.fail_after_substep == Some(i) {
                    self.fail_after_substep = None;
                    return Err(Error::Solver("injected post-substep failure".into()));
                }
            }
            Ok(())
        })();
        solver.substeps = configured;
        if let Some(soil) = &mut self.soil {
            soil.profile_submissions_finished();
        }
        #[cfg(test)]
        trace_soil_runtime("substeps submitted; waiting for dispatch validation");
        let completion = self
            .soil
            .as_ref()
            .map(|soil| {
                soil_completion::CompletionDeadline::new(
                    self.engine.device_health().unwrap(),
                    soil.completion_timeout,
                )
            })
            .transpose()?;
        let validation = if let Some(completion) = &completion {
            completion.wait(device, scope.pop())?
        } else {
            wait(device, scope.pop())?
        };
        #[cfg(test)]
        trace_soil_runtime("dispatch validation completed");
        result?;
        if let Some(error) = validation {
            return Err(Error::Gpu(format!(
                "GPU runtime dispatch validation: {error}"
            )));
        }
        let mut completed = substeps;
        let mut event_count = 0;
        let mut notification_bytes = 0;
        if scene.breakable_joints().next().is_some() {
            let reactions = compiled.reactions.as_ref().unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(
                &compiled.solver_status,
                0,
                &compiled.break_readback,
                0,
                4,
            );
            encoder.copy_buffer_to_buffer(
                &reactions.break_header,
                0,
                &compiled.break_readback,
                4,
                8,
            );
            queue.submit([encoder.finish()]);
            map_read_with_completion(device, &compiled.break_readback, completion.as_ref())?;
            let mapped = molla_core::gpu_readback!(compiled.break_readback.slice(..));
            let header = bytemuck::pod_read_unaligned::<[u32; 3]>(&mapped);
            drop(mapped);
            compiled.break_readback.unmap();
            notification_bytes = 12;
            if header[0] & !BREAK_STOP != 0 {
                return Err(compiled.failure(&self.engine, header[0] & !BREAK_STOP));
            }
            if header[0] & BREAK_STOP != 0 {
                completed = header[1] as usize;
                event_count = header[2] as u64;
                if completed == 0
                    || completed > substeps
                    || event_count == 0
                    || event_count * 112 > reactions.break_events.size()
                {
                    return Err(Error::Solver("invalid GPU break notification".into()));
                }
            } else if header[1] != 0 || header[2] != 0 {
                return Err(Error::Solver("uncommitted GPU break notification".into()));
            }
        }
        let output = &compiled.states[substeps % 2];
        let mut encoder = device.create_command_encoder(&Default::default());
        if let Some(soil) = &self.soil {
            soil.resolve_machine_profile(&mut encoder);
        }
        let buffers = [
            solver.gpu_status_buffer().unwrap(),
            output
                .body_q
                .device_buffer()
                .unwrap_or(solver.gpu_status_buffer().unwrap()),
            output
                .body_qd
                .device_buffer()
                .unwrap_or(solver.gpu_status_buffer().unwrap()),
            output
                .joint_q
                .device_buffer()
                .unwrap_or(solver.gpu_status_buffer().unwrap()),
            output
                .joint_qd
                .device_buffer()
                .unwrap_or(solver.gpu_status_buffer().unwrap()),
        ];
        for (i, buffer) in buffers.into_iter().enumerate() {
            let size = compiled.offsets[i + 1] - compiled.offsets[i];
            if size != 0 {
                encoder.copy_buffer_to_buffer(
                    buffer,
                    0,
                    &compiled.readback,
                    compiled.offsets[i],
                    size,
                );
            }
        }
        if let Some(distance) = &compiled.distance {
            let size = compiled.offsets[6] - compiled.offsets[5];
            if size != 0 {
                encoder.copy_buffer_to_buffer(
                    &distance.samples,
                    0,
                    &compiled.readback,
                    compiled.offsets[5],
                    size,
                );
            }
        }
        if let Some(reactions) = &compiled.reactions {
            for (section, buffer) in [(6, &reactions.means), (7, &reactions.samples)] {
                let size = compiled.offsets[section + 1] - compiled.offsets[section];
                if size != 0 {
                    encoder.copy_buffer_to_buffer(
                        buffer,
                        0,
                        &compiled.readback,
                        compiled.offsets[section],
                        size,
                    );
                }
            }
            if event_count != 0 {
                encoder.copy_buffer_to_buffer(
                    &reactions.break_events,
                    0,
                    &compiled.readback,
                    compiled.offsets[8],
                    event_count * 112,
                );
            }
        }
        if let Some(activation) = &compiled.activation {
            let size = compiled.offsets[10] - compiled.offsets[9];
            if size != 0 {
                encoder.copy_buffer_to_buffer(
                    activation.body_mask(),
                    0,
                    &compiled.readback,
                    compiled.offsets[9],
                    size,
                );
            }
        }
        if compiled.residues().is_some() {
            for (section, buffer) in [
                (10, solver.gpu_position_compensation_buffer(output)?),
                (11, solver.gpu_coordinate_compensation_buffer(output)?),
            ] {
                let size = compiled.offsets[section + 1] - compiled.offsets[section];
                if size != 0 {
                    encoder.copy_buffer_to_buffer(
                        buffer,
                        0,
                        &compiled.readback,
                        compiled.offsets[section],
                        size,
                    );
                }
            }
        }
        if solver.gpu_paired_velocity_delta_buffer().is_some() {
            for (section, buffer) in [
                (12, solver.gpu_paired_rates_buffer(output)?),
                (13, solver.gpu_body_velocity_compensation_buffer(output)?),
            ] {
                let size = compiled.offsets[section + 1] - compiled.offsets[section];
                if size != 0 {
                    encoder.copy_buffer_to_buffer(
                        buffer,
                        0,
                        &compiled.readback,
                        compiled.offsets[section],
                        size,
                    );
                }
            }
        }
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        compiled
            .readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        let mapping = if let Some(completion) = &completion {
            completion.poll(device)?;
            completion.receive(&rx)?
        } else {
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| Error::Gpu(e.to_string()))?;
            rx.recv().map_err(|e| Error::Gpu(e.to_string()))?
        };
        mapping.map_err(|e| Error::Gpu(e.to_string()))?;
        let mapped = molla_core::gpu_readback!(compiled.readback.slice(..));
        let status = bytemuck::pod_read_unaligned::<u32>(&mapped[..4]);
        let mut next = scene.state().clone_host()?;
        let section =
            |i: usize| &mapped[compiled.offsets[i] as usize..compiled.offsets[i + 1] as usize];
        let mask: Vec<u32> = section(9)
            .chunks_exact(4)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        let invalid_mask = mask.iter().any(|&v| v > 1)
            || (compiled.activation.is_some() && mask.len() != selected.len());
        let effective: Vec<bool> = if compiled.activation.is_some() {
            mask.iter().map(|&v| v == 1).collect()
        } else {
            selected.to_vec()
        };
        let awakened = scene
            .bodies()
            .filter_map(|(body, _)| {
                let index = scene.body_index(body).unwrap().index();
                (!selected[index] && effective.get(index) == Some(&true)).then_some(body)
            })
            .collect();
        let selected = effective.as_slice();
        decode::<_, GpuTransform>(section(1), next.body_q.host_mut()?, Into::into);
        decode::<_, GpuSpatialVector>(section(2), next.body_qd.host_mut()?, Into::into);
        decode::<_, f32>(section(3), next.joint_q.host_mut()?, Real::from);
        decode::<_, f32>(section(4), next.joint_qd.host_mut()?, Real::from);
        let paired_decode = if solver.gpu_paired_velocity_delta_buffer().is_some() {
            precision::decode_rates(&mut next, section(12), section(13))
        } else {
            Ok(())
        };
        for (pose, bytes) in next
            .body_q
            .host_mut()?
            .iter_mut()
            .zip(section(10).chunks_exact(16))
        {
            let low = bytemuck::pod_read_unaligned::<[f32; 4]>(bytes);
            pose.position += molla_math::Vec3::new(low[0] as Real, low[1] as Real, low[2] as Real);
        }
        for (value, bytes) in next
            .joint_q
            .host_mut()?
            .iter_mut()
            .zip(section(11).chunks_exact(4))
        {
            *value += bytemuck::pod_read_unaligned::<f32>(bytes) as Real;
        }
        let mut readings = ConstraintReadings::default();
        let mut last_active = BTreeMap::<JointHandle, u32>::new();
        if let Some(distance) = &compiled.distance {
            for (row_index, ((&owner, &cone), bytes)) in distance
                .owners
                .iter()
                .zip(&distance.angular_rows)
                .zip(section(5).chunks_exact(32))
                .enumerate()
            {
                if let Some(owner) = owner {
                    let joint = scene.joint(owner).unwrap();
                    if !selected[scene.body_index(joint.child).unwrap().index()]
                        || joint
                            .parent
                            .is_some_and(|p| !selected[scene.body_index(p).unwrap().index()])
                    {
                        continue;
                    }
                    let sample = bytemuck::pod_read_unaligned::<[f32; 4]>(&bytes[..16]);
                    let tag = bytemuck::pod_read_unaligned::<[u32; 4]>(&bytes[16..]);
                    let vector = molla_math::Vec3::new(
                        sample[0] as Real,
                        sample[1] as Real,
                        sample[2] as Real,
                    );
                    if distance.is_rigid_row(row_index) {
                        let reading = readings.rigid.entry(owner).or_default();
                        if cone {
                            reading.angular += vector;
                        } else {
                            reading.linear += vector;
                        }
                        continue;
                    }
                    if cone {
                        let reading = readings.cones.entry(owner).or_default();
                        reading.torque += vector;
                        reading.active |= tag[0] != 0;
                        continue;
                    }
                    let reading = readings.distance.entry(owner).or_default();
                    reading.force += vector;
                    if tag[0] == 0 || last_active.get(&owner).is_some_and(|&last| last > tag[1]) {
                        continue;
                    }
                    last_active.insert(owner, tag[1]);
                    match tag[0] {
                        1 => reading.active = DistanceBound::Minimum,
                        2 => reading.active = DistanceBound::Maximum,
                        3 => reading.active = DistanceBound::Equality,
                        _ => {}
                    }
                }
            }
        }
        let reactions = if status & !BREAK_STOP == 0 && capture {
            compiled
                .reactions
                .as_ref()
                .map(|r| r.decode(section(6), section(7), dt))
                .transpose()
        } else {
            Ok(None)
        };
        let breaks = if event_count != 0 {
            compiled.reactions.as_ref().unwrap().decode_breaks(
                &mapped[compiled.offsets[8] as usize
                    ..(compiled.offsets[8] + event_count * 112) as usize],
                sub_dt,
                completed,
            )
        } else {
            Ok(Vec::new())
        };
        drop(mapped);
        compiled.readback.unmap();
        if status & !BREAK_STOP != 0 {
            if let (Some(soil), Some(completion)) = (&self.soil, &completion)
                && let Some(detail) = soil.fault_detail(&self.engine, completion)?
            {
                return Err(Error::Solver(format!(
                    "GPU rigid runtime solver fault {}; {detail}",
                    status & !BREAK_STOP
                )));
            }
            return Err(compiled.failure(&self.engine, status & !BREAK_STOP));
        }
        paired_decode?;
        if invalid_mask {
            return Err(Error::Gpu("invalid GPU contact activation readout".into()));
        }
        let reactions = reactions?;
        let breaks = breaks?;
        self.engine.check_device()?;
        for (id, joint, limits) in crate::distance_constraints::point_constraints(scene) {
            if !selected[scene.body_index(joint.child).unwrap().index()]
                || joint
                    .parent
                    .is_some_and(|p| !selected[scene.body_index(p).unwrap().index()])
            {
                continue;
            }
            if !scene.joint_settings(id).unwrap().enabled
                || !scene.body(joint.child).unwrap().enabled
                || joint
                    .parent
                    .is_some_and(|p| !scene.body(p).unwrap().enabled)
            {
                continue;
            }
            let [a, b] = crate::distance_constraints::anchors(scene, &next, joint)?;
            let reading = readings.distance.entry(id).or_default();
            reading.separation = (b - a).length();
            reading.violation = limits
                .min
                .map_or(0.0, |v| (v - reading.separation).max(0.0))
                .max(
                    limits
                        .max
                        .map_or(0.0, |v| (reading.separation - v).max(0.0)),
                );
            reading.sample_dt = sub_dt * completed as Real;
            if !reading.force.is_finite() {
                return Err(Error::Solver("nonfinite GPU distance reaction".into()));
            }
        }
        for (id, joint, limits) in scene.cone_joints() {
            if !crate::distance_constraints::enabled(scene, id, joint)
                || !selected[scene.body_index(joint.child).unwrap().index()]
                || joint
                    .parent
                    .is_some_and(|p| !selected[scene.body_index(p).unwrap().index()])
            {
                continue;
            }
            let (_, geometry) = crate::cone_constraints::geometry(scene, &next, joint, limits)?;
            let reading = readings.cones.entry(id).or_default();
            reading.swing = geometry.swing;
            reading.violation = geometry.violation;
            reading.sample_dt = sub_dt * completed as Real;
            if !reading.torque.is_finite() {
                return Err(Error::Solver("nonfinite GPU cone reaction".into()));
            }
        }
        let contacts = compiled
            .contacts
            .as_ref()
            .map(|contacts| contacts.readings(&self.engine, completed, sub_dt, completion.as_ref()))
            .transpose()?
            .unwrap_or_default();
        for (index, &awake) in selected.iter().enumerate() {
            if !awake {
                next.body_q.host_mut()?[index] = scene.state().body_q.host()?[index];
                next.body_qd.host_mut()?[index] = scene.state().body_qd.host()?[index];
            }
        }
        let model = scene.model();
        let starts = model.articulation_start.host()?;
        let children = model.joint_child.host()?;
        let coords = model.joint_coord_offset.host()?;
        let dofs = model.joint_dof_offset.host()?;
        for (art, range) in starts.windows(2).enumerate() {
            if !active[art]
                && !children[range[0] as usize..range[1] as usize]
                    .iter()
                    .any(|&b| !selection.bodies[b as usize] && selected[b as usize])
            {
                let q = coords[range[0] as usize] as usize..coords[range[1] as usize] as usize;
                let qd = dofs[range[0] as usize] as usize..dofs[range[1] as usize] as usize;
                next.joint_q.host_mut()?[q.clone()]
                    .copy_from_slice(&scene.state().joint_q.host()?[q]);
                next.joint_qd.host_mut()?[qd.clone()]
                    .copy_from_slice(&scene.state().joint_qd.host()?[qd]);
            }
        }
        scene.commit_step(next)?;
        compiled.render_side = substeps % 2;
        if let (_, Some(precision)) = compiled.precision_parts() {
            precision.committed(scene.state(), substeps % 2)?;
        }
        self.diagnostics.current_model_bind_groups_created =
            solver.gpu_bind_groups_created().unwrap()
                + compiled
                    .distance
                    .as_ref()
                    .map_or(0, |d| d.bind_groups_created())
                + compiled
                    .loops
                    .as_ref()
                    .map_or(0, |l| l.bind_groups_created())
                + compiled
                    .reactions
                    .as_ref()
                    .map_or(0, |r| r.bind_groups_created)
                + u64::from(compiled.momentum.is_some()) * 2
                + u64::from(compiled.damping.is_some()) * 2;
        self.diagnostics.last_substeps = completed;
        self.diagnostics.last_state_readback_bytes =
            compiled.offsets[8] + compiled.offsets[10] - compiled.offsets[9] + compiled.offsets[14]
                - compiled.offsets[10]
                + event_count * 112
                + notification_bytes
                + compiled
                    .contacts
                    .as_ref()
                    .map_or(0, |c| c.readback_bytes(completed));
        self.diagnostics.last_state_upload_bytes =
            compiled.offsets[5] - 4 + scene.model().body_count as u64 * 32;
        Ok(Segment {
            readings,
            reactions,
            breaks,
            completed,
            contacts,
            awakened,
        })
    }
}

fn upload<T: BufferElement>(
    queue: &wgpu::Queue,
    buffer: &mut Buffer<T>,
    data: &[T],
    scratch: &mut Vec<u8>,
) -> Result<()> {
    scratch.clear();
    for value in data {
        scratch.extend_from_slice(bytemuck::bytes_of(&value.to_gpu()));
    }
    if !scratch.is_empty() {
        molla_core::gpu_write_buffer!(
            queue,
            buffer
                .device_buffer()
                .ok_or_else(|| Error::Gpu("missing runtime buffer".into()))?,
            0,
            scratch,
        );
    }
    buffer.mark_device_dirty();
    Ok(())
}

fn decode<T, G: bytemuck::Pod>(bytes: &[u8], output: &mut [T], convert: impl Fn(G) -> T) {
    for (value, bytes) in output
        .iter_mut()
        .zip(bytes.chunks_exact(std::mem::size_of::<G>()))
    {
        *value = convert(bytemuck::pod_read_unaligned(bytes));
    }
}

fn wait<T>(device: &wgpu::Device, future: impl std::future::Future<Output = T>) -> Result<T> {
    wait_with_progress(future, || {
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|e| Error::Gpu(e.to_string()))
    })
}

fn wait_with_progress<T>(
    future: impl std::future::Future<Output = T>,
    mut progress: impl FnMut() -> Result<()>,
) -> Result<T> {
    struct Wake(std::thread::Thread);
    impl std::task::Wake for Wake {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = std::task::Waker::from(Arc::new(Wake(std::thread::current())));
    let mut context = std::task::Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return Ok(value);
        }
        progress()?;
        if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return Ok(value);
        }
        std::thread::park_timeout(std::time::Duration::from_millis(1));
    }
}

#[cfg(test)]
mod wait_tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn backend_poll_may_consume_the_completion_wake() {
        let ready = AtomicBool::new(false);
        let wake = std::sync::Mutex::new(None::<std::task::Waker>);
        let future = std::future::poll_fn(|context| {
            if ready.load(Ordering::Acquire) {
                std::task::Poll::Ready(42)
            } else {
                *wake.lock().unwrap() = Some(context.waker().clone());
                std::task::Poll::Pending
            }
        });
        let result = wait_with_progress(future, || {
            ready.store(true, Ordering::Release);
            wake.lock().unwrap().take().unwrap().wake();
            std::thread::park();
            Ok(())
        })
        .unwrap();
        assert_eq!(result, 42);
    }
}
