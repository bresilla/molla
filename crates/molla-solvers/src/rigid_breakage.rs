use super::*;
use molla_sim::runtime::{JointBreakLimits, JointBreakTrigger, JointHandle};

#[derive(Clone, Debug, PartialEq)]
pub struct JointBreakEvent {
    /// Zero-based successful outer-step index; sample times are relative to this frame.
    pub frame: u64,
    pub joint: JointHandle,
    pub parent: Option<BodyHandle>,
    pub child: BodyHandle,
    pub trigger: JointBreakTrigger,
    pub limits: JointBreakLimits,
    pub sample: JointReactionSample,
}

type EffortKey = (BodyHandle, Option<JointHandle>, usize);

fn effort_keys(scene: &RigidScene) -> Result<Vec<EffortKey>> {
    let authored: BTreeMap<_, _> = scene
        .joints()
        .filter_map(|(id, _)| scene.joint_index(id).map(|index| (index.index(), id)))
        .collect();
    let offsets = scene.model().joint_dof_offset.host()?;
    let mut keys = Vec::with_capacity(scene.model().joint_dof_total);
    for (joint, &child) in scene.model().joint_child.host()?.iter().enumerate() {
        let body = scene
            .body_handle(molla_core::BodyId::from_index(child as usize))
            .unwrap();
        for dof in 0..(offsets[joint + 1] - offsets[joint]) as usize {
            keys.push((body, authored.get(&joint).copied(), dof));
        }
    }
    Ok(keys)
}

impl RigidWorld {
    /// Committed break events, retained until explicitly drained.
    pub fn joint_break_events(&self) -> &[JointBreakEvent] {
        &self.break_events
    }

    pub fn drain_joint_break_events(&mut self) -> Vec<JointBreakEvent> {
        std::mem::take(&mut self.break_events)
    }

    pub(super) fn step_breakable_cpu(
        &mut self,
        dt: Real,
        mut excluded: impl FnMut(Option<BodyHandle>, Option<BodyHandle>) -> bool,
        mut forces: impl FnMut(&RigidScene, &molla_sim::State, Real, &mut [SpatialVector]) -> Result<()>,
        custom_forces: bool,
    ) -> Result<()> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(Error::Solver("step dt must be positive and finite".into()));
        }
        if !self.devices.is_empty() || !self.wheels.is_empty() || !self.tracks.is_empty() {
            return Err(Error::Solver("automatic joint breaking with actuator devices, wheels or tracks is not yet supported".into()));
        }
        let substeps = self.solver.substeps.max(1);
        let sub_dt = dt / substeps as Real;
        if sub_dt == 0.0 {
            return Err(Error::Solver("breakable substep underflows".into()));
        }
        let configured = self.solver.substeps;
        let readout = self.reaction_readout;
        let changed = self.settling.changed_inputs(&self.scene, &self.wheels);
        self.quarantined = self.scene.quarantine_non_finite()?;
        let quarantined = self.quarantined.clone();
        let mut checkpoint = Some(self.scene.checkpoint_step()?);
        let mut original = None;
        let sleeping = self.sleeping.clone();
        let settling = self.settling.clone();
        let impacts = self.impacts.clone();
        let old_loop_reactions = self.devices.replace_reactions(BTreeMap::new());
        self.solver.substeps = 1;
        self.reaction_readout = true;
        let mut frame = ReactionFrame::default();
        let mut readings = ConstraintReadings::default();
        let mut contacts = Vec::new();
        let mut loop_reactions = BTreeMap::<JointHandle, crate::loop_forces::LoopReaction>::new();
        let mut efforts = BTreeMap::<EffortKey, crate::featherstone::JointEffort>::new();
        let mut events = Vec::new();
        let weight = 1.0 / substeps as Real;
        let result = (|| {
            for substep in 0..substeps {
                if !custom_forces && self.sleeping.any_valid(&self.scene, &self.wheels) {
                    self.step_sleeping(sub_dt, &mut excluded)?;
                } else {
                    self.step_with_forces_inner(sub_dt, &mut excluded, &mut forces, custom_forces)?;
                }
                contacts.extend_from_slice(self.solver.contact_forces());
                for (joint, reaction) in self.devices.replace_reactions(BTreeMap::new()) {
                    let mean = loop_reactions.entry(joint).or_default();
                    mean.force += reaction.force * weight;
                    mean.torque += reaction.torque * weight;
                }
                for (key, effort) in effort_keys(&self.scene)?
                    .into_iter()
                    .zip(self.solver.joint_efforts())
                {
                    let mean = efforts.entry(key).or_default();
                    mean.drive += effort.drive * weight;
                    mean.brake += effort.brake * weight;
                }
                readings.accumulate(
                    ConstraintReadings {
                        distance: std::mem::take(&mut self.distance_readings),
                        cones: std::mem::take(&mut self.cone_readings),
                        ..Default::default()
                    },
                    weight,
                    dt,
                );
                let mut broken = Vec::new();
                for mut sample in std::mem::take(&mut self.reaction_frame.samples) {
                    sample.substep = substep;
                    sample.time_offset = (substep + 1) as Real * sub_dt;
                    let limits = self.scene.joint_break_limits(sample.joint)?;
                    let norm = |v: Vec3| v.x.hypot(v.y).hypot(v.z);
                    if let Some(trigger) =
                        limits.trigger(norm(sample.child.linear), norm(sample.child.angular))?
                    {
                        let joint = self.scene.joint(sample.joint).unwrap();
                        events.push(JointBreakEvent {
                            frame: self.completed_steps,
                            joint: sample.joint,
                            parent: joint.parent,
                            child: joint.child,
                            trigger,
                            limits,
                            sample,
                        });
                        broken.push(sample.joint);
                    }
                    frame.add(sample, dt)?;
                }
                if !broken.is_empty() {
                    if original.is_none() {
                        let working = self.scene.fork()?;
                        self.scene.restore_step(checkpoint.take().unwrap())?;
                        original = Some(std::mem::replace(&mut self.scene, working));
                    }
                    self.scene.break_joints(&broken)?;
                    self.impacts.preserve_after_detachment(&self.scene);
                }
            }
            let means = effort_keys(&self.scene)?
                .into_iter()
                .map(|key| efforts.get(&key).copied().unwrap_or_default())
                .collect();
            self.solver.publish_frame_readouts(contacts, means);
            Ok(())
        })();
        self.solver.substeps = configured;
        self.reaction_readout = readout;
        self.quarantined = quarantined;
        if let Err(error) = result {
            if let Some(scene) = original {
                self.scene = scene;
            } else if let Some(checkpoint) = checkpoint {
                self.scene.restore_step(checkpoint)?;
            }
            self.sleeping = sleeping;
            self.settling = settling;
            self.impacts = impacts;
            self.devices.replace_reactions(old_loop_reactions);
            self.solver.publish_frame_readouts(Vec::new(), Vec::new());
            self.reaction_frame = ReactionFrame::default();
            self.reaction_stamp = None;
            self.distance_stamp = None;
            self.contact_revision = None;
            return Err(error);
        }
        self.distance_readings = readings.distance;
        self.devices.replace_reactions(loop_reactions);
        self.cone_readings = readings.cones;
        self.distance_stamp = Some(self.scene.dynamics_stamp());
        self.contact_revision = Some(self.scene.stamp());
        if readout {
            self.reaction_frame = frame;
            self.reaction_stamp = Some(self.scene.dynamics_stamp());
        } else {
            self.reaction_frame = ReactionFrame::default();
            self.reaction_stamp = None;
        }
        self.break_events.extend(events);
        if custom_forces {
            self.settling.clear_windows();
        } else {
            self.finish_settling(dt, &changed);
        }
        Ok(())
    }
}
