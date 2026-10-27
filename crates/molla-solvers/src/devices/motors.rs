//! Rotational and linear motors on revolute and prismatic tree joints, and
//! brakes on any of their axes.

use std::collections::BTreeMap;

use molla_core::{Error, Result};
use molla_math::Real;
use molla_sim::JointTargetMode;
use molla_sim::JointType;
use molla_sim::joint_control::{JointAxis, JointMotor, MotorModel};
use molla_sim::runtime::{JointHandle, RigidScene};

use super::{MotorCommand, MotorController, MotorLimits};
use crate::featherstone::JointEffort;

/// A motor or brake is addressed by its joint.
pub type MotorKey = JointHandle;

/// Velocity gain that makes the implicit drive reach the commanded speed
/// within a substep whenever the available force allows.
const VELOCITY_GAIN: Real = 1e10;

/// A rotational (revolute) or linear (prismatic) motor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotorSpec {
    pub limits: MotorLimits,
    /// Force or torque available to the motor (N or N·m).
    pub max_force: Real,
}

impl Default for MotorSpec {
    fn default() -> Self {
        Self {
            limits: MotorLimits::default(),
            max_force: 10.0,
        }
    }
}

/// A motor's joint state and effort over the last step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotorReading {
    pub position: Real,
    pub velocity: Real,
    pub command: MotorCommand,
    /// Speed the controller asked of the drive (zero under force control).
    pub commanded_velocity: Real,
    /// Force or torque the motor applied, averaged over the step.
    pub force: Real,
    /// Brake damping on the joint and the effort it applied.
    pub brake_damping: Real,
    pub brake_force: Real,
}

struct Motor {
    controller: MotorController,
    max_force: Real,
    written: Option<JointMotor>,
}

/// Motors and brakes of a rigid world.
#[derive(Default)]
pub struct Motors {
    motors: BTreeMap<MotorKey, Motor>,
    brakes: BTreeMap<MotorKey, Real>,
    readings: BTreeMap<MotorKey, MotorReading>,
}

fn axis_of(scene: &RigidScene, joint: JointHandle) -> Result<JointAxis> {
    let desc = scene
        .joint(joint)
        .ok_or_else(|| Error::Build("motor joint is missing".into()))?;
    if scene.joint_softness(joint)?.is_some() {
        return Err(Error::Build("motors and brakes need a rigid joint".into()));
    }
    match desc.kind {
        JointType::Revolute => Ok(JointAxis::AngX),
        JointType::Prismatic => Ok(JointAxis::LinX),
        _ => Err(Error::Build(
            "motors and brakes need a revolute or prismatic joint".into(),
        )),
    }
}

/// Coordinate, rate and DOF index of a revolute or prismatic tree joint.
fn coordinate(scene: &RigidScene, joint: JointHandle) -> Result<Option<(Real, Real, usize)>> {
    let Some(index) = scene.joint_index(joint) else {
        return Ok(None);
    };
    let model = scene.model();
    let coord = model.joint_coord_offset.host()?[index.index()] as usize;
    let dof = model.joint_dof_offset.host()?[index.index()] as usize;
    let state = scene.state();
    Ok(Some((
        state.joint_q.host()?[coord],
        state.joint_qd.host()?[dof],
        dof,
    )))
}

impl Motors {
    pub(super) fn is_empty(&self) -> bool {
        self.motors.is_empty() && self.brakes.is_empty()
    }

    /// Puts a motor on `joint`, holding its current position.
    pub fn insert(
        &mut self,
        scene: &RigidScene,
        joint: JointHandle,
        spec: MotorSpec,
    ) -> Result<()> {
        axis_of(scene, joint)?;
        if spec.max_force.is_nan()
            || spec.max_force < 0.0
            || spec.limits.max_velocity.is_nan()
            || spec.limits.max_velocity < 0.0
        {
            return Err(Error::Build(
                "motor force and velocity limits must be nonnegative".into(),
            ));
        }
        let mut controller = MotorController::new(spec.limits);
        let hold = coordinate(scene, joint)?.map_or(0.0, |(q, _, _)| q);
        controller.set_command(MotorCommand::Position(hold));
        self.motors.insert(
            joint,
            Motor {
                controller,
                max_force: spec.max_force,
                written: None,
            },
        );
        Ok(())
    }

    /// Takes the motor off `joint`, releasing its drive.
    pub fn remove(&mut self, scene: &mut RigidScene, joint: JointHandle) -> Result<()> {
        if self.motors.remove(&joint).is_some() && scene.joint(joint).is_some() {
            let axis = axis_of(scene, joint)?;
            scene.set_joint_motor(joint, axis, JointMotor::default())?;
        }
        self.readings.remove(&joint);
        Ok(())
    }

    pub fn contains(&self, joint: JointHandle) -> bool {
        self.motors.contains_key(&joint)
    }

    fn motor(&mut self, joint: JointHandle) -> Result<&mut Motor> {
        self.motors
            .get_mut(&joint)
            .ok_or_else(|| Error::Build("no motor on that joint".into()))
    }

    pub fn command(&mut self, joint: JointHandle, command: MotorCommand) -> Result<()> {
        let (MotorCommand::Position(v) | MotorCommand::Velocity(v) | MotorCommand::Force(v)) =
            command;
        if !v.is_finite() {
            return Err(Error::Build("motor command must be finite".into()));
        }
        self.motor(joint)?.controller.set_command(command);
        Ok(())
    }

    /// Speed position control runs at, up to the motor's maximum.
    pub fn set_speed(&mut self, joint: JointHandle, speed: Real) -> Result<()> {
        let motor = self.motor(joint)?;
        motor.controller.speed = speed.abs().min(motor.controller.limits.max_velocity);
        Ok(())
    }

    pub fn set_acceleration(
        &mut self,
        joint: JointHandle,
        acceleration: Option<Real>,
    ) -> Result<()> {
        self.motor(joint)?.controller.limits.acceleration = acceleration.map(Real::abs);
        Ok(())
    }

    pub fn set_pid(&mut self, joint: JointHandle, pid: [Real; 3]) -> Result<()> {
        self.motor(joint)?.controller.set_pid(pid);
        Ok(())
    }

    /// Force or torque available to the motor.
    pub fn set_max_force(&mut self, joint: JointHandle, max_force: Real) -> Result<()> {
        if max_force.is_nan() || max_force < 0.0 {
            return Err(Error::Build("motor force limit must be nonnegative".into()));
        }
        self.motor(joint)?.max_force = max_force;
        Ok(())
    }

    /// Puts a brake on `joint`, or changes its damping (N·m·s/rad or N·s/m);
    /// zero releases it and keeps reporting.
    pub fn set_brake(
        &mut self,
        scene: &mut RigidScene,
        joint: JointHandle,
        damping: Real,
    ) -> Result<()> {
        let axis = axis_of(scene, joint)?;
        scene.set_joint_brake(joint, axis, damping)?;
        self.brakes.insert(joint, damping);
        Ok(())
    }

    /// Takes the brake off `joint`.
    pub fn remove_brake(&mut self, scene: &mut RigidScene, joint: JointHandle) -> Result<()> {
        if self.brakes.remove(&joint).is_some() && scene.joint(joint).is_some() {
            let axis = axis_of(scene, joint)?;
            scene.set_joint_brake(joint, axis, 0.0)?;
        }
        self.readings.remove(&joint);
        Ok(())
    }

    /// The motor or brake of `joint` over the last step.
    pub fn reading(&self, joint: JointHandle) -> Option<MotorReading> {
        self.readings.get(&joint).copied()
    }

    pub(crate) fn before_step(&mut self, scene: &mut RigidScene, dt: Real) -> Result<()> {
        self.motors.retain(|joint, _| scene.joint(*joint).is_some());
        for (&joint, motor) in &mut self.motors {
            let Some((q, _, dof)) = coordinate(scene, joint)? else {
                continue;
            };
            let axis = axis_of(scene, joint)?;
            let model = scene.model();
            let stops = [model.joint_limit_lower.host()?[dof], model.joint_limit_upper.host()?[dof]];
            let drive = match motor.controller.velocity_within(q, dt, stops) {
                Some(velocity) => JointMotor {
                    mode: JointTargetMode::Velocity,
                    target_velocity: velocity,
                    damping: VELOCITY_GAIN,
                    max_force: motor.max_force,
                    model: MotorModel::Force,
                    ..JointMotor::default()
                },
                None => {
                    let MotorCommand::Force(force) = motor.controller.command else {
                        unreachable!("only force control has no speed");
                    };
                    JointMotor {
                        mode: JointTargetMode::Effort,
                        target_position: force.clamp(-motor.max_force, motor.max_force),
                        max_force: motor.max_force,
                        model: MotorModel::Force,
                        ..JointMotor::default()
                    }
                }
            };
            let wake = motor.written.is_none_or(|last| {
                last.mode != drive.mode
                    || last.max_force != drive.max_force
                    || (last.target_velocity - drive.target_velocity).abs() > 1e-6
                    || (last.target_position - drive.target_position).abs() > 1e-6
            });
            scene.set_joint_motor_with_wake(joint, axis, drive, wake)?;
            motor.written = Some(drive);
        }
        Ok(())
    }

    pub(crate) fn after_step(&mut self, scene: &RigidScene, efforts: &[JointEffort]) -> Result<()> {
        self.brakes.retain(|joint, _| scene.joint(*joint).is_some());
        self.readings.clear();
        let joints: Vec<JointHandle> = self
            .motors
            .keys()
            .chain(self.brakes.keys())
            .copied()
            .collect();
        for joint in joints {
            let Some((position, velocity, dof)) = coordinate(scene, joint)? else {
                continue;
            };
            let effort = efforts.get(dof).copied().unwrap_or_default();
            let (command, commanded_velocity, force) = match self.motors.get(&joint) {
                Some(motor) => (
                    motor.controller.command,
                    motor.written.map_or(0.0, |w| {
                        if w.mode == JointTargetMode::Velocity {
                            w.target_velocity
                        } else {
                            0.0
                        }
                    }),
                    effort.drive,
                ),
                None => (MotorCommand::Force(0.0), 0.0, 0.0),
            };
            self.readings.insert(
                joint,
                MotorReading {
                    position,
                    velocity,
                    command,
                    commanded_velocity,
                    force,
                    brake_damping: self.brakes.get(&joint).copied().unwrap_or(0.0),
                    brake_force: effort.brake,
                },
            );
        }
        Ok(())
    }
}
