//! 6-DOF spacecraft guidance, navigation and control (GNC) controllers.
//!
//! The zero-G counterpart to [`super::Copters`]: drives a spacecraft's
//! [`super::Thrusters`] and optional [`super::ReactionWheels`] to track a
//! body wrench, arrest linear/angular rates (`RateHold`), hold a 6-DOF pose
//! (`PoseHold`), follow a rendezvous trajectory (`Trajectory`), or desaturate
//! reaction wheels (`DesaturateWheels`) without disturbing attitude.
//!
//! When both reaction wheels and RCS thrusters are present, attitude torques
//! are allocated preferentially to the reaction wheels (zero propellant) and
//! any saturated overflow spills over to the RCS thrusters. Thruster allocation
//! solves the non-negative box-constrained least-squares QP `u ∈ [0, 1]^N`
//! via projected Gauss–Seidel, automatically routing around starved thrusters.

use std::collections::BTreeMap;

use molla_core::{Error, Result};
use molla_math::{Mat3, Quat, Real, Transform, Vec3};
use molla_sim::runtime::{BodyHandle, BodyKind, RigidScene};

use super::{ReactionWheelId, ReactionWheels, ThrusterId, Thrusters};

const QP_ITERATIONS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpacecraftId(pub u64);

/// High-level 6-DOF command for a spacecraft GNC controller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpacecraftCommand {
    /// All thrusters off and reaction wheel torques zeroed.
    Off,
    /// Direct wrench in the spacecraft control frame (force in N, torque in N·m).
    Wrench { force: Vec3, torque: Vec3 },
    /// Hold world-space linear velocity (m/s) and control-frame angular rate
    /// (rad/s) — e.g. `Vec3::ZERO` for kill-rot / velocity damping.
    RateHold {
        linear_velocity: Vec3,
        angular_velocity: Vec3,
    },
    /// Full 6-DOF station-keeping at world `position` and `orientation`.
    PoseHold { position: Vec3, orientation: Quat },
    /// Full 6-DOF rendezvous / docking trajectory tracking in world coordinates.
    Trajectory {
        position: Vec3,
        velocity: Vec3,
        acceleration: Vec3,
        orientation: Quat,
        angular_velocity: Vec3,
        angular_acceleration: Vec3,
    },
    /// Momentum dump: bleed reaction wheels toward zero speed while holding
    /// `orientation` (and optionally `position`) using RCS thrusters.
    DesaturateWheels {
        position: Option<Vec3>,
        orientation: Quat,
    },
}

/// Gains and velocity limits of a 6-DOF spacecraft controller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpacecraftLimits {
    /// Maximum linear translation speed commanded by `PoseHold` (m/s).
    pub max_linear_speed: Real,
    /// Maximum angular slew rate commanded by `PoseHold` (rad/s).
    pub max_angular_speed: Real,
    /// Position loop natural frequency `ω_p` (rad/s), critically damped.
    pub position_frequency: Real,
    /// Attitude loop natural frequency `ω_a` (rad/s), critically damped.
    pub attitude_frequency: Real,
    /// Linear velocity damping gain for `RateHold` (1/s).
    pub velocity_gain: Real,
    /// Angular rate damping gain for `RateHold` (1/s).
    pub angular_rate_gain: Real,
    /// Momentum bleed rate (1/s) during `DesaturateWheels`.
    pub desaturation_gain: Real,
    /// Relative weight on force tracking in the RCS QP allocator.
    pub force_weight: Real,
    /// Relative weight on torque tracking in the RCS QP allocator.
    pub torque_weight: Real,
    /// Small L2 penalty for minimum-fuel tie-breaking in the RCS QP allocator.
    pub regularization: Real,
}

impl Default for SpacecraftLimits {
    fn default() -> Self {
        Self {
            max_linear_speed: 5.0,
            max_angular_speed: 1.0,
            position_frequency: 0.8,
            attitude_frequency: 1.5,
            velocity_gain: 2.0,
            angular_rate_gain: 3.0,
            desaturation_gain: 0.5,
            force_weight: 1.0,
            torque_weight: 1.0,
            regularization: 1e-6,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpacecraftSpec {
    pub body: BodyHandle,
    /// Control frame in the body frame.
    pub frame: Transform,
    /// RCS thrusters attached to `body` (may be empty for wheel-only attitude).
    pub thrusters: Vec<ThrusterId>,
    /// Optional reaction wheels attached to `body`.
    pub reaction_wheels: Vec<ReactionWheelId>,
    pub limits: SpacecraftLimits,
}

/// A spacecraft controller's state and allocation summary after the last step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpacecraftReading {
    pub command: SpacecraftCommand,
    /// Desired force in the control frame (N).
    pub desired_force: Vec3,
    /// Desired torque in the control frame (N·m).
    pub desired_torque: Vec3,
    /// Allocated RCS force in the control frame (N).
    pub allocated_force: Vec3,
    /// Total allocated torque (reaction wheels + RCS) in the control frame (N·m).
    pub allocated_torque: Vec3,
    /// Portion of torque supplied by reaction wheels in the control frame (N·m).
    pub wheel_torque: Vec3,
    /// Portion of torque supplied by RCS thrusters in the control frame (N·m).
    pub rcs_torque: Vec3,
    /// Total stored angular momentum across attached reaction wheels (control frame, N·m·s).
    pub total_wheel_momentum: Vec3,
    /// `true` when actuators could not fully realize the requested wrench.
    pub saturated: bool,
}

struct Spacecraft {
    spec: SpacecraftSpec,
    command: SpacecraftCommand,
    reading: SpacecraftReading,
}

/// 6-DOF spacecraft GNC controllers of a rigid world.
#[derive(Default)]
pub struct SpacecraftControllers {
    next: u64,
    controllers: BTreeMap<SpacecraftId, Spacecraft>,
}

impl SpacecraftControllers {
    pub(super) fn is_empty(&self) -> bool {
        self.controllers.is_empty()
    }

    pub fn insert(
        &mut self,
        scene: &RigidScene,
        thrusters: &Thrusters,
        reaction_wheels: &ReactionWheels,
        spec: SpacecraftSpec,
    ) -> Result<SpacecraftId> {
        if scene.body(spec.body).is_none() {
            return Err(Error::Build("spacecraft body is missing".into()));
        }
        if !spec.frame.position.is_finite()
            || !spec.frame.rotation.is_finite()
            || (spec.frame.rotation.length_squared() - 1.0).abs() > 1e-4
        {
            return Err(Error::Build(
                "spacecraft frame must be valid and finite".into(),
            ));
        }
        validate_limits(&spec.limits)?;
        if spec.thrusters.is_empty() && spec.reaction_wheels.is_empty() {
            return Err(Error::Build(
                "spacecraft controller requires at least one thruster or reaction wheel".into(),
            ));
        }
        for &id in &spec.thrusters {
            let t = thrusters
                .spec(id)
                .ok_or_else(|| Error::Build("spacecraft thruster is missing".into()))?;
            if t.body != spec.body {
                return Err(Error::Build(
                    "spacecraft thrusters must be mounted on its body".into(),
                ));
            }
        }
        for &id in &spec.reaction_wheels {
            let w = reaction_wheels
                .spec(id)
                .ok_or_else(|| Error::Build("spacecraft reaction wheel is missing".into()))?;
            if w.body != spec.body {
                return Err(Error::Build(
                    "spacecraft reaction wheels must be mounted on its body".into(),
                ));
            }
        }
        self.next += 1;
        let id = SpacecraftId(self.next);
        self.controllers.insert(
            id,
            Spacecraft {
                spec,
                command: SpacecraftCommand::Off,
                reading: SpacecraftReading {
                    command: SpacecraftCommand::Off,
                    desired_force: Vec3::ZERO,
                    desired_torque: Vec3::ZERO,
                    allocated_force: Vec3::ZERO,
                    allocated_torque: Vec3::ZERO,
                    wheel_torque: Vec3::ZERO,
                    rcs_torque: Vec3::ZERO,
                    total_wheel_momentum: Vec3::ZERO,
                    saturated: false,
                },
            },
        );
        Ok(id)
    }

    pub fn remove(&mut self, id: SpacecraftId) -> bool {
        self.controllers.remove(&id).is_some()
    }

    pub fn command(&mut self, id: SpacecraftId, command: SpacecraftCommand) -> Result<()> {
        validate_command(command)?;
        self.controllers
            .get_mut(&id)
            .ok_or_else(|| Error::Build("no such spacecraft controller".into()))?
            .command = command;
        Ok(())
    }

    pub fn reading(&self, id: SpacecraftId) -> Option<SpacecraftReading> {
        self.controllers.get(&id).map(|c| c.reading)
    }

    pub fn spec(&self, id: SpacecraftId) -> Option<&SpacecraftSpec> {
        self.controllers.get(&id).map(|c| &c.spec)
    }

    pub fn active(&self) -> bool {
        self.controllers
            .values()
            .any(|c| !matches!(c.command, SpacecraftCommand::Off))
    }

    pub(crate) fn before_step(
        &mut self,
        scene: &RigidScene,
        thrusters: &mut Thrusters,
        reaction_wheels: &mut ReactionWheels,
        dt: Real,
    ) -> Result<()> {
        for craft in self.controllers.values_mut() {
            let Some(desc) = scene.body(craft.spec.body) else {
                continue;
            };
            if !desc.enabled || desc.kind != BodyKind::Dynamic {
                continue;
            }
            let body_pose = scene.body_pose(craft.spec.body)?;
            let body_vel = scene.body_velocity(craft.spec.body)?;
            let mass_props = scene.body_mass_properties(craft.spec.body)?;
            let to_ctrl = craft.spec.frame.rotation.conjugate();
            let ctrl_pose = body_pose.compose(&craft.spec.frame);
            let ctrl_rot_inv = ctrl_pose.rotation.conjugate();

            // Express body inertia and wheel momentum in the control frame.
            let r_cb = Mat3::from_quat(to_ctrl);
            let inertia_c = r_cb * mass_props.inertia * r_cb.transpose();
            let omega_c = ctrl_rot_inv * body_vel.angular;
            let wheel_h_c = to_ctrl * reaction_wheels.body_momentum(craft.spec.body);

            if matches!(craft.command, SpacecraftCommand::Off) {
                for &tid in &craft.spec.thrusters {
                    let _ = thrusters.set_throttle(tid, 0.0);
                }
                for &wid in &craft.spec.reaction_wheels {
                    let _ = reaction_wheels.set_torque(wid, 0.0);
                    if reaction_wheels.spec(wid).is_some_and(|s| s.gimbal.is_some()) {
                        let _ = reaction_wheels.set_gimbal_rate(wid, 0.0);
                    }
                }
                craft.reading = SpacecraftReading {
                    command: SpacecraftCommand::Off,
                    desired_force: Vec3::ZERO,
                    desired_torque: Vec3::ZERO,
                    allocated_force: Vec3::ZERO,
                    allocated_torque: Vec3::ZERO,
                    wheel_torque: Vec3::ZERO,
                    rcs_torque: Vec3::ZERO,
                    total_wheel_momentum: wheel_h_c,
                    saturated: false,
                };
                continue;
            }

            let lim = craft.spec.limits;
            let com_world = body_pose.transform_point(mass_props.local_com);
            let gyro_ff = omega_c.cross(inertia_c * omega_c + wheel_h_c);

            let (desired_force_c, desired_torque_c, desaturate) = match craft.command {
                SpacecraftCommand::Off => unreachable!(),
                SpacecraftCommand::Wrench { force, torque } => (force, torque, false),
                SpacecraftCommand::RateHold {
                    linear_velocity,
                    angular_velocity,
                } => {
                    let accel_w = (linear_velocity - body_vel.linear) * lim.velocity_gain;
                    let force_c = ctrl_rot_inv * (accel_w * mass_props.mass);
                    let alpha_c = (angular_velocity - omega_c) * lim.angular_rate_gain;
                    let torque_c = inertia_c * alpha_c + gyro_ff;
                    (force_c, torque_c, false)
                }
                SpacecraftCommand::PoseHold {
                    position,
                    orientation,
                } => {
                    let force_c = pose_hold_force(
                        com_world,
                        body_vel.linear,
                        position,
                        Vec3::ZERO,
                        Vec3::ZERO,
                        mass_props.mass,
                        ctrl_rot_inv,
                        &lim,
                    );
                    let torque_c = attitude_hold_torque(
                        ctrl_pose.rotation,
                        omega_c,
                        orientation.normalize(),
                        Vec3::ZERO,
                        Vec3::ZERO,
                        inertia_c,
                        gyro_ff,
                        &lim,
                    );
                    (force_c, torque_c, false)
                }
                SpacecraftCommand::Trajectory {
                    position,
                    velocity,
                    acceleration,
                    orientation,
                    angular_velocity,
                    angular_acceleration,
                } => {
                    let force_c = pose_hold_force(
                        com_world,
                        body_vel.linear,
                        position,
                        velocity,
                        acceleration,
                        mass_props.mass,
                        ctrl_rot_inv,
                        &lim,
                    );
                    let target_omega_c = ctrl_rot_inv * angular_velocity;
                    let target_alpha_c = ctrl_rot_inv * angular_acceleration;
                    let torque_c = attitude_hold_torque(
                        ctrl_pose.rotation,
                        omega_c,
                        orientation.normalize(),
                        target_omega_c,
                        target_alpha_c,
                        inertia_c,
                        gyro_ff,
                        &lim,
                    );
                    (force_c, torque_c, false)
                }
                SpacecraftCommand::DesaturateWheels {
                    position,
                    orientation,
                } => {
                    let force_c = match position {
                        Some(pos) => pose_hold_force(
                            com_world,
                            body_vel.linear,
                            pos,
                            Vec3::ZERO,
                            Vec3::ZERO,
                            mass_props.mass,
                            ctrl_rot_inv,
                            &lim,
                        ),
                        None => Vec3::ZERO,
                    };
                    let torque_c = attitude_hold_torque(
                        ctrl_pose.rotation,
                        omega_c,
                        orientation.normalize(),
                        Vec3::ZERO,
                        Vec3::ZERO,
                        inertia_c,
                        gyro_ff,
                        &lim,
                    );
                    (force_c, torque_c, true)
                }
            };

            // 1. Reaction wheel allocation (preferential torque or momentum dump).
            let mut wheel_torque_on_body_c = Vec3::ZERO;
            if !craft.spec.reaction_wheels.is_empty() {
                if desaturate {
                    for &wid in &craft.spec.reaction_wheels {
                        let (Some(wspec), Some(wread)) =
                            (reaction_wheels.spec(wid), reaction_wheels.reading(wid))
                        else {
                            continue;
                        };
                        let axis_c = to_ctrl * wread.spin_axis;
                        // Brake wheel speed toward 0; motor torque opposes wheel momentum.
                        let tau_w = (-lim.desaturation_gain * wread.momentum)
                            .clamp(-wspec.max_torque, wspec.max_torque);
                        reaction_wheels.set_torque(wid, tau_w)?;
                        // Reaction torque felt by the spacecraft is -tau_w * axis_c.
                        wheel_torque_on_body_c -= axis_c * tau_w;
                    }
                } else {
                    wheel_torque_on_body_c = allocate_reaction_wheels(
                        &craft.spec.reaction_wheels,
                        reaction_wheels,
                        to_ctrl,
                        desired_torque_c,
                        dt,
                    )?;
                }
            }

            // 2. RCS thruster allocation for linear force + residual torque.
            let rcs_target_torque_c = desired_torque_c - wheel_torque_on_body_c;
            let (allocated_force_c, rcs_torque_c) = if !craft.spec.thrusters.is_empty() {
                allocate_thrusters(
                    &craft.spec.thrusters,
                    thrusters,
                    to_ctrl,
                    mass_props.local_com,
                    desired_force_c,
                    rcs_target_torque_c,
                    &lim,
                )?
            } else {
                (Vec3::ZERO, Vec3::ZERO)
            };

            let allocated_torque_c = wheel_torque_on_body_c + rcs_torque_c;
            let force_err = (allocated_force_c - desired_force_c).length();
            let torque_err = (allocated_torque_c - desired_torque_c).length();
            let saturated = force_err > 1e-3 * desired_force_c.length().max(1.0)
                || torque_err > 1e-3 * desired_torque_c.length().max(1.0);

            craft.reading = SpacecraftReading {
                command: craft.command,
                desired_force: desired_force_c,
                desired_torque: desired_torque_c,
                allocated_force: allocated_force_c,
                allocated_torque: allocated_torque_c,
                wheel_torque: wheel_torque_on_body_c,
                rcs_torque: rcs_torque_c,
                total_wheel_momentum: wheel_h_c,
                saturated,
            };
        }
        Ok(())
    }
}

fn validate_limits(lim: &SpacecraftLimits) -> Result<()> {
    let positive = [
        lim.max_linear_speed,
        lim.max_angular_speed,
        lim.position_frequency,
        lim.attitude_frequency,
        lim.velocity_gain,
        lim.angular_rate_gain,
        lim.desaturation_gain,
        lim.force_weight,
        lim.torque_weight,
    ];
    if positive.iter().any(|v| !v.is_finite() || *v <= 0.0)
        || !lim.regularization.is_finite()
        || lim.regularization < 0.0
    {
        return Err(Error::Build(
            "spacecraft controller limits must be finite and positive".into(),
        ));
    }
    Ok(())
}

fn validate_command(command: SpacecraftCommand) -> Result<()> {
    let finite_quat = |q: Quat| q.is_finite() && q.length_squared() > 1e-12;
    let ok = match command {
        SpacecraftCommand::Off => true,
        SpacecraftCommand::Wrench { force, torque } => force.is_finite() && torque.is_finite(),
        SpacecraftCommand::RateHold {
            linear_velocity,
            angular_velocity,
        } => linear_velocity.is_finite() && angular_velocity.is_finite(),
        SpacecraftCommand::PoseHold {
            position,
            orientation,
        } => position.is_finite() && finite_quat(orientation),
        SpacecraftCommand::Trajectory {
            position,
            velocity,
            acceleration,
            orientation,
            angular_velocity,
            angular_acceleration,
        } => {
            position.is_finite()
                && velocity.is_finite()
                && acceleration.is_finite()
                && finite_quat(orientation)
                && angular_velocity.is_finite()
                && angular_acceleration.is_finite()
        }
        SpacecraftCommand::DesaturateWheels {
            position,
            orientation,
        } => position.is_none_or(|p| p.is_finite()) && finite_quat(orientation),
    };
    if !ok {
        return Err(Error::Build(
            "spacecraft command must be finite with valid orientation".into(),
        ));
    }
    Ok(())
}

/// Critically damped 3D translation controller with velocity saturation.
#[allow(clippy::too_many_arguments)]
fn pose_hold_force(
    position: Vec3,
    velocity: Vec3,
    target_pos: Vec3,
    target_vel: Vec3,
    target_accel: Vec3,
    mass: Real,
    ctrl_rot_inv: Quat,
    lim: &SpacecraftLimits,
) -> Vec3 {
    let pos_err = target_pos - position;
    let wn = lim.position_frequency;
    // Galilean-invariant velocity-bounded outer loop: saturate closing velocity
    // relative to `target_vel` so rendezvous with a fast-drifting orbital target
    // never stalls against world-frame speed limits.
    let mut closing_vel = pos_err * (0.5 * wn);
    let closing_speed = closing_vel.length();
    if closing_speed > lim.max_linear_speed {
        closing_vel *= lim.max_linear_speed / closing_speed;
    }
    let desired_vel = target_vel + closing_vel;
    let accel_w = target_accel + (desired_vel - velocity) * (2.0 * wn);
    ctrl_rot_inv * (accel_w * mass)
}

/// Critically damped SO(3) attitude controller with angular-rate saturation and
/// gyroscopic feedforward.
#[allow(clippy::too_many_arguments)]
fn attitude_hold_torque(
    current: Quat,
    omega_c: Vec3,
    target: Quat,
    target_omega_c: Vec3,
    target_alpha_c: Vec3,
    inertia_c: Mat3,
    gyro_ff: Vec3,
    lim: &SpacecraftLimits,
) -> Vec3 {
    // Error quaternion bringing current frame to target frame, expressed in
    // current control frame: q_err = current⁻¹ * target.
    let mut q_err = (current.conjugate() * target).normalize();
    if q_err.w < 0.0 {
        q_err = -q_err;
    }
    let v = Vec3::new(q_err.x, q_err.y, q_err.z);
    let sin_half = v.length();
    let rot_err = if sin_half > 1e-12 {
        v * (2.0 * sin_half.atan2(q_err.w) / sin_half)
    } else {
        v * 2.0
    };
    let wn = lim.attitude_frequency;
    let mut closing_omega = rot_err * (0.5 * wn);
    let closing_rate = closing_omega.length();
    if closing_rate > lim.max_angular_speed {
        closing_omega *= lim.max_angular_speed / closing_rate;
    }
    let desired_omega = target_omega_c + closing_omega;
    let alpha_c = target_alpha_c + (desired_omega - omega_c) * (2.0 * wn);
    inertia_c * alpha_c + gyro_ff
}

/// Allocates `desired_torque_c` (torque on the spacecraft body) across attached
/// single-gimbal CMGs (via Wie/Schaub Singularity-Robust Inverse steering) and
/// fixed-axis reaction wheels (via box-constrained projected Gauss–Seidel),
/// returning the total torque actually supplied on the spacecraft body.
fn allocate_reaction_wheels(
    ids: &[ReactionWheelId],
    wheels: &mut ReactionWheels,
    to_ctrl: Quat,
    desired_torque_c: Vec3,
    dt: Real,
) -> Result<Vec3> {
    let mut cmg_ids = Vec::new();
    let mut rw_ids = Vec::new();
    for &id in ids {
        if wheels.spec(id).is_some_and(|s| s.gimbal.is_some()) {
            cmg_ids.push(id);
        } else {
            rw_ids.push(id);
        }
    }

    // 1. Single-Gimbal CMG Singularity-Robust Inverse (SRI) steering law:
    //    For each CMG `j`, gimballing at `γ̇_j` produces body torque
    //    `c_j * γ̇_j` where `c_j = -(g_c × h_c)`.
    //    `γ̇ = A^T (A A^T + λ I)⁻¹ τ_req` with adaptive singularity damping `λ`.
    let mut body_torque = Vec3::ZERO;
    if !cmg_ids.is_empty() {
        let mut cols = Vec::with_capacity(cmg_ids.len());
        let mut max_rates = Vec::with_capacity(cmg_ids.len());
        let mut gram = Mat3::ZERO;
        let mut trace_scale = 0.0_f64;
        for &id in &cmg_ids {
            let (Some(spec), Some(read)) = (wheels.spec(id), wheels.reading(id)) else {
                cols.push(Vec3::ZERO);
                max_rates.push(0.0);
                continue;
            };
            let gimbal = spec.gimbal.unwrap();
            let g_c = to_ctrl * gimbal.axis;
            let h_c = (to_ctrl * read.spin_axis) * read.momentum;
            let cj = -g_c.cross(h_c);
            cols.push(cj);
            max_rates.push(gimbal.max_rate);
            trace_scale += cj.length_squared();
            gram.x_axis += cj * cj.x;
            gram.y_axis += cj * cj.y;
            gram.z_axis += cj * cj.z;
        }
        if trace_scale > 1e-12 {
            let avg = (trace_scale / 3.0).max(1e-12);
            let det_norm = (gram.determinant().max(0.0)) / (avg * avg * avg);
            // Adaptive Wie/Schaub SRI damping: activates smoothly near internal singularities.
            let lambda = avg * (0.04 * (-12.0 * det_norm).exp() + 1e-6);
            let damped = gram + Mat3::from_diagonal(Vec3::splat(lambda));
            let y = damped.inverse() * desired_torque_c;
            for (j, &id) in cmg_ids.iter().enumerate() {
                let rate = cols[j].dot(y).clamp(-max_rates[j], max_rates[j]);
                wheels.set_gimbal_rate(id, rate)?;
                // Maintain nominal constant rotor speed on SGCMGs.
                wheels.set_torque(id, 0.0)?;
                body_torque += cols[j] * rate;
            }
        } else {
            for &id in &cmg_ids {
                wheels.set_gimbal_rate(id, 0.0)?;
            }
        }
    }

    // 2. Fixed-axis reaction wheel allocation for any residual torque.
    let residual_target = desired_torque_c - body_torque;
    if rw_ids.is_empty() {
        return Ok(body_torque);
    }

    let n = rw_ids.len();
    let mut cols = Vec::with_capacity(n);
    let mut lo = Vec::with_capacity(n);
    let mut hi = Vec::with_capacity(n);
    for &id in &rw_ids {
        let (Some(spec), Some(read)) = (wheels.spec(id), wheels.reading(id)) else {
            cols.push(Vec3::ZERO);
            lo.push(0.0);
            hi.push(0.0);
            continue;
        };
        cols.push(-(to_ctrl * read.spin_axis));
        let max_pos = ((spec.max_speed - read.speed) * spec.inertia / dt)
            .clamp(0.0, spec.max_torque);
        let max_neg = ((-spec.max_speed - read.speed) * spec.inertia / dt)
            .clamp(-spec.max_torque, 0.0);
        lo.push(max_neg);
        hi.push(max_pos);
    }

    let reg = 1e-9;
    let mut tau = vec![0.0; n];
    for _ in 0..QP_ITERATIONS {
        for j in 0..n {
            let bj = cols[j];
            let diag = bj.length_squared() + reg;
            if diag <= reg || hi[j] <= lo[j] {
                tau[j] = 0.0;
                continue;
            }
            let mut residual = residual_target;
            for k in 0..n {
                if k != j {
                    residual -= cols[k] * tau[k];
                }
            }
            tau[j] = (bj.dot(residual) / diag).clamp(lo[j], hi[j]);
        }
    }

    for (j, &id) in rw_ids.iter().enumerate() {
        wheels.set_torque(id, tau[j])?;
        body_torque += cols[j] * tau[j];
    }
    Ok(body_torque)
}

/// Solves `min_{0 <= u <= 1} 1/2 ||W^(1/2)(B u - w)||^2 + (λ/2) ||u||^2` via
/// projected Gauss–Seidel and applies throttles to the RCS thrusters.
fn allocate_thrusters(
    ids: &[ThrusterId],
    thrusters: &mut Thrusters,
    to_ctrl: Quat,
    body_local_com: Vec3,
    desired_force_c: Vec3,
    desired_torque_c: Vec3,
    lim: &SpacecraftLimits,
) -> Result<(Vec3, Vec3)> {
    let n = ids.len();
    let mut bf = Vec::with_capacity(n);
    let mut bt = Vec::with_capacity(n);
    let mut usable = Vec::with_capacity(n);

    for &id in ids {
        let Some(spec) = thrusters.spec(id) else {
            bf.push(Vec3::ZERO);
            bt.push(Vec3::ZERO);
            usable.push(false);
            continue;
        };
        let active = spec.max_thrust > 0.0 && thrusters.has_propellant(id);
        usable.push(active);
        if active {
            let axis_b = spec.frame.rotation * Vec3::X;
            let arm_b = spec.frame.position - body_local_com;
            let force_c = (to_ctrl * axis_b) * spec.max_thrust;
            let torque_c = (to_ctrl * arm_b.cross(axis_b)) * spec.max_thrust;
            bf.push(force_c);
            bt.push(torque_c);
        } else {
            bf.push(Vec3::ZERO);
            bt.push(Vec3::ZERO);
        }
    }

    let wf = lim.force_weight;
    let wt = lim.torque_weight;
    let reg = lim.regularization;
    let mut u = vec![0.0; n];

    // Precompute upper/full symmetric normal matrix H and RHS g for speed.
    let mut diag = vec![0.0; n];
    let mut rhs = vec![0.0; n];
    for j in 0..n {
        if !usable[j] {
            continue;
        }
        diag[j] = wf * bf[j].length_squared() + wt * bt[j].length_squared() + reg;
        rhs[j] = wf * bf[j].dot(desired_force_c) + wt * bt[j].dot(desired_torque_c);
    }

    for _ in 0..QP_ITERATIONS {
        let mut max_delta: Real = 0.0;
        for j in 0..n {
            if !usable[j] || diag[j] <= reg {
                u[j] = 0.0;
                continue;
            }
            let mut sum = rhs[j];
            for k in 0..n {
                if k != j && u[k] != 0.0 {
                    let hjk = wf * bf[j].dot(bf[k]) + wt * bt[j].dot(bt[k]);
                    sum -= hjk * u[k];
                }
            }
            let next = (sum / diag[j]).clamp(0.0, 1.0);
            max_delta = max_delta.max((next - u[j]).abs());
            u[j] = next;
        }
        if max_delta < 1e-11 {
            break;
        }
    }

    let mut alloc_force = Vec3::ZERO;
    let mut alloc_torque = Vec3::ZERO;
    for (j, &id) in ids.iter().enumerate() {
        let throttle = if u[j] < 1e-9 { 0.0 } else { u[j] };
        thrusters.set_throttle(id, throttle)?;
        alloc_force += bf[j] * throttle;
        alloc_torque += bt[j] * throttle;
    }
    Ok((alloc_force, alloc_torque))
}
