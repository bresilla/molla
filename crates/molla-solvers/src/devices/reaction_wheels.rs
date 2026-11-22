//! Spacecraft reaction wheels and single-gimbal control moment gyroscopes
//! (SGCMGs): internal momentum-exchange actuators with exact 3D gyroscopic
//! cross-coupling.
//!
//! Each wheel spins about a body-fixed axis `axis` (or a gimballed axis when
//! [`GimbalSpec`] is attached) with axial moment of inertia `inertia`.
//! Accelerating the rotor at `τ_w` exerts `-τ_w · â`, gimballing a spinning
//! rotor at `γ̇` about `ĝ` exerts `-γ̇ (ĝ × h_w)`, and any body angular velocity
//! `ω` perpendicular to `h_w` produces the gyroscopic coupling torque
//! `-ω × h_w`. All three are solved jointly by an exact finite-rotation
//! momentum exchange `τ = -(q₁ h₁ - q₀ h₀) / Δt` so total angular momentum
//! `I_world · ω + h_w` is conserved across spin-ups, gimbal slews, and 3D
//! rotations.

use std::collections::BTreeMap;

use molla_core::{Error, Result};
use molla_math::{Quat, Real, SpatialVector, Vec3};
use molla_sim::State;
use molla_sim::runtime::{BodyHandle, BodyKind, RigidScene};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReactionWheelId(pub u64);

/// Optional rotor static & dynamic imbalance jitter parameters for a reaction
/// wheel or CMG.
///
/// A spinning rotor at angular rate `Ω` and phase `φ` exerts transverse
/// centrifugal force `F = U_s · Ω²` (applied at `mount_offset`) and dynamic
/// rocking torque `τ = U_d · Ω²` in the plane orthogonal to the spin axis.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImbalanceSpec {
    /// Static mass eccentricity `U_s = m · e` (kg·m), nonnegative.
    pub static_unbalance: Real,
    /// Dynamic product-of-inertia imbalance `U_d` (kg·m²), nonnegative.
    pub dynamic_unbalance: Real,
    /// Wheel mount position in the spacecraft body frame (m).
    pub mount_offset: Vec3,
    /// Initial rotor phase angle `φ₀` (rad).
    pub initial_phase: Real,
}

/// Optional single-gimbal specification turning a reaction wheel into a
/// single-gimbal control moment gyroscope (SGCMG).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GimbalSpec {
    /// Body-frame gimbal axis `ĝ`, orthogonal to the rotor spin axis `axis`.
    pub axis: Vec3,
    /// Maximum gimbal slew rate `|γ̇| <= max_rate` (rad/s), strictly positive.
    pub max_rate: Real,
    /// Initial gimbal angle `γ₀` (rad).
    pub initial_angle: Real,
}

/// What a reaction wheel is told to do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReactionWheelCommand {
    /// Apply motor torque `τ_w` (N·m) to the wheel rotor; the spacecraft body
    /// feels `-τ_w` along the spin axis.
    Torque(Real),
    /// Drive the wheel toward a target spin speed (rad/s) within `max_torque`
    /// (used for spin-up or momentum desaturation).
    Speed(Real),
}

impl Default for ReactionWheelCommand {
    fn default() -> Self {
        Self::Torque(0.0)
    }
}

/// A reaction wheel (or single-gimbal CMG when `gimbal` is `Some`) mounted on
/// `body`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReactionWheelSpec {
    pub body: BodyHandle,
    /// Spin axis in the body frame at `gimbal_angle = 0` (normalized on insertion).
    pub axis: Vec3,
    /// Axial moment of inertia of the rotor (kg·m²), strictly positive.
    pub inertia: Real,
    /// Maximum motor torque magnitude (N·m), nonnegative.
    pub max_torque: Real,
    /// Saturation spin speed limit `|Ω| <= max_speed` (rad/s), strictly positive.
    pub max_speed: Real,
    /// Initial rotor spin speed (rad/s), within `[-max_speed, max_speed]`.
    pub initial_speed: Real,
    /// Optional single-gimbal axis turning this wheel into an SGCMG.
    pub gimbal: Option<GimbalSpec>,
}

/// A reaction wheel / CMG's state and averaged torques over the last step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReactionWheelReading {
    /// Rotor spin speed relative to the spacecraft body (rad/s).
    pub speed: Real,
    /// Stored scalar angular momentum `I_w · Ω` along the current spin axis (N·m·s).
    pub momentum: Real,
    /// Current unit spin axis in the body frame (rotates with `gimbal_angle` for CMGs).
    pub spin_axis: Vec3,
    /// Current gimbal angle (rad; `0.0` for fixed reaction wheels).
    pub gimbal_angle: Real,
    /// Active gimbal slew rate (rad/s; `0.0` for fixed reaction wheels).
    pub gimbal_rate: Real,
    /// Actual motor torque delivered to the wheel (N·m), averaged over the step.
    pub applied_torque: Real,
    /// Gyroscopic coupling torque `-ω × h_w` applied to the body (N·m, world),
    /// averaged over the step.
    pub gyroscopic_torque: Vec3,
    /// Total torque applied to the body (N·m, world), averaged over the step.
    pub world_torque: Vec3,
    /// Transverse centrifugal jitter force from static rotor unbalance (N, world).
    pub jitter_force: Vec3,
    /// Transverse rocking jitter torque from dynamic rotor unbalance (N·m, world).
    pub jitter_torque: Vec3,
    /// `true` when the wheel hit its speed saturation limit during the step.
    pub saturated: bool,
}

struct ReactionWheel {
    spec: ReactionWheelSpec,
    imbalance: ImbalanceSpec,
    rotor_phase: Real,
    unit_axis: Vec3,
    gimbal_axis: Option<Vec3>,
    gimbal_angle: Real,
    gimbal_rate: Real,
    command: ReactionWheelCommand,
    speed: Real,
    reading: ReactionWheelReading,
}

impl ReactionWheel {
    fn spin_axis_at(&self, angle: Real) -> Vec3 {
        match self.gimbal_axis {
            Some(g) => (Quat::from_axis_angle(g, angle) * self.unit_axis).normalize(),
            None => self.unit_axis,
        }
    }

    fn current_axis(&self) -> Vec3 {
        self.spin_axis_at(self.gimbal_angle)
    }

    fn is_active(&self) -> bool {
        self.speed != 0.0
            || self.gimbal_rate != 0.0
            || match self.command {
                ReactionWheelCommand::Torque(t) => t != 0.0,
                ReactionWheelCommand::Speed(s) => s != 0.0,
            }
    }
}

/// Reaction wheels of a rigid world.
#[derive(Default)]
pub struct ReactionWheels {
    next: u64,
    wheels: BTreeMap<ReactionWheelId, ReactionWheel>,
}

impl ReactionWheels {
    pub(super) fn is_empty(&self) -> bool {
        self.wheels.is_empty()
    }

    pub fn insert(
        &mut self,
        scene: &RigidScene,
        mut spec: ReactionWheelSpec,
    ) -> Result<ReactionWheelId> {
        if scene.body(spec.body).is_none() {
            return Err(Error::Build("reaction wheel body is missing".into()));
        }
        if !spec.axis.is_finite() || spec.axis.length_squared() < 1e-18 {
            return Err(Error::Build(
                "reaction wheel axis must be finite and non-zero".into(),
            ));
        }
        let unit_axis = spec.axis.normalize();
        spec.axis = unit_axis;
        if !spec.inertia.is_finite() || spec.inertia <= 0.0 {
            return Err(Error::Build(
                "reaction wheel inertia must be finite and positive".into(),
            ));
        }
        if !spec.max_torque.is_finite() || spec.max_torque < 0.0 {
            return Err(Error::Build(
                "reaction wheel max_torque must be finite and nonnegative".into(),
            ));
        }
        if !spec.max_speed.is_finite() || spec.max_speed <= 0.0 {
            return Err(Error::Build(
                "reaction wheel max_speed must be finite and positive".into(),
            ));
        }
        if !spec.initial_speed.is_finite() || spec.initial_speed.abs() > spec.max_speed {
            return Err(Error::Build(
                "reaction wheel initial_speed must be finite and within max_speed".into(),
            ));
        }
        let (gimbal_axis, gimbal_angle) = match spec.gimbal {
            Some(mut g) => {
                if !g.axis.is_finite() || g.axis.length_squared() < 1e-18 {
                    return Err(Error::Build(
                        "CMG gimbal axis must be finite and non-zero".into(),
                    ));
                }
                let unit_g = g.axis.normalize();
                if unit_g.dot(unit_axis).abs() > 1e-6 {
                    return Err(Error::Build(
                        "CMG gimbal axis must be orthogonal to rotor spin axis".into(),
                    ));
                }
                if !g.max_rate.is_finite() || g.max_rate <= 0.0 || !g.initial_angle.is_finite() {
                    return Err(Error::Build(
                        "CMG gimbal max_rate must be positive and initial_angle finite".into(),
                    ));
                }
                g.axis = unit_g;
                spec.gimbal = Some(g);
                (Some(unit_g), g.initial_angle)
            }
            None => (None, 0.0),
        };
        self.next += 1;
        let id = ReactionWheelId(self.next);
        let speed = spec.initial_speed;
        let momentum = spec.inertia * speed;
        let spin_axis = match gimbal_axis {
            Some(g) => (Quat::from_axis_angle(g, gimbal_angle) * unit_axis).normalize(),
            None => unit_axis,
        };
        self.wheels.insert(
            id,
            ReactionWheel {
                spec,
                imbalance: ImbalanceSpec::default(),
                rotor_phase: 0.0,
                unit_axis,
                gimbal_axis,
                gimbal_angle,
                gimbal_rate: 0.0,
                command: ReactionWheelCommand::Torque(0.0),
                speed,
                reading: ReactionWheelReading {
                    speed,
                    momentum,
                    spin_axis,
                    gimbal_angle,
                    ..ReactionWheelReading::default()
                },
            },
        );
        Ok(id)
    }

    pub fn remove(&mut self, id: ReactionWheelId) -> bool {
        self.wheels.remove(&id).is_some()
    }

    pub fn command(&mut self, id: ReactionWheelId, command: ReactionWheelCommand) -> Result<()> {
        match command {
            ReactionWheelCommand::Torque(t) if !t.is_finite() => {
                return Err(Error::Build(
                    "reaction wheel torque command must be finite".into(),
                ));
            }
            ReactionWheelCommand::Speed(s) if !s.is_finite() => {
                return Err(Error::Build(
                    "reaction wheel speed command must be finite".into(),
                ));
            }
            _ => {}
        }
        self.wheels
            .get_mut(&id)
            .ok_or_else(|| Error::Build("no such reaction wheel".into()))?
            .command = command;
        Ok(())
    }

    /// Convenience for `command(id, ReactionWheelCommand::Torque(torque))`.
    pub fn set_torque(&mut self, id: ReactionWheelId, torque: Real) -> Result<()> {
        self.command(id, ReactionWheelCommand::Torque(torque))
    }

    /// Convenience for `command(id, ReactionWheelCommand::Speed(speed))`.
    pub fn set_speed(&mut self, id: ReactionWheelId, speed: Real) -> Result<()> {
        self.command(id, ReactionWheelCommand::Speed(speed))
    }

    /// Attach optional static/dynamic rotor imbalance jitter to a reaction wheel or CMG.
    pub fn set_imbalance(&mut self, id: ReactionWheelId, imbalance: ImbalanceSpec) -> Result<()> {
        if !imbalance.static_unbalance.is_finite()
            || imbalance.static_unbalance < 0.0
            || !imbalance.dynamic_unbalance.is_finite()
            || imbalance.dynamic_unbalance < 0.0
            || !imbalance.mount_offset.is_finite()
            || !imbalance.initial_phase.is_finite()
        {
            return Err(Error::Build(
                "reaction wheel imbalance parameters must be finite and nonnegative".into(),
            ));
        }
        let wheel = self
            .wheels
            .get_mut(&id)
            .ok_or_else(|| Error::Build("no such reaction wheel".into()))?;
        wheel.imbalance = imbalance;
        wheel.rotor_phase = imbalance.initial_phase;
        Ok(())
    }

    /// Command the gimbal slew rate `γ̇` (rad/s) on a single-gimbal CMG.
    pub fn set_gimbal_rate(&mut self, id: ReactionWheelId, rate: Real) -> Result<()> {
        if !rate.is_finite() {
            return Err(Error::Build("CMG gimbal rate must be finite".into()));
        }
        let wheel = self
            .wheels
            .get_mut(&id)
            .ok_or_else(|| Error::Build("no such reaction wheel".into()))?;
        let gimbal = wheel
            .spec
            .gimbal
            .ok_or_else(|| Error::Build("reaction wheel has no gimbal".into()))?;
        wheel.gimbal_rate = rate.clamp(-gimbal.max_rate, gimbal.max_rate);
        Ok(())
    }

    pub fn reading(&self, id: ReactionWheelId) -> Option<ReactionWheelReading> {
        self.wheels.get(&id).map(|w| w.reading)
    }

    pub fn spec(&self, id: ReactionWheelId) -> Option<&ReactionWheelSpec> {
        self.wheels.get(&id).map(|w| &w.spec)
    }

    /// Total stored wheel angular momentum in the body frame (N·m·s) for `body`.
    pub fn body_momentum(&self, body: BodyHandle) -> Vec3 {
        let mut total = Vec3::ZERO;
        for wheel in self.wheels.values() {
            if wheel.spec.body == body {
                total += wheel.current_axis() * (wheel.spec.inertia * wheel.speed);
            }
        }
        total
    }

    /// Whether any wheel is spinning (gyroscopic coupling) or commanded.
    pub fn active(&self) -> bool {
        self.wheels.values().any(ReactionWheel::is_active)
    }

    pub(crate) fn before_step(&mut self) {
        for wheel in self.wheels.values_mut() {
            wheel.reading.applied_torque = 0.0;
            wheel.reading.gyroscopic_torque = Vec3::ZERO;
            wheel.reading.world_torque = Vec3::ZERO;
            wheel.reading.jitter_force = Vec3::ZERO;
            wheel.reading.jitter_torque = Vec3::ZERO;
            wheel.reading.saturated = false;
        }
    }

    pub(crate) fn apply(
        &mut self,
        scene: &RigidScene,
        state: &State,
        wrenches: &mut [SpatialVector],
        sub_dt: Real,
        weight: Real,
    ) -> Result<()> {
        if self.wheels.is_empty() || !(sub_dt > 0.0) {
            return Ok(());
        }
        let poses = state.body_q.host()?;
        let velocities = state.body_qd.host()?;
        let inertias = scene.model().body_inertia.host()?;
        let coms = scene.model().body_com.host()?;

        // First pass: advance each wheel's rotor speed and collect per-body
        // momentum endpoints `(h_local_before, h_local_after)` so multi-wheel
        // gyrostats conserve total angular momentum `I_world ω + h_w` jointly.
        struct StepWheel {
            id: ReactionWheelId,
            body_idx: usize,
            tau_actual: Real,
            saturated: bool,
            omega_next: Real,
            gimbal_next: Real,
            axis_next: Vec3,
            h_local_before: Vec3,
            h_local_after: Vec3,
        }

        let mut active_wheels = Vec::new();
        let mut body_h_before: BTreeMap<usize, Vec3> = BTreeMap::new();
        let mut body_h_after: BTreeMap<usize, Vec3> = BTreeMap::new();

        for (&id, wheel) in &mut self.wheels {
            if !wheel.is_active() {
                continue;
            }
            let (Some(body), Some(desc)) = (
                scene.body_index(wheel.spec.body),
                scene.body(wheel.spec.body),
            ) else {
                continue;
            };
            if !desc.enabled || desc.kind != BodyKind::Dynamic {
                continue;
            }
            let spec = wheel.spec;
            let tau_cmd = match wheel.command {
                ReactionWheelCommand::Torque(t) => t.clamp(-spec.max_torque, spec.max_torque),
                ReactionWheelCommand::Speed(target) => {
                    let clamped = target.clamp(-spec.max_speed, spec.max_speed);
                    ((clamped - wheel.speed) * spec.inertia / sub_dt)
                        .clamp(-spec.max_torque, spec.max_torque)
                }
            };
            let omega_before = wheel.speed;
            let unclamped_next = omega_before + (tau_cmd / spec.inertia) * sub_dt;
            let omega_next = unclamped_next.clamp(-spec.max_speed, spec.max_speed);
            let tau_actual = (omega_next - omega_before) * spec.inertia / sub_dt;
            let saturated = (unclamped_next - omega_next).abs() > 1e-12
                || (omega_next.abs() >= spec.max_speed - 1e-12 && tau_cmd * omega_next > 0.0);

            let axis_before = wheel.current_axis();
            let gimbal_next = if wheel.gimbal_axis.is_some() {
                wheel.gimbal_angle + wheel.gimbal_rate * sub_dt
            } else {
                wheel.gimbal_angle
            };
            let axis_next = wheel.spin_axis_at(gimbal_next);

            let body_idx = body.index();
            let h_local_before = axis_before * (spec.inertia * omega_before);
            let h_local_after = axis_next * (spec.inertia * omega_next);
            *body_h_before.entry(body_idx).or_insert(Vec3::ZERO) += h_local_before;
            *body_h_after.entry(body_idx).or_insert(Vec3::ZERO) += h_local_after;
            active_wheels.push(StepWheel {
                id,
                body_idx,
                tau_actual,
                saturated,
                omega_next,
                gimbal_next,
                axis_next,
                h_local_before,
                h_local_after,
            });
        }

        // Second pass: predict each affected body's post-step orientation q₁
        // using a 3-iteration fixed-point solve for the exact finite-rotation
        // angular-momentum exchange `τ = -(q₁ h_after - q₀ h_before) / sub_dt`.
        let mut body_q_next: BTreeMap<usize, molla_math::Quat> = BTreeMap::new();
        for (&i, &h_b0) in &body_h_before {
            let h_b1 = body_h_after[&i];
            let q0 = poses[i].rotation;
            let omega0 = velocities[i].angular;
            let rot0 = molla_math::Mat3::from_quat(q0);
            let i_world = rot0 * inertias[i] * rot0.transpose();
            let inv_i = if i_world.determinant().abs() > 1e-18 {
                i_world.inverse()
            } else {
                molla_math::Mat3::ZERO
            };
            let h_w0 = q0 * h_b0;
            let body_gyro = omega0.cross(i_world * omega0);
            let mut omega1 = omega0;
            for _ in 0..3 {
                let q1 = (molla_math::Quat::from_scaled_axis(omega1 * sub_dt) * q0).normalize();
                let tau_total = -(q1 * h_b1 - h_w0) / sub_dt;
                omega1 = omega0 + (inv_i * (tau_total + wrenches[i].angular - body_gyro)) * sub_dt;
            }
            let q1 = (molla_math::Quat::from_scaled_axis(omega1 * sub_dt) * q0).normalize();
            body_q_next.insert(i, q1);
        }

        // Third pass: apply each wheel's exact momentum-conserving torque and
        // update readings.
        for step in active_wheels {
            let i = step.body_idx;
            let q0 = poses[i].rotation;
            let q1 = body_q_next[&i];
            let h0 = q0 * step.h_local_before;
            let h1 = q1 * step.h_local_after;
            let total_torque = -(h1 - h0) / sub_dt;

            let wheel = self.wheels.get_mut(&step.id).unwrap();
            let h_rot_only = q1 * step.h_local_before;
            let gyro_torque = -(h_rot_only - h0) / sub_dt;

            wrenches[i].angular += total_torque;
            // Optional static & dynamic rotor imbalance jitter (`F = U_s Ω²`, `τ = U_d Ω²`).
            let omega_avg = 0.5 * (wheel.speed + step.omega_next);
            wheel.rotor_phase = (wheel.rotor_phase + omega_avg * sub_dt).rem_euclid(std::f64::consts::TAU);
            let mut j_force = Vec3::ZERO;
            let mut j_torque = Vec3::ZERO;
            if wheel.imbalance.static_unbalance > 0.0 || wheel.imbalance.dynamic_unbalance > 0.0 {
                let s_ax = step.axis_next;
                let helper = if s_ax.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
                let u_b = s_ax.cross(helper).normalize();
                let v_b = s_ax.cross(u_b).normalize();
                let dir_b = u_b * wheel.rotor_phase.cos() + v_b * wheel.rotor_phase.sin();
                let w2 = omega_avg * omega_avg;
                let dir_w = q0 * dir_b;
                if wheel.imbalance.static_unbalance > 0.0 {
                    j_force = dir_w * (wheel.imbalance.static_unbalance * w2);
                    let arm = poses[i].transform_point(wheel.imbalance.mount_offset)
                        - poses[i].transform_point(coms[i]);
                    wrenches[i].linear += j_force;
                    wrenches[i].angular += arm.cross(j_force);
                }
                if wheel.imbalance.dynamic_unbalance > 0.0 {
                    j_torque = dir_w * (wheel.imbalance.dynamic_unbalance * w2);
                    wrenches[i].angular += j_torque;
                }
            }
            wheel.speed = step.omega_next;
            wheel.gimbal_angle = step.gimbal_next;
            let reading = &mut wheel.reading;
            reading.speed = step.omega_next;
            reading.momentum = wheel.spec.inertia * step.omega_next;
            reading.spin_axis = step.axis_next;
            reading.gimbal_angle = step.gimbal_next;
            reading.gimbal_rate = wheel.gimbal_rate;
            reading.applied_torque += step.tau_actual * weight;
            reading.gyroscopic_torque += gyro_torque * weight;
            reading.world_torque += total_torque * weight;
            reading.jitter_force += j_force * weight;
            reading.jitter_torque += j_torque * weight;
            reading.saturated |= step.saturated;
        }
        Ok(())
    }
}
