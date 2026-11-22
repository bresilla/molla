//! Spacecraft thrusters: reaction-control (RCS) and main-engine nozzles.
//!
//! Each thruster pushes its body along `+X` of its nozzle frame at the frame's
//! origin, producing both linear thrust and lever-arm torque about the body's
//! centre of mass. Thrusters optionally model first-order valve/chamber lag,
//! minimum impulse bits (MIB), timed pulse burns, propellant consumption from
//! specific impulse `I_sp`, and dynamic body mass depletion (Tsiolkovsky).

use std::collections::BTreeMap;

use molla_core::{Error, Result};
use molla_math::{GRAVITY, Real, SpatialVector, Transform, Vec3};
use molla_sim::{MassProperties, State};
use molla_sim::runtime::{BodyHandle, BodyKind, RigidScene};

/// Floor on body mass (kg) during propellant depletion so the body never hits
/// zero or negative mass.
const MIN_DRY_MASS: Real = 1e-3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ThrusterId(pub u64);

/// Valve modulation strategy converting continuous GNC throttle demands `[0, 1]`
/// into physical thruster valve commands.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum ThrusterModulation {
    /// Continuous analog throttle in `[0, 1]` (default).
    #[default]
    Continuous,
    /// Fixed-period Pulse-Width Modulation (PWM): latches duty `u` each cycle of
    /// length `period` (s) and opens the valve fully for `u * period`, suppressing
    /// pulses shorter than `min_pulse_width` (s).
    Pwm {
        period: Real,
        min_pulse_width: Real,
    },
    /// Classic aerospace Pulse-Width Pulse-Frequency (PWPF) modulator:
    /// first-order lag `τ_m ḟ + f = K_m (u - u_fb)` driving a Schmitt trigger
    /// with hysteresis thresholds `0 <= u_off < u_on`.
    Pwpf {
        gain_km: Real,
        time_constant_tm: Real,
        u_on: Real,
        u_off: Real,
    },
}

impl ThrusterModulation {
    /// Typical aerospace RCS PWPF tuning (crisp ~17 ms minimum pulse, ~2.5% deadband).
    pub fn aerospace_pwpf() -> Self {
        Self::Pwpf {
            gain_km: 10.0,
            time_constant_tm: 0.10,
            u_on: 0.25,
            u_off: 0.08,
        }
    }

    /// Standard 10 Hz RCS PWM modulator with 12 ms minimum valve on-time.
    pub fn rcs_pwm_10hz() -> Self {
        Self::Pwm {
            period: 0.10,
            min_pulse_width: 0.012,
        }
    }

    fn validated(self) -> Result<Self> {
        match self {
            Self::Continuous => Ok(self),
            Self::Pwm {
                period,
                min_pulse_width,
            } => {
                if !period.is_finite()
                    || period <= 0.0
                    || !min_pulse_width.is_finite()
                    || min_pulse_width < 0.0
                    || min_pulse_width > period
                {
                    return Err(Error::Build(
                        "PWM requires finite period > 0 and 0 <= min_pulse_width <= period".into(),
                    ));
                }
                Ok(self)
            }
            Self::Pwpf {
                gain_km,
                time_constant_tm,
                u_on,
                u_off,
            } => {
                if !gain_km.is_finite()
                    || gain_km <= 0.0
                    || !time_constant_tm.is_finite()
                    || time_constant_tm <= 0.0
                    || !u_on.is_finite()
                    || !u_off.is_finite()
                    || u_off < 0.0
                    || u_on <= u_off
                {
                    return Err(Error::Build(
                        "PWPF requires positive gain_km, time_constant_tm, and 0 <= u_off < u_on"
                            .into(),
                    ));
                }
                Ok(self)
            }
        }
    }
}

/// A thruster mounted on `body`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThrusterSpec {
    pub body: BodyHandle,
    /// Nozzle frame in the body frame: origin at the line-of-action point,
    /// `+X` pointing in the direction of applied thrust (opposite exhaust).
    pub frame: Transform,
    /// Full-throttle thrust magnitude (N).
    pub max_thrust: Real,
    /// Specific impulse `I_sp` (s). `None` gives unlimited propellant.
    pub specific_impulse: Option<Real>,
    /// Initial propellant reserve (kg); ignored when `specific_impulse` is `None`.
    pub propellant_mass: Real,
    /// When `true` and `specific_impulse` is `Some`, reduce the parent body's
    /// mass (and scale its inertia proportionally) as propellant is burned.
    pub deplete_body_mass: bool,
    /// Minimum impulse bit (N·s). When positive, opening the valve commits at
    /// least `min_impulse_bit / max_thrust` seconds of burn time.
    pub min_impulse_bit: Real,
    /// First-order thrust response time constant (s); `0.0` is instantaneous.
    pub response_time: Real,
}

impl ThrusterSpec {
    /// Convenience constructor for an ideal unlimited-propellant thruster.
    pub fn ideal(body: BodyHandle, frame: Transform, max_thrust: Real) -> Self {
        Self {
            body,
            frame,
            max_thrust,
            specific_impulse: None,
            propellant_mass: 0.0,
            deplete_body_mass: false,
            min_impulse_bit: 0.0,
            response_time: 0.0,
        }
    }
}

/// A thruster's state and averaged loads over the last step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ThrusterReading {
    /// Commanded throttle in `[0, 1]`.
    pub throttle: Real,
    /// Actual applied thrust magnitude (N), averaged over the last step.
    pub thrust: Real,
    /// Propellant mass flow rate (kg/s) over the last step.
    pub mass_flow: Real,
    /// Remaining propellant (kg), or `f64::INFINITY` when unlimited.
    pub propellant_remaining: Real,
    /// Cumulative impulse delivered since insertion (N·s).
    pub impulse_total: Real,
    /// World-space force applied to the body (N), averaged over the step.
    pub world_force: Vec3,
    /// World-space torque about the body's centre of mass (N·m), averaged over
    /// the step.
    pub world_torque: Vec3,
    /// `true` when firing was requested while propellant was exhausted.
    pub starved: bool,
    /// Discrete solenoid valve state at the end of the step (`true` = open).
    pub valve_open: bool,
    /// Cumulative discrete valve opening cycles (rising edges).
    pub valve_cycles: u64,
}

struct Thruster {
    spec: ThrusterSpec,
    /// Continuous throttle command `[0, 1]`.
    throttle: Real,
    /// Active timed pulse `(throttle, remaining_seconds)`.
    pulse: Option<(Real, Real)>,
    /// Previous continuous firing state (for MIB latching on valve open).
    was_firing: bool,
    /// Valve modulation mode (`Continuous`, `Pwm`, or `Pwpf`).
    modulation: ThrusterModulation,
    /// Internal PWM carrier phase in `[0, period)` (s).
    pwm_phase: Real,
    /// Latched on-time for the current PWM period (s).
    pwm_latched_on: Real,
    /// Internal PWPF first-order lag filter state `f`.
    pwpf_filter: Real,
    /// Current discrete valve state for PWM/PWPF modulation.
    valve_open: bool,
    /// Cumulative discrete valve rising-edge cycle count.
    valve_cycles: u64,
    /// Current chamber thrust magnitude (N) before propellant starvation clamp.
    chamber_thrust: Real,
    /// Effective thrust applied during the current step (N).
    step_thrust: Real,
    propellant_remaining: Real,
    impulse_total: Real,
    reading: ThrusterReading,
}

impl Thruster {
    fn has_propellant(&self) -> bool {
        self.spec.specific_impulse.is_none() || self.propellant_remaining > 0.0
    }

    fn is_active(&self) -> bool {
        self.throttle > 0.0
            || self.pulse.is_some_and(|(t, d)| t > 0.0 && d > 0.0)
            || self.valve_open
            || self.pwpf_filter.abs() > 1e-6
            || self.chamber_thrust > 1e-12
            || self.step_thrust > 0.0
    }
}

/// Thrusters of a rigid world.
#[derive(Default)]
pub struct Thrusters {
    next: u64,
    thrusters: BTreeMap<ThrusterId, Thruster>,
}

impl Thrusters {
    pub(super) fn is_empty(&self) -> bool {
        self.thrusters.is_empty()
    }

    pub fn insert(&mut self, scene: &RigidScene, spec: ThrusterSpec) -> Result<ThrusterId> {
        if scene.body(spec.body).is_none() {
            return Err(Error::Build("thruster body is missing".into()));
        }
        if !spec.frame.position.is_finite()
            || !spec.frame.rotation.is_finite()
            || (spec.frame.rotation.length_squared() - 1.0).abs() > 1e-4
        {
            return Err(Error::Build("thruster frame must be valid and finite".into()));
        }
        if !spec.max_thrust.is_finite() || spec.max_thrust < 0.0 {
            return Err(Error::Build(
                "thruster max_thrust must be finite and nonnegative".into(),
            ));
        }
        if let Some(isp) = spec.specific_impulse
            && (!isp.is_finite() || isp <= 0.0)
        {
            return Err(Error::Build(
                "thruster specific_impulse must be finite and positive".into(),
            ));
        }
        if !spec.propellant_mass.is_finite() || spec.propellant_mass < 0.0 {
            return Err(Error::Build(
                "thruster propellant_mass must be finite and nonnegative".into(),
            ));
        }
        if !spec.min_impulse_bit.is_finite() || spec.min_impulse_bit < 0.0 {
            return Err(Error::Build(
                "thruster min_impulse_bit must be finite and nonnegative".into(),
            ));
        }
        if !spec.response_time.is_finite() || spec.response_time < 0.0 {
            return Err(Error::Build(
                "thruster response_time must be finite and nonnegative".into(),
            ));
        }
        self.next += 1;
        let id = ThrusterId(self.next);
        let propellant_remaining = if spec.specific_impulse.is_some() {
            spec.propellant_mass
        } else {
            Real::INFINITY
        };
        self.thrusters.insert(
            id,
            Thruster {
                spec,
                throttle: 0.0,
                pulse: None,
                was_firing: false,
                modulation: ThrusterModulation::Continuous,
                pwm_phase: 0.0,
                pwm_latched_on: 0.0,
                pwpf_filter: 0.0,
                valve_open: false,
                valve_cycles: 0,
                chamber_thrust: 0.0,
                step_thrust: 0.0,
                propellant_remaining,
                impulse_total: 0.0,
                reading: ThrusterReading {
                    propellant_remaining,
                    ..ThrusterReading::default()
                },
            },
        );
        Ok(id)
    }

    pub fn remove(&mut self, id: ThrusterId) -> bool {
        self.thrusters.remove(&id).is_some()
    }

    /// Set continuous throttle in `[0, 1]`. Cancels any active timed pulse.
    pub fn set_throttle(&mut self, id: ThrusterId, throttle: Real) -> Result<()> {
        if !throttle.is_finite() {
            return Err(Error::Build("thruster throttle must be finite".into()));
        }
        let thruster = self
            .thrusters
            .get_mut(&id)
            .ok_or_else(|| Error::Build("no such thruster".into()))?;
        let clamped = throttle.clamp(0.0, 1.0);
        // When opening the valve on an MIB thruster, commit at least one
        // minimum impulse bit as a latched pulse if the caller later turns off
        // sooner.
        if matches!(thruster.modulation, ThrusterModulation::Continuous)
            && clamped > 0.0
            && !thruster.was_firing
            && thruster.spec.min_impulse_bit > 0.0
            && thruster.spec.max_thrust > 0.0
        {
            let min_duration =
                thruster.spec.min_impulse_bit / (thruster.spec.max_thrust * clamped);
            thruster.pulse = Some((clamped, min_duration));
        } else if clamped > 0.0 && matches!(thruster.modulation, ThrusterModulation::Continuous) {
            thruster.pulse = None;
        }
        thruster.throttle = clamped;
        thruster.was_firing = clamped > 0.0;
        Ok(())
    }

    /// Fire a timed pulse at `throttle` in `[0, 1]` for `duration` seconds
    /// (extended to satisfy `min_impulse_bit` when nonzero).
    pub fn fire_pulse(&mut self, id: ThrusterId, throttle: Real, duration: Real) -> Result<()> {
        if !throttle.is_finite() || !duration.is_finite() || duration < 0.0 {
            return Err(Error::Build(
                "thruster pulse throttle must be finite and duration nonnegative".into(),
            ));
        }
        let thruster = self
            .thrusters
            .get_mut(&id)
            .ok_or_else(|| Error::Build("no such thruster".into()))?;
        let clamped = throttle.clamp(0.0, 1.0);
        if clamped == 0.0 || (duration == 0.0 && thruster.spec.min_impulse_bit == 0.0) {
            thruster.pulse = None;
            return Ok(());
        }
        let min_duration = if thruster.spec.min_impulse_bit > 0.0 && thruster.spec.max_thrust > 0.0
        {
            thruster.spec.min_impulse_bit / (thruster.spec.max_thrust * clamped)
        } else {
            0.0
        };
        thruster.pulse = Some((clamped, duration.max(min_duration)));
        Ok(())
    }

    /// Configure valve modulation (`Continuous`, `Pwm`, or `Pwpf`) on a thruster.
    pub fn set_modulation(&mut self, id: ThrusterId, mode: ThrusterModulation) -> Result<()> {
        let mode = mode.validated()?;
        let thruster = self
            .thrusters
            .get_mut(&id)
            .ok_or_else(|| Error::Build("no such thruster".into()))?;
        thruster.modulation = mode;
        thruster.pwm_phase = 0.0;
        thruster.pwm_latched_on = 0.0;
        thruster.pwpf_filter = 0.0;
        thruster.valve_open = false;
        Ok(())
    }

    /// Replace the remaining propellant reserve (kg) for a finite-propellant
    /// thruster.
    pub fn refuel(&mut self, id: ThrusterId, propellant_mass: Real) -> Result<()> {
        if !propellant_mass.is_finite() || propellant_mass < 0.0 {
            return Err(Error::Build(
                "thruster propellant_mass must be finite and nonnegative".into(),
            ));
        }
        let thruster = self
            .thrusters
            .get_mut(&id)
            .ok_or_else(|| Error::Build("no such thruster".into()))?;
        if thruster.spec.specific_impulse.is_some() {
            thruster.propellant_remaining = propellant_mass;
            thruster.reading.propellant_remaining = propellant_mass;
        }
        Ok(())
    }

    pub fn reading(&self, id: ThrusterId) -> Option<ThrusterReading> {
        self.thrusters.get(&id).map(|t| t.reading)
    }

    pub fn spec(&self, id: ThrusterId) -> Option<&ThrusterSpec> {
        self.thrusters.get(&id).map(|t| &t.spec)
    }

    pub fn has_propellant(&self, id: ThrusterId) -> bool {
        self.thrusters
            .get(&id)
            .is_some_and(Thruster::has_propellant)
    }

    /// Whether any thruster is firing, pulsing, or ramping down.
    pub fn active(&self) -> bool {
        self.thrusters.values().any(Thruster::is_active)
    }

    /// Yields `(body, body_frame_nozzle, applied_thrust_n)` for every actively
    /// firing thruster in the current step (used by plume impingement models).
    pub(crate) fn active_nozzles(&self) -> impl Iterator<Item = (BodyHandle, Transform, Real)> + '_ {
        self.thrusters.values().filter_map(|t| {
            (t.step_thrust > 0.0).then_some((t.spec.body, t.spec.frame, t.step_thrust))
        })
    }

    pub(crate) fn before_step(&mut self, scene: &mut RigidScene, dt: Real) -> Result<()> {
        let mut mass_burned: BTreeMap<BodyHandle, Real> = BTreeMap::new();
        for thruster in self.thrusters.values_mut() {
            // 1. Modulate continuous throttle command through PWM or PWPF when enabled.
            let prev_valve = thruster.valve_open;
            let mut eff_throttle = match thruster.modulation {
                ThrusterModulation::Continuous => {
                    thruster.valve_open = thruster.throttle > 0.0;
                    if thruster.valve_open && !prev_valve {
                        thruster.valve_cycles += 1;
                    }
                    thruster.throttle
                }
                ThrusterModulation::Pwm {
                    period,
                    min_pulse_width,
                } => {
                    let sub_n = ((dt / 0.0005).ceil() as usize).clamp(1, 64);
                    let h = dt / sub_n as Real;
                    let mut on_time = 0.0;
                    for _ in 0..sub_n {
                        if thruster.pwm_phase < 1e-12 || thruster.pwm_phase >= period - 1e-12 {
                            thruster.pwm_phase = 0.0;
                            let desired_on = thruster.throttle * period;
                            thruster.pwm_latched_on = if desired_on >= min_pulse_width {
                                desired_on.min(period)
                            } else {
                                0.0
                            };
                        }
                        let open = thruster.pwm_phase < thruster.pwm_latched_on;
                        if open && !thruster.valve_open {
                            thruster.valve_cycles += 1;
                        }
                        thruster.valve_open = open;
                        if open {
                            on_time += h;
                        }
                        thruster.pwm_phase += h;
                        if thruster.pwm_phase >= period - 1e-12 {
                            thruster.pwm_phase = (thruster.pwm_phase - period).max(0.0);
                        }
                    }
                    (on_time / dt).clamp(0.0, 1.0)
                }
                ThrusterModulation::Pwpf {
                    gain_km,
                    time_constant_tm,
                    u_on,
                    u_off,
                } => {
                    let sub_n = ((dt / 0.0005).ceil() as usize).clamp(1, 64);
                    let h = dt / sub_n as Real;
                    let alpha = (-h / time_constant_tm).exp();
                    let mut on_time = 0.0;
                    for _ in 0..sub_n {
                        let u_fb = if thruster.valve_open { 1.0 } else { 0.0 };
                        let target_f = gain_km * (thruster.throttle - u_fb);
                        thruster.pwpf_filter =
                            target_f + (thruster.pwpf_filter - target_f) * alpha;
                        let next_open = if thruster.pwpf_filter >= u_on {
                            true
                        } else if thruster.pwpf_filter <= u_off {
                            false
                        } else {
                            thruster.valve_open
                        };
                        if next_open && !thruster.valve_open {
                            thruster.valve_cycles += 1;
                        }
                        thruster.valve_open = next_open;
                        if next_open {
                            on_time += h;
                        }
                    }
                    (on_time / dt).clamp(0.0, 1.0)
                }
            };
            // Combine modulated throttle and any latched MIB / timed pulse.
            if let Some((pulse_throttle, remaining)) = thruster.pulse {
                if remaining > 0.0 && pulse_throttle > 0.0 {
                    let burn = remaining.min(dt);
                    let pulse_eff = pulse_throttle * (burn / dt);
                    if pulse_eff > eff_throttle {
                        eff_throttle = pulse_eff;
                        thruster.valve_open = true;
                    }
                    let next_remaining = remaining - burn;
                    thruster.pulse = if next_remaining > 1e-12 {
                        Some((pulse_throttle, next_remaining))
                    } else {
                        None
                    };
                } else {
                    thruster.pulse = None;
                }
            }
            let target_thrust = thruster.spec.max_thrust * eff_throttle;
            let raw_thrust = if thruster.spec.response_time > 0.0 {
                let alpha = (-dt / thruster.spec.response_time).exp();
                let next = target_thrust + (thruster.chamber_thrust - target_thrust) * alpha;
                // Average thrust over the exponential step for accurate impulse.
                let mean = target_thrust
                    + (thruster.chamber_thrust - target_thrust)
                        * ((1.0 - alpha) * thruster.spec.response_time / dt);
                thruster.chamber_thrust = if next < 1e-12 && target_thrust == 0.0 {
                    0.0
                } else {
                    next
                };
                mean.max(0.0)
            } else {
                thruster.chamber_thrust = target_thrust;
                target_thrust
            };

            let (applied_thrust, mass_flow, starved) =
                if let Some(isp) = thruster.spec.specific_impulse {
                    let ve = isp * GRAVITY;
                    let wanted_dm = (raw_thrust / ve) * dt;
                    if thruster.propellant_remaining <= 0.0 {
                        thruster.chamber_thrust = 0.0;
                        (0.0, 0.0, eff_throttle > 0.0 || raw_thrust > 0.0)
                    } else if wanted_dm > thruster.propellant_remaining {
                        let dm = thruster.propellant_remaining;
                        let frac = if wanted_dm > 0.0 {
                            dm / wanted_dm
                        } else {
                            0.0
                        };
                        thruster.propellant_remaining = 0.0;
                        thruster.chamber_thrust = 0.0;
                        if thruster.spec.deplete_body_mass && dm > 0.0 {
                            *mass_burned.entry(thruster.spec.body).or_insert(0.0) += dm;
                        }
                        (raw_thrust * frac, dm / dt, true)
                    } else {
                        thruster.propellant_remaining -= wanted_dm;
                        if thruster.spec.deplete_body_mass && wanted_dm > 0.0 {
                            *mass_burned.entry(thruster.spec.body).or_insert(0.0) += wanted_dm;
                        }
                        (raw_thrust, wanted_dm / dt, false)
                    }
                } else {
                    (raw_thrust, 0.0, false)
                };

            thruster.step_thrust = applied_thrust;
            thruster.impulse_total += applied_thrust * dt;
            thruster.reading = ThrusterReading {
                throttle: eff_throttle,
                thrust: applied_thrust,
                mass_flow,
                propellant_remaining: thruster.propellant_remaining,
                impulse_total: thruster.impulse_total,
                world_force: Vec3::ZERO,
                world_torque: Vec3::ZERO,
                starved,
                valve_open: thruster.valve_open && !starved,
                valve_cycles: thruster.valve_cycles,
            };
        }

        // Apply batched per-body propellant mass depletion once per body.
        for (body, dm) in mass_burned {
            let Some(desc) = scene.body(body) else {
                continue;
            };
            if !desc.enabled || desc.kind != BodyKind::Dynamic || dm <= 0.0 {
                continue;
            }
            let current = scene.body_mass_properties(body)?;
            if current.mass <= MIN_DRY_MASS {
                continue;
            }
            let new_mass = (current.mass - dm).max(MIN_DRY_MASS);
            let scale = new_mass / current.mass;
            let updated = MassProperties {
                mass: new_mass,
                local_com: current.local_com,
                inertia: current.inertia * scale,
            };
            scene.set_body_mass_properties_with_wake(body, updated, false)?;
        }
        Ok(())
    }

    pub(crate) fn apply(
        &mut self,
        scene: &RigidScene,
        state: &State,
        wrenches: &mut [SpatialVector],
        weight: Real,
    ) -> Result<()> {
        if self.thrusters.is_empty() {
            return Ok(());
        }
        let poses = state.body_q.host()?;
        let centers = scene.model().body_com.host()?;
        for thruster in self.thrusters.values_mut() {
            let thrust = thruster.step_thrust;
            if thrust <= 0.0 {
                continue;
            }
            let (Some(body), Some(desc)) = (
                scene.body_index(thruster.spec.body),
                scene.body(thruster.spec.body),
            ) else {
                continue;
            };
            if !desc.enabled || desc.kind != BodyKind::Dynamic {
                continue;
            }
            let i = body.index();
            let nozzle = poses[i].compose(&thruster.spec.frame);
            let axis = nozzle.rotation * Vec3::X;
            let com = poses[i].transform_point(centers[i]);
            let arm = nozzle.position - com;
            let force = axis * thrust;
            let torque = arm.cross(force);
            wrenches[i].linear += force;
            wrenches[i].angular += torque;
            thruster.reading.world_force += force * weight;
            thruster.reading.world_torque += torque * weight;
        }
        Ok(())
    }
}
