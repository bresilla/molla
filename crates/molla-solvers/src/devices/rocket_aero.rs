//! 0°–180° full-regime launch-vehicle aerodynamics, actuated body flaps,
//! titanium grid fins, wind shear / turbulence, and Supersonic Retropropulsion
//! (SRP) aerodynamic drag-loss modeling.
//!
//! Grounded in classic NASA/AIAA semi-empirical rocket aerodynamics:
//! - **Allen–Perkins / Jorgensen (NASA TN D-6996, TR R-474) & Hoerner crossflow**:
//!   continuous axial & normal force + Mach-dependent Center of Pressure from `0°`
//!   vertical ascent through `90°` Starship belly-flop skydiving to `180°` tail-first
//!   Super Heavy booster reentry!
//! - **Actuated Control Surfaces**:
//!   - [`AeroSurfaceKind::BodyFlap`]: high-AoA flat-plate / crossflow body flaps
//!     (Starship's 2 forward + 2 aft elonerons) with real aerodynamic hinge torque.
//!   - [`AeroSurfaceKind::GridFin`]: lattice grid fins (Super Heavy's 4 titanium
//!     fins) capturing subsonic effectiveness, transonic choking drop around
//!     `M ∈ [0.9, 1.4]`, and high supersonic effectiveness.
//! - **Supersonic Retropropulsion (SRP) Drag Loss** (NASA / SpaceX Falcon 9 flight
//!   studies): reduces body axial pressure drag as a function of thrust coefficient
//!   `C_T = T / (q S_ref)` when engines fire forward into oncoming supersonic flow.

use std::collections::BTreeMap;

use molla_core::{Error, Result};
use molla_math::{Real, SpatialVector, Vec3};
use molla_sim::State;
use molla_sim::runtime::{BodyHandle, BodyKind, RigidScene};

use super::rocket::RocketPropulsion;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AeroBodyId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AeroSurfaceId(pub u64);

/// Atmospheric wind shear + deterministic multi-sine Dryden-like turbulence model.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WindField {
    /// Mean ground-level horizontal wind vector (m/s, world frame).
    pub surface_wind: Vec3,
    /// Jet-stream peak wind vector near `jet_altitude_m` (m/s, world frame).
    pub jet_stream_wind: Vec3,
    /// Peak jet-stream altitude (m; typically ~11,000 m).
    pub jet_altitude_m: Real,
    /// Jet-stream Gaussian vertical thickness (m; typically ~3,500 m).
    pub jet_sigma_m: Real,
    /// Gust / turbulence RMS amplitude (m/s).
    pub gust_amplitude_mps: Real,
}

impl WindField {
    pub fn calm() -> Self {
        Self::default()
    }

    pub fn boca_chica_moderate() -> Self {
        Self {
            surface_wind: Vec3::new(6.0, 0.0, -3.5),
            jet_stream_wind: Vec3::new(28.0, 0.0, 10.0),
            jet_altitude_m: 11_500.0,
            jet_sigma_m: 3_500.0,
            gust_amplitude_mps: 2.5,
        }
    }

    /// Mean atmospheric wind profile (surface wind + Gaussian altitude jet stream).
    /// Continuous stochastic turbulence (`gust_amplitude_mps`) is driven by the stateful
    /// MIL-F-8785C Dryden filter inside [`RocketAero`].
    pub fn wind_at(&self, _pos_world: Vec3, altitude_m: Real, _time_s: Real) -> Vec3 {
        let mut w = self.surface_wind;
        if self.jet_sigma_m > 1.0 {
            let z = (altitude_m - self.jet_altitude_m) / self.jet_sigma_m;
            w += self.jet_stream_wind * (-0.5 * z * z).exp();
        }
        w
    }
}

/// Cylindrical / ogive launch-vehicle body aerodynamic specification (0°–180° AoA).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AeroBodySpec {
    pub body: BodyHandle,
    /// Unit longitudinal axis pointing from aft toward nose in body frame (e.g. `+Y`).
    pub axis_local: Vec3,
    /// Vehicle outer diameter `D` (m; `9.0 m` for Starship / Super Heavy).
    pub diameter_m: Real,
    /// Total aerodynamic length `L` (m; `50.0 m` Ship, `71.0 m` Booster).
    pub length_m: Real,
    /// Subsonic axial drag coefficient nose-first (`α = 0°`, ~0.28–0.38).
    pub cd_axial_nose: Real,
    /// Subsonic axial drag coefficient tail-first (`α = 180°`, ~0.85–1.05).
    pub cd_axial_tail: Real,
    /// Subsonic cylinder crossflow drag `C_{d,n}` at `α = 90°` (~1.15–1.25).
    pub cd_crossflow_subsonic: Real,
    /// Supersonic/hypersonic cylinder crossflow drag `C_{d,n}` at `M_n > 1.5` (~1.45–1.60).
    pub cd_crossflow_supersonic: Real,
    /// Geometric Center of Pressure axial coordinate (m along `axis_local`) in `90°` crossflow.
    pub cp_crossflow_local: Real,
    /// Nose-first CP offset relative to `cp_crossflow_local` (m; positive = forward toward nose
    /// during `α = 0°` ascent -> real finless rocket aerodynamic instability, shifting aft at `α = 180°`).
    pub cp_axial_shift_m: Real,
    /// Rotational aerodynamic strip damping factor (dimensionless, ~0.8–1.2).
    pub rot_damping_scale: Real,
    /// Supersonic Retropropulsion (SRP) drag-loss shielding gain `k_srp` (~0.6–1.2; `0.0` disables).
    pub srp_drag_loss_gain: Real,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AeroBodyReading {
    pub dynamic_pressure_pa: Real,
    pub mach: Real,
    pub aoa_rad: Real,
    pub srp_drag_factor: Real,
    /// Rarefied Knudsen number `Kn = λ / D` (continuum `< 0.01`, transition `0.01..10`, free-molecular `> 10`).
    pub knudsen_number: Real,
    /// Forward aerodynamic Center-of-Pressure shift from Plume-Induced Flow Separation (PIFS, m).
    pub pifs_cp_shift_m: Real,
    /// Transonic Max-Q buffet torque magnitude (N·m).
    pub buffet_torque_nm: Real,
    pub world_force: Vec3,
    pub world_torque: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AeroSurfaceKind {
    /// Starship-style hinged body flap (flat-plate crossflow up to 90° deflection).
    BodyFlap,
    /// Super-Heavy-style titanium lattice grid fin (transonic choking dip around M ≈ 1.0–1.35).
    GridFin,
}

/// Actuated aerodynamic control surface (body flap or titanium grid fin) mounted on `body`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AeroSurfaceSpec {
    pub body: BodyHandle,
    pub kind: AeroSurfaceKind,
    /// Aerodynamic center of the surface in the parent body frame (m).
    pub center_local: Vec3,
    /// Unit hinge rotation axis in body frame (deflection `δ` rotates `normal_local` about `hinge_axis_local`).
    pub hinge_axis_local: Vec3,
    /// Unit surface normal at `δ = 0` in body frame.
    pub normal_local: Vec3,
    /// Effective planform area `S` (m²).
    pub area_m2: Real,
    /// Distance from hinge line to surface aerodynamic center (m, for hinge torque calculation).
    pub hinge_moment_arm_m: Real,
    /// Minimum & maximum deflection angles (rad).
    pub min_deflection_rad: Real,
    pub max_deflection_rad: Real,
    /// Maximum electric actuator deflection rate (rad/s).
    pub max_rate_rad_s: Real,
    /// Maximum actuator stall hinge torque (N·m); aerodynamic hinge loads above this limit stall/backdrive.
    pub max_hinge_torque_nm: Real,
    /// Peak flat-plate normal force coefficient `C_N,max` (~1.8–2.0 for flaps, ~2.2 for grid fins).
    pub cn_max: Real,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AeroSurfaceReading {
    pub deflection_rad: Real,
    pub local_aoa_rad: Real,
    pub hinge_torque_nm: Real,
    pub world_force: Vec3,
    pub world_torque: Vec3,
}

struct AeroBodyState {
    spec: AeroBodySpec,
    reading: AeroBodyReading,
}

struct AeroSurfaceState {
    spec: AeroSurfaceSpec,
    cmd_deflection_rad: Real,
    deflection_rad: Real,
    reading: AeroSurfaceReading,
}

/// Stateful MIL-F-8785C / MIL-STD-1797 Dryden continuous stochastic turbulence filter
/// (1st-order longitudinal + 2nd-order lateral/vertical shaping filters driven by deterministic white noise).
#[derive(Clone, Copy, Debug)]
struct DrydenFilter {
    rng: u64,
    u_g: Real,
    v_g: Real,
    v_dot: Real,
    w_g: Real,
    w_dot: Real,
    buffet_x: Real,
    buffet_z: Real,
}

impl Default for DrydenFilter {
    fn default() -> Self {
        Self {
            rng: 0xD27D_E410_9E37_79B9,
            u_g: 0.0,
            v_g: 0.0,
            v_dot: 0.0,
            w_g: 0.0,
            w_dot: 0.0,
            buffet_x: 0.0,
            buffet_z: 0.0,
        }
    }
}

impl DrydenFilter {
    fn next_normal(&mut self) -> Real {
        // Fast Box-Muller-free sum of 3 uniform [-1, 1] draws (variance = 1.0, bounded ±sqrt(3) -> 3-sigma safe!).
        let mut s = 0.0;
        for _ in 0..3 {
            let mut x = self.rng;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.rng = x;
            let u = (x as f64) * (2.0 / (u64::MAX as f64)) - 1.0;
            s += u;
        }
        s
    }

    fn step(&mut self, dt: Real, sigma_mps: Real) {
        if !(dt > 0.0) {
            return;
        }
        // Standard MIL-F-8785C characteristic turbulence frequencies (V/L ≈ 0.35..0.65 rad/s):
        let omega_u = 0.45;
        let omega_vw = 0.65;
        let zeta = 0.72;
        if sigma_mps <= 1e-6 {
            let decay = (-dt * 2.0).exp();
            self.u_g *= decay;
            self.v_g *= decay;
            self.v_dot *= decay;
            self.w_g *= decay;
            self.w_dot *= decay;
        } else {
            let sq_dt = dt.sqrt();
            // 1st-order Dryden longitudinal gust u_g
            let nu = self.next_normal();
            self.u_g += (-omega_u * self.u_g) * dt + sigma_mps * (2.0 * omega_u).sqrt() * nu * sq_dt;
            // 2nd-order Dryden lateral gust v_g (horizontal Z) & vertical gust w_g (Y, 0.5x amplitude)
            let nv = self.next_normal();
            let v_acc = -2.0 * zeta * omega_vw * self.v_dot - omega_vw * omega_vw * self.v_g
                + sigma_mps * (2.0 * omega_vw.powi(3)).sqrt() * nv / sq_dt.max(1e-3);
            self.v_dot += v_acc * dt;
            self.v_g += self.v_dot * dt;
            let nw = self.next_normal();
            let w_acc = -2.0 * zeta * omega_vw * self.w_dot - omega_vw * omega_vw * self.w_g
                + (0.45 * sigma_mps) * (2.0 * omega_vw.powi(3)).sqrt() * nw / sq_dt.max(1e-3);
            self.w_dot += w_acc * dt;
            self.w_g += self.w_dot * dt;
            let lim = 3.2 * sigma_mps;
            self.u_g = self.u_g.clamp(-lim, lim);
            self.v_g = self.v_g.clamp(-lim, lim);
            self.w_g = self.w_g.clamp(-0.5 * lim, 0.5 * lim);
        }
        // Band-limited transonic buffet stochastic noise (~8 Hz cutoff, unit RMS):
        let wb = 50.0;
        let a = (-wb * dt).exp();
        let gain = (1.0 - a * a).max(0.0).sqrt();
        self.buffet_x = self.buffet_x * a + self.next_normal() * gain;
        self.buffet_z = self.buffet_z * a + self.next_normal() * gain;
    }

    fn gust_world(&self) -> Vec3 {
        Vec3::new(self.u_g, self.w_g, self.v_g)
    }
}

/// Full-regime launch-vehicle aerodynamics device (`0°–180°` AoA + flaps + grid fins + SRP
/// + MIL-F-8785C Dryden turbulence + PIFS + transonic buffet + Knudsen rarefied bridging).
#[derive(Default)]
pub struct RocketAero {
    next_body: u64,
    next_surface: u64,
    time_s: Real,
    wind: WindField,
    dryden: DrydenFilter,
    bodies: BTreeMap<AeroBodyId, AeroBodyState>,
    surfaces: BTreeMap<AeroSurfaceId, AeroSurfaceState>,
}

impl RocketAero {
    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty() && self.surfaces.is_empty()
    }

    pub fn active(&self) -> bool {
        !self.bodies.is_empty() || !self.surfaces.is_empty()
    }

    pub fn set_wind(&mut self, wind: WindField) {
        self.wind = wind;
    }

    pub fn wind(&self) -> WindField {
        self.wind
    }

    pub fn insert_body(&mut self, scene: &RigidScene, mut spec: AeroBodySpec) -> Result<AeroBodyId> {
        if scene.body(spec.body).is_none() {
            return Err(Error::Build("aero body is missing".into()));
        }
        if !spec.axis_local.is_finite()
            || spec.axis_local.length_squared() < 1e-12
            || !spec.diameter_m.is_finite()
            || spec.diameter_m <= 0.0
            || !spec.length_m.is_finite()
            || spec.length_m <= 0.0
        {
            return Err(Error::Build("invalid rocket aero body specification".into()));
        }
        spec.axis_local = spec.axis_local.normalize();
        self.next_body += 1;
        let id = AeroBodyId(self.next_body);
        self.bodies.insert(
            id,
            AeroBodyState {
                spec,
                reading: AeroBodyReading::default(),
            },
        );
        Ok(id)
    }

    pub fn insert_surface(
        &mut self,
        scene: &RigidScene,
        mut spec: AeroSurfaceSpec,
    ) -> Result<AeroSurfaceId> {
        if scene.body(spec.body).is_none() {
            return Err(Error::Build("aero surface body is missing".into()));
        }
        if !spec.center_local.is_finite()
            || !spec.hinge_axis_local.is_finite()
            || spec.hinge_axis_local.length_squared() < 1e-12
            || !spec.normal_local.is_finite()
            || spec.normal_local.length_squared() < 1e-12
            || !spec.area_m2.is_finite()
            || spec.area_m2 <= 0.0
            || spec.min_deflection_rad > spec.max_deflection_rad
        {
            return Err(Error::Build("invalid aero surface specification".into()));
        }
        spec.hinge_axis_local = spec.hinge_axis_local.normalize();
        spec.normal_local = spec.normal_local.normalize();
        let init_d = 0.0_f64.clamp(spec.min_deflection_rad, spec.max_deflection_rad);
        self.next_surface += 1;
        let id = AeroSurfaceId(self.next_surface);
        self.surfaces.insert(
            id,
            AeroSurfaceState {
                spec,
                cmd_deflection_rad: init_d,
                deflection_rad: init_d,
                reading: AeroSurfaceReading {
                    deflection_rad: init_d,
                    ..AeroSurfaceReading::default()
                },
            },
        );
        Ok(id)
    }

    pub fn command_surface(&mut self, id: AeroSurfaceId, deflection_rad: Real) -> Result<()> {
        if !deflection_rad.is_finite() {
            return Err(Error::Build("surface deflection must be finite".into()));
        }
        let s = self
            .surfaces
            .get_mut(&id)
            .ok_or_else(|| Error::Build("no such aero surface".into()))?;
        s.cmd_deflection_rad = deflection_rad.clamp(s.spec.min_deflection_rad, s.spec.max_deflection_rad);
        Ok(())
    }

    /// Sets both the commanded and instantaneous surface deflection (for scenario initialization).
    pub fn set_surface_deflection(&mut self, id: AeroSurfaceId, deflection_rad: Real) -> Result<()> {
        if !deflection_rad.is_finite() {
            return Err(Error::Build("surface deflection must be finite".into()));
        }
        let s = self
            .surfaces
            .get_mut(&id)
            .ok_or_else(|| Error::Build("no such aero surface".into()))?;
        let d = deflection_rad.clamp(s.spec.min_deflection_rad, s.spec.max_deflection_rad);
        s.cmd_deflection_rad = d;
        s.deflection_rad = d;
        s.reading.deflection_rad = d;
        Ok(())
    }

    pub fn body_reading(&self, id: AeroBodyId) -> Option<AeroBodyReading> {
        self.bodies.get(&id).map(|b| b.reading)
    }

    pub fn surface_reading(&self, id: AeroSurfaceId) -> Option<AeroSurfaceReading> {
        self.surfaces.get(&id).map(|s| s.reading)
    }

    /// Returns `(body_aero_force_world, sum_surface_force_world, mean_surface_y_local)` for
    /// aeroelastic fuselage bending excitation in [`super::RocketStructure`].
    pub fn body_excitation_forces(&self, body: BodyHandle) -> (Vec3, Vec3, Real) {
        let mut f_body = Vec3::ZERO;
        for ab in self.bodies.values() {
            if ab.spec.body == body {
                f_body += ab.reading.world_force;
            }
        }
        let mut f_surf = Vec3::ZERO;
        let mut w_sum = 0.0;
        let mut y_sum = 0.0;
        for s in self.surfaces.values() {
            if s.spec.body == body {
                let mag = s.reading.world_force.length();
                f_surf += s.reading.world_force;
                w_sum += mag;
                y_sum += mag * s.spec.center_local.y;
            }
        }
        let y_avg = if w_sum > 1.0 { y_sum / w_sum } else { 0.0 };
        (f_body, f_surf, y_avg)
    }

    pub(crate) fn before_step(&mut self, dt: Real) {
        self.time_s += dt.max(0.0);
        self.dryden.step(dt.max(0.0), self.wind.gust_amplitude_mps);
        for b in self.bodies.values_mut() {
            b.reading.world_force = Vec3::ZERO;
            b.reading.world_torque = Vec3::ZERO;
            b.reading.buffet_torque_nm = 0.0;
        }
        for s in self.surfaces.values_mut() {
            // Rate-limited electric flap / grid-fin actuator advance (respecting stall torque).
            let step_lim = s.spec.max_rate_rad_s * dt.max(0.0);
            let mut delta = (s.cmd_deflection_rad - s.deflection_rad).clamp(-step_lim, step_lim);
            if s.spec.max_hinge_torque_nm > 0.0
                && s.reading.hinge_torque_nm.abs() > s.spec.max_hinge_torque_nm
                && delta * s.reading.hinge_torque_nm < 0.0
            {
                // Actuator stalled against opposing hinge aerodynamic load!
                delta *= 0.15;
            }
            s.deflection_rad = (s.deflection_rad + delta)
                .clamp(s.spec.min_deflection_rad, s.spec.max_deflection_rad);
            s.reading.deflection_rad = s.deflection_rad;
            s.reading.world_force = Vec3::ZERO;
            s.reading.world_torque = Vec3::ZERO;
            s.reading.hinge_torque_nm = 0.0;
        }
    }

    pub(crate) fn apply(
        &mut self,
        scene: &RigidScene,
        state: &State,
        rockets: &RocketPropulsion,
        wrenches: &mut [SpatialVector],
        weight: Real,
    ) -> Result<()> {
        if self.is_empty() {
            return Ok(());
        }
        let poses = state.body_q.host()?;
        let vels = state.body_qd.host()?;
        let coms = scene.model().body_com.host()?;

        // 1. Cylindrical Launch-Vehicle Bodies (0°–180° Jorgensen/Hoerner + Transonic + SRP).
        for ab in self.bodies.values_mut() {
            let (Some(bi), Some(desc)) = (scene.body_index(ab.spec.body), scene.body(ab.spec.body))
            else {
                continue;
            };
            if !desc.enabled || desc.kind != BodyKind::Dynamic {
                continue;
            }
            let i = bi.index();
            let bp = poses[i];
            let bv = vels[i];
            let com_w = bp.transform_point(coms[i]);
            let alt_m = rockets.altitude_at(com_w);
            let atmo = rockets.atmosphere_at(com_w);
            if atmo.density_kg_m3 < 1e-10 {
                ab.reading.dynamic_pressure_pa = 0.0;
                ab.reading.mach = 0.0;
                ab.reading.srp_drag_factor = 1.0;
                ab.reading.knudsen_number = 0.0;
                ab.reading.pifs_cp_shift_m = 0.0;
                continue;
            }
            let mut wind_w =
                rockets.corotating_atmosphere_velocity(com_w) + self.wind.wind_at(com_w, alt_m, self.time_s);
            // Add stateful MIL-F-8785C Dryden stochastic turbulence (fades cleanly above 25 km)!
            let turb_alt_scale = (-alt_m / 18_000.0).exp();
            wind_w += self.dryden.gust_world() * turb_alt_scale;
            let v_rel_w = bv.linear - wind_w;
            let speed = v_rel_w.length();
            if speed < 1e-3 {
                ab.reading.dynamic_pressure_pa = 0.0;
                ab.reading.mach = 0.0;
                ab.reading.srp_drag_factor = 1.0;
                ab.reading.knudsen_number = 0.0;
                ab.reading.pifs_cp_shift_m = 0.0;
                continue;
            }
            let q = 0.5 * atmo.density_kg_m3 * speed * speed;
            let mach = speed / atmo.speed_of_sound_mps.max(100.0);
            let axis_w = (bp.rotation * ab.spec.axis_local).normalize();
            let v_hat = v_rel_w / speed;
            let cos_a = axis_w.dot(v_hat).clamp(-1.0, 1.0);
            let aoa = cos_a.acos(); // 0 = nose-first, π/2 = belly-flop, π = tail-first!
            let sin_a = aoa.sin();
            let m_cross = mach * sin_a;

            // Rarefied-to-Continuum Knudsen Bridging (`Kn = λ / D`, `λ = k_λ / ρ` for Earth air or Mars CO₂)!
            let mean_free_path =
                rockets.atmosphere().mean_free_path_const() / atmo.density_kg_m3.max(1e-16);
            let kn = mean_free_path / ab.spec.diameter_m.max(0.1);
            let fm_blend = if kn > 1e-3 {
                let k12 = kn.powf(1.2);
                k12 / (0.35 + k12)
            } else {
                0.0
            };

            // Transonic wave-drag bump peaking near M = 1.08.
            let transonic_bump = 0.36 * (-((mach - 1.08) / 0.24).powi(2)).exp();
            let cd_axial_cont = if cos_a >= 0.0 {
                ab.spec.cd_axial_nose + transonic_bump
            } else {
                ab.spec.cd_axial_tail + transonic_bump * 0.75
            };
            let cd_axial_base = cd_axial_cont * (1.0 - fm_blend) + 2.2 * fm_blend;
            let s_ref = 0.25 * std::f64::consts::PI * ab.spec.diameter_m * ab.spec.diameter_m;
            let s_plan = ab.spec.diameter_m * ab.spec.length_m;

            // Supersonic Retropropulsion (SRP) aerodynamic axial drag-loss factor!
            let mut srp_factor = 1.0;
            if ab.spec.srp_drag_loss_gain > 0.0 && mach > 0.75 {
                let t_vec = rockets.body_total_thrust_world(ab.spec.body);
                let retro_thrust = (-t_vec.dot(v_hat)).max(0.0);
                if retro_thrust > 100.0 && q * s_ref > 1.0 {
                    let ct = retro_thrust / (q * s_ref);
                    let mach_ramp = ((mach - 0.75) / 0.50).clamp(0.0, 1.0);
                    srp_factor = 1.0 / (1.0 + ab.spec.srp_drag_loss_gain * mach_ramp * ct);
                }
            }

            // Plume-Induced Flow Separation (PIFS) during high-altitude powered ascent!
            // When the multi-engine exhaust plume underexpands (`P_e / P_a > 1.8`) in transonic/supersonic
            // flight, the swollen plume drives a separation shock upstream along the aft skirt, shifting CP
            // forward (`+pifs_cp_shift_m`) and reducing aft base drag slightly!
            let p_ratio = rockets.body_mean_pressure_ratio(ab.spec.body);
            let mut pifs_cp_shift_m = 0.0;
            let mut pifs_drag_scale = 1.0;
            if p_ratio > 1.8 && mach > 0.7 && cos_a > 0.2 {
                let mach_f = ((mach - 0.7) / 0.7).clamp(0.0, 1.0);
                let expand_f = (0.30 * (p_ratio / 1.8).ln()).tanh();
                let pifs_frac = mach_f * expand_f * (1.0 - fm_blend);
                pifs_cp_shift_m = ab.spec.length_m * 0.14 * pifs_frac;
                pifs_drag_scale = 1.0 - 0.20 * pifs_frac;
            }

            // Axial aerodynamic force along body axis (opposes axial velocity component).
            let f_axial_mag = q * s_ref * cd_axial_base * cos_a * cos_a.abs() * srp_factor * pifs_drag_scale;
            let f_axial_w = -axis_w * f_axial_mag;

            // Hoerner / Jorgensen crossflow normal force (continuous 0°..180° + Knudsen free-molecular bridge!).
            let cross_blend = ((m_cross - 0.45) / 0.75).clamp(0.0, 1.0);
            let cd_n_cont = ab.spec.cd_crossflow_subsonic * (1.0 - cross_blend)
                + ab.spec.cd_crossflow_supersonic * cross_blend;
            let cd_n = cd_n_cont * (1.0 - fm_blend) + 2.2 * fm_blend;
            let v_perp = v_rel_w - axis_w * v_rel_w.dot(axis_w);
            let v_perp_mag = v_perp.length();
            let f_normal_w = if v_perp_mag > 1e-6 {
                let n_dir = -v_perp / v_perp_mag; // Opposes crossflow velocity
                // Exact Munk-Jones potential slender-body term (`dC_N/dα = 2` -> `1.0 * sin(2α)`) + viscous crossflow:
                let f_pot = q * s_ref * 1.0 * (2.0 * aoa).sin().abs() * (0.5 * aoa).cos().abs() * (1.0 - fm_blend);
                let f_visc = q * s_plan * 0.88 * cd_n * sin_a * sin_a;
                n_dir * (f_pot + f_visc)
            } else {
                Vec3::ZERO
            };

            // Transonic Max-Q Buffet forcing (`M ∈ [0.80, 1.22]`, peaking at `M ≈ 1.02`)!
            let buffet_env = (-((mach - 1.02) / 0.14).powi(2)).exp() * (1.0 - fm_blend);
            let mut f_buffet_w = Vec3::ZERO;
            let mut tau_buffet_w = Vec3::ZERO;
            if buffet_env > 0.02 && q > 500.0 {
                let e1 = if axis_w.x.abs() < 0.9 {
                    (Vec3::X - axis_w * axis_w.x).normalize()
                } else {
                    (Vec3::Y - axis_w * axis_w.y).normalize()
                };
                let e2 = axis_w.cross(e1).normalize();
                let f_rms = 0.045 * q * s_ref * buffet_env;
                f_buffet_w = (e1 * self.dryden.buffet_x + e2 * self.dryden.buffet_z) * f_rms;
                // Applied near the forward interstage / ogive shoulder (`+0.32 L`)!
                let buffet_pt_w = bp.transform_point(ab.spec.axis_local * (0.32 * ab.spec.length_m));
                tau_buffet_w = (buffet_pt_w - com_w).cross(f_buffet_w);
            }

            let f_total = f_axial_w + f_normal_w + f_buffet_w;
            // Signed Center of Pressure shift: nose-first (`cos_a > 0`) shifts CP FORWARD toward ogive nose
            // (real finless rocket aerodynamic instability!), belly-flop (`cos_a = 0`) sits at geometric midbody,
            // and tail-first (`cos_a < 0`) shifts CP AFT toward the engine skirt (requiring top grid fins)!
            let cp_local_z =
                ab.spec.cp_crossflow_local + ab.spec.cp_axial_shift_m * cos_a + pifs_cp_shift_m;
            let cp_w = bp.transform_point(ab.spec.axis_local * cp_local_z);
            let mut tau_total = (cp_w - com_w).cross(f_axial_w + f_normal_w) + tau_buffet_w;

            // Crossflow rotational strip damping (exact integral along slender cylinder `[-L/2, +L/2]`):
            // Linear crossflow term scales with perpendicular airflow `v_perp_mag` (strictly 0 in pure axial flight!),
            // and quadratic spin-drag integral is `∫ 0.5 ρ C_d D |ω r| (ω r) r dr = (1/64) ρ C_d D L⁴ |ω| ω`!
            let omega_perp = bv.angular - axis_w * bv.angular.dot(axis_w);
            let w_mag = omega_perp.length();
            if w_mag > 1e-6 && ab.spec.rot_damping_scale > 0.0 {
                let l = ab.spec.length_m;
                let d = ab.spec.diameter_m;
                let c_lin = (1.0 / 12.0) * atmo.density_kg_m3 * cd_n * d * l * l * l * v_perp_mag;
                let c_quad = (1.0 / 64.0) * atmo.density_kg_m3 * cd_n * d * l.powi(4) * w_mag;
                tau_total -= omega_perp * ((c_lin + c_quad) * ab.spec.rot_damping_scale);
            }

            wrenches[i].linear += f_total;
            wrenches[i].angular += tau_total;
            ab.reading.dynamic_pressure_pa = q;
            ab.reading.mach = mach;
            ab.reading.aoa_rad = aoa;
            ab.reading.srp_drag_factor = srp_factor;
            ab.reading.knudsen_number = kn;
            ab.reading.pifs_cp_shift_m = pifs_cp_shift_m;
            ab.reading.buffet_torque_nm += tau_buffet_w.length() * weight;
            ab.reading.world_force += f_total * weight;
            ab.reading.world_torque += tau_total * weight;
        }

        // 2. Actuated Body Flaps & Titanium Grid Fins!
        for surf in self.surfaces.values_mut() {
            let (Some(bi), Some(desc)) =
                (scene.body_index(surf.spec.body), scene.body(surf.spec.body))
            else {
                continue;
            };
            if !desc.enabled || desc.kind != BodyKind::Dynamic {
                continue;
            }
            let i = bi.index();
            let bp = poses[i];
            let bv = vels[i];
            let com_w = bp.transform_point(coms[i]);
            let pt_w = bp.transform_point(surf.spec.center_local);
            let alt_m = rockets.altitude_at(pt_w);
            let atmo = rockets.atmosphere_at(pt_w);
            if atmo.density_kg_m3 < 1e-10 {
                continue;
            }
            let arm = pt_w - com_w;
            let turb_alt_scale = (-alt_m / 18_000.0).exp();
            let wind_w = rockets.corotating_atmosphere_velocity(pt_w)
                + self.wind.wind_at(pt_w, alt_m, self.time_s)
                + self.dryden.gust_world() * turb_alt_scale;
            let v_local_w = bv.linear + bv.angular.cross(arm) - wind_w;
            let speed = v_local_w.length();
            if speed < 1e-3 {
                continue;
            }
            let q = 0.5 * atmo.density_kg_m3 * speed * speed;
            let mach = speed / atmo.speed_of_sound_mps.max(100.0);

            // Rotate surface normal by current deflection angle `δ` about `hinge_axis_local`.
            let rot_d = molla_math::Quat::from_axis_angle(
                surf.spec.hinge_axis_local,
                surf.deflection_rad,
            );
            let n_w = (bp.rotation * (rot_d * surf.spec.normal_local)).normalize();
            let v_hat = v_local_w / speed;
            let sin_local = v_hat.dot(n_w).clamp(-1.0, 1.0); // Positive when air strikes `+n_w` face
            let local_aoa = sin_local.asin();

            // Aerodynamic normal force coefficient `C_N` by surface type:
            let cn = match surf.spec.kind {
                AeroSurfaceKind::BodyFlap => {
                    // Flat-plate high-AoA crossflow: F_n = q S C_N,max sin(α) |sin(α)| + linear lift term
                    let mach_boost = 1.0 + 0.18 * ((mach - 0.8) / 1.2).clamp(0.0, 1.0);
                    surf.spec.cn_max * mach_boost * (0.35 * sin_local + 0.65 * sin_local * sin_local.abs())
                }
                AeroSurfaceKind::GridFin => {
                    // Titanium grid fin: high slope at subsonic/supersonic, classic transonic choking dip
                    // around M ∈ [0.92, 1.38] where shocks choke the open lattice cells!
                    let transonic_choke = 1.0 - 0.52 * (-((mach - 1.12) / 0.22).powi(2)).exp();
                    let raw = (2.8 * local_aoa).clamp(-surf.spec.cn_max, surf.spec.cn_max);
                    raw * transonic_choke
                }
            };

            let f_n_mag = q * surf.spec.area_m2 * cn;
            let f_w = -n_w * f_n_mag;
            let tau_w = arm.cross(f_w);
            let hinge_torque = f_n_mag * surf.spec.hinge_moment_arm_m;

            wrenches[i].linear += f_w;
            wrenches[i].angular += tau_w;
            surf.reading.local_aoa_rad = local_aoa;
            surf.reading.hinge_torque_nm += hinge_torque * weight;
            surf.reading.world_force += f_w * weight;
            surf.reading.world_torque += tau_w * weight;
        }
        Ok(())
    }
}
