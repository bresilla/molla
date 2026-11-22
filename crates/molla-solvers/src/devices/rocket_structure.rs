//! Real-time launch-vehicle structural aeroelasticity (NASA SP-8036 / SP-8057)
//! and longitudinal Pogo propulsion-structure coupling (NASA SP-8055).
//!
//! Designed for 60+ FPS real-time rigid+flex simulation without FEM overhead:
//! - **Free-Free Euler-Bernoulli Fuselage Bending Modes (1st & 2nd modes, pitch & yaw)**:
//!   analytic normalized beam eigenfunctions `φ_1(ξ), φ_2(ξ)` and slopes `φ'_1(ξ), φ'_2(ξ)`
//!   with live propellant-mass-scaled frequencies `ω_i(t) = ω_{i,dry} sqrt(m_dry / m_total)`.
//!   Excited by lateral TVC gimbal forces and aerodynamic/control-surface normal loads,
//!   and physically feeds elastic slope rate `Σ φ'_i(y_imu) q̇_i` into mounted flight IMUs!
//! - **Longitudinal Pogo Coupled Structure-Propulsion Mode (NASA SP-8055)**:
//!   1st axial structural mode coupled to LOX downcomer fluid inertia `ρ L_pipe z̈_p` and
//!   turbopump pressure-to-thrust gain, damped by a configurable Pogo accumulator.
//! - **Digital Biquad Bending Notch Filter** (`BiquadNotchFilter`): reusable 2nd-order
//!   direct-form-II notch filter for GNC Control-Structure Interaction (CSI) stabilization.

use std::collections::BTreeMap;

use molla_core::{Error, Result};
use molla_math::{GRAVITY, Real, SpatialVector, Vec3};
use molla_sim::State;
use molla_sim::runtime::{BodyHandle, BodyKind, RigidScene};

use super::rocket::RocketPropulsion;
use super::rocket_aero::RocketAero;

/// Free-free Euler-Bernoulli beam eigenvalues `β_i L` for modes 1 and 2.
const BETA1_L: Real = 4.730_040_744_862_704;
const BETA2_L: Real = 7.853_204_624_095_838;

/// Normalized free-free Euler-Bernoulli beam mode shape `φ(u)` and dimensionless slope `dφ/du`
/// for `u = y / L + 0.5 ∈ [0, 1]` (mode 1 symmetric, mode 2 antisymmetric, normalized to `|φ(0)| ≈ 1`).
pub fn free_free_beam_mode(mode: usize, u_clamped: Real) -> (Real, Real) {
    let u = u_clamped.clamp(0.0, 1.0);
    let bl = if mode == 1 { BETA1_L } else { BETA2_L };
    let sigma = (bl.cosh() - bl.cos()) / (bl.sinh() - bl.sin());
    let x = bl * u;
    // Standard free-free eigenfunction: φ(u) = 0.5 * ((cosh x + cos x) - σ (sinh x + sin x))
    // At u = 0: φ(0) = 1.0; at u = 1: φ(1) = +1.0 (mode 1) or -1.0 (mode 2)!
    let phi = 0.5 * ((x.cosh() + x.cos()) - sigma * (x.sinh() + x.sin()));
    let dphi_du = 0.5 * bl * ((x.sinh() - x.sin()) - sigma * (x.cosh() + x.cos()));
    (phi, dphi_du)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FlexBodyId(pub u64);

/// Flexible launch-vehicle structural bending & longitudinal Pogo specification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlexBodySpec {
    pub body: BodyHandle,
    /// Unit longitudinal vehicle axis in body frame (e.g. `+Y`).
    pub axis_local: Vec3,
    /// Total structural length `L` (m; `71.0 m` Booster, `50.0 m` Ship).
    pub length_m: Real,
    /// Reference dry structural mass `m_dry` (kg) at which `mode1_dry_hz` / `mode2_dry_hz` are specified.
    pub dry_mass_kg: Real,
    /// 1st free-free fuselage bending natural frequency when empty (Hz; ~3.0 Hz Booster, ~3.8 Hz Ship).
    pub mode1_dry_hz: Real,
    /// 2nd free-free fuselage bending natural frequency when empty (Hz; ~7.5 Hz Booster, ~9.5 Hz Ship).
    pub mode2_dry_hz: Real,
    /// Modal structural damping ratios `[ζ_1, ζ_2]` (~0.015–0.03 for welded stainless steel tanks).
    pub bending_damping: [Real; 2],
    /// Axial coordinate of engine TVC thrust puck along `axis_local` (m, e.g. `-L/2 + 1.0`).
    pub engine_station_m: Real,
    /// Axial coordinate of the mounted flight IMU along `axis_local` (m; feeds slope rate `φ'(y_imu) q̇`).
    pub imu_station_m: Real,
    /// 1st longitudinal (axial) Pogo structural frequency when empty (Hz; ~18–24 Hz).
    pub pogo_dry_hz: Real,
    /// Longitudinal structural damping ratio `ζ_pogo` (~0.03).
    pub pogo_damping: Real,
    /// Effective LOX feedline downcomer column length `L_pipe` (m; ~35 m Booster, ~15 m Ship).
    pub feedline_length_m: Real,
    /// Feedline liquid density (kg/m³; LOX ~1140).
    pub feedline_density_kg_m3: Real,
    /// Pogo accumulator compliance/damping factor `κ_acc >= 0` (0 = undamped pogo risk, 2.5 = damped accumulator).
    pub pogo_accumulator_damping: Real,
    /// Pump inlet-pressure-to-thrust coupling gain `∂(ΔT/T) / ∂P_in` (1/Pa; ~3.5e-7).
    pub pogo_pump_gain_per_pa: Real,
}

impl FlexBodySpec {
    pub fn super_heavy(body: BodyHandle) -> Self {
        Self {
            body,
            axis_local: Vec3::Y,
            length_m: 71.0,
            dry_mass_kg: 275_000.0,
            mode1_dry_hz: 3.1,
            mode2_dry_hz: 7.8,
            bending_damping: [0.025, 0.030],
            engine_station_m: -34.7,
            imu_station_m: 26.0, // Forward interstage avionics bay!
            pogo_dry_hz: 19.5,
            pogo_damping: 0.045,
            feedline_length_m: 34.0,
            feedline_density_kg_m3: 1_140.0,
            pogo_accumulator_damping: 2.2,
            pogo_pump_gain_per_pa: 3.2e-7,
        }
    }

    pub fn starship_upper(body: BodyHandle) -> Self {
        Self {
            body,
            axis_local: Vec3::Y,
            length_m: 50.0,
            dry_mass_kg: 120_000.0,
            mode1_dry_hz: 3.8,
            mode2_dry_hz: 9.4,
            bending_damping: [0.028, 0.035],
            engine_station_m: -24.2,
            imu_station_m: 15.0, // Nose avionics bay!
            pogo_dry_hz: 23.0,
            pogo_damping: 0.05,
            feedline_length_m: 16.0,
            feedline_density_kg_m3: 1_140.0,
            pogo_accumulator_damping: 2.4,
            pogo_pump_gain_per_pa: 2.8e-7,
        }
    }
}

/// Live aeroelastic bending & longitudinal Pogo telemetry for one launch-vehicle stage.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlexBodyReading {
    /// Live propellant-mass-scaled 1st & 2nd fuselage bending frequencies (Hz).
    pub mode1_hz: Real,
    pub mode2_hz: Real,
    /// Generalized tip deflection amplitudes `[mode1_x, mode1_z, mode2_x, mode2_z]` (m).
    pub modal_disp_m: [Real; 4],
    /// Peak lateral tip elastic deflection magnitude (m).
    pub tip_deflection_m: Real,
    /// Elastic bending angular rate measured at the mounted flight IMU in body frame (rad/s).
    pub imu_bending_rate_body: Vec3,
    /// Elastic bending lateral acceleration at the mounted flight IMU in body frame (m/s²).
    pub imu_bending_accel_body: Vec3,
    /// Live longitudinal Pogo natural frequency (Hz).
    pub pogo_hz: Real,
    /// Longitudinal Pogo oscillation acceleration (g's).
    pub pogo_accel_g: Real,
    /// LOX downcomer Pogo pressure oscillation amplitude (Pa).
    pub pogo_pressure_pa: Real,
    /// Pogo-induced axial thrust modulation force applied to the stage (N).
    pub pogo_thrust_mod_n: Real,
}

struct FlexBodyState {
    spec: FlexBodySpec,
    // Modal coordinates [m1_x, m1_z, m2_x, m2_z] and velocities (m, m/s)
    q: [Real; 4],
    qd: [Real; 4],
    qdd: [Real; 4],
    // Longitudinal Pogo state (m, m/s, m/s²)
    z_pogo: Real,
    zd_pogo: Real,
    zdd_pogo: Real,
    prev_axial_thrust_n: Real,
    reading: FlexBodyReading,
}

/// Real-time launch-vehicle structural bending & NASA SP-8055 Pogo dynamics device.
#[derive(Default)]
pub struct RocketStructure {
    next_id: u64,
    bodies: BTreeMap<FlexBodyId, FlexBodyState>,
    by_handle: BTreeMap<BodyHandle, FlexBodyId>,
}

impl RocketStructure {
    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    pub fn active(&self) -> bool {
        !self.bodies.is_empty()
    }

    pub fn insert(&mut self, scene: &RigidScene, mut spec: FlexBodySpec) -> Result<FlexBodyId> {
        if scene.body(spec.body).is_none() {
            return Err(Error::Build("flex structure body is missing".into()));
        }
        if !spec.axis_local.is_finite()
            || spec.axis_local.length_squared() < 1e-12
            || !(spec.length_m > 0.0)
            || !(spec.dry_mass_kg > 0.0)
            || !(spec.mode1_dry_hz > 0.0)
            || !(spec.mode2_dry_hz > 0.0)
        {
            return Err(Error::Build("invalid flex body specification".into()));
        }
        spec.axis_local = spec.axis_local.normalize();
        self.next_id += 1;
        let id = FlexBodyId(self.next_id);
        let init_reading = FlexBodyReading {
            mode1_hz: spec.mode1_dry_hz,
            mode2_hz: spec.mode2_dry_hz,
            pogo_hz: spec.pogo_dry_hz,
            ..FlexBodyReading::default()
        };
        self.bodies.insert(
            id,
            FlexBodyState {
                spec,
                q: [0.0; 4],
                qd: [0.0; 4],
                qdd: [0.0; 4],
                z_pogo: 0.0,
                zd_pogo: 0.0,
                zdd_pogo: 0.0,
                prev_axial_thrust_n: 0.0,
                reading: init_reading,
            },
        );
        self.by_handle.insert(spec.body, id);
        Ok(id)
    }

    pub fn reading(&self, id: FlexBodyId) -> Option<FlexBodyReading> {
        self.bodies.get(&id).map(|b| b.reading)
    }

    pub fn reading_for_body(&self, body: BodyHandle) -> Option<FlexBodyReading> {
        self.by_handle.get(&body).and_then(|id| self.reading(*id))
    }

    /// Elastic bending angular rate physically seen by the flight IMU mounted on `body` (body frame, rad/s).
    pub fn imu_bending_rate_body(&self, body: BodyHandle) -> Vec3 {
        self.reading_for_body(body)
            .map_or(Vec3::ZERO, |r| r.imu_bending_rate_body)
    }

    /// Live 1st bending mode frequency (Hz) for adaptive GNC notch filter tracking.
    pub fn mode1_hz_for_body(&self, body: BodyHandle) -> Option<Real> {
        self.reading_for_body(body).map(|r| r.mode1_hz)
    }

    pub(crate) fn apply(
        &mut self,
        scene: &RigidScene,
        state: &State,
        rockets: &RocketPropulsion,
        rocket_aero: &RocketAero,
        wrenches: &mut [SpatialVector],
        sub_dt: Real,
        weight: Real,
    ) -> Result<()> {
        if self.is_empty() || !(sub_dt > 0.0) {
            return Ok(());
        }
        let poses = state.body_q.host()?;
        let masses = scene.model().body_mass.host()?;

        for fb in self.bodies.values_mut() {
            let (Some(bi), Some(desc)) = (scene.body_index(fb.spec.body), scene.body(fb.spec.body))
            else {
                continue;
            };
            if !desc.enabled || desc.kind != BodyKind::Dynamic {
                continue;
            }
            let i = bi.index();
            let bp = poses[i];
            let mass_now = masses[i].max(fb.spec.dry_mass_kg * 0.5);
            let mass_scale = (fb.spec.dry_mass_kg / mass_now).sqrt().clamp(0.30, 1.35);

            // Live propellant-mass-scaled modal frequencies:
            let f1 = fb.spec.mode1_dry_hz * mass_scale;
            let f2 = fb.spec.mode2_dry_hz * mass_scale;
            let w1 = std::f64::consts::TAU * f1;
            let w2 = std::f64::consts::TAU * f2;

            // Generalized modal mass of a normalized free-free beam (`∫ ρ φ² dx ≈ 0.25 M`):
            let m_gen1 = (0.26 * mass_now).max(1_000.0);
            let m_gen2 = (0.24 * mass_now).max(1_000.0);

            // Build orthonormal lateral axes `(e1, e2)` perpendicular to `axis_local` (for `+Y`, `e1 = +X, e2 = +Z`).
            let ax = fb.spec.axis_local;
            let e1 = if ax.x.abs() < 0.9 {
                Vec3::X - ax * ax.x
            } else {
                Vec3::Y - ax * ax.y
            }
            .normalize();
            let e2 = ax.cross(e1).normalize();

            // Gather live lateral excitation forces in body frame from propulsion (TVC + side loads)
            // and aerodynamics (body normal + transonic buffet + fins/flaps)!
            let f_tvc_w = rockets.body_total_thrust_world(fb.spec.body);
            let f_tvc_b = bp.rotation.conjugate() * f_tvc_w;
            let f_tvc_1 = f_tvc_b.dot(e1);
            let f_tvc_2 = f_tvc_b.dot(e2);
            let f_axial_thrust = f_tvc_b.dot(ax).max(0.0);

            let (f_aero_w, f_surf_w, surf_y_avg) =
                rocket_aero.body_excitation_forces(fb.spec.body);
            let f_aero_b = bp.rotation.conjugate() * f_aero_w;
            let f_surf_b = bp.rotation.conjugate() * f_surf_w;

            let l = fb.spec.length_m;
            let u_eng = (fb.spec.engine_station_m / l + 0.5).clamp(0.0, 1.0);
            let u_cp = 0.48; // Midbody aerodynamic load station
            let u_surf = (surf_y_avg / l + 0.5).clamp(0.0, 1.0);
            let u_imu = (fb.spec.imu_station_m / l + 0.5).clamp(0.0, 1.0);

            let (phi1_eng, _) = free_free_beam_mode(1, u_eng);
            let (phi2_eng, _) = free_free_beam_mode(2, u_eng);
            let (phi1_cp, _) = free_free_beam_mode(1, u_cp);
            let (phi2_cp, _) = free_free_beam_mode(2, u_cp);
            let (phi1_surf, _) = free_free_beam_mode(1, u_surf);
            let (phi2_surf, _) = free_free_beam_mode(2, u_surf);
            let (phi1_imu, dphi1_imu) = free_free_beam_mode(1, u_imu);
            let (phi2_imu, dphi2_imu) = free_free_beam_mode(2, u_imu);

            let gen_f = [
                phi1_eng * f_tvc_1 + phi1_cp * f_aero_b.dot(e1) + phi1_surf * f_surf_b.dot(e1),
                phi1_eng * f_tvc_2 + phi1_cp * f_aero_b.dot(e2) + phi1_surf * f_surf_b.dot(e2),
                phi2_eng * f_tvc_1 + phi2_cp * f_aero_b.dot(e1) + phi2_surf * f_surf_b.dot(e1),
                phi2_eng * f_tvc_2 + phi2_cp * f_aero_b.dot(e2) + phi2_surf * f_surf_b.dot(e2),
            ];
            let omegas = [w1, w1, w2, w2];
            let zetas = [
                fb.spec.bending_damping[0],
                fb.spec.bending_damping[0],
                fb.spec.bending_damping[1],
                fb.spec.bending_damping[1],
            ];
            let mgens = [m_gen1, m_gen1, m_gen2, m_gen2];

            // Semi-implicit symplectic integration of the 4 bending oscillators (unconditionally stable for sub_dt <= 1/120 s!).
            let max_q = 0.012 * l; // Structural yield cap (~1.2% of fuselage length)
            for k in 0..4 {
                let w = omegas[k];
                let c = 2.0 * zetas[k] * w;
                let acc = gen_f[k] / mgens[k] - c * fb.qd[k] - w * w * fb.q[k];
                let qd_new = fb.qd[k] + acc * sub_dt;
                let mut q_new = fb.q[k] + qd_new * sub_dt;
                let mut qd_clamped = qd_new;
                if q_new.abs() > max_q {
                    q_new = q_new.clamp(-max_q, max_q);
                    qd_clamped *= 0.5;
                }
                fb.qdd[k] = acc;
                fb.qd[k] = qd_clamped;
                fb.q[k] = q_new;
            }

            // Compute physical slope rate `dθ/dt = (1/L) Σ (dφ_i/du)(u_imu) q̇_i` at IMU station!
            // Lateral displacement `δ_1` along `e1` with axial slope `dδ_1/dy` is a rotation about `e1 × ax = -e2`
            // (for `ax=+Y, e1=+X, e2=+Z`: positive `dδ_x/dy` tilts `+Y` toward `+X`, which is `-Z` rotation!).
            let slope_rate_e1 = (dphi1_imu * fb.qd[0] + dphi2_imu * fb.qd[2]) / l;
            let slope_rate_e2 = (dphi1_imu * fb.qd[1] + dphi2_imu * fb.qd[3]) / l;
            let imu_bend_rate_b = e2 * (-slope_rate_e1) + e1 * slope_rate_e2;
            let imu_bend_acc_b = e1 * (phi1_imu * fb.qdd[0] + phi2_imu * fb.qdd[2])
                + e2 * (phi1_imu * fb.qdd[1] + phi2_imu * fb.qdd[3]);

            // Free-free Euler-Bernoulli modes satisfy `∫ ρ φ_i dx = 0` and `∫ ρ x φ_i dx = 0`,
            // so elastic bending strictly conserves rigid-body linear & angular momentum in free flight
            // and couples to the vehicle through real IMU elastic slope-rate / acceleration feedthrough!
            let _ = (phi1_eng, phi2_eng, m_gen1, m_gen2);

            // NASA SP-8055 Longitudinal Pogo Coupled Structure-Propulsion Dynamics!
            let f_pogo = fb.spec.pogo_dry_hz * mass_scale;
            let w_pogo = std::f64::consts::TAU * f_pogo;
            let m_pogo = (0.35 * mass_now).max(2_000.0);
            // Thrust transient (ignition/staging/throttle steps) + Pogo pressure feedback drives the axial mode!
            let d_thrust = f_axial_thrust - fb.prev_axial_thrust_n;
            fb.prev_axial_thrust_n = f_axial_thrust;
            let atten = 1.0 + fb.spec.pogo_accumulator_damping.max(0.0);
            let p_pogo_pa =
                -(fb.spec.feedline_density_kg_m3 * fb.spec.feedline_length_m * fb.zdd_pogo) / atten;
            let pogo_thrust_mod = if f_axial_thrust > 1.0e4 {
                (f_axial_thrust * fb.spec.pogo_pump_gain_per_pa * p_pogo_pa)
                    .clamp(-0.08 * f_axial_thrust, 0.08 * f_axial_thrust)
            } else {
                0.0
            };
            let zeta_eff = fb.spec.pogo_damping * (1.0 + 0.45 * fb.spec.pogo_accumulator_damping);
            let f_pogo_drive = d_thrust * 0.25 + pogo_thrust_mod;
            let zdd = f_pogo_drive / m_pogo
                - 2.0 * zeta_eff * w_pogo * fb.zd_pogo
                - w_pogo * w_pogo * fb.z_pogo;
            // Use implicit-like denominator for stability at ~20 Hz with sub_dt ~ 1/120 s!
            let denom = 1.0 + 2.0 * zeta_eff * w_pogo * sub_dt + (w_pogo * sub_dt).powi(2);
            let zd_new = (fb.zd_pogo
                + (f_pogo_drive / m_pogo - w_pogo * w_pogo * fb.z_pogo) * sub_dt)
                / denom;
            let z_new = (fb.z_pogo + zd_new * sub_dt).clamp(-0.05, 0.05);
            fb.zdd_pogo = zdd.clamp(-12.0 * GRAVITY, 12.0 * GRAVITY);
            fb.zd_pogo = zd_new;
            fb.z_pogo = z_new;

            if pogo_thrust_mod.abs() > 1.0 {
                wrenches[i].linear += (bp.rotation * ax) * pogo_thrust_mod;
            }

            let tip_x = fb.q[0] + fb.q[2];
            let tip_z = fb.q[1] + fb.q[3];
            fb.reading.mode1_hz = f1;
            fb.reading.mode2_hz = f2;
            fb.reading.modal_disp_m = fb.q;
            fb.reading.tip_deflection_m = (tip_x * tip_x + tip_z * tip_z).sqrt();
            fb.reading.imu_bending_rate_body = imu_bend_rate_b;
            fb.reading.imu_bending_accel_body = imu_bend_acc_b;
            fb.reading.pogo_hz = f_pogo;
            fb.reading.pogo_accel_g = fb.zdd_pogo.abs() / GRAVITY;
            fb.reading.pogo_pressure_pa = p_pogo_pa;
            fb.reading.pogo_thrust_mod_n =
                fb.reading.pogo_thrust_mod_n * (1.0 - weight) + pogo_thrust_mod * weight;
        }
        Ok(())
    }
}

/// Real-time 2nd-order digital Biquad Notch Filter for GNC bending-mode CSI rejection.
///
/// Tracks a time-varying target notch frequency `f0_hz` (e.g. live propellant-mass-scaled
/// 1st fuselage bending frequency) with quality factor `q` (typically `2.0..4.0`).
#[derive(Clone, Copy, Debug)]
pub struct BiquadNotchFilter {
    z1: Real,
    z2: Real,
    b0: Real,
    b1: Real,
    b2: Real,
    a1: Real,
    a2: Real,
}

impl Default for BiquadNotchFilter {
    fn default() -> Self {
        Self::new(2.0, 3.0, 60.0)
    }
}

impl BiquadNotchFilter {
    pub fn new(f0_hz: Real, q: Real, sample_hz: Real) -> Self {
        let mut f = Self {
            z1: 0.0,
            z2: 0.0,
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
        f.retune(f0_hz, q, sample_hz);
        f
    }

    /// Updates biquad coefficients cleanly without resetting filter memory (bumpless gain scheduling!).
    pub fn retune(&mut self, f0_hz: Real, q: Real, sample_hz: Real) {
        let fs = sample_hz.max(10.0);
        let fc = f0_hz.clamp(0.2, 0.45 * fs);
        let w0 = std::f64::consts::TAU * fc / fs;
        let alpha = w0.sin() / (2.0 * q.max(0.5));
        let cos_w0 = w0.cos();
        let a0 = 1.0 + alpha;
        self.b0 = 1.0 / a0;
        self.b1 = (-2.0 * cos_w0) / a0;
        self.b2 = 1.0 / a0;
        self.a1 = (-2.0 * cos_w0) / a0;
        self.a2 = (1.0 - alpha) / a0;
    }

    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    pub fn filter(&mut self, x: Real) -> Real {
        // Direct Form II Transposed: numerically smooth even under live frequency retuning!
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}
