//! Liquid-propellant rocket propulsion, altitude-dependent nozzle gas dynamics,
//! 2-axis Thrust Vector Control (TVC) with Tail-Wags-Dog (TWD) inertial coupling,
//! multi-tank shifting COM / inertia tensors, and NASA SP-8009 acceleration-driven
//! pendulum propellant slosh.
//!
//! Designed for full-scale launch vehicles (e.g. Starship + Super Heavy 33-engine
//! clusters):
//! - **US Standard Atmosphere 1976** (`us76_atmosphere`) from 0 to 86 km seamlessly
//!   blended into the NRLMSISE-00 thermosphere table (`nrlmsise_density_kg_m3`) above
//!   86 km for exact ambient pressure `P_a(h)`, density `ρ(h)`, temperature `T(h)`,
//!   and speed of sound `a(h)`.
//! - **Nozzle Gas Dynamics**: exact `F = ṁ v_e + (P_e - P_a) A_e`, minimum throttle
//!   floor, turbopump spool-up / shutdown lag, Schmucker/Summerfield overexpansion
//!   flow separation with unsteady lateral side-loads, and Prandtl–Pack shock-cell
//!   spacing `λ = 1.22 D_e sqrt(M_e² - 1)`.
//! - **2-Axis Electric TVC + Tail-Wags-Dog (NASA SP-8015)**: second-order rate- and
//!   angle-limited gimbal actuators applying exact nonlinear inertial reaction force
//!   `-m_e (α_g × l_c + ω_g × (ω_g × l_c))` and counter-torque `-I_e α_g`.
//! - **Multi-Tank Dynamic Mass Properties + Pendulum Slosh (NASA SP-8009 / Dodge)**:
//!   tracks dry vehicle mass + arbitrary cylindrical propellant tanks (main + header),
//!   recomputing exact total mass, shifting center of mass, and full 3×3 parallel-axis
//!   inertia tensor every step while integrating acceleration-scaled first-mode liquid
//!   slosh (`ω_s² = 1.841 a_eff / R * tanh(1.841 h / R)`).

use std::collections::BTreeMap;

use molla_core::{Error, Result};
use molla_math::{GRAVITY, Mat3, Quat, Real, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyHandle, BodyKind, RigidScene};
use molla_sim::{MassProperties, State};

use super::space_env::{SolarActivity, nrlmsise_density_kg_m3};

const AIR_R: Real = 287.052_87;
const AIR_GAMMA: Real = 1.4;
const BESSEL_XI11: Real = 1.841_183_781; // First root of J_1'(x) = 0 (cylindrical tank fundamental slosh mode)

/// Atmospheric thermodynamic state at geometric altitude `h` (m).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtmosphereState {
    pub altitude_m: Real,
    pub temperature_k: Real,
    pub pressure_pa: Real,
    pub density_kg_m3: Real,
    pub speed_of_sound_mps: Real,
}

/// US Standard Atmosphere 1976 layers `(base_alt_m, base_temp_k, lapse_k_per_m, base_pressure_pa)`.
const US76_LAYERS: &[(Real, Real, Real, Real)] = &[
    (0.0, 288.15, -0.0065, 101_325.0),
    (11_000.0, 216.65, 0.0, 22_632.06),
    (20_000.0, 216.65, 0.0010, 5_474.889),
    (32_000.0, 228.65, 0.0028, 868.0187),
    (47_000.0, 270.65, 0.0, 110.9063),
    (51_000.0, 270.65, -0.0028, 66.93887),
    (71_000.0, 214.65, -0.0020, 3.956420),
    (84_852.0, 186.946, 0.0, 0.3733836),
];

/// Evaluates US Standard Atmosphere 1976 from 0 to 86 km and blends smoothly into
/// the NRLMSISE-00 thermosphere density table above 86 km.
pub fn us76_atmosphere(altitude_m: Real) -> AtmosphereState {
    let h = altitude_m.max(0.0);
    if h <= 86_000.0 {
        let mut layer = US76_LAYERS[0];
        for &l in US76_LAYERS {
            if h >= l.0 {
                layer = l;
            } else {
                break;
            }
        }
        let (h0, t0, lapse, p0) = layer;
        let dh = h - h0;
        let (temp, pressure) = if lapse.abs() > 1e-12 {
            let t = (t0 + lapse * dh).max(120.0);
            let p = p0 * (t / t0).powf(-GRAVITY / (AIR_R * lapse));
            (t, p)
        } else {
            let t = t0;
            let p = p0 * (-GRAVITY * dh / (AIR_R * t)).exp();
            (t, p)
        };
        let density = pressure / (AIR_R * temp);
        let speed_of_sound = (AIR_GAMMA * AIR_R * temp).sqrt();
        AtmosphereState {
            altitude_m: h,
            temperature_k: temp,
            pressure_pa: pressure,
            density_kg_m3: density,
            speed_of_sound_mps: speed_of_sound,
        }
    } else {
        let density = nrlmsise_density_kg_m3(h, SolarActivity::Moderate);
        let temp = (186.9 + ((h - 86_000.0) / 150_000.0).clamp(0.0, 1.0) * 750.0).min(1000.0);
        let pressure = density * AIR_R * temp;
        let speed_of_sound = (AIR_GAMMA * AIR_R * temp).sqrt();
        AtmosphereState {
            altitude_m: h,
            temperature_k: temp,
            pressure_pa: pressure,
            density_kg_m3: density,
            speed_of_sound_mps: speed_of_sound,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TankId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RocketEngineId(pub u64);

/// Cylindrical propellant tank mounted on a dynamic rocket stage, contributing
/// exact live mass, shifting center of mass, 3×3 parallel-axis inertia, and
/// NASA SP-8009 acceleration-driven lateral pendulum slosh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PropellantTankSpec {
    pub body: BodyHandle,
    /// Bottom (aft) center of the cylindrical tank in the parent body frame (m).
    pub base_local: Vec3,
    /// Unit longitudinal axis pointing from tank bottom toward top (e.g. `+Y` or `+X`).
    pub axis_local: Vec3,
    /// Tank inner radius `R` (m).
    pub radius_m: Real,
    /// Tank cylindrical height `H` (m).
    pub height_m: Real,
    /// Liquid propellant bulk density `ρ_liq` (kg/m³, e.g. LOX ~1140, LCH4 ~422, blended ~830).
    pub liquid_density_kg_m3: Real,
    /// Initial propellant mass inside this tank (kg).
    pub initial_mass_kg: Real,
    /// Fraction of liquid mass participating in fundamental lateral slosh mode (`0.0..=0.8`).
    pub slosh_mass_fraction: Real,
    /// Viscous baffle damping ratio `ζ_s` (`0.005..0.15`; ringed anti-slosh baffles ~0.04–0.08).
    pub slosh_damping_ratio: Real,
    /// Residual microgravity capillary stiffness per unit slosh mass (`ω_0²`, rad²/s²).
    pub capillary_omega2: Real,
}

/// WGS-84 Earth sidereal rotation rate `Ω_E` (rad/s).
pub const EARTH_OMEGA_RAD_S: Real = 7.292_115_9e-5;

/// Computes Earth's angular velocity vector in a local Y-up, +X-East, -Z-North launch frame
/// at geodetic latitude `latitude_deg` (e.g. `25.997°` for Starbase Boca Chica).
pub fn earth_rotation_at_latitude(latitude_deg: Real) -> Vec3 {
    let lat = latitude_deg.to_radians();
    Vec3::new(0.0, EARTH_OMEGA_RAD_S * lat.sin(), -EARTH_OMEGA_RAD_S * lat.cos())
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PropellantTankReading {
    pub mass_remaining_kg: Real,
    pub fill_fraction: Real,
    pub fill_height_m: Real,
    pub liquid_com_local: Vec3,
    pub slosh_mass_kg: Real,
    pub slosh_freq_hz: Real,
    pub slosh_disp_m: Vec3,
    pub slosh_force_world: Vec3,
    pub slosh_torque_world: Vec3,
    /// Live autogenous ullage pressure inside the tank (Pa).
    pub ullage_pressure_pa: Real,
    /// Live static pressure at the tank outlet / turbopump inlet flange (Pa).
    pub inlet_pressure_pa: Real,
    /// Available Net Positive Suction Head `NPSH_a` at pump inlet (m).
    pub npsh_available_m: Real,
    /// Turbopump cavitation thrust/flow retention factor (`1.0` = healthy, `< 1.0` = cavitating!).
    pub cavitation_factor: Real,
}

/// Full-scale liquid rocket engine specification (e.g. SpaceX Raptor 2/3 SL or RVac).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RocketEngineSpec {
    pub body: BodyHandle,
    /// Optional tank(s) supplying this engine (proportional mass draw across linked tanks).
    /// Empty slice / `None` gives unlimited propellant with no mass change.
    pub tank: Option<TankId>,
    /// Optional secondary tank (e.g. fuel tank alongside oxidizer tank, or header tank fallback).
    pub secondary_tank: Option<TankId>,
    /// Fraction of mass flow drawn from `tank` when `secondary_tank` is `Some` (e.g. 0.78 LOX / 0.22 CH4).
    pub primary_mass_fraction: Real,
    /// Gimbal pivot frame in body coordinates: `+X` is nominal thrust direction (opposite exhaust),
    /// gimbal pitch rotates about local `Z`, gimbal yaw rotates about local `Y`.
    pub frame: Transform,
    /// Rated vacuum thrust at `throttle = 1.0` (N).
    pub vacuum_thrust_n: Real,
    /// Vacuum specific impulse `I_sp,vac` (s).
    pub vacuum_isp_s: Real,
    /// Chamber stagnation pressure `P_c` at `throttle = 1.0` (Pa, e.g. 30–35 MPa for Raptor 2/3).
    pub chamber_pressure_pa: Real,
    /// Nozzle exit diameter `D_e` (m; ~1.30 m Raptor SL, ~2.30 m RVac).
    pub exit_diameter_m: Real,
    /// Exhaust gas ratio of specific heats `γ` (~1.20–1.24 for hot methalox).
    pub exhaust_gamma: Real,
    /// Design exit Mach number `M_e` (~3.4 Raptor SL, ~4.8 RVac).
    pub exit_mach: Real,
    /// Design exit static pressure at `throttle = 1.0` (Pa; ~65–85 kPa Raptor SL, ~15 kPa RVac).
    pub exit_pressure_pa: Real,
    /// Minimum stable throttle fraction (e.g. `0.25..0.40`). Commands below this shut the engine off.
    pub min_throttle: Real,
    /// Turbopump spool-up time constant (s, e.g. 0.18 s).
    pub startup_tau_s: Real,
    /// Shutdown tail-off time constant (s, e.g. 0.12 s).
    pub shutdown_tau_s: Real,
    /// Maximum gimbal deflection cone half-angle (rad; `0.0` = fixed outer/RVac engine).
    pub max_gimbal_rad: Real,
    /// Maximum gimbal angular rate (rad/s; e.g. `0.8–1.4 rad/s` electric TVC).
    pub max_gimbal_rate_rad_s: Real,
    /// TVC actuator closed-loop natural frequency (rad/s; e.g. 35 rad/s ≈ 5.5 Hz).
    pub gimbal_actuator_omega: Real,
    /// Engine assembly mass participating in Tail-Wags-Dog (kg; ~1,525–1,630 kg for Raptor).
    pub engine_mass_kg: Real,
    /// Distance from gimbal pivot aft to engine assembly COM (m; ~0.85 m).
    pub engine_cg_aft_m: Real,
    /// Engine transverse moment of inertia about gimbal pivot (kg·m²; ~1,600 kg·m² for Raptor).
    pub engine_inertia_kgm2: Real,
}

impl RocketEngineSpec {
    /// SpaceX Raptor 3 Sea-Level gimbaled engine (`2,746 kN` vac / `~2,500 kN` SL, 350 bar, ±15° TVC).
    pub fn raptor3_sea_level(
        body: BodyHandle,
        frame: Transform,
        lox_tank: Option<TankId>,
        ch4_tank: Option<TankId>,
        gimbal_max_rad: Real,
    ) -> Self {
        Self {
            body,
            tank: lox_tank,
            secondary_tank: ch4_tank,
            primary_mass_fraction: 0.782, // Methalox O/F ≈ 3.6:1
            frame,
            vacuum_thrust_n: 2_746_000.0,
            vacuum_isp_s: 363.0,
            chamber_pressure_pa: 35_000_000.0,
            exit_diameter_m: 1.30,
            exhaust_gamma: 1.22,
            exit_mach: 3.45,
            exit_pressure_pa: 72_000.0,
            min_throttle: 0.28,
            startup_tau_s: 0.18,
            shutdown_tau_s: 0.11,
            max_gimbal_rad: gimbal_max_rad,
            max_gimbal_rate_rad_s: 1.35,
            gimbal_actuator_omega: 38.0,
            engine_mass_kg: 1_525.0,
            engine_cg_aft_m: 0.85,
            engine_inertia_kgm2: 1_550.0,
        }
    }

    /// SpaceX Raptor 3 Vacuum fixed engine (`2,850 kN` vac, 380 s vac Isp, 2.30 m bell, fixed 0°).
    pub fn raptor3_vacuum(
        body: BodyHandle,
        frame: Transform,
        lox_tank: Option<TankId>,
        ch4_tank: Option<TankId>,
    ) -> Self {
        Self {
            body,
            tank: lox_tank,
            secondary_tank: ch4_tank,
            primary_mass_fraction: 0.782,
            frame,
            vacuum_thrust_n: 2_850_000.0,
            vacuum_isp_s: 380.0,
            chamber_pressure_pa: 35_000_000.0,
            exit_diameter_m: 2.30,
            exhaust_gamma: 1.21,
            exit_mach: 4.85,
            exit_pressure_pa: 14_500.0,
            min_throttle: 0.35,
            startup_tau_s: 0.22,
            shutdown_tau_s: 0.14,
            max_gimbal_rad: 0.0,
            max_gimbal_rate_rad_s: 0.0,
            gimbal_actuator_omega: 30.0,
            engine_mass_kg: 1_950.0,
            engine_cg_aft_m: 1.35,
            engine_inertia_kgm2: 3_800.0,
        }
    }

    pub fn exit_area_m2(&self) -> Real {
        0.25 * std::f64::consts::PI * self.exit_diameter_m * self.exit_diameter_m
    }
}

/// Live telemetry & plume diagnostics of one rocket engine over the last step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RocketEngineReading {
    pub ignited: bool,
    pub failed: bool,
    pub starved: bool,
    /// Actual chamber throttle fraction `[0, 1]` after spool-up/tail-off lag.
    pub chamber_throttle: Real,
    /// Net applied thrust magnitude `ṁ v_e + (P_e - P_a) A_e` (N).
    pub thrust_n: Real,
    /// Effective instantaneous specific impulse `F / (ṁ g₀)` (s).
    pub effective_isp_s: Real,
    /// Total propellant mass flow rate `ṁ` (kg/s).
    pub mass_flow_kg_s: Real,
    /// Actual gimbal deflection `(pitch_z, yaw_y)` (rad).
    pub gimbal_rad: [Real; 2],
    /// Actual gimbal angular rate `(pitch_dot, yaw_dot)` (rad/s).
    pub gimbal_rate_rad_s: [Real; 2],
    /// `true` when Schmucker/Summerfield overexpansion flow separation is active!
    pub flow_separated: bool,
    /// Unsteady lateral side-load force magnitude from flow separation (N).
    pub side_load_n: Real,
    /// Tail-Wags-Dog inertial torque magnitude injected on parent stage (N·m).
    pub twd_torque_nm: Real,
    /// Prandtl–Pack shock-cell wavelength `λ` (m) for exhaust rendering.
    pub shock_cell_length_m: Real,
    /// Nozzle exit-to-ambient pressure ratio `P_e / P_a` (>1 underexpanded / vacuum bloom).
    pub pressure_ratio: Real,
    /// `true` if upstream tank NPSH dropped below pump requirement (real turbopump cavitation!).
    pub cavitating: bool,
    /// World-space applied thrust vector (N).
    pub world_force: Vec3,
    /// World-space applied torque about stage COM (N·m).
    pub world_torque: Vec3,
}

/// Command for one rocket engine.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RocketEngineCommand {
    /// Ignition switch (`false` commands shutdown tail-off).
    pub ignited: bool,
    /// Desired throttle in `[0, 1]` (clamped to `[min_throttle, 1.0]` when ignited).
    pub throttle: Real,
    /// Desired 2-axis gimbal angles `[pitch_about_local_z, yaw_about_local_y]` (rad).
    pub gimbal_rad: [Real; 2],
}

struct BodyDryMass {
    dry: MassProperties,
    prev_vel_world: Option<Vec3>,
    specific_accel_world: Vec3,
}

struct TankState {
    spec: PropellantTankSpec,
    mass_kg: Real,
    /// Regulated target autogenous ullage pressure (Pa; default 4.8 bar).
    target_ullage_pa: Real,
    /// Live ullage pressure (Pa).
    ullage_pa: Real,
    /// Liquid saturation vapor pressure at subcooled operating point (Pa; ~1.1 bar).
    vapor_pressure_pa: Real,
    /// Required pump inducer Net Positive Suction Head `NPSH_r` (m; ~14 m).
    npsh_required_m: Real,
    /// Current cavitation thrust/flow retention factor `[0.25, 1.0]`.
    cavitation_factor: Real,
    /// Slosh displacement in the 2D plane perpendicular to `axis_local` (body frame, m).
    slosh_disp_body: Vec3,
    slosh_vel_body: Vec3,
    reading: PropellantTankReading,
}

struct EngineState {
    spec: RocketEngineSpec,
    cmd: RocketEngineCommand,
    failed: bool,
    chamber_throttle: Real,
    gimbal: [Real; 2],
    gimbal_rate: [Real; 2],
    gimbal_accel: [Real; 2],
    step_thrust_n: Real,
    step_side_load_local: Vec3,
    step_twd_force_local: Vec3,
    step_twd_torque_local: Vec3,
    phase_seed: Real,
    reading: RocketEngineReading,
}

/// Multi-stage liquid rocket propulsion, tank mass-shift, slosh, ullage/NPSH cavitation,
/// TVC/TWD, rotating-Earth Coriolis, and ground/interstage plume impingement device.
#[derive(Default)]
pub struct RocketPropulsion {
    next_tank: u64,
    next_engine: u64,
    time_s: Real,
    /// Reference ground elevation `y_ground` (m) for altitude `h = pos.y - y_ground`
    /// and supersonic ground-impingement cushion. Set `None` to disable ground cushion.
    ground_y: Option<Real>,
    /// Altitude offset added to `pos.y` for atmosphere lookup (e.g. pad elevation).
    altitude_offset_m: Real,
    /// Optional spherical planet `(center, radius)` for orbital altitude calculation
    /// (overrides flat `pos.y` when `Some`).
    planet: Option<(Vec3, Real)>,
    /// Optional planetary rotation vector `Ω` in world frame (rad/s) for exact Coriolis `-2 m (Ω × v)`.
    planet_omega_world: Vec3,
    dry_masses: BTreeMap<BodyHandle, BodyDryMass>,
    tanks: BTreeMap<TankId, TankState>,
    engines: BTreeMap<RocketEngineId, EngineState>,
    /// Optional hot-staging / interstage impingement target pairs `(upper_engine_body, lower_stage_body, lower_dome_local, dome_radius)`.
    interstage_targets: Vec<(BodyHandle, BodyHandle, Vec3, Real)>,
    last_interstage_blast_n: Real,
    last_ground_cushion_n: Real,
}

impl RocketPropulsion {
    pub fn is_empty(&self) -> bool {
        self.tanks.is_empty() && self.engines.is_empty()
    }

    pub fn active(&self) -> bool {
        !self.tanks.is_empty() || !self.engines.is_empty()
    }

    pub fn set_flat_ground(&mut self, ground_y: Option<Real>, altitude_offset_m: Real) {
        self.ground_y = ground_y;
        self.altitude_offset_m = altitude_offset_m;
        self.planet = None;
    }

    pub fn set_spherical_planet(&mut self, center: Vec3, radius_m: Real) {
        self.planet = Some((center, radius_m));
        self.ground_y = None;
    }

    /// Enables rotating-planet Coriolis force `-2 m (Ω × v)` on all registered rocket stages.
    pub fn set_earth_rotation(&mut self, omega_world: Vec3) {
        self.planet_omega_world = omega_world;
    }

    /// Configures a tank's autogenous ullage pressure target, propellant vapor pressure,
    /// and turbopump required Net Positive Suction Head (`NPSH_r`, m) for real cavitation dynamics.
    pub fn set_tank_pressurization(
        &mut self,
        id: TankId,
        target_ullage_pa: Real,
        vapor_pressure_pa: Real,
        npsh_required_m: Real,
    ) -> Result<()> {
        let t = self
            .tanks
            .get_mut(&id)
            .ok_or_else(|| Error::Build("no such propellant tank".into()))?;
        t.target_ullage_pa = target_ullage_pa.max(0.0);
        t.ullage_pa = target_ullage_pa.max(0.0);
        t.vapor_pressure_pa = vapor_pressure_pa.max(0.0);
        t.npsh_required_m = npsh_required_m.max(0.0);
        Ok(())
    }

    /// Register a lower-stage dome that receives hot-staging exhaust blast force
    /// whenever engines mounted on `upper_body` fire down onto it within 30 m.
    pub fn register_interstage_blast_target(
        &mut self,
        upper_body: BodyHandle,
        lower_body: BodyHandle,
        dome_center_local: Vec3,
        dome_radius_m: Real,
    ) {
        self.interstage_targets
            .push((upper_body, lower_body, dome_center_local, dome_radius_m));
    }

    pub fn last_interstage_blast_n(&self) -> Real {
        self.last_interstage_blast_n
    }

    pub fn last_ground_cushion_n(&self) -> Real {
        self.last_ground_cushion_n
    }

    /// Register the dry structural mass properties of a stage body before inserting
    /// its propellant tanks so total mass, COM, and 3×3 inertia stay exact as fuel drains.
    pub fn register_stage_dry_mass(
        &mut self,
        scene: &RigidScene,
        body: BodyHandle,
        dry: MassProperties,
    ) -> Result<()> {
        if scene.body(body).is_none() || !dry.mass.is_finite() || dry.mass <= 0.0 {
            return Err(Error::Build("invalid rocket stage dry mass".into()));
        }
        let vel = scene.body_velocity(body).ok().map(|v| v.linear);
        self.dry_masses.insert(
            body,
            BodyDryMass {
                dry,
                prev_vel_world: vel,
                specific_accel_world: Vec3::ZERO,
            },
        );
        Ok(())
    }

    pub fn insert_tank(&mut self, scene: &mut RigidScene, mut spec: PropellantTankSpec) -> Result<TankId> {
        if scene.body(spec.body).is_none() {
            return Err(Error::Build("propellant tank body is missing".into()));
        }
        if !spec.base_local.is_finite()
            || !spec.axis_local.is_finite()
            || spec.axis_local.length_squared() < 1e-12
            || !spec.radius_m.is_finite()
            || spec.radius_m <= 0.0
            || !spec.height_m.is_finite()
            || spec.height_m <= 0.0
            || !spec.liquid_density_kg_m3.is_finite()
            || spec.liquid_density_kg_m3 <= 0.0
            || !spec.initial_mass_kg.is_finite()
            || spec.initial_mass_kg < 0.0
            || !(0.0..=0.85).contains(&spec.slosh_mass_fraction)
            || !spec.slosh_damping_ratio.is_finite()
            || spec.slosh_damping_ratio < 0.0
        {
            return Err(Error::Build("invalid propellant tank specification".into()));
        }
        spec.axis_local = spec.axis_local.normalize();
        // Auto-capture dry mass if caller didn't explicitly register it yet.
        if !self.dry_masses.contains_key(&spec.body) {
            let dry = scene.body_mass_properties(spec.body)?;
            let vel = scene.body_velocity(spec.body).ok().map(|v| v.linear);
            self.dry_masses.insert(
                spec.body,
                BodyDryMass {
                    dry,
                    prev_vel_world: vel,
                    specific_accel_world: Vec3::ZERO,
                },
            );
        }
        self.next_tank += 1;
        let id = TankId(self.next_tank);
        self.tanks.insert(
            id,
            TankState {
                spec,
                mass_kg: spec.initial_mass_kg,
                target_ullage_pa: 480_000.0,
                ullage_pa: 480_000.0,
                vapor_pressure_pa: 110_000.0,
                npsh_required_m: 14.0,
                cavitation_factor: 1.0,
                slosh_disp_body: Vec3::ZERO,
                slosh_vel_body: Vec3::ZERO,
                reading: PropellantTankReading {
                    mass_remaining_kg: spec.initial_mass_kg,
                    ullage_pressure_pa: 480_000.0,
                    inlet_pressure_pa: 480_000.0,
                    npsh_available_m: 32.0,
                    cavitation_factor: 1.0,
                    ..PropellantTankReading::default()
                },
            },
        );
        self.recompute_body_mass_properties(scene, spec.body)?;
        Ok(id)
    }

    pub fn insert_engine(&mut self, scene: &RigidScene, spec: RocketEngineSpec) -> Result<RocketEngineId> {
        if scene.body(spec.body).is_none() {
            return Err(Error::Build("rocket engine body is missing".into()));
        }
        if !spec.vacuum_thrust_n.is_finite()
            || spec.vacuum_thrust_n <= 0.0
            || !spec.vacuum_isp_s.is_finite()
            || spec.vacuum_isp_s <= 0.0
            || !spec.exit_diameter_m.is_finite()
            || spec.exit_diameter_m <= 0.0
            || !spec.exit_mach.is_finite()
            || spec.exit_mach <= 1.0
            || !spec.min_throttle.is_finite()
            || !(0.0..=1.0).contains(&spec.min_throttle)
            || !spec.max_gimbal_rad.is_finite()
            || spec.max_gimbal_rad < 0.0
        {
            return Err(Error::Build("invalid rocket engine specification".into()));
        }
        self.next_engine += 1;
        let id = RocketEngineId(self.next_engine);
        self.engines.insert(
            id,
            EngineState {
                spec,
                cmd: RocketEngineCommand::default(),
                failed: false,
                chamber_throttle: 0.0,
                gimbal: [0.0, 0.0],
                gimbal_rate: [0.0, 0.0],
                gimbal_accel: [0.0, 0.0],
                step_thrust_n: 0.0,
                step_side_load_local: Vec3::ZERO,
                step_twd_force_local: Vec3::ZERO,
                step_twd_torque_local: Vec3::ZERO,
                phase_seed: (self.next_engine as Real) * 1.618_033_988,
                reading: RocketEngineReading::default(),
            },
        );
        Ok(id)
    }

    pub fn command_engine(&mut self, id: RocketEngineId, cmd: RocketEngineCommand) -> Result<()> {
        let eng = self
            .engines
            .get_mut(&id)
            .ok_or_else(|| Error::Build("no such rocket engine".into()))?;
        if !cmd.throttle.is_finite()
            || !cmd.gimbal_rad[0].is_finite()
            || !cmd.gimbal_rad[1].is_finite()
        {
            return Err(Error::Build("rocket engine command must be finite".into()));
        }
        eng.cmd = cmd;
        Ok(())
    }

    pub fn set_engine_failed(&mut self, id: RocketEngineId, failed: bool) -> Result<()> {
        let eng = self
            .engines
            .get_mut(&id)
            .ok_or_else(|| Error::Build("no such rocket engine".into()))?;
        eng.failed = failed;
        Ok(())
    }

    /// Reassign an engine's propellant source tanks (e.g. switching Starship's
    /// 3 center Raptors from main tanks to landing header tanks before flip!).
    pub fn set_engine_tanks(
        &mut self,
        id: RocketEngineId,
        primary: Option<TankId>,
        secondary: Option<TankId>,
    ) -> Result<()> {
        let eng = self
            .engines
            .get_mut(&id)
            .ok_or_else(|| Error::Build("no such rocket engine".into()))?;
        eng.spec.tank = primary;
        eng.spec.secondary_tank = secondary;
        Ok(())
    }

    pub fn set_tank_mass(&mut self, scene: &mut RigidScene, id: TankId, mass_kg: Real) -> Result<()> {
        let body = {
            let t = self
                .tanks
                .get_mut(&id)
                .ok_or_else(|| Error::Build("no such propellant tank".into()))?;
            t.mass_kg = mass_kg.max(0.0);
            t.spec.body
        };
        self.recompute_body_mass_properties(scene, body)
    }

    pub fn engine_reading(&self, id: RocketEngineId) -> Option<RocketEngineReading> {
        self.engines.get(&id).map(|e| e.reading)
    }

    pub fn engine_spec(&self, id: RocketEngineId) -> Option<&RocketEngineSpec> {
        self.engines.get(&id).map(|e| &e.spec)
    }

    pub fn tank_reading(&self, id: TankId) -> Option<PropellantTankReading> {
        self.tanks.get(&id).map(|t| t.reading)
    }

    /// Total remaining propellant mass across all tanks mounted on `body` (kg).
    pub fn body_propellant_kg(&self, body: BodyHandle) -> Real {
        self.tanks
            .values()
            .filter(|t| t.spec.body == body)
            .map(|t| t.mass_kg)
            .sum()
    }

    /// Total world-space thrust currently generated by all engines on `body` (N).
    pub fn body_total_thrust_world(&self, body: BodyHandle) -> Vec3 {
        let mut sum = Vec3::ZERO;
        for e in self.engines.values() {
            if e.spec.body == body {
                sum += e.reading.world_force;
            }
        }
        sum
    }

    /// Thrust-weighted mean nozzle exit-to-ambient pressure ratio `P_e / P_a` on `body`
    /// (used by `RocketAero` for Plume-Induced Flow Separation / PIFS at altitude).
    pub fn body_mean_pressure_ratio(&self, body: BodyHandle) -> Real {
        let mut w_sum = 0.0;
        let mut pr_sum = 0.0;
        for e in self.engines.values() {
            if e.spec.body == body && e.reading.thrust_n > 1.0 {
                w_sum += e.reading.thrust_n;
                pr_sum += e.reading.thrust_n * e.reading.pressure_ratio;
            }
        }
        if w_sum > 1.0 { pr_sum / w_sum } else { 0.0 }
    }

    pub fn altitude_at(&self, pos_world: Vec3) -> Real {
        if let Some((center, r_p)) = self.planet {
            ((pos_world - center).length() - r_p).max(0.0)
        } else {
            let base = self.ground_y.unwrap_or(0.0);
            (pos_world.y - base + self.altitude_offset_m).max(0.0)
        }
    }

    /// Exact parallel-axis recombination of dry stage mass + all live liquid tanks!
    fn recompute_body_mass_properties(&self, scene: &mut RigidScene, body: BodyHandle) -> Result<()> {
        let Some(dry_entry) = self.dry_masses.get(&body) else {
            return Ok(());
        };
        let dry = dry_entry.dry;
        let mut total_mass = dry.mass;
        let mut weighted_com = dry.local_com * dry.mass;

        // Collect cylinder properties for every non-empty tank on `body`.
        let mut tank_parts: Vec<(Real, Vec3, Mat3)> = Vec::new();
        for t in self.tanks.values() {
            if t.spec.body != body || t.mass_kg <= 1e-6 {
                continue;
            }
            let area = std::f64::consts::PI * t.spec.radius_m * t.spec.radius_m;
            let fill_h = (t.mass_kg / (t.spec.liquid_density_kg_m3 * area))
                .clamp(0.02, t.spec.height_m);
            let com_t = t.spec.base_local + t.spec.axis_local * (0.5 * fill_h);
            let r = t.spec.radius_m;
            let i_axial = 0.5 * t.mass_kg * r * r;
            let i_trans = t.mass_kg * (3.0 * r * r + fill_h * fill_h) / 12.0;
            // Build inertia tensor aligned with `axis_local`.
            let u = t.spec.axis_local;
            let uut = Mat3::from_cols(u * u.x, u * u.y, u * u.z);
            let i_cyl = Mat3::IDENTITY * i_trans + uut * (i_axial - i_trans);
            total_mass += t.mass_kg;
            weighted_com += com_t * t.mass_kg;
            tank_parts.push((t.mass_kg, com_t, i_cyl));
        }

        let new_com = weighted_com / total_mass.max(1e-6);
        let parallel_shift = |m: Real, d: Vec3| -> Mat3 {
            let d2 = d.length_squared();
            let ddt = Mat3::from_cols(d * d.x, d * d.y, d * d.z);
            (Mat3::IDENTITY * d2 - ddt) * m
        };
        let mut total_inertia = dry.inertia + parallel_shift(dry.mass, dry.local_com - new_com);
        for (m_t, com_t, i_cyl) in tank_parts {
            total_inertia += i_cyl + parallel_shift(m_t, com_t - new_com);
        }

        scene.set_body_mass_properties_with_wake(
            body,
            MassProperties {
                mass: total_mass,
                local_com: new_com,
                inertia: total_inertia,
            },
            false,
        )
    }

    pub(crate) fn before_step(&mut self, scene: &mut RigidScene, dt: Real) -> Result<()> {
        if self.is_empty() || !(dt > 0.0) {
            return Ok(());
        }
        self.time_s += dt;
        self.last_interstage_blast_n = 0.0;
        self.last_ground_cushion_n = 0.0;

        // Track bodies whose propellant changed so we recompute COM & 3×3 inertia cleanly once per body.
        let mut dirty_bodies: Vec<BodyHandle> = Vec::new();
        let mut mark_dirty = |b: BodyHandle| {
            if !dirty_bodies.contains(&b) {
                dirty_bodies.push(b);
            }
        };

        // Update body specific accelerations (for NASA SP-8009 slosh natural frequency).
        let g_world = scene.gravity();
        for (&body, entry) in &mut self.dry_masses {
            if let Ok(v) = scene.body_velocity(body) {
                if let Some(v0) = entry.prev_vel_world {
                    entry.specific_accel_world = (v.linear - v0) / dt - g_world;
                }
                entry.prev_vel_world = Some(v.linear);
            }
        }

        // Advance every rocket engine's turbopump spool, 2-axis electric TVC, nozzle gas dynamics,
        // Schmucker flow separation, and Tail-Wags-Dog inertial wrench!
        let planet = self.planet;
        let ground_y = self.ground_y;
        let altitude_offset_m = self.altitude_offset_m;
        let alt_fn = |pos_world: Vec3| -> Real {
            if let Some((center, r_p)) = planet {
                ((pos_world - center).length() - r_p).max(0.0)
            } else {
                let base = ground_y.unwrap_or(0.0);
                (pos_world.y - base + altitude_offset_m).max(0.0)
            }
        };
        let mut tank_draws: BTreeMap<TankId, Real> = BTreeMap::new();
        for eng in self.engines.values_mut() {
            let Ok(pose) = scene.body_pose(eng.spec.body) else {
                continue;
            };
            let alt_m = alt_fn(pose.position);
            let atmo = us76_atmosphere(alt_m);

            // 1. 2-Axis TVC electric actuator second-order dynamics (angle & rate limited).
            let max_g = eng.spec.max_gimbal_rad;
            let mut cmd_g = eng.cmd.gimbal_rad;
            let cmd_norm = (cmd_g[0] * cmd_g[0] + cmd_g[1] * cmd_g[1]).sqrt();
            if cmd_norm > max_g && cmd_norm > 1e-12 {
                let s = max_g / cmd_norm;
                cmd_g[0] *= s;
                cmd_g[1] *= s;
            }
            if max_g > 0.0 {
                let wn = eng.spec.gimbal_actuator_omega.max(5.0);
                let zeta = 0.85;
                for k in 0..2 {
                    let prev_rate = eng.gimbal_rate[k];
                    let acc = wn * wn * (cmd_g[k] - eng.gimbal[k]) - 2.0 * zeta * wn * prev_rate;
                    let mut next_rate = (prev_rate + acc * dt)
                        .clamp(-eng.spec.max_gimbal_rate_rad_s, eng.spec.max_gimbal_rate_rad_s);
                    let mut next_g = eng.gimbal[k] + next_rate * dt;
                    if next_g.abs() > max_g {
                        next_g = next_g.clamp(-max_g, max_g);
                        next_rate = 0.0;
                    }
                    eng.gimbal_accel[k] = (next_rate - prev_rate) / dt;
                    eng.gimbal_rate[k] = next_rate;
                    eng.gimbal[k] = next_g;
                }
            } else {
                eng.gimbal = [0.0, 0.0];
                eng.gimbal_rate = [0.0, 0.0];
                eng.gimbal_accel = [0.0, 0.0];
            }

            // 2. Check propellant availability & upstream turbopump NPSH cavitation in linked tank(s).
            let (has_fuel, cav_factor) = match (eng.spec.tank, eng.spec.secondary_tank) {
                (Some(t1), Some(t2)) => match (self.tanks.get(&t1), self.tanks.get(&t2)) {
                    (Some(a), Some(b)) => (
                        a.mass_kg > 1e-3 && b.mass_kg > 1e-3,
                        a.cavitation_factor.min(b.cavitation_factor),
                    ),
                    _ => (false, 1.0),
                },
                (Some(t1), None) => match self.tanks.get(&t1) {
                    Some(a) => (a.mass_kg > 1e-3, a.cavitation_factor),
                    None => (false, 1.0),
                },
                _ => (true, 1.0),
            };
            let cavitating = cav_factor < 0.995 && eng.chamber_throttle > 1e-3;

            let wants_burn = eng.cmd.ignited && !eng.failed && eng.cmd.throttle >= eng.spec.min_throttle;
            let starved = wants_burn && !has_fuel;
            let target_throttle = if wants_burn && has_fuel {
                eng.cmd.throttle.clamp(eng.spec.min_throttle, 1.0)
            } else {
                0.0
            };

            let tau = if target_throttle > eng.chamber_throttle {
                eng.spec.startup_tau_s.max(1e-4)
            } else {
                eng.spec.shutdown_tau_s.max(1e-4)
            };
            let alpha = (-dt / tau).exp();
            eng.chamber_throttle = target_throttle + (eng.chamber_throttle - target_throttle) * alpha;
            if eng.chamber_throttle < 1e-4 && target_throttle == 0.0 {
                eng.chamber_throttle = 0.0;
            }

            // 3. Nozzle Gas Dynamics: F = ṁ v_e + (P_e - P_a) A_e + Schmucker Flow Separation + NPSH Cavitation!
            let eff_throttle = eng.chamber_throttle * cav_factor;
            let ae = eng.spec.exit_area_m2();
            let mdot_rated = eng.spec.vacuum_thrust_n / (eng.spec.vacuum_isp_s * GRAVITY);
            let mdot = mdot_rated * eff_throttle;
            let p_e = eng.spec.exit_pressure_pa * eff_throttle;
            let p_a = atmo.pressure_pa;
            // Schmucker (1973) separation pressure ratio: P_sep / P_a = (1.88 M_e - 1)^(-0.64)
            let schmucker_ratio = (1.88 * eng.spec.exit_mach - 1.0).max(1.2).powf(-0.64);
            let p_sep = (schmucker_ratio * p_a).min(0.42 * p_a);
            let mut flow_separated = false;
            let mut side_load_local = Vec3::ZERO;
            let net_thrust_n = if eff_throttle > 1e-4 {
                let f_vac = eng.spec.vacuum_thrust_n * eff_throttle;
                if p_a > 100.0 && p_e < p_sep {
                    flow_separated = true;
                    // Flow separates inside the bell where local pressure hits P_sep,
                    // capping overexpansion thrust loss & inducing unsteady lateral side-load!
                    let sep_frac = (p_e / p_sep.max(1.0)).clamp(0.35, 1.0);
                    let ae_eff = ae * sep_frac;
                    let loss = p_a * ae_eff;
                    let separated_area = ae * (1.0 - sep_frac);
                    let side_mag = 0.045 * (p_a - p_e).max(0.0) * separated_area;
                    let osc_y = (self.time_s * 37.0 + eng.phase_seed).sin();
                    let osc_z = (self.time_s * 53.0 + eng.phase_seed * 1.7).cos();
                    side_load_local = Vec3::new(0.0, side_mag * osc_y, side_mag * osc_z);
                    (f_vac - loss).max(0.0)
                } else {
                    (f_vac - p_a * ae).max(0.0)
                }
            } else {
                0.0
            };

            // 4. Propellant drawdown.
            let dm = mdot * dt;
            if dm > 0.0 {
                match (eng.spec.tank, eng.spec.secondary_tank) {
                    (Some(t1), Some(t2)) => {
                        let f1 = eng.spec.primary_mass_fraction.clamp(0.0, 1.0);
                        *tank_draws.entry(t1).or_insert(0.0) += dm * f1;
                        *tank_draws.entry(t2).or_insert(0.0) += dm * (1.0 - f1);
                    }
                    (Some(t1), None) => {
                        *tank_draws.entry(t1).or_insert(0.0) += dm;
                    }
                    _ => {}
                }
            }

            // 5. Tail-Wags-Dog (TWD) exact inertial reaction force & counter-torque in nozzle frame!
            //    Nozzle COM is at `l_c = (-engine_cg_aft_m, 0, 0)` (aft of gimbal pivot).
            //    Gimbal angular vel ω_g = (0, yaw_dot, pitch_dot), α_g = (0, yaw_ddot, pitch_ddot).
            let omega_g = Vec3::new(0.0, eng.gimbal_rate[1], eng.gimbal_rate[0]);
            let alpha_g = Vec3::new(0.0, eng.gimbal_accel[1], eng.gimbal_accel[0]);
            let l_c = Vec3::new(-eng.spec.engine_cg_aft_m, 0.0, 0.0);
            let a_cg_rel = alpha_g.cross(l_c) + omega_g.cross(omega_g.cross(l_c));
            eng.step_twd_force_local = -a_cg_rel * eng.spec.engine_mass_kg;
            eng.step_twd_torque_local = -alpha_g * eng.spec.engine_inertia_kgm2;
            eng.step_thrust_n = net_thrust_n;
            eng.step_side_load_local = side_load_local;

            // Prandtl–Pack shock-cell spacing λ = 1.22 D_e sqrt(M_e² - 1)
            let shock_len = 1.22
                * eng.spec.exit_diameter_m
                * (eng.spec.exit_mach * eng.spec.exit_mach - 1.0).max(0.0).sqrt();
            let pressure_ratio = if p_a > 1.0 { p_e / p_a } else { 1_000.0 };
            let eff_isp = if mdot > 1e-6 {
                net_thrust_n / (mdot * GRAVITY)
            } else {
                0.0
            };

            eng.reading = RocketEngineReading {
                ignited: eng.chamber_throttle > 1e-3,
                failed: eng.failed,
                starved,
                chamber_throttle: eng.chamber_throttle,
                thrust_n: net_thrust_n,
                effective_isp_s: eff_isp,
                mass_flow_kg_s: mdot,
                gimbal_rad: eng.gimbal,
                gimbal_rate_rad_s: eng.gimbal_rate,
                flow_separated,
                side_load_n: side_load_local.length(),
                twd_torque_nm: eng.step_twd_torque_local.length(),
                shock_cell_length_m: shock_len,
                pressure_ratio,
                cavitating,
                world_force: Vec3::ZERO,
                world_torque: Vec3::ZERO,
            };
        }

        // Commit propellant burns to tanks, update pneumatic autogenous ullage / hydrostatic NPSH,
        // and recompute affected stage mass properties!
        for (&tid, tank) in self.tanks.iter_mut() {
            let dm = tank_draws.get(&tid).copied().unwrap_or(0.0);
            if dm > 0.0 {
                tank.mass_kg = (tank.mass_kg - dm).max(0.0);
                mark_dirty(tank.spec.body);
            }
            let area = std::f64::consts::PI * tank.spec.radius_m * tank.spec.radius_m;
            let fill_h = (tank.mass_kg / (tank.spec.liquid_density_kg_m3 * area))
                .clamp(0.0, tank.spec.height_m);
            let ullage_h = (tank.spec.height_m - fill_h).max(0.15);
            // Adiabatic ullage expansion during rapid drain + vigorous slosh collapse, restored by
            // autogenous pressurization regulator valve (`tau_reg ≈ 0.55 s`)!
            let dv_over_v = (dm / (tank.spec.liquid_density_kg_m3 * area * ullage_h)).clamp(0.0, 0.25);
            let slosh_frac = (tank.slosh_disp_body.length() / tank.spec.radius_m).clamp(0.0, 0.8);
            let slosh_drop = slosh_frac * slosh_frac * 0.14 * dt;
            tank.ullage_pa *= (1.0 - 1.3 * dv_over_v - slosh_drop).clamp(0.5, 1.0);
            let reg_alpha = (-dt / 0.55).exp();
            tank.ullage_pa =
                tank.target_ullage_pa + (tank.ullage_pa - tank.target_ullage_pa) * reg_alpha;

            // Signed axial specific acceleration along tank axis (`> 0` settles propellant into aft sump!).
            let a_axial_signed = if let (Ok(bp), Some(d)) = (
                scene.body_pose(tank.spec.body),
                self.dry_masses.get(&tank.spec.body),
            ) {
                (bp.rotation.conjugate() * d.specific_accel_world).dot(tank.spec.axis_local)
            } else {
                GRAVITY
            };
            let hydro_pa = tank.spec.liquid_density_kg_m3 * a_axial_signed * fill_h;
            let mdot_t = dm / dt.max(1e-6);
            // Feedline friction drop scaling with `mdot² / ρ` (realistic ~20–40 kPa at full 33-Raptor flow!).
            let line_loss_pa = (0.0012 * mdot_t * mdot_t / tank.spec.liquid_density_kg_m3).min(85_000.0);
            let p_inlet = (tank.ullage_pa + hydro_pa - line_loss_pa).max(0.0);
            let npsh_a = ((p_inlet - tank.vapor_pressure_pa)
                / (tank.spec.liquid_density_kg_m3 * GRAVITY))
                .max(0.0);
            let cav = if tank.npsh_required_m <= 1e-6 || npsh_a >= tank.npsh_required_m {
                1.0
            } else {
                (0.25 + 0.75 * (npsh_a / tank.npsh_required_m)).clamp(0.25, 1.0)
            };
            tank.cavitation_factor = cav;
            tank.reading.ullage_pressure_pa = tank.ullage_pa;
            tank.reading.inlet_pressure_pa = p_inlet;
            tank.reading.npsh_available_m = npsh_a;
            tank.reading.cavitation_factor = cav;
        }
        for body in dirty_bodies {
            self.recompute_body_mass_properties(scene, body)?;
        }

        // Reset per-step tank force/torque accumulators.
        for tank in self.tanks.values_mut() {
            tank.reading.slosh_force_world = Vec3::ZERO;
            tank.reading.slosh_torque_world = Vec3::ZERO;
        }
        Ok(())
    }

    pub(crate) fn apply(
        &mut self,
        scene: &RigidScene,
        state: &State,
        wrenches: &mut [SpatialVector],
        sub_dt: Real,
        weight: Real,
    ) -> Result<()> {
        if self.is_empty() {
            return Ok(());
        }
        let poses = state.body_q.host()?;
        let vels = state.body_qd.host()?;
        let coms = scene.model().body_com.host()?;
        let masses = scene.model().body_mass.host()?;

        // 0. Rotating-Earth Coriolis Force `-2 m (Ω × v)` on every registered rocket stage!
        if self.planet_omega_world.length_squared() > 0.0 {
            let omega = self.planet_omega_world;
            for &body in self.dry_masses.keys() {
                if let (Some(bi), Some(desc)) = (scene.body_index(body), scene.body(body))
                    && desc.enabled
                    && desc.kind == BodyKind::Dynamic
                {
                    let idx = bi.index();
                    let f_cor = omega.cross(vels[idx].linear) * (-2.0 * masses[idx]);
                    wrenches[idx].linear += f_cor;
                }
            }
        }

        // 1. Apply Rocket Engine Thrust + Gimbal TVC + Side-Loads + TWD + Ground/Interstage Blast!
        let mut active_plumes: Vec<(BodyHandle, Vec3, Vec3, Real, Real)> = Vec::new();
        for eng in self.engines.values_mut() {
            let (Some(bi), Some(desc)) =
                (scene.body_index(eng.spec.body), scene.body(eng.spec.body))
            else {
                continue;
            };
            if !desc.enabled || desc.kind != BodyKind::Dynamic {
                continue;
            }
            let i = bi.index();
            let bp = poses[i];
            let com_w = bp.transform_point(coms[i]);
            let base_frame = bp.compose(&eng.spec.frame);
            let gimbal_q = (Quat::from_rotation_z(eng.gimbal[0])
                * Quat::from_rotation_y(eng.gimbal[1]))
            .normalize();
            let nozzle_rot = (base_frame.rotation * gimbal_q).normalize();

            let thrust_local = Vec3::new(eng.step_thrust_n, 0.0, 0.0) + eng.step_side_load_local;
            let mut f_world =
                nozzle_rot * thrust_local + base_frame.rotation * eng.step_twd_force_local;
            let arm = base_frame.position - com_w;
            let mut tau_world = arm.cross(f_world) + base_frame.rotation * eng.step_twd_torque_local;

            // Ground-impingement overpressure cushion when firing near the launch pad / catch pad!
            if let Some(gy) = self.ground_y
                && eng.step_thrust_n > 1.0
            {
                let exhaust_dir = -(nozzle_rot * Vec3::X);
                let h_noz = base_frame.position.y - gy;
                let h_max = 10.0 * eng.spec.exit_diameter_m;
                if exhaust_dir.y < -0.35 && h_noz > 0.1 && h_noz < h_max {
                    let prox = (1.0 - h_noz / h_max).powi(2);
                    // ~5%–8% reflected acoustic/stagnation ground cushion near pad surface.
                    let f_cushion = eng.step_thrust_n * 0.065 * prox * (-exhaust_dir.y);
                    let fc_vec = Vec3::Y * f_cushion;
                    f_world += fc_vec;
                    tau_world += arm.cross(fc_vec);
                    self.last_ground_cushion_n += f_cushion * weight;
                }
            }

            if f_world.length_squared() > 0.0 || tau_world.length_squared() > 0.0 {
                wrenches[i].linear += f_world;
                wrenches[i].angular += tau_world;
                eng.reading.world_force += f_world * weight;
                eng.reading.world_torque += tau_world * weight;
            }
            if eng.step_thrust_n > 1.0 && !self.interstage_targets.is_empty() {
                let exh_w = -(nozzle_rot * Vec3::X);
                active_plumes.push((
                    eng.spec.body,
                    base_frame.position,
                    exh_w,
                    eng.step_thrust_n,
                    eng.spec.exit_diameter_m,
                ));
            }
        }

        // 2. Interstage Hot-Staging Plume Blast Impingement onto Lower-Stage Dome!
        if !active_plumes.is_empty() {
            for &(upper_b, lower_b, dome_local, dome_r) in &self.interstage_targets {
                let (Some(li), Some(ldesc)) = (scene.body_index(lower_b), scene.body(lower_b))
                else {
                    continue;
                };
                if !ldesc.enabled || ldesc.kind != BodyKind::Dynamic {
                    continue;
                }
                let li = li.index();
                let lp = poses[li];
                let dome_w = lp.transform_point(dome_local);
                let l_com = lp.transform_point(coms[li]);
                for &(src_b, noz_w, exh_w, thrust_n, d_e) in &active_plumes {
                    if src_b != upper_b {
                        continue;
                    }
                    let rel = dome_w - noz_w;
                    let dist = rel.length();
                    if !(0.2..=32.0).contains(&dist) {
                        continue;
                    }
                    let u = rel / dist;
                    let align = exh_w.dot(u);
                    if align < 0.45 {
                        continue;
                    }
                    // Ventilated hot-stage ring deflects ~55% radially; remaining axial blast decays with distance.
                    let spread_r = (0.5 * d_e + 0.30 * dist).max(0.5);
                    let capture_frac = ((dome_r * dome_r) / (spread_r * spread_r)).min(1.0);
                    let f_blast_mag = thrust_n * 0.42 * align.powi(2) * capture_frac / (1.0 + 0.08 * dist);
                    let f_blast = exh_w * f_blast_mag;
                    wrenches[li].linear += f_blast;
                    wrenches[li].angular += (dome_w - l_com).cross(f_blast);
                    self.last_interstage_blast_n += f_blast_mag * weight;
                }
            }
        }

        // 3. Acceleration-Driven NASA SP-8009 Cylindrical-Tank Pendulum Slosh!
        if sub_dt > 0.0 {
            for tank in self.tanks.values_mut() {
                let (Some(bi), Some(desc)) =
                    (scene.body_index(tank.spec.body), scene.body(tank.spec.body))
                else {
                    continue;
                };
                if !desc.enabled || desc.kind != BodyKind::Dynamic {
                    continue;
                }
                let area = std::f64::consts::PI * tank.spec.radius_m * tank.spec.radius_m;
                let max_mass = area * tank.spec.height_m * tank.spec.liquid_density_kg_m3;
                let fill_frac = if max_mass > 1e-6 {
                    (tank.mass_kg / max_mass).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let fill_h = (tank.mass_kg / (tank.spec.liquid_density_kg_m3 * area))
                    .clamp(0.0, tank.spec.height_m);
                let liq_com_local = tank.spec.base_local + tank.spec.axis_local * (0.5 * fill_h);

                // Slosh mass vanishes cleanly when tank is nearly empty or completely full (ullage=0).
                let ullage_factor = (4.0 * fill_frac * (1.0 - fill_frac)).clamp(0.0, 1.0).sqrt();
                let m_slosh = tank.mass_kg * tank.spec.slosh_mass_fraction * ullage_factor;
                if m_slosh < 1.0 {
                    tank.slosh_disp_body = Vec3::ZERO;
                    tank.slosh_vel_body = Vec3::ZERO;
                    tank.reading.mass_remaining_kg = tank.mass_kg;
                    tank.reading.fill_fraction = fill_frac;
                    tank.reading.fill_height_m = fill_h;
                    tank.reading.liquid_com_local = liq_com_local;
                    tank.reading.slosh_mass_kg = 0.0;
                    tank.reading.slosh_freq_hz = 0.0;
                    tank.reading.slosh_disp_m = Vec3::ZERO;
                    continue;
                }

                let i = bi.index();
                let bp = poses[i];
                let com_w = bp.transform_point(coms[i]);
                let a_spec_w = self
                    .dry_masses
                    .get(&tank.spec.body)
                    .map_or(Vec3::ZERO, |d| d.specific_accel_world);
                let a_spec_b = bp.rotation.conjugate() * a_spec_w;
                let ax = tank.spec.axis_local;
                let a_axial = a_spec_b.dot(ax).abs();
                let a_lat_b = a_spec_b - ax * a_spec_b.dot(ax);

                // NASA SP-8009 fundamental antisymmetric slosh mode frequency:
                //   ω_s² = (ξ_11 * a_axial / R) * tanh(ξ_11 * h / R) + ω_capillary²
                let h_over_r = (fill_h / tank.spec.radius_m).max(0.05);
                let omega2 = (BESSEL_XI11 * a_axial / tank.spec.radius_m)
                    * (BESSEL_XI11 * h_over_r).tanh()
                    + tank.spec.capillary_omega2;
                let omega = omega2.max(0.04).sqrt();
                let c_over_m = 2.0 * tank.spec.slosh_damping_ratio * omega;

                // Equation of motion in tank lateral plane: q̈ + 2ζω q̇ + ω² q = -a_lat
                let mut acc_s = -a_lat_b - tank.slosh_disp_body * omega2 - tank.slosh_vel_body * c_over_m;
                // Keep strictly orthogonal to tank axis.
                acc_s -= ax * acc_s.dot(ax);
                tank.slosh_vel_body += acc_s * sub_dt;
                tank.slosh_disp_body += tank.slosh_vel_body * sub_dt;

                // Clamp against cylindrical tank wall (`max_disp = 0.65 R`) with inelastic wall baffle.
                let max_d = 0.65 * tank.spec.radius_m;
                let d_mag = tank.slosh_disp_body.length();
                if d_mag > max_d {
                    let n = tank.slosh_disp_body / d_mag;
                    tank.slosh_disp_body = n * max_d;
                    let vn = tank.slosh_vel_body.dot(n);
                    if vn > 0.0 {
                        tank.slosh_vel_body -= n * (1.25 * vn);
                    }
                }

                // Reaction force exerted by sloshing fluid onto the tank wall (Newton's 3rd Law).
                let f_wall_b = (tank.slosh_disp_body * omega2 + tank.slosh_vel_body * c_over_m) * m_slosh;
                let f_wall_w = bp.rotation * f_wall_b;
                // Apply at the liquid free-surface effective slosh elevation (~0.75 fill_h).
                let slosh_pt_w =
                    bp.transform_point(tank.spec.base_local + ax * (0.75 * fill_h));
                let tau_w = (slosh_pt_w - com_w).cross(f_wall_w);
                wrenches[i].linear += f_wall_w;
                wrenches[i].angular += tau_w;

                tank.reading.mass_remaining_kg = tank.mass_kg;
                tank.reading.fill_fraction = fill_frac;
                tank.reading.fill_height_m = fill_h;
                tank.reading.liquid_com_local = liq_com_local;
                tank.reading.slosh_mass_kg = m_slosh;
                tank.reading.slosh_freq_hz = omega / std::f64::consts::TAU;
                tank.reading.slosh_disp_m = tank.slosh_disp_body;
                tank.reading.slosh_force_world += f_wall_w * weight;
                tank.reading.slosh_torque_world += tau_w * weight;
            }
        }
        Ok(())
    }
}
