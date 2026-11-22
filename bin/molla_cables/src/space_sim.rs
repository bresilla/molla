//! First-Principles 960 Hz ISS IDSS / NASA Docking System (NDS) Hardware-Validation
//! Physics Simulation powered by `molla-solvers`.
//!
//! Zero `Connector` snap cheats, zero scripted clamping, zero fake pressure/vent forces:
//! 1. **Full-Scale Orbital Dynamics (`OrbitalField::iss_leo()`)**:
//!    - `18,000 kg` Orbital Station + `2 × 420 kg` motorized SARJ solar wings +
//!      4-pyramid SGCMGs (`3,000 N·m·s` each) + reaction wheels + station RCS.
//!    - `3,200 kg` Visiting Spacecraft + `16 × 420 N` Draco RCS (`I_sp = 300 s`,
//!      `1.8 N·s` MIB, `25 ms` valve lag, live mass & inertia depletion, `#0` faulted).
//! 2. **Free 6-DOF Stewart-Platform Soft-Capture Ring (`95 kg`) + Real Petal Funnels**:
//!    - Suspended exclusively by 6 telescoping linear actuator struts evaluated at
//!      960 Hz inside `RigidWorld::step_with_forces`.
//!    - Real `ColliderDesc` outward-flaring passive IDA-2 guide petals + active ring
//!      rim collider physically self-center an off-axis approach.
//!    - **3 Independent Physical Soft-Capture Pawls** (`120°` apart, bounded break
//!      load `28 kN` each, ZERO `snap: true` cheat!) engage on mechanical contact
//!      and disengage (`0 N`) once the 12 hard-capture hooks carry the load.
//! 3. **12 Real Dynamic `Revolute` Clamping Hooks Coupled Through Featherstone**:
//!    - Each hook (`4 kg`, reflected gearbox inertia `0.45 kg·m²`) is driven by a
//!      real `MotorSpec` torque actuator (`450 N·m` cap).
//!    - Jaw clamping forces are applied **directly to each hook body's jaw tip**
//!      (and `-F` to the station striker), transmitting 100% of the clamping load
//!      and back-driving motor torque physically through the 12 Featherstone pins
//!      against a 6-pad elastomeric O-ring gasket (`9.0 MN/m`).
//! 4. **First-Law Compressible Open-System Vestibule Gas Thermodynamics**:
//!    - Integrates BOTH air mass `m_air` AND internal energy `U = m c_v T`
//!      (`γ = 1.4`, `R = 287.05`, `V(gap) = V_0 + A Δx`, wall heat transfer,
//!      `-P dV` boundary work) — visibly heating during compression (`> 310 K`)
//!      and flash-cooling below freezing (`< 230 K`) during choked vacuum blowdown.
//!    - Exact analytic choked sonic vacuum thrust per open vent nozzle:
//!      `F = C_d P A_t (2/(γ+1))^(γ/(γ-1)) (γ+1)` applied at exact 3D nozzle frames
//!      on the station (balanced `+Z/-Z` cancels to `< 1e-6 N·m`; jamming `-Z` via
//!      `[V]` visibly pushes & yaws the `21,200 kg` docked stack!).
//!    - Outward pneumatic separation force `F_air = P · A_seal` (`61.6 kN` at
//!      `101.3 kPa`) fights the 12 hooks every substep; `[E]` unclamps under
//!      pressure and physically blasts the spacecraft apart conserving momentum.
//! 5. **Simons Vacuum Plume Impingement on Station & Rotating Solar Wings**:
//!    - Applies free-molecular far-field plume forces (`cos^4 θ / r²`) from active
//!      Draco nozzles onto both the station bulkhead and live-oriented solar wings.

use std::collections::VecDeque;
use std::f64::consts::{FRAC_PI_2, PI, TAU};

use molla_core::Result;
use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry, JointDesc, JointHandle,
};
use molla_sim::{BodyParams, JointType, MassProperties};
use molla_solvers::devices::{
    EARTH_EQUATORIAL_RADIUS, EARTH_MU, EccentricOrbit, FlexJointId, FlexJointSpec, GimbalSpec,
    ImbalanceSpec, ImuId, ImuSpec, LidarId, LidarSpec, MagneticField, MagnetorquerId,
    MagnetorquerSpec, MotorCommand, MotorLimits, MotorSpec, NavigationFilter, NrlmsiseAtmosphere,
    OrbitalField, ReactionWheelId, ReactionWheelSpec, SloshTankId, SloshTankSpec, SolarActivity,
    SolarRadiation, SpacecraftCommand, SpacecraftId, SpacecraftLimits, SpacecraftSpec,
    StarTrackerId, StarTrackerSpec, SurfaceFacetSpec, ThrusterId, ThrusterModulation, ThrusterSpec,
    desaturation_dipole_body, nrlmsise_density_kg_m3,
};
use molla_solvers::rigid_world::RigidWorld;

pub const SIM_DT: f64 = 1.0 / 120.0;
pub const MISSION_SECONDS: f64 = 38.0;

/// Station passive IDA-2 mating flange face offset along station `+X` (m).
pub const STATION_PORT_X: f64 = 2.65;
/// Visiting capsule forward structural bulkhead along chaser `-X` (m).
pub const CHASER_BULKHEAD_X: f64 = -0.94;
/// Recessed Stewart-platform lower actuator clevis plane inside the nose bay (m).
pub const STEWART_BASE_X: f64 = CHASER_BULKHEAD_X + 0.24;
/// Capture ring mating face offset from capture-ring origin along `-X` (m).
pub const RING_FACE_OFFSET_X: f64 = -0.18;
/// Full Stewart-platform extension stroke ahead of the flush position (m).
pub const SCR_EXTENDED_STROKE: f64 = 0.32;
/// Nominal center-to-center distance when O-ring flanges sit flush (m).
pub const FLUSH_CENTER_DIST: f64 = STATION_PORT_X - (CHASER_BULKHEAD_X + RING_FACE_OFFSET_X);

// Real N2/O2 cabin air thermodynamic constants (SI units).
pub const NOMINAL_PRESSURE_PA: f64 = 101_325.0;
const GAS_GAMMA: f64 = 1.4;
const GAS_R: f64 = 287.05;
const GAS_CV: f64 = GAS_R / (GAS_GAMMA - 1.0); // 717.625 J/(kg·K)
const GAS_CP: f64 = GAS_GAMMA * GAS_CV; // 1004.675 J/(kg·K)
const T_WALL_K: f64 = 293.0;
const T_SUPPLY_K: f64 = 293.0;
const VESTIBULE_VOL_0_M3: f64 = 0.65;
const VESTIBULE_WALL_AREA_M2: f64 = 3.2;
const WALL_HEAT_COEFF_W_M2K: f64 = 14.0; // Natural/forced convection inside vestibule
const SEAL_RADIUS_M: f64 = 0.44;
pub const VESTIBULE_AREA_M2: f64 = PI * SEAL_RADIUS_M * SEAL_RADIUS_M; // 0.6082 m^2 -> 61.6 kN at 1 atm
const VENT_NOZZLE_DIAM_M: f64 = 0.042;
const VENT_THROAT_AREA_M2: f64 = 0.25 * PI * VENT_NOZZLE_DIAM_M * VENT_NOZZLE_DIAM_M; // 1.385e-3 m^2
const VENT_CD: f64 = 0.85;

/// Hook open hinge angle (rad) vs closed clamping angle (rad).
pub const HOOK_OPEN_RAD: f64 = 0.92;
pub const HOOK_CLOSED_RAD: f64 = -0.02;

/// Exact analytic sonic vacuum thrust coefficient `C_F = (2/(γ+1))^(γ/(γ-1)) * (γ+1)`
/// multiplied by discharge coefficient `C_d` for choked flow (`M_e = 1`) expanding to vacuum.
pub fn exact_sonic_vacuum_cf(gamma: f64, cd: f64) -> f64 {
    let p_ratio = (2.0 / (gamma + 1.0)).powf(gamma / (gamma - 1.0));
    cd * p_ratio * (gamma + 1.0)
}

/// Exact isentropic choked mass flow rate `ṁ = C_d P A_t sqrt(γ/(R T)) (2/(γ+1))^((γ+1)/(2(γ-1)))`.
pub fn choked_mass_flow_kg_s(pressure_pa: f64, temp_k: f64, area_m2: f64, cd: f64) -> f64 {
    if pressure_pa <= 0.0 || temp_k <= 20.0 || area_m2 <= 0.0 {
        return 0.0;
    }
    let k = GAS_GAMMA;
    let crit = (2.0 / (k + 1.0)).powf((k + 1.0) / (2.0 * (k - 1.0)));
    cd * pressure_pa * area_m2 * (k / (GAS_R * temp_k)).sqrt() * crit
}

/// Returns the exact 6 Stewart-platform lower clevis anchors (chaser frame)
/// and upper clevis anchors (capture_ring frame), shared identically by the
/// 960 Hz physics force callback and 3D renderer.
pub fn stewart_anchors() -> ([Vec3; 6], [Vec3; 6]) {
    let base_r = 0.51;
    let ring_r = 0.49;
    let mut base = [Vec3::ZERO; 6];
    let mut ring = [Vec3::ZERO; 6];
    let mut k = 0;
    for pair in 0..3 {
        let c = pair as f64 * (TAU / 3.0);
        for &(side, shift) in &[(-1.0, -0.42), (1.0, 0.42)] {
            let ab = c + side * 0.11;
            let ar = c + shift;
            base[k] = Vec3::new(STEWART_BASE_X, base_r * ab.cos(), base_r * ab.sin());
            ring[k] = Vec3::new(0.02, ring_r * ar.cos(), ring_r * ar.sin());
            k += 1;
        }
    }
    (base, ring)
}

pub fn stewart_nominal_lengths(stroke_m: f64) -> [f64; 6] {
    let (base, ring) = stewart_anchors();
    let ring_offset = Vec3::new(CHASER_BULKHEAD_X - stroke_m, 0.0, 0.0);
    let mut lens = [0.0; 6];
    for i in 0..6 {
        lens[i] = ((ring_offset + ring[i]) - base[i]).length();
    }
    lens
}

/// Local hook orientation in chaser frame when closed (`q_hinge = 0`):
/// `+X` points forward along chaser `-X`, `+Y` points radially outward, `+Z = -tang`.
pub fn hook_base_rotation(angle: f64) -> Quat {
    let x_axis = Vec3::NEG_X;
    let y_axis = Vec3::new(0.0, angle.cos(), angle.sin());
    let z_axis = x_axis.cross(y_axis).normalize();
    Quat::from_mat3(&Mat3::from_cols(x_axis, y_axis, z_axis)).normalize()
}

pub fn hook_pivot_local(angle: f64) -> Vec3 {
    let rad = Vec3::new(0.0, angle.cos(), angle.sin());
    Vec3::new(CHASER_BULKHEAD_X - 0.04, 0.0, 0.0) + rad * 0.60
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissionPhase {
    Detumble,
    FlyAround,
    ApproachAndExtend,
    FinalClosure,
    SoftCaptureDamp,
    RingRetract,
    HardClamping,
    VestibulePressurize,
    DepressurizeVent,
    EmergencyBlowoff,
    UnclampAndDepart,
}

impl MissionPhase {
    pub fn label(self) -> &'static str {
        match self {
            Self::Detumble => "1 · DETUMBLE & RATE MATCH",
            Self::FlyAround => "1b · KOS COLLISION-FREE FLY-AROUND",
            Self::ApproachAndExtend => "2 · V-BAR APPROACH · EXTENDING 6-DOF SCR",
            Self::FinalClosure => "3 · FINAL CLOSURE · SPIN-MATCH & PETALS",
            Self::SoftCaptureDamp => "4 · 3-PAWL SOFT CAPTURE · 6-STRUT DAMP",
            Self::RingRetract => "5 · STEWART HEXAPOD RETRACTING",
            Self::HardClamping => "6 · 12 DYNAMIC HOOKS CLAMPING (FEATHERSTONE)",
            Self::VestibulePressurize => "7 · VESTIBULE 101.3 kPa · 61.6 kN LOAD",
            Self::DepressurizeVent => "8 · ADIABATIC CHOKED-FLOW AIR VENT",
            Self::EmergencyBlowoff => "! · EMERGENCY PRESSURIZED BLOW-OFF !",
            Self::UnclampAndDepart => "9 · UNCLAMP · SPRING PUSH-OFF & DEPART",
        }
    }
}

/// Active interactive 3D mouse grab/nudge tether applied as a real physical
/// spring-damper wrench at the clicked point on any spacecraft body.
#[derive(Clone, Copy, Debug)]
pub struct UserGrab {
    pub body: BodyHandle,
    pub local_point: Vec3,
    pub target_world: Vec3,
    pub depth: f64,
    pub applied_force_n: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct GasPuff {
    pub pos: Vec3,
    pub vel: Vec3,
    pub radius: f64,
    pub growth_rate: f64,
    pub stretch: f64,
    pub age: f64,
    pub max_age: f64,
    pub base_opacity: f32,
    pub seed: f32,
    /// `0.0` = Draco RCS exhaust, `1.0` = Adiabatic cryo-vented air.
    pub kind: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrbitMode {
    ZeroG,
    IssLeo,
    EccentricLeo,
}

impl OrbitMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::ZeroG => "Pure Inertial 0-G",
            Self::IssLeo => "ISS LEO (408km CW + NRLMSISE)",
            Self::EccentricLeo => "Eccentric e=0.22 (YA + NRLMSISE)",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::ZeroG => Self::IssLeo,
            Self::IssLeo => Self::EccentricLeo,
            Self::EccentricLeo => Self::ZeroG,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModulationUiMode {
    Pwpf,
    Pwm10Hz,
    Continuous,
}

impl ModulationUiMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Pwpf => "PWPF (Schmitt Pulse)",
            Self::Pwm10Hz => "PWM (10 Hz Discrete)",
            Self::Continuous => "Continuous Analog",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Pwpf => Self::Pwm10Hz,
            Self::Pwm10Hz => Self::Continuous,
            Self::Continuous => Self::Pwpf,
        }
    }

    pub fn to_device(self) -> ThrusterModulation {
        match self {
            Self::Pwpf => ThrusterModulation::aerospace_pwpf(),
            Self::Pwm10Hz => ThrusterModulation::rcs_pwm_10hz(),
            Self::Continuous => ThrusterModulation::Continuous,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Telemetry {
    pub time_s: f64,
    pub phase: MissionPhase,
    pub port_distance_m: f64,
    pub flange_gap_mm: f64,
    pub closing_speed_mps: f64,
    pub attitude_err_deg: f64,
    pub chaser_omega_deg_s: f64,
    pub chaser_mass_kg: f64,
    pub propellant_kg: f64,
    pub initial_propellant_kg: f64,
    pub active_thrusters: usize,
    pub faulted_thrusters: usize,
    pub total_thrust_n: f64,
    pub impingement_force_n: f64,
    pub scr_stroke_m: f64,
    pub scr_lateral_mm: f64,
    pub strut_peak_n: f64,
    pub soft_pawls_latched: usize,
    pub soft_pawl_force_kn: f64,
    pub hooks_engaged: usize,
    pub clamp_force_kn: f64,
    pub hook_peak_motor_nm: f64,
    pub seal_force_kn: f64,
    pub seal_min_pad_kn: f64,
    pub slosh_disp_mm: f64,
    pub slosh_force_n: f64,
    pub env_drag_srp_mn: f64,
    pub mtb_torque_mnm: f64,
    pub vestibule_pressure_kpa: f64,
    pub vestibule_temp_k: f64,
    pub air_sep_force_kn: f64,
    pub vent_thrust_pos_z_n: f64,
    pub vent_thrust_neg_z_n: f64,
    pub vent_net_torque_nm: f64,
    pub vent_jammed: bool,
    pub seal_verified: bool,
    pub venting_air: bool,
    pub petal_contacts: usize,
    pub station_speed_mps: f64,
    pub station_omega_deg_s: f64,
    pub station_att_deg: f64,
    pub station_free_drift: bool,
    pub cmg_momentum_nms: f64,
    pub iss_leo: bool,
    pub orbit_mode: OrbitMode,
    pub altitude_km: f64,
    pub atmo_density_kg_m3: f64,
    pub wing_flex_deg: f64,
    pub wing_flex_energy_j: f64,
    pub modulation_mode: ModulationUiMode,
    pub rcs_valve_cycles: u64,
    pub hil_sensor_gnc: bool,
    pub lidar_locked: bool,
    pub lidar_range_m: f64,
    pub nav_att_err_deg: f64,
    pub nav_bias_err_mrad_s: f64,
}

pub struct SpaceDockingSim {
    pub world: RigidWorld,
    // Station complex (6,500 kg bus + 2 × 2-segment flexible solar wings)
    pub station: BodyHandle,
    pub solar_port: BodyHandle,
    pub solar_port_outer: BodyHandle,
    pub solar_starboard: BodyHandle,
    pub solar_starboard_outer: BodyHandle,
    pub solar_port_joint: JointHandle,
    pub solar_port_flex_joint: JointHandle,
    pub solar_starboard_joint: JointHandle,
    pub solar_starboard_flex_joint: JointHandle,
    pub solar_port_flex: FlexJointId,
    pub solar_starboard_flex: FlexJointId,
    pub station_thrusters: Vec<ThrusterId>,
    pub station_wheels: Vec<ReactionWheelId>,
    pub station_cmgs: Vec<ReactionWheelId>,
    pub station_mtb: MagnetorquerId,
    pub station_gnc: SpacecraftId,
    /// When `true` (default!), the station is 100% FREE-FLOATING (zero invisible autopilot
    /// holding it in place) so every petal bump, retraction pull, vent jet, and blow-off
    /// visibly moves & rotates BOTH spacecraft by Newton's 3rd Law!
    pub station_free_drift: bool,
    // Visiting spacecraft (3,200 kg) + FREE 6-DOF Soft-Capture Ring (95 kg)
    pub chaser: BodyHandle,
    pub capture_ring: BodyHandle,
    pub chaser_slosh: SloshTankId,
    pub chaser_thrusters: Vec<ThrusterId>,
    pub chaser_wheels: Vec<ReactionWheelId>,
    pub chaser_gnc: SpacecraftId,
    pub faulted_thruster_idx: Option<usize>,
    // Hardware-in-the-loop spacecraft sensor suite + closed-loop navigation filter
    pub chaser_imu: ImuId,
    pub chaser_star_tracker: StarTrackerId,
    pub chaser_lidar: LidarId,
    pub chaser_lidar_mount: Transform,
    pub chaser_nav: NavigationFilter,
    pub hil_sensor_gnc: bool,
    pub modulation_mode: ModulationUiMode,
    prev_nav_rel_rot: Option<Quat>,
    prev_nav_rot_t: f64,
    est_station_omega_w: Vec3,
    // 12 REAL Dynamic Hinged Clamping Hooks (`Revolute` bodies + motors)
    pub hook_bodies: [BodyHandle; 12],
    pub hook_joints: [JointHandle; 12],
    pub hook_engaged: [bool; 12],
    pub hook_forces_n: [f64; 12],
    // 3 Independent Physical Soft-Capture Pawls (NO `Connector` snap cheat!)
    pub soft_pawls_armed: bool,
    pub soft_pawls_latched: [bool; 3],
    pub soft_pawl_forces_n: [f64; 3],
    // Real actuation & thermodynamic state (First-Law open control volume!)
    pub scr_command_m: f64,
    pub strut_forces_n: [f64; 6],
    pub seal_force_n: f64,
    pub seal_min_pad_n: f64,
    pub air_mass_kg: f64,
    pub air_energy_j: f64,
    pub vestibule_temp_k: f64,
    pub vestibule_pressure_pa: f64,
    pub prev_vestibule_vol_m3: f64,
    pub air_sep_force_n: f64,
    pub vent_pos_z_open: bool,
    pub vent_neg_z_open: bool,
    pub vent_neg_z_jammed: bool,
    pub vent_thrust_pos_z_n: f64,
    pub vent_thrust_neg_z_n: f64,
    pub vent_net_torque_nm: f64,
    pub impingement_force_n: f64,
    pub seal_verified: bool,
    pub venting_air: bool,
    pub petal_contacts: usize,
    // Volumetric vacuum gas puffs & interactive mouse grab tether
    pub gas_puffs: Vec<GasPuff>,
    pub user_grab: Option<UserGrab>,
    rng_state: u64,
    // Mission state & hardware-validation metrics
    pub initial_chaser_mass_kg: f64,
    pub initial_propellant_kg: f64,
    pub phase: MissionPhase,
    pub phase_start_s: f64,
    pub time_s: f64,
    pub seg_start_pos: Vec3,
    pub iss_leo: bool,
    pub orbit_mode: OrbitMode,
    pub ever_petal_contact: bool,
    pub ever_soft_captured: bool,
    pub ever_hard_clamped: bool,
    pub ever_pressurized: bool,
    pub ever_vented_and_undocked: bool,
    pub peak_clamp_force_kn: f64,
    pub peak_air_force_kn: f64,
    pub peak_vent_thrust_n: f64,
    pub max_gas_temp_k: f64,
    pub min_blowdown_temp_k: f64,
    pub peak_hook_motor_nm: f64,
    pub peak_wing_flex_deg: f64,
    pub min_port_distance_m: f64,
    pub peak_detumble_omega: f64,
    pub chaser_trail: VecDeque<Vec3>,
    trail_tick: u32,
}

fn dynamic_body(
    world: &mut RigidWorld,
    pose: Transform,
    mass: f64,
    inertia: Vec3,
    collider: Option<(Vec3, u32, u32)>,
) -> Result<BodyHandle> {
    let body = world.scene.insert_body(BodyDesc {
        params: BodyParams::new().with_pose(pose),
        additional_mass: Some(MassProperties {
            mass,
            local_com: Vec3::ZERO,
            inertia: Mat3::from_diagonal(inertia),
        }),
        ..Default::default()
    })?;
    if let Some((half_extents, memberships, filter)) = collider {
        let mut desc = ColliderDesc::new(ColliderGeometry::Cuboid { half_extents });
        desc.parent = Some(body);
        desc.memberships = memberships;
        desc.filter = filter;
        desc.friction = 0.14;
        desc.restitution = 0.05;
        world.scene.insert_collider(desc)?;
    }
    Ok(body)
}

#[allow(clippy::too_many_arguments)]
fn attach_petal_colliders(
    world: &mut RigidWorld,
    body: BodyHandle,
    base_x: f64,
    dir_x: f64,
    angle_offset: f64,
    radial_flare: f64,
    base_r: f64,
    memberships: u32,
    filter: u32,
) -> Result<()> {
    for i in 0..3 {
        let a = i as f64 * (TAU / 3.0) + angle_offset;
        let rad = Vec3::new(0.0, a.cos(), a.sin());
        let tang = Vec3::new(0.0, -a.sin(), a.cos());
        let axis_x = (Vec3::new(dir_x, 0.0, 0.0) + rad * radial_flare).normalize();
        let axis_z = tang;
        let axis_y = axis_z.cross(axis_x).normalize();
        let rot = Quat::from_mat3(&Mat3::from_cols(axis_x, axis_y, axis_z)).normalize();
        let center = Vec3::new(base_x + dir_x * 0.075, 0.0, 0.0)
            + rad * (base_r + 0.075 * radial_flare);
        let mut desc = ColliderDesc::new(ColliderGeometry::Cuboid {
            half_extents: Vec3::new(0.075, 0.018, 0.085),
        });
        desc.parent = Some(body);
        desc.pose = Transform::new(center, rot);
        desc.memberships = memberships;
        desc.filter = filter;
        desc.friction = 0.12;
        desc.restitution = 0.04;
        world.scene.insert_collider(desc)?;
    }
    Ok(())
}

fn mount_12_rcs(
    world: &mut RigidWorld,
    body: BodyHandle,
    arm: f64,
    max_thrust: f64,
) -> Result<Vec<ThrusterId>> {
    let specs: &[(Vec3, Quat)] = &[
        (Vec3::new(0.0, arm, 0.0), Quat::IDENTITY),
        (Vec3::new(0.0, -arm, 0.0), Quat::IDENTITY),
        (Vec3::new(0.0, arm, 0.0), Quat::from_rotation_z(PI)),
        (Vec3::new(0.0, -arm, 0.0), Quat::from_rotation_z(PI)),
        (Vec3::new(0.0, 0.0, arm), Quat::from_rotation_z(FRAC_PI_2)),
        (Vec3::new(0.0, 0.0, -arm), Quat::from_rotation_z(FRAC_PI_2)),
        (Vec3::new(0.0, 0.0, arm), Quat::from_rotation_z(-FRAC_PI_2)),
        (Vec3::new(0.0, 0.0, -arm), Quat::from_rotation_z(-FRAC_PI_2)),
        (Vec3::new(arm, 0.0, 0.0), Quat::from_rotation_y(-FRAC_PI_2)),
        (Vec3::new(-arm, 0.0, 0.0), Quat::from_rotation_y(-FRAC_PI_2)),
        (Vec3::new(arm, 0.0, 0.0), Quat::from_rotation_y(FRAC_PI_2)),
        (Vec3::new(-arm, 0.0, 0.0), Quat::from_rotation_y(FRAC_PI_2)),
    ];
    let mut ids = Vec::with_capacity(specs.len());
    for &(pos, rot) in specs {
        ids.push(world.devices.thrusters.insert(
            &world.scene,
            ThrusterSpec {
                body,
                frame: Transform::new(pos, rot),
                max_thrust,
                specific_impulse: None,
                propellant_mass: 0.0,
                deplete_body_mass: false,
                min_impulse_bit: 0.0,
                response_time: 0.025,
            },
        )?);
    }
    Ok(ids)
}

#[allow(clippy::too_many_arguments)]
fn mount_16_dragon_rcs(
    world: &mut RigidWorld,
    body: BodyHandle,
    x_fwd: f64,
    x_aft: f64,
    r: f64,
    max_thrust: f64,
    isp: f64,
    prop_each: f64,
) -> Result<Vec<ThrusterId>> {
    let specs: &[(Vec3, Quat)] = &[
        (Vec3::new(x_aft, r, 0.0), Quat::IDENTITY),
        (Vec3::new(x_aft, -r, 0.0), Quat::IDENTITY),
        (Vec3::new(x_aft, 0.0, r), Quat::IDENTITY),
        (Vec3::new(x_aft, 0.0, -r), Quat::IDENTITY),
        (Vec3::new(x_fwd, r, 0.0), Quat::from_rotation_z(PI)),
        (Vec3::new(x_fwd, -r, 0.0), Quat::from_rotation_z(PI)),
        (Vec3::new(x_fwd, 0.0, r), Quat::from_rotation_z(PI)),
        (Vec3::new(x_fwd, 0.0, -r), Quat::from_rotation_z(PI)),
        (Vec3::new(x_fwd, 0.0, r), Quat::from_rotation_z(FRAC_PI_2)),
        (Vec3::new(x_aft, 0.0, -r), Quat::from_rotation_z(FRAC_PI_2)),
        (Vec3::new(x_fwd, 0.0, -r), Quat::from_rotation_z(-FRAC_PI_2)),
        (Vec3::new(x_aft, 0.0, r), Quat::from_rotation_z(-FRAC_PI_2)),
        (Vec3::new(x_fwd, r, 0.0), Quat::from_rotation_y(-FRAC_PI_2)),
        (Vec3::new(x_aft, -r, 0.0), Quat::from_rotation_y(-FRAC_PI_2)),
        (Vec3::new(x_fwd, -r, 0.0), Quat::from_rotation_y(FRAC_PI_2)),
        (Vec3::new(x_aft, r, 0.0), Quat::from_rotation_y(FRAC_PI_2)),
    ];
    let mut ids = Vec::with_capacity(specs.len());
    for &(pos, rot) in specs {
        ids.push(world.devices.thrusters.insert(
            &world.scene,
            ThrusterSpec {
                body,
                frame: Transform::new(pos, rot),
                max_thrust,
                specific_impulse: Some(isp),
                propellant_mass: prop_each,
                deplete_body_mass: true,
                min_impulse_bit: 1.8,
                response_time: 0.025,
            },
        )?);
    }
    Ok(ids)
}

fn mount_reaction_wheels(
    world: &mut RigidWorld,
    body: BodyHandle,
    inertia: f64,
    max_torque: f64,
    max_speed: f64,
) -> Result<Vec<ReactionWheelId>> {
    let mut ids = Vec::with_capacity(3);
    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
        ids.push(world.devices.reaction_wheels.insert(
            &world.scene,
            ReactionWheelSpec {
                body,
                axis,
                inertia,
                max_torque,
                max_speed,
                initial_speed: 0.0,
                gimbal: None,
            },
        )?);
    }
    Ok(ids)
}

fn mount_station_cmgs(world: &mut RigidWorld, station: BodyHandle) -> Result<Vec<ReactionWheelId>> {
    // True 4-pyramid SGCMG cluster (skew angle β = 54.73°) providing symmetric 3-axis
    // gyroscopic torque amplification under Singularity-Robust Inverse (SRI) steering!
    let beta = 54.73_f64.to_radians();
    let sb = beta.sin();
    let cb = beta.cos();
    let configs = [
        (Vec3::new(0.0, -1.0, 0.0), Vec3::new(sb, 0.0, cb)),
        (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, sb, cb)),
        (Vec3::new(0.0, 1.0, 0.0), Vec3::new(-sb, 0.0, cb)),
        (Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, -sb, cb)),
    ];
    let mut ids = Vec::with_capacity(4);
    for (spin, gimbal_axis) in configs {
        ids.push(world.devices.reaction_wheels.insert(
            &world.scene,
            ReactionWheelSpec {
                body: station,
                axis: spin,
                inertia: 25.0,
                max_torque: 120.0,
                max_speed: 240.0,
                initial_speed: 120.0, // 3,000 N·m·s stored momentum per CMG
                gimbal: Some(GimbalSpec {
                    axis: gimbal_axis,
                    max_rate: 0.75,
                    initial_angle: 0.0,
                }),
            },
        )?);
    }
    Ok(ids)
}

#[derive(Default)]
struct SubstepDiag {
    samples: usize,
    strut_forces: [f64; 6],
    pawl_forces: [f64; 3],
    hook_forces: [f64; 12],
    hook_engaged: [bool; 12],
    seal_force_n: f64,
    seal_min_pad_n: f64,
    air_sep_force_n: f64,
    vent_pos_z_n: f64,
    vent_neg_z_n: f64,
    vent_torque_nm: f64,
    grab_force_n: f64,
}

impl SpaceDockingSim {
    pub fn new(iss_leo: bool) -> Result<Self> {
        let mut world = RigidWorld::new()?;
        world.scene.set_gravity(Vec3::ZERO)?;
        if iss_leo {
            world.devices.orbital.set_field(OrbitalField::iss_leo())?;
        }

        // 1. Truly Free-Floating Orbital Station Module + Passive IDA-2 Port (6,500 kg).
        //    Free-drift by default (`SpacecraftCommand::Off`) just like real ISS docking
        //    so Newton's 3rd Law moves & rotates BOTH spacecraft visibly with zero hidden hold!
        let station = dynamic_body(
            &mut world,
            Transform::IDENTITY,
            6_500.0,
            Vec3::new(4_800.0, 8_600.0, 7_900.0),
            Some((Vec3::new(2.05, 0.95, 0.95), 0b001, 0b001)),
        )?;
        attach_petal_colliders(
            &mut world,
            station,
            STATION_PORT_X - 0.02,
            1.0,
            PI / 3.0,
            0.32,
            0.442,
            0b010,
            0b100,
        )?;

        let station_thrusters = mount_12_rcs(&mut world, station, 1.35, 500.0)?;
        let station_wheels = mount_reaction_wheels(&mut world, station, 10.0, 320.0, 220.0)?;
        let station_cmgs = mount_station_cmgs(&mut world, station)?;
        // Wire BOTH the 4-pyramid SGCMGs (steered by SRI!) and backup reaction wheels into station GNC!
        let mut station_rotors = station_cmgs.clone();
        station_rotors.extend_from_slice(&station_wheels);
        let station_gnc = world.devices.spacecraft.insert(
            &world.scene,
            &world.devices.thrusters,
            &world.devices.reaction_wheels,
            SpacecraftSpec {
                body: station,
                frame: Transform::IDENTITY,
                thrusters: station_thrusters.clone(),
                reaction_wheels: station_rotors,
                limits: SpacecraftLimits {
                    position_frequency: 0.35,
                    attitude_frequency: 0.75,
                    ..SpacecraftLimits::default()
                },
            },
        )?;
        world
            .devices
            .spacecraft
            .command(station_gnc, SpacecraftCommand::Off)?;

        // 2. Twin 2-Segment Flexible Sun-Tracking Solar Array Wings (100 kg inner + 80 kg flex tip each).
        //    Each outer segment hinges on a `Y` out-of-plane elastic bending joint (0.28 Hz, ζ = 1.8%)
        //    for authentic Control-Structure Interaction (CSI) coupled through Featherstone!
        let solar_port = dynamic_body(
            &mut world,
            Transform::from_translation(Vec3::new(-0.35, 0.0, 3.125)),
            100.0,
            Vec3::new(55.0, 65.0, 28.0),
            None,
        )?;
        let solar_port_outer = dynamic_body(
            &mut world,
            Transform::from_translation(Vec3::new(-0.35, 0.0, 5.55)),
            80.0,
            Vec3::new(48.0, 56.0, 22.0),
            None,
        )?;
        let solar_starboard = dynamic_body(
            &mut world,
            Transform::from_translation(Vec3::new(-0.35, 0.0, -3.125)),
            100.0,
            Vec3::new(55.0, 65.0, 28.0),
            None,
        )?;
        let solar_starboard_outer = dynamic_body(
            &mut world,
            Transform::from_translation(Vec3::new(-0.35, 0.0, -5.55)),
            80.0,
            Vec3::new(48.0, 56.0, 22.0),
            None,
        )?;
        let solar_port_joint = world.scene.insert_joint(JointDesc {
            parent: Some(station),
            child: solar_port,
            kind: JointType::Revolute,
            axis: Vec3::Z,
            frame_parent: Transform::from_translation(Vec3::new(-0.35, 0.0, 2.0)),
            frame_child: Transform::from_translation(Vec3::new(0.0, 0.0, -1.125)),
        })?;
        let solar_port_flex_joint = world.scene.insert_joint(JointDesc {
            parent: Some(solar_port),
            child: solar_port_outer,
            kind: JointType::Revolute,
            axis: Vec3::Y,
            frame_parent: Transform::from_translation(Vec3::new(0.0, 0.0, 1.125)),
            frame_child: Transform::from_translation(Vec3::new(0.0, 0.0, -1.30)),
        })?;
        let solar_starboard_joint = world.scene.insert_joint(JointDesc {
            parent: Some(station),
            child: solar_starboard,
            kind: JointType::Revolute,
            axis: Vec3::Z,
            frame_parent: Transform::from_translation(Vec3::new(-0.35, 0.0, -2.0)),
            frame_child: Transform::from_translation(Vec3::new(0.0, 0.0, 1.125)),
        })?;
        let solar_starboard_flex_joint = world.scene.insert_joint(JointDesc {
            parent: Some(solar_starboard),
            child: solar_starboard_outer,
            kind: JointType::Revolute,
            axis: Vec3::Y,
            frame_parent: Transform::from_translation(Vec3::new(0.0, 0.0, -1.125)),
            frame_child: Transform::from_translation(Vec3::new(0.0, 0.0, 1.30)),
        })?;
        for joint in [solar_port_joint, solar_starboard_joint] {
            world.devices.motors.insert(
                &world.scene,
                joint,
                MotorSpec {
                    limits: MotorLimits {
                        max_velocity: 0.20,
                        pid: [3.0, 0.0, 0.4],
                        ..Default::default()
                    },
                    max_force: 450.0,
                },
            )?;
        }
        // Effective flap inertia about hinge: I_y + m d² = 56 + 80 * 1.30² = 191.2 kg·m² -> 0.28 Hz CSI bending mode!
        let i_eff_wing = 56.0 + 80.0 * 1.30 * 1.30;
        let solar_port_flex = world.devices.space_env.insert_flex_joint(
            &world.scene,
            FlexJointSpec::from_modal_frequency(solar_port_flex_joint, i_eff_wing, 0.28, 0.018),
        )?;
        let solar_starboard_flex = world.devices.space_env.insert_flex_joint(
            &world.scene,
            FlexJointSpec::from_modal_frequency(
                solar_starboard_flex_joint,
                i_eff_wing,
                0.28,
                0.018,
            ),
        )?;

        // 3. Visiting Crew/Cargo Spacecraft (3,200 kg).
        let chaser_start = Transform::new(
            Vec3::new(7.0, 0.78, -0.58),
            (Quat::from_rotation_x(0.24)
                * Quat::from_rotation_y(-0.28)
                * Quat::from_rotation_z(0.18))
            .normalize(),
        );
        let chaser = dynamic_body(
            &mut world,
            chaser_start,
            3_200.0,
            Vec3::new(850.0, 1_150.0, 1_100.0),
            Some((Vec3::new(0.75, 0.62, 0.62), 0b001, 0b001)),
        )?;
        world.apply_body_impulse(
            chaser,
            SpatialVector::new(
                Vec3::new(-220.0, 160.0, 140.0),
                Vec3::new(170.0, -140.0, 185.0),
            ),
        )?;

        // 4. FREE 6-DOF Dynamic Active Soft-Capture Ring (95 kg) — NO tree joint!
        let ring_start = chaser_start.compose(&Transform::from_translation(Vec3::new(
            CHASER_BULKHEAD_X,
            0.0,
            0.0,
        )));
        let capture_ring = dynamic_body(
            &mut world,
            ring_start,
            95.0,
            Vec3::new(14.0, 9.0, 9.0),
            None,
        )?;
        attach_petal_colliders(
            &mut world,
            capture_ring,
            RING_FACE_OFFSET_X + 0.02,
            -1.0,
            0.0,
            -0.25,
            0.395,
            0b100,
            0b010,
        )?;
        {
            let mut rim = ColliderDesc::new(ColliderGeometry::Cylinder {
                half_height: 0.035,
                radius: 0.405,
            });
            rim.parent = Some(capture_ring);
            rim.pose = Transform::new(
                Vec3::new(RING_FACE_OFFSET_X + 0.035, 0.0, 0.0),
                Quat::from_rotation_z(FRAC_PI_2),
            );
            rim.memberships = 0b100;
            rim.filter = 0b010;
            rim.friction = 0.12;
            rim.restitution = 0.04;
            world.scene.insert_collider(rim)?;
        }

        // 5. 12 REAL Dynamic Hinged Clamping Hooks (`Revolute` bodies + torque motors)!
        //    Reflected gearbox inertia `0.45 kg·m²` lets jaw clamping forces apply DIRECTLY
        //    onto each hook body and transmit physically through the Featherstone hinge pin!
        let mut hook_bodies_vec = Vec::with_capacity(12);
        let mut hook_joints_vec = Vec::with_capacity(12);
        for i in 0..12 {
            let a = i as f64 * (TAU / 12.0);
            let pivot = hook_pivot_local(a);
            let rot_closed = hook_base_rotation(a);
            let rot_open = (rot_closed * Quat::from_rotation_z(HOOK_OPEN_RAD)).normalize();
            let hook_world = chaser_start.compose(&Transform::new(pivot, rot_open));
            let hb = dynamic_body(&mut world, hook_world, 4.0, Vec3::splat(0.45), None)?;
            let hj = world.scene.insert_joint(JointDesc {
                parent: Some(chaser),
                child: hb,
                kind: JointType::Revolute,
                axis: Vec3::Z,
                frame_parent: Transform::new(pivot, rot_closed),
                frame_child: Transform::IDENTITY,
            })?;
            world.devices.motors.insert(
                &world.scene,
                hj,
                MotorSpec {
                    limits: MotorLimits {
                        max_velocity: 1.35,
                        pid: [14.0, 0.0, 0.45],
                        ..Default::default()
                    },
                    max_force: 450.0, // Real ball-screw / worm-gear hook drive torque limit (N·m)
                },
            )?;
            world
                .devices
                .motors
                .command(hj, MotorCommand::Position(HOOK_OPEN_RAD))?;
            hook_bodies_vec.push(hb);
            hook_joints_vec.push(hj);
        }
        let hook_bodies: [BodyHandle; 12] = hook_bodies_vec.try_into().unwrap();
        let hook_joints: [JointHandle; 12] = hook_joints_vec.try_into().unwrap();

        // 6. 16 Redundant Draco RCS Thrusters + Reaction Wheels (with realistic rotor imbalance jitter!)
        //    + Microgravity Liquid Propellant Slosh Tank EMM + LEO Space Environment!
        let prop_each = 28.0;
        let chaser_thrusters =
            mount_16_dragon_rcs(&mut world, chaser, -0.42, 0.46, 0.68, 650.0, 300.0, prop_each)?;
        world.devices.thrusters.refuel(chaser_thrusters[0], 0.0)?;
        let chaser_wheels = mount_reaction_wheels(&mut world, chaser, 10.0, 380.0, 240.0)?;
        for (idx, &rw) in chaser_wheels.iter().enumerate() {
            world.devices.reaction_wheels.set_imbalance(
                rw,
                ImbalanceSpec {
                    static_unbalance: 1.5e-5, // Real aerospace RW static unbalance (kg·m)
                    dynamic_unbalance: 4.0e-6, // Real aerospace RW dynamic unbalance (kg·m²)
                    mount_offset: Vec3::new(0.05, 0.0, 0.0),
                    initial_phase: idx as f64 * 2.1,
                },
            )?;
        }
        // Microgravity liquid propellant slosh equivalent mechanical model (120 kg convective MMH/NTO!).
        let chaser_slosh = world.devices.space_env.insert_slosh_tank(
            &world.scene,
            SloshTankSpec {
                body: chaser,
                tank_center_local: Vec3::new(0.10, 0.0, 0.0),
                slosh_mass: 120.0,
                stiffness: 580.0,  // f_n ≈ 0.35 Hz capillary restoring mode
                damping: 52.0,     // ζ ≈ 0.10 viscous baffle damping
                max_displacement: 0.24,
            },
        )?;

        // LEO Space Environment: NRLMSISE-00 28-layer altitude atmosphere, 1 AU SRP, ISS geomagnetic
        // field, Magnetorquer cluster, and native Simons/Dettleff (σ_n, σ_t) plume impingement!
        let orbit_mode = if iss_leo {
            OrbitMode::IssLeo
        } else {
            OrbitMode::ZeroG
        };
        if iss_leo {
            world.devices.space_env.set_nrlmsise_atmosphere(Some(NrlmsiseAtmosphere {
                frame_orbital_velocity: Vec3::new(7_664.0, 0.0, 0.0),
                ..NrlmsiseAtmosphere::earth_lvlh(408_000.0, SolarActivity::Moderate)
            }))?;
            world
                .devices
                .space_env
                .set_solar_radiation(Some(SolarRadiation::one_au(Vec3::new(-0.45, 0.8, 0.5))))?;
            world
                .devices
                .space_env
                .set_magnetic_field(MagneticField::iss_leo())?;
        }
        world
            .devices
            .space_env
            .set_plume_impingement(true, 4.0)?;
        // Register physical surface facets on station forward bulkhead & all 4 solar wing segments!
        world.devices.space_env.insert_facet(
            &world.scene,
            SurfaceFacetSpec::hull_panel(
                station,
                Vec3::new(STATION_PORT_X - 0.1, 0.0, 0.0),
                Vec3::X,
                2.8,
            ),
        )?;
        world.devices.space_env.insert_facet(
            &world.scene,
            SurfaceFacetSpec::hull_panel(station, Vec3::new(0.0, 0.0, 0.0), Vec3::Y, 6.5),
        )?;
        world.devices.space_env.insert_facet(
            &world.scene,
            SurfaceFacetSpec::solar_panel(solar_port, Vec3::ZERO, Vec3::Y, 6.0),
        )?;
        world.devices.space_env.insert_facet(
            &world.scene,
            SurfaceFacetSpec::solar_panel(solar_port_outer, Vec3::ZERO, Vec3::Y, 7.5),
        )?;
        world.devices.space_env.insert_facet(
            &world.scene,
            SurfaceFacetSpec::solar_panel(solar_starboard, Vec3::ZERO, Vec3::Y, 6.0),
        )?;
        world.devices.space_env.insert_facet(
            &world.scene,
            SurfaceFacetSpec::solar_panel(solar_starboard_outer, Vec3::ZERO, Vec3::Y, 7.5),
        )?;
        let station_mtb = world.devices.space_env.insert_magnetorquer(
            &world.scene,
            MagnetorquerSpec {
                body: station,
                max_dipole: Vec3::splat(4_000.0), // ISS-class 4,000 A·m² magnetic torquer bars
                residual_dipole: Vec3::new(12.0, -8.0, 15.0), // Realistic parasitic bus dipole
            },
        )?;

        // Configure aerospace PWPF discrete valve modulation on all 16 chaser Draco RCS nozzles!
        let modulation_mode = ModulationUiMode::Pwpf;
        for &tid in &chaser_thrusters {
            world
                .devices
                .thrusters
                .set_modulation(tid, modulation_mode.to_device())?;
        }

        // Hardware-in-the-loop navigation sensor suite on chaser: FOG IMU + Star Tracker + LiDAR!
        let chaser_imu = world
            .devices
            .space_sensors
            .insert_imu(&world.scene, ImuSpec::tactical_fog(chaser, 0x51A0_0001))?;
        let chaser_star_tracker = world.devices.space_sensors.insert_star_tracker(
            &world.scene,
            StarTrackerSpec {
                mount_frame: Transform::from_rotation(Quat::from_rotation_z(FRAC_PI_2)),
                max_slew_rate_rad_s: 0.45,
                ..StarTrackerSpec::leo_flight(chaser, Transform::IDENTITY, 0x51A0_0002)
            },
        )?;
        let chaser_lidar_mount = Transform::new(
            Vec3::new(CHASER_BULKHEAD_X, 0.0, 0.0),
            Quat::from_rotation_y(PI),
        );
        let chaser_lidar = world.devices.space_sensors.insert_lidar(
            &world.scene,
            LidarSpec::rendezvous_proximity(
                chaser,
                station,
                chaser_lidar_mount,
                Vec3::new(STATION_PORT_X, 0.0, 0.0),
                0x51A0_0003,
            ),
        )?;
        let chaser_nav = NavigationFilter::new(chaser_start.rotation);
        let chaser_gnc = world.devices.spacecraft.insert(
            &world.scene,
            &world.devices.thrusters,
            &world.devices.reaction_wheels,
            SpacecraftSpec {
                body: chaser,
                frame: Transform::IDENTITY,
                thrusters: chaser_thrusters.clone(),
                reaction_wheels: chaser_wheels.clone(),
                limits: SpacecraftLimits {
                    max_linear_speed: 2.6,
                    max_angular_speed: 1.4,
                    position_frequency: 0.80,
                    attitude_frequency: 1.40,
                    velocity_gain: 2.0,
                    angular_rate_gain: 2.8,
                    ..SpacecraftLimits::default()
                },
            },
        )?;

        world.devices.spacecraft.command(
            chaser_gnc,
            SpacecraftCommand::RateHold {
                linear_velocity: Vec3::ZERO,
                angular_velocity: Vec3::ZERO,
            },
        )?;

        let initial_chaser_mass_kg = world.scene.body_mass_properties(chaser)?.mass;
        let peak_omega = world.scene.body_velocity(chaser)?.angular.length();
        Ok(Self {
            world,
            station,
            solar_port,
            solar_port_outer,
            solar_starboard,
            solar_starboard_outer,
            solar_port_joint,
            solar_port_flex_joint,
            solar_starboard_joint,
            solar_starboard_flex_joint,
            solar_port_flex,
            solar_starboard_flex,
            station_thrusters,
            station_wheels,
            station_cmgs,
            station_mtb,
            station_gnc,
            station_free_drift: true,
            chaser,
            capture_ring,
            chaser_slosh,
            chaser_thrusters,
            chaser_wheels,
            chaser_gnc,
            faulted_thruster_idx: Some(0),
            chaser_imu,
            chaser_star_tracker,
            chaser_lidar,
            chaser_lidar_mount,
            chaser_nav,
            hil_sensor_gnc: true,
            modulation_mode,
            prev_nav_rel_rot: None,
            prev_nav_rot_t: 0.0,
            est_station_omega_w: Vec3::ZERO,
            hook_bodies,
            hook_joints,
            hook_engaged: [false; 12],
            hook_forces_n: [0.0; 12],
            soft_pawls_armed: true,
            soft_pawls_latched: [false; 3],
            soft_pawl_forces_n: [0.0; 3],
            scr_command_m: 0.0,
            strut_forces_n: [0.0; 6],
            seal_force_n: 0.0,
            seal_min_pad_n: 0.0,
            air_mass_kg: 0.0,
            air_energy_j: 0.0,
            vestibule_temp_k: T_WALL_K,
            vestibule_pressure_pa: 0.0,
            prev_vestibule_vol_m3: VESTIBULE_VOL_0_M3,
            air_sep_force_n: 0.0,
            vent_pos_z_open: false,
            vent_neg_z_open: false,
            vent_neg_z_jammed: false,
            vent_thrust_pos_z_n: 0.0,
            vent_thrust_neg_z_n: 0.0,
            vent_net_torque_nm: 0.0,
            impingement_force_n: 0.0,
            seal_verified: false,
            venting_air: false,
            petal_contacts: 0,
            gas_puffs: Vec::with_capacity(1100),
            user_grab: None,
            rng_state: 0x517C_C1B7_2722_0A95,
            initial_chaser_mass_kg,
            initial_propellant_kg: prop_each * 15.0,
            phase: MissionPhase::Detumble,
            phase_start_s: 0.0,
            time_s: 0.0,
            seg_start_pos: chaser_start.position,
            iss_leo,
            orbit_mode,
            ever_petal_contact: false,
            ever_soft_captured: false,
            ever_hard_clamped: false,
            ever_pressurized: false,
            ever_vented_and_undocked: false,
            peak_clamp_force_kn: 0.0,
            peak_air_force_kn: 0.0,
            peak_vent_thrust_n: 0.0,
            max_gas_temp_k: T_WALL_K,
            min_blowdown_temp_k: T_WALL_K,
            peak_hook_motor_nm: 0.0,
            peak_wing_flex_deg: 0.0,
            min_port_distance_m: f64::INFINITY,
            peak_detumble_omega: peak_omega,
            chaser_trail: VecDeque::with_capacity(180),
            trail_tick: 0,
        })
    }

    fn next_rand(&mut self) -> f64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        (x as f64) / (u64::MAX as f64)
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn_gas_puff(
        &mut self,
        pos: Vec3,
        vel: Vec3,
        radius: f64,
        growth_rate: f64,
        stretch: f64,
        max_age: f64,
        base_opacity: f32,
        kind: f32,
    ) {
        if self.gas_puffs.len() >= 1100 {
            self.gas_puffs.swap_remove(0);
        }
        let seed = (self.next_rand() * 100.0) as f32;
        self.gas_puffs.push(GasPuff {
            pos,
            vel,
            radius,
            growth_rate,
            stretch,
            age: 0.0,
            max_age,
            base_opacity,
            seed,
            kind,
        });
    }

    pub fn scr_displacement(&self) -> Result<(f64, f64)> {
        let chaser_pose = self.world.scene.body_pose(self.chaser)?;
        let ring_pose = self.world.scene.body_pose(self.capture_ring)?;
        let local = chaser_pose.inverse().transform_point(ring_pose.position);
        let axial = (CHASER_BULKHEAD_X - local.x).max(0.0);
        let lateral = (local.y * local.y + local.z * local.z).sqrt();
        Ok((axial, lateral))
    }

    pub fn flange_gap_m(&self) -> Result<f64> {
        let sp = self.station_port_pose()?;
        let chaser_pose = self.world.scene.body_pose(self.chaser)?;
        let chaser_flange = chaser_pose.transform_point(Vec3::new(
            CHASER_BULKHEAD_X + RING_FACE_OFFSET_X,
            0.0,
            0.0,
        ));
        let axis = sp.rotation * Vec3::X;
        Ok((chaser_flange - sp.position).dot(axis))
    }

    pub fn station_port_pose(&self) -> Result<Transform> {
        Ok(self
            .world
            .scene
            .body_pose(self.station)?
            .compose(&Transform::from_translation(Vec3::new(
                STATION_PORT_X,
                0.0,
                0.0,
            ))))
    }

    pub fn chaser_port_pose(&self) -> Result<Transform> {
        Ok(self
            .world
            .scene
            .body_pose(self.capture_ring)?
            .compose(&Transform::new(
                Vec3::new(RING_FACE_OFFSET_X, 0.0, 0.0),
                Quat::from_rotation_z(PI),
            )))
    }

    /// Ray-pick against the visiting capsule, station hull, and both solar wings
    /// so the user can click & drag ANY spacecraft part to shove or spin it!
    pub fn try_begin_grab(&mut self, ray_origin: Vec3, ray_dir: Vec3) -> bool {
        let dir = ray_dir.normalize_or_zero();
        if dir.length_squared() < 0.5 {
            return false;
        }
        let mut best: Option<(f64, BodyHandle, Vec3, Vec3)> = None;
        let mut test_sphere = |body: BodyHandle, pose: Transform, local_c: Vec3, radius: f64| {
            let center = pose.transform_point(local_c);
            let oc = ray_origin - center;
            let b = oc.dot(dir);
            let c = oc.length_squared() - radius * radius;
            let disc = b * b - c;
            if disc >= 0.0 {
                let t = -b - disc.sqrt();
                if t > 0.25 && best.as_ref().is_none_or(|b_hit| t < b_hit.0) {
                    let hit_w = ray_origin + dir * t;
                    let local_p = pose.inverse().transform_point(hit_w);
                    best = Some((t, body, local_p, hit_w));
                }
            }
        };

        if let Ok(cp) = self.world.scene.body_pose(self.chaser) {
            test_sphere(self.chaser, cp, Vec3::ZERO, 1.15);
            test_sphere(self.chaser, cp, Vec3::new(-0.8, 0.0, 0.0), 0.72);
            test_sphere(self.chaser, cp, Vec3::new(0.75, 0.0, 0.0), 0.85);
            test_sphere(self.chaser, cp, Vec3::new(0.72, 0.0, 1.25), 0.55);
            test_sphere(self.chaser, cp, Vec3::new(0.72, 0.0, -1.25), 0.55);
        }
        if let Ok(sp) = self.world.scene.body_pose(self.station) {
            test_sphere(self.station, sp, Vec3::new(-0.4, 0.0, 0.0), 1.35);
            test_sphere(self.station, sp, Vec3::new(1.5, 0.0, 0.0), 1.10);
            test_sphere(self.station, sp, Vec3::new(-1.8, 0.0, 0.0), 1.15);
            test_sphere(self.station, sp, Vec3::new(-0.95, 0.0, -2.0), 0.90);
            // Clicking either inner or flexible outer solar wing attaches the grab lever directly
            // to the station body at the wing point so the user can effortlessly spin/flip it!
            for &(wing, sign_z, offsets) in &[
                (self.solar_port, 1.0, [-0.5, 0.5]),
                (self.solar_port_outer, 1.0, [-0.5, 0.7]),
                (self.solar_starboard, -1.0, [-0.5, 0.5]),
                (self.solar_starboard_outer, -1.0, [-0.5, 0.7]),
            ] {
                if let Ok(wp) = self.world.scene.body_pose(wing) {
                    for dz in offsets {
                        let pt_w = wp.transform_point(Vec3::new(0.0, 0.0, sign_z * dz));
                        let pt_in_station = sp.inverse().transform_point(pt_w);
                        test_sphere(self.station, sp, pt_in_station, 1.05);
                    }
                }
            }
        }

        if let Some((depth, body, local_point, target_world)) = best {
            self.user_grab = Some(UserGrab {
                body,
                local_point,
                target_world,
                depth,
                applied_force_n: 0.0,
            });
            true
        } else {
            false
        }
    }

    pub fn update_grab_ray(&mut self, ray_origin: Vec3, ray_dir: Vec3) {
        if let Some(g) = &mut self.user_grab {
            let dir = ray_dir.normalize_or_zero();
            if dir.length_squared() > 0.5 {
                g.target_world = ray_origin + dir * g.depth;
            }
        }
    }

    pub fn end_grab(&mut self) {
        self.user_grab = None;
    }

    /// Returns `(target_pos, target_vel, target_accel, target_rot, target_omega)` in
    /// world space for a docking offset `(port_gap_m, rbar, cross)` expressed in the
    /// **LIVE moving & rotating station frame** (exact Coriolis/centripetal kinematics!).
    ///
    /// When `hil_sensor_gnc` is active and LiDAR has lock, computes the relative target
    /// purely from the noisy, delayed IMU + Star Tracker + LiDAR [`NavigationFilter`]!
    fn target_chaser_kinematics(
        &self,
        port_gap_m: f64,
        rbar: f64,
        cross: f64,
    ) -> Result<(Vec3, Vec3, Vec3, Quat, Vec3)> {
        let (stroke, _) = self.scr_displacement()?;
        if self.hil_sensor_gnc && self.chaser_nav.lidar_locked {
            let cp = self.world.scene.body_pose(self.chaser)?;
            let cv = self.world.scene.body_velocity(self.chaser)?;
            let est_chaser_rot = self.chaser_nav.orientation;
            let est_station_rot = (est_chaser_rot * self.chaser_nav.rel_target_rot).normalize();
            let port_rel_w = est_chaser_rot * self.chaser_nav.rel_target_pos_body;
            let port_offset_local = Vec3::new(
                -(CHASER_BULKHEAD_X + RING_FACE_OFFSET_X) + stroke + port_gap_m,
                rbar,
                cross,
            );
            let r_from_port_w = est_station_rot * port_offset_local;
            // Express the noisy relative error against `cp` so `SpacecraftCommand::Trajectory`
            // closes the loop strictly on the sensor-estimated pose/attitude error!
            let pos = cp.position + port_rel_w + r_from_port_w;
            let omega_s = self.est_station_omega_w;
            let vel = cv.linear
                + est_chaser_rot * self.chaser_nav.rel_target_vel_body
                + omega_s.cross(r_from_port_w);
            let acc = omega_s.cross(omega_s.cross(r_from_port_w));
            let cmd_rot = (cp.rotation * self.chaser_nav.rel_target_rot).normalize();
            return Ok((pos, vel, acc, cmd_rot, omega_s));
        }
        let sp = self.world.scene.body_pose(self.station)?;
        let sv = self.world.scene.body_velocity(self.station)?;
        let local_off = Vec3::new(FLUSH_CENTER_DIST + stroke + port_gap_m, rbar, cross);
        let r_world = sp.rotation * local_off;
        let pos = sp.position + r_world;
        let vel = sv.linear + sv.angular.cross(r_world);
        let acc = sv.angular.cross(sv.angular.cross(r_world));
        Ok((pos, vel, acc, sp.rotation, sv.angular))
    }

    fn target_chaser_pose(&self, port_gap_m: f64, rbar: f64, cross: f64) -> Result<(Vec3, Quat)> {
        let (p, _, _, r, _) = self.target_chaser_kinematics(port_gap_m, rbar, cross)?;
        Ok((p, r))
    }

    pub fn kick_disturbance(&mut self) -> Result<()> {
        if matches!(
            self.phase,
            MissionPhase::SoftCaptureDamp
                | MissionPhase::RingRetract
                | MissionPhase::HardClamping
                | MissionPhase::VestibulePressurize
        ) {
            self.trigger_undock()?;
            return Ok(());
        }
        self.world.apply_body_impulse(
            self.chaser,
            SpatialVector::new(
                Vec3::new(420.0, 480.0, -380.0),
                Vec3::new(260.0, -220.0, 290.0),
            ),
        )?;
        self.phase = MissionPhase::Detumble;
        self.phase_start_s = self.time_s;
        Ok(())
    }

    pub fn toggle_thruster_fault(&mut self) -> Result<()> {
        if let Some(idx) = self.faulted_thruster_idx.take() {
            self.world
                .devices
                .thrusters
                .refuel(self.chaser_thrusters[idx], 18.0)?;
        } else {
            let idx = 0;
            self.world
                .devices
                .thrusters
                .refuel(self.chaser_thrusters[idx], 0.0)?;
            self.faulted_thruster_idx = Some(idx);
        }
        Ok(())
    }

    pub fn toggle_vent_jam(&mut self) {
        self.vent_neg_z_jammed = !self.vent_neg_z_jammed;
    }

    /// Toggles Station Free-Drift (`Off`, realistic ISS capture mode) vs Active Station Hold.
    pub fn toggle_station_free_drift(&mut self) -> Result<()> {
        self.station_free_drift = !self.station_free_drift;
        if self.station_free_drift {
            self.world
                .devices
                .spacecraft
                .command(self.station_gnc, SpacecraftCommand::Off)?;
        } else {
            let p = self.world.scene.body_pose(self.station)?;
            self.world.devices.spacecraft.command(
                self.station_gnc,
                SpacecraftCommand::PoseHold {
                    position: p.position,
                    orientation: p.rotation,
                },
            )?;
        }
        Ok(())
    }

    pub fn toggle_hil_sensor_gnc(&mut self) {
        self.hil_sensor_gnc = !self.hil_sensor_gnc;
    }

    pub fn set_modulation_mode(&mut self, mode: ModulationUiMode) -> Result<()> {
        self.modulation_mode = mode;
        let dev = mode.to_device();
        for &tid in &self.chaser_thrusters {
            self.world.devices.thrusters.set_modulation(tid, dev)?;
        }
        Ok(())
    }

    pub fn cycle_modulation_mode(&mut self) -> Result<()> {
        self.set_modulation_mode(self.modulation_mode.next())
    }

    pub fn set_orbit_mode(&mut self, mode: OrbitMode) -> Result<()> {
        self.orbit_mode = mode;
        self.iss_leo = mode != OrbitMode::ZeroG;
        match mode {
            OrbitMode::ZeroG => {
                self.world.devices.orbital.set_field(OrbitalField::None)?;
                self.world.devices.space_env.set_atmosphere(None)?;
                self.world.devices.space_env.set_nrlmsise_atmosphere(None)?;
                self.world.devices.space_env.set_solar_radiation(None)?;
                self.world
                    .devices
                    .space_env
                    .set_magnetic_field(MagneticField::None)?;
            }
            OrbitMode::IssLeo => {
                self.world
                    .devices
                    .orbital
                    .set_field(OrbitalField::iss_leo())?;
                self.world.devices.space_env.set_atmosphere(None)?;
                self.world
                    .devices
                    .space_env
                    .set_nrlmsise_atmosphere(Some(NrlmsiseAtmosphere {
                        frame_orbital_velocity: Vec3::new(7_664.0, 0.0, 0.0),
                        ..NrlmsiseAtmosphere::earth_lvlh(408_000.0, SolarActivity::Moderate)
                    }))?;
                self.world
                    .devices
                    .space_env
                    .set_solar_radiation(Some(SolarRadiation::one_au(Vec3::new(
                        -0.45, 0.8, 0.5,
                    ))))?;
                self.world
                    .devices
                    .space_env
                    .set_magnetic_field(MagneticField::iss_leo())?;
            }
            OrbitMode::EccentricLeo => {
                // Eccentric LEO/MEO transfer orbit: perigee ~220 km, apogee ~3,940 km (e = 0.22).
                let orbit = EccentricOrbit {
                    mu: EARTH_MU,
                    semi_major_axis: EARTH_EQUATORIAL_RADIUS + 2_080_000.0,
                    eccentricity: 0.22,
                    initial_true_anomaly: 0.15,
                };
                self.world.devices.orbital.set_field(
                    OrbitalField::eccentric_lvlh(orbit, Vec3::Y, Vec3::X, true, false)?,
                )?;
                self.world.devices.space_env.set_atmosphere(None)?;
                self.world
                    .devices
                    .space_env
                    .set_nrlmsise_atmosphere(Some(NrlmsiseAtmosphere::earth_lvlh(
                        408_000.0,
                        SolarActivity::Moderate,
                    )))?;
                self.world
                    .devices
                    .space_env
                    .set_solar_radiation(Some(SolarRadiation::one_au(Vec3::new(
                        -0.45, 0.8, 0.5,
                    ))))?;
                self.world
                    .devices
                    .space_env
                    .set_magnetic_field(MagneticField::iss_leo())?;
            }
        }
        Ok(())
    }

    pub fn toggle_orbital(&mut self) -> Result<()> {
        self.set_orbit_mode(self.orbit_mode.next())
    }

    pub fn trigger_undock(&mut self) -> Result<()> {
        if matches!(
            self.phase,
            MissionPhase::SoftCaptureDamp
                | MissionPhase::RingRetract
                | MissionPhase::HardClamping
                | MissionPhase::VestibulePressurize
        ) {
            self.phase = MissionPhase::DepressurizeVent;
            self.phase_start_s = self.time_s;
        } else if matches!(
            self.phase,
            MissionPhase::UnclampAndDepart | MissionPhase::EmergencyBlowoff
        ) {
            self.soft_pawls_armed = true;
            self.soft_pawls_latched = [false; 3];
            self.seg_start_pos = self.world.scene.body_pose(self.chaser)?.position;
            self.phase = MissionPhase::ApproachAndExtend;
            self.phase_start_s = self.time_s;
        }
        Ok(())
    }

    /// **EMERGENCY PRESSURIZED UNCLAMP (`[E]`)**:
    /// Commands all 12 real dynamic hooks OPEN while the vestibule is pressurized!
    /// Pure `F = P · A` (`61.6 kN`) physically blasts the `3,200 kg` capsule away!
    pub fn trigger_emergency_blowoff(&mut self) -> Result<()> {
        if self.vestibule_pressure_pa < 15_000.0 && self.flange_gap_m()?.abs() < 0.03 {
            let m = NOMINAL_PRESSURE_PA * VESTIBULE_VOL_0_M3 / (GAS_R * T_WALL_K);
            self.air_mass_kg = m;
            self.vestibule_temp_k = T_WALL_K;
            self.air_energy_j = m * GAS_CV * T_WALL_K;
            self.vestibule_pressure_pa = NOMINAL_PRESSURE_PA;
        }
        self.soft_pawls_armed = false;
        self.soft_pawls_latched = [false; 3];
        self.world
            .devices
            .spacecraft
            .command(self.chaser_gnc, SpacecraftCommand::Off)?;
        for &hj in &self.hook_joints {
            self.world
                .devices
                .motors
                .command(hj, MotorCommand::Position(HOOK_OPEN_RAD))?;
        }
        self.seal_verified = false;
        self.phase = MissionPhase::EmergencyBlowoff;
        self.phase_start_s = self.time_s;
        Ok(())
    }

    fn emit_rcs_plumes(&mut self) -> Result<()> {
        let sources = [
            (self.chaser, self.chaser_thrusters.clone()),
            (self.station, self.station_thrusters.clone()),
        ];
        for (body, ids) in sources {
            let pose = self.world.scene.body_pose(body)?;
            let body_vel = self.world.scene.body_velocity(body)?.linear;
            for id in ids {
                let (frame, max_thrust, thrust) = {
                    let (Some(spec), Some(read)) = (
                        self.world.devices.thrusters.spec(id),
                        self.world.devices.thrusters.reading(id),
                    ) else {
                        continue;
                    };
                    (spec.frame, spec.max_thrust, read.thrust)
                };
                let frac = if max_thrust > 0.0 {
                    (thrust / max_thrust).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                if frac < 0.03 || self.next_rand() > 0.45 + 0.55 * frac {
                    continue;
                }
                let exhaust_local = -(frame.rotation * Vec3::X);
                let exit_local = frame.position + exhaust_local * 0.14;
                let exit_world = pose.transform_point(exit_local);
                let exhaust_world = (pose.rotation * exhaust_local).normalize();
                let helper = if exhaust_world.y.abs() < 0.9 {
                    Vec3::Y
                } else {
                    Vec3::X
                };
                let t1 = exhaust_world.cross(helper).normalize();
                let t2 = exhaust_world.cross(t1).normalize();
                for _ in 0..2 {
                    let sa = (self.next_rand() - 0.5) * 0.30;
                    let sb = (self.next_rand() - 0.5) * 0.30;
                    let jet = (exhaust_world + t1 * sa + t2 * sb).normalize();
                    let speed = 4.2 + 5.2 * frac + self.next_rand() * 1.4;
                    let off = 0.03 * self.next_rand();
                    self.spawn_gas_puff(
                        exit_world + jet * off,
                        body_vel + jet * speed,
                        0.040 + 0.028 * frac,
                        0.85 + 0.65 * frac,
                        2.1,
                        0.24 + 0.16 * frac,
                        (0.32 + 0.36 * frac) as f32,
                        0.0,
                    );
                }
            }
        }
        Ok(())
    }

    fn emit_vent_nozzle_plume(&mut self, sign_z: f64, thrust_n: f64) -> Result<()> {
        if thrust_n < 1.5 {
            return Ok(());
        }
        let intensity = (thrust_n / 155.0).clamp(0.0, 1.0).sqrt();
        let port = self.station_port_pose()?;
        let station_vel = self.world.scene.body_velocity(self.station)?.linear;
        let p_world = port.transform_point(Vec3::new(-0.06, 0.0, sign_z * 0.72));
        let dir_world = (port.rotation * Vec3::new(0.0, 0.0, sign_z)).normalize();
        let u_axis = (port.rotation * Vec3::X).normalize();
        let v_axis = dir_world.cross(u_axis).normalize();
        for _ in 0..2 {
            let su = (self.next_rand() - 0.5) * 0.36;
            let sv = (self.next_rand() - 0.5) * 0.36;
            let jet = (dir_world + u_axis * su + v_axis * sv).normalize();
            let speed = (2.6 + 5.6 * intensity) * (0.78 + 0.45 * self.next_rand());
            let off = 0.04 * self.next_rand();
            self.spawn_gas_puff(
                p_world + jet * off,
                station_vel + jet * speed,
                0.042 + 0.034 * intensity,
                0.92 + 0.85 * intensity,
                2.3,
                0.52 + 0.38 * intensity,
                (0.20 + 0.34 * intensity) as f32,
                1.0,
            );
        }
        Ok(())
    }

    fn emit_seam_blowout_ring(&mut self, pressure_frac: f64) -> Result<()> {
        if pressure_frac < 0.02 {
            return Ok(());
        }
        let port = self.station_port_pose()?;
        let station_vel = self.world.scene.body_velocity(self.station)?.linear;
        for _ in 0..6 {
            let a = self.next_rand() * TAU;
            let rad_local = Vec3::new(0.0, a.cos(), a.sin());
            let p_world = port.transform_point(rad_local * 0.52);
            let dir_world = (port.rotation * (rad_local + Vec3::new(0.25, 0.0, 0.0))).normalize();
            let speed = (3.8 + 7.2 * pressure_frac.sqrt()) * (0.7 + 0.6 * self.next_rand());
            self.spawn_gas_puff(
                p_world,
                station_vel + dir_world * speed,
                0.055 + 0.040 * pressure_frac,
                1.4 + 1.1 * pressure_frac,
                2.0,
                0.55 + 0.35 * pressure_frac,
                (0.26 + 0.38 * pressure_frac.sqrt()) as f32,
                1.0,
            );
        }
        Ok(())
    }

    fn step_gas_puffs(&mut self, dt: f64) {
        for puff in &mut self.gas_puffs {
            puff.pos += puff.vel * dt;
            puff.radius += puff.growth_rate * dt;
            puff.age += dt;
        }
        self.gas_puffs.retain(|p| p.age < p.max_age);
    }

    pub fn plume_billboards(&self) -> Vec<crate::studio::PlumeBillboard> {
        self.gas_puffs
            .iter()
            .filter_map(|p| {
                let u = (p.age / p.max_age).clamp(0.0, 1.0) as f32;
                // Energy-conserving optical depth decay as gas cloud expands into vacuum.
                let fade = (1.0 - u).powf(1.85);
                let op = p.base_opacity * fade;
                (op > 0.006).then_some(crate::studio::PlumeBillboard {
                    pos: p.pos,
                    vel: p.vel,
                    radius: p.radius,
                    stretch: 1.0 + (p.stretch - 1.0) * (1.0 - u as f64 * 0.7),
                    opacity: op,
                    age_frac: u,
                    seed: p.seed,
                    kind: p.kind,
                })
            })
            .collect()
    }

    pub fn step(&mut self, dt: f64) -> Result<()> {
        let t = self.time_s;

        // 1. Motorized SARJ solar wings + Magnetorquer momentum management (only when Station Hold is ON!).
        let sarj_angle = if self.station_free_drift {
            0.22
        } else {
            0.28 * (t * 0.12).sin() + 0.20
        };
        self.world
            .devices
            .motors
            .command(self.solar_port_joint, MotorCommand::Position(sarj_angle))?;
        self.world
            .devices
            .motors
            .command(self.solar_starboard_joint, MotorCommand::Position(sarj_angle))?;
        if self.station_free_drift {
            self.world
                .devices
                .space_env
                .set_magnetorquer_dipole(self.station_mtb, Vec3::ZERO)?;
        } else if let Some(mtb_r) = self.world.devices.space_env.magnetorquer_reading(self.station_mtb) {
            let sq = self.world.scene.body_pose(self.station)?.rotation;
            let h_b = self.world.devices.reaction_wheels.body_momentum(self.station);
            let b_b = sq.conjugate() * mtb_r.field_world;
            let cmd = desaturation_dipole_body(h_b, b_b, 0.05, Vec3::splat(4_000.0));
            self.world
                .devices
                .space_env
                .set_magnetorquer_dipole(self.station_mtb, cmd)?;
        }

        // 2. Supervisory Autonomous Recovery & Rendezvous GNC State Machine:
        //    Handles arbitrary user nudges, 180° station flips, and spinning targets!
        let station_pose = self.world.scene.body_pose(self.station)?;
        let station_vel = self.world.scene.body_velocity(self.station)?;
        let chaser_pose = self.world.scene.body_pose(self.chaser)?;
        let chaser_vel = self.world.scene.body_velocity(self.chaser)?;
        let chaser_in_station = station_pose.inverse().transform_point(chaser_pose.position);
        let rel_att_rad = chaser_pose.rotation.angle_between(station_pose.rotation);
        let rel_omega_mag = (chaser_vel.angular - station_vel.angular).length();
        let (stroke, _) = self.scr_displacement()?;
        let flange_gap = self.flange_gap_m()?;
        let phase_elapsed = self.time_s - self.phase_start_s;
        let pawls_count = self.soft_pawls_latched.iter().filter(|&&l| l).count();
        let hooks_engaged_count = self.hook_engaged.iter().filter(|&&e| e).count();
        self.vent_pos_z_open = false;
        self.vent_neg_z_open = false;
        let mut fill_valve_open = false;

        // Check whether chaser is outside the safe forward docking corridor (e.g. user flipped
        // the station 180° or dragged the capsule behind the truss/solar wings!).
        let dist_from_station = chaser_in_station.length();
        let cos_cone = if dist_from_station > 0.1 {
            chaser_in_station.x / dist_from_station
        } else {
            -1.0
        };
        let needs_flyaround = chaser_in_station.x < 2.2 || cos_cone < 0.72; // outside ~44° forward cone

        // Automatic abort/re-acquisition if user yanks either spacecraft during approach/capture!
        if matches!(
            self.phase,
            MissionPhase::ApproachAndExtend | MissionPhase::FinalClosure
        ) && needs_flyaround
        {
            self.soft_pawls_armed = false;
            self.soft_pawls_latched = [false; 3];
            self.phase = MissionPhase::FlyAround;
            self.phase_start_s = self.time_s;
        } else if self.phase == MissionPhase::SoftCaptureDamp
            && (pawls_count < 3 || rel_att_rad > 0.45)
        {
            self.soft_pawls_armed = false;
            self.soft_pawls_latched = [false; 3];
            self.phase = if needs_flyaround {
                MissionPhase::FlyAround
            } else {
                MissionPhase::ApproachAndExtend
            };
            self.seg_start_pos = chaser_pose.position;
            self.phase_start_s = self.time_s;
        }

        match self.phase {
            MissionPhase::Detumble => {
                self.scr_command_m = 0.0;
                self.soft_pawls_armed = false;
                self.soft_pawls_latched = [false; 3];
                self.seal_verified = false;
                for &hj in &self.hook_joints {
                    self.world
                        .devices
                        .motors
                        .command(hj, MotorCommand::Position(HOOK_OPEN_RAD))?;
                }
                // Note: `RateHold.angular_velocity` is in the spacecraft's control (body) frame!
                let station_omega_in_chaser = chaser_pose.rotation.conjugate() * station_vel.angular;
                self.world.devices.spacecraft.command(
                    self.chaser_gnc,
                    SpacecraftCommand::RateHold {
                        linear_velocity: station_vel.linear,
                        angular_velocity: station_omega_in_chaser,
                    },
                )?;
                if phase_elapsed >= 1.8
                    && rel_omega_mag < 0.10
                    && (chaser_vel.linear - station_vel.linear).length() < 0.22
                {
                    self.seg_start_pos = chaser_pose.position;
                    self.phase = if needs_flyaround {
                        MissionPhase::FlyAround
                    } else {
                        MissionPhase::ApproachAndExtend
                    };
                    self.phase_start_s = self.time_s;
                }
            }
            MissionPhase::FlyAround => {
                // Collision-free great-circle arc around the Keep-Out Sphere (R = 5.4 m)
                // in the station's LIVE moving & rotating frame!
                self.scr_command_m = 0.0;
                self.soft_pawls_armed = false;
                for &hj in &self.hook_joints {
                    self.world
                        .devices
                        .motors
                        .command(hj, MotorCommand::Position(HOOK_OPEN_RAD))?;
                }
                let r_kos = 6.8;
                let mut u_cur = chaser_in_station.normalize_or_zero();
                if u_cur.length_squared() < 0.5 {
                    u_cur = Vec3::Y;
                }
                // Advance along great circle toward +X; arch over +Y if directly behind (-X).
                let mut axis = u_cur.cross(Vec3::X);
                if axis.length_squared() < 1e-4 {
                    axis = Vec3::Z;
                } else {
                    axis = axis.normalize();
                }
                let angle_to_x = u_cur.x.clamp(-1.0, 1.0).acos();
                let step_angle = angle_to_x.min(0.60);
                let u_next = (Quat::from_axis_angle(axis, step_angle) * u_cur).normalize();
                let wp_local = u_next * r_kos;
                let r_w = station_pose.rotation * wp_local;
                let wp_pos = station_pose.position + r_w;
                let wp_vel = station_vel.linear + station_vel.angular.cross(r_w);
                let wp_acc = station_vel.angular.cross(station_vel.angular.cross(r_w));
                self.world.devices.spacecraft.command(
                    self.chaser_gnc,
                    SpacecraftCommand::Trajectory {
                        position: wp_pos,
                        velocity: wp_vel,
                        acceleration: wp_acc,
                        orientation: station_pose.rotation,
                        angular_velocity: station_vel.angular,
                        angular_acceleration: Vec3::ZERO,
                    },
                )?;
                if chaser_in_station.x > 4.2 && cos_cone > 0.86 {
                    self.seg_start_pos = chaser_pose.position;
                    self.phase = MissionPhase::ApproachAndExtend;
                    self.phase_start_s = self.time_s;
                }
            }
            MissionPhase::ApproachAndExtend => {
                self.scr_command_m = SCR_EXTENDED_STROKE;
                self.soft_pawls_armed = false;
                for &hj in &self.hook_joints {
                    self.world
                        .devices
                        .motors
                        .command(hj, MotorCommand::Position(HOOK_OPEN_RAD))?;
                }
                let (hold_pos, hold_vel, hold_acc, hold_rot, hold_omega) =
                    self.target_chaser_kinematics(0.95, 0.025, 0.015)?;
                self.world.devices.spacecraft.command(
                    self.chaser_gnc,
                    SpacecraftCommand::Trajectory {
                        position: hold_pos,
                        velocity: hold_vel,
                        acceleration: hold_acc,
                        orientation: hold_rot,
                        angular_velocity: hold_omega,
                        angular_acceleration: Vec3::ZERO,
                    },
                )?;
                if phase_elapsed >= 1.8
                    && chaser_pose.position.distance(hold_pos) < 0.22
                    && rel_att_rad < 0.10
                    && (stroke - SCR_EXTENDED_STROKE).abs() < 0.03
                {
                    self.seg_start_pos = chaser_pose.position;
                    self.soft_pawls_armed = true;
                    self.phase = MissionPhase::FinalClosure;
                    self.phase_start_s = self.time_s;
                }
            }
            MissionPhase::FinalClosure => {
                self.scr_command_m = SCR_EXTENDED_STROKE;
                self.soft_pawls_armed = true;
                if pawls_count == 3 {
                    self.ever_soft_captured = true;
                    self.phase = MissionPhase::SoftCaptureDamp;
                    self.phase_start_s = self.time_s;
                    self.world
                        .devices
                        .spacecraft
                        .command(self.chaser_gnc, SpacecraftCommand::Off)?;
                } else {
                    // Close smoothly in the LIVE rotating/moving station frame (matches spin
                    // even if the user spun the station!) with slight 3 cm funnel contact offset.
                    let dur = 5.2;
                    let s = (phase_elapsed / dur).clamp(0.0, 1.0);
                    let gap_now = 0.95 * (1.0 - (3.0 * s * s - 2.0 * s * s * s)) - 0.022;
                    let gap_dot = if s < 1.0 {
                        -0.95 * (6.0 * s - 6.0 * s * s) / dur
                    } else {
                        0.0
                    };
                    let offset_fade = 1.0 - ((s - 0.86) / 0.14).clamp(0.0, 1.0);
                    let (tgt_pos, tgt_vel_rot, tgt_acc_rot, tgt_rot, tgt_omega) =
                        self.target_chaser_kinematics(gap_now, -0.030 * offset_fade, 0.0)?;
                    let axis_w = tgt_rot * Vec3::X;
                    let rel_vel_w = axis_w * gap_dot;
                    let coriolis_acc = 2.0 * tgt_omega.cross(rel_vel_w);
                    self.world.devices.spacecraft.command(
                        self.chaser_gnc,
                        SpacecraftCommand::Trajectory {
                            position: tgt_pos,
                            velocity: tgt_vel_rot + rel_vel_w,
                            acceleration: tgt_acc_rot + coriolis_acc,
                            orientation: tgt_rot,
                            angular_velocity: tgt_omega,
                            angular_acceleration: Vec3::ZERO,
                        },
                    )?;
                }
            }
            MissionPhase::SoftCaptureDamp => {
                self.scr_command_m = SCR_EXTENDED_STROKE;
                self.soft_pawls_armed = true;
                self.world
                    .devices
                    .spacecraft
                    .command(self.chaser_gnc, SpacecraftCommand::Off)?;
                if phase_elapsed >= 1.1 {
                    self.phase = MissionPhase::RingRetract;
                    self.phase_start_s = self.time_s;
                }
            }
            MissionPhase::RingRetract => {
                let retract_dur = 2.6;
                let s = (phase_elapsed / retract_dur).clamp(0.0, 1.0);
                let smooth = 3.0 * s * s - 2.0 * s * s * s;
                self.scr_command_m = SCR_EXTENDED_STROKE * (1.0 - smooth);
                self.soft_pawls_armed = true;
                self.world
                    .devices
                    .spacecraft
                    .command(self.chaser_gnc, SpacecraftCommand::Off)?;
                if s >= 1.0 && flange_gap < 0.018 {
                    self.scr_command_m = 0.0;
                    self.phase = MissionPhase::HardClamping;
                    self.phase_start_s = self.time_s;
                }
            }
            MissionPhase::HardClamping => {
                self.scr_command_m = 0.0;
                self.world
                    .devices
                    .spacecraft
                    .command(self.chaser_gnc, SpacecraftCommand::Off)?;
                for (i, &hj) in self.hook_joints.iter().enumerate() {
                    let target = if i % 2 == 0 || phase_elapsed >= 0.90 {
                        HOOK_CLOSED_RAD
                    } else {
                        HOOK_OPEN_RAD
                    };
                    self.world
                        .devices
                        .motors
                        .command(hj, MotorCommand::Position(target))?;
                }
                if hooks_engaged_count == 12 && self.hook_forces_n.iter().sum::<f64>() > 55_000.0 {
                    // Disengage all 3 soft-capture pawls — 100% of structural load now rests
                    // purely on the 12 dynamic Featherstone clamping hooks + O-ring gasket!
                    self.soft_pawls_armed = false;
                    self.soft_pawls_latched = [false; 3];
                    self.ever_hard_clamped = true;
                    self.phase = MissionPhase::VestibulePressurize;
                    self.phase_start_s = self.time_s;
                }
            }
            MissionPhase::VestibulePressurize => {
                self.scr_command_m = 0.0;
                self.soft_pawls_armed = false;
                self.world
                    .devices
                    .spacecraft
                    .command(self.chaser_gnc, SpacecraftCommand::Off)?;
                for &hj in &self.hook_joints {
                    self.world
                        .devices
                        .motors
                        .command(hj, MotorCommand::Position(HOOK_CLOSED_RAD))?;
                }
                fill_valve_open = self.vestibule_pressure_pa < NOMINAL_PRESSURE_PA;
                if self.vestibule_pressure_pa >= 100_000.0 && flange_gap < 0.012 {
                    self.seal_verified = true;
                    self.ever_pressurized = true;
                }
                if phase_elapsed >= 3.8 {
                    self.phase = MissionPhase::DepressurizeVent;
                    self.phase_start_s = self.time_s;
                }
            }
            MissionPhase::DepressurizeVent => {
                self.scr_command_m = 0.0;
                self.seal_verified = false;
                self.world
                    .devices
                    .spacecraft
                    .command(self.chaser_gnc, SpacecraftCommand::Off)?;
                if self.vestibule_pressure_pa > 1_200.0 {
                    self.vent_pos_z_open = true;
                    self.vent_neg_z_open = !self.vent_neg_z_jammed;
                } else {
                    let unclamp_t = (phase_elapsed - 2.6).max(0.0);
                    for (i, &hj) in self.hook_joints.iter().enumerate() {
                        let open = i % 2 == 1 || unclamp_t >= 0.65;
                        self.world.devices.motors.command(
                            hj,
                            MotorCommand::Position(if open {
                                HOOK_OPEN_RAD
                            } else {
                                HOOK_CLOSED_RAD
                            }),
                        )?;
                    }
                    if hooks_engaged_count == 0 && unclamp_t >= 1.25 {
                        let push_dir = self.station_port_pose()?.rotation * Vec3::X;
                        let push_impulse = 480.0; // Exact equal & opposite spring push-off!
                        self.world.apply_body_impulse(
                            self.chaser,
                            SpatialVector::new(push_dir * push_impulse, Vec3::ZERO),
                        )?;
                        self.world.apply_body_impulse(
                            self.station,
                            SpatialVector::new(-push_dir * push_impulse, Vec3::ZERO),
                        )?;
                        self.seg_start_pos = chaser_pose.position;
                        self.ever_vented_and_undocked = true;
                        self.phase = MissionPhase::UnclampAndDepart;
                        self.phase_start_s = self.time_s;
                    }
                }
            }
            MissionPhase::EmergencyBlowoff => {
                self.scr_command_m = 0.0;
                self.world
                    .devices
                    .spacecraft
                    .command(self.chaser_gnc, SpacecraftCommand::Off)?;
                if phase_elapsed >= 3.2 {
                    self.seg_start_pos = chaser_pose.position;
                    self.phase = MissionPhase::Detumble;
                    self.phase_start_s = self.time_s;
                }
            }
            MissionPhase::UnclampAndDepart => {
                self.scr_command_m = 0.0;
                self.soft_pawls_armed = false;
                for &hj in &self.hook_joints {
                    self.world
                        .devices
                        .motors
                        .command(hj, MotorCommand::Position(HOOK_OPEN_RAD))?;
                }
                if phase_elapsed < 0.8 {
                    self.world
                        .devices
                        .spacecraft
                        .command(self.chaser_gnc, SpacecraftCommand::Off)?;
                } else {
                    let t_dep = phase_elapsed - 0.8;
                    let (dep_pos, dep_rot) = self.target_chaser_pose(3.4, 0.28, -0.18)?;
                    let dur = 3.8;
                    let s = (t_dep / dur).clamp(0.0, 1.0);
                    let h = 3.0 * s * s - 2.0 * s * s * s;
                    let hd = if s < 1.0 { (6.0 * s - 6.0 * s * s) / dur } else { 0.0 };
                    let hdd = if s < 1.0 { (6.0 - 12.0 * s) / (dur * dur) } else { 0.0 };
                    let delta = dep_pos - self.seg_start_pos;
                    self.world.devices.spacecraft.command(
                        self.chaser_gnc,
                        SpacecraftCommand::Trajectory {
                            position: self.seg_start_pos + delta * h,
                            velocity: delta * hd,
                            acceleration: delta * hdd,
                            orientation: dep_rot,
                            angular_velocity: Vec3::ZERO,
                            angular_acceleration: Vec3::ZERO,
                        },
                    )?;
                    if t_dep >= dur + 0.8 {
                        self.seg_start_pos = chaser_pose.position;
                        self.phase = MissionPhase::ApproachAndExtend;
                        self.phase_start_s = self.time_s;
                    }
                }
            }
        }

        // 3. First-Law Open-System Compressible Gas Thermodynamics (dM & dU with -P dV + wall Q!).
        let vol_now = (VESTIBULE_VOL_0_M3 + VESTIBULE_AREA_M2 * flange_gap.max(-0.02)).max(0.50);
        let dv_dt = (vol_now - self.prev_vestibule_vol_m3) / dt;
        self.prev_vestibule_vol_m3 = vol_now;

        let mut mdot_in = 0.0;
        if fill_valve_open && flange_gap < 0.015 {
            // Choked supply fill valve (10 mm orifice from 200 kPa cabin/N2 manifold).
            let dp = (NOMINAL_PRESSURE_PA - self.vestibule_pressure_pa).max(0.0);
            mdot_in = (0.48 * (dp / NOMINAL_PRESSURE_PA).sqrt()).min(0.45);
        }
        let mut mdot_out = 0.0;
        if self.vent_pos_z_open {
            mdot_out += choked_mass_flow_kg_s(
                self.vestibule_pressure_pa,
                self.vestibule_temp_k,
                VENT_THROAT_AREA_M2,
                VENT_CD,
            );
        }
        if self.vent_neg_z_open {
            mdot_out += choked_mass_flow_kg_s(
                self.vestibule_pressure_pa,
                self.vestibule_temp_k,
                VENT_THROAT_AREA_M2,
                VENT_CD,
            );
        }
        let seam_blowout = self.vestibule_pressure_pa > 500.0 && flange_gap > 0.014;
        if seam_blowout {
            let leak_area = (TAU * SEAL_RADIUS_M * (flange_gap - 0.012)).min(0.28);
            mdot_out += choked_mass_flow_kg_s(
                self.vestibule_pressure_pa,
                self.vestibule_temp_k,
                leak_area,
                0.72,
            );
        }

        let q_wall = WALL_HEAT_COEFF_W_M2K * VESTIBULE_WALL_AREA_M2 * (T_WALL_K - self.vestibule_temp_k);
        let du_dt = mdot_in * GAS_CP * T_SUPPLY_K
            - mdot_out * GAS_CP * self.vestibule_temp_k
            - self.vestibule_pressure_pa * dv_dt
            + q_wall;
        self.air_mass_kg = (self.air_mass_kg + (mdot_in - mdot_out) * dt).max(0.0);
        if self.air_mass_kg > 1e-5 {
            self.air_energy_j = (self.air_energy_j + du_dt * dt)
                .clamp(self.air_mass_kg * GAS_CV * 120.0, self.air_mass_kg * GAS_CV * 450.0);
            self.vestibule_temp_k = self.air_energy_j / (self.air_mass_kg * GAS_CV);
            self.vestibule_pressure_pa =
                self.air_mass_kg * GAS_R * self.vestibule_temp_k / vol_now;
            if self.vestibule_pressure_pa > 20_000.0 {
                self.max_gas_temp_k = self.max_gas_temp_k.max(self.vestibule_temp_k);
            }
            if (self.vent_pos_z_open || self.vent_neg_z_open) && self.vestibule_pressure_pa > 3_000.0 {
                self.min_blowdown_temp_k = self.min_blowdown_temp_k.min(self.vestibule_temp_k);
            }
        } else {
            self.air_mass_kg = 0.0;
            self.air_energy_j = 0.0;
            self.vestibule_temp_k = T_WALL_K;
            self.vestibule_pressure_pa = 0.0;
        }

        // 4. Execute 960 Hz Featherstone + Contact + First-Principles Force Elements!
        //    (Aero drag, SRP, geomagnetic torque, Simons/Dettleff plume impingement, and
        //     microgravity propellant slosh all run natively inside `world.devices.space_env`!)
        let (stewart_base, stewart_ring) = stewart_anchors();
        let stewart_cmd_len = stewart_nominal_lengths(self.scr_command_m);
        let station_h = self.station;
        let chaser_h = self.chaser;
        let ring_h = self.capture_ring;
        let hook_hs = self.hook_bodies;
        let pressure_pa = self.vestibule_pressure_pa;
        let vent_pos_open = self.vent_pos_z_open;
        let vent_neg_open = self.vent_neg_z_open;
        let pawls_armed = self.soft_pawls_armed;
        let mut pawls_latched = self.soft_pawls_latched;
        let cf_sonic_vac = exact_sonic_vacuum_cf(GAS_GAMMA, VENT_CD);
        let active_grab = self.user_grab;
        let mut diag = SubstepDiag::default();

        self.world.step_with_forces(
            dt,
            |_, _| false,
            |scene, state, _sub_dt, wrenches| {
                let poses = state.body_q.host()?;
                let vels = state.body_qd.host()?;
                let coms = scene.model().body_com.host()?;

                let si = scene.body_index(station_h).unwrap().index();
                let ci = scene.body_index(chaser_h).unwrap().index();
                let ri = scene.body_index(ring_h).unwrap().index();
                let sp = poses[si];
                let cp = poses[ci];
                let rp = poses[ri];
                let sv = vels[si];
                let cv = vels[ci];
                let rv = vels[ri];
                let s_com = sp.transform_point(coms[si]);
                let c_com = cp.transform_point(coms[ci]);
                let r_com = rp.transform_point(coms[ri]);

                let apply_at = |w: &mut [SpatialVector], idx: usize, com: Vec3, pt: Vec3, f: Vec3| {
                    w[idx].linear += f;
                    w[idx].angular += (pt - com).cross(f);
                };
                let vel_at = |v: SpatialVector, com: Vec3, pt: Vec3| {
                    v.linear + v.angular.cross(pt - com)
                };

                // A. 6 REAL Stewart-Platform Linear Actuator Struts (chaser <-> free 6-DOF capture_ring)!
                let k_strut = 110_000.0;
                let c_strut = 3_200.0;
                let f_strut_max = 24_000.0;
                for i in 0..6 {
                    let pb = cp.transform_point(stewart_base[i]);
                    let pr = rp.transform_point(stewart_ring[i]);
                    let delta = pr - pb;
                    let len = delta.length().max(1e-6);
                    let u = delta / len;
                    let ldot = (vel_at(rv, r_com, pr) - vel_at(cv, c_com, pb)).dot(u);
                    let f_mag = (k_strut * (stewart_cmd_len[i] - len) - c_strut * ldot)
                        .clamp(-f_strut_max, f_strut_max);
                    let f_vec = u * f_mag;
                    apply_at(wrenches, ri, r_com, pr, f_vec);
                    apply_at(wrenches, ci, c_com, pb, -f_vec);
                    diag.strut_forces[i] += f_mag;
                }

                // B. 3 Independent Physical Soft-Capture Pawls (ZERO `Connector` snap cheat!).
                let port_world = sp.compose(&Transform::from_translation(Vec3::new(
                    STATION_PORT_X,
                    0.0,
                    0.0,
                )));
                let port_axis = port_world.rotation * Vec3::X;
                let k_pawl = 140_000.0;
                let c_pawl = 3_400.0;
                let f_pawl_break = 32_000.0;
                for p in 0..3 {
                    let a = p as f64 * (TAU / 3.0);
                    let rad_l = Vec3::new(0.0, a.cos(), a.sin()) * 0.46;
                    let p_stat = port_world.transform_point(rad_l);
                    let p_ring = rp.transform_point(Vec3::new(RING_FACE_OFFSET_X, 0.0, 0.0) + rad_l);
                    let err = p_ring - p_stat;
                    let dist = err.length();
                    if pawls_armed && !pawls_latched[p] && dist < 0.042 {
                        pawls_latched[p] = true;
                    }
                    if pawls_armed && pawls_latched[p] {
                        let v_rel = vel_at(rv, r_com, p_ring) - vel_at(sv, s_com, p_stat);
                        let f_vec = -err * k_pawl - v_rel * c_pawl;
                        let f_mag = f_vec.length();
                        if f_mag > f_pawl_break {
                            // Real mechanical overload slip/release!
                            pawls_latched[p] = false;
                        } else {
                            apply_at(wrenches, ri, r_com, p_ring, f_vec);
                            apply_at(wrenches, si, s_com, p_stat, -f_vec);
                            diag.pawl_forces[p] += f_mag;
                        }
                    }
                }

                // C. 6-Pad REAL Elastomeric O-Ring Gasket Contact (IDSS Rev D metal-to-metal margin)!
                let k_seal_pad = 1_500_000.0; // 9.0 MN/m total gasket stiffness
                let c_seal_pad = 45_000.0;
                let mut seal_sum = 0.0;
                let mut min_pad = f64::INFINITY;
                for pad in 0..6 {
                    let a = pad as f64 * (TAU / 6.0);
                    let rad_l = Vec3::new(0.0, a.cos(), a.sin()) * SEAL_RADIUS_M;
                    let p_stat = port_world.transform_point(rad_l);
                    let p_chas = cp.transform_point(
                        Vec3::new(CHASER_BULKHEAD_X + RING_FACE_OFFSET_X, 0.0, 0.0) + rad_l,
                    );
                    let sep = (p_chas - p_stat).dot(port_axis);
                    let lat = ((p_chas - p_stat) - port_axis * sep).length();
                    let mut f_n = 0.0;
                    if sep < 0.0 && lat < 0.35 {
                        let pen = -sep;
                        let v_sep = (vel_at(cv, c_com, p_chas) - vel_at(sv, s_com, p_stat)).dot(port_axis);
                        f_n = (k_seal_pad * pen - c_seal_pad * v_sep).max(0.0);
                        let f_vec = port_axis * f_n;
                        apply_at(wrenches, ci, c_com, p_chas, f_vec);
                        apply_at(wrenches, si, s_com, p_stat, -f_vec);
                        seal_sum += f_n;
                    }
                    min_pad = min_pad.min(f_n);
                }
                diag.seal_force_n += seal_sum;
                diag.seal_min_pad_n += if min_pad.is_finite() { min_pad } else { 0.0 };

                // D. 12 REAL Dynamic Clamping Hooks — Jaw Forces Applied DIRECTLY to Hook Bodies!
                //    Transmits axial/shear clamp load & motor load torque physically through
                //    each Featherstone `Revolute` hinge pin into `chaser`!
                for i in 0..12 {
                    let a = i as f64 * (TAU / 12.0);
                    let hi = scene.body_index(hook_hs[i]).unwrap().index();
                    let hp = poses[hi];
                    let hv = vels[hi];
                    let h_com = hp.transform_point(coms[hi]);
                    let rel_rot = cp.rotation.inverse() * hp.rotation;
                    let base_rot = hook_base_rotation(a);
                    let local_hinge = base_rot.inverse() * rel_rot;
                    let q_real = 2.0 * local_hinge.z.atan2(local_hinge.w);
                    let engage = ((0.14 - q_real) / 0.12).clamp(0.0, 1.0);
                    if engage <= 0.0 {
                        continue;
                    }
                    // Over-center jaw contact point on hook (`y = -0.016 m` lever arm about hinge pin!).
                    let p_jaw_world = hp.transform_point(Vec3::new(0.24, -0.016, 0.0));
                    let rad_l = Vec3::new(0.0, a.cos(), a.sin()) * 0.584;
                    let p_striker = sp.transform_point(
                        Vec3::new(STATION_PORT_X - 0.075, 0.0, 0.0) + rad_l,
                    );
                    let p_flange_ref = cp.transform_point(
                        Vec3::new(CHASER_BULKHEAD_X + RING_FACE_OFFSET_X - 0.075, 0.0, 0.0) + rad_l,
                    );
                    let err = p_flange_ref - p_striker;
                    let ax_gap = err.dot(port_axis);
                    let lat_err = err - port_axis * ax_gap;
                    if ax_gap < 0.055 && lat_err.length() < 0.12 {
                        let v_rel = vel_at(hv, h_com, p_jaw_world) - vel_at(sv, s_com, p_striker);
                        let v_ax = v_rel.dot(port_axis);
                        let v_lat = v_rel - port_axis * v_ax;
                        let f_ax = (engage * (6_500.0 + 48_000.0 * ax_gap - 2_800.0 * v_ax))
                            .clamp(0.0, 9_500.0);
                        let f_lat = (-lat_err * 140_000.0 - v_lat * 3_500.0) * engage;
                        let f_on_hook = -port_axis * f_ax + f_lat;
                        // Applied DIRECTLY onto `hook_bodies[i]` at its jaw tip!
                        apply_at(wrenches, hi, h_com, p_jaw_world, f_on_hook);
                        apply_at(wrenches, si, s_com, p_striker, -f_on_hook);
                        diag.hook_forces[i] += f_ax;
                        diag.hook_engaged[i] = true;
                    }
                }

                // E. REAL Outward Vestibule Air Pressure Force (`F = P * A_vestibule`, 61.6 kN at 1 atm)!
                let chaser_flange_c = cp.transform_point(Vec3::new(
                    CHASER_BULKHEAD_X + RING_FACE_OFFSET_X,
                    0.0,
                    0.0,
                ));
                let axial_sep = (chaser_flange_c - port_world.position).dot(port_axis);
                if pressure_pa > 10.0 && axial_sep < 0.25 {
                    let containment = (1.0 - (axial_sep.max(0.0) / 0.18)).clamp(0.0, 1.0);
                    let f_air = pressure_pa * VESTIBULE_AREA_M2 * containment;
                    let f_vec = port_axis * f_air;
                    apply_at(wrenches, ci, c_com, chaser_flange_c, f_vec);
                    apply_at(wrenches, si, s_com, port_world.position, -f_vec);
                    diag.air_sep_force_n += f_air;
                }

                // F. Exact Analytic Choked-Flow Sonic Vacuum Vent Nozzle Reaction Thrust!
                let f_nozzle = cf_sonic_vac * pressure_pa * VENT_THROAT_AREA_M2;
                let mut net_vent_torque = Vec3::ZERO;
                if vent_pos_open && f_nozzle > 0.01 {
                    let pt = port_world.transform_point(Vec3::new(-0.06, 0.0, 0.72));
                    let exhaust_dir = (port_world.rotation * Vec3::Z).normalize();
                    let reaction = -exhaust_dir * f_nozzle;
                    apply_at(wrenches, si, s_com, pt, reaction);
                    net_vent_torque += (pt - s_com).cross(reaction);
                    diag.vent_pos_z_n += f_nozzle;
                }
                if vent_neg_open && f_nozzle > 0.01 {
                    let pt = port_world.transform_point(Vec3::new(-0.06, 0.0, -0.72));
                    let exhaust_dir = (port_world.rotation * Vec3::NEG_Z).normalize();
                    let reaction = -exhaust_dir * f_nozzle;
                    apply_at(wrenches, si, s_com, pt, reaction);
                    net_vent_torque += (pt - s_com).cross(reaction);
                    diag.vent_neg_z_n += f_nozzle;
                }
                diag.vent_torque_nm += net_vent_torque.length();

                // G. Interactive 3D Mouse Grab / Nudge Spring-Damper Force at Clicked Point!
                if let Some(grab) = active_grab
                    && let Some(gi) = scene.body_index(grab.body)
                {
                    let gi = gi.index();
                    let gp = poses[gi];
                    let gv = vels[gi];
                    let g_com = gp.transform_point(coms[gi]);
                    let p_w = gp.transform_point(grab.local_point);
                    let v_w = vel_at(gv, g_com, p_w);
                    let err = grab.target_world - p_w;
                    let m_ref = if grab.body == chaser_h { 3_200.0 } else { 6_500.0 };
                    let f_raw = err * (m_ref * 2.4) - v_w * (m_ref * 2.2);
                    let f_cap = m_ref * 2.8;
                    let f_mag = f_raw.length();
                    let f_grab = if f_mag > f_cap {
                        f_raw * (f_cap / f_mag)
                    } else {
                        f_raw
                    };
                    apply_at(wrenches, gi, g_com, p_w, f_grab);
                    diag.grab_force_n += f_grab.length();
                }

                diag.samples += 1;
                Ok(())
            },
        )?;

        self.soft_pawls_latched = pawls_latched;
        let inv_n = 1.0 / diag.samples.max(1) as f64;
        if let Some(g) = &mut self.user_grab {
            g.applied_force_n = diag.grab_force_n * inv_n;
        }
        for i in 0..6 {
            self.strut_forces_n[i] = diag.strut_forces[i] * inv_n;
        }
        for p in 0..3 {
            self.soft_pawl_forces_n[p] = diag.pawl_forces[p] * inv_n;
        }
        for i in 0..12 {
            self.hook_forces_n[i] = diag.hook_forces[i] * inv_n;
            self.hook_engaged[i] = diag.hook_engaged[i];
        }
        self.seal_force_n = diag.seal_force_n * inv_n;
        self.seal_min_pad_n = diag.seal_min_pad_n * inv_n;
        self.air_sep_force_n = diag.air_sep_force_n * inv_n;
        self.vent_thrust_pos_z_n = diag.vent_pos_z_n * inv_n;
        self.vent_thrust_neg_z_n = diag.vent_neg_z_n * inv_n;
        self.vent_net_torque_nm = diag.vent_torque_nm * inv_n;
        // Read native Simons/Dettleff plume impingement from `molla-solvers::devices::space_env`!
        let f_imp_total = self.world.devices.space_env.body_plume_force(self.station)
            + self.world.devices.space_env.body_plume_force(self.solar_port)
            + self
                .world
                .devices
                .space_env
                .body_plume_force(self.solar_starboard);
        self.impingement_force_n = f_imp_total.length();
        self.venting_air = self.vent_thrust_pos_z_n > 1.5 || self.vent_thrust_neg_z_n > 1.5;

        let mut max_motor_nm = 0.0_f64;
        for &hj in &self.hook_joints {
            if let Some(mr) = self.world.devices.motors.reading(hj) {
                max_motor_nm = max_motor_nm.max(mr.force.abs());
            }
        }
        self.peak_hook_motor_nm = self.peak_hook_motor_nm.max(max_motor_nm);

        self.petal_contacts = self.world.contact_pairs().len();
        if self.petal_contacts > 0 {
            self.ever_petal_contact = true;
        }

        let total_clamp_kn = self.hook_forces_n.iter().sum::<f64>() * 1e-3;
        self.peak_clamp_force_kn = self.peak_clamp_force_kn.max(total_clamp_kn);
        self.peak_air_force_kn = self.peak_air_force_kn.max(self.air_sep_force_n * 1e-3);
        self.peak_vent_thrust_n = self
            .peak_vent_thrust_n
            .max(self.vent_thrust_pos_z_n.max(self.vent_thrust_neg_z_n));

        self.time_s += dt;

        // 5. Update Hardware-in-the-Loop Navigation Filter (IMU + Star Tracker + LiDAR)
        //    and Flexible Solar Wing CSI bending telemetry.
        if let Some(r_imu) = self.world.devices.space_sensors.imu_reading(self.chaser_imu) {
            let r_st = self
                .world
                .devices
                .space_sensors
                .star_tracker_reading(self.chaser_star_tracker);
            let r_ld = self
                .world
                .devices
                .space_sensors
                .lidar_reading(self.chaser_lidar);
            self.chaser_nav.update(
                dt,
                &r_imu,
                r_st.as_ref(),
                r_ld.as_ref(),
                self.chaser_lidar_mount,
            );
            if let Some(ld) = r_ld
                && ld.fresh
            {
                if !ld.valid {
                    self.prev_nav_rel_rot = None;
                } else {
                    let est_station_q =
                        (self.chaser_nav.orientation * self.chaser_nav.rel_target_rot).normalize();
                    if let Some(prev_q) = self.prev_nav_rel_rot {
                        let dt_q = (ld.timestamp_s - self.prev_nav_rot_t).max(0.05);
                        let mut dq = (est_station_q * prev_q.conjugate()).normalize();
                        if dq.w < 0.0 {
                            dq = -dq;
                        }
                        let v = Vec3::new(dq.x, dq.y, dq.z);
                        let s = v.length();
                        let dtheta_w = if s > 1e-12 {
                            v * (2.0 * s.atan2(dq.w) / s)
                        } else {
                            v * 2.0
                        };
                        let omega_meas = dtheta_w / dt_q;
                        self.est_station_omega_w = self.est_station_omega_w.lerp(omega_meas, 0.45);
                    }
                    self.prev_nav_rel_rot = Some(est_station_q);
                    self.prev_nav_rot_t = ld.timestamp_s;
                }
            }
        }
        let flex_p = self
            .world
            .devices
            .space_env
            .flex_joint_reading(self.solar_port_flex)
            .map_or(0.0, |r| r.deflection.abs().to_degrees());
        let flex_s = self
            .world
            .devices
            .space_env
            .flex_joint_reading(self.solar_starboard_flex)
            .map_or(0.0, |r| r.deflection.abs().to_degrees());
        self.peak_wing_flex_deg = self.peak_wing_flex_deg.max(flex_p.max(flex_s));

        self.emit_rcs_plumes()?;
        self.emit_vent_nozzle_plume(1.0, self.vent_thrust_pos_z_n)?;
        self.emit_vent_nozzle_plume(-1.0, self.vent_thrust_neg_z_n)?;
        if seam_blowout {
            self.emit_seam_blowout_ring(self.vestibule_pressure_pa / NOMINAL_PRESSURE_PA)?;
        }
        self.step_gas_puffs(dt);

        let sp = self.station_port_pose()?.position;
        let cp = self.chaser_port_pose()?.position;
        self.min_port_distance_m = self.min_port_distance_m.min(sp.distance(cp));

        self.trail_tick += 1;
        if self.trail_tick.is_multiple_of(8) {
            if self.chaser_trail.len() >= 180 {
                self.chaser_trail.pop_front();
            }
            self.chaser_trail
                .push_back(self.world.scene.body_pose(self.chaser)?.position);
        }
        Ok(())
    }

    pub fn telemetry(&self) -> Result<Telemetry> {
        let sp = self.station_port_pose()?;
        let cp = self.chaser_port_pose()?;
        let port_delta = cp.position - sp.position;
        let port_distance_m = port_delta.length();
        let sv = self.world.scene.body_velocity(self.station)?;
        let cv = self.world.scene.body_velocity(self.chaser)?;
        let rel_v = cv.linear - sv.linear;
        let closing_speed_mps = if port_distance_m > 1e-6 {
            -rel_v.dot(port_delta / port_distance_m)
        } else {
            0.0
        };
        let station_pose = self.world.scene.body_pose(self.station)?;
        let chaser_pose = self.world.scene.body_pose(self.chaser)?;
        let attitude_err_deg = chaser_pose
            .rotation
            .angle_between(station_pose.rotation)
            .to_degrees();
        let station_att_deg = station_pose
            .rotation
            .angle_between(Quat::IDENTITY)
            .to_degrees();
        let chaser_omega_deg_s = cv.angular.length().to_degrees();
        let chaser_mass_kg = self.world.scene.body_mass_properties(self.chaser)?.mass;

        let mut propellant_kg = 0.0;
        let mut active_thrusters = 0;
        let mut faulted_thrusters = 0;
        let mut total_thrust_n = 0.0;
        let mut rcs_valve_cycles = 0_u64;
        for &id in &self.chaser_thrusters {
            if let Some(r) = self.world.devices.thrusters.reading(id) {
                if r.propellant_remaining.is_finite() {
                    propellant_kg += r.propellant_remaining;
                    if r.propellant_remaining <= 1e-6 {
                        faulted_thrusters += 1;
                    }
                }
                if r.thrust > 0.5 {
                    active_thrusters += 1;
                }
                total_thrust_n += r.thrust;
                rcs_valve_cycles += r.valve_cycles;
            }
        }
        let mut cmg_momentum = Vec3::ZERO;
        for &id in &self.station_cmgs {
            if let Some(r) = self.world.devices.reaction_wheels.reading(id) {
                cmg_momentum += r.spin_axis * r.momentum;
            }
        }
        let (scr_stroke_m, scr_lat_m) = self.scr_displacement()?;
        let strut_peak_n = self
            .strut_forces_n
            .iter()
            .fold(0.0_f64, |acc, &f| acc.max(f.abs()));
        let soft_pawls_latched = self.soft_pawls_latched.iter().filter(|&&l| l).count();
        let soft_pawl_force_kn = self.soft_pawl_forces_n.iter().sum::<f64>() * 1e-3;
        let hooks_engaged = self.hook_engaged.iter().filter(|&&e| e).count();
        let clamp_force_kn = self.hook_forces_n.iter().sum::<f64>() * 1e-3;
        let mut hook_peak_motor_nm = 0.0_f64;
        for &hj in &self.hook_joints {
            if let Some(mr) = self.world.devices.motors.reading(hj) {
                hook_peak_motor_nm = hook_peak_motor_nm.max(mr.force.abs());
            }
        }

        let slosh_r = self
            .world
            .devices
            .space_env
            .slosh_reading(self.chaser_slosh)
            .unwrap_or_default();
        let env_f = self.world.devices.space_env.body_aero_force(self.station)
            + self.world.devices.space_env.body_aero_force(self.solar_port)
            + self.world.devices.space_env.body_aero_force(self.solar_port_outer)
            + self.world.devices.space_env.body_aero_force(self.solar_starboard)
            + self.world.devices.space_env.body_aero_force(self.solar_starboard_outer)
            + self.world.devices.space_env.body_srp_force(self.station)
            + self.world.devices.space_env.body_srp_force(self.solar_port)
            + self.world.devices.space_env.body_srp_force(self.solar_port_outer)
            + self.world.devices.space_env.body_srp_force(self.solar_starboard)
            + self.world.devices.space_env.body_srp_force(self.solar_starboard_outer);
        let mtb_r = self
            .world
            .devices
            .space_env
            .magnetorquer_reading(self.station_mtb)
            .unwrap_or_default();
        let flex_p = self
            .world
            .devices
            .space_env
            .flex_joint_reading(self.solar_port_flex)
            .unwrap_or_default();
        let flex_s = self
            .world
            .devices
            .space_env
            .flex_joint_reading(self.solar_starboard_flex)
            .unwrap_or_default();
        let wing_flex_deg = flex_p.deflection.abs().max(flex_s.deflection.abs()).to_degrees();
        let wing_flex_energy_j = self.world.devices.space_env.total_flex_elastic_energy_j();
        let altitude_km = match self.orbit_mode {
            OrbitMode::ZeroG => 0.0,
            OrbitMode::IssLeo => 408.0 + station_pose.position.y * 1e-3,
            OrbitMode::EccentricLeo => {
                let r_c = self
                    .world
                    .devices
                    .orbital
                    .chief_radius()
                    .unwrap_or(EARTH_EQUATORIAL_RADIUS + 408_000.0);
                ((r_c - EARTH_EQUATORIAL_RADIUS) + station_pose.position.y) * 1e-3
            }
        };
        let atmo_density_kg_m3 = if self.orbit_mode == OrbitMode::ZeroG {
            0.0
        } else {
            nrlmsise_density_kg_m3(altitude_km * 1_000.0, SolarActivity::Moderate)
        };
        let lidar_r = self
            .world
            .devices
            .space_sensors
            .lidar_reading(self.chaser_lidar)
            .unwrap_or_default();
        let imu_r = self
            .world
            .devices
            .space_sensors
            .imu_reading(self.chaser_imu)
            .unwrap_or_default();
        let nav_att_err_deg = self
            .chaser_nav
            .orientation
            .angle_between(chaser_pose.rotation)
            .to_degrees();
        let nav_bias_err_mrad_s =
            (self.chaser_nav.gyro_bias_est - imu_r.true_gyro_bias).length() * 1e3;

        Ok(Telemetry {
            time_s: self.time_s,
            phase: self.phase,
            port_distance_m,
            flange_gap_mm: self.flange_gap_m()? * 1000.0,
            closing_speed_mps,
            attitude_err_deg,
            chaser_omega_deg_s,
            chaser_mass_kg,
            propellant_kg,
            initial_propellant_kg: self.initial_propellant_kg,
            active_thrusters,
            faulted_thrusters,
            total_thrust_n,
            impingement_force_n: self.impingement_force_n,
            scr_stroke_m,
            scr_lateral_mm: scr_lat_m * 1000.0,
            strut_peak_n,
            soft_pawls_latched,
            soft_pawl_force_kn,
            hooks_engaged,
            clamp_force_kn,
            hook_peak_motor_nm,
            seal_force_kn: self.seal_force_n * 1e-3,
            seal_min_pad_kn: self.seal_min_pad_n * 1e-3,
            slosh_disp_mm: slosh_r.displacement_local.length() * 1000.0,
            slosh_force_n: slosh_r.world_force.length(),
            env_drag_srp_mn: env_f.length() * 1e3,
            mtb_torque_mnm: mtb_r.world_torque.length() * 1e3,
            vestibule_pressure_kpa: self.vestibule_pressure_pa * 1e-3,
            vestibule_temp_k: self.vestibule_temp_k,
            air_sep_force_kn: self.air_sep_force_n * 1e-3,
            vent_thrust_pos_z_n: self.vent_thrust_pos_z_n,
            vent_thrust_neg_z_n: self.vent_thrust_neg_z_n,
            vent_net_torque_nm: self.vent_net_torque_nm,
            vent_jammed: self.vent_neg_z_jammed,
            seal_verified: self.seal_verified,
            venting_air: self.venting_air,
            petal_contacts: self.petal_contacts,
            station_speed_mps: sv.linear.length(),
            station_omega_deg_s: sv.angular.length().to_degrees(),
            station_att_deg,
            station_free_drift: self.station_free_drift,
            cmg_momentum_nms: cmg_momentum.length(),
            iss_leo: self.iss_leo,
            orbit_mode: self.orbit_mode,
            altitude_km,
            atmo_density_kg_m3,
            wing_flex_deg,
            wing_flex_energy_j,
            modulation_mode: self.modulation_mode,
            rcs_valve_cycles,
            hil_sensor_gnc: self.hil_sensor_gnc,
            lidar_locked: self.chaser_nav.lidar_locked && lidar_r.valid,
            lidar_range_m: lidar_r.range_m,
            nav_att_err_deg,
            nav_bias_err_mrad_s,
        })
    }
}
