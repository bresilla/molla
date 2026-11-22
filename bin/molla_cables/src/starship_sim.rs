//! Full-Scale 2-Stage SpaceX Starship + Super Heavy + Mechazilla Physics Simulation
//! powered by `molla-solvers` (`RocketPropulsion`, `RocketAero`, `SpaceSensors`).
//!
//! 100% First-Principles Physics (ZERO scripted flight paths):
//! - **Super Heavy Booster (`71 m`, `9 m` Ø, `275 t` dry)**:
//!   - 33 Raptor 3 Sea-Level engines (`3 center` + `10 inner` ±15° electric TVC + TWD +
//!     `20 outer` fixed), `LOX` & `CH4` cylindrical main tanks with shifting COM/inertia
//!     and NASA SP-8009 acceleration-driven pendulum slosh, 4 titanium grid fins with
//!     transonic choking, SRP drag loss, and 2 physical Mechazilla catch pins.
//! - **Starship Upper Stage (`50 m`, `9 m` Ø, `120 t` dry)**:
//!   - 6 Raptor 3 engines (`3 SL` ±15° electric TVC + TWD + `3 RVac` fixed vacuum bells
//!     with Schmucker overexpansion flow separation), Main + Header `LOX`/`CH4` tanks,
//!     and 4 actuated aerodynamic body flaps (2 forward + 2 aft) for 90° belly-flop
//!     skydiving and 90°→0° flip-and-burn pinpoint landing.
//! - **Starbase Mechazilla Tower (`140 m`)**:
//!   - Orbital Launch Mount (OLM) with physical hold-down clamps, 2 articulated chopstick
//!     catch arms (`catch_arm_left`, `catch_arm_right`) driven by real `MotorSpec` actuators
//!     that physically close and bear the `~330 t` Booster's weight on catch pins!

use std::collections::VecDeque;
use std::f64::consts::{FRAC_PI_2, PI, TAU};

use molla_core::Result;
use molla_math::{GRAVITY, Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, BodyHandle, BodyKind, JointDesc, JointHandle};
use molla_sim::{BodyParams, JointType, MassProperties};
use molla_solvers::devices::{
    AeroBodyId, AeroBodySpec, AeroSurfaceId, AeroSurfaceKind, AeroSurfaceSpec, BiquadNotchFilter,
    FlexBodyId, FlexBodySpec, ImuId, ImuSpec, MARS_GRAVITY, MARS_OMEGA_RAD_S, MOON_GRAVITY,
    MotorCommand, MotorLimits, MotorSpec, PlanetAtmosphere, PropellantTankSpec,
    RocketEngineCommand, RocketEngineId, RocketEngineSpec, TankId, ThrusterId, ThrusterSpec,
    WindField, earth_rotation_at_latitude, planet_rotation_at_latitude, us76_atmosphere,
};
use molla_solvers::rigid_world::RigidWorld;

pub const SIM_DT: f64 = 1.0 / 60.0;

// Full-scale SI Starbase geometry (meters).
pub const BOOSTER_LEN: f64 = 71.0;
pub const BOOSTER_HALF_LEN: f64 = BOOSTER_LEN * 0.5; // 35.5 m
pub const SHIP_LEN: f64 = 50.0;
pub const SHIP_HALF_LEN: f64 = SHIP_LEN * 0.5; // 25.0 m
pub const VEHICLE_DIAM: f64 = 9.0;
pub const OLM_DECK_Y: f64 = 20.0;
pub const HOTSTAGE_RING_H: f64 = 1.8;
pub const BOOSTER_CATCH_PIN_LOCAL_Y: f64 = 26.5;
pub const CHOPSTICK_CATCH_Y: f64 = 104.0; // Chopsticks sit at y = 104 m (booster COM at 77.5 m)
pub const SHIP_PAD_X: f64 = 260.0;
pub const SHIP_PAD_Z: f64 = 0.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StarshipScenario {
    FullStackMission,
    HotStaging,
    BoosterCatch,
    ShipFlipLanding,
    MarsSrpLanding,
    LunarHlsDescent,
}

impl StarshipScenario {
    pub fn label(self) -> &'static str {
        match self {
            Self::FullStackMission => "[1] EARTH · FULL 2-STAGE MISSION",
            Self::HotStaging => "[2] EARTH · HOT-STAGING SEPARATION",
            Self::BoosterCatch => "[3] EARTH · SUPER HEAVY CATCH",
            Self::ShipFlipLanding => "[4] EARTH · SHIP BELLY-FLOP & FLIP",
            Self::MarsSrpLanding => "[5] MARS · SUPERSONIC SRP LANDING",
            Self::LunarHlsDescent => "[6] MOON · LUNAR HLS DESCENT",
        }
    }

    pub fn is_earth(self) -> bool {
        !matches!(self, Self::MarsSrpLanding | Self::LunarHlsDescent)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoosterPhase {
    PadIgnition,
    Ascent33Engines,
    MecoHotStaging,
    FlipAndBoostback,
    GridFinEntry,
    CatchBurn13,
    CatchHover3,
    CaughtInChopsticks,
}

impl BoosterPhase {
    pub fn label(self) -> &'static str {
        match self {
            Self::PadIgnition => "PAD · 33-RAPTOR IGNITION",
            Self::Ascent33Engines => "ASCENT · 33 RAPTORS + MAX-Q",
            Self::MecoHotStaging => "HOT-STAGING · 3 CENTER + BLAST",
            Self::FlipAndBoostback => "BOOSTBACK · 13-ENGINE RETURN",
            Self::GridFinEntry => "ENTRY · 4 TITANIUM GRID FINS",
            Self::CatchBurn13 => "CATCH BURN · 13 RAPTORS + SRP",
            Self::CatchHover3 => "TOWER APPROACH · 3 RAPTORS",
            Self::CaughtInChopsticks => "MECHAZILLA CHOPSTICKS CAUGHT!",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShipPhase {
    StackedOnBooster,
    HotStageIgnition,
    ShipAscent6Engines,
    ApogeeCoastFlipToBelly,
    BellyFlopSkydiving,
    FlipAndBurn3Engines,
    LandingBurn2To1,
    LunarBrakingRaptors,
    LunarHlsFinalLanding,
    LandedOnPad,
}

impl ShipPhase {
    pub fn label(self) -> &'static str {
        match self {
            Self::StackedOnBooster => "STACKED ON SUPER HEAVY",
            Self::HotStageIgnition => "HOT-STAGE · 6 RAPTORS LIT",
            Self::ShipAscent6Engines => "SHIP ASCENT · 3 SL + 3 RVAC",
            Self::ApogeeCoastFlipToBelly => "HEADER TANKS · PITCH TO BELLY",
            Self::BellyFlopSkydiving => "90° BELLY-FLOP · 4 BODY FLAPS",
            Self::FlipAndBurn3Engines => "FLIP & SRP BURN · 3 RAPTORS",
            Self::LandingBurn2To1 => "FINAL TOUCHDOWN · 2→1 RAPTOR",
            Self::LunarBrakingRaptors => "LUNAR BRAKING · VAC RAPTORS",
            Self::LunarHlsFinalLanding => "LUNAR HLS · 4 MID-BODY THRUSTERS",
            Self::LandedOnPad => "STARSHIP TOUCHDOWN CONFIRMED!",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindPreset {
    Calm,
    Moderate,
    StrongGusts,
}

impl WindPreset {
    pub fn label(self) -> &'static str {
        match self {
            Self::Calm => "Calm (0 m/s)",
            Self::Moderate => "Boca Chica (6 m/s + Jet)",
            Self::StrongGusts => "High Gusts (12 m/s + Shear)",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Calm => Self::Moderate,
            Self::Moderate => Self::StrongGusts,
            Self::StrongGusts => Self::Calm,
        }
    }

    pub fn to_wind(self) -> WindField {
        match self {
            Self::Calm => WindField::calm(),
            Self::Moderate => WindField::boca_chica_moderate(),
            Self::StrongGusts => WindField {
                surface_wind: Vec3::new(11.0, 0.0, -7.0),
                jet_stream_wind: Vec3::new(38.0, 0.0, 16.0),
                jet_altitude_m: 11_000.0,
                jet_sigma_m: 3_500.0,
                gust_amplitude_mps: 6.5,
            },
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PlumePuff {
    pub pos: Vec3,
    pub vel: Vec3,
    pub radius: f64,
    pub growth_rate: f64,
    pub stretch: f64,
    pub age: f64,
    pub max_age: f64,
    pub base_opacity: f32,
    pub seed: f32,
    pub kind: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct UserNudge {
    pub body: BodyHandle,
    pub local_point: Vec3,
    pub target_world: Vec3,
    pub depth: f64,
    pub force_n: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct StarshipTelemetry {
    pub time_s: f64,
    pub scenario: StarshipScenario,
    pub booster_phase: BoosterPhase,
    pub ship_phase: ShipPhase,
    pub booster_alt_m: f64,
    pub booster_speed_mps: f64,
    pub booster_vy_mps: f64,
    pub booster_dist_tower_m: f64,
    pub booster_tilt_deg: f64,
    pub booster_mach: f64,
    pub booster_q_kpa: f64,
    pub booster_thrust_mn: f64,
    pub booster_active_engines: usize,
    pub booster_prop_t: f64,
    pub booster_mass_t: f64,
    pub booster_com_y_m: f64,
    pub booster_slosh_hz: f64,
    pub booster_twd_knm: f64,
    pub booster_srp_factor: f64,
    pub booster_mode1_hz: f64,
    pub booster_tip_cm: f64,
    pub booster_pogo_g: f64,
    pub booster_ullage_bar: f64,
    pub booster_pifs_m: f64,
    pub booster_buffet_knm: f64,
    pub ship_alt_m: f64,
    pub ship_speed_mps: f64,
    pub ship_vy_mps: f64,
    pub ship_dist_pad_m: f64,
    pub ship_tilt_deg: f64,
    pub ship_aoa_deg: f64,
    pub ship_mach: f64,
    pub ship_q_kpa: f64,
    pub ship_thrust_mn: f64,
    pub ship_active_engines: usize,
    pub ship_prop_t: f64,
    pub ship_header_t: f64,
    pub ship_mass_t: f64,
    pub ship_com_y_m: f64,
    pub ship_slosh_hz: f64,
    pub ship_mode1_hz: f64,
    pub ship_tip_cm: f64,
    pub ship_pogo_g: f64,
    pub ship_ullage_bar: f64,
    pub ship_rvac_separated: bool,
    pub ship_flap_deg: [f64; 4],
    pub interstage_blast_mn: f64,
    pub ground_cushion_mn: f64,
    pub chopsticks_closed_frac: f64,
    pub wind_preset: WindPreset,
    pub failed_engines: usize,
    pub tvc_enabled: bool,
    pub booster_gimbal_deg: f64,
    pub ship_gimbal_deg: f64,
}

pub struct StarshipSim {
    pub world: RigidWorld,
    pub scenario: StarshipScenario,
    pub time_s: f64,

    // Main bodies
    pub tower_base: BodyHandle,
    pub chopstick_left: BodyHandle,
    pub chopstick_right: BodyHandle,
    pub chopstick_left_joint: JointHandle,
    pub chopstick_right_joint: JointHandle,
    pub booster: BodyHandle,
    pub ship: BodyHandle,

    // Booster propulsion (33 Raptors: 0..3 center, 3..13 inner gimbal, 13..33 outer fixed)
    pub booster_lox: TankId,
    pub booster_ch4: TankId,
    pub booster_engines: Vec<RocketEngineId>,
    pub booster_aero: AeroBodyId,
    pub booster_grid_fins: [AeroSurfaceId; 4], // [+X, -X, +Z, -Z]
    pub booster_flex: FlexBodyId,
    pub booster_imu: ImuId,

    // Ship propulsion (6 Raptors: 0..3 SL center gimbal, 3..6 RVac outer fixed)
    pub ship_main_lox: TankId,
    pub ship_main_ch4: TankId,
    pub ship_hdr_lox: TankId,
    pub ship_hdr_ch4: TankId,
    pub ship_engines: Vec<RocketEngineId>,
    /// 4 Mid-body HLS Lunar Landing Thrusters (mounted high at y = +8 m for 1/6 g soft touchdown).
    pub ship_hls_thrusters: Vec<RocketEngineId>,
    /// 4 Real Physical Cold-Gas N2 RCS Thrusters (nose + aft pure-couple pairs, finite propellant, valve lag).
    pub ship_rcs_thrusters: [ThrusterId; 4],
    pub ship_aero: AeroBodyId,
    pub ship_flaps: [AeroSurfaceId; 4], // [fwd_L(+Z), fwd_R(-Z), aft_L(+Z), aft_R(-Z)]
    pub ship_flex: FlexBodyId,
    pub ship_imu: ImuId,
    pub ship_using_headers: bool,
    pub planet_gravity: f64,
    pub atmosphere: PlanetAtmosphere,
    /// Toggle closed-loop TVC / attitude control (`false` locks all gimbals at 0° -> raw open-loop physics!).
    pub tvc_enabled: bool,

    // Hardware-in-the-loop GNC state: adaptive bending notch + roll-off filters & wind observers
    booster_notch: [BiquadNotchFilter; 2],
    ship_notch: [BiquadNotchFilter; 2],
    booster_rate_lpf: Vec3,
    ship_rate_lpf: Vec3,
    booster_wind_int: Vec3,
    ship_wind_int: Vec3,
    ship_landing_sl_count: usize,

    // Physical clamps / mechanical state
    pub hold_down_engaged: bool,
    pub interstage_latched: bool,
    pub chopsticks_cmd_rad: f64,
    pub booster_caught: bool,
    pub ship_landed: bool,

    // Mission phases & diagnostics
    pub booster_phase: BoosterPhase,
    pub booster_phase_start_s: f64,
    pub ship_phase: ShipPhase,
    pub ship_phase_start_s: f64,
    pub wind_preset: WindPreset,
    pub failed_engine_count: usize,

    // Validation flags & peaks
    pub ever_lifted_off: bool,
    pub ever_hot_staged: bool,
    pub ever_booster_caught: bool,
    pub ever_ship_bellyflop: bool,
    pub ever_ship_flipped: bool,
    pub ever_ship_landed: bool,
    pub peak_max_q_kpa: f64,
    pub peak_interstage_blast_mn: f64,
    pub peak_twd_knm: f64,
    pub ship_touchdown_speed_mps: f64,
    pub ship_touchdown_tilt_deg: f64,
    pub booster_catch_speed_mps: f64,
    pub booster_catch_offset_m: f64,

    // Visual plumes, trails & interactive nudge
    pub plumes: Vec<PlumePuff>,
    pub booster_trail: VecDeque<Vec3>,
    pub ship_trail: VecDeque<Vec3>,
    pub user_nudge: Option<UserNudge>,
    rng: u64,
    trail_tick: u32,
}

fn insert_stage_body(
    world: &mut RigidWorld,
    pose: Transform,
    vel: SpatialVector,
    dry_mass: f64,
    dry_inertia: Vec3,
) -> Result<BodyHandle> {
    let h = world.scene.insert_body(BodyDesc {
        params: BodyParams::new().with_pose(pose),
        additional_mass: Some(MassProperties {
            mass: dry_mass,
            local_com: Vec3::ZERO,
            inertia: Mat3::from_diagonal(dry_inertia),
        }),
        ..BodyDesc::default()
    })?;
    world.scene.set_body_velocity(h, vel)?;
    Ok(h)
}

impl StarshipSim {
    pub fn new(scenario: StarshipScenario) -> Result<Self> {
        let mut world = RigidWorld::new()?;
        let (planet_gravity, atmosphere, planet_omega, wind_preset, wind_field) = match scenario {
            StarshipScenario::MarsSrpLanding => (
                MARS_GRAVITY,
                PlanetAtmosphere::MarsCospar,
                planet_rotation_at_latitude(MARS_OMEGA_RAD_S, 18.4), // Jezero Crater latitude
                WindPreset::Moderate,
                WindField {
                    surface_wind: Vec3::new(10.0, 0.0, -4.5),
                    jet_stream_wind: Vec3::new(32.0, 0.0, 8.0),
                    jet_altitude_m: 15_000.0,
                    jet_sigma_m: 5_000.0,
                    gust_amplitude_mps: 3.0,
                },
            ),
            StarshipScenario::LunarHlsDescent => (
                MOON_GRAVITY,
                PlanetAtmosphere::Vacuum,
                planet_rotation_at_latitude(2.6617e-6, -85.0), // Lunar South Pole Shackleton rim
                WindPreset::Calm,
                WindField::calm(),
            ),
            _ => (
                GRAVITY,
                PlanetAtmosphere::EarthUs76,
                earth_rotation_at_latitude(25.997), // Starbase Boca Chica
                WindPreset::Moderate,
                WindPreset::Moderate.to_wind(),
            ),
        };
        world.scene.set_gravity(Vec3::new(0.0, -planet_gravity, 0.0))?;
        world.devices.rockets.set_flat_ground(Some(0.0), 0.0);
        world.devices.rockets.set_atmosphere(atmosphere);
        world.devices.rockets.set_earth_rotation(planet_omega);
        world.devices.rocket_aero.set_wind(wind_field);

        // 1. Mechazilla 140 m Tower + 2 Articulated Featherstone Chopstick Arms!
        //    Tower sits at x = -14.0 m, z = 0.0 m; chopsticks reach forward to x = 0.0 m
        //    at y = CHOPSTICK_CATCH_Y (104 m), pivoting around vertical Y hinges!
        let tower_base = world.scene.insert_body(BodyDesc {
            kind: BodyKind::Fixed,
            params: BodyParams::new().with_pose(Transform::from_translation(Vec3::new(-14.0, 70.0, 0.0))),
            ..BodyDesc::default()
        })?;
        let arm_mass = 45_000.0;
        let arm_inertia = Vec3::new(0.4e6, 2.2e6, 2.2e6);
        let chopstick_left = insert_stage_body(
            &mut world,
            Transform::from_translation(Vec3::new(-6.0, CHOPSTICK_CATCH_Y, 6.2)),
            SpatialVector::ZERO,
            arm_mass,
            arm_inertia,
        )?;
        let chopstick_right = insert_stage_body(
            &mut world,
            Transform::from_translation(Vec3::new(-6.0, CHOPSTICK_CATCH_Y, -6.2)),
            SpatialVector::ZERO,
            arm_mass,
            arm_inertia,
        )?;
        let chopstick_left_joint = world.scene.insert_joint(JointDesc {
            parent: Some(tower_base),
            child: chopstick_left,
            kind: JointType::Revolute,
            axis: Vec3::Y,
            frame_parent: Transform::from_translation(Vec3::new(2.0, CHOPSTICK_CATCH_Y - 70.0, 4.8)),
            frame_child: Transform::from_translation(Vec3::new(-6.0, 0.0, 0.0)),
        })?;
        let chopstick_right_joint = world.scene.insert_joint(JointDesc {
            parent: Some(tower_base),
            child: chopstick_right,
            kind: JointType::Revolute,
            axis: Vec3::Y,
            frame_parent: Transform::from_translation(Vec3::new(2.0, CHOPSTICK_CATCH_Y - 70.0, -4.8)),
            frame_child: Transform::from_translation(Vec3::new(-6.0, 0.0, 0.0)),
        })?;
        for &j in &[chopstick_left_joint, chopstick_right_joint] {
            world.devices.motors.insert(
                &world.scene,
                j,
                MotorSpec {
                    limits: MotorLimits {
                        max_velocity: 0.45,
                        pid: [5.0, 0.0, 0.8],
                        ..Default::default()
                    },
                    max_force: 4.0e7,
                },
            )?;
        }

        // 2. Scenario Initial Conditions (all real full-scale SI coordinates & velocities!).
        let (
            b_pose,
            b_vel,
            b_lox_kg,
            b_ch4_kg,
            s_pose,
            s_vel,
            s_main_lox_kg,
            s_main_ch4_kg,
            s_hdr_lox_kg,
            s_hdr_ch4_kg,
            hold_down_engaged,
            interstage_latched,
            booster_phase,
            ship_phase,
            ship_using_headers,
        ) = match scenario {
            StarshipScenario::FullStackMission => {
                // Full 121 m stack sitting on the 20 m Orbital Launch Mount (RTLS suborbital test load
                // so liftoff -> Max-Q -> hot-stage -> booster catch AND ship belly-flop flip-landing
                // all play out cleanly in ~2.5 minutes!).
                let by = OLM_DECK_Y + BOOSTER_HALF_LEN;
                let sy = by + BOOSTER_HALF_LEN + HOTSTAGE_RING_H + SHIP_HALF_LEN;
                (
                    Transform::from_translation(Vec3::new(0.0, by, 0.0)),
                    SpatialVector::ZERO,
                    1_280_000.0,
                    360_000.0,
                    Transform::from_translation(Vec3::new(0.0, sy, 0.0)),
                    SpatialVector::ZERO,
                    380_000.0,
                    110_000.0,
                    48_000.0,
                    14_000.0,
                    true,
                    true,
                    BoosterPhase::PadIgnition,
                    ShipPhase::StackedOnBooster,
                    false,
                )
            }
            StarshipScenario::HotStaging => {
                // Directly at MECO (h = 4.2 km, Mach 1.08, slight 11° downrange tilt).
                let rot = Quat::from_rotation_z(-0.18);
                let up = rot * Vec3::Y;
                let bp = Vec3::new(420.0, 4_200.0, 0.0);
                let sp = bp + up * (BOOSTER_HALF_LEN + HOTSTAGE_RING_H + SHIP_HALF_LEN);
                let v0 = up * 340.0;
                (
                    Transform::new(bp, rot),
                    SpatialVector::new(v0, Vec3::ZERO),
                    175_000.0,
                    52_000.0,
                    Transform::new(sp, rot),
                    SpatialVector::new(v0, Vec3::ZERO),
                    340_000.0,
                    100_000.0,
                    48_000.0,
                    14_000.0,
                    false,
                    true,
                    BoosterPhase::MecoHotStaging,
                    ShipPhase::HotStageIgnition,
                    false,
                )
            }
            StarshipScenario::BoosterCatch => {
                // Super Heavy returning tail-first at 2,400 m altitude, 220 m downrange, -155 m/s!
                let rot = Quat::from_rotation_z(-0.10);
                (
                    Transform::new(Vec3::new(190.0, 2_400.0, -35.0), rot),
                    SpatialVector::new(Vec3::new(-18.0, -155.0, 3.0), Vec3::ZERO),
                    115_000.0,
                    34_000.0,
                    // Park ship safely landed on Pad B while focusing on Booster catch.
                    Transform::from_translation(Vec3::new(SHIP_PAD_X, SHIP_HALF_LEN, SHIP_PAD_Z)),
                    SpatialVector::ZERO,
                    0.0,
                    0.0,
                    8_000.0,
                    2_500.0,
                    false,
                    false,
                    BoosterPhase::GridFinEntry,
                    ShipPhase::LandedOnPad,
                    true,
                )
            }
            StarshipScenario::ShipFlipLanding => {
                // Starship at 2,150 m in 90° horizontal belly-flop (belly heatshield `-X` facing down `-Y`,
                // dorsal `+X` facing up `+Y`, nose `+Y` facing `+X` toward Pad B at x = 260 m)!
                let rot_belly = (Quat::from_rotation_y(PI) * Quat::from_rotation_z(FRAC_PI_2)).normalize();
                (
                    // Park booster caught in Mechazilla chopsticks!
                    Transform::from_translation(Vec3::new(0.0, CHOPSTICK_CATCH_Y - BOOSTER_CATCH_PIN_LOCAL_Y, 0.0)),
                    SpatialVector::ZERO,
                    12_000.0,
                    4_000.0,
                    Transform::new(Vec3::new(SHIP_PAD_X - 115.0, 2_150.0, 22.0), rot_belly),
                    SpatialVector::new(Vec3::new(4.0, -72.0, -2.5), Vec3::new(0.03, -0.02, 0.04)),
                    0.0,
                    0.0,
                    46_000.0,
                    13_500.0,
                    false,
                    false,
                    BoosterPhase::CaughtInChopsticks,
                    ShipPhase::BellyFlopSkydiving,
                    true,
                )
            }
            StarshipScenario::MarsSrpLanding => {
                // Loaded Mars Cargo Starship entering thin 699 Pa Martian CO2 at Mach 1.37 supersonic
                // in 90° belly-flop, requiring high-altitude Supersonic Retropropulsion (SRP) flip-and-burn!
                let rot_belly = (Quat::from_rotation_y(PI) * Quat::from_rotation_z(FRAC_PI_2)).normalize();
                (
                    Transform::from_translation(Vec3::new(-5_000.0, CHOPSTICK_CATCH_Y, 0.0)),
                    SpatialVector::ZERO,
                    12_000.0,
                    4_000.0,
                    Transform::new(Vec3::new(SHIP_PAD_X - 580.0, 5_800.0, 10.0), rot_belly),
                    SpatialVector::new(Vec3::new(18.0, -305.0, -1.0), Vec3::new(0.02, -0.02, 0.03)),
                    0.0,
                    0.0,
                    82_000.0,
                    23_500.0,
                    false,
                    false,
                    BoosterPhase::CaughtInChopsticks,
                    ShipPhase::BellyFlopSkydiving,
                    true,
                )
            }
            StarshipScenario::LunarHlsDescent => {
                // Artemis Lunar HLS Starship approaching South Pole landing site in pure vacuum (1/6 g):
                // pitched ~48° retro-braking on main vacuum Raptors, then cutting Raptors and
                // descending softly on 4 mid-body HLS Landing Thrusters right onto the South Pole pad!
                let rot_brake = Quat::from_rotation_z(0.84);
                (
                    Transform::from_translation(Vec3::new(-5_000.0, CHOPSTICK_CATCH_Y, 0.0)),
                    SpatialVector::ZERO,
                    12_000.0,
                    4_000.0,
                    Transform::new(Vec3::new(SHIP_PAD_X - 245.0, 680.0, 10.0), rot_brake),
                    SpatialVector::new(Vec3::new(62.0, -28.0, -1.0), Vec3::ZERO),
                    0.0,
                    0.0,
                    56_000.0,
                    16_000.0,
                    false,
                    false,
                    BoosterPhase::CaughtInChopsticks,
                    ShipPhase::LunarBrakingRaptors,
                    true,
                )
            }
        };

        // 3. Super Heavy Booster Body, Dry Mass, LOX/CH4 Tanks, 33 Raptors, 4 Grid Fins & IMU!
        let booster_dry_mass = 275_000.0;
        let booster_dry_inertia = Vec3::new(1.15e8, 3.4e6, 1.15e8);
        let booster = insert_stage_body(&mut world, b_pose, b_vel, booster_dry_mass, booster_dry_inertia)?;
        world.devices.rockets.register_stage_dry_mass(
            &world.scene,
            booster,
            MassProperties {
                mass: booster_dry_mass,
                local_com: Vec3::new(0.0, -5.5, 0.0), // Heavy 33-engine aft thrust puck!
                inertia: Mat3::from_diagonal(booster_dry_inertia),
            },
        )?;
        let booster_lox = world.devices.rockets.insert_tank(
            &mut world.scene,
            PropellantTankSpec {
                body: booster,
                base_local: Vec3::new(0.0, -32.0, 0.0),
                axis_local: Vec3::Y,
                radius_m: 4.45,
                height_m: 34.0,
                liquid_density_kg_m3: 1_140.0,
                initial_mass_kg: b_lox_kg,
                slosh_mass_fraction: 0.22,
                slosh_damping_ratio: 0.06,
                capillary_omega2: 0.15,
            },
        )?;
        let booster_ch4 = world.devices.rockets.insert_tank(
            &mut world.scene,
            PropellantTankSpec {
                body: booster,
                base_local: Vec3::new(0.0, 2.5, 0.0),
                axis_local: Vec3::Y,
                radius_m: 4.45,
                height_m: 28.0,
                liquid_density_kg_m3: 422.0,
                initial_mass_kg: b_ch4_kg,
                slosh_mass_fraction: 0.20,
                slosh_damping_ratio: 0.06,
                capillary_omega2: 0.15,
            },
        )?;

        // 33 Raptor 3 Sea-Level Engines (3 center + 10 inner gimbal ±15°, 20 outer fixed 0°)!
        let up_rot = Quat::from_rotation_z(FRAC_PI_2);
        let mut booster_engines = Vec::with_capacity(33);
        let mut add_ring = |count: usize, radius: f64, phase0: f64, max_gimbal: f64| -> Result<()> {
            for k in 0..count {
                let a = phase0 + (k as f64) * (TAU / count as f64);
                let frame = Transform::new(
                    Vec3::new(radius * a.cos(), -BOOSTER_HALF_LEN + 0.8, radius * a.sin()),
                    up_rot,
                );
                let id = world.devices.rockets.insert_engine(
                    &world.scene,
                    RocketEngineSpec::raptor3_sea_level(
                        booster,
                        frame,
                        Some(booster_lox),
                        Some(booster_ch4),
                        max_gimbal,
                    ),
                )?;
                booster_engines.push(id);
            }
            Ok(())
        };
        add_ring(3, 1.05, 0.0, 15.0_f64.to_radians())?; // 0..3: center gimballing
        add_ring(10, 2.45, 0.18, 15.0_f64.to_radians())?; // 3..13: inner gimballing
        add_ring(20, 3.85, 0.0, 0.0)?; // 13..33: outer fixed ring

        let booster_aero = world.devices.rocket_aero.insert_body(
            &world.scene,
            AeroBodySpec {
                body: booster,
                axis_local: Vec3::Y,
                diameter_m: VEHICLE_DIAM,
                length_m: BOOSTER_LEN,
                cd_axial_nose: 0.34,
                cd_axial_tail: 0.96,
                cd_crossflow_subsonic: 1.20,
                cd_crossflow_supersonic: 1.52,
                cp_crossflow_local: 0.0,
                cp_axial_shift_m: 10.5,
                rot_damping_scale: 1.0,
                srp_drag_loss_gain: 0.85,
            },
        )?;
        // 4 Titanium Grid Fins at y = +29.5 m ([+X, -X, +Z, -Z]).
        let grid_configs = [
            (Vec3::new(4.85, 29.5, 0.0), Vec3::X, Vec3::Z),
            (Vec3::new(-4.85, 29.5, 0.0), Vec3::X, Vec3::Z),
            (Vec3::new(0.0, 29.5, 4.85), Vec3::Z, Vec3::X),
            (Vec3::new(0.0, 29.5, -4.85), Vec3::Z, Vec3::X),
        ];
        let mut grid_vec = Vec::with_capacity(4);
        for (center, hinge, normal) in grid_configs {
            grid_vec.push(world.devices.rocket_aero.insert_surface(
                &world.scene,
                AeroSurfaceSpec {
                    body: booster,
                    kind: AeroSurfaceKind::GridFin,
                    center_local: center,
                    hinge_axis_local: hinge,
                    normal_local: normal,
                    area_m2: 7.2,
                    hinge_moment_arm_m: 0.35,
                    min_deflection_rad: -22.0_f64.to_radians(),
                    max_deflection_rad: 22.0_f64.to_radians(),
                    max_rate_rad_s: 1.8,
                    max_hinge_torque_nm: 650_000.0,
                    cn_max: 2.3,
                },
            )?);
        }
        let booster_grid_fins: [AeroSurfaceId; 4] = grid_vec.try_into().unwrap();
        let booster_flex = world
            .devices
            .rocket_structure
            .insert(&world.scene, FlexBodySpec::super_heavy(booster))?;
        let booster_imu = world
            .devices
            .space_sensors
            .insert_imu(&world.scene, ImuSpec::tactical_fog(booster, 0xB005_0001))?;
        world
            .devices
            .rockets
            .set_tank_pressurization(booster_lox, 460_000.0, 105_000.0, 14.0)?;
        world
            .devices
            .rockets
            .set_tank_pressurization(booster_ch4, 480_000.0, 115_000.0, 14.0)?;

        // 4. Starship Upper Stage Body, Dry Mass, Main + Header Tanks, 6 Raptors (3 SL + 3 RVac), 4 HLS Thrusters, 4 Body Flaps!
        // Mars Cargo Starship carries ~85 t payload (so 1 Raptor at ~31% throttle cleanly matches 3.72 m/s² Mars weight!).
        let ship_dry_mass = if scenario == StarshipScenario::MarsSrpLanding {
            205_000.0
        } else {
            120_000.0
        };
        let ship_dry_inertia = if scenario == StarshipScenario::MarsSrpLanding {
            Vec3::new(4.1e7, 2.4e6, 4.1e7)
        } else {
            Vec3::new(2.6e7, 1.5e6, 2.6e7)
        };
        let ship = insert_stage_body(&mut world, s_pose, s_vel, ship_dry_mass, ship_dry_inertia)?;
        world.devices.rockets.register_stage_dry_mass(
            &world.scene,
            ship,
            MassProperties {
                mass: ship_dry_mass,
                local_com: Vec3::new(0.0, -3.5, 0.0), // Heavy 6-Raptor aft skirt & TPS
                inertia: Mat3::from_diagonal(ship_dry_inertia),
            },
        )?;
        let ship_main_lox = world.devices.rockets.insert_tank(
            &mut world.scene,
            PropellantTankSpec {
                body: ship,
                base_local: Vec3::new(0.0, -21.0, 0.0),
                axis_local: Vec3::Y,
                radius_m: 4.4,
                height_m: 15.0,
                liquid_density_kg_m3: 1_140.0,
                initial_mass_kg: s_main_lox_kg,
                slosh_mass_fraction: 0.22,
                slosh_damping_ratio: 0.06,
                capillary_omega2: 0.18,
            },
        )?;
        let ship_main_ch4 = world.devices.rockets.insert_tank(
            &mut world.scene,
            PropellantTankSpec {
                body: ship,
                base_local: Vec3::new(0.0, -5.5, 0.0),
                axis_local: Vec3::Y,
                radius_m: 4.4,
                height_m: 12.0,
                liquid_density_kg_m3: 422.0,
                initial_mass_kg: s_main_ch4_kg,
                slosh_mass_fraction: 0.20,
                slosh_damping_ratio: 0.06,
                capillary_omega2: 0.18,
            },
        )?;
        // Nose LOX header + Mid CH4 header tanks (keeps belly-flop COM near aerodynamic center!).
        let ship_hdr_lox = world.devices.rockets.insert_tank(
            &mut world.scene,
            PropellantTankSpec {
                body: ship,
                base_local: Vec3::new(0.0, 14.5, 0.0),
                axis_local: Vec3::Y,
                radius_m: 1.85,
                height_m: 4.5,
                liquid_density_kg_m3: 1_140.0,
                initial_mass_kg: s_hdr_lox_kg,
                slosh_mass_fraction: 0.18,
                slosh_damping_ratio: 0.08,
                capillary_omega2: 0.35,
            },
        )?;
        let ship_hdr_ch4 = world.devices.rockets.insert_tank(
            &mut world.scene,
            PropellantTankSpec {
                body: ship,
                base_local: Vec3::new(0.0, -5.0, 0.0),
                axis_local: Vec3::Y,
                radius_m: 1.9,
                height_m: 3.5,
                liquid_density_kg_m3: 422.0,
                initial_mass_kg: s_hdr_ch4_kg,
                slosh_mass_fraction: 0.18,
                slosh_damping_ratio: 0.08,
                capillary_omega2: 0.35,
            },
        )?;

        let (sl_t1, sl_t2) = if ship_using_headers {
            (Some(ship_hdr_lox), Some(ship_hdr_ch4))
        } else {
            (Some(ship_main_lox), Some(ship_main_ch4))
        };
        let mut ship_engines = Vec::with_capacity(6);
        // 3 Center Sea-Level Gimballing Raptors (±15° TVC!) with realistic ±30 ms turbopump spool spread!
        let sl_spool_tau = [0.155, 0.185, 0.215];
        for k in 0..3 {
            let a = (k as f64) * (TAU / 3.0);
            let frame = Transform::new(
                Vec3::new(1.10 * a.cos(), -SHIP_HALF_LEN + 0.8, 1.10 * a.sin()),
                up_rot,
            );
            let mut spec =
                RocketEngineSpec::raptor3_sea_level(ship, frame, sl_t1, sl_t2, 15.0_f64.to_radians());
            spec.startup_tau_s = sl_spool_tau[k];
            ship_engines.push(world.devices.rockets.insert_engine(&world.scene, spec)?);
        }
        // 3 Outer Fixed RVac Vacuum Raptors (2.30 m bells, 380 s Vac Isp!)
        for k in 0..3 {
            let a = (k as f64) * (TAU / 3.0) + PI / 3.0;
            let frame = Transform::new(
                Vec3::new(2.85 * a.cos(), -SHIP_HALF_LEN + 0.6, 2.85 * a.sin()),
                up_rot,
            );
            ship_engines.push(world.devices.rockets.insert_engine(
                &world.scene,
                RocketEngineSpec::raptor3_vacuum(ship, frame, Some(ship_main_lox), Some(ship_main_ch4)),
            )?);
        }
        // 4 Mid-Body HLS Lunar Landing Thrusters (y = +8.0 m, 220 kN vac each, 10:1 deep throttle, ±12° TVC)
        let mut ship_hls_thrusters = Vec::with_capacity(4);
        for k in 0..4 {
            let a = (k as f64) * FRAC_PI_2;
            let frame = Transform::new(
                Vec3::new(4.75 * a.cos(), 8.0, 4.75 * a.sin()),
                up_rot,
            );
            ship_hls_thrusters.push(world.devices.rockets.insert_engine(
                &world.scene,
                RocketEngineSpec::hls_lunar_landing_thruster(
                    ship,
                    frame,
                    Some(ship_hdr_lox),
                    Some(ship_hdr_ch4),
                ),
            )?);
        }
        // 4 Real Physical Cold-Gas N2 RCS Thrusters (nose + aft pure-couple yaw pairs):
        // `+X` of thruster frame is applied thrust direction (`Quat::from_rotation_y(-PI/2)` maps `+X -> +Z`,
        // `Quat::from_rotation_y(+PI/2)` maps `+X -> -Z`).
        // Pair (0, 1) generates `+tau_x` (`+480 kN·m` max); Pair (2, 3) generates `-tau_x` (`-480 kN·m` max).
        let rot_pos_z = Quat::from_rotation_y(-FRAC_PI_2);
        let rot_neg_z = Quat::from_rotation_y(FRAC_PI_2);
        let rcs_mounts = [
            Transform::new(Vec3::new(0.0, 20.0, -2.2), rot_pos_z),  // 0: Nose +Z -> +tau_x
            Transform::new(Vec3::new(0.0, -20.0, 4.5), rot_neg_z),  // 1: Aft  -Z -> +tau_x
            Transform::new(Vec3::new(0.0, 20.0, 2.2), rot_neg_z),   // 2: Nose -Z -> -tau_x
            Transform::new(Vec3::new(0.0, -20.0, -4.5), rot_pos_z), // 3: Aft  +Z -> -tau_x
        ];
        let mut rcs_ids = Vec::with_capacity(4);
        for frame in rcs_mounts {
            rcs_ids.push(world.devices.thrusters.insert(
                &world.scene,
                ThrusterSpec {
                    body: ship,
                    frame,
                    max_thrust: 12_000.0,
                    specific_impulse: Some(70.0), // Cold-gas N2 Isp
                    propellant_mass: 220.0,
                    deplete_body_mass: false,
                    min_impulse_bit: 120.0,
                    response_time: 0.025,
                },
            )?);
        }
        let ship_rcs_thrusters: [ThrusterId; 4] = rcs_ids.try_into().unwrap();

        // Register Hot-Staging Blast Impingement: Ship's 6 Raptors blast Booster's top dome!
        world.devices.rockets.register_interstage_blast_target(
            ship,
            booster,
            Vec3::new(0.0, BOOSTER_HALF_LEN, 0.0),
            4.5,
        );

        let ship_aero = world.devices.rocket_aero.insert_body(
            &world.scene,
            AeroBodySpec {
                body: ship,
                axis_local: Vec3::Y,
                diameter_m: VEHICLE_DIAM,
                length_m: SHIP_LEN,
                cd_axial_nose: 0.30,
                cd_axial_tail: 0.90,
                cd_crossflow_subsonic: 1.22,
                cd_crossflow_supersonic: 1.54,
                cp_crossflow_local: 0.5,
                cp_axial_shift_m: 8.5,
                rot_damping_scale: 1.15,
                srp_drag_loss_gain: 0.75,
            },
        )?;
        // 4 Actuated Body Flaps: [fwd_L(+Z), fwd_R(-Z), aft_L(+Z), aft_R(-Z)].
        // Normal `-X` faces belly airflow at `δ = 0`; by right-hand rule, positive `δ ∈ [0, 75°]`
        // about `+Y` on `+Z` flaps (and `-Y` on `-Z` flaps) folds each flap dorsally toward `+X`!
        let flap_specs = [
            (Vec3::new(0.0, 14.5, 5.2), Vec3::Y, 24.0, 1.3),       // fwd_L (+Z)
            (Vec3::new(0.0, 14.5, -5.2), Vec3::NEG_Y, 24.0, 1.3),  // fwd_R (-Z)
            (Vec3::new(0.0, -15.5, 5.9), Vec3::Y, 44.0, 1.9),      // aft_L (+Z)
            (Vec3::new(0.0, -15.5, -5.9), Vec3::NEG_Y, 44.0, 1.9), // aft_R (-Z)
        ];
        let mut flap_vec = Vec::with_capacity(4);
        let init_flaps_deg = [22.0_f64, 22.0, 56.0, 56.0];
        for (idx, (center, hinge, area, arm)) in flap_specs.into_iter().enumerate() {
            let sid = world.devices.rocket_aero.insert_surface(
                &world.scene,
                AeroSurfaceSpec {
                    body: ship,
                    kind: AeroSurfaceKind::BodyFlap,
                    center_local: center,
                    hinge_axis_local: hinge,
                    normal_local: Vec3::NEG_X,
                    area_m2: area,
                    hinge_moment_arm_m: arm,
                    min_deflection_rad: 0.0,
                    max_deflection_rad: 75.0_f64.to_radians(),
                    max_rate_rad_s: 1.6,
                    max_hinge_torque_nm: 1.8e6,
                    cn_max: 1.95,
                },
            )?;
            world
                .devices
                .rocket_aero
                .command_surface(sid, init_flaps_deg[idx].to_radians())?;
            flap_vec.push(sid);
        }
        let ship_flaps: [AeroSurfaceId; 4] = flap_vec.try_into().unwrap();
        let ship_flex = world
            .devices
            .rocket_structure
            .insert(&world.scene, FlexBodySpec::starship_upper(ship))?;
        let ship_imu = world
            .devices
            .space_sensors
            .insert_imu(&world.scene, ImuSpec::tactical_fog(ship, 0x5419_0001))?;
        // Main tanks at 4.5 bar vs high-pressure 5.6 bar landing header tanks (prevents flip cavitation!).
        world
            .devices
            .rockets
            .set_tank_pressurization(ship_main_lox, 450_000.0, 105_000.0, 14.0)?;
        world
            .devices
            .rockets
            .set_tank_pressurization(ship_main_ch4, 460_000.0, 115_000.0, 14.0)?;
        world
            .devices
            .rockets
            .set_tank_pressurization(ship_hdr_lox, 560_000.0, 105_000.0, 12.0)?;
        world
            .devices
            .rockets
            .set_tank_pressurization(ship_hdr_ch4, 560_000.0, 115_000.0, 12.0)?;

        let chopsticks_open_rad = 0.34;
        let chopsticks_cmd_rad = if booster_phase == BoosterPhase::CaughtInChopsticks {
            0.0
        } else {
            chopsticks_open_rad
        };
        world
            .devices
            .motors
            .command(chopstick_left_joint, MotorCommand::Position(chopsticks_cmd_rad))?;
        world
            .devices
            .motors
            .command(chopstick_right_joint, MotorCommand::Position(-chopsticks_cmd_rad))?;

        Ok(Self {
            world,
            scenario,
            time_s: 0.0,
            tower_base,
            chopstick_left,
            chopstick_right,
            chopstick_left_joint,
            chopstick_right_joint,
            booster,
            ship,
            booster_lox,
            booster_ch4,
            booster_engines,
            booster_aero,
            booster_grid_fins,
            booster_flex,
            booster_imu,
            ship_main_lox,
            ship_main_ch4,
            ship_hdr_lox,
            ship_hdr_ch4,
            ship_engines,
            ship_hls_thrusters,
            ship_rcs_thrusters,
            ship_aero,
            ship_flaps,
            ship_flex,
            ship_imu,
            ship_using_headers,
            planet_gravity,
            atmosphere,
            tvc_enabled: true,
            booster_notch: [
                BiquadNotchFilter::new(2.2, 2.8, 1.0 / SIM_DT),
                BiquadNotchFilter::new(2.2, 2.8, 1.0 / SIM_DT),
            ],
            ship_notch: [
                BiquadNotchFilter::new(3.2, 2.8, 1.0 / SIM_DT),
                BiquadNotchFilter::new(3.2, 2.8, 1.0 / SIM_DT),
            ],
            booster_rate_lpf: Vec3::ZERO,
            ship_rate_lpf: Vec3::ZERO,
            booster_wind_int: Vec3::ZERO,
            ship_wind_int: Vec3::ZERO,
            ship_landing_sl_count: 3,
            hold_down_engaged,
            interstage_latched,
            chopsticks_cmd_rad,
            booster_caught: booster_phase == BoosterPhase::CaughtInChopsticks,
            ship_landed: ship_phase == ShipPhase::LandedOnPad,
            booster_phase,
            booster_phase_start_s: 0.0,
            ship_phase,
            ship_phase_start_s: 0.0,
            wind_preset,
            failed_engine_count: 0,
            ever_lifted_off: scenario != StarshipScenario::FullStackMission,
            ever_hot_staged: matches!(scenario, StarshipScenario::BoosterCatch | StarshipScenario::ShipFlipLanding),
            ever_booster_caught: booster_phase == BoosterPhase::CaughtInChopsticks,
            ever_ship_bellyflop: ship_phase == ShipPhase::BellyFlopSkydiving,
            ever_ship_flipped: false,
            ever_ship_landed: ship_phase == ShipPhase::LandedOnPad,
            peak_max_q_kpa: 0.0,
            peak_interstage_blast_mn: 0.0,
            peak_twd_knm: 0.0,
            ship_touchdown_speed_mps: 0.0,
            ship_touchdown_tilt_deg: 0.0,
            booster_catch_speed_mps: 0.0,
            booster_catch_offset_m: 0.0,
            plumes: Vec::with_capacity(900),
            booster_trail: VecDeque::with_capacity(240),
            ship_trail: VecDeque::with_capacity(240),
            user_nudge: None,
            rng: 0x57A5_4190_CAFE_BEEF,
            trail_tick: 0,
        })
    }

    fn next_rand(&mut self) -> f64 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        (x as f64) / (u64::MAX as f64)
    }

    pub fn cycle_wind(&mut self) {
        self.wind_preset = self.wind_preset.next();
        self.world.devices.rocket_aero.set_wind(self.wind_preset.to_wind());
    }

    /// Injects a real Raptor/HLS engine-out failure on an ACTUALLY FIRING engine (creates immediate asymmetric torque!).
    pub fn inject_engine_out(&mut self) -> Result<()> {
        let candidates: Vec<RocketEngineId> = match self.scenario {
            StarshipScenario::ShipFlipLanding
            | StarshipScenario::MarsSrpLanding
            | StarshipScenario::LunarHlsDescent => self
                .ship_hls_thrusters
                .iter()
                .chain(self.ship_engines.iter())
                .copied()
                .collect(),
            _ => self
                .booster_engines
                .iter()
                .chain(self.ship_engines.iter())
                .copied()
                .collect(),
        };
        // Prefer failing a currently lit engine so the user sees the real asymmetric torque kick immediately!
        for &id in &candidates {
            if let Some(r) = self.world.devices.rockets.engine_reading(id)
                && !r.failed
                && r.ignited
            {
                self.world.devices.rockets.set_engine_failed(id, true)?;
                self.failed_engine_count += 1;
                return Ok(());
            }
        }
        // Fallback if engines aren't lit yet (e.g. during skydiving): fail the first healthy candidate!
        for &id in &candidates {
            if let Some(r) = self.world.devices.rockets.engine_reading(id)
                && !r.failed
            {
                self.world.devices.rockets.set_engine_failed(id, true)?;
                self.failed_engine_count += 1;
                break;
            }
        }
        Ok(())
    }

    /// Interactive 3D mouse grab picking on Booster or Ship (in full SI world coordinates!).
    pub fn try_begin_nudge(&mut self, ray_origin: Vec3, ray_dir: Vec3) -> bool {
        let dir = ray_dir.normalize_or_zero();
        if dir.length_squared() < 0.5 {
            return false;
        }
        let mut best: Option<(f64, BodyHandle, Vec3, Vec3)> = None;
        let mut test = |body: BodyHandle, pose: Transform, half_len: f64| {
            for k in -3..=3 {
                let ly = (k as f64 / 3.0) * (half_len * 0.85);
                let center = pose.transform_point(Vec3::new(0.0, ly, 0.0));
                let oc = ray_origin - center;
                let b = oc.dot(dir);
                let r = 8.0;
                let c = oc.length_squared() - r * r;
                let disc = b * b - c;
                if disc >= 0.0 {
                    let t = -b - disc.sqrt();
                    if t > 5.0 && best.as_ref().is_none_or(|h| t < h.0) {
                        let hit_w = ray_origin + dir * t;
                        let local_p = pose.inverse().transform_point(hit_w);
                        best = Some((t, body, local_p, hit_w));
                    }
                }
            }
        };
        if let Ok(bp) = self.world.scene.body_pose(self.booster) {
            test(self.booster, bp, BOOSTER_HALF_LEN);
        }
        if let Ok(sp) = self.world.scene.body_pose(self.ship) {
            test(self.ship, sp, SHIP_HALF_LEN);
        }
        if let Some((depth, body, local_point, target_world)) = best {
            self.user_nudge = Some(UserNudge {
                body,
                local_point,
                target_world,
                depth,
                force_n: 0.0,
            });
            true
        } else {
            false
        }
    }

    pub fn update_nudge_ray(&mut self, ray_origin: Vec3, ray_dir: Vec3) {
        if let Some(n) = &mut self.user_nudge {
            let dir = ray_dir.normalize_or_zero();
            if dir.length_squared() > 0.5 {
                n.target_world = ray_origin + dir * n.depth;
            }
        }
    }

    pub fn end_nudge(&mut self) {
        self.user_nudge = None;
    }

    /// Reads the live tactical FOG IMU angular rate (which includes physical elastic
    /// fuselage bending feedthrough from `RocketStructure`), runs the adaptive 2-axis
    /// biquad notch filter at `mode1_hz` PLUS a high-frequency roll-off filter (`fc = 1.6 Hz`)
    /// to gain-stabilize higher bending modes (`mode2+`) under real IMU transport latency!
    fn filtered_imu_rate_body(
        world: &RigidWorld,
        body: BodyHandle,
        imu: ImuId,
        pose: Transform,
        vel: SpatialVector,
        notch: &mut [BiquadNotchFilter; 2],
        lpf_state: &mut Vec3,
    ) -> Vec3 {
        let raw_gyro = world
            .devices
            .space_sensors
            .imu_reading(imu)
            .filter(|r| r.angular_velocity.length_squared().is_finite())
            .map_or_else(|| pose.rotation.conjugate() * vel.angular, |r| {
                // On step 0 before first `after_step`, `imu_reading` is zero; blend cleanly!
                if r.angular_velocity.length_squared() < 1e-16 && vel.angular.length_squared() > 1e-6 {
                    pose.rotation.conjugate() * vel.angular
                } else {
                    r.angular_velocity
                }
            });
        // `SpaceSensors` already injects physical fuselage elastic bending feedthrough
        // (`RocketStructure::imu_bending_rate_body`) directly into `ImuReading::angular_velocity`!
        if let Some(f1) = world.devices.rocket_structure.mode1_hz_for_body(body) {
            notch[0].retune(f1, 2.2, 1.0 / SIM_DT);
            notch[1].retune(f1, 2.2, 1.0 / SIM_DT);
        }
        let notched = Vec3::new(notch[0].filter(raw_gyro.x), raw_gyro.y, notch[1].filter(raw_gyro.z));
        // NASA SP-8036 bending roll-off filter (fc ≈ 1.8 Hz) gain-stabilizes mode 2+ under IMU latency:
        let alpha = 1.0 - (-TAU * 1.8 * SIM_DT).exp();
        *lpf_state = *lpf_state + (notched - *lpf_state) * alpha;
        *lpf_state
    }

    /// Computes gain-scheduled 2-axis TVC gimbal `[pitch_z, yaw_y]` + per-engine roll bias
    /// using live vehicle transverse inertia `I_trans`, COM-to-nozzle moment arm `L_arm`,
    /// active gimballing thrust `T_gimbal`, and notch-filtered tactical IMU angular rates!
    fn compute_tvc_attitude_gimbal(
        pose: Transform,
        omega_b: Vec3,
        mp: MassProperties,
        desired_up_world: Vec3,
        gimbal_thrust_n: f64,
        engine_y_local: f64,
        omega_n: f64,
        zeta: f64,
        max_rad: f64,
    ) -> ([f64; 2], f64) {
        let up_b = Vec3::Y;
        let des_up_b = (pose.rotation.conjugate() * desired_up_world).normalize_or_zero();
        // Rotation error vector in body frame bringing `+Y` toward `des_up_b`:
        let cross_err = up_b.cross(des_up_b);
        let dot_err = up_b.dot(des_up_b).clamp(-1.0, 1.0);
        let ang = dot_err.acos();
        let err_b = if cross_err.length_squared() > 1e-10 {
            cross_err.normalize() * ang
        } else if dot_err < 0.0 {
            Vec3::Z * PI
        } else {
            Vec3::ZERO
        };
        // Desired angular acceleration in body frame:
        let alpha_cmd = err_b * (omega_n * omega_n) - omega_b * (2.0 * zeta * omega_n);
        let ix = mp.inertia.x_axis.x.max(1.0e5);
        let iz = mp.inertia.z_axis.z.max(1.0e5);
        // Signed lever arm (`> 0` for aft Raptors below COM, `< 0` for upper-ring Lunar HLS thrusters
        // above COM) so TVC gimbal sign automatically inverts cleanly for upper-mounted thrusters!
        let raw_lever = mp.local_com.y - engine_y_local;
        let lever_m = if raw_lever >= 0.0 {
            raw_lever.max(4.0)
        } else {
            raw_lever.min(-4.0)
        };
        let tvc_gain = gimbal_thrust_n.max(1.0e5) * lever_m;
        // Verified TVC signs:
        //   +g0 (pitch_z) -> -tau_z (when lever > 0) ==> g0 = -(I_z * alpha_z) / (T * lever)
        //   +g1 (yaw_y)   -> +tau_x (when lever > 0) ==> g1 = +(I_x * alpha_x) / (T * lever)
        let g0 = (-(iz * alpha_cmd.z) / tvc_gain).clamp(-max_rad, max_rad);
        let g1 = ((ix * alpha_cmd.x) / tvc_gain).clamp(-max_rad, max_rad);
        let roll_cmd = (-omega_b.y * 0.45).clamp(-0.08, 0.08);
        ([g0, g1], roll_cmd)
    }

    fn command_booster_cluster(
        &mut self,
        active_count: usize,
        throttle: f64,
        base_gimbal: [f64; 2],
        roll_bias: f64,
    ) -> Result<()> {
        let eff_gimbal = if self.tvc_enabled { base_gimbal } else { [0.0, 0.0] };
        let eff_roll = if self.tvc_enabled { roll_bias } else { 0.0 };
        for (idx, &eid) in self.booster_engines.iter().enumerate() {
            let lit = idx < active_count && throttle > 0.01;
            let mut g = if idx < 13 { eff_gimbal } else { [0.0, 0.0] };
            if idx < 3 && eff_roll.abs() > 1e-4 {
                // Exact tangential roll vectoring on the 3 center engines:
                // F_xz = (-T g0, 0, -T g1) -> tau_y = T r (-g0 sin a + g1 cos a) = +T r eff_roll!
                let a = (idx as f64) * (TAU / 3.0);
                g[0] -= eff_roll * a.sin();
                g[1] += eff_roll * a.cos();
            }
            self.world.devices.rockets.command_engine(
                eid,
                RocketEngineCommand {
                    ignited: lit,
                    throttle: if lit { throttle.clamp(0.28, 1.0) } else { 0.0 },
                    gimbal_rad: g,
                },
            )?;
        }
        Ok(())
    }

    fn command_ship_cluster(
        &mut self,
        sl_count: usize,
        sl_throttle: f64,
        rvac_lit: bool,
        rvac_throttle: f64,
        base_gimbal: [f64; 2],
        roll_bias: f64,
    ) -> Result<()> {
        // Select symmetric subsets when staging 3 -> 2 -> 1 SL engines!
        let sl_active = |idx: usize| -> bool {
            match sl_count {
                0 => false,
                1 => idx == 0,
                2 => idx == 1 || idx == 2,
                _ => true,
            }
        };
        // Compute live COM-pointing gimbal trim for each active SL engine ONLY when TVC is active!
        // When TVC is locked OFF (`[T]`), gimbals stay strictly at `0°` so off-center engines spin freely!
        let com_l = self
            .world
            .scene
            .body_mass_properties(self.ship)
            .map_or(Vec3::ZERO, |m| m.local_com);
        let eff_gimbal = if self.tvc_enabled { base_gimbal } else { [0.0, 0.0] };
        let eff_roll = if self.tvc_enabled { roll_bias } else { 0.0 };
        for (idx, &eid) in self.ship_engines.iter().enumerate() {
            if idx < 3 {
                let lit = sl_active(idx) && sl_throttle > 0.01;
                let mut g = eff_gimbal;
                if self.tvc_enabled && lit && sl_count < 3 {
                    if let Some(spec) = self.world.devices.rockets.engine_spec(eid) {
                        let d = spec.frame.position - com_l;
                        let dy = (-d.y).max(5.0);
                        // Thrust line points through COM: +g0 tilts thrust toward -X (cancels +d.x),
                        // +g1 tilts thrust toward -Z (cancels +d.z)!
                        g[0] += (d.x / dy).atan();
                        g[1] += (d.z / dy).atan();
                    }
                }
                if sl_count >= 2 && eff_roll.abs() > 1e-4 {
                    let a = (idx as f64) * (TAU / 3.0);
                    g[0] -= eff_roll * a.sin();
                    g[1] += eff_roll * a.cos();
                }
                self.world.devices.rockets.command_engine(
                    eid,
                    RocketEngineCommand {
                        ignited: lit,
                        throttle: if lit { sl_throttle.clamp(0.28, 1.0) } else { 0.0 },
                        gimbal_rad: g,
                    },
                )?;
            } else {
                let lit = rvac_lit && rvac_throttle > 0.01;
                self.world.devices.rockets.command_engine(
                    eid,
                    RocketEngineCommand {
                        ignited: lit,
                        throttle: if lit { rvac_throttle.clamp(0.35, 1.0) } else { 0.0 },
                        gimbal_rad: [0.0, 0.0],
                    },
                )?;
            }
        }
        Ok(())
    }

    fn command_ship_hls_thrusters(
        &mut self,
        lit: bool,
        base_throttle: f64,
        base_gimbal: [f64; 2],
        diff_thr_xz: [f64; 2],
        roll_bias: f64,
    ) -> Result<()> {
        let eff_gimbal = if self.tvc_enabled { base_gimbal } else { [0.0, 0.0] };
        let eff_diff = if self.tvc_enabled { diff_thr_xz } else { [0.0, 0.0] };
        let eff_roll = if self.tvc_enabled { roll_bias } else { 0.0 };
        for (idx, &eid) in self.ship_hls_thrusters.iter().enumerate() {
            let a = (idx as f64) * FRAC_PI_2;
            // k=0 (+X): +diff_z; k=1 (+Z): +diff_x; k=2 (-X): -diff_z; k=3 (-Z): -diff_x
            let thr_off = eff_diff[1] * a.cos() + eff_diff[0] * a.sin();
            let mut g = eff_gimbal;
            if lit && eff_roll.abs() > 1e-4 {
                g[0] -= eff_roll * a.sin();
                g[1] += eff_roll * a.cos();
            }
            self.world.devices.rockets.command_engine(
                eid,
                RocketEngineCommand {
                    ignited: lit && base_throttle > 0.01,
                    throttle: if lit && base_throttle > 0.01 {
                        (base_throttle + thr_off).clamp(0.10, 1.0)
                    } else {
                        0.0
                    },
                    gimbal_rad: g,
                },
            )?;
        }
        Ok(())
    }

    fn switch_ship_to_headers(&mut self) -> Result<()> {
        if self.ship_using_headers {
            return Ok(());
        }
        self.ship_using_headers = true;
        for idx in 0..3 {
            self.world.devices.rockets.set_engine_tanks(
                self.ship_engines[idx],
                Some(self.ship_hdr_lox),
                Some(self.ship_hdr_ch4),
            )?;
        }
        Ok(())
    }

    fn step_booster_gnc(&mut self) -> Result<()> {
        let bp = self.world.scene.body_pose(self.booster)?;
        let bv = self.world.scene.body_velocity(self.booster)?;
        let mp = self.world.scene.body_mass_properties(self.booster)?;
        let mass = mp.mass;
        let elapsed = self.time_s - self.booster_phase_start_s;
        let alt_base = (bp.position.y - BOOSTER_HALF_LEN).max(0.0);
        let up_w = bp.rotation * Vec3::Y;
        let tilt_rad = up_w.y.clamp(-1.0, 1.0).acos();
        let com_w = bp.transform_point(mp.local_com);
        let pin_pos_w = bp.transform_point(Vec3::new(0.0, BOOSTER_CATCH_PIN_LOCAL_Y, 0.0));
        let catch_target = Vec3::new(0.0, CHOPSTICK_CATCH_Y, 0.0);
        // Guide the COM horizontally to tower centerline so pitching never injects
        // non-minimum-phase +43 m catch-pin lever-arm feedback into the outer loop!
        let com_horiz_err = Vec3::new(com_w.x - catch_target.x, 0.0, com_w.z - catch_target.z);
        let pin_horiz_err = Vec3::new(pin_pos_w.x - catch_target.x, 0.0, pin_pos_w.z - catch_target.z);
        let eng_y = -BOOSTER_HALF_LEN + 0.8;
        let omega_b_filt = Self::filtered_imu_rate_body(
            &self.world,
            self.booster,
            self.booster_imu,
            bp,
            bv,
            &mut self.booster_notch,
            &mut self.booster_rate_lpf,
        );

        // Zero grid fins by default; overwritten in GridFinEntry.
        let mut fin_cmds = [0.0_f64; 4];

        match self.booster_phase {
            BoosterPhase::PadIgnition => {
                self.hold_down_engaged = true;
                self.interstage_latched = true;
                self.chopsticks_cmd_rad = 0.36;
                // Staggered 33-Raptor ignition ramp!
                let n_lit = if elapsed < 0.35 {
                    3
                } else if elapsed < 0.70 {
                    13
                } else {
                    33
                };
                self.command_booster_cluster(n_lit, 0.92, [0.0, 0.0], 0.0)?;
                let thrust_y = self.world.devices.rockets.body_total_thrust_world(self.booster).y;
                let ship_m = self.world.scene.body_mass_properties(self.ship)?.mass;
                if elapsed > 1.15 && thrust_y > (mass + ship_m) * GRAVITY * 1.25 {
                    self.hold_down_engaged = false;
                    self.ever_lifted_off = true;
                    self.booster_phase = BoosterPhase::Ascent33Engines;
                    self.booster_phase_start_s = self.time_s;
                }
            }
            BoosterPhase::Ascent33Engines => {
                self.hold_down_engaged = false;
                self.interstage_latched = true;
                self.chopsticks_cmd_rad = 0.36;
                // Smooth gravity turn toward +X (10° pitchover after clearing tower!).
                let pitch_prog = ((alt_base - 180.0) / 3_500.0).clamp(0.0, 1.0);
                let pitch_angle = 0.16 * pitch_prog;
                let des_up = Vec3::new(pitch_angle.sin(), pitch_angle.cos(), 0.0).normalize();
                // Throttle bucket around Max-Q (~20..35 kPa), full thrust otherwise.
                let q_pa = self
                    .world
                    .devices
                    .rocket_aero
                    .body_reading(self.booster_aero)
                    .map_or(0.0, |r| r.dynamic_pressure_pa);
                let thr = if q_pa > 22_000.0 { 0.78 } else { 0.95 };
                let t_gimbal = 13.0 * 2_480_000.0 * thr;
                let (g, r_b) = Self::compute_tvc_attitude_gimbal(
                    bp, omega_b_filt, mp, des_up, t_gimbal, eng_y, 0.95, 1.05, 0.18,
                );
                self.command_booster_cluster(33, thr, g, r_b)?;

                let prop_rem = self.world.devices.rockets.body_propellant_kg(self.booster);
                if alt_base > 4_400.0 || prop_rem < 260_000.0 {
                    self.booster_phase = BoosterPhase::MecoHotStaging;
                    self.booster_phase_start_s = self.time_s;
                    self.ship_phase = ShipPhase::HotStageIgnition;
                    self.ship_phase_start_s = self.time_s;
                }
            }
            BoosterPhase::MecoHotStaging => {
                // Cut outer 30 engines; hold 3 center Raptors at 45% during Ship hot-stage ignition!
                let t_gimbal = 3.0 * 2_480_000.0 * 0.45;
                let (g, r_b) = Self::compute_tvc_attitude_gimbal(
                    bp, omega_b_filt, mp, up_w, t_gimbal, eng_y, 0.85, 1.05, 0.16,
                );
                self.command_booster_cluster(3, 0.45, g, r_b)?;
                if elapsed > 0.55 {
                    self.interstage_latched = false;
                    self.ever_hot_staged = true;
                }
                if elapsed > 2.4 {
                    self.booster_phase = BoosterPhase::FlipAndBoostback;
                    self.booster_phase_start_s = self.time_s;
                }
            }
            BoosterPhase::FlipAndBoostback => {
                // Pitch back toward Starbase tower (x=0, z=0) and fire 13 inner engines to reverse downrange velocity!
                let to_tower = Vec3::new(-com_w.x, 0.0, -com_w.z);
                let dist_h = to_tower.length();
                let v_h = Vec3::new(bv.linear.x, 0.0, bv.linear.z);
                let v_toward = if dist_h > 1.0 {
                    v_h.dot(to_tower / dist_h)
                } else {
                    0.0
                };
                let target_v_return = (dist_h * 0.055).clamp(15.0, 95.0);
                let need_boostback = dist_h > 90.0 && v_toward < target_v_return && elapsed < 14.0;
                if need_boostback {
                    let dir_h = to_tower.normalize_or_zero();
                    let des_up = (dir_h * 0.65 + Vec3::Y * 0.76).normalize();
                    let t_gimbal = 13.0 * 2_480_000.0 * 0.72;
                    let (g, r_b) = Self::compute_tvc_attitude_gimbal(
                        bp, omega_b_filt, mp, des_up, t_gimbal, eng_y, 0.95, 1.08, 0.24,
                    );
                    self.command_booster_cluster(13, 0.72, g, r_b)?;
                } else {
                    self.command_booster_cluster(0, 0.0, [0.0, 0.0], 0.0)?;
                    self.booster_phase = BoosterPhase::GridFinEntry;
                    self.booster_phase_start_s = self.time_s;
                }
            }
            BoosterPhase::GridFinEntry => {
                self.chopsticks_cmd_rad = 0.36;
                // Tail-first aerodynamic descent steered by 4 titanium grid fins!
                let h_err = (pin_pos_w.y - catch_target.y).max(1.0);
                let v_down = (-bv.linear.y).max(10.0);
                let t_go = (h_err / v_down).clamp(2.0, 35.0);
                let zem = Vec3::new(
                    com_w.x + bv.linear.x * t_go,
                    0.0,
                    com_w.z + bv.linear.z * t_go,
                );
                let des_tilt_x = (zem.x * 0.0035).clamp(-0.26, 0.26);
                let des_tilt_z = (zem.z * 0.0035).clamp(-0.26, 0.26);
                let des_up = Vec3::new(des_tilt_x, 1.0, des_tilt_z).normalize();

                let des_up_b = (bp.rotation.conjugate() * des_up).normalize_or_zero();
                let err_b = Vec3::Y.cross(des_up_b);
                let tau_des = err_b * 1.4 - omega_b_filt * 1.9;

                let d_x = (-tau_des.x).clamp(-0.34, 0.34);
                let d_z = (-tau_des.z).clamp(-0.34, 0.34);
                let d_roll = (-omega_b_filt.y * 0.5).clamp(-0.15, 0.15);
                fin_cmds = [d_x + d_roll, d_x - d_roll, d_z + d_roll, d_z - d_roll];
                self.command_booster_cluster(0, 0.0, [0.0, 0.0], 0.0)?;

                let a_13_net = (13.0 * 2_450_000.0 * 0.68 / mass - GRAVITY).max(18.0);
                let h_ignite = (v_down * v_down) / (2.0 * a_13_net) + 210.0;
                if bv.linear.y < -15.0 && (h_err <= h_ignite || alt_base < 680.0) {
                    self.booster_phase = BoosterPhase::CatchBurn13;
                    self.booster_phase_start_s = self.time_s;
                }
            }
            BoosterPhase::CatchBurn13 => {
                self.chopsticks_cmd_rad = 0.32;
                let h_err = (pin_pos_w.y - catch_target.y).max(0.0);
                let v_down = (-bv.linear.y).max(0.0);
                let v_target = (2.0 * 11.5 * (h_err - 110.0).max(0.0)).sqrt().clamp(16.0, 170.0);
                let ay_cmd = GRAVITY + (v_down - v_target) * 1.55;
                let v_h = Vec3::new(bv.linear.x, 0.0, bv.linear.z);
                let a_horiz = (-com_horiz_err * 0.085 - v_h * 0.65).clamp_length_max(4.5);
                let a_vec = Vec3::new(a_horiz.x, ay_cmd.max(4.0), a_horiz.z);
                let des_up = a_vec.normalize();
                let f_req = a_vec.length() * mass;
                let thr = (f_req / (13.0 * 2_480_000.0)).clamp(0.30, 1.0);
                let t_gimbal = 13.0 * 2_480_000.0 * thr;
                let (g, r_b) = Self::compute_tvc_attitude_gimbal(
                    bp, omega_b_filt, mp, des_up, t_gimbal, eng_y, 1.25, 1.12, 0.24,
                );
                self.command_booster_cluster(13, thr, g, r_b)?;

                if v_down < 46.0 || h_err < 185.0 {
                    self.booster_phase = BoosterPhase::CatchHover3;
                    self.booster_phase_start_s = self.time_s;
                }
            }
            BoosterPhase::CatchHover3 => {
                let v_h = Vec3::new(bv.linear.x, 0.0, bv.linear.z);
                // Anti-windup online wind/disturbance observer: only integrates once near hover
                // so transit approach never winds up the integrator!
                if com_horiz_err.length() < 7.0 && v_h.length() < 2.8 {
                    self.booster_wind_int = (self.booster_wind_int - com_horiz_err * (0.014 * SIM_DT))
                        .clamp_length_max(0.55);
                }
                // Smooth continuous approach corridor: holds slightly above chopstick catch plane while
                // removing lateral offset, then descends monotonically into the jaws with zero step bounce!
                let align_hold_y = ((com_horiz_err.length() - 2.2) * 1.5).clamp(0.0, 10.0);
                let h_err = pin_pos_w.y - (catch_target.y + align_hold_y);
                let v_des = if h_err > 14.0 {
                    -((2.0 * 3.8 * (h_err - 3.0)).sqrt()).clamp(3.0, 42.0)
                } else {
                    (-h_err * 0.36 - if align_hold_y < 0.5 { 1.1 } else { 0.0 }).clamp(-3.2, 2.5)
                };
                let ay_cmd = (GRAVITY + (v_des - bv.linear.y) * 2.2).clamp(3.5, 22.0);
                // Overdamped outer COM translation loop + wind observer inside notch-filtered TVC inner loop!
                let a_horiz = (-com_horiz_err * 0.090 - v_h * 0.72 + self.booster_wind_int)
                    .clamp_length_max(if com_horiz_err.length() < 10.0 { 2.0 } else { 3.5 });
                let a_vec = Vec3::new(a_horiz.x, ay_cmd, a_horiz.z);
                let des_up = a_vec.normalize();
                let f_req = a_vec.length() * mass;
                let thr = (f_req / (3.0 * 2_480_000.0)).clamp(0.28, 1.0);
                let t_gimbal = 3.0 * 2_480_000.0 * thr;
                let (g, r_b) = Self::compute_tvc_attitude_gimbal(
                    bp, omega_b_filt, mp, des_up, t_gimbal, eng_y, 1.15, 1.16, 0.24,
                );
                self.command_booster_cluster(3, thr, g, r_b)?;

                let h_pin_err = pin_pos_w.y - catch_target.y;
                if h_pin_err < 24.0 && com_horiz_err.length() < 8.5 {
                    self.chopsticks_cmd_rad = 0.04;
                } else {
                    self.chopsticks_cmd_rad = 0.34;
                }

                // Physical Mechazilla catch: catch pins inside chopstick jaws at low speed & upright!
                if h_pin_err.abs() < 2.8 && pin_horiz_err.length() < 3.0 && bv.linear.length() < 3.8 && tilt_rad < 0.11 {
                    self.booster_caught = true;
                    self.ever_booster_caught = true;
                    self.booster_catch_speed_mps = bv.linear.length();
                    self.booster_catch_offset_m = pin_horiz_err.length();
                    self.chopsticks_cmd_rad = 0.0;
                    self.command_booster_cluster(0, 0.0, [0.0, 0.0], 0.0)?;
                    self.booster_phase = BoosterPhase::CaughtInChopsticks;
                    self.booster_phase_start_s = self.time_s;
                }
            }
            BoosterPhase::CaughtInChopsticks => {
                self.chopsticks_cmd_rad = 0.0;
                self.command_booster_cluster(0, 0.0, [0.0, 0.0], 0.0)?;
            }
        }

        for i in 0..4 {
            self.world
                .devices
                .rocket_aero
                .command_surface(self.booster_grid_fins[i], fin_cmds[i])?;
        }
        self.world.devices.motors.command(
            self.chopstick_left_joint,
            MotorCommand::Position(self.chopsticks_cmd_rad),
        )?;
        self.world.devices.motors.command(
            self.chopstick_right_joint,
            MotorCommand::Position(-self.chopsticks_cmd_rad),
        )?;
        Ok(())
    }

    fn step_ship_gnc(&mut self) -> Result<()> {
        let sp = self.world.scene.body_pose(self.ship)?;
        let sv = self.world.scene.body_velocity(self.ship)?;
        let mp = self.world.scene.body_mass_properties(self.ship)?;
        let mass = mp.mass;
        let elapsed = self.time_s - self.ship_phase_start_s;
        let alt_base = (sp.position.y - SHIP_HALF_LEN).max(0.0);
        let up_w = sp.rotation * Vec3::Y;
        let tilt_rad = up_w.y.clamp(-1.0, 1.0).acos();
        let com_w = sp.transform_point(mp.local_com);
        let pad_pos = Vec3::new(SHIP_PAD_X, SHIP_HALF_LEN, SHIP_PAD_Z);
        let com_horiz_err = Vec3::new(com_w.x - pad_pos.x, 0.0, com_w.z - pad_pos.z);
        let eng_y = -SHIP_HALF_LEN + 0.8;
        let rot_belly = (Quat::from_rotation_y(PI) * Quat::from_rotation_z(FRAC_PI_2)).normalize();
        let omega_b_filt = Self::filtered_imu_rate_body(
            &self.world,
            self.ship,
            self.ship_imu,
            sp,
            sv,
            &mut self.ship_notch,
            &mut self.ship_rate_lpf,
        );

        let mut rcs_yaw_cmd_nm = 0.0_f64;
        let mut flap_cmds = [70.0_f64.to_radians(); 4]; // Folded flat during ascent!
        let mut hls_active = false;
        let is_mars = self.atmosphere == PlanetAtmosphere::MarsCospar;
        let is_lunar = self.atmosphere == PlanetAtmosphere::Vacuum;
        let g_local = self.planet_gravity;
        let raptor_ref_n = if is_mars || is_lunar { 2_740_000.0 } else { 2_480_000.0 };

        match self.ship_phase {
            ShipPhase::StackedOnBooster => {
                self.command_ship_cluster(0, 0.0, false, 0.0, [0.0, 0.0], 0.0)?;
            }
            ShipPhase::HotStageIgnition => {
                // Ignite all 6 Raptors (3 SL + 3 RVac) atop Booster's hot-stage ring!
                let t_gimbal = 3.0 * 2_480_000.0 * 0.90;
                let (g, r_b) = Self::compute_tvc_attitude_gimbal(
                    sp, omega_b_filt, mp, up_w, t_gimbal, eng_y, 1.0, 1.1, 0.18,
                );
                self.command_ship_cluster(3, 0.90, true, 0.90, g, r_b)?;
                if !self.interstage_latched && elapsed > 2.2 {
                    self.ship_phase = ShipPhase::ShipAscent6Engines;
                    self.ship_phase_start_s = self.time_s;
                }
            }
            ShipPhase::ShipAscent6Engines => {
                // Climb steeply toward high-altitude suborbital test apogee (~10 km) aimed back over Pad B!
                let to_pad = Vec3::new(pad_pos.x - com_w.x, 0.0, pad_pos.z - com_w.z);
                let des_up = (Vec3::Y + to_pad.normalize_or_zero() * (to_pad.length() * 0.0003).clamp(0.0, 0.25))
                    .normalize();
                let t_gimbal = 3.0 * 2_480_000.0 * 0.85;
                let (g, r_b) = Self::compute_tvc_attitude_gimbal(
                    sp, omega_b_filt, mp, des_up, t_gimbal, eng_y, 1.05, 1.1, 0.20,
                );
                self.command_ship_cluster(3, 0.85, true, 0.85, g, r_b)?;
                let main_prop = self.world.devices.rockets.tank_reading(self.ship_main_lox).map_or(0.0, |r| r.mass_remaining_kg);
                if elapsed > 12.0 || alt_base > 9_500.0 || main_prop < 12_000.0 {
                    self.switch_ship_to_headers()?;
                    self.command_ship_cluster(0, 0.0, false, 0.0, [0.0, 0.0], 0.0)?;
                    self.ship_phase = ShipPhase::ApogeeCoastFlipToBelly;
                    self.ship_phase_start_s = self.time_s;
                }
            }
            ShipPhase::ApogeeCoastFlipToBelly => {
                self.switch_ship_to_headers()?;
                // Pitch smoothly into horizontal 90° belly-flop orientation (belly `-X` facing down `-Y`)!
                let q_err = (sp.rotation.conjugate() * rot_belly).normalize();
                let mut e_v = Vec3::new(q_err.x, q_err.y, q_err.z);
                if q_err.w < 0.0 {
                    e_v = -e_v;
                }
                let ang = 2.0 * e_v.length().atan2(q_err.w.abs());
                flap_cmds = [
                    22.0_f64.to_radians(),
                    22.0_f64.to_radians(),
                    56.0_f64.to_radians(),
                    56.0_f64.to_radians(),
                ];
                if ang > 0.25 && elapsed < 6.0 {
                    let axis_b = e_v.normalize_or_zero();
                    let g0 = (-(axis_b.z * 0.35 - omega_b_filt.z * 0.45)).clamp(-0.24, 0.24);
                    let g1 = (axis_b.x * 0.35 - omega_b_filt.x * 0.45).clamp(-0.24, 0.24);
                    self.command_ship_cluster(1, 0.30, false, 0.0, [g0, g1], 0.0)?;
                } else {
                    self.command_ship_cluster(0, 0.0, false, 0.0, [0.0, 0.0], 0.0)?;
                }
                if sv.linear.y < -15.0 || elapsed > 10.0 {
                    self.ever_ship_bellyflop = true;
                    self.ship_phase = ShipPhase::BellyFlopSkydiving;
                    self.ship_phase_start_s = self.time_s;
                }
            }
            ShipPhase::BellyFlopSkydiving => {
                self.ever_ship_bellyflop = true;
                self.command_ship_cluster(0, 0.0, false, 0.0, [0.0, 0.0], 0.0)?;

                // 4-Flap 6-DOF Skydiver Controller + Cold-Gas Nose Yaw Thruster!
                // On Earth, trigger flip at h = 600 m (~145 m lead); on Mars, thin 699 Pa CO2
                // keeps Starship supersonic (~320 m/s), requiring high-altitude SRP flip at h = 3,900 m!
                let flip_alt_m = if is_mars { 4_050.0 } else { 600.0 };
                let flip_lead_x = if is_mars { 360.0 } else { 145.0 };
                let v_down = (-sv.linear.y).max(20.0);
                let t_go = ((alt_base - flip_alt_m).max(25.0) / v_down).clamp(2.0, 30.0);
                let zem_x = (com_w.x - (pad_pos.x - flip_lead_x)) + sv.linear.x * t_go * 0.65;
                let zem_z = (com_w.z - pad_pos.z) + sv.linear.z * t_go * 0.65;

                // World-frame glide tilt:
                let pitch_bias = (zem_x * 0.0026).clamp(-0.26, 0.26);
                let roll_bias = (-zem_z * 0.0026).clamp(-0.24, 0.24);
                let target_rot = (Quat::from_rotation_x(roll_bias)
                    * Quat::from_rotation_z(pitch_bias)
                    * rot_belly)
                .normalize();

                let mut q_err = (sp.rotation.conjugate() * target_rot).normalize();
                if q_err.w < 0.0 {
                    q_err = -q_err;
                }
                let ev = Vec3::new(q_err.x, q_err.y, q_err.z);
                let s = ev.length();
                let att_err_b = if s > 1e-9 {
                    ev * (2.0 * s.atan2(q_err.w) / s)
                } else {
                    ev * 2.0
                };

                // Real Starship cold-gas N2 nose/aft RCS pods keep yaw aligned to landing flip plane!
                rcs_yaw_cmd_nm =
                    (att_err_b.x * 2.2e6 - omega_b_filt.x * 3.2e6).clamp(-4.8e5, 4.8e5);

                let u_pitch_z = (att_err_b.z * 1.55 - omega_b_filt.z * 2.20).clamp(-0.65, 0.65);
                let u_roll_y = (att_err_b.y * 1.35 - omega_b_filt.y * 1.90).clamp(-0.48, 0.48);

                // Asymmetric real-Starship trim: small fwd flaps open ~22°, huge aft flaps folded ~56°!
                let nom_fwd = 22.0_f64.to_radians();
                let nom_aft = 56.0_f64.to_radians();
                let d_fl = (nom_fwd + u_pitch_z - u_roll_y).clamp(0.0, 75.0_f64.to_radians());
                let d_fr = (nom_fwd + u_pitch_z + u_roll_y).clamp(0.0, 75.0_f64.to_radians());
                let d_al = (nom_aft - u_pitch_z - u_roll_y).clamp(0.0, 75.0_f64.to_radians());
                let d_ar = (nom_aft - u_pitch_z + u_roll_y).clamp(0.0, 75.0_f64.to_radians());
                flap_cmds = [d_fl, d_fr, d_al, d_ar];

                if alt_base <= flip_alt_m && sv.linear.y < -25.0 {
                    self.ship_landing_sl_count = 3;
                    self.ship_phase = ShipPhase::FlipAndBurn3Engines;
                    self.ship_phase_start_s = self.time_s;
                }
            }
            ShipPhase::FlipAndBurn3Engines => {
                // Iconic Starship Flip (& Mars Supersonic Retropropulsion!) Maneuver!
                flap_cmds = [
                    8.0_f64.to_radians(),
                    8.0_f64.to_radians(),
                    75.0_f64.to_radians(),
                    75.0_f64.to_radians(),
                ];
                let v_h = Vec3::new(sv.linear.x, 0.0, sv.linear.z);
                let a_horiz = if is_mars {
                    (-com_horiz_err * 0.012 - v_h * 0.55).clamp_length_max(4.5)
                } else {
                    (-com_horiz_err * 0.065 - v_h * 0.50).clamp_length_max(4.0)
                };
                let des_up = Vec3::new(a_horiz.x, 12.0, a_horiz.z).normalize();
                let v_down = (-sv.linear.y).max(0.0);
                // Keep throttle low while horizontal (> 50° tilt) so Raptors whip Starship upright
                // with minimal sideways drift, then ramp to 92% once upright for SRP braking!
                let thr = if tilt_rad > 0.85 {
                    0.36
                } else if v_down > 28.0 {
                    0.92
                } else {
                    0.42
                };
                let t_gimbal = 3.0 * raptor_ref_n * thr;
                let (g, r_b) = Self::compute_tvc_attitude_gimbal(
                    sp, omega_b_filt, mp, des_up, t_gimbal, eng_y, 1.32, 1.05, 0.26,
                );
                self.command_ship_cluster(3, thr, false, 0.0, g, r_b)?;

                if tilt_rad < 0.34 && elapsed > 1.6 {
                    self.ever_ship_flipped = true;
                    self.ship_phase = ShipPhase::LandingBurn2To1;
                    self.ship_phase_start_s = self.time_s;
                }
            }
            ShipPhase::LandingBurn2To1 => {
                self.ever_ship_flipped = true;
                flap_cmds = [72.0_f64.to_radians(); 4];
                let v_h = Vec3::new(sv.linear.x, 0.0, sv.linear.z);
                // Anti-windup online wind/disturbance observer: integrates only once settled near pad!
                if com_horiz_err.length() < 8.0 && v_h.length() < 2.8 {
                    self.ship_wind_int = (self.ship_wind_int - com_horiz_err * (0.014 * SIM_DT))
                        .clamp_length_max(0.55);
                }
                let hold_h = ((com_horiz_err.length() - 3.8) * 1.1).clamp(0.0, 20.0);
                let h_err = alt_base - hold_h;
                let brake_a = if is_mars { 12.2 } else { 4.4 };
                let v_cap = if is_mars { 320.0 } else { 45.0 };
                let v_des = if h_err > 12.0 {
                    -((2.0 * brake_a * (h_err - 2.0)).sqrt()).clamp(2.5, v_cap)
                } else {
                    (-h_err * 0.42 - if hold_h < 0.4 { 1.3 } else { 0.0 }).clamp(-3.8, 2.5)
                };
                let ay_cmd = (g_local + (v_des - sv.linear.y) * 2.2).clamp(g_local * 0.25, 28.0);
                let a_horiz = if is_mars {
                    // Cascade velocity-limited + tilt-angle-capped horizontal loop:
                    // High-altitude SRP braking uses up to 22° tilt (6.5 m/s²) to cancel post-flip drift fast,
                    // then tightens smoothly to 8.5° tilt for dead-center Martian touchdown!
                    let (v_h_cap, max_tilt, max_a) = if alt_base > 60.0 {
                        (24.0, 22.0_f64.to_radians(), 6.5)
                    } else if com_horiz_err.length() > 16.0 {
                        (10.0, 16.0_f64.to_radians(), 2.0)
                    } else {
                        (4.0, 8.5_f64.to_radians(), 1.2)
                    };
                    let v_h_des = (-com_horiz_err * 0.18).clamp_length_max(v_h_cap);
                    ((v_h_des - v_h) * 0.72 + self.ship_wind_int)
                        .clamp_length_max((ay_cmd * max_tilt.tan()).min(max_a))
                } else {
                    (-com_horiz_err * 0.095 - v_h * 0.74 + self.ship_wind_int)
                        .clamp_length_max(if alt_base < 18.0 { 1.8 } else { 4.4 })
                };
                let a_vec = Vec3::new(a_horiz.x, ay_cmd, a_horiz.z);
                let des_up = a_vec.normalize();
                let f_req = a_vec.length() * mass;

                // Hysteretic discrete engine staging (3 -> 2 -> 1) prevents Raptor on/off chatter!
                self.ship_landing_sl_count = match self.ship_landing_sl_count {
                    3 if f_req < 4_100_000.0 => 2,
                    2 if f_req > 4_800_000.0 => 3,
                    2 if f_req < 2_050_000.0 => 1,
                    1 if f_req > 2_380_000.0 => 2,
                    n => n.clamp(1, 3),
                };
                let n_sl = self.ship_landing_sl_count;
                let thr = (f_req / ((n_sl as f64) * raptor_ref_n)).clamp(0.28, 1.0);
                let t_gimbal = (n_sl as f64) * raptor_ref_n * thr;
                let (g, r_b) = Self::compute_tvc_attitude_gimbal(
                    sp, omega_b_filt, mp, des_up, t_gimbal, eng_y, 1.25, 1.14, 0.24,
                );
                self.command_ship_cluster(n_sl, thr, false, 0.0, g, r_b)?;

                if alt_base < 0.85 && sv.linear.length() < 4.5 && tilt_rad < 0.14 {
                    self.ship_landed = true;
                    self.ever_ship_landed = true;
                    self.ship_touchdown_speed_mps = sv.linear.length();
                    self.ship_touchdown_tilt_deg = tilt_rad.to_degrees();
                    self.command_ship_cluster(0, 0.0, false, 0.0, [0.0, 0.0], 0.0)?;
                    self.ship_phase = ShipPhase::LandedOnPad;
                    self.ship_phase_start_s = self.time_s;
                }
            }
            ShipPhase::LunarBrakingRaptors => {
                // Artemis Lunar HLS Powered Descent Braking in pure vacuum (1/6 g, zero aero)!
                // Fires 1-2 vacuum-Isp SL Raptors to kill horizontal orbital speed (92 m/s -> < 15 m/s)
                // and arrest initial sink rate, then cuts main Raptors (since 1 Raptor @ 28% min throttle
                // is ~2.7x lunar weight!) and hands off to the 4 Mid-Body HLS Landing Thrusters!
                flap_cmds = [72.0_f64.to_radians(); 4];
                let v_h = Vec3::new(sv.linear.x, 0.0, sv.linear.z);
                let ay_cmd = (g_local + (-18.0 - sv.linear.y) * 0.55).clamp(1.6, 8.0);
                let a_horiz = (-com_horiz_err * 0.010 - v_h * 0.45).clamp_length_max(9.5);
                let a_vec = Vec3::new(a_horiz.x, ay_cmd, a_horiz.z);
                let des_up = a_vec.normalize();
                let f_req = a_vec.length() * mass;
                let (n_sl, thr) = if f_req > 2_450_000.0 {
                    (2, (f_req / (2.0 * raptor_ref_n)).clamp(0.28, 1.0))
                } else {
                    (1, (f_req / raptor_ref_n).clamp(0.28, 1.0))
                };
                let t_gimbal = (n_sl as f64) * raptor_ref_n * thr;
                let (g, r_b) = Self::compute_tvc_attitude_gimbal(
                    sp, omega_b_filt, mp, des_up, t_gimbal, eng_y, 1.15, 1.15, 0.24,
                );
                self.command_ship_cluster(n_sl, thr, false, 0.0, g, r_b)?;

                // Cut main Raptors as soon as horizontal orbital speed is killed (or upward bounce begins)!
                if (v_h.length() < 9.5 && sv.linear.y > -22.0) || sv.linear.y > -2.5 || alt_base < 220.0 {
                    self.ever_ship_flipped = true;
                    self.ship_phase = ShipPhase::LunarHlsFinalLanding;
                    self.ship_phase_start_s = self.time_s;
                }
            }
            ShipPhase::LunarHlsFinalLanding => {
                // Final Lunar South Pole Touchdown on 4 Mid-Body HLS Landing Thrusters (y = +8 m)!
                // Main Raptors cut off to protect lunar regolith from cratering & satisfy 1/6 g throttle floor!
                hls_active = true;
                self.ever_ship_flipped = true;
                flap_cmds = [72.0_f64.to_radians(); 4];
                self.command_ship_cluster(0, 0.0, false, 0.0, [0.0, 0.0], 0.0)?;

                let v_h = Vec3::new(sv.linear.x, 0.0, sv.linear.z);
                let hold_h = ((com_horiz_err.length() - 4.0) * 0.75).clamp(0.0, 26.0);
                let h_err = alt_base - hold_h;
                let v_des = if h_err > 14.0 {
                    -((2.0 * 0.90 * (h_err - 2.0)).sqrt()).clamp(2.2, 22.0)
                } else {
                    (-h_err * 0.36 - if hold_h < 0.5 { 1.1 } else { 0.0 }).clamp(-3.0, 2.0)
                };
                // Keep vertical thrust floor >= 1.15 m/s² (~0.24 throttle) so 4 HLS pods ALWAYS retain
                // crisp differential throttling + TVC attitude authority throughout lunar descent!
                let ay_cmd = (g_local + (v_des - sv.linear.y) * 1.85).clamp(1.15, 4.4);
                // Cascade velocity-limited + tilt-angle-capped Lunar HLS horizontal loop (never exceeds 15° tilt!)
                let v_h_des = (-com_horiz_err * 0.11)
                    .clamp_length_max(if alt_base < 18.0 { 2.5 } else { 4.8 });
                let max_tilt = if alt_base < 16.0 {
                    8.0_f64.to_radians()
                } else {
                    15.0_f64.to_radians()
                };
                let a_horiz = ((v_h_des - v_h) * 0.60)
                    .clamp_length_max((ay_cmd * max_tilt.tan()).min(0.95));
                let a_vec = Vec3::new(a_horiz.x, ay_cmd, a_horiz.z);
                let des_up = a_vec.normalize();
                let f_req = a_vec.length() * mass;
                // Keep base_thr inside [0.22, 0.88] so 10%-min-throttle HLS pods ALWAYS have full diff-throttle authority!
                let base_thr = (f_req / (4.0 * 220_000.0)).clamp(0.22, 0.88);
                let t_hls = 4.0 * 220_000.0 * base_thr;

                // Clean physical 50/50 torque allocation: 50% signed TVC (upper ring lever < 0) + 50% diff throttle!
                let (g_full, r_b) = Self::compute_tvc_attitude_gimbal(
                    sp, omega_b_filt, mp, des_up, t_hls, 8.0, 1.15, 1.25, 0.21,
                );
                let g = [g_full[0] * 0.5, g_full[1] * 0.5];
                let des_up_b = (sp.rotation.conjugate() * des_up).normalize_or_zero();
                let cross_e = Vec3::Y.cross(des_up_b);
                let ang_e = Vec3::Y.dot(des_up_b).clamp(-1.0, 1.0).acos();
                let err_b = if cross_e.length_squared() > 1e-10 {
                    cross_e.normalize() * ang_e
                } else {
                    Vec3::ZERO
                };
                let alpha_c = (err_b * (1.15 * 1.15) - omega_b_filt * (2.0 * 1.25 * 1.15)) * 0.5;
                let denom_diff = 2.0 * 4.75 * 220_000.0;
                let diff_z = (mp.inertia.z_axis.z * alpha_c.z / denom_diff).clamp(-0.32, 0.32);
                let diff_x = (-mp.inertia.x_axis.x * alpha_c.x / denom_diff).clamp(-0.32, 0.32);
                self.command_ship_hls_thrusters(true, base_thr, g, [diff_x, diff_z], r_b)?;

                if alt_base < 0.85 && sv.linear.length() < 3.5 && tilt_rad < 0.12 {
                    self.ship_landed = true;
                    self.ever_ship_landed = true;
                    self.ship_touchdown_speed_mps = sv.linear.length();
                    self.ship_touchdown_tilt_deg = tilt_rad.to_degrees();
                    self.command_ship_hls_thrusters(false, 0.0, [0.0, 0.0], [0.0, 0.0], 0.0)?;
                    self.ship_phase = ShipPhase::LandedOnPad;
                    self.ship_phase_start_s = self.time_s;
                }
            }
            ShipPhase::LandedOnPad => {
                flap_cmds = [72.0_f64.to_radians(); 4];
                self.command_ship_cluster(0, 0.0, false, 0.0, [0.0, 0.0], 0.0)?;
            }
        }
        if !hls_active {
            self.command_ship_hls_thrusters(false, 0.0, [0.0, 0.0], [0.0, 0.0], 0.0)?;
        }

        // Command real physical Cold-Gas N2 RCS pods (0..2 = +tau_x couple, 2..4 = -tau_x couple; max 480 kN·m)!
        let u_rcs = if self.tvc_enabled {
            (rcs_yaw_cmd_nm / 480_000.0).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let thr_pos = u_rcs.max(0.0);
        let thr_neg = (-u_rcs).max(0.0);
        let rcs_throttles = [thr_pos, thr_pos, thr_neg, thr_neg];
        for i in 0..4 {
            self.world
                .devices
                .thrusters
                .set_throttle(self.ship_rcs_thrusters[i], rcs_throttles[i])?;
        }

        for i in 0..4 {
            self.world
                .devices
                .rocket_aero
                .command_surface(self.ship_flaps[i], flap_cmds[i])?;
        }
        Ok(())
    }

    pub fn step(&mut self, dt: f64) -> Result<()> {
        self.step_booster_gnc()?;
        self.step_ship_gnc()?;

        // Execute Featherstone + RocketPropulsion + RocketAero + physical OLM hold-down clamps,
        // interstage structural ring clamp, Mechazilla chopstick catch-pin cradle, ground pad contact,
        // and interactive 3D user nudge!
        let booster_h = self.booster;
        let ship_h = self.ship;
        let hold_down = self.hold_down_engaged;
        let interstage = self.interstage_latched;
        let caught = self.booster_caught;
        let nudge = self.user_nudge;
        let mut nudge_f_accum = 0.0_f64;
        let mut samples = 0usize;

        self.world.step_with_forces(
            dt,
            |_, _| false,
            |scene, state, _sub_dt, wrenches| {
                let poses = state.body_q.host()?;
                let vels = state.body_qd.host()?;
                let coms = scene.model().body_com.host()?;
                let masses = scene.model().body_mass.host()?;

                let bi = scene.body_index(booster_h).unwrap().index();
                let si = scene.body_index(ship_h).unwrap().index();
                let bp = poses[bi];
                let sp = poses[si];
                let bv = vels[bi];
                let sv = vels[si];
                let b_com = bp.transform_point(coms[bi]);
                let s_com = sp.transform_point(coms[si]);

                let apply_at = |w: &mut [SpatialVector], idx: usize, com: Vec3, pt: Vec3, f: Vec3| {
                    w[idx].linear += f;
                    w[idx].angular += (pt - com).cross(f);
                };
                let vel_at = |v: SpatialVector, com: Vec3, pt: Vec3| v.linear + v.angular.cross(pt - com);

                // A. Orbital Launch Mount (OLM) Hold-Down Clamps (4-point perimeter skirt support).
                if hold_down {
                    for k in 0..4 {
                        let a = k as f64 * FRAC_PI_2;
                        let local_pt = Vec3::new(4.5 * a.cos(), -BOOSTER_HALF_LEN, 4.5 * a.sin());
                        let anchor_w = Vec3::new(4.5 * a.cos(), OLM_DECK_Y, 4.5 * a.sin());
                        let pt_w = bp.transform_point(local_pt);
                        let v_w = vel_at(bv, b_com, pt_w);
                        let f = (anchor_w - pt_w) * 1.4e8 - v_w * 2.2e7;
                        apply_at(wrenches, bi, b_com, pt_w, f);
                    }
                }

                // B. Interstage Hot-Stage Ring Structural Clamp (4 perimeter pins linking Booster top -> Ship base).
                if interstage {
                    for k in 0..4 {
                        let a = k as f64 * FRAC_PI_2;
                        let b_pt_l = Vec3::new(4.5 * a.cos(), BOOSTER_HALF_LEN + HOTSTAGE_RING_H, 4.5 * a.sin());
                        let s_pt_l = Vec3::new(4.5 * a.cos(), -SHIP_HALF_LEN, 4.5 * a.sin());
                        let b_pt_w = bp.transform_point(b_pt_l);
                        let s_pt_w = sp.transform_point(s_pt_l);
                        let vb = vel_at(bv, b_com, b_pt_w);
                        let vs = vel_at(sv, s_com, s_pt_w);
                        let f_on_ship = (b_pt_w - s_pt_w) * 9.0e7 + (vb - vs) * 1.6e7;
                        apply_at(wrenches, si, s_com, s_pt_w, f_on_ship);
                        apply_at(wrenches, bi, b_com, b_pt_w, -f_on_ship);
                    }
                }

                // C. Mechazilla Chopstick Catch-Pin Physical Cradle (supports 275+ t Booster on 2 steel pins!).
                if caught {
                    for &sign_z in &[-1.0, 1.0] {
                        let pin_l = Vec3::new(0.0, BOOSTER_CATCH_PIN_LOCAL_Y, sign_z * 4.8);
                        let pin_w = bp.transform_point(pin_l);
                        let rest_w = Vec3::new(0.0, CHOPSTICK_CATCH_Y, sign_z * 4.8);
                        let vp = vel_at(bv, b_com, pin_w);
                        let f = (rest_w - pin_w) * 1.2e7 - vp * 3.8e6;
                        apply_at(wrenches, bi, b_com, pin_w, f);
                    }
                }

                // D. Ground / Pad B Contact (aft skirt + nose/top perimeter, strictly `pt_w.y < 0.0` + Coulomb friction!).
                for &(idx, pose, vel, com, half_l) in &[
                    (bi, bp, bv, b_com, BOOSTER_HALF_LEN),
                    (si, sp, sv, s_com, SHIP_HALF_LEN),
                ] {
                    let m = masses[idx];
                    for &end_y in &[-half_l, half_l] {
                        let rad = if end_y < 0.0 { 4.5 } else { 2.2 };
                        for k in 0..4 {
                            let a = k as f64 * FRAC_PI_2;
                            let pt_l = Vec3::new(rad * a.cos(), end_y, rad * a.sin());
                            let pt_w = pose.transform_point(pt_l);
                            if pt_w.y < 0.0 {
                                let pen = -pt_w.y;
                                let vp = vel_at(vel, com, pt_w);
                                let k_g = m * 65.0;
                                let c_g = m * 14.0;
                                let fy = (k_g * pen - c_g * vp.y).max(0.0);
                                // True Coulomb friction cone: `|f_t| <= mu * fy` (strictly zero when rebounding!).
                                let v_t = Vec3::new(vp.x, 0.0, vp.z);
                                let f_fric = (-v_t * (m * 4.0)).clamp_length_max(0.85 * fy);
                                apply_at(wrenches, idx, com, pt_w, Vec3::new(f_fric.x, fy, f_fric.z));
                            }
                        }
                    }
                }

                // E. Interactive 3D Mouse Nudge Force!
                if let Some(n) = nudge
                    && let Some(ni) = scene.body_index(n.body)
                {
                    let ni = ni.index();
                    let np = poses[ni];
                    let nv = vels[ni];
                    let n_com = np.transform_point(coms[ni]);
                    let p_w = np.transform_point(n.local_point);
                    let v_w = vel_at(nv, n_com, p_w);
                    let m_ref = masses[ni];
                    let f_raw = (n.target_world - p_w) * (m_ref * 0.45) - v_w * (m_ref * 0.35);
                    let f_cap = m_ref * 18.0;
                    let fm = f_raw.length();
                    let f_app = if fm > f_cap { f_raw * (f_cap / fm) } else { f_raw };
                    apply_at(wrenches, ni, n_com, p_w, f_app);
                    nudge_f_accum += f_app.length();
                }

                samples += 1;
                Ok(())
            },
        )?;

        if let Some(n) = &mut self.user_nudge {
            n.force_n = nudge_f_accum / samples.max(1) as f64;
        }
        self.time_s += dt;

        // Track hardware-validation peaks.
        if let Some(r) = self.world.devices.rocket_aero.body_reading(self.booster_aero) {
            self.peak_max_q_kpa = self.peak_max_q_kpa.max(r.dynamic_pressure_pa * 1e-3);
        }
        self.peak_interstage_blast_mn = self
            .peak_interstage_blast_mn
            .max(self.world.devices.rockets.last_interstage_blast_n() * 1e-6);
        for &eid in self.booster_engines.iter().chain(self.ship_engines.iter()) {
            if let Some(r) = self.world.devices.rockets.engine_reading(eid) {
                self.peak_twd_knm = self.peak_twd_knm.max(r.twd_torque_nm * 1e-3);
            }
        }

        self.emit_raptor_plumes(dt)?;
        self.trail_tick += 1;
        if self.trail_tick.is_multiple_of(4) {
            if self.booster_trail.len() >= 240 {
                self.booster_trail.pop_front();
            }
            if self.ship_trail.len() >= 240 {
                self.ship_trail.pop_front();
            }
            self.booster_trail.push_back(self.world.scene.body_pose(self.booster)?.position);
            self.ship_trail.push_back(self.world.scene.body_pose(self.ship)?.position);
        }
        Ok(())
    }

    fn emit_raptor_plumes(&mut self, dt: f64) -> Result<()> {
        let bp = self.world.scene.body_pose(self.booster)?;
        let bv = self.world.scene.body_velocity(self.booster)?.linear;
        let sp = self.world.scene.body_pose(self.ship)?;
        let sv = self.world.scene.body_velocity(self.ship)?.linear;
        let b_lit = self
            .booster_engines
            .iter()
            .filter(|&&e| self.world.devices.rockets.engine_reading(e).is_some_and(|r| r.ignited))
            .count();
        let b_stride = if b_lit > 14 { 3 } else { 1 };
        let b_ids = self.booster_engines.clone();
        let s_ids = self.ship_engines.clone();
        let hls_ids = self.ship_hls_thrusters.clone();

        // Spawn supersonic methalox exhaust puffs (`kind = 2.0`) scaled by Prandtl–Pack pressure ratio!
        let mut emit_for = |ids: &[RocketEngineId], pose: Transform, vel: Vec3, stride: usize| {
            for (idx, &eid) in ids.iter().enumerate() {
                if !idx.is_multiple_of(stride) {
                    continue;
                }
                let (Some(spec), Some(read)) = (
                    self.world.devices.rockets.engine_spec(eid),
                    self.world.devices.rockets.engine_reading(eid),
                ) else {
                    continue;
                };
                if read.chamber_throttle < 0.05 {
                    continue;
                }
                let base_f = pose.compose(&spec.frame);
                let gq = (Quat::from_rotation_z(read.gimbal_rad[0])
                    * Quat::from_rotation_y(read.gimbal_rad[1]))
                .normalize();
                let exh_w = -((base_f.rotation * gq) * Vec3::X).normalize();
                let exit_w = base_f.position + exh_w * 1.4;
                let bloom = (read.pressure_ratio.clamp(0.4, 20.0)).sqrt();
                let r0 = spec.exit_diameter_m
                    * (0.65 + 0.35 * read.chamber_throttle)
                    * (0.7 + 0.3 * bloom);
                let spd = 110.0 + 95.0 * read.chamber_throttle;
                let seed = (self.next_rand() * 100.0) as f32;
                let off = self.next_rand() * 1.2;
                if self.plumes.len() >= 900 {
                    self.plumes.swap_remove(0);
                }
                self.plumes.push(PlumePuff {
                    pos: exit_w + exh_w * off,
                    vel: vel + exh_w * spd,
                    radius: r0,
                    growth_rate: (8.5 + 10.0 * bloom) * read.chamber_throttle,
                    stretch: (3.2 - 0.5 * (bloom - 1.0).clamp(0.0, 2.0)).clamp(1.6, 3.6),
                    age: 0.0,
                    max_age: 0.24 + 0.14 * read.chamber_throttle,
                    base_opacity: (0.48 + 0.34 * read.chamber_throttle) as f32,
                    seed,
                    kind: 2.0,
                });
            }
        };

        emit_for(&b_ids, bp, bv, b_stride);
        emit_for(&s_ids, sp, sv, 1);
        emit_for(&hls_ids, sp, sv, 1);

        // Emit visible white cold-gas N2 RCS thruster puffs whenever any physical RCS pod fires!
        let rcs_ids = self.ship_rcs_thrusters;
        for tid in rcs_ids {
            let (Some(spec), Some(read)) = (
                self.world.devices.thrusters.spec(tid),
                self.world.devices.thrusters.reading(tid),
            ) else {
                continue;
            };
            if read.thrust < 200.0 {
                continue;
            }
            let base_f = sp.compose(&spec.frame);
            let exh_w = -(base_f.rotation * Vec3::X).normalize();
            let seed = (self.next_rand() * 100.0) as f32;
            if self.plumes.len() >= 900 {
                self.plumes.swap_remove(0);
            }
            self.plumes.push(PlumePuff {
                pos: base_f.position + exh_w * 0.4,
                vel: sv + exh_w * 55.0,
                radius: 0.45,
                growth_rate: 7.5,
                stretch: 2.2,
                age: 0.0,
                max_age: 0.22,
                base_opacity: 0.42,
                seed,
                kind: 1.0,
            });
        }

        for p in &mut self.plumes {
            p.pos += p.vel * dt;
            // Ground-pad exhaust steam deflection outward along y = 0!
            if p.pos.y < 1.5 && p.vel.y < 0.0 {
                p.pos.y = 1.5;
                let rad = Vec3::new(p.pos.x, 0.0, p.pos.z).normalize_or_zero();
                let horiz_dir = if rad.length_squared() > 0.1 { rad } else { Vec3::X };
                p.vel = horiz_dir * (p.vel.length() * 0.55) + Vec3::Y * 4.0;
                p.growth_rate *= 1.6;
            }
            p.radius += p.growth_rate * dt;
            p.age += dt;
        }
        self.plumes.retain(|p| p.age < p.max_age);
        Ok(())
    }

    pub fn telemetry(&self) -> Result<StarshipTelemetry> {
        let bp = self.world.scene.body_pose(self.booster)?;
        let bv = self.world.scene.body_velocity(self.booster)?;
        let bmp = self.world.scene.body_mass_properties(self.booster)?;
        let sp = self.world.scene.body_pose(self.ship)?;
        let sv = self.world.scene.body_velocity(self.ship)?;
        let smp = self.world.scene.body_mass_properties(self.ship)?;

        let b_aero = self
            .world
            .devices
            .rocket_aero
            .body_reading(self.booster_aero)
            .unwrap_or_default();
        let s_aero = self
            .world
            .devices
            .rocket_aero
            .body_reading(self.ship_aero)
            .unwrap_or_default();

        let mut b_lit = 0;
        let mut b_thrust = 0.0;
        let mut b_twd = 0.0_f64;
        let mut b_gimbal_deg = 0.0_f64;
        for &e in &self.booster_engines {
            if let Some(r) = self.world.devices.rockets.engine_reading(e) {
                if r.ignited {
                    b_lit += 1;
                    b_gimbal_deg = b_gimbal_deg
                        .max(r.gimbal_rad[0].hypot(r.gimbal_rad[1]).to_degrees());
                }
                b_thrust += r.thrust_n;
                b_twd = b_twd.max(r.twd_torque_nm);
            }
        }
        let mut s_lit = 0;
        let mut s_thrust = 0.0;
        let mut rvac_sep = false;
        let mut s_gimbal_deg = 0.0_f64;
        for (i, &e) in self.ship_engines.iter().enumerate() {
            if let Some(r) = self.world.devices.rockets.engine_reading(e) {
                if r.ignited {
                    s_lit += 1;
                    s_gimbal_deg = s_gimbal_deg
                        .max(r.gimbal_rad[0].hypot(r.gimbal_rad[1]).to_degrees());
                }
                s_thrust += r.thrust_n;
                if i >= 3 && r.flow_separated {
                    rvac_sep = true;
                }
            }
        }
        for &e in &self.ship_hls_thrusters {
            if let Some(r) = self.world.devices.rockets.engine_reading(e) {
                if r.ignited {
                    s_lit += 1;
                    s_gimbal_deg = s_gimbal_deg
                        .max(r.gimbal_rad[0].hypot(r.gimbal_rad[1]).to_degrees());
                }
                s_thrust += r.thrust_n;
            }
        }

        let b_lox = self
            .world
            .devices
            .rockets
            .tank_reading(self.booster_lox)
            .unwrap_or_default();
        let s_hdr = self
            .world
            .devices
            .rockets
            .tank_reading(self.ship_hdr_lox)
            .unwrap_or_default();
        let s_hdr_m = self
            .world
            .devices
            .rockets
            .tank_reading(self.ship_hdr_ch4)
            .map_or(0.0, |r| r.mass_remaining_kg);

        let mut flap_deg = [0.0; 4];
        for i in 0..4 {
            flap_deg[i] = self
                .world
                .devices
                .rocket_aero
                .surface_reading(self.ship_flaps[i])
                .map_or(0.0, |r| r.deflection_rad.to_degrees());
        }

        let b_pin = bp.transform_point(Vec3::new(0.0, BOOSTER_CATCH_PIN_LOCAL_Y, 0.0));
        let b_dist_tower = Vec3::new(b_pin.x, 0.0, b_pin.z).length();
        let s_dist_pad = Vec3::new(sp.position.x - SHIP_PAD_X, 0.0, sp.position.z - SHIP_PAD_Z).length();
        let b_flex = self
            .world
            .devices
            .rocket_structure
            .reading(self.booster_flex)
            .unwrap_or_default();
        let s_flex = self
            .world
            .devices
            .rocket_structure
            .reading(self.ship_flex)
            .unwrap_or_default();
        let _atmo = us76_atmosphere((bp.position.y - BOOSTER_HALF_LEN).max(0.0));

        Ok(StarshipTelemetry {
            time_s: self.time_s,
            scenario: self.scenario,
            booster_phase: self.booster_phase,
            ship_phase: self.ship_phase,
            booster_alt_m: (bp.position.y - BOOSTER_HALF_LEN).max(0.0),
            booster_speed_mps: bv.linear.length(),
            booster_vy_mps: bv.linear.y,
            booster_dist_tower_m: b_dist_tower,
            booster_tilt_deg: (bp.rotation * Vec3::Y).y.clamp(-1.0, 1.0).acos().to_degrees(),
            booster_mach: b_aero.mach,
            booster_q_kpa: b_aero.dynamic_pressure_pa * 1e-3,
            booster_thrust_mn: b_thrust * 1e-6,
            booster_active_engines: b_lit,
            booster_prop_t: self.world.devices.rockets.body_propellant_kg(self.booster) * 1e-3,
            booster_mass_t: bmp.mass * 1e-3,
            booster_com_y_m: bmp.local_com.y,
            booster_slosh_hz: b_lox.slosh_freq_hz,
            booster_twd_knm: b_twd * 1e-3,
            booster_srp_factor: b_aero.srp_drag_factor,
            booster_mode1_hz: b_flex.mode1_hz,
            booster_tip_cm: b_flex.tip_deflection_m * 100.0,
            booster_pogo_g: b_flex.pogo_accel_g,
            booster_ullage_bar: b_lox.ullage_pressure_pa * 1e-5,
            booster_pifs_m: b_aero.pifs_cp_shift_m,
            booster_buffet_knm: b_aero.buffet_torque_nm * 1e-3,
            ship_alt_m: (sp.position.y - SHIP_HALF_LEN).max(0.0),
            ship_speed_mps: sv.linear.length(),
            ship_vy_mps: sv.linear.y,
            ship_dist_pad_m: s_dist_pad,
            ship_tilt_deg: (sp.rotation * Vec3::Y).y.clamp(-1.0, 1.0).acos().to_degrees(),
            ship_aoa_deg: s_aero.aoa_rad.to_degrees(),
            ship_mach: s_aero.mach,
            ship_q_kpa: s_aero.dynamic_pressure_pa * 1e-3,
            ship_thrust_mn: s_thrust * 1e-6,
            ship_active_engines: s_lit,
            ship_prop_t: self.world.devices.rockets.body_propellant_kg(self.ship) * 1e-3,
            ship_header_t: (s_hdr.mass_remaining_kg + s_hdr_m) * 1e-3,
            ship_mass_t: smp.mass * 1e-3,
            ship_com_y_m: smp.local_com.y,
            ship_slosh_hz: s_hdr.slosh_freq_hz,
            ship_mode1_hz: s_flex.mode1_hz,
            ship_tip_cm: s_flex.tip_deflection_m * 100.0,
            ship_pogo_g: s_flex.pogo_accel_g,
            ship_ullage_bar: s_hdr.ullage_pressure_pa * 1e-5,
            ship_rvac_separated: rvac_sep,
            ship_flap_deg: flap_deg,
            interstage_blast_mn: self.world.devices.rockets.last_interstage_blast_n() * 1e-6,
            ground_cushion_mn: self.world.devices.rockets.last_ground_cushion_n() * 1e-6,
            chopsticks_closed_frac: (1.0 - self.chopsticks_cmd_rad / 0.36).clamp(0.0, 1.0),
            wind_preset: self.wind_preset,
            failed_engines: self.failed_engine_count,
            tvc_enabled: self.tvc_enabled,
            booster_gimbal_deg: b_gimbal_deg,
            ship_gimbal_deg: s_gimbal_deg,
        })
    }
}
