//! Coherent aerospace, orbital, and launch-vehicle physics facade for `molla`
//! (paralleling [`crate::vehicle`] for ground vehicles).
//!
//! Organizes all real-time spaceflight & rocket physics devices exported by
//! `molla-solvers` into clean domain submodules for downstream integrators
//! (such as `gearbox` machines, RL environments, and HIL testbeds):
//! - [`orbital`] — central gravity, $J_2\text{–}J_4$ zonal harmonics, Clohessy–Wiltshire,
//!   and Yamanaka–Ankersen eccentric relative motion.
//! - [`propulsion`] — liquid rocket engines (`RocketPropulsion`, nozzle gas dynamics,
//!   Schmucker flow separation, 2-axis TVC + Tail-Wags-Dog, shifting-COM tanks,
//!   NASA SP-8009 slosh, pneumatic ullage/NPSH cavitation, hot-staging blast, Coriolis),
//!   RCS `Thrusters` (valve lag, MIB, PWM/PWPF), and `ReactionWheels` / SGCMGs.
//! - [`aero`] — 0°–180° Jorgensen/Hoerner launch aerodynamics, SRP drag loss, PIFS,
//!   transonic Max-Q buffet, MIL-F-8785C Dryden turbulence, Knudsen rarefied bridging,
//!   hinged body flaps, and titanium grid fins.
//! - [`structure`] — Euler–Bernoulli free-free fuselage bending (CSI), NASA SP-8055
//!   longitudinal Pogo coupling, adaptive `BiquadNotchFilter`, and flexible appendages.
//! - [`environment`] — NRLMSISE-00 atmosphere, SRP, gravity-gradient, magnetorquers,
//!   and faceted drag / Dettleff plume impingement.
//! - [`sensors`] — hardware-grade IMU (with automatic elastic bending feedthrough),
//!   star tracker, rendezvous LiDAR, and latency-compensated `NavigationFilter`.
//! - [`gnc`] — 6-DOF spacecraft attitude & rendezvous controllers.

pub mod orbital {
    pub use molla_solvers::devices::{
        EARTH_EQUATORIAL_RADIUS, EARTH_J2, EARTH_J3, EARTH_J4, EARTH_MU, EccentricOrbit,
        J2Perturbation, MARS_MU, MOON_MU, Orbital, OrbitalField, OrbitalReading, YamanakaAnkersen,
        ZonalHarmonics,
    };
}

pub mod propulsion {
    pub use molla_solvers::devices::{
        AtmosphereState, EARTH_OMEGA_RAD_S, GimbalSpec, ImbalanceSpec, MARS_GRAVITY,
        MARS_OMEGA_RAD_S, MOON_GRAVITY, PlanetAtmosphere, PropellantTankReading,
        PropellantTankSpec, ReactionWheelCommand, ReactionWheelId, ReactionWheelReading,
        ReactionWheelSpec, ReactionWheels, RocketEngineCommand, RocketEngineId,
        RocketEngineReading, RocketEngineSpec, RocketPropulsion, TankId, ThrusterId,
        ThrusterModulation, ThrusterReading, ThrusterSpec, Thrusters, earth_rotation_at_latitude,
        mars_atmosphere, planet_rotation_at_latitude, us76_atmosphere,
    };
}

pub mod aero {
    pub use molla_solvers::devices::{
        AeroBodyId, AeroBodyReading, AeroBodySpec, AeroSurfaceId, AeroSurfaceKind,
        AeroSurfaceReading, AeroSurfaceSpec, RocketAero, WindField,
    };
}

pub mod structure {
    pub use molla_solvers::devices::{
        BiquadNotchFilter, FlexBodyId, FlexBodyReading, FlexBodySpec, FlexJointId,
        FlexJointReading, FlexJointSpec, RocketStructure, SloshTankId, SloshTankReading,
        SloshTankSpec, free_free_beam_mode,
    };
}

pub mod environment {
    pub use molla_solvers::devices::{
        EARTH_MAGNETIC_MU0_M_OVER_4PI, FacetId, FacetReading, MagneticField, MagnetorquerId,
        MagnetorquerReading, MagnetorquerSpec, NrlmsiseAtmosphere, OrbitalAtmosphere,
        SOLAR_PRESSURE_1AU, SolarActivity, SolarRadiation, SpaceEnvironment, SurfaceFacetSpec,
        desaturation_dipole_body, nrlmsise_density_kg_m3,
    };
}

pub mod sensors {
    pub use molla_solvers::devices::{
        ImuId, ImuReading, ImuSpec, LidarId, LidarReading, LidarSpec, NavigationFilter,
        SpaceSensors, StarTrackerId, StarTrackerReading, StarTrackerSpec,
    };
}

pub mod gnc {
    pub use molla_solvers::devices::{
        SpacecraftCommand, SpacecraftControllers, SpacecraftId, SpacecraftLimits,
        SpacecraftReading, SpacecraftSpec,
    };
}

pub use aero::*;
pub use environment::*;
pub use gnc::*;
pub use orbital::*;
pub use propulsion::*;
pub use sensors::*;
pub use structure::*;
