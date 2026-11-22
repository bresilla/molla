//! Low-Earth-Orbit (LEO) space environment effectors and spacecraft internal
//! dynamics: faceted atmospheric drag, solar radiation pressure (SRP) with
//! cylindrical Earth eclipse, geomagnetic field & magnetorquers (MTBs),
//! Simons/Dettleff free-molecular thruster plume impingement (`σ_n`, `σ_t`),
//! and microgravity liquid propellant slosh equivalent mechanical models.

use std::collections::BTreeMap;
use std::f64::consts::TAU;

use molla_core::{Error, Result};
use molla_math::{Real, SpatialVector, Vec3};
use molla_sim::joint_control::{JointAxis, JointMotor, MotorModel};
use molla_sim::runtime::{BodyHandle, BodyKind, JointHandle, RigidScene};
use molla_sim::{JointTargetMode, JointType, State};

use super::orbital::EARTH_EQUATORIAL_RADIUS;
use super::{Orbital, OrbitalField, Thrusters};
use crate::featherstone::JointEffort;

/// Nominal 1 AU solar radiation pressure constant (N/m²).
pub const SOLAR_PRESSURE_1AU: Real = 4.56e-6;
/// Earth's magnetic dipole moment magnitude (`~7.94 × 10²² A·m²` -> `μ₀/(4π) * M ≈ 7.94e15 T·m³`).
pub const EARTH_MAGNETIC_MU0_M_OVER_4PI: Real = 7.94e15;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FacetId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MagnetorquerId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SloshTankId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FlexJointId(pub u64);

/// Solar & geomagnetic activity regime (`F10.7` / `Ap`) scaling thermospheric
/// density above ~140 km in [`nrlmsise_density_kg_m3`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SolarActivity {
    /// Solar minimum (`F10.7 ≈ 70 SFU`).
    Low,
    /// Nominal moderate solar cycle (`F10.7 ≈ 150 SFU`).
    #[default]
    Moderate,
    /// Solar maximum (`F10.7 ≈ 220 SFU`).
    High,
}

/// Vallado / CIRA-72 / NRLMSISE-00 piecewise-exponential reference atmosphere
/// layers `(base_alt_km, base_density_kg_m3, scale_height_km)` from 0 to 1,000+ km.
const NRLMSISE_LAYERS: &[(Real, Real, Real)] = &[
    (0.0, 1.225, 7.249),
    (25.0, 3.899e-2, 6.349),
    (30.0, 1.774e-2, 6.682),
    (40.0, 3.972e-3, 7.554),
    (50.0, 1.057e-3, 8.382),
    (60.0, 3.206e-4, 7.714),
    (70.0, 8.770e-5, 6.549),
    (80.0, 1.905e-5, 5.799),
    (90.0, 3.396e-6, 5.382),
    (100.0, 5.297e-7, 5.877),
    (110.0, 9.661e-8, 7.263),
    (120.0, 2.438e-8, 9.473),
    (130.0, 8.484e-9, 12.636),
    (140.0, 3.845e-9, 16.149),
    (150.0, 2.070e-9, 22.523),
    (180.0, 5.464e-10, 29.740),
    (200.0, 2.789e-10, 37.105),
    (250.0, 7.248e-11, 45.546),
    (300.0, 2.418e-11, 53.628),
    (350.0, 9.518e-12, 53.298),
    (400.0, 3.725e-12, 58.515),
    (450.0, 1.585e-12, 60.828),
    (500.0, 6.967e-13, 63.822),
    (600.0, 1.454e-13, 71.835),
    (700.0, 3.614e-14, 88.667),
    (800.0, 1.170e-14, 124.64),
    (900.0, 5.245e-15, 181.05),
    (1000.0, 3.019e-15, 268.00),
];

/// Evaluates Earth atmospheric mass density `ρ(h)` (kg/m³) at geometric altitude
/// `altitude_m` (m) using the 28-layer piecewise-exponential NRLMSISE-00 /
/// Vallado thermosphere table with solar activity modulation.
pub fn nrlmsise_density_kg_m3(altitude_m: Real, activity: SolarActivity) -> Real {
    let h_km = (altitude_m * 1e-3).max(0.0);
    let mut layer = NRLMSISE_LAYERS[0];
    for &entry in NRLMSISE_LAYERS {
        if h_km >= entry.0 {
            layer = entry;
        } else {
            break;
        }
    }
    let (h0, rho0, scale_h) = layer;
    let rho_nom = rho0 * (-(h_km - h0) / scale_h).exp();
    // Smooth thermospheric solar-cycle bulge above 140 km (exosphere temperature variation).
    let ramp = ((h_km - 140.0) / 260.0).clamp(0.0, 1.0);
    let factor = match activity {
        SolarActivity::Low => (1.0 - 0.55 * ramp).max(0.35),
        SolarActivity::Moderate => 1.0,
        SolarActivity::High => 1.0 + 1.40 * ramp,
    };
    rho_nom * factor
}

/// Altitude-varying NRLMSISE-00 piecewise-exponential orbital atmosphere model.
///
/// Automatically couples with [`OrbitalField::CentralGravity`],
/// [`OrbitalField::CentralGravityZonal`], and [`OrbitalField::EccentricRelative`]
/// so eccentric orbits experience realistic perigee-dominant atmospheric drag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NrlmsiseAtmosphere {
    /// Planet equatorial radius `R_e` (m; defaults to [`EARTH_EQUATORIAL_RADIUS`]).
    pub planet_radius_m: Real,
    /// Optional inertial planet centre (for `CentralGravity` / `CentralGravityZonal`).
    /// When `None`, uses `Orbital::chief_radius()` (for `EccentricRelative`) or
    /// `reference_altitude_m` (for circular LVLH).
    pub planet_center: Option<Vec3>,
    /// Nominal geometric altitude (m) used when `planet_center` is `None` and the
    /// orbital field has no eccentric chief radius.
    pub reference_altitude_m: Real,
    /// LVLH frame orbital velocity through the atmosphere (m/s); for
    /// `CentralGravity` set to `Vec3::ZERO` (uses body inertial velocity minus
    /// atmospheric co-rotation).
    pub frame_orbital_velocity: Vec3,
    /// Planet angular rotation vector `ω_planet` (rad/s) for co-rotating
    /// atmosphere in inertial `CentralGravity` frames (`Vec3::ZERO` disables).
    pub planet_angular_velocity: Vec3,
    /// Solar / geomagnetic activity level (`F10.7`).
    pub solar_activity: SolarActivity,
}

impl NrlmsiseAtmosphere {
    /// Earth inertial atmosphere for `CentralGravity` / `CentralGravityZonal`.
    pub fn earth_inertial(planet_center: Vec3, pole_axis: Vec3, activity: SolarActivity) -> Self {
        let omega_earth = 7.292_115e-5; // rad/s sidereal rotation
        Self {
            planet_radius_m: EARTH_EQUATORIAL_RADIUS,
            planet_center: Some(planet_center),
            reference_altitude_m: 400_000.0,
            frame_orbital_velocity: Vec3::ZERO,
            planet_angular_velocity: pole_axis.normalize_or_zero() * omega_earth,
            solar_activity: activity,
        }
    }

    /// Earth eccentric/circular LVLH atmosphere automatically tracking chief altitude.
    pub fn earth_lvlh(reference_altitude_m: Real, activity: SolarActivity) -> Self {
        Self {
            planet_radius_m: EARTH_EQUATORIAL_RADIUS,
            planet_center: None,
            reference_altitude_m,
            frame_orbital_velocity: Vec3::ZERO,
            planet_angular_velocity: Vec3::ZERO,
            solar_activity: activity,
        }
    }
}

/// Implicit structural flexible-appendage hinge spring-damper specification
/// (e.g. 0.1–0.5 Hz solar array bending modes for Control-Structure Interaction).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlexJointSpec {
    /// Articulated revolute or prismatic joint representing the elastic bending hinge.
    pub joint: JointHandle,
    /// Equilibrium / unstressed joint angle (rad) or displacement (m).
    pub rest_angle: Real,
    /// Torsional spring stiffness `k` (N·m/rad) or linear stiffness (N/m).
    pub stiffness: Real,
    /// Modal structural viscous damping `c` (N·m·s/rad) or (N·s/m).
    pub damping: Real,
    /// Maximum elastic restoring torque/force magnitude (N·m or N); defaults to `f64::INFINITY`.
    pub max_torque: Real,
}

impl FlexJointSpec {
    /// Construct a flexible bending hinge directly from an appendage's effective
    /// inertia `I_eff` (kg·m²), target modal frequency `frequency_hz` (Hz), and
    /// structural damping ratio `damping_ratio` `ζ` (typically `0.005..0.04`).
    pub fn from_modal_frequency(
        joint: JointHandle,
        effective_inertia: Real,
        frequency_hz: Real,
        damping_ratio: Real,
    ) -> Self {
        let omega_n = TAU * frequency_hz;
        let stiffness = omega_n * omega_n * effective_inertia;
        let damping = 2.0 * damping_ratio * omega_n * effective_inertia;
        Self {
            joint,
            rest_angle: 0.0,
            stiffness,
            damping,
            max_torque: Real::INFINITY,
        }
    }
}

/// Elastic bending state and stored modal energy of a [`FlexJointSpec`] over the last step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlexJointReading {
    /// Deflection angle `θ - θ₀` (rad) or displacement (m).
    pub deflection: Real,
    /// Deflection rate `θ̇` (rad/s or m/s).
    pub rate: Real,
    /// Restoring elastic + damping torque/force applied by the hinge (N·m or N).
    pub torque: Real,
    /// Stored elastic potential energy `½ k (θ - θ₀)²` (J).
    pub elastic_energy_j: Real,
}

/// A planar aerodynamic / optical / plume-impingement surface facet fixed to a
/// dynamic spacecraft body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceFacetSpec {
    pub body: BodyHandle,
    /// Center of pressure of the facet in the body frame (m).
    pub local_center: Vec3,
    /// Outward unit normal in the body frame (normalized on insertion).
    pub local_normal: Vec3,
    /// Surface area (m²), strictly positive.
    pub area: Real,
    /// Free-molecular aerodynamic drag coefficient (`~2.0..2.2` in LEO).
    pub drag_cd: Real,
    /// Specular optical reflectivity `ρ_s ∈ [0, 1]` (`ρ_s + ρ_d <= 1`).
    pub reflectivity_spec: Real,
    /// Diffuse optical reflectivity `ρ_d ∈ [0, 1]` (`ρ_s + ρ_d <= 1`).
    pub reflectivity_diff: Real,
    /// Dettleff normal momentum accommodation coefficient `σ_n ∈ [0, 1]`.
    pub sigma_n: Real,
    /// Dettleff tangential momentum accommodation coefficient `σ_t ∈ [0, 1]`.
    pub sigma_t: Real,
    /// If `true` (e.g. thin solar array wings), both `+n` and `-n` sides receive
    /// aero, SRP, and plume loads.
    pub double_sided: bool,
}

impl SurfaceFacetSpec {
    /// Typical LEO photovoltaic solar array panel (`double_sided = true`).
    pub fn solar_panel(body: BodyHandle, local_center: Vec3, local_normal: Vec3, area: Real) -> Self {
        Self {
            body,
            local_center,
            local_normal,
            area,
            drag_cd: 2.2,
            reflectivity_spec: 0.25,
            reflectivity_diff: 0.08,
            sigma_n: 0.85,
            sigma_t: 0.85,
            double_sided: true,
        }
    }

    /// Typical white ceramic / Kapton spacecraft bulkhead (`double_sided = false`).
    pub fn hull_panel(body: BodyHandle, local_center: Vec3, local_normal: Vec3, area: Real) -> Self {
        Self {
            body,
            local_center,
            local_normal,
            area,
            drag_cd: 2.1,
            reflectivity_spec: 0.15,
            reflectivity_diff: 0.55,
            sigma_n: 0.90,
            sigma_t: 0.90,
            double_sided: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FacetReading {
    pub aero_force: Vec3,
    pub srp_force: Vec3,
    pub plume_force: Vec3,
    pub in_eclipse: bool,
}

/// Rarefied orbital atmosphere model (free-molecular LEO drag).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbitalAtmosphere {
    /// Ambient atmospheric density `ρ` (kg/m³), nonnegative.
    pub density: Real,
    /// Mean orbital velocity of the LVLH reference frame through the atmosphere (m/s).
    /// Relative gas velocity striking a body is `-(body_vel + frame_orbital_velocity)`.
    pub frame_orbital_velocity: Vec3,
}

impl OrbitalAtmosphere {
    /// Nominal 408 km ISS LEO atmosphere (`ρ ≈ 3.8e-12 kg/m³`, `v_orb = 7,664 m/s` along `+X`).
    pub fn iss_408km() -> Self {
        Self {
            density: 3.8e-12,
            frame_orbital_velocity: Vec3::new(7_664.0, 0.0, 0.0),
        }
    }
}

/// Solar radiation pressure (SRP) environment with optional cylindrical Earth eclipse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolarRadiation {
    /// Solar radiation pressure `P_0` (N/m²; `4.56e-6` at 1 AU).
    pub pressure: Real,
    /// Unit vector pointing from the scene toward the Sun (world frame).
    pub sun_direction: Vec3,
    /// Optional occulting planet `(center_world, radius_m)` for cylindrical umbra eclipse.
    pub eclipse_occluder: Option<(Vec3, Real)>,
}

impl SolarRadiation {
    pub fn one_au(sun_direction: Vec3) -> Self {
        Self {
            pressure: SOLAR_PRESSURE_1AU,
            sun_direction: sun_direction.normalize_or_zero(),
            eclipse_occluder: None,
        }
    }
}

/// Ambient geomagnetic field model (Tesla).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum MagneticField {
    #[default]
    None,
    /// Uniform local field vector `B` (Tesla, world frame) — standard for LVLH proximity ops.
    Uniform(Vec3),
    /// Centered geomagnetic dipole: `B(r) = (μ₀ / 4π r³) (3 (m·r̂) r̂ - m)`,
    /// where `dipole_moment_scaled` is `(μ₀ / 4π) * m_Earth` in `T·m³`.
    Dipole {
        center: Vec3,
        dipole_moment_scaled: Vec3,
    },
}

impl MagneticField {
    /// Representative ISS LEO geomagnetic field (`~32 µT` inclined).
    pub fn iss_leo() -> Self {
        Self::Uniform(Vec3::new(1.4e-5, -2.5e-5, 1.5e-5))
    }

    pub fn field_at(self, world_pos: Vec3) -> Vec3 {
        match self {
            Self::None => Vec3::ZERO,
            Self::Uniform(b) => b,
            Self::Dipole {
                center,
                dipole_moment_scaled,
            } => {
                let r = world_pos - center;
                let r2 = r.length_squared();
                if r2 < 1.0 {
                    return Vec3::ZERO;
                }
                let r_mag = r2.sqrt();
                let r_hat = r / r_mag;
                let inv_r3 = 1.0 / (r2 * r_mag);
                (r_hat * (3.0 * dipole_moment_scaled.dot(r_hat)) - dipole_moment_scaled) * inv_r3
            }
        }
    }
}

/// A 3-axis magnetorquer (MTB) rod cluster or residual magnetic dipole on `body`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MagnetorquerSpec {
    pub body: BodyHandle,
    /// Maximum commandable magnetic dipole magnitude per axis (A·m²).
    pub max_dipole: Vec3,
    /// Fixed residual parasitic magnetic dipole in the body frame (A·m²).
    pub residual_dipole: Vec3,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MagnetorquerReading {
    /// Active total body-frame magnetic dipole (commanded + residual, A·m²).
    pub total_dipole_body: Vec3,
    /// Local geomagnetic field sampled at the body (Tesla, world frame).
    pub field_world: Vec3,
    /// Resulting magnetic torque `m_world × B_world` applied to the body (N·m, world).
    pub world_torque: Vec3,
}

/// Computes the classic B-cross magnetorquer dipole command (A·m², body frame)
/// for propellant-free angular-momentum desaturation (`τ = m × B` opposes `h_excess`).
pub fn desaturation_dipole_body(
    excess_momentum_body: Vec3,
    field_body: Vec3,
    gain: Real,
    max_dipole: Vec3,
) -> Vec3 {
    let b2 = field_body.length_squared();
    if b2 < 1e-18 || !(gain > 0.0) {
        return Vec3::ZERO;
    }
    let raw = (excess_momentum_body.cross(field_body)) * (gain / b2);
    Vec3::new(
        raw.x.clamp(-max_dipole.x.abs(), max_dipole.x.abs()),
        raw.y.clamp(-max_dipole.y.abs(), max_dipole.y.abs()),
        raw.z.clamp(-max_dipole.z.abs(), max_dipole.z.abs()),
    )
}

/// Microgravity liquid propellant slosh equivalent mechanical model (3-DOF
/// capillary spring-damper convective mass coupled to `body` by Newton's 3rd Law).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SloshTankSpec {
    pub body: BodyHandle,
    /// Equilibrium center of the propellant tank in the body frame (m).
    pub tank_center_local: Vec3,
    /// Convective sloshing liquid mass `m_1` (kg), strictly positive.
    pub slosh_mass: Real,
    /// Capillary / surface-tension restoring stiffness `k` (N/m), positive.
    pub stiffness: Real,
    /// Viscous boundary-layer & baffle damping `c` (N·s/m), nonnegative.
    pub damping: Real,
    /// Spherical baffle / tank-wall travel stop radius (m), strictly positive.
    pub max_displacement: Real,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SloshTankReading {
    /// Current slosh mass displacement from `tank_center_local` in the body frame (m).
    pub displacement_local: Vec3,
    /// Current slosh mass relative velocity in the body frame (m/s).
    pub velocity_local: Vec3,
    /// Reaction force exerted by the sloshing liquid on the spacecraft body (N, world).
    pub world_force: Vec3,
    /// Reaction torque exerted about the spacecraft COM (N·m, world).
    pub world_torque: Vec3,
}

struct Facet {
    spec: SurfaceFacetSpec,
    reading: FacetReading,
}

struct Magnetorquer {
    spec: MagnetorquerSpec,
    command_body: Vec3,
    reading: MagnetorquerReading,
}

struct SloshTank {
    spec: SloshTankSpec,
    initialized: bool,
    pos_world: Vec3,
    vel_world: Vec3,
    reading: SloshTankReading,
}

struct FlexJoint {
    spec: FlexJointSpec,
    axis: JointAxis,
    configured: bool,
    reading: FlexJointReading,
}

/// Space environment & spacecraft internal-dynamics device suite.
#[derive(Default)]
pub struct SpaceEnvironment {
    next_facet: u64,
    next_mtb: u64,
    next_slosh: u64,
    next_flex: u64,
    atmosphere: Option<OrbitalAtmosphere>,
    nrlmsise: Option<NrlmsiseAtmosphere>,
    solar_radiation: Option<SolarRadiation>,
    magnetic_field: MagneticField,
    plume_impingement: bool,
    plume_beta: Real,
    facets: BTreeMap<FacetId, Facet>,
    magnetorquers: BTreeMap<MagnetorquerId, Magnetorquer>,
    slosh_tanks: BTreeMap<SloshTankId, SloshTank>,
    flex_joints: BTreeMap<FlexJointId, FlexJoint>,
}

impl SpaceEnvironment {
    pub(super) fn is_empty(&self) -> bool {
        self.facets.is_empty()
            && self.magnetorquers.is_empty()
            && self.slosh_tanks.is_empty()
            && self.flex_joints.is_empty()
            && self.atmosphere.is_none()
            && self.nrlmsise.is_none()
            && self.solar_radiation.is_none()
            && matches!(self.magnetic_field, MagneticField::None)
    }

    pub fn active(&self) -> bool {
        (!self.facets.is_empty()
            && (self.atmosphere.is_some()
                || self.nrlmsise.is_some()
                || self.solar_radiation.is_some()
                || self.plume_impingement))
            || (!self.magnetorquers.is_empty()
                && !matches!(self.magnetic_field, MagneticField::None))
            || !self.slosh_tanks.is_empty()
    }

    /// Active unit Sun direction (world frame) if solar radiation is configured.
    pub fn sun_direction(&self) -> Option<Vec3> {
        self.solar_radiation.map(|s| s.sun_direction)
    }

    /// Configure the altitude-varying NRLMSISE-00 piecewise-exponential atmosphere.
    pub fn set_nrlmsise_atmosphere(&mut self, model: Option<NrlmsiseAtmosphere>) -> Result<()> {
        if let Some(m) = model
            && (!m.planet_radius_m.is_finite()
                || m.planet_radius_m <= 0.0
                || !m.reference_altitude_m.is_finite()
                || !m.frame_orbital_velocity.is_finite()
                || !m.planet_angular_velocity.is_finite()
                || m.planet_center.is_some_and(|c| !c.is_finite()))
        {
            return Err(Error::Build("invalid NRLMSISE atmosphere parameters".into()));
        }
        self.nrlmsise = model;
        Ok(())
    }

    pub fn set_atmosphere(&mut self, atmosphere: Option<OrbitalAtmosphere>) -> Result<()> {
        if let Some(a) = atmosphere
            && (!a.density.is_finite()
                || a.density < 0.0
                || !a.frame_orbital_velocity.is_finite())
        {
            return Err(Error::Build("invalid orbital atmosphere parameters".into()));
        }
        self.atmosphere = atmosphere;
        Ok(())
    }

    pub fn set_solar_radiation(&mut self, srp: Option<SolarRadiation>) -> Result<()> {
        if let Some(mut s) = srp {
            if !s.pressure.is_finite()
                || s.pressure < 0.0
                || !s.sun_direction.is_finite()
                || s.sun_direction.length_squared() < 1e-18
            {
                return Err(Error::Build("invalid solar radiation parameters".into()));
            }
            s.sun_direction = s.sun_direction.normalize();
            if let Some((c, r)) = s.eclipse_occluder
                && (!c.is_finite() || !r.is_finite() || r <= 0.0)
            {
                return Err(Error::Build("invalid eclipse occluder".into()));
            }
            self.solar_radiation = Some(s);
        } else {
            self.solar_radiation = None;
        }
        Ok(())
    }

    pub fn set_magnetic_field(&mut self, field: MagneticField) -> Result<()> {
        match field {
            MagneticField::None => {}
            MagneticField::Uniform(b) if !b.is_finite() => {
                return Err(Error::Build("magnetic field must be finite".into()));
            }
            MagneticField::Dipole {
                center,
                dipole_moment_scaled,
            } if !center.is_finite() || !dipole_moment_scaled.is_finite() => {
                return Err(Error::Build("magnetic dipole must be finite".into()));
            }
            _ => {}
        }
        self.magnetic_field = field;
        Ok(())
    }

    /// Enables or disables automatic Simons/Dettleff free-molecular plume
    /// impingement from all active [`Thrusters`] onto all registered [`SurfaceFacetSpec`]s.
    pub fn set_plume_impingement(&mut self, enabled: bool, simons_beta: Real) -> Result<()> {
        if !simons_beta.is_finite() || simons_beta < 1.0 {
            return Err(Error::Build(
                "Simons plume exponent beta must be finite and >= 1.0".into(),
            ));
        }
        self.plume_impingement = enabled;
        self.plume_beta = simons_beta;
        Ok(())
    }

    pub fn insert_facet(
        &mut self,
        scene: &RigidScene,
        mut spec: SurfaceFacetSpec,
    ) -> Result<FacetId> {
        if scene.body(spec.body).is_none() {
            return Err(Error::Build("surface facet body is missing".into()));
        }
        if !spec.local_center.is_finite()
            || !spec.local_normal.is_finite()
            || spec.local_normal.length_squared() < 1e-18
            || !spec.area.is_finite()
            || spec.area <= 0.0
            || !spec.drag_cd.is_finite()
            || spec.drag_cd < 0.0
            || !spec.reflectivity_spec.is_finite()
            || !spec.reflectivity_diff.is_finite()
            || spec.reflectivity_spec < 0.0
            || spec.reflectivity_diff < 0.0
            || spec.reflectivity_spec + spec.reflectivity_diff > 1.0 + 1e-9
            || !spec.sigma_n.is_finite()
            || !spec.sigma_t.is_finite()
            || !(0.0..=1.0).contains(&spec.sigma_n)
            || !(0.0..=1.0).contains(&spec.sigma_t)
        {
            return Err(Error::Build("invalid surface facet specification".into()));
        }
        spec.local_normal = spec.local_normal.normalize();
        self.next_facet += 1;
        let id = FacetId(self.next_facet);
        self.facets.insert(
            id,
            Facet {
                spec,
                reading: FacetReading::default(),
            },
        );
        Ok(id)
    }

    pub fn insert_magnetorquer(
        &mut self,
        scene: &RigidScene,
        spec: MagnetorquerSpec,
    ) -> Result<MagnetorquerId> {
        if scene.body(spec.body).is_none() {
            return Err(Error::Build("magnetorquer body is missing".into()));
        }
        if !spec.max_dipole.is_finite()
            || spec.max_dipole.min_element() < 0.0
            || !spec.residual_dipole.is_finite()
        {
            return Err(Error::Build("invalid magnetorquer specification".into()));
        }
        self.next_mtb += 1;
        let id = MagnetorquerId(self.next_mtb);
        self.magnetorquers.insert(
            id,
            Magnetorquer {
                spec,
                command_body: Vec3::ZERO,
                reading: MagnetorquerReading::default(),
            },
        );
        Ok(id)
    }

    pub fn set_magnetorquer_dipole(
        &mut self,
        id: MagnetorquerId,
        dipole_body: Vec3,
    ) -> Result<()> {
        if !dipole_body.is_finite() {
            return Err(Error::Build("magnetorquer dipole must be finite".into()));
        }
        let mtb = self
            .magnetorquers
            .get_mut(&id)
            .ok_or_else(|| Error::Build("no such magnetorquer".into()))?;
        let lim = mtb.spec.max_dipole;
        mtb.command_body = Vec3::new(
            dipole_body.x.clamp(-lim.x, lim.x),
            dipole_body.y.clamp(-lim.y, lim.y),
            dipole_body.z.clamp(-lim.z, lim.z),
        );
        Ok(())
    }

    pub fn insert_slosh_tank(
        &mut self,
        scene: &RigidScene,
        spec: SloshTankSpec,
    ) -> Result<SloshTankId> {
        if scene.body(spec.body).is_none() {
            return Err(Error::Build("slosh tank body is missing".into()));
        }
        if !spec.tank_center_local.is_finite()
            || !spec.slosh_mass.is_finite()
            || spec.slosh_mass <= 0.0
            || !spec.stiffness.is_finite()
            || spec.stiffness <= 0.0
            || !spec.damping.is_finite()
            || spec.damping < 0.0
            || !spec.max_displacement.is_finite()
            || spec.max_displacement <= 0.0
        {
            return Err(Error::Build("invalid slosh tank specification".into()));
        }
        let pose = scene.body_pose(spec.body)?;
        let vel = scene.body_velocity(spec.body)?;
        let mass_props = scene.body_mass_properties(spec.body)?;
        let com_world = pose.transform_point(mass_props.local_com);
        let pos_world = pose.transform_point(spec.tank_center_local);
        let vel_world = vel.linear + vel.angular.cross(pos_world - com_world);
        self.next_slosh += 1;
        let id = SloshTankId(self.next_slosh);
        self.slosh_tanks.insert(
            id,
            SloshTank {
                spec,
                initialized: true,
                pos_world,
                vel_world,
                reading: SloshTankReading::default(),
            },
        );
        Ok(id)
    }

    pub fn facet_reading(&self, id: FacetId) -> Option<FacetReading> {
        self.facets.get(&id).map(|f| f.reading)
    }

    pub fn magnetorquer_reading(&self, id: MagnetorquerId) -> Option<MagnetorquerReading> {
        self.magnetorquers.get(&id).map(|m| m.reading)
    }

    pub fn slosh_reading(&self, id: SloshTankId) -> Option<SloshTankReading> {
        self.slosh_tanks.get(&id).map(|s| s.reading)
    }

    /// Insert an implicit structural flexible-appendage bending joint
    /// (e.g. hinged solar array segment for 0.1–0.5 Hz CSI modes).
    pub fn insert_flex_joint(
        &mut self,
        scene: &RigidScene,
        spec: FlexJointSpec,
    ) -> Result<FlexJointId> {
        let desc = scene
            .joint(spec.joint)
            .ok_or_else(|| Error::Build("flex joint is missing".into()))?;
        let axis = match desc.kind {
            JointType::Revolute => JointAxis::AngX,
            JointType::Prismatic => JointAxis::LinX,
            _ => {
                return Err(Error::Build(
                    "flex joint requires a revolute or prismatic tree joint".into(),
                ));
            }
        };
        if !spec.rest_angle.is_finite()
            || !spec.stiffness.is_finite()
            || spec.stiffness < 0.0
            || !spec.damping.is_finite()
            || spec.damping < 0.0
            || spec.max_torque.is_nan()
            || spec.max_torque < 0.0
        {
            return Err(Error::Build("invalid flex joint specification".into()));
        }
        self.next_flex += 1;
        let id = FlexJointId(self.next_flex);
        self.flex_joints.insert(
            id,
            FlexJoint {
                spec,
                axis,
                configured: false,
                reading: FlexJointReading::default(),
            },
        );
        Ok(id)
    }

    pub fn flex_joint_reading(&self, id: FlexJointId) -> Option<FlexJointReading> {
        self.flex_joints.get(&id).map(|f| f.reading)
    }

    /// Total stored elastic strain energy across all flexible appendage joints (J).
    pub fn total_flex_elastic_energy_j(&self) -> Real {
        self.flex_joints
            .values()
            .map(|f| f.reading.elastic_energy_j)
            .sum()
    }

    /// World-frame linear momentum `m_slosh * v_slosh` of an internal slosh mass
    /// (for exact system momentum conservation accounting).
    pub fn slosh_momentum_world(&self, id: SloshTankId) -> Option<Vec3> {
        self.slosh_tanks
            .get(&id)
            .map(|s| s.vel_world * s.spec.slosh_mass)
    }

    /// Total averaged plume impingement force magnitude across all facets on `body`.
    pub fn body_plume_force(&self, body: BodyHandle) -> Vec3 {
        let mut total = Vec3::ZERO;
        for f in self.facets.values() {
            if f.spec.body == body {
                total += f.reading.plume_force;
            }
        }
        total
    }

    pub fn body_aero_force(&self, body: BodyHandle) -> Vec3 {
        let mut total = Vec3::ZERO;
        for f in self.facets.values() {
            if f.spec.body == body {
                total += f.reading.aero_force;
            }
        }
        total
    }

    pub fn body_srp_force(&self, body: BodyHandle) -> Vec3 {
        let mut total = Vec3::ZERO;
        for f in self.facets.values() {
            if f.spec.body == body {
                total += f.reading.srp_force;
            }
        }
        total
    }

    pub(crate) fn before_step(&mut self, scene: &mut RigidScene) -> Result<()> {
        for f in self.facets.values_mut() {
            f.reading = FacetReading::default();
        }
        for m in self.magnetorquers.values_mut() {
            m.reading.world_torque = Vec3::ZERO;
        }
        for s in self.slosh_tanks.values_mut() {
            s.reading.world_force = Vec3::ZERO;
            s.reading.world_torque = Vec3::ZERO;
        }
        self.flex_joints
            .retain(|_, fj| scene.joint(fj.spec.joint).is_some());
        for fj in self.flex_joints.values_mut() {
            if !fj.configured {
                let drive = JointMotor {
                    mode: JointTargetMode::Position,
                    target_position: fj.spec.rest_angle,
                    target_velocity: 0.0,
                    stiffness: fj.spec.stiffness,
                    damping: fj.spec.damping,
                    max_force: fj.spec.max_torque,
                    model: MotorModel::Force,
                };
                scene.set_joint_motor_with_wake(fj.spec.joint, fj.axis, drive, false)?;
                fj.configured = true;
            }
        }
        Ok(())
    }

    pub(crate) fn after_step(&mut self, scene: &RigidScene, efforts: &[JointEffort]) -> Result<()> {
        if self.flex_joints.is_empty() {
            return Ok(());
        }
        let model = scene.model();
        let coord_offs = model.joint_coord_offset.host()?;
        let dof_offs = model.joint_dof_offset.host()?;
        let state = scene.state();
        let joint_q = state.joint_q.host()?;
        let joint_qd = state.joint_qd.host()?;
        for fj in self.flex_joints.values_mut() {
            let Some(ji) = scene.joint_index(fj.spec.joint) else {
                continue;
            };
            let coord = coord_offs[ji.index()] as usize;
            let dof = dof_offs[ji.index()] as usize;
            let q = joint_q[coord];
            let qd = joint_qd[dof];
            let defl = q - fj.spec.rest_angle;
            let effort = efforts.get(dof).copied().unwrap_or_default().drive;
            fj.reading = FlexJointReading {
                deflection: defl,
                rate: qd,
                torque: effort,
                elastic_energy_j: 0.5 * fj.spec.stiffness * defl * defl,
            };
        }
        Ok(())
    }

    pub(crate) fn apply(
        &mut self,
        scene: &RigidScene,
        state: &State,
        thrusters: &Thrusters,
        orbital: &Orbital,
        wrenches: &mut [SpatialVector],
        sub_dt: Real,
        weight: Real,
    ) -> Result<()> {
        if !self.active() {
            return Ok(());
        }
        let poses = state.body_q.host()?;
        let velocities = state.body_qd.host()?;
        let coms = scene.model().body_com.host()?;

        // Snapshot active thrusters for Simons/Dettleff plume impingement if enabled.
        let mut plume_sources: Vec<(BodyHandle, Vec3, Vec3, Real)> = Vec::new();
        if self.plume_impingement && !self.facets.is_empty() {
            for (body, frame, thrust) in thrusters.active_nozzles() {
                let Some(bi) = scene.body_index(body) else {
                    continue;
                };
                let bp = poses[bi.index()];
                let noz_w = bp.transform_point(frame.position);
                let exh_w = -(bp.rotation * frame.rotation * Vec3::X);
                plume_sources.push((body, noz_w, exh_w, thrust));
            }
        }

        // 1. Faceted LEO Aero Drag, SRP (+ Eclipse), and Simons/Dettleff Plume Impingement.
        let beta = if self.plume_beta >= 1.0 {
            self.plume_beta
        } else {
            4.0
        };
        let simons_norm = (beta + 2.0) / (2.0 * TAU);

        for facet in self.facets.values_mut() {
            let (Some(bi), Some(desc)) = (
                scene.body_index(facet.spec.body),
                scene.body(facet.spec.body),
            ) else {
                continue;
            };
            if !desc.enabled || desc.kind != BodyKind::Dynamic {
                continue;
            }
            let i = bi.index();
            let bp = poses[i];
            let bv = velocities[i];
            let com_w = bp.transform_point(coms[i]);
            let pos_w = bp.transform_point(facet.spec.local_center);
            let n_w = (bp.rotation * facet.spec.local_normal).normalize();
            let arm = pos_w - com_w;

            // 1a. Rarefied LEO Atmospheric Drag (NRLMSISE-00 exponential table or constant ρ).
            let mut f_aero = Vec3::ZERO;
            let aero_sample: Option<(Real, Vec3)> = if let Some(nrl) = self.nrlmsise {
                let (alt_m, v_rel) = if let Some(center) = nrl.planet_center {
                    let r_vec = pos_w - center;
                    let alt = (r_vec.length() - nrl.planet_radius_m).max(0.0);
                    let v_atmo = nrl.planet_angular_velocity.cross(r_vec);
                    (alt, bv.linear + bv.angular.cross(arm) - v_atmo)
                } else {
                    let (alt, frame_v) = match *orbital.field() {
                        OrbitalField::EccentricRelative {
                            radial_axis,
                            along_track_axis,
                            origin,
                            ..
                        } => {
                            let r_chief = orbital.chief_radius().unwrap_or(
                                nrl.planet_radius_m + nrl.reference_altitude_m,
                            );
                            let spd_chief = orbital.chief_speed().unwrap_or(0.0);
                            let dr = (pos_w - origin).dot(radial_axis);
                            let fv = if nrl.frame_orbital_velocity.length_squared() > 0.0 {
                                nrl.frame_orbital_velocity
                            } else {
                                along_track_axis * spd_chief
                            };
                            ((r_chief - nrl.planet_radius_m + dr).max(0.0), fv)
                        }
                        OrbitalField::ClohessyWiltshire {
                            radial_axis,
                            origin,
                            ..
                        } => {
                            let dr = (pos_w - origin).dot(radial_axis);
                            (
                                (nrl.reference_altitude_m + dr).max(0.0),
                                nrl.frame_orbital_velocity,
                            )
                        }
                        _ => (nrl.reference_altitude_m.max(0.0), nrl.frame_orbital_velocity),
                    };
                    (alt, bv.linear + bv.angular.cross(arm) + frame_v)
                };
                Some((nrlmsise_density_kg_m3(alt_m, nrl.solar_activity), v_rel))
            } else if let Some(atmo) = self.atmosphere
                && atmo.density > 0.0
            {
                Some((
                    atmo.density,
                    bv.linear + bv.angular.cross(arm) + atmo.frame_orbital_velocity,
                ))
            } else {
                None
            };
            if let Some((density, v_body)) = aero_sample
                && density > 0.0
            {
                let speed2 = v_body.length_squared();
                if speed2 > 1e-12 {
                    let speed = speed2.sqrt();
                    let v_hat = v_body / speed;
                    let proj = if facet.spec.double_sided {
                        n_w.dot(v_hat).abs()
                    } else {
                        n_w.dot(v_hat).max(0.0)
                    };
                    if proj > 0.0 {
                        f_aero = -v_hat
                            * (0.5 * density * facet.spec.drag_cd * facet.spec.area * proj * speed2);
                    }
                }
            }

            // 1b. Solar Radiation Pressure (exact specular + diffuse + cylindrical umbra eclipse).
            let mut f_srp = Vec3::ZERO;
            let mut eclipsed = false;
            if let Some(srp) = self.solar_radiation
                && srp.pressure > 0.0
            {
                if let Some((occ_c, occ_r)) = srp.eclipse_occluder {
                    let rel = pos_w - occ_c;
                    let along = rel.dot(srp.sun_direction);
                    if along < 0.0 {
                        let perp2 = (rel - srp.sun_direction * along).length_squared();
                        if perp2 < occ_r * occ_r {
                            eclipsed = true;
                        }
                    }
                }
                if !eclipsed {
                    let dot_sn = n_w.dot(srp.sun_direction);
                    let (cos_th, active_n) = if facet.spec.double_sided {
                        if dot_sn >= 0.0 {
                            (dot_sn, n_w)
                        } else {
                            (-dot_sn, -n_w)
                        }
                    } else {
                        (dot_sn.max(0.0), n_w)
                    };
                    if cos_th > 0.0 {
                        let rs = facet.spec.reflectivity_spec;
                        let rd = facet.spec.reflectivity_diff;
                        let coeff = srp.pressure * facet.spec.area * cos_th;
                        f_srp = -(srp.sun_direction * (1.0 - rs)
                            + active_n * (2.0 * (rs * cos_th + rd / 3.0)))
                            * coeff;
                    }
                }
            }

            // 1c. Simons + Dettleff Free-Molecular Plume Impingement (`σ_n`, `σ_t`).
            let mut f_plume = Vec3::ZERO;
            if !plume_sources.is_empty() {
                for &(src_body, noz_w, exh_w, thrust) in &plume_sources {
                    if src_body == facet.spec.body {
                        continue;
                    }
                    let to_f = pos_w - noz_w;
                    let dist2 = to_f.length_squared();
                    if dist2 < 0.0625 || dist2 > 100.0 {
                        continue;
                    }
                    let dist = dist2.sqrt();
                    let u_inc = to_f / dist; // Unit incident gas stream direction
                    let cos_plume = exh_w.dot(u_inc);
                    if cos_plume <= 0.15 {
                        continue;
                    }
                    let dot_un = -u_inc.dot(n_w);
                    let (cos_alpha, active_n) = if facet.spec.double_sided {
                        if dot_un >= 0.0 {
                            (dot_un, n_w)
                        } else {
                            (-dot_un, -n_w)
                        }
                    } else {
                        (dot_un.max(0.0), n_w)
                    };
                    if cos_alpha <= 1e-4 {
                        continue;
                    }
                    // Simons dynamic momentum flux `q = (1/2) ρ u²` at `(r, θ)` (Pa).
                    let q = thrust * simons_norm * cos_plume.powf(beta) / dist2;
                    let sn = facet.spec.sigma_n;
                    let st = facet.spec.sigma_t;
                    // Dettleff hyperthermal free-molecular normal pressure & tangential shear:
                    //   p_n = q * [ 2(2 - σ_n) cos²α + σ_n * 0.28 * cosα ] along `-active_n`
                    //   p_t = q * 2 σ_t cosα * sinα along surface-projected flow tangent `t_hat`.
                    let p_n = q * (2.0 * (2.0 - sn) * cos_alpha * cos_alpha + sn * 0.28 * cos_alpha);
                    let tang = u_inc + active_n * cos_alpha;
                    let sin_alpha = tang.length();
                    let f_n = -active_n * (p_n * facet.spec.area);
                    let f_t = if sin_alpha > 1e-6 {
                        (tang / sin_alpha) * (q * 2.0 * st * cos_alpha * sin_alpha * facet.spec.area)
                    } else {
                        Vec3::ZERO
                    };
                    let f_local = f_n + f_t;
                    // Energy-safe cap: single facet cannot intercept more than 2× thruster force.
                    let mag = f_local.length();
                    let cap = 2.0 * thrust;
                    f_plume += if mag > cap { f_local * (cap / mag) } else { f_local };
                }
            }

            let f_total = f_aero + f_srp + f_plume;
            if f_total.length_squared() > 0.0 {
                wrenches[i].linear += f_total;
                wrenches[i].angular += arm.cross(f_total);
            }
            facet.reading.aero_force += f_aero * weight;
            facet.reading.srp_force += f_srp * weight;
            facet.reading.plume_force += f_plume * weight;
            facet.reading.in_eclipse = eclipsed;
        }

        // 2. Geomagnetic Field + Magnetorquers & Residual Dipoles (`τ = m × B`).
        if !matches!(self.magnetic_field, MagneticField::None) {
            for mtb in self.magnetorquers.values_mut() {
                let (Some(bi), Some(desc)) = (
                    scene.body_index(mtb.spec.body),
                    scene.body(mtb.spec.body),
                ) else {
                    continue;
                };
                if !desc.enabled || desc.kind != BodyKind::Dynamic {
                    continue;
                }
                let i = bi.index();
                let bp = poses[i];
                let com_w = bp.transform_point(coms[i]);
                let b_world = self.magnetic_field.field_at(com_w);
                let m_body = mtb.command_body + mtb.spec.residual_dipole;
                let m_world = bp.rotation * m_body;
                let torque = m_world.cross(b_world);
                wrenches[i].angular += torque;
                mtb.reading.total_dipole_body = m_body;
                mtb.reading.field_world = b_world;
                mtb.reading.world_torque += torque * weight;
            }
        }

        // 3. Microgravity Propellant Slosh EMM (3-DOF capillary spring-damper mass).
        if sub_dt > 0.0 {
            for tank in self.slosh_tanks.values_mut() {
                let (Some(bi), Some(desc)) = (
                    scene.body_index(tank.spec.body),
                    scene.body(tank.spec.body),
                ) else {
                    continue;
                };
                if !desc.enabled || desc.kind != BodyKind::Dynamic {
                    continue;
                }
                let i = bi.index();
                let bp = poses[i];
                let bv = velocities[i];
                let com_w = bp.transform_point(coms[i]);
                let eq_w = bp.transform_point(tank.spec.tank_center_local);
                let arm = eq_w - com_w;
                let v_eq_w = bv.linear + bv.angular.cross(arm);

                if !tank.initialized {
                    tank.pos_world = eq_w;
                    tank.vel_world = v_eq_w;
                    tank.initialized = true;
                }
                let disp_w = tank.pos_world - eq_w;
                let rel_v_w = tank.vel_world - v_eq_w;

                // Linear capillary restoring spring + viscous baffle damper + wall stop penalty.
                let mut f_on_slosh = -disp_w * tank.spec.stiffness - rel_v_w * tank.spec.damping;
                let r = disp_w.length();
                if r > tank.spec.max_displacement {
                    let n = disp_w / r;
                    let over = r - tank.spec.max_displacement;
                    let k_wall = tank.spec.stiffness * 25.0;
                    let c_wall = tank.spec.damping * 5.0;
                    let vn = rel_v_w.dot(n);
                    f_on_slosh -= n * (k_wall * over + c_wall * vn.max(0.0));
                }

                // Symplectic Euler advance for the internal convective slosh mass.
                tank.vel_world += (f_on_slosh / tank.spec.slosh_mass) * sub_dt;
                tank.pos_world += tank.vel_world * sub_dt;

                // Exact Newton's 3rd Law reaction force & lever-arm torque on the spacecraft!
                let f_on_body = -f_on_slosh;
                let tau_on_body = arm.cross(f_on_body);
                wrenches[i].linear += f_on_body;
                wrenches[i].angular += tau_on_body;

                let rot_inv = bp.rotation.conjugate();
                tank.reading.displacement_local = rot_inv * (tank.pos_world - eq_w);
                tank.reading.velocity_local = rot_inv * (tank.vel_world - v_eq_w);
                tank.reading.world_force += f_on_body * weight;
                tank.reading.world_torque += tau_on_body * weight;
            }
        }

        Ok(())
    }
}
