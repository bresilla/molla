//! molla-vehicle — wheeled-vehicle simulation for molla.
//!
//! Built per `PLAN_ENGINE.md` Track V. The vehicle is assembled from
//! parts molla already provides (rigid bodies, joints) plus the
//! vehicle-specific physics in this crate. Modules are added stage by
//! stage:
//!
//! - **V0** — engine prerequisites: a [`Terrain`] query abstraction and
//!   a 1-DOF rotational [`Shaft`] (the driveline building block).
//! - **V1/V2** — `tire` force-element models.
//! - **V3** — `suspension` templates.
//! - **V4** — `powertrain` (engine map, transmission, driveline).

pub mod deformable;
pub mod granular;
pub mod powertrain;
pub mod real_tire;
pub mod shaft;
pub mod suspension;
pub mod sysid;
pub mod terrain;
pub mod tire;
pub mod vehicle;

pub use deformable::DeformableTerrain;
pub use granular::{GrainObstacle, GranularBed};
pub use powertrain::{Driveline, DriverInput, Engine, Powertrain, Transmission, ackermann_steer};
pub use real_tire::{
    DynamicTireCarcass, DynamicTireCarcassParams, RealTireContact, RealTireParams, RealWheelState,
    SurfaceMaterial, TetraTireFem, TetraTireFemGpuSnapshot, TetraTireFemParams, TireContactCell,
    evaluate_brush_tire, solve_loaded_tire_hub_height,
};
pub use shaft::Shaft;
pub use suspension::{Suspension, SuspensionKind};
pub use sysid::{drive_distance, identify_friction};
pub use terrain::{FlatTerrain, HeightfieldTerrain, Terrain};
pub use tire::{
    Pac89Tire, RigidTire, TireModel, TireOutput, TmSimpleTire, WheelContact, magic_formula,
    magic_formula_deriv, wheel_ground_wrench,
};
pub use vehicle::{Corner, Vehicle};
