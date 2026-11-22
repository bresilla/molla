//! Oriented elastic fiber volumes, progressive bonds, and transactional cuts.

mod bundle;
mod contact;
mod coupled;
mod cpu;
#[cfg(feature = "gpu")]
mod gpu;
mod model;
mod patch;
mod splinter;
mod topology;

pub use bundle::{BundleDesc, FiberBundle};
pub use coupled::CoupledConfig;
pub use cpu::{ConstraintRow, PointGrab, StepConfig, StepReport, bond_rows, step_cpu};
#[cfg(feature = "gpu")]
pub use gpu::{CoupledDiagnostics, GpuStrandLimits, GpuStrands};
pub use model::{Bond, BondMaterial, Fiber, Pin, StrandModel};
pub use patch::BondPatch;
pub use splinter::{SplinterReport, SplinterTransaction};
pub use topology::{CutReport, CutTransaction};

use molla_core::Error;

fn invalid(message: &str) -> Error {
    Error::Solver(format!("strands: {message}"))
}
