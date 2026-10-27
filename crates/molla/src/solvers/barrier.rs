//! Barrier-Newton contact solver (roadmap C0) — penetration-free
//! deformables with strong (ABD) rigid coupling. See
//! `book/src/solvers/barrier_newton.md`.
pub use molla_solvers::BarrierSolver;
pub use molla_solvers::barrier_contact::{
    AffineBody, BarrierObstacle, BarrierParams, BarrierShell, BarrierSpring, BarrierStepStats,
    BarrierTet,
};
