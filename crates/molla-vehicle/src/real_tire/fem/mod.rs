//! `TetraTireFem` volumetric tetrahedral FEM tire model.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

pub mod builders;
pub mod diagnostics;
pub mod math;
pub mod newton_builders;
pub mod newton_reports;
pub mod params;
pub mod sidewall;
pub mod solver;
pub mod state;
