//! Per-feature report families extracted out of `real_tire/mod.rs`.
//!
//! Each child module groups a coherent block of report types and their
//! deterministic builder functions. Moved as part of PLAN_REALTIRE Stream A.

pub mod convergence;
pub mod physical_shape;
pub mod soil;
pub mod thermal;
pub mod triangle;
pub mod vehicle;
