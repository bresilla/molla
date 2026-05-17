//! `molla` — pure-Rust, wgpu-accelerated physics engine.
//!
//! A portable alternative to NVIDIA Newton, built on `wgpu` so it runs on
//! AMD, Intel, Apple Silicon, and (eventually) the web.
//!
//! The crate is structured around four pillars borrowed from Newton:
//! - [`sim::Model`]    — static, immutable simulation description.
//! - [`sim::State`]    — time-varying positions, velocities, forces.
//! - [`sim::Control`]  — user-supplied joint targets and applied wrenches.
//! - [`sim::Contacts`] — collision pipeline output, consumed by solvers.
//!
//! Host-side math is `f64` (see [`math`]). GPU kernels run in `f32` because
//! WGSL has no `f64`. Conversions happen exactly once, at the buffer
//! boundary in [`storage::Buffer`].

#![cfg_attr(docsrs, feature(doc_auto_cfg))]

pub mod actuators;
pub mod autograd;
pub mod compute;
pub use molla_core as core;
pub use molla_sim as sim;
pub mod fem;
pub mod geometry;
pub mod kernels;
pub mod math;
pub mod optim;
pub mod sensors;
pub mod solvers;
pub mod storage;
pub mod utils;

pub mod prelude;
