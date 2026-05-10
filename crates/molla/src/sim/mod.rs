//! Core simulation types: [`Model`], [`State`], [`Control`], [`Contacts`].
//!
//! These four types form the contract every solver consumes:
//! [`Model`] is built once via [`ModelBuilder`] and then immutable;
//! [`State`] holds the time-varying body / joint / particle data
//! that solvers ping-pong between substeps; [`Control`] carries
//! user-supplied joint targets and external wrenches; [`Contacts`]
//! is the collision pipeline's output buffer.
//!
//! Forward kinematics on a [`Model`] + [`State`] is exposed as
//! [`eval_fk`]. Joint topology and per-DOF flags live in [`flags`]
//! and [`JointType`] / [`JointTargetMode`].

mod articulation;
mod builder;
mod contacts;
mod control;
pub mod flags;
mod joint;
mod model;
mod state;

pub use articulation::eval_fk;
pub use builder::{BodyParams, MaterialNeoHookean, ModelBuilder, ShapeGeometry, TetMeshId};
pub use contacts::Contacts;
pub use control::Control;
pub use flags::{BodyFlags, ParticleFlags, ShapeFlags};
pub use joint::{EqualityType, JointTargetMode, JointType};
pub use model::{Heightfield, Model, SdfShape, TriMesh};
pub use state::State;
