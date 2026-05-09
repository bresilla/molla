//! Core simulation types: [`Model`], [`State`], [`Control`], [`Contacts`].
//!
//! See PLAN.md §10–11.

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
