//! Core simulation types: [`Model`], [`State`], [`Control`], [`Contacts`].
//!
//! See PLAN.md §10–11.

mod builder;
mod contacts;
mod control;
pub mod flags;
mod model;
mod state;

pub use builder::{BodyParams, ModelBuilder, ShapeGeometry};
pub use contacts::Contacts;
pub use control::Control;
pub use flags::{BodyFlags, ParticleFlags, ShapeFlags};
pub use model::Model;
pub use state::State;
