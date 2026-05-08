//! Core simulation types: [`Model`], [`State`], [`Control`], [`Contacts`].
//!
//! Stage 0 ships the type skeletons and field declarations — see PLAN.md
//! §10. Builder + actual array population land in stage 1.

mod contacts;
mod control;
mod model;
mod state;

pub use contacts::Contacts;
pub use control::Control;
pub use model::Model;
pub use state::State;
