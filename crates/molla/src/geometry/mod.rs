//! Geometry & collision pipeline.
//!
//! Stage 4.0 ships a CPU-only brute-force broad phase, sphere-sphere
//! and sphere-plane narrow phase, and the [`collide`] entry-point
//! that fills a [`crate::sim::Contacts`].

mod aabb;
mod broad_phase;
mod narrow_phase;
mod pipeline;
mod shape_world;

pub use aabb::Aabb;
pub use broad_phase::{brute_force_pairs, BroadPhase, ShapePair};
pub use narrow_phase::{pair_contact, ContactManifold, MAX_MANIFOLD_POINTS};
pub use pipeline::{collide, CollisionPipeline, STATIC_BODY};
pub use shape_world::{shape_world_aabb, shape_world_transform};
