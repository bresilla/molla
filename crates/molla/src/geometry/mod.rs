//! Geometry & collision pipeline.
//!
//! Stage 4.0 ships a CPU-only brute-force broad phase, sphere-sphere
//! and sphere-plane narrow phase, and the [`collide`] entry-point
//! that fills a [`crate::sim::Contacts`].

mod aabb;
mod broad_phase;
mod bvh;
pub mod contact_extras;
mod epa;
mod gjk;
pub mod mesh_bvh;
mod narrow_phase;
mod pipeline;
mod shape_world;
mod support;

pub use contact_extras::{
    build_contact_id_table, contact_id, match_contacts, reduce_contacts,
};

pub use aabb::Aabb;
pub use broad_phase::{brute_force_pairs, BroadPhase, ShapePair};
pub use narrow_phase::{pair_contact, ContactManifold, MAX_MANIFOLD_POINTS};
pub use pipeline::{collide, CollisionPipeline, STATIC_BODY};
pub use bvh::{Bvh, BvhNode as BroadBvhNode};
pub use epa::{epa, EpaContact};
pub use gjk::{gjk, GjkDistance, GjkResult, GjkSimplex, SupportPoint};
pub use mesh_bvh::{BvhNode as MeshBvhNode, MeshBvh};
pub use shape_world::{shape_world_aabb, shape_world_transform};
pub use support::{
    support, support_box, support_capsule, support_cylinder, support_minkowski,
    support_sphere,
};
