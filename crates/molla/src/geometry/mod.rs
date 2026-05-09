//! Geometry & collision pipeline.
//!
//! Stage 4.0 ships a CPU-only brute-force broad phase, sphere-sphere
//! and sphere-plane narrow phase, and the [`collide`] entry-point
//! that fills a [`crate::sim::Contacts`].

mod aabb;
mod broad_phase;
mod bvh;
pub mod contact_extras;
mod decimate;
mod epa;
mod gjk;
mod hull;
pub mod hydroelastic;
mod marching_cubes;
pub mod mesh_bvh;
mod narrow_phase;
mod pipeline;
pub mod sdf_bake;
mod shape_world;
mod spatial_hash;
pub mod sparse_grid;
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
    support, support_box, support_capsule, support_cone, support_convex_hull,
    support_cylinder, support_minkowski, support_sphere, support_with_hulls,
};

// ── Stage 21: mesh authoring tools ─────────────────────────────────
pub use decimate::{decimate, decimate_lod_chain};
pub use hull::{convex_hull, ConvexHullMesh};
pub use marching_cubes::marching_cubes;
pub use sdf_bake::{bake_mesh_to_sdf, closest_point_on_triangle, SdfGrid};
pub use spatial_hash::SpatialHashGrid;

// ── Stage 24e: NanoVDB-equivalent sparse-volume storage ────────────
pub use sparse_grid::{
    bake_mesh_to_sparse_sdf, sparse_sdf_from_dense, InternalNode, Leaf, SparseGrid,
    SparseGridGpu, SparseSdf, INTERNAL_DIM, INTERNAL_MASK_U32S, INTERNAL_SHIFT, LEAF_DIM,
    LEAF_MASK_U32S, LEAF_SHIFT,
};

// ── Stage 22.1: hydroelastic contact ───────────────────────────────
pub use hydroelastic::{
    compute_hydroelastic_contact, sample_plane_patch, sample_sdf_gradient,
    sample_sdf_trilinear, sample_sphere_surface, HydroelasticConfig, HydroelasticContact,
    HydroelasticManifold, HydroelasticPolicy, SurfaceSample,
};
#[cfg(feature = "gpu")]
pub use hydroelastic::HydroelasticKernel;
