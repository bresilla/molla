//! Geometry & collision pipeline.
//!
//! Top-level entry point is [`collide`], which runs the full broad-
//! and narrow-phase pipeline and writes contacts into a
//! [`crate::sim::Contacts`]. Building blocks include:
//!
//! - **Broad phase**: brute-force pairs ([`brute_force_pairs`]),
//!   top-down [`Bvh`], and a GPU LBVH (in [`crate::kernels`]).
//! - **Narrow phase**: per-pair [`pair_contact`] dispatch, with
//!   GJK/EPA ([`gjk`], [`epa`]) and MPR ([`mpr_collide`]) for convex
//!   pairs, dedicated routines for sphere/box/capsule/cylinder/plane,
//!   and triangle-mesh BVH traversal ([`MeshBvh`]).
//! - **Soft contact**: cross-mesh VT/EE pipeline ([`SoftSurface`],
//!   [`SoftBvh`], [`vt_ccd`]) — see Stage 33.
//! - **Hydroelastic**: rigid-vs-rigid pressure-field contact
//!   ([`compute_hydroelastic_contact`]).
//! - **Mesh authoring**: [`convex_hull`], [`marching_cubes`],
//!   [`bake_mesh_to_sdf`], and the sparse-volume helpers in
//!   [`sparse_grid`].

mod aabb;
mod broad_phase;
mod bvh;
pub mod contact_extras;
pub mod contact_reduction;
mod decimate;
mod epa;
pub mod gaussian_splat;
mod gjk;
mod hull;
pub mod hydroelastic;
mod marching_cubes;
pub mod mesh_bvh;
pub mod mpr;
mod narrow_phase;
mod pipeline;
pub mod sdf_bake;
mod shape_world;
pub mod soft_bvh;
pub mod soft_ccd;
pub mod soft_surface;
pub mod sparse_grid;
mod spatial_hash;
mod support;
pub mod tet_bvh;

pub use contact_extras::{build_contact_id_table, contact_id, match_contacts, reduce_contacts};
// Stage 28: alternative narrow-phase + standalone contact-reduction primitive.
pub use contact_reduction::{
    Contact as ReducedContact, reduce_contacts as reduce_contacts_per_pair, reduce_with_history,
};
pub use mpr::{Manifold as MprManifold, ShapeSupport, SupportFn, mpr_collide};

pub use aabb::Aabb;
pub use broad_phase::{BroadPhase, ShapePair, brute_force_pairs};
pub use bvh::{Bvh, BvhNode as BroadBvhNode};
pub use epa::{EpaContact, epa};
pub use narrow_phase::{ContactManifold, MAX_MANIFOLD_POINTS, pair_contact};
pub use pipeline::{CollisionPipeline, NarrowPhaseBackend, STATIC_BODY, collide};
// ── Stage 44 v1: 3D Gaussian splat primitive for the differentiable
// raytrace renderer (forward-only). ───────────────────────────────
pub use gaussian_splat::{
    GaussianSplat3d, SplatParam, canonical_ray_max_response, composite_forward_cpu,
    composite_rgb_cpu, fd_grad_one_splat, gaussian_response, map_ray_to_canonical,
    splat_camera_distance,
};
// ── Stage 44 v3: view-dependent (spherical-harmonics) splats. ──────
pub use gaussian_splat::{
    GaussianSplatSh3d, SH_BAND_2_COUNT, SH_C0, SH_C1, SH_C2A, SH_C2B, SH_C2C, SH_C2D, SH_C2E,
    ShCoeffParam, composite_sh_forward_cpu, composite_sh_rgb_cpu, eval_sh, fd_grad_sh_one_splat,
    sh_basis,
};
// ── Stage 44 v4: band-3 SH (Kerbl 2023) + diffuse/view-dep split. ──
pub use gaussian_splat::{
    GaussianSplatShBand3d, GaussianSplatShDiffuseView3d, SH_BAND_3_COUNT, SH_C3A, SH_C3B, SH_C3C,
    SH_C3D, SH_C3E, SH_C3F, SH_C3G, SH_VIEW_DEP_COUNT, ShBand3CoeffParam,
    composite_sh_band3_forward_cpu, composite_sh_band3_rgb_cpu,
    composite_sh_diffuse_view_forward_cpu, composite_sh_diffuse_view_rgb_cpu, eval_sh_band3,
    eval_sh_diffuse_view, fd_grad_sh_band3_one_splat, sh_basis_band3, sh_view_basis,
};
pub use gjk::{GjkDistance, GjkResult, GjkSimplex, SupportPoint, gjk};
// ── Stage 38d: shared constraint-graph coloring utility ────────────
//
// The implementation lives in `molla-math` because it is a generic graph
// utility used by simulation and compute code, not geometry-specific state.
// Keep this geometry re-export as a compatibility path for existing callers.
pub use crate::math::{color_constraints, greedy_color, group_by_color, mcs_color};
pub use mesh_bvh::{BvhNode as MeshBvhNode, MeshBvh};
// ── Stage 20.4: per-tet AABB for tet-vs-tet collision ──────────────
pub use tet_bvh::{
    TetAabb, recompute_tet_aabbs, tet_closest_face, tet_mesh_root_aabb, tet_meshes_aabb_overlap,
};

// ── Stage 33: cross-mesh soft-vs-soft VT contact infrastructure ────
pub use soft_bvh::SoftBvh;
pub use soft_surface::{
    ClothMeshId, INVALID_VERT, SoftSurface, SoftSurfaceKind, boundary_facets,
    compute_surface_edges, compute_tet_surface_triangles,
};
// ── Stage 33.4: VT-CCD root-finding for fast-impact tunnelling ─────
pub use shape_world::{shape_world_aabb, shape_world_transform};
pub use soft_ccd::{vt_ccd, vt_ccd_normal};
pub use support::{
    support, support_box, support_capsule, support_cone, support_convex_hull, support_cylinder,
    support_minkowski, support_sphere, support_with_hulls,
};

// ── Stage 21: mesh authoring tools ─────────────────────────────────
pub use decimate::{decimate, decimate_lod_chain};
pub use hull::{ConvexHullMesh, convex_hull};
pub use marching_cubes::marching_cubes;
pub use sdf_bake::{SdfGrid, bake_mesh_to_sdf, closest_point_on_triangle};
pub use spatial_hash::SpatialHashGrid;

// ── Stage 24e: NanoVDB-equivalent sparse-volume storage ────────────
// ── Stage 24e.1: GPU root hashing (`root_hash3`) ───────────────────
pub use sparse_grid::{
    INTERNAL_DIM, INTERNAL_MASK_U32S, INTERNAL_SHIFT, InternalNode, LEAF_DIM, LEAF_MASK_U32S,
    LEAF_SHIFT, Leaf, SparseGrid, SparseGridGpu, SparseSdf, bake_mesh_to_sparse_sdf, root_hash3,
    sparse_sdf_from_dense,
};

// ── Stage 22.1: hydroelastic contact ───────────────────────────────
#[cfg(feature = "gpu")]
pub use hydroelastic::HydroelasticKernel;
pub use hydroelastic::{
    HydroelasticConfig, HydroelasticContact, HydroelasticManifold, HydroelasticPolicy,
    SurfaceSample, compute_hydroelastic_contact, sample_plane_patch, sample_sdf_gradient,
    sample_sdf_trilinear, sample_sphere_surface,
};
// ── Stage 39D: hydroelastic contact polish (anchor + aggregate-stiffness) ──
pub use hydroelastic::{
    AggregateStiffnessConfig, AggregatedContact, AnchorReducedContact, PatchSample,
    StiffnessReduction, StiffnessSample, aggregate_by_bucket, manifold_to_samples,
    reduce_with_anchor,
};
