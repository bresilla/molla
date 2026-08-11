//! Geometry and collision primitives.

pub mod aabb;
pub mod broad_phase;
pub mod bvh;
pub mod contact_extras;
pub mod contact_reduction;
pub mod decimate;
pub mod epa;
#[cfg(feature = "gpu")]
mod epa_gpu;
pub mod gaussian_splat;
pub mod gjk;
#[cfg(feature = "gpu")]
mod gjk_gpu;
pub mod hull;
pub mod hydroelastic;
#[cfg(feature = "gpu")]
mod lbvh;
pub mod marching_cubes;
#[cfg(feature = "gpu")]
mod marching_cubes_gpu;
pub mod mesh_bvh;
#[cfg(feature = "gpu")]
mod mesh_bvh_raycast_gpu;
pub mod mpr;
#[cfg(feature = "gpu")]
mod mpr_gpu;
pub mod narrow_phase;
pub mod pipeline;
pub mod sdf_bake;
#[cfg(feature = "gpu")]
mod sdf_bake_gpu;
pub mod self_contact;
#[cfg(feature = "gpu")]
mod self_contact_gpu;
pub mod shape_world;
pub mod soft_bvh;
#[cfg(feature = "gpu")]
mod soft_bvh_gpu;
pub mod soft_ccd;
#[cfg(feature = "gpu")]
mod soft_ccd_gpu;
pub mod sparse_grid;
pub mod spatial_hash;
pub mod support;
pub mod tet_bvh;
#[cfg(feature = "gpu")]
mod vbd_ccd_gpu;

pub use aabb::Aabb;
pub use broad_phase::{BroadPhase, ShapePair, brute_force_pairs};
pub use bvh::{Bvh, BvhNode as BroadBvhNode};
pub use contact_extras::{build_contact_id_table, contact_id, match_contacts, reduce_contacts};
pub use contact_reduction::{
    Contact as ReducedContact, reduce_contacts as reduce_contacts_per_pair, reduce_with_history,
};
pub use decimate::{decimate, decimate_lod_chain};
pub use epa::{EpaContact, epa};
#[cfg(feature = "gpu")]
pub use epa_gpu::{
    EPA_OUT_STRIDE, EPA_STATUS_FALLBACK, EPA_STATUS_NONE, EPA_STATUS_OK, EpaKernel, EpaPairResult,
    decode_epa_out,
};
pub use gaussian_splat::{
    GaussianSplat3d, GaussianSplatSh3d, GaussianSplatShBand3d, GaussianSplatShDiffuseView3d,
    SH_BAND_2_COUNT, SH_BAND_3_COUNT, SH_C0, SH_C1, SH_C2A, SH_C2B, SH_C2C, SH_C2D, SH_C2E, SH_C3A,
    SH_C3B, SH_C3C, SH_C3D, SH_C3E, SH_C3F, SH_C3G, SH_VIEW_DEP_COUNT, ShBand3CoeffParam,
    ShCoeffParam, SplatParam, canonical_ray_max_response, composite_forward_cpu, composite_rgb_cpu,
    composite_sh_band3_forward_cpu, composite_sh_band3_rgb_cpu,
    composite_sh_diffuse_view_forward_cpu, composite_sh_diffuse_view_rgb_cpu,
    composite_sh_forward_cpu, composite_sh_rgb_cpu, eval_sh, eval_sh_band3, eval_sh_diffuse_view,
    fd_grad_one_splat, fd_grad_sh_band3_one_splat, fd_grad_sh_one_splat, gaussian_response,
    map_ray_to_canonical, sh_basis, sh_basis_band3, sh_view_basis, splat_camera_distance,
};
pub use gjk::{GjkDistance, GjkResult, GjkSimplex, SupportPoint, gjk};
#[cfg(feature = "gpu")]
pub use gjk_gpu::{GJK_SIMPLEX_STRIDE, GjkKernel};
pub use hull::{ConvexHullMesh, convex_hull};
#[cfg(feature = "gpu")]
pub use hydroelastic::HydroelasticKernel;
pub use hydroelastic::{
    AggregateStiffnessConfig, AggregatedContact, AnchorReducedContact, HydroelasticConfig,
    HydroelasticContact, HydroelasticManifold, HydroelasticPolicy, PatchSample, StiffnessReduction,
    StiffnessSample, SurfaceSample, aggregate_by_bucket, compute_hydroelastic_contact,
    manifold_to_samples, reduce_with_anchor, sample_plane_patch, sample_sdf_gradient,
    sample_sdf_trilinear, sample_sphere_surface,
};
#[cfg(feature = "gpu")]
pub use lbvh::{LbvhBroadPhase, LbvhPairs, collect_shape_aabbs};
pub use marching_cubes::marching_cubes;
#[cfg(feature = "gpu")]
pub use marching_cubes_gpu::marching_cubes_gpu;
pub use mesh_bvh::{BvhNode as MeshBvhNode, MeshBvh};
#[cfg(feature = "gpu")]
pub use mesh_bvh_raycast_gpu::{MeshBvhGpu, MeshBvhRaycastKernel};
pub use mpr::{Manifold as MprManifold, ShapeSupport, SupportFn, mpr_collide};
#[cfg(feature = "gpu")]
pub use mpr_gpu::{
    MPR_DEFAULT_TOLERANCE, MPR_OUT_STRIDE, MPR_STATUS_FALLBACK, MPR_STATUS_NONE, MPR_STATUS_OK,
    MprKernel, MprPairResult, decode_mpr_out,
};
pub use narrow_phase::{ContactManifold, MAX_MANIFOLD_POINTS, pair_contact};
pub use pipeline::{CollisionPipeline, NarrowPhaseBackend, STATIC_BODY, collide};
pub use sdf_bake::{SdfGrid, bake_mesh_to_sdf, closest_point_on_triangle};
#[cfg(feature = "gpu")]
pub use sdf_bake_gpu::{SdfBakeKernel, bake_mesh_to_sdf_gpu};
pub use self_contact::{SelfContactCandidate, detect_vertex_triangle_self_contacts};
#[cfg(feature = "gpu")]
pub use self_contact_gpu::{SelfContactKernel, detect_vertex_triangle_self_contacts_gpu};
pub use shape_world::{shape_world_aabb, shape_world_transform};
pub use soft_bvh::SoftBvh;
#[cfg(feature = "gpu")]
pub use soft_bvh_gpu::soft_bvh_query_points_swept_gpu;
pub use soft_ccd::{vt_ccd, vt_ccd_normal};
#[cfg(feature = "gpu")]
pub use soft_ccd_gpu::{SoftCcdKernel, VtCcdQuery, vt_ccd_batch_gpu};
pub use sparse_grid::{
    INTERNAL_DIM, INTERNAL_MASK_U32S, INTERNAL_SHIFT, InternalNode, LEAF_DIM, LEAF_MASK_U32S,
    LEAF_SHIFT, Leaf, SparseGrid, SparseGridGpu, SparseSdf, bake_mesh_to_sparse_sdf, root_hash3,
    sparse_sdf_from_dense,
};
pub use spatial_hash::SpatialHashGrid;
pub use support::{
    support, support_box, support_capsule, support_cone, support_convex_hull, support_cylinder,
    support_ellipsoid, support_minkowski, support_sphere, support_triangle, support_with_hulls,
};
pub use tet_bvh::{
    TetAabb, TetBvh, TetBvhNode, recompute_tet_aabbs, tet_closest_face, tet_mesh_root_aabb,
    tet_meshes_aabb_overlap,
};
#[cfg(feature = "gpu")]
pub use vbd_ccd_gpu::vbd_ccd_earliest_toi_gpu;
