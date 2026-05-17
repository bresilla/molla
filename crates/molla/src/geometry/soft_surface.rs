//! Facade compatibility for soft-surface data and FEM boundary facets.
//!
//! Raw soft-surface model descriptors live in `molla-sim`; FEM boundary
//! extraction lives in `molla-fem`.

pub use molla_fem::boundary_facets;
pub use molla_sim::{
    ClothMeshId, INVALID_VERT, SoftSurface, SoftSurfaceKind, compute_surface_edges,
    compute_tet_surface_triangles,
};
