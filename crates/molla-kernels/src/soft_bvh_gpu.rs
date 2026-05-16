//! GPU LBVH wrapper for soft-surface tris — Stage 33.2.
//!
//! Builds a Karras-style LBVH over every soft-surface triangle each
//! substep, **without** a host-side `particle_q.download`:
//!
//! 1. `compute_tri_aabbs` (in `shaders/soft_vt_contact.wgsl`): one
//!    thread per tri reads the three vert positions from `particle_q`
//!    and writes the per-tri AABB **directly into the LBVH's
//!    `primitive_aabb` buffer**.
//! 2. [`LbvhBroadPhase::refit_in_place`] consumes those AABBs to refit
//!    the cached BVH topology in place.
//!
//! Topology is seeded once at construction using AABBs derived from
//! `Model.initial_particle_q` (no `State` download required); the
//! refit pass reuses that topology for every subsequent substep. A
//! drift of >2× in scene extent invalidates the Morton codes — at
//! that point the caller should rebuild the wrapper.
//!
//! The downstream Stage 33.2 VT contact gather kernel reads the LBVH
//! `nodes` + `sorted_idx` buffers via the public accessors on
//! [`LbvhBroadPhase`].

use std::sync::Arc;

use molla_core::{Error, Result};
use molla_math::Vec3;
use molla_sim::Model;

use super::LbvhBroadPhase;

/// GPU LBVH refit driver for soft surfaces.
///
/// Holds the underlying [`LbvhBroadPhase`] (which owns the per-tri
/// `primitive_aabb` storage; the `compute_tri_aabbs` kernel writes
/// directly into it).
pub struct SoftBvhGpu {
    /// Number of soft tris this driver was built for. Stays constant
    /// for the lifetime of the wrapper — rebuild on topology change.
    tri_count: usize,
    /// Number of soft surfaces — for parity-check that the model's
    /// registry hasn't drifted between substeps.
    surface_count: usize,
    /// Cached LBVH driver. Reused across substeps so the radix-sort +
    /// Morton + build cost is paid once on construction, then every
    /// subsequent refit reuses the topology.
    lbvh: LbvhBroadPhase,
}

impl SoftBvhGpu {
    /// Construct the wrapper sized for `model.soft_surface_tris.len()`
    /// tris and seed the LBVH topology from
    /// `model.initial_particle_q`.
    ///
    /// The seed AABBs are deliberately permissive (per-tri bounding
    /// boxes derived from the initial particle positions); subsequent
    /// substeps refit the existing topology in place against the live
    /// `state.particle_q` via [`Self::refit`].
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, model: &Model) -> Result<Self> {
        let tri_count = model.soft_surface_tris.len();
        let surface_count = model.soft_surfaces.len();
        // LBVH internally sizes for `prim_capacity = max(2)`. Empty
        // soft-surface registries shouldn't dispatch this kernel at
        // all but we still construct a minimal LBVH so the rest of
        // the GPU contact pipeline doesn't crash.
        let mut lbvh = LbvhBroadPhase::new(device.clone(), queue.clone(), tri_count.max(2), 64)?;
        // Seed the LBVH topology from the initial particle positions.
        // We need this so `refit_in_place` (called from
        // [`Self::refit`]) has a valid `cached_topology`. Worth the
        // one-time host walk because it's CPU-only at construction
        // time (no per-substep cost).
        if tri_count >= 1 {
            let q0 = model.initial_particle_q.host()?;
            let mut aabbs: Vec<(Vec3, Vec3)> = Vec::with_capacity(tri_count);
            for tri in &model.soft_surface_tris {
                let v0 = q0[tri[0] as usize];
                let v1 = q0[tri[1] as usize];
                let v2 = q0[tri[2] as usize];
                aabbs.push((v0.min(v1).min(v2), v0.max(v1).max(v2)));
            }
            // Inflate the seed scene bounds: soft bodies will move
            // around (gravity, contact response). A 4× inflation gives
            // headroom before Morton-code resolution loss kicks in.
            // We pad by inflating each AABB to the same centroid +
            // a fraction of the scene's diagonal — cheap one-time
            // pad to avoid drift during the first few seconds of sim.
            // (`refit_in_place` is tolerant of moderate drift; this
            // keeps Morton codes well-distributed for a longer run.)
            //
            // For v1 we just use the raw seed AABBs — the caller can
            // rebuild the wrapper if a long sim drifts the bounds.
            let _ = lbvh.dispatch(&aabbs)?;
        }
        Ok(Self {
            tri_count,
            surface_count,
            lbvh,
        })
    }

    /// Number of soft tris this driver was built for.
    pub fn tri_count(&self) -> usize {
        self.tri_count
    }

    /// Read-access to the underlying LBVH driver — the gather kernel
    /// reads its `nodes` + `sorted_idx` + `pair_count` buffers.
    pub fn lbvh(&self) -> &LbvhBroadPhase {
        &self.lbvh
    }

    /// Mutable access to the underlying LBVH driver — needed for
    /// dispatching `refit_in_place` from external orchestrators.
    /// Most callers should prefer [`Self::refit`].
    pub fn lbvh_mut(&mut self) -> &mut LbvhBroadPhase {
        &mut self.lbvh
    }

    /// Run the per-substep refit. **The caller must dispatch
    /// `compute_tri_aabbs` (in `shaders/soft_vt_contact.wgsl`) first**
    /// so the LBVH's `primitive_aabb` buffer holds the current
    /// per-tri AABBs. This wrapper only orchestrates the LBVH side
    /// (`refit_aabbs` + `find_pairs`).
    ///
    /// Topology must already be seeded (handled by [`Self::new`]).
    pub fn refit(&mut self, model: &Model) -> Result<()> {
        if self.tri_count == 0 {
            return Ok(());
        }
        // Sanity: model's tri / surface count hasn't drifted since
        // construction. If it has, the caller needs to rebuild this
        // driver from scratch.
        if model.soft_surface_tris.len() != self.tri_count {
            return Err(Error::Gpu(format!(
                "SoftBvhGpu: tri count drift ({} → {}) — rebuild the driver",
                self.tri_count,
                model.soft_surface_tris.len()
            )));
        }
        if model.soft_surfaces.len() != self.surface_count {
            return Err(Error::Gpu(format!(
                "SoftBvhGpu: surface count drift ({} → {}) — rebuild the driver",
                self.surface_count,
                model.soft_surfaces.len()
            )));
        }
        // Refit using the AABBs already on the device (written by the
        // caller's `compute_tri_aabbs` dispatch). Zero host bounce.
        self.lbvh.refit_in_place(self.tri_count as u32)
    }
}
