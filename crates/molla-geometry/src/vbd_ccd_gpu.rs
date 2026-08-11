//! GPU vertex-triangle CCD pass for VBD soft contact. Composes the two
//! validated GPU building blocks — the swept-AABB broad phase
//! ([`soft_bvh_query_points_swept_gpu`](crate::soft_bvh_gpu::soft_bvh_query_points_swept_gpu))
//! and the VT-CCD narrow phase
//! ([`vt_ccd_batch_gpu`](crate::soft_ccd_gpu::vt_ccd_batch_gpu)) — into
//! the earliest time-of-impact per moving vertex. This is the GPU path
//! for the VBD pre-substep CCD sweep that previously ran only on the CPU.

use crate::soft_bvh::SoftBvh;
use crate::soft_bvh_gpu::soft_bvh_query_points_swept_gpu;
use crate::soft_ccd_gpu::{VtCcdQuery, vt_ccd_batch_gpu};
use molla_core::Result;
use molla_math::{Real, Vec3};

/// Earliest VT-CCD time-of-impact in `[0, 1]` for each `query_vertices`
/// entry over the timestep `verts_t0 → verts_t1`, or `None` if that
/// vertex hits nothing. Adjacent (incident) triangles are skipped, so a
/// vertex never collides with a face it belongs to. `bvh` must already
/// be refit to the swept AABBs (`SoftBvh::refit_swept*`).
pub fn vbd_ccd_earliest_toi_gpu(
    engine: &molla_compute::Engine,
    bvh: &SoftBvh,
    verts_t0: &[Vec3],
    verts_t1: &[Vec3],
    tris: &[[u32; 3]],
    query_vertices: &[u32],
    margin: Real,
) -> Result<Vec<Option<Real>>> {
    if query_vertices.is_empty() {
        return Ok(Vec::new());
    }

    // Broad phase: candidate tris per moving vertex.
    let queries: Vec<(Vec3, Vec3, Real)> = query_vertices
        .iter()
        .map(|&vi| (verts_t0[vi as usize], verts_t1[vi as usize], margin))
        .collect();
    let candidates = soft_bvh_query_points_swept_gpu(engine, bvh, &queries)?;

    // Narrow phase: one VT-CCD query per (vertex, non-incident candidate).
    let mut ccd_queries: Vec<VtCcdQuery> = Vec::new();
    let mut owner: Vec<usize> = Vec::new();
    for (qi, cand) in candidates.iter().enumerate() {
        let vi = query_vertices[qi];
        for &t in cand {
            let tri = tris[t as usize];
            if tri.contains(&vi) {
                continue;
            }
            ccd_queries.push(VtCcdQuery {
                v_at_t0: verts_t0[vi as usize],
                v_at_t1: verts_t1[vi as usize],
                tri_at_t0: [
                    verts_t0[tri[0] as usize],
                    verts_t0[tri[1] as usize],
                    verts_t0[tri[2] as usize],
                ],
                tri_at_t1: [
                    verts_t1[tri[0] as usize],
                    verts_t1[tri[1] as usize],
                    verts_t1[tri[2] as usize],
                ],
            });
            owner.push(qi);
        }
    }

    let mut out = vec![None; query_vertices.len()];
    if ccd_queries.is_empty() {
        return Ok(out);
    }
    let tois = vt_ccd_batch_gpu(engine, &ccd_queries)?;
    for (k, toi) in tois.iter().enumerate() {
        if let Some(t) = toi {
            let slot = &mut out[owner[k]];
            *slot = Some(slot.map_or(*t, |e: Real| e.min(*t)));
        }
    }
    Ok(out)
}
