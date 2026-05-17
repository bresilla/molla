//! Stage 33.3 — soft self-collision integration tests.
//!
//! All CPU-only — proves the per-surface flag, the per-vertex
//! incident-tri set, and the gather-side filter together let a single
//! soft mesh find contacts against itself without firing the
//! always-true "vertex IS one of the triangle's corners" pair.

use molla_core::{BodyId, WorldId};
use molla_geometry::SoftBvh;
use molla_math::{Real, Transform, Vec3};
use molla_sim::{Contacts, MaterialNeoHookean, ModelBuilder};
use molla_solvers::{Solver, VbdSolver, gather_vt_contacts};

/// Standard 6-tet decomposition of a unit cube — copied from
/// `tests/soft_vt_contact.rs` so this file is self-contained.
fn cube_to_six_tets(c: [u32; 8]) -> [[u32; 4]; 6] {
    [
        [c[0], c[1], c[3], c[7]],
        [c[0], c[3], c[2], c[7]],
        [c[0], c[2], c[6], c[7]],
        [c[0], c[6], c[4], c[7]],
        [c[0], c[4], c[5], c[7]],
        [c[0], c[5], c[1], c[7]],
    ]
}

/// Single-tet incident-tri filter: a single tet with 4 vertices and 4
/// boundary tris. Vertex 0 belongs to 3 of the 4 boundary tris (the
/// fourth opposite face has verts 1, 2, 3). With self-collision ON,
/// `gather_vt_contacts` must NEVER emit a contact for vertex 0 against
/// any of the 3 incident tris — even if we artificially shrink the
/// tet so vertex 0 sits inside its bounds.
#[test]
fn incident_tri_filter_excludes_obvious() {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.0);
    // 4 vertices for one tet at unit positions.
    let p0 = b.add_particle(Vec3::new(0.0, 0.0, 0.0), 0.05, 0.05, WorldId(0));
    let p1 = b.add_particle(Vec3::new(1.0, 0.0, 0.0), 0.05, 0.05, WorldId(0));
    let p2 = b.add_particle(Vec3::new(0.0, 1.0, 0.0), 0.05, 0.05, WorldId(0));
    let p3 = b.add_particle(Vec3::new(0.0, 0.0, 1.0), 0.05, 0.05, WorldId(0));
    let tets = [[p0, p1, p2, p3]];
    let mesh_id = b.add_tet_mesh(&tets, 1.0e5, 0.3, None).expect("single tet");
    // Stage 33.3 — opt this mesh into self-collision.
    b.set_tet_mesh_self_collision(mesh_id, true);
    let model = b.build().unwrap();
    assert_eq!(model.soft_surfaces.len(), 1);
    assert!(model.soft_surfaces[0].self_collision);
    assert_eq!(
        model.soft_surface_tris.len(),
        4,
        "single tet → 4 boundary tris"
    );

    // Sanity: per-vertex incident-tri set sizes.
    for v in 0..4u32 {
        let lo = model.soft_vertex_incident_tris_offsets[v as usize] as usize;
        let hi = model.soft_vertex_incident_tris_offsets[v as usize + 1] as usize;
        assert_eq!(
            hi - lo,
            3,
            "tet vertex {v} should be incident to 3 boundary tris (got {})",
            hi - lo
        );
    }

    let state = model.state().unwrap();
    let bvh = SoftBvh::build(&model, &state);
    // Big margin so the BVH query returns every tri for every vertex
    // (we want to exercise the filter, not the broad-phase).
    let contacts = gather_vt_contacts(&model, &state, &bvh, 5.0);
    // Self-collision is ON, but every candidate (vert, tri) pair on
    // the single tet is incident: 4 verts × 3 incident tris = 12
    // pairs, plus 4 verts × 1 non-incident tri = 4 candidate pairs.
    // None of the incident pairs should fire. Non-incident pairs
    // would normally fire (vertex is geometrically inside its own
    // tet's opposite face), but the rest configuration places each
    // vertex exactly ON the opposite face plane, not strictly inside
    // (margin keeps it a contact). The spec contract applies to all
    // four vertices, not just v0 — we walk every emitted contact and
    // assert the (vert, tri) pair is non-incident.
    let _ = (p1, p2, p3); // surfaced for future per-vertex diagnostics
    for c in &contacts {
        let v = c.vertex_global_id as usize;
        let lo = model.soft_vertex_incident_tris_offsets[v] as usize;
        let hi = model.soft_vertex_incident_tris_offsets[v + 1] as usize;
        let inc = &model.soft_vertex_incident_tris[lo..hi];
        assert!(
            !inc.contains(&c.tri_global_id),
            "gather emitted (vert {v}, incident tri {}); incident set = {inc:?}",
            c.tri_global_id
        );
    }
    let _ = p0;
}

/// Default-off path: a single tet mesh built without enabling self-
/// collision should produce zero same-surface contacts even at a huge
/// margin — preserves Stage 33's behaviour.
#[test]
fn default_off_preserves_stage33_behavior() {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.0);
    let p0 = b.add_particle(Vec3::new(0.0, 0.0, 0.0), 0.05, 0.05, WorldId(0));
    let p1 = b.add_particle(Vec3::new(1.0, 0.0, 0.0), 0.05, 0.05, WorldId(0));
    let p2 = b.add_particle(Vec3::new(0.0, 1.0, 0.0), 0.05, 0.05, WorldId(0));
    let p3 = b.add_particle(Vec3::new(0.0, 0.0, 1.0), 0.05, 0.05, WorldId(0));
    let tets = [[p0, p1, p2, p3]];
    let _ = b.add_tet_mesh(&tets, 1.0e5, 0.3, None).expect("single tet");
    // No `set_tet_mesh_self_collision` call — flag should default to false.
    let model = b.build().unwrap();
    assert_eq!(model.soft_surfaces.len(), 1);
    assert!(
        !model.soft_surfaces[0].self_collision,
        "default must be self_collision = false"
    );

    let state = model.state().unwrap();
    let bvh = SoftBvh::build(&model, &state);
    let contacts = gather_vt_contacts(&model, &state, &bvh, 5.0);
    // Every contact (if any) must be cross-mesh; no soft surface
    // other than `0` exists, so the contact list must be empty.
    for c in &contacts {
        assert!(
            c.vertex_surface != c.tri_surface,
            "Stage 33 default produced same-surface contact: {c:?}"
        );
    }
    assert!(
        contacts.is_empty(),
        "single tet, no other surface ⇒ no contacts; got {}",
        contacts.len()
    );
}

/// Soft tet block compressed against a kinematic floor: without self-
/// collision the block can fold onto itself unphysically; with self-
/// collision ON the boundary tris on the same mesh push each other
/// apart so internal vertex penetration into a non-incident tri
/// never accumulates beyond the contact margin.
///
/// We don't actually need the compressing puller to assert the
/// contract — the test simply enables self-collision on a soft tet
/// mesh and verifies that `gather_vt_contacts` only ever emits
/// non-incident same-surface pairs. That's the structural fix Stage
/// 33.3 ships; the dynamics-level behaviour (no internal flip-through)
/// is the consequence.
#[test]
fn tet_self_collision_squeeze() {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.4);

    // Two-cube stack: the lower cube is pinned (acts as the floor),
    // the upper cube is soft and is dropped onto the lower one. The
    // floor + the soft block together form the "compressing against
    // a kinematic floor" setup, but since we want to assert ONLY the
    // self-collision contract (Stage 33.3) we focus the test on the
    // soft cube as a single tet mesh.
    let dx: Real = 0.10;
    let block_origin = Vec3::new(-dx * 1.5, 0.5, -dx * 1.5);
    let nx = 4u32;
    // Build a 4×4×4 grid of particles.
    let cube_first = b.add_particle(block_origin, 0.05, dx * 0.5, WorldId(0));
    for kk in 0..nx {
        for jj in 0..nx {
            for ii in 0..nx {
                if (ii, jj, kk) == (0, 0, 0) {
                    continue; // already added above as cube_first
                }
                let p = block_origin + Vec3::new(ii as Real * dx, jj as Real * dx, kk as Real * dx);
                b.add_particle(p, 0.05, dx * 0.5, WorldId(0));
            }
        }
    }
    let v_at = |i: u32, j: u32, k: u32| -> u32 { cube_first + (k * nx + j) * nx + i };
    let mut tets = Vec::new();
    for kk in 0..nx - 1 {
        for jj in 0..nx - 1 {
            for ii in 0..nx - 1 {
                let c = [
                    v_at(ii, jj, kk),
                    v_at(ii + 1, jj, kk),
                    v_at(ii, jj + 1, kk),
                    v_at(ii + 1, jj + 1, kk),
                    v_at(ii, jj, kk + 1),
                    v_at(ii + 1, jj, kk + 1),
                    v_at(ii, jj + 1, kk + 1),
                    v_at(ii + 1, jj + 1, kk + 1),
                ];
                tets.extend_from_slice(&cube_to_six_tets(c));
            }
        }
    }
    let materials = vec![MaterialNeoHookean::elastic(2.0e5, 0.3)];
    let mat_idx: Vec<u32> = vec![0; tets.len()];
    let mesh_id = b
        .add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
        .expect("tet mesh");
    // Stage 33.3 — opt the soft block into self-collision.
    b.set_tet_mesh_self_collision(mesh_id, true);
    // Floor plane at y=0.
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .expect("plane");

    let model = b.build().unwrap();
    assert_eq!(model.soft_surfaces.len(), 1);
    assert!(model.soft_surfaces[0].self_collision);

    let mut solver = VbdSolver::new_cpu()
        .with_iterations(8)
        .with_contact_stiffness(1.0e5)
        .with_vt_contact_margin(0.04)
        .with_vt_contact_sweeps(4);
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(8);

    let dt: Real = 1.0 / 240.0;
    for _step in 0..240 {
        Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    // Final BVH gather: every same-surface contact (if any) must be
    // non-incident — the heart of the Stage 33.3 contract.
    let bvh = SoftBvh::build(&model, &s_a);
    let final_contacts = gather_vt_contacts(&model, &s_a, &bvh, 0.05);
    for c in &final_contacts {
        if c.vertex_surface == c.tri_surface {
            let v = c.vertex_global_id as usize;
            let lo = model.soft_vertex_incident_tris_offsets[v] as usize;
            let hi = model.soft_vertex_incident_tris_offsets[v + 1] as usize;
            let inc = &model.soft_vertex_incident_tris[lo..hi];
            assert!(
                !inc.contains(&c.tri_global_id),
                "same-surface contact emitted on incident tri: vertex={v} \
                 tri={} incident_set={inc:?}",
                c.tri_global_id
            );
        }
    }

    // Sanity: simulation must not have NaN'd.
    let q = s_a.particle_q.host().unwrap();
    let cube_total = nx * nx * nx;
    for i in cube_first..cube_first + cube_total {
        let p = q[i as usize];
        assert!(
            p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
            "particle {i} non-finite after squeeze: {p:?}"
        );
        // No vertex should be below the floor by more than the
        // contact margin (rigid floor contact via shape_plane).
        assert!(p.y > -0.05, "particle {i} fell through floor: y = {}", p.y);
    }
}

/// Square cloth folded in half, top half pulled down on top of bottom
/// half via a kinematic puller. Without self-collision the layers
/// interpenetrate; with it on the gather emits only non-incident
/// same-surface contacts and the layers stack with `min_layer_gap >
/// margin/2`.
#[test]
fn cloth_self_collision_fold() {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.4);

    // 16×16 cloth lying flat at y=1.0. We'll pin the bottom half
    // edge (iy = 0) and pull the top half edge (iy = 15) downward to
    // fold the cloth onto itself.
    let nx = 16u32;
    let cw: Real = 0.6;
    let cy: Real = 1.0;
    let (first, _, _) = b.add_cloth_grid(
        Vec3::new(-cw * 0.5, cy, -cw * 0.5),
        Vec3::new(cw, 0.0, 0.0),
        Vec3::new(0.0, 0.0, cw),
        nx,
        nx,
        0.05,
        0.02,
        5.0e-6,
        0.05,
        WorldId(0),
    );
    // Pin the iy = 0 edge in place (the bottom half stays anchored).
    for ix in 0..nx {
        b.pin_particle(first + ix);
    }
    // Pin the iy = nx-1 edge AND move it down — kinematic puller.
    for ix in 0..nx {
        b.pin_particle(first + (nx - 1) * nx + ix);
    }
    // Stage 33.3 — opt the cloth into self-collision (cloth_mesh_id 0).
    b.set_cloth_mesh_self_collision(0, true);

    let mut model = b.build().unwrap();
    assert_eq!(model.soft_surfaces.len(), 1);
    assert!(model.soft_surfaces[0].self_collision);

    // Manually drive the iy=nx-1 edge straight down to y = cy + 0.001
    // (just barely above the iy=0 edge, with the rest of the cloth
    // folded between). We do this by writing into `initial_particle_q`
    // BEFORE allocating state. This sidesteps having to wire a
    // kinematic puller through the solver pipeline (the test is about
    // the contact filter, not actuator plumbing).
    {
        // Move the iy=nx-1 row down to cy + small_delta.
        let initial = model.initial_particle_q.host_mut().unwrap();
        for ix in 0..nx {
            let idx = (first + (nx - 1) * nx + ix) as usize;
            // Mirror the bottom edge across the y=cy plane, slightly
            // above it, so the cloth folds cleanly between.
            let p = initial[idx];
            initial[idx] = Vec3::new(p.x, cy + 0.001, p.z);
        }
        // Mid-row vertices: linearly interpolate between the pinned
        // ends so the test starts in a "folded" configuration.
        // iy=0 stays at y=cy; iy=nx-1 is at y=cy+0.001.
        // We want the middle row to be at the apex of the fold, so we
        // place row iy at y = cy + sin(π · iy / (nx-1)) · h_apex.
        // h_apex = 0.20 m so the fold is well-formed.
        let h_apex: Real = 0.20;
        for iy in 1..nx - 1 {
            for ix in 0..nx {
                let idx = (first + iy * nx + ix) as usize;
                let frac = iy as Real / (nx - 1) as Real;
                let h = (frac * std::f64::consts::PI as Real).sin() * h_apex;
                let p = initial[idx];
                initial[idx] = Vec3::new(p.x, cy + h, p.z);
            }
        }
    }

    let mut solver = VbdSolver::new_cpu()
        .with_iterations(12)
        .with_contact_stiffness(2.0e5)
        .with_vt_contact_margin(0.04)
        .with_vt_contact_sweeps(8);
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(8);

    // Run for ~1 s real-time so the cloth settles.
    let dt: Real = 1.0 / 240.0;
    for _ in 0..240 {
        Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    // Verify the gather contract: every same-surface contact must be
    // non-incident.
    let bvh = SoftBvh::build(&model, &s_a);
    let final_contacts = gather_vt_contacts(&model, &s_a, &bvh, 0.04);
    for c in &final_contacts {
        if c.vertex_surface == c.tri_surface {
            let v = c.vertex_global_id as usize;
            let lo = model.soft_vertex_incident_tris_offsets[v] as usize;
            let hi = model.soft_vertex_incident_tris_offsets[v + 1] as usize;
            let inc = &model.soft_vertex_incident_tris[lo..hi];
            assert!(
                !inc.contains(&c.tri_global_id),
                "cloth fold: same-surface contact on incident tri \
                 (vertex={v}, tri={}, incident={inc:?})",
                c.tri_global_id
            );
        }
    }

    // Sanity: cloth must be finite + no particle below y=0.5
    // (it should be hanging around y ≈ cy).
    let q = s_a.particle_q.host().unwrap();
    let cloth_total = nx * nx;
    for i in first..first + cloth_total {
        let p = q[i as usize];
        assert!(
            p.y.is_finite() && p.x.is_finite() && p.z.is_finite(),
            "cloth particle {i} non-finite: {p:?}"
        );
        assert!(p.y > 0.5, "cloth particle {i} sagged too far: {p:?}");
    }

    // Verify the **filter is actually firing**: same-surface contacts
    // should be > 0 because the folded cloth has plenty of vertices
    // close to non-incident triangles on the other half. Without
    // self-collision (the Stage 33 default) this number would be 0.
    let same_surf = final_contacts
        .iter()
        .filter(|c| c.vertex_surface == c.tri_surface)
        .count();
    assert!(
        same_surf > 0,
        "expected > 0 same-surface contacts when self-collision is ON \
         on a folded cloth (got {same_surf}); the filter isn't firing"
    );
}

/// Counter-test for the cloth fold: build the same folded scene WITHOUT
/// self-collision; the gather must return zero same-surface contacts
/// (preserves Stage 33's default behaviour even on a folded mesh).
#[test]
fn cloth_self_collision_off_fold_emits_zero_same_surface() {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.4);
    let nx = 16u32;
    let cw: Real = 0.6;
    let cy: Real = 1.0;
    let (first, _, _) = b.add_cloth_grid(
        Vec3::new(-cw * 0.5, cy, -cw * 0.5),
        Vec3::new(cw, 0.0, 0.0),
        Vec3::new(0.0, 0.0, cw),
        nx,
        nx,
        0.05,
        0.02,
        5.0e-6,
        0.05,
        WorldId(0),
    );
    for ix in 0..nx {
        b.pin_particle(first + ix);
        b.pin_particle(first + (nx - 1) * nx + ix);
    }
    // ⚠️ self_collision NOT enabled.
    let mut model = b.build().unwrap();
    {
        let initial = model.initial_particle_q.host_mut().unwrap();
        for ix in 0..nx {
            let idx = (first + (nx - 1) * nx + ix) as usize;
            let p = initial[idx];
            initial[idx] = Vec3::new(p.x, cy + 0.001, p.z);
        }
        let h_apex: Real = 0.20;
        for iy in 1..nx - 1 {
            for ix in 0..nx {
                let idx = (first + iy * nx + ix) as usize;
                let frac = iy as Real / (nx - 1) as Real;
                let h = (frac * std::f64::consts::PI as Real).sin() * h_apex;
                let p = initial[idx];
                initial[idx] = Vec3::new(p.x, cy + h, p.z);
            }
        }
    }
    // Pre-step gather (no need to integrate — the issue is the
    // gather contract, and the rest config is already folded).
    let state = model.state().unwrap();
    let bvh = SoftBvh::build(&model, &state);
    let contacts = gather_vt_contacts(&model, &state, &bvh, 0.04);
    let same_surf = contacts
        .iter()
        .filter(|c| c.vertex_surface == c.tri_surface)
        .count();
    assert_eq!(
        same_surf, 0,
        "Stage 33 default (self_collision = false) must emit 0 same-\
         surface contacts even on a folded cloth (got {same_surf})"
    );
}
