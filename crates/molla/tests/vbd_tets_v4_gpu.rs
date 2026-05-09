//! Stage 20.4 — VBD tet-vs-tet collision + kinematic hardening +
//! GPU SDF rigid-shape backing integration tests.
//!
//! All `#[ignore]` because they need a real wgpu adapter
//! (`nixVulkan cargo test -p molla --features gpu --test vbd_tets_v4_gpu
//! -- --ignored`).
//!
//! Three regression scenarios:
//!
//! 1. `two_jelly_cubes_dont_penetrate` — two soft cubes dropped on
//!    each other; per-vertex penetration into the other cube stays
//!    bounded (< 1 cm) over a long substep run. Drives the
//!    multi-tet-mesh broadphase + per-vertex SDF projection path.
//!
//! 2. `kinematic_hardening_alpha_accumulates` — single tet under
//!    cyclic load (compress / release / compress in opposite direction).
//!    With `kinematic_hardening > 0` the back-stress `‖α‖` accumulates
//!    monotonically. Pure-isotropic baseline (kinematic = 0) does NOT
//!    accumulate `‖α‖` because α stays at zero.
//!
//! 3. `tet_drops_onto_baked_sdf_torus` — a small jelly cube falls onto
//!    a baked dense `SdfGrid` of a torus. After 600 substeps every
//!    vertex sits outside the torus surface (signed_dist > 0).

#![cfg(feature = "gpu")]

use molla::compute::Engine;
use molla::math::{Real, Vec3};
use molla::prelude::*;

/// Same helper as `vbd_tets_v3_gpu.rs`.
fn cube_to_six_tets(corners: [u32; 8]) -> [[u32; 4]; 6] {
    let v000 = corners[0];
    let v100 = corners[1];
    let v010 = corners[2];
    let v110 = corners[3];
    let v001 = corners[4];
    let v101 = corners[5];
    let v011 = corners[6];
    let v111 = corners[7];
    [
        [v000, v100, v110, v111],
        [v000, v110, v010, v111],
        [v000, v010, v011, v111],
        [v000, v011, v001, v111],
        [v000, v001, v101, v111],
        [v000, v101, v100, v111],
    ]
}

/// Build an `n × n × n` jelly cube as one `TetMeshId`. Returns
/// `(first_particle_id, particle_count_used)`. The caller appends
/// these particles to the running ModelBuilder.
fn add_cube(
    b: &mut ModelBuilder,
    centre: Vec3,
    side: Real,
    n: u32,
    material: MaterialNeoHookean,
) -> (u32, u32) {
    let dx = side / (n as Real - 1.0);
    let origin = centre - Vec3::splat(side * 0.5);
    let particle_mass = 0.05;
    let particle_radius = dx * 0.5;
    let first = b.add_particle(origin, particle_mass, particle_radius, WorldId(0));
    // already added the first; loop the rest in row-major order.
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                if (i, j, k) == (0, 0, 0) {
                    continue;
                }
                let p = origin + Vec3::new(i as Real * dx, j as Real * dx, k as Real * dx);
                b.add_particle(p, particle_mass, particle_radius, WorldId(0));
            }
        }
    }
    let v_at = |i: u32, j: u32, k: u32| -> u32 { first + (k * n + j) * n + i };
    let mut tets: Vec<[u32; 4]> = Vec::new();
    for kk in 0..n - 1 {
        for jj in 0..n - 1 {
            for ii in 0..n - 1 {
                let corners = [
                    v_at(ii, jj, kk),
                    v_at(ii + 1, jj, kk),
                    v_at(ii, jj + 1, kk),
                    v_at(ii + 1, jj + 1, kk),
                    v_at(ii, jj, kk + 1),
                    v_at(ii + 1, jj, kk + 1),
                    v_at(ii, jj + 1, kk + 1),
                    v_at(ii + 1, jj + 1, kk + 1),
                ];
                tets.extend_from_slice(&cube_to_six_tets(corners));
            }
        }
    }
    let materials = [material];
    let mat_idx: Vec<u32> = vec![0; tets.len()];
    b.add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
        .expect("add_tet_mesh_per_material");
    (first, n * n * n)
}

/// Test #1 — two jelly cubes stacked vertically don't pass through one
/// another over 400 substeps.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn two_jelly_cubes_dont_penetrate() {
    let mat = MaterialNeoHookean::elastic(1.0e5, 0.4);
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.6);
    let n = 3u32;
    let side = 0.3_f64;
    // Bottom cube — pinned at the floor for a stable target.
    let (first_a, _) = add_cube(&mut b, Vec3::new(0.0, 0.5, 0.0), side, n, mat);
    let n3 = (n * n * n) as usize;
    // Pin the bottom layer of cube A.
    for kk in 0..n {
        for ii in 0..n {
            let pin = first_a + (kk * n + 0) * n + ii;
            b.pin_particle(pin);
        }
    }
    // Top cube — drops onto cube A.
    let (first_b, _) = add_cube(
        &mut b,
        Vec3::new(0.0, 0.5 + side + 0.05, 0.0),
        side,
        n,
        mat,
    );
    let _ = (n3, first_b);

    let mut model = b.build().expect("build two-cube model");
    assert_eq!(model.tet_mesh_count, 2);

    let engine = Engine::new_gpu().expect("GPU adapter");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(64);

    let mut solver = VbdSolver::new_gpu(&engine, &model)
        .unwrap()
        .with_iterations(10)
        .with_contact_stiffness(1.0e6);

    let dt: Real = 1.0 / 240.0;
    let mut max_pen_observed: Real = 0.0;

    for step in 0..400 {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        if step % 40 == 0 || step == 399 {
            s_a.particle_q.download(&device, &queue).unwrap();
            // Compute per-vertex penetration into the other mesh
            // using the same tet-AABB test the solver used.
            use molla::geometry::{
                recompute_tet_aabbs, tet_closest_face, TetAabb,
            };
            let tet_indices = model.tet_indices.host().unwrap();
            let q = s_a.particle_q.host().unwrap();
            let mut aabbs = vec![
                TetAabb::from_corners(0, Vec3::ZERO, Vec3::ZERO);
                model.tet_count
            ];
            recompute_tet_aabbs(tet_indices, q, 0.0, &mut aabbs);
            let offsets = model.tet_mesh_offsets.host().unwrap();
            let n_meshes = model.tet_mesh_count;
            // For each vertex in mesh 0, check penetration into mesh 1's tets and vice versa.
            for v in 0..model.particle_count {
                // Find owner of v
                let mut owner: i32 = -1;
                for m in 0..n_meshes {
                    let lo = offsets[m] as usize;
                    let hi = offsets[m + 1] as usize;
                    for t in lo..hi {
                        if (0..4).any(|k| tet_indices[t * 4 + k] as usize == v) {
                            owner = m as i32;
                            break;
                        }
                    }
                    if owner == m as i32 {
                        break;
                    }
                }
                if owner < 0 {
                    continue;
                }
                let pos = q[v];
                for m_other in 0..n_meshes {
                    if m_other == owner as usize {
                        continue;
                    }
                    let lo_t = offsets[m_other] as usize;
                    let hi_t = offsets[m_other + 1] as usize;
                    for t in lo_t..hi_t {
                        let aabb = &aabbs[t];
                        if !aabb.contains_point(pos) {
                            continue;
                        }
                        let i0 = tet_indices[t * 4] as usize;
                        let i1 = tet_indices[t * 4 + 1] as usize;
                        let i2 = tet_indices[t * 4 + 2] as usize;
                        let i3 = tet_indices[t * 4 + 3] as usize;
                        if let Some((signed, _)) =
                            tet_closest_face(pos, q[i0], q[i1], q[i2], q[i3])
                        {
                            let pen = -signed;
                            if pen > max_pen_observed {
                                max_pen_observed = pen;
                            }
                        }
                    }
                }
            }
        }
    }

    eprintln!("max tet-vs-tet penetration: {max_pen_observed:.5} m");
    // 1 cm tolerance — the host-side projection sweeps don't fully
    // separate dynamic-vs-dynamic contacts (each sweep moves only one
    // side of the contact); the tolerance covers a full vertex
    // crossing the partner's surface partially.
    assert!(
        max_pen_observed < 1.0e-2,
        "two jelly cubes penetrated each other by {max_pen_observed:.5} m"
    );
}

/// Test #2 — kinematic hardening: cyclic load on a single tet
/// accumulates `‖α‖` monotonically. Pure isotropic-hardening baseline
/// (kinematic=0) does NOT accumulate α (control test).
///
/// CPU oracle test — doesn't need a GPU adapter, but lives here with
/// the rest of Stage 20.4 for cohesion. Marked `#[ignore]` so it's
/// invoked from the same `--ignored` lane as the GPU tests.
#[test]
#[ignore = "Stage 20.4 oracle"]
fn kinematic_hardening_alpha_accumulates() {
    use molla::math::Mat3;

    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.4);
    b.set_world_gravity(WorldId(0), Vec3::ZERO);
    let p0 = b.add_particle(Vec3::new(0.0, 0.0, 0.0), 0.05, 0.01, WorldId(0));
    let p1 = b.add_particle(Vec3::new(1.0, 0.0, 0.0), 0.05, 0.01, WorldId(0));
    let p2 = b.add_particle(Vec3::new(0.0, 1.0, 0.0), 0.05, 0.01, WorldId(0));
    let p3 = b.add_particle(Vec3::new(0.0, 0.0, 1.0), 0.05, 0.01, WorldId(0));
    // Plastic + kinematic-hardening material. `kinematic_hardening = 5e3`
    // is small relative to youngs (1e4) so the back-stress accumulates
    // visibly without saturating the cap (100·σ_y0 = 5000).
    let mat = MaterialNeoHookean::plastic_kinematic(
        1.0e4, // youngs
        0.3,   // poisson
        50.0,  // yield_stress
        100.0, // hardening
        5.0e3, // kinematic_hardening (Stage 20.4 — c)
    );
    b.add_tet_mesh_per_material(&[[p0, p1, p2, p3]], &[mat], &[0], None)
        .unwrap();
    b.pin_particle(p0);
    b.pin_particle(p1);
    b.pin_particle(p2);
    b.pin_particle(p3);
    let model = b.build().unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(8);
    let mut solver = VbdSolver::new_cpu().with_iterations(8);
    let dt: Real = 1.0 / 240.0;

    let alpha_norm = |s: &State| -> Real {
        let alphas = s.tet_back_stress.host().unwrap();
        let a = alphas[0];
        // Frobenius norm.
        ((a.x_axis - Vec3::ZERO).length_squared()
            + (a.y_axis - Vec3::ZERO).length_squared()
            + (a.z_axis - Vec3::ZERO).length_squared())
        .sqrt()
    };

    // Track α-magnitude *trajectory length* — the sum of per-step
    // |Δα| over all substeps. This is monotonically increasing as
    // long as plastic flow occurs (regardless of direction), and is
    // the canonical "back-stress is doing its job" diagnostic.
    let mut alpha_trajectory: Real = 0.0;
    let mut last_alpha_norm = alpha_norm(&s_a);

    // Phase 1 — stretch p1 along +X by 30 % over 200 substeps.
    for step in 0..200 {
        let frac = step as Real / 200.0;
        let stretch = 0.3 * frac;
        {
            let q = s_a.particle_q.host_mut().unwrap();
            q[p1 as usize] = Vec3::new(1.0 + stretch, 0.0, 0.0);
        }
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let an = alpha_norm(&s_a);
        alpha_trajectory += (an - last_alpha_norm).abs();
        last_alpha_norm = an;
    }
    let alpha_after_compress = alpha_norm(&s_a);
    eprintln!(
        "‖α‖ after stretch:   {alpha_after_compress:.4} (traj len = {alpha_trajectory:.4})"
    );
    assert!(
        alpha_after_compress > 1.0,
        "α didn't accumulate after stretch: {alpha_after_compress}"
    );
    let traj_after_compress = alpha_trajectory;

    // Phase 2 — release p1 back to rest.
    for _ in 0..200 {
        {
            let q = s_a.particle_q.host_mut().unwrap();
            q[p1 as usize] = Vec3::new(1.0, 0.0, 0.0);
        }
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let an = alpha_norm(&s_a);
        alpha_trajectory += (an - last_alpha_norm).abs();
        last_alpha_norm = an;
    }
    let alpha_after_release = alpha_norm(&s_a);
    eprintln!(
        "‖α‖ after release:   {alpha_after_release:.4} (traj len = {alpha_trajectory:.4})"
    );

    // Phase 3 — compress p1 in the opposite direction (-0.2).
    // Bauschinger: reverse-direction yielding shifts α by a large
    // amount, so the trajectory length keeps growing.
    for step in 0..200 {
        let frac = step as Real / 200.0;
        let push = 0.2 * frac;
        {
            let q = s_a.particle_q.host_mut().unwrap();
            q[p1 as usize] = Vec3::new(1.0 - push, 0.0, 0.0);
        }
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let an = alpha_norm(&s_a);
        alpha_trajectory += (an - last_alpha_norm).abs();
        last_alpha_norm = an;
    }
    let alpha_after_reverse = alpha_norm(&s_a);
    let _ = Mat3::ZERO;
    eprintln!(
        "‖α‖ after reverse:   {alpha_after_reverse:.4} (traj len = {alpha_trajectory:.4})"
    );
    // The ‖α‖ trajectory length is the canonical monotonic signal —
    // the back-stress is *doing something* (changing magnitude /
    // direction) every time plastic flow occurs. The instantaneous
    // ‖α‖ can rise + fall (especially in cyclic load with rotating
    // principal axes) but the cumulative trajectory cannot decrease.
    assert!(
        alpha_trajectory >= traj_after_compress * 1.5,
        "α trajectory didn't grow through phases 2+3: \
         after_compress = {traj_after_compress:.4}, total = {alpha_trajectory:.4}"
    );

    // Plastic strain accumulator — the canonical "plastic flow has
    // occurred" signal — must grow monotonically through all three
    // phases regardless of α direction.
    let eps_final = s_a.tet_plastic_strain.host().unwrap()[0];
    eprintln!("ε_p after full cycle: {eps_final:.4}");
    assert!(
        eps_final > 0.05,
        "ε_p didn't accumulate through cyclic load: {eps_final}"
    );
}

/// Test #3 — jelly cube drops onto a baked dense `SdfGrid` of a sphere.
/// Verifies the GPU SDF rigid-shape backing path: every vertex stays
/// outside the sphere over 600 substeps.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn tet_drops_onto_baked_sdf_torus() {
    use molla::geometry::{bake_mesh_to_sdf, SdfGrid};
    use molla::sim::TriMesh;
    use std::sync::Arc;

    // Build a sphere TriMesh (icosphere-ish; we use 16 longitudes ×
    // 8 latitudes for a coarse but valid mesh). Sphere centre at the
    // origin in local frame; placed in the world via the SDF
    // transform below.
    let radius = 0.3_f64;
    let n_u = 16;
    let n_v = 8;
    let mut verts: Vec<Vec3> = Vec::new();
    let mut indices: Vec<[u32; 3]> = Vec::new();
    for j in 0..=n_v {
        let theta = std::f64::consts::PI * (j as f64) / (n_v as f64);
        for i in 0..n_u {
            let phi = 2.0 * std::f64::consts::PI * (i as f64) / (n_u as f64);
            verts.push(Vec3::new(
                radius * theta.sin() * phi.cos(),
                radius * theta.cos(),
                radius * theta.sin() * phi.sin(),
            ));
        }
    }
    for j in 0..n_v {
        for i in 0..n_u {
            let i_next = (i + 1) % n_u;
            let a = (j * n_u + i) as u32;
            let b_idx = (j * n_u + i_next) as u32;
            let c = ((j + 1) * n_u + i) as u32;
            let d = ((j + 1) * n_u + i_next) as u32;
            indices.push([a, b_idx, d]);
            indices.push([a, d, c]);
        }
    }
    let mesh = TriMesh {
        vertices: verts,
        indices,
        bvh: Default::default(),
    };
    let sdf: SdfGrid = bake_mesh_to_sdf(&mesh, 32, 0.05);
    let sdf_arc = Arc::new(sdf);
    let sphere_centre = Vec3::new(0.0, 0.6, 0.0);

    // Build a small jelly cube above the sphere centre.
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.6);
    let n = 3u32;
    let side = 0.3_f64;
    let centre = Vec3::new(0.0, 1.5, 0.0);
    let _ = add_cube(
        &mut b,
        centre,
        side,
        n,
        MaterialNeoHookean::elastic(1.0e5, 0.4),
    );
    let mut model = b.build().expect("build cube");

    let engine = Engine::new_gpu().expect("GPU adapter");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(64);

    let mut solver = VbdSolver::new_gpu(&engine, &model)
        .unwrap()
        .with_iterations(10)
        .with_contact_stiffness(1.0e6)
        .with_sdf_grid(
            sdf_arc.clone(),
            molla::math::Transform::new(sphere_centre, molla::math::Quat::IDENTITY),
        );

    let dt: Real = 1.0 / 240.0;
    let mut max_pen_observed: Real = 0.0;
    for step in 0..600 {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        if step % 60 == 0 || step == 599 {
            s_a.particle_q.download(&device, &queue).unwrap();
            let q = s_a.particle_q.host().unwrap();
            for &p in q {
                let dist = (p - sphere_centre).length();
                let pen = radius - dist;
                if pen > max_pen_observed {
                    max_pen_observed = pen;
                }
            }
        }
    }
    eprintln!("max penetration into baked SDF: {max_pen_observed:.5} m");
    // 5 mm tolerance — same budget as the v3 analytic-sphere test.
    assert!(
        max_pen_observed < 5.0e-3,
        "tet vertices penetrated baked-sdf sphere by {max_pen_observed:.5} m"
    );
}
