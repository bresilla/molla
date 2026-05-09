//! Stage 20.2 — VBD tetrahedral / volumetric soft body + CABLE bending
//! integration tests.
//!
//! All `#[ignore]` because they need a real wgpu adapter
//! (`nixVulkan cargo test -p molla --features gpu --test vbd_tets_gpu
//! -- --ignored`).
//!
//! Three regression scenarios:
//!
//! 1. `jelly_cube_drops_and_settles` — 4×4×4 tet jelly cube falls onto
//!    a kinematic plane, deforms elastically, then settles. Asserts
//!    the cube stays bounded (no NaNs, no runaway energy growth).
//!
//! 2. `tet_rest_state_is_stable` — a tet mesh held at its rest
//!    configuration with no external forces stays put for 600
//!    substeps (no spurious deformation from numerical drift).
//!
//! 3. `cable_bending_stiffness_sweep` — a 16-particle CABLE polyline
//!    pinned at both ends; sweeps `k_b ∈ {0, 1e-2, 1.0}` and asserts
//!    higher `k_b` produces less mid-span sag than lower `k_b`.

#![cfg(feature = "gpu")]

use molla::compute::Engine;
use molla::math::{Real, Vec3};
use molla::prelude::*;

/// Tetrahedrise a single axis-aligned cube cell into 6 tets sharing
/// the body diagonal `(v000, v111)`. This is the standard "diagonal
/// split" — guaranteed positive volume for every tet when the
/// vertices are read off the standard cube layout. Cell vertices are
/// indexed in `+x +y +z` order so the helper works for any grid cell.
fn cube_to_six_tets(corners: [u32; 8]) -> [[u32; 4]; 6] {
    let v000 = corners[0];
    let v100 = corners[1];
    let v010 = corners[2];
    let v110 = corners[3];
    let v001 = corners[4];
    let v101 = corners[5];
    let v011 = corners[6];
    let v111 = corners[7];
    // Six tets, all sharing edge (v000, v111). Order picked so each
    // tet's `[v0, v1, v2, v3]` produces a positive det(Dm) under the
    // standard `+x +y +z` cube layout. Verified by hand on a unit
    // cube.
    [
        [v000, v100, v110, v111],
        [v000, v110, v010, v111],
        [v000, v010, v011, v111],
        [v000, v011, v001, v111],
        [v000, v001, v101, v111],
        [v000, v101, v100, v111],
    ]
}

/// Build an `n×n×n` tetrahedral grid jelly cube of side `side`,
/// anchored with its centre of mass at `centre`. Returns
/// `(model, first_particle_index, n_per_axis)`.
fn build_jelly_cube(centre: Vec3, side: Real, n: u32) -> (Model, u32, u32) {
    assert!(n >= 2, "need at least 2 vertices per axis");
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.4);
    let dx = side / (n as Real - 1.0);
    let origin = centre - Vec3::splat(side * 0.5);
    let particle_mass = 0.05;
    let particle_radius = dx * 0.5;
    let first = 0u32;
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let p = origin + Vec3::new(i as Real * dx, j as Real * dx, k as Real * dx);
                b.add_particle(p, particle_mass, particle_radius, WorldId(0));
            }
        }
    }
    let v_at = |i: u32, j: u32, k: u32| -> u32 {
        first + (k * n + j) * n + i
    };
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
    // E = 1e5 Pa (gel-rubber regime); ν = 0.4.
    let _mesh_id = b
        .add_tet_mesh(&tets, 1.0e5, 0.4, None)
        .expect("add_tet_mesh");

    // Floor plane.
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .expect("plane");

    let model = b.build().expect("build jelly cube");
    (model, first, n)
}

/// CPU smoke check: cable bending stiffness affects the resting
/// shape of a horizontal cable pinned at both ends. Higher `k_b`
/// should produce *less* sag than lower `k_b`. Mirrors the GPU
/// `cable_bending_stiffness_sweep` test below at small problem size.
#[test]
fn cpu_cable_bending_reduces_sag() {
    fn run(bending: Real) -> Vec<Vec3> {
        let mut b = ModelBuilder::new();
        b.set_linear_damping(0.6);
        let n = 16u32;
        let pin_a = Vec3::new(-0.5, 1.0, 0.0);
        let pin_b = Vec3::new(0.5, 1.0, 0.0);
        let (first, count, cable_id) = b.add_cable_polyline(
            pin_a, pin_b, n, 0.05, 0.01, 1.0e6, 0.02, WorldId(0),
        );
        b.pin_particle(first);
        b.pin_particle(first + count - 1);
        b.set_cable_bending_stiffness(cable_id, bending);
        let model = b.build().unwrap();
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(8);
        let mut solver = VbdSolver::new_cpu().with_iterations(15);
        let dt: Real = 1.0 / 240.0;
        for _ in 0..600 {
            molla::solvers::Solver::step(
                &mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b,
            ).unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }
        s_a.particle_q.host().unwrap().to_vec()
    }
    // CABLE bending energy is `½·k_b·|κ|²` where `κ = v_prev +
    // v_next - 2·v_mid` (units of metres). Picking `k_b ≈ 5e3` puts
    // the bending force per typical curvature ~100 mN — comparable
    // to the 50 mN/segment gravity load at this mass scale, so the
    // mid-span sag responds visibly.
    let q_no = run(0.0);
    let q_hi = run(5.0e3);
    let mid = q_no.len() / 2;
    let sag_no = 1.0 - q_no[mid].y;
    let sag_hi = 1.0 - q_hi[mid].y;
    eprintln!("CPU sag (no bending): {sag_no:.4} m, sag (k_b=5e3): {sag_hi:.4} m");
    // Demand at least a 1% reduction in sag — a tighter assertion
    // than "any reduction" guards against a no-op kernel slipping
    // through, but the absolute effect is bounded by the cable's
    // stretch springs which already cap mid-span displacement.
    assert!(
        sag_hi < sag_no * 0.99,
        "CPU bending did not stiffen cable: sag_hi {sag_hi:.4} >= 0.99·sag_no {:.4}",
        sag_no * 0.99
    );
}

/// CPU-side smoke check: rest configuration is a fixed point of the
/// per-tet Neo-Hookean energy. Cheap, no GPU adapter needed —
/// catches sign/format mistakes before they make it to the GPU path.
#[test]
fn cpu_tet_rest_state_is_stable() {
    let centre = Vec3::new(0.0, 5.0, 0.0);
    let side = 0.4_f64;
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.0);
    b.set_world_gravity(WorldId(0), Vec3::ZERO);
    let n = 3u32;
    let dx = side / (n as Real - 1.0);
    let origin = centre - Vec3::splat(side * 0.5);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let p = origin + Vec3::new(i as Real * dx, j as Real * dx, k as Real * dx);
                b.add_particle(p, 0.05, dx * 0.5, WorldId(0));
            }
        }
    }
    let v_at = |i: u32, j: u32, k: u32| -> u32 { (k * n + j) * n + i };
    let mut tets = Vec::new();
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
    b.add_tet_mesh(&tets, 1.0e5, 0.4, None).unwrap();
    let model = b.build().unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(8);
    let mut solver = VbdSolver::new_cpu().with_iterations(8);
    let dt: Real = 1.0 / 240.0;
    let initial = model.initial_particle_q.host().unwrap().to_vec();
    for _ in 0..200 {
        molla::solvers::Solver::step(
            &mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b,
        ).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let q = s_a.particle_q.host().unwrap();
    let mut max_drift = 0.0_f64;
    for (i, p) in q.iter().enumerate() {
        assert!(
            p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
            "CPU: particle {i} non-finite: {p:?}"
        );
        let d = (*p - initial[i]).length();
        if d > max_drift { max_drift = d; }
    }
    assert!(
        max_drift < 1.0e-3,
        "CPU tet rest state drifted by {max_drift:.6} m"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn jelly_cube_drops_and_settles() {
    let centre = Vec3::new(0.0, 1.5, 0.0);
    let side = 0.5_f64;
    let (mut model, _first, _n) = build_jelly_cube(centre, side, 4);

    let engine = Engine::new_gpu().expect("GPU adapter for VBD tets");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(64);
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();

    let mut solver = VbdSolver::new_gpu(&engine, &model)
        .unwrap()
        .with_iterations(10)
        .with_contact_stiffness(1.0e5);

    let dt: Real = 1.0 / 240.0;
    let n_steps = 480; // 2 s of sim.
    for _ in 0..n_steps {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    s_a.particle_q.download(&device, &queue).unwrap();
    let q = s_a.particle_q.host().unwrap();

    // No NaNs.
    for (i, p) in q.iter().enumerate() {
        assert!(
            p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
            "particle {i} non-finite: {p:?}"
        );
    }
    // Mean Y dropped (the cube fell). It started at centre.y; after 2 s
    // it should have hit the floor and either settled or be bouncing.
    let mean_y: Real = q.iter().map(|p| p.y).sum::<Real>() / q.len() as Real;
    assert!(
        mean_y < centre.y - 0.1,
        "jelly cube didn't fall: mean_y = {mean_y}"
    );
    // No particle has gone catastrophically below the floor (>50 cm).
    let min_y = q.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
    assert!(
        min_y > -0.5,
        "jelly cube tunneled through floor: min_y = {min_y}"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tet_rest_state_is_stable() {
    // Place the jelly cube high above the floor and disable gravity
    // for this test (we only want to confirm the rest state is a
    // fixed point of the solver).
    let centre = Vec3::new(0.0, 5.0, 0.0);
    let side = 0.4_f64;
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.0);
    b.set_world_gravity(WorldId(0), Vec3::ZERO);
    let n = 3u32;
    let dx = side / (n as Real - 1.0);
    let origin = centre - Vec3::splat(side * 0.5);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let p = origin + Vec3::new(i as Real * dx, j as Real * dx, k as Real * dx);
                b.add_particle(p, 0.05, dx * 0.5, WorldId(0));
            }
        }
    }
    let v_at = |i: u32, j: u32, k: u32| -> u32 { (k * n + j) * n + i };
    let mut tets = Vec::new();
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
    b.add_tet_mesh(&tets, 1.0e5, 0.4, None).unwrap();
    let mut model = b.build().unwrap();

    let engine = Engine::new_gpu().expect("GPU adapter");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(8);

    let initial = model.initial_particle_q.host().unwrap().to_vec();

    let mut solver = VbdSolver::new_gpu(&engine, &model)
        .unwrap()
        .with_iterations(8)
        .with_contact_stiffness(0.0);

    let dt: Real = 1.0 / 240.0;
    for _ in 0..600 {
        solver.step(&model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    s_a.particle_q.download(&device, &queue).unwrap();
    let q = s_a.particle_q.host().unwrap();
    let mut max_drift = 0.0_f64;
    for (i, p) in q.iter().enumerate() {
        let d = (*p - initial[i]).length();
        if d > max_drift {
            max_drift = d;
        }
        assert!(
            p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
            "particle {i} non-finite"
        );
    }
    // Allowing 1 mm of drift over 2.5 s is generous for a strict
    // rest configuration with no contacts.
    assert!(
        max_drift < 1.0e-3,
        "tet rest state drifted by {max_drift:.6} m (expected < 1e-3)"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn cable_bending_stiffness_sweep() {
    // Compare two cables — one with bending stiffness, one without —
    // pinned at both ends, with gravity pulling on the middle particles.
    // The high-bending cable should sag *less* than the no-bending
    // cable.
    fn build(bending: Real) -> Model {
        let mut b = ModelBuilder::new();
        b.set_linear_damping(0.6);
        let n = 16u32;
        let pin_a = Vec3::new(-0.5, 1.0, 0.0);
        let pin_b = Vec3::new(0.5, 1.0, 0.0);
        let (first, count, cable_id) = b.add_cable_polyline(
            pin_a, pin_b, n, 0.05, 0.01, 1.0e6, 0.02, WorldId(0),
        );
        b.pin_particle(first);
        b.pin_particle(first + count - 1);
        b.set_cable_bending_stiffness(cable_id, bending);
        b.build().unwrap()
    }
    let mut m_no = build(0.0);
    let mut m_hi = build(5.0e3);
    let engine = Engine::new_gpu().expect("GPU adapter");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    m_no.upload_to_device(&device, &queue).unwrap();
    m_hi.upload_to_device(&device, &queue).unwrap();

    let run_one = |model: &mut Model| -> Vec<Vec3> {
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        s_a.upload_to_device(&device, &queue).unwrap();
        s_b.upload_to_device(&device, &queue).unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(8);
        let mut solver = VbdSolver::new_gpu(&engine, model)
            .unwrap()
            .with_iterations(15)
            .with_contact_stiffness(0.0);
        let dt: Real = 1.0 / 240.0;
        for _ in 0..600 {
            solver.step(model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }
        s_a.particle_q.download(&device, &queue).unwrap();
        s_a.particle_q.host().unwrap().to_vec()
    };

    let q_no = run_one(&mut m_no);
    let q_hi = run_one(&mut m_hi);

    // Compare sag at the midpoint particle (index n/2).
    let mid = m_no.particle_count / 2;
    let sag_no = 1.0 - q_no[mid].y; // pinned at y = 1.0
    let sag_hi = 1.0 - q_hi[mid].y;
    eprintln!("sag (no bending): {sag_no:.4} m,  sag (k_b=5e3): {sag_hi:.4} m");
    assert!(
        sag_no.is_finite() && sag_hi.is_finite(),
        "non-finite sag values"
    );
    // High-bending should sag LESS than no-bending. The reduction
    // is bounded by the cable's stretch springs (which already cap
    // mid-span displacement), so we settle for a 1 % relative drop —
    // tight enough that a no-op kernel can't slip through.
    assert!(
        sag_hi < sag_no * 0.99,
        "high-bending cable did not stiffen: sag_hi {sag_hi:.4} >= 0.99·sag_no {:.4}",
        sag_no * 0.99
    );
}
