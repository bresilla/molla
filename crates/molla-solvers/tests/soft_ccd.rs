//! Stage 33.4 — VT-CCD root finding + pre-substep CCD pass tests.
//!
//! CPU-only — proves both the math (vertex-triangle TOI on toy
//! geometry) and the integration with VBD (fast-impact tunnelling
//! gated by the velocity threshold).

use molla_core::WorldId;
use molla_geometry::vt_ccd;
use molla_math::{Real, Vec3};
use molla_sim::{Contacts, MaterialNeoHookean, ModelBuilder};
use molla_solvers::{Solver, VbdSolver};

/// Six-tet decomposition of a unit-cell cube — same vertex order as
/// the Stage 33 tests' helper.
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

#[test]
fn vt_ccd_finds_simple_pass_through() {
    // Vertex starts at y=-0.5 and moves to y=+0.5. Triangle is a
    // horizontal sheet at y=0 covering [-1, 1] × [-1, 1]. Static.
    let v0 = Vec3::new(0.0, -0.5, 0.0);
    let v1 = Vec3::new(0.0, 0.5, 0.0);
    let tri0 = [
        Vec3::new(-1.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    let toi = vt_ccd(v0, v1, tri0, tri0).expect("must find a TOI");
    println!("[CCD] simple_pass_through TOI = {toi}");
    assert!((toi - 0.5).abs() < 1.0e-3, "expected TOI ≈ 0.5, got {toi}");
}

#[test]
fn vt_ccd_returns_none_when_no_crossing() {
    // Same geometry as above but vertex moves from y=-0.5 to y=-0.1
    // — never crosses the y=0 plane.
    let v0 = Vec3::new(0.0, -0.5, 0.0);
    let v1 = Vec3::new(0.0, -0.1, 0.0);
    let tri0 = [
        Vec3::new(-1.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    assert!(
        vt_ccd(v0, v1, tri0, tri0).is_none(),
        "vertex never crosses plane — must return None"
    );
}

#[test]
fn vt_ccd_returns_none_when_outside_tri() {
    // Triangle in [0, 1] × [0, 1] on y=0; vertex passes y=0 at
    // x=10 (way outside). The plane crossing is real but the
    // projection lies far outside the triangle, so the CCD must
    // reject the root.
    let v0 = Vec3::new(10.0, -0.5, 0.0);
    let v1 = Vec3::new(10.0, 0.5, 0.0);
    let tri0 = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    assert!(
        vt_ccd(v0, v1, tri0, tri0).is_none(),
        "vertex passes plane far outside tri — must return None"
    );
}

#[test]
fn vt_ccd_returns_none_for_grazing_parallel_motion() {
    // The vertex moves tangentially across the triangle footprint but
    // stays a fixed small distance above the plane. This is a common
    // near-contact case for cloth/soft CCD: the root finder must not
    // turn "almost touching while sliding" into a false impact.
    let eps: Real = 1.0e-3;
    let v0 = Vec3::new(-0.5, eps, 0.0);
    let v1 = Vec3::new(0.5, eps, 0.0);
    let tri0 = [
        Vec3::new(-1.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];

    assert!(
        vt_ccd(v0, v1, tri0, tri0).is_none(),
        "grazing parallel motion above the triangle must not produce a TOI"
    );
}

#[test]
fn vt_ccd_returns_none_when_separating_from_triangle_plane() {
    // The vertex starts above the triangle and moves farther away. A
    // broad-phase sweep may overlap the triangle AABB, but the signed
    // plane distance never crosses zero, so CCD must not snap it back.
    let v0 = Vec3::new(0.0, 0.05, 0.0);
    let v1 = Vec3::new(0.0, 0.35, 0.0);
    let tri0 = [
        Vec3::new(-1.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];

    assert!(
        vt_ccd(v0, v1, tri0, tri0).is_none(),
        "separating motion away from the triangle plane must not produce a TOI"
    );
}

/// Fast-impact tunnelling test: a small soft tet cube launched
/// downward at 5 m/s toward a horizontal pinned cloth strip. With
/// CCD off the cube tunnels (cube min_y goes WAY below cloth_y); with
/// CCD on the cube stays close to the cloth (the per-vertex snap
/// holds the bottom verts at the contact while the tet's elastic
/// network absorbs the residual momentum).
///
/// We use a **soft** tet material so the bottom verts that get
/// snapped to TOI dominate the substep: a stiff cube would have its
/// top verts re-launch the bottom verts via elastic restoration in
/// the same substep (per-vertex CCD doesn't see the top verts move
/// inside the substep). Stage 33.4.1 follow-up could add whole-mesh
/// CCD; v1 ships per-vertex CCD only.
#[test]
fn fast_soft_impact_no_tunnel() {
    /// Build the model. Returns (model, cloth_first, cube_first,
    /// cube_total, cloth_count, cloth_y).
    fn build_model() -> (molla_sim::Model, u32, u32, u32, u32, Real) {
        let mut b = ModelBuilder::new();
        b.set_linear_damping(0.20);

        let cloth_n = 16u32;
        let cloth_w: Real = 0.6;
        let cloth_y: Real = 0.5;
        let (cloth_first, _, _) = b.add_cloth_grid(
            Vec3::new(-cloth_w * 0.5, cloth_y, -cloth_w * 0.5),
            Vec3::new(cloth_w, 0.0, 0.0),
            Vec3::new(0.0, 0.0, cloth_w),
            cloth_n,
            cloth_n,
            0.05,   // particle mass
            0.02,   // particle radius
            5.0e-6, // stretch compliance (stiff)
            0.05,   // max stretch
            WorldId(0),
        );
        // Pin all four edges so the cloth is taut.
        for ix in 0..cloth_n {
            b.pin_particle(cloth_first + ix);
            b.pin_particle(cloth_first + (cloth_n - 1) * cloth_n + ix);
            b.pin_particle(cloth_first + ix * cloth_n);
            b.pin_particle(cloth_first + ix * cloth_n + cloth_n - 1);
        }

        // Soft small tet cube above the cloth, launched downward at
        // 5 m/s.
        let nx = 3u32;
        let dx: Real = 0.05;
        let cube_lo_y: Real = 0.65; // 15 cm above the cloth.
        let cube_origin = Vec3::new(
            -dx * (nx as Real - 1.0) * 0.5,
            cube_lo_y,
            -dx * (nx as Real - 1.0) * 0.5,
        );
        for kk in 0..nx {
            for jj in 0..nx {
                for ii in 0..nx {
                    let p =
                        cube_origin + Vec3::new(ii as Real * dx, jj as Real * dx, kk as Real * dx);
                    b.add_particle(p, 0.05, dx * 0.4, WorldId(0));
                }
            }
        }
        let cloth_count = cloth_n * cloth_n;
        let v_at =
            |i: u32, j: u32, k: u32| -> u32 { cloth_first + cloth_count + (k * nx + j) * nx + i };
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
                    let cell_tets = cube_to_six_tets(c);
                    tets.extend_from_slice(&cell_tets);
                }
            }
        }
        let materials = vec![MaterialNeoHookean::elastic(2.0e5, 0.3)];
        let mat_idx: Vec<u32> = vec![0; tets.len()];
        let _ = b
            .add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
            .expect("tet mesh");

        let model = b.build().unwrap();
        let cube_total = nx * nx * nx;
        let cube_first = cloth_first + cloth_count;
        (
            model,
            cloth_first,
            cube_first,
            cube_total,
            cloth_count,
            cloth_y,
        )
    }

    /// Run the simulation for the given number of substeps with CCD
    /// on/off; returns (final cube_min_y, cloth_min_y, mid_sim
    /// minimum cube_min_y).
    fn run(
        model: &molla_sim::Model,
        ccd: bool,
        cube_first: u32,
        cube_total: u32,
        cloth_first: u32,
        cloth_count: u32,
    ) -> (Real, Real, Real) {
        let mut solver = VbdSolver::new_cpu()
            .with_iterations(8)
            .with_contact_stiffness(1.0e5)
            .with_vt_contact_margin(0.02)
            .with_vt_contact_sweeps(8)
            .with_ccd(ccd)
            .with_ccd_velocity_threshold(0.5);
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(8);

        // Set initial cube velocity to (0, -5, 0).
        {
            let qd = s_a.particle_qd.host_mut().unwrap();
            for i in cube_first..cube_first + cube_total {
                qd[i as usize] = Vec3::new(0.0, -5.0, 0.0);
            }
        }

        let dt: Real = 1.0 / 120.0;
        // 60 substeps = 0.5 s — enough for the cube to either tunnel
        // or stop at the cloth.
        let n_steps = 60;
        let mut min_cube_y_seen = Real::INFINITY;
        for _ in 0..n_steps {
            Solver::step(&mut solver, model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
            let q = s_a.particle_q.host().unwrap();
            let cube_min_y = (cube_first..cube_first + cube_total)
                .map(|i| q[i as usize].y)
                .fold(Real::INFINITY, |a, b| a.min(b));
            if cube_min_y < min_cube_y_seen {
                min_cube_y_seen = cube_min_y;
            }
        }
        let q = s_a.particle_q.host().unwrap();
        let cube_min_y = (cube_first..cube_first + cube_total)
            .map(|i| q[i as usize].y)
            .fold(Real::INFINITY, |a, b| a.min(b));
        let cloth_min_y = (cloth_first..cloth_first + cloth_count)
            .map(|i| q[i as usize].y)
            .fold(Real::INFINITY, |a, b| a.min(b));
        (cube_min_y, cloth_min_y, min_cube_y_seen)
    }

    let (model, cloth_first, cube_first, cube_total, cloth_count, cloth_y) = build_model();

    // Without CCD — fast cube tunnels through the cloth.
    let (no_ccd_cube_y, no_ccd_cloth_y, no_ccd_min) = run(
        &model,
        false,
        cube_first,
        cube_total,
        cloth_first,
        cloth_count,
    );
    println!(
        "[CCD-OFF] cube_min_y={no_ccd_cube_y:.4} cloth_min_y={no_ccd_cloth_y:.4} \
         min_cube_y_seen={no_ccd_min:.4} (cloth_y0={cloth_y})"
    );
    // With CCD — fast cube stops at contact.
    let (ccd_cube_y, ccd_cloth_y, ccd_min) = run(
        &model,
        true,
        cube_first,
        cube_total,
        cloth_first,
        cloth_count,
    );
    println!(
        "[CCD-ON ] cube_min_y={ccd_cube_y:.4} cloth_min_y={ccd_cloth_y:.4} \
         min_cube_y_seen={ccd_min:.4} (cloth_y0={cloth_y})"
    );

    // Strict tunneling check: WITHOUT CCD the cube must end up well
    // below the cloth surface (it punched through). 30 cm below
    // initial cloth height is plenty of margin.
    assert!(
        no_ccd_min < cloth_y - 0.20,
        "without CCD the cube should tunnel; got min cube_y={no_ccd_min}, cloth_y={cloth_y}"
    );

    // WITH CCD the cube must stop at the contact: at no point during
    // the simulation should the cube's lowest vertex have descended
    // below the cloth. We give a tiny tolerance (1 cm) for the snap
    // SNAP_EPS placement + the brief moment when CCD is reacting to
    // an in-progress crossing, but the cube must NOT have tunneled.
    assert!(
        ccd_min > cloth_y - 0.01,
        "with CCD the cube must stop at the cloth (min distance > 0); \
         got min_cube_y_seen={ccd_min}, cloth_y={cloth_y}"
    );
    let _ = ccd_cube_y; // surfaced via println above for diagnostics
    let _ = ccd_cloth_y;
}

/// CCD velocity threshold gate test. A slow-moving vertex (well below
/// the threshold) should NOT trigger any `vt_ccd` invocations — the
/// per-vertex BVH query is skipped entirely.
#[test]
fn ccd_velocity_threshold_skips_slow() {
    // Build a minimal scene: 4×4 cloth + 2×2×2 jelly cube at near-zero
    // velocity. With a 1.0 m/s threshold, the slow cube should never
    // engage CCD even though both surfaces are registered.
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.05);
    let cloth_n = 4u32;
    let cloth_w: Real = 0.4;
    let cloth_y: Real = 0.5;
    let (cloth_first, _, _) = b.add_cloth_grid(
        Vec3::new(-cloth_w * 0.5, cloth_y, -cloth_w * 0.5),
        Vec3::new(cloth_w, 0.0, 0.0),
        Vec3::new(0.0, 0.0, cloth_w),
        cloth_n,
        cloth_n,
        0.05,
        0.02,
        1.0e-5,
        0.05,
        WorldId(0),
    );
    for ix in 0..cloth_n {
        b.pin_particle(cloth_first + ix);
        b.pin_particle(cloth_first + (cloth_n - 1) * cloth_n + ix);
    }
    let nx = 2u32;
    let dx: Real = 0.05;
    // Cube sits JUST above the cloth so a single-substep CCD query
    // immediately sees the swept tri AABBs. The threshold gate is
    // what we're testing — placement is irrelevant otherwise.
    let cube_origin = Vec3::new(
        -dx * (nx as Real - 1.0) * 0.5,
        cloth_y + 0.02,
        -dx * (nx as Real - 1.0) * 0.5,
    );
    for kk in 0..nx {
        for jj in 0..nx {
            for ii in 0..nx {
                let p = cube_origin + Vec3::new(ii as Real * dx, jj as Real * dx, kk as Real * dx);
                b.add_particle(p, 0.05, dx * 0.4, WorldId(0));
            }
        }
    }
    let cloth_count = cloth_n * cloth_n;
    let v_at =
        |i: u32, j: u32, k: u32| -> u32 { cloth_first + cloth_count + (k * nx + j) * nx + i };
    let mut tets = Vec::new();
    let c = [
        v_at(0, 0, 0),
        v_at(1, 0, 0),
        v_at(0, 1, 0),
        v_at(1, 1, 0),
        v_at(0, 0, 1),
        v_at(1, 0, 1),
        v_at(0, 1, 1),
        v_at(1, 1, 1),
    ];
    let cell_tets = cube_to_six_tets(c);
    tets.extend_from_slice(&cell_tets);
    let materials = vec![MaterialNeoHookean::elastic(2.0e5, 0.3)];
    let mat_idx: Vec<u32> = vec![0; tets.len()];
    let _ = b
        .add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
        .expect("tet mesh");
    let model = b.build().unwrap();

    // Solver with CCD on, threshold 1.0 m/s.
    let mut solver = VbdSolver::new_cpu()
        .with_iterations(8)
        .with_contact_stiffness(1.0e5)
        .with_vt_contact_margin(0.02)
        .with_ccd(true)
        .with_ccd_velocity_threshold(1.0);
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(8);

    // Set the cube to a SLOW velocity (0.1 m/s downward) — well below
    // the 1.0 m/s threshold.
    let cube_first = cloth_first + cloth_count;
    let cube_total = nx * nx * nx;
    {
        let qd = s_a.particle_qd.host_mut().unwrap();
        for i in cube_first..cube_first + cube_total {
            qd[i as usize] = Vec3::new(0.0, -0.1, 0.0);
        }
    }

    let dt: Real = 1.0 / 240.0;
    // Run a single substep — CCD invocations counter should remain 0.
    Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
    let inv = solver.last_ccd_invocations();
    let snaps = solver.last_ccd_snaps();
    println!(
        "[CCD-THRESHOLD] slow cube (0.1 m/s @ threshold 1.0): \
         invocations={inv} snaps={snaps}"
    );
    assert_eq!(
        inv, 0,
        "slow cube must skip CCD, got {inv} vt_ccd invocations"
    );
    assert_eq!(snaps, 0, "slow cube must not be snapped, got {snaps}");

    // Now bump the cube to a fast velocity (5 m/s) — this MUST trigger
    // CCD invocations (proves the threshold gate is the only thing
    // suppressing them above).
    let mut s_c = model.state().unwrap();
    let mut s_d = model.state().unwrap();
    {
        let qd = s_c.particle_qd.host_mut().unwrap();
        for i in cube_first..cube_first + cube_total {
            qd[i as usize] = Vec3::new(0.0, -5.0, 0.0);
        }
    }
    Solver::step(&mut solver, &model, &s_c, &control, &contacts, dt, &mut s_d).unwrap();
    let inv_fast = solver.last_ccd_invocations();
    println!("[CCD-THRESHOLD] fast cube (5 m/s): invocations={inv_fast}");
    assert!(
        inv_fast > 0,
        "fast cube must engage CCD, got {inv_fast} invocations"
    );
}
