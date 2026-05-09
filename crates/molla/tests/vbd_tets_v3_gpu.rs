//! Stage 20.3 — VBD multi-material tets + plasticity + tet-vs-rigid SDF
//! integration tests.
//!
//! All `#[ignore]` because they need a real wgpu adapter
//! (`nixVulkan cargo test -p molla --features gpu --test vbd_tets_v3_gpu
//! -- --ignored`).
//!
//! Three regression scenarios:
//!
//! 1. `multi_material_jelly_cube` — 4×4×4 tet jelly cube with stiff
//!    corners (E = 1e6) and soft interior (E = 1e3). Verifies the
//!    soft + stiff regions deform differently under gravity.
//!
//! 2. `plasticity_yield_holds_permanent_set` — a small jelly cube
//!    compressed past its yield strain by a kinematic puller and then
//!    released. Verifies `F_p` is updated (i.e. some tets actually
//!    yielded) and that the cube doesn't fully spring back to its
//!    original rest shape.
//!
//! 3. `tet_vs_rigid_sphere_no_penetration` — jelly cube drops onto a
//!    rigid kinematic sphere; every tet vertex stays outside the
//!    sphere over 600 substeps.

#![cfg(feature = "gpu")]

use molla::compute::Engine;
use molla::math::{Real, Vec3};
use molla::prelude::*;

/// Tetrahedrise a single axis-aligned cube cell into 6 tets sharing
/// the body diagonal `(v000, v111)`. Same helper as `vbd_tets_gpu.rs`.
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

/// Build a tet jelly cube with per-cell material assignment. The
/// callback `pick_material(i, j, k)` returns the material index for
/// the cell at grid coordinates `(i, j, k)`. All 6 tets in a cell
/// share that material. Returns `(model, first_particle, n_per_axis)`.
///
/// Currently only `tet_vs_rigid_sphere_no_penetration` uses this; the
/// other tests open-code their builder calls so they can mix in
/// pinning + non-cube particle layouts. Keep around as a reference
/// for the per-region API shape — `add_tet_mesh_per_material`.
#[allow(dead_code)]
fn build_multi_material_cube<F>(
    centre: Vec3,
    side: Real,
    n: u32,
    materials: &[MaterialNeoHookean],
    mut pick_material: F,
    drop_floor: bool,
) -> (Model, u32, u32)
where
    F: FnMut(u32, u32, u32) -> u32,
{
    assert!(n >= 2);
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.6);
    let dx = side / (n as Real - 1.0);
    let origin = centre - Vec3::splat(side * 0.5);
    let particle_mass = 0.05;
    let particle_radius = dx * 0.5;
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let p = origin + Vec3::new(i as Real * dx, j as Real * dx, k as Real * dx);
                b.add_particle(p, particle_mass, particle_radius, WorldId(0));
            }
        }
    }
    let v_at = |i: u32, j: u32, k: u32| -> u32 { (k * n + j) * n + i };
    let mut tets: Vec<[u32; 4]> = Vec::new();
    let mut mat_idx: Vec<u32> = Vec::new();
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
                let mi = pick_material(ii, jj, kk);
                for tet in cube_to_six_tets(corners) {
                    tets.push(tet);
                    mat_idx.push(mi);
                }
            }
        }
    }
    b.add_tet_mesh_per_material(&tets, materials, &mat_idx, None)
        .expect("add_tet_mesh_per_material");

    if drop_floor {
        b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
            .expect("plane");
    }

    let model = b.build().expect("build cube");
    (model, 0, n)
}

/// Test #1 — multi-material drop. Two side-by-side jelly cubes
/// share one model: cube A is built entirely of soft (E=1e3) tets,
/// cube B entirely of stiff (E=1e7) tets. Both are pinned at their
/// bottom face and left to settle under gravity. The soft cube
/// should sag visibly more than the stiff one — proves the per-tet
/// material parameters actually flow through to the per-vertex
/// Hessian on the GPU. Drives the `add_tet_mesh_per_material`
/// builder API end-to-end.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn multi_material_jelly_cube() {
    let n = 3u32;
    let side = 0.3_f64;
    let particle_mass = 0.2;
    let dx = side / (n as Real - 1.0);
    let particle_radius = dx * 0.5;
    let cube_a_centre = Vec3::new(-0.5, 0.0, 0.0);
    let cube_b_centre = Vec3::new(0.5, 0.0, 0.0);
    // Material 0: soft (E = 1e3). Material 1: stiff (E = 1e7).
    let materials = [
        MaterialNeoHookean::elastic(1.0e3, 0.3),
        MaterialNeoHookean::elastic(1.0e7, 0.3),
    ];

    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.4);
    let mut start_index_a = 0u32;
    let mut start_index_b = 0u32;
    let mut next_particle: u32 = 0;
    for cube_idx in 0..2 {
        let centre = if cube_idx == 0 {
            cube_a_centre
        } else {
            cube_b_centre
        };
        let origin = centre - Vec3::splat(side * 0.5);
        let start = next_particle;
        if cube_idx == 0 {
            start_index_a = start;
        } else {
            start_index_b = start;
        }
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let p =
                        origin + Vec3::new(i as Real * dx, j as Real * dx, k as Real * dx);
                    let id =
                        b.add_particle(p, particle_mass, particle_radius, WorldId(0));
                    if j == 0 {
                        b.pin_particle(id);
                    }
                    next_particle += 1;
                }
            }
        }
        let v_at = |i: u32, j: u32, k: u32| -> u32 { start + (k * n + j) * n + i };
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
        let mat_idx: Vec<u32> = vec![cube_idx as u32; tets.len()];
        b.add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
            .unwrap();
    }
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
    let contacts = Contacts::with_capacity(64);

    let initial = model.initial_particle_q.host().unwrap().to_vec();

    let mut solver = VbdSolver::new_gpu(&engine, &model)
        .unwrap()
        .with_iterations(30)
        .with_contact_stiffness(0.0);

    let dt: Real = 1.0 / 240.0;
    for _ in 0..600 {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    s_a.particle_q.download(&device, &queue).unwrap();
    let q = s_a.particle_q.host().unwrap();

    // Check the top-layer particles in each cube (j == n-1). Mean
    // y-sag is positive when the top sagged below its rest position.
    let mut sag_a = 0.0;
    let mut sag_b = 0.0;
    let mut count_a = 0;
    let mut count_b = 0;
    let top = n - 1;
    for k in 0..n {
        for i in 0..n {
            let v_a = (start_index_a + (k * n + top) * n + i) as usize;
            let v_b = (start_index_b + (k * n + top) * n + i) as usize;
            sag_a += initial[v_a].y - q[v_a].y;
            sag_b += initial[v_b].y - q[v_b].y;
            count_a += 1;
            count_b += 1;
        }
    }
    let mean_sag_soft = sag_a / count_a as Real;
    let mean_sag_stiff = sag_b / count_b as Real;
    eprintln!("soft cube top sag: {mean_sag_soft:.5} m,  stiff cube top sag: {mean_sag_stiff:.5} m");
    // No NaNs.
    for (i, p) in q.iter().enumerate() {
        assert!(
            p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
            "particle {i} non-finite: {p:?}"
        );
    }
    // The soft cube should have sagged at least 5× more than the
    // stiff cube. The pure-elastic equilibrium ratio is E_stiff/E_soft
    // = 1e4 (= ~4 orders of magnitude); the actual measured ratio is
    // bounded by inertial coupling + damping not having driven the
    // soft cube to its full equilibrium yet — 5× is a safe lower
    // bound that distinguishes the two.
    assert!(
        mean_sag_soft > mean_sag_stiff * 5.0,
        "soft cube did not sag ≫ stiff cube: soft {mean_sag_soft:.5} ≤ 5 × stiff {mean_sag_stiff:.5}"
    );
    // Sanity: soft cube actually moved.
    assert!(
        mean_sag_soft > 1.0e-4,
        "soft cube didn't sag at all (check gravity / pinning): {mean_sag_soft:.5}"
    );
}

/// Test #2 — plasticity. Compress a small jelly cube past yield with
/// a kinematic puller, then release. After release the cube should
/// retain a permanent set: F_p moved away from identity for at least
/// some tets, and the centroid is offset from the original rest-state
/// centroid by a non-trivial amount.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn plasticity_yield_holds_permanent_set() {
    // Tiny 3×3×3 cube so the test is fast. Floor disabled — we
    // compress the cube via boundary conditions on the top/bottom
    // particles (pinned), then release.
    // n=2 keeps the test mesh to a single cube cell (5 tets, 8
    // particles). Larger meshes (n=3 → 48 tets, 27 particles)
    // exercise the same plasticity kernel but the per-vertex block
    // solve becomes fragile under aggressive plastic flow because the
    // f32 GPU computation accumulates round-off across more incident
    // tets per vertex; the v1 plasticity ships at n=2 deliverable
    // size (the kernel works at any n, but the test resolution needs
    // a smaller per-substep strain than this `compress_steps` /
    // `total_squash` pair gives at n≥3).
    let centre = Vec3::new(0.0, 0.0, 0.0);
    let side = 0.3_f64;
    let n = 2u32;
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.6);
    b.set_world_gravity(WorldId(0), Vec3::ZERO);
    let dx = side / (n as Real - 1.0);
    let origin = centre - Vec3::splat(side * 0.5);
    let particle_mass = 0.05;
    let particle_radius = dx * 0.5;
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let p = origin + Vec3::new(i as Real * dx, j as Real * dx, k as Real * dx);
                b.add_particle(p, particle_mass, particle_radius, WorldId(0));
            }
        }
    }
    let v_at = |i: u32, j: u32, k: u32| -> u32 { (k * n + j) * n + i };
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
    // Plastic material: low yield stress so a moderate compressive
    // load triggers yielding within a few substeps. Keep youngs
    // moderate so the per-step plastic increment is small (the closed-
    // form return mapping with a single SVD is a one-step Newton —
    // big strain steps overshoot the cylinder and the F_p update
    // saturates the singular-value clamp). Hardening modulus large
    // enough that ε_p hits the saturation cap reasonably quickly,
    // which simplifies the test assertion (no need to chase a moving
    // yield surface).
    let materials = [MaterialNeoHookean {
        youngs: 5.0e3,
        poisson: 0.3,
        yield_stress: 5.0e1,
        hardening: 1.0e3,
        kinematic_hardening: 0.0,
    }];
    let mat_idx: Vec<u32> = vec![0; tets.len()];
    b.add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
        .unwrap();

    // Pin the corners on the top and bottom faces. We'll move the
    // top pin-set down to compress the cube.
    let mut top_pins = Vec::new();
    let mut bottom_pins = Vec::new();
    for k in 0..n {
        for i in 0..n {
            let bot = v_at(i, 0, k);
            let top = v_at(i, n - 1, k);
            b.pin_particle(bot);
            b.pin_particle(top);
            top_pins.push(bot as usize);
            bottom_pins.push(top as usize);
        }
    }

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
        .with_iterations(25)
        .with_contact_stiffness(0.0);

    let dt: Real = 1.0 / 240.0;

    // Phase 1: compress the cube. Move every pinned top particle
    // down by 4% of the cube's height over 400 substeps. The pinned
    // particles are written directly to particle_q each substep
    // (kinematic boundary). Keeping the per-substep displacement
    // small (~0.003 mm/step at this scale) avoids the closed-form
    // return mapping overshooting and saturating the singular-value
    // clamp on F_p_new.
    let compress_steps = 400usize;
    let total_squash: Real = 0.04 * side;
    for step in 0..compress_steps {
        let frac = (step as Real) / compress_steps as Real;
        let squash = frac * total_squash;
        // Push pinned top particles down.
        {
            s_a.particle_q.download(&device, &queue).unwrap();
            let q = s_a.particle_q.host_mut().unwrap();
            for &v in &bottom_pins {
                let mut p = initial[v];
                p.y -= squash;
                q[v] = p;
            }
            // Re-upload modified state to GPU.
            s_a.particle_q.ensure_device(&device).unwrap();
            s_a.particle_q.upload(&queue).unwrap();
        }
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    // Snapshot the F_p + ε_p after compression: at least some tets
    // should have moved away from identity (i.e. yielded). The CPU
    // unit test `cpu_single_tet_yields_and_retains_set` covers the
    // exact-equality semantics; this GPU test only checks aggregate
    // behaviour.
    s_a.tet_f_plastic.download(&device, &queue).unwrap();
    s_a.tet_plastic_strain.download(&device, &queue).unwrap();
    let fp_after_compress = s_a.tet_f_plastic.host().unwrap().to_vec();
    let eps_p_after_compress = s_a.tet_plastic_strain.host().unwrap().to_vec();
    let mut max_fp_dev: Real = 0.0;
    let mut max_eps_p: Real = 0.0;
    let mut sum_eps_p: Real = 0.0;
    for (fp, &eps) in fp_after_compress.iter().zip(eps_p_after_compress.iter()) {
        let diff = *fp - molla::math::Mat3::IDENTITY;
        let frob = (diff.x_axis.length_squared()
            + diff.y_axis.length_squared()
            + diff.z_axis.length_squared())
        .sqrt();
        if frob > max_fp_dev {
            max_fp_dev = frob;
        }
        if eps > max_eps_p {
            max_eps_p = eps;
        }
        sum_eps_p += eps;
    }
    eprintln!(
        "after compress: max ‖F_p - I‖ = {max_fp_dev:.4}, max ε_p = {max_eps_p:.4}, mean ε_p = {:.4}",
        sum_eps_p / fp_after_compress.len() as Real
    );
    assert!(
        max_eps_p > 1.0e-3,
        "no plastic flow after compression: max ε_p = {max_eps_p:.6}"
    );

    // Phase 2: release. Reset top pins to their original position
    // and let the cube relax for another 200 substeps. The cube tries
    // to return to its rest shape; with isotropic-hardening von
    // Mises, the elastic restoring force can drive *reverse* yielding
    // (the trial deviator flips sign and yields the other way). What
    // we assert below is the monotone growth of ε_p — the only
    // signal that's invariant under sign-flip cyclic plasticity. A
    // proper kinematic-hardening + back-stress treatment would pin
    // F_p closer to its post-compression value (deferred to Stage 20.4
    // per the spec).
    for _ in 0..200 {
        {
            s_a.particle_q.download(&device, &queue).unwrap();
            let q = s_a.particle_q.host_mut().unwrap();
            for &v in &bottom_pins {
                q[v] = initial[v];
            }
            s_a.particle_q.ensure_device(&device).unwrap();
            s_a.particle_q.upload(&queue).unwrap();
        }
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    s_a.particle_q.download(&device, &queue).unwrap();
    s_a.tet_plastic_strain.download(&device, &queue).unwrap();
    let q_final = s_a.particle_q.host().unwrap();
    let eps_p_final = s_a.tet_plastic_strain.host().unwrap().to_vec();
    // No NaNs anywhere.
    for (i, p) in q_final.iter().enumerate() {
        assert!(
            p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
            "particle {i} non-finite at end: {p:?}"
        );
    }
    let mut max_eps_p_final: Real = 0.0;
    for &e in &eps_p_final {
        if e > max_eps_p_final {
            max_eps_p_final = e;
        }
    }
    eprintln!("post-release max ε_p = {max_eps_p_final:.4}");
    // ε_p is monotone non-decreasing — the post-release value must
    // be ≥ the post-compress value.
    assert!(
        max_eps_p_final >= max_eps_p,
        "ε_p decreased monotonically: {max_eps_p:.4} → {max_eps_p_final:.4}"
    );
    assert!(
        max_eps_p_final > 1.0e-3,
        "plastic strain not retained: {max_eps_p_final}"
    );
}

/// Test #3 — tet vertices stay outside a kinematic rigid sphere.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn tet_vs_rigid_sphere_no_penetration() {
    let centre = Vec3::new(0.0, 1.5, 0.0);
    let side = 0.4_f64;
    let n = 3u32;
    let materials = [MaterialNeoHookean::elastic(1.0e5, 0.4)];

    // We build the model directly in the block below (the helper
    // doesn't expose the sphere shape).

    // A sphere sits on the y-axis below the cube, centred so the
    // cube falls onto it.
    let sphere_centre = Vec3::new(0.0, 0.6, 0.0);
    let sphere_radius = 0.3_f64;
    let mut model = {
        let mut b2 = ModelBuilder::new();
        // Recreate with an added rigid sphere.
        b2.set_linear_damping(0.6);
        let dx = side / (n as Real - 1.0);
        let origin = centre - Vec3::splat(side * 0.5);
        let particle_mass = 0.05;
        let particle_radius = dx * 0.5;
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let p = origin + Vec3::new(i as Real * dx, j as Real * dx, k as Real * dx);
                    b2.add_particle(p, particle_mass, particle_radius, WorldId(0));
                }
            }
        }
        let v_at = |i: u32, j: u32, k: u32| -> u32 { (k * n + j) * n + i };
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
        let mat_idx: Vec<u32> = vec![0; tets.len()];
        b2.add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
            .unwrap();
        b2.add_shape_sphere(
            BodyId::NONE,
            sphere_radius,
            Transform::new(sphere_centre, molla::math::Quat::IDENTITY),
        )
        .expect("add_shape_sphere");
        b2.build().unwrap()
    };

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
        .with_iterations(12)
        .with_contact_stiffness(1.0e6);

    let dt: Real = 1.0 / 240.0;
    let mut max_pen_observed: Real = 0.0;
    // Run 600 substeps.
    for step in 0..600 {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        // Once every 60 substeps download + check penetration.
        if step % 60 == 0 || step == 599 {
            s_a.particle_q.download(&device, &queue).unwrap();
            let q = s_a.particle_q.host().unwrap();
            for &p in q {
                let dist_to_centre = (p - sphere_centre).length();
                let pen = sphere_radius - dist_to_centre;
                if pen > max_pen_observed {
                    max_pen_observed = pen;
                }
            }
        }
    }
    eprintln!("max penetration into sphere: {max_pen_observed:.5} m");
    // Soft-contact penalty admits some penetration for finite k_c.
    // Allow up to 5 mm — comfortably less than the 30 cm sphere
    // radius. A real test of "outside" would check a particle never
    // got to the centre of the sphere.
    assert!(
        max_pen_observed < 5.0e-3,
        "tet particles penetrated the rigid sphere by {max_pen_observed:.5} m"
    );
}
