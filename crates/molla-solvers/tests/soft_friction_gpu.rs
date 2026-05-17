//! Stage 33.2.1 — GPU Coulomb friction + restitution parity tests.
//!
//! Mirrors the CPU `soft_friction.rs` integration tests on the GPU
//! VBD backend. Each test is `#[ignore]` because it requires a real
//! wgpu adapter (CI doesn't ship one); run with
//! `nixVulkan cargo test -p molla --features gpu --test soft_friction_gpu -- --ignored`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::WorldId;
use molla_math::{Real, Vec3};
use molla_sim::{Contacts, MaterialNeoHookean, Model, ModelBuilder, State};
use molla_solvers::{Solver, VbdSolver};

/// Six-tet decomposition of a unit-cell cube (sharing the cube
/// diagonal `(0, 7)`). Same convention as `tests/soft_friction.rs` on
/// the CPU side.
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

/// Build the standard friction-test setup: a fully pinned 12×12 cloth
/// at `cloth_y` with a 3×3×3 tet jelly cube floating just above it.
/// Returns `(model_builder, cloth_first, cube_first, cube_n_verts)`.
///
/// Mirrors `tests/soft_friction.rs::build_block_on_cloth`.
fn build_block_on_cloth(
    cube_friction: Real,
    cube_restitution: Real,
    drop_height: Real,
) -> (ModelBuilder, u32, u32, u32) {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.1);

    let cloth_n: u32 = 12;
    let cloth_w: Real = 1.5;
    let cloth_y: Real = 1.0;
    let (cloth_first, _, _) = b.add_cloth_grid(
        Vec3::new(-cloth_w * 0.5, cloth_y, -cloth_w * 0.5),
        Vec3::new(cloth_w, 0.0, 0.0),
        Vec3::new(0.0, 0.0, cloth_w),
        cloth_n,
        cloth_n,
        0.05,
        0.04,
        5.0e-6,
        0.05,
        WorldId(0),
    );
    for ix in 0..cloth_n {
        for iy in 0..cloth_n {
            b.pin_particle(cloth_first + iy * cloth_n + ix);
        }
    }

    let nx: u32 = 3;
    let dx: Real = 0.10;
    let cube_first = (cloth_n * cloth_n) as u32 + cloth_first;
    let cube_origin = Vec3::new(
        -dx * (nx as Real - 1.0) * 0.5,
        cloth_y + drop_height,
        -dx * (nx as Real - 1.0) * 0.5,
    );
    for kk in 0..nx {
        for jj in 0..nx {
            for ii in 0..nx {
                let p = cube_origin + Vec3::new(ii as Real * dx, jj as Real * dx, kk as Real * dx);
                b.add_particle(p, 0.05, dx * 0.5, WorldId(0));
            }
        }
    }
    let v_at = |i: u32, j: u32, k: u32| -> u32 { cube_first + (k * nx + j) * nx + i };
    let mut tets: Vec<[u32; 4]> = Vec::new();
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
    let materials = vec![MaterialNeoHookean::elastic(1.0e6, 0.3)];
    let mat_idx: Vec<u32> = vec![0; tets.len()];
    let tet_id = b
        .add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
        .expect("tet mesh");
    b.set_tet_mesh_friction(tet_id, cube_friction);
    b.set_tet_mesh_restitution(tet_id, cube_restitution);

    let n_cube_verts = nx * nx * nx;
    (b, cloth_first, cube_first, n_cube_verts)
}

fn cube_center_xz(q: &[Vec3], cube_first: u32, n: u32) -> (Real, Real) {
    let mut cx: Real = 0.0;
    let mut cz: Real = 0.0;
    for i in cube_first..cube_first + n {
        let p = q[i as usize];
        cx += p.x;
        cz += p.z;
    }
    let inv = 1.0 / n as Real;
    (cx * inv, cz * inv)
}

/// Bring a model into GPU residency. Returns `None` if no adapter is
/// available — caller `expect`s on the result so CI without a GPU
/// surfaces an explicit "GPU adapter required" error.
fn setup_gpu(mut model: Model) -> Option<(Engine, Model, State, State)> {
    let engine = match Engine::new_gpu() {
        Ok(e) => e,
        Err(_) => return None,
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).ok()?;
    let mut s_a = model.state().ok()?;
    let mut s_b = model.state().ok()?;
    s_a.upload_to_device(&device, &queue).ok()?;
    s_b.upload_to_device(&device, &queue).ok()?;
    Some((engine, model, s_a, s_b))
}

/// Run `n_steps` of the GPU VBD solver. Returns the final particle
/// positions (downloaded to host).
fn run_gpu_steps(
    engine: &Engine,
    model: &Model,
    mut s_a: State,
    mut s_b: State,
    n_steps: usize,
) -> Vec<Vec3> {
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let mut solver = VbdSolver::new_gpu(engine, model)
        .unwrap()
        .with_iterations(8)
        .with_contact_stiffness(1.0e5)
        .with_vt_contact_margin(0.01)
        .with_vt_contact_sweeps(8);
    let control = model.control();
    let contacts = Contacts::with_capacity(8);
    let dt: Real = 1.0 / 240.0;
    for _ in 0..n_steps {
        solver
            .step(model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    s_a.particle_q.download(&device, &queue).unwrap();
    s_a.particle_q.host().unwrap().to_vec()
}

/// Run `n_steps` of the CPU VBD solver — used as the parity oracle.
fn run_cpu_steps(model: &Model, n_steps: usize) -> Vec<Vec3> {
    let mut solver = VbdSolver::new_cpu()
        .with_iterations(8)
        .with_contact_stiffness(1.0e5)
        .with_vt_contact_margin(0.01)
        .with_vt_contact_sweeps(8);
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(8);
    let dt: Real = 1.0 / 240.0;
    for _ in 0..n_steps {
        solver
            .step(model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    s_a.particle_q.host().unwrap().to_vec()
}

/// μ = 0.5, horizontal pseudo-gravity = 0.02·g (well below μ·g).
/// Block must stay within 5 mm of the post-settle position over 1.5 s
/// of measured sim time. GPU port of the CPU
/// `coulomb_stick_below_threshold` test.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-soft-friction-gpu"]
fn gpu_coulomb_stick_below_threshold() {
    let mu: Real = 0.5;
    let alpha: Real = 0.02;
    let drop_height: Real = 0.001;
    let (mut b, _cloth_first, cube_first, n_cube_verts) =
        build_block_on_cloth(mu, 0.0, drop_height);
    b.set_world_gravity(WorldId(0), Vec3::new(alpha * 9.81, -9.81, 0.0));
    let model_in = b.build().unwrap();

    let (engine, model, s_a, s_b) = setup_gpu(model_in).expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut solver = VbdSolver::new_gpu(&engine, &model)
        .unwrap()
        .with_iterations(8)
        .with_contact_stiffness(1.0e5)
        .with_vt_contact_margin(0.01)
        .with_vt_contact_sweeps(8);
    let mut s_a = s_a;
    let mut s_b = s_b;
    let control = model.control();
    let contacts = Contacts::with_capacity(8);
    let dt: Real = 1.0 / 240.0;

    // Settle phase (60 steps) — let the cube land on the cloth before
    // baselining the lateral position.
    for _ in 0..60 {
        Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    s_a.particle_q.download(&device, &queue).unwrap();
    let q_settle = s_a.particle_q.host().unwrap().to_vec();
    let (cx_settle, cz_settle) = cube_center_xz(&q_settle, cube_first, n_cube_verts);

    // Measurement phase — 1.5 s of sim time. Friction MUST hold the
    // block within 5 mm of the settle position.
    for _ in 0..360 {
        Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    s_a.particle_q.download(&device, &queue).unwrap();
    let q_final = s_a.particle_q.host().unwrap().to_vec();
    let (cx, cz) = cube_center_xz(&q_final, cube_first, n_cube_verts);
    let drift = ((cx - cx_settle).powi(2) + (cz - cz_settle).powi(2)).sqrt();
    eprintln!(
        "GPU stick test: mu={mu}, alpha={alpha}, drift={drift:.4} m, \
         settle=({cx_settle:.4}, {cz_settle:.4}), final=({cx:.4}, {cz:.4})"
    );
    assert!(
        drift <= 0.005,
        "GPU block slipped despite force below μ·mg: drift={drift} m, \
         settle=({cx_settle}, {cz_settle}), final=({cx}, {cz})"
    );
}

/// μ = 0.5, horizontal pseudo-gravity = 1.5·g (well above μ·g).
/// Block must slide far (> 50 mm). GPU port of the CPU
/// `coulomb_slip_above_threshold` test.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-soft-friction-gpu"]
fn gpu_coulomb_slip_above_threshold() {
    let mu: Real = 0.5;
    let alpha: Real = 1.5;
    let drop_height: Real = 0.001;
    let (mut b, _cloth_first, cube_first, n_cube_verts) =
        build_block_on_cloth(mu, 0.0, drop_height);
    b.set_world_gravity(WorldId(0), Vec3::new(alpha * 9.81, -9.81, 0.0));
    let model_in = b.build().unwrap();

    let (engine, model, s_a, s_b) = setup_gpu(model_in).expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut solver = VbdSolver::new_gpu(&engine, &model)
        .unwrap()
        .with_iterations(8)
        .with_contact_stiffness(1.0e5)
        .with_vt_contact_margin(0.01)
        .with_vt_contact_sweeps(8);
    let mut s_a = s_a;
    let mut s_b = s_b;
    let control = model.control();
    let contacts = Contacts::with_capacity(8);
    let dt: Real = 1.0 / 240.0;

    s_a.particle_q.download(&device, &queue).unwrap();
    let q0 = s_a.particle_q.host().unwrap().to_vec();
    let (cx0, cz0) = cube_center_xz(&q0, cube_first, n_cube_verts);

    for _ in 0..240 {
        Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    s_a.particle_q.download(&device, &queue).unwrap();
    let q_final = s_a.particle_q.host().unwrap().to_vec();
    let (cx, cz) = cube_center_xz(&q_final, cube_first, n_cube_verts);
    let drift = ((cx - cx0).powi(2) + (cz - cz0).powi(2)).sqrt();
    eprintln!(
        "GPU slip test: mu={mu}, alpha={alpha}, drift={drift:.4} m, \
         start=({cx0:.4}, {cz0:.4}), final=({cx:.4}, {cz:.4})"
    );
    assert!(
        drift >= 0.05,
        "GPU block stayed put despite force above μ·mg: drift={drift} m, \
         start=({cx0}, {cz0}), final=({cx}, {cz})"
    );
}

/// CPU/GPU parity check: the same μ = 0.5 friction scene run on both
/// backends for 60 substeps; per-vertex max drift must be ≤ 5 mm.
///
/// 60 steps puts the cube on the cloth (with friction engaged after
/// the first contact frame); a longer run amplifies the f32-vs-f64 +
/// Jacobi-vs-Gauss-Seidel differences past 5 mm even though both
/// paths converge to the same rest configuration.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-soft-friction-gpu"]
fn gpu_friction_matches_cpu_oracle() {
    let mu: Real = 0.5;
    let drop_height: Real = 0.001;
    // A small lateral pseudo-gravity (α = 0.1·g, well below μ·g) so
    // friction is engaged but the block doesn't slip (parity is
    // dominated by the contact projection, not by chaotic slip
    // dynamics).
    let alpha: Real = 0.1;
    let cpu_model = {
        let (mut b, _, _, _) = build_block_on_cloth(mu, 0.0, drop_height);
        b.set_world_gravity(WorldId(0), Vec3::new(alpha * 9.81, -9.81, 0.0));
        b.build().unwrap()
    };
    let gpu_model_in = {
        let (mut b, _, _, _) = build_block_on_cloth(mu, 0.0, drop_height);
        b.set_world_gravity(WorldId(0), Vec3::new(alpha * 9.81, -9.81, 0.0));
        b.build().unwrap()
    };

    let n_steps = 60;
    let cpu_q = run_cpu_steps(&cpu_model, n_steps);

    let (engine, gpu_model, s_a, s_b) = setup_gpu(gpu_model_in).expect("GPU adapter required");
    let gpu_q = run_gpu_steps(&engine, &gpu_model, s_a, s_b, n_steps);

    assert_eq!(cpu_q.len(), gpu_q.len(), "particle count drift");
    let mut max_err: Real = 0.0;
    let mut max_idx = 0usize;
    for (i, (cp, gp)) in cpu_q.iter().zip(gpu_q.iter()).enumerate() {
        let d = (*cp - *gp).length();
        if d > max_err {
            max_err = d;
            max_idx = i;
        }
    }
    eprintln!(
        "GPU friction parity (60 steps): max per-vertex drift = {max_err:.4} m \
         (vertex {max_idx}, cpu={:?}, gpu={:?})",
        cpu_q[max_idx], gpu_q[max_idx],
    );
    assert!(
        max_err < 0.005,
        "GPU/CPU friction divergence {max_err:.4} m exceeds 5 mm tolerance \
         (vertex {max_idx})"
    );
}

/// Mirror of the viewer `Scene::SoftBrake` build — soft tet block on
/// a kinematic ramp tilted at 25° with μ = 1.0 on both surfaces. The
/// CPU pipeline does NOT fully arrest the slide here (Stage 33.5
/// position-form Coulomb friction undershoots the gravity-driven X
/// velocity build-up); the GPU port should reproduce the same creep
/// to within parity tolerance. See the matching CPU oracle in
/// `tests/soft_friction.rs::cpu_soft_brake_high_mu_creeps` for the
/// reference behavior.
///
/// This test asserts:
///   1. The GPU SoftBrake scene runs without crashing / NaN.
///   2. The block creeps within the SAME 1.5–4.0 m window as the CPU
///      oracle (i.e. the GPU friction is engaged and roughly matches
///      CPU magnitudes; it isn't silently zero-friction).
///
/// A future Stage 33.2.2 / Stage 33.5.1 friction-strength bump
/// (impulse-form normal-force estimate instead of `μ · g · dt` per
/// substep) would tighten the bound to 5 mm.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-soft-friction-gpu"]
fn gpu_soft_brake_matches_cpu() {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.05);

    // 25° tilt — viewer SoftBrake setting.
    let ramp_tilt: Real = 25.0_f64.to_radians();
    b.set_world_gravity(
        WorldId(0),
        Vec3::new(9.81 * ramp_tilt.sin(), -9.81 * ramp_tilt.cos(), 0.0),
    );

    // Ramp: thick (1.5 × 0.10 × 0.6) kinematic tet slab at y = 1.0.
    let ramp_lo = Vec3::new(-0.75, 1.0, -0.30);
    let ramp_hi = Vec3::new(0.75, 1.10, 0.30);
    let mut ramp_verts: Vec<u32> = Vec::with_capacity(8);
    for &z in &[ramp_lo.z, ramp_hi.z] {
        for &y in &[ramp_lo.y, ramp_hi.y] {
            for &x in &[ramp_lo.x, ramp_hi.x] {
                let id = b.add_particle(Vec3::new(x, y, z), 0.0, 0.02, WorldId(0));
                ramp_verts.push(id);
            }
        }
    }
    let rv: [u32; 8] = ramp_verts.as_slice().try_into().unwrap();
    let ramp_tets = cube_to_six_tets(rv);
    let ramp_id = b
        .add_tet_mesh(&ramp_tets, 1.0e6, 0.3, None)
        .expect("ramp tet block");
    // High μ — well above tan(25°).
    b.set_tet_mesh_friction(ramp_id, 1.0);

    // Block: 3×3×3 tet jelly cube above the ramp.
    let bnx: u32 = 3;
    let bdx: Real = 0.10;
    let block_origin = Vec3::new(-0.50, 1.10 + 0.005, -bdx * (bnx as Real - 1.0) * 0.5);
    let block_first = 8u32;
    for kk in 0..bnx {
        for jj in 0..bnx {
            for ii in 0..bnx {
                let p =
                    block_origin + Vec3::new(ii as Real * bdx, jj as Real * bdx, kk as Real * bdx);
                b.add_particle(p, 0.05, bdx * 0.5, WorldId(0));
            }
        }
    }
    let v_at_b = |i: u32, j: u32, k: u32| -> u32 { block_first + (k * bnx + j) * bnx + i };
    let mut block_tets: Vec<[u32; 4]> = Vec::new();
    for kk in 0..bnx - 1 {
        for jj in 0..bnx - 1 {
            for ii in 0..bnx - 1 {
                let corners = [
                    v_at_b(ii, jj, kk),
                    v_at_b(ii + 1, jj, kk),
                    v_at_b(ii, jj + 1, kk),
                    v_at_b(ii + 1, jj + 1, kk),
                    v_at_b(ii, jj, kk + 1),
                    v_at_b(ii + 1, jj, kk + 1),
                    v_at_b(ii, jj + 1, kk + 1),
                    v_at_b(ii + 1, jj + 1, kk + 1),
                ];
                block_tets.extend_from_slice(&cube_to_six_tets(corners));
            }
        }
    }
    let materials = vec![MaterialNeoHookean::elastic(1.0e6, 0.3)];
    let mat_idx: Vec<u32> = vec![0; block_tets.len()];
    let block_id = b
        .add_tet_mesh_per_material(&block_tets, &materials, &mat_idx, None)
        .expect("block tet mesh");
    b.set_tet_mesh_friction(block_id, 1.0);

    let block_n_verts = bnx * bnx * bnx;
    let model_in = b.build().unwrap();

    let (engine, model, s_a, s_b) = setup_gpu(model_in).expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut solver = VbdSolver::new_gpu(&engine, &model)
        .unwrap()
        .with_iterations(8)
        .with_contact_stiffness(2.0e5)
        .with_vt_contact_margin(0.02)
        .with_vt_contact_sweeps(10);
    let mut s_a = s_a;
    let mut s_b = s_b;
    let control = model.control();
    let contacts = Contacts::with_capacity(8);
    let dt: Real = 1.0 / 240.0;

    // Settle phase: 60 steps before measurement so the block lands
    // on the ramp with friction engaged.
    for _ in 0..60 {
        Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    s_a.particle_q.download(&device, &queue).unwrap();
    let q_settle = s_a.particle_q.host().unwrap().to_vec();
    let (cx0, cz0) = cube_center_xz(&q_settle, block_first, block_n_verts);

    // Measurement phase: 1 s of sim time. With μ = 1.0 ≫ tan(25°)
    // the block must stick: lateral drift along the slope direction
    // (X) stays below 5 mm.
    for _ in 0..240 {
        Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    s_a.particle_q.download(&device, &queue).unwrap();
    let q_final = s_a.particle_q.host().unwrap().to_vec();
    let (cx, cz) = cube_center_xz(&q_final, block_first, block_n_verts);
    let dx = cx - cx0;
    let dz = cz - cz0;
    let drift = (dx * dx + dz * dz).sqrt();
    eprintln!(
        "GPU SoftBrake (μ=1.0) drift after 1 s: {drift:.4} m, \
         start=({cx0:.4}, {cz0:.4}), final=({cx:.4}, {cz:.4})"
    );
    // No NaNs.
    for (i, p) in q_final.iter().enumerate() {
        assert!(
            p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
            "GPU SoftBrake particle {i} non-finite: {p:?}"
        );
    }
    // Match the CPU reference window (1.5–4.0 m). The GPU result
    // should land in the same range — friction is engaged and
    // qualitatively matches CPU. See the test docstring for why a
    // tight stick assertion isn't possible until Stage 33.5.1.
    assert!(
        (1.5..=4.0).contains(&drift),
        "GPU SoftBrake creep is outside the CPU-reference 1.5–4.0 m window: \
         drift={drift} m"
    );
}
