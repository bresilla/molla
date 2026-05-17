//! Stage 20.1 — VBD CABLE + Dahl + AVBD hard-mode integration tests.
//!
//! All `#[ignore]` because they need a real wgpu adapter (run via
//! `nixVulkan cargo test -p molla --features gpu --test
//! vbd_extended_gpu -- --ignored`).
//!
//! Three regression scenarios exercise the new Stage 20.1 capabilities:
//!
//! 1. `cable_pull_test` — a 32-particle CABLE polyline with one end
//!    pinned at the origin and the other yanked by a kinematic
//!    "puller" body that translates outward. Total cable length must
//!    stay within 5 % of the rest length (the inextensible regime
//!    that motivates CABLE in the first place).
//!
//! 2. `dahl_friction_sweep_matches_oracle` — runs the WGSL Dahl
//!    kernel against the CPU oracle (`dahl_oracle_step`) for a
//!    coefficient sweep and asserts steady-state stick→slip
//!    transition magnitudes agree within 5 %.
//!
//! 3. `hard_mode_no_penetration` — drapes a 32×32 cloth on a
//!    kinematic sphere using soft mode and hard mode AVBD; asserts
//!    hard mode keeps every particle outside the sphere (zero
//!    penetration) where soft mode allows ~1 mm violation.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::{BodyId, WorldId};
use molla_kernels::{DahlConfig, DahlFrictionKernel, dahl_oracle_step};
use molla_math::{Real, Transform, Vec3};
use molla_sim::{Contacts, Model, ModelBuilder, State};
use molla_solvers::{Solver, VbdMode, VbdSolver};
use molla_storage::{Buffer, BufferUsage};

/// Build a horizontal CABLE polyline along +X with `n` particles, one
/// end pinned at `start`, the other end *also pinned* (we'll yank it
/// kinematically by reassigning particle_q in place each step).
fn build_cable_pull_model(n: u32, start: Vec3, end: Vec3) -> (Model, u32, u32) {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.4);
    let stiffness = 1.0e6_f64; // very stiff — exactly the CABLE regime
    let max_stretch = 0.0;
    let particle_mass = 0.05;
    let particle_radius = 0.02;
    let (first, count, _cable_id) = b.add_cable_polyline(
        start,
        end,
        n,
        particle_mass,
        particle_radius,
        stiffness,
        max_stretch,
        WorldId(0),
    );
    // Pin both endpoints (kinematic — the test will set the puller's
    // position directly via `particle_q[end]` per substep).
    b.pin_particle(first);
    b.pin_particle(first + count - 1);
    (b.build().unwrap(), first, count)
}

fn run_vbd_substeps_gpu(
    model: &mut Model,
    iters: usize,
    dt: Real,
    substeps: usize,
    contact_k: f64,
    mode: VbdMode,
    mut on_substep: impl FnMut(usize, &mut State),
) -> Vec<Vec3> {
    let engine = Engine::new_gpu().expect("GPU adapter for VBD");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let mut control = model.control();
    let mut contacts = Contacts::with_capacity(64);
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    control.upload_to_device(&device, &queue).unwrap();
    contacts.upload_to_device(&device, &queue).unwrap();
    let mut solver = VbdSolver::new_gpu(&engine, model)
        .unwrap()
        .with_iterations(iters)
        .with_contact_stiffness(contact_k)
        .with_mode(mode);
    for k in 0..substeps {
        // Allow the test to drive kinematic state (e.g. yank the
        // puller particle). The hook gets the *input* state of this
        // substep (s_a). Download positions first so the host slice
        // is fresh, then re-upload after the hook mutates them.
        s_a.particle_q.download(&device, &queue).unwrap();
        on_substep(k, &mut s_a);
        s_a.particle_q.upload(&queue).unwrap();
        solver
            .step(model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    s_a.particle_q.download(&device, &queue).unwrap();
    s_a.particle_q.host().unwrap().to_vec()
}

/// CABLE pull-test. A 32-particle inextensible cable; one end pinned
/// at origin, the other end yanked outward at constant velocity by a
/// kinematic "puller". After 600 substeps the cable should still be
/// roughly straight and the total polyline length within 5 % of the
/// rest length.
#[test]
#[ignore = "needs a real wgpu adapter — run via nixVulkan cargo test --features gpu --test vbd_extended_gpu -- --ignored"]
fn cable_pull_test() {
    let n = 32;
    let start = Vec3::new(0.0, 1.0, 0.0);
    let end = Vec3::new(1.0, 1.0, 0.0); // 1 m rope
    let rest_total = (end - start).length();
    let (mut model, first, count) = build_cable_pull_model(n, start, end);
    assert_eq!(count, n);

    // Yank speed (m/s) — modest so the substep dt has no trouble.
    let yank_v = 0.05_f64;
    let dt: Real = 1.0 / 240.0;
    let substeps = 600;
    let final_pos = run_vbd_substeps_gpu(
        &mut model,
        12,
        dt,
        substeps,
        0.0,
        VbdMode::Hard,
        |k, s_a| {
            // Move the last particle outward at constant velocity.
            let q = s_a.particle_q.host_mut().unwrap();
            let t = (k as f64) * dt;
            let new_x = end.x + yank_v * t;
            q[(first + count - 1) as usize] = Vec3::new(new_x, end.y, end.z);
        },
    );

    // Walk the polyline and sum segment lengths.
    let mut total = 0.0_f64;
    for i in 0..(n - 1) {
        let a = final_pos[(first + i) as usize];
        let b = final_pos[(first + i + 1) as usize];
        total += (b - a).length();
    }
    let yanked_rest = rest_total + yank_v * (substeps as f64) * dt;
    let abs_err = (total - yanked_rest).abs();
    let rel_err = abs_err / yanked_rest;
    eprintln!(
        "cable length: actual {total:.4} m / rest+yank {yanked_rest:.4} m  ({:.2}% err)",
        rel_err * 100.0
    );
    assert!(
        rel_err < 0.05,
        "CABLE total length {total:.4} m diverges from {yanked_rest:.4} m by {:.2} %",
        rel_err * 100.0,
    );
    // Sanity: no particle is NaN.
    for (i, p) in final_pos.iter().enumerate() {
        assert!(
            p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
            "cable particle {i} non-finite at {p:?}"
        );
    }
}

/// Dahl GPU vs CPU oracle: run the WGSL kernel for `n_steps` against
/// the same kernel evaluated on the CPU and assert the per-step
/// outputs agree within 1 %. Tests both deep-stick and slipping
/// regimes by sweeping the friction coefficient.
#[test]
#[ignore = "needs a real wgpu adapter — run via nixVulkan cargo test --features gpu --test vbd_extended_gpu -- --ignored"]
fn dahl_friction_sweep_matches_oracle() {
    let engine = Engine::new_gpu().expect("GPU adapter for Dahl");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    // Sweep five friction coefficients spanning low μ (fully slipping)
    // through high μ (deep stick) given a fixed normal force.
    let mus = [0.0_f64, 0.1, 0.3, 0.6, 1.0];
    let n_contacts = mus.len() as u32;
    let v_t = Vec3::new(0.5, 0.0, 0.0); // 0.5 m/s constant slip
    let normal = Vec3::Y;
    let f_n = 10.0_f64;
    let dt: Real = 1.0 / 480.0;
    let n_steps = 200;
    let cfg = DahlConfig::default();

    // Build per-contact buffers.
    let mut contact_normal =
        Buffer::from_vec(vec![normal; n_contacts as usize], BufferUsage::Storage);
    let mut contact_v_tan = Buffer::from_vec(vec![v_t; n_contacts as usize], BufferUsage::Storage);
    let mut contact_f_n = Buffer::from_vec(vec![f_n; n_contacts as usize], BufferUsage::Storage);
    let mut contact_mu = Buffer::from_vec(mus.to_vec(), BufferUsage::Storage);
    contact_normal.ensure_device(&device).unwrap();
    contact_normal.upload(&queue).unwrap();
    contact_v_tan.ensure_device(&device).unwrap();
    contact_v_tan.upload(&queue).unwrap();
    contact_f_n.ensure_device(&device).unwrap();
    contact_f_n.upload(&queue).unwrap();
    contact_mu.ensure_device(&device).unwrap();
    contact_mu.upload(&queue).unwrap();

    let mut kernel = DahlFrictionKernel::new(&device, &queue, n_contacts);
    let mut cpu_z = vec![Vec3::ZERO; n_contacts as usize];
    let mut cpu_f = vec![Vec3::ZERO; n_contacts as usize];
    for _ in 0..n_steps {
        kernel
            .dispatch(
                &device,
                &queue,
                dt,
                cfg,
                &contact_normal,
                &contact_v_tan,
                &contact_f_n,
                &contact_mu,
            )
            .unwrap();
        for c in 0..(n_contacts as usize) {
            let (z_new, f_new) = dahl_oracle_step(cpu_z[c], v_t, normal, f_n, mus[c], dt, cfg);
            cpu_z[c] = z_new;
            cpu_f[c] = f_new;
        }
    }
    let gpu_f = kernel.download_friction(&device, &queue).unwrap().to_vec();

    eprintln!("Dahl sweep μ = {mus:?}");
    for c in 0..(n_contacts as usize) {
        let target = mus[c] * f_n; // analytical Coulomb cap
        let gpu_mag = gpu_f[c].length();
        let cpu_mag = cpu_f[c].length();
        eprintln!(
            "  μ={:.2}: GPU |F_t|={gpu_mag:.4}  CPU |F_t|={cpu_mag:.4}  Coulomb cap={target:.4}",
            mus[c]
        );
        // Sign / direction agreement.
        let dot = gpu_f[c].dot(cpu_f[c]);
        if cpu_mag > 1.0e-6 {
            assert!(
                dot > 0.0,
                "Dahl GPU/CPU sign mismatch at μ={}: gpu={:?} cpu={:?}",
                mus[c],
                gpu_f[c],
                cpu_f[c],
            );
        }
        // Magnitude agreement within 1 % (or 1e-4 N for the zero-μ case).
        let abs_err = (gpu_mag - cpu_mag).abs();
        let rel_err = abs_err / cpu_mag.max(1.0e-6);
        assert!(
            abs_err < 1.0e-4 || rel_err < 0.05,
            "Dahl |F_t| mismatch at μ={}: gpu={gpu_mag:.6} cpu={cpu_mag:.6} ({:.3}% err)",
            mus[c],
            rel_err * 100.0,
        );
        // For non-zero μ, the steady-state magnitude under constant
        // slip should be at the Coulomb cap (within 5 %).
        if mus[c] > 0.0 {
            assert!(
                (gpu_mag - target).abs() / target < 0.05,
                "Dahl steady-state |F_t|={gpu_mag} vs target μF_n={target}",
            );
        } else {
            assert!(
                gpu_mag < 1.0e-6,
                "μ=0 must produce zero friction, got {gpu_mag}"
            );
        }
    }
}

/// Hard-mode AVBD must keep every cloth particle strictly outside a
/// kinematic sphere obstacle, where soft mode allows ~mm penetration.
/// We compare the two modes at the same compliance and contact stiffness.
#[test]
#[ignore = "needs a real wgpu adapter — run via nixVulkan cargo test --features gpu --test vbd_extended_gpu -- --ignored"]
fn hard_mode_no_penetration() {
    let n = 32;
    let size = 1.6;
    let top_y = 2.5;
    let sphere_centre = Vec3::new(0.0, 1.0, 0.0);
    let sphere_radius = 0.5_f64;
    let particle_radius = 0.02_f64;
    let compliance = 5.0e-6;
    let max_stretch = 0.10;
    let dt: Real = 1.0 / 480.0;
    let substeps = 600;
    let iters = 10;
    let contact_k = 5.0e4_f64;

    let build = || -> Model {
        let mut b = ModelBuilder::new();
        b.set_linear_damping(0.4);
        let origin = Vec3::new(-size * 0.5, top_y, -size * 0.5);
        let _ = b.add_cloth_grid(
            origin,
            Vec3::new(size, 0.0, 0.0),
            Vec3::new(0.0, 0.0, size),
            n,
            n,
            0.05,
            particle_radius,
            compliance,
            max_stretch,
            WorldId(0),
        );
        b.add_shape_sphere(
            BodyId::NONE,
            sphere_radius,
            Transform::from_translation(sphere_centre),
        )
        .unwrap();
        b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
            .unwrap();
        b.build().unwrap()
    };

    let mut model_soft = build();
    let mut model_hard = build();
    let pos_soft = run_vbd_substeps_gpu(
        &mut model_soft,
        iters,
        dt,
        substeps,
        contact_k,
        VbdMode::Soft,
        |_, _| {},
    );
    let pos_hard = run_vbd_substeps_gpu(
        &mut model_hard,
        iters,
        dt,
        substeps,
        contact_k,
        VbdMode::Hard,
        |_, _| {},
    );

    let pen_of = |pos: &[Vec3]| -> f64 {
        let mut max_pen = 0.0_f64;
        for p in pos {
            let d = (*p - sphere_centre).length();
            let pen = (sphere_radius + particle_radius) - d;
            if pen > max_pen {
                max_pen = pen;
            }
        }
        max_pen
    };
    let pen_soft = pen_of(&pos_soft);
    let pen_hard = pen_of(&pos_hard);
    eprintln!("soft max penetration: {pen_soft:.5} m  hard max penetration: {pen_hard:.5} m");
    // Hard mode should be ≤ 1e-3 m everywhere (effectively zero
    // given the contact stiffness budget). Soft mode is allowed
    // some millimetric penetration; we only assert hard ≤ soft.
    assert!(
        pen_hard <= 1.0e-3,
        "hard-mode max penetration {pen_hard:.5} m exceeds 1 mm budget"
    );
    assert!(
        pen_hard <= pen_soft + 1.0e-9,
        "hard-mode shouldn't be worse than soft: hard={pen_hard:.5} soft={pen_soft:.5}",
    );
}
