//! Stage 36.5 Slice E — GPU wave transient parity tests.
//!
//! Verifies that [`molla::fem::WaveStepperGpu::step`] produces the
//! same `(u, v, a)` triple as the CPU [`molla::fem::solve_wave_transient`]
//! on a pinned-boundary 2D wave PDE.
//!
//! All tests are `#[ignore]` because they need a real wgpu adapter.
//! Run with `nixVulkan cargo test -p molla --test fem_wave_gpu -- --ignored`.

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla::compute::Engine;
use molla::fem::{
    assemble_global, compute_element_mass, compute_element_wave_stiffness, solve_wave_transient,
    BoundaryCondition, FemMesh, FunctionSpace, WaveStepperGpu, NEWMARK_BETA_DEFAULT,
    NEWMARK_GAMMA_DEFAULT,
};
use molla::math::sparse::BsrMatrix;
use molla::math::Real;

/// Serialise wgpu adapter ownership across the tests in this binary
/// — same pattern as `tests/fem_gpu.rs`.
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn assemble_mass(mesh: &FemMesh) -> BsrMatrix {
    assemble_global(mesh, FunctionSpace::P1Nodal, |c, v| {
        compute_element_mass(c, v)
    })
    .expect("mass assembly")
}

fn assemble_wave_stiffness(mesh: &FemMesh, c2: Real) -> BsrMatrix {
    assemble_global(mesh, FunctionSpace::P1Nodal, |c, v| {
        compute_element_wave_stiffness(c, v, c2)
    })
    .expect("stiffness assembly")
}

/// Build the WarpWave-style problem: `n_grid × n_grid` triangle mesh
/// on the unit square, all boundary verts pinned to zero, Gaussian
/// initial displacement at the centre.
fn build_warpwave(n_grid: usize) -> (FemMesh, Vec<BoundaryCondition>, Vec<Real>) {
    let mesh = FemMesh::tri_grid_2d(n_grid, n_grid, 1.0, 1.0);
    let mut bcs = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        if p.x <= 1e-9 || p.x >= 1.0 - 1e-9 || p.y <= 1e-9 || p.y >= 1.0 - 1e-9 {
            bcs.push(BoundaryCondition::Dirichlet {
                vertex: i as u32,
                value: 0.0,
            });
        }
    }
    let mut u_init = vec![0.0_f64 as Real; mesh.vert_count()];
    for (i, p) in mesh.vertices.iter().enumerate() {
        let dx = p.x - 0.5;
        let dy = p.y - 0.5;
        let r2 = dx * dx + dy * dy;
        u_init[i] = (-r2 * 50.0).exp();
    }
    // Re-zero the boundary so the initial state is BC-consistent.
    for bc in &bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = bc {
            u_init[*vertex as usize] = *value;
        }
    }
    (mesh, bcs, u_init)
}

#[test]
#[ignore = "needs a real wgpu adapter — run via nixVulkan cargo test"]
fn gpu_wave_step_matches_cpu_oracle() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // 16×16 grid — small enough to converge fast, large enough that
    // the f32 round-off doesn't dominate the comparison.
    let (mesh, bcs, u_init) = build_warpwave(16);
    let m = assemble_mass(&mesh);
    let k = assemble_wave_stiffness(&mesh, 1.0);

    let dt: Real = 0.005;
    let beta = NEWMARK_BETA_DEFAULT;
    let gamma = NEWMARK_GAMMA_DEFAULT;
    let n = mesh.vert_count();

    // Initial state: u = Gaussian, v = a = 0.
    let mut u_cpu = u_init.clone();
    let mut v_cpu = vec![0.0; n];
    let mut a_cpu = vec![0.0; n];
    let mut u_gpu = u_init.clone();
    let mut v_gpu = vec![0.0; n];
    let mut a_gpu = vec![0.0; n];

    let engine = Engine::new_gpu().expect("gpu");
    let stepper = WaveStepperGpu::new(&engine, &mesh, &m, &k, dt, beta, gamma, &bcs)
        .expect("WaveStepperGpu::new");

    // Step 50 times and compare every step.
    let n_steps = 50;
    for step in 0..n_steps {
        let cpu = solve_wave_transient(&m, &k, &u_cpu, &v_cpu, &a_cpu, dt, beta, gamma, &bcs);
        let gpu = stepper.step(&u_gpu, &v_gpu, &a_gpu).expect("gpu step");

        // Per-vertex max-abs error on u_new.
        let mut max_du: Real = 0.0;
        let mut max_dv: Real = 0.0;
        let mut max_da: Real = 0.0;
        for i in 0..n {
            max_du = max_du.max((cpu.u_new[i] - gpu.u_new[i]).abs());
            max_dv = max_dv.max((cpu.v_new[i] - gpu.v_new[i]).abs());
            max_da = max_da.max((cpu.a_new[i] - gpu.a_new[i]).abs());
        }
        // f32 GPU + f64 CPU + 50 steps of error accumulation — 1e-3
        // is loose but well within the published GPU FEM precision
        // budget.
        assert!(
            max_du < 1e-3,
            "step {step}: max |Δu| = {max_du:.3e} (CPU iters = {}, GPU iters = {})",
            cpu.iters,
            gpu.iters,
        );
        assert!(max_dv < 1e-2, "step {step}: max |Δv| = {max_dv:.3e}");
        assert!(max_da < 1.0, "step {step}: max |Δa| = {max_da:.3e}");

        u_cpu = cpu.u_new;
        v_cpu = cpu.v_new;
        a_cpu = cpu.a_new;
        u_gpu = gpu.u_new;
        v_gpu = gpu.v_new;
        a_gpu = gpu.a_new;
    }

    // Final-step diagnostic.
    let mut final_du: Real = 0.0;
    for i in 0..n {
        final_du = final_du.max((u_cpu[i] - u_gpu[i]).abs());
    }
    eprintln!(
        "gpu_wave_step_matches_cpu_oracle: n = {n}, {n_steps} steps, final max |Δu| = {final_du:.3e}",
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run via nixVulkan cargo test"]
fn gpu_wave_stepper_accepts_nonzero_dirichlet() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // Slice G: nonzero Dirichlet is now supported via the precomputed
    // column shift. Verify the stepper builds AND that one step
    // matches the CPU oracle on a problem with a non-zero pinned
    // boundary value.
    let mesh = FemMesh::tri_grid_2d(8, 8, 1.0, 1.0);
    let m = assemble_mass(&mesh);
    let k = assemble_wave_stiffness(&mesh, 1.0);
    let n = mesh.vert_count();
    // Pin the corner to a nonzero displacement.
    let bcs = vec![BoundaryCondition::Dirichlet {
        vertex: 0,
        value: 0.5,
    }];

    let engine = Engine::new_gpu().expect("gpu");
    let stepper = WaveStepperGpu::new(
        &engine,
        &mesh,
        &m,
        &k,
        0.01,
        NEWMARK_BETA_DEFAULT,
        NEWMARK_GAMMA_DEFAULT,
        &bcs,
    )
    .expect("stepper builds with nonzero Dirichlet");

    // Initial state: BC-consistent (pinned vertex at 0.5).
    let mut u_init = vec![0.0; n];
    u_init[0] = 0.5;
    let v0 = vec![0.0; n];
    let a0 = vec![0.0; n];

    let cpu = solve_wave_transient(
        &m,
        &k,
        &u_init,
        &v0,
        &a0,
        0.01,
        NEWMARK_BETA_DEFAULT,
        NEWMARK_GAMMA_DEFAULT,
        &bcs,
    );
    let gpu = stepper.step(&u_init, &v0, &a0).expect("step");
    let mut max_du: Real = 0.0;
    for i in 0..n {
        max_du = max_du.max((cpu.u_new[i] - gpu.u_new[i]).abs());
    }
    assert!(
        max_du < 1e-3,
        "nonzero-Dirichlet single-step max |Δu| = {max_du:.3e}",
    );
    assert!((gpu.u_new[0] - 0.5).abs() < 1e-5, "pinned DoF must hold");
}
