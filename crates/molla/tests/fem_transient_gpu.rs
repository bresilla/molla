//! Stage 36.5 Slice F — GPU heat-transient parity test.
//!
//! Verifies that [`molla::fem::TransientStepperGpu::step`] produces
//! the same per-vertex temperature as the CPU
//! [`molla::fem::solve_transient`] over many backward-Euler steps.
//!
//! Tests are `#[ignore]` because they need a real wgpu adapter.

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla::compute::Engine;
use molla::fem::{
    assemble_global, compute_element_mass, compute_element_stiffness, solve_transient,
    BoundaryCondition, FemMesh, FunctionSpace, TransientStepperGpu,
};
use molla::math::sparse::BsrMatrix;
use molla::math::Real;

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn assemble_mass(mesh: &FemMesh) -> BsrMatrix {
    assemble_global(mesh, FunctionSpace::P1Nodal, |c, v| {
        compute_element_mass(c, v)
    })
    .expect("mass assembly")
}

fn assemble_stiffness(mesh: &FemMesh, alpha: Real) -> BsrMatrix {
    assemble_global(mesh, FunctionSpace::P1Nodal, |c, v| {
        compute_element_stiffness(c, v, alpha)
    })
    .expect("stiffness assembly")
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn gpu_transient_step_matches_cpu_oracle() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // 8³ tet cube with bottom face pinned to zero — same recipe as
    // the existing fem_v1 transient heat test, just driven through
    // both CPU + GPU paths simultaneously.
    let nx = 8usize;
    let mesh = FemMesh::tet_grid_3d(nx, nx, nx, 1.0, 1.0, 1.0);
    let n = mesh.vert_count();
    let alpha: Real = 1.0;

    let m = assemble_mass(&mesh);
    let k = assemble_stiffness(&mesh, alpha);

    // Pin bottom face (z = 0) to zero, no source.
    let mut bcs = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        if p.z <= 1e-9 {
            bcs.push(BoundaryCondition::Dirichlet {
                vertex: i as u32,
                value: 0.0,
            });
        }
    }

    // Initial state: linear gradient from top to bottom (units don't
    // matter — we only compare CPU vs GPU).
    let mut u_cpu: Vec<Real> = (0..n).map(|i| mesh.vertices[i].z).collect();
    let mut u_gpu = u_cpu.clone();
    let f = vec![0.0; n];

    let dt: Real = 0.01;

    let engine = Engine::new_gpu().expect("gpu");
    let stepper = TransientStepperGpu::new(&engine, &mesh, &m, &k, dt, &bcs)
        .expect("TransientStepperGpu::new");

    let n_steps = 50;
    for step in 0..n_steps {
        let cpu = solve_transient(&m, &k, &f, &u_cpu, dt, &bcs);
        let gpu = stepper.step(&u_gpu, &f).expect("gpu transient step");

        let mut max_du: Real = 0.0;
        for i in 0..n {
            max_du = max_du.max((cpu[i] - gpu.x[i]).abs());
        }
        assert!(
            max_du < 1e-3,
            "step {step}: max |Δu| = {max_du:.3e} (GPU iters = {})",
            gpu.iters,
        );

        u_cpu = cpu;
        u_gpu = gpu.x;
    }

    let mut final_du: Real = 0.0;
    for i in 0..n {
        final_du = final_du.max((u_cpu[i] - u_gpu[i]).abs());
    }
    eprintln!(
        "gpu_transient_step_matches_cpu_oracle: n = {n}, {n_steps} steps, final max |Δu| = {final_du:.3e}",
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn gpu_transient_step_with_nonzero_dirichlet_matches_cpu() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // Slice G: nonzero Dirichlet now supported. This exercise mirrors
    // the FemHeat viewer scene — bottom face cold (T=0), top face
    // hot (T=1). Verify GPU/CPU parity over many backward-Euler
    // steps with the heat conducting through.
    let nx = 8usize;
    let mesh = FemMesh::tet_grid_3d(nx, nx, nx, 1.0, 1.0, 1.0);
    let n = mesh.vert_count();
    let m = assemble_mass(&mesh);
    let k = assemble_stiffness(&mesh, 1.0);

    let mut bcs = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        if p.z <= 1e-9 {
            bcs.push(BoundaryCondition::Dirichlet {
                vertex: i as u32,
                value: 0.0,
            });
        } else if p.z >= 1.0 - 1e-9 {
            bcs.push(BoundaryCondition::Dirichlet {
                vertex: i as u32,
                value: 1.0,
            });
        }
    }

    let mut u_cpu = vec![0.0_f64 as Real; n];
    for bc in &bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = bc {
            u_cpu[*vertex as usize] = *value;
        }
    }
    let mut u_gpu = u_cpu.clone();
    let f = vec![0.0; n];
    let dt: Real = 0.01;

    let engine = Engine::new_gpu().expect("gpu");
    let stepper = TransientStepperGpu::new(&engine, &mesh, &m, &k, dt, &bcs).expect("stepper");

    let n_steps = 30;
    for step in 0..n_steps {
        let cpu = solve_transient(&m, &k, &f, &u_cpu, dt, &bcs);
        let gpu = stepper.step(&u_gpu, &f).expect("gpu step");
        let mut max_du: Real = 0.0;
        for i in 0..n {
            max_du = max_du.max((cpu[i] - gpu.x[i]).abs());
        }
        assert!(
            max_du < 1e-3,
            "step {step}: max |Δu| = {max_du:.3e} (GPU iters = {})",
            gpu.iters,
        );
        u_cpu = cpu;
        u_gpu = gpu.x;
    }
    eprintln!(
        "gpu_transient_step_with_nonzero_dirichlet_matches_cpu: n = {n}, {n_steps} steps OK",
    );
}
