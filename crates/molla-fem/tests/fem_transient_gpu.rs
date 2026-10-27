//! Stage 36.5 Slice F + Slice H — GPU heat-transient parity test.
//!
//! Verifies that [`molla_fem::TransientStepperGpu::step`] produces
//! the same per-vertex temperature as the CPU
//! [`molla_fem::solve_transient`] over many backward-Euler steps.
//!
//! **Slice H** changed the GPU stepper to keep `u` resident across
//! timesteps — the new test pattern loops `gpu.step()` for N
//! iterations with no per-step host upload, then reads back the
//! final `u` and compares to the CPU oracle.
//!
//! Tests are `#[ignore]` because they need a real wgpu adapter.

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla_compute::Engine;
use molla_fem::{
    BoundaryCondition, FemMesh, FunctionSpace, TransientStepperGpu, assemble_global,
    compute_element_mass, compute_element_stiffness, solve_transient,
};
use molla_math::Real;
use molla_math::sparse::BsrMatrix;

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

    // Initial state: linear gradient from top to bottom.
    let u_init: Vec<Real> = (0..n).map(|i| mesh.vertices[i].z).collect();
    let mut u_cpu = u_init.clone();
    let f = vec![0.0; n];

    let dt: Real = 0.01;

    let engine = Engine::new_gpu().expect("gpu");
    let mut stepper = TransientStepperGpu::new(&engine, &mesh, &m, &k, dt, &bcs, &u_init)
        .expect("TransientStepperGpu::new");

    let n_steps = 50;
    for _ in 0..n_steps {
        let cpu = solve_transient(&m, &k, &f, &u_cpu, dt, &bcs);
        let _gpu_stats = stepper.step().expect("gpu transient step");
        u_cpu = cpu;
    }

    let u_gpu = stepper.read_u().expect("gpu read_u");
    let mut max_du: Real = 0.0;
    for i in 0..n {
        max_du = max_du.max((u_cpu[i] - u_gpu[i]).abs());
    }
    eprintln!(
        "gpu_transient_step_matches_cpu_oracle: n = {n}, {n_steps} steps, final max |Δu| = {max_du:.3e}",
    );
    assert!(max_du < 1e-3, "final max |Δu| = {max_du:.3e}");
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn gpu_transient_step_with_nonzero_dirichlet_matches_cpu() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // Slice G: nonzero Dirichlet now supported. Slice H: stepper is
    // persistent — verify GPU/CPU parity over many backward-Euler
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

    let mut u_init = vec![0.0_f64 as Real; n];
    for bc in &bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = bc {
            u_init[*vertex as usize] = *value;
        }
    }
    let mut u_cpu = u_init.clone();
    let f = vec![0.0; n];
    let dt: Real = 0.01;

    let engine = Engine::new_gpu().expect("gpu");
    let mut stepper =
        TransientStepperGpu::new(&engine, &mesh, &m, &k, dt, &bcs, &u_init).expect("stepper");

    let n_steps = 30;
    for _ in 0..n_steps {
        let cpu = solve_transient(&m, &k, &f, &u_cpu, dt, &bcs);
        let _gpu_stats = stepper.step().expect("gpu step");
        u_cpu = cpu;
    }
    let u_gpu = stepper.read_u().expect("read_u");
    let mut max_du: Real = 0.0;
    for i in 0..n {
        max_du = max_du.max((u_cpu[i] - u_gpu[i]).abs());
    }
    eprintln!(
        "gpu_transient_step_with_nonzero_dirichlet_matches_cpu: n = {n}, {n_steps} steps, max |Δu| = {max_du:.3e}",
    );
    assert!(max_du < 1e-3, "final max |Δu| = {max_du:.3e}");
}

/// Interleave a scalar CSR system into `dim` decoupled components:
/// scalar entry `(i, j, v)` becomes `(i·dim + c, j·dim + c, v)` for
/// every component `c`. The result is a valid vector-field system
/// whose per-component dynamics are exactly the scalar system's, so
/// the scalar CPU solver run once per component is an exact oracle.
fn interleave_csr(a: &BsrMatrix, dim: usize) -> BsrMatrix {
    let n = a.nrows();
    let mut row_offsets = Vec::with_capacity(n * dim + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_offsets.push(0usize);
    for i in 0..n {
        for c in 0..dim {
            for s in a.row_offsets[i]..a.row_offsets[i + 1] {
                col_idx.push(a.col_idx[s] * dim + c);
                values.push(a.values[s]);
            }
            row_offsets.push(col_idx.len());
        }
    }
    BsrMatrix {
        b: 1,
        block_rows: n * dim,
        block_cols: n * dim,
        row_offsets,
        col_idx,
        values,
    }
}

/// INSPECTION B2 — vector Dirichlet BCs on the GPU transient stepper.
/// Before the fix, a `DirichletVector` with more than one component
/// was silently truncated to `values[0]` in release builds; now the
/// stepper infers `dim = nrows / vert_count` and pins every component.
/// Each bottom-face vertex is pinned to a *different* value per
/// component — the exact case the truncation destroyed.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn gpu_transient_vector_dirichlet_matches_componentwise_cpu() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let nx = 6usize;
    let mesh = FemMesh::tet_grid_3d(nx, nx, nx, 1.0, 1.0, 1.0);
    let nv = mesh.vert_count();
    let dim = 3usize;
    let m = assemble_mass(&mesh);
    let k = assemble_stiffness(&mesh, 1.0);
    let m3 = interleave_csr(&m, dim);
    let k3 = interleave_csr(&k, dim);

    let pin_vals = [0.0 as Real, 0.5, 1.0];
    let bottom: Vec<u32> = mesh
        .vertices
        .iter()
        .enumerate()
        .filter(|(_, p)| p.z <= 1e-9)
        .map(|(i, _)| i as u32)
        .collect();
    let bcs3: Vec<BoundaryCondition> = bottom
        .iter()
        .map(|&v| BoundaryCondition::DirichletVector {
            vertex: v,
            values: pin_vals.to_vec(),
        })
        .collect();

    // Initial state: component c = z + 0.25·c, pinned rows preset to
    // their BC value so the state starts BC-consistent.
    let mut u3_init = vec![0.0 as Real; nv * dim];
    for i in 0..nv {
        for c in 0..dim {
            u3_init[i * dim + c] = mesh.vertices[i].z + 0.25 * c as Real;
        }
    }
    for &v in &bottom {
        for c in 0..dim {
            u3_init[v as usize * dim + c] = pin_vals[c];
        }
    }

    let dt: Real = 0.01;
    let engine = Engine::new_gpu().expect("gpu");
    let mut stepper = TransientStepperGpu::new(&engine, &mesh, &m3, &k3, dt, &bcs3, &u3_init)
        .expect("vector TransientStepperGpu::new");

    let n_steps = 40;
    for _ in 0..n_steps {
        stepper.step().expect("gpu transient step");
    }
    let u3_gpu = stepper.read_u().expect("gpu read_u");

    // Oracle: the interleaved system is `dim` decoupled copies of the
    // scalar system — run the scalar CPU path once per component.
    let f = vec![0.0; nv];
    let mut max_du: Real = 0.0;
    for c in 0..dim {
        let bcs_c: Vec<BoundaryCondition> = bottom
            .iter()
            .map(|&v| BoundaryCondition::Dirichlet {
                vertex: v,
                value: pin_vals[c],
            })
            .collect();
        let mut u_c: Vec<Real> = (0..nv).map(|i| u3_init[i * dim + c]).collect();
        for _ in 0..n_steps {
            u_c = solve_transient(&m, &k, &f, &u_c, dt, &bcs_c);
        }
        for i in 0..nv {
            max_du = max_du.max((u_c[i] - u3_gpu[i * dim + c]).abs());
        }
    }
    eprintln!(
        "gpu_transient_vector_dirichlet: nv = {nv}, dim = {dim}, {n_steps} steps, \
         max |Δu| = {max_du:.3e}",
    );
    assert!(max_du < 1e-3, "final max |Δu| = {max_du:.3e}");

    // The truncation bug's smoking gun: every pinned component must
    // hold its own value, not component 0's.
    for &v in &bottom {
        for c in 0..dim {
            let got = u3_gpu[v as usize * dim + c];
            assert!(
                (got - pin_vals[c]).abs() < 1e-5,
                "pinned component must hold: vertex {v} comp {c} = {got}, want {}",
                pin_vals[c]
            );
        }
    }
}
