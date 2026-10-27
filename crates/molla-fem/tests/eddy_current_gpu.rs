//! GPU eddy-current steady solve vs the CPU `solve_eddy_current_steady`.
//!
//! Run with: `make test-gpu P=molla-fem T=eddy_current_gpu`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_fem::eddy_current::{EddyCurrentConfig, solve_eddy_current_steady};
use molla_fem::eddy_current_gpu::solve_eddy_current_steady_gpu;
use molla_fem::magneto::compute_curl_per_cell;
use molla_fem::mesh::{FemCell, FemMesh};
use molla_fem::nedelec_element::NedelecDofMap;
use molla_math::Vec3;

fn centroid(cell: &FemCell, verts: &[Vec3]) -> Vec3 {
    let FemCell::Tet(v) = cell else {
        return Vec3::ZERO;
    };
    (verts[v[0] as usize] + verts[v[1] as usize] + verts[v[2] as usize] + verts[v[3] as usize])
        / 4.0
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_eddy_current_matches_cpu() {
    let mesh = FemMesh::tet_grid_3d(3, 3, 3, 1.0, 1.0, 1.0);
    let n_cells = mesh.cells.len();

    let mut j = vec![Vec3::ZERO; n_cells];
    for (i, cell) in mesh.cells.iter().enumerate() {
        let c = centroid(cell, &mesh.vertices);
        if c.z > 0.3 && c.z < 0.7 {
            j[i] = Vec3::new(0.0, 1.0, 0.0);
        }
    }
    let nu = vec![1.0; n_cells];
    let sigma = vec![1.0; n_cells]; // σ ≠ 0 → genuine coupled eddy system
    let j_zero = vec![Vec3::ZERO; n_cells];
    let cfg = EddyCurrentConfig {
        omega: 1.5,
        ..Default::default()
    };

    let cpu = solve_eddy_current_steady(&mesh, &j, &j_zero, &nu, &sigma, &[], &cfg).expect("cpu");

    let engine = Engine::new_gpu().expect("gpu");
    let gpu = solve_eddy_current_steady_gpu(&engine, &mesh, &j, &j_zero, &nu, &sigma, &[], &cfg)
        .expect("gpu");

    // Compare the gauge-invariant curl of both real and imaginary parts.
    let map = NedelecDofMap::build(&mesh);
    for (label, cpu_a, gpu_a) in [("re", &cpu.a_re, &gpu.a_re), ("im", &cpu.a_im, &gpu.a_im)] {
        let bc = compute_curl_per_cell(&mesh, &map, cpu_a);
        let bg = compute_curl_per_cell(&mesh, &map, gpu_a);
        let mut max_err = 0.0f64;
        let mut scale = 0.0f64;
        for (c, g) in bc.iter().zip(bg.iter()) {
            max_err = max_err.max((*c - *g).length());
            scale = scale.max(c.length());
        }
        let rel = max_err / scale.max(1.0e-30);
        assert!(
            rel < 2.0e-3,
            "curl({label}) rel mismatch {rel} (abs {max_err})"
        );
    }
    // Non-degenerate: the real curl field is non-trivial.
    let b_re = compute_curl_per_cell(&mesh, &map, &cpu.a_re);
    assert!(
        b_re.iter().map(|v| v.length()).fold(0.0, f64::max) > 1.0e-3,
        "degenerate: curl ~ 0",
    );
}
