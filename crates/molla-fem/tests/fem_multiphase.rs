//! CPU oracle gates for level-set / continuum-surface-force multiphase pieces.
//!
//! These are reduced physics checks, not a full two-phase Navier-Stokes solve:
//! they keep the curvature/delta/CSF building blocks honest until a larger
//! coupled solver or GPU path is worth adding.

use molla_fem::csf_surface_tension_force_2d;
use molla_math::Real;

fn circle_level_set(n: usize, radius: Real) -> (Vec<Real>, Real) {
    let dx = 1.0 / n as Real;
    let mut phi = vec![0.0; n * n];
    for j in 0..n {
        let y = (j as Real + 0.5) * dx - 0.5;
        for i in 0..n {
            let x = (i as Real + 0.5) * dx - 0.5;
            phi[j * n + i] = (x * x + y * y).sqrt() - radius;
        }
    }
    (phi, dx)
}

#[test]
fn level_set_csf_force_is_interface_local_symmetric_and_sigma_linear() {
    let n = 64usize;
    let radius = 0.21;
    let (phi, dx) = circle_level_set(n, radius);
    let sigma = 0.7;
    let f = csf_surface_tension_force_2d(&phi, n, n, dx, sigma).unwrap();
    let f2 = csf_surface_tension_force_2d(&phi, n, n, dx, 2.0 * sigma).unwrap();

    let mut net = [0.0, 0.0];
    let mut net_abs = 0.0;
    let mut interface_count = 0usize;
    let mut far_max = 0.0;
    let mut sigma_rel_err: Real = 0.0;
    let eps = 1.5 * dx;
    for (k, force) in f.iter().enumerate() {
        let mag = (force[0] * force[0] + force[1] * force[1]).sqrt();
        net[0] += force[0];
        net[1] += force[1];
        net_abs += mag;
        if phi[k].abs() < eps {
            interface_count += 1;
            let doubled = [2.0 * force[0], 2.0 * force[1]];
            let err = ((f2[k][0] - doubled[0]).abs() + (f2[k][1] - doubled[1]).abs())
                / (doubled[0].abs() + doubled[1].abs()).max(1.0e-12);
            sigma_rel_err = sigma_rel_err.max(err);
        } else if phi[k].abs() > 2.5 * eps {
            far_max = f64::max(far_max, mag);
        }
    }

    let net_mag = (net[0] * net[0] + net[1] * net[1]).sqrt();
    let right_i = ((0.5 + radius) / dx).floor() as usize;
    let center_j = n / 2;
    let right_force = f[center_j * n + right_i];
    let left_i = ((0.5 - radius) / dx).floor() as usize;
    let left_force = f[center_j * n + left_i];

    eprintln!(
        "CSF circle: interface_count={interface_count}, net_mag={net_mag:.4e}, net_abs={net_abs:.4e}, far_max={far_max:.4e}, sigma_rel_err={sigma_rel_err:.4e}, right={right_force:?}, left={left_force:?}"
    );
    assert!(
        interface_count > n,
        "CSF force should be active on the circular interface band"
    );
    assert!(
        net_mag < net_abs * 0.04,
        "symmetric circle should have near-zero net CSF force: net={net:?}, net_abs={net_abs}"
    );
    assert!(
        far_max < 1.0e-12,
        "CSF force should be compactly supported near the level-set interface: far_max={far_max}"
    );
    assert!(
        sigma_rel_err < 1.0e-12,
        "CSF force should scale linearly with surface tension σ: rel_err={sigma_rel_err}"
    );
    assert!(
        right_force[0] > 0.0 && left_force[0] < 0.0,
        "circle CSF force should point along ±normal on opposite sides: right={right_force:?}, left={left_force:?}"
    );
}
