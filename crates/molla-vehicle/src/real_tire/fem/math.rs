//! Tetra-FEM math, mesh build, and constraint-projection helpers.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use super::super::internal::{add, cross, dot, finite_or, finite_vec3, normalize, scale, sub};
use super::super::{SidewallDihedral, StructuralEdge, TetraElement, tet_is_sidewall_laminate};

pub(crate) fn quad_dihedral_angle(points: &[[f64; 3]], ids: [usize; 4]) -> f64 {
    let [a, b, c, d] = ids;
    let n0 = normalize(cross(sub(points[b], points[a]), sub(points[c], points[a])));
    let n1 = normalize(cross(sub(points[c], points[a]), sub(points[d], points[a])));
    dot(n0, n1).clamp(-1.0, 1.0).acos()
}

pub(crate) fn mat_vec(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

pub(crate) fn add_holzapfel_fiber_stress(
    p_stress: &mut [[f64; 3]; 3],
    f_def: [[f64; 3]; 3],
    a0: [f64; 3],
    k1: f64,
    k2: f64,
) {
    if k1 <= 0.0 {
        return;
    }
    let f_a0 = mat_vec(f_def, a0);
    let i4 = dot(f_a0, f_a0);
    if i4 <= 1.0 {
        return;
    }
    let stretch = i4 - 1.0;
    let exp_factor = if k2 > 0.0 {
        (k2 * stretch * stretch).min(30.0).exp()
    } else {
        1.0
    };
    let coeff = 2.0 * k1 * stretch * exp_factor;
    // P_aniso[i,j] = coeff · (F·a₀)_i · a₀[j]
    for r in 0..3 {
        for c in 0..3 {
            p_stress[r][c] += coeff * f_a0[r] * a0[c];
        }
    }
}

pub(crate) fn det3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

pub(crate) fn inv3(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let d = det3(m);
    if d.abs() < 1.0e-12 {
        return None;
    }
    let inv_d = 1.0 / d;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_d,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_d,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_d,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_d,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_d,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_d,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_d,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_d,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_d,
        ],
    ])
}

pub(crate) fn transpose3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

pub(crate) fn lame_from_young_poisson(young: f64, poisson: f64) -> (f64, f64) {
    let nu = poisson.clamp(-0.9, 0.49);
    let mu = young / (2.0 * (1.0 + nu));
    let lambda = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    (mu, lambda)
}

pub(crate) fn make_tetra_element(
    mut ids: [usize; 4],
    rest: &[[f64; 3]],
    mu: f64,
    lambda: f64,
    fiber_dir: [f64; 3],
    fiber_k1: f64,
    fiber_k2: f64,
    fiber_dir_b: [f64; 3],
    fiber_k1_b: f64,
    fiber_k2_b: f64,
    fiber_dir_c: [f64; 3],
    fiber_k1_c: f64,
    fiber_k2_c: f64,
    fiber_dir_d: [f64; 3],
    fiber_k1_d: f64,
    fiber_k2_d: f64,
) -> Option<TetraElement> {
    let mut dm = tetra_dm(ids, rest);
    let mut det = det3(dm);
    if det.abs() < 1.0e-12 {
        return None;
    }
    if det < 0.0 {
        ids.swap(2, 3);
        dm = tetra_dm(ids, rest);
        det = det3(dm);
    }
    let inv = inv3(dm)?;
    let g1 = [inv[0][0], inv[0][1], inv[0][2]];
    let g2 = [inv[1][0], inv[1][1], inv[1][2]];
    let g3 = [inv[2][0], inv[2][1], inv[2][2]];
    let g0 = scale(add(add(g1, g2), g3), -1.0);
    Some(TetraElement {
        ids,
        grads: [g0, g1, g2, g3],
        volume: det.abs() / 6.0,
        mu,
        lambda,
        fiber_dir,
        fiber_k1,
        fiber_k2,
        fiber_dir_b,
        fiber_k1_b,
        fiber_k2_b,
        fiber_dir_c,
        fiber_k1_c,
        fiber_k2_c,
        fiber_dir_d,
        fiber_k1_d,
        fiber_k2_d,
    })
}

pub(crate) fn tetra_dm(ids: [usize; 4], rest: &[[f64; 3]]) -> [[f64; 3]; 3] {
    let x0 = rest[ids[0]];
    let x1 = rest[ids[1]];
    let x2 = rest[ids[2]];
    let x3 = rest[ids[3]];
    [
        [x1[0] - x0[0], x2[0] - x0[0], x3[0] - x0[0]],
        [x1[1] - x0[1], x2[1] - x0[1], x3[1] - x0[1]],
        [x1[2] - x0[2], x2[2] - x0[2], x3[2] - x0[2]],
    ]
}

pub(crate) fn tetra_signed_volume(ids: [usize; 4], pos: &[[f64; 3]]) -> f64 {
    det3(tetra_dm(ids, pos)) / 6.0
}

pub(crate) fn project_edge_length_constraints(
    pos: &mut [[f64; 3]],
    vel: &mut [[f64; 3]],
    mass: &[f64],
    edges: &[StructuralEdge],
    compliance: f64,
    dt: f64,
) -> f64 {
    if edges.is_empty() || compliance.is_infinite() || dt <= 0.0 {
        return 0.0;
    }
    let alpha = compliance.max(0.0) / (dt * dt).max(1.0e-12);
    let mut max_correction: f64 = 0.0;
    for edge in edges {
        let [a, b] = edge.ids;
        if a >= pos.len() || b >= pos.len() || a >= mass.len() || b >= mass.len() {
            continue;
        }
        let dx = sub(pos[b], pos[a]);
        let len = dot(dx, dx).sqrt();
        if len <= 1.0e-9 || !len.is_finite() {
            continue;
        }
        let c = len - edge.rest_length;
        if c.abs() <= edge.rest_length.max(1.0e-9) * 1.0e-5 {
            continue;
        }
        let n = scale(dx, 1.0 / len);
        let wa = 1.0 / mass[a].max(1.0e-9);
        let wb = 1.0 / mass[b].max(1.0e-9);
        let denom = wa + wb + alpha;
        if denom <= 1.0e-18 || !denom.is_finite() {
            continue;
        }
        let lambda = -c / denom;
        let dpa = scale(n, -wa * lambda);
        let dpb = scale(n, wb * lambda);
        max_correction = max_correction
            .max(dot(dpa, dpa).sqrt())
            .max(dot(dpb, dpb).sqrt());
        pos[a] = add(pos[a], dpa);
        pos[b] = add(pos[b], dpb);
        for (axis, component) in dpa.iter().enumerate() {
            vel[a][axis] += component / dt;
            vel[a][axis] = finite_or(vel[a][axis], 0.0).clamp(-12.0, 12.0);
        }
        for (axis, component) in dpb.iter().enumerate() {
            vel[b][axis] += component / dt;
            vel[b][axis] = finite_or(vel[b][axis], 0.0).clamp(-12.0, 12.0);
        }
    }
    max_correction
}

pub(crate) fn quad_dihedral_angle_local(p: &[[f64; 3]; 4]) -> f64 {
    let n0 = normalize(cross(sub(p[1], p[0]), sub(p[2], p[0])));
    let n1 = normalize(cross(sub(p[2], p[0]), sub(p[3], p[0])));
    dot(n0, n1).clamp(-1.0, 1.0).acos()
}

pub(crate) fn project_sidewall_dihedral_constraints(
    pos: &mut [[f64; 3]],
    vel: &mut [[f64; 3]],
    mass: &[f64],
    dihedrals: &[SidewallDihedral],
    compliance: f64,
    dt: f64,
) -> f64 {
    if compliance.is_infinite() || dt <= 0.0 || dihedrals.is_empty() {
        return 0.0;
    }
    let alpha = compliance.max(0.0) / (dt * dt).max(1.0e-12);
    let eps = 1.0e-5;
    let mut max_correction: f64 = 0.0;
    for diag in dihedrals {
        let ids = diag.ids;
        let mut local = [pos[ids[0]], pos[ids[1]], pos[ids[2]], pos[ids[3]]];
        if !local.iter().all(|p| finite_vec3(*p)) {
            continue;
        }
        let angle = quad_dihedral_angle_local(&local);
        let c = angle - diag.rest_angle_rad;
        if !c.is_finite() || c.abs() < 1.0e-5 {
            continue;
        }
        let mut grads = [[0.0; 3]; 4];
        for vi in 0..4 {
            for axis in 0..3 {
                local[vi][axis] += eps;
                let plus = quad_dihedral_angle_local(&local);
                local[vi][axis] -= 2.0 * eps;
                let minus = quad_dihedral_angle_local(&local);
                local[vi][axis] += eps;
                grads[vi][axis] = finite_or((plus - minus) / (2.0 * eps), 0.0).clamp(-1.0e4, 1.0e4);
            }
        }
        let mut denom = alpha;
        for vi in 0..4 {
            denom += (1.0 / mass[ids[vi]].max(1.0e-9)) * dot(grads[vi], grads[vi]);
        }
        if denom <= 1.0e-18 || !denom.is_finite() {
            continue;
        }
        let lambda = -c / denom;
        if !lambda.is_finite() {
            continue;
        }
        for vi in 0..4 {
            let id = ids[vi];
            let inv_m = 1.0 / mass[id].max(1.0e-9);
            let dp = scale(grads[vi], inv_m * lambda);
            if !finite_vec3(dp) {
                continue;
            }
            max_correction = max_correction.max(dot(dp, dp).sqrt());
            pos[id] = add(pos[id], dp);
            for (axis, component) in dp.iter().enumerate() {
                vel[id][axis] += component / dt;
                vel[id][axis] = finite_or(vel[id][axis], 0.0).clamp(-12.0, 12.0);
            }
        }
    }
    max_correction
}

pub(crate) fn project_sidewall_ply_strain_constraints(
    pos: &mut [[f64; 3]],
    vel: &mut [[f64; 3]],
    mass: &[f64],
    tets: &[TetraElement],
    rest: &[[f64; 3]],
    hub: [f64; 3],
    stretch_limit: f64,
    compliance: f64,
    dt: f64,
) -> f64 {
    if compliance.is_infinite() || dt <= 0.0 || tets.is_empty() {
        return 0.0;
    }
    let stretch_limit = finite_or(stretch_limit, 1.35).clamp(1.01, 3.0);
    let alpha = compliance.max(0.0) / (dt * dt).max(1.0e-12);
    let mut max_correction: f64 = 0.0;
    for tet in tets {
        if !tet_is_sidewall_laminate(tet, rest, hub) {
            continue;
        }
        let mut f_def = [[0.0; 3]; 3];
        for a in 0..4 {
            let id = tet.ids[a];
            if id >= pos.len() || id >= mass.len() {
                continue;
            }
            for row in 0..3 {
                for col in 0..3 {
                    f_def[row][col] += pos[id][row] * tet.grads[a][col];
                }
            }
        }
        if det3(f_def).abs() <= 1.0e-12 {
            continue;
        }
        for dir in [
            tet.fiber_dir,
            tet.fiber_dir_b,
            tet.fiber_dir_c,
            tet.fiber_dir_d,
        ] {
            let fa = mat_vec(f_def, dir);
            let stretch = dot(fa, fa).sqrt();
            let c = stretch - stretch_limit;
            if !c.is_finite() || c <= 1.0e-6 || stretch <= 1.0e-9 {
                continue;
            }
            let stretch_grad_dir = scale(fa, 1.0 / stretch);
            let mut grads = [[0.0; 3]; 4];
            let mut denom = alpha;
            for a in 0..4 {
                let id = tet.ids[a];
                if id >= mass.len() {
                    continue;
                }
                let scalar = dot(tet.grads[a], dir);
                grads[a] = scale(stretch_grad_dir, scalar);
                denom += (1.0 / mass[id].max(1.0e-9)) * dot(grads[a], grads[a]);
            }
            if denom <= 1.0e-18 || !denom.is_finite() {
                continue;
            }
            let lambda = -c / denom;
            if !lambda.is_finite() {
                continue;
            }
            for (a, grad) in grads.iter().enumerate() {
                let id = tet.ids[a];
                if id >= pos.len() || id >= mass.len() {
                    continue;
                }
                let inv_m = 1.0 / mass[id].max(1.0e-9);
                let dp = scale(*grad, inv_m * lambda);
                if !finite_vec3(dp) {
                    continue;
                }
                max_correction = max_correction.max(dot(dp, dp).sqrt());
                pos[id] = add(pos[id], dp);
                for (axis, component) in dp.iter().enumerate() {
                    vel[id][axis] += component / dt;
                    vel[id][axis] = finite_or(vel[id][axis], 0.0).clamp(-12.0, 12.0);
                }
            }
        }
    }
    max_correction
}

pub(crate) fn project_to_ground(v: [f64; 3], n: [f64; 3]) -> [f64; 3] {
    let d = dot(v, n);
    [v[0] - d * n[0], v[1] - d * n[1], v[2] - d * n[2]]
}
