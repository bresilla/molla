//! Newton-family report builders + supporting tetra-element / patch helpers
//! and the moving-flat-terrain fixture used by stiffness reports.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use crate::terrain::{FlatTerrain, Terrain};

use super::super::internal::{add, dot, finite_nonnegative, finite_or, finite_vec3, scale, sub};
use super::super::{
    TetraElement, TetraFemDiagonalNewtonReport, TetraFemElementNewtonReport,
    TetraFemHardeningEnvelopeReport, TetraFemMatrixFreeNewtonPreconditionerReport,
    TetraFemMatrixFreeNewtonScaleReport, TetraFemMatrixFreeNewtonSolveReport,
    TetraFemMatrixFreeNewtonStepReport, TetraFemPatchNewtonReport, TetraFemSolverStiffnessReport,
    TetraFemTinyMeshNewtonReport, TetraTireFem, TetraTireFemParams, add_holzapfel_fiber_stress,
    det3, inv3, lame_from_young_poisson, make_tetra_element, mat_vec, tetra_signed_volume,
    tire_nodes_finite, transpose3,
};

fn tire_max_abs_node_position(tire: &TetraTireFem) -> f64 {
    tire.nodes()
        .iter()
        .flat_map(|p| p.iter())
        .map(|v| v.abs())
        .filter(|v| v.is_finite())
        .fold(0.0_f64, f64::max)
}

fn tire_kinetic_energy(tire: &TetraTireFem) -> f64 {
    tire.mass
        .iter()
        .zip(tire.vel.iter())
        .map(|(m, v)| 0.5 * m * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
        .filter(|e| e.is_finite())
        .sum()
}

#[derive(Clone, Copy, Debug)]
struct TireSolverMovingFlatTerrain {
    speed_x: f64,
    friction: f64,
}

impl Terrain for TireSolverMovingFlatTerrain {
    fn height(&self, _x: f64, _z: f64) -> f64 {
        0.0
    }

    fn normal(&self, _x: f64, _z: f64) -> [f64; 3] {
        [0.0, 1.0, 0.0]
    }

    fn friction(&self) -> f64 {
        finite_nonnegative(self.friction)
    }

    fn surface_velocity(&self, _x: f64, _z: f64) -> [f64; 3] {
        [finite_or(self.speed_x, 0.0), 0.0, 0.0]
    }
}

/// Build a deterministic report for the current semi-implicit stiffness guard.
pub fn tetra_fem_solver_stiffness_report() -> TetraFemSolverStiffnessReport {
    let make_stiff_tire = |semi_implicit_stiffness_scale: f64| {
        let mut tire = TetraTireFem::new(
            0.0,
            360_000.0,
            TetraTireFemParams {
                segments: 12,
                radial_layers: 3,
                width_layers: 3,
                pressure_pa: 360_000.0,
                young_modulus: 1.4e7,
                sidewall_young_modulus: 9.0e6,
                tread_young_modulus: 2.2e7,
                bead_young_modulus: 4.0e7,
                node_damping: 2.0,
                xpbd_volume_iterations: 0,
                xpbd_structural_iterations: 0,
                semi_implicit_stiffness_scale,
                ..TetraTireFemParams::high_pressure_reference()
            },
        );
        for (i, p) in tire.pos.iter_mut().enumerate() {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            p[0] += 0.025 * sign;
            p[1] += 0.018 * sign;
        }
        tire
    };

    let mut explicit = make_stiff_tire(0.0);
    let mut semi = make_stiff_tire(1.5);
    let free_space = FlatTerrain::new(-10.0, 1.0);
    explicit.step_on_terrain(&free_space, 1.0 / 120.0);
    semi.step_on_terrain(&free_space, 1.0 / 120.0);

    let mut high_pressure = TetraTireFem::new(
        0.0,
        420_000.0,
        TetraTireFemParams {
            segments: 16,
            radial_layers: 3,
            width_layers: 3,
            downforce: 3_500.0,
            pressure_pa: 420_000.0,
            ..TetraTireFemParams::high_pressure_reference()
        },
    );
    let rolling_terrain = TireSolverMovingFlatTerrain {
        speed_x: 28.0,
        friction: 1.0,
    };
    for _ in 0..240 {
        high_pressure.step_on_terrain(&rolling_terrain, 1.0 / 480.0);
    }
    let high_pressure_max_radius = high_pressure
        .tread_nodes()
        .iter()
        .map(|&id| {
            let p = high_pressure.nodes()[id];
            ((p[0] - high_pressure.hub()[0]).powi(2) + (p[1] - high_pressure.hub()[1]).powi(2))
                .sqrt()
        })
        .filter(|v| v.is_finite())
        .fold(0.0_f64, f64::max);
    let explicit_speed = explicit.max_node_speed_m_per_s();
    let semi_speed = semi.max_node_speed_m_per_s();
    let finite_outputs = tire_nodes_finite(&explicit)
        && tire_nodes_finite(&semi)
        && tire_nodes_finite(&high_pressure);
    TetraFemSolverStiffnessReport {
        explicit_max_node_speed_m_per_s: explicit_speed,
        semi_implicit_max_node_speed_m_per_s: semi_speed,
        speed_ratio_semi_over_explicit: semi_speed / explicit_speed.max(1.0e-9),
        semi_implicit_stiffness_scale: semi.semi_implicit_stiffness_scale(),
        high_pressure_max_node_speed_m_per_s: high_pressure.max_node_speed_m_per_s(),
        high_pressure_radius_ratio: high_pressure_max_radius
            / high_pressure.params.outer_radius.max(1.0e-9),
        high_pressure_contact_patch_m: high_pressure.contact_patch_width(),
        finite_outputs,
    }
}

fn tire_newton_mode_energy(mass: f64, k_linear: f64, k_cubic: f64, x: f64, v: f64) -> f64 {
    0.5 * mass * v * v + 0.5 * k_linear * x * x + 0.25 * k_cubic * x.powi(4)
}

fn tire_newton_mode_residual_norm(
    mass: f64,
    damping: f64,
    k_linear: f64,
    k_cubic: f64,
    external_force: f64,
    dt: f64,
    x0: f64,
    v0: f64,
    x: f64,
    v: f64,
) -> f64 {
    let r_position = x - x0 - dt * v;
    let internal_force = k_linear * x + k_cubic * x.powi(3);
    let r_momentum = mass * (v - v0) - dt * (external_force - damping * v - internal_force);
    (r_position * r_position + r_momentum * r_momentum).sqrt()
}

/// Build the first residual/Jacobian Newton solve gate for a stiff nonlinear
/// tire-FEM mode.
pub fn tetra_fem_diagonal_newton_report() -> TetraFemDiagonalNewtonReport {
    let dt_s = 1.0_f64 / 45.0;
    let mass_kg = 2.1_f64;
    let damping_n_s_per_m = 22.0_f64;
    let linear_stiffness_n_per_m = 1.5e5_f64;
    let cubic_stiffness_n_per_m3 = 2.5e7_f64;
    let external_force_n = -3_500.0_f64;
    let initial_displacement_m = 0.055_f64;
    let initial_velocity_m_per_s = 18.0_f64;

    let initial_internal_force = linear_stiffness_n_per_m * initial_displacement_m
        + cubic_stiffness_n_per_m3 * initial_displacement_m.powi(3);
    let explicit_accel =
        (external_force_n - damping_n_s_per_m * initial_velocity_m_per_s - initial_internal_force)
            / mass_kg;
    let explicit_velocity_m_per_s = initial_velocity_m_per_s + explicit_accel * dt_s;
    let explicit_displacement_m = initial_displacement_m + initial_velocity_m_per_s * dt_s;

    // Start Newton from the explicit predictor, then solve the coupled
    // backward-Euler displacement/momentum residual:
    //   r_x = xₙ₊₁ - xₙ - dt·vₙ₊₁
    //   r_v = m(vₙ₊₁-vₙ) - dt(Fext - c vₙ₊₁ - kxₙ₊₁ - qxₙ₊₁³)
    let mut x = explicit_displacement_m;
    let mut v = explicit_velocity_m_per_s;
    let initial_residual_norm = tire_newton_mode_residual_norm(
        mass_kg,
        damping_n_s_per_m,
        linear_stiffness_n_per_m,
        cubic_stiffness_n_per_m3,
        external_force_n,
        dt_s,
        initial_displacement_m,
        initial_velocity_m_per_s,
        x,
        v,
    );
    let mut min_tangent_n_per_m = f64::INFINITY;
    let mut iteration_count = 0usize;
    let mut converged = false;
    for iteration in 0..12 {
        let internal_force = linear_stiffness_n_per_m * x + cubic_stiffness_n_per_m3 * x.powi(3);
        let tangent = linear_stiffness_n_per_m + 3.0 * cubic_stiffness_n_per_m3 * x * x;
        min_tangent_n_per_m = min_tangent_n_per_m.min(tangent);
        let r_position = x - initial_displacement_m - dt_s * v;
        let r_momentum = mass_kg * (v - initial_velocity_m_per_s)
            - dt_s * (external_force_n - damping_n_s_per_m * v - internal_force);
        let residual_norm = (r_position * r_position + r_momentum * r_momentum).sqrt();
        iteration_count = iteration + 1;
        if residual_norm <= 1.0e-9 {
            converged = true;
            break;
        }

        let a = 1.0;
        let b = -dt_s;
        let c = dt_s * tangent;
        let d = mass_kg + dt_s * damping_n_s_per_m;
        let det = a * d - b * c;
        if det.abs() <= 1.0e-18 || !det.is_finite() {
            break;
        }
        let dx = (-r_position * d + b * r_momentum) / det;
        let dv = (c * r_position - a * r_momentum) / det;
        x += dx;
        v += dv;
        if !x.is_finite() || !v.is_finite() {
            break;
        }
    }
    let final_residual_norm = tire_newton_mode_residual_norm(
        mass_kg,
        damping_n_s_per_m,
        linear_stiffness_n_per_m,
        cubic_stiffness_n_per_m3,
        external_force_n,
        dt_s,
        initial_displacement_m,
        initial_velocity_m_per_s,
        x,
        v,
    );
    converged |= final_residual_norm <= 1.0e-7;
    let explicit_energy_j = tire_newton_mode_energy(
        mass_kg,
        linear_stiffness_n_per_m,
        cubic_stiffness_n_per_m3,
        explicit_displacement_m,
        explicit_velocity_m_per_s,
    );
    let implicit_energy_j = tire_newton_mode_energy(
        mass_kg,
        linear_stiffness_n_per_m,
        cubic_stiffness_n_per_m3,
        x,
        v,
    );

    TetraFemDiagonalNewtonReport {
        dt_s,
        mass_kg,
        damping_n_s_per_m,
        linear_stiffness_n_per_m,
        cubic_stiffness_n_per_m3,
        external_force_n,
        initial_displacement_m,
        initial_velocity_m_per_s,
        explicit_displacement_m,
        explicit_velocity_m_per_s,
        newton_displacement_m: x,
        newton_velocity_m_per_s: v,
        initial_residual_norm,
        final_residual_norm,
        residual_reduction_ratio: final_residual_norm / initial_residual_norm.max(1.0e-30),
        iteration_count,
        converged,
        min_tangent_n_per_m,
        explicit_energy_j,
        implicit_energy_j,
        energy_ratio_implicit_over_explicit: implicit_energy_j / explicit_energy_j.max(1.0e-30),
        finite_outputs: explicit_energy_j.is_finite()
            && implicit_energy_j.is_finite()
            && final_residual_norm.is_finite()
            && initial_residual_norm.is_finite()
            && min_tangent_n_per_m.is_finite(),
    }
}

fn flatten_tetra_nodes(nodes: [[f64; 3]; 4]) -> [f64; 12] {
    let mut out = [0.0; 12];
    for a in 0..4 {
        for c in 0..3 {
            out[a * 3 + c] = nodes[a][c];
        }
    }
    out
}

fn unflatten_tetra_nodes(values: &[f64; 12]) -> [[f64; 3]; 4] {
    let mut out = [[0.0; 3]; 4];
    for a in 0..4 {
        for c in 0..3 {
            out[a][c] = values[a * 3 + c];
        }
    }
    out
}

fn tetra_element_internal_forces(tet: &TetraElement, pos: [[f64; 3]; 4]) -> [[f64; 3]; 4] {
    let mut f_def = [[0.0; 3]; 3];
    for (a, p) in pos.iter().enumerate() {
        for row in 0..3 {
            for col in 0..3 {
                f_def[row][col] += p[row] * tet.grads[a][col];
            }
        }
    }
    let Some(inv_f) = inv3(f_def) else {
        return [[0.0; 3]; 4];
    };
    let f_inv_t = transpose3(inv_f);
    let j = det3(f_def).max(0.20);
    let ln_j = j.ln();
    let mut p_stress = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            p_stress[r][c] =
                tet.mu * (f_def[r][c] - f_inv_t[r][c]) + tet.lambda * ln_j * f_inv_t[r][c];
        }
    }
    add_holzapfel_fiber_stress(
        &mut p_stress,
        f_def,
        tet.fiber_dir,
        tet.fiber_k1,
        tet.fiber_k2,
    );
    add_holzapfel_fiber_stress(
        &mut p_stress,
        f_def,
        tet.fiber_dir_b,
        tet.fiber_k1_b,
        tet.fiber_k2_b,
    );
    add_holzapfel_fiber_stress(
        &mut p_stress,
        f_def,
        tet.fiber_dir_c,
        tet.fiber_k1_c,
        tet.fiber_k2_c,
    );
    add_holzapfel_fiber_stress(
        &mut p_stress,
        f_def,
        tet.fiber_dir_d,
        tet.fiber_k1_d,
        tet.fiber_k2_d,
    );
    let mut forces = [[0.0; 3]; 4];
    for (a, force) in forces.iter_mut().enumerate() {
        *force = scale(mat_vec(p_stress, tet.grads[a]), -tet.volume);
    }
    forces
}

fn tetra_element_elastic_energy(tet: &TetraElement, pos: [[f64; 3]; 4]) -> f64 {
    let mut f_def = [[0.0; 3]; 3];
    for (a, p) in pos.iter().enumerate() {
        for row in 0..3 {
            for col in 0..3 {
                f_def[row][col] += p[row] * tet.grads[a][col];
            }
        }
    }
    let j = det3(f_def).max(0.20);
    let ln_j = j.ln();
    let mut i_c = 0.0;
    for row in &f_def {
        for value in row {
            i_c += value * value;
        }
    }
    let psi = 0.5 * tet.mu * (i_c - 3.0) - tet.mu * ln_j + 0.5 * tet.lambda * ln_j * ln_j;
    finite_nonnegative(psi * tet.volume)
}

fn tetra_element_newton_residual(
    tet: &TetraElement,
    x0: [[f64; 3]; 4],
    v0: [[f64; 3]; 4],
    x: [f64; 12],
    mass_per_node: f64,
    damping: f64,
    external: [[f64; 3]; 4],
    dt: f64,
) -> [f64; 12] {
    let pos = unflatten_tetra_nodes(&x);
    let internal = tetra_element_internal_forces(tet, pos);
    let mut residual = [0.0; 12];
    for a in 0..4 {
        for c in 0..3 {
            let idx = a * 3 + c;
            let v = (pos[a][c] - x0[a][c]) / dt;
            residual[idx] = mass_per_node * (v - v0[a][c])
                - dt * (internal[a][c] + external[a][c] - damping * v);
        }
    }
    residual
}

fn vec12_norm(values: &[f64; 12]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn vec12_max_abs(values: &[f64; 12]) -> f64 {
    values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
}

fn solve_12x12(mut a: [[f64; 12]; 12], mut b: [f64; 12]) -> Option<([f64; 12], f64)> {
    let mut min_pivot_abs = f64::INFINITY;
    for col in 0..12 {
        let mut pivot = col;
        let mut pivot_abs = a[col][col].abs();
        for row in (col + 1)..12 {
            let value = a[row][col].abs();
            if value > pivot_abs {
                pivot = row;
                pivot_abs = value;
            }
        }
        if pivot_abs <= 1.0e-14 || !pivot_abs.is_finite() {
            return None;
        }
        min_pivot_abs = min_pivot_abs.min(pivot_abs);
        if pivot != col {
            a.swap(pivot, col);
            b.swap(pivot, col);
        }
        let pivot_value = a[col][col];
        for row in (col + 1)..12 {
            let factor = a[row][col] / pivot_value;
            a[row][col] = 0.0;
            for k in (col + 1)..12 {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0; 12];
    for row in (0..12).rev() {
        let mut rhs = b[row];
        for col in (row + 1)..12 {
            rhs -= a[row][col] * x[col];
        }
        x[row] = rhs / a[row][row];
        if !x[row].is_finite() {
            return None;
        }
    }
    Some((x, min_pivot_abs))
}

fn tetra_element_total_energy(
    tet: &TetraElement,
    x0: [[f64; 3]; 4],
    pos: [[f64; 3]; 4],
    mass_per_node: f64,
    dt: f64,
) -> f64 {
    let kinetic = pos
        .iter()
        .zip(x0.iter())
        .map(|(p, p0)| {
            let v = [
                (p[0] - p0[0]) / dt,
                (p[1] - p0[1]) / dt,
                (p[2] - p0[2]) / dt,
            ];
            0.5 * mass_per_node * dot(v, v)
        })
        .sum::<f64>();
    kinetic + tetra_element_elastic_energy(tet, pos)
}

/// Build the first coupled element-level Newton solve gate for the tire FEM.
pub fn tetra_fem_element_newton_report() -> TetraFemElementNewtonReport {
    let rest = [
        [0.0, 0.0, 0.0],
        [0.090, 0.0, 0.0],
        [0.0, 0.070, 0.0],
        [0.0, 0.0, 0.055],
    ];
    let (mu, lambda) = lame_from_young_poisson(9.0e6, 0.46);
    let tet = make_tetra_element(
        [0, 1, 2, 3],
        &rest,
        mu,
        lambda,
        [0.0; 3],
        0.0,
        0.0,
        [0.0; 3],
        0.0,
        0.0,
        [0.0; 3],
        0.0,
        0.0,
        [0.0; 3],
        0.0,
        0.0,
    )
    .expect("fixture tetra should be valid");
    let x0 = [
        [0.000, 0.002, 0.000],
        [0.105, -0.006, 0.004],
        [-0.007, 0.054, 0.006],
        [0.005, -0.004, 0.071],
    ];
    let v0 = [
        [1.4, -1.8, 0.5],
        [-1.2, 1.5, -0.4],
        [0.8, -1.1, 0.7],
        [-0.9, 1.3, -0.6],
    ];
    let external = [
        [0.0, -120.0, 0.0],
        [45.0, -160.0, -20.0],
        [-35.0, -140.0, 25.0],
        [15.0, -135.0, -10.0],
    ];
    let dt_s = 1.0 / 90.0;
    let mass_per_node = 1.15;
    let damping = 18.0;
    let mut x = flatten_tetra_nodes([
        add(x0[0], scale(v0[0], dt_s)),
        add(x0[1], scale(v0[1], dt_s)),
        add(x0[2], scale(v0[2], dt_s)),
        add(x0[3], scale(v0[3], dt_s)),
    ]);
    let initial_residual =
        tetra_element_newton_residual(&tet, x0, v0, x, mass_per_node, damping, external, dt_s);
    let initial_residual_norm = vec12_norm(&initial_residual);
    let explicit_pos = unflatten_tetra_nodes(&x);
    let explicit_energy_j = tetra_element_total_energy(&tet, x0, explicit_pos, mass_per_node, dt_s);
    let mut min_pivot_abs = f64::INFINITY;
    let mut max_newton_correction_m: f64 = 0.0;
    let mut jacobian_offdiag_ratio: f64 = 0.0;
    let mut iteration_count = 0usize;
    let mut converged = false;
    for iteration in 0..12 {
        let residual =
            tetra_element_newton_residual(&tet, x0, v0, x, mass_per_node, damping, external, dt_s);
        let residual_norm = vec12_norm(&residual);
        iteration_count = iteration + 1;
        if residual_norm <= 1.0e-9 {
            converged = true;
            break;
        }

        let mut jac = [[0.0; 12]; 12];
        for col in 0..12 {
            let eps = 1.0e-6 * x[col].abs().max(1.0);
            let mut xp = x;
            xp[col] += eps;
            let rp = tetra_element_newton_residual(
                &tet,
                x0,
                v0,
                xp,
                mass_per_node,
                damping,
                external,
                dt_s,
            );
            for row in 0..12 {
                jac[row][col] = (rp[row] - residual[row]) / eps;
            }
        }
        let diag_sum = (0..12).map(|i| jac[i][i].abs()).sum::<f64>().max(1.0e-30);
        let offdiag_sum = (0..12)
            .flat_map(|row| (0..12).map(move |col| (row, col)))
            .filter(|(row, col)| row != col)
            .map(|(row, col)| jac[row][col].abs())
            .sum::<f64>();
        jacobian_offdiag_ratio = jacobian_offdiag_ratio.max(offdiag_sum / diag_sum);
        let rhs = residual.map(|value| -value);
        let Some((dx, min_pivot)) = solve_12x12(jac, rhs) else {
            break;
        };
        min_pivot_abs = min_pivot_abs.min(min_pivot);
        max_newton_correction_m = max_newton_correction_m.max(vec12_max_abs(&dx));
        for i in 0..12 {
            x[i] += dx[i];
        }
        if !x.iter().all(|value| value.is_finite()) {
            break;
        }
    }
    let final_residual =
        tetra_element_newton_residual(&tet, x0, v0, x, mass_per_node, damping, external, dt_s);
    let final_residual_norm = vec12_norm(&final_residual);
    converged |= final_residual_norm <= 1.0e-6;
    let implicit_pos = unflatten_tetra_nodes(&x);
    let implicit_energy_j = tetra_element_total_energy(&tet, x0, implicit_pos, mass_per_node, dt_s);
    let final_volume_ratio = tetra_signed_volume([0, 1, 2, 3], &implicit_pos) / tet.volume;
    TetraFemElementNewtonReport {
        coupled_dof_count: 12,
        element_volume_m3: tet.volume,
        dt_s,
        young_modulus_pa: 9.0e6,
        poisson_ratio: 0.46,
        initial_residual_norm,
        final_residual_norm,
        residual_reduction_ratio: final_residual_norm / initial_residual_norm.max(1.0e-30),
        iteration_count,
        converged,
        max_newton_correction_m,
        min_pivot_abs,
        jacobian_offdiag_ratio,
        explicit_energy_j,
        implicit_energy_j,
        energy_ratio_implicit_over_explicit: implicit_energy_j / explicit_energy_j.max(1.0e-30),
        final_volume_ratio,
        finite_outputs: initial_residual_norm.is_finite()
            && final_residual_norm.is_finite()
            && min_pivot_abs.is_finite()
            && jacobian_offdiag_ratio.is_finite()
            && explicit_energy_j.is_finite()
            && implicit_energy_j.is_finite()
            && final_volume_ratio.is_finite()
            && x.iter().all(|value| value.is_finite()),
    }
}

pub(crate) fn patch_flatten_nodes(nodes: &[[f64; 3]]) -> Vec<f64> {
    let mut out = Vec::with_capacity(nodes.len() * 3);
    for node in nodes {
        out.extend_from_slice(node);
    }
    out
}

pub(crate) fn patch_unflatten_nodes(values: &[f64]) -> Vec<[f64; 3]> {
    values
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect()
}

fn patch_internal_forces(tets: &[TetraElement], pos: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut forces = vec![[0.0; 3]; pos.len()];
    for tet in tets {
        let local_pos = [
            pos[tet.ids[0]],
            pos[tet.ids[1]],
            pos[tet.ids[2]],
            pos[tet.ids[3]],
        ];
        let local_forces = tetra_element_internal_forces(tet, local_pos);
        for a in 0..4 {
            let id = tet.ids[a];
            forces[id] = add(forces[id], local_forces[a]);
        }
    }
    forces
}

fn patch_elastic_energy(tets: &[TetraElement], pos: &[[f64; 3]]) -> f64 {
    tets.iter()
        .map(|tet| {
            tetra_element_elastic_energy(
                tet,
                [
                    pos[tet.ids[0]],
                    pos[tet.ids[1]],
                    pos[tet.ids[2]],
                    pos[tet.ids[3]],
                ],
            )
        })
        .sum()
}

fn patch_total_energy(
    tets: &[TetraElement],
    x0: &[[f64; 3]],
    pos: &[[f64; 3]],
    mass_per_node: f64,
    dt: f64,
) -> f64 {
    let kinetic = pos
        .iter()
        .zip(x0.iter())
        .map(|(p, p0)| {
            let v = [
                (p[0] - p0[0]) / dt,
                (p[1] - p0[1]) / dt,
                (p[2] - p0[2]) / dt,
            ];
            0.5 * mass_per_node * dot(v, v)
        })
        .sum::<f64>();
    kinetic + patch_elastic_energy(tets, pos)
}

pub(crate) fn patch_total_energy_with_masses(
    tets: &[TetraElement],
    x0: &[[f64; 3]],
    pos: &[[f64; 3]],
    masses: &[f64],
    dt: f64,
) -> f64 {
    let kinetic = pos
        .iter()
        .zip(x0.iter())
        .enumerate()
        .map(|(id, (p, p0))| {
            let v = [
                (p[0] - p0[0]) / dt,
                (p[1] - p0[1]) / dt,
                (p[2] - p0[2]) / dt,
            ];
            0.5 * masses.get(id).copied().unwrap_or(1.0).max(1.0e-9) * dot(v, v)
        })
        .sum::<f64>();
    kinetic + patch_elastic_energy(tets, pos)
}

fn patch_newton_residual(
    tets: &[TetraElement],
    x0: &[[f64; 3]],
    v0: &[[f64; 3]],
    x: &[f64],
    mass_per_node: f64,
    damping: f64,
    external: &[[f64; 3]],
    dt: f64,
) -> Vec<f64> {
    let pos = patch_unflatten_nodes(x);
    let internal = patch_internal_forces(tets, &pos);
    let mut residual = vec![0.0; x.len()];
    for a in 0..pos.len() {
        for c in 0..3 {
            let idx = a * 3 + c;
            let v = (pos[a][c] - x0[a][c]) / dt;
            residual[idx] = mass_per_node * (v - v0[a][c])
                - dt * (internal[a][c] + external[a][c] - damping * v);
        }
    }
    residual
}

pub(crate) fn patch_newton_residual_with_masses(
    tets: &[TetraElement],
    x0: &[[f64; 3]],
    v0: &[[f64; 3]],
    x: &[f64],
    masses: &[f64],
    damping: f64,
    external: &[[f64; 3]],
    dt: f64,
) -> Vec<f64> {
    let pos = patch_unflatten_nodes(x);
    let internal = patch_internal_forces(tets, &pos);
    let mut residual = vec![0.0; x.len()];
    for a in 0..pos.len() {
        let mass = masses.get(a).copied().unwrap_or(1.0).max(1.0e-9);
        for c in 0..3 {
            let idx = a * 3 + c;
            let v = (pos[a][c] - x0[a][c]) / dt;
            residual[idx] =
                mass * (v - v0[a][c]) - dt * (internal[a][c] + external[a][c] - damping * v);
        }
    }
    residual
}

pub(crate) fn vec_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn vec_dot_slice(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(av, bv)| av * bv).sum::<f64>()
}

pub(crate) fn vec_max_abs(values: &[f64]) -> f64 {
    values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
}

fn solve_matrix_free_gmres<F>(
    mut apply: F,
    rhs: &[f64],
    max_iter: usize,
    relative_tolerance: f64,
) -> (Vec<f64>, f64, usize, usize, bool)
where
    F: FnMut(&[f64]) -> Vec<f64>,
{
    let n = rhs.len();
    let beta = vec_norm(rhs);
    if beta <= 1.0e-30 || !beta.is_finite() {
        return (vec![0.0; n], beta, 0, 0, true);
    }

    let mut basis = Vec::with_capacity(max_iter + 1);
    basis.push(rhs.iter().map(|value| value / beta).collect::<Vec<_>>());
    let mut h = vec![vec![0.0; max_iter]; max_iter + 1];
    let mut best_dx = vec![0.0; n];
    let mut best_residual_norm = beta;
    let mut best_iteration_count = 0usize;
    let mut apply_count = 0usize;

    for j in 0..max_iter {
        apply_count += 1;
        let mut w = apply(&basis[j]);
        if w.len() != n || !w.iter().all(|value| value.is_finite()) {
            break;
        }
        for i in 0..=j {
            h[i][j] = vec_dot_slice(&w, &basis[i]);
            for (wk, vik) in w.iter_mut().zip(basis[i].iter()) {
                *wk -= h[i][j] * vik;
            }
        }
        h[j + 1][j] = vec_norm(&w);
        if h[j + 1][j].is_finite() && h[j + 1][j] > 1.0e-14 && j + 1 < max_iter {
            basis.push(
                w.iter()
                    .map(|value| value / h[j + 1][j])
                    .collect::<Vec<_>>(),
            );
        }

        let k = j + 1;
        let mut normal = vec![vec![0.0; k]; k];
        let mut normal_rhs = vec![0.0; k];
        for row in 0..k {
            normal_rhs[row] = beta * h[0][row];
            for col in 0..k {
                normal[row][col] = (0..=k).map(|i| h[i][row] * h[i][col]).sum::<f64>();
            }
            normal[row][row] += 1.0e-18 * normal[row][row].abs().max(1.0);
        }
        let Some((y, _)) = solve_dense(normal, normal_rhs) else {
            break;
        };
        let mut dx = vec![0.0; n];
        for col in 0..k {
            for (dxi, vi) in dx.iter_mut().zip(basis[col].iter()) {
                *dxi += y[col] * vi;
            }
        }
        if !dx.iter().all(|value| value.is_finite()) {
            break;
        }

        apply_count += 1;
        let jdx = apply(&dx);
        if jdx.len() != n || !jdx.iter().all(|value| value.is_finite()) {
            break;
        }
        let linear_residual = rhs
            .iter()
            .zip(jdx.iter())
            .map(|(rv, av)| rv - av)
            .collect::<Vec<_>>();
        let linear_residual_norm = vec_norm(&linear_residual);
        if linear_residual_norm < best_residual_norm {
            best_residual_norm = linear_residual_norm;
            best_dx = dx;
            best_iteration_count = k;
        }
        if linear_residual_norm / beta <= relative_tolerance {
            return (
                best_dx,
                best_residual_norm,
                best_iteration_count,
                apply_count,
                true,
            );
        }
        if h[j + 1][j] <= 1.0e-14 || j + 1 >= max_iter {
            break;
        }
    }

    let converged = best_residual_norm / beta <= relative_tolerance;
    (
        best_dx,
        best_residual_norm,
        best_iteration_count,
        apply_count,
        converged,
    )
}

fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<(Vec<f64>, f64)> {
    let n = b.len();
    let mut min_pivot_abs = f64::INFINITY;
    for col in 0..n {
        let mut pivot = col;
        let mut pivot_abs = a[col][col].abs();
        for row in (col + 1)..n {
            let value = a[row][col].abs();
            if value > pivot_abs {
                pivot = row;
                pivot_abs = value;
            }
        }
        if pivot_abs <= 1.0e-14 || !pivot_abs.is_finite() {
            return None;
        }
        min_pivot_abs = min_pivot_abs.min(pivot_abs);
        if pivot != col {
            a.swap(pivot, col);
            b.swap(pivot, col);
        }
        let pivot_value = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot_value;
            a[row][col] = 0.0;
            for k in (col + 1)..n {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut rhs = b[row];
        for col in (row + 1)..n {
            rhs -= a[row][col] * x[col];
        }
        x[row] = rhs / a[row][row];
        if !x[row].is_finite() {
            return None;
        }
    }
    Some((x, min_pivot_abs))
}

/// Build a connected multi-tetra Newton solve gate for the tire FEM.
pub fn tetra_fem_patch_newton_report() -> TetraFemPatchNewtonReport {
    let rest = vec![
        [0.0, 0.0, 0.0],
        [0.090, 0.0, 0.0],
        [0.0, 0.075, 0.0],
        [0.0, 0.0, 0.065],
        [0.105, 0.080, 0.070],
    ];
    let (mu, lambda) = lame_from_young_poisson(8.5e6, 0.46);
    let tets = vec![
        make_tetra_element(
            [0, 1, 2, 3],
            &rest,
            mu,
            lambda,
            [0.0; 3],
            0.0,
            0.0,
            [0.0; 3],
            0.0,
            0.0,
            [0.0; 3],
            0.0,
            0.0,
            [0.0; 3],
            0.0,
            0.0,
        )
        .expect("fixture tetra 0 should be valid"),
        make_tetra_element(
            [1, 2, 3, 4],
            &rest,
            mu,
            lambda,
            [0.0; 3],
            0.0,
            0.0,
            [0.0; 3],
            0.0,
            0.0,
            [0.0; 3],
            0.0,
            0.0,
            [0.0; 3],
            0.0,
            0.0,
        )
        .expect("fixture tetra 1 should be valid"),
    ];
    let x0 = vec![
        [0.000, 0.002, 0.000],
        [0.103, -0.006, 0.004],
        [-0.007, 0.056, 0.006],
        [0.005, -0.004, 0.074],
        [0.111, 0.091, 0.061],
    ];
    let v0 = vec![
        [1.4, -1.7, 0.5],
        [-1.2, 1.5, -0.4],
        [0.8, -1.1, 0.7],
        [-0.9, 1.3, -0.6],
        [1.1, -1.4, 0.3],
    ];
    let external = vec![
        [0.0, -105.0, 0.0],
        [45.0, -155.0, -20.0],
        [-35.0, -135.0, 25.0],
        [15.0, -130.0, -10.0],
        [-25.0, -145.0, 30.0],
    ];
    let dt_s = 1.0 / 110.0;
    let mass_per_node = 1.10;
    let damping = 20.0;
    let predictor: Vec<[f64; 3]> = x0
        .iter()
        .zip(v0.iter())
        .map(|(p, v)| add(*p, scale(*v, dt_s)))
        .collect();
    let mut x = patch_flatten_nodes(&predictor);
    let initial_residual =
        patch_newton_residual(&tets, &x0, &v0, &x, mass_per_node, damping, &external, dt_s);
    let initial_residual_norm = vec_norm(&initial_residual);
    let explicit_energy_j = patch_total_energy(&tets, &x0, &predictor, mass_per_node, dt_s);
    let mut min_pivot_abs = f64::INFINITY;
    let mut max_newton_correction_m: f64 = 0.0;
    let mut jacobian_offdiag_ratio: f64 = 0.0;
    let mut iteration_count = 0usize;
    let mut converged = false;
    for iteration in 0..16 {
        let residual =
            patch_newton_residual(&tets, &x0, &v0, &x, mass_per_node, damping, &external, dt_s);
        let residual_norm = vec_norm(&residual);
        iteration_count = iteration + 1;
        if residual_norm <= 1.0e-8 {
            converged = true;
            break;
        }
        let n = x.len();
        let mut jac = vec![vec![0.0; n]; n];
        for col in 0..n {
            let eps = 1.0e-6 * x[col].abs().max(1.0);
            let mut xp = x.clone();
            xp[col] += eps;
            let rp = patch_newton_residual(
                &tets,
                &x0,
                &v0,
                &xp,
                mass_per_node,
                damping,
                &external,
                dt_s,
            );
            for row in 0..n {
                jac[row][col] = (rp[row] - residual[row]) / eps;
            }
        }
        let diag_sum = (0..n).map(|i| jac[i][i].abs()).sum::<f64>().max(1.0e-30);
        let mut offdiag_sum = 0.0;
        for row in 0..n {
            for col in 0..n {
                if row != col {
                    offdiag_sum += jac[row][col].abs();
                }
            }
        }
        jacobian_offdiag_ratio = jacobian_offdiag_ratio.max(offdiag_sum / diag_sum);
        let rhs = residual.iter().map(|value| -value).collect::<Vec<_>>();
        let Some((dx, min_pivot)) = solve_dense(jac, rhs) else {
            break;
        };
        min_pivot_abs = min_pivot_abs.min(min_pivot);
        max_newton_correction_m = max_newton_correction_m.max(vec_max_abs(&dx));
        for (value, delta) in x.iter_mut().zip(dx.iter()) {
            *value += delta;
        }
        if !x.iter().all(|value| value.is_finite()) {
            break;
        }
    }
    let final_residual =
        patch_newton_residual(&tets, &x0, &v0, &x, mass_per_node, damping, &external, dt_s);
    let final_residual_norm = vec_norm(&final_residual);
    converged |= final_residual_norm <= 1.0e-4;
    let implicit_pos = patch_unflatten_nodes(&x);
    let implicit_energy_j = patch_total_energy(&tets, &x0, &implicit_pos, mass_per_node, dt_s);
    let rest_volume_m3 = tets.iter().map(|tet| tet.volume).sum::<f64>();
    let mut min_final_volume_ratio = f64::INFINITY;
    let mut max_final_volume_ratio = 0.0_f64;
    for tet in &tets {
        let volume = tetra_signed_volume(tet.ids, &implicit_pos).abs() / tet.volume.max(1.0e-30);
        min_final_volume_ratio = min_final_volume_ratio.min(volume);
        max_final_volume_ratio = max_final_volume_ratio.max(volume);
    }
    let final_internal = patch_internal_forces(&tets, &implicit_pos);
    let shared_node_internal_force_n = [1usize, 2, 3]
        .iter()
        .map(|&id| dot(final_internal[id], final_internal[id]).sqrt())
        .sum::<f64>()
        / 3.0;

    TetraFemPatchNewtonReport {
        node_count: rest.len(),
        tetra_count: tets.len(),
        coupled_dof_count: rest.len() * 3,
        shared_node_count: 3,
        rest_volume_m3,
        dt_s,
        initial_residual_norm,
        final_residual_norm,
        residual_reduction_ratio: final_residual_norm / initial_residual_norm.max(1.0e-30),
        iteration_count,
        converged,
        max_newton_correction_m,
        min_pivot_abs,
        jacobian_offdiag_ratio,
        shared_node_internal_force_n,
        min_final_volume_ratio,
        max_final_volume_ratio,
        explicit_energy_j,
        implicit_energy_j,
        energy_ratio_implicit_over_explicit: implicit_energy_j / explicit_energy_j.max(1.0e-30),
        finite_outputs: initial_residual_norm.is_finite()
            && final_residual_norm.is_finite()
            && min_pivot_abs.is_finite()
            && jacobian_offdiag_ratio.is_finite()
            && shared_node_internal_force_n.is_finite()
            && min_final_volume_ratio.is_finite()
            && max_final_volume_ratio.is_finite()
            && explicit_energy_j.is_finite()
            && implicit_energy_j.is_finite()
            && x.iter().all(|value| value.is_finite()),
    }
}

/// Build the first tiny full-mesh Newton gate from an actual annular tire mesh.
pub fn tetra_fem_tiny_mesh_newton_report() -> TetraFemTinyMeshNewtonReport {
    let tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams {
            segments: 8,
            radial_layers: 2,
            width_layers: 2,
            pressure_pa: 180_000.0,
            young_modulus: 2.4e6,
            sidewall_young_modulus: 1.8e6,
            tread_young_modulus: 3.2e6,
            bead_young_modulus: 5.5e6,
            node_damping: 24.0,
            ..TetraTireFemParams::realtime_viewer()
        },
    );
    let node_count = tire.rest.len();
    let mut x0 = tire.rest.clone();
    let mut v0 = vec![[0.0; 3]; node_count];
    let mut external = vec![[0.0; 3]; node_count];
    for (id, p) in x0.iter_mut().enumerate() {
        let a = id as f64;
        p[0] += 0.0025 * (a * 1.7).sin();
        p[1] += 0.0020 * (a * 1.1).cos();
        p[2] += 0.0018 * (a * 0.9).sin();
        v0[id] = [
            0.35 * (a * 0.8).cos(),
            -0.28 + 0.10 * (a * 0.6).sin(),
            0.22 * (a * 1.2).cos(),
        ];
        external[id][1] = -6.0 * tire.mass[id].max(1.0e-6);
    }
    for &id in &tire.tread_nodes {
        external[id][1] -= 18.0;
        external[id][0] += 4.0;
    }
    for &(id, _) in &tire.bead_offsets {
        external[id][1] += 10.0;
    }
    let dt_s = 1.0 / 240.0;
    let predictor: Vec<[f64; 3]> = x0
        .iter()
        .zip(v0.iter())
        .map(|(p, v)| add(*p, scale(*v, dt_s)))
        .collect();
    let mut x = patch_flatten_nodes(&predictor);
    let initial_residual = patch_newton_residual_with_masses(
        &tire.tets,
        &x0,
        &v0,
        &x,
        &tire.mass,
        tire.params.node_damping,
        &external,
        dt_s,
    );
    let initial_residual_norm = vec_norm(&initial_residual);
    let explicit_energy_j =
        patch_total_energy_with_masses(&tire.tets, &x0, &predictor, &tire.mass, dt_s);
    let mut min_pivot_abs = f64::INFINITY;
    let mut max_newton_correction_m: f64 = 0.0;
    let mut jacobian_offdiag_ratio: f64 = 0.0;
    let mut iteration_count = 0usize;
    let mut converged = false;
    for iteration in 0..10 {
        let residual = patch_newton_residual_with_masses(
            &tire.tets,
            &x0,
            &v0,
            &x,
            &tire.mass,
            tire.params.node_damping,
            &external,
            dt_s,
        );
        let residual_norm = vec_norm(&residual);
        iteration_count = iteration + 1;
        if residual_norm <= 1.0e-5 {
            converged = true;
            break;
        }
        let n = x.len();
        let mut jac = vec![vec![0.0; n]; n];
        for col in 0..n {
            let eps = 1.0e-6 * x[col].abs().max(1.0);
            let mut xp = x.clone();
            xp[col] += eps;
            let rp = patch_newton_residual_with_masses(
                &tire.tets,
                &x0,
                &v0,
                &xp,
                &tire.mass,
                tire.params.node_damping,
                &external,
                dt_s,
            );
            for row in 0..n {
                jac[row][col] = (rp[row] - residual[row]) / eps;
            }
        }
        let diag_sum = (0..n).map(|i| jac[i][i].abs()).sum::<f64>().max(1.0e-30);
        let mut offdiag_sum = 0.0;
        for row in 0..n {
            for col in 0..n {
                if row != col {
                    offdiag_sum += jac[row][col].abs();
                }
            }
        }
        jacobian_offdiag_ratio = jacobian_offdiag_ratio.max(offdiag_sum / diag_sum);
        let rhs = residual.iter().map(|value| -value).collect::<Vec<_>>();
        let Some((dx, min_pivot)) = solve_dense(jac, rhs) else {
            break;
        };
        min_pivot_abs = min_pivot_abs.min(min_pivot);
        max_newton_correction_m = max_newton_correction_m.max(vec_max_abs(&dx));
        for (value, delta) in x.iter_mut().zip(dx.iter()) {
            *value += delta;
        }
        if !x.iter().all(|value| value.is_finite()) {
            break;
        }
    }
    let final_residual = patch_newton_residual_with_masses(
        &tire.tets,
        &x0,
        &v0,
        &x,
        &tire.mass,
        tire.params.node_damping,
        &external,
        dt_s,
    );
    let final_residual_norm = vec_norm(&final_residual);
    converged |= final_residual_norm <= 1.0e-3;
    let implicit_pos = patch_unflatten_nodes(&x);
    let implicit_energy_j =
        patch_total_energy_with_masses(&tire.tets, &x0, &implicit_pos, &tire.mass, dt_s);
    let rest_volume_m3 = tire.tets.iter().map(|tet| tet.volume).sum::<f64>();
    let mut min_final_volume_ratio = f64::INFINITY;
    let mut max_final_volume_ratio = 0.0_f64;
    for tet in &tire.tets {
        let ratio = tetra_signed_volume(tet.ids, &implicit_pos).abs() / tet.volume.max(1.0e-30);
        min_final_volume_ratio = min_final_volume_ratio.min(ratio);
        max_final_volume_ratio = max_final_volume_ratio.max(ratio);
    }
    TetraFemTinyMeshNewtonReport {
        node_count,
        tetra_count: tire.tets.len(),
        coupled_dof_count: node_count * 3,
        tread_node_count: tire.tread_nodes.len(),
        bead_node_count: tire.bead_offsets.len(),
        rest_volume_m3,
        dt_s,
        initial_residual_norm,
        final_residual_norm,
        residual_reduction_ratio: final_residual_norm / initial_residual_norm.max(1.0e-30),
        iteration_count,
        converged,
        max_newton_correction_m,
        min_pivot_abs,
        jacobian_offdiag_ratio,
        min_final_volume_ratio,
        max_final_volume_ratio,
        explicit_energy_j,
        implicit_energy_j,
        energy_ratio_implicit_over_explicit: implicit_energy_j / explicit_energy_j.max(1.0e-30),
        finite_outputs: initial_residual_norm.is_finite()
            && final_residual_norm.is_finite()
            && min_pivot_abs.is_finite()
            && jacobian_offdiag_ratio.is_finite()
            && min_final_volume_ratio.is_finite()
            && max_final_volume_ratio.is_finite()
            && explicit_energy_j.is_finite()
            && implicit_energy_j.is_finite()
            && x.iter().all(|value| value.is_finite()),
    }
}

pub fn tetra_fem_matrix_free_newton_scale_report() -> TetraFemMatrixFreeNewtonScaleReport {
    let tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams {
            pressure_pa: 180_000.0,
            young_modulus: 2.2e6,
            sidewall_young_modulus: 1.7e6,
            tread_young_modulus: 3.0e6,
            bead_young_modulus: 5.0e6,
            node_damping: 22.0,
            ..TetraTireFemParams::realtime_viewer().with_resolution(14, 3, 3)
        },
    );
    let node_count = tire.rest.len();
    let mut x0 = tire.rest.clone();
    let mut v0 = vec![[0.0; 3]; node_count];
    let mut external = vec![[0.0; 3]; node_count];
    for (id, p) in x0.iter_mut().enumerate() {
        let a = id as f64;
        p[0] += 0.0018 * (a * 1.37).sin();
        p[1] += 0.0016 * (a * 0.73).cos();
        p[2] += 0.0014 * (a * 0.91).sin();
        v0[id] = [
            0.18 * (a * 0.41).cos(),
            -0.16 + 0.06 * (a * 0.53).sin(),
            0.12 * (a * 0.67).cos(),
        ];
        external[id][1] = -4.0 * tire.mass[id].max(1.0e-6);
    }
    for &id in &tire.tread_nodes {
        external[id][1] -= 8.0;
        external[id][0] += 2.5;
    }
    for &(id, _) in &tire.bead_offsets {
        external[id][1] += 4.0;
    }

    let dt_s = 1.0 / 300.0;
    let predictor: Vec<[f64; 3]> = x0
        .iter()
        .zip(v0.iter())
        .map(|(p, v)| add(*p, scale(*v, dt_s)))
        .collect();
    let x = patch_flatten_nodes(&predictor);
    let residual = patch_newton_residual_with_masses(
        &tire.tets,
        &x0,
        &v0,
        &x,
        &tire.mass,
        tire.params.node_damping,
        &external,
        dt_s,
    );
    let residual_norm = vec_norm(&residual);
    let mut direction = vec![0.0; x.len()];
    for (i, value) in direction.iter_mut().enumerate() {
        let a = i as f64;
        *value = 0.65 * (a * 0.113).sin() + 0.35 * (a * 0.071).cos();
    }
    let direction_norm = vec_norm(&direction).max(1.0e-30);
    for value in &mut direction {
        *value /= direction_norm;
    }
    let eps = 1.0e-5;
    let x_perturbed = x
        .iter()
        .zip(direction.iter())
        .map(|(value, dir)| value + eps * dir)
        .collect::<Vec<_>>();
    let residual_perturbed = patch_newton_residual_with_masses(
        &tire.tets,
        &x0,
        &v0,
        &x_perturbed,
        &tire.mass,
        tire.params.node_damping,
        &external,
        dt_s,
    );
    let jv = residual_perturbed
        .iter()
        .zip(residual.iter())
        .map(|(rp, r)| (rp - r) / eps)
        .collect::<Vec<_>>();
    let jv_norm = vec_norm(&jv);
    let dof = x.len();
    let dense_matrix_bytes = (dof * dof * std::mem::size_of::<f64>()) as f64;
    let matrix_free_working_bytes =
        ((x.len() + direction.len() + residual.len() + residual_perturbed.len() + jv.len())
            * std::mem::size_of::<f64>()) as f64;
    TetraFemMatrixFreeNewtonScaleReport {
        node_count,
        tetra_count: tire.tets.len(),
        coupled_dof_count: dof,
        tread_node_count: tire.tread_nodes.len(),
        bead_node_count: tire.bead_offsets.len(),
        dt_s,
        residual_norm,
        jv_norm,
        directional_step_norm: eps,
        dense_matrix_bytes,
        matrix_free_working_bytes,
        dense_over_matrix_free_bytes: dense_matrix_bytes / matrix_free_working_bytes.max(1.0),
        residual_evaluations: 2,
        finite_outputs: x.iter().all(|value| value.is_finite())
            && residual.iter().all(|value| value.is_finite())
            && residual_perturbed.iter().all(|value| value.is_finite())
            && jv.iter().all(|value| value.is_finite()),
    }
}

pub fn tetra_fem_matrix_free_newton_solve_report() -> TetraFemMatrixFreeNewtonSolveReport {
    let tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams {
            pressure_pa: 180_000.0,
            young_modulus: 2.2e6,
            sidewall_young_modulus: 1.7e6,
            tread_young_modulus: 3.0e6,
            bead_young_modulus: 5.0e6,
            node_damping: 22.0,
            ..TetraTireFemParams::realtime_viewer().with_resolution(14, 3, 3)
        },
    );
    let node_count = tire.rest.len();
    let mut x0 = tire.rest.clone();
    let mut v0 = vec![[0.0; 3]; node_count];
    let mut external = vec![[0.0; 3]; node_count];
    for (id, p) in x0.iter_mut().enumerate() {
        let a = id as f64;
        p[0] += 0.0018 * (a * 1.37).sin();
        p[1] += 0.0016 * (a * 0.73).cos();
        p[2] += 0.0014 * (a * 0.91).sin();
        v0[id] = [
            0.18 * (a * 0.41).cos(),
            -0.16 + 0.06 * (a * 0.53).sin(),
            0.12 * (a * 0.67).cos(),
        ];
        external[id][1] = -4.0 * tire.mass[id].max(1.0e-6);
    }
    for &id in &tire.tread_nodes {
        external[id][1] -= 8.0;
        external[id][0] += 2.5;
    }
    for &(id, _) in &tire.bead_offsets {
        external[id][1] += 4.0;
    }

    let dt_s = 1.0 / 300.0;
    let predictor = x0
        .iter()
        .zip(v0.iter())
        .map(|(p, v)| add(*p, scale(*v, dt_s)))
        .collect::<Vec<_>>();
    let x = patch_flatten_nodes(&predictor);
    let residual_at = |positions: &[f64]| {
        patch_newton_residual_with_masses(
            &tire.tets,
            &x0,
            &v0,
            positions,
            &tire.mass,
            tire.params.node_damping,
            &external,
            dt_s,
        )
    };
    let residual = residual_at(&x);
    let initial_residual_norm = vec_norm(&residual);
    let rhs = residual.iter().map(|value| -value).collect::<Vec<_>>();
    let eps = 1.0e-5;
    let base_x = x.clone();
    let base_residual = residual.clone();
    let (dx, linear_residual_norm, gmres_iteration_count, jv_evaluation_count, linear_converged) =
        solve_matrix_free_gmres(
            |direction| {
                let perturbed = base_x
                    .iter()
                    .zip(direction.iter())
                    .map(|(value, dir)| value + eps * dir)
                    .collect::<Vec<_>>();
                let perturbed_residual = residual_at(&perturbed);
                perturbed_residual
                    .iter()
                    .zip(base_residual.iter())
                    .map(|(rp, r)| (rp - r) / eps)
                    .collect::<Vec<_>>()
            },
            &rhs,
            24,
            0.35,
        );

    let mut best_alpha = 0.0;
    let mut best_corrected_residual = initial_residual_norm;
    let mut best_x = x.clone();
    for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625] {
        let candidate = x
            .iter()
            .zip(dx.iter())
            .map(|(value, delta)| value + alpha * delta)
            .collect::<Vec<_>>();
        let candidate_residual = residual_at(&candidate);
        let candidate_residual_norm = vec_norm(&candidate_residual);
        if candidate_residual_norm.is_finite() && candidate_residual_norm < best_corrected_residual
        {
            best_alpha = alpha;
            best_corrected_residual = candidate_residual_norm;
            best_x = candidate;
        }
    }
    let correction = best_x
        .iter()
        .zip(x.iter())
        .map(|(new_value, old_value)| new_value - old_value)
        .collect::<Vec<_>>();
    let correction_norm_m = vec_norm(&correction);
    let max_correction_m = vec_max_abs(&correction);
    let dof = x.len();
    let dense_matrix_bytes = (dof * dof * std::mem::size_of::<f64>()) as f64;
    let matrix_free_working_bytes =
        ((x.len() * (gmres_iteration_count.max(1) + 6)) * std::mem::size_of::<f64>()) as f64;

    TetraFemMatrixFreeNewtonSolveReport {
        node_count,
        tetra_count: tire.tets.len(),
        coupled_dof_count: dof,
        dt_s,
        initial_residual_norm,
        linear_residual_norm,
        corrected_residual_norm: best_corrected_residual,
        residual_reduction_ratio: best_corrected_residual / initial_residual_norm.max(1.0e-30),
        gmres_iteration_count,
        jv_evaluation_count,
        correction_norm_m,
        max_correction_m,
        line_search_alpha: best_alpha,
        dense_matrix_bytes,
        matrix_free_working_bytes,
        dense_over_matrix_free_bytes: dense_matrix_bytes / matrix_free_working_bytes.max(1.0),
        linear_converged,
        finite_outputs: initial_residual_norm.is_finite()
            && linear_residual_norm.is_finite()
            && best_corrected_residual.is_finite()
            && dx.iter().all(|value| value.is_finite())
            && best_x.iter().all(|value| value.is_finite()),
    }
}

pub fn tetra_fem_matrix_free_newton_preconditioner_report()
-> TetraFemMatrixFreeNewtonPreconditionerReport {
    let tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams {
            pressure_pa: 180_000.0,
            young_modulus: 2.2e6,
            sidewall_young_modulus: 1.7e6,
            tread_young_modulus: 3.0e6,
            bead_young_modulus: 5.0e6,
            node_damping: 22.0,
            ..TetraTireFemParams::realtime_viewer().with_resolution(14, 3, 3)
        },
    );
    let node_count = tire.rest.len();
    let mut x0 = tire.rest.clone();
    let mut v0 = vec![[0.0; 3]; node_count];
    let mut external = vec![[0.0; 3]; node_count];
    for (id, p) in x0.iter_mut().enumerate() {
        let a = id as f64;
        p[0] += 0.0018 * (a * 1.37).sin();
        p[1] += 0.0016 * (a * 0.73).cos();
        p[2] += 0.0014 * (a * 0.91).sin();
        v0[id] = [
            0.18 * (a * 0.41).cos(),
            -0.16 + 0.06 * (a * 0.53).sin(),
            0.12 * (a * 0.67).cos(),
        ];
        external[id][1] = -4.0 * tire.mass[id].max(1.0e-6);
    }
    for &id in &tire.tread_nodes {
        external[id][1] -= 8.0;
        external[id][0] += 2.5;
    }
    for &(id, _) in &tire.bead_offsets {
        external[id][1] += 4.0;
    }

    let dt_s = 1.0 / 300.0;
    let predictor = x0
        .iter()
        .zip(v0.iter())
        .map(|(p, v)| add(*p, scale(*v, dt_s)))
        .collect::<Vec<_>>();
    let x = patch_flatten_nodes(&predictor);
    let residual_at = |positions: &[f64]| {
        patch_newton_residual_with_masses(
            &tire.tets,
            &x0,
            &v0,
            positions,
            &tire.mass,
            tire.params.node_damping,
            &external,
            dt_s,
        )
    };
    let residual = residual_at(&x);
    let initial_residual_norm = vec_norm(&residual);
    let rhs = residual.iter().map(|value| -value).collect::<Vec<_>>();
    let eps = 1.0e-5;
    let base_x = x.clone();
    let base_residual = residual.clone();
    let apply_jv = |direction: &[f64]| {
        let perturbed = base_x
            .iter()
            .zip(direction.iter())
            .map(|(value, dir)| value + eps * dir)
            .collect::<Vec<_>>();
        let perturbed_residual = residual_at(&perturbed);
        perturbed_residual
            .iter()
            .zip(base_residual.iter())
            .map(|(rp, r)| (rp - r) / eps)
            .collect::<Vec<_>>()
    };

    let (_unpre_dx, unpreconditioned_linear_residual_norm, unpreconditioned_iteration_count, _, _) =
        solve_matrix_free_gmres(|direction| apply_jv(direction), &rhs, 16, 0.35);

    let dof = x.len();
    let mut diagonal = vec![0.0; dof];
    let mut positive_count = 0usize;
    let mut min_abs_diagonal = f64::INFINITY;
    let mut max_abs_diagonal = 0.0_f64;
    for i in 0..dof {
        let mut direction = vec![0.0; dof];
        direction[i] = 1.0;
        let jv = apply_jv(&direction);
        let value = jv.get(i).copied().unwrap_or(0.0);
        diagonal[i] = value;
        if value > 0.0 {
            positive_count += 1;
        }
        let abs = value.abs();
        if abs.is_finite() {
            min_abs_diagonal = min_abs_diagonal.min(abs);
            max_abs_diagonal = max_abs_diagonal.max(abs);
        }
    }
    let preconditioner = diagonal
        .iter()
        .map(|value| {
            if value.is_finite() && value.abs() > 1.0e-9 {
                1.0 / value
            } else {
                1.0
            }
        })
        .collect::<Vec<_>>();
    let (
        preconditioned_y,
        preconditioned_linear_residual_norm,
        preconditioned_iteration_count,
        _preconditioned_jv_count,
        _preconditioned_converged,
    ) = solve_matrix_free_gmres(
        |y| {
            let dx = y
                .iter()
                .zip(preconditioner.iter())
                .map(|(value, inv_diag)| value * inv_diag)
                .collect::<Vec<_>>();
            apply_jv(&dx)
        },
        &rhs,
        16,
        0.35,
    );
    let dx = preconditioned_y
        .iter()
        .zip(preconditioner.iter())
        .map(|(value, inv_diag)| value * inv_diag)
        .collect::<Vec<_>>();

    let mut best_residual_norm = initial_residual_norm;
    let mut best_x = x.clone();
    for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625] {
        let candidate = x
            .iter()
            .zip(dx.iter())
            .map(|(value, delta)| value + alpha * delta)
            .collect::<Vec<_>>();
        let candidate_residual_norm = vec_norm(&residual_at(&candidate));
        if candidate_residual_norm.is_finite() && candidate_residual_norm < best_residual_norm {
            best_residual_norm = candidate_residual_norm;
            best_x = candidate;
        }
    }
    let correction = best_x
        .iter()
        .zip(x.iter())
        .map(|(new_value, old_value)| new_value - old_value)
        .collect::<Vec<_>>();
    let dense_matrix_bytes = (dof * dof * std::mem::size_of::<f64>()) as f64;
    let matrix_free_working_bytes =
        ((dof * (preconditioned_iteration_count.max(1) + 7)) * std::mem::size_of::<f64>()) as f64;

    TetraFemMatrixFreeNewtonPreconditionerReport {
        node_count,
        tetra_count: tire.tets.len(),
        coupled_dof_count: dof,
        dt_s,
        diagonal_sample_count: dof,
        positive_diagonal_fraction: positive_count as f64 / dof.max(1) as f64,
        min_abs_diagonal,
        max_abs_diagonal,
        unpreconditioned_iteration_count,
        preconditioned_iteration_count,
        unpreconditioned_linear_residual_norm,
        preconditioned_linear_residual_norm,
        corrected_residual_norm: best_residual_norm,
        residual_reduction_ratio: best_residual_norm / initial_residual_norm.max(1.0e-30),
        preconditioned_correction_norm_m: vec_norm(&correction),
        max_preconditioned_correction_m: vec_max_abs(&correction),
        diagonal_jv_evaluation_count: dof,
        dense_matrix_bytes,
        matrix_free_working_bytes,
        dense_over_matrix_free_bytes: dense_matrix_bytes / matrix_free_working_bytes.max(1.0),
        finite_outputs: initial_residual_norm.is_finite()
            && diagonal.iter().all(|value| value.is_finite())
            && preconditioner.iter().all(|value| value.is_finite())
            && dx.iter().all(|value| value.is_finite())
            && best_x.iter().all(|value| value.is_finite()),
    }
}

/// Shared CPU matrix-free Newton outcome.
///
/// Used by both `tetra_fem_matrix_free_newton_step_report` (report-only) and
/// `TetraTireFem::step_on_terrain_matrix_free_newton_debug` (callable opt-in
/// timestep). Centralises the predictor build, residual / Jv probing, Jacobi
/// preconditioner sampling, GMRES correction, and line-searched candidate so
/// both call sites stay in lockstep.
pub(crate) struct MatrixFreeNewtonOutcome {
    pub(crate) dof: usize,
    pub(crate) predictor: Vec<[f64; 3]>,
    pub(crate) initial_residual_norm: f64,
    pub(crate) diagonal: Vec<f64>,
    pub(crate) dx: Vec<f64>,
    pub(crate) best_x: Vec<f64>,
    pub(crate) best_alpha: f64,
    pub(crate) best_residual_norm: f64,
    pub(crate) krylov_iteration_count: usize,
    pub(crate) correction: Vec<f64>,
}

pub(crate) fn matrix_free_newton_correction(
    tets: &[TetraElement],
    x0: &[[f64; 3]],
    v0: &[[f64; 3]],
    masses: &[f64],
    node_damping: f64,
    external: &[[f64; 3]],
    dt_s: f64,
) -> MatrixFreeNewtonOutcome {
    let predictor: Vec<[f64; 3]> = x0
        .iter()
        .zip(v0.iter())
        .map(|(p, v)| add(*p, scale(*v, dt_s)))
        .collect();
    let x = patch_flatten_nodes(&predictor);
    let residual_at = |positions: &[f64]| {
        patch_newton_residual_with_masses(
            tets,
            x0,
            v0,
            positions,
            masses,
            node_damping,
            external,
            dt_s,
        )
    };
    let residual = residual_at(&x);
    let initial_residual_norm = vec_norm(&residual);
    let rhs = residual.iter().map(|value| -value).collect::<Vec<_>>();
    let eps = 1.0e-5;
    let base_x = x.clone();
    let base_residual = residual.clone();
    let apply_jv = |direction: &[f64]| {
        let perturbed = base_x
            .iter()
            .zip(direction.iter())
            .map(|(value, dir)| value + eps * dir)
            .collect::<Vec<_>>();
        let perturbed_residual = residual_at(&perturbed);
        perturbed_residual
            .iter()
            .zip(base_residual.iter())
            .map(|(rp, r)| (rp - r) / eps)
            .collect::<Vec<_>>()
    };

    let dof = x.len();
    let mut diagonal = vec![0.0; dof];
    for i in 0..dof {
        let mut direction = vec![0.0; dof];
        direction[i] = 1.0;
        let jv = apply_jv(&direction);
        diagonal[i] = jv.get(i).copied().unwrap_or(1.0);
    }
    let preconditioner = diagonal
        .iter()
        .map(|value| {
            if value.is_finite() && value.abs() > 1.0e-9 {
                1.0 / value
            } else {
                1.0
            }
        })
        .collect::<Vec<_>>();
    let (preconditioned_y, _linear_residual_norm, krylov_iteration_count, _, _) =
        solve_matrix_free_gmres(
            |y| {
                let dx = y
                    .iter()
                    .zip(preconditioner.iter())
                    .map(|(value, inv_diag)| value * inv_diag)
                    .collect::<Vec<_>>();
                apply_jv(&dx)
            },
            &rhs,
            16,
            0.35,
        );
    let dx = preconditioned_y
        .iter()
        .zip(preconditioner.iter())
        .map(|(value, inv_diag)| value * inv_diag)
        .collect::<Vec<_>>();
    let mut best_alpha = 0.0;
    let mut best_residual_norm = initial_residual_norm;
    let mut best_x = x.clone();
    for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625] {
        let candidate = x
            .iter()
            .zip(dx.iter())
            .map(|(value, delta)| value + alpha * delta)
            .collect::<Vec<_>>();
        let candidate_residual_norm = vec_norm(&residual_at(&candidate));
        if candidate_residual_norm.is_finite() && candidate_residual_norm < best_residual_norm {
            best_alpha = alpha;
            best_residual_norm = candidate_residual_norm;
            best_x = candidate;
        }
    }
    let correction = best_x
        .iter()
        .zip(x.iter())
        .map(|(new_value, old_value)| new_value - old_value)
        .collect::<Vec<_>>();
    MatrixFreeNewtonOutcome {
        dof,
        predictor,
        initial_residual_norm,
        diagonal,
        dx,
        best_x,
        best_alpha,
        best_residual_norm,
        krylov_iteration_count,
        correction,
    }
}

pub fn tetra_fem_matrix_free_newton_step_report() -> TetraFemMatrixFreeNewtonStepReport {
    let tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams {
            pressure_pa: 180_000.0,
            young_modulus: 2.2e6,
            sidewall_young_modulus: 1.7e6,
            tread_young_modulus: 3.0e6,
            bead_young_modulus: 5.0e6,
            node_damping: 22.0,
            ..TetraTireFemParams::realtime_viewer().with_resolution(14, 3, 3)
        },
    );
    let node_count = tire.rest.len();
    let mut x0 = tire.rest.clone();
    let mut v0 = vec![[0.0; 3]; node_count];
    let mut external = vec![[0.0; 3]; node_count];
    for (id, p) in x0.iter_mut().enumerate() {
        let a = id as f64;
        p[0] += 0.0018 * (a * 1.37).sin();
        p[1] += 0.0016 * (a * 0.73).cos();
        p[2] += 0.0014 * (a * 0.91).sin();
        v0[id] = [
            0.18 * (a * 0.41).cos(),
            -0.16 + 0.06 * (a * 0.53).sin(),
            0.12 * (a * 0.67).cos(),
        ];
        external[id][1] = -4.0 * tire.mass[id].max(1.0e-6);
    }
    for &id in &tire.tread_nodes {
        external[id][1] -= 8.0;
        external[id][0] += 2.5;
    }
    for &(id, _) in &tire.bead_offsets {
        external[id][1] += 4.0;
    }

    let dt_s = 1.0 / 300.0;
    let outcome = matrix_free_newton_correction(
        &tire.tets,
        &x0,
        &v0,
        &tire.mass,
        tire.params.node_damping,
        &external,
        dt_s,
    );
    let corrected = patch_unflatten_nodes(&outcome.best_x);
    let mut max_velocity_m_per_s = 0.0_f64;
    for (p, p0) in corrected.iter().zip(x0.iter()) {
        let v = scale(sub(*p, *p0), 1.0 / dt_s);
        max_velocity_m_per_s = max_velocity_m_per_s.max(dot(v, v).sqrt());
    }
    let mut min_volume_ratio = f64::INFINITY;
    let mut max_volume_ratio = 0.0_f64;
    for tet in &tire.tets {
        let ratio = tetra_signed_volume(tet.ids, &corrected).abs() / tet.volume.max(1.0e-30);
        min_volume_ratio = min_volume_ratio.min(ratio);
        max_volume_ratio = max_volume_ratio.max(ratio);
    }
    let explicit_energy_j =
        patch_total_energy_with_masses(&tire.tets, &x0, &outcome.predictor, &tire.mass, dt_s);
    let implicit_energy_j =
        patch_total_energy_with_masses(&tire.tets, &x0, &corrected, &tire.mass, dt_s);

    TetraFemMatrixFreeNewtonStepReport {
        node_count,
        tetra_count: tire.tets.len(),
        coupled_dof_count: outcome.dof,
        dt_s,
        initial_residual_norm: outcome.initial_residual_norm,
        corrected_residual_norm: outcome.best_residual_norm,
        residual_reduction_ratio: outcome.best_residual_norm
            / outcome.initial_residual_norm.max(1.0e-30),
        line_search_alpha: outcome.best_alpha,
        krylov_iteration_count: outcome.krylov_iteration_count,
        diagonal_jv_evaluation_count: outcome.dof,
        correction_norm_m: vec_norm(&outcome.correction),
        max_correction_m: vec_max_abs(&outcome.correction),
        max_velocity_m_per_s,
        min_volume_ratio,
        max_volume_ratio,
        explicit_energy_j,
        implicit_energy_j,
        energy_ratio_implicit_over_explicit: implicit_energy_j / explicit_energy_j.max(1.0e-30),
        finite_outputs: outcome.initial_residual_norm.is_finite()
            && outcome.best_residual_norm.is_finite()
            && outcome.diagonal.iter().all(|value| value.is_finite())
            && outcome.dx.iter().all(|value| value.is_finite())
            && outcome.best_x.iter().all(|value| value.is_finite())
            && corrected.iter().all(|p| finite_vec3(*p)),
    }
}

pub fn tetra_fem_matrix_free_newton_timestep_report() -> TetraFemMatrixFreeNewtonStepReport {
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams {
            pressure_pa: 180_000.0,
            young_modulus: 2.2e6,
            sidewall_young_modulus: 1.7e6,
            tread_young_modulus: 3.0e6,
            bead_young_modulus: 5.0e6,
            node_damping: 22.0,
            downforce: 480.0,
            ..TetraTireFemParams::realtime_viewer().with_resolution(14, 3, 3)
        },
    );
    for (id, p) in tire.pos.iter_mut().enumerate() {
        let a = id as f64;
        p[0] += 0.0018 * (a * 1.37).sin();
        p[1] += 0.0016 * (a * 0.73).cos();
        p[2] += 0.0014 * (a * 0.91).sin();
        tire.vel[id] = [
            0.18 * (a * 0.41).cos(),
            -0.16 + 0.06 * (a * 0.53).sin(),
            0.12 * (a * 0.67).cos(),
        ];
    }
    for &id in &tire.tread_nodes {
        tire.vel[id][0] += 0.08;
    }
    tire.external_force_xyz = [40.0, -80.0, 0.0];
    let terrain = FlatTerrain::new(-10.0, tire.params.surface_material.mu_static);
    tire.step_on_terrain_matrix_free_newton_debug(&terrain, 1.0 / 300.0)
}

pub fn tetra_fem_hardening_envelope_report() -> TetraFemHardeningEnvelopeReport {
    let mut max_abs_node_position_m = 0.0_f64;

    let mut high_load = TetraTireFem::new(
        0.0,
        45_000.0,
        TetraTireFemParams {
            segments: 16,
            radial_layers: 3,
            width_layers: 3,
            downforce: 8_500.0,
            pressure_pa: 45_000.0,
            ..Default::default()
        },
    );
    for _ in 0..360 {
        high_load.step(1.0 / 360.0);
    }
    max_abs_node_position_m = max_abs_node_position_m.max(tire_max_abs_node_position(&high_load));
    let high_load_hub_floor_margin_m = high_load.hub()[1] - high_load.params.inner_radius * 0.75;

    let mut high_pressure = TetraTireFem::new(
        0.0,
        420_000.0,
        TetraTireFemParams {
            segments: 16,
            radial_layers: 3,
            width_layers: 3,
            downforce: 3_500.0,
            pressure_pa: 420_000.0,
            ..Default::default()
        },
    );
    let rolling_terrain = TireSolverMovingFlatTerrain {
        speed_x: 28.0,
        friction: 1.0,
    };
    for _ in 0..360 {
        high_pressure.step_on_terrain(&rolling_terrain, 1.0 / 480.0);
    }
    max_abs_node_position_m =
        max_abs_node_position_m.max(tire_max_abs_node_position(&high_pressure));
    let high_pressure_max_radius = high_pressure
        .tread_nodes()
        .iter()
        .map(|&id| {
            let p = high_pressure.nodes()[id];
            ((p[0] - high_pressure.hub()[0]).powi(2) + (p[1] - high_pressure.hub()[1]).powi(2))
                .sqrt()
        })
        .filter(|v| v.is_finite())
        .fold(0.0_f64, f64::max);

    let long_run_steps = 900usize;
    let mut long_run = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams {
            segments: 12,
            radial_layers: 3,
            width_layers: 3,
            downforce: 4_200.0,
            ..Default::default()
        },
    );
    for _ in 0..long_run_steps {
        long_run.step(1.25e-4);
    }
    max_abs_node_position_m = max_abs_node_position_m.max(tire_max_abs_node_position(&long_run));

    let mut energy_tire = TetraTireFem::new(
        0.0,
        170_000.0,
        TetraTireFemParams {
            segments: 12,
            radial_layers: 3,
            width_layers: 3,
            downforce: 5_000.0,
            ..Default::default()
        },
    );
    for (i, v) in energy_tire.vel.iter_mut().enumerate() {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        *v = [0.35 * sign, -0.25, 0.20 * sign];
    }
    let initial_energy = tire_kinetic_energy(&energy_tire).max(1.0e-9);
    let mut peak_energy = initial_energy;
    for _ in 0..600 {
        energy_tire.step(1.25e-4);
        peak_energy = peak_energy.max(tire_kinetic_energy(&energy_tire));
    }
    let final_energy = tire_kinetic_energy(&energy_tire);
    max_abs_node_position_m = max_abs_node_position_m.max(tire_max_abs_node_position(&energy_tire));

    let finite_outputs = tire_nodes_finite(&high_load)
        && tire_nodes_finite(&high_pressure)
        && tire_nodes_finite(&long_run)
        && tire_nodes_finite(&energy_tire)
        && initial_energy.is_finite()
        && peak_energy.is_finite()
        && final_energy.is_finite();

    TetraFemHardeningEnvelopeReport {
        high_load_hub_floor_margin_m,
        high_load_patch_width_m: high_load.contact_patch_width(),
        high_load_max_node_speed_m_per_s: high_load.max_node_speed_m_per_s(),
        high_pressure_radius_ratio: high_pressure_max_radius
            / high_pressure.params.outer_radius.max(1.0e-9),
        high_pressure_max_node_speed_m_per_s: high_pressure.max_node_speed_m_per_s(),
        long_run_steps,
        long_run_hub_deflection_m: long_run.hub_deflection_m(),
        long_run_max_node_speed_m_per_s: long_run.max_node_speed_m_per_s(),
        energy_peak_over_initial: peak_energy / initial_energy.max(1.0e-9),
        energy_final_over_peak: final_energy / peak_energy.max(1.0e-9),
        max_abs_node_position_m,
        finite_outputs,
    }
}
