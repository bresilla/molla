//! Newton-family reports for the tetra FEM solver (stiffness, diagonal,
//! element-Newton, patch-Newton, tiny-mesh-Newton, matrix-free Newton
//! scale/solve/preconditioner/step), plus sidewall laminate coupling +
//! ply-strain XPBD + dihedral XPBD effect reports.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use super::super::TireRigAcceptanceCheck;
use super::super::internal::bool_metric;

/// Executable coupling gate for the current sidewall laminate approximation.
///
/// This is still not the final laminated shell/carcass solve. It makes the
/// existing tetra sidewall ply stack harder to overclaim by perturbing the
/// tire in deterministic radial, axial, and folded modes, then checking that
/// crossed bias plies stay balanced, radial/axial helper plies actually pick
/// up energy in their intended modes, and the structural projection reduces
/// a measured fold.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraSidewallLaminateCouplingReport {
    pub sidewall_element_count: usize,
    pub balanced_bias_a_energy_j: f64,
    pub balanced_bias_b_energy_j: f64,
    pub crossed_pair_relative_imbalance: f64,
    pub radial_helper_energy_j: f64,
    pub axial_helper_energy_j: f64,
    pub folded_ply_energy_j: f64,
    pub folded_max_fold_drift_rad: f64,
    pub projected_max_fold_drift_rad: f64,
    pub projected_fold_ratio: f64,
    pub finite_outputs: bool,
}

impl TetraSidewallLaminateCouplingReport {
    pub fn is_finite(self) -> bool {
        self.balanced_bias_a_energy_j.is_finite()
            && self.balanced_bias_b_energy_j.is_finite()
            && self.crossed_pair_relative_imbalance.is_finite()
            && self.radial_helper_energy_j.is_finite()
            && self.axial_helper_energy_j.is_finite()
            && self.folded_ply_energy_j.is_finite()
            && self.folded_max_fold_drift_rad.is_finite()
            && self.projected_max_fold_drift_rad.is_finite()
            && self.projected_fold_ratio.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.sidewall_element_count >= 16
            && self.balanced_bias_a_energy_j > 1.0e-6
            && self.balanced_bias_b_energy_j > 1.0e-6
            && self.crossed_pair_relative_imbalance <= 0.08
            && self.radial_helper_energy_j > 1.0e-6
            && self.axial_helper_energy_j > 1.0e-6
            && self.folded_ply_energy_j > 1.0e-6
            && self.folded_max_fold_drift_rad >= 0.02
            && self.projected_max_fold_drift_rad < self.folded_max_fold_drift_rad
            && self.projected_fold_ratio <= 0.90
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "sidewall_laminate_coupling_outputs_finite",
                "sidewall laminate coupling diagnostics should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "sidewall_laminate_coupling_has_sidewall_elements",
                "sidewall laminate coupling should cover actual sidewall tetrahedra",
                "sidewall_element_count",
                self.sidewall_element_count as f64,
                16.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "sidewall_laminate_coupling_crossed_pair_balanced",
                "mirrored ± bias plies should carry near-balanced tensile energy under symmetric radial bulge",
                "crossed_pair_relative_imbalance",
                self.crossed_pair_relative_imbalance,
                0.08,
            ),
            TireRigAcceptanceCheck::at_least(
                "sidewall_laminate_coupling_radial_helper_active",
                "radial helper ply should pick up tensile energy under radial sidewall bulge",
                "radial_helper_energy_j",
                self.radial_helper_energy_j,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_least(
                "sidewall_laminate_coupling_axial_helper_active",
                "axial helper ply should pick up tensile energy under widthwise sidewall stretch",
                "axial_helper_energy_j",
                self.axial_helper_energy_j,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_least(
                "sidewall_laminate_coupling_fold_energy_positive",
                "folded sidewall fixture should put positive tensile energy into the laminate stack",
                "folded_ply_energy_j",
                self.folded_ply_energy_j,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_most(
                "sidewall_laminate_coupling_projected_fold_bounded",
                "structural projection should reduce the same folded sidewall used for laminate energy",
                "projected_over_folded_max_deviation",
                self.projected_fold_ratio,
                0.90,
            ),
        ]
    }
}

/// Executable XPBD guardrail for sidewall ply over-stretch.
///
/// This is a force/constraint-producing step toward the laminated sidewall
/// solve: the current tetra sidewall ply families still drive stress through
/// the volumetric material, but this projection treats sidewall ply stretch as
/// a hard structural envelope and reduces excessive tensile `|F·a0|` before a
/// folded/over-bulged tire can be accepted. It remains a compact guardrail,
/// not a full layered shell.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraSidewallPlyStrainXpbdReport {
    pub sidewall_element_count: usize,
    pub stretch_limit: f64,
    pub pre_max_tensile_strain: f64,
    pub post_max_tensile_strain: f64,
    pub strain_reduction_ratio: f64,
    pub max_projection_correction_m: f64,
    pub finite_outputs: bool,
}

impl TetraSidewallPlyStrainXpbdReport {
    pub fn is_finite(self) -> bool {
        self.stretch_limit.is_finite()
            && self.pre_max_tensile_strain.is_finite()
            && self.post_max_tensile_strain.is_finite()
            && self.strain_reduction_ratio.is_finite()
            && self.max_projection_correction_m.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.sidewall_element_count >= 16
            && self.pre_max_tensile_strain > 0.15
            && self.post_max_tensile_strain < self.pre_max_tensile_strain
            && self.post_max_tensile_strain <= (self.stretch_limit - 1.0) * 1.08
            && self.strain_reduction_ratio <= 0.75
            && self.max_projection_correction_m > 1.0e-6
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "sidewall_ply_strain_xpbd_outputs_finite",
                "sidewall ply-strain XPBD diagnostics should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "sidewall_ply_strain_xpbd_has_sidewall_elements",
                "sidewall ply-strain XPBD should cover actual sidewall tetrahedra",
                "sidewall_element_count",
                self.sidewall_element_count as f64,
                16.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "sidewall_ply_strain_xpbd_fixture_overstretches",
                "sidewall ply-strain fixture should create measurable tensile over-stretch before projection",
                "pre_max_tensile_strain",
                self.pre_max_tensile_strain,
                0.15,
            ),
            TireRigAcceptanceCheck::at_most(
                "sidewall_ply_strain_xpbd_reduces_tensile_strain",
                "sidewall ply-strain XPBD should reduce max tensile ply strain",
                "post_over_pre_max_tensile_strain",
                self.strain_reduction_ratio,
                0.75,
            ),
            TireRigAcceptanceCheck::at_most(
                "sidewall_ply_strain_xpbd_bounds_tensile_strain",
                "sidewall ply-strain XPBD should project the fixture back near the configured stretch envelope",
                "post_max_tensile_strain",
                self.post_max_tensile_strain,
                (self.stretch_limit - 1.0) * 1.08,
            ),
            TireRigAcceptanceCheck::at_least(
                "sidewall_ply_strain_xpbd_applies_correction",
                "sidewall ply-strain XPBD should apply a nonzero structural correction in the fixture",
                "max_projection_correction_m",
                self.max_projection_correction_m,
                1.0e-6,
            ),
        ]
    }
}

/// Non-calibration guard for the current sidewall dihedral XPBD approximation.
///
/// This does not claim a full laminated shell solve. It makes the current
/// fold-angle projection executable in the standard rig by deliberately
/// perturbing sidewall surface nodes, measuring the sidewall quad dihedral
/// drift, then applying only the structural XPBD pass and checking that the
/// measured fold error is reduced while all diagnostics stay finite.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraSidewallDihedralXpbdEffectReport {
    pub quad_count: usize,
    pub folded_max_abs_deviation_rad: f64,
    pub stabilized_max_abs_deviation_rad: f64,
    pub reduction_ratio: f64,
    pub finite_outputs: bool,
}

impl TetraSidewallDihedralXpbdEffectReport {
    pub fn is_finite(self) -> bool {
        self.folded_max_abs_deviation_rad.is_finite()
            && self.stabilized_max_abs_deviation_rad.is_finite()
            && self.reduction_ratio.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.quad_count >= 16
            && self.folded_max_abs_deviation_rad >= 0.02
            && self.stabilized_max_abs_deviation_rad < self.folded_max_abs_deviation_rad
            && self.reduction_ratio <= 0.90
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "sidewall_dihedral_xpbd_has_quads",
                "sidewall dihedral XPBD should cover actual sidewall quads",
                "quad_count",
                self.quad_count as f64,
                16.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "sidewall_dihedral_xpbd_perturbs_fold",
                "sidewall dihedral XPBD fixture should introduce measurable fold-angle drift before projection",
                "folded_max_abs_deviation_rad",
                self.folded_max_abs_deviation_rad,
                0.02,
            ),
            TireRigAcceptanceCheck::at_least(
                "sidewall_dihedral_xpbd_outputs_finite",
                "sidewall dihedral XPBD diagnostics should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "sidewall_dihedral_xpbd_reduces_fold",
                "sidewall dihedral XPBD projection should reduce measured fold-angle drift",
                "stabilized_over_folded_max_deviation",
                self.reduction_ratio,
                0.90,
            ),
            TireRigAcceptanceCheck::at_most(
                "sidewall_dihedral_xpbd_stabilized_below_folded",
                "projected sidewall fold drift should be below the deliberately folded state",
                "stabilized_minus_folded_rad",
                self.stabilized_max_abs_deviation_rad - self.folded_max_abs_deviation_rad,
                -1.0e-9,
            ),
        ]
    }
}

/// Non-calibration solver-depth guard for the current tetra-FEM stabilizer.
///
/// This is not a full Newton/implicit solve. It makes the current diagonal
/// semi-implicit stiffness denominator executable in the standard rig: a
/// deliberately disturbed stiff/high-pressure tire should produce lower node
/// velocity spikes with the stabilizer enabled, and a short high-pressure
/// rolling case should stay finite and radius-bounded.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraFemSolverStiffnessReport {
    pub explicit_max_node_speed_m_per_s: f64,
    pub semi_implicit_max_node_speed_m_per_s: f64,
    pub speed_ratio_semi_over_explicit: f64,
    pub semi_implicit_stiffness_scale: f64,
    pub high_pressure_max_node_speed_m_per_s: f64,
    pub high_pressure_radius_ratio: f64,
    pub high_pressure_contact_patch_m: f64,
    pub finite_outputs: bool,
}

impl TetraFemSolverStiffnessReport {
    pub fn is_finite(self) -> bool {
        self.explicit_max_node_speed_m_per_s.is_finite()
            && self.semi_implicit_max_node_speed_m_per_s.is_finite()
            && self.speed_ratio_semi_over_explicit.is_finite()
            && self.semi_implicit_stiffness_scale.is_finite()
            && self.high_pressure_max_node_speed_m_per_s.is_finite()
            && self.high_pressure_radius_ratio.is_finite()
            && self.high_pressure_contact_patch_m.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.semi_implicit_stiffness_scale > 0.0
            && self.speed_ratio_semi_over_explicit <= 0.80
            && self.high_pressure_max_node_speed_m_per_s <= 12.0
            && self.high_pressure_radius_ratio <= 1.35
            && self.high_pressure_contact_patch_m >= 1.0e-5
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "fem_solver_stiffness_outputs_finite",
                "semi-implicit FEM stiffness diagnostics should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_solver_stiffness_scale_active",
                "semi-implicit stiffness denominator should be active in the stiff-rubber case",
                "semi_implicit_stiffness_scale",
                self.semi_implicit_stiffness_scale,
                1.0e-9,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_solver_stiffness_reduces_velocity_spike",
                "semi-implicit stiffness should damp stiff-rubber velocity spikes versus explicit update",
                "speed_ratio_semi_over_explicit",
                self.speed_ratio_semi_over_explicit,
                0.80,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_solver_high_pressure_speed_bounded",
                "high-pressure rolling tire should keep node speeds bounded in the realtime envelope",
                "high_pressure_max_node_speed_m_per_s",
                self.high_pressure_max_node_speed_m_per_s,
                12.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_solver_high_pressure_radius_bounded",
                "high-pressure rolling tire should not balloon unbounded without a true Newton solve",
                "high_pressure_radius_ratio",
                self.high_pressure_radius_ratio,
                1.35,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_solver_high_pressure_patch_exists",
                "high-pressure rolling tire should still form a nonzero supported contact patch",
                "high_pressure_contact_patch_m",
                self.high_pressure_contact_patch_m,
                1.0e-5,
            ),
        ]
    }
}

/// First reusable Newton-style implicit solve gate for stiff tire modes.
///
/// The full tetrahedral tire is still not solved with a coupled Newton system,
/// but this report adds an executable backward-Euler Newton solve for a
/// nonlinear radial rubber mode. It is intentionally data-free: the fixture
/// compares the same stiff spring mode under explicit Euler and a true
/// residual/Jacobian Newton solve, then exposes convergence, residual, tangent,
/// and energy accounting before the method is promoted into the full FEM.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraFemDiagonalNewtonReport {
    pub dt_s: f64,
    pub mass_kg: f64,
    pub damping_n_s_per_m: f64,
    pub linear_stiffness_n_per_m: f64,
    pub cubic_stiffness_n_per_m3: f64,
    pub external_force_n: f64,
    pub initial_displacement_m: f64,
    pub initial_velocity_m_per_s: f64,
    pub explicit_displacement_m: f64,
    pub explicit_velocity_m_per_s: f64,
    pub newton_displacement_m: f64,
    pub newton_velocity_m_per_s: f64,
    pub initial_residual_norm: f64,
    pub final_residual_norm: f64,
    pub residual_reduction_ratio: f64,
    pub iteration_count: usize,
    pub converged: bool,
    pub min_tangent_n_per_m: f64,
    pub explicit_energy_j: f64,
    pub implicit_energy_j: f64,
    pub energy_ratio_implicit_over_explicit: f64,
    pub finite_outputs: bool,
}

impl TetraFemDiagonalNewtonReport {
    pub fn is_finite(self) -> bool {
        self.dt_s.is_finite()
            && self.mass_kg.is_finite()
            && self.damping_n_s_per_m.is_finite()
            && self.linear_stiffness_n_per_m.is_finite()
            && self.cubic_stiffness_n_per_m3.is_finite()
            && self.external_force_n.is_finite()
            && self.initial_displacement_m.is_finite()
            && self.initial_velocity_m_per_s.is_finite()
            && self.explicit_displacement_m.is_finite()
            && self.explicit_velocity_m_per_s.is_finite()
            && self.newton_displacement_m.is_finite()
            && self.newton_velocity_m_per_s.is_finite()
            && self.initial_residual_norm.is_finite()
            && self.final_residual_norm.is_finite()
            && self.residual_reduction_ratio.is_finite()
            && self.min_tangent_n_per_m.is_finite()
            && self.explicit_energy_j.is_finite()
            && self.implicit_energy_j.is_finite()
            && self.energy_ratio_implicit_over_explicit.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.converged
            && self.iteration_count > 0
            && self.iteration_count <= 12
            && self.initial_residual_norm > 1.0
            && self.final_residual_norm <= 1.0e-7
            && self.residual_reduction_ratio <= 1.0e-8
            && self.min_tangent_n_per_m > 0.0
            && self.energy_ratio_implicit_over_explicit <= 0.35
            && self.explicit_velocity_m_per_s.abs() > self.newton_velocity_m_per_s.abs() * 2.0
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "fem_diagonal_newton_outputs_finite",
                "diagonal Newton implicit fixture should keep all outputs finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_diagonal_newton_converged",
                "Newton fixture should converge its backward-Euler residual",
                "converged",
                bool_metric(self.converged),
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_diagonal_newton_iteration_budget",
                "Newton fixture should converge within the small realtime/debug iteration budget",
                "iteration_count",
                self.iteration_count as f64,
                12.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_diagonal_newton_residual_reduced",
                "Newton iterations should reduce the nonlinear residual by many orders of magnitude",
                "residual_reduction_ratio",
                self.residual_reduction_ratio,
                1.0e-8,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_diagonal_newton_final_residual_small",
                "final backward-Euler residual should be numerically small",
                "final_residual_norm",
                self.final_residual_norm,
                1.0e-7,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_diagonal_newton_positive_tangent",
                "nonlinear rubber tangent should remain positive in the tested stiff mode",
                "min_tangent_n_per_m",
                self.min_tangent_n_per_m,
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_diagonal_newton_energy_bounded",
                "implicit Newton step should avoid the explicit stiff-mode energy spike",
                "energy_ratio_implicit_over_explicit",
                self.energy_ratio_implicit_over_explicit,
                0.35,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_diagonal_newton_explicit_overshoot_visible",
                "fixture should show why an implicit solve is needed by making explicit velocity overshoot visible",
                "abs_explicit_over_newton_velocity",
                self.explicit_velocity_m_per_s.abs()
                    / self.newton_velocity_m_per_s.abs().max(1.0e-9),
                2.0,
            ),
        ]
    }
}

/// Coupled element-level Newton gate for the next FEM solver-depth step.
///
/// Unlike `TetraFemDiagonalNewtonReport`, this fixture does not collapse the
/// tire to one scalar mode. It builds one actual Neo-Hookean tetrahedral
/// element, keeps all four node positions coupled through the deformation
/// gradient, assembles a finite-difference residual/Jacobian for the 12
/// backward-Euler position unknowns, and solves that small nonlinear system
/// with Newton iterations. This is still a small patch solve rather than the
/// full tire mesh Newton solve, but it validates the residual/Jacobian shape
/// needed to promote the full FEM from explicit/semi-implicit guardrails.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraFemElementNewtonReport {
    pub coupled_dof_count: usize,
    pub element_volume_m3: f64,
    pub dt_s: f64,
    pub young_modulus_pa: f64,
    pub poisson_ratio: f64,
    pub initial_residual_norm: f64,
    pub final_residual_norm: f64,
    pub residual_reduction_ratio: f64,
    pub iteration_count: usize,
    pub converged: bool,
    pub max_newton_correction_m: f64,
    pub min_pivot_abs: f64,
    pub jacobian_offdiag_ratio: f64,
    pub explicit_energy_j: f64,
    pub implicit_energy_j: f64,
    pub energy_ratio_implicit_over_explicit: f64,
    pub final_volume_ratio: f64,
    pub finite_outputs: bool,
}

impl TetraFemElementNewtonReport {
    pub fn is_finite(self) -> bool {
        self.element_volume_m3.is_finite()
            && self.dt_s.is_finite()
            && self.young_modulus_pa.is_finite()
            && self.poisson_ratio.is_finite()
            && self.initial_residual_norm.is_finite()
            && self.final_residual_norm.is_finite()
            && self.residual_reduction_ratio.is_finite()
            && self.max_newton_correction_m.is_finite()
            && self.min_pivot_abs.is_finite()
            && self.jacobian_offdiag_ratio.is_finite()
            && self.explicit_energy_j.is_finite()
            && self.implicit_energy_j.is_finite()
            && self.energy_ratio_implicit_over_explicit.is_finite()
            && self.final_volume_ratio.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.coupled_dof_count == 12
            && self.converged
            && self.iteration_count > 0
            && self.iteration_count <= 12
            && self.initial_residual_norm > 1.0
            && self.final_residual_norm <= 1.0e-6
            && self.residual_reduction_ratio <= 1.0e-8
            && self.max_newton_correction_m <= 0.20
            && self.min_pivot_abs > 1.0e-8
            && self.jacobian_offdiag_ratio >= 0.05
            && self.energy_ratio_implicit_over_explicit <= 0.75
            && self.final_volume_ratio >= 0.55
            && self.final_volume_ratio <= 1.65
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::between(
                "fem_element_newton_coupled_dofs",
                "element Newton fixture should solve all four tetra nodes as a coupled 12-DoF system",
                "coupled_dof_count",
                self.coupled_dof_count as f64,
                12.0,
                12.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_element_newton_outputs_finite",
                "coupled element Newton outputs should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_element_newton_converged",
                "coupled element Newton solve should converge its backward-Euler residual",
                "converged",
                bool_metric(self.converged),
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_element_newton_iteration_budget",
                "coupled element Newton solve should fit inside the small debug iteration budget",
                "iteration_count",
                self.iteration_count as f64,
                12.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_element_newton_residual_reduced",
                "coupled element Newton iterations should reduce the residual by many orders of magnitude",
                "residual_reduction_ratio",
                self.residual_reduction_ratio,
                1.0e-8,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_element_newton_final_residual_small",
                "coupled element final residual should be numerically small",
                "final_residual_norm",
                self.final_residual_norm,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_element_newton_offdiag_coupling_present",
                "Jacobian should contain meaningful off-diagonal coupling between tetra nodes/axes",
                "jacobian_offdiag_ratio",
                self.jacobian_offdiag_ratio,
                0.05,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_element_newton_pivots_nonzero",
                "linearized Newton system should have non-zero pivots",
                "min_pivot_abs",
                self.min_pivot_abs,
                1.0e-8,
            ),
            TireRigAcceptanceCheck::between(
                "fem_element_newton_volume_bounded",
                "coupled Newton step should keep the tetra volume inside a broad physical envelope",
                "final_volume_ratio",
                self.final_volume_ratio,
                0.55,
                1.65,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_element_newton_energy_bounded",
                "coupled implicit element solve should avoid the explicit stiff-element energy spike",
                "energy_ratio_implicit_over_explicit",
                self.energy_ratio_implicit_over_explicit,
                0.75,
            ),
        ]
    }
}

/// Multi-element patch Newton gate for promoting solver-depth toward the full tire mesh.
///
/// This extends the single-element Newton fixture to a tiny connected tetra
/// patch with shared nodes. The solve still runs on CPU and is intentionally
/// compact enough for unit tests, but it exercises the same kind of global
/// residual coupling that a full tire-mesh Newton solve will need: element
/// forces accumulate onto shared nodes, the finite-difference Jacobian has
/// off-diagonal node coupling, and one Newton solve advances the whole patch.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraFemPatchNewtonReport {
    pub node_count: usize,
    pub tetra_count: usize,
    pub coupled_dof_count: usize,
    pub shared_node_count: usize,
    pub rest_volume_m3: f64,
    pub dt_s: f64,
    pub initial_residual_norm: f64,
    pub final_residual_norm: f64,
    pub residual_reduction_ratio: f64,
    pub iteration_count: usize,
    pub converged: bool,
    pub max_newton_correction_m: f64,
    pub min_pivot_abs: f64,
    pub jacobian_offdiag_ratio: f64,
    pub shared_node_internal_force_n: f64,
    pub min_final_volume_ratio: f64,
    pub max_final_volume_ratio: f64,
    pub explicit_energy_j: f64,
    pub implicit_energy_j: f64,
    pub energy_ratio_implicit_over_explicit: f64,
    pub finite_outputs: bool,
}

impl TetraFemPatchNewtonReport {
    pub fn is_finite(self) -> bool {
        self.rest_volume_m3.is_finite()
            && self.dt_s.is_finite()
            && self.initial_residual_norm.is_finite()
            && self.final_residual_norm.is_finite()
            && self.residual_reduction_ratio.is_finite()
            && self.max_newton_correction_m.is_finite()
            && self.min_pivot_abs.is_finite()
            && self.jacobian_offdiag_ratio.is_finite()
            && self.shared_node_internal_force_n.is_finite()
            && self.min_final_volume_ratio.is_finite()
            && self.max_final_volume_ratio.is_finite()
            && self.explicit_energy_j.is_finite()
            && self.implicit_energy_j.is_finite()
            && self.energy_ratio_implicit_over_explicit.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.node_count >= 5
            && self.tetra_count >= 2
            && self.coupled_dof_count == self.node_count * 3
            && self.shared_node_count >= 3
            && self.converged
            && self.iteration_count > 0
            && self.iteration_count <= 16
            && self.initial_residual_norm > 1.0
            && self.final_residual_norm <= 1.0e-4
            && self.residual_reduction_ratio <= 1.0e-7
            && self.max_newton_correction_m <= 0.20
            && self.min_pivot_abs > 1.0e-8
            && self.jacobian_offdiag_ratio >= 0.05
            && self.shared_node_internal_force_n > 1.0
            && self.min_final_volume_ratio >= 0.50
            && self.max_final_volume_ratio <= 1.80
            && self.energy_ratio_implicit_over_explicit <= 0.80
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "fem_patch_newton_multiple_tets",
                "patch Newton fixture should solve more than one tetrahedron",
                "tetra_count",
                self.tetra_count as f64,
                2.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_patch_newton_shared_nodes",
                "patch Newton fixture should accumulate element forces through shared nodes",
                "shared_node_count",
                self.shared_node_count as f64,
                3.0,
            ),
            TireRigAcceptanceCheck::between(
                "fem_patch_newton_coupled_dofs",
                "patch Newton fixture should solve the whole connected patch as one coupled position system",
                "coupled_dof_count",
                self.coupled_dof_count as f64,
                (self.node_count * 3) as f64,
                (self.node_count * 3) as f64,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_patch_newton_outputs_finite",
                "multi-element patch Newton outputs should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_patch_newton_converged",
                "multi-element patch Newton solve should converge its backward-Euler residual",
                "converged",
                bool_metric(self.converged),
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_patch_newton_iteration_budget",
                "multi-element patch Newton solve should fit inside a small debug iteration budget",
                "iteration_count",
                self.iteration_count as f64,
                16.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_patch_newton_residual_reduced",
                "multi-element patch Newton iterations should reduce the residual by many orders of magnitude",
                "residual_reduction_ratio",
                self.residual_reduction_ratio,
                1.0e-7,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_patch_newton_final_residual_small",
                "multi-element patch final residual should be numerically small",
                "final_residual_norm",
                self.final_residual_norm,
                1.0e-4,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_patch_newton_offdiag_coupling_present",
                "patch Jacobian should contain meaningful off-diagonal coupling across nodes/elements",
                "jacobian_offdiag_ratio",
                self.jacobian_offdiag_ratio,
                0.05,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_patch_newton_shared_node_force_accumulates",
                "shared nodes should receive non-zero accumulated internal force from the connected patch",
                "shared_node_internal_force_n",
                self.shared_node_internal_force_n,
                1.0,
            ),
            TireRigAcceptanceCheck::between(
                "fem_patch_newton_volume_bounded",
                "all patch tetra volumes should remain inside a broad physical envelope",
                "min_final_volume_ratio",
                self.min_final_volume_ratio,
                0.50,
                1.80,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_patch_newton_energy_bounded",
                "implicit patch solve should avoid the explicit stiff-patch energy spike",
                "energy_ratio_implicit_over_explicit",
                self.energy_ratio_implicit_over_explicit,
                0.80,
            ),
        ]
    }
}

/// Tiny full-mesh Newton gate using an actual `TetraTireFem` topology.
///
/// This is the next solver-depth step after the connected two-tetra patch: it
/// takes the smallest allowed annular tire mesh, includes all its nodes and
/// tetrahedra in one residual/Jacobian Newton solve, and reports mesh-level
/// convergence/volume/energy gates. It is intentionally CPU/debug only for
/// now; PLAN_REALTIRE still requires promotion to the production tire mesh and
/// GPU parity before the full Newton item is considered complete.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraFemTinyMeshNewtonReport {
    pub node_count: usize,
    pub tetra_count: usize,
    pub coupled_dof_count: usize,
    pub tread_node_count: usize,
    pub bead_node_count: usize,
    pub rest_volume_m3: f64,
    pub dt_s: f64,
    pub initial_residual_norm: f64,
    pub final_residual_norm: f64,
    pub residual_reduction_ratio: f64,
    pub iteration_count: usize,
    pub converged: bool,
    pub max_newton_correction_m: f64,
    pub min_pivot_abs: f64,
    pub jacobian_offdiag_ratio: f64,
    pub min_final_volume_ratio: f64,
    pub max_final_volume_ratio: f64,
    pub explicit_energy_j: f64,
    pub implicit_energy_j: f64,
    pub energy_ratio_implicit_over_explicit: f64,
    pub finite_outputs: bool,
}

impl TetraFemTinyMeshNewtonReport {
    pub fn is_finite(self) -> bool {
        self.rest_volume_m3.is_finite()
            && self.dt_s.is_finite()
            && self.initial_residual_norm.is_finite()
            && self.final_residual_norm.is_finite()
            && self.residual_reduction_ratio.is_finite()
            && self.max_newton_correction_m.is_finite()
            && self.min_pivot_abs.is_finite()
            && self.jacobian_offdiag_ratio.is_finite()
            && self.min_final_volume_ratio.is_finite()
            && self.max_final_volume_ratio.is_finite()
            && self.explicit_energy_j.is_finite()
            && self.implicit_energy_j.is_finite()
            && self.energy_ratio_implicit_over_explicit.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.node_count >= 32
            && self.tetra_count >= 32
            && self.coupled_dof_count == self.node_count * 3
            && self.tread_node_count >= 16
            && self.bead_node_count >= 16
            && self.converged
            && self.iteration_count > 0
            && self.iteration_count <= 10
            && self.initial_residual_norm > 1.0
            && self.final_residual_norm <= 1.0e-3
            && self.residual_reduction_ratio <= 1.0e-6
            && self.max_newton_correction_m <= 0.20
            && self.min_pivot_abs > 1.0e-8
            && self.jacobian_offdiag_ratio >= 0.02
            && self.min_final_volume_ratio >= 0.45
            && self.max_final_volume_ratio <= 1.95
            && self.energy_ratio_implicit_over_explicit <= 0.95
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "fem_tiny_mesh_newton_full_mesh_nodes",
                "tiny-mesh Newton fixture should solve an actual annular tire mesh, not one tetra",
                "node_count",
                self.node_count as f64,
                32.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_tiny_mesh_newton_full_mesh_tets",
                "tiny-mesh Newton fixture should include the tire mesh tetrahedra",
                "tetra_count",
                self.tetra_count as f64,
                32.0,
            ),
            TireRigAcceptanceCheck::between(
                "fem_tiny_mesh_newton_coupled_dofs",
                "tiny-mesh Newton should solve all mesh node positions as one coupled system",
                "coupled_dof_count",
                self.coupled_dof_count as f64,
                (self.node_count * 3) as f64,
                (self.node_count * 3) as f64,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_tiny_mesh_newton_outputs_finite",
                "tiny-mesh Newton outputs should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_tiny_mesh_newton_converged",
                "tiny-mesh Newton solve should converge its mesh residual",
                "converged",
                bool_metric(self.converged),
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_tiny_mesh_newton_residual_reduced",
                "tiny-mesh Newton iterations should reduce the residual by a broad mesh-level envelope",
                "residual_reduction_ratio",
                self.residual_reduction_ratio,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_tiny_mesh_newton_final_residual_small",
                "tiny-mesh final residual should be small enough for a debug full-mesh Newton gate",
                "final_residual_norm",
                self.final_residual_norm,
                1.0e-3,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_tiny_mesh_newton_offdiag_coupling_present",
                "tiny-mesh Jacobian should contain cross-node/tet coupling",
                "jacobian_offdiag_ratio",
                self.jacobian_offdiag_ratio,
                0.02,
            ),
            TireRigAcceptanceCheck::between(
                "fem_tiny_mesh_newton_volume_bounded",
                "tiny-mesh Newton should keep all tetra volumes inside a broad debug envelope",
                "min_final_volume_ratio",
                self.min_final_volume_ratio,
                0.45,
                1.95,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_tiny_mesh_newton_energy_bounded",
                "tiny-mesh implicit solve should avoid increasing explicit predictor energy",
                "energy_ratio_implicit_over_explicit",
                self.energy_ratio_implicit_over_explicit,
                0.95,
            ),
        ]
    }
}

/// Matrix-free Newton scaling gate for a larger tire FEM topology.
///
/// This is **not** the production implicit tire solve yet. It is the next
/// solver-depth step after the tiny dense Newton fixture: evaluate the
/// backward-Euler residual on a larger annular tire mesh, compute one
/// finite-difference Jacobian-vector product without building a dense
/// `dof × dof` matrix, and report the dense-memory cost that this avoids.
/// The result is a CPU/debug readiness gate for the planned sparse or
/// matrix-free Newton/PCG GPU path.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraFemMatrixFreeNewtonScaleReport {
    pub node_count: usize,
    pub tetra_count: usize,
    pub coupled_dof_count: usize,
    pub tread_node_count: usize,
    pub bead_node_count: usize,
    pub dt_s: f64,
    pub residual_norm: f64,
    pub jv_norm: f64,
    pub directional_step_norm: f64,
    pub dense_matrix_bytes: f64,
    pub matrix_free_working_bytes: f64,
    pub dense_over_matrix_free_bytes: f64,
    pub residual_evaluations: usize,
    pub finite_outputs: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TetraFemMatrixFreeNewtonSolveReport {
    pub node_count: usize,
    pub tetra_count: usize,
    pub coupled_dof_count: usize,
    pub dt_s: f64,
    pub initial_residual_norm: f64,
    pub linear_residual_norm: f64,
    pub corrected_residual_norm: f64,
    pub residual_reduction_ratio: f64,
    pub gmres_iteration_count: usize,
    pub jv_evaluation_count: usize,
    pub correction_norm_m: f64,
    pub max_correction_m: f64,
    pub line_search_alpha: f64,
    pub dense_matrix_bytes: f64,
    pub matrix_free_working_bytes: f64,
    pub dense_over_matrix_free_bytes: f64,
    pub linear_converged: bool,
    pub finite_outputs: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TetraFemMatrixFreeNewtonPreconditionerReport {
    pub node_count: usize,
    pub tetra_count: usize,
    pub coupled_dof_count: usize,
    pub dt_s: f64,
    pub diagonal_sample_count: usize,
    pub positive_diagonal_fraction: f64,
    pub min_abs_diagonal: f64,
    pub max_abs_diagonal: f64,
    pub unpreconditioned_iteration_count: usize,
    pub preconditioned_iteration_count: usize,
    pub unpreconditioned_linear_residual_norm: f64,
    pub preconditioned_linear_residual_norm: f64,
    pub corrected_residual_norm: f64,
    pub residual_reduction_ratio: f64,
    pub preconditioned_correction_norm_m: f64,
    pub max_preconditioned_correction_m: f64,
    pub diagonal_jv_evaluation_count: usize,
    pub dense_matrix_bytes: f64,
    pub matrix_free_working_bytes: f64,
    pub dense_over_matrix_free_bytes: f64,
    pub finite_outputs: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TetraFemMatrixFreeNewtonStepReport {
    pub node_count: usize,
    pub tetra_count: usize,
    pub coupled_dof_count: usize,
    pub dt_s: f64,
    pub initial_residual_norm: f64,
    pub corrected_residual_norm: f64,
    pub residual_reduction_ratio: f64,
    pub line_search_alpha: f64,
    pub krylov_iteration_count: usize,
    pub diagonal_jv_evaluation_count: usize,
    pub correction_norm_m: f64,
    pub max_correction_m: f64,
    pub max_velocity_m_per_s: f64,
    pub min_volume_ratio: f64,
    pub max_volume_ratio: f64,
    pub explicit_energy_j: f64,
    pub implicit_energy_j: f64,
    pub energy_ratio_implicit_over_explicit: f64,
    pub finite_outputs: bool,
}

impl TetraFemMatrixFreeNewtonScaleReport {
    pub fn is_finite(self) -> bool {
        self.dt_s.is_finite()
            && self.residual_norm.is_finite()
            && self.jv_norm.is_finite()
            && self.directional_step_norm.is_finite()
            && self.dense_matrix_bytes.is_finite()
            && self.matrix_free_working_bytes.is_finite()
            && self.dense_over_matrix_free_bytes.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.node_count > 64
            && self.tetra_count > 96
            && self.coupled_dof_count == self.node_count * 3
            && self.tread_node_count > 0
            && self.bead_node_count > 0
            && self.residual_norm > 1.0
            && self.jv_norm > 1.0e-3
            && self.directional_step_norm > 1.0e-6
            && self.dense_matrix_bytes > 1.0e6
            && self.dense_over_matrix_free_bytes > 8.0
            && self.residual_evaluations == 2
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "fem_matrix_free_newton_larger_mesh_nodes",
                "matrix-free Newton scaling gate should use a larger tire mesh than the tiny dense fixture",
                "node_count",
                self.node_count as f64,
                65.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_matrix_free_newton_larger_mesh_tets",
                "matrix-free Newton scaling gate should include a non-tiny tetra count",
                "tetra_count",
                self.tetra_count as f64,
                97.0,
            ),
            TireRigAcceptanceCheck::between(
                "fem_matrix_free_newton_coupled_dofs",
                "matrix-free Newton scaling gate should still cover all tire node DOFs",
                "coupled_dof_count",
                self.coupled_dof_count as f64,
                (self.node_count * 3) as f64,
                (self.node_count * 3) as f64,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_matrix_free_newton_outputs_finite",
                "matrix-free residual/Jv diagnostics should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_matrix_free_newton_residual_nonzero",
                "larger-mesh backward-Euler residual should be non-zero for the disturbed predictor",
                "residual_norm",
                self.residual_norm,
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_matrix_free_newton_jv_nonzero",
                "finite-difference matrix-free Jacobian-vector product should be non-zero",
                "jv_norm",
                self.jv_norm,
                1.0e-3,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_matrix_free_newton_dense_matrix_large",
                "dense Newton matrix for this mesh should already be large enough to justify matrix-free scaling",
                "dense_matrix_bytes",
                self.dense_matrix_bytes,
                1.0e6,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_matrix_free_newton_memory_saves_dense",
                "matrix-free residual/Jv working memory should be much smaller than a dense Jacobian",
                "dense_over_matrix_free_bytes",
                self.dense_over_matrix_free_bytes,
                8.0,
            ),
            TireRigAcceptanceCheck::between(
                "fem_matrix_free_newton_two_residual_evals",
                "matrix-free Jv probe should use exactly base and perturbed residual evaluations",
                "residual_evaluations",
                self.residual_evaluations as f64,
                2.0,
                2.0,
            ),
        ]
    }
}

impl TetraFemMatrixFreeNewtonSolveReport {
    pub fn is_finite(self) -> bool {
        self.dt_s.is_finite()
            && self.initial_residual_norm.is_finite()
            && self.linear_residual_norm.is_finite()
            && self.corrected_residual_norm.is_finite()
            && self.residual_reduction_ratio.is_finite()
            && self.correction_norm_m.is_finite()
            && self.max_correction_m.is_finite()
            && self.line_search_alpha.is_finite()
            && self.dense_matrix_bytes.is_finite()
            && self.matrix_free_working_bytes.is_finite()
            && self.dense_over_matrix_free_bytes.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.node_count > 64
            && self.tetra_count > 96
            && self.coupled_dof_count == self.node_count * 3
            && self.initial_residual_norm > 1.0
            && self.corrected_residual_norm < self.initial_residual_norm
            && self.residual_reduction_ratio <= 0.80
            && self.gmres_iteration_count >= 2
            && self.gmres_iteration_count <= 32
            && self.jv_evaluation_count >= self.gmres_iteration_count
            && self.correction_norm_m > 1.0e-8
            && self.max_correction_m < 0.25
            && self.line_search_alpha > 0.0
            && self.dense_matrix_bytes > 1.0e6
            && self.dense_over_matrix_free_bytes > 8.0
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "fem_matrix_free_newton_solve_larger_mesh_nodes",
                "matrix-free Newton solve should use a larger tire mesh than the tiny dense fixture",
                "node_count",
                self.node_count as f64,
                65.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_matrix_free_newton_solve_larger_mesh_tets",
                "matrix-free Newton solve should include a non-tiny tetra count",
                "tetra_count",
                self.tetra_count as f64,
                97.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_matrix_free_newton_solve_outputs_finite",
                "matrix-free Newton solve diagnostics should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_matrix_free_newton_solve_initial_residual_nonzero",
                "larger-mesh disturbed predictor should have a non-zero residual before correction",
                "initial_residual_norm",
                self.initial_residual_norm,
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_matrix_free_newton_solve_residual_reduced",
                "matrix-free Krylov correction should reduce the nonlinear residual after line search",
                "residual_reduction_ratio",
                self.residual_reduction_ratio,
                0.80,
            ),
            TireRigAcceptanceCheck::between(
                "fem_matrix_free_newton_solve_iteration_count",
                "matrix-free linear solve should perform a bounded number of Krylov iterations",
                "gmres_iteration_count",
                self.gmres_iteration_count as f64,
                2.0,
                32.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_matrix_free_newton_solve_jv_evaluations",
                "matrix-free solve should be driven by Jacobian-vector products rather than dense assembly",
                "jv_evaluation_count",
                self.jv_evaluation_count as f64,
                self.gmres_iteration_count as f64,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_matrix_free_newton_solve_correction_nonzero",
                "matrix-free solve should produce a non-zero position correction",
                "correction_norm_m",
                self.correction_norm_m,
                1.0e-8,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_matrix_free_newton_solve_correction_bounded",
                "line-searched matrix-free correction should stay inside a broad debug displacement bound",
                "max_correction_m",
                self.max_correction_m,
                0.25,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_matrix_free_newton_solve_memory_saves_dense",
                "matrix-free linear solve working memory should be much smaller than a dense Jacobian",
                "dense_over_matrix_free_bytes",
                self.dense_over_matrix_free_bytes,
                8.0,
            ),
        ]
    }
}

impl TetraFemMatrixFreeNewtonPreconditionerReport {
    pub fn is_finite(self) -> bool {
        self.dt_s.is_finite()
            && self.positive_diagonal_fraction.is_finite()
            && self.min_abs_diagonal.is_finite()
            && self.max_abs_diagonal.is_finite()
            && self.unpreconditioned_linear_residual_norm.is_finite()
            && self.preconditioned_linear_residual_norm.is_finite()
            && self.corrected_residual_norm.is_finite()
            && self.residual_reduction_ratio.is_finite()
            && self.preconditioned_correction_norm_m.is_finite()
            && self.max_preconditioned_correction_m.is_finite()
            && self.dense_matrix_bytes.is_finite()
            && self.matrix_free_working_bytes.is_finite()
            && self.dense_over_matrix_free_bytes.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.node_count > 64
            && self.tetra_count > 96
            && self.coupled_dof_count == self.node_count * 3
            && self.diagonal_sample_count == self.coupled_dof_count
            && self.positive_diagonal_fraction >= 0.80
            && self.min_abs_diagonal > 1.0e-6
            && self.max_abs_diagonal > self.min_abs_diagonal
            && self.preconditioned_iteration_count >= 1
            && self.preconditioned_iteration_count <= 24
            && self.preconditioned_iteration_count <= self.unpreconditioned_iteration_count
            && self.preconditioned_linear_residual_norm
                <= self.unpreconditioned_linear_residual_norm
            && self.residual_reduction_ratio <= 0.80
            && self.preconditioned_correction_norm_m > 1.0e-8
            && self.max_preconditioned_correction_m < 0.25
            && self.diagonal_jv_evaluation_count == self.coupled_dof_count
            && self.dense_matrix_bytes > 1.0e6
            && self.dense_over_matrix_free_bytes > 8.0
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "fem_matrix_free_newton_preconditioner_larger_mesh_nodes",
                "matrix-free Newton preconditioner should use a larger tire mesh than the tiny dense fixture",
                "node_count",
                self.node_count as f64,
                65.0,
            ),
            TireRigAcceptanceCheck::between(
                "fem_matrix_free_newton_preconditioner_samples_all_dofs",
                "diagonal preconditioner should sample every coupled tire DOF",
                "diagonal_sample_count",
                self.diagonal_sample_count as f64,
                self.coupled_dof_count as f64,
                self.coupled_dof_count as f64,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_matrix_free_newton_preconditioner_positive_diagonal",
                "matrix-free diagonal preconditioner should find mostly positive backward-Euler tangent entries",
                "positive_diagonal_fraction",
                self.positive_diagonal_fraction,
                0.80,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_matrix_free_newton_preconditioner_diag_nonzero",
                "matrix-free diagonal entries should be non-zero enough for Jacobi-style scaling",
                "min_abs_diagonal",
                self.min_abs_diagonal,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_matrix_free_newton_preconditioner_iteration_not_worse",
                "diagonal-preconditioned Krylov solve should not take more iterations than the unpreconditioned probe",
                "preconditioned_minus_unpreconditioned_iters",
                self.preconditioned_iteration_count as f64
                    - self.unpreconditioned_iteration_count as f64,
                0.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_matrix_free_newton_preconditioner_residual_not_worse",
                "diagonal-preconditioned Krylov residual should not be worse than the unpreconditioned probe",
                "preconditioned_over_unpreconditioned_linear_residual",
                self.preconditioned_linear_residual_norm
                    / self.unpreconditioned_linear_residual_norm.max(1.0e-30),
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_matrix_free_newton_preconditioner_nonlinear_residual_reduced",
                "line-searched preconditioned correction should reduce the nonlinear tire residual",
                "residual_reduction_ratio",
                self.residual_reduction_ratio,
                0.80,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_matrix_free_newton_preconditioner_correction_nonzero",
                "preconditioned Krylov solve should produce a non-zero position correction",
                "preconditioned_correction_norm_m",
                self.preconditioned_correction_norm_m,
                1.0e-8,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_matrix_free_newton_preconditioner_correction_bounded",
                "preconditioned correction should remain inside a broad debug displacement bound",
                "max_preconditioned_correction_m",
                self.max_preconditioned_correction_m,
                0.25,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_matrix_free_newton_preconditioner_memory_saves_dense",
                "preconditioner plus Krylov vectors should stay far smaller than dense Jacobian assembly",
                "dense_over_matrix_free_bytes",
                self.dense_over_matrix_free_bytes,
                8.0,
            ),
        ]
    }
}

impl TetraFemMatrixFreeNewtonStepReport {
    pub fn is_finite(self) -> bool {
        self.dt_s.is_finite()
            && self.initial_residual_norm.is_finite()
            && self.corrected_residual_norm.is_finite()
            && self.residual_reduction_ratio.is_finite()
            && self.line_search_alpha.is_finite()
            && self.correction_norm_m.is_finite()
            && self.max_correction_m.is_finite()
            && self.max_velocity_m_per_s.is_finite()
            && self.min_volume_ratio.is_finite()
            && self.max_volume_ratio.is_finite()
            && self.explicit_energy_j.is_finite()
            && self.implicit_energy_j.is_finite()
            && self.energy_ratio_implicit_over_explicit.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.node_count > 64
            && self.tetra_count > 96
            && self.coupled_dof_count == self.node_count * 3
            && self.corrected_residual_norm < self.initial_residual_norm
            && self.residual_reduction_ratio <= 0.80
            && self.line_search_alpha > 0.0
            && self.krylov_iteration_count >= 1
            && self.diagonal_jv_evaluation_count == self.coupled_dof_count
            && self.correction_norm_m > 1.0e-8
            && self.max_correction_m < 0.25
            && self.max_velocity_m_per_s < 30.0
            && self.min_volume_ratio >= 0.35
            && self.max_volume_ratio <= 2.50
            && self.energy_ratio_implicit_over_explicit <= 1.05
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "fem_matrix_free_newton_step_larger_mesh_nodes",
                "matrix-free Newton step should run on a larger tire mesh than the tiny dense fixture",
                "node_count",
                self.node_count as f64,
                65.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_matrix_free_newton_step_residual_reduced",
                "line-searched matrix-free implicit step should reduce the nonlinear residual",
                "residual_reduction_ratio",
                self.residual_reduction_ratio,
                0.80,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_matrix_free_newton_step_alpha_positive",
                "implicit correction should accept a positive line-search step",
                "line_search_alpha",
                self.line_search_alpha,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_matrix_free_newton_step_correction_nonzero",
                "implicit step should apply a non-zero correction to the predictor",
                "correction_norm_m",
                self.correction_norm_m,
                1.0e-8,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_matrix_free_newton_step_correction_bounded",
                "implicit step correction should stay inside a broad displacement bound",
                "max_correction_m",
                self.max_correction_m,
                0.25,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_matrix_free_newton_step_velocity_bounded",
                "implicit step velocity reconstruction should stay inside a broad debug bound",
                "max_velocity_m_per_s",
                self.max_velocity_m_per_s,
                30.0,
            ),
            TireRigAcceptanceCheck::between(
                "fem_matrix_free_newton_step_volume_bounded",
                "implicit step should keep all tetra volumes inside the broad debug envelope",
                "min_volume_ratio",
                self.min_volume_ratio,
                0.35,
                2.50,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_matrix_free_newton_step_energy_bounded",
                "implicit step should not increase predictor energy beyond the broad debug envelope",
                "energy_ratio_implicit_over_explicit",
                self.energy_ratio_implicit_over_explicit,
                1.05,
            ),
            TireRigAcceptanceCheck::between(
                "fem_matrix_free_newton_step_samples_all_dofs",
                "implicit step preconditioner should sample every coupled tire DOF",
                "diagonal_jv_evaluation_count",
                self.diagonal_jv_evaluation_count as f64,
                self.coupled_dof_count as f64,
                self.coupled_dof_count as f64,
            ),
        ]
    }
}
