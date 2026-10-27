//! Host-side contract test for the GPU matrix-free Newton tire-FEM debug
//! timestep.
//!
//! The hardware path is not implemented yet. This test pins the dispatch
//! shape returned by [`tire_fem_matrix_free_newton_gpu_contract`] so the
//! WGSL kernel cannot silently drift from the CPU reference in
//! `molla_vehicle::TetraTireFem::step_on_terrain_matrix_free_newton_debug`
//! once it lands.
//!
//! The whole `tire_fem` module is gated behind the `gpu` feature in
//! `crates/molla-kernels/src/lib.rs`, so this test mirrors that gate. It is
//! still a CPU-only test — the contract is pure host-side data describing the
//! GPU dispatch shape — and runs under plain `cargo test --features gpu`
//! without needing a wgpu adapter.

#![cfg(feature = "gpu")]

use molla_kernels::{
    TIRE_FEM_MATRIX_FREE_NEWTON_DEFAULT_KRYLOV_BUDGET,
    TIRE_FEM_MATRIX_FREE_NEWTON_LINE_SEARCH_ALPHAS, TIRE_FEM_MATRIX_FREE_NEWTON_SHADER,
    TireFemMatrixFreeNewtonGpuStatus, tire_fem_matrix_free_newton_gpu_contract,
};

#[test]
fn tire_fem_matrix_free_newton_gpu_contract_pins_dispatch_shape() {
    let node_count: u32 = 128;
    let contract = tire_fem_matrix_free_newton_gpu_contract(
        node_count,
        TIRE_FEM_MATRIX_FREE_NEWTON_DEFAULT_KRYLOV_BUDGET,
    );
    assert_eq!(
        contract.shader_label,
        "molla::tire_fem_matrix_free_newton::step_debug_contract"
    );
    assert_eq!(contract.workgroup_size, 64);
    assert_eq!(contract.node_count, node_count);
    assert_eq!(contract.coupled_dof_count, node_count * 3);
    assert_eq!(contract.residual_workgroup_count, 2);
    assert_eq!(contract.jv_probe_workgroup_count, 2);
    assert_eq!(contract.diagonal_workgroup_count, 6);
    assert_eq!(contract.correction_workgroup_count, 2);
    assert_eq!(
        contract.line_search_alpha_count,
        TIRE_FEM_MATRIX_FREE_NEWTON_LINE_SEARCH_ALPHAS
    );
    assert_eq!(
        contract.krylov_iteration_budget,
        TIRE_FEM_MATRIX_FREE_NEWTON_DEFAULT_KRYLOV_BUDGET
    );
    assert_eq!(
        contract.line_search_residual_invocations,
        TIRE_FEM_MATRIX_FREE_NEWTON_LINE_SEARCH_ALPHAS + 1
    );
    assert_eq!(
        contract.status,
        TireFemMatrixFreeNewtonGpuStatus::Live,
        "WGSL shader, host-side kernel, and ignored hardware test are all wired"
    );
}

#[test]
fn tire_fem_matrix_free_newton_shader_defines_contract_entry_points() {
    let shader = TIRE_FEM_MATRIX_FREE_NEWTON_SHADER;
    for entry in [
        "fn residual_eval",
        "fn jv_probe",
        "fn diagonal_extract",
        "fn apply_correction",
    ] {
        assert!(
            shader.contains(entry),
            "matrix-free Newton shader must declare {entry} so the host dispatch contract is honoured"
        );
    }
    assert!(
        shader.contains("@compute @workgroup_size(WORKGROUP_SIZE)")
            || shader.contains("@compute @workgroup_size(64)"),
        "shader entry points must use the contract workgroup size"
    );
    for binding in [
        "rest_pos",
        "x0_pos",
        "v0_vel",
        "mass_in",
        "external_in",
        "tet_indices",
        "tet_grads",
        "tet_volume",
        "tet_mu",
        "tet_lambda",
        "candidate_pos",
        "direction_in",
        "residual_out",
        "jv_out",
        "diagonal_out",
        "pos_out",
    ] {
        assert!(
            shader.contains(binding),
            "matrix-free Newton shader must expose binding `{binding}`"
        );
    }
}

#[test]
fn tire_fem_matrix_free_newton_gpu_contract_rounds_workgroups_up() {
    let contract = tire_fem_matrix_free_newton_gpu_contract(
        65,
        TIRE_FEM_MATRIX_FREE_NEWTON_DEFAULT_KRYLOV_BUDGET,
    );
    assert_eq!(contract.residual_workgroup_count, 2);
    assert_eq!(contract.jv_probe_workgroup_count, 2);
    assert_eq!(contract.correction_workgroup_count, 2);
    let dof: u32 = 65 * 3;
    let expected_dof_wg = dof.div_ceil(contract.workgroup_size);
    assert_eq!(contract.diagonal_workgroup_count, expected_dof_wg);
}

#[test]
fn tire_fem_matrix_free_newton_gpu_contract_clamps_krylov_budget_above_zero() {
    let contract = tire_fem_matrix_free_newton_gpu_contract(32, 0);
    assert!(
        contract.krylov_iteration_budget >= 1,
        "krylov budget should never collapse to zero (would skip GMRES correction)"
    );
}
