//! Host-side contract test for the PLAN_TOURQUE GPU torque/spin extension.
//!
//! Stage G part 1 only wires the host-side parameter fields through
//! [`TireFemGpuParams`]. Stage G part 2 will land the WGSL kernel
//! integration of the spin DOF + spin-coupled bristle slip + dynamic-hub
//! XZ integration. This test pins the contract surface so the kernel
//! cannot drift from the CPU reference in
//! `molla_vehicle::real_tire::TetraTireFem::integrate_spin_dof` once it
//! lands.

#![cfg(feature = "gpu")]

use molla_kernels::{
    TIRE_FEM_TORQUE_BRAKE_SOFT_SHOULDER_RAD_S, TIRE_FEM_TORQUE_HUB_MASS_MIN_KG, TireFemGpuParams,
    TireFemTorqueGpuStatus, tire_fem_torque_gpu_contract,
};

#[test]
fn tire_fem_torque_gpu_contract_pins_field_names_and_constants() {
    let contract = tire_fem_torque_gpu_contract();
    assert_eq!(contract.feature_label, "molla::tire_fem::torque_spin_api");
    assert_eq!(
        contract.uniform_field_names,
        ["hub_omega", "drive_torque_n_m", "brake_torque_n_m"]
    );
    assert_eq!(
        contract.brake_soft_shoulder_rad_s,
        TIRE_FEM_TORQUE_BRAKE_SOFT_SHOULDER_RAD_S
    );
    assert_eq!(contract.hub_mass_min_kg, TIRE_FEM_TORQUE_HUB_MASS_MIN_KG);
    // Stage G part 1 ships the WGSL `phys5` row; Stage G part 2 ships
    // the spin-coupled XPBD contact contribution in tire_fem.wgsl;
    // Stage G part 3 adds the hardware lane that exercises it
    // (`tire_fem_gpu_spin_pushes_grounded_tread_backward_in_contact`).
    // PLAN_REALTIRE Stream C G4 then closed the three-leg CPU/GPU parity
    // lane — node FEM, dynamic hub, and contact-driven rolling with
    // sub-cm hub agreement — which advances the contract to
    // LiveValidated (the 40-wheel scaling-ready state).
    assert_eq!(
        contract.status,
        TireFemTorqueGpuStatus::LiveValidated,
        "torque/spin GPU stack is wired end-to-end and parity-validated against CPU"
    );
}

#[test]
fn tire_fem_gpu_params_carry_torque_spin_fields_zeroed_by_default() {
    let params = TireFemGpuParams::default();
    assert_eq!(params.hub_omega, 0.0);
    assert_eq!(params.drive_torque_n_m, 0.0);
    assert_eq!(params.brake_torque_n_m, 0.0);
}

#[test]
fn tire_fem_gpu_params_torque_spin_fields_round_trip() {
    let mut params = TireFemGpuParams::default();
    params.hub_omega = 4.5;
    params.drive_torque_n_m = 120.0;
    params.brake_torque_n_m = 60.0;
    assert_eq!(params.hub_omega, 4.5);
    assert_eq!(params.drive_torque_n_m, 120.0);
    assert_eq!(params.brake_torque_n_m, 60.0);
}

#[test]
fn tire_fem_torque_contract_shoulder_matches_cpu_constant() {
    // The brake soft-sign shoulder must agree byte-for-byte with the
    // constant the CPU `integrate_spin_dof` bakes in
    // (`omega / (|omega| + 0.5)`), or GPU/CPU drift the moment Stage G
    // part 2 ships.
    let contract = tire_fem_torque_gpu_contract();
    assert!(
        (contract.brake_soft_shoulder_rad_s - 0.5).abs() < 1.0e-6,
        "brake shoulder constant must remain 0.5 rad/s"
    );
}
