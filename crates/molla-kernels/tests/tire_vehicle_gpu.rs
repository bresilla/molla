//! GPU vehicle-level tire yaw coupling tests.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::{
    TireBrushGpuKernel, TireFemContactFeedbackOutput, TireFemContactFeedbackResidentOutput,
    TireFemGpuBatchAtlasLayout, TireFemGpuBatchAtlasRow, TireFemGpuBatchDispatchPlan,
    TireVehicleBrushRowsFromSpinGpuInput, TireVehicleBrushRowsGpuInput,
    TireVehiclePlanarFromBrushGpuInput, TireVehiclePlanarGpuInput, TireVehicleSuspensionGpuInput,
    TireVehicleVerticalGpuInput, TireVehicleWheelKinematicsFromSpinGpuInput,
    TireVehicleWheelKinematicsGpuInput, TireVehicleWheelSpinGpuInput, TireVehicleYawGpuInput,
    TireVehicleYawGpuKernel, tire_vehicle_brush_rows_from_spin_reference,
    tire_vehicle_brush_rows_from_spin_suspension_reference, tire_vehicle_brush_rows_reference,
    tire_vehicle_planar_from_brush_reference,
    tire_vehicle_planar_from_contact_reduction_rows_reference,
    tire_vehicle_planar_from_feedback_output_reference, tire_vehicle_planar_reference,
    tire_vehicle_suspension_reference, tire_vehicle_vertical_from_suspension_reference,
    tire_vehicle_wheel_kinematics_from_spin_reference, tire_vehicle_wheel_kinematics_reference,
    tire_vehicle_wheel_spin_reference, tire_vehicle_wheel_spin_state_reference,
    tire_vehicle_yaw_reference,
};

fn try_engine() -> Option<Engine> {
    Engine::new_gpu().ok()
}

fn sample_yaw_input() -> TireVehicleYawGpuInput {
    TireVehicleYawGpuInput {
        wheel_offsets: vec![
            [1.2, 0.0, 0.75, 0.0],
            [1.2, 0.0, -0.75, 0.0],
            [-1.2, 0.0, 0.75, 0.0],
            [-1.2, 0.0, -0.75, 0.0],
        ],
        tire_wrenches: vec![
            [120.0, 0.0, 640.0, 18.0],
            [90.0, 0.0, 580.0, 15.0],
            [-40.0, 0.0, -180.0, 6.0],
            [-55.0, 0.0, -210.0, 4.0],
        ],
        yaw: 0.10,
        yaw_rate: 0.25,
        yaw_inertia: 1_800.0,
        yaw_damping: 35.0,
        dt: 1.0 / 120.0,
    }
}

fn sample_planar_input() -> TireVehiclePlanarGpuInput {
    TireVehiclePlanarGpuInput {
        wheel_offsets: vec![
            [1.2, 0.0, 0.75, 0.0],
            [1.2, 0.0, -0.75, 0.0],
            [-1.2, 0.0, 0.75, 0.0],
            [-1.2, 0.0, -0.75, 0.0],
        ],
        tire_wrenches: vec![
            [320.0, 0.0, 40.0, 18.0],
            [280.0, 0.0, 30.0, 15.0],
            [260.0, 0.0, -20.0, 6.0],
            [240.0, 0.0, -15.0, 4.0],
        ],
        position_xz: [2.0, -1.0],
        velocity_xz: [4.0, 0.25],
        yaw: 0.10,
        yaw_rate: 0.25,
        mass: 1_240.0,
        yaw_inertia: 1_800.0,
        linear_damping: 12.0,
        yaw_damping: 35.0,
        dt: 1.0 / 120.0,
    }
}

fn sample_planar_from_brush_input() -> TireVehiclePlanarFromBrushGpuInput {
    let planar = sample_planar_input();
    TireVehiclePlanarFromBrushGpuInput {
        wheel_offsets: planar.wheel_offsets,
        position_xz: planar.position_xz,
        velocity_xz: planar.velocity_xz,
        yaw: planar.yaw,
        yaw_rate: planar.yaw_rate,
        mass: planar.mass,
        yaw_inertia: planar.yaw_inertia,
        linear_damping: planar.linear_damping,
        yaw_damping: planar.yaw_damping,
        dt: planar.dt,
    }
}

fn sample_wheel_spin_input() -> TireVehicleWheelSpinGpuInput {
    TireVehicleWheelSpinGpuInput {
        wheel_state: vec![
            [28.0, 0.32, 1.2, 0.0],
            [-12.0, 0.32, 1.1, 0.0],
            [5.0, 0.35, 1.6, 0.0],
            [0.0, 0.30, 1.4, 0.0],
        ],
        wheel_commands: vec![
            [240.0, 15.0, 620.0, 0.0],
            [120.0, 30.0, -180.0, 0.0],
            [0.0, 80.0, 110.0, 0.0],
            [180.0, 20.0, 0.0, 0.0],
        ],
        dt: 1.0 / 240.0,
    }
}

fn sample_wheel_kinematics_input() -> TireVehicleWheelKinematicsGpuInput {
    TireVehicleWheelKinematicsGpuInput {
        wheel_mounts: vec![
            [1.3, -0.2, 0.75, 0.32],
            [1.3, -0.2, -0.75, 0.32],
            [-1.3, -0.2, 0.75, 0.32],
            [-1.3, -0.2, -0.75, 0.32],
        ],
        wheel_state: vec![
            [31.0, 0.18, 0.0, 0.0],
            [30.0, 0.18, 0.0, 0.0],
            [29.0, 0.0, 0.0, 0.0],
            [28.0, 0.0, 0.0, 0.0],
        ],
        velocity_xz: [6.0, 0.8],
        yaw: 0.25,
        yaw_rate: 0.35,
    }
}

fn sample_wheel_kinematics_from_spin_input() -> TireVehicleWheelKinematicsFromSpinGpuInput {
    let base = sample_wheel_kinematics_input();
    TireVehicleWheelKinematicsFromSpinGpuInput {
        wheel_mounts: base.wheel_mounts,
        wheel_controls: base
            .wheel_state
            .iter()
            .map(|row| [row[1], 0.0, 0.0, 0.0])
            .collect(),
        velocity_xz: base.velocity_xz,
        yaw: base.yaw,
        yaw_rate: base.yaw_rate,
    }
}

fn sample_suspension_input() -> TireVehicleSuspensionGpuInput {
    TireVehicleSuspensionGpuInput {
        suspension_geometry: vec![
            [-0.2, 0.32, 0.35, 0.315],
            [-0.2, 0.32, 0.35, 0.315],
            [-0.18, 0.33, 0.34, 0.306],
            [-0.18, 0.33, 0.34, 0.306],
        ],
        suspension_params: vec![
            [35_000.0, 4_000.0, 0.0, 0.0],
            [35_000.0, 4_000.0, 0.02, 0.0],
            [32_000.0, 3_500.0, -0.03, 0.0],
            [32_000.0, 3_500.0, 0.04, 0.0],
        ],
        chassis_y: 0.786,
        chassis_vy: -0.15,
    }
}

fn sample_vertical_input() -> TireVehicleVerticalGpuInput {
    TireVehicleVerticalGpuInput {
        y: sample_suspension_input().chassis_y,
        vy: sample_suspension_input().chassis_vy,
        mass: 1_240.0,
        gravity: 9.81,
        vertical_damping: 150.0,
        dt: sample_planar_from_brush_input().dt,
    }
}

fn sample_brush_rows_input() -> TireVehicleBrushRowsGpuInput {
    TireVehicleBrushRowsGpuInput {
        chassis_pose: [2.0, 0.78, -1.0, 0.25],
        chassis_velocity: [6.0, -0.15, 0.8, 0.35],
        wheel_mounts: vec![
            [1.3, -0.2, 0.75, 0.32],
            [1.3, -0.2, -0.75, 0.32],
            [-1.3, -0.18, 0.75, 0.33],
            [-1.3, -0.18, -0.75, 0.33],
        ],
        wheel_state: vec![
            [31.0, 0.18, 0.22, 180_000.0],
            [30.0, 0.18, 0.22, 180_000.0],
            [29.0, 0.0, 0.24, 170_000.0],
            [28.0, 0.0, 0.24, 170_000.0],
        ],
        tire_rows: vec![[48.0, 280_000.0, 25_000.0, 0.5]; 4],
        material_rows: vec![[0.9, 0.75, 1_200.0, 18.0]; 4],
        extra_rows: vec![[0.0, 0.0, 1.0, 0.02]; 4],
        suspension_rows: vec![
            [0.10, 0.15, 0.0, 1.0],
            [0.10, 0.15, 0.02, 1.0],
            [0.08, 0.15, -0.03, 1.0],
            [0.13, 0.15, 0.04, 1.0],
        ],
    }
}

fn sample_brush_rows_from_spin_input(
    spin_state_rows: Vec<[f32; 4]>,
) -> TireVehicleBrushRowsFromSpinGpuInput {
    let base = sample_brush_rows_input();
    TireVehicleBrushRowsFromSpinGpuInput {
        chassis_pose: base.chassis_pose,
        chassis_velocity: base.chassis_velocity,
        spin_state_rows,
        wheel_mounts: base.wheel_mounts,
        wheel_controls: base
            .wheel_state
            .iter()
            .map(|row| [row[1], row[2], row[3], 0.0])
            .collect(),
        tire_rows: base.tire_rows,
        material_rows: base.material_rows,
        extra_rows: base.extra_rows,
        suspension_rows: base.suspension_rows,
    }
}

#[test]
fn tire_vehicle_yaw_reference_sums_tire_forces_and_aligning_moments() {
    let input = sample_yaw_input();
    let out = tire_vehicle_yaw_reference(&input).expect("yaw reference");
    let expected_mz: f32 = input
        .wheel_offsets
        .iter()
        .zip(input.tire_wrenches.iter())
        .map(|(off, wrench)| off[0] * wrench[2] - off[2] * wrench[0] + wrench[3])
        .sum();
    assert!((out.yaw_moment - expected_mz).abs() < 1.0e-4);
    assert!(out.finite);
    assert!(
        out.yaw_rate > input.yaw_rate,
        "positive total tire yaw moment should increase yaw rate: {out:?}"
    );

    let shader = include_str!("../shaders/tire_vehicle_yaw.wgsl");
    assert!(shader.contains("off.x * force.z - off.z * force.x + tire_mz"));
    assert!(shader.contains("yaw_out[0]"));
}

#[test]
fn tire_vehicle_planar_reference_integrates_force_translation_and_yaw() {
    let input = sample_planar_input();
    let out = tire_vehicle_planar_reference(&input).expect("planar reference");
    let expected_fx: f32 = input.tire_wrenches.iter().map(|w| w[0]).sum();
    let expected_fz: f32 = input.tire_wrenches.iter().map(|w| w[2]).sum();
    let expected_mz: f32 = input
        .wheel_offsets
        .iter()
        .zip(input.tire_wrenches.iter())
        .map(|(off, wrench)| off[0] * wrench[2] - off[2] * wrench[0] + wrench[3])
        .sum();
    assert!((out.force_xz[0] - expected_fx).abs() < 1.0e-4);
    assert!((out.force_xz[1] - expected_fz).abs() < 1.0e-4);
    assert!((out.yaw_moment - expected_mz).abs() < 1.0e-4);
    assert!(out.velocity_xz[0] > input.velocity_xz[0]);
    assert!(out.position_xz[0] > input.position_xz[0]);
    assert!(out.finite);

    let shader = include_str!("../shaders/tire_vehicle_yaw.wgsl");
    assert!(shader.contains("fn integrate_planar"));
    assert!(shader.contains("yaw_out[2] = vec4<f32>(total_fx, total_fz"));
}

#[test]
fn tire_vehicle_planar_from_brush_reference_uses_world_force_and_aligning_moment() {
    let input = sample_planar_from_brush_input();
    let mut brush_outputs = Vec::with_capacity(input.wheel_offsets.len());
    for i in 0..input.wheel_offsets.len() {
        brush_outputs.push(molla_kernels::BrushWheelOutput {
            world_force: [120.0 + i as f32 * 30.0, 0.0, 40.0 - i as f32 * 12.0, 0.0],
            diag2: [0.0, 7.0 + i as f32, 0.0, 0.0],
            ..Default::default()
        });
    }
    let out =
        tire_vehicle_planar_from_brush_reference(&input, &brush_outputs).expect("brush planar ref");
    let expected_fx: f32 = brush_outputs.iter().map(|row| row.world_force[0]).sum();
    let expected_fz: f32 = brush_outputs.iter().map(|row| row.world_force[2]).sum();
    let expected_mz: f32 = input
        .wheel_offsets
        .iter()
        .zip(brush_outputs.iter())
        .map(|(off, brush)| {
            off[0] * brush.world_force[2] - off[2] * brush.world_force[0] + brush.diag2[1]
        })
        .sum();
    assert!((out.force_xz[0] - expected_fx).abs() < 1.0e-5);
    assert!((out.force_xz[1] - expected_fz).abs() < 1.0e-5);
    assert!((out.yaw_moment - expected_mz).abs() < 1.0e-5);
    assert!(out.finite);

    let shader = include_str!("../shaders/tire_vehicle_yaw.wgsl");
    assert!(shader.contains("fn integrate_planar_from_brush_outputs"));
    assert!(shader.contains("fn integrate_planar_state_from_brush_outputs"));
    assert!(shader.contains("fn integrate_planar_from_contact_reduction_outputs"));
    assert!(shader.contains("fn integrate_planar_state_from_contact_reduction_outputs"));
    assert!(shader.contains("fn integrate_planar_state_from_feedback_force_rows"));
    assert!(shader.contains("vertical_rows[tire].xy"));
    assert!(shader.contains("fn build_brush_rows_from_spin_suspension_planar_state"));
    assert!(shader.contains("fn build_brush_rows_from_spin_suspension_chassis_state"));
    assert!(shader.contains("fn integrate_vertical_state_from_suspension"));
    assert!(shader.contains("fn compute_suspension_from_vertical_state"));
    assert!(shader.contains("base + 6u"));
    assert!(shader.contains("diag2.y"));
    let brush_source = include_str!("../src/tire_brush.rs");
    assert!(brush_source.contains("TireBrushResidentOutput"));
    assert!(brush_source.contains("dispatch_input_buffer_resident"));
    let vehicle_source = include_str!("../src/tire_vehicle.rs");
    assert!(vehicle_source.contains("TireVehiclePlanarResidentState"));
    assert!(vehicle_source.contains("step_planar_resident_from_brush_output"));
    assert!(vehicle_source.contains("dispatch_planar_from_contact_reduction_output_blocking"));
    assert!(vehicle_source.contains("step_planar_resident_from_contact_reduction_output"));
    assert!(vehicle_source.contains("tire_vehicle_planar_from_feedback_output_reference"));
    assert!(vehicle_source.contains("step_planar_resident_from_feedback_output"));
    assert!(vehicle_source.contains("TireVehicleResidentChassisBrushOwner"));
    assert!(vehicle_source.contains("step_resident_chassis_brush_owner"));
    assert!(vehicle_source.contains("step_resident_chassis_brush_owner_from_planar_state"));
    assert!(vehicle_source.contains("dispatch_brush_rows_from_spin_suspension_planar_resident"));
    assert!(vehicle_source.contains("TireVehicleVerticalResidentState"));
    assert!(vehicle_source.contains("step_suspension_resident_from_vertical_state"));
    assert!(vehicle_source.contains("step_vertical_resident_from_suspension"));
    assert!(vehicle_source.contains("create_resident_chassis_brush_owner_with_vertical"));
    assert!(vehicle_source.contains("step_resident_chassis_brush_owner_from_resident_state"));
    assert!(vehicle_source.contains("dispatch_brush_rows_from_spin_suspension_chassis_resident"));
    let fem_source = include_str!("../src/tire_fem.rs");
    assert!(fem_source.contains("step_resident_vehicle_fem_hubs_from_brush_owner"));
    assert!(fem_source.contains("step_resident_atlas_hubs_from_brush_output"));
}

#[test]
fn tire_vehicle_planar_from_contact_reduction_reference_uses_world_force_and_moment() {
    let input = sample_planar_from_brush_input();
    let mut rows = Vec::with_capacity(input.wheel_offsets.len() * 4);
    for i in 0..input.wheel_offsets.len() {
        rows.push([0.0, 0.0, 0.0, 0.0]);
        rows.push([0.0, 0.0, 0.0, 0.0]);
        rows.push([90.0 + i as f32 * 25.0, 0.0, 35.0 - i as f32 * 8.0, 0.0]);
        rows.push([0.0, 5.0 + i as f32 * 1.5, 0.0, 0.0]);
    }
    let out = tire_vehicle_planar_from_contact_reduction_rows_reference(&input, &rows)
        .expect("contact reduction planar ref");
    let expected_fx: f32 = rows.chunks_exact(4).map(|chunk| chunk[2][0]).sum();
    let expected_fz: f32 = rows.chunks_exact(4).map(|chunk| chunk[2][2]).sum();
    let expected_mz: f32 = input
        .wheel_offsets
        .iter()
        .zip(rows.chunks_exact(4))
        .map(|(off, chunk)| off[0] * chunk[2][2] - off[2] * chunk[2][0] + chunk[3][1])
        .sum();
    assert!((out.force_xz[0] - expected_fx).abs() < 1.0e-5);
    assert!((out.force_xz[1] - expected_fz).abs() < 1.0e-5);
    assert!((out.yaw_moment - expected_mz).abs() < 1.0e-5);
    assert!(out.finite);
    assert!(
        tire_vehicle_planar_from_contact_reduction_rows_reference(&input, &rows[..rows.len() - 1])
            .is_err(),
        "packed contact-reduction rows must be exactly 4 rows per wheel"
    );
}

#[test]
fn tire_vehicle_planar_from_feedback_reference_uses_atlas_force_rows() {
    let mut input = sample_planar_from_brush_input();
    input.wheel_offsets.truncate(2);
    let layout = TireFemGpuBatchAtlasLayout {
        plan: TireFemGpuBatchDispatchPlan {
            tire_count: 2,
            total_node_count: 5,
            total_tet_count: 2,
            total_tread_count: 0,
            total_bead_count: 0,
            total_xpbd_edge_count: 0,
            total_xpbd_dihedral_count: 0,
            max_nodes_per_tire: 3,
            node_workgroup_count: 1,
            persistent_state_bytes: (5 * core::mem::size_of::<[f32; 4]>()) as u64,
        },
        rows: vec![
            TireFemGpuBatchAtlasRow {
                tire_index: 0,
                node_offset: 0,
                node_count: 3,
                ..Default::default()
            },
            TireFemGpuBatchAtlasRow {
                tire_index: 1,
                node_offset: 3,
                node_count: 2,
                ..Default::default()
            },
        ],
    };
    let feedback = TireFemContactFeedbackOutput {
        velocities: vec![[0.0; 4]; 5],
        bristles: Vec::new(),
        forces: vec![
            [10.0, 0.0, 1.0, 0.0],
            [20.0, 0.0, 2.0, 0.0],
            [30.0, 0.0, 3.0, 0.0],
            [-5.0, 0.0, 11.0, 0.0],
            [-7.0, 0.0, 13.0, 0.0],
        ],
    };
    let out = tire_vehicle_planar_from_feedback_output_reference(&input, &feedback, &layout)
        .expect("feedback planar ref");
    assert_eq!(out.force_xz, [48.0, 30.0]);
    let expected_mz = input.wheel_offsets[0][0] * 6.0 - input.wheel_offsets[0][2] * 60.0
        + input.wheel_offsets[1][0] * 24.0
        - input.wheel_offsets[1][2] * -12.0;
    assert!((out.yaw_moment - expected_mz).abs() < 1.0e-5);
    assert!(out.finite);

    let mut bad_layout = layout.clone();
    bad_layout.rows[1].node_count = 3;
    assert!(
        tire_vehicle_planar_from_feedback_output_reference(&input, &feedback, &bad_layout).is_err(),
        "atlas force ranges must be validated before summing"
    );
}

#[test]
fn tire_vehicle_wheel_spin_reference_matches_vehicle_equation() {
    let input = sample_wheel_spin_input();
    let out = tire_vehicle_wheel_spin_reference(&input).expect("wheel spin reference");
    assert_eq!(out.rows.len(), input.wheel_state.len());
    assert!(out.finite);
    for ((state, command), row) in input
        .wheel_state
        .iter()
        .zip(input.wheel_commands.iter())
        .zip(out.rows.iter())
    {
        let spin = state[0];
        let radius = state[1];
        let inertia = state[2];
        let brake = command[1] * spin.tanh();
        let expected_net = command[0] - brake - command[2] * radius;
        let expected_accel = expected_net / inertia;
        let expected_spin = spin + expected_accel * input.dt;
        assert!((row[0] - expected_spin).abs() < 1.0e-5);
        assert!((row[1] - expected_accel).abs() < 1.0e-5);
        assert!((row[2] - expected_net).abs() < 1.0e-5);
        assert!(row[3] > 0.5);
    }

    let shader = include_str!("../shaders/tire_vehicle_yaw.wgsl");
    assert!(shader.contains("fn integrate_wheel_spin"));
    assert!(shader.contains("drive - brake - tire_fx * radius"));
}

#[test]
fn tire_vehicle_wheel_spin_state_reference_preserves_resident_state_layout() {
    let input = sample_wheel_spin_input();
    let diagnostic = tire_vehicle_wheel_spin_reference(&input).expect("wheel spin diagnostic");
    let state =
        tire_vehicle_wheel_spin_state_reference(&input).expect("wheel spin resident state ref");
    assert_eq!(state.rows.len(), input.wheel_state.len());
    assert!(state.finite);
    for ((input_state, diagnostic_row), state_row) in input
        .wheel_state
        .iter()
        .zip(diagnostic.rows.iter())
        .zip(state.rows.iter())
    {
        assert!((state_row[0] - diagnostic_row[0]).abs() < 1.0e-6);
        assert!((state_row[1] - input_state[1]).abs() < 1.0e-6);
        assert!((state_row[2] - input_state[2]).abs() < 1.0e-6);
        assert!(state_row[3] > 0.5);
    }

    let shader = include_str!("../shaders/tire_vehicle_yaw.wgsl");
    assert!(shader.contains("fn integrate_wheel_spin_state"));
    assert!(shader.contains("x=next_spin_rad_s, y=wheel_radius_m"));
    let source = include_str!("../src/tire_vehicle.rs");
    assert!(source.contains("TireVehicleWheelSpinResidentBatch"));
    assert!(source.contains("step_wheel_spin_resident"));
    assert!(source.contains("readback_wheel_spin_resident"));
}

#[test]
fn tire_vehicle_wheel_kinematics_reference_matches_vehicle_contact_frame() {
    let input = sample_wheel_kinematics_input();
    let out = tire_vehicle_wheel_kinematics_reference(&input).expect("wheel kinematics reference");
    let n = input.wheel_mounts.len();
    assert_eq!(out.rows.len(), n * 2);
    assert!(out.finite);

    let cy = input.yaw.cos();
    let sy = input.yaw.sin();
    for i in 0..n {
        let mount = input.wheel_mounts[i];
        let state = input.wheel_state[i];
        let off_x = mount[0] * cy - mount[2] * sy;
        let off_z = mount[0] * sy + mount[2] * cy;
        let corner_vx = input.velocity_xz[0] - input.yaw_rate * off_z;
        let corner_vz = input.velocity_xz[1] + input.yaw_rate * off_x;
        let heading = input.yaw + state[1];
        let long_x = heading.cos();
        let long_z = heading.sin();
        let lat_x = -heading.sin();
        let lat_z = heading.cos();
        let v_long = corner_vx * long_x + corner_vz * long_z;
        let v_lat = corner_vx * lat_x + corner_vz * lat_z;
        let denom = v_long.abs().max(0.5);
        let kappa = (state[0] * mount[3] - v_long) / denom;
        let alpha = (v_lat / denom).atan();
        let kin = out.rows[i];
        let basis = out.rows[n + i];
        assert!((kin[0] - v_long).abs() < 1.0e-5);
        assert!((kin[1] - v_lat).abs() < 1.0e-5);
        assert!((kin[2] - kappa).abs() < 1.0e-5);
        assert!((kin[3] - alpha).abs() < 1.0e-5);
        assert!((basis[0] - long_x).abs() < 1.0e-6);
        assert!((basis[1] - long_z).abs() < 1.0e-6);
        assert!((basis[2] - lat_x).abs() < 1.0e-6);
        assert!((basis[3] - lat_z).abs() < 1.0e-6);
    }

    assert!(
        out.rows[0][3].abs() > out.rows[2][3].abs(),
        "front steered wheel should have larger slip angle than rear: {:?}",
        out.rows
    );

    let shader = include_str!("../shaders/tire_vehicle_yaw.wgsl");
    assert!(shader.contains("fn compute_wheel_kinematics"));
    assert!(shader.contains("let heading = yaw + wheel.y"));
    assert!(shader.contains("yaw_out[n + i]"));
}

#[test]
fn tire_vehicle_wheel_kinematics_from_spin_reference_uses_resident_spin_state() {
    let spin_rows = tire_vehicle_wheel_spin_state_reference(&sample_wheel_spin_input())
        .expect("spin state reference")
        .rows;
    let input = sample_wheel_kinematics_from_spin_input();
    let out = tire_vehicle_wheel_kinematics_from_spin_reference(&input, &spin_rows)
        .expect("kinematics from resident spin ref");
    let n = input.wheel_mounts.len();
    assert_eq!(out.rows.len(), n * 2);
    assert!(out.finite);

    let base = sample_wheel_kinematics_input();
    let stale = tire_vehicle_wheel_kinematics_reference(&base).expect("stale CPU spin reference");
    assert!(
        (out.rows[0][2] - stale.rows[0][2]).abs() > 1.0e-3,
        "resident kinematics should use stepped GPU spin state, not stale CPU spin"
    );

    let shader = include_str!("../shaders/tire_vehicle_yaw.wgsl");
    assert!(shader.contains("fn compute_wheel_kinematics_from_spin_state"));
    assert!(shader.contains("resident wheel-spin state"));
    let source = include_str!("../src/tire_vehicle.rs");
    assert!(source.contains("dispatch_wheel_kinematics_from_spin_resident_blocking"));
    assert!(source.contains("pipeline_wheel_kinematics_from_spin"));
}

#[test]
fn tire_vehicle_suspension_reference_matches_vehicle_spring_damper() {
    let input = sample_suspension_input();
    let out = tire_vehicle_suspension_reference(&input).expect("suspension reference");
    assert_eq!(out.rows.len(), input.suspension_geometry.len());
    assert!(out.finite);
    for ((geo, params), row) in input
        .suspension_geometry
        .iter()
        .zip(input.suspension_params.iter())
        .zip(out.rows.iter())
    {
        let attach_world_y = input.chassis_y + geo[0];
        let hub_y = params[2] + geo[1];
        let compression = geo[2] - (attach_world_y - hub_y);
        let travel = compression.clamp(0.0, geo[3]);
        let travel_rate = -input.chassis_vy;
        let load = (params[0] * travel + params[1] * travel_rate).max(0.0);
        assert!((row[0] - load).abs() < 1.0e-3);
        assert!((row[1] - travel).abs() < 1.0e-6);
        assert!((row[2] - travel_rate).abs() < 1.0e-6);
        assert!(row[3] > 0.5);
    }
    assert!(
        out.rows[1][0] > out.rows[0][0],
        "higher ground under same corner should increase normal load: {:?}",
        out.rows
    );

    let shader = include_str!("../shaders/tire_vehicle_yaw.wgsl");
    assert!(shader.contains("fn compute_suspension"));
    assert!(shader.contains("stiffness * travel + damping * travel_rate"));
}

#[test]
fn tire_vehicle_vertical_reference_integrates_suspension_load_against_gravity() {
    let suspension = tire_vehicle_suspension_reference(&sample_suspension_input())
        .expect("suspension reference");
    let input = sample_vertical_input();
    let out = tire_vehicle_vertical_from_suspension_reference(&input, &suspension.rows)
        .expect("vertical reference");
    let total_load: f32 = suspension.rows.iter().map(|row| row[0]).sum();
    let expected_force = total_load - input.mass * input.gravity;
    let expected_vy = (input.mass * input.vy + expected_force * input.dt)
        / (input.mass + input.vertical_damping * input.dt);
    let expected_y = input.y + expected_vy * input.dt;
    assert!((out.force_y - expected_force).abs() < 1.0e-3);
    assert!((out.vy - expected_vy).abs() < 1.0e-5);
    assert!((out.y - expected_y).abs() < 1.0e-5);
    assert!(out.finite);

    let shader = include_str!("../shaders/tire_vehicle_yaw.wgsl");
    assert!(shader.contains("partial_fy"));
    assert!(shader.contains("force_y = partial_fy[0] - mass * gravity"));
}

#[test]
fn tire_vehicle_brush_rows_reference_builds_gpu_brush_inputs() {
    let input = sample_brush_rows_input();
    let out = tire_vehicle_brush_rows_reference(&input).expect("brush rows reference");
    assert_eq!(out.wheels.len(), input.wheel_mounts.len());
    assert!(out.finite);

    let first = out.wheels[0];
    let heading = input.chassis_pose[3] + input.wheel_state[0][1];
    assert!((first.heading[0] - heading.cos()).abs() < 1.0e-6);
    assert!((first.heading[2] - heading.sin()).abs() < 1.0e-6);
    assert!((first.spin[0] - input.wheel_state[0][0]).abs() < 1.0e-6);
    assert!((first.spin[1] - input.wheel_mounts[0][3]).abs() < 1.0e-6);
    assert!((first.spin[2] - input.wheel_state[0][2]).abs() < 1.0e-6);
    assert!((first.tire[0] - input.tire_rows[0][0]).abs() < 1.0e-6);
    assert!((first.material[0] - input.material_rows[0][0]).abs() < 1.0e-6);
    assert!((first.extra[2] - input.extra_rows[0][2]).abs() < 1.0e-6);
    assert!(
        out.wheels[1].hub_pos[1] > out.wheels[0].hub_pos[1],
        "higher terrain should raise hub when travel is comparable: {:?} {:?}",
        out.wheels[0].hub_pos,
        out.wheels[1].hub_pos
    );

    let shader = include_str!("../shaders/tire_vehicle_yaw.wgsl");
    assert!(shader.contains("fn build_brush_rows"));
    assert!(shader.contains("yaw_out[base_out + 0u]"));
    assert!(shader.contains("yaw_out[base_out + 6u]"));

    let source = include_str!("../src/tire_vehicle.rs");
    assert!(source.contains("dispatch_brush_rows_resident"));
    assert!(source.contains("TireVehicleBrushRowsResidentOutput"));
    let brush_source = include_str!("../src/tire_brush.rs");
    assert!(brush_source.contains("dispatch_input_buffer_blocking"));
    assert!(brush_source.contains("already packed in a GPU-resident"));
}

#[test]
fn tire_vehicle_brush_rows_from_spin_reference_uses_resident_spin_state() {
    let spin_rows = tire_vehicle_wheel_spin_state_reference(&sample_wheel_spin_input())
        .expect("spin state reference")
        .rows;
    let input = sample_brush_rows_from_spin_input(spin_rows.clone());
    let out =
        tire_vehicle_brush_rows_from_spin_reference(&input).expect("brush rows from spin ref");
    assert_eq!(out.wheels.len(), input.wheel_mounts.len());
    assert!(out.finite);
    for (wheel, spin) in out.wheels.iter().zip(spin_rows.iter()) {
        assert!((wheel.spin[0] - spin[0]).abs() < 1.0e-6);
        assert!((wheel.spin[1] - spin[1]).abs() < 1.0e-6);
    }
    assert!(
        (out.wheels[0].spin[0] - sample_brush_rows_input().wheel_state[0][0]).abs() > 1.0e-3,
        "brush row should use stepped resident spin, not stale CPU wheel_state"
    );

    let shader = include_str!("../shaders/tire_vehicle_yaw.wgsl");
    assert!(shader.contains("fn build_brush_rows_from_spin_state"));
    assert!(shader.contains("resident wheel-spin state"));
    let source = include_str!("../src/tire_vehicle.rs");
    assert!(source.contains("dispatch_brush_rows_from_spin_resident"));
    assert!(source.contains("spin_batch.current_state_buffer()"));
}

#[test]
fn tire_vehicle_brush_rows_from_spin_suspension_reference_uses_resident_travel() {
    let spin_rows = tire_vehicle_wheel_spin_state_reference(&sample_wheel_spin_input())
        .expect("spin state reference")
        .rows;
    let suspension_rows = tire_vehicle_suspension_reference(&sample_suspension_input())
        .expect("suspension reference")
        .rows;
    let mut input = sample_brush_rows_from_spin_input(spin_rows);
    let stale_travel = input.suspension_rows[0][0];
    input.suspension_rows[0][0] = 0.0;
    let out = tire_vehicle_brush_rows_from_spin_suspension_reference(&input, &suspension_rows)
        .expect("brush rows from resident spin+suspension ref");
    assert!(out.finite);
    let expected_hub_y =
        input.suspension_rows[0][2] + out.wheels[0].spin[1] - suspension_rows[0][1];
    assert!((out.wheels[0].hub_pos[1] - expected_hub_y).abs() < 1.0e-6);
    assert!(
        (out.wheels[0].hub_pos[1]
            - (input.suspension_rows[0][2] + out.wheels[0].spin[1] - stale_travel))
            .abs()
            > 1.0e-4,
        "resident suspension travel should override stale CPU travel"
    );

    let shader = include_str!("../shaders/tire_vehicle_yaw.wgsl");
    assert!(shader.contains("fn build_brush_rows_from_spin_and_suspension_state"));
    assert!(shader.contains("resident_rows[i]"));
    let source = include_str!("../src/tire_vehicle.rs");
    assert!(source.contains("TireVehicleSuspensionResidentBatch"));
    assert!(source.contains("TireVehicleResidentBrushOwner"));
    assert!(source.contains("dispatch_brush_rows_from_spin_suspension_resident"));
    assert!(source.contains("step_resident_brush_owner"));
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_vehicle_yaw_gpu_matches_reference() {
    let engine = try_engine().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireVehicleYawGpuKernel::new(&device);
    let input = sample_yaw_input();
    let expected = tire_vehicle_yaw_reference(&input).expect("reference");
    let got = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("GPU yaw");
    assert!(got.finite);
    assert!((got.yaw_moment - expected.yaw_moment).abs() < 2.0e-4);
    assert!((got.yaw_rate - expected.yaw_rate).abs() < 2.0e-5);
    assert!((got.yaw - expected.yaw).abs() < 2.0e-5);
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_vehicle_planar_gpu_matches_reference() {
    let engine = try_engine().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireVehicleYawGpuKernel::new(&device);
    let input = sample_planar_input();
    let expected = tire_vehicle_planar_reference(&input).expect("reference");
    let got = kernel
        .dispatch_planar_blocking(&device, &queue, &input)
        .expect("GPU planar vehicle");
    assert!(got.finite);
    assert!((got.force_xz[0] - expected.force_xz[0]).abs() < 2.0e-4);
    assert!((got.force_xz[1] - expected.force_xz[1]).abs() < 2.0e-4);
    assert!((got.yaw_moment - expected.yaw_moment).abs() < 2.0e-4);
    assert!((got.velocity_xz[0] - expected.velocity_xz[0]).abs() < 2.0e-5);
    assert!((got.velocity_xz[1] - expected.velocity_xz[1]).abs() < 2.0e-5);
    assert!((got.position_xz[0] - expected.position_xz[0]).abs() < 2.0e-5);
    assert!((got.position_xz[1] - expected.position_xz[1]).abs() < 2.0e-5);
    assert!((got.yaw_rate - expected.yaw_rate).abs() < 2.0e-5);
    assert!((got.yaw - expected.yaw).abs() < 2.0e-5);
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_vehicle_planar_contact_reduction_gpu_matches_reference() {
    let engine = try_engine().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireVehicleYawGpuKernel::new(&device);
    let input = sample_planar_from_brush_input();
    let mut rows = Vec::<[f32; 4]>::with_capacity(input.wheel_offsets.len() * 4);
    for i in 0..input.wheel_offsets.len() {
        rows.push([0.0, 0.0, 0.0, 0.0]);
        rows.push([0.0, 0.0, 0.0, 0.0]);
        rows.push([90.0 + i as f32 * 25.0, 0.0, 35.0 - i as f32 * 8.0, 0.0]);
        rows.push([0.0, 5.0 + i as f32 * 1.5, 0.0, 0.0]);
    }
    let expected = tire_vehicle_planar_from_contact_reduction_rows_reference(&input, &rows)
        .expect("reference");
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_vehicle contact reduction rows"),
        size: core::mem::size_of_val(rows.as_slice()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytemuck::cast_slice(&rows));
    let got = kernel
        .dispatch_planar_from_contact_reduction_output_blocking(
            &device,
            &queue,
            &input,
            &buf,
            input.wheel_offsets.len(),
        )
        .expect("GPU planar from contact reduction");
    assert!(got.finite);
    assert!((got.force_xz[0] - expected.force_xz[0]).abs() < 2.0e-4);
    assert!((got.force_xz[1] - expected.force_xz[1]).abs() < 2.0e-4);
    assert!((got.yaw_moment - expected.yaw_moment).abs() < 2.0e-4);
    assert!((got.velocity_xz[0] - expected.velocity_xz[0]).abs() < 2.0e-5);
    assert!((got.velocity_xz[1] - expected.velocity_xz[1]).abs() < 2.0e-5);
    assert!((got.position_xz[0] - expected.position_xz[0]).abs() < 2.0e-5);
    assert!((got.position_xz[1] - expected.position_xz[1]).abs() < 2.0e-5);
    assert!((got.yaw_rate - expected.yaw_rate).abs() < 2.0e-5);
    assert!((got.yaw - expected.yaw).abs() < 2.0e-5);
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_vehicle_planar_resident_state_ping_pongs_from_contact_reduction_output() {
    let engine = try_engine().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireVehicleYawGpuKernel::new(&device);
    let input = sample_planar_from_brush_input();
    let mut rows = Vec::<[f32; 4]>::with_capacity(input.wheel_offsets.len() * 4);
    for i in 0..input.wheel_offsets.len() {
        rows.push([0.0, 0.0, 0.0, 0.0]);
        rows.push([0.0, 0.0, 0.0, 0.0]);
        rows.push([90.0 + i as f32 * 25.0, 0.0, 35.0 - i as f32 * 8.0, 0.0]);
        rows.push([0.0, 5.0 + i as f32 * 1.5, 0.0, 0.0]);
    }
    let expected = tire_vehicle_planar_from_contact_reduction_rows_reference(&input, &rows)
        .expect("reference");
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_vehicle resident contact reduction rows"),
        size: core::mem::size_of_val(rows.as_slice()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytemuck::cast_slice(&rows));
    let mut state = kernel
        .create_planar_resident_state(&device, &queue, &input, input.wheel_offsets.len())
        .expect("resident planar state");
    kernel
        .step_planar_resident_from_contact_reduction_output(
            &device,
            &queue,
            &mut state,
            &buf,
            input.wheel_offsets.len(),
            input.dt,
        )
        .expect("resident planar from contact reduction");
    let got = kernel
        .readback_planar_resident_state(&device, &queue, &state)
        .expect("resident planar readback");
    assert!(got.finite);
    assert!((got.force_xz[0] - expected.force_xz[0]).abs() < 2.0e-4);
    assert!((got.force_xz[1] - expected.force_xz[1]).abs() < 2.0e-4);
    assert!((got.yaw_moment - expected.yaw_moment).abs() < 2.0e-4);
    assert!((got.velocity_xz[0] - expected.velocity_xz[0]).abs() < 2.0e-5);
    assert!((got.velocity_xz[1] - expected.velocity_xz[1]).abs() < 2.0e-5);
    assert!((got.position_xz[0] - expected.position_xz[0]).abs() < 2.0e-5);
    assert!((got.position_xz[1] - expected.position_xz[1]).abs() < 2.0e-5);
    assert!((got.yaw_rate - expected.yaw_rate).abs() < 2.0e-5);
    assert!((got.yaw - expected.yaw).abs() < 2.0e-5);
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_vehicle_planar_resident_state_ping_pongs_from_feedback_output() {
    let engine = try_engine().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireVehicleYawGpuKernel::new(&device);
    let mut input = sample_planar_from_brush_input();
    input.wheel_offsets.truncate(2);
    let layout = TireFemGpuBatchAtlasLayout {
        plan: TireFemGpuBatchDispatchPlan {
            tire_count: 2,
            total_node_count: 5,
            total_tet_count: 2,
            total_tread_count: 0,
            total_bead_count: 0,
            total_xpbd_edge_count: 0,
            total_xpbd_dihedral_count: 0,
            max_nodes_per_tire: 3,
            node_workgroup_count: 1,
            persistent_state_bytes: (5 * core::mem::size_of::<[f32; 4]>()) as u64,
        },
        rows: vec![
            TireFemGpuBatchAtlasRow {
                tire_index: 0,
                node_offset: 0,
                node_count: 3,
                ..Default::default()
            },
            TireFemGpuBatchAtlasRow {
                tire_index: 1,
                node_offset: 3,
                node_count: 2,
                ..Default::default()
            },
        ],
    };
    let feedback = TireFemContactFeedbackOutput {
        velocities: vec![[0.0; 4]; 5],
        bristles: vec![[0.0; 4]; 5],
        forces: vec![
            [10.0, 0.0, 1.0, 0.0],
            [20.0, 0.0, 2.0, 0.0],
            [30.0, 0.0, 3.0, 0.0],
            [-5.0, 0.0, 11.0, 0.0],
            [-7.0, 0.0, 13.0, 0.0],
        ],
    };
    let expected = tire_vehicle_planar_from_feedback_output_reference(&input, &feedback, &layout)
        .expect("reference");
    let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
    let force_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_vehicle resident feedback force rows"),
        size: vec4_bytes(feedback.forces.len()),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&force_buf, 0, bytemuck::cast_slice(&feedback.forces));
    let velocities = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_vehicle resident feedback dummy velocities"),
        size: vec4_bytes(feedback.velocities.len()),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bristles = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_vehicle resident feedback dummy bristles"),
        size: vec4_bytes(feedback.bristles.len()),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let resident = TireFemContactFeedbackResidentOutput {
        node_count: feedback.forces.len(),
        bristle_count: feedback.bristles.len(),
        velocities,
        bristles,
        forces: force_buf,
    };
    let mut state = kernel
        .create_planar_resident_state(&device, &queue, &input, input.wheel_offsets.len())
        .expect("resident planar state");
    kernel
        .step_planar_resident_from_feedback_output(
            &device, &queue, &mut state, &resident, &layout, input.dt,
        )
        .expect("resident planar from feedback");
    let got = kernel
        .readback_planar_resident_state(&device, &queue, &state)
        .expect("resident planar readback");
    assert!(got.finite);
    assert!((got.force_xz[0] - expected.force_xz[0]).abs() < 2.0e-4);
    assert!((got.force_xz[1] - expected.force_xz[1]).abs() < 2.0e-4);
    assert!((got.yaw_moment - expected.yaw_moment).abs() < 2.0e-4);
    assert!((got.velocity_xz[0] - expected.velocity_xz[0]).abs() < 2.0e-5);
    assert!((got.velocity_xz[1] - expected.velocity_xz[1]).abs() < 2.0e-5);
    assert!((got.position_xz[0] - expected.position_xz[0]).abs() < 2.0e-5);
    assert!((got.position_xz[1] - expected.position_xz[1]).abs() < 2.0e-5);
    assert!((got.yaw_rate - expected.yaw_rate).abs() < 2.0e-5);
    assert!((got.yaw - expected.yaw).abs() < 2.0e-5);
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_vehicle_wheel_spin_gpu_matches_reference() {
    let engine = try_engine().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireVehicleYawGpuKernel::new(&device);
    let input = sample_wheel_spin_input();
    let expected = tire_vehicle_wheel_spin_reference(&input).expect("reference");
    let got = kernel
        .dispatch_wheel_spin_blocking(&device, &queue, &input)
        .expect("GPU wheel spin");
    assert!(got.finite);
    assert_eq!(got.rows.len(), expected.rows.len());
    for (a, b) in got.rows.iter().zip(expected.rows.iter()) {
        assert!((a[0] - b[0]).abs() < 2.0e-5);
        assert!((a[1] - b[1]).abs() < 2.0e-4);
        assert!((a[2] - b[2]).abs() < 2.0e-4);
        assert_eq!(a[3] > 0.5, b[3] > 0.5);
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_vehicle_wheel_spin_resident_batch_ping_pongs_state() {
    let engine = try_engine().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireVehicleYawGpuKernel::new(&device);
    let input = sample_wheel_spin_input();
    let expected = tire_vehicle_wheel_spin_state_reference(&input).expect("reference");
    let mut batch = kernel
        .create_wheel_spin_resident_batch(&device, &queue, &input)
        .expect("resident wheel-spin batch");
    assert_eq!(batch.wheel_count(), input.wheel_state.len());
    kernel
        .step_wheel_spin_resident(&device, &queue, &mut batch, input.dt)
        .expect("resident wheel-spin step");
    let got = kernel
        .readback_wheel_spin_resident(&device, &queue, &batch)
        .expect("resident wheel-spin readback");
    assert!(got.finite);
    assert_eq!(got.rows.len(), expected.rows.len());
    for (a, b) in got.rows.iter().zip(expected.rows.iter()) {
        for k in 0..4 {
            assert!(
                (a[k] - b[k]).abs() < 2.0e-5,
                "resident state mismatch gpu={a:?} expected={b:?}"
            );
        }
    }

    let second_input = TireVehicleWheelSpinGpuInput {
        wheel_state: expected.rows.clone(),
        wheel_commands: vec![[0.0, 10.0, 80.0, 0.0]; expected.rows.len()],
        dt: input.dt * 0.5,
    };
    kernel
        .upload_wheel_spin_resident_commands(&queue, &batch, &second_input.wheel_commands)
        .expect("upload resident wheel-spin commands");
    kernel
        .step_wheel_spin_resident(&device, &queue, &mut batch, second_input.dt)
        .expect("second resident wheel-spin step");
    let second_expected =
        tire_vehicle_wheel_spin_state_reference(&second_input).expect("second reference");
    let second_got = kernel
        .readback_wheel_spin_resident(&device, &queue, &batch)
        .expect("second resident readback");
    assert!(second_got.finite);
    for (a, b) in second_got.rows.iter().zip(second_expected.rows.iter()) {
        for k in 0..4 {
            assert!(
                (a[k] - b[k]).abs() < 2.0e-5,
                "second resident state mismatch gpu={a:?} expected={b:?}"
            );
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_vehicle_resident_spin_state_feeds_brush_gpu_without_staging() {
    let engine = try_engine().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let vehicle_kernel = TireVehicleYawGpuKernel::new(&device);
    let brush_kernel = TireBrushGpuKernel::new(&device);
    let spin_input = sample_wheel_spin_input();
    let spin_state = tire_vehicle_wheel_spin_state_reference(&spin_input)
        .expect("resident spin state reference")
        .rows;
    let brush_input = sample_brush_rows_from_spin_input(spin_state);
    let expected_rows =
        tire_vehicle_brush_rows_from_spin_reference(&brush_input).expect("reference rows");
    let expected = brush_kernel
        .dispatch_blocking(&device, &queue, &expected_rows.wheels, None)
        .expect("staged brush dispatch");
    let mut spin_batch = vehicle_kernel
        .create_wheel_spin_resident_batch(&device, &queue, &spin_input)
        .expect("resident spin batch");
    vehicle_kernel
        .step_wheel_spin_resident(&device, &queue, &mut spin_batch, spin_input.dt)
        .expect("resident spin step");
    let resident_rows = vehicle_kernel
        .dispatch_brush_rows_from_spin_resident(&device, &queue, &spin_batch, &brush_input)
        .expect("resident spin-to-brush rows");
    let got = brush_kernel
        .dispatch_input_buffer_blocking(
            &device,
            &queue,
            &resident_rows.buffer,
            resident_rows.wheel_count,
            None,
        )
        .expect("resident brush dispatch");
    assert_eq!(got.len(), expected.len());
    for (a, b) in got.iter().zip(expected.iter()) {
        for (a_row, b_row) in [
            (&a.forces, &b.forces),
            (&a.diag1, &b.diag1),
            (&a.diag2, &b.diag2),
            (&a.diag3, &b.diag3),
            (&a.diag4, &b.diag4),
            (&a.diag5, &b.diag5),
            (&a.world_force, &b.world_force),
        ] {
            for k in 0..4 {
                let tol = 2.5e-3_f32.max(b_row[k].abs() * 2.0e-5);
                assert!(
                    (a_row[k] - b_row[k]).abs() < tol,
                    "resident spin-to-brush output mismatch got={a_row:?} expected={b_row:?} tol={tol}"
                );
            }
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_vehicle_resident_spin_and_suspension_feed_brush_gpu_without_staging() {
    let engine = try_engine().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let vehicle_kernel = TireVehicleYawGpuKernel::new(&device);
    let brush_kernel = TireBrushGpuKernel::new(&device);
    let spin_input = sample_wheel_spin_input();
    let suspension_input = sample_suspension_input();
    let spin_rows = tire_vehicle_wheel_spin_state_reference(&spin_input)
        .expect("resident spin state reference")
        .rows;
    let suspension_rows = tire_vehicle_suspension_reference(&suspension_input)
        .expect("suspension reference")
        .rows;
    let brush_input = sample_brush_rows_from_spin_input(spin_rows);
    let expected_rows =
        tire_vehicle_brush_rows_from_spin_suspension_reference(&brush_input, &suspension_rows)
            .expect("reference spin+suspension rows");
    let expected = brush_kernel
        .dispatch_blocking(&device, &queue, &expected_rows.wheels, None)
        .expect("staged brush dispatch");
    let mut spin_batch = vehicle_kernel
        .create_wheel_spin_resident_batch(&device, &queue, &spin_input)
        .expect("resident spin batch");
    vehicle_kernel
        .step_wheel_spin_resident(&device, &queue, &mut spin_batch, spin_input.dt)
        .expect("resident spin step");
    let suspension_batch = vehicle_kernel
        .create_suspension_resident_batch(&device, &queue, &suspension_input)
        .expect("resident suspension batch");
    vehicle_kernel
        .step_suspension_resident(
            &device,
            &queue,
            &suspension_batch,
            suspension_input.chassis_y,
            suspension_input.chassis_vy,
        )
        .expect("resident suspension step");
    let resident_rows = vehicle_kernel
        .dispatch_brush_rows_from_spin_suspension_resident(
            &device,
            &queue,
            &spin_batch,
            &suspension_batch,
            &brush_input,
        )
        .expect("resident spin+suspension-to-brush rows");
    let got = brush_kernel
        .dispatch_input_buffer_blocking(
            &device,
            &queue,
            &resident_rows.buffer,
            resident_rows.wheel_count,
            None,
        )
        .expect("resident brush dispatch");
    assert_eq!(got.len(), expected.len());
    for (a, b) in got.iter().zip(expected.iter()) {
        for (a_row, b_row) in [
            (&a.forces, &b.forces),
            (&a.diag1, &b.diag1),
            (&a.diag2, &b.diag2),
            (&a.diag3, &b.diag3),
            (&a.diag4, &b.diag4),
            (&a.diag5, &b.diag5),
            (&a.world_force, &b.world_force),
        ] {
            for k in 0..4 {
                let tol = 2.5e-3_f32.max(b_row[k].abs() * 2.0e-5);
                assert!(
                    (a_row[k] - b_row[k]).abs() < tol,
                    "resident spin+suspension brush output mismatch got={a_row:?} expected={b_row:?} tol={tol}"
                );
            }
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_vehicle_resident_brush_owner_sequences_spin_suspension_and_brush() {
    let engine = try_engine().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let vehicle_kernel = TireVehicleYawGpuKernel::new(&device);
    let brush_kernel = TireBrushGpuKernel::new(&device);
    let spin_input = sample_wheel_spin_input();
    let suspension_input = sample_suspension_input();
    let spin_rows = tire_vehicle_wheel_spin_state_reference(&spin_input)
        .expect("resident spin state reference")
        .rows;
    let suspension_rows = tire_vehicle_suspension_reference(&suspension_input)
        .expect("suspension reference")
        .rows;
    let brush_input = sample_brush_rows_from_spin_input(spin_rows);
    let expected_rows =
        tire_vehicle_brush_rows_from_spin_suspension_reference(&brush_input, &suspension_rows)
            .expect("reference owner rows");
    let expected = brush_kernel
        .dispatch_blocking(&device, &queue, &expected_rows.wheels, None)
        .expect("staged brush dispatch");
    let mut owner = vehicle_kernel
        .create_resident_brush_owner(&device, &queue, &spin_input, &suspension_input, brush_input)
        .expect("resident brush owner");
    assert_eq!(owner.wheel_count(), spin_input.wheel_state.len());
    let resident_rows = vehicle_kernel
        .step_resident_brush_owner(
            &device,
            &queue,
            &mut owner,
            spin_input.dt,
            sample_brush_rows_input().chassis_pose,
            sample_brush_rows_input().chassis_velocity,
            suspension_input.chassis_y,
            suspension_input.chassis_vy,
        )
        .expect("resident brush-owner step");
    let got = brush_kernel
        .dispatch_input_buffer_blocking(
            &device,
            &queue,
            &resident_rows.buffer,
            resident_rows.wheel_count,
            None,
        )
        .expect("resident owner brush dispatch");
    assert_eq!(got.len(), expected.len());
    for (a, b) in got.iter().zip(expected.iter()) {
        for (a_row, b_row) in [
            (&a.forces, &b.forces),
            (&a.diag1, &b.diag1),
            (&a.diag2, &b.diag2),
            (&a.diag3, &b.diag3),
            (&a.diag4, &b.diag4),
            (&a.diag5, &b.diag5),
            (&a.world_force, &b.world_force),
        ] {
            for k in 0..4 {
                let tol = 2.5e-3_f32.max(b_row[k].abs() * 2.0e-5);
                assert!(
                    (a_row[k] - b_row[k]).abs() < tol,
                    "resident brush-owner output mismatch got={a_row:?} expected={b_row:?} tol={tol}"
                );
            }
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_vehicle_resident_brush_output_feeds_planar_gpu_without_staging() {
    let engine = try_engine().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let vehicle_kernel = TireVehicleYawGpuKernel::new(&device);
    let brush_kernel = TireBrushGpuKernel::new(&device);
    let spin_input = sample_wheel_spin_input();
    let suspension_input = sample_suspension_input();
    let spin_rows = tire_vehicle_wheel_spin_state_reference(&spin_input)
        .expect("resident spin state reference")
        .rows;
    let brush_input = sample_brush_rows_from_spin_input(spin_rows);
    let mut owner = vehicle_kernel
        .create_resident_brush_owner(&device, &queue, &spin_input, &suspension_input, brush_input)
        .expect("resident brush owner");
    let brush_rows = vehicle_kernel
        .step_resident_brush_owner(
            &device,
            &queue,
            &mut owner,
            spin_input.dt,
            sample_brush_rows_input().chassis_pose,
            sample_brush_rows_input().chassis_velocity,
            suspension_input.chassis_y,
            suspension_input.chassis_vy,
        )
        .expect("resident brush-owner rows");
    let staged_brush = brush_kernel
        .dispatch_input_buffer_blocking(
            &device,
            &queue,
            &brush_rows.buffer,
            brush_rows.wheel_count,
            None,
        )
        .expect("staged brush output for reference");
    let expected =
        tire_vehicle_planar_from_brush_reference(&sample_planar_from_brush_input(), &staged_brush)
            .expect("planar brush reference");
    let resident_brush = brush_kernel
        .dispatch_input_buffer_resident(
            &device,
            &queue,
            &brush_rows.buffer,
            brush_rows.wheel_count,
            None,
        )
        .expect("resident brush output");
    let got = vehicle_kernel
        .dispatch_planar_from_brush_output_blocking(
            &device,
            &queue,
            &sample_planar_from_brush_input(),
            &resident_brush.buffer,
            resident_brush.wheel_count,
        )
        .expect("resident brush output to planar chassis");
    assert!(got.finite);
    let tol_fx = 2.5e-3_f32.max(expected.force_xz[0].abs() * 2.0e-5);
    let tol_fz = 2.5e-3_f32.max(expected.force_xz[1].abs() * 2.0e-5);
    let tol_mz = 2.5e-3_f32.max(expected.yaw_moment.abs() * 2.0e-5);
    assert!(
        (got.force_xz[0] - expected.force_xz[0]).abs() < tol_fx,
        "Fx mismatch got={got:?} expected={expected:?} tol={tol_fx}"
    );
    assert!(
        (got.force_xz[1] - expected.force_xz[1]).abs() < tol_fz,
        "Fz mismatch got={got:?} expected={expected:?} tol={tol_fz}"
    );
    assert!(
        (got.yaw_moment - expected.yaw_moment).abs() < tol_mz,
        "Mz mismatch got={got:?} expected={expected:?} tol={tol_mz}"
    );
    assert!((got.velocity_xz[0] - expected.velocity_xz[0]).abs() < 2.0e-5);
    assert!((got.velocity_xz[1] - expected.velocity_xz[1]).abs() < 2.0e-5);
    assert!((got.position_xz[0] - expected.position_xz[0]).abs() < 2.0e-5);
    assert!((got.position_xz[1] - expected.position_xz[1]).abs() < 2.0e-5);
    assert!((got.yaw_rate - expected.yaw_rate).abs() < 2.0e-5);
    assert!((got.yaw - expected.yaw).abs() < 2.0e-5);
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_vehicle_planar_resident_state_ping_pongs_from_brush_output() {
    let engine = try_engine().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let vehicle_kernel = TireVehicleYawGpuKernel::new(&device);
    let brush_kernel = TireBrushGpuKernel::new(&device);
    let spin_input = sample_wheel_spin_input();
    let suspension_input = sample_suspension_input();
    let spin_rows = tire_vehicle_wheel_spin_state_reference(&spin_input)
        .expect("resident spin state reference")
        .rows;
    let brush_input = sample_brush_rows_from_spin_input(spin_rows);
    let mut owner = vehicle_kernel
        .create_resident_brush_owner(&device, &queue, &spin_input, &suspension_input, brush_input)
        .expect("resident brush owner");
    let brush_rows = vehicle_kernel
        .step_resident_brush_owner(
            &device,
            &queue,
            &mut owner,
            spin_input.dt,
            sample_brush_rows_input().chassis_pose,
            sample_brush_rows_input().chassis_velocity,
            suspension_input.chassis_y,
            suspension_input.chassis_vy,
        )
        .expect("resident brush-owner rows");
    let staged_brush = brush_kernel
        .dispatch_input_buffer_blocking(
            &device,
            &queue,
            &brush_rows.buffer,
            brush_rows.wheel_count,
            None,
        )
        .expect("staged brush output for reference");
    let input = sample_planar_from_brush_input();
    let expected = tire_vehicle_planar_from_brush_reference(&input, &staged_brush)
        .expect("planar brush reference");
    let resident_brush = brush_kernel
        .dispatch_input_buffer_resident(
            &device,
            &queue,
            &brush_rows.buffer,
            brush_rows.wheel_count,
            None,
        )
        .expect("resident brush output");
    let mut planar = vehicle_kernel
        .create_planar_resident_state(&device, &queue, &input, resident_brush.wheel_count)
        .expect("resident planar state");
    assert_eq!(planar.wheel_count(), resident_brush.wheel_count);
    vehicle_kernel
        .step_planar_resident_from_brush_output(
            &device,
            &queue,
            &mut planar,
            &resident_brush.buffer,
            resident_brush.wheel_count,
            input.dt,
        )
        .expect("resident planar step");
    let got = vehicle_kernel
        .readback_planar_resident_state(&device, &queue, &planar)
        .expect("resident planar readback");
    assert!(got.finite);
    assert!((got.force_xz[0] - expected.force_xz[0]).abs() < expected.force_xz[0].abs() * 2.0e-5);
    assert!(
        (got.force_xz[1] - expected.force_xz[1]).abs()
            < 2.5e-3_f32.max(expected.force_xz[1].abs() * 2.0e-5)
    );
    assert!(
        (got.yaw_moment - expected.yaw_moment).abs()
            < 2.5e-3_f32.max(expected.yaw_moment.abs() * 2.0e-5)
    );
    assert!((got.velocity_xz[0] - expected.velocity_xz[0]).abs() < 2.0e-5);
    assert!((got.velocity_xz[1] - expected.velocity_xz[1]).abs() < 2.0e-5);
    assert!((got.position_xz[0] - expected.position_xz[0]).abs() < 2.0e-5);
    assert!((got.position_xz[1] - expected.position_xz[1]).abs() < 2.0e-5);
    assert!((got.yaw_rate - expected.yaw_rate).abs() < 2.0e-5);
    assert!((got.yaw - expected.yaw).abs() < 2.0e-5);
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_vehicle_resident_chassis_brush_owner_sequences_brush_and_planar_state() {
    let engine = try_engine().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let vehicle_kernel = TireVehicleYawGpuKernel::new(&device);
    let brush_kernel = TireBrushGpuKernel::new(&device);
    let planar_input = sample_planar_from_brush_input();
    let mut spin_input = sample_wheel_spin_input();
    spin_input.dt = planar_input.dt;
    let suspension_input = sample_suspension_input();
    let spin_rows = tire_vehicle_wheel_spin_state_reference(&spin_input)
        .expect("resident spin state reference")
        .rows;
    let suspension_rows = tire_vehicle_suspension_reference(&suspension_input)
        .expect("suspension reference")
        .rows;
    let brush_input = sample_brush_rows_from_spin_input(spin_rows);
    let expected_rows =
        tire_vehicle_brush_rows_from_spin_suspension_reference(&brush_input, &suspension_rows)
            .expect("reference owner rows");
    let expected_brush = brush_kernel
        .dispatch_blocking(&device, &queue, &expected_rows.wheels, None)
        .expect("staged brush dispatch");
    let expected = tire_vehicle_planar_from_brush_reference(&planar_input, &expected_brush)
        .expect("planar brush reference");
    let mut owner = vehicle_kernel
        .create_resident_chassis_brush_owner(
            &device,
            &queue,
            &spin_input,
            &suspension_input,
            brush_input,
            &planar_input,
        )
        .expect("resident chassis+brush owner");
    assert_eq!(owner.wheel_count(), spin_input.wheel_state.len());
    assert_eq!(owner.planar.wheel_count(), spin_input.wheel_state.len());
    let resident_brush = vehicle_kernel
        .step_resident_chassis_brush_owner(
            &brush_kernel,
            &device,
            &queue,
            &mut owner,
            planar_input.dt,
            sample_brush_rows_input().chassis_pose,
            sample_brush_rows_input().chassis_velocity,
            suspension_input.chassis_y,
            suspension_input.chassis_vy,
            None,
        )
        .expect("resident chassis+brush step");
    assert_eq!(resident_brush.wheel_count, spin_input.wheel_state.len());
    let got = vehicle_kernel
        .readback_planar_resident_state(&device, &queue, &owner.planar)
        .expect("resident planar readback");
    assert!(got.finite);
    assert!(
        (got.force_xz[0] - expected.force_xz[0]).abs()
            < 2.5e-3_f32.max(expected.force_xz[0].abs() * 2.0e-5),
        "Fx mismatch got={got:?} expected={expected:?}"
    );
    assert!(
        (got.force_xz[1] - expected.force_xz[1]).abs()
            < 2.5e-3_f32.max(expected.force_xz[1].abs() * 2.0e-5),
        "Fz mismatch got={got:?} expected={expected:?}"
    );
    assert!(
        (got.yaw_moment - expected.yaw_moment).abs()
            < 2.5e-3_f32.max(expected.yaw_moment.abs() * 2.0e-5),
        "Mz mismatch got={got:?} expected={expected:?}"
    );
    assert!((got.velocity_xz[0] - expected.velocity_xz[0]).abs() < 2.0e-5);
    assert!((got.velocity_xz[1] - expected.velocity_xz[1]).abs() < 2.0e-5);
    assert!((got.position_xz[0] - expected.position_xz[0]).abs() < 2.0e-5);
    assert!((got.position_xz[1] - expected.position_xz[1]).abs() < 2.0e-5);
    assert!((got.yaw_rate - expected.yaw_rate).abs() < 2.0e-5);
    assert!((got.yaw - expected.yaw).abs() < 2.0e-5);
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_vehicle_resident_chassis_brush_owner_feeds_planar_state_back_to_brush_rows() {
    let engine = try_engine().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let vehicle_kernel = TireVehicleYawGpuKernel::new(&device);
    let brush_kernel = TireBrushGpuKernel::new(&device);
    let planar_input = sample_planar_from_brush_input();
    let mut spin_input = sample_wheel_spin_input();
    spin_input.dt = planar_input.dt;
    let suspension_input = sample_suspension_input();
    let spin_rows = tire_vehicle_wheel_spin_state_reference(&spin_input)
        .expect("resident spin state reference")
        .rows;
    let suspension_rows = tire_vehicle_suspension_reference(&suspension_input)
        .expect("suspension reference")
        .rows;
    let mut brush_input = sample_brush_rows_from_spin_input(spin_rows.clone());
    brush_input.chassis_pose = [-50.0, suspension_input.chassis_y, 25.0, -1.25];
    brush_input.chassis_velocity = [-9.0, suspension_input.chassis_vy, 7.0, -0.75];

    let mut expected_input = brush_input.clone();
    expected_input.chassis_pose = [
        planar_input.position_xz[0],
        suspension_input.chassis_y,
        planar_input.position_xz[1],
        planar_input.yaw,
    ];
    expected_input.chassis_velocity = [
        planar_input.velocity_xz[0],
        suspension_input.chassis_vy,
        planar_input.velocity_xz[1],
        planar_input.yaw_rate,
    ];
    let expected_rows =
        tire_vehicle_brush_rows_from_spin_suspension_reference(&expected_input, &suspension_rows)
            .expect("reference rows from resident planar state");
    let expected_brush = brush_kernel
        .dispatch_blocking(&device, &queue, &expected_rows.wheels, None)
        .expect("staged brush output reference");
    let expected_planar = tire_vehicle_planar_from_brush_reference(&planar_input, &expected_brush)
        .expect("planar reference");

    let mut owner = vehicle_kernel
        .create_resident_chassis_brush_owner(
            &device,
            &queue,
            &spin_input,
            &suspension_input,
            brush_input,
            &planar_input,
        )
        .expect("resident chassis+brush owner");
    vehicle_kernel
        .step_resident_chassis_brush_owner_from_planar_state(
            &brush_kernel,
            &device,
            &queue,
            &mut owner,
            planar_input.dt,
            suspension_input.chassis_y,
            suspension_input.chassis_vy,
            None,
        )
        .expect("closed-loop resident chassis+brush step");
    let got = vehicle_kernel
        .readback_planar_resident_state(&device, &queue, &owner.planar)
        .expect("resident planar readback");
    assert!(got.finite);
    assert!(
        (got.force_xz[0] - expected_planar.force_xz[0]).abs()
            < 2.5e-3_f32.max(expected_planar.force_xz[0].abs() * 2.0e-5),
        "Fx mismatch got={got:?} expected={expected_planar:?}"
    );
    assert!(
        (got.force_xz[1] - expected_planar.force_xz[1]).abs()
            < 2.5e-3_f32.max(expected_planar.force_xz[1].abs() * 2.0e-5),
        "Fz mismatch got={got:?} expected={expected_planar:?}"
    );
    assert!(
        (got.yaw_moment - expected_planar.yaw_moment).abs()
            < 2.5e-3_f32.max(expected_planar.yaw_moment.abs() * 2.0e-5),
        "Mz mismatch got={got:?} expected={expected_planar:?}"
    );
    assert!((got.velocity_xz[0] - expected_planar.velocity_xz[0]).abs() < 2.0e-5);
    assert!((got.velocity_xz[1] - expected_planar.velocity_xz[1]).abs() < 2.0e-5);
    assert!((got.position_xz[0] - expected_planar.position_xz[0]).abs() < 2.0e-5);
    assert!((got.position_xz[1] - expected_planar.position_xz[1]).abs() < 2.0e-5);
    assert!((got.yaw_rate - expected_planar.yaw_rate).abs() < 2.0e-5);
    assert!((got.yaw - expected_planar.yaw).abs() < 2.0e-5);
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_vehicle_resident_chassis_brush_owner_uses_resident_planar_and_vertical_state() {
    let engine = try_engine().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let vehicle_kernel = TireVehicleYawGpuKernel::new(&device);
    let brush_kernel = TireBrushGpuKernel::new(&device);
    let planar_input = sample_planar_from_brush_input();
    let vertical_input = sample_vertical_input();
    let mut spin_input = sample_wheel_spin_input();
    spin_input.dt = planar_input.dt;
    let mut suspension_input = sample_suspension_input();
    suspension_input.chassis_y = -5.0;
    suspension_input.chassis_vy = 3.0;
    let spin_rows = tire_vehicle_wheel_spin_state_reference(&spin_input)
        .expect("resident spin state reference")
        .rows;
    let mut expected_suspension_input = suspension_input.clone();
    expected_suspension_input.chassis_y = vertical_input.y;
    expected_suspension_input.chassis_vy = vertical_input.vy;
    let suspension_rows = tire_vehicle_suspension_reference(&expected_suspension_input)
        .expect("resident vertical suspension reference")
        .rows;
    let mut brush_input = sample_brush_rows_from_spin_input(spin_rows.clone());
    brush_input.chassis_pose = [99.0, -9.0, 88.0, 1.75];
    brush_input.chassis_velocity = [-17.0, 11.0, 13.0, -2.5];

    let mut expected_input = brush_input.clone();
    expected_input.chassis_pose = [
        planar_input.position_xz[0],
        vertical_input.y,
        planar_input.position_xz[1],
        planar_input.yaw,
    ];
    expected_input.chassis_velocity = [
        planar_input.velocity_xz[0],
        vertical_input.vy,
        planar_input.velocity_xz[1],
        planar_input.yaw_rate,
    ];
    let expected_rows =
        tire_vehicle_brush_rows_from_spin_suspension_reference(&expected_input, &suspension_rows)
            .expect("reference rows from resident chassis state");
    let expected_brush = brush_kernel
        .dispatch_blocking(&device, &queue, &expected_rows.wheels, None)
        .expect("staged brush output reference");
    let expected_planar = tire_vehicle_planar_from_brush_reference(&planar_input, &expected_brush)
        .expect("planar reference");
    let expected_vertical =
        tire_vehicle_vertical_from_suspension_reference(&vertical_input, &suspension_rows)
            .expect("vertical reference");

    let mut owner = vehicle_kernel
        .create_resident_chassis_brush_owner_with_vertical(
            &device,
            &queue,
            &spin_input,
            &suspension_input,
            brush_input,
            &planar_input,
            &vertical_input,
        )
        .expect("resident full chassis+brush owner");
    let resident_brush = vehicle_kernel
        .step_resident_chassis_brush_owner_from_resident_state(
            &brush_kernel,
            &device,
            &queue,
            &mut owner,
            planar_input.dt,
            None,
        )
        .expect("resident full chassis+brush step");
    assert_eq!(resident_brush.wheel_count, spin_input.wheel_state.len());
    let got_planar = vehicle_kernel
        .readback_planar_resident_state(&device, &queue, &owner.planar)
        .expect("resident planar readback");
    let got_vertical = vehicle_kernel
        .readback_vertical_resident_state(&device, &queue, &owner.vertical)
        .expect("resident vertical readback");
    assert!(got_planar.finite);
    assert!(got_vertical.finite);
    assert!(
        (got_planar.force_xz[0] - expected_planar.force_xz[0]).abs()
            < 2.5e-3_f32.max(expected_planar.force_xz[0].abs() * 2.0e-5),
        "Fx mismatch got={got_planar:?} expected={expected_planar:?}"
    );
    assert!(
        (got_planar.force_xz[1] - expected_planar.force_xz[1]).abs()
            < 2.5e-3_f32.max(expected_planar.force_xz[1].abs() * 2.0e-5),
        "Fz mismatch got={got_planar:?} expected={expected_planar:?}"
    );
    assert!(
        (got_planar.yaw_moment - expected_planar.yaw_moment).abs()
            < 2.5e-3_f32.max(expected_planar.yaw_moment.abs() * 2.0e-5),
        "Mz mismatch got={got_planar:?} expected={expected_planar:?}"
    );
    assert!((got_planar.velocity_xz[0] - expected_planar.velocity_xz[0]).abs() < 2.0e-5);
    assert!((got_planar.velocity_xz[1] - expected_planar.velocity_xz[1]).abs() < 2.0e-5);
    assert!((got_planar.position_xz[0] - expected_planar.position_xz[0]).abs() < 2.0e-5);
    assert!((got_planar.position_xz[1] - expected_planar.position_xz[1]).abs() < 2.0e-5);
    assert!((got_planar.yaw_rate - expected_planar.yaw_rate).abs() < 2.0e-5);
    assert!((got_planar.yaw - expected_planar.yaw).abs() < 2.0e-5);
    assert!((got_vertical.force_y - expected_vertical.force_y).abs() < 2.0e-3);
    assert!((got_vertical.vy - expected_vertical.vy).abs() < 2.0e-5);
    assert!((got_vertical.y - expected_vertical.y).abs() < 2.0e-5);
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_vehicle_resident_spin_state_feeds_kinematics_gpu_without_staging() {
    let engine = try_engine().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireVehicleYawGpuKernel::new(&device);
    let spin_input = sample_wheel_spin_input();
    let expected_spin = tire_vehicle_wheel_spin_state_reference(&spin_input)
        .expect("resident spin state reference")
        .rows;
    let kinematics_input = sample_wheel_kinematics_from_spin_input();
    let expected =
        tire_vehicle_wheel_kinematics_from_spin_reference(&kinematics_input, &expected_spin)
            .expect("reference kinematics from spin");
    let mut spin_batch = kernel
        .create_wheel_spin_resident_batch(&device, &queue, &spin_input)
        .expect("resident spin batch");
    kernel
        .step_wheel_spin_resident(&device, &queue, &mut spin_batch, spin_input.dt)
        .expect("resident spin step");
    let got = kernel
        .dispatch_wheel_kinematics_from_spin_resident_blocking(
            &device,
            &queue,
            &spin_batch,
            &kinematics_input,
        )
        .expect("resident spin-to-kinematics");
    assert!(got.finite);
    assert_eq!(got.rows.len(), expected.rows.len());
    for (a, b) in got.rows.iter().zip(expected.rows.iter()) {
        for k in 0..4 {
            assert!(
                (a[k] - b[k]).abs() < 2.0e-5,
                "resident spin-to-kinematics mismatch got={a:?} expected={b:?}"
            );
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_vehicle_wheel_kinematics_gpu_matches_reference() {
    let engine = try_engine().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireVehicleYawGpuKernel::new(&device);
    let input = sample_wheel_kinematics_input();
    let expected = tire_vehicle_wheel_kinematics_reference(&input).expect("reference");
    let got = kernel
        .dispatch_wheel_kinematics_blocking(&device, &queue, &input)
        .expect("GPU wheel kinematics");
    assert!(got.finite);
    assert_eq!(got.rows.len(), expected.rows.len());
    for (a, b) in got.rows.iter().zip(expected.rows.iter()) {
        for k in 0..4 {
            assert!(
                (a[k] - b[k]).abs() < 2.0e-5,
                "row mismatch gpu={a:?} expected={b:?}"
            );
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_vehicle_suspension_gpu_matches_reference() {
    let engine = try_engine().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireVehicleYawGpuKernel::new(&device);
    let input = sample_suspension_input();
    let expected = tire_vehicle_suspension_reference(&input).expect("reference");
    let got = kernel
        .dispatch_suspension_blocking(&device, &queue, &input)
        .expect("GPU suspension");
    assert!(got.finite);
    assert_eq!(got.rows.len(), expected.rows.len());
    for (a, b) in got.rows.iter().zip(expected.rows.iter()) {
        assert!((a[0] - b[0]).abs() < 2.0e-3);
        assert!((a[1] - b[1]).abs() < 2.0e-5);
        assert!((a[2] - b[2]).abs() < 2.0e-5);
        assert_eq!(a[3] > 0.5, b[3] > 0.5);
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_vehicle_vertical_resident_state_ping_pongs_from_suspension_rows() {
    let engine = try_engine().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireVehicleYawGpuKernel::new(&device);
    let suspension_input = sample_suspension_input();
    let vertical_input = sample_vertical_input();
    let expected_suspension =
        tire_vehicle_suspension_reference(&suspension_input).expect("suspension reference");
    let expected =
        tire_vehicle_vertical_from_suspension_reference(&vertical_input, &expected_suspension.rows)
            .expect("vertical reference");
    let suspension = kernel
        .create_suspension_resident_batch(&device, &queue, &suspension_input)
        .expect("resident suspension");
    let mut vertical = kernel
        .create_vertical_resident_state(
            &device,
            &queue,
            &vertical_input,
            suspension_input.suspension_geometry.len(),
        )
        .expect("resident vertical state");
    assert_eq!(
        vertical.wheel_count(),
        suspension_input.suspension_geometry.len()
    );
    kernel
        .step_suspension_resident_from_vertical_state(&device, &queue, &suspension, &vertical)
        .expect("resident suspension from vertical");
    let got_suspension = kernel
        .readback_suspension_resident(&device, &queue, &suspension)
        .expect("resident suspension readback");
    assert_eq!(got_suspension.rows.len(), expected_suspension.rows.len());
    for (a, b) in got_suspension
        .rows
        .iter()
        .zip(expected_suspension.rows.iter())
    {
        assert!((a[0] - b[0]).abs() < 2.0e-3);
        assert!((a[1] - b[1]).abs() < 2.0e-5);
        assert!((a[2] - b[2]).abs() < 2.0e-5);
    }
    kernel
        .step_vertical_resident_from_suspension(
            &device,
            &queue,
            &mut vertical,
            suspension.rows_buffer(),
            suspension.wheel_count(),
            vertical_input.dt,
        )
        .expect("resident vertical step");
    let got = kernel
        .readback_vertical_resident_state(&device, &queue, &vertical)
        .expect("resident vertical readback");
    assert!(got.finite);
    assert!((got.force_y - expected.force_y).abs() < 2.0e-3);
    assert!((got.vy - expected.vy).abs() < 2.0e-5);
    assert!((got.y - expected.y).abs() < 2.0e-5);
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_vehicle_brush_rows_gpu_matches_reference() {
    let engine = try_engine().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireVehicleYawGpuKernel::new(&device);
    let input = sample_brush_rows_input();
    let expected = tire_vehicle_brush_rows_reference(&input).expect("reference");
    let got = kernel
        .dispatch_brush_rows_blocking(&device, &queue, &input)
        .expect("GPU brush rows");
    assert!(got.finite);
    assert_eq!(got.wheels.len(), expected.wheels.len());
    for (a, b) in got.wheels.iter().zip(expected.wheels.iter()) {
        for (a_row, b_row) in [
            (&a.hub_pos, &b.hub_pos),
            (&a.hub_vel, &b.hub_vel),
            (&a.heading, &b.heading),
            (&a.spin, &b.spin),
            (&a.tire, &b.tire),
            (&a.material, &b.material),
            (&a.extra, &b.extra),
        ] {
            for k in 0..4 {
                assert!((a_row[k] - b_row[k]).abs() < 2.0e-5);
            }
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_vehicle_brush_rows_resident_feed_brush_gpu_matches_staged_rows() {
    let engine = try_engine().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let vehicle_kernel = TireVehicleYawGpuKernel::new(&device);
    let brush_kernel = TireBrushGpuKernel::new(&device);
    let input = sample_brush_rows_input();
    let expected_rows = tire_vehicle_brush_rows_reference(&input).expect("reference rows");
    let expected = brush_kernel
        .dispatch_blocking(&device, &queue, &expected_rows.wheels, None)
        .expect("staged brush dispatch");
    let resident = vehicle_kernel
        .dispatch_brush_rows_resident(&device, &queue, &input)
        .expect("resident brush rows");
    let got = brush_kernel
        .dispatch_input_buffer_blocking(
            &device,
            &queue,
            &resident.buffer,
            resident.wheel_count,
            None,
        )
        .expect("resident brush dispatch");
    assert_eq!(got.len(), expected.len());
    for (a, b) in got.iter().zip(expected.iter()) {
        for (a_row, b_row) in [
            (&a.forces, &b.forces),
            (&a.diag1, &b.diag1),
            (&a.diag2, &b.diag2),
            (&a.diag3, &b.diag3),
            (&a.diag4, &b.diag4),
            (&a.diag5, &b.diag5),
            (&a.world_force, &b.world_force),
        ] {
            for k in 0..4 {
                let tol = 2.5e-3_f32.max(b_row[k].abs() * 2.0e-5);
                assert!(
                    (a_row[k] - b_row[k]).abs() < tol,
                    "resident brush output mismatch got={a_row:?} expected={b_row:?} tol={tol}"
                );
            }
        }
    }
}
