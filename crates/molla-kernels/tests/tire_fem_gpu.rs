//! GPU smoke/parity test for the volumetric tire FEM node kernel.
//!
//! Ignored by default because it needs a real wgpu adapter:
//! `cargo test -p molla-kernels --features gpu --test tire_fem_gpu -- --ignored`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::{
    BrushWheelOutput, TIRE_FEM_ATLAS_SHADER, TireBrushGpuKernel, TireFemContactFeedbackGpuTerrain,
    TireFemContactFeedbackInput, TireFemContactFeedbackKernel, TireFemContactFeedbackOutput,
    TireFemContactFeedbackParams, TireFemContactReductionGpuTerrain, TireFemContactReductionInput,
    TireFemContactReductionKernel, TireFemContactReductionOutput, TireFemContactReductionParams,
    TireFemContactSdfPrimitive, TireFemContactTerrain, TireFemContactTriangle, TireFemGpuInput,
    TireFemGpuKernel, TireFemGpuParams, TireFemHubParams, TireTransientGpuOutput,
    TireVehicleBrushRowsFromSpinGpuInput, TireVehiclePlanarFromBrushGpuInput,
    TireVehicleSuspensionGpuInput, TireVehicleVerticalGpuInput, TireVehicleWheelSpinGpuInput,
    TireVehicleYawGpuKernel, tire_fem_contact_terrain_with_transient_surface_velocity,
    tire_fem_contact_triangle_bvh_rows, tire_fem_feedback_params_with_transient,
    tire_fem_gpu_batch_atlas_binding_plan, tire_fem_gpu_batch_atlas_buffers,
    tire_fem_gpu_batch_atlas_dispatch_contract, tire_fem_gpu_batch_atlas_layout,
    tire_fem_gpu_batch_atlas_offset_row, tire_fem_gpu_batch_atlas_param_row,
    tire_fem_gpu_batch_dispatch_plan, tire_fem_gpu_input_from_tetra_fem,
    tire_fem_hub_params_with_brush_outputs, tire_fem_hub_params_with_contact_reduction,
    tire_fem_hub_params_with_feedback_output, tire_fem_transient_feedback_command,
};

fn try_engine() -> Option<Engine> {
    Engine::new_gpu().ok()
}

fn large_triangle_contact_terrain() -> TireFemContactTerrain {
    let mut triangles = Vec::new();
    for ix in 0..24 {
        for iz in 0..18 {
            let x = ix as f32 * 0.42;
            let z = iz as f32 * 0.37;
            let friction = if (ix + iz) % 2 == 0 { 0.72 } else { 0.58 };
            let p00 = [x, 0.0, z];
            let p10 = [x + 0.38, 0.0, z + 0.01];
            let p01 = [x - 0.01, 0.0, z + 0.34];
            let p11 = [x + 0.39, 0.0, z + 0.35];
            // Winding is intentionally upward-facing so the hardware stress
            // checks the triangle BVH traversal without depending on flat
            // heightfield fallback contact.
            triangles.push(TireFemContactTriangle::new(p00, p11, p10, friction));
            triangles.push(TireFemContactTriangle::new(p00, p01, p11, friction));
        }
    }
    TireFemContactTerrain {
        origin: [0.0, 0.0],
        spacing: 1.0,
        nx: 1,
        nz: 1,
        heights: vec![-10.0],
        friction: 0.05,
        friction_map: None,
        water_film_map: None,
        sinkage_map: None,
        soil_rolling_resistance_scale: 1.0,
        surface_velocity: [0.0; 3],
        surface_velocity_map: None,
        sdf_primitives: Vec::new(),
        triangles,
    }
}

fn tet_volume(p: &[[f32; 4]], ids: [usize; 4]) -> f32 {
    let x0 = p[ids[0]];
    let x1 = p[ids[1]];
    let x2 = p[ids[2]];
    let x3 = p[ids[3]];
    let a = [x1[0] - x0[0], x1[1] - x0[1], x1[2] - x0[2]];
    let b = [x2[0] - x0[0], x2[1] - x0[1], x2[2] - x0[2]];
    let c = [x3[0] - x0[0], x3[1] - x0[1], x3[2] - x0[2]];
    let cross_bc = [
        b[1] * c[2] - b[2] * c[1],
        b[2] * c[0] - b[0] * c[2],
        b[0] * c[1] - b[1] * c[0],
    ];
    (a[0] * cross_bc[0] + a[1] * cross_bc[1] + a[2] * cross_bc[2]) / 6.0
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[test]
fn tire_fem_contact_feedback_accepts_transient_relaxed_kinematics() {
    let transient = TireTransientGpuOutput {
        effective: [0.25, 12.0_f32.to_radians(), 0.02, 1.0],
        deflection: [0.04, 0.015, 0.0, 0.0],
    };
    let params = TireFemContactFeedbackParams {
        rolling_radius: 0.32,
        omega: 0.0,
        ..Default::default()
    };
    let updated = tire_fem_feedback_params_with_transient(params, &transient, 8.0);
    let command = tire_fem_transient_feedback_command(&transient, 8.0, 0.32);
    assert!(
        (updated.omega - command.wheel_omega_rad_per_s).abs() < 1.0e-5,
        "FEM feedback omega should be sourced from transient effective slip"
    );
    assert!(
        updated.omega > 8.0 / 0.32,
        "positive relaxed kappa should increase wheel angular speed"
    );

    let terrain = TireFemContactTerrain {
        origin: [-1.0, -1.0],
        spacing: 0.1,
        nx: 2,
        nz: 2,
        heights: vec![0.0; 4],
        friction: 1.0,
        friction_map: None,
        water_film_map: None,
        sinkage_map: None,
        soil_rolling_resistance_scale: 0.0,
        surface_velocity: [0.0; 3],
        surface_velocity_map: None,
        sdf_primitives: Vec::new(),
        triangles: Vec::new(),
    };
    let terrain =
        tire_fem_contact_terrain_with_transient_surface_velocity(terrain, &transient, 8.0, 0.32);
    assert!(
        terrain.surface_velocity[0] < -7.0 && terrain.surface_velocity[2] < -1.0,
        "effective alpha should become moving-ground lateral velocity: {:?}",
        terrain.surface_velocity
    );
}

#[test]
fn tire_fem_contact_feedback_exposes_gpu_resident_no_readback_path() {
    let source = include_str!("../src/tire_fem.rs");
    let shader = include_str!("../shaders/tire_fem_contact_feedback.wgsl");
    let resident = source
        .split("pub fn dispatch_resident_with_gpu_terrain")
        .nth(1)
        .expect("resident GPU-terrain feedback dispatch should exist");
    let resident = resident
        .split("fn validate_input")
        .next()
        .expect("resident dispatch body should be before validation helpers");
    assert!(resident.contains("TireFemContactFeedbackResidentOutput"));
    assert!(resident.contains("velocities: vel_out_buf"));
    assert!(resident.contains("bristles: bristle_out_buf"));
    assert!(resident.contains("forces: force_out_buf"));
    assert!(
        !resident.contains("stage_vec4") && !resident.contains("read_vec4"),
        "resident FEM feedback path must not stage/read back GPU buffers"
    );
    assert!(
        source.contains("surface_velocity_map: Option"),
        "FEM feedback terrain should expose optional spatial surface velocity maps"
    );
    assert!(
        source.contains("readback_resident_feedback_output")
            && source.contains("TireFemContactFeedbackResidentOutput")
            && source.contains("resident readback vel"),
        "resident FEM feedback should centralize the optional viewer/telemetry readback seam"
    );
    assert!(
        source.contains("storage_entry(13, true)") && shader.contains("@binding(13)"),
        "FEM feedback GPU binding contract should include terrain surface velocity maps"
    );
    assert!(
        shader.contains("fn sample_terrain_surface_velocity")
            && shader.contains("u.terrain_dims.w & 8u")
            && shader.contains("let terrain_velocity = sample_terrain_surface_velocity(p.x, p.z);"),
        "FEM feedback shader should sample per-node spatial surface velocity before computing relative slip"
    );
}

/// PLAN_REALTIRE Stream C G4 prep — `tire_fem_gpu_input_from_tetra_fem`
/// builds a complete GPU node-FEM input payload from a live `TetraTireFem`.
/// This test runs on CPU (no wgpu adapter) and just verifies the conversion
/// preserves the mesh topology + hub state. It is the foundation for the
/// future CPU-vs-GPU hub-trajectory parity hardware lane: build both sides
/// from the same `TetraTireFem`, dispatch in parallel, compare hub
/// trajectories within tolerance.
#[test]
fn tire_fem_gpu_input_from_tetra_fem_preserves_snapshot_and_hub() {
    use molla_vehicle::{TetraTireFem, TetraTireFemParams};
    let params = TetraTireFemParams::realtime_viewer();
    let mut tire = TetraTireFem::new(0.0, 180_000.0, params);
    tire.set_drive_torque_n_m(12.5);
    tire.set_brake_torque_n_m(3.25);
    let snapshot = tire.gpu_snapshot();
    let base_params = TireFemGpuParams {
        dt: 1.0 / 8000.0,
        node_damping: 22.0,
        ..Default::default()
    };
    let input = tire_fem_gpu_input_from_tetra_fem(&tire, base_params);

    // Mesh topology should match the snapshot exactly.
    assert_eq!(input.rest_positions.len(), snapshot.rest_positions.len());
    assert_eq!(input.positions.len(), snapshot.positions.len());
    assert_eq!(input.velocities.len(), snapshot.velocities.len());
    assert_eq!(input.tet_indices.len(), snapshot.tet_indices.len());
    assert_eq!(input.tet_grads.len(), snapshot.tet_grads.len());
    assert_eq!(input.tet_volumes.len(), snapshot.tet_volumes.len());
    assert_eq!(input.bead_ids.len(), snapshot.bead_ids.len());
    assert_eq!(input.tread_ids.len(), snapshot.tread_ids.len());
    assert_eq!(
        input.xpbd_edge_indices.len(),
        snapshot.xpbd_edge_indices.len()
    );
    assert_eq!(
        input.xpbd_dihedral_indices.len(),
        snapshot.xpbd_dihedral_indices.len()
    );
    assert_eq!(
        input.node_tet_offsets.len(),
        snapshot.node_tet_offsets.len()
    );

    // Hub state should propagate from the live tire (NOT from base_params).
    let hub_cpu = tire.hub();
    assert!(
        (input.params.hub[0] as f64 - hub_cpu[0]).abs() < 1.0e-5
            && (input.params.hub[1] as f64 - hub_cpu[1]).abs() < 1.0e-5
            && (input.params.hub[2] as f64 - hub_cpu[2]).abs() < 1.0e-5,
        "hub position must come from tire.hub(), got {:?} vs {:?}",
        input.params.hub,
        hub_cpu
    );
    assert!((input.params.drive_torque_n_m - 12.5).abs() < 1.0e-4);
    assert!((input.params.brake_torque_n_m - 3.25).abs() < 1.0e-4);
    assert!((input.params.hub_omega - tire.hub_omega() as f32).abs() < 1.0e-6);
    assert!((input.params.pressure_pa - tire.pressure() as f32).abs() < 1.0e-2);

    // base_params should pass through for fields the conversion does not touch.
    assert!((input.params.dt - 1.0 / 8000.0).abs() < 1.0e-9);
    assert!((input.params.node_damping - 22.0).abs() < 1.0e-6);
}

/// PLAN_REALTIRE Stream C G4 — CPU-vs-GPU FEM node parity hardware lane.
///
/// Build a single tire on both CPU and GPU from the same `TetraTireFem`
/// state. Pin the hub kinematic well above the floor (no contact this
/// step), advance the CPU via `step_on_terrain` and the GPU via
/// `step_persistent` for N substeps, then compare per-node positions and
/// velocities. The two paths share constitutive law (Neo-Hookean +
/// Holzapfel) and integration shape (semi-implicit Euler with damping),
/// so they should track within FP tolerance over moderate horizons.
///
/// This is the first leg of the eventual full parity lane (which will
/// also exercise the dynamic hub integrator and contact branch).
/// `tire_fem_gpu_input_from_tetra_fem` is the bridge it depends on.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_cpu_vs_gpu_node_parity_no_contact() {
    use molla_vehicle::{TetraTireFem, TetraTireFemParams};
    let Some(engine) = try_engine() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemGpuKernel::new(&device);

    // Realtime-viewer preset is the lowest-resolution mesh that still has all
    // ply families, bead, and cavity wired up — keeps the test fast while
    // exercising every code path in `step_persistent`. Hub is pinned high so
    // contact does not trigger and we can isolate the FEM kernel.
    // Zero the bead damping to match the GPU node kernel — `step_persistent`
    // applies bead_stiffness · (target − pos) but does not have a bead-damping
    // term (that lives in the separate hub kernel). For a node-only parity
    // test the two sides have to use the same bead force law.
    let params = TetraTireFemParams {
        bead_damping: 0.0,
        ..TetraTireFemParams::realtime_viewer()
    };
    let mut cpu_tire = TetraTireFem::new(0.0, 180_000.0, params);
    cpu_tire.set_hub_kinematic([0.0, 1.5, 0.0], [0.0; 3]);
    let dt = 1.0 / 8_000.0;

    // GPU side mirrors the same tire. The bridge supplies hub + cavity
    // pressure + torque from the CPU tire; the base params carry dt and
    // damping that the snapshot does not see.
    let base_gpu_params = TireFemGpuParams {
        dt: dt as f32,
        node_damping: 18.0,
        bead_stiffness: 5.5e4,
        xpbd_volume_compliance: 5.0e-10,
        xpbd_bead_compliance: 5.0e-5,
        semi_implicit_stiffness_scale: 0.0,
        ..Default::default()
    };
    let gpu_input = tire_fem_gpu_input_from_tetra_fem(&cpu_tire, base_gpu_params);
    let mut gpu_mesh = kernel
        .create_persistent(&device, &queue, &gpu_input)
        .expect("create persistent GPU mesh");

    // Tolerance shape: this is a smoke / parity test, not a bit-exact
    // comparison. CPU is f64, GPU is f32, and the XPBD passes can iterate
    // in different orders between the two. Empirically the no-contact path
    // drifts ~1 mm per substep at this resolution, dominated by cavity-
    // pressure + XPBD iteration order rather than FP noise. The test asserts
    // the drift does not blow up — i.e. that CPU and GPU stay in the same
    // ballpark over a short horizon. Tighter parity requires either
    // matching the XPBD iteration loop exactly between the two backends or
    // wiring the GPU through the dynamic-hub kernel as well, both of which
    // are follow-up work tracked in PROGRESS.md.
    // Empirical: drift starts at ~6 mm by step 8 and grows to ~26 mm by
    // step 40. The dominant source is f32-vs-f64 + XPBD pass ordering;
    // fixing it requires either splitting the GPU XPBD into separate
    // dispatches or moving the CPU into a single combined pass.
    let pos_tol = 0.06_f32;
    let vel_tol = 0.05_f32;

    // Use a flat terrain at Y = -100 m so CPU's contact branch never triggers;
    // gravity + bead + cavity are the only active forces, matching GPU
    // `step_persistent` (which has no contact branch in this kernel).
    let terrain = molla_vehicle::terrain::FlatTerrain::new(-100.0, 1.0);

    for step in 0..40 {
        // Extract GPU params from the CPU state BEFORE advancing the CPU
        // step, so both sides apply the same hub / cavity / torque inputs
        // to *their own* current mesh state — not state from one substep
        // ahead. This eliminates one source of linear drift identified
        // during the first-leg parity bring-up.
        let params = tire_fem_gpu_input_from_tetra_fem(&cpu_tire, base_gpu_params).params;
        cpu_tire.step_on_terrain(&terrain, dt);
        kernel.step_persistent(&device, &queue, &mut gpu_mesh, params);

        // Sample every 8 substeps to keep readback cost reasonable.
        if step % 8 == 7 {
            let gpu_state = kernel
                .readback_state(&device, &queue, &gpu_mesh)
                .expect("readback");
            let cpu_pos = cpu_tire.nodes();
            assert_eq!(
                gpu_state.positions.len(),
                cpu_pos.len(),
                "node count mismatch at step {step}"
            );
            let mut max_pos_diff = 0.0_f32;
            for (gpu_p, cpu_p) in gpu_state.positions.iter().zip(cpu_pos.iter()) {
                for axis in 0..3 {
                    let dp = (gpu_p[axis] - cpu_p[axis] as f32).abs();
                    max_pos_diff = max_pos_diff.max(dp);
                }
            }
            assert!(
                max_pos_diff.is_finite(),
                "step {step}: non-finite position drift"
            );
            assert!(
                max_pos_diff < pos_tol,
                "step {step}: per-node position drift {max_pos_diff} exceeds tol {pos_tol}"
            );
            let _ = vel_tol; // velocity comparison deferred until a CPU
            // `velocities()` accessor lands.
        }
    }
}

/// PLAN_REALTIRE Stream C G4 — second leg: hub-trajectory parity.
///
/// Builds on the first-leg node-parity test by also dispatching the GPU
/// hub kernel (`tire_fem_hub.wgsl`) each substep. CPU lets the dynamic
/// hub integrator advance under gravity + downforce + bead reactions;
/// GPU mirrors the same. Compares the hub-Y trajectory (the only
/// vertical DoF that moves in this fixture) over a short horizon.
///
/// At hub kinematic this just verifies hub-Y matches; at hub dynamic
/// (`set_hub_dynamic`) it exercises the full coupled integrator on
/// both sides. Combined with the first-leg, this proves the GPU stack
/// matches CPU through the FEM step *and* the hub equation, which is
/// what's needed to flip `TireFemTorqueGpuStatus` toward `LiveValidated`.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_cpu_vs_gpu_hub_y_parity_dynamic_hub() {
    use molla_vehicle::{TetraTireFem, TetraTireFemParams};
    let Some(engine) = try_engine() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemGpuKernel::new(&device);

    let params = TetraTireFemParams::realtime_viewer();
    let mut cpu_tire = TetraTireFem::new(0.0, 180_000.0, params);
    cpu_tire.set_hub_dynamic();
    let dt = 1.0 / 8_000.0;
    let downforce = params.downforce as f32;
    let hub_mass = params.hub_mass as f32;
    let hub_floor_y = (params.inner_radius * 0.75) as f32;

    let base_gpu_params = TireFemGpuParams {
        dt: dt as f32,
        node_damping: params.node_damping as f32,
        bead_stiffness: params.bead_stiffness as f32,
        xpbd_volume_compliance: params.xpbd_volume_compliance as f32,
        xpbd_bead_compliance: params.xpbd_bead_compliance as f32,
        semi_implicit_stiffness_scale: params.semi_implicit_stiffness_scale as f32,
        ..Default::default()
    };
    let gpu_input = tire_fem_gpu_input_from_tetra_fem(&cpu_tire, base_gpu_params);
    let mut gpu_mesh = kernel
        .create_persistent(&device, &queue, &gpu_input)
        .expect("create persistent GPU mesh");

    let hub_params = TireFemHubParams {
        dt: dt as f32,
        bead_stiffness: params.bead_stiffness as f32,
        bead_damping: params.bead_damping as f32,
        hub_mass,
        hub_damping: 260.0,
        downforce,
        gravity_y: -9.81,
        hub_floor_active: 1.0,
        hub_floor_y,
        external_force_xyz: [0.0; 3],
        lateral_damping: 80.0,
    };

    // Terrain at Y=0 so hub_floor catches things. Hub will sag under
    // downforce until floor stops it; CPU and GPU should track.
    let terrain = molla_vehicle::terrain::FlatTerrain::new(0.0, 1.0);

    // Run 24 substeps (3 ms simulated). Long enough for the hub to
    // visibly sag toward the floor; short enough to keep node drift
    // bounded.
    for step in 0..24 {
        let node_params = tire_fem_gpu_input_from_tetra_fem(&cpu_tire, base_gpu_params).params;
        cpu_tire.step_on_terrain(&terrain, dt);
        kernel.step_persistent(&device, &queue, &mut gpu_mesh, node_params);
        kernel.step_hub_persistent(&device, &queue, &gpu_mesh, hub_params);

        if step % 6 == 5 {
            let gpu_hub = kernel
                .readback_hub(&device, &queue, &gpu_mesh)
                .expect("readback_hub");
            // hub_state[0] = (hub_x, hub_y, hub_z, hub_v_y).
            let cpu_hub = cpu_tire.hub();
            let dy = (gpu_hub[1] - cpu_hub[1] as f32).abs();
            // 1 cm absolute tolerance on hub-Y over 3 ms simulated. The
            // hub coupling is downforce + gravity + Σ bead reactions;
            // bead reactions depend on node positions which themselves
            // drift ~1 mm/substep. So a sub-cm hub agreement is the
            // expected regime here.
            assert!(
                dy.is_finite() && dy < 0.01,
                "step {step}: GPU/CPU hub_y drift {dy} exceeds 1 cm tol \
                 (gpu={}, cpu={})",
                gpu_hub[1],
                cpu_hub[1]
            );
        }
    }
}

/// PLAN_REALTIRE Stream C G4 — third leg: contact-driven rolling parity.
///
/// Completes the G4 lane: leg 1 proved node-FEM parity without contact,
/// leg 2 proved hub-Y parity with the dynamic hub, this leg rolls a
/// spinning tire on flat ground on both backends and compares the
/// longitudinal hub trajectory the contact patch produces.
///
/// CPU chain: `step_on_terrain` runs the spin-coupled XPBD contact +
/// bristle brush in `step_inner`, accumulates `net_contact_fx`, and
/// integrates it into `hub_v_xz` — that's rolling propulsion.
///
/// GPU chain (the wiring this leg adds): `step_persistent` (node FEM +
/// spin-coupled Coulomb contact) → `TireFemContactFeedbackKernel`
/// (persistent GPU bristles + brush force per tread node, the GPU
/// counterpart of the CPU bristle term) → the summed tread wrench feeds
/// `TireFemHubParams::external_force_xyz` → `step_hub_persistent`
/// integrates the hub. Bristle state threads through the blocking
/// dispatch from substep to substep; feedback node velocities are
/// uploaded back into the persistent mesh so GPU nodes feel the brush
/// like CPU nodes do.
///
/// Known structural differences absorbed by the tolerance: the CPU clips
/// brush + Coulomb in a single friction cone while the GPU applies them
/// in two stages; CPU caps bristles by penetration, the feedback kernel
/// by `normal_load_per_tread`; f32 vs f64. The per-substep param
/// re-bridge keeps cavity pressure, hub pose, and `hub_omega` in sync
/// (the cavity refresh is the drift fix called out in PROGRESS.md).
#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_cpu_vs_gpu_rolling_contact_hub_x_parity() {
    use molla_vehicle::{SurfaceMaterial, TetraTireFem, TetraTireFemParams};
    let Some(engine) = try_engine() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemGpuKernel::new(&device);
    let feedback_kernel = TireFemContactFeedbackKernel::new(&device);

    // Match the CPU friction cone to the GPU node kernel's hard-coded
    // mu values so the Coulomb stage agrees between backends.
    let params = TetraTireFemParams {
        surface_material: SurfaceMaterial {
            mu_static: 1.0,
            mu_dynamic: 0.85,
            bristle_damping: 5.0e4,
            rolling_resistance: 0.0,
            cohesion: 0.0,
        },
        ..TetraTireFemParams::realtime_viewer()
    };
    let mut cpu_tire = TetraTireFem::new(0.0, 180_000.0, params);
    let dt = 1.0 / 8_000.0;
    let terrain = molla_vehicle::terrain::FlatTerrain::new(0.0, 1.0);

    // Settle into stationary contact on the CPU first so both backends
    // start from the same loaded, equilibrated state. The mesh sinks
    // only at its node-damping terminal velocity (~cm/s), so instead of
    // simulating the slow free descent, pin the hub at a loaded height
    // (a few mm of patch interference), let the contact patch form, then
    // release the hub to dynamic and let it find equilibrium.
    cpu_tire.set_hub_kinematic([0.0, params.outer_radius * 0.982, 0.0], [0.0; 3]);
    for _ in 0..1_600 {
        cpu_tire.step_on_terrain(&terrain, dt);
    }
    cpu_tire.set_hub_dynamic();
    // The mesh descends at its node-damping terminal velocity (~cm/s),
    // so keep settling until the patch actually touches the ground —
    // rolling parity on a hovering tire would compare zero against zero.
    let mut settled_in_contact = false;
    for chunk in 0..20 {
        for _ in 0..1_600 {
            cpu_tire.step_on_terrain(&terrain, dt);
        }
        let nodes = cpu_tire.nodes();
        let touching = cpu_tire
            .tread_nodes()
            .iter()
            .filter(|&&id| nodes[id][1] < 1.0e-3)
            .count();
        if touching >= 2 {
            settled_in_contact = true;
            eprintln!("settled into contact after {} chunks", chunk + 1);
            break;
        }
    }
    assert!(
        settled_in_contact,
        "tire must reach ground contact during settle"
    );

    // Start the wheel already rolling with sustained drive torque so the
    // contact patch produces meaningful longitudinal force from the
    // first shared substep.
    cpu_tire.set_hub_spin(6.0);
    cpu_tire.set_drive_torque_n_m(120.0);

    let base_gpu_params = TireFemGpuParams {
        dt: dt as f32,
        node_damping: params.node_damping as f32,
        bead_stiffness: params.bead_stiffness as f32,
        xpbd_volume_compliance: params.xpbd_volume_compliance as f32,
        xpbd_bead_compliance: params.xpbd_bead_compliance as f32,
        semi_implicit_stiffness_scale: params.semi_implicit_stiffness_scale as f32,
        ..Default::default()
    };
    let gpu_input = tire_fem_gpu_input_from_tetra_fem(&cpu_tire, base_gpu_params);
    let masses: Vec<f32> = gpu_input.masses.clone();
    let tread_ids: Vec<u32> = gpu_input.tread_ids.clone();
    let mut gpu_mesh = kernel
        .create_persistent(&device, &queue, &gpu_input)
        .expect("create persistent GPU mesh");

    let hub_mass = params.hub_mass as f32;
    let hub_floor_y = (params.inner_radius * 0.75) as f32;
    let base_hub_params = TireFemHubParams {
        dt: dt as f32,
        bead_stiffness: params.bead_stiffness as f32,
        bead_damping: params.bead_damping as f32,
        hub_mass,
        hub_damping: 260.0,
        downforce: params.downforce as f32,
        gravity_y: -9.81,
        hub_floor_active: 1.0,
        hub_floor_y,
        external_force_xyz: [0.0; 3],
        lateral_damping: 80.0,
    };

    // Per-tread normal load for the feedback cone: the same
    // load-normalization `evaluate_tread_contact` uses — total downforce
    // spread over the tread nodes currently inside the contact band.
    let contact_band = 0.012_f32;
    let in_band = |tire: &TetraTireFem| -> usize {
        let nodes = tire.nodes();
        tire.tread_nodes()
            .iter()
            .filter(|&&id| (nodes[id][1] as f32) < contact_band)
            .count()
            .max(1)
    };
    let initial_contact_count = in_band(&cpu_tire);
    {
        let nodes = cpu_tire.nodes();
        let mut heights: Vec<f64> = cpu_tire
            .tread_nodes()
            .iter()
            .map(|&id| nodes[id][1])
            .collect();
        heights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "settle probe: hub={:?} lowest tread heights={:?}",
            cpu_tire.hub(),
            &heights[..heights.len().min(12)]
        );
    }
    assert!(
        initial_contact_count >= 3,
        "fixture must settle into real contact before rolling: {initial_contact_count} nodes"
    );

    // Persistent GPU bristle state threads through the blocking feedback
    // dispatch. CPU bristles are ≈0 after the stationary settle, so both
    // backends start with relaxed bristles.
    let mut gpu_bristles: Vec<[f32; 4]> = vec![[0.0; 4]; tread_ids.len()];

    let cpu_hub_start = cpu_tire.hub();
    let mut gpu_hub_x_prev = cpu_hub_start[0] as f32;
    let omega_start = cpu_tire.hub_omega();

    for step in 0..96 {
        // Bridge CPU state BEFORE stepping either side: hub pose, ω,
        // torque, and the live cavity pressure all enter both backends
        // for the same substep (PROGRESS.md drift fix).
        let bridged = tire_fem_gpu_input_from_tetra_fem(&cpu_tire, base_gpu_params);
        let node_params = bridged.params;
        let hub_pre = cpu_tire.hub();
        let hub_vel_pre = cpu_tire.hub_velocity();
        let omega_pre = cpu_tire.hub_omega();
        let load_per_tread = (params.downforce / in_band(&cpu_tire) as f64) as f32;

        cpu_tire.step_on_terrain(&terrain, dt);
        kernel.step_persistent(&device, &queue, &mut gpu_mesh, node_params);

        // GPU brush stage: persistent-bristle contact feedback on the
        // post-step node state, mirroring the CPU bristle integration
        // that ran inside `step_inner`.
        let gpu_state = kernel
            .readback_state(&device, &queue, &gpu_mesh)
            .expect("readback state");
        let feedback = feedback_kernel
            .dispatch_blocking(
                &device,
                &queue,
                &TireFemContactFeedbackInput {
                    positions: gpu_state.positions.clone(),
                    velocities: gpu_state.velocities.clone(),
                    masses: masses.clone(),
                    tread_ids: tread_ids.clone(),
                    bristles: gpu_bristles.clone(),
                    terrain: None,
                    params: TireFemContactFeedbackParams {
                        dt: dt as f32,
                        contact_band,
                        ground_y: 0.0,
                        omega: omega_pre as f32,
                        hub: [hub_pre[0] as f32, hub_pre[1] as f32, hub_pre[2] as f32],
                        hub_vel: [
                            hub_vel_pre[0] as f32,
                            hub_vel_pre[1] as f32,
                            hub_vel_pre[2] as f32,
                        ],
                        heading: [1.0, 0.0, 0.0],
                        rolling_radius: params.outer_radius as f32,
                        normal_load_per_tread: load_per_tread,
                        area_per_tread: node_params.pressure_area_per_tread_node,
                        tread_shear_stiffness: params.tread_shear_stiffness as f32,
                        mu_static: 1.0,
                        mu_dynamic: 0.85,
                        bristle_damping: 5.0e4,
                        cohesion: 0.0,
                        bristle_relax_rate: params.bristle_relax_rate as f32,
                    },
                },
            )
            .expect("contact feedback dispatch");
        let wrench = feedback.patch_accounting().total_force;
        gpu_bristles = feedback.bristles;

        // Brush impulses must reach the GPU nodes like the CPU bristle
        // term reaches `self.vel` inside the contact projection.
        kernel.upload_state(
            &queue,
            &gpu_mesh,
            &gpu_state.positions,
            &feedback.velocities,
        );

        // The patch wrench rolls the hub. X/Z only — the normal
        // component reaches the hub through the mesh + bead reduction,
        // same as on CPU.
        let hub_params = TireFemHubParams {
            external_force_xyz: [wrench[0], 0.0, wrench[2]],
            ..base_hub_params
        };
        kernel.step_hub_persistent(&device, &queue, &gpu_mesh, hub_params);

        if step % 8 == 7 {
            let gpu_hub = kernel
                .readback_hub(&device, &queue, &gpu_mesh)
                .expect("readback_hub");
            let cpu_hub = cpu_tire.hub();
            let dx = (gpu_hub[0] - cpu_hub[0] as f32).abs();
            let dy = (gpu_hub[1] - cpu_hub[1] as f32).abs();
            assert!(
                dx.is_finite() && dy.is_finite(),
                "step {step}: non-finite hub drift"
            );
            // Sub-cm agreement on both hub axes under rolling — the G4
            // closure gate from PROGRESS.md.
            assert!(
                dx < 0.01,
                "step {step}: rolling hub-X drift {dx} exceeds 1 cm \
                 (gpu={}, cpu={})",
                gpu_hub[0],
                cpu_hub[0]
            );
            assert!(
                dy < 0.01,
                "step {step}: hub-Y drift {dy} exceeds 1 cm under rolling \
                 (gpu={}, cpu={})",
                gpu_hub[1],
                cpu_hub[1]
            );
            gpu_hub_x_prev = gpu_hub[0];
        }
    }

    // The parity above must not pass vacuously: the tire has to have
    // actually rolled forward on both backends, and the spin DOF must
    // have accelerated under sustained drive torque.
    let cpu_dx = cpu_tire.hub()[0] - cpu_hub_start[0];
    let gpu_dx = gpu_hub_x_prev as f64 - cpu_hub_start[0];
    assert!(
        cpu_dx > 2.0e-4,
        "CPU tire should roll forward under spin + drive torque: dx={cpu_dx}"
    );
    assert!(
        gpu_dx > 0.0,
        "GPU tire should roll forward under spin + drive torque: dx={gpu_dx}"
    );
    assert!(
        cpu_tire.hub_omega() > omega_start,
        "drive torque should keep accelerating omega: start={omega_start} end={}",
        cpu_tire.hub_omega()
    );
}

/// G4 follow-up — on-device contact-wrench reduction.
///
/// `hub_reduce_with_tread_wrench` must integrate the hub identically to
/// the host path that reads back the contact-feedback force rows, sums
/// them on the CPU, and passes the X/Z total through
/// `external_force_xyz`. Two identical persistent meshes get the same
/// synthetic per-node force rows; one steps through
/// `step_hub_persistent` with the host-summed wrench, the other through
/// `step_hub_persistent_with_tread_forces` with the raw rows in a GPU
/// buffer. The hub states must agree to f32 reduction-order noise.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_hub_tread_wrench_reduction_matches_host_sum() {
    use molla_vehicle::{TetraTireFem, TetraTireFemParams};
    use wgpu::util::DeviceExt;
    let Some(engine) = try_engine() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemGpuKernel::new(&device);

    let params = TetraTireFemParams::realtime_viewer();
    let cpu_tire = TetraTireFem::new(0.0, 180_000.0, params);
    let dt = 1.0 / 8_000.0;
    let base_gpu_params = TireFemGpuParams {
        dt: dt as f32,
        ..Default::default()
    };
    let gpu_input = tire_fem_gpu_input_from_tetra_fem(&cpu_tire, base_gpu_params);
    let node_count = gpu_input.positions.len();

    let mesh_host = kernel
        .create_persistent(&device, &queue, &gpu_input)
        .expect("create host-path mesh");
    let mesh_device = kernel
        .create_persistent(&device, &queue, &gpu_input)
        .expect("create device-path mesh");

    // Synthetic per-node force rows with a deterministic non-uniform
    // pattern across X/Y/Z. The Y entries must be ignored by the
    // on-device reduction, so give them large values to catch a leak.
    let mut force_rows: Vec<[f32; 4]> = vec![[0.0; 4]; node_count];
    let mut host_sum_x = 0.0_f64;
    let mut host_sum_z = 0.0_f64;
    for (i, row) in force_rows.iter_mut().enumerate() {
        let s = (i as f32 * 0.37).sin();
        row[0] = 3.5 * s;
        row[1] = 500.0 + i as f32;
        row[2] = -2.0 * s * s;
        host_sum_x += row[0] as f64;
        host_sum_z += row[2] as f64;
    }
    let force_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test tread force rows"),
        contents: bytemuck::cast_slice(&force_rows),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let hub_params = TireFemHubParams {
        dt: dt as f32,
        bead_stiffness: params.bead_stiffness as f32,
        bead_damping: params.bead_damping as f32,
        hub_mass: params.hub_mass as f32,
        hub_damping: 260.0,
        downforce: params.downforce as f32,
        gravity_y: -9.81,
        hub_floor_active: 1.0,
        hub_floor_y: (params.inner_radius * 0.75) as f32,
        external_force_xyz: [0.0; 3],
        lateral_damping: 80.0,
    };

    for step in 0..4 {
        let host_params = TireFemHubParams {
            external_force_xyz: [host_sum_x as f32, 0.0, host_sum_z as f32],
            ..hub_params
        };
        kernel.step_hub_persistent(&device, &queue, &mesh_host, host_params);
        kernel.step_hub_persistent_with_tread_forces(
            &device,
            &queue,
            &mesh_device,
            hub_params,
            &force_buf,
            node_count as u32,
        );

        let hub_host = kernel
            .readback_hub(&device, &queue, &mesh_host)
            .expect("readback host hub");
        let hub_device = kernel
            .readback_hub(&device, &queue, &mesh_device)
            .expect("readback device hub");
        for slot in 0..6 {
            let diff = (hub_host[slot] - hub_device[slot]).abs();
            assert!(
                diff < 1.0e-5,
                "step {step} slot {slot}: on-device wrench reduction diverged from \
                 host sum: host={:?} device={:?}",
                hub_host,
                hub_device
            );
        }
        // The wrench must actually move the hub laterally, or this test
        // is comparing two no-ops.
        if step == 3 {
            assert!(
                hub_device[4].abs() > 1.0e-7,
                "synthetic wrench should produce hub X velocity: {:?}",
                hub_device
            );
        }
    }
}

/// PLAN_REALTIRE Stream C — 40-wheel scaling smoke.
///
/// Build 40 independent `TetraTireFem` tires (10 machines × 4 wheels),
/// hand them all to the resident atlas batch path, and step them through
/// 24 substeps. Asserts every mesh stays finite and on/above the floor.
/// Not a parity test — that lives in the two `cpu_vs_gpu_*` tests above.
/// This is the "does the fused batch dispatch survive 40 tires"
/// hardware smoke that the PLAN_REALTIRE Stream C goal calls for.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_resident_atlas_batch_steps_forty_wheels() {
    use molla_vehicle::{TetraTireFem, TetraTireFemParams};
    let Some(engine) = try_engine() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemGpuKernel::new(&device);

    let params = TetraTireFemParams::realtime_viewer();
    let dt = 1.0 / 8_000.0_f64;
    let base_gpu_params = TireFemGpuParams {
        dt: dt as f32,
        node_damping: params.node_damping as f32,
        bead_stiffness: params.bead_stiffness as f32,
        xpbd_volume_compliance: params.xpbd_volume_compliance as f32,
        xpbd_bead_compliance: params.xpbd_bead_compliance as f32,
        ..Default::default()
    };

    // 40 wheels arranged 10 × 4. Stagger the hubs so each tire occupies
    // its own slice of world XZ; the FEM atlas doesn't care about
    // overlap (terrain is queried independently per tire), but the
    // stagger keeps the test thinking about "10 vehicles" rather than
    // "40 tires in one giant pile".
    let mut inputs: Vec<TireFemGpuInput> = Vec::with_capacity(40);
    for veh in 0..10 {
        for wheel in 0..4 {
            let hub_x_base = veh as f64 * 4.0;
            let along_axle = match wheel {
                0 | 2 => -1.4,
                _ => 1.4,
            };
            let hub_x = hub_x_base + along_axle;
            let hub_z = match wheel {
                0 | 1 => -0.7,
                _ => 0.7,
            };
            // `TetraTireFem::new(hub_x, ...)` builds the rest configuration
            // around that hub X; passing 0 and then `set_hub_kinematic`
            // moves the hub WITHOUT updating the mesh, which leaves the
            // FEM rest mesh thousands of times deformed relative to the
            // current positions. Build the tire at its final hub X to
            // keep rest aligned with positions.
            let mut tire = TetraTireFem::new(hub_x, 180_000.0, params);
            tire.set_hub_kinematic([hub_x, 1.0, hub_z], [0.0; 3]);
            inputs.push(tire_fem_gpu_input_from_tetra_fem(&tire, base_gpu_params));
        }
    }
    assert_eq!(inputs.len(), 40);

    let mut batch = kernel
        .create_resident_atlas_batch(&device, &queue, &inputs)
        .expect("create resident atlas batch");
    let step_params: Vec<TireFemGpuParams> = inputs.iter().map(|i| i.params).collect();

    // Run 24 substeps (3 ms simulated). At 40 wheels × ~150 nodes/tire =
    // 6 000 nodes per dispatch, the fused atlas path covers in one
    // workgroup-grid what the per-tire path would need 40 separate
    // dispatches for. Validates the production scaling-path is live.
    for _ in 0..24 {
        kernel
            .step_resident_atlas_batch(&device, &queue, &mut batch, &step_params)
            .expect("step_resident_atlas_batch");
    }

    let outputs = kernel
        .readback_resident_atlas_batch(&device, &queue, &batch)
        .expect("readback");
    assert_eq!(outputs.len(), 40);
    for (wheel_idx, out) in outputs.iter().enumerate() {
        assert_eq!(out.positions.len(), inputs[wheel_idx].positions.len());
        for p in &out.positions {
            assert!(
                p.iter().all(|x| x.is_finite()),
                "wheel {wheel_idx}: non-finite node position after 24 substeps × 40 wheels: {p:?}"
            );
            // Hub Y was 1.0 with no downforce; some sag under gravity
            // before bead+FEM catches it is expected. Just assert the
            // mesh stayed finite and above the floor (Y=0) — this is a
            // 40-wheel atlas-batch dispatch SMOKE test, not a
            // correctness gate on the sagged shape. Correctness lives
            // in the two `cpu_vs_gpu_*` tests.
            assert!(
                p[1] > 0.0,
                "wheel {wheel_idx}: node Y={} went below the floor in atlas batch substep: {p:?}",
                p[1]
            );
        }
    }
}

#[test]
fn tire_fem_reduced_world_wrench_feeds_gpu_hub_external_force() {
    let params = TireFemHubParams {
        external_force_xyz: [10.0, -2.0, 3.0],
        ..Default::default()
    };
    let reduction = TireFemContactReductionOutput {
        normal_load: 900.0,
        contact_count: 4,
        patch_length: 0.25,
        contact_center_x: 0.02,
        min_x: -0.10,
        max_x: 0.15,
        weight_sum: 1.0,
        mean_friction: 0.9,
        mean_water_film_m: 0.0,
        mean_sinkage_m: 0.0,
        world_force: [120.0, 800.0, -45.0],
        world_moment: [0.0, 12.0, -4.0],
    };
    let coupled = tire_fem_hub_params_with_contact_reduction(params, &reduction, 0.5);
    assert_eq!(coupled.external_force_xyz, [70.0, 398.0, -19.5]);

    let hub_shader = include_str!("../shaders/tire_fem_hub.wgsl");
    assert!(
        hub_shader.contains("reaction_y - downforce + gravity_y * hub_mass + h.extf.y"),
        "GPU hub integration must apply the Y component of reduced contact force"
    );
}

#[test]
fn tire_fem_brush_outputs_feed_gpu_hub_external_force_params() {
    let base = vec![
        TireFemHubParams {
            external_force_xyz: [1.0, 2.0, 3.0],
            ..TireFemHubParams::default()
        },
        TireFemHubParams {
            external_force_xyz: [-4.0, 5.0, -6.0],
            ..TireFemHubParams::default()
        },
    ];
    let a = BrushWheelOutput {
        world_force: [100.0, 200.0, -50.0, 0.0],
        ..Default::default()
    };
    let b = BrushWheelOutput {
        world_force: [-20.0, 40.0, 80.0, 0.0],
        ..Default::default()
    };
    let coupled = tire_fem_hub_params_with_brush_outputs(&base, &[a, b], 0.25)
        .expect("brush-to-FEM hub handoff");
    assert_eq!(coupled[0].external_force_xyz, [26.0, 52.0, -9.5]);
    assert_eq!(coupled[1].external_force_xyz, [-9.0, 15.0, 14.0]);
    assert_eq!(
        coupled[0].dt, base[0].dt,
        "base FEM hub params are preserved"
    );

    let source = include_str!("../src/tire_fem.rs");
    assert!(source.contains("tire_fem_hub_params_with_brush_outputs"));
    assert!(source.contains("BrushWheelOutput"));
    assert!(source.contains("step_resident_atlas_hubs"));
    assert!(source.contains("update_resident_atlas_hub_params_from_brush_output"));
    assert!(source.contains("step_resident_atlas_hubs_with_resident_params"));
    assert!(source.contains("step_resident_atlas_hubs_from_brush_output"));
    assert!(source.contains("step_resident_vehicle_fem_hubs_from_brush_owner"));
    assert!(source.contains("step_resident_vehicle_fem_atlas_from_brush_owner"));
    assert!(source.contains("TireResidentVehicleFemAtlasOwner"));
    assert!(source.contains("create_resident_vehicle_fem_atlas_owner"));
    assert!(source.contains("step_resident_vehicle_fem_atlas_owner"));
    assert!(source.contains("update_resident_atlas_hub_params_from_contact_reduction_output"));
    assert!(source.contains("step_resident_atlas_hubs_from_contact_reduction_output"));
    assert!(source.contains("readback_resident_atlas_hub_param_rows"));
    let atlas_shader = include_str!("../shaders/tire_fem_atlas.wgsl");
    assert!(atlas_shader.contains("hub_params_from_contact_reduction_outputs"));
    assert!(atlas_shader.contains("contact_hub_outputs[tire].world_force_sinkage.xyz"));
}

#[test]
fn tire_fem_feedback_force_rows_feed_gpu_hub_external_force_params() {
    let inputs = vec![simple_tire_fem_input(0.0), simple_tire_fem_input(3.0)];
    let layout = tire_fem_gpu_batch_atlas_layout(&inputs).expect("atlas layout");
    let base = vec![
        TireFemHubParams {
            external_force_xyz: [1.0, 2.0, 3.0],
            ..TireFemHubParams::default()
        },
        TireFemHubParams {
            external_force_xyz: [-4.0, 5.0, -6.0],
            ..TireFemHubParams::default()
        },
    ];
    let feedback = TireFemContactFeedbackOutput {
        velocities: vec![[0.0; 4]; layout.plan.total_node_count],
        bristles: Vec::new(),
        forces: vec![
            [10.0, 0.0, 1.0, 0.0],
            [20.0, 1.0, 2.0, 0.0],
            [30.0, 2.0, 3.0, 0.0],
            [40.0, 3.0, 4.0, 0.0],
            [-1.0, 5.0, 10.0, 0.0],
            [-2.0, 6.0, 20.0, 0.0],
            [-3.0, 7.0, 30.0, 0.0],
            [-4.0, 8.0, 40.0, 0.0],
        ],
    };
    let coupled = tire_fem_hub_params_with_feedback_output(&base, &feedback, &layout, 0.5)
        .expect("feedback force-to-FEM hub handoff");
    assert_eq!(coupled[0].external_force_xyz, [51.0, 5.0, 8.0]);
    assert_eq!(coupled[1].external_force_xyz, [-9.0, 18.0, 44.0]);

    let source = include_str!("../src/tire_fem.rs");
    assert!(source.contains("tire_fem_hub_params_with_feedback_output"));
    assert!(source.contains("update_resident_atlas_hub_params_from_feedback_output"));
    assert!(source.contains("step_resident_atlas_hubs_from_feedback_output"));
    assert!(source.contains("feedback_output.forces"));
    let atlas_shader = include_str!("../shaders/tire_fem_atlas.wgsl");
    assert!(atlas_shader.contains("hub_params_from_feedback_force_rows"));
    assert!(atlas_shader.contains("feedback_hub_forces[i].xyz"));
    assert!(atlas_shader.contains("offsets.node.x"));
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l > 1.0e-7 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        [1.0, 0.0, 0.0]
    }
}

fn sub4(a: [f32; 4], b: [f32; 4]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn quad_dihedral_angle(p: &[[f32; 4]], ids: [usize; 4]) -> f32 {
    let x0 = p[ids[0]];
    let x1 = p[ids[1]];
    let x2 = p[ids[2]];
    let x3 = p[ids[3]];
    let n0 = normalize3(cross3(sub4(x1, x0), sub4(x2, x0)));
    let n1 = normalize3(cross3(sub4(x2, x0), sub4(x3, x0)));
    (n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2])
        .clamp(-1.0, 1.0)
        .acos()
}

fn simple_tet_fiber_stretch(pos: &[[f32; 4]], dir: [f32; 3]) -> f32 {
    let grads = [
        [-1.0_f32, -1.0, -1.0],
        [1.0_f32, 0.0, 0.0],
        [0.0_f32, 1.0, 0.0],
        [0.0_f32, 0.0, 1.0],
    ];
    let mut f = [[0.0_f32; 3]; 3];
    for a in 0..4 {
        for row in 0..3 {
            for col in 0..3 {
                f[row][col] += pos[a][row] * grads[a][col];
            }
        }
    }
    let fa = [
        f[0][0] * dir[0] + f[0][1] * dir[1] + f[0][2] * dir[2],
        f[1][0] * dir[0] + f[1][1] * dir[1] + f[1][2] * dir[2],
        f[2][0] * dir[0] + f[2][1] * dir[1] + f[2][2] * dir[2],
    ];
    (fa[0] * fa[0] + fa[1] * fa[1] + fa[2] * fa[2]).sqrt()
}

fn simple_tire_fem_input(x_shift: f32) -> TireFemGpuInput {
    TireFemGpuInput {
        rest_positions: vec![
            [x_shift + 0.0, 0.0, 0.0, 0.0],
            [x_shift + 1.0, 0.0, 0.0, 0.0],
            [x_shift + 0.0, 1.0, 0.0, 0.0],
            [x_shift + 0.0, 0.0, 1.0, 0.0],
        ],
        positions: vec![
            [x_shift + 0.0, 0.05, 0.0, 0.0],
            [x_shift + 1.0, 0.05, 0.0, 0.0],
            [x_shift + 0.0, 1.05, 0.0, 0.0],
            [x_shift + 0.0, 0.05, 1.0, 0.0],
        ],
        velocities: vec![[0.0; 4]; 4],
        masses: vec![0.25; 4],
        tet_indices: vec![0, 1, 2, 3],
        tet_grads: vec![
            [-1.0, -1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        tet_volumes: vec![1.0 / 6.0],
        tet_mu: vec![3_400.0],
        tet_lambda: vec![23_000.0],
        tet_fiber_dir: vec![[0.0, 0.0, 0.0, 0.0]],
        tet_fiber_k1: vec![0.0],
        tet_fiber_k2: vec![0.0],
        tet_fiber_dir_b: vec![[0.0, 0.0, 0.0, 0.0]],
        tet_fiber_k1_b: vec![0.0],
        tet_fiber_k2_b: vec![0.0],
        tet_fiber_dir_c: vec![[0.0, 0.0, 0.0, 0.0]],
        tet_fiber_k1_c: vec![0.0],
        tet_fiber_k2_c: vec![0.0],
        tet_fiber_dir_d: vec![[0.0, 0.0, 0.0, 0.0]],
        tet_fiber_k1_d: vec![0.0],
        tet_fiber_k2_d: vec![0.0],
        node_tet_offsets: vec![0, 1, 2, 3, 4],
        node_tet_entries: vec![0, 1, 2, 3],
        xpbd_edge_indices: vec![],
        xpbd_edge_params: vec![],
        node_edge_offsets: vec![],
        node_edge_entries: vec![],
        xpbd_dihedral_indices: vec![],
        xpbd_dihedral_params: vec![],
        node_dihedral_offsets: vec![],
        node_dihedral_entries: vec![],
        bead_ids: vec![],
        bead_offsets: vec![],
        tread_ids: vec![2],
        params: TireFemGpuParams {
            dt: 1.0 / 600.0,
            young_modulus: 1.0e4,
            pressure_pa: 1.0e3,
            pressure_area_per_tread_node: 0.01,
            hub: [x_shift, 0.25, 0.0],
            ..Default::default()
        },
    }
}

#[test]
fn tire_fem_batch_dispatch_plan_accounts_for_fused_atlas_scaling() {
    let inputs = vec![
        simple_tire_fem_input(0.0),
        simple_tire_fem_input(3.0),
        simple_tire_fem_input(-3.0),
    ];
    let plan = tire_fem_gpu_batch_dispatch_plan(&inputs).expect("batch dispatch plan");
    assert!(
        plan.is_nonempty(),
        "batch plan should be non-empty: {plan:?}"
    );
    assert_eq!(plan.tire_count, 3);
    assert_eq!(plan.total_node_count, 12);
    assert_eq!(plan.total_tet_count, 3);
    assert_eq!(plan.total_tread_count, 3);
    assert_eq!(plan.total_xpbd_edge_count, 0);
    assert_eq!(plan.total_xpbd_dihedral_count, 0);
    assert_eq!(plan.max_nodes_per_tire, 4);
    assert_eq!(plan.node_workgroup_count, 1);
    assert!(
        plan.persistent_state_bytes
            > (plan.total_node_count * core::mem::size_of::<[f32; 4]>()) as u64,
        "fused-atlas plan should account for state plus topology/material buffers: {plan:?}"
    );
}

#[test]
fn tire_fem_gpu_shaders_include_sidewall_ply_strain_xpbd_guardrail() {
    let source = include_str!("../src/tire_fem.rs");
    let shader = include_str!("../shaders/tire_fem.wgsl");
    let atlas_shader = include_str!("../shaders/tire_fem_atlas.wgsl");
    assert!(
        source.contains("xpbd_sidewall_ply_stretch_limit")
            && source.contains("xpbd_sidewall_ply_compliance")
            && source.contains("params.xpbd_sidewall_ply_stretch_limit")
            && source.contains("params.xpbd_sidewall_ply_compliance"),
        "TireFemGpuParams and uniform rows should carry sidewall ply-strain XPBD knobs"
    );
    for (name, text) in [("single", shader), ("atlas", atlas_shader)] {
        assert!(
            text.contains("Stage (f4)")
                && text.contains("ply_limit")
                && text.contains("ply_compliance")
                && text.contains("tet_fiber_dir_c")
                && text.contains("tet_fiber_dir_d")
                && text.contains("abs(dot(tet_fiber_dir_c")
                && text.contains("let c_ply = stretch - ply_limit")
                && text.contains("dp_ply_sum")
                && text.contains("p_next = p_next + dp_ply"),
            "{name} tire FEM shader should apply sidewall ply-strain XPBD from fibre stretch"
        );
    }
}

#[test]
fn tire_fem_batch_atlas_layout_records_contiguous_offsets_for_fused_dispatch() {
    let mut large = simple_tire_fem_input(6.0);
    large.positions.push([6.0, 0.42, 0.0, 1.0]);
    large.rest_positions.push([6.0, 0.42, 0.0, 1.0]);
    large.velocities.push([0.0, 0.0, 0.0, 0.0]);
    large.masses.push(1.0);
    large.node_tet_offsets = vec![0, 1, 2, 3, 4, 4];

    let inputs = vec![
        simple_tire_fem_input(0.0),
        large,
        simple_tire_fem_input(-6.0),
    ];
    let layout = tire_fem_gpu_batch_atlas_layout(&inputs).expect("batch atlas layout");
    assert!(layout.plan.is_nonempty());
    assert!(
        layout.is_contiguous(),
        "fused atlas layout should be contiguous and account for all rows: {layout:?}"
    );
    assert_eq!(layout.rows.len(), 3);
    assert_eq!(layout.rows[0].node_offset, 0);
    assert_eq!(layout.rows[0].xpbd_edge_offset, 0);
    assert_eq!(layout.rows[0].xpbd_dihedral_offset, 0);
    assert_eq!(layout.rows[1].node_offset, inputs[0].positions.len());
    assert_eq!(layout.rows[1].xpbd_edge_offset, 0);
    assert_eq!(layout.rows[1].xpbd_dihedral_offset, 0);
    assert_eq!(
        layout.rows[2].node_offset,
        inputs[0].positions.len() + inputs[1].positions.len()
    );
    assert_eq!(layout.rows[1].node_count, 5);
    assert_eq!(
        layout.rows.last().unwrap().node_offset + layout.rows.last().unwrap().node_count,
        layout.plan.total_node_count
    );
    assert!(
        layout
            .rows
            .windows(2)
            .all(|pair| pair[1].node_workgroup_offset >= pair[0].node_workgroup_offset),
        "node workgroup offsets should be monotone for future fused dispatch: {layout:?}"
    );
}

fn structural_tire_fem_input(x_shift: f32) -> TireFemGpuInput {
    let mut input = simple_tire_fem_input(x_shift);
    input.bead_ids = vec![0, 1];
    input.bead_offsets = vec![[0.0, -0.25, 0.0, 0.0], [1.0, -0.25, 0.0, 0.0]];
    input.xpbd_edge_indices = vec![[0, 1, 2, 0], [2, 3, 1, 0]];
    input.xpbd_edge_params = vec![[1.0, 2.0e-5, 0.0, 0.0], [1.4, 3.0e-5, 0.0, 0.0]];
    input.node_edge_offsets = vec![0, 1, 2, 3, 4];
    input.node_edge_entries = vec![0, 1, 2, 3];
    input.xpbd_dihedral_indices = vec![[0, 1, 2, 3]];
    input.xpbd_dihedral_params = vec![[0.85, 4.0e-5, 0.0, 0.0]];
    input.node_dihedral_offsets = vec![0, 1, 2, 3, 4];
    input.node_dihedral_entries = vec![0, 1, 2, 3];
    input
}

#[test]
fn tire_fem_batch_atlas_buffers_rewrite_topology_indices_for_fused_dispatch() {
    let inputs = vec![
        structural_tire_fem_input(0.0),
        simple_tire_fem_input(4.0),
        structural_tire_fem_input(-4.0),
    ];
    let atlas = tire_fem_gpu_batch_atlas_buffers(&inputs).expect("batch atlas buffers");
    atlas.validate().expect("atlas buffers self-validate");
    assert!(
        atlas.storage_bytes()
            > ((atlas.positions.len() + atlas.velocities.len()) * core::mem::size_of::<[f32; 4]>())
                as u64,
        "atlas storage accounting should include topology, routing, params, and XPBD rows"
    );
    assert!(
        atlas.layout.is_contiguous(),
        "packed atlas should keep the same contiguous row contract: {:?}",
        atlas.layout
    );
    assert_eq!(atlas.layout.plan.total_xpbd_edge_count, 4);
    assert_eq!(atlas.layout.plan.total_xpbd_dihedral_count, 2);
    assert_eq!(atlas.layout.rows[0].xpbd_edge_offset, 0);
    assert_eq!(atlas.layout.rows[0].xpbd_edge_count, 2);
    assert_eq!(atlas.layout.rows[1].xpbd_edge_offset, 2);
    assert_eq!(atlas.layout.rows[1].xpbd_edge_count, 0);
    assert_eq!(atlas.layout.rows[2].xpbd_edge_offset, 2);
    assert_eq!(atlas.layout.rows[2].xpbd_edge_count, 2);
    assert_eq!(atlas.layout.rows[2].xpbd_dihedral_offset, 1);
    assert_eq!(atlas.positions.len(), 12);
    assert_eq!(
        atlas.node_tire_indices,
        vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    );
    assert_eq!(atlas.tet_tire_indices, vec![0, 1, 2]);
    assert_eq!(atlas.tread_tire_indices, vec![0, 1, 2]);
    assert_eq!(atlas.bead_tire_indices, vec![0, 0, 2, 2]);
    assert_eq!(
        atlas.tet_indices,
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    );
    assert_eq!(atlas.bead_ids, vec![0, 1, 8, 9]);
    assert_eq!(atlas.tread_ids, vec![2, 6, 10]);
    assert_eq!(
        atlas.node_tet_offsets,
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    );
    assert_eq!(
        atlas.node_tet_entries,
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    );
    assert_eq!(
        atlas.xpbd_edge_indices,
        vec![[0, 1, 2, 0], [2, 3, 1, 0], [8, 9, 2, 0], [10, 11, 1, 0],]
    );
    assert_eq!(
        atlas.node_edge_offsets,
        vec![0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 6, 7, 8]
    );
    assert_eq!(atlas.node_edge_entries, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(
        atlas.xpbd_dihedral_indices,
        vec![[0, 1, 2, 3], [8, 9, 10, 11]]
    );
    assert_eq!(
        atlas.node_dihedral_offsets,
        vec![0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 6, 7, 8]
    );
    assert_eq!(atlas.node_dihedral_entries, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(atlas.param_rows.len(), 3);
    assert_eq!(
        atlas.param_rows[0],
        tire_fem_gpu_batch_atlas_param_row(&inputs[0]).unwrap()
    );
    assert_eq!(atlas.param_rows[1].counts, [4, 1, 0, 1]);
    assert_eq!(atlas.param_rows[2].counts, [4, 1, 2, 1]);
    assert_eq!(atlas.param_rows[2].counts2, [2, 1, 0, 0]);
    assert_eq!(atlas.param_rows[1].phys2[1], 4.0);
    assert_eq!(atlas.param_rows[2].phys2[1], -4.0);
    assert_eq!(atlas.offset_rows.len(), 3);
    assert_eq!(
        atlas.offset_rows[0],
        tire_fem_gpu_batch_atlas_offset_row(atlas.layout.rows[0]).unwrap()
    );
    assert_eq!(atlas.offset_rows[0].node, [0, 4, 0, 1]);
    assert_eq!(atlas.offset_rows[1].node, [4, 4, 1, 1]);
    assert_eq!(atlas.offset_rows[2].node, [8, 4, 2, 1]);
    assert_eq!(atlas.offset_rows[0].contact, [0, 1, 0, 2]);
    assert_eq!(atlas.offset_rows[1].contact, [1, 1, 2, 0]);
    assert_eq!(atlas.offset_rows[2].contact, [2, 1, 2, 2]);
    assert_eq!(atlas.offset_rows[2].xpbd, [2, 2, 1, 1]);
    assert_eq!(atlas.params.len(), 3);
    assert_eq!(atlas.params[1].hub[0], 4.0);

    let bindings = tire_fem_gpu_batch_atlas_binding_plan(&atlas).expect("atlas binding plan");
    assert_eq!(bindings.len(), 44);
    for (expected, binding) in bindings.iter().enumerate() {
        assert_eq!(binding.binding as usize, expected);
        assert!(
            binding.bytes >= 4,
            "wgpu storage buffers need non-zero size"
        );
    }
    assert_eq!(bindings[0].name, "rest_pos");
    assert_eq!(bindings[12].name, "pos_out");
    assert!(!bindings[12].read_only && !bindings[13].read_only && !bindings[14].read_only);
    assert_eq!(bindings[20].name, "hub_state");
    assert_eq!(bindings[20].rows, atlas.layout.plan.tire_count * 2);
    assert_eq!(bindings[38].name, "node_tire_indices");
    assert_eq!(bindings[38].rows, atlas.layout.plan.total_node_count);
    assert_eq!(bindings[42].name, "param_rows");
    assert_eq!(bindings[42].rows, atlas.layout.plan.tire_count);
    assert_eq!(bindings[43].name, "offset_rows");
    assert_eq!(bindings[43].rows, atlas.layout.plan.tire_count);
    let contract =
        tire_fem_gpu_batch_atlas_dispatch_contract(&atlas).expect("atlas dispatch contract");
    assert_eq!(
        contract.shader_label,
        "molla::tire_fem_atlas::step_nodes_atlas_contract"
    );
    assert_eq!(contract.entry_point, "step_nodes_atlas_contract");
    assert_eq!(contract.workgroup_size, 64);
    assert_eq!(
        contract.workgroup_count,
        atlas.layout.plan.node_workgroup_count
    );
    assert_eq!(
        contract.total_node_count,
        atlas.layout.plan.total_node_count
    );
    assert_eq!(contract.binding_count, bindings.len());
    assert!(TIRE_FEM_ATLAS_SHADER.contains("fn step_nodes_atlas_contract"));
    assert!(TIRE_FEM_ATLAS_SHADER.contains("@compute @workgroup_size(64)"));
    assert!(TIRE_FEM_ATLAS_SHADER.contains("struct AtlasParamRow"));
    assert!(TIRE_FEM_ATLAS_SHADER.contains("struct AtlasOffsetRow"));
    for binding in &bindings {
        let declaration = format!("@binding({})", binding.binding);
        assert!(
            TIRE_FEM_ATLAS_SHADER.contains(&declaration),
            "atlas shader contract should declare {declaration} for {}",
            binding.name
        );
        assert!(
            TIRE_FEM_ATLAS_SHADER.contains(binding.name),
            "atlas shader contract should include binding name {}",
            binding.name
        );
    }
    assert!(
        TIRE_FEM_ATLAS_SHADER.contains("node_tire_indices[i]")
            && TIRE_FEM_ATLAS_SHADER.contains("param_rows[tire]")
            && TIRE_FEM_ATLAS_SHADER.contains("offset_rows[tire]"),
        "atlas contract entry must route nodes through tire ownership rows"
    );
    assert!(
        TIRE_FEM_ATLAS_SHADER.contains("let F = transpose(mat3x3<f32>(f0, f1, f2))")
            && TIRE_FEM_ATLAS_SHADER.contains("tet_mu[t] + params.phys3.z")
            && TIRE_FEM_ATLAS_SHADER.contains("add_fiber_stress")
            && TIRE_FEM_ATLAS_SHADER.contains("node_tet_offsets[i]")
            && TIRE_FEM_ATLAS_SHADER.contains("hub_state[tire * 2u]")
            && TIRE_FEM_ATLAS_SHADER.contains("xpbd_edge_indices[eidx]")
            && TIRE_FEM_ATLAS_SHADER.contains("xpbd_dihedral_indices[didx]")
            && TIRE_FEM_ATLAS_SHADER.contains("pos_out[i] = vec4<f32>(p_next, 0.0)")
            && !TIRE_FEM_ATLAS_SHADER.contains("pos_out[i] = pos_in[i]"),
        "atlas shader should now contain the physical FEM node update, not only the old routing-copy contract"
    );

    for (idx, original) in inputs.iter().enumerate() {
        let extracted = atlas.extract_input(idx).expect("extract tire input");
        assert_eq!(extracted.rest_positions, original.rest_positions);
        assert_eq!(extracted.positions, original.positions);
        assert_eq!(extracted.velocities, original.velocities);
        assert_eq!(extracted.masses, original.masses);
        assert_eq!(extracted.tet_indices, original.tet_indices);
        assert_eq!(extracted.tet_grads, original.tet_grads);
        assert_eq!(extracted.tet_volumes, original.tet_volumes);
        assert_eq!(extracted.tet_mu, original.tet_mu);
        assert_eq!(extracted.tet_lambda, original.tet_lambda);
        assert_eq!(extracted.bead_ids, original.bead_ids);
        assert_eq!(extracted.bead_offsets, original.bead_offsets);
        assert_eq!(extracted.tread_ids, original.tread_ids);
        assert_eq!(extracted.node_tet_offsets, original.node_tet_offsets);
        assert_eq!(extracted.node_tet_entries, original.node_tet_entries);
        assert_eq!(extracted.xpbd_edge_indices, original.xpbd_edge_indices);
        assert_eq!(extracted.xpbd_edge_params, original.xpbd_edge_params);
        assert_eq!(extracted.node_edge_offsets, original.node_edge_offsets);
        assert_eq!(extracted.node_edge_entries, original.node_edge_entries);
        assert_eq!(
            extracted.xpbd_dihedral_indices,
            original.xpbd_dihedral_indices
        );
        assert_eq!(
            extracted.xpbd_dihedral_params,
            original.xpbd_dihedral_params
        );
        assert_eq!(
            extracted.node_dihedral_offsets,
            original.node_dihedral_offsets
        );
        assert_eq!(
            extracted.node_dihedral_entries,
            original.node_dihedral_entries
        );
        assert_eq!(extracted.params.hub, original.params.hub);
    }
    assert!(
        atlas.extract_input(99).is_err(),
        "extract_input should reject out-of-range tire rows"
    );

    let positions = (0..12)
        .map(|i| [i as f32, 10.0 + i as f32, 20.0 + i as f32, 1.0])
        .collect::<Vec<_>>();
    let velocities = (0..12)
        .map(|i| [30.0 + i as f32, 40.0 + i as f32, 50.0 + i as f32, 0.0])
        .collect::<Vec<_>>();
    let forces = (0..12)
        .map(|i| [60.0 + i as f32, 70.0 + i as f32, 80.0 + i as f32, 0.0])
        .collect::<Vec<_>>();
    let outputs = atlas
        .split_outputs(&positions, &velocities, &forces)
        .expect("split fused atlas outputs");
    assert_eq!(outputs.len(), 3);
    assert_eq!(outputs[0].positions, positions[0..4]);
    assert_eq!(outputs[1].velocities, velocities[4..8]);
    assert_eq!(outputs[2].forces, forces[8..12]);
    let joined = atlas
        .join_outputs(&outputs)
        .expect("join per-tire outputs back to atlas rows");
    assert_eq!(joined.positions, positions);
    assert_eq!(joined.velocities, velocities);
    assert_eq!(joined.forces, forces);
    assert_eq!(
        atlas
            .split_output_rows(&joined)
            .expect("split joined atlas rows")[2]
            .forces,
        outputs[2].forces
    );
    let mut bad_outputs = outputs.clone();
    bad_outputs[1].positions.pop();
    assert!(
        atlas.join_outputs(&bad_outputs).is_err(),
        "join_outputs must reject per-tire rows with the wrong node count"
    );
    let bad_positions = positions[0..11].to_vec();
    assert!(
        atlas
            .split_outputs(&bad_positions, &velocities, &forces)
            .is_err(),
        "splitter must reject truncated fused-atlas readback rows"
    );

    let source = include_str!("../src/tire_fem.rs");
    let lib_source = include_str!("../src/lib.rs");
    assert!(
        lib_source.contains("TireFemGpuAtlasBatch"),
        "resident fused atlas owner should be exported from molla-kernels"
    );
    assert!(
        !source.contains("later GPU-atlas/fused-dispatch path"),
        "resident fused atlas docs should not describe the implemented atlas owner as future work"
    );
    let reference_loop = source
        .split("pub fn dispatch_atlas_reference_loop")
        .nth(1)
        .expect("atlas reference-loop dispatch should exist");
    let reference_loop = reference_loop
        .split("pub fn create_persistent")
        .next()
        .expect("reference-loop dispatch should be before persistent path");
    assert!(reference_loop.contains("atlas.extract_input(tire_index)"));
    assert!(reference_loop.contains("self.dispatch_blocking(device, queue, &input)"));
    assert!(reference_loop.contains("atlas.join_outputs(&outputs)"));

    let constructor = source
        .split("pub fn new(device: &wgpu::Device) -> Self")
        .nth(1)
        .expect("TireFemGpuKernel constructor should exist");
    let constructor = constructor
        .split("pub fn dispatch_blocking")
        .next()
        .expect("constructor should precede blocking dispatch");
    assert!(constructor.contains("TIRE_FEM_ATLAS_SHADER"));
    assert!(constructor.contains("atlas_contract_layout_io"));
    assert!(constructor.contains("atlas_contract_pipeline"));
    assert!(constructor.contains("step_nodes_atlas_contract"));
    assert!(constructor.contains("atlas_hub_layout_io"));
    assert!(constructor.contains("atlas_hub_pipeline"));
    assert!(constructor.contains("hub_reduce_atlas"));

    let contract_dispatch = source
        .split("pub fn dispatch_atlas_contract_blocking")
        .nth(1)
        .expect("one-dispatch atlas contract path should exist");
    let contract_dispatch = contract_dispatch
        .split("pub fn dispatch_atlas_reference_loop")
        .next()
        .expect("contract dispatch should precede reference loop");
    assert!(contract_dispatch.contains("atlas.validate()?"));
    assert!(contract_dispatch.contains("self.atlas_contract_layout_io"));
    assert!(contract_dispatch.contains("self.atlas_contract_pipeline"));
    assert!(contract_dispatch.contains("storage_buffer_from_slice"));
    assert!(contract_dispatch.contains("tire_fem_gpu_batch_atlas_binding_plan(atlas)?"));
    assert!(contract_dispatch.contains("tire_fem_gpu_batch_atlas_dispatch_contract(atlas)?"));
    assert!(contract_dispatch.contains("pass.dispatch_workgroups(contract.workgroup_count, 1, 1)"));
    assert!(contract_dispatch.contains("read_vec4(device, &stage_pos"));
    assert!(contract_dispatch.contains("read_vec4(device, &stage_vel"));
    assert!(contract_dispatch.contains("read_vec4(device, &stage_force"));

    let batch_dispatch = source
        .split("pub fn dispatch_batch_atlas_blocking")
        .nth(1)
        .expect("high-level fused batch atlas dispatch should exist");
    let batch_dispatch = batch_dispatch
        .split("pub fn dispatch_atlas_reference_loop")
        .next()
        .expect("batch atlas dispatch should precede reference loop");
    assert!(batch_dispatch.contains("tire_fem_gpu_batch_atlas_buffers(inputs)?"));
    assert!(
        batch_dispatch.contains("self.dispatch_atlas_contract_blocking(device, queue, &atlas)?")
    );
    assert!(batch_dispatch.contains("atlas.split_output_rows(&rows)"));

    let resident_create = source
        .split("pub fn create_resident_atlas_batch")
        .nth(1)
        .expect("resident fused atlas owner should exist");
    let resident_create = resident_create
        .split("pub fn step_hub_persistent")
        .next()
        .expect("resident atlas creation should precede hub stepping");
    assert!(resident_create.contains("TireFemGpuAtlasBatch"));
    assert!(resident_create.contains("bg_a_to_b"));
    assert!(resident_create.contains("bg_b_to_a"));
    assert!(resident_create.contains("atlas_hub_bg_a"));
    assert!(resident_create.contains("atlas_hub_bg_b"));
    assert!(resident_create.contains("atlas_hub_param_rows_buf"));
    assert!(resident_create.contains("tire_fem_gpu_batch_atlas_buffers(inputs)?"));

    let resident_step = source
        .split("pub fn step_resident_atlas_batch")
        .nth(1)
        .expect("resident fused atlas step should exist");
    let resident_step = resident_step
        .split("pub fn readback_resident_atlas_rows")
        .next()
        .expect("resident step should precede resident readback");
    assert!(resident_step.contains("queue.write_buffer("));
    assert!(resident_step.contains("batch.param_rows_buf"));
    assert!(resident_step.contains("batch.hub_state_buf"));
    assert!(resident_step.contains("self.atlas_contract_pipeline"));
    assert!(resident_step.contains("batch.bg_b_to_a"));
    assert!(resident_step.contains("batch.bg_a_to_b"));
    assert!(resident_step.contains("batch.parity = !batch.parity"));
    assert!(resident_step.contains("pub fn step_resident_atlas_hubs"));
    assert!(resident_step.contains("self.atlas_hub_pipeline"));
    assert!(resident_step.contains("batch.atlas_hub_bg_b"));
    assert!(resident_step.contains("batch.atlas_hub_bg_a"));
    assert!(resident_step.contains("pass.dispatch_workgroups(tire_count as u32, 1, 1)"));
    assert!(
        !resident_step.contains("atlas_hub_param_rows_buf")
            || resident_step.contains("TireFemHubParams"),
        "resident atlas hub stepping should be driven by explicit hub params, not hidden scene state"
    );

    let resident_upload = source
        .split("pub fn upload_resident_atlas_batch_state")
        .nth(1)
        .expect("resident fused atlas state upload should exist");
    let resident_upload = resident_upload
        .split("pub fn readback_resident_atlas_rows")
        .next()
        .expect("resident upload should precede resident readback");
    assert!(resident_upload.contains("batch.atlas.join_outputs(outputs)?"));
    assert!(resident_upload.contains("batch.pos_b"));
    assert!(resident_upload.contains("batch.pos_a"));
    assert!(resident_upload.contains("queue.write_buffer(current_pos"));
    assert!(resident_upload.contains("queue.write_buffer(current_vel"));
    assert!(resident_upload.contains("queue.write_buffer(&batch.force_buf"));
    assert!(resident_upload.contains("pub fn upload_resident_atlas_rows"));
    assert!(resident_upload.contains("batch.atlas.split_output_rows(rows)?"));
    assert!(resident_upload.contains("pub fn upload_resident_atlas_tire_state"));
    assert!(resident_upload.contains("row.node_offset"));
    assert!(resident_upload.contains("byte_offset"));
    assert!(resident_upload.contains("pub fn upload_resident_atlas_hub_rows"));
    assert!(resident_upload.contains("pub fn upload_resident_atlas_tire_hub"));
    assert!(resident_upload.contains("batch.hub_state_buf"));
    assert!(resident_upload.contains("tire_count * 2"));

    let resident_hub_readback = source
        .split("pub fn readback_resident_atlas_hub_rows")
        .nth(1)
        .expect("resident fused atlas hub readback should exist");
    let resident_hub_readback = resident_hub_readback
        .split("pub fn readback_resident_atlas_batch")
        .next()
        .expect("hub readback should precede resident batch readback");
    assert!(resident_hub_readback.contains("batch.hub_state_buf"));
    assert!(resident_hub_readback.contains("read_vec4(device, &stage_hub"));

    let atlas_shader = include_str!("../shaders/tire_fem_atlas.wgsl");
    assert!(atlas_shader.contains("struct AtlasHubParamRow"));
    assert!(atlas_shader.contains("@binding(44)"));
    assert!(atlas_shader.contains("fn hub_reduce_atlas"));
    assert!(atlas_shader.contains("hub_state[hub_row]"));
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_batch_atlas_contract_dispatch_routes_packed_rows_on_gpu() {
    let Some(engine) = try_engine() else {
        eprintln!("skipping atlas contract hardware dispatch: no wgpu adapter");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemGpuKernel::new(&device);
    let inputs = vec![
        structural_tire_fem_input(0.0),
        simple_tire_fem_input(4.0),
        structural_tire_fem_input(-4.0),
    ];
    let atlas = tire_fem_gpu_batch_atlas_buffers(&inputs).expect("batch atlas buffers");

    let rows = kernel
        .dispatch_atlas_contract_blocking(&device, &queue, &atlas)
        .expect("one-dispatch atlas physical dispatch");
    let batch_rows = kernel
        .dispatch_batch_atlas_blocking(&device, &queue, &inputs)
        .expect("high-level one-dispatch atlas batch");
    let reference = kernel
        .dispatch_atlas_reference_loop(&device, &queue, &atlas)
        .expect("per-tire reference dispatch");
    let split = atlas
        .split_output_rows(&rows)
        .expect("split one-dispatch atlas rows");
    assert_eq!(split.len(), batch_rows.len());
    for (split_tire, batch_tire) in split.iter().zip(&batch_rows) {
        assert_eq!(split_tire.positions, batch_tire.positions);
        assert_eq!(split_tire.velocities, batch_tire.velocities);
        assert_eq!(split_tire.forces, batch_tire.forces);
    }
    let mut resident = kernel
        .create_resident_atlas_batch(&device, &queue, &inputs)
        .expect("resident fused atlas batch");
    assert_eq!(resident.tire_count(), inputs.len());
    assert_eq!(
        resident.total_node_count(),
        atlas.layout.plan.total_node_count
    );
    let params = inputs.iter().map(|input| input.params).collect::<Vec<_>>();
    kernel
        .step_resident_atlas_batch(&device, &queue, &mut resident, &params)
        .expect("resident fused atlas step");
    let resident_rows = kernel
        .readback_resident_atlas_rows(&device, &queue, &resident)
        .expect("resident fused atlas readback");
    assert_eq!(resident_rows.positions.len(), rows.positions.len());
    assert_eq!(resident_rows.velocities.len(), rows.velocities.len());
    assert_eq!(resident_rows.forces.len(), rows.forces.len());
    let resident_split = kernel
        .readback_resident_atlas_batch(&device, &queue, &resident)
        .expect("resident fused atlas split readback");
    assert_eq!(resident_split.len(), inputs.len());
    for (node, (resident_p, blocking_p)) in resident_rows
        .positions
        .iter()
        .zip(rows.positions.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (resident_p[axis] - blocking_p[axis]).abs() < 1.0e-4,
                "resident atlas position should match blocking atlas at node {node} axis {axis}: resident={resident_p:?} blocking={blocking_p:?}"
            );
        }
    }
    kernel
        .upload_resident_atlas_rows(&queue, &resident, &rows)
        .expect("reseed resident atlas from packed rows");
    let reseeded_rows = kernel
        .readback_resident_atlas_rows(&device, &queue, &resident)
        .expect("resident fused atlas readback after reseed");
    assert_eq!(reseeded_rows.positions, rows.positions);
    assert_eq!(reseeded_rows.velocities, rows.velocities);
    assert_eq!(reseeded_rows.forces, rows.forces);
    let mut tire_one = resident_split[1].clone();
    for pos in &mut tire_one.positions {
        pos[1] += 0.025;
    }
    kernel
        .upload_resident_atlas_tire_state(&queue, &resident, 1, &tire_one)
        .expect("reseed one tire in resident atlas");
    let one_tire_rows = kernel
        .readback_resident_atlas_rows(&device, &queue, &resident)
        .expect("resident fused atlas readback after single-tire reseed");
    let tire_one_range = atlas.layout.rows[1].node_offset
        ..atlas.layout.rows[1].node_offset + atlas.layout.rows[1].node_count;
    assert_eq!(
        &one_tire_rows.positions[tire_one_range.clone()],
        tire_one.positions.as_slice()
    );
    assert_eq!(
        &one_tire_rows.positions[0..atlas.layout.rows[0].node_count],
        &rows.positions[0..atlas.layout.rows[0].node_count],
        "single-tire reseed must not modify preceding tire rows"
    );
    let hub_rows = vec![
        [1.0, 2.0, 3.0, 0.0],
        [0.1, 0.2, 0.3, 0.0],
        [4.0, 5.0, 6.0, 0.0],
        [0.4, 0.5, 0.6, 0.0],
        [7.0, 8.0, 9.0, 0.0],
        [0.7, 0.8, 0.9, 0.0],
    ];
    kernel
        .upload_resident_atlas_hub_rows(&queue, &resident, &hub_rows)
        .expect("upload packed resident hub rows");
    assert_eq!(
        kernel
            .readback_resident_atlas_hub_rows(&device, &queue, &resident)
            .expect("read back packed resident hub rows"),
        hub_rows
    );
    kernel
        .upload_resident_atlas_tire_hub(&queue, &resident, 1, [10.0, 11.0, 12.0], [1.0, 1.1, 1.2])
        .expect("upload one tire resident hub rows");
    let hub_rows_after_one = kernel
        .readback_resident_atlas_hub_rows(&device, &queue, &resident)
        .expect("read back resident hub rows after one-tire hub upload");
    assert_eq!(hub_rows_after_one[2], [10.0, 11.0, 12.0, 0.0]);
    assert_eq!(hub_rows_after_one[3], [1.0, 1.1, 1.2, 0.0]);
    assert_eq!(hub_rows_after_one[0], hub_rows[0]);
    let moving_hub_rows = (0..inputs.len())
        .flat_map(|_| [[0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        .collect::<Vec<_>>();
    kernel
        .upload_resident_atlas_hub_rows(&queue, &resident, &moving_hub_rows)
        .expect("upload packed resident hub rows before fused hub step");
    let driven_hub_params = inputs
        .iter()
        .map(|_| TireFemHubParams {
            dt: 1.0 / 60.0,
            bead_stiffness: 0.0,
            bead_damping: 0.0,
            hub_mass: 10.0,
            hub_damping: 0.0,
            downforce: 0.0,
            gravity_y: 0.0,
            hub_floor_active: 0.0,
            hub_floor_y: 0.0,
            external_force_xyz: [60.0, 0.0, 0.0],
            lateral_damping: 0.0,
        })
        .collect::<Vec<_>>();
    kernel
        .step_resident_atlas_hubs(&device, &queue, &resident, &driven_hub_params)
        .expect("fused resident atlas hub integration");
    let driven_hub_rows = kernel
        .readback_resident_atlas_hub_rows(&device, &queue, &resident)
        .expect("read back resident hub rows after fused hub integration");
    for tire in 0..inputs.len() {
        let pos = driven_hub_rows[tire * 2];
        let vel = driven_hub_rows[tire * 2 + 1];
        assert!(
            pos[0] > moving_hub_rows[tire * 2][0],
            "fused hub step should advance tire {tire} hub x under positive external force: pos={pos:?}"
        );
        assert!(
            vel[0] > 0.0,
            "fused hub step should write positive x velocity for tire {tire}: vel={vel:?}"
        );
    }
    assert_eq!(rows.forces.len(), atlas.layout.plan.total_node_count);
    assert_eq!(rows.positions.len(), reference.positions.len());
    assert_eq!(rows.velocities.len(), reference.velocities.len());
    assert_eq!(rows.forces.len(), reference.forces.len());
    for (node, (atlas_p, reference_p)) in rows
        .positions
        .iter()
        .zip(reference.positions.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (atlas_p[axis] - reference_p[axis]).abs() < 1.0e-4,
                "atlas fused position should match per-tire reference at node {node} axis {axis}: atlas={atlas_p:?} reference={reference_p:?}"
            );
        }
    }
    for (node, (atlas_v, reference_v)) in rows
        .velocities
        .iter()
        .zip(reference.velocities.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (atlas_v[axis] - reference_v[axis]).abs() < 1.0e-4,
                "atlas fused velocity should match per-tire reference at node {node} axis {axis}: atlas={atlas_v:?} reference={reference_v:?}"
            );
        }
    }
    for (node, (atlas_f, reference_f)) in
        rows.forces.iter().zip(reference.forces.iter()).enumerate()
    {
        for axis in 0..3 {
            assert!(
                (atlas_f[axis] - reference_f[axis]).abs() < 1.0e-3,
                "atlas fused force should match per-tire reference at node {node} axis {axis}: atlas={atlas_f:?} reference={reference_f:?}"
            );
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_resident_atlas_hub_params_from_brush_output_gpu() {
    let Some(engine) = try_engine() else {
        eprintln!("skipping resident atlas brush-to-hub hardware dispatch: no wgpu adapter");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemGpuKernel::new(&device);
    let inputs = vec![simple_tire_fem_input(0.0), simple_tire_fem_input(4.0)];
    let resident = kernel
        .create_resident_atlas_batch(&device, &queue, &inputs)
        .expect("resident atlas");
    let base = vec![
        TireFemHubParams {
            external_force_xyz: [1.0, 2.0, 3.0],
            lateral_damping: 11.0,
            ..TireFemHubParams::default()
        },
        TireFemHubParams {
            external_force_xyz: [-4.0, 5.0, -6.0],
            lateral_damping: 17.0,
            ..TireFemHubParams::default()
        },
    ];
    let a = BrushWheelOutput {
        world_force: [100.0, 200.0, -50.0, 0.0],
        ..Default::default()
    };
    let b = BrushWheelOutput {
        world_force: [-20.0, 40.0, 80.0, 0.0],
        ..Default::default()
    };
    let brush_rows = [a, b];
    let brush_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_fem_gpu test brush output rows"),
        size: core::mem::size_of_val(&brush_rows) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&brush_buf, 0, bytemuck::cast_slice(&brush_rows));
    kernel
        .update_resident_atlas_hub_params_from_brush_output(
            &device,
            &queue,
            &resident,
            &brush_buf,
            brush_rows.len(),
            &base,
            0.25,
        )
        .expect("resident brush-to-hub-param GPU update");
    let rows = kernel
        .readback_resident_atlas_hub_param_rows(&device, &queue, &resident)
        .expect("read resident hub param rows");
    assert_eq!(rows.len(), brush_rows.len());
    assert_eq!(rows[0].extf, [26.0, 52.0, -9.5, 11.0]);
    assert_eq!(rows[1].extf, [-9.0, 15.0, 14.0, 17.0]);
    assert_eq!(rows[0].cfg[3], base[0].dt);
    assert_eq!(rows[1].cfg[3], base[1].dt);
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_resident_atlas_hubs_step_from_brush_output_gpu() {
    let Some(engine) = try_engine() else {
        eprintln!("skipping resident atlas brush-to-hub step hardware dispatch: no wgpu adapter");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemGpuKernel::new(&device);
    let inputs = vec![simple_tire_fem_input(0.0), simple_tire_fem_input(4.0)];
    let resident = kernel
        .create_resident_atlas_batch(&device, &queue, &inputs)
        .expect("resident atlas");
    let base = vec![
        TireFemHubParams {
            dt: 1.0 / 60.0,
            bead_stiffness: 0.0,
            bead_damping: 0.0,
            hub_mass: 10.0,
            hub_damping: 0.0,
            gravity_y: 0.0,
            downforce: 0.0,
            external_force_xyz: [0.0, 0.0, 0.0],
            lateral_damping: 0.0,
            ..TireFemHubParams::default()
        },
        TireFemHubParams {
            dt: 1.0 / 60.0,
            bead_stiffness: 0.0,
            bead_damping: 0.0,
            hub_mass: 10.0,
            hub_damping: 0.0,
            gravity_y: 0.0,
            downforce: 0.0,
            external_force_xyz: [0.0, 0.0, 0.0],
            lateral_damping: 0.0,
            ..TireFemHubParams::default()
        },
    ];
    let a = BrushWheelOutput {
        world_force: [600.0, 0.0, 0.0, 0.0],
        ..Default::default()
    };
    let b = BrushWheelOutput {
        world_force: [-300.0, 0.0, 0.0, 0.0],
        ..Default::default()
    };
    let brush_rows = [a, b];
    let brush_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_fem_gpu test brush output rows for hub step"),
        size: core::mem::size_of_val(&brush_rows) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&brush_buf, 0, bytemuck::cast_slice(&brush_rows));
    kernel
        .step_resident_atlas_hubs_from_brush_output(
            &device,
            &queue,
            &resident,
            &brush_buf,
            brush_rows.len(),
            &base,
            1.0,
        )
        .expect("resident brush-to-hub-param GPU update plus hub step");
    let param_rows = kernel
        .readback_resident_atlas_hub_param_rows(&device, &queue, &resident)
        .expect("read resident hub param rows after hub step");
    assert_eq!(param_rows[0].extf[0], 600.0);
    assert_eq!(param_rows[1].extf[0], -300.0);
    let hub_rows = kernel
        .readback_resident_atlas_hub_rows(&device, &queue, &resident)
        .expect("read resident hub rows after brush-driven hub step");
    assert!(
        hub_rows[1][0] > 0.0,
        "first hub should gain positive x velocity from resident brush load: {hub_rows:?}"
    );
    assert!(
        hub_rows[3][0] < 0.0,
        "second hub should gain negative x velocity from resident brush load: {hub_rows:?}"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_resident_vehicle_fem_atlas_substep_sequences_nodes_brush_and_hubs_gpu() {
    let Some(engine) = try_engine() else {
        eprintln!("skipping resident vehicle/FEM atlas substep hardware dispatch: no wgpu adapter");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let fem_kernel = TireFemGpuKernel::new(&device);
    let vehicle_kernel = TireVehicleYawGpuKernel::new(&device);
    let brush_kernel = TireBrushGpuKernel::new(&device);

    let tire_inputs = vec![simple_tire_fem_input(0.0), simple_tire_fem_input(4.0)];
    let mut resident = fem_kernel
        .create_resident_atlas_batch(&device, &queue, &tire_inputs)
        .expect("resident atlas");
    let fem_params = tire_inputs
        .iter()
        .map(|input| TireFemGpuParams {
            dt: 1.0 / 600.0,
            hub: input.params.hub,
            pressure_pa: input.params.pressure_pa,
            ..input.params
        })
        .collect::<Vec<_>>();
    let base_hub_params = tire_inputs
        .iter()
        .map(|_| TireFemHubParams {
            dt: 1.0 / 600.0,
            bead_stiffness: 0.0,
            bead_damping: 0.0,
            hub_mass: 10.0,
            hub_damping: 0.0,
            gravity_y: 0.0,
            downforce: 0.0,
            external_force_xyz: [120.0, 0.0, 0.0],
            lateral_damping: 0.0,
            ..TireFemHubParams::default()
        })
        .collect::<Vec<_>>();

    let spin_input = TireVehicleWheelSpinGpuInput {
        wheel_state: vec![[24.0, 0.32, 1.2, 0.0], [18.0, 0.32, 1.1, 0.0]],
        wheel_commands: vec![[40.0, 0.0, 0.0, 0.0], [40.0, 0.0, 0.0, 0.0]],
        dt: 1.0 / 600.0,
    };
    let suspension_input = TireVehicleSuspensionGpuInput {
        suspension_geometry: vec![[-0.2, 0.32, 0.35, 0.315]; 2],
        suspension_params: vec![[25_000.0, 2_500.0, 0.0, 0.0]; 2],
        chassis_y: 0.40,
        chassis_vy: 0.0,
    };
    let brush_input = TireVehicleBrushRowsFromSpinGpuInput {
        chassis_pose: [0.0, 0.40, 0.0, 0.0],
        chassis_velocity: [0.5, 0.0, 0.0, 0.0],
        spin_state_rows: spin_input.wheel_state.clone(),
        wheel_mounts: vec![[0.0, -0.20, 0.0, 0.32], [4.0, -0.20, 0.0, 0.32]],
        wheel_controls: vec![[0.0, 0.20, 180_000.0, 0.0]; 2],
        tire_rows: vec![[48.0, 280_000.0, 25_000.0, 0.5]; 2],
        material_rows: vec![[0.9, 0.75, 1_200.0, 18.0]; 2],
        extra_rows: vec![[0.0, 0.0, 1.0, 0.0]; 2],
        suspension_rows: vec![[0.10, 0.0, 0.0, 1.0]; 2],
    };
    let planar_input = TireVehiclePlanarFromBrushGpuInput {
        wheel_offsets: vec![[0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0]],
        position_xz: [0.0, 0.0],
        velocity_xz: [0.0, 0.0],
        yaw: 0.0,
        yaw_rate: 0.0,
        mass: 900.0,
        yaw_inertia: 1_000.0,
        linear_damping: 0.0,
        yaw_damping: 0.0,
        dt: 1.0 / 600.0,
    };
    let vertical_input = TireVehicleVerticalGpuInput {
        y: 0.40,
        vy: 0.0,
        mass: 900.0,
        gravity: 9.81,
        vertical_damping: 0.0,
        dt: 1.0 / 600.0,
    };
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
        .expect("resident vehicle owner");

    let brush = fem_kernel
        .step_resident_vehicle_fem_atlas_from_brush_owner(
            &vehicle_kernel,
            &brush_kernel,
            &device,
            &queue,
            &mut owner,
            &mut resident,
            &fem_params,
            1.0 / 600.0,
            None,
            &base_hub_params,
            1.0,
        )
        .expect("resident vehicle/FEM atlas substep");
    assert_eq!(brush.wheel_count, tire_inputs.len());
    let hub_rows = fem_kernel
        .readback_resident_atlas_hub_rows(&device, &queue, &resident)
        .expect("resident hub rows");
    assert!(
        hub_rows[1][0] > 0.0 && hub_rows[3][0] > 0.0,
        "coherent substep should advance both hubs from base/brush force: {hub_rows:?}"
    );
    let planar = vehicle_kernel
        .readback_planar_resident_state(&device, &queue, &owner.planar)
        .expect("resident planar readback");
    assert!(
        planar.finite,
        "resident chassis state should stay finite: {planar:?}"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_resident_vehicle_fem_atlas_owner_steps_gpu() {
    let Some(engine) = try_engine() else {
        eprintln!("skipping resident vehicle/FEM atlas owner hardware dispatch: no wgpu adapter");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let fem_kernel = TireFemGpuKernel::new(&device);
    let vehicle_kernel = TireVehicleYawGpuKernel::new(&device);
    let brush_kernel = TireBrushGpuKernel::new(&device);

    let tire_inputs = vec![simple_tire_fem_input(0.0), simple_tire_fem_input(4.0)];
    let spin_input = TireVehicleWheelSpinGpuInput {
        wheel_state: vec![[24.0, 0.32, 1.2, 0.0], [18.0, 0.32, 1.1, 0.0]],
        wheel_commands: vec![[40.0, 0.0, 0.0, 0.0], [40.0, 0.0, 0.0, 0.0]],
        dt: 1.0 / 600.0,
    };
    let suspension_input = TireVehicleSuspensionGpuInput {
        suspension_geometry: vec![[-0.2, 0.32, 0.35, 0.315]; 2],
        suspension_params: vec![[25_000.0, 2_500.0, 0.0, 0.0]; 2],
        chassis_y: 0.40,
        chassis_vy: 0.0,
    };
    let brush_input = TireVehicleBrushRowsFromSpinGpuInput {
        chassis_pose: [0.0, 0.40, 0.0, 0.0],
        chassis_velocity: [0.5, 0.0, 0.0, 0.0],
        spin_state_rows: spin_input.wheel_state.clone(),
        wheel_mounts: vec![[0.0, -0.20, 0.0, 0.32], [4.0, -0.20, 0.0, 0.32]],
        wheel_controls: vec![[0.0, 0.20, 180_000.0, 0.0]; 2],
        tire_rows: vec![[48.0, 280_000.0, 25_000.0, 0.5]; 2],
        material_rows: vec![[0.9, 0.75, 1_200.0, 18.0]; 2],
        extra_rows: vec![[0.0, 0.0, 1.0, 0.0]; 2],
        suspension_rows: vec![[0.10, 0.0, 0.0, 1.0]; 2],
    };
    let planar_input = TireVehiclePlanarFromBrushGpuInput {
        wheel_offsets: vec![[0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0]],
        position_xz: [0.0, 0.0],
        velocity_xz: [0.0, 0.0],
        yaw: 0.0,
        yaw_rate: 0.0,
        mass: 900.0,
        yaw_inertia: 1_000.0,
        linear_damping: 0.0,
        yaw_damping: 0.0,
        dt: 1.0 / 600.0,
    };
    let vertical_input = TireVehicleVerticalGpuInput {
        y: 0.40,
        vy: 0.0,
        mass: 900.0,
        gravity: 9.81,
        vertical_damping: 0.0,
        dt: 1.0 / 600.0,
    };
    let mut owner = fem_kernel
        .create_resident_vehicle_fem_atlas_owner(
            &vehicle_kernel,
            &device,
            &queue,
            &tire_inputs,
            &spin_input,
            &suspension_input,
            brush_input,
            &planar_input,
            &vertical_input,
        )
        .expect("combined resident owner");
    assert_eq!(owner.tire_count(), tire_inputs.len());

    let fem_params = tire_inputs
        .iter()
        .map(|input| TireFemGpuParams {
            dt: 1.0 / 600.0,
            hub: input.params.hub,
            pressure_pa: input.params.pressure_pa,
            ..input.params
        })
        .collect::<Vec<_>>();
    let base_hub_params = tire_inputs
        .iter()
        .map(|_| TireFemHubParams {
            dt: 1.0 / 600.0,
            bead_stiffness: 0.0,
            bead_damping: 0.0,
            hub_mass: 10.0,
            hub_damping: 0.0,
            gravity_y: 0.0,
            downforce: 0.0,
            external_force_xyz: [120.0, 0.0, 0.0],
            lateral_damping: 0.0,
            ..TireFemHubParams::default()
        })
        .collect::<Vec<_>>();

    let brush = fem_kernel
        .step_resident_vehicle_fem_atlas_owner(
            &vehicle_kernel,
            &brush_kernel,
            &device,
            &queue,
            &mut owner,
            &fem_params,
            1.0 / 600.0,
            None,
            &base_hub_params,
            1.0,
        )
        .expect("combined resident owner step");
    assert_eq!(brush.wheel_count, tire_inputs.len());
    let hub_rows = fem_kernel
        .readback_resident_atlas_hub_rows(&device, &queue, owner.fem())
        .expect("owner resident hub rows");
    assert!(
        hub_rows[1][0] > 0.0 && hub_rows[3][0] > 0.0,
        "combined owner should advance both hubs from base/brush force: {hub_rows:?}"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_resident_atlas_hubs_step_from_contact_reduction_output_gpu() {
    let Some(engine) = try_engine() else {
        eprintln!(
            "skipping resident atlas contact-reduction-to-hub hardware dispatch: no wgpu adapter"
        );
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemGpuKernel::new(&device);
    let inputs = vec![simple_tire_fem_input(0.0), simple_tire_fem_input(4.0)];
    let resident = kernel
        .create_resident_atlas_batch(&device, &queue, &inputs)
        .expect("resident atlas");
    let base = vec![
        TireFemHubParams {
            dt: 1.0 / 60.0,
            bead_stiffness: 0.0,
            bead_damping: 0.0,
            hub_mass: 10.0,
            hub_damping: 0.0,
            gravity_y: 0.0,
            downforce: 0.0,
            external_force_xyz: [1.0, 2.0, 3.0],
            lateral_damping: 11.0,
            ..TireFemHubParams::default()
        },
        TireFemHubParams {
            dt: 1.0 / 60.0,
            bead_stiffness: 0.0,
            bead_damping: 0.0,
            hub_mass: 10.0,
            hub_damping: 0.0,
            gravity_y: 0.0,
            downforce: 0.0,
            external_force_xyz: [-4.0, 5.0, -6.0],
            lateral_damping: 17.0,
            ..TireFemHubParams::default()
        },
    ];
    let contact_rows: [[f32; 4]; 8] = [
        [0.0, 0.0, 0.0, 0.0],       // tire 0 diagnostics a
        [0.0, 0.0, 0.0, 0.0],       // tire 0 diagnostics b
        [100.0, 200.0, -50.0, 0.0], // tire 0 world_force + sinkage
        [0.0, 0.0, 0.0, 0.0],       // tire 0 world_moment + water
        [0.0, 0.0, 0.0, 0.0],       // tire 1 diagnostics a
        [0.0, 0.0, 0.0, 0.0],       // tire 1 diagnostics b
        [-20.0, 40.0, 80.0, 0.0],   // tire 1 world_force + sinkage
        [0.0, 0.0, 0.0, 0.0],       // tire 1 world_moment + water
    ];
    let contact_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_fem_gpu test contact reduction rows for hub step"),
        size: core::mem::size_of_val(&contact_rows) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&contact_buf, 0, bytemuck::cast_slice(&contact_rows));
    kernel
        .step_resident_atlas_hubs_from_contact_reduction_output(
            &device,
            &queue,
            &resident,
            &contact_buf,
            inputs.len(),
            &base,
            0.25,
        )
        .expect("resident contact-reduction-to-hub-param GPU update plus hub step");
    let param_rows = kernel
        .readback_resident_atlas_hub_param_rows(&device, &queue, &resident)
        .expect("read resident hub param rows after contact-reduction update");
    assert_eq!(param_rows[0].extf, [26.0, 52.0, -9.5, 11.0]);
    assert_eq!(param_rows[1].extf, [-9.0, 15.0, 14.0, 17.0]);
    let hub_rows = kernel
        .readback_resident_atlas_hub_rows(&device, &queue, &resident)
        .expect("read resident hub rows after contact-reduction-driven hub step");
    assert!(
        hub_rows[1][0] > 0.0,
        "first hub should gain positive x velocity from resident contact reduction force: {hub_rows:?}"
    );
    assert!(
        hub_rows[3][0] < 0.0,
        "second hub should gain negative x velocity from resident contact reduction force: {hub_rows:?}"
    );
}

#[test]
fn tire_fem_contact_triangle_exports_gpu_broadphase_sphere() {
    let tri = TireFemContactTriangle::new([-2.0, 0.0, 1.0], [4.0, 0.0, 1.0], [1.0, 0.0, 7.0], 0.8);
    let (center, radius) = tri.horizontal_bounding_sphere();
    assert!(
        (center[0] - 1.0).abs() < 1.0e-6 && (center[1] - 3.0).abs() < 1.0e-6,
        "triangle broadphase center should be the XZ centroid: {center:?}"
    );
    for p in [tri.a, tri.b, tri.c] {
        let dx = p[0] - center[0];
        let dz = p[2] - center[1];
        assert!(
            (dx * dx + dz * dz).sqrt() <= radius + 1.0e-6,
            "triangle broadphase sphere must enclose all vertices: center={center:?} radius={radius} p={p:?}"
        );
    }
}

#[test]
fn tire_fem_contact_triangle_exports_gpu_bvh_atlas_rows() {
    let mut triangles = Vec::new();
    for ix in 0..5 {
        for iz in 0..5 {
            let x = ix as f32 * 3.0;
            let z = iz as f32 * 3.0;
            triangles.push(TireFemContactTriangle::new(
                [x, 0.2, z],
                [x + 1.0, 0.2, z],
                [x, 0.2, z + 1.0],
                0.8,
            ));
        }
    }
    let (rows, node_count, index_rows) = tire_fem_contact_triangle_bvh_rows(&triangles);
    assert!(node_count > 1, "large triangle sets should build a BVH");
    assert_eq!(index_rows, (triangles.len() as u32).div_ceil(4));
    assert_eq!(
        rows.len(),
        node_count as usize * 3 + index_rows as usize,
        "BVH atlas rows should be node rows followed by packed index rows"
    );
    assert_eq!(rows[0][0], 0.0, "root min x should cover all triangles");
    assert_eq!(rows[0][2], 0.0, "root min z should cover all triangles");
    assert!(
        rows[0][3] >= 13.0 && rows[1][1] >= 13.0,
        "root max x/z should cover the sparse triangle grid: {:?} {:?}",
        rows[0],
        rows[1]
    );
    assert!(
        rows[1][2] < 0.5,
        "root should be an internal node for this grid"
    );
}

#[test]
fn tire_fem_contact_triangle_bvh_large_atlas_stress_keeps_packed_rows_valid() {
    let triangles = large_triangle_contact_terrain().triangles;
    let (rows, node_count, index_rows) = tire_fem_contact_triangle_bvh_rows(&triangles);
    assert_eq!(triangles.len(), 24 * 18 * 2);
    assert!(
        node_count > 128,
        "large mesh should build a deep BVH rather than a small linear atlas: nodes={node_count}"
    );
    assert_eq!(index_rows, (triangles.len() as u32).div_ceil(4));
    assert_eq!(rows.len(), node_count as usize * 3 + index_rows as usize);
    assert!(
        rows[0][0] <= -0.01 && rows[0][2] <= 0.0,
        "root min x/z should cover mesh origin and staggered triangle vertices: row0={:?}",
        rows[0]
    );
    assert!(
        rows[0][3] > 9.5 && rows[1][1] > 6.5,
        "root max x/z should cover the full triangle terrain: row0={:?} row1={:?}",
        rows[0],
        rows[1]
    );
    let mut leaf_index_total = 0usize;
    let mut leaf_count = 0usize;
    for node_id in 0..node_count as usize {
        let row0 = rows[node_id * 3];
        let row1 = rows[node_id * 3 + 1];
        let row2 = rows[node_id * 3 + 2];
        assert!(
            row0.iter().all(|v| v.is_finite())
                && row1.iter().all(|v| v.is_finite())
                && row2.iter().all(|v| v.is_finite()),
            "BVH node rows must stay finite: node={node_id} {row0:?} {row1:?} {row2:?}"
        );
        assert!(row0[0] <= row0[3], "min x <= max x for node {node_id}");
        assert!(row0[1] <= row1[0], "min y <= max y for node {node_id}");
        assert!(row0[2] <= row1[1], "min z <= max z for node {node_id}");
        if row1[2] > 0.5 {
            let start = row1[3] as usize;
            let count = row2[0] as usize;
            assert!(
                count > 0 && count <= 8,
                "leaf size must stay bounded: node={node_id} count={count}"
            );
            leaf_count += 1;
            leaf_index_total += count;
            assert!(
                start + count <= triangles.len(),
                "leaf index range must stay inside packed index rows: start={start} count={count}"
            );
        } else {
            let left = row1[3] as usize;
            let right = row2[0] as usize;
            assert!(left < node_count as usize && right < node_count as usize);
        }
    }
    assert!(leaf_count > 64, "stress atlas should have many leaves");
    assert_eq!(
        leaf_index_total,
        triangles.len(),
        "leaf ranges should account for every triangle exactly once"
    );
    let index_base = node_count as usize * 3;
    let mut seen = vec![false; triangles.len()];
    for i in 0..triangles.len() {
        let row = rows[index_base + i / 4];
        let tri_id = row[i % 4] as usize;
        assert!(
            tri_id < triangles.len(),
            "packed triangle index out of range: {tri_id}"
        );
        assert!(
            !seen[tri_id],
            "triangle index appears twice in packed atlas: {tri_id}"
        );
        seen[tri_id] = true;
    }
    assert!(seen.iter().all(|v| *v));
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_contact_reduction_large_triangle_bvh_dispatch_stress() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemContactReductionKernel::new(&device);
    let terrain = large_triangle_contact_terrain();
    let (_, node_count, index_rows) = tire_fem_contact_triangle_bvh_rows(&terrain.triangles);
    assert!(
        node_count > 128 && index_rows > 128,
        "large triangle hardware lane should exercise a deep packed atlas: nodes={node_count} index_rows={index_rows}"
    );

    let input = TireFemContactReductionInput {
        positions: vec![
            [0.20, -0.020, 0.17, 0.0],
            [4.82, -0.018, 3.13, 0.0],
            [8.18, -0.022, 5.72, 0.0],
            [9.86, -0.019, 6.46, 0.0],
        ],
        tread_ids: vec![0, 1, 2, 3],
        terrain: Some(terrain),
        params: TireFemContactReductionParams {
            hub: [5.0, 0.30, 3.0],
            heading: [1.0, 0.0, 0.0],
            contact_band: 0.05,
            ground_y: -10.0,
            downforce: 960.0,
        },
    };
    let out = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("large triangle-BVH contact reduction");
    assert_eq!(
        out.contact_count, 4,
        "all four far-apart tread nodes should contact finite triangles through the BVH atlas: {out:?}"
    );
    assert!(
        (out.normal_load - 960.0).abs() < 1.0e-3 && out.world_force[1] > 900.0,
        "triangle contacts should carry the requested downforce with upward world force: {out:?}"
    );
    assert!(
        out.mean_friction > 0.55 && out.mean_friction < 0.75,
        "mean friction should come from triangle rows, not the fallback heightfield friction: {out:?}"
    );
}

#[test]
fn tire_fem_contact_shaders_use_weighted_triangle_manifold_sampling() {
    let reduce = include_str!("../shaders/tire_fem_contact_reduce.wgsl");
    let feedback = include_str!("../shaders/tire_fem_contact_feedback.wgsl");
    for (name, shader) in [
        ("contact reduction", reduce),
        ("contact feedback", feedback),
    ] {
        assert!(
            shader.contains("var tri_weight_sum = 0.0;"),
            "{name} shader should accumulate all finite-triangle contacts inside the node band"
        );
        assert!(
            shader.contains("tri_normal_sum += candidate.normal * w;"),
            "{name} shader should blend contact normals instead of choosing one triangle seam normal"
        );
        assert!(
            shader.contains("tri_friction_sum += candidate.friction * w;"),
            "{name} shader should blend per-triangle friction across manifold patches"
        );
        assert!(
            shader.contains(
                "tri_min_distance = min(tri_min_distance, candidate.distance - tri_margin);"
            ),
            "{name} shader should preserve deepest manifold penetration for contact gating"
        );
        assert!(
            shader.contains("safe_normalize(tri_normal_sum, best.normal)"),
            "{name} shader should emit a normalized weighted manifold normal"
        );
    }
}

#[test]
fn tire_fem_contact_feedback_output_reports_patch_force_and_bristle_accounting() {
    let out = TireFemContactFeedbackOutput {
        velocities: vec![[0.0; 4]; 4],
        bristles: vec![
            [0.010, 0.000, 0.0, 0.0],
            [0.000, -0.020, 0.0, 0.0],
            [0.0; 4],
        ],
        forces: vec![[10.0, 50.0, 0.0, 0.0], [-4.0, 25.0, 3.0, 0.0], [0.0; 4]],
    };
    let accounting = out.patch_accounting();
    assert!(accounting.is_finite());
    assert_eq!(accounting.force_row_count, 2);
    assert_eq!(accounting.bristle_row_count, 2);
    assert!((accounting.total_force[0] - 6.0).abs() < 1.0e-6);
    assert!((accounting.total_force[1] - 75.0).abs() < 1.0e-6);
    assert!((accounting.total_force[2] - 3.0).abs() < 1.0e-6);
    assert!(accounting.mean_force_n > 25.0);
    assert!(accounting.max_force_n > accounting.mean_force_n);
    assert!((accounting.mean_bristle_deflection_m - 0.015).abs() < 1.0e-6);
    assert!((accounting.max_bristle_deflection_m - 0.020).abs() < 1.0e-6);
}

fn max_speed(output: &molla_kernels::TireFemGpuOutput) -> f32 {
    output
        .velocities
        .iter()
        .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
        .fold(0.0_f32, f32::max)
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_advances_one_tetra_without_nan() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemGpuKernel::new(&device);

    // One positively-oriented tetra. Gradients are the constant shape-function
    // gradients for x0=(0,0,0), x1=(1,0,0), x2=(0,1,0), x3=(0,0,1).
    let input = TireFemGpuInput {
        rest_positions: vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        positions: vec![
            [0.0, 0.05, 0.0, 0.0],
            [1.0, 0.05, 0.0, 0.0],
            [0.0, 1.05, 0.0, 0.0],
            [0.0, 0.05, 1.0, 0.0],
        ],
        velocities: vec![[0.0; 4]; 4],
        masses: vec![0.25; 4],
        tet_indices: vec![0, 1, 2, 3],
        tet_grads: vec![
            [-1.0, -1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        tet_volumes: vec![1.0 / 6.0],
        tet_mu: vec![3_400.0],
        tet_lambda: vec![23_000.0],
        tet_fiber_dir: vec![[0.0, 0.0, 0.0, 0.0]],
        tet_fiber_k1: vec![0.0],
        tet_fiber_k2: vec![0.0],
        tet_fiber_dir_b: vec![[0.0, 0.0, 0.0, 0.0]],
        tet_fiber_k1_b: vec![0.0],
        tet_fiber_k2_b: vec![0.0],
        tet_fiber_dir_c: vec![[0.0, 0.0, 0.0, 0.0]],
        tet_fiber_k1_c: vec![0.0],
        tet_fiber_k2_c: vec![0.0],
        tet_fiber_dir_d: vec![[0.0, 0.0, 0.0, 0.0]],
        tet_fiber_k1_d: vec![0.0],
        tet_fiber_k2_d: vec![0.0],
        // CSR adjacency for a single tet across 4 nodes: each node has
        // exactly one entry packing `(tet_index << 2) | local_vertex_id`.
        node_tet_offsets: vec![0, 1, 2, 3, 4],
        node_tet_entries: vec![0, 1, 2, 3],
        xpbd_edge_indices: vec![],
        xpbd_edge_params: vec![],
        node_edge_offsets: vec![],
        node_edge_entries: vec![],
        xpbd_dihedral_indices: vec![],
        xpbd_dihedral_params: vec![],
        node_dihedral_offsets: vec![],
        node_dihedral_entries: vec![],
        bead_ids: vec![],
        bead_offsets: vec![],
        tread_ids: vec![2],
        params: TireFemGpuParams {
            dt: 1.0 / 600.0,
            young_modulus: 1.0e4,
            pressure_pa: 1.0e3,
            pressure_area_per_tread_node: 0.01,
            hub: [0.0, 0.25, 0.0],
            ..Default::default()
        },
    };

    let out = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("tire FEM GPU dispatch");
    assert_eq!(out.positions.len(), 4);
    assert_eq!(out.velocities.len(), 4);
    assert_eq!(out.forces.len(), 4);
    for p in &out.positions {
        assert!(
            p.iter().all(|x| x.is_finite()),
            "non-finite position: {p:?}"
        );
        assert!(p[1] >= 0.0, "floor contact should clamp y >= 0: {p:?}");
    }
}

/// Stage (f2) — GPU parity for the CPU diagonal semi-implicit stiffness
/// stabilizer. With the same deformed stiff tetra and no gravity/contact, the
/// semi-implicit denominator should damp the elastic velocity spike.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_semi_implicit_stiffness_damps_velocity_spike() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemGpuKernel::new(&device);

    let make_input = |semi_implicit_stiffness_scale: f32| {
        let mut input = simple_tire_fem_input(0.0);
        input.positions[1][0] += 0.35;
        input.positions[2][1] += 0.20;
        input.tet_mu[0] = 2.0e6;
        input.tet_lambda[0] = 8.0e6;
        input.params.dt = 1.0 / 60.0;
        input.params.gravity_y = 0.0;
        input.params.node_damping = 0.0;
        input.params.pressure_pa = 0.0;
        input.params.pressure_area_per_tread_node = 0.0;
        input.params.cavity_radial_force_per_bead = 0.0;
        input.params.xpbd_volume_compliance = 1.0e30;
        input.params.xpbd_bead_compliance = 1.0e30;
        input.params.semi_implicit_stiffness_scale = semi_implicit_stiffness_scale;
        input
    };

    let explicit = kernel
        .dispatch_blocking(&device, &queue, &make_input(0.0))
        .expect("explicit tire FEM GPU dispatch");
    let semi = kernel
        .dispatch_blocking(&device, &queue, &make_input(50.0))
        .expect("semi-implicit tire FEM GPU dispatch");
    let explicit_speed = max_speed(&explicit);
    let semi_speed = max_speed(&semi);
    assert!(
        semi_speed < explicit_speed * 0.55,
        "semi-implicit GPU stiffness should damp stiff-rubber velocity spike: explicit={explicit_speed} semi={semi_speed}"
    );
}

/// Stage (c3) — the extra third/fourth ply-family buffers are not just ABI
/// padding: nonzero cap/radial fibre families should change the GPU force.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_third_fourth_ply_families_change_force() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemGpuKernel::new(&device);

    let mut base = simple_tire_fem_input(0.0);
    base.positions[1][0] = 1.30;
    base.positions[2][1] = 1.25;
    base.tet_mu[0] = 0.0;
    base.tet_lambda[0] = 0.0;
    base.params.gravity_y = 0.0;
    base.params.pressure_pa = 0.0;
    base.params.pressure_area_per_tread_node = 0.0;
    base.params.cavity_radial_force_per_bead = 0.0;
    base.params.xpbd_volume_compliance = 1.0e30;
    base.params.xpbd_bead_compliance = 1.0e30;

    let mut plied = base.clone();
    plied.tet_fiber_dir_c[0] = [1.0, 0.0, 0.0, 0.0];
    plied.tet_fiber_k1_c[0] = 12_000.0;
    plied.tet_fiber_k2_c[0] = 0.0;
    plied.tet_fiber_dir_d[0] = [0.0, 1.0, 0.0, 0.0];
    plied.tet_fiber_k1_d[0] = 8_000.0;
    plied.tet_fiber_k2_d[0] = 0.0;

    let base_out = kernel
        .dispatch_blocking(&device, &queue, &base)
        .expect("base tire FEM GPU dispatch");
    let plied_out = kernel
        .dispatch_blocking(&device, &queue, &plied)
        .expect("plied tire FEM GPU dispatch");
    let force_norm = |forces: &[[f32; 4]]| {
        forces
            .iter()
            .map(|f| (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt())
            .fold(0.0_f32, f32::max)
    };
    let base_force = force_norm(&base_out.forces);
    let plied_force = force_norm(&plied_out.forces);
    assert!(
        plied_force > base_force + 100.0,
        "third/fourth ply families should contribute GPU elastic force: base={base_force} plied={plied_force}"
    );
}

/// Stage (g) — persistent buffer path runs multiple substeps without
/// re-uploading mesh data and produces the same kind of finite output.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_persistent_runs_multiple_substeps() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemGpuKernel::new(&device);
    let input = TireFemGpuInput {
        rest_positions: vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        positions: vec![
            [0.0, 0.05, 0.0, 0.0],
            [1.0, 0.05, 0.0, 0.0],
            [0.0, 1.05, 0.0, 0.0],
            [0.0, 0.05, 1.0, 0.0],
        ],
        velocities: vec![[0.0; 4]; 4],
        masses: vec![0.25; 4],
        tet_indices: vec![0, 1, 2, 3],
        tet_grads: vec![
            [-1.0, -1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        tet_volumes: vec![1.0 / 6.0],
        tet_mu: vec![3_400.0],
        tet_lambda: vec![23_000.0],
        tet_fiber_dir: vec![[0.0, 0.0, 0.0, 0.0]],
        tet_fiber_k1: vec![0.0],
        tet_fiber_k2: vec![0.0],
        tet_fiber_dir_b: vec![[0.0, 0.0, 0.0, 0.0]],
        tet_fiber_k1_b: vec![0.0],
        tet_fiber_k2_b: vec![0.0],
        tet_fiber_dir_c: vec![[0.0, 0.0, 0.0, 0.0]],
        tet_fiber_k1_c: vec![0.0],
        tet_fiber_k2_c: vec![0.0],
        tet_fiber_dir_d: vec![[0.0, 0.0, 0.0, 0.0]],
        tet_fiber_k1_d: vec![0.0],
        tet_fiber_k2_d: vec![0.0],
        node_tet_offsets: vec![0, 1, 2, 3, 4],
        node_tet_entries: vec![0, 1, 2, 3],
        xpbd_edge_indices: vec![],
        xpbd_edge_params: vec![],
        node_edge_offsets: vec![],
        node_edge_entries: vec![],
        xpbd_dihedral_indices: vec![],
        xpbd_dihedral_params: vec![],
        node_dihedral_offsets: vec![],
        node_dihedral_entries: vec![],
        bead_ids: vec![],
        bead_offsets: vec![],
        tread_ids: vec![2],
        params: TireFemGpuParams {
            dt: 1.0 / 600.0,
            young_modulus: 1.0e4,
            pressure_pa: 1.0e3,
            pressure_area_per_tread_node: 0.01,
            hub: [0.0, 0.25, 0.0],
            ..Default::default()
        },
    };
    let mut mesh = kernel
        .create_persistent(&device, &queue, &input)
        .expect("create_persistent");
    let params = input.params;
    // Run several substeps without intermediate readback.
    for _ in 0..8 {
        kernel.step_persistent(&device, &queue, &mut mesh, params);
    }
    let out = kernel
        .readback_state(&device, &queue, &mesh)
        .expect("readback_state");
    assert_eq!(out.positions.len(), 4);
    for p in &out.positions {
        assert!(
            p.iter().all(|x| x.is_finite()),
            "non-finite position after 8 persistent substeps: {p:?}"
        );
        assert!(p[1] >= 0.0, "y >= 0 after persistent substeps: {p:?}");
    }
}

/// Stage (g2) — persistent batch path keeps multiple tire meshes resident and
/// advances them through one reusable batch API.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_persistent_batch_steps_multiple_tires() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemGpuKernel::new(&device);

    let inputs = vec![simple_tire_fem_input(0.0), simple_tire_fem_input(3.0)];
    let params: Vec<_> = inputs.iter().map(|input| input.params).collect();
    let mut batch = kernel
        .create_persistent_batch(&device, &queue, &inputs)
        .expect("create_persistent_batch");
    assert_eq!(batch.len(), 2);
    assert_eq!(batch.total_node_count(), 8);

    for _ in 0..6 {
        kernel
            .step_persistent_batch(&device, &queue, &mut batch, &params)
            .expect("batch step");
    }
    let out = kernel
        .readback_batch(&device, &queue, &batch)
        .expect("batch readback");
    assert_eq!(out.len(), 2);
    for (tire_idx, tire) in out.iter().enumerate() {
        assert_eq!(tire.positions.len(), 4);
        for p in &tire.positions {
            assert!(
                p.iter().all(|x| x.is_finite()),
                "non-finite batched tire {tire_idx} position: {p:?}"
            );
            assert!(
                p[1] >= 0.0,
                "batched tire {tire_idx} floor projection failed: {p:?}"
            );
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_xpbd_volume_projection_reduces_compression() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemGpuKernel::new(&device);

    let positions = vec![
        [0.0, 0.05, 0.0, 0.0],
        [1.0, 0.05, 0.0, 0.0],
        [0.0, 1.05, 0.0, 0.0],
        [0.0, 0.05, 0.45, 0.0],
    ];
    let input_volume = tet_volume(&positions, [0, 1, 2, 3]);
    let rest_volume = 1.0 / 6.0;
    assert!(input_volume < rest_volume * 0.5);

    let input = TireFemGpuInput {
        rest_positions: vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        positions,
        velocities: vec![[0.0; 4]; 4],
        masses: vec![0.25; 4],
        tet_indices: vec![0, 1, 2, 3],
        tet_grads: vec![
            [-1.0, -1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        tet_volumes: vec![rest_volume],
        tet_mu: vec![0.0],
        tet_lambda: vec![0.0],
        tet_fiber_dir: vec![[0.0; 4]],
        tet_fiber_k1: vec![0.0],
        tet_fiber_k2: vec![0.0],
        tet_fiber_dir_b: vec![[0.0; 4]],
        tet_fiber_k1_b: vec![0.0],
        tet_fiber_k2_b: vec![0.0],
        tet_fiber_dir_c: vec![[0.0; 4]],
        tet_fiber_k1_c: vec![0.0],
        tet_fiber_k2_c: vec![0.0],
        tet_fiber_dir_d: vec![[0.0; 4]],
        tet_fiber_k1_d: vec![0.0],
        tet_fiber_k2_d: vec![0.0],
        node_tet_offsets: vec![0, 1, 2, 3, 4],
        node_tet_entries: vec![0, 1, 2, 3],
        xpbd_edge_indices: vec![],
        xpbd_edge_params: vec![],
        node_edge_offsets: vec![],
        node_edge_entries: vec![],
        xpbd_dihedral_indices: vec![],
        xpbd_dihedral_params: vec![],
        node_dihedral_offsets: vec![],
        node_dihedral_entries: vec![],
        bead_ids: vec![],
        bead_offsets: vec![],
        tread_ids: vec![],
        params: TireFemGpuParams {
            dt: 1.0 / 240.0,
            gravity_y: 0.0,
            node_damping: 0.0,
            pressure_pa: 0.0,
            pressure_area_per_tread_node: 0.0,
            xpbd_volume_compliance: 0.0,
            ..Default::default()
        },
    };

    let out = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("tire FEM GPU XPBD volume dispatch");
    let output_volume = tet_volume(&out.positions, [0, 1, 2, 3]);
    assert!(
        (rest_volume - output_volume).abs() < (rest_volume - input_volume).abs() * 0.75,
        "GPU XPBD volume projection should reduce compression: input={input_volume} output={output_volume} rest={rest_volume}"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_xpbd_bead_projection_pulls_node_toward_rim_mount() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemGpuKernel::new(&device);

    let target = [0.0_f32, 0.12, 0.0];
    let positions = vec![
        [0.30, 0.22, 0.08, 0.0],
        [1.0, 0.22, 0.0, 0.0],
        [0.0, 1.22, 0.0, 0.0],
        [0.0, 0.22, 1.0, 0.0],
    ];
    let dist = |p: [f32; 4]| -> f32 {
        let dx = p[0] - target[0];
        let dy = p[1] - target[1];
        let dz = p[2] - target[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    };
    let before = dist(positions[0]);

    let input = TireFemGpuInput {
        rest_positions: vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        positions,
        velocities: vec![[0.0; 4]; 4],
        masses: vec![0.25; 4],
        tet_indices: vec![0, 1, 2, 3],
        tet_grads: vec![
            [-1.0, -1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        tet_volumes: vec![1.0 / 6.0],
        tet_mu: vec![0.0],
        tet_lambda: vec![0.0],
        tet_fiber_dir: vec![[0.0; 4]],
        tet_fiber_k1: vec![0.0],
        tet_fiber_k2: vec![0.0],
        tet_fiber_dir_b: vec![[0.0; 4]],
        tet_fiber_k1_b: vec![0.0],
        tet_fiber_k2_b: vec![0.0],
        tet_fiber_dir_c: vec![[0.0; 4]],
        tet_fiber_k1_c: vec![0.0],
        tet_fiber_k2_c: vec![0.0],
        tet_fiber_dir_d: vec![[0.0; 4]],
        tet_fiber_k1_d: vec![0.0],
        tet_fiber_k2_d: vec![0.0],
        node_tet_offsets: vec![0, 1, 2, 3, 4],
        node_tet_entries: vec![0, 1, 2, 3],
        xpbd_edge_indices: vec![],
        xpbd_edge_params: vec![],
        node_edge_offsets: vec![],
        node_edge_entries: vec![],
        xpbd_dihedral_indices: vec![],
        xpbd_dihedral_params: vec![],
        node_dihedral_offsets: vec![],
        node_dihedral_entries: vec![],
        bead_ids: vec![0],
        bead_offsets: vec![[target[0], target[1], target[2], 0.0]],
        tread_ids: vec![],
        params: TireFemGpuParams {
            dt: 1.0 / 240.0,
            gravity_y: 0.0,
            node_damping: 0.0,
            bead_stiffness: 0.0,
            pressure_pa: 0.0,
            pressure_area_per_tread_node: 0.0,
            hub: [0.0, 0.0, 0.0],
            xpbd_volume_compliance: 1.0e30,
            xpbd_bead_compliance: 0.0,
            ..Default::default()
        },
    };

    let out = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("tire FEM GPU XPBD bead dispatch");
    let after = dist(out.positions[0]);
    assert!(
        after < before * 0.25,
        "GPU XPBD bead projection should pull bead node toward rim mount: before={before} after={after}"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_xpbd_structural_edge_reduces_stretch() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemGpuKernel::new(&device);

    let positions = vec![
        [0.0, 0.20, 0.0, 0.0],
        [1.5, 0.20, 0.0, 0.0],
        [0.0, 1.20, 0.0, 0.0],
        [0.0, 0.20, 1.0, 0.0],
    ];
    let edge_len = |p: &[[f32; 4]]| -> f32 {
        let dx = p[1][0] - p[0][0];
        let dy = p[1][1] - p[0][1];
        let dz = p[1][2] - p[0][2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    };
    let before = edge_len(&positions);
    let rest_len = 1.0_f32;

    let input = TireFemGpuInput {
        rest_positions: vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        positions,
        velocities: vec![[0.0; 4]; 4],
        masses: vec![0.25; 4],
        tet_indices: vec![0, 1, 2, 3],
        tet_grads: vec![
            [-1.0, -1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        tet_volumes: vec![1.0 / 6.0],
        tet_mu: vec![0.0],
        tet_lambda: vec![0.0],
        tet_fiber_dir: vec![[0.0; 4]],
        tet_fiber_k1: vec![0.0],
        tet_fiber_k2: vec![0.0],
        tet_fiber_dir_b: vec![[0.0; 4]],
        tet_fiber_k1_b: vec![0.0],
        tet_fiber_k2_b: vec![0.0],
        tet_fiber_dir_c: vec![[0.0; 4]],
        tet_fiber_k1_c: vec![0.0],
        tet_fiber_k2_c: vec![0.0],
        tet_fiber_dir_d: vec![[0.0; 4]],
        tet_fiber_k1_d: vec![0.0],
        tet_fiber_k2_d: vec![0.0],
        node_tet_offsets: vec![0, 1, 2, 3, 4],
        node_tet_entries: vec![0, 1, 2, 3],
        xpbd_edge_indices: vec![[0, 1, 0, 0]],
        xpbd_edge_params: vec![[rest_len, 0.0, 0.0, 0.0]],
        node_edge_offsets: vec![0, 1, 2, 2, 2],
        node_edge_entries: vec![0, 1],
        xpbd_dihedral_indices: vec![],
        xpbd_dihedral_params: vec![],
        node_dihedral_offsets: vec![],
        node_dihedral_entries: vec![],
        bead_ids: vec![],
        bead_offsets: vec![],
        tread_ids: vec![],
        params: TireFemGpuParams {
            dt: 1.0 / 240.0,
            gravity_y: 0.0,
            node_damping: 0.0,
            pressure_pa: 0.0,
            pressure_area_per_tread_node: 0.0,
            xpbd_volume_compliance: 1.0e30,
            xpbd_bead_compliance: 1.0e30,
            ..Default::default()
        },
    };

    let out = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("tire FEM GPU XPBD structural edge dispatch");
    let after = edge_len(&out.positions);
    assert!(
        (after - rest_len).abs() < (before - rest_len).abs() * 0.75,
        "GPU XPBD structural edge should reduce stretch: before={before} after={after} rest={rest_len}"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_sidewall_dihedral_projection_reduces_fold_angle() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemGpuKernel::new(&device);

    let rest = vec![
        [0.0, 0.20, 0.0, 0.0],
        [1.0, 0.20, 0.0, 0.0],
        [0.0, 1.20, 0.0, 0.0],
        [0.0, 0.20, 1.0, 0.0],
    ];
    let mut positions = rest.clone();
    positions[3][0] += 0.35;
    positions[3][1] += 0.20;
    positions[3][2] += 0.45;
    let rest_angle = quad_dihedral_angle(&rest, [0, 1, 2, 3]);
    let before = (quad_dihedral_angle(&positions, [0, 1, 2, 3]) - rest_angle).abs();

    let input = TireFemGpuInput {
        rest_positions: rest,
        positions,
        velocities: vec![[0.0; 4]; 4],
        masses: vec![0.25; 4],
        tet_indices: vec![0, 1, 2, 3],
        tet_grads: vec![
            [-1.0, -1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        tet_volumes: vec![1.0 / 6.0],
        tet_mu: vec![0.0],
        tet_lambda: vec![0.0],
        tet_fiber_dir: vec![[0.0; 4]],
        tet_fiber_k1: vec![0.0],
        tet_fiber_k2: vec![0.0],
        tet_fiber_dir_b: vec![[0.0; 4]],
        tet_fiber_k1_b: vec![0.0],
        tet_fiber_k2_b: vec![0.0],
        tet_fiber_dir_c: vec![[0.0; 4]],
        tet_fiber_k1_c: vec![0.0],
        tet_fiber_k2_c: vec![0.0],
        tet_fiber_dir_d: vec![[0.0; 4]],
        tet_fiber_k1_d: vec![0.0],
        tet_fiber_k2_d: vec![0.0],
        node_tet_offsets: vec![0, 1, 2, 3, 4],
        node_tet_entries: vec![0, 1, 2, 3],
        xpbd_edge_indices: vec![],
        xpbd_edge_params: vec![],
        node_edge_offsets: vec![],
        node_edge_entries: vec![],
        xpbd_dihedral_indices: vec![[0, 1, 2, 3]],
        xpbd_dihedral_params: vec![[rest_angle, 0.0, 0.0, 0.0]],
        node_dihedral_offsets: vec![0, 1, 2, 3, 4],
        node_dihedral_entries: vec![0, 1, 2, 3],
        bead_ids: vec![],
        bead_offsets: vec![],
        tread_ids: vec![],
        params: TireFemGpuParams {
            dt: 1.0 / 240.0,
            gravity_y: 0.0,
            node_damping: 0.0,
            pressure_pa: 0.0,
            pressure_area_per_tread_node: 0.0,
            xpbd_volume_compliance: 1.0e30,
            xpbd_bead_compliance: 1.0e30,
            ..Default::default()
        },
    };

    let out = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("tire FEM GPU sidewall dihedral XPBD dispatch");
    let after = (quad_dihedral_angle(&out.positions, [0, 1, 2, 3]) - rest_angle).abs();
    assert!(
        after < before * 0.90,
        "GPU sidewall dihedral XPBD should reduce fold-angle error: before={before} after={after}"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_sidewall_ply_strain_projection_reduces_overstretch() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemGpuKernel::new(&device);

    let rest = vec![
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
    ];
    let positions = vec![
        [1.0, 0.0, 0.0, 0.0],
        [2.85, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
    ];
    let before = simple_tet_fiber_stretch(&positions, [1.0, 0.0, 0.0]);
    assert!(
        before > 1.5,
        "fixture should over-stretch radial ply: {before}"
    );

    let input = TireFemGpuInput {
        rest_positions: rest,
        positions,
        velocities: vec![[0.0; 4]; 4],
        masses: vec![0.25; 4],
        tet_indices: vec![0, 1, 2, 3],
        tet_grads: vec![
            [-1.0, -1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        tet_volumes: vec![1.0 / 6.0],
        tet_mu: vec![0.0],
        tet_lambda: vec![0.0],
        tet_fiber_dir: vec![[1.0, 0.18, 0.0, 0.0]],
        tet_fiber_k1: vec![1.0],
        tet_fiber_k2: vec![0.0],
        tet_fiber_dir_b: vec![[1.0, -0.18, 0.0, 0.0]],
        tet_fiber_k1_b: vec![1.0],
        tet_fiber_k2_b: vec![0.0],
        tet_fiber_dir_c: vec![[1.0, 0.0, 0.0, 0.0]],
        tet_fiber_k1_c: vec![1.0],
        tet_fiber_k2_c: vec![0.0],
        tet_fiber_dir_d: vec![[0.0, 0.0, 1.0, 0.0]],
        tet_fiber_k1_d: vec![1.0],
        tet_fiber_k2_d: vec![0.0],
        node_tet_offsets: vec![0, 1, 2, 3, 4],
        node_tet_entries: vec![0, 1, 2, 3],
        xpbd_edge_indices: vec![],
        xpbd_edge_params: vec![],
        node_edge_offsets: vec![],
        node_edge_entries: vec![],
        xpbd_dihedral_indices: vec![],
        xpbd_dihedral_params: vec![],
        node_dihedral_offsets: vec![],
        node_dihedral_entries: vec![],
        bead_ids: vec![],
        bead_offsets: vec![],
        tread_ids: vec![],
        params: TireFemGpuParams {
            dt: 1.0 / 240.0,
            gravity_y: 0.0,
            node_damping: 0.0,
            pressure_pa: 0.0,
            pressure_area_per_tread_node: 0.0,
            xpbd_volume_compliance: 1.0e30,
            xpbd_bead_compliance: 1.0e30,
            xpbd_sidewall_ply_stretch_limit: 1.15,
            xpbd_sidewall_ply_compliance: 0.0,
            ..Default::default()
        },
    };

    let out = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("tire FEM GPU sidewall ply-strain XPBD dispatch");
    let after = simple_tet_fiber_stretch(&out.positions, [1.0, 0.0, 0.0]);
    assert!(
        after < before * 0.90,
        "GPU sidewall ply-strain XPBD should reduce tensile ply over-stretch: before={before} after={after}"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_contact_feedback_updates_bristles_and_node_velocity() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemContactFeedbackKernel::new(&device);

    let input = TireFemContactFeedbackInput {
        positions: vec![
            // Tread node inside the contact band.
            [0.0, 0.010, 0.0, 0.0],
            // Non-tread node should pass through unchanged.
            [0.0, 0.300, 0.0, 0.0],
        ],
        velocities: vec![[0.0; 4], [0.25, 0.0, 0.0, 0.0]],
        masses: vec![2.0, 2.0],
        tread_ids: vec![0],
        bristles: vec![[0.0; 4]],
        terrain: None,
        params: TireFemContactFeedbackParams {
            dt: 1.0 / 120.0,
            contact_band: 0.05,
            ground_y: 0.0,
            omega: 16.0,
            rolling_radius: 0.30,
            heading: [1.0, 0.0, 0.0],
            normal_load_per_tread: 800.0,
            area_per_tread: 0.01,
            tread_shear_stiffness: 2.0e5,
            mu_static: 1.0,
            mu_dynamic: 0.8,
            bristle_damping: 2.0e3,
            ..Default::default()
        },
    };

    let out = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("tire FEM GPU contact feedback");
    assert_eq!(out.velocities.len(), 2);
    assert_eq!(out.bristles.len(), 1);
    assert_eq!(out.forces.len(), 2);
    assert!(
        out.bristles[0][0] > 0.0,
        "drive slip should write persistent longitudinal bristle: {:?}",
        out.bristles[0]
    );
    assert!(
        out.forces[0][0] > 0.0 && out.forces[0][1] > 0.0,
        "contact feedback should produce forward + normal force: {:?}",
        out.forces[0]
    );
    assert!(
        out.velocities[0][0] > input.velocities[0][0],
        "contact impulse should accelerate tread node forward: before={:?} after={:?}",
        input.velocities[0],
        out.velocities[0]
    );
    assert!(
        (out.velocities[1][0] - input.velocities[1][0]).abs() < 1.0e-6,
        "non-tread node velocity should pass through unchanged"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_contact_feedback_samples_heightfield_friction_and_normals() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemContactFeedbackKernel::new(&device);

    let base = TireFemContactFeedbackInput {
        positions: vec![
            // Low-friction cell at x=0.
            [0.0, 0.010, 0.0, 0.0],
            // High-friction cell at x=1.
            [1.0, 0.010, 0.0, 0.0],
        ],
        velocities: vec![[0.0; 4]; 2],
        masses: vec![2.0; 2],
        tread_ids: vec![0, 1],
        bristles: vec![[0.0; 4]; 2],
        terrain: Some(TireFemContactTerrain {
            origin: [0.0, 0.0],
            spacing: 1.0,
            nx: 2,
            nz: 2,
            heights: vec![0.0, 0.0, 0.0, 0.0],
            friction: 1.0,
            friction_map: Some(vec![0.2, 1.0, 0.2, 1.0]),
            water_film_map: None,
            sinkage_map: None,
            soil_rolling_resistance_scale: 1.0,
            surface_velocity: [0.0; 3],
            surface_velocity_map: None,
            sdf_primitives: Vec::new(),
            triangles: Vec::new(),
        }),
        params: TireFemContactFeedbackParams {
            dt: 1.0 / 120.0,
            contact_band: 0.05,
            omega: 40.0,
            rolling_radius: 0.30,
            heading: [1.0, 0.0, 0.0],
            normal_load_per_tread: 800.0,
            area_per_tread: 0.01,
            tread_shear_stiffness: 4.0e5,
            mu_static: 1.0,
            mu_dynamic: 0.8,
            bristle_damping: 5.0e3,
            ..Default::default()
        },
    };
    let out = kernel
        .dispatch_blocking(&device, &queue, &base)
        .expect("heightfield/friction feedback");
    assert!(
        out.forces[1][0] > out.forces[0][0] * 2.0,
        "friction map should allow more longitudinal force on high-mu cell: low={:?} high={:?}",
        out.forces[0],
        out.forces[1]
    );

    let sloped = TireFemContactFeedbackInput {
        positions: vec![[1.0, 0.110, 0.0, 0.0]],
        velocities: vec![[0.0; 4]],
        masses: vec![2.0],
        tread_ids: vec![0],
        bristles: vec![[0.0; 4]],
        terrain: Some(TireFemContactTerrain {
            origin: [0.0, 0.0],
            spacing: 1.0,
            nx: 2,
            nz: 2,
            heights: vec![0.0, 0.1, 0.0, 0.1],
            friction: 1.0,
            friction_map: None,
            water_film_map: None,
            sinkage_map: None,
            soil_rolling_resistance_scale: 1.0,
            surface_velocity: [0.0; 3],
            surface_velocity_map: None,
            sdf_primitives: Vec::new(),
            triangles: Vec::new(),
        }),
        params: TireFemContactFeedbackParams {
            dt: 1.0 / 120.0,
            contact_band: 0.05,
            omega: 0.0,
            heading: [1.0, 0.0, 0.0],
            normal_load_per_tread: 800.0,
            area_per_tread: 0.01,
            ..Default::default()
        },
    };
    let slope_out = kernel
        .dispatch_blocking(&device, &queue, &sloped)
        .expect("sloped heightfield feedback");
    assert!(
        slope_out.forces[0][0] < -1.0 && slope_out.forces[0][1] > 100.0,
        "heightfield normal should tilt contact force on slope: {:?}",
        slope_out.forces[0]
    );

    let dry_film = TireFemContactFeedbackInput {
        positions: vec![[0.0, 0.010, 0.0, 0.0]],
        velocities: vec![[0.0; 4]],
        masses: vec![2.0],
        tread_ids: vec![0],
        bristles: vec![[0.0; 4]],
        terrain: Some(TireFemContactTerrain {
            origin: [0.0, 0.0],
            spacing: 1.0,
            nx: 2,
            nz: 2,
            heights: vec![0.0; 4],
            friction: 1.0,
            friction_map: None,
            water_film_map: None,
            sinkage_map: None,
            soil_rolling_resistance_scale: 1.0,
            surface_velocity: [0.0; 3],
            surface_velocity_map: None,
            sdf_primitives: Vec::new(),
            triangles: Vec::new(),
        }),
        params: TireFemContactFeedbackParams {
            dt: 1.0 / 120.0,
            contact_band: 0.05,
            omega: 40.0,
            rolling_radius: 0.30,
            heading: [1.0, 0.0, 0.0],
            normal_load_per_tread: 800.0,
            area_per_tread: 0.01,
            tread_shear_stiffness: 4.0e5,
            mu_static: 1.0,
            mu_dynamic: 0.8,
            bristle_damping: 5.0e3,
            ..Default::default()
        },
    };
    let mut wet_film = dry_film.clone();
    wet_film.terrain.as_mut().unwrap().water_film_map = Some(vec![0.004; 4]);
    let dry_film_out = kernel
        .dispatch_blocking(&device, &queue, &dry_film)
        .expect("dry-film feedback");
    let wet_film_out = kernel
        .dispatch_blocking(&device, &queue, &wet_film)
        .expect("wet-film feedback");
    assert!(
        wet_film_out.forces[0][0] < dry_film_out.forces[0][0] * 0.85,
        "water-film map should reduce GPU FEM feedback shear: dry={:?} wet={:?}",
        dry_film_out.forces[0],
        wet_film_out.forces[0]
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_contact_feedback_samples_sdf_side_normal() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemContactFeedbackKernel::new(&device);

    let input = TireFemContactFeedbackInput {
        positions: vec![[0.34, 0.30, 0.0, 0.0]],
        velocities: vec![[0.0; 4]],
        masses: vec![4.0],
        tread_ids: vec![0],
        bristles: vec![[0.0; 4]],
        terrain: Some(TireFemContactTerrain {
            origin: [0.0, 0.0],
            spacing: 1.0,
            nx: 1,
            nz: 1,
            heights: vec![0.0],
            friction: 1.0,
            friction_map: None,
            water_film_map: None,
            sinkage_map: None,
            soil_rolling_resistance_scale: 1.0,
            surface_velocity: [0.0; 3],
            surface_velocity_map: None,
            sdf_primitives: vec![TireFemContactSdfPrimitive::sphere(
                [0.0, 0.30, 0.0],
                0.35,
                0.55,
            )],
            triangles: Vec::new(),
        }),
        params: TireFemContactFeedbackParams {
            dt: 1.0 / 240.0,
            contact_band: 0.05,
            omega: 0.0,
            heading: [1.0, 0.0, 0.0],
            normal_load_per_tread: 700.0,
            area_per_tread: 0.01,
            ..Default::default()
        },
    };

    let out = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("sdf side contact feedback");
    assert!(
        out.forces[0][0] > 500.0,
        "sphere SDF side contact should push along +x: {:?}",
        out.forces[0]
    );
    assert!(
        out.forces[0][1].abs() < 5.0,
        "sphere side normal should not behave like flat ground: {:?}",
        out.forces[0]
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_contact_feedback_samples_triangle_mesh_patch() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemContactFeedbackKernel::new(&device);

    let input = TireFemContactFeedbackInput {
        positions: vec![[0.52, 0.45, -0.02, 0.0]],
        velocities: vec![[0.0; 4]],
        masses: vec![4.0],
        tread_ids: vec![0],
        bristles: vec![[0.0; 4]],
        terrain: Some(TireFemContactTerrain {
            origin: [0.0, 0.0],
            spacing: 1.0,
            nx: 1,
            nz: 1,
            heights: vec![0.0],
            friction: 1.0,
            friction_map: None,
            water_film_map: None,
            sinkage_map: None,
            soil_rolling_resistance_scale: 1.0,
            surface_velocity: [0.0; 3],
            surface_velocity_map: None,
            sdf_primitives: Vec::new(),
            triangles: vec![
                TireFemContactTriangle::new(
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    0.42,
                ),
                TireFemContactTriangle::new(
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0],
                    0.42,
                ),
            ],
        }),
        params: TireFemContactFeedbackParams {
            dt: 1.0 / 240.0,
            contact_band: 0.10,
            omega: 0.0,
            heading: [1.0, 0.0, 0.0],
            normal_load_per_tread: 700.0,
            area_per_tread: 0.01,
            ..Default::default()
        },
    };

    let out = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("triangle-mesh contact feedback");
    assert!(
        out.forces[0][0] < -300.0 && out.forces[0][1] > 300.0,
        "triangle ramp contact should use closest ramp normal, not flat ground: {:?}",
        out.forces[0]
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_contact_feedback_uses_terrain_surface_velocity() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemContactFeedbackKernel::new(&device);

    let mut input = TireFemContactFeedbackInput {
        positions: vec![[0.0, 0.010, 0.0, 0.0]],
        velocities: vec![[0.0; 4]],
        masses: vec![2.0],
        tread_ids: vec![0],
        bristles: vec![[0.0; 4]],
        terrain: Some(TireFemContactTerrain {
            origin: [0.0, 0.0],
            spacing: 1.0,
            nx: 1,
            nz: 1,
            heights: vec![0.0],
            friction: 1.0,
            friction_map: None,
            water_film_map: None,
            sinkage_map: None,
            soil_rolling_resistance_scale: 1.0,
            surface_velocity: [0.0; 3],
            surface_velocity_map: None,
            sdf_primitives: Vec::new(),
            triangles: Vec::new(),
        }),
        params: TireFemContactFeedbackParams {
            dt: 1.0 / 120.0,
            contact_band: 0.05,
            omega: 0.0,
            heading: [1.0, 0.0, 0.0],
            normal_load_per_tread: 800.0,
            area_per_tread: 0.01,
            tread_shear_stiffness: 4.0e5,
            mu_static: 1.0,
            mu_dynamic: 0.8,
            bristle_damping: 5.0e3,
            ..Default::default()
        },
    };

    let static_out = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("static-terrain feedback");
    input.terrain.as_mut().expect("terrain").surface_velocity = [4.0, 0.0, 0.0];
    let moving_out = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("moving-terrain feedback");

    assert!(
        static_out.forces[0][0].abs() < 1.0e-4,
        "static ground with no wheel spin should have no longitudinal shear: {:?}",
        static_out.forces[0]
    );
    assert!(
        moving_out.forces[0][0] > 100.0,
        "moving ground should enter relative slip and drag tread in +x: {:?}",
        moving_out.forces[0]
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_contact_feedback_samples_spatial_surface_velocity_map() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemContactFeedbackKernel::new(&device);

    let input = TireFemContactFeedbackInput {
        positions: vec![
            // Left cell has static ground.
            [0.0, 0.010, 0.0, 0.0],
            // Right cell is a moving belt/flow patch.
            [1.0, 0.010, 0.0, 0.0],
        ],
        velocities: vec![[0.0; 4]; 2],
        masses: vec![2.0; 2],
        tread_ids: vec![0, 1],
        bristles: vec![[0.0; 4]; 2],
        terrain: Some(TireFemContactTerrain {
            origin: [0.0, 0.0],
            spacing: 1.0,
            nx: 2,
            nz: 2,
            heights: vec![0.0; 4],
            friction: 1.0,
            friction_map: None,
            water_film_map: None,
            sinkage_map: None,
            soil_rolling_resistance_scale: 1.0,
            surface_velocity: [0.0; 3],
            surface_velocity_map: Some(vec![
                [0.0, 0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0, 0.0],
            ]),
            sdf_primitives: Vec::new(),
            triangles: Vec::new(),
        }),
        params: TireFemContactFeedbackParams {
            dt: 1.0 / 120.0,
            contact_band: 0.05,
            omega: 0.0,
            heading: [1.0, 0.0, 0.0],
            normal_load_per_tread: 800.0,
            area_per_tread: 0.01,
            tread_shear_stiffness: 4.0e5,
            mu_static: 1.0,
            mu_dynamic: 0.8,
            bristle_damping: 5.0e3,
            ..Default::default()
        },
    };

    let out = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("spatial velocity terrain feedback");
    assert!(
        out.forces[0][0].abs() < 1.0e-4,
        "static cell should not create longitudinal shear with zero wheel spin: {:?}",
        out.forces[0]
    );
    assert!(
        out.forces[1][0] > 100.0,
        "moving velocity-map cell should drag only the right tread node in +x: left={:?} right={:?}",
        out.forces[0],
        out.forces[1]
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_contact_feedback_consumes_gpu_resident_surface_velocity_map() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemContactFeedbackKernel::new(&device);

    let input = TireFemContactFeedbackInput {
        positions: vec![[0.0, 0.010, 0.0, 0.0], [1.0, 0.010, 0.0, 0.0]],
        velocities: vec![[0.0; 4]; 2],
        masses: vec![2.0; 2],
        tread_ids: vec![0, 1],
        bristles: vec![[0.0; 4]; 2],
        terrain: None,
        params: TireFemContactFeedbackParams {
            dt: 1.0 / 120.0,
            contact_band: 0.05,
            omega: 0.0,
            heading: [1.0, 0.0, 0.0],
            normal_load_per_tread: 800.0,
            area_per_tread: 0.01,
            tread_shear_stiffness: 4.0e5,
            mu_static: 1.0,
            mu_dynamic: 0.8,
            bristle_damping: 5.0e3,
            ..Default::default()
        },
    };
    let mk_f32 = |label: &str, data: &[f32]| {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (data.len().max(1) * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));
        buffer
    };
    let mk_vec4 = |label: &str, data: &[[f32; 4]]| {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (data.len().max(1) * core::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));
        buffer
    };
    let heights = mk_f32("test.feedback.gpu_velocity.heights", &[0.0; 4]);
    let velocity_map = mk_vec4(
        "test.feedback.gpu_velocity.map",
        &[
            [0.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
        ],
    );

    let out = kernel
        .dispatch_blocking_with_gpu_terrain(
            &device,
            &queue,
            &input,
            &TireFemContactFeedbackGpuTerrain {
                origin: [0.0, 0.0],
                spacing: 1.0,
                nx: 2,
                nz: 2,
                heights: &heights,
                friction: 1.0,
                friction_map: None,
                water_film_map: None,
                sinkage_map: None,
                surface_velocity_map: Some(&velocity_map),
                soil_rolling_resistance_scale: 1.0,
                surface_velocity: [0.0; 3],
            },
        )
        .expect("GPU-resident velocity-map feedback");
    assert!(
        out.forces[0][0].abs() < 1.0e-4 && out.forces[1][0] > 100.0,
        "GPU-resident velocity map should affect only the moving cell: left={:?} right={:?}",
        out.forces[0],
        out.forces[1]
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_contact_feedback_samples_deformable_sinkage_for_rolling_loss() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemContactFeedbackKernel::new(&device);

    let mut input = TireFemContactFeedbackInput {
        positions: vec![[0.0, 0.010, 0.0, 0.0]],
        velocities: vec![[2.0, 0.0, 0.0, 0.0]],
        masses: vec![2.0],
        tread_ids: vec![0],
        bristles: vec![[0.0; 4]],
        terrain: Some(TireFemContactTerrain {
            origin: [0.0, 0.0],
            spacing: 1.0,
            nx: 1,
            nz: 1,
            heights: vec![0.0],
            friction: 1.0,
            friction_map: None,
            water_film_map: None,
            sinkage_map: None,
            soil_rolling_resistance_scale: 1.0,
            surface_velocity: [0.0; 3],
            surface_velocity_map: None,
            sdf_primitives: Vec::new(),
            triangles: Vec::new(),
        }),
        params: TireFemContactFeedbackParams {
            dt: 1.0 / 120.0,
            contact_band: 0.05,
            omega: 2.0 / 0.32,
            hub_vel: [2.0, 0.0, 0.0],
            heading: [1.0, 0.0, 0.0],
            rolling_radius: 0.32,
            normal_load_per_tread: 800.0,
            area_per_tread: 0.01,
            tread_shear_stiffness: 4.0e5,
            mu_static: 1.0,
            mu_dynamic: 0.8,
            bristle_damping: 5.0e3,
            ..Default::default()
        },
    };

    let hard_out = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("hard-terrain feedback");
    input.terrain.as_mut().expect("terrain").sinkage_map = Some(vec![0.04]);
    let soft_out = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("sinkage-terrain feedback");

    assert!(
        soft_out.forces[0][0] < hard_out.forces[0][0] - 50.0,
        "deformable sinkage should add Bekker-style rolling loss against forward travel: hard={:?} soft={:?}",
        hard_out.forces[0],
        soft_out.forces[0]
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_contact_feedback_consumes_gpu_resident_terrain_buffers() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemContactFeedbackKernel::new(&device);

    let input = TireFemContactFeedbackInput {
        positions: vec![[0.0, 0.010, 0.0, 0.0]],
        velocities: vec![[2.0, 0.0, 0.0, 0.0]],
        masses: vec![2.0],
        tread_ids: vec![0],
        bristles: vec![[0.0; 4]],
        terrain: None,
        params: TireFemContactFeedbackParams {
            dt: 1.0 / 120.0,
            contact_band: 0.05,
            omega: 2.0 / 0.32,
            hub_vel: [2.0, 0.0, 0.0],
            heading: [1.0, 0.0, 0.0],
            rolling_radius: 0.32,
            normal_load_per_tread: 800.0,
            area_per_tread: 0.01,
            tread_shear_stiffness: 4.0e5,
            mu_static: 1.0,
            mu_dynamic: 0.8,
            bristle_damping: 5.0e3,
            ..Default::default()
        },
    };

    let mk = |label: &str, data: &[f32]| {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (data.len().max(1) * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));
        buffer
    };
    let heights = mk("test.gpu_terrain.heights", &[0.0]);
    let sinkage_hard = mk("test.gpu_terrain.sinkage_hard", &[0.0]);
    let sinkage_soft = mk("test.gpu_terrain.sinkage_soft", &[0.04]);
    let hard = kernel
        .dispatch_blocking_with_gpu_terrain(
            &device,
            &queue,
            &input,
            &TireFemContactFeedbackGpuTerrain {
                origin: [0.0, 0.0],
                spacing: 1.0,
                nx: 1,
                nz: 1,
                heights: &heights,
                friction: 1.0,
                friction_map: None,
                water_film_map: None,
                sinkage_map: Some(&sinkage_hard),
                soil_rolling_resistance_scale: 1.0,
                surface_velocity: [0.0; 3],
                surface_velocity_map: None,
            },
        )
        .expect("hard GPU-terrain feedback");
    let soft = kernel
        .dispatch_blocking_with_gpu_terrain(
            &device,
            &queue,
            &input,
            &TireFemContactFeedbackGpuTerrain {
                origin: [0.0, 0.0],
                spacing: 1.0,
                nx: 1,
                nz: 1,
                heights: &heights,
                friction: 1.0,
                friction_map: None,
                water_film_map: None,
                sinkage_map: Some(&sinkage_soft),
                soil_rolling_resistance_scale: 1.0,
                surface_velocity: [0.0; 3],
                surface_velocity_map: None,
            },
        )
        .expect("soft GPU-terrain feedback");
    assert!(
        soft.forces[0][0] < hard.forces[0][0] - 50.0,
        "GPU-resident sinkage terrain should be consumed directly by FEM contact feedback: hard={:?} soft={:?}",
        hard.forces[0],
        soft.forces[0]
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_contact_reduction_reports_patch_diagnostics() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemContactReductionKernel::new(&device);

    let input = TireFemContactReductionInput {
        positions: vec![
            [-0.5, 0.020, 0.0, 0.0],
            [0.5, 0.040, 0.0, 0.0],
            [0.0, 0.120, 0.0, 0.0],
            [0.8, 0.090, 0.0, 0.0],
        ],
        tread_ids: vec![0, 1, 2, 3],
        terrain: None,
        params: TireFemContactReductionParams {
            hub: [0.0, 0.32, 0.0],
            heading: [1.0, 0.0, 0.0],
            contact_band: 0.05,
            ground_y: 0.0,
            downforce: 300.0,
        },
    };
    let out = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("contact reduction");
    assert_eq!(out.contact_count, 2);
    assert!((out.normal_load - 300.0).abs() < 1.0e-3);
    assert!((out.patch_length - 1.0).abs() < 1.0e-4);
    // Weights are 0.6 for x=-0.5 and 0.2 for x=+0.5, so center is -0.25.
    assert!((out.contact_center_x + 0.25).abs() < 2.0e-4);
    assert!((out.min_x + 0.5).abs() < 1.0e-4);
    assert!((out.max_x - 0.5).abs() < 1.0e-4);
    assert!((out.weight_sum - 0.8).abs() < 1.0e-4);
    assert!((out.world_force[0]).abs() < 1.0e-4);
    assert!((out.world_force[1] - 300.0).abs() < 1.0e-3);
    assert!((out.world_force[2]).abs() < 1.0e-4);
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_contact_reduction_samples_sdf_primitives() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemContactReductionKernel::new(&device);

    let input = TireFemContactReductionInput {
        positions: vec![
            // This point is just inside the side of the SDF sphere but far
            // above the flat heightfield, so closest-surface SDF contact must
            // win over vertical ground projection.
            [0.34, 0.30, 0.0, 0.0],
            [0.0, 0.72, 0.0, 0.0],
        ],
        tread_ids: vec![0, 1],
        terrain: Some(TireFemContactTerrain {
            origin: [0.0, 0.0],
            spacing: 1.0,
            nx: 1,
            nz: 1,
            heights: vec![0.0],
            friction: 1.0,
            friction_map: None,
            water_film_map: None,
            sinkage_map: None,
            soil_rolling_resistance_scale: 1.0,
            surface_velocity: [0.0; 3],
            surface_velocity_map: None,
            sdf_primitives: vec![TireFemContactSdfPrimitive::sphere(
                [0.0, 0.30, 0.0],
                0.35,
                0.55,
            )],
            triangles: Vec::new(),
        }),
        params: TireFemContactReductionParams {
            hub: [0.0, 0.30, 0.0],
            heading: [1.0, 0.0, 0.0],
            contact_band: 0.05,
            ground_y: 0.0,
            downforce: 500.0,
        },
    };

    let out = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("sdf contact reduction");
    assert_eq!(out.contact_count, 1);
    assert!((out.normal_load - 500.0).abs() < 1.0e-3);
    assert!(
        out.world_force[0] > 450.0 && out.world_force[1].abs() < 5.0,
        "SDF sphere side normal should dominate world force: {:?}",
        out.world_force
    );
    assert!(
        (out.mean_friction - 0.55).abs() < 2.0e-4,
        "SDF primitive friction should win when SDF surface wins: {}",
        out.mean_friction
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_contact_reduction_samples_triangle_mesh_patch() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemContactReductionKernel::new(&device);

    let input = TireFemContactReductionInput {
        positions: vec![
            // Below the y=x ramp and just outside the finite triangle's z=0
            // edge. Flat/heightfield contact would miss this point.
            [0.52, 0.45, -0.02, 0.0],
            // Far above the ramp; should not count.
            [0.0, 1.20, 0.0, 0.0],
        ],
        tread_ids: vec![0, 1],
        terrain: Some(TireFemContactTerrain {
            origin: [0.0, 0.0],
            spacing: 1.0,
            nx: 1,
            nz: 1,
            heights: vec![0.0],
            friction: 1.0,
            friction_map: None,
            water_film_map: None,
            sinkage_map: None,
            soil_rolling_resistance_scale: 1.0,
            surface_velocity: [0.0; 3],
            surface_velocity_map: None,
            sdf_primitives: Vec::new(),
            triangles: vec![
                TireFemContactTriangle::new(
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    0.42,
                ),
                TireFemContactTriangle::new(
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0],
                    0.42,
                ),
            ],
        }),
        params: TireFemContactReductionParams {
            hub: [0.0, 0.50, 0.0],
            heading: [1.0, 0.0, 0.0],
            contact_band: 0.10,
            ground_y: 0.0,
            downforce: 500.0,
        },
    };

    let out = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("triangle-mesh contact reduction");
    assert_eq!(out.contact_count, 1);
    assert!((out.normal_load - 500.0).abs() < 1.0e-3);
    assert!(
        out.world_force[0] < -300.0 && out.world_force[1] > 300.0,
        "triangle ramp normal should dominate world force: {:?}",
        out.world_force
    );
    assert!(
        (out.mean_friction - 0.42).abs() < 2.0e-4,
        "triangle friction should be reported when triangle contact wins: {}",
        out.mean_friction
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_contact_reduction_samples_heightfield_and_friction() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemContactReductionKernel::new(&device);

    let input = TireFemContactReductionInput {
        positions: vec![
            // At x=0,z=0 ground height is 0.03 -> clearance 0.01, weight 0.8.
            [0.0, 0.040, 0.0, 0.0],
            // At x=1,z=0 ground height is 0.00 -> clearance 0.04, weight 0.2.
            [1.0, 0.040, 0.0, 0.0],
            // Above band; should not count.
            [0.0, 0.090, 1.0, 0.0],
        ],
        tread_ids: vec![0, 1, 2],
        terrain: Some(TireFemContactTerrain {
            origin: [0.0, 0.0],
            spacing: 1.0,
            nx: 2,
            nz: 2,
            heights: vec![0.03, 0.0, 0.0, 0.0],
            friction: 1.0,
            friction_map: Some(vec![0.4, 1.0, 1.0, 1.0]),
            water_film_map: Some(vec![0.004, 0.0, 0.0, 0.0]),
            sinkage_map: Some(vec![0.020, 0.0, 0.0, 0.0]),
            soil_rolling_resistance_scale: 1.0,
            surface_velocity: [0.0; 3],
            surface_velocity_map: None,
            sdf_primitives: Vec::new(),
            triangles: Vec::new(),
        }),
        params: TireFemContactReductionParams {
            hub: [0.0, 0.32, 0.0],
            heading: [1.0, 0.0, 0.0],
            contact_band: 0.05,
            ground_y: 0.0,
            downforce: 500.0,
        },
    };
    let out = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("heightfield contact reduction");
    assert_eq!(out.contact_count, 2);
    assert!((out.normal_load - 500.0).abs() < 1.0e-3);
    assert!((out.patch_length - 1.0).abs() < 1.0e-4);
    assert!((out.contact_center_x - 0.2).abs() < 2.0e-4);
    assert!((out.weight_sum - 1.0).abs() < 1.0e-4);
    // Weighted by contact weights: (0.8*0.4 + 0.2*1.0) / 1.0 = 0.52.
    assert!((out.mean_friction - 0.52).abs() < 2.0e-4);
    // Weighted by the same contact weights: 0.8 * 4mm = 3.2mm.
    assert!((out.mean_water_film_m - 0.0032).abs() < 2.0e-5);
    // Same weights: 0.8 * 20mm = 16mm of sampled deformable-terrain rut.
    assert!((out.mean_sinkage_m - 0.016).abs() < 2.0e-5);
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_contact_reduction_consumes_gpu_resident_terrain_buffers() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemContactReductionKernel::new(&device);

    let input = TireFemContactReductionInput {
        positions: vec![
            [0.0, 0.040, 0.0, 0.0],
            [1.0, 0.040, 0.0, 0.0],
            [0.0, 0.090, 1.0, 0.0],
        ],
        tread_ids: vec![0, 1, 2],
        terrain: None,
        params: TireFemContactReductionParams {
            hub: [0.0, 0.32, 0.0],
            heading: [1.0, 0.0, 0.0],
            contact_band: 0.05,
            ground_y: 0.0,
            downforce: 500.0,
        },
    };
    let mk = |label: &str, data: &[f32]| {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (data.len().max(1) * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));
        buffer
    };
    let heights = mk("test.reduction.gpu_terrain.heights", &[0.03, 0.0, 0.0, 0.0]);
    let friction = mk("test.reduction.gpu_terrain.friction", &[0.4, 1.0, 1.0, 1.0]);
    let water = mk("test.reduction.gpu_terrain.water", &[0.004, 0.0, 0.0, 0.0]);
    let sinkage = mk(
        "test.reduction.gpu_terrain.sinkage",
        &[0.020, 0.0, 0.0, 0.0],
    );
    let out = kernel
        .dispatch_blocking_with_gpu_terrain(
            &device,
            &queue,
            &input,
            &TireFemContactReductionGpuTerrain {
                origin: [0.0, 0.0],
                spacing: 1.0,
                nx: 2,
                nz: 2,
                heights: &heights,
                friction: 1.0,
                friction_map: Some(&friction),
                water_film_map: Some(&water),
                sinkage_map: Some(&sinkage),
                soil_rolling_resistance_scale: 1.0,
                surface_velocity: [0.0; 3],
                surface_velocity_map: None,
            },
        )
        .expect("GPU-resident terrain contact reduction");
    assert_eq!(out.contact_count, 2);
    assert!((out.normal_load - 500.0).abs() < 1.0e-3);
    assert!((out.mean_friction - 0.52).abs() < 2.0e-4);
    assert!((out.mean_water_film_m - 0.0032).abs() < 2.0e-5);
    assert!((out.mean_sinkage_m - 0.016).abs() < 2.0e-5);
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_contact_reduction_accumulates_heightfield_world_wrench() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemContactReductionKernel::new(&device);

    let input = TireFemContactReductionInput {
        positions: vec![
            [0.0, 0.020, 0.0, 0.0],
            [1.0, 0.120, 0.0, 0.0],
            [0.0, 0.020, 1.0, 0.0],
        ],
        tread_ids: vec![0, 1, 2],
        terrain: Some(TireFemContactTerrain {
            origin: [0.0, 0.0],
            spacing: 1.0,
            nx: 2,
            nz: 2,
            // Plane h = 0.1*x, so upward normal has negative x component.
            heights: vec![0.0, 0.1, 0.0, 0.1],
            friction: 1.0,
            friction_map: None,
            water_film_map: None,
            sinkage_map: None,
            soil_rolling_resistance_scale: 1.0,
            surface_velocity: [0.0; 3],
            surface_velocity_map: None,
            sdf_primitives: Vec::new(),
            triangles: Vec::new(),
        }),
        params: TireFemContactReductionParams {
            hub: [0.0, 0.32, 0.0],
            heading: [1.0, 0.0, 0.0],
            contact_band: 0.05,
            ground_y: 0.0,
            downforce: 400.0,
        },
    };
    let out = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("heightfield wrench contact reduction");
    assert_eq!(out.contact_count, 3);
    assert!((out.normal_load - 400.0).abs() < 1.0e-3);
    assert!(
        out.world_force[0] < -10.0,
        "slope normal should push backward in x: {:?}",
        out.world_force
    );
    assert!(
        out.world_force[1] > 390.0,
        "normal world force should keep large upward component: {:?}",
        out.world_force
    );
    assert!(
        out.world_moment.iter().all(|v| v.is_finite()),
        "world moment should be finite: {:?}",
        out.world_moment
    );
}

/// PLAN_TOURQUE Stage G part 3 — confirm the WGSL `phys5` row is wired
/// end-to-end. With a tread node placed below the contact plane, a
/// positive `hub_omega` must push it through the XPBD friction code in
/// the −X direction relative to the same dispatch with `omega = 0`,
/// matching the CPU sign convention pinned by
/// `tetra_tire_fem_spin_creates_backward_tread_slip_at_contact` in
/// `crates/molla-vehicle/src/real_tire.rs`.
fn tire_fem_input_with_grounded_tread(omega: f32) -> TireFemGpuInput {
    let mut input = simple_tire_fem_input(0.0);
    // Zero out the FEM internal forces so this single-step test is
    // dominated by gravity + the XPBD contact branch we care about.
    input.tet_mu = vec![0.0];
    input.tet_lambda = vec![0.0];
    // Put the tread node (index 2) below the contact plane y=0 so the
    // XPBD contact branch fires immediately. Other nodes stay near the
    // rest tet (the FEM force is zeroed so deformation doesn't matter).
    input.positions[2] = [0.0, -0.02, 0.0, 0.0];
    input.velocities[2] = [0.0, -0.05, 0.0, 0.0];
    input.params.dt = 1.0 / 240.0;
    input.params.hub = [0.0, 0.25, 0.0];
    input.params.hub_omega = omega;
    input.params.young_modulus = 0.0;
    input.params.pressure_pa = 0.0;
    input.params.node_damping = 0.0;
    input.params.floor_stiffness = 0.0;
    input.params.floor_damping = 0.0;
    input.params.gravity_y = 0.0;
    // Disable every XPBD projection too — they fire on rest-pose
    // deviation regardless of `tet_mu`, and would push the contact
    // node back above y=0 before the friction branch we're testing.
    input.params.xpbd_volume_compliance = 1.0e30;
    input.params.xpbd_bead_compliance = 1.0e30;
    input.params.xpbd_sidewall_ply_compliance = 1.0e30;
    input
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_spin_pushes_grounded_tread_backward_in_contact() {
    let Some(engine) = try_engine() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemGpuKernel::new(&device);

    // Baseline: omega = 0. Whatever lateral shift happens, treat as the
    // zero-spin reference.
    let zero_out = kernel
        .dispatch_blocking(&device, &queue, &tire_fem_input_with_grounded_tread(0.0))
        .expect("baseline dispatch");
    // Positive spin: should push the contact tread node in −X relative
    // to baseline. The sign comes from `v_spin = ω · r_y` with the
    // tread node's `r_y = pos.y − hub.y = −0.02 − 0.25 = −0.27` and
    // positive ω → v_spin.x < 0. The omega value must be large enough
    // for `|v_spin| · dt` to exceed the static-friction cap
    // `μ_s · pen`, otherwise the XPBD branch sticks and the test
    // sees zero motion in both directions.
    let spin_out = kernel
        .dispatch_blocking(&device, &queue, &tire_fem_input_with_grounded_tread(80.0))
        .expect("spin dispatch");
    let baseline_x = zero_out.positions[2][0];
    let spin_x = spin_out.positions[2][0];
    let dx = spin_x - baseline_x;
    assert!(
        dx.is_finite() && spin_x.is_finite() && baseline_x.is_finite(),
        "tread node X must stay finite under both dispatches: baseline={baseline_x} spin={spin_x}"
    );
    assert!(
        dx < 0.0,
        "positive hub_omega should push the grounded tread node in −X via the WGSL phys5 contact path; \
         baseline_x={baseline_x}, spin_x={spin_x}, dx={dx}"
    );
}

/// F2 (INSPECTION.md) — zero-host-readback substep loop.
///
/// The resident chain
/// `step_persistent → dispatch_resident → step_hub_persistent_with_tread_forces`
/// must reproduce the blocking readback chain
/// (`readback_state → dispatch_blocking → upload_state → host wrench
/// sum → step_hub_persistent`) on the G4 rolling-contact scenario:
/// same shader, same per-substep uniforms, so the only admissible
/// divergence is the on-device wrench reduction order (f32 noise,
/// amplified by 96 substeps of dynamics). The CPU tire steps alongside
/// purely as the shared per-substep parameter bridge, exactly like the
/// G4 parity test.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_resident_contact_loop_matches_blocking_chain() {
    use molla_vehicle::{SurfaceMaterial, TetraTireFem, TetraTireFemParams};
    let Some(engine) = try_engine() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemGpuKernel::new(&device);
    let feedback_kernel = TireFemContactFeedbackKernel::new(&device);

    let params = TetraTireFemParams {
        surface_material: SurfaceMaterial {
            mu_static: 1.0,
            mu_dynamic: 0.85,
            bristle_damping: 5.0e4,
            rolling_resistance: 0.0,
            cohesion: 0.0,
        },
        ..TetraTireFemParams::realtime_viewer()
    };
    let mut cpu_tire = TetraTireFem::new(0.0, 180_000.0, params);
    let dt = 1.0 / 8_000.0;
    let terrain = molla_vehicle::terrain::FlatTerrain::new(0.0, 1.0);

    // Same settle recipe as the G4 rolling parity test.
    cpu_tire.set_hub_kinematic([0.0, params.outer_radius * 0.982, 0.0], [0.0; 3]);
    for _ in 0..1_600 {
        cpu_tire.step_on_terrain(&terrain, dt);
    }
    cpu_tire.set_hub_dynamic();
    let mut settled_in_contact = false;
    for _ in 0..20 {
        for _ in 0..1_600 {
            cpu_tire.step_on_terrain(&terrain, dt);
        }
        let nodes = cpu_tire.nodes();
        if cpu_tire
            .tread_nodes()
            .iter()
            .filter(|&&id| nodes[id][1] < 1.0e-3)
            .count()
            >= 2
        {
            settled_in_contact = true;
            break;
        }
    }
    assert!(settled_in_contact, "tire must settle into contact");
    cpu_tire.set_hub_spin(6.0);
    cpu_tire.set_drive_torque_n_m(120.0);

    let base_gpu_params = TireFemGpuParams {
        dt: dt as f32,
        node_damping: params.node_damping as f32,
        bead_stiffness: params.bead_stiffness as f32,
        xpbd_volume_compliance: params.xpbd_volume_compliance as f32,
        xpbd_bead_compliance: params.xpbd_bead_compliance as f32,
        semi_implicit_stiffness_scale: params.semi_implicit_stiffness_scale as f32,
        ..Default::default()
    };
    let gpu_input = tire_fem_gpu_input_from_tetra_fem(&cpu_tire, base_gpu_params);
    let masses: Vec<f32> = gpu_input.masses.clone();
    let tread_ids: Vec<u32> = gpu_input.tread_ids.clone();
    let n_nodes = gpu_input.positions.len() as u32;

    // Two identical persistent meshes: one for each chain.
    let mut mesh_blocking = kernel
        .create_persistent(&device, &queue, &gpu_input)
        .expect("blocking mesh");
    let mut mesh_resident = kernel
        .create_persistent(&device, &queue, &gpu_input)
        .expect("resident mesh");
    let mut resident = feedback_kernel
        .create_resident(
            &device,
            &queue,
            &mesh_resident,
            &vec![[0.0_f32; 4]; tread_ids.len()],
        )
        .expect("resident contact state");

    let hub_mass = params.hub_mass as f32;
    let hub_floor_y = (params.inner_radius * 0.75) as f32;
    let base_hub_params = TireFemHubParams {
        dt: dt as f32,
        bead_stiffness: params.bead_stiffness as f32,
        bead_damping: params.bead_damping as f32,
        hub_mass,
        hub_damping: 260.0,
        downforce: params.downforce as f32,
        gravity_y: -9.81,
        hub_floor_active: 1.0,
        hub_floor_y,
        external_force_xyz: [0.0; 3],
        lateral_damping: 80.0,
    };

    let contact_band = 0.012_f32;
    let in_band = |tire: &TetraTireFem| -> usize {
        let nodes = tire.nodes();
        tire.tread_nodes()
            .iter()
            .filter(|&&id| (nodes[id][1] as f32) < contact_band)
            .count()
            .max(1)
    };

    let mut blocking_bristles: Vec<[f32; 4]> = vec![[0.0; 4]; tread_ids.len()];
    let roll_start_x = cpu_tire.hub()[0] as f32;

    for step in 0..96 {
        let bridged = tire_fem_gpu_input_from_tetra_fem(&cpu_tire, base_gpu_params);
        let node_params = bridged.params;
        let hub_pre = cpu_tire.hub();
        let hub_vel_pre = cpu_tire.hub_velocity();
        let omega_pre = cpu_tire.hub_omega();
        let load_per_tread = (params.downforce / in_band(&cpu_tire) as f64) as f32;
        let fb_params = TireFemContactFeedbackParams {
            dt: dt as f32,
            contact_band,
            ground_y: 0.0,
            omega: omega_pre as f32,
            hub: [hub_pre[0] as f32, hub_pre[1] as f32, hub_pre[2] as f32],
            hub_vel: [
                hub_vel_pre[0] as f32,
                hub_vel_pre[1] as f32,
                hub_vel_pre[2] as f32,
            ],
            heading: [1.0, 0.0, 0.0],
            rolling_radius: params.outer_radius as f32,
            normal_load_per_tread: load_per_tread,
            area_per_tread: node_params.pressure_area_per_tread_node,
            tread_shear_stiffness: params.tread_shear_stiffness as f32,
            mu_static: 1.0,
            mu_dynamic: 0.85,
            bristle_damping: 5.0e4,
            cohesion: 0.0,
            bristle_relax_rate: params.bristle_relax_rate as f32,
        };
        cpu_tire.step_on_terrain(&terrain, dt);

        // ── Chain A: blocking readback loop (the pre-F2 pattern).
        kernel.step_persistent(&device, &queue, &mut mesh_blocking, node_params);
        let gpu_state = kernel
            .readback_state(&device, &queue, &mesh_blocking)
            .expect("readback state");
        let feedback = feedback_kernel
            .dispatch_blocking(
                &device,
                &queue,
                &TireFemContactFeedbackInput {
                    positions: gpu_state.positions.clone(),
                    velocities: gpu_state.velocities.clone(),
                    masses: masses.clone(),
                    tread_ids: tread_ids.clone(),
                    bristles: blocking_bristles.clone(),
                    terrain: None,
                    params: fb_params,
                },
            )
            .expect("blocking feedback");
        let wrench = feedback.patch_accounting().total_force;
        blocking_bristles = feedback.bristles;
        kernel.upload_state(
            &queue,
            &mesh_blocking,
            &gpu_state.positions,
            &feedback.velocities,
        );
        kernel.step_hub_persistent(
            &device,
            &queue,
            &mesh_blocking,
            TireFemHubParams {
                external_force_xyz: [wrench[0], 0.0, wrench[2]],
                ..base_hub_params
            },
        );

        // ── Chain B: zero-readback resident loop.
        kernel.step_persistent(&device, &queue, &mut mesh_resident, node_params);
        feedback_kernel
            .dispatch_resident(&device, &queue, &mesh_resident, &mut resident, fb_params)
            .expect("resident feedback");
        kernel.step_hub_persistent_with_tread_forces(
            &device,
            &queue,
            &mesh_resident,
            base_hub_params,
            resident.force_rows(),
            n_nodes,
        );

        if step % 8 == 7 {
            let hub_a = kernel
                .readback_hub(&device, &queue, &mesh_blocking)
                .expect("hub A");
            let hub_b = kernel
                .readback_hub(&device, &queue, &mesh_resident)
                .expect("hub B");
            let dx = (hub_a[0] - hub_b[0]).abs();
            let dy = (hub_a[1] - hub_b[1]).abs();
            eprintln!(
                "step {step}: blocking hub=({:.6},{:.6}) resident hub=({:.6},{:.6}) d=({dx:.2e},{dy:.2e})",
                hub_a[0], hub_a[1], hub_b[0], hub_b[1]
            );
            assert!(
                dx < 5.0e-3 && dy < 5.0e-3,
                "step {step}: resident chain hub diverged from blocking chain \
                 (dx={dx}, dy={dy}) — only reduction-order f32 noise is admissible"
            );
        }
    }

    // Full-state agreement at the end: same shader, same inputs — the
    // node states may differ only through the hub-coupling noise.
    let state_a = kernel
        .readback_state(&device, &queue, &mesh_blocking)
        .expect("final state A");
    let state_b = kernel
        .readback_state(&device, &queue, &mesh_resident)
        .expect("final state B");
    let mut max_dp = 0.0_f32;
    for (a, b) in state_a.positions.iter().zip(state_b.positions.iter()) {
        for c in 0..3 {
            max_dp = max_dp.max((a[c] - b[c]).abs());
        }
    }
    let resident_bristles = resident
        .read_bristles(&device, &queue)
        .expect("resident bristles");
    let mut max_db = 0.0_f32;
    for (a, b) in blocking_bristles.iter().zip(resident_bristles.iter()) {
        for c in 0..3 {
            max_db = max_db.max((a[c] - b[c]).abs());
        }
    }
    eprintln!("final: max node |Δp| = {max_dp:.3e}, max bristle |Δ| = {max_db:.3e}");
    assert!(max_dp < 1.0e-3, "node positions diverged: {max_dp}");
    assert!(max_db < 1.0e-4, "bristle state diverged: {max_db}");

    // Non-vacuous: the resident tire must actually have rolled
    // forward from where the settle left it (the hub settles slightly
    // off-origin, so absolute x is not the right gate).
    let hub_b = kernel
        .readback_hub(&device, &queue, &mesh_resident)
        .expect("hub B final");
    assert!(
        hub_b[0] > roll_start_x + 1.0e-4,
        "resident tire should roll forward under spin + drive torque: \
         x={} start={roll_start_x}",
        hub_b[0]
    );
}

/// F2 (INSPECTION.md) — the 40-wheel realtime *measurement* the atlas
/// smoke test never took. Times the fused-atlas substep dispatch for
/// 10 vehicles × 4 wheels and reports substeps/second against the
/// realtime budget (`dt = 1/8000` ⇒ 8 000 substeps/s). The hard gate
/// is deliberately loose (catastrophic-regression floor); the printed
/// realtime factor is the number that matters — record it in
/// PROGRESS.md when the target hardware changes. Run with `RELEASE=1`:
///   make test-gpu P=molla-kernels T=tire_fem_gpu RELEASE=1 A=forty_wheel_walltime
#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_atlas_forty_wheel_walltime() {
    use molla_vehicle::{TetraTireFem, TetraTireFemParams};
    let Some(engine) = try_engine() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemGpuKernel::new(&device);

    let params = TetraTireFemParams::realtime_viewer();
    let dt = 1.0 / 8_000.0_f64;
    let base_gpu_params = TireFemGpuParams {
        dt: dt as f32,
        node_damping: params.node_damping as f32,
        bead_stiffness: params.bead_stiffness as f32,
        xpbd_volume_compliance: params.xpbd_volume_compliance as f32,
        xpbd_bead_compliance: params.xpbd_bead_compliance as f32,
        ..Default::default()
    };

    let mut inputs: Vec<TireFemGpuInput> = Vec::with_capacity(40);
    let mut total_nodes = 0usize;
    for veh in 0..10 {
        for wheel in 0..4 {
            let hub_x_base = veh as f64 * 4.0;
            let along_axle = match wheel {
                0 | 2 => -1.4,
                _ => 1.4,
            };
            let hub_x = hub_x_base + along_axle;
            let hub_z = match wheel {
                0 | 1 => -0.7,
                _ => 0.7,
            };
            let mut tire = TetraTireFem::new(hub_x, 180_000.0, params);
            tire.set_hub_kinematic([hub_x, 1.0, hub_z], [0.0; 3]);
            let input = tire_fem_gpu_input_from_tetra_fem(&tire, base_gpu_params);
            total_nodes += input.positions.len();
            inputs.push(input);
        }
    }
    let mut batch = kernel
        .create_resident_atlas_batch(&device, &queue, &inputs)
        .expect("create resident atlas batch");
    let step_params: Vec<TireFemGpuParams> = inputs.iter().map(|i| i.params).collect();

    // Warm-up (pipeline compilation, first-dispatch overheads).
    for _ in 0..32 {
        kernel
            .step_resident_atlas_batch(&device, &queue, &mut batch, &step_params)
            .expect("warm-up substep");
    }
    let _ = kernel
        .readback_resident_atlas_batch(&device, &queue, &batch)
        .expect("warm-up sync");

    // Timed run: N substeps, then a readback to force completion so
    // the clock covers GPU execution, not just submission.
    let n_steps = 2_000u32;
    let t0 = std::time::Instant::now();
    for _ in 0..n_steps {
        kernel
            .step_resident_atlas_batch(&device, &queue, &mut batch, &step_params)
            .expect("timed substep");
    }
    let outputs = kernel
        .readback_resident_atlas_batch(&device, &queue, &batch)
        .expect("timed sync");
    let elapsed = t0.elapsed();

    for (wheel_idx, out) in outputs.iter().enumerate() {
        for p in &out.positions {
            assert!(
                p.iter().all(|x| x.is_finite()),
                "wheel {wheel_idx}: non-finite position after timed run"
            );
        }
    }

    let secs = elapsed.as_secs_f64();
    let substeps_per_sec = n_steps as f64 / secs.max(1.0e-9);
    // Realtime budget: dt = 1/8000 ⇒ 8000 substeps of the WHOLE
    // 40-wheel batch per simulated second.
    let realtime_factor = substeps_per_sec / 8_000.0;
    eprintln!(
        "atlas 40-wheel walltime: {total_nodes} nodes, {n_steps} substeps in {:.3} s \
         → {substeps_per_sec:.0} substeps/s → realtime factor {realtime_factor:.2}x \
         (>= 1.0 means 40 live FEM tires run realtime at dt = 1/8000)",
        secs
    );
    // Catastrophic-regression floor only — absolute throughput is
    // hardware-dependent and belongs in the printed record, not a gate.
    assert!(
        realtime_factor > 0.05,
        "atlas substep throughput collapsed: {substeps_per_sec:.0} substeps/s \
         ({realtime_factor:.3}x realtime)"
    );
}
