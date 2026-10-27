// T1/T2 — Volumetric tetra-FEM tire carcass dynamics with inflation pressure.
//
// Five tires sit side-by-side: four at fixed inflation pressures (0, 80,
// 200, 320 kPa) plus one whose pressure cycles 0 → 350 kPa → 0 so the
// audience sees one full inflation/deflation transition. Each tire is a true
// volumetric tetrahedral FEM carcass from `molla_vehicle::TetraTireFem`, with
// an annular tire volume split into tetrahedra:
//
//   * circumferential × radial × width nodes form a real 3-D tire body.
//   * each hexahedral parameter cell is split into tetrahedra.
//   * Neo-Hookean tetra FEM stress produces the internal rubber forces.
//   * anisotropic cord/belt fibres stiffen tread, sidewall and bead regions.
//   * the bead/inner radius is coupled to a 1-DoF loaded hub.
//   * a sealed cavity gas law pushes on the bead-side cavity surface.
//   * tread nodes use zero-gap contact plus Coulomb/brush friction.
//   * a constant downward hub load (2.6 kN) — fixed so the only visible
//     variable between tires is inflation pressure.
//   * pressure changes cavity load and membrane tangent stiffness, so
//     high-pressure tires remain rounder while low-pressure tires squash
//     into wide patches.
//
// The hub is locked in X / Z (1-DoF, free in Y) — what you see is the
// dynamic deflection of the carcass under a fixed load at the five
// pressures. Higher pressure ⇒ stiffer casing ⇒ less deflection ⇒ small
// contact patch. Lower pressure ⇒ casing folds, tire pancakes, wide
// patch. The leftmost "deflated" tire flattens toward the rim while every
// tire keeps the same undeformed radius.
//
// This is *not* a Pacejka force-element model — it's the rubber +
// air carcass FEM behind it. The contact patch and carcass motion are
// emergent from the tetrahedral volume dynamics.

const TIRE_SEGMENTS: usize = 32;
const TIRE_RADIUS: f64 = 0.32; // 13" rim + 195/65/R13 tyre, roughly
const TIRE_INNER_RADIUS: f64 = 0.12;
const HUB_RADIUS: f64 = 0.07; // visual only — sidewall extends to here
const TIRE_WIDTH: f64 = 0.185; // effective tire width (m)
const REFERENCE_PRESSURE_PA: f64 = 320_000.0;
const TIRE_DYNAMICS_VEHICLE_CHASSIS_Y: f32 = 0.50;
const TIRE_DYNAMICS_VEHICLE_ATTACH_Y: f32 = -0.20;
const TIRE_DYNAMICS_BRUSH_FORCE_SCALE: f32 = 0.02;

fn tire_carcass_params(pressure: f64) -> molla_vehicle::TetraTireFemParams {
    molla_vehicle::TetraTireFemParams {
        outer_radius: TIRE_RADIUS,
        inner_radius: TIRE_INNER_RADIUS,
        width: TIRE_WIDTH,
        segments: TIRE_SEGMENTS,
        radial_layers: 3,
        width_layers: 3,
        pressure_pa: pressure,
        downforce: 2_600.0,
        young_modulus: 4.0e6,
        sidewall_young_modulus: 2.0e6,
        tread_young_modulus: 8.0e6,
        bead_young_modulus: 2.5e7,
        density: 1_100.0,
        node_damping: 55.0,
        bead_stiffness: 5.8e4,
        pressure_load_scale: 0.0,
        ..Default::default()
    }
}

fn tire_visual_hub_height_for_load(pressure: f64, load_n: f64) -> f64 {
    let params = molla_vehicle::RealTireParams::passenger_radial()
        .with_geometry_and_pressure(TIRE_RADIUS, TIRE_WIDTH, pressure)
        .with_segments(64);
    let terrain = molla_vehicle::FlatTerrain::new(0.0, 0.9);
    molla_vehicle::solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, load_n)
}

fn tire_visual_hub_height(pressure: f64) -> f64 {
    tire_visual_hub_height_for_load(pressure, 2_600.0)
}

fn build_tire_dynamics_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("tire dynamics floor");
    (b.build().expect("tire dynamics model"), Vec::new())
}

#[derive(Clone, Debug, Default)]
struct TireRigReplayPanelState {
    run: Option<molla_vehicle::TireRigRunOutput>,
    selected: usize,
    show_csv: bool,
}

fn tire_rig_replay_expected_behavior(scenario_id: &str) -> &'static str {
    match scenario_id {
        "load_pressure_high" => {
            "High pressure should carry the same load with a shorter, stiffer patch."
        }
        "load_pressure_low" => {
            "Lower pressure should visibly grow the contact patch under the same load."
        }
        "longitudinal_slip_dry" => {
            "Dry asphalt should transmit much more drive force than ice at the same slip command."
        }
        "longitudinal_slip_ice" => {
            "Ice should cap force hard; high slip should not magically create traction."
        }
        "ice_large_patch_grip" => {
            "Even with a large low-pressure patch, ice should still slide and cap force far below the same tire on dry asphalt."
        }
        "adhesion_sliding_continuity" => {
            "Adhesion-to-sliding checks should sweep through the friction transition without force, aligning moment, or sign-convention jumps."
        }
        "camber_pneumatic_trail" => {
            "Camber/slip angle should produce lateral force plus a pneumatic-trail aligning moment."
        }
        "load_combined_slip_law" => {
            "Load/combined-slip checks should show sub-linear load sensitivity, low-pressure low-μ stiffness gain, one shared friction budget, and smooth saturation."
        }
        "wet_film_slip" => {
            "A water film should reduce available shear compared with the dry-asphalt case."
        }
        "loose_soil_slip" => {
            "Loose soil should show higher rolling loss / lower usable drive than dry asphalt."
        }
        "soil_normal_shear_coupling" => {
            "Bekker soil coupling should show normal load creating support/sinkage while driven slip adds shear rutting and rolling-loss feedback."
        }
        "moving_ground_belt" => {
            "A moving belt should change relative slip and report non-zero terrain work."
        }
        "heightfield_surface_velocity_map" => {
            "A heightfield surface-velocity map should change tire force and terrain work locally, proving flowing ground is sampled per contact position."
        }
        "curb_envelope_contact" => {
            "The heightfield curb should change the terrain normal and keep a multi-cell patch."
        }
        "cleat_envelope_contact" => {
            "The narrow cleat should lift the tire on a raised strip while retaining a distributed patch and visible horizontal force."
        }
        "rolling_obstacle_approach" => {
            "The FEM tire should approach the bump with an already-loaded extended tread patch."
        }
        "rolling_obstacle_crest" => {
            "At the bump crest, deformed tread-node contact should remain distributed instead of collapsing to one point."
        }
        "rolling_obstacle_exit" => {
            "After the bump, the FEM tire should recover continuously without losing finite patch/force telemetry."
        }
        "thermal_wear_cycle" => {
            "Sustained slip should heat/wear the tire, then cooling should reduce the surface temperature."
        }
        "thermal_wear_heat_peak" => {
            "The heat-peak replay should capture the hot sliding phase with layered temperature, wear, and dissipated-power telemetry."
        }
        "thermal_wear_cooling_recovery" => {
            "The cooling-recovery replay should show zero-slip cooling that lowers surface temperature without healing tread wear."
        }
        "thermal_abuse_envelope" => {
            "Cold, overheated, worn, and abused tire states should reduce grip through recorded thermal/wear state, not hidden viewer multipliers."
        }
        "thermal_wear_power_temperature" => {
            "Wear sensitivity should increase tread loss with both sliding power and hot surface temperature using explicit energy/thermal state."
        }
        "worn_surface_sensitivity" => {
            "Worn tread should reduce wet-film, ice, and loose-soil drive more than dry asphalt through tire wear and surface state."
        }
        "condition_failure_modes" => {
            "Condition failure modes should derive labels, μ, carcass stiffness, rolling loss, and force loss from thermal/wear state."
        }
        "wheel_spin_torque_balance" => {
            "Driven wheel spin should follow I·ωdot = motor torque - tire reaction torque - bearing loss, making slip an outcome of torque balance."
        }
        "vehicle_longitudinal_motion" => {
            "Vehicle speed should integrate from summed tire drive force after rolling loss and aerodynamic drag, not from a viewer velocity shortcut."
        }
        "vehicle_dynamic_load_transfer" => {
            "Dynamic vehicle load transfer should move normal load, patch length, force capacity, and suspension travel between axles without a viewer-only pose shortcut."
        }
        "thermal_rolling_carcass" => {
            "Sustained zero-slip rolling should heat the carcass through rolling resistance and casing flex, raising the pressure/stiffness factor without relying on sliding wear."
        }
        "thermal_brake_rim" => {
            "Brake/rim heat should enter the rim first, then conduct into the carcass and nudge pressure/stiffness without pretending it came from tread sliding."
        }
        "thermal_wear_convergence" => {
            "Thermal/wear convergence should keep dissipated power, surface temperature, and wear response stable as contact segments are refined."
        }
        "layered_transient_envelope" => {
            "Layered transient checks should show fast tread, slower belt, slowest carcass response, hot/soft/load slowdown, and force buildup over time."
        }
        "fem_solver_budget" => {
            "FEM solver-budget telemetry should expose substeps, XPBD iterations, node speeds, volume error, and bead drift for the same loaded tire state."
        }
        "fem_solver_stiffness_stabilizer" => {
            "Semi-implicit stiffness stabilization should reduce stiff-rubber velocity spikes while keeping high-pressure rolling finite and supported."
        }
        "fem_diagonal_newton_solver" => {
            "The diagonal Newton fixture should solve a stiff nonlinear tire mode with a real backward-Euler residual/Jacobian, reducing the residual and avoiding the explicit energy spike."
        }
        "fem_element_newton_solver" => {
            "The coupled tetra-element Newton fixture should solve all four element nodes together, proving off-diagonal residual/Jacobian coupling before the full tire mesh Newton solve lands."
        }
        "fem_patch_newton_solver" => {
            "The multi-tetra patch Newton fixture should solve connected elements with shared nodes, proving residual coupling across a small mesh patch before full tire-mesh Newton lands."
        }
        "fem_tiny_mesh_newton_solver" => {
            "The tiny full-mesh Newton fixture should solve every node of the smallest annular tetra tire mesh together, proving mesh-level residual/Jacobian plumbing before production full-mesh Newton and GPU parity."
        }
        "fem_matrix_free_newton_scale" => {
            "Matrix-free Newton scaling should probe a larger tire mesh residual/Jv without assembling a dense Jacobian while keeping GPU implicit-solver status debug-only."
        }
        "fem_matrix_free_newton_solve" => {
            "Matrix-free Newton solve should use Jv-driven Krylov iterations plus a line search to reduce the larger tire-mesh residual, while still being labelled CPU/debug until GPU sparse parity exists."
        }
        "fem_matrix_free_newton_preconditioner" => {
            "Matrix-free Newton preconditioner should sample the larger tire-mesh tangent diagonal without dense assembly, then prove the scaled Krylov correction is not worse before GPU sparse parity exists."
        }
        "fem_matrix_free_newton_step" => {
            "Matrix-free Newton step should line-search the preconditioned correction into a bounded tire state, reducing residual while keeping velocity, volume, and energy inside broad debug envelopes."
        }
        "fem_matrix_free_newton_timestep" => {
            "Matrix-free Newton timestep should exercise the opt-in TetraTireFem debug step through the shared matrix_free_newton_correction helper, with TetraTireFemSolverMode::MatrixFreeNewtonDebug (and the default-flipped TetraTireFemParams::high_pressure_implicit preset) routing step_on_terrain to it. The GPU parity row is now CpuAndGpu: tire_fem_matrix_free_newton.wgsl is wrapped by TireFemMatrixFreeNewtonGpuKernel, the host-orchestrated TireFemMatrixFreeNewtonGpuKernel::run_newton_outer_blocking drives a full GMRES+Jacobi+line-search outer iteration on GPU primitives, and both are validated on real hardware by tire_fem_matrix_free_newton_gpu_residual_matches_cpu_oracle and tire_fem_matrix_free_newton_gpu_outer_iteration_reduces_residual."
        }
        "fem_deformation_force_shape" => {
            "FEM deformation-force checks should show the same wheel command producing different tire force when low/high pressure changes the actual tetra tire shape."
        }
        "fem_contact_slip_patch" => {
            "FEM contact slip/patch checks should show low pressure growing the actual tread-node patch and high slip increasing sliding/bristle cells and heat."
        }
        "dem_soil_feedback" => {
            "DEM soil feedback should show tire contact cells transferring normal and shear impulse into grains, with slip moving grains more than rolling."
        }
        "mpm_soil_feedback" => {
            "MPM soil feedback should compact material points and export stronger horizontal material velocity under driven slip."
        }
        "pressure_patch_envelope" => {
            "The same vertical load should produce a longer/lower-pressure footprint than the high-pressure baseline."
        }
        "surface_grip_envelope" => {
            "Ice should cap usable drive force far below dry asphalt at the same slip command."
        }
        "wet_film_envelope" => {
            "A water-film surface should reduce drive force relative to dry asphalt while preserving explicit film telemetry."
        }
        "surface_transition_sequence" => {
            "A banded dry/ice/wet/packed-dirt/loose-soil/moving-belt pass should change force smoothly and expose each terrain state's contribution, including moving-belt force/work from relative ground velocity."
        }
        "rolling_loss_drag" => {
            "Rolling loss, soft-soil loss, standing-water loss, and aero drag should remain explicit positive loss terms."
        }
        "reference_shape_longitudinal_dry" => {
            "Reference-shape diagnostics should confirm smooth longitudinal force buildup/saturation without becoming the force source."
        }
        "reference_shape_combined_dry" => {
            "Combined-slip diagnostics should confirm longitudinal and lateral forces share one friction budget."
        }
        "step_torque_response" => {
            "A torque step should build effective slip and drive force with relaxation lag instead of teleporting to steady state."
        }
        "step_brake_response" => {
            "A brake step should reduce wheel speed while effective braking slip lags the raw command and respects the friction budget."
        }
        "step_steer_response" => {
            "A steering step should build lateral force and tread yaw gradually from transient tire state."
        }
        "frame_sign_convention" => {
            "Frame/sign checks should prove drive, brake, lateral, camber, trail, and moving-ground work signs before trusting tire forces."
        }
        "construction_sanity" => {
            "Static construction checks should keep loaded radius, sidewall bulge, bead drift, tread-belt support, and pressure footprint believable before dynamic traction is judged."
        }
        "contact_patch_convergence" => {
            "Patch convergence should show finite pressure bins, load accounting, force, and trail trends as contact resolution changes."
        }
        "fem_substep_convergence" => {
            "FEM substep convergence should keep hub height, deformed patch length, contact force, and tetra volume diagnostics in a broad stable envelope as solver substeps are refined."
        }
        "heightfield_resolution_convergence" => {
            "Heightfield resolution convergence should keep terrain height/normal, tire patch length, and contact force stable as the same bump is sampled on finer grids."
        }
        "road_condition_state" => {
            "Wet, ice, hot/worn, moving-ground, and loose-soil behavior should come from recorded physical state rather than surface-name multipliers."
        }
        "realism_source_classification" => {
            "Realism classification should separate native force-producing physics from diagnostics, numerical stabilizers, and visualization helpers."
        }
        "cavity_bead_sanity" => {
            "Cavity and bead checks should keep gas pressure, bead support, and decomposed radial/axial/tangential rim-seat drift bounded instead of hiding leaks or bead slip."
        }
        "cavity_coupling_effect" => {
            "Cavity coupling checks should make the named pressure-to-carcass proxy visible: zero coupling removes bead load, nominal coupling loads the carcass, and stronger coupling increases the same gas-law path."
        }
        "bead_seat_load_cases" => {
            "Bead-seat load cases should keep radial, axial, and tangential rim-seat drift bounded under inflation, vertical load, lateral load, and high pressure."
        }
        "sidewall_laminate_sanity" => {
            "Sidewall laminate checks should keep crossed ply families present while fold drift and ply stretch stay within the realtime tire envelope."
        }
        "sidewall_dynamic_load_cases" => {
            "Sidewall dynamic cases should keep laminate fold and ply strain bounded under vertical, lateral, and steered slip-angle loads."
        }
        "sidewall_laminate_coupling" => {
            "Sidewall laminate coupling checks should load the crossed bias plies and radial/axial helper plies in deterministic bulge, width-stretch, and fold fixtures, then prove the same folded sidewall is reduced by the structural projection."
        }
        "sidewall_ply_strain_xpbd" => {
            "Sidewall ply-strain XPBD checks should deliberately over-stretch the sidewall plies, then prove a structural projection reduces and bounds the tensile ply strain instead of only reporting the layout."
        }
        "sidewall_dihedral_xpbd_effect" => {
            "Sidewall dihedral XPBD checks should deliberately fold sidewall quads, then prove the structural projection reduces measured fold-angle drift without hiding the laminated-sidewall approximation."
        }
        "triangle_seam_manifold" => {
            "Triangle seam checks should keep FEM tread contact multi-patch/manifold-aware instead of collapsing to a single point at mesh seams."
        }
        "triangle_surface_stress" => {
            "Triangle surface stress checks should sample low/high-friction plate interiors plus the seam and preserve material contrast across the manifold."
        }
        "triangle_surface_force" => {
            "Triangle surface force checks should prove low/high-friction triangle materials and seam manifolds change real brush force and sliding/bristle diagnostics."
        }
        "triangle_density_convergence" => {
            "Triangle-density convergence should keep height, normal, seam material span, contact patch, and brush force stable as the same two-plate terrain is subdivided."
        }
        "triangle_large_manifold" => {
            "Large triangle-manifold checks should prove the BVH narrows a sizeable mesh while seam material span, multi-patch contact, and FEM tread-node feedback remain active."
        }
        "low_speed_robustness" => {
            "Low-speed checks should keep slip, relaxation, and brush forces finite through launch, reversal, and near-zero rolling speed."
        }
        _ => {
            "Internal tire-rig snapshot; use force, patch, heat, and pass/fail data to falsify the model."
        }
    }
}

fn tire_rig_replay_case_label(scenario_id: &str) -> &'static str {
    match scenario_id {
        "load_pressure_high" => "Pressure/load · high pressure",
        "load_pressure_low" => "Pressure/load · low pressure",
        "longitudinal_slip_dry" => "Surface slip · dry asphalt",
        "longitudinal_slip_ice" => "Surface slip · ice",
        "ice_large_patch_grip" => "Surface slip · ice large patch",
        "adhesion_sliding_continuity" => "Slip/contact · adhesion to sliding",
        "camber_pneumatic_trail" => "Slip/camber · pneumatic trail",
        "load_combined_slip_law" => "Slip/load · combined law",
        "wet_film_slip" => "Surface slip · wet film",
        "loose_soil_slip" => "Deformable surface · loose soil",
        "moving_ground_belt" => "Terrain motion · moving belt",
        "heightfield_surface_velocity_map" => "Terrain motion · heightfield velocity map",
        "curb_envelope_contact" => "Heightfield obstacle · curb",
        "cleat_envelope_contact" => "Heightfield obstacle · cleat",
        "rolling_obstacle_approach" => "Obstacle sequence · approach",
        "rolling_obstacle_crest" => "Obstacle sequence · crest",
        "rolling_obstacle_exit" => "Obstacle sequence · exit",
        "thermal_wear_cycle" => "Thermal/wear · slip cycle",
        "thermal_wear_heat_peak" => "Thermal/wear · heat peak",
        "thermal_wear_cooling_recovery" => "Thermal/wear · cooling recovery",
        "thermal_abuse_envelope" => "Thermal/wear · abuse envelope",
        "thermal_wear_power_temperature" => "Thermal/wear · power/temp sensitivity",
        "worn_surface_sensitivity" => "Thermal/wear · worn surface sensitivity",
        "condition_failure_modes" => "Thermal/wear · condition failure modes",
        "wheel_spin_torque_balance" => "Wheel dynamics · torque balance",
        "vehicle_longitudinal_motion" => "Vehicle dynamics · longitudinal motion",
        "vehicle_dynamic_load_transfer" => "Vehicle dynamics · load transfer",
        "thermal_rolling_carcass" => "Thermal/wear · rolling carcass heat",
        "thermal_brake_rim" => "Thermal/wear · brake rim heat",
        "thermal_wear_convergence" => "Thermal/wear · convergence",
        "layered_transient_envelope" => "Transient · layered tread/belt/carcass",
        "fem_solver_budget" => "Diagnostics · FEM solver budget",
        "fem_solver_stiffness_stabilizer" => "Diagnostics · FEM stiffness stabilizer",
        "fem_diagonal_newton_solver" => "Diagnostics · FEM diagonal Newton",
        "fem_element_newton_solver" => "Diagnostics · FEM element Newton",
        "fem_patch_newton_solver" => "Diagnostics · FEM patch Newton",
        "fem_tiny_mesh_newton_solver" => "Diagnostics · FEM tiny-mesh Newton",
        "fem_matrix_free_newton_scale" => "Diagnostics · FEM matrix-free Newton",
        "fem_matrix_free_newton_solve" => "Diagnostics · FEM matrix-free solve",
        "fem_matrix_free_newton_preconditioner" => "Diagnostics · FEM matrix-free preconditioner",
        "fem_matrix_free_newton_step" => "Diagnostics · FEM matrix-free step",
        "fem_matrix_free_newton_timestep" => "Diagnostics · FEM matrix-free timestep",
        "fem_deformation_force_shape" => "Diagnostics · FEM deformation force",
        "fem_contact_slip_patch" => "Diagnostics · FEM slip patch",
        "soil_normal_shear_coupling" => "Deformable surface · normal/shear coupling",
        "dem_soil_feedback" => "Deformable surface · DEM feedback",
        "mpm_soil_feedback" => "Deformable surface · MPM feedback",
        "pressure_patch_envelope" => "Envelope · pressure patch",
        "surface_grip_envelope" => "Envelope · ice grip cap",
        "wet_film_envelope" => "Envelope · wet-film grip",
        "surface_transition_sequence" => "Envelope · surface transition",
        "rolling_loss_drag" => "Envelope · rolling/aero loss",
        "reference_shape_longitudinal_dry" => "Diagnostics · longitudinal shape",
        "reference_shape_combined_dry" => "Diagnostics · combined-slip shape",
        "step_torque_response" => "Transient · torque step",
        "step_brake_response" => "Transient · brake step",
        "step_steer_response" => "Transient · steer step",
        "frame_sign_convention" => "Diagnostics · frame/sign convention",
        "construction_sanity" => "Diagnostics · construction sanity",
        "contact_patch_convergence" => "Diagnostics · patch convergence",
        "fem_substep_convergence" => "Diagnostics · FEM substep convergence",
        "heightfield_resolution_convergence" => "Diagnostics · heightfield resolution",
        "road_condition_state" => "Diagnostics · road-condition state",
        "realism_source_classification" => "Diagnostics · realism source labels",
        "cavity_bead_sanity" => "Diagnostics · cavity/bead sanity",
        "cavity_coupling_effect" => "Diagnostics · cavity coupling",
        "bead_seat_load_cases" => "Diagnostics · bead-seat load cases",
        "sidewall_laminate_sanity" => "Diagnostics · sidewall laminate",
        "sidewall_dynamic_load_cases" => "Diagnostics · sidewall dynamic loads",
        "sidewall_laminate_coupling" => "Diagnostics · sidewall laminate coupling",
        "sidewall_ply_strain_xpbd" => "Diagnostics · sidewall ply-strain XPBD",
        "sidewall_dihedral_xpbd_effect" => "Diagnostics · sidewall dihedral XPBD",
        "triangle_seam_manifold" => "Diagnostics · triangle seam manifold",
        "triangle_surface_stress" => "Diagnostics · triangle surface stress",
        "triangle_surface_force" => "Diagnostics · triangle surface force",
        "triangle_density_convergence" => "Diagnostics · triangle density",
        "triangle_large_manifold" => "Diagnostics · large triangle manifold",
        "low_speed_robustness" => "Diagnostics · low-speed robustness",
        _ => "Internal tire-rig snapshot",
    }
}

fn tire_rig_report_matches_snapshot(report_scenario_id: &str, snapshot_scenario_id: &str) -> bool {
    report_scenario_id == snapshot_scenario_id
        || (report_scenario_id == "rolling_obstacle_sequence"
            && matches!(
                snapshot_scenario_id,
                "rolling_obstacle_approach" | "rolling_obstacle_crest" | "rolling_obstacle_exit"
            ))
}

fn tire_known_approximation_panel(ui: &mut egui::Ui) {
    egui::CollapsingHeader::new("Known tire approximations")
        .default_open(false)
        .show(ui, |ui| {
            ui.add(
                egui::Label::new(
                    "Blunt do-not-overclaim ledger from molla-vehicle. These are \
                     model-quality gaps, not hidden viewer force multipliers.",
                )
                .wrap(),
            );
            for item in molla_vehicle::tire_known_approximation_ledger() {
                ui.separator();
                ui.horizontal(|ui| {
                    ui.monospace(item.item_id);
                    ui.small(format!(
                        "{} · {}",
                        item.status.label(),
                        item.gpu_status.label()
                    ));
                });
                ui.small(format!("Current: {}", item.current_model));
                ui.small(format!("Consequence: {}", item.consequence));
                ui.small(format!("Next: {}", item.next_step));
            }
        });
}

fn tire_rig_replay_panel(ui: &mut egui::Ui, state: &mut TireRigReplayPanelState) {
    egui::CollapsingHeader::new("Internal tire-rig replay")
        .default_open(false)
        .show(ui, |ui| {
            ui.add(
                egui::Label::new(
                    "Runs the built-in non-calibration rig suite from molla-vehicle. \
                     It replays deterministic load/pressure, ice, wet, soil, curb, \
                     camber, and heat/wear cases; it does not fit coefficients or \
                     import target curves.",
                )
                .wrap(),
            );
            ui.horizontal(|ui| {
                if ui.button("Run / refresh rig suite").clicked() || state.run.is_none() {
                    state.run = Some(molla_vehicle::tire_rig_default_non_calibration_run());
                    state.selected = 0;
                }
                ui.checkbox(&mut state.show_csv, "show CSV");
            });

            let Some(run) = &state.run else {
                return;
            };
            if run.snapshots.is_empty() {
                ui.small("Rig suite produced no snapshots.");
                return;
            }
            state.selected = state.selected.min(run.snapshots.len() - 1);
            egui::ComboBox::from_label("Replay case")
                .selected_text(tire_rig_replay_case_label(
                    &run.snapshots[state.selected].scenario_id,
                ))
                .show_ui(ui, |ui| {
                    for (i, snapshot) in run.snapshots.iter().enumerate() {
                        ui.selectable_value(
                            &mut state.selected,
                            i,
                            tire_rig_replay_case_label(&snapshot.scenario_id),
                        );
                    }
                });

            let snapshot = &run.snapshots[state.selected];
            ui.separator();
            readout(ui, "Runner", &run.runner_id);
            readout(ui, "Seed", &format!("{}", run.seed));
            readout(ui, "Suite pass", if run.passed() { "yes" } else { "NO" });
            readout(
                ui,
                "Sensitivity pass",
                &format!(
                    "{}/{}",
                    run.sensitivity_points
                        .iter()
                        .filter(|point| point.passed)
                        .count(),
                    run.sensitivity_points.len()
                ),
            );
            readout(
                ui,
                "Vehicle sanity",
                &format!(
                    "{}/{}",
                    run.vehicle_sanity_cases
                        .iter()
                        .filter(|case| case.acceptance_passed())
                        .count(),
                    run.vehicle_sanity_cases.len()
                ),
            );
            egui::CollapsingHeader::new("Sensitivity map")
                .default_open(false)
                .show(ui, |ui| {
                    ui.small(
                        "Broad directional checks only: these are not fitted target curves.",
                    );
                    for point in &run.sensitivity_points {
                        ui.horizontal(|ui| {
                            ui.label(if point.passed { "✓" } else { "!" });
                            ui.monospace(point.axis.label());
                            ui.label(format!(
                                "{} {:.4} → {:.4} ({} margin {:+.4})",
                                point.metric_label,
                                point.baseline_metric,
                                point.comparison_metric,
                                point.input_unit,
                                point.direction_margin()
                            ));
                        });
                    }
                });
            egui::CollapsingHeader::new("Vehicle sanity cases")
                .default_open(false)
                .show(ui, |ui| {
                    ui.small(
                        "Four-wheel falsification cases built from reusable tire APIs, not scene-local hacks.",
                    );
                    for case in &run.vehicle_sanity_cases {
                        let checks = case.acceptance_checks();
                        let passed = case.acceptance_passed();
                        ui.horizontal(|ui| {
                            ui.label(if passed { "✓" } else { "!" });
                            ui.monospace(case.scenario_id.as_str());
                            ui.label(format!(
                                "{:.1} kN total, patch {:.1} mm, slip {:+.2}, checks {}/{}",
                                case.total_vehicle_load_n / 1000.0,
                                case.observed_patch_length_m * 1000.0,
                                case.observed_slip_ratio,
                                checks.iter().filter(|check| check.passed).count(),
                                checks.len()
                            ));
                        });
                        ui.small(case.expected_behavior.as_str());
                        for check in checks.iter().filter(|check| !check.passed) {
                            ui.small(format!(
                                "! {}: {} = {:.4}",
                                check.check_id, check.metric_name, check.observed_value
                            ));
                        }
                    }
                });
            egui::CollapsingHeader::new("Viewer target coverage")
                .default_open(false)
                .show(ui, |ui| {
                    let coverage = run.viewer_target_coverage_report();
                    ui.small(
                        "PLAN_REALTIRE viewer targets mapped to deterministic replay / vehicle evidence.",
                    );
                    for check in &coverage.checks {
                        ui.horizontal(|ui| {
                            ui.label(if check.passed { "✓" } else { "!" });
                            ui.monospace(check.target_id.as_str());
                        });
                        ui.small(check.expectation.as_str());
                        ui.small(format!("evidence: {}", check.evidence));
                    }
                });
            egui::CollapsingHeader::new("Replay artifact coverage")
                .default_open(false)
                .show(ui, |ui| {
                    let coverage = run.replay_artifact_coverage_report();
                    ui.small(
                        "Compact replay snapshot audit: commands, terrain, provenance, \
                         force/moment, patch, thermal/wear, solver budget, CPU/GPU columns, \
                         and known-approximation rows must stay present for reproducible \
                         falsification.",
                    );
                    ui.horizontal(|ui| {
                        ui.label(if coverage.passed() { "✓" } else { "!" });
                        ui.monospace("replay_artifact_coverage");
                        ui.label(format!(
                            "{} snapshots · {} header columns · max row {} B",
                            coverage.snapshot_count,
                            coverage.csv_header_columns,
                            coverage.max_csv_row_bytes
                        ));
                    });
                    ui.small(format!(
                        "command_rows={} terrain_rows={} provenance_rows={} force_rows={} patch_rows={}",
                        coverage.command_input_rows,
                        coverage.terrain_state_rows,
                        coverage.provenance_rows,
                        coverage.force_moment_rows,
                        coverage.patch_telemetry_rows
                    ));
                    ui.small(format!(
                        "thermal/wear_rows={} solver_budget_rows={} cpu_gpu_delta_columns={} known_approximations={}",
                        coverage.thermal_wear_rows,
                        coverage.solver_budget_rows,
                        coverage.cpu_gpu_delta_columns_present,
                        coverage.known_approximation_rows
                    ));
                    for check in coverage.acceptance_checks() {
                        ui.horizontal(|ui| {
                            ui.label(if check.passed { "✓" } else { "!" });
                            ui.monospace(check.check_id.as_str());
                            ui.small(format!(
                                "{} = {:.4}",
                                check.metric_name, check.observed_value
                            ));
                        });
                        ui.small(check.expectation.as_str());
                    }
                });
            egui::CollapsingHeader::new("Text artifact reports")
                .default_open(false)
                .show(ui, |ui| {
                    let rig_text = run.to_text_report();
                    let gpu_text = molla_vehicle::tire_gpu_parity_text_report();
                    let approximation_text = molla_vehicle::tire_known_approximation_text_report();
                    let sensitivity_text = molla_vehicle::tire_rig_standard_sensitivity_text_report();
                    ui.small(
                        "Compact text artifacts exported by molla-vehicle for logs/CI. \
                         These mirror the viewer truth labels so replay files carry \
                         GPU parity, approximation, sensitivity, and preset-governance context.",
                    );
                    ui.horizontal(|ui| {
                        ui.label(if rig_text.contains("non_calibration=true")
                            && rig_text.contains("gpu_parity=")
                            && rig_text.contains("known_approximation=")
                            && rig_text.contains("preset_governance=")
                        {
                            "✓"
                        } else {
                            "!"
                        });
                        ui.monospace("tire_rig_run_text_report");
                        ui.small(format!("{} lines", rig_text.lines().count()));
                    });
                    ui.horizontal(|ui| {
                        ui.label(if !gpu_text.is_empty() && gpu_text.contains("status=") {
                            "✓"
                        } else {
                            "!"
                        });
                        ui.monospace("tire_gpu_parity_text_report");
                        ui.small(format!("{} rows", gpu_text.lines().count()));
                    });
                    ui.horizontal(|ui| {
                        ui.label(if !approximation_text.is_empty()
                            && approximation_text.contains("next_step=")
                        {
                            "✓"
                        } else {
                            "!"
                        });
                        ui.monospace("tire_known_approximation_text_report");
                        ui.small(format!("{} rows", approximation_text.lines().count()));
                    });
                    ui.horizontal(|ui| {
                        ui.label(if !sensitivity_text.is_empty()
                            && sensitivity_text.contains("metric=")
                        {
                            "✓"
                        } else {
                            "!"
                        });
                        ui.monospace("tire_rig_standard_sensitivity_text_report");
                        ui.small(format!("{} rows", sensitivity_text.lines().count()));
                    });
                });
            readout(ui, "Physics path", &snapshot.physics_path);
            ui.add(egui::Label::new(tire_rig_replay_expected_behavior(&snapshot.scenario_id)).wrap());
            egui::CollapsingHeader::new("Acceptance envelopes")
                .default_open(false)
                .show(ui, |ui| {
                    ui.small(
                        "Broad non-calibration pass/fail checks for this replay snapshot. \
                         These are qualitative sanity envelopes, not fitted target data.",
                    );
                    let mut shown = 0usize;
                    for report in run
                        .reports
                        .iter()
                        .filter(|report| {
                            tire_rig_report_matches_snapshot(
                                &report.snapshot.scenario_id,
                                &snapshot.scenario_id,
                            )
                        })
                    {
                        shown += 1;
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label(if report.passed() { "✓" } else { "!" });
                            ui.monospace(report.snapshot.scenario_id.as_str());
                            ui.small(format!(
                                "{}/{} checks pass",
                                report.checks.iter().filter(|check| check.passed).count(),
                                report.checks.len()
                            ));
                        });
                        for check in &report.checks {
                            ui.horizontal(|ui| {
                                ui.label(if check.passed { "✓" } else { "!" });
                                ui.monospace(check.check_id.as_str());
                                ui.label(format!(
                                    "{} = {:.4}",
                                    check.metric_name, check.observed_value
                                ));
                                if let Some(min) = check.min_value {
                                    ui.small(format!("min {:.4}", min));
                                }
                                if let Some(max) = check.max_value {
                                    ui.small(format!("max {:.4}", max));
                                }
                            });
                            ui.small(check.expectation.as_str());
                        }
                    }
                    if shown == 0 {
                        ui.small("No acceptance report is attached to this selected snapshot.");
                    }
                });
            readout(ui, "Tire preset", &snapshot.tire_preset);
            readout(ui, "Surface", &snapshot.surface_preset);
            readout(ui, "Terrain", &snapshot.terrain_preset);
            egui::CollapsingHeader::new("Snapshot provenance / commands")
                .default_open(false)
                .show(ui, |ui| {
                    ui.small(
                        "Replay inputs and provenance tags recorded with the snapshot. \
                         These make hand-tuned presets and numerical solver knobs explicit.",
                    );
                    readout(
                        ui,
                        "Provenance tire/surface/solver",
                        &format!(
                            "{} / {} / {}",
                            snapshot.tire_parameter_provenance.label(),
                            snapshot.surface_parameter_provenance.label(),
                            snapshot.solver_parameter_provenance.label()
                        ),
                    );
                    readout(
                        ui,
                        "Hub height / deflection",
                        &format!(
                            "{:.1} / {:.1} mm",
                            snapshot.hub_height_m * 1000.0,
                            snapshot.deflection_m * 1000.0
                        ),
                    );
                    readout(ui, "Speed", &format!("{:.2} m/s", snapshot.speed_m_per_s));
                    readout(
                        ui,
                        "Camber command",
                        &format!("{:+.2}°", snapshot.camber_angle_command_rad.to_degrees()),
                    );
                    readout(
                        ui,
                        "Terrain height",
                        &format!("{:+.3} m", snapshot.terrain_height_m),
                    );
                    readout(
                        ui,
                        "Terrain normal",
                        &format!(
                            "[{:+.2}, {:+.2}, {:+.2}]",
                            snapshot.terrain_normal[0],
                            snapshot.terrain_normal[1],
                            snapshot.terrain_normal[2]
                        ),
                    );
                    readout(
                        ui,
                        "Terrain surface velocity",
                        &format!(
                            "[{:+.2}, {:+.2}, {:+.2}] m/s",
                            snapshot.terrain_surface_velocity_m_per_s[0],
                            snapshot.terrain_surface_velocity_m_per_s[1],
                            snapshot.terrain_surface_velocity_m_per_s[2]
                        ),
                    );
                });
            readout(
                ui,
                "Moving ground work",
                &format!("{:+.0} W", snapshot.moving_ground_power_w),
            );
            readout(ui, "Load", &format!("{:.1} kN", snapshot.load_n / 1000.0));
            readout(ui, "Pressure", &format!("{:.0} kPa", snapshot.pressure_pa / 1000.0));
            readout(
                ui,
                "Slip κ / α",
                &format!(
                    "{:+.3} / {:+.1}°",
                    snapshot.slip_ratio_command,
                    snapshot.slip_angle_command_rad.to_degrees()
                ),
            );
            readout(
                ui,
                "Force Fx/Fy/Fz",
                &format!(
                    "{:+.1}/{:+.1}/{:+.1} kN",
                    snapshot.telemetry.force_n[0] / 1000.0,
                    snapshot.telemetry.force_n[2] / 1000.0,
                    snapshot.telemetry.force_n[1] / 1000.0
                ),
            );
            readout(
                ui,
                "Patch",
                &format!(
                    "{:.1} mm  {} cells",
                    snapshot.telemetry.patch.patch_length_m * 1000.0,
                    snapshot.telemetry.patch.contact_count
                ),
            );
            readout(
                ui,
                "Friction use",
                &format!(
                    "{:.1}/{:.1} kN",
                    snapshot.telemetry.consumed_friction_budget_n / 1000.0,
                    snapshot.telemetry.available_friction_budget_n / 1000.0
                ),
            );
            egui::CollapsingHeader::new("Patch / friction telemetry")
                .default_open(false)
                .show(ui, |ui| {
                    ui.small(
                        "Unit-labelled contact-patch accounting from the selected replay \
                         snapshot. This makes adhesion/sliding and pressure accounting visible.",
                    );
                    let patch = snapshot.telemetry.patch;
                    readout(
                        ui,
                        "Contact / sliding cells",
                        &format!("{}/{}", patch.contact_count, patch.sliding_count),
                    );
                    readout(
                        ui,
                        "Contact area",
                        &format!("{:.1} cm²", patch.contact_area_m2 * 10_000.0),
                    );
                    readout(
                        ui,
                        "Mean / peak pressure",
                        &format!(
                            "{:.1} / {:.1} kPa",
                            patch.mean_normal_pressure_pa / 1000.0,
                            patch.peak_normal_pressure_pa / 1000.0
                        ),
                    );
                    readout(
                        ui,
                        "Center / resultant / trail",
                        &format!(
                            "{:+.1} / {:+.1} / {:+.1} mm",
                            patch.contact_center_x_m * 1000.0,
                            patch.force_resultant_x_m * 1000.0,
                            patch.pneumatic_trail_m * 1000.0
                        ),
                    );
                    readout(
                        ui,
                        "Adhesion / sliding",
                        &format!(
                            "{:.0}% / {:.0}%",
                            patch.adhesion_fraction * 100.0,
                            patch.sliding_fraction * 100.0
                        ),
                    );
                    readout(
                        ui,
                        "Slip speeds",
                        &format!(
                            "{:+.2} long, {:+.2} lat m/s",
                            snapshot.telemetry.slip_speed_m_per_s,
                            snapshot.telemetry.lateral_slip_speed_m_per_s
                        ),
                    );
                    readout(
                        ui,
                        "Effective stiffness",
                        &format!(
                            "{:.1} kN/m long, {:.1} kN/rad corner",
                            snapshot.telemetry.effective_longitudinal_stiffness_n_per_m / 1000.0,
                            snapshot.telemetry.effective_cornering_stiffness_n_per_rad / 1000.0
                        ),
                    );
                    readout(
                        ui,
                        "Rolling resistance",
                        &format!("{:+.1} N", snapshot.telemetry.rolling_resistance_force_n),
                    );
                });
            let energy_audit = snapshot.energy_sign_audit();
            readout(
                ui,
                "Energy/sign pass",
                if energy_audit.passed() { "yes" } else { "NO" },
            );
            readout(
                ui,
                "Friction residual",
                &format!("{:+.1} N", energy_audit.friction_budget_residual_n),
            );
            readout(
                ui,
                "Moving-ground power",
                &format!(
                    "{:+.0} W ({})",
                    energy_audit.moving_ground_power_w,
                    if energy_audit.moving_ground_work_sign_ok {
                        "sign ok"
                    } else {
                        "SIGN FAIL"
                    }
                ),
            );
            readout(
                ui,
                "Trail / Mz",
                &format!(
                    "{:.1} mm / {:+.1} N·m",
                    snapshot.telemetry.patch.pneumatic_trail_m * 1000.0,
                    snapshot.telemetry.aligning_moment_nm
                ),
            );
            readout(
                ui,
                "Trail sign audit",
                &format!(
                    "expected {:+.1} N·m, reported {:+.1} N·m ({})",
                    energy_audit.expected_pneumatic_aligning_moment_nm,
                    energy_audit.reported_aligning_moment_nm,
                    if energy_audit.aligning_moment_sign_ok {
                        "sign ok"
                    } else {
                        "SIGN FAIL"
                    }
                ),
            );
            let dimensionless = snapshot.dimensionless_sanity_metrics(TIRE_RADIUS);
            egui::CollapsingHeader::new("Dimensionless sanity")
                .default_open(false)
                .show(ui, |ui| {
                    ui.small(
                        "Broad data-free ratios: these catch impossible forces, pressure \
                         accounting, slip energy, and CPU/GPU parity without fitting a tire.",
                    );
                    readout(ui, "|Fx|/Fz", &format!("{:.3}", dimensionless.abs_fx_over_fz));
                    readout(ui, "|Fy|/Fz", &format!("{:.3}", dimensionless.abs_fy_over_fz));
                    readout(
                        ui,
                        "Friction budget ratio",
                        &format!("{:.3}", dimensionless.consumed_over_available_friction),
                    );
                    readout(
                        ui,
                        "Patch length/R",
                        &format!("{:.3}", dimensionless.patch_length_over_radius),
                    );
                    readout(
                        ui,
                        "Pressure area/load",
                        &format!("{:.3}", dimensionless.contact_pressure_area_over_load),
                    );
                    readout(
                        ui,
                        "Slip energy/(load·R)",
                        &format!("{:.3}", dimensionless.dissipated_energy_over_load_radius),
                    );
                    if let Some(relative) = dimensionless.cpu_gpu_relative_force_error {
                        readout(ui, "CPU/GPU relative error", &format!("{:.3}%", relative * 100.0));
                    } else {
                        readout(ui, "CPU/GPU relative error", "not recorded for this replay");
                    }
                });
            readout(
                ui,
                "Dissipated",
                &format!(
                    "{:.0} W  {:.2} J",
                    snapshot.telemetry.dissipated_power_w,
                    snapshot.telemetry.dissipated_energy_j
                ),
            );
            if let Some(surface_c) = snapshot.surface_temperature_c {
                readout(ui, "Surface temp", &format!("{surface_c:.1} °C"));
            }
            egui::CollapsingHeader::new("Thermal / wear snapshot")
                .default_open(false)
                .show(ui, |ui| {
                    ui.small(
                        "State-derived condition telemetry recorded by the replay snapshot. \
                         These values are model state, not hidden grip multipliers.",
                    );
                    if let Some(surface_c) = snapshot.surface_temperature_c {
                        readout(ui, "Surface temperature", &format!("{surface_c:.1} °C"));
                    } else {
                        readout(ui, "Surface temperature", "not recorded for this replay");
                    }
                    if let Some(tread_c) = snapshot.tread_temperature_c {
                        readout(ui, "Tread temperature", &format!("{tread_c:.1} °C"));
                    } else {
                        readout(ui, "Tread temperature", "not recorded for this replay");
                    }
                    if let Some(carcass_c) = snapshot.carcass_temperature_c {
                        readout(ui, "Carcass temperature", &format!("{carcass_c:.1} °C"));
                    } else {
                        readout(ui, "Carcass temperature", "not recorded for this replay");
                    }
                    if let Some(depth_m) = snapshot.tread_depth_m {
                        readout(ui, "Tread depth", &format!("{:.2} mm", depth_m * 1000.0));
                    } else {
                        readout(ui, "Tread depth", "not recorded for this replay");
                    }
                    if let Some(wear) = snapshot.wear_ratio {
                        readout(ui, "Wear ratio", &format!("{:.1}%", wear * 100.0));
                    } else {
                        readout(ui, "Wear ratio", "not recorded for this replay");
                    }
                });
            if let Some(wear) = snapshot.wear_ratio {
                readout(ui, "Wear ratio", &format!("{:.1}%", wear * 100.0));
            }
            if let Some(delta) = snapshot.cpu_gpu_force_delta_n {
                readout(
                    ui,
                    "CPU/GPU force Δ",
                    &format!(
                        "[{:+.1}, {:+.1}, {:+.1}] N",
                        delta[0], delta[1], delta[2]
                    ),
                );
                if let Some(relative) = snapshot.cpu_gpu_relative_force_error {
                    readout(
                        ui,
                        "CPU/GPU rel Δ",
                        &format!("{:.3}%", relative * 100.0),
                    );
                }
            } else {
                readout(ui, "CPU/GPU force Δ", "not recorded for this replay");
            }
            if let Some(budget) = snapshot.solver_budget {
                readout(
                    ui,
                    "FEM solver",
                    &format!(
                        "{} substeps, XPBD v/s {}/{}, vmax {:.2} m/s",
                        budget.substeps_per_frame,
                        budget.xpbd_volume_iterations,
                        budget.xpbd_structural_iterations,
                        budget.max_node_speed_m_per_s
                    ),
                );
                readout(
                    ui,
                    "FEM errors",
                    &format!(
                        "vol {:.3e} m³, bead {:.2} mm, corr {:.2}/{:.2}/{:.2} mm",
                        budget.max_tet_volume_error_m3,
                        budget.max_bead_mount_error_m * 1000.0,
                        budget.last_xpbd_volume_correction_m * 1000.0,
                        budget.last_xpbd_bead_correction_m * 1000.0,
                        budget.last_xpbd_structural_correction_m * 1000.0
                    ),
                );
            }
            ui.separator();
            tire_known_approximation_panel(ui);
            if state.show_csv {
                ui.separator();
                ui.monospace(molla_vehicle::TireRigSnapshot::CSV_HEADER);
                ui.monospace(snapshot.to_csv_row());
            }
        });
}

/// Per-frame state for `Scene::TireDynamics`. Four volumetric tetra-FEM
/// carcasses at fixed inflation pressures plus a sweep counter for one
/// animated "demo" tire that cycles its pressure to make the inflation/
/// deflation transition visible.
#[derive(Resource)]
struct TireDynamicsState {
    tires: Vec<molla_vehicle::TetraTireFem>,
    /// Viewer runs the tetra node update on the new WGSL kernel by default.
    use_gpu: bool,
    compare_cpu: bool,
    resident_cavity: bool,
    kernel: Option<molla::kernels::TireFemGpuKernel>,
    /// Stage (g) — per-tire persistent GPU buffers. `None` until the
    /// first GPU step builds them; reused thereafter.
    persistent_meshes: Vec<Option<molla::kernels::TireFemPersistentMesh>>,
    /// Stage (D/f) — preferred many-tire resident atlas owner. This keeps all
    /// Scene-120 tire topology/state in one GPU atlas and advances nodes +
    /// hubs through fused dispatches.
    use_resident_atlas: bool,
    resident_atlas: Option<molla::kernels::TireFemGpuAtlasBatch>,
    /// Stage (vehicle/FEM adoption) — when the resident atlas path is active,
    /// prefer the combined resident vehicle+brush+FEM owner so the viewer uses
    /// the same long-lived runtime object as the reusable vehicle tire path.
    use_resident_vehicle_owner: bool,
    resident_vehicle_fem: Option<molla::kernels::TireResidentVehicleFemAtlasOwner>,
    prepared_chassis: Option<molla::kernels::PreparedTireVehicleFemStep>,
    prepared_readback: Option<molla::kernels::PreparedTireVehicleFemReadback>,
    vehicle_kernel: Option<molla::kernels::TireVehicleYawGpuKernel>,
    brush_kernel: Option<molla::kernels::TireBrushGpuKernel>,
    last_brush_wheels: usize,
    last_resident_vehicle_valid: bool,
    last_resident_vehicle_x_m: f32,
    last_resident_vehicle_z_m: f32,
    last_resident_vehicle_y_m: f32,
    last_resident_vehicle_yaw_rad: f32,
    last_resident_vehicle_vx_m_s: f32,
    last_resident_vehicle_vz_m_s: f32,
    last_resident_vehicle_vy_m_s: f32,
    last_resident_vehicle_yaw_rate_rad_s: f32,
    last_resident_wheel_spin_finite: bool,
    last_resident_suspension_finite: bool,
    last_backend: &'static str,
    last_gpu_error: Option<String>,
    /// Pressures (Pa) the four `tires` were created with. The fifth
    /// entry (`animated_pressure`) drives `tires[4]`.
    pressures_pa: [f64; 5],
    /// Elapsed scene time (s) — drives the animated tire's sweep.
    elapsed: f32,
    /// Scene-120 CPU reference deltas for the animated tire when the GPU FEM
    /// path is active. This is a diagnostic only; the viewer still advances
    /// with the GPU state.
    last_gpu_delta_valid: bool,
    last_gpu_delta_node_m: f32,
    last_gpu_delta_hub_m: f32,
    last_gpu_delta_cavity_ratio: f32,
    replay_panel: TireRigReplayPanelState,
}

#[derive(Clone, Copy, Debug, Default)]
struct TireDynamicsResidentVehicleTelemetry {
    valid: bool,
    x_m: f32,
    z_m: f32,
    y_m: f32,
    yaw_rad: f32,
    vx_m_s: f32,
    vz_m_s: f32,
    vy_m_s: f32,
    yaw_rate_rad_s: f32,
    wheel_spin_finite: bool,
    suspension_finite: bool,
}

fn tire_dynamics_resident_vehicle_telemetry(
    frame: &molla::kernels::TireVehicleFemFrame,
) -> TireDynamicsResidentVehicleTelemetry {
    let planar = frame.planar_rows();
    let vertical = frame.vertical_row();
    let spin_finite = frame.spin_rows().iter().all(|r| r[3] > 0.5);
    let suspension_finite = frame.suspension_rows().iter().all(|r| r[3] > 0.5);
    TireDynamicsResidentVehicleTelemetry {
        valid: planar[1][3] > 0.5 && vertical[3] > 0.5 && spin_finite && suspension_finite,
        x_m: planar[0][0],
        z_m: planar[0][1],
        y_m: vertical[0],
        yaw_rad: planar[1][0],
        vx_m_s: planar[0][2],
        vz_m_s: planar[0][3],
        vy_m_s: vertical[1],
        yaw_rate_rad_s: planar[1][1],
        wheel_spin_finite: spin_finite,
        suspension_finite,
    }
}

fn tire_dynamics_validate_frame(
    tires: &[molla_vehicle::TetraTireFem],
    frame: &molla::kernels::TireVehicleFemFrame,
) -> Result<(), &'static str> {
    if !tire_dynamics_resident_vehicle_telemetry(frame).valid
        || frame.hub_rows().len() != 2 * tires.len()
        || frame.spin_rows().len() != tires.len()
        || frame.suspension_rows().len() != tires.len()
    {
        return Err("invalid resident vehicle frame");
    }
    for rows in [
        frame.hub_rows(),
        frame.planar_rows(),
        frame.spin_rows(),
        frame.suspension_rows(),
        std::slice::from_ref(&frame.vertical_row()),
    ] {
        if !rows.iter().flatten().all(|v| v.is_finite()) {
            return Err("nonfinite resident vehicle frame");
        }
    }
    for (i, tire) in tires.iter().enumerate() {
        let nodes = tire.cavity_surface().node_count;
        for rows in [frame.positions(i), frame.velocities(i)] {
            if !rows.is_some_and(|rows| {
                rows.len() == nodes && rows.iter().flatten().all(|v| v.is_finite())
            }) {
                return Err("invalid resident tire frame");
            }
        }
    }
    if frame.positions(tires.len()).is_some() {
        return Err("resident tire frame count mismatch");
    }
    Ok(())
}

impl TireDynamicsState {
    fn clear_prepared(&mut self) {
        self.prepared_chassis = None;
        self.prepared_readback = None;
    }
}

impl Default for TireDynamicsState {
    fn default() -> Self {
        Self {
            tires: Vec::new(),
            use_gpu: true,
            compare_cpu: false,
            resident_cavity: false,
            kernel: None,
            persistent_meshes: Vec::new(),
            use_resident_atlas: true,
            resident_atlas: None,
            use_resident_vehicle_owner: true,
            resident_vehicle_fem: None,
            prepared_chassis: None,
            prepared_readback: None,
            vehicle_kernel: None,
            brush_kernel: None,
            last_brush_wheels: 0,
            last_resident_vehicle_valid: false,
            last_resident_vehicle_x_m: 0.0,
            last_resident_vehicle_z_m: 0.0,
            last_resident_vehicle_y_m: 0.0,
            last_resident_vehicle_yaw_rad: 0.0,
            last_resident_vehicle_vx_m_s: 0.0,
            last_resident_vehicle_vz_m_s: 0.0,
            last_resident_vehicle_vy_m_s: 0.0,
            last_resident_vehicle_yaw_rate_rad_s: 0.0,
            last_resident_wheel_spin_finite: false,
            last_resident_suspension_finite: false,
            last_backend: "GPU tetra FEM (resident vehicle+FEM owner)",
            last_gpu_error: None,
            pressures_pa: [0.0, 80_000.0, 200_000.0, 320_000.0, 0.0],
            elapsed: 0.0,
            last_gpu_delta_valid: false,
            last_gpu_delta_node_m: 0.0,
            last_gpu_delta_hub_m: 0.0,
            last_gpu_delta_cavity_ratio: 0.0,
            replay_panel: Default::default(),
        }
    }
}

fn tire_dynamics_cpu_reference(
    state: &TireDynamicsState,
    tire_idx: usize,
) -> Option<molla_vehicle::TetraTireFem> {
    if !state.use_gpu || !state.compare_cpu || tire_idx != 4 {
        return None;
    }
    state.tires.get(tire_idx).cloned()
}

fn update_tire_dynamics_example(
    time: Res<Time>,
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut state: ResMut<TireDynamicsState>,
) {
    if sim.scene != Scene::TireDynamics {
        return;
    }
    if state.tires.is_empty() {
        // Hubs spaced 1.5 m apart along X. Snapshot the pressures
        // first so the push loop doesn't alias `state`.
        let xs = [-2.5_f64, -1.0, 0.5, 2.0, 3.5];
        let pressures = state.pressures_pa;
        for (i, &x) in xs.iter().enumerate() {
            state.tires.push(molla_vehicle::TetraTireFem::new(
                x,
                pressures[i],
                tire_carcass_params(pressures[i]),
            ));
        }
        state.persistent_meshes = (0..state.tires.len()).map(|_| None).collect();
        state.resident_atlas = None;
        state.resident_vehicle_fem = None;
        state.elapsed = 0.0;
        state.clear_prepared();
    }
    if run.paused {
        return;
    }
    state.elapsed += time.delta_secs();
    // Animated tire pressure cycles 0 → 350 kPa → 0 with a 12-s period
    // so the audience sees one full inflate/deflate transition.
    let phase = (state.elapsed as f64 / 12.0).rem_euclid(1.0);
    let pulse = if phase < 0.5 {
        phase * 2.0
    } else {
        2.0 - phase * 2.0
    };
    let animated = pulse * 350_000.0;
    state.pressures_pa[4] = animated;
    if let Some(t) = state.tires.get_mut(4) {
        t.set_pressure(animated);
    }
    let dt = 1.0 / 120.0;
    if state.use_gpu {
        let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) else {
            state.last_backend = "CPU fallback (no GPU queue)";
            for tire in state.tires.iter_mut() {
                tire.step(dt);
            }
            return;
        };
        if state.kernel.is_none() {
            state.kernel = Some(molla::kernels::TireFemGpuKernel::new(device));
        }
        let Some(kernel) = state.kernel.take() else {
            return;
        };
        let mut gpu_ok = true;
        let mut gpu_error: Option<String> = None;
        let mut gpu_delta: Option<(f32, f32, f32)> = None;
        state.last_resident_vehicle_valid = false;
        state.last_resident_wheel_spin_finite = false;
        state.last_resident_suspension_finite = false;
        let gpu_substeps = 24;
        let gpu_dt = dt / gpu_substeps as f64;
        if state.persistent_meshes.len() != state.tires.len() {
            state.persistent_meshes = (0..state.tires.len()).map(|_| None).collect();
        }
        if state.use_resident_atlas && state.use_resident_vehicle_owner {
            let cpu_reference = tire_dynamics_cpu_reference(&state, 4);
            if state.vehicle_kernel.is_none() {
                state.vehicle_kernel = Some(molla::kernels::TireVehicleYawGpuKernel::new(device));
            }
            if state.brush_kernel.is_none() {
                state.brush_kernel = Some(molla::kernels::TireBrushGpuKernel::new(device));
            }
            let needs_rebuild = state
                .resident_vehicle_fem
                .as_ref()
                .map(|owner| owner.tire_count() != state.tires.len())
                .unwrap_or(true);
            if needs_rebuild {
                state.clear_prepared();
                let inputs = state
                    .tires
                    .iter()
                    .map(|tire| build_tire_fem_gpu_input(tire, gpu_dt))
                    .collect::<Vec<_>>();
                let spin_input = tire_dynamics_spin_input(&state.tires, gpu_dt);
                let suspension_input = tire_dynamics_suspension_input(&state.tires);
                let brush_input = tire_dynamics_brush_input(&state.tires);
                let planar_input = tire_dynamics_planar_input(&state.tires, gpu_dt);
                let vertical_input = tire_dynamics_vertical_input(gpu_dt);
                if let Some(vehicle_kernel) = state.vehicle_kernel.as_ref() {
                    match kernel.create_resident_vehicle_fem_atlas_owner(
                        vehicle_kernel,
                        device,
                        queue,
                        &inputs,
                        &spin_input,
                        &suspension_input,
                        brush_input,
                        &planar_input,
                        &vertical_input,
                    ) {
                        Ok(owner) => {
                            state.resident_vehicle_fem = Some(owner);
                            state.resident_atlas = None;
                            for slot in state.persistent_meshes.iter_mut() {
                                *slot = None;
                            }
                        }
                        Err(err) => {
                            gpu_ok = false;
                            gpu_error = Some(err.to_string());
                        }
                    }
                } else {
                    gpu_ok = false;
                    gpu_error = Some("resident vehicle kernel unavailable".to_owned());
                }
            }
            if gpu_ok {
                let params = state
                    .tires
                    .iter()
                    .map(|tire| {
                        if state.resident_cavity {
                            tire_fem_gpu_params_with_load(tire, gpu_dt, 0.)
                        } else {
                            tire_fem_gpu_params(tire, gpu_dt)
                        }
                    })
                    .collect::<Vec<_>>();
                let hub_params = state
                    .tires
                    .iter()
                    .map(|tire| tire_fem_hub_params(tire, gpu_dt))
                    .collect::<Vec<_>>();
                if let Some(mut owner) = state.resident_vehicle_fem.take() {
                    let mut prepared_chassis = state.prepared_chassis.take();
                    let mut prepared_readback = state.prepared_readback.take();
                    let pressures = state.pressures_pa;
                    for (i, pressure) in pressures.iter().copied().enumerate() {
                        if let Some(row) = owner.vehicle.brush.brush_input.wheel_controls.get_mut(i)
                        {
                            row[2] = pressure as f32;
                        }
                    }
                    if let (Some(vehicle_kernel), Some(brush_kernel)) =
                        (state.vehicle_kernel.as_ref(), state.brush_kernel.as_ref())
                    {
                        let mut last_brush_wheels = 0;
                        let prepare_result = (|| -> molla::core::Result<()> {
                            if prepared_chassis.is_none() {
                                prepared_readback = None;
                                prepared_chassis =
                                    Some(molla::kernels::PreparedTireVehicleFemStep::new(
                                        &sim.engine,
                                        &kernel,
                                        vehicle_kernel,
                                        brush_kernel,
                                        &owner,
                                        None,
                                        gpu_dt as f32,
                                        &params,
                                        &hub_params,
                                        TIRE_DYNAMICS_BRUSH_FORCE_SCALE,
                                        64 * 1024,
                                    )?);
                                if state.resident_cavity {
                                    let surfaces = state
                                        .tires
                                        .iter()
                                        .map(|t| t.cavity_surface())
                                        .collect::<Vec<_>>();
                                    prepared_chassis.as_mut().unwrap().prepare_cavities(
                                        &sim.engine,
                                        &owner,
                                        &surfaces,
                                        1024 * 1024,
                                    )?;
                                }
                            }
                            prepared_chassis.as_mut().unwrap().update_configuration(
                                &sim.engine,
                                &owner,
                                &params,
                                &hub_params,
                                TIRE_DYNAMICS_BRUSH_FORCE_SCALE,
                            )?;
                            if prepared_readback.is_none() {
                                prepared_readback =
                                    Some(molla::kernels::PreparedTireVehicleFemReadback::new(
                                        &sim.engine,
                                        prepared_chassis.as_ref().unwrap(),
                                        &owner,
                                        1024 * 1024,
                                    )?);
                            }
                            Ok(())
                        })();
                        if let Err(err) = prepare_result {
                            gpu_ok = false;
                            gpu_error = Some(err.to_string());
                        }
                        if gpu_ok && state.resident_cavity {
                            match prepared_chassis.as_mut().unwrap().step_cavity_frame(
                                &sim.engine,
                                &mut owner,
                                gpu_substeps,
                            ) {
                                Ok(brush) => last_brush_wheels = brush.wheel_count,
                                Err(err) => {
                                    gpu_ok = false;
                                    gpu_error = Some(err.to_string());
                                }
                            }
                        }
                        for _ in 0..if state.resident_cavity {
                            0
                        } else {
                            gpu_substeps
                        } {
                            if !gpu_ok {
                                break;
                            }
                            match prepared_chassis
                                .as_ref()
                                .unwrap()
                                .step(&sim.engine, &mut owner)
                            {
                                Ok(brush) => {
                                    last_brush_wheels = brush.wheel_count;
                                }
                                Err(err) => {
                                    gpu_ok = false;
                                    gpu_error = Some(err.to_string());
                                    break;
                                }
                            }
                        }
                        if gpu_ok {
                            match prepared_readback.as_mut().unwrap().read_blocking(
                                &sim.engine,
                                prepared_chassis.as_ref().unwrap(),
                                &owner,
                            ) {
                                Ok(frame) => {
                                    if let Err(err) =
                                        tire_dynamics_validate_frame(&state.tires, frame)
                                    {
                                        gpu_ok = false;
                                        gpu_error = Some(err.to_owned());
                                    } else {
                                        let telemetry =
                                            tire_dynamics_resident_vehicle_telemetry(frame);
                                        state.last_resident_vehicle_valid = telemetry.valid;
                                        state.last_resident_vehicle_x_m = telemetry.x_m;
                                        state.last_resident_vehicle_z_m = telemetry.z_m;
                                        state.last_resident_vehicle_y_m = telemetry.y_m;
                                        state.last_resident_vehicle_yaw_rad = telemetry.yaw_rad;
                                        state.last_resident_vehicle_vx_m_s = telemetry.vx_m_s;
                                        state.last_resident_vehicle_vz_m_s = telemetry.vz_m_s;
                                        state.last_resident_vehicle_vy_m_s = telemetry.vy_m_s;
                                        state.last_resident_vehicle_yaw_rate_rad_s =
                                            telemetry.yaw_rate_rad_s;
                                        state.last_resident_wheel_spin_finite =
                                            telemetry.wheel_spin_finite;
                                        state.last_resident_suspension_finite =
                                            telemetry.suspension_finite;
                                        for tire_idx in 0..state.tires.len() {
                                            let hub_after = resident_atlas_hub_state(
                                                frame.hub_rows(),
                                                tire_idx,
                                            );
                                            if let Err(err) = state.tires[tire_idx]
                                                .apply_gpu_node_state(
                                                    frame.positions(tire_idx).unwrap(),
                                                    frame.velocities(tire_idx).unwrap(),
                                                    hub_after,
                                                )
                                            {
                                                gpu_ok = false;
                                                gpu_error = Some(err.to_string());
                                                break;
                                            }
                                        }
                                        if let Some(mut cpu_reference) = cpu_reference {
                                            cpu_reference.step(dt);
                                            if let Some(animated) = state.tires.get(4) {
                                                gpu_delta = Some(tire_fem_cpu_gpu_delta(
                                                    animated,
                                                    &cpu_reference,
                                                ));
                                            }
                                        }
                                    }
                                }
                                Err(err) => {
                                    gpu_ok = false;
                                    gpu_error = Some(err.to_string());
                                }
                            }
                        }
                        state.last_brush_wheels = last_brush_wheels;
                    } else {
                        gpu_ok = false;
                        gpu_error = Some("resident vehicle/brush kernels unavailable".to_owned());
                    }
                    state.prepared_chassis = if gpu_ok { prepared_chassis } else { None };
                    state.prepared_readback = if gpu_ok { prepared_readback } else { None };
                    state.resident_vehicle_fem = Some(owner);
                } else {
                    gpu_ok = false;
                    gpu_error = Some("resident vehicle+FEM owner unavailable".to_owned());
                }
            }
        } else if state.use_resident_atlas {
            let cpu_reference = tire_dynamics_cpu_reference(&state, 4);
            state.resident_vehicle_fem = None;
            state.clear_prepared();
            let needs_rebuild = state
                .resident_atlas
                .as_ref()
                .map(|batch| batch.tire_count() != state.tires.len())
                .unwrap_or(true);
            if needs_rebuild {
                let inputs = state
                    .tires
                    .iter()
                    .map(|tire| build_tire_fem_gpu_input(tire, gpu_dt))
                    .collect::<Vec<_>>();
                match kernel.create_resident_atlas_batch(device, queue, &inputs) {
                    Ok(batch) => {
                        state.resident_atlas = Some(batch);
                        for slot in state.persistent_meshes.iter_mut() {
                            *slot = None;
                        }
                    }
                    Err(err) => {
                        gpu_ok = false;
                        gpu_error = Some(err.to_string());
                    }
                }
            }
            if gpu_ok {
                let params = state
                    .tires
                    .iter()
                    .map(|tire| tire_fem_gpu_params(tire, gpu_dt))
                    .collect::<Vec<_>>();
                let hub_params = state
                    .tires
                    .iter()
                    .map(|tire| tire_fem_hub_params(tire, gpu_dt))
                    .collect::<Vec<_>>();
                if let Some(batch) = state.resident_atlas.as_mut() {
                    for _ in 0..gpu_substeps {
                        if let Err(err) =
                            kernel.step_resident_atlas_batch(device, queue, batch, &params)
                        {
                            gpu_ok = false;
                            gpu_error = Some(err.to_string());
                            break;
                        }
                        if let Err(err) =
                            kernel.step_resident_atlas_hubs(device, queue, batch, &hub_params)
                        {
                            gpu_ok = false;
                            gpu_error = Some(err.to_string());
                            break;
                        }
                    }
                    if gpu_ok {
                        let outputs =
                            match kernel.readback_resident_atlas_batch(device, queue, batch) {
                                Ok(outputs) => outputs,
                                Err(err) => {
                                    gpu_ok = false;
                                    gpu_error = Some(err.to_string());
                                    Vec::new()
                                }
                            };
                        let hub_rows = if gpu_ok {
                            match kernel.readback_resident_atlas_hub_rows(device, queue, batch) {
                                Ok(rows) => rows,
                                Err(err) => {
                                    gpu_ok = false;
                                    gpu_error = Some(err.to_string());
                                    Vec::new()
                                }
                            }
                        } else {
                            Vec::new()
                        };
                        if gpu_ok {
                            for (tire_idx, out) in outputs.iter().enumerate() {
                                let hub_after = resident_atlas_hub_state(&hub_rows, tire_idx);
                                if let Err(err) = state.tires[tire_idx].apply_gpu_node_state(
                                    &out.positions,
                                    &out.velocities,
                                    hub_after,
                                ) {
                                    gpu_ok = false;
                                    gpu_error = Some(err.to_string());
                                    break;
                                }
                            }
                            if let Some(mut cpu_reference) = cpu_reference {
                                cpu_reference.step(dt);
                                if let Some(animated) = state.tires.get(4) {
                                    gpu_delta =
                                        Some(tire_fem_cpu_gpu_delta(animated, &cpu_reference));
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Index-based loop so we can borrow `tires[i]` and
            // `persistent_meshes[i]` separately each iteration.
            for tire_idx in 0..state.tires.len() {
                let cpu_reference = tire_dynamics_cpu_reference(&state, tire_idx);
                // Stage (g) — build the persistent buffers once per tire.
                if state.persistent_meshes[tire_idx].is_none() {
                    let tire = &state.tires[tire_idx];
                    match build_persistent_mesh(&kernel, device, queue, tire, gpu_dt) {
                        Ok(mesh) => state.persistent_meshes[tire_idx] = Some(mesh),
                        Err(err) => {
                            gpu_ok = false;
                            gpu_error = Some(err);
                            break;
                        }
                    }
                }
                let params = tire_fem_gpu_params(&state.tires[tire_idx], gpu_dt);
                let hub_params = tire_fem_hub_params(&state.tires[tire_idx], gpu_dt);
                let (out, hub_after) = {
                    let mesh = state.persistent_meshes[tire_idx].as_mut().unwrap();
                    for _ in 0..gpu_substeps {
                        kernel.step_persistent(device, queue, mesh, params);
                        kernel.step_hub_persistent(device, queue, mesh, hub_params);
                    }
                    let state_out = match kernel.readback_state(device, queue, mesh) {
                        Ok(o) => o,
                        Err(err) => {
                            gpu_ok = false;
                            gpu_error = Some(err.to_string());
                            break;
                        }
                    };
                    let hub_state = match kernel.readback_hub(device, queue, mesh) {
                        Ok(h) => h,
                        Err(err) => {
                            gpu_ok = false;
                            gpu_error = Some(err.to_string());
                            break;
                        }
                    };
                    (state_out, hub_state)
                };
                // The CPU tire's `apply_gpu_node_step` does its own hub
                // integration; since the GPU now owns the hub, just sync
                // the position/velocity arrays and the hub back to the CPU.
                if let Err(err) = state.tires[tire_idx].apply_gpu_node_state(
                    &out.positions,
                    &out.velocities,
                    hub_after,
                ) {
                    gpu_ok = false;
                    gpu_error = Some(err.to_string());
                    break;
                }
                if let Some(mut cpu_reference) = cpu_reference {
                    cpu_reference.step(dt);
                    gpu_delta = Some(tire_fem_cpu_gpu_delta(
                        &state.tires[tire_idx],
                        &cpu_reference,
                    ));
                }
            }
        }
        if gpu_ok {
            state.last_backend = if state.use_resident_atlas {
                if state.use_resident_vehicle_owner {
                    "GPU tetra FEM (resident vehicle+FEM owner)"
                } else {
                    "GPU tetra FEM (resident atlas)"
                }
            } else {
                "GPU tetra FEM (persistent)"
            };
            state.last_gpu_error = None;
            if let Some((node, hub, cavity)) = gpu_delta {
                state.last_gpu_delta_valid = true;
                state.last_gpu_delta_node_m = node;
                state.last_gpu_delta_hub_m = hub;
                state.last_gpu_delta_cavity_ratio = cavity;
            } else {
                state.last_gpu_delta_valid = false;
                state.last_gpu_delta_node_m = 0.0;
                state.last_gpu_delta_hub_m = 0.0;
                state.last_gpu_delta_cavity_ratio = 0.0;
            }
        } else {
            state.last_backend = "CPU fallback (GPU error)";
            state.last_gpu_error = gpu_error;
            state.last_gpu_delta_valid = false;
            state.last_gpu_delta_node_m = 0.0;
            state.last_gpu_delta_hub_m = 0.0;
            state.last_gpu_delta_cavity_ratio = 0.0;
            // Tear down persistent buffers — they may be inconsistent.
            for slot in state.persistent_meshes.iter_mut() {
                *slot = None;
            }
            state.resident_atlas = None;
            state.resident_vehicle_fem = None;
            state.last_brush_wheels = 0;
            state.last_resident_vehicle_valid = false;
            state.clear_prepared();
            for tire in state.tires.iter_mut() {
                tire.step(dt);
            }
        }
        state.kernel = Some(kernel);
    } else {
        state.last_backend = "CPU tetra FEM";
        state.last_gpu_error = None;
        state.last_gpu_delta_valid = false;
        state.last_gpu_delta_node_m = 0.0;
        state.last_gpu_delta_hub_m = 0.0;
        state.last_gpu_delta_cavity_ratio = 0.0;
        state.last_brush_wheels = 0;
        for tire in state.tires.iter_mut() {
            tire.step(dt);
        }
    }
}

/// Stage (g) — build the persistent GPU buffers for one tire. The snapshot
/// here is consumed only to seed the buffers; later substeps read/write
/// the GPU-side state directly via the ping-pong path.
fn build_persistent_mesh(
    kernel: &molla::kernels::TireFemGpuKernel,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tire: &molla_vehicle::TetraTireFem,
    initial_dt: f64,
) -> Result<molla::kernels::TireFemPersistentMesh, String> {
    let input = build_tire_fem_gpu_input(tire, initial_dt);
    kernel
        .create_persistent(device, queue, &input)
        .map_err(|e| e.to_string())
}

fn build_tire_fem_gpu_input(
    tire: &molla_vehicle::TetraTireFem,
    initial_dt: f64,
) -> molla::kernels::TireFemGpuInput {
    let snap = tire.gpu_snapshot();
    molla::kernels::TireFemGpuInput {
        rest_positions: snap.rest_positions,
        positions: snap.positions,
        velocities: snap.velocities,
        masses: snap.masses,
        tet_indices: snap.tet_indices,
        tet_grads: snap.tet_grads,
        tet_volumes: snap.tet_volumes,
        tet_mu: snap.tet_mu,
        tet_lambda: snap.tet_lambda,
        tet_fiber_dir: snap.tet_fiber_dir,
        tet_fiber_k1: snap.tet_fiber_k1,
        tet_fiber_k2: snap.tet_fiber_k2,
        tet_fiber_dir_b: snap.tet_fiber_dir_b,
        tet_fiber_k1_b: snap.tet_fiber_k1_b,
        tet_fiber_k2_b: snap.tet_fiber_k2_b,
        tet_fiber_dir_c: snap.tet_fiber_dir_c,
        tet_fiber_k1_c: snap.tet_fiber_k1_c,
        tet_fiber_k2_c: snap.tet_fiber_k2_c,
        tet_fiber_dir_d: snap.tet_fiber_dir_d,
        tet_fiber_k1_d: snap.tet_fiber_k1_d,
        tet_fiber_k2_d: snap.tet_fiber_k2_d,
        node_tet_offsets: snap.node_tet_offsets,
        node_tet_entries: snap.node_tet_entries,
        xpbd_edge_indices: snap.xpbd_edge_indices,
        xpbd_edge_params: snap.xpbd_edge_params,
        node_edge_offsets: snap.node_edge_offsets,
        node_edge_entries: snap.node_edge_entries,
        xpbd_dihedral_indices: snap.xpbd_dihedral_indices,
        xpbd_dihedral_params: snap.xpbd_dihedral_params,
        node_dihedral_offsets: snap.node_dihedral_offsets,
        node_dihedral_entries: snap.node_dihedral_entries,
        bead_ids: snap.bead_ids,
        bead_offsets: snap.bead_offsets,
        tread_ids: snap.tread_ids,
        params: tire_fem_gpu_params(tire, initial_dt),
    }
}

fn tire_fem_gpu_params(
    tire: &molla_vehicle::TetraTireFem,
    gpu_dt: f64,
) -> molla::kernels::TireFemGpuParams {
    let snap_cavity = tire.cavity_radial_force_per_bead_n() as f32;
    tire_fem_gpu_params_with_load(tire, gpu_dt, snap_cavity)
}

fn tire_fem_gpu_params_with_load(
    tire: &molla_vehicle::TetraTireFem,
    gpu_dt: f64,
    cavity_load: f32,
) -> molla::kernels::TireFemGpuParams {
    let hub = tire.hub();
    molla::kernels::TireFemGpuParams {
        dt: gpu_dt as f32,
        young_modulus: 4.0e6,
        poisson_ratio: 0.46,
        pressure_pa: tire.pressure() as f32,
        node_damping: 55.0,
        bead_stiffness: 5.8e4,
        floor_stiffness: 2.8e5,
        floor_damping: 1_200.0,
        pressure_area_per_tread_node: 0.0,
        hub: [hub[0] as f32, hub[1] as f32, hub[2] as f32],
        gravity_y: -9.81,
        cavity_radial_force_per_bead: cavity_load,
        belt_proxy_mu: 0.0, // Stage (c) — Holzapfel belt covers it.
        xpbd_volume_compliance: 5.0e-10,
        xpbd_bead_compliance: 5.0e-5,
        semi_implicit_stiffness_scale: tire.semi_implicit_stiffness_scale() as f32,
        xpbd_sidewall_ply_stretch_limit: 1.35,
        xpbd_sidewall_ply_compliance: 5.0e-3,
        // PLAN_TOURQUE Stage G part 2 — mirror the CPU TetraTireFem
        // spin/torque state into the GPU uniform row so a host-side
        // drive_torque integration shows up on GPU too.
        hub_omega: tire.hub_omega() as f32,
        drive_torque_n_m: tire.drive_torque_n_m() as f32,
        brake_torque_n_m: tire.brake_torque_n_m() as f32,
    }
}

fn tire_fem_hub_params(
    tire: &molla_vehicle::TetraTireFem,
    gpu_dt: f64,
) -> molla::kernels::TireFemHubParams {
    molla::kernels::TireFemHubParams {
        dt: gpu_dt as f32,
        bead_stiffness: 5.8e4,
        bead_damping: 650.0,
        hub_mass: 22.0,
        hub_damping: 260.0,
        downforce: 2_600.0,
        gravity_y: -9.81,
        hub_floor_active: 1.0,
        hub_floor_y: (tire.outer_radius() * 0.18) as f32,
        external_force_xyz: [0.0; 3],
        // Heavy lateral damping pins X/Z so the viewer's five
        // tires stay side-by-side; cornering scenes set this
        // far lower.
        lateral_damping: 2_500.0,
    }
}

fn tire_dynamics_spin_input(
    tires: &[molla_vehicle::TetraTireFem],
    gpu_dt: f64,
) -> molla::kernels::TireVehicleWheelSpinGpuInput {
    molla::kernels::TireVehicleWheelSpinGpuInput {
        wheel_state: tires
            .iter()
            .map(|tire| [0.0, tire.outer_radius() as f32, 1.2, 1.0])
            .collect(),
        wheel_commands: tires.iter().map(|_| [0.0; 4]).collect(),
        dt: gpu_dt as f32,
    }
}

fn tire_dynamics_suspension_input(
    tires: &[molla_vehicle::TetraTireFem],
) -> molla::kernels::TireVehicleSuspensionGpuInput {
    molla::kernels::TireVehicleSuspensionGpuInput {
        suspension_geometry: tires
            .iter()
            .map(|tire| {
                [
                    TIRE_DYNAMICS_VEHICLE_ATTACH_Y,
                    tire.outer_radius() as f32,
                    0.05,
                    0.055,
                ]
            })
            .collect(),
        suspension_params: tires
            .iter()
            .map(|_| [30_000.0, 2_000.0, 0.0, 0.0])
            .collect(),
        chassis_y: TIRE_DYNAMICS_VEHICLE_CHASSIS_Y,
        chassis_vy: 0.0,
    }
}

fn tire_dynamics_brush_input(
    tires: &[molla_vehicle::TetraTireFem],
) -> molla::kernels::TireVehicleBrushRowsFromSpinGpuInput {
    molla::kernels::TireVehicleBrushRowsFromSpinGpuInput {
        chassis_pose: [0.0, TIRE_DYNAMICS_VEHICLE_CHASSIS_Y, 0.0, 0.0],
        chassis_velocity: [0.0; 4],
        spin_state_rows: tires
            .iter()
            .map(|tire| [0.0, tire.outer_radius() as f32, 1.2, 1.0])
            .collect(),
        wheel_mounts: tires
            .iter()
            .map(|tire| {
                [
                    tire.hub()[0] as f32,
                    TIRE_DYNAMICS_VEHICLE_ATTACH_Y,
                    tire.hub()[2] as f32,
                    tire.outer_radius() as f32,
                ]
            })
            .collect(),
        wheel_controls: tires
            .iter()
            .map(|tire| [0.0, tire.width() as f32, tire.pressure() as f32, 0.0])
            .collect(),
        tire_rows: tires
            .iter()
            .map(|_| [64.0, 25_000.0, 120_000.0, 0.50])
            .collect(),
        material_rows: tires
            .iter()
            .map(|_| [0.90, 0.78, 80_000.0, 0.018])
            .collect(),
        extra_rows: tires.iter().map(|_| [0.0, 0.0, 1.0, 0.0]).collect(),
        suspension_rows: tires.iter().map(|_| [0.04, 0.0, 0.0, 1.0]).collect(),
    }
}

fn tire_dynamics_planar_input(
    tires: &[molla_vehicle::TetraTireFem],
    gpu_dt: f64,
) -> molla::kernels::TireVehiclePlanarFromBrushGpuInput {
    molla::kernels::TireVehiclePlanarFromBrushGpuInput {
        wheel_offsets: tires
            .iter()
            .map(|tire| [tire.hub()[0] as f32, 0.0, tire.hub()[2] as f32, 0.0])
            .collect(),
        position_xz: [0.0, 0.0],
        velocity_xz: [0.0, 0.0],
        yaw: 0.0,
        yaw_rate: 0.0,
        mass: 600.0,
        yaw_inertia: 1_800.0,
        linear_damping: 4_000.0,
        yaw_damping: 2_000.0,
        dt: gpu_dt as f32,
    }
}

fn tire_dynamics_vertical_input(gpu_dt: f64) -> molla::kernels::TireVehicleVerticalGpuInput {
    molla::kernels::TireVehicleVerticalGpuInput {
        y: TIRE_DYNAMICS_VEHICLE_CHASSIS_Y,
        vy: 0.0,
        mass: 600.0,
        gravity: 9.81,
        vertical_damping: 2_000.0,
        dt: gpu_dt as f32,
    }
}

fn resident_atlas_hub_state(hub_rows: &[[f32; 4]], tire_idx: usize) -> [f32; 8] {
    let hub = hub_rows.get(tire_idx * 2).copied().unwrap_or([0.0; 4]);
    let vel = hub_rows.get(tire_idx * 2 + 1).copied().unwrap_or([0.0; 4]);
    [
        hub[0], hub[1], hub[2], hub[3], vel[0], vel[1], vel[2], vel[3],
    ]
}

fn tire_fem_cpu_gpu_delta(
    gpu: &molla_vehicle::TetraTireFem,
    cpu: &molla_vehicle::TetraTireFem,
) -> (f32, f32, f32) {
    let mut max_node_delta = 0.0_f64;
    for (a, b) in gpu.nodes().iter().zip(cpu.nodes().iter()) {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        max_node_delta = max_node_delta.max((dx * dx + dy * dy + dz * dz).sqrt());
    }
    let gh = gpu.hub();
    let ch = cpu.hub();
    let hub_delta =
        ((gh[0] - ch[0]).powi(2) + (gh[1] - ch[1]).powi(2) + (gh[2] - ch[2]).powi(2)).sqrt();
    let gd = gpu.diagnostics();
    let cd = cpu.diagnostics();
    (
        max_node_delta as f32,
        hub_delta as f32,
        (gd.cavity_volume_ratio - cd.cavity_volume_ratio).abs() as f32,
    )
}

fn draw_tire_dynamics_gizmos(sim: Res<Sim>, state: Res<TireDynamicsState>, mut gizmos: Gizmos) {
    if sim.scene != Scene::TireDynamics {
        return;
    }
    let ring_color = Color::srgb(0.28, 0.55, 0.95);
    let contact_color = Color::srgb(0.95, 0.50, 0.30);
    for (idx, tire) in state.tires.iter().enumerate() {
        let pressure = state.pressures_pa[idx];
        let hub = tire.hub();
        let visual_hub_y = tire_visual_hub_height(pressure) as f32;
        let visual_angle = -(state.elapsed * 1.6 + idx as f32 * 0.45);

        // IMPORTANT: use the same obvious pressure silhouette as the working
        // traction scene. The tetra FEM backend is still stepped/read out, but
        // the viewer comparison must make inflation state immediately visible:
        // deflated = flat/wide patch, inflated = round/small patch.
        draw_traction_wheel(
            &mut gizmos,
            Vec3::new(hub[0] as f32, visual_hub_y, 0.0),
            pressure as f32,
            visual_angle,
            ring_color,
            contact_color,
        );

        // Extra explicit footprint stripe so pressure differences are not
        // ambiguous when the camera is pulled back.
        let q = (pressure as f32 / REFERENCE_PRESSURE_PA as f32).clamp(0.0, 1.0);
        let patch = 0.08 + (1.0 - q).powf(0.70) * 0.46;
        let half_patch = patch * 0.5;
        let half_w = TIRE_WIDTH as f32 * 0.5;
        for &z in &[-half_w, half_w] {
            gizmos.line(
                Vec3::new(hub[0] as f32 - half_patch, 0.004, z),
                Vec3::new(hub[0] as f32 + half_patch, 0.004, z),
                contact_color,
            );
        }
        // Pressure bar above each tire — height ∝ pressure / 400 kPa.
        let p_kpa = pressure / 1000.0;
        let bar_h = (p_kpa / 400.0).clamp(0.0, 1.5) as f32;
        let bar_base = Vec3::new(hub[0] as f32, 1.5, 0.0);
        let bar_top = bar_base + Vec3::new(0.0, bar_h, 0.0);
        gizmos.line(bar_base, bar_top, Color::srgb(0.95, 0.85, 0.30));

        // FEM diagnostic bars from the actual tetra tire state, not the
        // simplified silhouette: cyan shows sealed-cavity volume ratio, and
        // magenta shows the latest CPU XPBD correction magnitude / GPU volume
        // residual. These make pressure/cavity/constraint health visible
        // while the tire comparison is running.
        let diag = tire.diagnostics();
        let cavity_h = diag.cavity_volume_ratio.clamp(0.0, 1.8) as f32 * 0.45;
        let cavity_base = Vec3::new(hub[0] as f32 - 0.18, 1.5, 0.08);
        gizmos.line(
            cavity_base,
            cavity_base + Vec3::Y * cavity_h,
            Color::srgb(0.15, 0.85, 1.0),
        );
        let xpbd_mag = diag
            .last_xpbd_volume_correction_m
            .max(diag.last_xpbd_bead_correction_m)
            .max(diag.last_xpbd_structural_correction_m)
            .max(diag.max_tet_volume_error_m3.cbrt());
        let xpbd_h = (xpbd_mag as f32 / 0.025).clamp(0.0, 1.4) * 0.45;
        let xpbd_base = Vec3::new(hub[0] as f32 + 0.18, 1.5, 0.08);
        gizmos.line(
            xpbd_base,
            xpbd_base + Vec3::Y * xpbd_h,
            Color::srgb(1.0, 0.25, 0.95),
        );
    }
    // Pressure-axis ticks above the tires (50, 150, 300 kPa references).
    let tick_color = Color::srgb(0.4, 0.4, 0.45);
    for &kpa in &[100.0_f32, 200.0, 300.0] {
        let y = 1.5 + (kpa / 400.0).clamp(0.0, 1.5);
        gizmos.line(Vec3::new(-3.0, y, 0.0), Vec3::new(4.0, y, 0.0), tick_color);
    }
}

fn tire_dynamics_pane(ui: &mut egui::Ui, state: &mut TireDynamicsState) {
    ui.label("Tire pressure comparison");
    let was_gpu = state.use_gpu;
    let _ = ui.checkbox(&mut state.use_gpu, "Use GPU tire FEM kernel");
    if was_gpu != state.use_gpu {
        state.last_gpu_error = None;
        state.resident_atlas = None;
        state.resident_vehicle_fem = None;
        state.clear_prepared();
        for slot in state.persistent_meshes.iter_mut() {
            *slot = None;
        }
    }
    if state.use_gpu {
        if ui
            .checkbox(&mut state.compare_cpu, "Compare CPU reference (extra work)")
            .changed()
        {
            state.last_gpu_delta_valid = false;
        }
        let was_resident = state.use_resident_atlas;
        let _ = ui.checkbox(
            &mut state.use_resident_atlas,
            "Use resident GPU atlas batch",
        );
        if was_resident != state.use_resident_atlas {
            state.last_gpu_error = None;
            state.resident_atlas = None;
            state.resident_vehicle_fem = None;
            state.clear_prepared();
            for slot in state.persistent_meshes.iter_mut() {
                *slot = None;
            }
        }
        if state.use_resident_atlas {
            let was_vehicle_owner = state.use_resident_vehicle_owner;
            let _ = ui.checkbox(
                &mut state.use_resident_vehicle_owner,
                "Use resident vehicle+FEM owner",
            );
            if was_vehicle_owner != state.use_resident_vehicle_owner {
                state.last_gpu_error = None;
                state.resident_atlas = None;
                state.resident_vehicle_fem = None;
                state.clear_prepared();
                for slot in state.persistent_meshes.iter_mut() {
                    *slot = None;
                }
            }
            if state.use_resident_vehicle_owner {
                if ui
                    .checkbox(
                        &mut state.resident_cavity,
                        "Resident cavity loads (experimental)",
                    )
                    .changed()
                {
                    state.resident_vehicle_fem = None;
                    state.clear_prepared();
                    state.last_gpu_error = None;
                }
                if state.resident_cavity {
                    ui.small("Experimental: trajectory parity gate failed. Blocking display readback remains.");
                }
            }
        }
    }
    readout(ui, "Backend", state.last_backend);
    readout(ui, "Tires", &format!("{}", state.tires.len()));
    if state.use_gpu && state.use_resident_atlas && state.use_resident_vehicle_owner {
        readout(
            ui,
            "Resident brush wheels",
            &format!("{}", state.last_brush_wheels),
        );
        readout(
            ui,
            "Resident chassis state",
            if state.last_resident_vehicle_valid {
                "finite"
            } else {
                "not yet validated"
            },
        );
        readout(
            ui,
            "Resident chassis x/z/yaw",
            &format!(
                "{:+.3} / {:+.3} m / {:+.3} rad",
                state.last_resident_vehicle_x_m,
                state.last_resident_vehicle_z_m,
                state.last_resident_vehicle_yaw_rad
            ),
        );
        readout(
            ui,
            "Resident chassis v",
            &format!(
                "{:+.3}/{:+.3}/{:+.3} m/s, yaw {:+.3} rad/s",
                state.last_resident_vehicle_vx_m_s,
                state.last_resident_vehicle_vy_m_s,
                state.last_resident_vehicle_vz_m_s,
                state.last_resident_vehicle_yaw_rate_rad_s
            ),
        );
        readout(
            ui,
            "Resident y/spin/susp",
            &format!(
                "{:+.3} m, spin {}, suspension {}",
                state.last_resident_vehicle_y_m,
                if state.last_resident_wheel_spin_finite {
                    "finite"
                } else {
                    "pending"
                },
                if state.last_resident_suspension_finite {
                    "finite"
                } else {
                    "pending"
                }
            ),
        );
        ui.small(format!(
            "Brush→hub force scale {:.3} keeps Scene 120 primarily a pressure/FEM comparison while exercising the live resident vehicle+brush handoff.",
            TIRE_DYNAMICS_BRUSH_FORCE_SCALE
        ));
    }
    if let Some(tire) = state.tires.first() {
        readout(ui, "Tets/tire", &format!("{}", tire.tetra_count()));
    }
    if let Some(tire) = state.tires.get(4).or_else(|| state.tires.first()) {
        let d = tire.diagnostics();
        ui.separator();
        ui.label("Animated tire FEM diagnostics");
        readout(
            ui,
            "Pressure",
            &format!("{:.0} kPa", d.pressure_pa / 1000.0),
        );
        readout(
            ui,
            "Cavity V",
            &format!(
                "{:.4} / {:.4} m³ ({:.0}%)",
                d.cavity_volume_m3,
                d.cavity_rest_volume_m3,
                d.cavity_volume_ratio * 100.0
            ),
        );
        readout(
            ui,
            "Cavity P_eff",
            &format!("{:.0} kPa", d.effective_cavity_pressure_pa / 1000.0),
        );
        readout(
            ui,
            "Cavity bead F",
            &format!("{:.2} N/node", d.cavity_radial_force_per_bead_n),
        );
        readout(
            ui,
            "Tet vol residual",
            &format!("{:.2} cm³", d.max_tet_volume_error_m3 * 1.0e6),
        );
        readout(
            ui,
            "XPBD Δ vol/bead/struct",
            &format!(
                "{:.2}/{:.2}/{:.2} mm",
                d.last_xpbd_volume_correction_m * 1000.0,
                d.last_xpbd_bead_correction_m * 1000.0,
                d.last_xpbd_structural_correction_m * 1000.0
            ),
        );
        ui.small("Yellow=pressure, cyan=cavity volume ratio, magenta=XPBD correction/residual.");
    }
    if !state.compare_cpu {
        readout(ui, "CPU comparison", "disabled (opt-in)");
    } else if state.last_gpu_delta_valid {
        readout(
            ui,
            "GPU Δ node",
            &format!("{:.2} mm", state.last_gpu_delta_node_m * 1000.0),
        );
        readout(
            ui,
            "GPU Δ hub",
            &format!("{:.2} mm", state.last_gpu_delta_hub_m * 1000.0),
        );
        readout(
            ui,
            "GPU Δ cavity",
            &format!("{:.3}%", state.last_gpu_delta_cavity_ratio * 100.0),
        );
    } else if state.use_gpu {
        readout(ui, "GPU Δ FEM", "not available");
    }
    if let Some(err) = &state.last_gpu_error {
        ui.colored_label(egui::Color32::from_rgb(255, 120, 80), err);
    }
    ui.separator();
    tire_rig_replay_panel(ui, &mut state.replay_panel);
    ui.small("120 is now drawn with the same visible pressure silhouette as 121: deflated tires flatten into a wide patch, inflated tires stay round. The FEM backend still runs for diagnostics.");
}
