//! `molla_viewer` — Bevy app that runs a molla scene and visualises it.
//!
//! Scenes are picked from the dropdown in the left rail (no hotkeys).
//!
//! Mouse:
//!   * **LMB-click+drag**   grab a body and drag it (spring-pull force)
//!   * MMB-drag             pan the camera focus (XZ plane)
//!   * MMB+LMB / MMB+RMB    lift the focus (vertical Y)
//!   * LMB+RMB-drag         orbit yaw + pitch
//!   * Scroll               zoom
//!
//! Global keys:
//!   * T           toggle "activity view" — sleeping bodies turn grey
//!   * W           toggle wireframe view
//!   * Space       pause / resume the simulation
//!   * R           reset poses to the initial drop
//!   * Esc         quit

#[cfg(any(feature = "profile", feature = "log_diag"))]
use bevy::diagnostic::LogDiagnosticsPlugin;
use bevy::diagnostic::{
    DiagnosticsStore, EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin,
};
use bevy::light::{CascadeShadowConfigBuilder, NotShadowCaster};
use bevy::pbr::wireframe::{Wireframe, WireframeConfig, WireframePlugin};
use bevy::prelude::*;
#[cfg(not(feature = "profile"))]
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy::window::PrimaryWindow;
use bevy_egui::{
    EguiContexts, EguiPlugin, EguiPrimaryContextPass, EguiTextureHandle, EguiUserTextures, egui,
};
use bevy_frost::container::Normal;
use bevy_frost::pane::{Pane2, PaneAnchor, PaneResize, RailZone};
use bevy_frost::pod::Pod;
use bevy_frost::prelude::*;
use bevy_frost::ribbon::{
    RibbonCluster, RibbonDef, RibbonDrag, RibbonEdge, RibbonGlyph, RibbonItem, RibbonMode,
    RibbonOpen, RibbonPlacement, RibbonRole, draw_assembly, find_item,
};
use bevy_frost::style::AccentColor;
use bevy_frost::widget::{button, readout};
use bevy_glacial::prelude::*;
use molla::actuators::{
    Actuator, ClampingDcMotor, ClampingDcMotorConfig, ClampingMaxEffort, ClampingMaxEffortConfig,
    ClampingStage, ControllerNeuralLstm, ControllerNeuralLstmConfig, ControllerNeuralMlp,
    ControllerNeuralMlpConfig, ControllerPd, ControllerPdConfig, ControllerPipelinePd,
    ControllerPipelinePdConfig, ControllerStage, Delay, DelayConfig, LstmWeights, MlpWeights,
};
use molla::autograd::{DiffArm, FeatherstoneRevoluteChain, JacobianScratch};
use molla::compute::{Engine, MpmConfig, MpmRheology, MpmSimulation, MpmTimeStep};
use molla::fem;
use molla::geometry::CollisionPipeline;
use molla::math::{Quat as MQuat, Vec3 as MVec3};
use molla::prelude as mp;
use molla::sensors::{
    AerialLidar, AerialScanMetadata, BodyContactReading, BoundingBox2D, CameraChannels,
    ContactSensor, Dataset, ImuMount, ImuReading, ImuSensor, Lidar, RayHit, Raycaster,
    ScanMetadata, TiledCamera, TiledCameraConfig,
};
use molla::solvers::{
    FeatherstoneSolver, IkConfig, IkObjective, IkSolver, Solver as _, SphConfig, SphSimulation,
    box_tank_walls,
};

#[cfg(not(target_arch = "wasm32"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Match Newton's XPBD cloth example pattern: many small substeps
// give the constraint solver enough Gauss-Seidel sweeps per frame to
// propagate corrections corner-to-corner across a 36×36 cloth (≈50
// cells diagonal). At 2 × 1/120 dt we had 40 total sweeps per frame —
// not enough; cloth stretched and the strain limit zigzagged.
const FIXED_DT: f64 = 1.0 / 600.0;
const SUBSTEPS_PER_FRAME: usize = 10;

const PLANE_RESTITUTION: f64 = 0.05;
const PLANE_FRICTION: f64 = 0.9;
/// Global linear damping (1/s) applied to body velocities each step.
/// Models air drag / rolling resistance — without it spheres roll
/// forever and cubes hit by impulses skid off into the distance.
const LINEAR_DAMPING: f64 = 0.30;

// ── Ribbon / pane identifiers ────────────────────────────────────
// Each `Pane2::new(...)` takes a unique id; ribbons reference the
// same ids so clicking a ribbon button toggles the matching pane.
const PANE_MAIN: &str = "molla_main";
/// Single right-rail pane id. Its title and body change per scene
/// (`Lidar` shows lidar config; `IMU` shows live readings; etc.) but
/// it's the SAME pane / ribbon button — never two on the same rail.
const PANE_SCENE: &str = "molla_scene";

const RIBBON_LEFT: &str = "molla_ribbon_left";
const RIBBON_RIGHT: &str = "molla_ribbon_right";

/// Static description of the ribbon rails. molla currently uses two
/// — left for the always-present global pane, right for per-scene
/// settings (Lidar today; future sensors / actuators will register
/// more items here).
const RIBBONS: &[RibbonDef] = &[
    RibbonDef {
        id: RIBBON_LEFT,
        edge: RibbonEdge::Left,
        role: RibbonRole::Panel,
        mode: RibbonMode::ThreeSided,
        draggable: false,
        accepts: &[RIBBON_RIGHT],
    },
    RibbonDef {
        id: RIBBON_RIGHT,
        edge: RibbonEdge::Right,
        role: RibbonRole::Panel,
        mode: RibbonMode::ThreeSided,
        draggable: false,
        accepts: &[RIBBON_LEFT],
    },
];

/// Ribbon items map a pane id → ribbon edge / cluster / icon. Adding
/// a new pane is one entry here + one `if is_open(id) { Pane2::new(...).show() }`
/// arm in `control_pane`.
const RIBBON_ITEMS: &[RibbonItem] = &[
    RibbonItem {
        id: PANE_MAIN,
        ribbon: RIBBON_LEFT,
        cluster: RibbonCluster::Start,
        slot: 0,
        glyph: RibbonGlyph::Icon("apps"),
        tooltip: "molla controls",
        child_ribbon: None,
        role: None,
    },
    RibbonItem {
        id: PANE_SCENE,
        ribbon: RIBBON_RIGHT,
        cluster: RibbonCluster::Start,
        slot: 0,
        glyph: RibbonGlyph::Icon("settings"),
        tooltip: "Description and per-example settings (changes with the active scene)",
        child_ribbon: None,
        role: None,
    },
];

// --- Mouse grab tuning ---------------------------------------------
// Velocity-based pick: each frame we set the picked body's linear
// velocity so its anchor point moves toward the cursor target at a
// bounded speed. Force-based picks (spring + damper through external
// force) overshoot violently with chained equality constraints — the
// rope was the canary. Velocity-based picks are what Rapier / Bullet /
// PhysX use for interactive grabs because they're mass-independent
// and never produce impulses larger than the configured cap.
//
// `LAG_TIME` is the time we'd take to close the gap if there were no
// speed cap — i.e. v = distance / lag_time. 0.08 s feels snappy.
// `MAX_SPEED` caps the anchor speed so we never tunnel a body
// through a wall. 14 m/s is fast but well below the contact margin
// safety budget (margin 0.02 m × 240 Hz substep = 4.8 m/s recovery).
const GRAB_LAG_TIME: f64 = 0.08;
const GRAB_MAX_SPEED: f64 = 14.0;

// --- Activity view (Muli-style) ------------------------------------
// A body whose combined kinematic norm |v| + ω·R is below this is
// considered "asleep" and shown in grey. Above → coloured. R is a
// rough body radius — 0.3m here matches the typical scene scale.
const ACTIVITY_THRESHOLD: f64 = 0.08;
const ACTIVITY_OMEGA_RADIUS: f64 = 0.30;

include!("examples/01_pile.rs");
include!("examples/02_rope.rs");
include!("examples/03_tower.rs");
include!("examples/04_cloth.rs");
include!("examples/05_lidar.rs");
include!("examples/06_imu.rs");
include!("examples/07_contact.rs");
include!("examples/08_arm.rs");
include!("examples/09_shapes.rs");
include!("examples/10_gpu_ik_arm.rs");
include!("examples/11_gpu_narrow_phase.rs");
include!("examples/12_camera.rs");
include!("examples/13_actuator.rs");
include!("examples/14_quadrotor.rs");
include!("examples/15_broad_phase.rs");
include!("examples/16_vbd_cloth.rs");
include!("examples/17_reference_pendulum.rs");
include!("examples/18_diff_arm.rs");
include!("examples/19_diff_arm6.rs");
include!("examples/20_mesh_tools.rs");
include!("examples/21_neural_policy.rs");
include!("examples/22_neural_lstm.rs");
include!("examples/23_cable.rs");
include!("examples/24_mpm.rs");
include!("examples/25_jelly.rs");
include!("examples/26_pendulum.rs");
include!("examples/27_joint_sampler.rs");
include!("examples/28_heightfield.rs");
include!("examples/29_conveyor.rs");
include!("examples/103_vehicle.rs");
include!("examples/104_sph_dam_break.rs");
include!("examples/105_lidar_v2.rs");
include!("examples/106_aerial_lidar_pass.rs");
include!("examples/107_solar_sky.rs");
include!("examples/108_synthetic_annotation.rs");
include!("examples/109_tire_model_lab.rs");
include!("examples/110_vehicle_offroad.rs");
include!("examples/118_bekker_sinkage.rs");
include!("examples/119_soil_compaction.rs");
include!("examples/120_tire_dynamics.rs");
include!("examples/121_tire_traction.rs");
include!("examples/30_pyramid.rs");
include!("examples/31_cloth_bending.rs");
include!("examples/114_cloth_rollers.rs");
include!("examples/32_cloth_twist.rs");
include!("examples/33_cable_pile.rs");
include!("examples/34_cable_y_junction.rs");
include!("examples/35_mpm_snow_ball.rs");
include!("examples/36_mpm_two_way.rs");
include!("examples/104_cable_bundle_hysteresis.rs");
include!("examples/105_nut_bolt_sdf.rs");
include!("examples/106_nut_bolt_hydro.rs");
include!("examples/37_cartpole.rs");
include!("examples/38_cartpole_swarm.rs");
include!("examples/39_fourbar.rs");
include!("examples/40_diff_ball.rs");
include!("examples/41_diff_spring.rs");
include!("examples/107_diffsim_cloth.rs");
include!("examples/108_diffsim_softbody.rs");
include!("examples/42_brick_wall.rs");
include!("examples/43_softbody_drop.rs");
include!("examples/44_cloth_collar.rs");
include!("examples/45_fast_soft_impact.rs");
include!("examples/46_soft_brake.rs");
include!("examples/47_fem_heat.rs");
include!("examples/49_fem_heat_p2.rs");
include!("examples/48_fem_beam.rs");
include!("examples/51_fem_diff_beam.rs");
include!("examples/50_fem_stokes.rs");

fn main() {
    let geometry = WindowGeometry::load("molla_viewer");
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(geometry.to_window("molla viewer")),
                ..default()
            })
            .set(bevy::log::LogPlugin {
                level: bevy::log::Level::INFO,
                filter: "info,wgpu=error,bevy_render=error,bevy_winit=error,naga=warn".into(),
                ..default()
            }),
    )
    .add_plugins(EguiPlugin::default())
    .add_plugins(FrameTimeDiagnosticsPlugin::default())
    .add_plugins(EntityCountDiagnosticsPlugin::default())
    .add_plugins(WireframePlugin::default());
    #[cfg(not(feature = "profile"))]
    app.add_plugins(RenderDiagnosticsPlugin);
    #[cfg(any(feature = "profile", feature = "log_diag"))]
    {
        let mut filter = bevy::platform::collections::HashSet::default();
        filter.insert(FrameTimeDiagnosticsPlugin::FPS.clone());
        filter.insert(FrameTimeDiagnosticsPlugin::FRAME_TIME.clone());
        app.add_plugins(LogDiagnosticsPlugin {
            debug: false,
            wait_duration: std::time::Duration::from_secs(1),
            filter: Some(filter),
        });
    }
    app.add_plugins(FrostPlugin)
        .add_plugins(
            GlacialPlugins
                .build()
                .disable::<TransformGizmoPlugin>()
                .disable::<SelectionRingPlugin>(),
        )
        .add_plugins(WindowSettingsPlugin::new("molla_viewer"))
        .insert_resource(ClearColor(Color::srgb(0.05, 0.07, 0.12)))
        .insert_resource(GroundGrid {
            visible: true,
            color: Color::srgba(0.30, 0.38, 0.50, 0.42),
        })
        .insert_resource(SimRunState::default())
        .insert_resource(SimStats::default())
        .insert_resource(PendingSceneSwitch(None))
        .insert_resource(SceneDescriptionPaneState::default())
        .insert_resource(MouseGrab::default())
        .insert_resource(ActivityView::default())
        .insert_resource(WireframeOn(false))
        .insert_resource(LidarConfig::default())
        .insert_resource(LidarState::default())
        .insert_resource(ImuExampleState::default())
        .insert_resource(ContactExampleState::default())
        .insert_resource(GpuNpExampleState::default())
        .insert_resource(GpuBpExampleState::default())
        .insert_resource(ArmExampleState::default())
        .insert_resource(GpuIkArmState::default())
        .insert_resource(CameraExampleState::default())
        .insert_resource(ActuatorExampleState::default())
        .insert_resource(QuadrotorState::default())
        .insert_resource(VbdClothExampleState::default())
        .insert_resource(ReferencePendulumState::default())
        .insert_resource(DiffArmExampleState::default())
        .insert_resource(DiffArm6ExampleState::default())
        .insert_resource(MeshToolsExampleState::default())
        .insert_resource(NeuralPolicyExampleState::default())
        .insert_resource(NeuralLstmExampleState::default())
        .insert_resource(CableExampleState::default())
        .insert_resource(MpmExampleState::default())
        .insert_resource(JellyExampleState::default())
        .insert_resource(PendulumExampleState::default())
        .insert_resource(ConveyorExampleState::default())
        .insert_resource(VehicleExampleState::default())
        .insert_resource(SphExampleState::default())
        .insert_resource(LidarV2ExampleState::default())
        .insert_resource(AerialLidarExampleState::default())
        .insert_resource(SolarSkyExampleState::default())
        .insert_resource(SynthAnnotationExampleState::default())
        .insert_resource(TireLabExampleState::default())
        .insert_resource(VehicleOffroadExampleState::default())
        .insert_resource(BekkerSinkageState::default())
        .insert_resource(SoilCompactionState::default())
        .insert_resource(TireDynamicsState::default())
        .insert_resource(TireTractionState::default())
        .insert_resource(ClothBendingExampleState::default())
        .insert_resource(ClothTwistExampleState::default())
        .insert_resource(CartPoleExampleState::default())
        .insert_resource(FourBarExampleState::default())
        .insert_resource(DiffBallExampleState::default())
        .insert_resource(DiffSpringExampleState::default())
        .insert_resource(FemHeatExampleState::default())
        .insert_resource(FemHeatP2ExampleState::default())
        .insert_resource(FemBeamExampleState::default())
        .insert_resource(FemDiffBeamExampleState::default())
        .insert_resource(FemStokesExampleState::default())
        .insert_resource(WarpExampleStates::default())
        // Ribbon state — open the main control pane on startup so
        // the user sees something immediately. Other panes (lidar)
        // start closed and are toggled via the right-rail button.
        .insert_resource({
            let mut o = RibbonOpen::default();
            o.set(RIBBON_LEFT, PANE_MAIN);
            // The right rail has a single pane whose body changes
            // per scene. Pre-open it so the moment the user picks
            // Lidar or IMU the settings are visible — for scenes
            // with no per-example settings the ribbon button is
            // disabled and the pane simply doesn't render.
            o.set(RIBBON_RIGHT, PANE_SCENE);
            o
        })
        .init_resource::<RibbonPlacement>()
        .init_resource::<RibbonDrag>()
        .insert_resource(WireframeConfig {
            // Always false — we don't want a *global* wireframe (would
            // catch bevy_glacial's ground grid mesh). We tag individual
            // physics-mesh entities with the `Wireframe` component
            // instead via `apply_wireframe`.
            global: false,
            default_color: Color::srgb(0.95, 0.95, 0.95),
        });
    let initial_scene = Scene::Pile;
    let (sim, visuals) = Sim::build_scene(initial_scene);
    app.insert_resource(sim)
        .insert_resource(BodyVisuals(visuals))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                (
                    handle_input,
                    apply_scene_switch,
                    update_mouse_grab,
                    // Pre-step controllers run BEFORE step_simulation
                    // so their joint_force / body_f writes are picked
                    // up by the solver.
                    update_arm_example,
                    update_actuator_example,
                    update_neural_policy_example,
                    update_neural_lstm_example,
                    update_gpu_ik_arm,
                    update_quadrotor,
                    update_reference_pendulum_example,
                    update_diff_arm,
                    update_diff_arm6,
                    update_cable_example,
                    update_mpm_example,
                    update_jelly_example,
                )
                    .chain(),
                // Newton-port pre-step controllers/integrators. Split
                // out of the main chain to dodge Bevy's tuple-chain
                // tuple-arity ceiling. Ordered before
                // `step_simulation` via the outer .chain().
                (
                    update_pendulum_example,
                    update_cartpole_example,
                    update_conveyor_example,
                    update_vehicle_example,
                    update_sph_example,
                    update_cloth_rollers,
                    update_cloth_twist_example,
                    update_fourbar_example,
                    update_diff_ball_example,
                    update_diff_spring_example,
                    update_cloth_collar_example,
                    update_fem_heat_example,
                    update_fem_heat_p2_example,
                    update_fem_beam_example,
                    update_fem_diff_beam_example,
                    update_fem_stokes_example,
                )
                    .chain(),
                (step_simulation, sync_transforms, sync_cloth_mesh).chain(),
                (
                    update_lidar,
                    update_imu_example,
                    update_contact_example,
                    update_camera_example,
                    update_activity_colors,
                    update_mesh_tools_visibility,
                    apply_wireframe,
                    draw_grab_gizmo,
                    draw_lidar_gizmos,
                    draw_contact_gizmos,
                    draw_diff_arm_gizmos,
                    draw_diff_arm6_gizmos,
                    draw_cable_gizmos,
                    draw_mpm_gizmos,
                    draw_jelly_gizmos,
                    draw_diff_spring_gizmos,
                    draw_diff_ball_gizmos,
                    draw_fem_heat_gizmos,
                    draw_fem_heat_p2_gizmos,
                    draw_fem_beam_gizmos,
                )
                    .chain(),
                (draw_fem_stokes_gizmos,).chain(),
            )
                .chain(),
        )
        // Stage 36.4 — kept in its own `add_systems` call so the chain
        // tuple above stays under Bevy's 20-element ceiling.
        .add_systems(Update, draw_fem_diff_beam_gizmos)
        .add_systems(Update, draw_cloth_rollers_gizmos)
        // PLAN_HELIOS scenes — each viewer scene has its own update +
        // draw pair; registered separately to stay under the chain
        // tuple-arity ceiling.
        .add_systems(Update, update_lidar_v2_example)
        .add_systems(Update, draw_lidar_v2_gizmos)
        .add_systems(Update, update_aerial_lidar_example)
        .add_systems(Update, draw_aerial_lidar_gizmos)
        .add_systems(Update, update_solar_sky_example)
        .add_systems(Update, draw_solar_sky_gizmos)
        .add_systems(Update, update_synth_annotation_example)
        .add_systems(Update, draw_synth_annotation_gizmos)
        .add_systems(Update, update_tire_lab_example)
        .add_systems(Update, draw_tire_lab_gizmos)
        .add_systems(Update, update_vehicle_offroad_example)
        .add_systems(Update, draw_vehicle_offroad_gizmos)
        .add_systems(Update, update_bekker_sinkage_example)
        .add_systems(Update, draw_bekker_sinkage_gizmos)
        .add_systems(Update, update_soil_compaction_example)
        .add_systems(Update, draw_soil_compaction_gizmos)
        .add_systems(Update, update_tire_dynamics_example)
        .add_systems(Update, draw_tire_dynamics_gizmos)
        .add_systems(Update, update_tire_traction_example)
        .add_systems(Update, draw_tire_traction_gizmos)
        // Warp-port scenes — registered in three groups to dodge the
        // 20-element tuple ceiling. Each tuple is independent (no
        // ordering constraints between them) so we don't bother
        // .chain()'ing them together.
        .add_systems(
            Update,
            (
                update_warp_raycast,
                update_warp_raymarch,
                update_warp_mesh,
                update_warp_mesh_intersect,
                update_warp_sample_mesh,
                update_warp_marching_cubes,
                update_warp_nvdb,
                update_warp_tile_fft,
                update_warp_tile_mlp,
                update_warp_tile_nbody,
                update_warp_tile_cholesky,
                update_warp_repulsion,
                update_warp_fem_diffusion,
                update_warp_fem_diffusion_3d,
                update_warp_fem_convdiff,
                update_warp_fem_convdiff_dg,
            ),
        )
        .add_systems(
            Update,
            (
                update_warp_fem_taylor,
                update_warp_fem_kh,
                update_warp_fem_streamlines,
                update_warp_fem_deformed,
                update_warp_fem_distortion,
                update_warp_fem_elastic_opt,
                update_warp_fem_mixed_elasticity,
                update_warp_fem_nonconforming_contact,
                update_warp_fft_fluid,
                update_warp_distortion_3d,
                update_warp_apic_fluid,
                update_warp_wave,
                update_warp_diff_ray,
                update_warp_tile_filter,
                update_warp_tile_compaction,
                update_adam_landscape,
                update_warp_tile_matmul,
                update_warp_tile_convolution,
                update_warp_tile_mcgp,
            ),
        )
        .add_systems(
            Update,
            (
                draw_warp_raycast_gizmos,
                draw_warp_sample_mesh_gizmos,
                draw_warp_marching_cubes_gizmos,
                draw_warp_tile_nbody_gizmos,
                draw_warp_repulsion_gizmos,
                draw_warp_fem_diffusion_gizmos,
                draw_warp_fem_diffusion_3d_gizmos,
                draw_warp_fem_convdiff_gizmos,
                draw_warp_fem_convdiff_dg_gizmos,
                draw_warp_fem_taylor_gizmos,
                draw_warp_fem_kh_gizmos,
                draw_warp_fem_streamlines_gizmos,
                draw_warp_fem_deformed_gizmos,
                draw_warp_fem_mixed_elasticity_gizmos,
                draw_warp_fem_nonconforming_contact_gizmos,
                draw_warp_fft_fluid_gizmos,
                draw_warp_apic_fluid_gizmos,
                draw_warp_wave_gizmos,
                draw_warp_diff_ray_gizmos,
            ),
        )
        .add_systems(Update, draw_warp_tile_filter_gizmos)
        .add_systems(Update, draw_warp_tile_compaction_gizmos)
        .add_systems(Update, draw_adam_landscape_gizmos)
        .add_systems(Update, update_warp_fem_stokes_transfer)
        .add_systems(Update, draw_warp_fem_stokes_transfer_gizmos)
        .add_systems(Update, draw_warp_tile_matmul_gizmos)
        .add_systems(Update, draw_warp_tile_convolution_gizmos)
        .add_systems(Update, draw_warp_tile_mcgp_gizmos)
        // Newton-port follow-ups (Stage 39).
        .add_systems(
            Update,
            (
                update_cloth_poker_cards,
                update_softbody_hanging,
                update_cable_twist,
                draw_cloth_poker_cards_gizmos,
                draw_softbody_hanging_gizmos,
                draw_cable_twist_gizmos,
            ),
        )
        .add_systems(
            Update,
            (
                update_ik_custom,
                update_diffsim_drone,
                update_warp_fft_poisson_2d,
                draw_ik_custom_gizmos,
                draw_diffsim_drone_gizmos,
                draw_warp_fft_poisson_2d_gizmos,
            ),
        )
        .add_systems(Update, update_ik_cube_stacking)
        .add_systems(Update, draw_ik_cube_stacking_gizmos)
        .add_systems(Update, update_fluid_checkpoint)
        .add_systems(Update, draw_fluid_checkpoint_gizmos)
        .add_systems(Update, update_ns_perturbation)
        .add_systems(Update, draw_ns_perturbation_gizmos)
        // Stage 39A — implicit MPM Bingham (Scene 93).
        .add_systems(Update, update_implicit_mpm_bingham)
        .add_systems(Update, draw_implicit_mpm_bingham_gizmos)
        // Stage 39C — Stam stable fluid (Scene 94).
        .add_systems(Update, update_warp_stable_fluid)
        .add_systems(Update, draw_warp_stable_fluid_gizmos)
        // Stages 36/41 — jelly damping sweep (Scene 95).
        .add_systems(Update, update_jelly_damping_fall)
        .add_systems(Update, draw_jelly_damping_fall_gizmos)
        // Stage 43 — DG Burgers shock (Scene 96).
        .add_systems(Update, update_dg_burgers)
        .add_systems(Update, draw_dg_burgers_gizmos)
        // Stage 43 v2 — shallow water dam break (Scene 97).
        .add_systems(Update, update_swe_dam_break)
        .add_systems(Update, draw_swe_dam_break_gizmos)
        // Stage 40 — implicit MPM 27-stencil (Scene 98).
        .add_systems(Update, update_implicit_mpm_full27)
        .add_systems(Update, draw_implicit_mpm_full27_gizmos)
        // Stage 44 v2 — Gaussian-splat pose recovery (Scene 99).
        .add_systems(Update, update_splat_pose_recovery)
        .add_systems(Update, draw_splat_pose_recovery_gizmos)
        // Stage 45 v2 — AMR heat (Scene 100).
        .add_systems(Update, update_amr_heat)
        .add_systems(Update, draw_amr_heat_gizmos)
        // Stage 46 — magnetostatics (Scene 101).
        .add_systems(Update, update_magnetostatics)
        .add_systems(Update, draw_magnetostatics_gizmos)
        // Tier C3 — phase-field fracture (Scene 102).
        .add_systems(Update, draw_phase_field_fracture_gizmos)
        .add_systems(EguiPrimaryContextPass, control_pane)
        .run();
}

#[derive(Component)]
struct MollaBody(mp::BodyId);

#[derive(Resource, Default)]
struct SimRunState {
    paused: bool,
    step_once: bool,
}

#[derive(Resource)]
struct SceneDescriptionPaneState {
    visible: bool,
}

impl Default for SceneDescriptionPaneState {
    fn default() -> Self {
        // Visible by default — the per-scene description is the point
        // of the pane; hiding it behind a click left users staring at
        // a bare toggle button.
        Self { visible: true }
    }
}

/// Per-frame physics timing. Each EMA in microseconds.
#[derive(Resource, Default)]
struct SimStats {
    smoothed_step_us: f64,
    smoothed_solve_us: f64,
    smoothed_collide_us: f64,
    smoothed_download_us: f64,
    last_contacts: u32,
    arm_logged: bool,
}

/// Bundle of the three ribbon resources so `control_pane` stays
/// under Bevy's 16-system-param ceiling.
#[derive(bevy::ecs::system::SystemParam)]
struct RibbonParams<'w> {
    open: ResMut<'w, RibbonOpen>,
    placement: ResMut<'w, RibbonPlacement>,
    drag: ResMut<'w, RibbonDrag>,
}

/// Bundle of every per-scene runtime resource. `apply_scene_switch`
/// resets each one on a scene change; bundling keeps the system under
/// Bevy's 16-system-param tuple ceiling.
#[derive(bevy::ecs::system::SystemParam)]
struct SceneStates<'w> {
    lidar: ResMut<'w, LidarState>,
    imu: ResMut<'w, ImuExampleState>,
    contact: ResMut<'w, ContactExampleState>,
    arm: ResMut<'w, ArmExampleState>,
    actuator: ResMut<'w, ActuatorExampleState>,
    gpu_ik: ResMut<'w, GpuIkArmState>,
    camera: ResMut<'w, CameraExampleState>,
    stats: ResMut<'w, SimStats>,
    gpu_np: ResMut<'w, GpuNpExampleState>,
    gpu_bp: ResMut<'w, GpuBpExampleState>,
    quad: ResMut<'w, QuadrotorState>,
    /// Native reference-pendulum demo state. Bundled here (rather than
    /// added as a fresh `ResMut` parameter) because both
    /// `apply_scene_switch` and `control_pane` are at the 16-tuple
    /// ceiling — see PLAN.md Stage 22's "Bevy plumbing constraint"
    /// note for details.
    reference_pendulum: ResMut<'w, ReferencePendulumState>,
    /// Stage-23 differentiable-arm demo state. Same bundling rationale.
    diff_arm: ResMut<'w, DiffArmExampleState>,
    /// Stage-23.1 differentiable-Featherstone (6-DOF) demo state.
    /// Bundled here for the same reason as `diff_arm` — keeps both
    /// `apply_scene_switch` and `control_pane` under the 16-element
    /// SystemParam ceiling.
    diff_arm6: ResMut<'w, DiffArm6ExampleState>,
    /// Stage-21 mesh-tools demo state. Same bundling rationale.
    mesh: ResMut<'w, MeshToolsExampleState>,
    /// Stage-18.1 neural-policy demo state. Same bundling rationale —
    /// avoids pushing `apply_scene_switch` past Bevy's 16-tuple
    /// SystemParam ceiling.
    neural: ResMut<'w, NeuralPolicyExampleState>,
    /// Stage-31 neural-LSTM demo state. Same bundling rationale.
    neural_lstm: ResMut<'w, NeuralLstmExampleState>,
    /// Stage-20.1 CABLE demo state. Same bundling rationale.
    cable: ResMut<'w, CableExampleState>,
    /// Stage-24d MPM (sand) demo state. Bundled per the standard
    /// SystemParam ceiling rule.
    mpm: ResMut<'w, MpmExampleState>,
    /// Stage-20.2 JELLY tetrahedral soft-body demo state. Bundled
    /// here for the same reason as `cable` — keeps both
    /// `apply_scene_switch` and `control_pane` under the 16-element
    /// SystemParam ceiling.
    jelly: ResMut<'w, JellyExampleState>,
    // Newton-port scenes (Group A-E).
    pendulum: ResMut<'w, PendulumExampleState>,
    conveyor: ResMut<'w, ConveyorExampleState>,
    cloth_bend: ResMut<'w, ClothBendingExampleState>,
    cloth_twist: ResMut<'w, ClothTwistExampleState>,
    cartpole: ResMut<'w, CartPoleExampleState>,
    fourbar: ResMut<'w, FourBarExampleState>,
    diff_ball: ResMut<'w, DiffBallExampleState>,
    diff_spring: ResMut<'w, DiffSpringExampleState>,
    /// Stage 36 v1 — FEM heat-equation demo state. Holds the FEM
    /// mesh, mass / stiffness matrices, BCs, and current
    /// temperature field. The matrices survive scene switches; only
    /// the temperature field is reset on entry.
    fem_heat: ResMut<'w, FemHeatExampleState>,
    /// Stage 36.3 — P2 (quadratic-nodal) heat-equation demo state.
    fem_heat_p2: ResMut<'w, FemHeatP2ExampleState>,
    /// Stage 36.1 — FEM elasticity beam demo state.
    fem_beam: ResMut<'w, FemBeamExampleState>,
    /// Stage 36.4 — Differentiable FEM beam demo state.
    fem_diff_beam: ResMut<'w, FemDiffBeamExampleState>,
    /// Stage 36.2 — FEM Stokes lid-driven cavity demo state.
    fem_stokes: ResMut<'w, FemStokesExampleState>,
}

/// Bundle of per-scene mutable state used by `control_pane`. Same
/// resources `SceneStates` exposes for `apply_scene_switch`, but
/// reborrowed independently because Bevy's SystemParam tuple has a
/// hard 16-element ceiling and `control_pane` already runs near it.
#[derive(bevy::ecs::system::SystemParam)]
struct PaneSceneStates<'w> {
    arm: ResMut<'w, ArmExampleState>,
    gpu_ik: ResMut<'w, GpuIkArmState>,
    gpu_np: ResMut<'w, GpuNpExampleState>,
    gpu_bp: ResMut<'w, GpuBpExampleState>,
    camera: ResMut<'w, CameraExampleState>,
    actuator: ResMut<'w, ActuatorExampleState>,
    quad: ResMut<'w, QuadrotorState>,
    vbd: ResMut<'w, VbdClothExampleState>,
    /// Native reference-pendulum demo state. Same rationale as the
    /// matching field on `SceneStates` — bundled to stay under the
    /// 16-tuple SystemParam ceiling.
    reference_pendulum: ResMut<'w, ReferencePendulumState>,
    /// Stage-23 differentiable-arm demo state. Same rationale.
    diff_arm: ResMut<'w, DiffArmExampleState>,
    /// Stage-23.1 differentiable-Featherstone demo state. Same rationale.
    diff_arm6: ResMut<'w, DiffArm6ExampleState>,
    /// Stage-21 mesh-tools demo state. Same rationale.
    mesh: ResMut<'w, MeshToolsExampleState>,
    /// Stage-18.1 neural-policy demo state. Same rationale.
    neural: ResMut<'w, NeuralPolicyExampleState>,
    /// Stage-31 neural-LSTM demo state. Same rationale.
    neural_lstm: ResMut<'w, NeuralLstmExampleState>,
    /// Stage-20.1 CABLE demo state. Same rationale.
    cable: ResMut<'w, CableExampleState>,
    /// Stage-24d MPM (sand) demo state. Same rationale.
    mpm: ResMut<'w, MpmExampleState>,
    /// Stage-20.2 JELLY tetrahedral soft-body demo state. Same
    /// rationale as `cable` — bundled to stay under the 16-tuple
    /// SystemParam ceiling.
    jelly: ResMut<'w, JellyExampleState>,
    // Newton-port scenes (Group A-E) — pane state for the right rail.
    pendulum: ResMut<'w, PendulumExampleState>,
    conveyor: ResMut<'w, ConveyorExampleState>,
    cloth_bend: ResMut<'w, ClothBendingExampleState>,
    cloth_twist: ResMut<'w, ClothTwistExampleState>,
    cartpole: ResMut<'w, CartPoleExampleState>,
    fourbar: ResMut<'w, FourBarExampleState>,
    diff_ball: ResMut<'w, DiffBallExampleState>,
    diff_spring: ResMut<'w, DiffSpringExampleState>,
    /// Stage 36 v1 — FEM heat-equation demo state. Same bundling
    /// rationale as the other scenes — keeps `control_pane` under
    /// Bevy's 16-tuple SystemParam ceiling via the `PaneSceneStates`
    /// derive.
    fem_heat: ResMut<'w, FemHeatExampleState>,
    /// Stage 36.3 — P2 (quadratic-nodal) FEM heat demo state.
    fem_heat_p2: ResMut<'w, FemHeatP2ExampleState>,
    /// Stage 36.1 — FEM elasticity beam demo state.
    fem_beam: ResMut<'w, FemBeamExampleState>,
    /// Stage 36.4 — Differentiable FEM beam demo state.
    fem_diff_beam: ResMut<'w, FemDiffBeamExampleState>,
    /// Stage 36.2 — FEM Stokes lid-driven cavity demo state.
    fem_stokes: ResMut<'w, FemStokesExampleState>,
    /// T4 — pressure/weight-controlled tire traction comparison.
    tire_traction: ResMut<'w, TireTractionState>,
    /// T1-T3 — tire tetra FEM backend/readouts.
    tire_dynamics: ResMut<'w, TireDynamicsState>,
}

/// Owns the molla simulation. **GPU-first** per the project rule —
/// `Engine::new_gpu()` is hard-required at startup, no CPU fallback.
#[derive(Resource)]
struct Sim {
    scene: Scene,
    model: mp::Model,
    state_a: mp::State,
    state_b: mp::State,
    control: mp::Control,
    contacts: mp::Contacts,
    solver: SolverBackend,
    pipeline: CollisionPipeline,
    engine: Engine,
    adapter_label: String,
    /// Stage-24d MPM (sand) sim. Lazy-built when entering
    /// `Scene::Mpm`; lives outside the standard solver pipeline.
    mpm: Option<MpmSimulation>,
    /// Last-read MPM particle positions for the renderer.
    mpm_particles: Vec<[f32; 3]>,
    /// Stage-24d.1 — kinematic rigid sphere(s) for two-way coupling
    /// demo. Each entry is `(center_xyz, radius, velocity_xyz)`.
    mpm_rigid_spheres: Vec<([f32; 3], f32, [f32; 3])>,
}

/// Per-scene solver dispatch. Each variant implements the same
/// `Solver::step` signature, so callers don't need to care which
/// backend the active scene picked.
enum SolverBackend {
    Xpbd(mp::XpbdSolver),
    Featherstone(FeatherstoneSolver),
    /// Stage-20 Vertex Block Descent solver. Used by `Scene::VbdCloth`
    /// for high-fidelity cloth + soft-on-rigid coupling.
    Vbd(mp::VbdSolver),
}

impl SolverBackend {
    fn step(
        &mut self,
        model: &mp::Model,
        state_in: &mp::State,
        control: &mp::Control,
        contacts: &mp::Contacts,
        dt: f64,
        state_out: &mut mp::State,
    ) -> molla::core::Result<()> {
        match self {
            Self::Xpbd(s) => s.step(model, state_in, control, contacts, dt, state_out),
            Self::Featherstone(s) => s.step(model, state_in, control, contacts, dt, state_out),
            Self::Vbd(s) => s.step(model, state_in, control, contacts, dt, state_out),
        }
    }
}

#[derive(Component, Copy, Clone)]
enum BodyVisual {
    Sphere(f32),
    /// Box half-extents along x/y/z (m).
    Box(Vec3),
    Capsule {
        half_h: f32,
        radius: f32,
    },
    /// Cone with apex at +Y·half_h, base disc at -Y·half_h with `radius`.
    Cone {
        half_h: f32,
        radius: f32,
    },
    /// Ellipsoid with per-axis semi-axes `(half_axes.x/y/z)` (m).
    /// Rendered as a unit `Sphere` mesh with a non-uniform scale of
    /// `2·half_axes` baked into the entity's transform.
    Ellipsoid {
        half_axes: Vec3,
    },
    /// Gaussian splat — render-only, never participates in
    /// collision. Drawn as a billboarded sphere whose radius equals
    /// the largest principal stddev derived from
    /// `covariance_inv⁻¹`, tinted by `color` (RGBA, alpha = density).
    /// Used by particle-cloud / point-cloud sensors.
    /// Stage 30: stub — viewer renders as a fixed-size billboarded
    /// sphere; full anisotropic-Gaussian rendering deferred.
    #[allow(dead_code)]
    GaussianSplat {
        centre: Vec3,
        /// Inverse covariance matrix (3 columns × 3 rows). Currently
        /// the viewer only consumes its determinant magnitude as a
        /// stand-in radius; full rendering deferred.
        covariance_inv: [Vec3; 3],
        color: bevy::prelude::Vec4,
    },
}

#[derive(Resource)]
struct BodyVisuals(Vec<BodyVisual>);

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Scene {
    Pile,
    Rope,
    Tower,
    Cloth,
    /// Lidar example — kinematic obstacle scene scanned by a GPU
    /// `Raycaster`, hits drawn as gizmo lines. Right-rail pane carries
    /// the per-scene config (origin, FOV, ray count, range).
    Lidar,
    /// IMU example — a single dynamic sphere bouncing on the floor
    /// with a GPU `ImuSensor` mounted at its centre. Right-rail pane
    /// shows the live accelerometer + gyroscope readout.
    Imu,
    /// Contact example — a few stacked bodies on a plane with a GPU
    /// `ContactSensor` running. Right-rail pane shows per-body
    /// contact counts; arrow gizmos point along the net push
    /// direction at each body's contact centroid.
    Contact,
    /// Arm example — 3-link revolute chain driven by a GPU PD
    /// actuator. Featherstone solver (not XPBD). Right-rail pane has
    /// per-joint target sliders and an auto-sine toggle so you can
    /// see the controller chase a moving target.
    Arm,
    /// Stage-13 specialty-shapes demo — drops cones, convex hulls,
    /// and spheres onto a wavy heightfield with a static torus
    /// (SDF) sticking out of the middle. Exercises every primitive
    /// shipped in Stage 13 in a single scene.
    Shapes,
    /// Stage-16 GPU IK demo — 7-DOF Franka-style arm chases a
    /// draggable target marker via [`IkSolver::solve_gpu`]. Right-rail
    /// pane carries XYZ sliders + live solver telemetry (iters,
    /// residual, ms-per-solve). No dynamics — purely kinematic.
    GpuIkArm,
    /// Stage-15 GPU narrow-phase demo — pile of mixed primitives
    /// with a toggle between CPU and full-GPU narrow phase. Right
    /// rail shows the per-frame breakdown of GPU vs CPU narrow-phase
    /// calls so the user can see the GPU pipeline take over.
    GpuNarrowPhase,
    /// Stage-14 multi-channel raytraced camera sensor demo. A bowl
    /// of vivid-coloured primitives sits on a plane; a `TiledCamera`
    /// orbits around them. The right-rail pane shows all five
    /// output channels (color / depth / normal / albedo / shape index)
    /// as live thumbnails read back from the GPU each frame.
    Camera,
    /// Stage-18 composable actuator pipeline demo — single revolute
    /// joint driven by `Actuator { Delay → ControllerPipelinePd →
    /// ClampingDcMotor }`. Right-rail pane has sliders for delay,
    /// motor stall torque, motor no-load speed, and joint target;
    /// shows commanded torque vs delivered torque live.
    Actuator,
    /// Stage-17 multi-DOF Featherstone demo — quadrotor with a FREE
    /// joint root + 4 fixed rotor bodies. PD altitude + attitude
    /// control writes per-rotor thrusts that flow into Featherstone
    /// via `state.body_f` (external wrenches). Hotkey: `Q`.
    Quadrotor,
    /// Stage-19 GPU LBVH broad-phase demo — a 32×32×4 = 4096 sphere
    /// shower, with a CPU-SAP ↔ GPU-LBVH toggle in the right-rail
    /// pane and live readouts of pair count + ms/frame for the
    /// active backend.
    BroadPhase,
    /// Stage-20 high-resolution VBD cloth demo. 32×32 cloth drapes
    /// over a kinematic sphere using Vertex Block Descent (VBD) +
    /// Augmented VBD soft contacts. Right-rail pane carries
    /// stretch / bend / contact stiffness sliders, an inner-iter
    /// count slider, and a toggle to swap between VBD and the
    /// Stage-12 XPBD baseline for side-by-side comparison.
    /// Hotkey `V`.
    VbdCloth,
    /// Native reference double-pendulum scene. Procedural/native so the viewer
    /// remains physics-only with no external-wrapper dependency. Hotkey `J`.
    ReferencePendulum,
    /// Stage-23 differentiable-simulation demo. A 2-link planar arm
    /// reaches a draggable XYZ target via gradient descent on an
    /// open-loop torque sequence. Right-rail pane carries the target
    /// XYZ sliders, an iteration counter + residual readout, a
    /// "Re-run optimisation" button, and a toggle that overlays an
    /// arrow-trace of EE positions across iterations. Hotkey `G`.
    DiffArm,
    /// Stage-23.1 differentiable-Featherstone demo. A 6-DOF Franka-style
    /// serial arm reaches a draggable XYZ target via reverse-mode AD
    /// through Featherstone's three sweeps. Right-rail pane mirrors
    /// `DiffArm`'s layout — target XYZ sliders, residual readout, EE
    /// trajectory overlay — plus an "iters to converge" telemetry row.
    /// Hotkey `H` (for "Hexa-DOF").
    DiffArm6,
    /// Stage-21 mesh-utility showcase. Pure visualisation — no
    /// physics step. Right-rail pane has three panels:
    ///   * **Decimate** — original icosphere vs QEM-decimated LOD
    ///     side-by-side (toggle to swap which one is rendered).
    ///   * **Bake → MC** — mesh → SDF → marching-cubes round-trip,
    ///     overlay toggle for the extracted surface.
    ///   * **Hull** — convex hull of the same mesh as a wireframe
    ///     overlay.
    /// Hotkey `M`.
    MeshTools,
    /// Stage-18.1 neural-policy demo. Cart-pole with a hard-coded
    /// 3-layer MLP (4→32→32→1) acting as a balance controller —
    /// reads `[cart_x, cart_xdot, pole_theta, pole_thetadot]`,
    /// emits a single cart-force command. The MLP runs entirely
    /// on-device via `ControllerNeuralMlp`. Right-rail pane shows
    /// the live MLP I/O readout. Hotkey `N`.
    NeuralPolicy,
    /// Stage-31 neural-LSTM demo. Same cart-pole topology as
    /// `NeuralPolicy`, but driven by a single-cell LSTM (4→16→1)
    /// instead of an MLP. Hidden + cell state survive across
    /// substeps inside `ControllerNeuralLstm`. The right-rail pane
    /// shows the live hidden-state vector + the linear-projection
    /// output. The bundled weights are tuned to express the same
    /// LQR-style balance policy via the linear projection (the
    /// recurrent gates are mostly inactive — see
    /// `build_neural_lstm_weights` for the rationale). Hotkey `L`.
    NeuralLstm,
    /// Stage-20.1 CABLE polyline demo. A 32-particle pinned cable is
    /// yanked by an orbiting kinematic puller body, draping over a
    /// static cylinder so soft-on-rigid contact + Dahl friction +
    /// hard-mode AVBD (toggleable) come into play. Right-rail pane
    /// carries stretch / friction / mode sliders. Hotkey `C`.
    Cable,
    /// Stage-24d v1 — explicit MPM (granular sand). A small column of
    /// particles is poured into a flat bin; sliders on the right pane
    /// expose the friction angle, Young's modulus and particle count.
    /// The MPM simulation lives outside the standard solver pipeline
    /// (it has its own particles + grid storage); `step_simulation`
    /// short-circuits and runs `update_mpm_example` instead. Hotkey
    /// `P` (Particle/MPM).
    Mpm,
    /// Stage-20.2 JELLY tetrahedral soft body. An 8×8×8 tet jelly
    /// cube falls onto a static plane and bounces off a kinematic
    /// puller body. Right-rail pane carries Young's modulus +
    /// Poisson-ratio sliders, a wireframe toggle (shows tet edges),
    /// and a "show CABLE comparison" toggle that overlays a small
    /// cable for bending behaviour A/B. Hotkey `Y`.
    Jelly,
    // ── Newton-port scenes (Group A: basic kinematics + stacking) ──
    /// 1-DoF revolute pendulum + a 3-link chain pendulum side by side.
    /// Featherstone-stepped. Port of `example_basic_pendulum.py`.
    Pendulum,
    /// 4 columns side-by-side, each showing one joint kind in motion:
    /// revolute, prismatic, ball, distance (XPBD equality). Port of
    /// `example_basic_joints.py`.
    JointSampler,
    /// 32×32 sinusoidal heightfield with 30 mixed primitives dropped
    /// on top. Port of `example_basic_heightfield.py`.
    Heightfield,
    /// Kinematic conveyor surface (long box with prescribed velocity)
    /// + dropped boxes that slide along. Port of
    /// `example_basic_conveyor.py`.
    Conveyor,
    /// Tall sphere/box pyramid being struck by a heavy projectile.
    /// Larger and more dramatic than `Tower`. Port of `example_pyramid.py`.
    Pyramid,
    // ── Group B: cloth + cable variations ──
    /// Hanging cloth strip with a live "bending stiffness" slider.
    /// Side-by-side comparison of two strips. Port of
    /// `example_cloth_bending.py`.
    ClothBending,
    /// Reduced procedural cloth-over-rollers scene with kinematic roller
    /// bodies and VBD cloth. Port target for Newton `example_cloth_rollers.py`.
    ClothRollers,
    /// Cloth pinned along TOP and BOTTOM edges; bottom edge rotates
    /// slowly to twist the cloth. Port of `example_cloth_twist.py`.
    ClothTwist,
    /// Several cable polylines dropped into a bowl, watch them pile
    /// up. CABLE bending stiffness. Port of `example_cable_pile.py`.
    CablePile,
    /// Three cables meeting at a Y junction (CONNECT equality at the
    /// central particle). Port of `example_cable_y_junction.py`.
    CableYJunction,
    /// Procedural bundle version of Newton's cable hysteresis example. Several
    /// parallel CABLE strands are pinned into a bundle while the scene audit
    /// points at the Dahl hysteresis oracle/GPU lane; full contact-coupled
    /// bundled Dahl integration remains a follow-up.
    CableBundleHysteresis,
    /// Procedural nut/bolt dense-contact scene using six kinematic socket
    /// walls, a dynamic cylinder bolt, and analytic torus SDF rings as
    /// thread-like contact lips. No imported meshes/assets.
    NutBoltSdf,
    /// Hydroelastic audit variant of the procedural nut/bolt contact case.
    /// The viewer geometry stays asset-free while hydroelastic force/moment
    /// correctness is proven by reduced patch oracles and GPU hydro kernels.
    NutBoltHydro,
    // ── Group C: MPM extensions ──
    /// Ball of snow MPM particles dropped onto ground; uses
    /// `MpmRheology::Snow`. Port of `example_mpm_snow_ball.py`.
    MpmSnowBall,
    /// Heavy rigid sphere drops INTO a pit of MPM sand (uses existing
    /// `mpm_rigid_spheres` two-way coupling). Port of
    /// `example_mpm_twoway_coupling.py`.
    MpmTwoWay,
    /// Procedural three-region MPM material showcase for
    /// `example_mpm_multi_material.py`: sand/snow/viscous particle groups are
    /// laid out side by side and rendered distinctly. Current runtime uses the
    /// existing single-rheology MPM solver while the CPU oracle proves
    /// branch-level multi-material dispatch; per-particle GPU material ids are
    /// the follow-up.
    MpmMultiMaterial,
    /// Reduced procedural MPM beam twist scene. Initializes a twisted beam-like
    /// particle block and colors torsional angle; true torsional constitutive
    /// response remains a follow-up.
    MpmBeamTwist,
    // ── Group D: diffsim + control ──
    /// Cart-pole built procedurally (cart on prismatic + pole on
    /// revolute). PD controller with target slider. Port of
    /// `example_robot_cartpole.py` / `example_selection_cartpole.py`.
    CartPole,
    /// 16 cart-poles in a row, each independently balanced.
    /// Demonstrates batched controllers. Port of
    /// `example_selection_cartpole.py`.
    CartPoleSwarm,
    /// Closed kinematic loop (4-bar linkage) using `JointMimic` to
    /// enforce closure. Driver joint is one revolute; other 3 mimic
    /// via polynomial. Port of `example_kamino_basic_fourbar.py`.
    FourBar,
    /// Ball is launched with `(v, angle)` initial condition; target
    /// hoop sits some distance away; gradient descent on miss
    /// distance. Port of `example_diffsim_ball.py`.
    DiffBall,
    /// Particle inside a "cage" of springs; gradient of position w.r.t.
    /// spring rest lengths; gradient-descent design optimisation.
    /// Port of `example_diffsim_spring_cage.py`.
    DiffSpring,
    /// Reduced differentiable cloth example: a pinned procedural cloth patch
    /// plus CPU oracle tests for a low-dimensional pin/force optimisation.
    /// Forward cloth uses the existing VBD cloth GPU lane where practical;
    /// adjoint/gradient proof is CPU until GPU cloth adjoints exist.
    DiffsimCloth,
    /// Reduced differentiable tet-softbody example. Live viewer uses a small
    /// procedural tet block; CPU oracle tests cover low-dimensional target
    /// deformation gradients until GPU softbody adjoints exist.
    DiffsimSoftBody,
    // ── Group E: sensors / multi-env ──
    /// 6×10 brick wall (rectangular boxes) being struck by a heavy
    /// projectile from one side. Wide, not tall. Port of
    /// `example_brick_stacking.py` (geometry only — no robot arm).
    BrickWall,
    /// Tetrahedral soft cube dropped onto a hanging VBD cloth. Tests
    /// soft-on-soft contact at the engine layer. Port of
    /// `example_softbody_dropping_to_cloth.py`.
    SoftbodyDrop,
    /// Stage 33.3 — pinned cloth tube collapsing onto itself.
    ClothCollar,
    /// Stage 33.4 — fast-impact CCD demo (cube into pinned cloth strip).
    FastSoftImpact,
    /// Stage 33.5 — soft tet block sliding down a soft tet ramp;
    /// friction-coefficient slider sweeps stick → slip → slide.
    SoftBrake,
    /// Stage 36 v1 — FEM heat-equation demo. 16×16×16 tet cube with
    /// Dirichlet BC `T(z=0) = 0`, `T(z=1) = 1`; backward-Euler stepping
    /// of `∂T/∂t = α∇²T`. Right-rail pane carries diffusivity α + dt
    /// sliders, residual readout, and a "Reset to T=0" button. Pure
    /// CPU — runs entirely outside the standard solver pipeline.
    FemHeat,
    /// Stage 36.3 — same heat equation as `FemHeat`, but with P2
    /// (quadratic-nodal) elements: 1 DoF per vertex + 1 DoF per edge.
    FemHeatP2,
    /// Stage 36.1 — FEM elasticity cantilever-beam demo. 20×4×4 tet
    /// beam pinned at one end, downward traction on the other end.
    FemBeam,
    /// Stage 36.4 — Differentiable FEM beam (gradient descent).
    FemDiffBeam,
    /// Stage 36.2 — FEM Stokes lid-driven cavity (Taylor-Hood + Uzawa-CG).
    FemStokes,
    // ── Warp-port scenes (Group A: core primitives) ─────────────────
    /// Port of Warp `core/example_raycast.py` — fire a grid of rays at
    /// a kinematic primitive scene via the GPU `Raycaster`, render
    /// hit points + normals as gizmos.
    WarpRaycast,
    /// Port of `core/example_raymarch.py` — host CPU SDF raymarch into
    /// a 128×128 image, displayed in the right-rail pane.
    WarpRaymarch,
    /// Port of `core/example_mesh.py` — animated `TriMesh` cube with
    /// procedural rotate + breathe scale.
    WarpMesh,
    /// Port of `core/example_mesh_intersect.py` — two cubes whose
    /// overlap is reported via a centre-gap / axis-aligned overlap
    /// check (the cubes only translate, never rotate).
    WarpMeshIntersect,
    /// Port of `core/example_sample_mesh.py` — uniformly random
    /// barycentric samples on a TriMesh sphere, drawn as gizmo points.
    WarpSampleMesh,
    /// Port of `core/example_marching_cubes.py` — bake an animated
    /// SDF (torus + box smooth-min), extract via marching cubes, render.
    WarpMarchingCubes,
    /// Port of `core/example_nvdb.py` — sparse-grid insert + lookup
    /// at random voxels; show insert/lookup latency in the pane.
    WarpNvdb,
    /// Port of `core/example_dem.py` — MPM Sand rheology with high
    /// friction angle (~45°); column poured into a bin, settles at
    /// the angle of repose.
    WarpDem,
    /// Port of `core/example_fluid.py` — MPM with very low friction
    /// angle (~5°), behaves fluid-like.
    WarpFluid,
    /// Port of `core/example_sph.py` — same MPM-as-fluid pattern as
    /// `WarpFluid`, just a different particle layout. For molla's real
    /// WCSPH solver see [`Scene::SphDamBreak`].
    WarpSph,
    // ── Warp-port scenes (Group B: tile / GPU compute primitives) ──
    /// Port of `tile/example_tile_fft.py` — run molla's GPU FFT on a
    /// 256-sample sinusoid and show input + magnitude in the pane.
    WarpTileFft,
    /// Port of `tile/example_tile_mlp.py` — single forward pass of a
    /// `ControllerNeuralMlp` on 16 inputs; show outputs in the pane.
    WarpTileMlp,
    /// Port of `tile/example_tile_nbody.py` — host-driven N-body with
    /// inverse-square gravity between 64 particles drawn as gizmos.
    WarpTileNbody,
    /// Port of `tile/example_tile_cholesky.py` — dense CPU Cholesky on
    /// a 16×16 SPD matrix; verify `‖L Lᵀ − A‖ < 1e-10`.
    WarpTileCholesky,
    /// Port of `optim/example_particle_repulsion.py` — gradient-
    /// descent repulsion of 32 particles via a stochastic (SPSA)
    /// gradient estimate from `molla::compute::tile_mcgp`.
    WarpParticleRepulsion,
    // ── Warp-port scenes (Group C: FEM examples) ────────────────────
    /// Port of `fem/example_diffusion.py` — 2D triangular-mesh heat
    /// equation with Dirichlet BCs on the vertical edges.
    WarpFemDiffusion,
    /// Port of `fem/example_diffusion_3d.py` — 3D tet-mesh diffusion
    /// with **time-varying** sinusoidal Dirichlet BCs.
    WarpFemDiffusion3D,
    /// Port of `fem/example_convection_diffusion.py` — steady
    /// advection-diffusion: u·∇T = α∇²T on a 2D tri mesh.
    WarpFemConvDiff,
    /// Port target for `fem/example_convection_diffusion_dg.py` —
    /// reduced DG-P0 advection-diffusion with limiter telemetry.
    WarpFemConvDiffDg,
    /// Port of `fem/example_taylor_green.py` — Stokes Taylor-Green
    /// vortex initial condition; analytic decay shown in pane.
    WarpFemTaylorGreen,
    /// Port of `fem/example_kelvin_helmholtz.py` — Stokes shear layer
    /// with sinusoidal perturbation; velocity arrows over time.
    WarpFemKelvinHelmholtz,
    /// Port of `fem/example_streamlines.py` — Stokes lid-driven cavity
    /// with tracer particles advected through the velocity field.
    WarpFemStreamlines,
    /// Reduced port target for `fem/example_stokes_transfer.py` — transfer a
    /// procedurally sampled Stokes/FEM velocity grid to tracer particles.
    WarpFemStokesTransfer,
    /// Port of `fem/example_deformed_geometry.py` — heat equation on
    /// a curved (annular) tri mesh.
    WarpFemDeformedGeometry,
    /// Port of `fem/example_distortion_energy.py` — linear elasticity
    /// where the loss is the distortion energy; strain heatmap.
    WarpFemDistortion,
    /// Port of `fem/example_elastic_shape_optimization.py` — same
    /// FemDiffBeam topology with a **shape parameter** (beam height)
    /// being optimised by adjoint.
    WarpFemElasticOpt,
    /// Reduced mixed-elasticity scene for near-incompressible material:
    /// compares displacement-only locking against a pressure-split response.
    WarpFemMixedElasticity,
    /// Reduced nonconforming FEM contact/load-transfer scene. Shows a contact
    /// point from a coarse source surface transferred to nonmatching target
    /// surface nodes while preserving resultant force and moment.
    WarpFemNonconformingContact,
    // ── Warp-port scenes (Group D: bonus stretch) ───────────────────
    /// Port of `core/example_fft_poisson_navier_stokes_2d.py` — 2D
    /// Stokes solve as a fallback for the FFT-Poisson NS coupling.
    WarpFftFluid,
    /// Bonus 3D distortion-energy example (companion to
    /// `WarpFemDistortion`).
    WarpDistortion3D,
    /// Stage 37.6 — APIC FEM-MPM hybrid coupling demo. Drops 200
    /// particles into a tet-meshed cube box; particles transfer
    /// momentum to mesh nodes via APIC weights, mesh provides
    /// viscous + boundary smoothing, particles read updated node
    /// velocities back. Right-rail pane: viscosity slider, particle
    /// count readout, mean kinetic energy readout, "Reset" button.
    WarpApicFluid,
    /// Stage 37.1 — wave PDE `u_tt = c²∇²u` on a 32×32 P1 tri mesh,
    /// integrated implicitly via Newmark-β. Initial Gaussian pulse
    /// at (0.3, 0.3); four edges pinned. Renders as height-field gizmos.
    WarpWave,
    /// Stage 37.7 — differentiable raycasting. Single sphere, target
    /// position (red wireframe); finite-difference adjoint runs
    /// gradient descent on sphere position to match a target depth
    /// image.
    WarpDiffRay,
    /// Stage 37.8 — image filtering / convolution demo. 32×32
    /// procedural input image; filter dropdown picks box / gaussian
    /// / laplacian / identity; CPU/GPU toggle; side-by-side render
    /// of input + output. Wires `molla::compute::tile_convolution`
    /// and `molla::compute::tile_filtering`.
    WarpTileFilter,
    /// Stage 37.8 — stream-compaction demo. 256 random 3D points,
    /// "keep if inside radius R" predicate; the compaction primitive
    /// surfaces only the survivor indices. Wires
    /// `molla::compute::tile_stream_compaction`.
    WarpTileCompaction,
    /// Stage 38c — Adam optimization on a 2D Rosenbrock landscape.
    /// Adam steps from a random init toward the global min at (1, 1)
    /// while the scene renders the loss landscape colored by `log(1+J)`
    /// and the optimization trajectory as a line. Pane exposes lr, β1,
    /// β2, weight-decay. Wires `molla::optim::{AdamConfig, AdamState}`.
    AdamLandscape,
    /// Newton port — 52 stiff VBD cloth "cards" stacked into a deck
    /// on top of a static cube, with a kinematic sphere swinging in
    /// from the side to knock the stack over.
    ClothPokerCards,
    /// Newton port — 4 tetrahedral soft bodies hanging from pinned
    /// faces, each with a different Young's modulus (1×, 2×, 5×, 10×).
    SoftbodyHanging,
    /// Newton port — a single cable pinned at both ends; one endpoint
    /// rotates about its tangent each frame to inject twist that
    /// propagates along the cable.
    CableTwist,
    /// Newton port — collision-avoidance IK on the GpuIkArm Franka.
    /// Sphere obstacle becomes a per-link position-residual row.
    IkCustom,
    /// Reduced Newton cube-stacking IK scene. GPU IK drives the procedural
    /// arm through pick/place waypoints while in-view procedural cubes are
    /// stacked and audited for finite support/contact residuals.
    IkCubeStacking,
    /// Newton port — differentiable-simulation MPC over a simplified
    /// point-mass quadrotor. Adam over FD gradients of a 50-step rollout.
    DiffsimDrone,
    /// Reduced Warp `optim/example_fluid_checkpoint.py` parity scene:
    /// checkpointed differentiable fluid-like rollout over a scalar field.
    FluidCheckpointOptimization,
    /// Reduced Warp `optim/example_navier_stokes_perturbation.py` parity scene:
    /// optimize a small flow perturbation parameter against a target response.
    NavierStokesPerturbationOptimization,
    /// Warp port — 2D FFT-Poisson pressure solve on a 32×32 periodic
    /// grid with a moving Gaussian source. Wires `solve_pressure_poisson_fft`.
    WarpFftPoisson2d,
    /// Stage 37.8 — standalone GEMM `C = A · B` demo. M=64, N=32, K=128
    /// with deterministic-hash inputs; CPU/GPU toggle; side-by-side
    /// render of A, B, C as colored grids. Wires
    /// `molla::compute::tile_matmul::{matmul_cpu, TileMatmulKernel}`.
    WarpTileMatmul,
    /// Stage 37.8 — standalone separable 2D convolution demo. 64×64
    /// procedural input; 5-tap kernel exposed as five user-editable
    /// coefficients (each axis uses the same kernel — the molla API
    /// is symmetric-separable). CPU/GPU toggle; side-by-side render
    /// of input + output. Wires `molla::compute::tile_convolution::{
    /// convolve_separable_cpu, TileConvolutionKernel, Kernel5}`.
    WarpTileConvolution,
    /// Stage 37.8 — Monte Carlo Laplace demo via walk-on-spheres on
    /// the unit square `[-1, 1]²`. Boundary value `g(x, y) = x²·y² −
    /// 0.5·(x² + y²)` is sampled at 16×16 = 256 interior points; each
    /// point's value is estimated by `mcgp_estimate`. Renders as a
    /// grid of points colored by their estimated solution.
    WarpTileMcgp,
    /// Stage 39A — Implicit MPM with Bingham rheology. A spherical
    /// blob of yield-limited fluid drops onto a fixed floor. The
    /// implicit grid solve runs on the CPU at `dt` ≈ 10× the
    /// explicit-CFL limit; per-particle Bingham return mapping
    /// projects the trial strain rate onto the yield surface and a
    /// Bingham-aware Coulomb floor friction arrests the residual
    /// lateral flow at rest. Pane exposes `yield_stress` + `viscosity`
    /// sliders + step / reset controls.
    ImplicitMpmBingham,
    /// Stage 39C — 2D Stam stable-fluid solver on a periodic grid.
    /// Velocity magnitude rendered as a height-field of small gizmo
    /// segments; dye source at the centre drives an unsteady plume
    /// that twists under a swirling body force. Pane: viscosity,
    /// source amplitude, force amplitude, reset.
    WarpStableFluid,
    /// Stage 36/41 — four jelly cubes (tet meshes) dropped onto a
    /// plane. Same base elasticity (E, ν) across all four blocks;
    /// each block is colour-ramped to indicate a damping-sweep slot
    /// (target ζ ∈ {0.1, 0.5, 1.0, 2.0}/s). The global linear
    /// damping is set to 0.5/s (one value from the sweep); per-block
    /// damping is a Stage 41 v2 follow-up (needs a per-particle
    /// slot). VBD tet solver via
    /// `add_tet_mesh_per_material`.
    JellyDampingFall,
    /// Stage 43 — 1D Burgers' equation `∂u/∂t + ∂(u²/2)/∂x = 0` on
    /// `[0, 1]` with smooth IC `u₀(x) = ½ + ½ sin(2πx)`. CPU
    /// Lax-Friedrichs finite-volume scheme + minmod slope limiter
    /// (analogous to the Barth-Jespersen limiter the molla DG-P1
    /// path uses on the tri mesh); shock develops around
    /// `t ≈ 1/π ≈ 0.318`. Renders the current solution profile +
    /// ghost trace of past snapshots.
    DgBurgersShock,
    /// Stage 43 v2 — 2D radial dam break for the shallow-water
    /// equations. Initial state: `h = h_in` inside a circular dam,
    /// `h = h_out` outside, `u = v = 0`. CPU Lax-Friedrichs flux on
    /// a 48×48 Cartesian grid (HLLC is the v3 follow-up). Water
    /// height rendered as a coloured height-field of gizmo lines.
    ShallowWaterDamBreak,
    /// Stage 40 — implicit MPM with the full 27-point Newton
    /// Hessian. Same Bingham blob as #93 but with ~5k particles and
    /// the `HessianStencil::Full27` path, demonstrating the higher
    /// fidelity solve. The Stage 40 v3 GPU device kernel ships in a
    /// follow-up — this scene wires the CPU path whose per-iter cost
    /// mirrors the device kernel.
    ImplicitMpmFull27,
    /// Stage 44 v2 — host-CPU differentiable Gaussian-splat camera
    /// pose recovery. Renders a 96-splat scene at a known camera
    /// pose to build the target image; perturbs the camera; runs an
    /// FD-based gradient descent on the 6-DoF pose vector each frame
    /// using `composite_rgb_cpu` from `molla::geometry`. The pane
    /// shows the current and target views as thumbnails plus a loss
    /// curve.
    SplatPoseRecovery,
    /// Stage 45 v2 — AMR heat equation. A moving Gaussian heat
    /// source orbits the unit square; refinement criterion =
    /// proximity to the source. After each refinement we re-solve
    /// `K · T = f` on the flattened leaf mesh and draw the refined
    /// cells + temperature colormap.
    AmrHeat,
    /// Stage 46 — magnetostatics. A short single-turn current loop
    /// (the simplest "solenoid") drives a steady B-field via
    /// `solve_magnetostatics_steady`. The cell-summed `|A|` magnitude
    /// (Nedelec edge potential) is rendered on the `z = 0.5` cut
    /// plane as a coloured sphere field — a proxy for `‖B‖` until
    /// the public curl-on-Ned-0 helper lands.
    Magnetostatics,
    /// Tier C3 — phase-field fracture scalar-damage oracle. Draws the
    /// P1 AT2 damage solve on a small plate as a height/colour field:
    /// high prescribed tensile history along the centreline produces
    /// a diffuse crack band while the Jacobi oracle enforces
    /// `0 <= d <= 1` and irreversibility.
    PhaseFieldFracture,
    /// V6 — a drivable wheeled vehicle (`molla-vehicle`): a chassis box
    /// on four wheels, driven on a flat plane with the arrow keys.
    Vehicle,
    /// Stage F1-F5 — a real WCSPH dam break. A tall fluid column held
    /// in an open tank collapses under gravity; the GPU `SphSimulation`
    /// (Tait EOS + Akinci boundary particles) advances it every frame.
    SphDamBreak,
    /// Stage H4-H6 — terrestrial LiDAR scan with a `(θ, φ)` grid. A
    /// `Lidar` sensor sweeps a procedural scene; hits are coloured by
    /// `t`, the (row, col) back-pointer connects each return to its
    /// scan cell, and the cone-sample sub-rays demonstrate H5.
    LidarV2,
    /// Stage H7 — aerial LiDAR pass. Pulse origins are
    /// Halton-distributed on a footprint AABB at altitude; rays fall
    /// nadir onto a TriMesh ground and an obstacle, the resulting
    /// point cloud is drawn as gizmos.
    AerialLidarPass,
    /// Stage H13 — outdoor scene lit by `molla_math::solar_position`.
    /// A small set of cubes / spheres sits on a ground; the directional
    /// light direction is driven by `(lat, lon, date_time)` so the sun
    /// arcs across the sky with the time-of-day knob.
    SolarSky,
    /// Stage H9 — synthetic-annotation dataset preview. A scene with
    /// class-labelled bodies is rendered via the `TiledCamera`; the
    /// per-instance bounding boxes are overlaid as gizmos so the
    /// downstream dataset export is visually verifiable.
    SyntheticAnnotation,
    /// V1-V2 — tire-model lab. The three `molla-vehicle` tire models
    /// (`RigidTire`, `TmSimpleTire`, `Pac89Tire`) are evaluated over
    /// a slip-ratio sweep and a slip-angle sweep; the resulting Fx /
    /// Fy curves are drawn as 3-D line plots so the differences are
    /// visible at a glance.
    TireModelLab,
    /// V3-V5 — wheeled vehicle on a procedural heightfield. The same
    /// `molla_vehicle::Vehicle` as `Scene::Vehicle`, but driven over
    /// rolling hills rather than a flat plane so the suspension
    /// kinematics and the heightfield-terrain coupling are visible.
    VehicleOffroad,
    /// V5 — Bekker pressure-sinkage curve. A loading plate is pressed
    /// into `DeformableTerrain` at a ramping load; the (pressure,
    /// sinkage) trace is drawn as a 3-D line strip so the canonical
    /// SCM curve is visible. Uses the new `bekker_n` exponent for a
    /// concave sandy-soil shape.
    BekkerSinkage,
    /// V5-deep — DEM multi-pass soil compaction. A heavy roller
    /// sweeps a `GranularBed` back and forth; per-pass rut depth is
    /// recorded and plotted as a 3-D bar chart so the Proctor-style
    /// "diminishing returns" compaction curve becomes visible.
    SoilCompaction,
    /// T1-T3 — deformable-tire tetra FEM. Five true volumetric tire
    /// meshes run side-by-side at fixed inflation pressures (0, 80,
    /// 200, 320 kPa) plus one animated pressure sweep; contact-patch
    /// width emerges from pressure-loaded tetrahedral carcass dynamics
    /// rather than a Pacejka force element.
    TireDynamics,
    /// T4 — traction comparison under heavy payload. Two simplified
    /// car rigs receive the same wheel torque on low-friction ground:
    /// over-inflated tires spin with a tiny patch, while partially
    /// inflated tires get enough patch area to pull the load forward.
    TireTraction,
}

impl Scene {
    /// All known scenes, in dropdown order.
    const ALL: &'static [Scene] = &[
        Scene::Pile,
        Scene::Rope,
        Scene::Tower,
        Scene::Cloth,
        Scene::Lidar,
        Scene::Imu,
        Scene::Contact,
        Scene::Arm,
        Scene::Shapes,
        Scene::GpuIkArm,
        Scene::GpuNarrowPhase,
        Scene::Camera,
        Scene::Actuator,
        Scene::Quadrotor,
        Scene::BroadPhase,
        Scene::VbdCloth,
        Scene::ReferencePendulum,
        Scene::DiffArm,
        Scene::DiffArm6,
        Scene::MeshTools,
        Scene::NeuralPolicy,
        Scene::NeuralLstm,
        Scene::Cable,
        Scene::Mpm,
        Scene::Jelly,
        // Newton-port scenes.
        Scene::Pendulum,
        Scene::JointSampler,
        Scene::Heightfield,
        Scene::Conveyor,
        Scene::Pyramid,
        Scene::ClothBending,
        Scene::ClothRollers,
        Scene::ClothTwist,
        Scene::CablePile,
        Scene::CableYJunction,
        Scene::CableBundleHysteresis,
        Scene::NutBoltSdf,
        Scene::NutBoltHydro,
        Scene::MpmSnowBall,
        Scene::MpmTwoWay,
        Scene::MpmMultiMaterial,
        Scene::MpmBeamTwist,
        Scene::CartPole,
        Scene::CartPoleSwarm,
        Scene::FourBar,
        Scene::DiffBall,
        Scene::DiffSpring,
        Scene::DiffsimCloth,
        Scene::DiffsimSoftBody,
        Scene::BrickWall,
        Scene::SoftbodyDrop,
        Scene::ClothCollar,
        Scene::FastSoftImpact,
        Scene::SoftBrake,
        Scene::FemHeat,
        Scene::FemBeam,
        Scene::FemHeatP2,
        Scene::FemDiffBeam,
        Scene::FemStokes,
        // ── Warp ports ─────────────────────────────────────────────
        Scene::WarpRaycast,
        Scene::WarpRaymarch,
        Scene::WarpMesh,
        Scene::WarpMeshIntersect,
        Scene::WarpSampleMesh,
        Scene::WarpMarchingCubes,
        Scene::WarpNvdb,
        Scene::WarpDem,
        Scene::WarpFluid,
        Scene::WarpSph,
        Scene::WarpTileFft,
        Scene::WarpTileMlp,
        Scene::WarpTileNbody,
        Scene::WarpTileCholesky,
        Scene::WarpParticleRepulsion,
        Scene::WarpFemDiffusion,
        Scene::WarpFemDiffusion3D,
        Scene::WarpFemConvDiff,
        Scene::WarpFemConvDiffDg,
        Scene::WarpFemTaylorGreen,
        Scene::WarpFemKelvinHelmholtz,
        Scene::WarpFemStreamlines,
        Scene::WarpFemStokesTransfer,
        Scene::WarpFemDeformedGeometry,
        Scene::WarpFemDistortion,
        Scene::WarpFemElasticOpt,
        Scene::WarpFemMixedElasticity,
        Scene::WarpFemNonconformingContact,
        Scene::WarpFftFluid,
        Scene::WarpDistortion3D,
        Scene::WarpApicFluid,
        Scene::WarpWave,
        Scene::WarpDiffRay,
        Scene::WarpTileFilter,
        Scene::WarpTileCompaction,
        Scene::AdamLandscape,
        // Newton-port follow-ups.
        Scene::ClothPokerCards,
        Scene::SoftbodyHanging,
        Scene::CableTwist,
        Scene::IkCustom,
        Scene::IkCubeStacking,
        Scene::DiffsimDrone,
        Scene::FluidCheckpointOptimization,
        Scene::NavierStokesPerturbationOptimization,
        Scene::WarpFftPoisson2d,
        Scene::WarpTileMatmul,
        Scene::WarpTileConvolution,
        Scene::WarpTileMcgp,
        Scene::ImplicitMpmBingham,
        Scene::WarpStableFluid,
        Scene::JellyDampingFall,
        Scene::DgBurgersShock,
        Scene::ShallowWaterDamBreak,
        Scene::ImplicitMpmFull27,
        Scene::SplatPoseRecovery,
        Scene::AmrHeat,
        Scene::Magnetostatics,
        Scene::PhaseFieldFracture,
        Scene::Vehicle,
        Scene::SphDamBreak,
        Scene::LidarV2,
        Scene::AerialLidarPass,
        Scene::SolarSky,
        Scene::SyntheticAnnotation,
        Scene::TireModelLab,
        Scene::VehicleOffroad,
        Scene::BekkerSinkage,
        Scene::SoilCompaction,
        Scene::TireDynamics,
        Scene::TireTraction,
    ];

    fn label(self) -> &'static str {
        match self {
            Scene::Pile => "01 Pile",
            Scene::Rope => "02 Rope",
            Scene::Tower => "03 Tower",
            Scene::Cloth => "04 Cloth",
            Scene::Lidar => "05 Lidar (raycast)",
            Scene::Imu => "06 IMU (accel + gyro)",
            Scene::Contact => "07 Contact (per-body)",
            Scene::Arm => "08 Arm (PD actuator)",
            Scene::Shapes => "09 Shapes (cone / hull / heightfield / SDF)",
            Scene::GpuIkArm => "10 GPU IK arm (Stage 16)",
            Scene::GpuNarrowPhase => "11 GPU narrow phase (Stage 15)",
            Scene::Camera => "12 Camera (5-channel raytrace)",
            Scene::Actuator => "13 Actuator pipeline (Stage 18)",
            Scene::Quadrotor => "14 Quadrotor (Stage 17 multi-DOF)",
            Scene::BroadPhase => "15 GPU broad phase (Stage 19)",
            Scene::VbdCloth => "16 VBD cloth (Stage 20)",
            Scene::ReferencePendulum => "17 Native double pendulum",
            Scene::DiffArm => "18 Differentiable arm (Stage 23)",
            Scene::DiffArm6 => "19 Differentiable Featherstone (Stage 23.1)",
            Scene::MeshTools => "20 Mesh tools (Stage 21)",
            Scene::NeuralPolicy => "21 Neural policy (Stage 18.1)",
            Scene::NeuralLstm => "22 Neural LSTM (Stage 31)",
            Scene::Cable => "23 Cable (Stage 20.1)",
            Scene::Mpm => "24 MPM sand (Stage 24d)",
            Scene::Jelly => "25 Jelly (Stage 20.3)",
            Scene::Pendulum => "26 Pendulum (single + 3-link)",
            Scene::JointSampler => "27 Joint sampler (revolute/prismatic/ball)",
            Scene::Heightfield => "28 Heightfield (waves + dropped primitives)",
            Scene::Conveyor => "29 Conveyor (kinematic belt)",
            Scene::Pyramid => "30 Pyramid (heavy projectile)",
            Scene::ClothBending => "31 Cloth bending (k_b A/B)",
            Scene::ClothRollers => "114 Cloth rollers",
            Scene::ClothTwist => "32 Cloth twist (rotating bottom edge)",
            Scene::CablePile => "33 Cable pile (CABLE bending)",
            Scene::CableYJunction => "34 Cable Y-junction (CONNECT)",
            Scene::CableBundleHysteresis => "104 Cable bundle hysteresis",
            Scene::NutBoltSdf => "105 Nut/bolt SDF contact",
            Scene::NutBoltHydro => "106 Nut/bolt hydroelastic",
            Scene::MpmSnowBall => "35 MPM snow ball (Stomakhin)",
            Scene::MpmTwoWay => "36 MPM two-way (sand + rigid sphere)",
            Scene::MpmMultiMaterial => "103 MPM multi-material",
            Scene::MpmBeamTwist => "115 MPM beam twist",
            Scene::CartPole => "37 Cart-pole (PD balance)",
            Scene::CartPoleSwarm => "38 Cart-pole swarm (16 envs)",
            Scene::FourBar => "39 Four-bar linkage (JointMimic closure)",
            Scene::DiffBall => "40 Diffsim ball (projectile gradient)",
            Scene::DiffSpring => "41 Diffsim spring cage (rest-length gradient)",
            Scene::DiffsimCloth => "107 Diffsim cloth",
            Scene::DiffsimSoftBody => "108 Diffsim soft body",
            Scene::BrickWall => "42 Brick wall (wide, projectile)",
            Scene::SoftbodyDrop => "43 Softbody onto cloth (jelly + VBD cloth)",
            Scene::ClothCollar => "44 Cloth collar (self-collision)",
            Scene::FastSoftImpact => "45 Fast soft impact (CCD)",
            Scene::SoftBrake => "46 Soft brake (friction sweep)",
            Scene::FemHeat => "47 FEM heat (Stage 36 v1)",
            Scene::FemBeam => "48 FEM elasticity (Stage 36.1)",
            Scene::FemHeatP2 => "49 FEM heat P2 (Stage 36.3)",
            Scene::FemStokes => "50 FEM Stokes (Stage 36.2)",
            Scene::FemDiffBeam => "51 FEM diff beam (Stage 36.4)",
            Scene::WarpRaycast => "52 Warp raycast",
            Scene::WarpRaymarch => "53 Warp raymarch",
            Scene::WarpMesh => "54 Warp mesh (animated TriMesh)",
            Scene::WarpMeshIntersect => "55 Warp mesh-intersect (overlap)",
            Scene::WarpSampleMesh => "56 Warp sample-mesh (CDF)",
            Scene::WarpMarchingCubes => "57 Warp marching cubes",
            Scene::WarpNvdb => "58 Warp NanoVDB (sparse grid)",
            Scene::WarpDem => "59 Warp DEM (MPM sand)",
            Scene::WarpFluid => "60 Warp fluid (MPM low-friction)",
            Scene::WarpSph => "61 Warp SPH (MPM-as-fluid)",
            Scene::WarpTileFft => "62 Warp tile FFT",
            Scene::WarpTileMlp => "63 Warp tile MLP",
            Scene::WarpTileNbody => "64 Warp tile N-body",
            Scene::WarpTileCholesky => "65 Warp tile Cholesky",
            Scene::WarpParticleRepulsion => "66 Warp particle repulsion (SPSA)",
            Scene::WarpFemDiffusion => "67 Warp FEM 2D diffusion",
            Scene::WarpFemDiffusion3D => "68 Warp FEM 3D diffusion (animated)",
            Scene::WarpFemConvDiff => "69 Warp FEM conv-diff",
            Scene::WarpFemConvDiffDg => "109 Warp FEM conv-diff DG",
            Scene::WarpFemTaylorGreen => "70 Warp FEM Taylor-Green",
            Scene::WarpFemKelvinHelmholtz => "71 Warp FEM Kelvin-Helmholtz",
            Scene::WarpFemStreamlines => "72 Warp FEM streamlines",
            Scene::WarpFemStokesTransfer => "112 Warp FEM Stokes transfer",
            Scene::WarpFemDeformedGeometry => "73 Warp FEM deformed geometry",
            Scene::WarpFemDistortion => "74 Warp FEM distortion energy",
            Scene::WarpFemElasticOpt => "75 Warp FEM elastic shape opt",
            Scene::WarpFemMixedElasticity => "110 Warp FEM mixed elasticity",
            Scene::WarpFemNonconformingContact => "111 Warp FEM nonconforming contact",
            Scene::WarpFftFluid => "76 Warp FFT fluid (Stokes fallback)",
            Scene::WarpDistortion3D => "77 Warp distortion 3D",
            Scene::WarpApicFluid => "79 Warp APIC fluid (Stage 37.6 hybrid)",
            Scene::WarpWave => "78 Warp wave (Stage 37.1)",
            Scene::WarpDiffRay => "80 Warp diff raycast (Stage 37.7)",
            Scene::WarpTileFilter => "81 Warp tile filter (Stage 37.8)",
            Scene::WarpTileCompaction => "82 Warp tile compaction (Stage 37.8)",
            Scene::AdamLandscape => "83 Adam optimization landscape (Stage 38c)",
            Scene::ClothPokerCards => "84 Cloth poker cards (Newton port)",
            Scene::SoftbodyHanging => "85 Softbody hanging (E sweep)",
            Scene::CableTwist => "86 Cable twist (tangent rotate)",
            Scene::IkCustom => "87 IK custom (collision avoidance)",
            Scene::IkCubeStacking => "113 IK cube stacking",
            Scene::DiffsimDrone => "88 Diffsim drone (MPC)",
            Scene::FluidCheckpointOptimization => "116 Fluid checkpoint optimization",
            Scene::NavierStokesPerturbationOptimization => {
                "117 Navier-Stokes perturbation optimization"
            }
            Scene::WarpFftPoisson2d => "89 Warp FFT Poisson 2D",
            Scene::WarpTileMatmul => "90 Warp tile matmul (Stage 37.8)",
            Scene::WarpTileConvolution => "91 Warp tile convolution (Stage 37.8)",
            Scene::WarpTileMcgp => "92 Warp tile MCGP (Stage 37.8)",
            Scene::ImplicitMpmBingham => "93 Implicit MPM Bingham (Stage 39A)",
            Scene::WarpStableFluid => "94 Warp stable fluid (Stage 39C)",
            Scene::JellyDampingFall => "95 Jelly damping sweep (Stage 36/41)",
            Scene::DgBurgersShock => "96 DG Burgers shock (Stage 43)",
            Scene::ShallowWaterDamBreak => "97 Shallow water dam break (Stage 43 v2)",
            Scene::ImplicitMpmFull27 => "98 Implicit MPM 27-stencil (Stage 40)",
            Scene::SplatPoseRecovery => "99 Splat pose recovery (Stage 44 v2)",
            Scene::AmrHeat => "100 AMR heat (Stage 45 v2)",
            Scene::Magnetostatics => "101 Magnetostatics (Stage 46)",
            Scene::PhaseFieldFracture => "102 Phase-field fracture (Tier C3)",
            Scene::Vehicle => "103 Vehicle (drivable car)",
            Scene::SphDamBreak => "104 SPH dam break (WCSPH)",
            Scene::LidarV2 => "105 LiDAR v2 (scan grid + cone)",
            Scene::AerialLidarPass => "106 Aerial LiDAR pass",
            Scene::SolarSky => "107 Solar sky (sun-position lighting)",
            Scene::SyntheticAnnotation => "108 Synthetic annotation (bboxes)",
            Scene::TireModelLab => "109 Tire model lab (Fx/Fy curves)",
            Scene::VehicleOffroad => "110 Vehicle off-road (heightfield)",
            Scene::BekkerSinkage => "118 Bekker pressure-sinkage curve",
            Scene::SoilCompaction => "119 Soil compaction (DEM passes)",
            Scene::TireDynamics => "120 Tire dynamics (inflation, contact)",
            Scene::TireTraction => "121 Tire traction (pressure vs slip)",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Scene::Pile => {
                "Drops a mixed pile of rigid bodies onto the floor to exercise basic contact, friction, and settling. The expected behavior is a noisy stack that loses energy and comes to rest without exploding or drifting away."
            }
            Scene::Rope => {
                "Builds a chain-like rope with distance constraints and lets it sag under gravity. It should bend and swing like a flexible strand while keeping neighboring particles at roughly fixed spacing."
            }
            Scene::Tower => {
                "Stacks rigid blocks into a simple tower and lets the contact solver stabilize it. The tower may wobble or collapse, but it should do so through contacts rather than numerical blow-up."
            }
            Scene::Cloth => {
                "Runs the baseline XPBD cloth patch with pinned vertices and gravity. The cloth should drape smoothly, preserve approximate edge lengths, and show the older cloth path before the higher-fidelity VBD scenes."
            }
            Scene::Lidar => {
                "Scans a procedural obstacle scene with the GPU raycaster and draws ray hits as gizmos. Changing the right-panel range, FOV, or ray count should immediately change the fan of visible hits."
            }
            Scene::Imu => {
                "Mounts an IMU sensor on a dynamic sphere bouncing on the floor. The right panel should show accelerometer and gyro readings that change as the body accelerates, collides, and rotates."
            }
            Scene::Contact => {
                "Shows a small stack of bodies with a GPU contact sensor reporting per-body contacts. The expected behavior is visible contact arrows and counts that update as bodies touch, separate, and settle."
            }
            Scene::Arm => {
                "Runs a six-link articulated arm driven by PD targets through the Featherstone solver. Joint sliders or the auto-sine mode should move the arm smoothly toward commanded angles."
            }
            Scene::Shapes => {
                "Drops cones, convex hulls, spheres, and other primitives onto a wavy heightfield with an SDF torus in the middle. This is mainly a primitive/contact coverage scene: all shapes should collide plausibly with the terrain."
            }
            Scene::GpuIkArm => {
                "Uses the GPU IK solver on a Franka-style arm chasing a target marker. There is no dynamics step here; the arm should kinematically follow target sliders while reporting residual and iteration telemetry."
            }
            Scene::GpuNarrowPhase => {
                "Compares CPU and GPU narrow-phase collision handling on a pile of mixed primitives. When the GPU path is selected, the scene should still produce stable contacts while the panel reports GPU narrow-phase work."
            }
            Scene::Camera => {
                "Renders a procedural scene through a tiled GPU camera sensor. The right panel should show color, depth, normal, albedo, and shape-index thumbnails that update as the camera orbits."
            }
            Scene::Actuator => {
                "Demonstrates a composable actuator pipeline on a single revolute joint: delay, PD control, and motor clamping. The joint should chase its target while commanded and delivered torques are reported separately."
            }
            Scene::Quadrotor => {
                "Simulates a simple free-flying quadrotor body controlled by altitude and attitude PD loops. Rotor thrusts should keep it near a commanded hover rather than immediately falling or tumbling."
            }
            Scene::BroadPhase => {
                "Runs a large shower of spheres through either CPU sweep-and-prune or GPU LBVH broad phase. The visible scene is simple, but the important behavior is stable candidate-pair generation and timing telemetry."
            }
            Scene::VbdCloth => {
                "Runs the high-resolution Vertex Block Descent cloth scene draping over a kinematic sphere. Stretch, bend, contact stiffness, and inner iterations should visibly affect cloth stability and contact behavior."
            }
            Scene::ReferencePendulum => {
                "Runs a native procedural double-pendulum reference scene. It should swing chaotically but continuously, providing a simple articulated baseline without any external wrapper or asset dependency."
            }
            Scene::DiffArm => {
                "Optimizes an open-loop torque sequence for a planar two-link arm to reach a target. Re-running optimization should reduce end-effector error and optionally draw the trajectory over iterations."
            }
            Scene::DiffArm6 => {
                "Runs the differentiable Featherstone arm example with a six-DOF serial chain. It should optimize toward the target in joint space while reporting residuals and convergence telemetry."
            }
            Scene::MeshTools => {
                "Visualizes mesh-processing utilities rather than a live physics solve. It compares an original procedural mesh, a decimated version, a marching-cubes SDF round trip, and a convex-hull overlay."
            }
            Scene::NeuralPolicy => {
                "Drives a cart-pole with a small on-device MLP controller. The pole's rest pose hangs straight down (θ=0); after a perturbation the controller should damp the swing and recenter the cart, while the right panel shows controller inputs and output force."
            }
            Scene::NeuralLstm => {
                "Drives the same cart-pole topology with a single-cell LSTM policy. Hidden and cell state should evolve over time, and the projected output should act as a cart force that damps the pole back toward its hanging rest."
            }
            Scene::Cable => {
                "Runs a pinned cable pulled around a cylinder with soft contact and Dahl-style friction. The strand should drape, slide, and resist motion without stretching excessively."
            }
            Scene::Mpm => {
                "Pours explicit MPM sand particles into a bin. Friction angle, stiffness, and particle count should alter how the pile flows and settles while remaining bounded."
            }
            Scene::Jelly => {
                "Drops a tetrahedral jelly cube onto a plane with optional wireframe and cable comparison overlays. The block should deform elastically, bounce or damp, and preserve a recognizable volume."
            }
            Scene::Pendulum => {
                "Shows a single revolute pendulum beside a three-link chain pendulum using Featherstone dynamics. Both should swing under gravity with constrained joints and no visible link separation."
            }
            Scene::JointSampler => {
                "Displays several joint types side by side: revolute, prismatic, ball, and distance/equality-style motion. It is a quick visual check that each constraint type moves along its intended degrees of freedom."
            }
            Scene::Heightfield => {
                "Drops mixed primitives onto a sinusoidal heightfield. Bodies should collide with the wavy terrain, slide into troughs, and settle without falling through the surface."
            }
            Scene::Conveyor => {
                "Uses a kinematic belt to push dropped boxes along a long surface. The boxes should receive tangential motion from the belt while still colliding and stacking normally."
            }
            Scene::Pyramid => {
                "Builds a large sphere/box pyramid and strikes it with a heavy projectile. The expected behavior is a dramatic but finite collapse driven by rigid contacts."
            }
            Scene::ClothBending => {
                "Compares two hanging cloth strips with independently controlled bending compliance. The visible difference should be stiffness: one strip can fold more easily while the other resists curvature."
            }
            Scene::ClothRollers => {
                "Places a VBD cloth patch over several kinematic rollers. Roller speed should be reported and visualized; full frictional velocity transfer is still a partial solver-side target."
            }
            Scene::ClothTwist => {
                "Pins cloth along opposite edges while rotating one edge to inject twist. The cloth should wind up, wrinkle, and transmit torsion without tearing numerically."
            }
            Scene::CablePile => {
                "Drops multiple flexible cables into a bowl-like support so they pile together. The scene should show cable bending, contact, and settling in a compact heap."
            }
            Scene::CableYJunction => {
                "Builds three cables joined at a central Y connection. The branch junction should remain connected while the free strands swing and settle."
            }
            Scene::CableBundleHysteresis => {
                "Shows several pinned cable strands arranged as a bundle and tied to the Dahl hysteresis oracle. The visual bundle is procedural; full contact-coupled bundled hysteresis remains partial."
            }
            Scene::NutBoltSdf => {
                "Uses a procedural socket, cylinder bolt, and analytic SDF rings to mimic dense threaded nut/bolt contact without imported meshes. The bolt should stay caged by the surrounding geometry and thread-like lips."
            }
            Scene::NutBoltHydro => {
                "Shows the hydroelastic-audit variant of the procedural nut/bolt setup. Geometry is visible in the viewer, while reduced hydroelastic force and moment preservation provide the main correctness proof."
            }
            Scene::MpmSnowBall => {
                "Drops an MPM snow ball using a Stomakhin-style snow rheology. It should deform and clump more like snow than dry sand or liquid."
            }
            Scene::MpmTwoWay => {
                "Drops a heavy rigid sphere into a pit of MPM sand with two-way coupling. The sphere should displace particles and the particle bed should react against the rigid body."
            }
            Scene::MpmMultiMaterial => {
                "Lays out sand, snow, and viscous/Bingham-like MPM particle regions with distinct colors. The current live runtime is still mostly single-rheology, so use this as a visual/material-dispatch parity scene rather than full per-particle GPU material mixing."
            }
            Scene::MpmBeamTwist => {
                "Initializes a beam-shaped MPM particle block with a procedural twist and angle-based colors. It should make torsional initialization visible, while true torsional constitutive response remains future work."
            }
            Scene::CartPole => {
                "Runs a procedural cart-pole with a PD target. The cart should move along its rail and the pole should respond to the controller rather than behaving like an unconstrained falling rod."
            }
            Scene::CartPoleSwarm => {
                "Runs sixteen cart-pole environments side by side with independent controllers. The row should demonstrate batched procedural control with each pole trying to stay balanced."
            }
            Scene::FourBar => {
                "Builds a four-bar linkage with mimic joints enforcing a closed kinematic loop. The driver joint should move the linkage while maintaining loop closure."
            }
            Scene::DiffBall => {
                "Optimizes a projectile launch velocity and angle so a ball approaches a target hoop. Iterations should reduce miss distance while preserving a ballistic trajectory."
            }
            Scene::DiffSpring => {
                "Optimizes spring rest lengths inside a small particle cage to change its final position. The scene should show differentiable design behavior: loss decreases as spring parameters update."
            }
            Scene::DiffsimCloth => {
                "Shows a reduced differentiable cloth example with a pinned procedural patch. The visible cloth is forward-simulated; the gradient proof is a small CPU oracle for pin or force optimization."
            }
            Scene::DiffsimSoftBody => {
                "Shows a reduced differentiable tetrahedral soft-body block. It is meant to connect soft-body deformation to small optimization oracles, not yet to claim full GPU adjoint parity."
            }
            Scene::BrickWall => {
                "Builds a wide brick wall and hits it with a projectile. Bricks should separate through contact impulses and form a finite debris pile."
            }
            Scene::SoftbodyDrop => {
                "Drops a tetrahedral soft cube onto a hanging VBD cloth. The key behavior is soft-on-soft contact: both bodies should deform and interact without tunneling badly."
            }
            Scene::ClothCollar => {
                "Runs a pinned cloth tube or collar collapsing onto itself. It is mainly a self-collision stress test, so folds should collide instead of freely passing through each other."
            }
            Scene::FastSoftImpact => {
                "Fires a fast cube or impactor into a pinned cloth strip to exercise soft-body CCD. The cloth may deform sharply, but the impact should not simply tunnel through."
            }
            Scene::SoftBrake => {
                "Slides a soft tetrahedral block down a soft ramp with adjustable friction. Sweeping friction should move the behavior between slip, stick, and slower sliding."
            }
            Scene::FemHeat => {
                "Solves a P1 FEM heat equation on a tetrahedral cube with fixed temperatures on opposite boundaries. The temperature field should diffuse smoothly between cold and hot sides."
            }
            Scene::FemBeam => {
                "Runs a FEM elasticity cantilever beam pinned on one end and pulled down on the other. The beam should bend under load and respond to material sliders."
            }
            Scene::FemHeatP2 => {
                "Runs the same heat-equation idea with quadratic P2 elements. Compared with the P1 heat scene, this checks higher-order nodal/edge degrees of freedom and smooth diffusion."
            }
            Scene::FemDiffBeam => {
                "Optimizes a FEM beam parameter or control to reduce a deformation loss. The expected behavior is a visible target/response comparison plus decreasing optimization error."
            }
            Scene::FemStokes => {
                "Solves a FEM Stokes lid-driven cavity problem. The velocity field should form a recirculating flow driven by the moving lid and remain stable under solver iterations."
            }
            Scene::WarpRaycast => {
                "Ports Warp's raycast example: a grid of rays is fired at procedural primitives and hit points/normals are drawn. It should reveal which surfaces the GPU raycaster sees."
            }
            Scene::WarpRaymarch => {
                "Raymarches a procedural signed-distance field into a small image shown in the right pane. This is a CPU SDF rendering example, so the important behavior is the rendered SDF silhouette and normals."
            }
            Scene::WarpMesh => {
                "Animates a procedural triangle mesh with rotation and breathing scale. It is mostly a mesh update/rendering sanity check rather than a contact-heavy physics scene."
            }
            Scene::WarpMeshIntersect => {
                "Moves two procedural axis-aligned cube meshes toward and away from each other and reports their overlap with a centre-gap test. The status should flip to INTERSECTING when the cubes touch."
            }
            Scene::WarpSampleMesh => {
                "Samples points uniformly across a procedural triangle mesh surface. The expected behavior is a stable cloud of samples distributed over the visible mesh."
            }
            Scene::WarpMarchingCubes => {
                "Builds an animated SDF from simple shapes and extracts a surface with marching cubes. The mesh should change smoothly as the underlying field moves."
            }
            Scene::WarpNvdb => {
                "Exercises sparse-grid insertion and lookup with NanoVDB-like data. The right panel reports lookup/insert telemetry while the scene shows active sparse samples."
            }
            Scene::WarpDem => {
                "Uses MPM sand with a high friction angle to mimic a DEM-style granular pile. It should form a steeper, more stable angle of repose than the fluid-like scenes."
            }
            Scene::WarpFluid => {
                "Uses the MPM path with a very low friction angle to behave fluid-like. Particles should spread and flow more readily than dry granular sand."
            }
            Scene::WarpSph => {
                "Shows a fluid-like particle layout using the existing MPM-as-fluid path, not a separate SPH solver. For molla's real weakly-compressible SPH solver, see the SPH dam break scene instead."
            }
            Scene::WarpTileFft => {
                "Runs a tile FFT/DFT-style signal example and plots the magnitude spectrum in the right pane. A sinusoidal input should produce clear frequency peaks."
            }
            Scene::WarpTileMlp => {
                "Runs a small tile MLP forward pass on deterministic inputs. The panel should expose the input/output values so the compute primitive can be checked without a physics scene."
            }
            Scene::WarpTileNbody => {
                "Runs a small host-driven N-body gravity simulation. Particles should attract each other while pairwise momentum and center-of-mass behavior remain bounded."
            }
            Scene::WarpTileCholesky => {
                "Performs a dense Cholesky factorization on a small SPD matrix and reports reconstruction error. This is a compute-kernel check rather than a visual dynamics scene."
            }
            Scene::WarpParticleRepulsion => {
                "Optimizes particle positions with a repulsion loss using a stochastic (SPSA) gradient estimate. Points should spread apart over iterations while the loss decreases."
            }
            Scene::WarpFemDiffusion => {
                "Solves 2D FEM heat diffusion on a triangular mesh with fixed boundary values. The visible scalar field should smooth out from boundary constraints."
            }
            Scene::WarpFemDiffusion3D => {
                "Solves 3D FEM diffusion with animated Dirichlet boundary conditions. The volume/point field should follow the changing boundary temperature without becoming unstable."
            }
            Scene::WarpFemConvDiff => {
                "Runs a continuous FEM convection-diffusion solve with advection and diffusion competing. The field should be transported along the flow while diffusion smooths gradients."
            }
            Scene::WarpFemConvDiffDg => {
                "Runs a reduced DG/P0 convection-diffusion variant with limiter telemetry. It should stay bounded in advection-dominated cases and show when the limiter clamps overshoot."
            }
            Scene::WarpFemTaylorGreen => {
                "Shows a Taylor-Green vortex/Stokes initial condition with analytic decay. Velocity arrows should form a structured vortex pattern that damps predictably over time."
            }
            Scene::WarpFemKelvinHelmholtz => {
                "Shows a shear layer with sinusoidal perturbation in a FEM/Stokes-style flow scene. The expected behavior is evolving velocity structure around the perturbed interface."
            }
            Scene::WarpFemStreamlines => {
                "Advects tracer particles through an analytic vortex velocity field. Tracers should follow the flow rather than behaving like independently simulated rigid bodies."
            }
            Scene::WarpFemStokesTransfer => {
                "Transfers a procedural Stokes/FEM velocity grid onto tracer particles. Constant fields should be preserved and tracers should move according to the sampled velocity field."
            }
            Scene::WarpFemDeformedGeometry => {
                "Runs heat diffusion on a curved or annular deformed geometry. The field should respect the curved mesh rather than looking like a flat-grid solve."
            }
            Scene::WarpFemDistortion => {
                "Visualizes distortion energy or strain over an elastic FEM shape. Higher-strain regions should appear clearly in the heatmap as the geometry deforms."
            }
            Scene::WarpFemElasticOpt => {
                "Optimizes an elastic beam shape parameter to reduce a FEM objective. The scene should show the shape or response changing as the optimization progresses."
            }
            Scene::WarpFemMixedElasticity => {
                "Compares a near-incompressible displacement-only proxy with a mixed pressure/displacement response. The mixed version should avoid the severe locking indicated by the simpler proxy."
            }
            Scene::WarpFemNonconformingContact => {
                "Shows reduced load transfer between nonmatching FEM contact surfaces. The important behavior is force and moment preservation across incompatible source and target patches."
            }
            Scene::WarpFftFluid => {
                "Runs a 2D FFT-Poisson/Stokes fallback for fluid pressure projection. The field should remain periodic and smooth while showing pressure/velocity structure from the source."
            }
            Scene::WarpDistortion3D => {
                "Extends the distortion-energy visualization into a 3D geometry. It should highlight strain or distortion in the volume, similar to the 2D FEM distortion scene."
            }
            Scene::WarpApicFluid => {
                "Demonstrates APIC-style particle/mesh velocity transfer in a FEM-MPM hybrid box. Particles should exchange momentum with grid or mesh nodes and report kinetic energy."
            }
            Scene::WarpWave => {
                "Integrates a wave equation on a triangular mesh from an initial pulse. The height-field view should show waves propagating and reflecting from constrained boundaries."
            }
            Scene::WarpDiffRay => {
                "Optimizes a sphere position so a differentiable raycast depth image matches a target. The current and target positions/images should converge as loss decreases."
            }
            Scene::WarpTileFilter => {
                "Applies selectable image filters to a procedural image. The right panel should show input and filtered output so box, Gaussian, Laplacian, and identity modes are easy to compare."
            }
            Scene::WarpTileCompaction => {
                "Runs stream compaction on a set of random points using a radius predicate. Only survivor indices should be highlighted or counted after compaction."
            }
            Scene::AdamLandscape => {
                "Runs Adam on a 2D Rosenbrock landscape and draws the optimization path. The marker should move toward the global minimum while the panel exposes optimizer parameters."
            }
            Scene::ClothPokerCards => {
                "Stacks many stiff VBD cloth cards on a cube and swings a kinematic sphere into them. The cards should behave like bendable sheets and collapse through contact."
            }
            Scene::SoftbodyHanging => {
                "Hangs several tetrahedral soft bodies with different Young's modulus values. Stiffer blocks should sag less than softer ones under gravity."
            }
            Scene::CableTwist => {
                "Pins a cable at both ends while rotating one endpoint around the tangent. Twist should propagate along the strand, though full twist-DOF parity is still partial."
            }
            Scene::IkCustom => {
                "Runs collision-avoidance IK on the GPU arm with a sphere obstacle represented as residuals. The arm should attempt to reach the target while keeping links away from the obstacle."
            }
            Scene::IkCubeStacking => {
                "Moves the GPU IK arm through procedural pick, lift, place, and retreat waypoints while cubes are shown stacking. Full gripper/contact-coupled stacking remains partial, so use the description to check the intended choreography."
            }
            Scene::DiffsimDrone => {
                "Runs a simplified differentiable quadrotor MPC over a point-mass model. Adam should reduce final position error over a short rollout while keeping the control sequence bounded."
            }
            Scene::FluidCheckpointOptimization => {
                "Runs a reduced checkpointed fluid-like scalar rollout and optimizes a source amplitude. It should save bounded checkpoints, replay deterministically, and reduce target-field loss."
            }
            Scene::NavierStokesPerturbationOptimization => {
                "Optimizes a scalar perturbation parameter in a reduced Navier-Stokes-like flow response. The scene should show the perturbation changing the flow and the loss decreasing over iterations."
            }
            Scene::WarpFftPoisson2d => {
                "Solves a 2D periodic Poisson pressure problem with an FFT-based method. A moving Gaussian source should produce a smooth pressure field on the grid."
            }
            Scene::WarpTileMatmul => {
                "Runs a standalone tile matrix multiplication and displays matrices as colored grids. It is a compute correctness scene: output should match deterministic CPU/GPU expectations."
            }
            Scene::WarpTileConvolution => {
                "Runs separable 2D convolution with editable five-tap kernel coefficients. Changing the kernel should visibly alter the filtered image while preserving the side-by-side input/output layout."
            }
            Scene::WarpTileMcgp => {
                "Estimates a Laplace solution with Monte Carlo walk-on-spheres sampling. The colored point grid should approximate the boundary-value solution over the square domain."
            }
            Scene::ImplicitMpmBingham => {
                "Runs an implicit MPM Bingham blob with yield stress and viscosity controls. The material should flow under load but arrest residual motion when stresses fall below yield."
            }
            Scene::WarpStableFluid => {
                "Runs a 2D Stam stable-fluid solver with dye source and swirling body force. Velocity magnitude and plume motion should evolve smoothly under viscosity and force sliders."
            }
            Scene::JellyDampingFall => {
                "Drops several jelly cubes with a damping sweep indicated by color. The cubes should deform similarly but settle with different apparent damping behavior."
            }
            Scene::DgBurgersShock => {
                "Solves 1D Burgers' equation with a finite-volume/DG-style limiter. The smooth initial wave should steepen into a shock while the limiter keeps the profile bounded."
            }
            Scene::ShallowWaterDamBreak => {
                "Runs a radial shallow-water dam-break on a 2D grid. Water height should spread outward from the initial circular dam and form a visible wave front."
            }
            Scene::ImplicitMpmFull27 => {
                "Runs the implicit MPM Bingham setup with the full 27-point Hessian stencil. It is meant as a higher-fidelity implicit solve path, so behavior should be similar but more solver-intensive than the simpler Bingham scene."
            }
            Scene::SplatPoseRecovery => {
                "Optimizes a camera pose against a target Gaussian-splat render. The current thumbnail and target thumbnail should become more similar as the pose loss decreases."
            }
            Scene::AmrHeat => {
                "Solves heat diffusion on an adaptively refined 2D mesh with a moving heat source. Cells near the source should refine and the temperature colormap should follow the source."
            }
            Scene::Magnetostatics => {
                "Solves a small steady magnetostatics problem for a current loop. The rendered cut-plane should show a structured vector-potential or field-magnitude proxy around the loop."
            }
            Scene::PhaseFieldFracture => {
                "Visualizes a reduced phase-field fracture/damage solve on a plate. Damage should stay between zero and one, be irreversible, and concentrate along the expected crack band."
            }
            Scene::Vehicle => {
                "A drivable four-wheeled vehicle on flat ground. The arrow keys throttle, brake, and steer it; the chassis should accelerate, corner, and settle on its suspension the way a car does."
            }
            Scene::SphDamBreak => {
                "A real weakly-compressible SPH dam break. A tall water column held against one wall collapses, surges across the tank floor, and runs up the far wall before sloshing back. Driven by the GPU WCSPH solver — not the MPM-as-fluid path."
            }
            Scene::LidarV2 => {
                "A terrestrial LiDAR scan over analytical primitives + a TriMesh ground. The new `Lidar` sensor lays out a (θ, φ) grid; hits are drawn as small spheres coloured by distance. The right pane exposes scan-grid size, beam divergence, and sub-ray count so cone widening is visible on the fly."
            }
            Scene::AerialLidarPass => {
                "An aerial LiDAR pass over a TriMesh ground with a couple of obstacles. Pulse origins are Halton-distributed across the footprint AABB and fired nadir; the resulting point cloud is drawn over the ground plane, density scaling with the pulses-per-m² knob."
            }
            Scene::SolarSky => {
                "An outdoor scene lit by `molla_math::solar_position`. A flat ground plus a few cubes sits at a (lat, lon); the sun direction is computed from a slider-driven date/time and used to shade the bodies, so the lighting visibly arcs across the sky as you scrub time."
            }
            Scene::SyntheticAnnotation => {
                "A synthetic-annotation dataset preview. Three labelled bodies (cube, sphere, capsule) sit on a ground; their per-instance bounding boxes — derived from the `TiledCamera`'s SHAPE_INDEX channel via the new `Dataset` post-process — are drawn as wireframe gizmos to make the dataset export visually auditable."
            }
            Scene::TireModelLab => {
                "A wheel-model laboratory. The three molla-vehicle tire models — `RigidTire`, `TmSimpleTire`, `Pac89Tire` — are evaluated across a slip-ratio sweep and a slip-angle sweep; the resulting longitudinal (Fx) and lateral (Fy) force curves are drawn as 3-D line plots so the difference between rigid, TMsimple, and Pacejka becomes visible. Right-rail knobs vary the normal load (Fz) and friction coefficient."
            }
            Scene::VehicleOffroad => {
                "The same drivable wheeled vehicle as Scene::Vehicle, but rolling over a procedural heightfield. The hills are deterministic and large enough to load the suspension; the heightfield grid is drawn as a wireframe so the wheel-ground contact is auditable. Arrow keys drive."
            }
            Scene::BekkerSinkage => {
                "A loading plate is pressed into a Bekker SCM soil patch at a continuously-ramping load; the resulting (pressure, sinkage) trace is plotted as a 3-D line strip so the canonical pressure-sinkage curve is visible. Uses the new `bekker_n` exponent on `DeformableTerrain` for a concave sandy-soil shape (`p = stiffness · z^n`)."
            }
            Scene::SoilCompaction => {
                "A heavy kinematic roller sweeps back and forth over a DEM granular bed. After every completed pass the rut depth at the bed centre is sampled and added to a bar chart — the soil compacts more on each pass with diminishing returns, the Proctor-style compaction curve made visible. Real grain rearrangement, two-way roller↔grain coupling via `GrainObstacle`."
            }
            Scene::TireDynamics => {
                "Five volumetric tetra-FEM tire carcasses run side-by-side at five inflation pressures (0, 80, 200, 320 kPa fixed + one animated). Each tire uses `molla_vehicle::TetraTireFem`: a circumferential × radial × width annular tire volume split into tetrahedra, with linear tetra FEM rubber forces, bead/rim coupling, pressure traction, floor contact, and a fixed hub load. The contact patch emerges from the tetra mesh — higher pressure stays rounder with a smaller patch; lower pressure squashes into a wider footprint."
            }
            Scene::TireTraction => {
                "Three simplified cars receive the same wheel torque on a low-friction track. The over-inflated red car has a tiny contact patch and mostly spins; the partially-inflated blue car has a wider patch and pulls forward when the surface can support it; the green car is user-controlled from the right rail with tire-pressure, payload, tire stiffness, and dirt→ice slipperiness sliders. Green bars show effective patch/traction, red bars show slip."
            }
        }
    }

    /// Which dynamics solver this scene wants. Most scenes use XPBD;
    /// the Arm example needs Featherstone for proper joint dynamics.
    /// `GpuIkArm` reuses the Featherstone backend (any articulation
    /// solver works since `step_simulation` skips this scene anyway).
    /// `VbdCloth` uses the Stage-20 Vertex Block Descent solver.
    /// The historical wrapper slot now routes to Featherstone so it remains a
    /// native procedural physics scene.
    fn solver_kind(self) -> SolverKind {
        match self {
            Scene::Arm
            | Scene::GpuIkArm
            | Scene::Actuator
            | Scene::Quadrotor
            | Scene::NeuralPolicy
            | Scene::NeuralLstm
            // Newton-port articulated scenes use Featherstone for proper
            // joint dynamics. `Conveyor` doesn't have an articulation but
            // we use Featherstone-friendly construction for the kinematic
            // belt; XPBD is fine.
            | Scene::Pendulum
            | Scene::JointSampler
            | Scene::CartPole
            | Scene::CartPoleSwarm
            | Scene::FourBar
            | Scene::IkCubeStacking => SolverKind::Featherstone,
            Scene::VbdCloth
            | Scene::Cable
            | Scene::Jelly
            | Scene::ClothBending
            | Scene::ClothRollers
            | Scene::ClothTwist
            | Scene::DiffsimCloth
            | Scene::DiffsimSoftBody
            | Scene::CablePile
            | Scene::CableYJunction
            | Scene::CableBundleHysteresis
            | Scene::SoftbodyDrop
            | Scene::ClothCollar
            | Scene::FastSoftImpact
            | Scene::SoftBrake
            | Scene::ClothPokerCards
            | Scene::SoftbodyHanging
            | Scene::CableTwist
            | Scene::JellyDampingFall => SolverKind::Vbd,
            Scene::ReferencePendulum => SolverKind::Featherstone,
            // DiffArm / DiffArm6 are purely host-driven (the optimiser
            // writes body_q each frame); pick the cheap-to-construct
            // XPBD solver — `step_simulation` skips these scenes anyway.
            _ => SolverKind::Xpbd,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum SolverKind {
    Xpbd,
    Featherstone,
    Vbd,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum PhysicsAuditStatus {
    /// Real physics is implemented and covered by a CPU/default test or a
    /// dedicated GPU target where relevant.
    Complete,
    /// Some real physics exists, but parity with the Warp/Newton reference case
    /// is incomplete, GPU coverage is incomplete, or the scene is mostly a
    /// reduced procedural substitute.
    Partial,
    /// Label/UI exists, but the physics case is not yet implemented as an
    /// engine-backed simulation.
    #[allow(dead_code)]
    Placeholder,
    /// Real physics exists, but meaningful verification is GPU-only or too long
    /// for default `make test`; it must be run through an explicit make target.
    ManualGpu,
    /// Real physics exists and is tested, but a GPU implementation is not
    /// meaningful for the current reduced/demo formulation or has an explicit
    /// architecture reason to stay on CPU.
    CpuOnly,
}

impl PhysicsAuditStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::Partial => "partial",
            Self::Placeholder => "placeholder",
            Self::ManualGpu => "manual GPU",
            Self::CpuOnly => "CPU-only",
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct ScenePhysicsAudit {
    status: PhysicsAuditStatus,
    reference_case: &'static str,
    cpu_proof: &'static str,
    gpu_proof: &'static str,
    notes: &'static str,
}

impl Scene {
    /// Physics-only audit for the viewer registry.
    ///
    /// This is deliberately separate from labels/builders: a scene may exist in
    /// the UI while only partially implementing the corresponding Warp/Newton
    /// physics case. Keeping the table in code gives us an exhaustive match so
    /// new scenes cannot be added without being classified.
    fn physics_audit(self) -> ScenePhysicsAudit {
        use PhysicsAuditStatus::{Complete, CpuOnly, ManualGpu, Partial};
        match self {
            Scene::Pile => ScenePhysicsAudit {
                status: Complete,
                reference_case: "rigid pile / stacking",
                cpu_proof: "scene_smoke + scene_correctness pile",
                gpu_proof: "make test-gpu / make test-xpbd-gpu",
                notes: "XPBD rigid contact stack",
            },
            Scene::Rope => ScenePhysicsAudit {
                status: Complete,
                reference_case: "rope/particle constraints",
                cpu_proof: "scene_correctness rope floor/taut gates",
                gpu_proof: "make test-particles-gpu",
                notes: "XPBD particle/equality coverage",
            },
            Scene::Tower => ScenePhysicsAudit {
                status: Complete,
                reference_case: "rigid tower stability",
                cpu_proof: "scene_correctness tower",
                gpu_proof: "make test-xpbd-gpu",
                notes: "stress stack under gravity",
            },
            Scene::Cloth => ScenePhysicsAudit {
                status: Complete,
                reference_case: "XPBD cloth",
                cpu_proof: "scene_smoke cloth",
                gpu_proof: "make test-particles-gpu",
                notes: "particle cloth path",
            },
            Scene::Lidar | Scene::WarpRaycast => ScenePhysicsAudit {
                status: Complete,
                reference_case: "raycast sensor / Warp raycast",
                cpu_proof: "raycast_adjoint + scene smoke",
                gpu_proof: "make test-raycast-gpu",
                notes: "GPU ray sensor has dedicated parity tests",
            },
            Scene::Imu => ScenePhysicsAudit {
                status: Complete,
                reference_case: "IMU sensor",
                cpu_proof: "scene_smoke imu",
                gpu_proof: "make test-imu-gpu",
                notes: "GPU IMU tests cover gravity/freefall/gyro axes",
            },
            Scene::Contact => ScenePhysicsAudit {
                status: Complete,
                reference_case: "contact sensor",
                cpu_proof: "scene_smoke contact",
                gpu_proof: "make test-contact-gpu",
                notes: "GPU contact sensor tests cover zero/one/stacked contacts",
            },
            Scene::Arm | Scene::Actuator => ScenePhysicsAudit {
                status: Complete,
                reference_case: "PD/PID actuator articulation",
                cpu_proof: "actuator and Featherstone tests",
                gpu_proof: "make test-actuator-gpu / make test-actuator-pipeline-gpu",
                notes: "Featherstone + actuator pipeline",
            },
            Scene::Shapes | Scene::Heightfield => ScenePhysicsAudit {
                status: Complete,
                reference_case: "primitive/SDF/heightfield contact",
                cpu_proof: "heightfield_and_sdf + shape tests",
                gpu_proof: "GPU narrow-phase targets for supported primitives",
                notes: "SDF/heightfield remain partly CPU contact paths",
            },
            Scene::GpuIkArm => ScenePhysicsAudit {
                status: Complete,
                reference_case: "GPU IK arm",
                cpu_proof: "ik_franka CPU IK/Jacobian tests + scene smoke",
                gpu_proof: "make test-articulation-gpu",
                notes: "Franka-style IK has CPU oracle and GPU multi-seed/Jacobian/IK parity gates",
            },
            Scene::IkCustom => ScenePhysicsAudit {
                status: Partial,
                reference_case: "IK with collision-avoidance",
                cpu_proof: "ik_franka + viewer collision-objective smoke",
                gpu_proof: "make test-articulation-gpu for base IK path",
                notes: "collision avoidance is a reduced synthesized position-residual objective, not full Newton objective parity",
            },
            Scene::IkCubeStacking => ScenePhysicsAudit {
                status: Partial,
                reference_case: "Newton ik/example_ik_cube_stacking.py",
                cpu_proof: "ik_cube_stacking_targets_are_reachable_and_audited + ik_cube_stacking_stack_is_supported_and_finite + ik_cube_stacking_reports_obstacle_residuals",
                gpu_proof: "make test-articulation-gpu for the GPU IK arm solve; cube contact/stacking is reduced viewer telemetry, not a coupled GPU manipulation solve",
                notes: "procedural cubes and pick/place waypoints are shown in the viewer; full gripper contact, coupled stacking dynamics, and obstacle-aware GPU IK objective remain future work",
            },
            Scene::GpuNarrowPhase | Scene::BroadPhase => ScenePhysicsAudit {
                status: Complete,
                reference_case: "GPU broad/narrow phase",
                cpu_proof: "geometry broad/narrow tests",
                gpu_proof: "make test-gpu and geometry GPU targets",
                notes: "GPU fallback must stay visible in telemetry",
            },
            Scene::Camera => ScenePhysicsAudit {
                status: Complete,
                reference_case: "tiled camera sensor",
                cpu_proof: "scene_smoke camera",
                gpu_proof: "make test-tiled-camera-gpu",
                notes: "GPU camera channels covered by sensor tests",
            },
            Scene::Quadrotor => ScenePhysicsAudit {
                status: Complete,
                reference_case: "procedural quadrotor control",
                cpu_proof: "scene_smoke quadrotor + quadrotor_pd_mixer_balances_hover_and_attitude_torques",
                gpu_proof: "make test-featherstone-multidof-gpu / make test-featherstone-gpu",
                notes: "native Featherstone FREE-root quadrotor control; differentiable MPC is tracked by DiffsimDrone",
            },
            Scene::VbdCloth => ScenePhysicsAudit {
                status: Complete,
                reference_case: "VBD cloth",
                cpu_proof: "vbd_cloth_cpu pinned/wrap/contact tests",
                gpu_proof: "make test-vbd-cloth-gpu",
                notes: "core VBD cloth path has CPU contact/stability gates and GPU parity lanes",
            },
            Scene::ClothBending | Scene::ClothTwist => ScenePhysicsAudit {
                status: Complete,
                reference_case: "VBD cloth bending/twist response",
                cpu_proof: "vbd_cloth_bending_compliance_changes_cantilever_sag / vbd_cloth_bending_springs_change_twist_response",
                gpu_proof: "make test-vbd-cloth-gpu / make test-vbd-extended-gpu",
                notes: "bending and twist-response variants are test-backed reduced cloth physics scenes",
            },
            Scene::ClothPokerCards => ScenePhysicsAudit {
                status: Complete,
                reference_case: "Newton cloth-over-card/box contact",
                cpu_proof: "vbd_cloth_cpu_drapes_over_kinematic_box_without_penetration",
                gpu_proof: "make test-vbd-cloth-gpu",
                notes: "procedural cloth-over-box/card contact is complete; garment/Style3D asset parity is out of scope",
            },
            Scene::ReferencePendulum => ScenePhysicsAudit {
                status: Complete,
                reference_case: "native procedural double pendulum",
                cpu_proof: "scene smoke / Featherstone articulation tests",
                gpu_proof: "make test-featherstone-gpu",
                notes: "external wrappers are explicitly out of scope",
            },
            Scene::DiffsimCloth => ScenePhysicsAudit {
                status: Partial,
                reference_case: "Newton diffsim/example_diffsim_cloth.py",
                cpu_proof: "diffsim_cloth_reduced_pin_gradient_matches_finite_difference + diffsim_cloth_reduced_optimization_reduces_loss",
                gpu_proof: "make test-vbd-cloth-gpu for forward cloth; no GPU cloth adjoint yet",
                notes: "procedural pinned cloth viewer scene plus reduced CPU differentiable oracle; full cloth adjoint/MPC parity remains future work",
            },
            Scene::DiffsimSoftBody => ScenePhysicsAudit {
                status: Partial,
                reference_case: "Newton diffsim/example_diffsim_soft_body.py",
                cpu_proof: "diffsim_softbody_reduced_velocity_gradient_matches_finite_difference + diffsim_softbody_reduced_optimization_reduces_loss",
                gpu_proof: "make test-vbd-tets-gpu for forward tet softbody; no GPU softbody adjoint yet",
                notes: "procedural tet block viewer scene plus reduced CPU differentiable oracle; full tet adjoint/material optimization parity remains future work",
            },
            Scene::DiffArm | Scene::DiffArm6 | Scene::DiffBall | Scene::DiffSpring => {
                ScenePhysicsAudit {
                    status: Complete,
                    reference_case: "differentiable rigid/articulation examples",
                    cpu_proof: "autograd/diffsim tests",
                    gpu_proof: "GPU forward/parity where solver-owned",
                    notes: "CPU adjoints are accepted where GPU adjoints are not practical",
                }
            }
            Scene::MeshTools
            | Scene::WarpMesh
            | Scene::WarpMeshIntersect
            | Scene::WarpSampleMesh
            | Scene::WarpMarchingCubes => ScenePhysicsAudit {
                status: Complete,
                reference_case: "mesh, hull, SDF, marching-cubes geometry",
                cpu_proof: "mesh_utils / geometry tests",
                gpu_proof: "mesh raycast/GJK GPU targets where supported",
                notes: "geometry-processing physics support, not a scene-loading task",
            },
            Scene::NeuralPolicy | Scene::NeuralLstm => ScenePhysicsAudit {
                status: Complete,
                reference_case: "neural actuator/control",
                cpu_proof: "actuator/controller tests",
                gpu_proof: "make test-actuator-pipeline-gpu",
                notes: "procedural cart-pole controller",
            },
            Scene::Cable | Scene::CablePile | Scene::CableYJunction => ScenePhysicsAudit {
                status: Complete,
                reference_case: "Newton cable/cable-pile/y-junction",
                cpu_proof: "joint_cable + vbd_cables_cpu tests",
                gpu_proof: "make test-vbd-tets-gpu / make test-vbd-extended-gpu",
                notes: "stretch/bend cable chains, piles, and branching are covered by CPU gates plus cable GPU lanes",
            },
            Scene::CableBundleHysteresis => ScenePhysicsAudit {
                status: Partial,
                reference_case: "Newton cable bundle hysteresis",
                cpu_proof: "cable_bundle_hysteresis_loop_has_positive_area + dahl_oracle tests",
                gpu_proof: "make test-vbd-extended-gpu",
                notes: "procedural bundle scene plus Dahl hysteresis oracle exist; full contact-coupled bundled Dahl integration remains future work",
            },
            Scene::ClothRollers => ScenePhysicsAudit {
                status: Partial,
                reference_case: "Newton cloth/example_cloth_rollers.py",
                cpu_proof: "cloth_rollers_scene_has_kinematic_rollers_and_cloth + cloth_roller_surface_velocity_transfers_tangential_speed + cloth_rollers_support_gap_oracle_is_finite",
                gpu_proof: "make test-vbd-cloth-gpu for the forward cloth lane; roller friction/kinematic velocity coupling remains a reduced CPU oracle",
                notes: "procedural VBD cloth plus kinematic roller bodies are in the viewer; full roller friction/tangential-velocity coupling needs stronger solver-side parity",
            },
            Scene::NutBoltSdf => ScenePhysicsAudit {
                status: Partial,
                reference_case: "Newton contacts/example_nut_bolt_sdf.py",
                cpu_proof: "procedural_socket_bolt_dense_contact_stays_finite_and_caged + nut_bolt_sdf_thread_ring_has_expected_signs",
                gpu_proof: "make test-contact-reduction-gpu / make test-sparse-grid-gpu for reduction/grid lanes; analytic SDF narrow phase is CPU-only today",
                notes: "procedural socket/bolt plus analytic torus SDF thread lips; no imported threaded mesh, and full threaded SDF GPU contact remains future work",
            },
            Scene::NutBoltHydro => ScenePhysicsAudit {
                status: Partial,
                reference_case: "Newton contacts/example_nut_bolt_hydro.py",
                cpu_proof: "plane_patch_hydroelastic_force_grows_with_penetration_depth + hydroelastic_offset_patch_reduction_preserves_force_and_moment + procedural_socket_bolt_dense_contact_stays_finite_and_caged",
                gpu_proof: "make test-hydroelastic-gpu",
                notes: "hydroelastic patch force/reduction is proven with reduced CPU/GPU kernels; the viewer reuses the procedural nut/bolt contact geometry until hydroelastic forces are coupled into the live solver",
            },
            Scene::CableTwist => ScenePhysicsAudit {
                status: Partial,
                reference_case: "Newton cable twist",
                cpu_proof: "reduced_cable_twist_stiffness_controls_torsional_return",
                gpu_proof: "no full twist DoF/kernel yet; VBD cable bending remains covered by make test-vbd-tets-gpu",
                notes: "twist has a reduced scalar CPU oracle only; full articulated cable twist remains future work",
            },
            Scene::MpmBeamTwist => ScenePhysicsAudit {
                status: Partial,
                reference_case: "Newton mpm/example_mpm_beam_twist.py",
                cpu_proof: "mpm_beam_twist_initialization_has_bounded_torsion + mpm_beam_twist_color_tracks_angle + mpm_beam_twist_audit_is_partial",
                gpu_proof: "make test-mpm-gpu / make test-mpm-v3-gpu for existing forward MPM lanes; dedicated torsional beam constitutive setup remains future work",
                notes: "viewer initializes a twisted procedural beam and torsion coloring over the current MPM runtime; true torsional material response/parity is not complete",
            },
            Scene::Mpm
            | Scene::MpmSnowBall
            | Scene::MpmTwoWay
            | Scene::MpmMultiMaterial
            | Scene::WarpDem
            | Scene::WarpFluid
            | Scene::WarpSph
            | Scene::ImplicitMpmBingham
            | Scene::ImplicitMpmFull27 => ScenePhysicsAudit {
                status: ManualGpu,
                reference_case: "MPM granular/fluid/snow/two-way/implicit",
                cpu_proof: "implicit_mpm and reduced MPM tests; reduced_multi_material_oracle_dispatches_distinct_branches for multi-material",
                gpu_proof: "make test-mpm-gpu / make test-mpm-v3-gpu",
                notes: "multi-material viewer currently visualizes material regions over the single-rheology MPM runtime; per-particle material-id GPU dispatch remains future work",
            },
            Scene::Jelly
            | Scene::SoftbodyDrop
            | Scene::ClothCollar
            | Scene::FastSoftImpact
            | Scene::SoftBrake
            | Scene::SoftbodyHanging
            | Scene::JellyDampingFall => ScenePhysicsAudit {
                status: Complete,
                reference_case: "VBD tet/softbody/contact/friction/CCD",
                cpu_proof: "soft_vt_contact / soft_friction / soft_ccd tests",
                gpu_proof: "make test-vbd-tets-gpu / make test-soft-vt-gpu / make test-soft-friction-gpu where supported",
                notes: "EE/CCD pieces may remain CPU-only until practical GPU kernels exist",
            },
            Scene::Pendulum | Scene::JointSampler | Scene::CartPole | Scene::CartPoleSwarm => {
                ScenePhysicsAudit {
                    status: Complete,
                    reference_case: "basic articulated mechanics/control",
                    cpu_proof: "Featherstone and scene smoke tests",
                    gpu_proof: "make test-featherstone-gpu",
                    notes: "procedural articulated stress tests",
                }
            }
            Scene::Conveyor => ScenePhysicsAudit {
                status: Complete,
                reference_case: "kinematic conveyor friction",
                cpu_proof: "scene_correctness conveyor",
                gpu_proof: "XPBD GPU friction parity targets as applicable",
                notes: "XPBD support-load/kinematic-velocity fix is covered",
            },
            Scene::Pyramid | Scene::BrickWall | Scene::FourBar => ScenePhysicsAudit {
                status: Complete,
                reference_case: "stacking / closed-chain stress",
                cpu_proof: "scene_correctness + equality_joint_mimic tests",
                gpu_proof: "make test-equality-gpu / make test-equality-joint-mimic-gpu",
                notes: "hard-contact and closure stress scenes",
            },
            Scene::FemHeat
            | Scene::FemHeatP2
            | Scene::WarpFemDiffusion
            | Scene::WarpFemDiffusion3D => ScenePhysicsAudit {
                status: Complete,
                reference_case: "FEM heat/diffusion",
                cpu_proof: "fem_v1 / fem_p2 heat tests and viewer heat recipe",
                gpu_proof: "make test-fem-gpu / make test-fem-transient-gpu",
                notes: "procedural P1/P2 heat diffusion has CPU oracles and explicit GPU assembly/transient lanes",
            },
            Scene::FemBeam | Scene::FemDiffBeam | Scene::WarpFemElasticOpt => ScenePhysicsAudit {
                status: Complete,
                reference_case: "FEM elasticity / differentiable shape optimization",
                cpu_proof: "fem_elasticity / fem_diff_optim / fem_shape_optim / fem_adjoint tests",
                gpu_proof: "make test-fem-elasticity-gpu",
                notes: "elasticity and reduced shape/load optimization are tested; GPU lane covers assembly/solve parity",
            },
            Scene::WarpFemMixedElasticity => ScenePhysicsAudit {
                status: Partial,
                reference_case: "Warp fem/example_mixed_elasticity.py",
                cpu_proof: "warp_fem_mixed_elasticity_reduced_near_incompressible_stays_finite + warp_fem_mixed_elasticity_pressure_split_beats_locking",
                gpu_proof: "make test-fem-elasticity-gpu for displacement elasticity assembly; mixed pressure GPU solve remains future work",
                notes: "reduced pressure-split oracle and viewer comparison exist; full mixed finite-element pressure/displacement solve is not implemented yet",
            },
            Scene::WarpFemNonconformingContact => ScenePhysicsAudit {
                status: Partial,
                reference_case: "Warp fem/example_nonconforming_contact.py",
                cpu_proof: "fem_nonconforming* projection/penalty tests + fem_surface_contact + warp_fem_nonconforming_contact_patch_preserves_force_and_moment",
                gpu_proof: "CPU oracle only; no nonconforming contact GPU kernel yet",
                notes: "reduced nonmatching-surface load-transfer scene preserves force/moment; full nonconforming contact solve/GPU parity remains future work",
            },
            Scene::WarpFemStokesTransfer => ScenePhysicsAudit {
                status: Partial,
                reference_case: "Warp fem/example_stokes_transfer.py",
                cpu_proof: "warp_fem_stokes_transfer_preserves_constant_field + warp_fem_stokes_transfer_linear_rotation_is_bounded + warp_fem_stokes_transfer_tracers_follow_velocity",
                gpu_proof: "make test-fem-apic-gpu for reusable APIC/field-transfer GPU lanes; full Stokes-to-particle transfer GPU scene remains future work",
                notes: "reduced bilinear Stokes/FEM velocity-grid transfer drives tracers in the viewer; this is not yet a coupled FEM Stokes GPU transfer pipeline",
            },
            Scene::FemStokes | Scene::WarpFemKelvinHelmholtz => ScenePhysicsAudit {
                status: Complete,
                reference_case: "Stokes / Navier-Stokes PDE scenes",
                cpu_proof: "fem_stokes / fem_ns_transient tests",
                gpu_proof: "make test-fem-stokes-gpu / make test-fem-ns-gpu",
                notes: "lid-driven Stokes (Uzawa-CG) and Kelvin-Helmholtz (FEM Navier-Stokes step) run live PDE solvers",
            },
            Scene::WarpFemTaylorGreen | Scene::WarpFemStreamlines => ScenePhysicsAudit {
                status: Partial,
                reference_case: "Warp fem/example_taylor_green.py / example_streamlines.py",
                cpu_proof: "scene-build smoke only — these scenes evaluate a closed-form field; no PDE solver is invoked",
                gpu_proof: "n/a for the analytic formulation; FEM Stokes/NS GPU lanes are make test-fem-stokes-gpu / make test-fem-ns-gpu",
                notes: "reduced analytic visualizations — Taylor-Green renders the closed-form Stokes decay envelope and streamlines advect tracers through an analytic vortex; neither runs a live FEM solve",
            },
            Scene::WarpFemConvDiff => ScenePhysicsAudit {
                status: Complete,
                reference_case: "FEM convection-diffusion",
                cpu_proof: "fem_convection tests",
                gpu_proof: "CPU-owned reduced FEM convection; no dedicated GPU convection target yet",
                notes: "procedural CPU convection-diffusion is complete; GPU promotion can be future work",
            },
            Scene::WarpFemConvDiffDg => ScenePhysicsAudit {
                status: Partial,
                reference_case: "Warp fem/example_convection_diffusion_dg.py",
                cpu_proof: "warp_fem_convdiff_dg_reduced_step_stays_bounded_and_monotone + warp_fem_convdiff_dg_limiter_clamps_overshoot",
                gpu_proof: "make test-fem-dg-swe-gpu for reusable DG flux/limiter GPU lanes; scalar conv-diff DG GPU kernel remains future work",
                notes: "reduced DG-P0 advection-diffusion viewer scene separate from continuous SUPG scene; full DG convection-diffusion assembly/GPU parity remains future work",
            },
            Scene::WarpFemDeformedGeometry | Scene::WarpFemDistortion | Scene::WarpDistortion3D => {
                ScenePhysicsAudit {
                    status: Complete,
                    reference_case: "FEM deformed-geometry/distortion energy",
                    cpu_proof: "fem_distortion tests",
                    gpu_proof: "CPU-owned geometric energy oracle; not a GPU throughput path",
                    notes: "distortion/deformed-geometry scenes are small CPU energy/geometry oracles",
                }
            }
            Scene::DgBurgersShock | Scene::ShallowWaterDamBreak => ScenePhysicsAudit {
                status: Complete,
                reference_case: "DG Burgers/shallow-water",
                cpu_proof: "fem_dg / fem_dg_advanced / fem_dg_v2 / fem_dg_v3 tests",
                gpu_proof: "make test-fem-dg-swe-gpu",
                notes: "DG scalar/SWE physics has CPU shock/dam-break oracles and GPU flux/limiter parity",
            },
            Scene::AmrHeat => ScenePhysicsAudit {
                status: Complete,
                reference_case: "AMR heat",
                cpu_proof: "fem_amr* tests",
                gpu_proof: "CPU-owned AMR adapt/coarsen/warm-start path; no dedicated GPU AMR target yet",
                notes: "AMR is a topology/adaptivity feature with CPU oracles; GPU solve lanes remain separate",
            },
            Scene::Magnetostatics => ScenePhysicsAudit {
                status: Complete,
                reference_case: "magnetostatics / eddy-current FEM",
                cpu_proof: "fem_magnetostatics / fem_eddy_current* tests",
                gpu_proof: "CPU-owned Nedelec/AMS/MG/AMG path; no dedicated GPU Nedelec target yet",
                notes: "electromagnetic FEM is currently CPU oracle coverage with multilevel solver gates",
            },
            Scene::PhaseFieldFracture => ScenePhysicsAudit {
                status: Complete,
                reference_case: "phase-field fracture",
                cpu_proof: "fem_phase_field tests",
                gpu_proof: "CPU-owned fracture oracle; no dedicated GPU phase-field target yet",
                notes: "reduced AT2 phase-field fracture is complete in CPU lane",
            },
            Scene::Vehicle => ScenePhysicsAudit {
                status: CpuOnly,
                reference_case: "wheeled vehicle on flat terrain",
                cpu_proof: "molla-vehicle crate test suite",
                gpu_proof: "CPU-only vehicle model; tyre/suspension/powertrain run on the host",
                notes: "drivable car: arrow keys throttle/brake/steer; molla-vehicle force-element tyres",
            },
            Scene::SphDamBreak => ScenePhysicsAudit {
                status: ManualGpu,
                reference_case: "WCSPH dam break (Tait EOS + Akinci boundary particles)",
                cpu_proof: "sph_oracle f64 CPU oracle; SPH CPU/GPU parity tests",
                gpu_proof: "make test-gpu P=molla-solvers T=sph_gpu / T=sph_boundary_gpu",
                notes: "real GPU WCSPH solver — column collapse held by static boundary particles; F1-F5 forward pipeline",
            },
            Scene::LidarV2 => ScenePhysicsAudit {
                status: ManualGpu,
                reference_case: "terrestrial LiDAR scan-grid (θ × φ) + cone sub-rays",
                cpu_proof: "lidar.rs validate + cone_sample_into property tests",
                gpu_proof: "make test-gpu P=molla-sensors T=lidar_scan_grid_gpu / T=lidar_multi_return_gpu / T=raycast_trimesh_gpu",
                notes: "H4 + H5 + H6 viewer scene — exercises ScanMetadata, beam-divergence cone, and TriMesh ray hits (H1)",
            },
            Scene::AerialLidarPass => ScenePhysicsAudit {
                status: ManualGpu,
                reference_case: "aerial LiDAR over a TriMesh ground",
                cpu_proof: "aerial_validate inline tests + halton 2D unit-square spread",
                gpu_proof: "make test-gpu P=molla-sensors T=aerial_lidar_gpu",
                notes: "H7 viewer scene — Halton-distributed pulses on the footprint upper face, nadir-only by default",
            },
            Scene::SolarSky => ScenePhysicsAudit {
                status: CpuOnly,
                reference_case: "NOAA solar-position lighting for an outdoor scene",
                cpu_proof: "molla-math solar.rs property tests (sun east AM / west PM, unit-length dir, deterministic)",
                gpu_proof: "CPU-only sun-direction; lighting feeds Bevy's DirectionalLight",
                notes: "H13 viewer scene — date/time slider drives the directional light",
            },
            Scene::SyntheticAnnotation => ScenePhysicsAudit {
                status: ManualGpu,
                reference_case: "labelled-scene render + per-instance bbox extraction",
                cpu_proof: "dataset.rs bbox-invariant and annotate_checked tests",
                gpu_proof: "make test-gpu P=molla-sensors T=dataset_export_gpu",
                notes: "H9 viewer scene — TiledCamera SHAPE_INDEX channel + Dataset post-process, bboxes drawn as wireframes",
            },
            Scene::TireModelLab => ScenePhysicsAudit {
                status: CpuOnly,
                reference_case: "force-element tire models (Rigid / TMsimple / Pac89)",
                cpu_proof: "molla-vehicle tire module unit tests; magic-formula derivative parity",
                gpu_proof: "CPU-only tire-model evaluation; the lab is a visual readout, not a sim",
                notes: "V1-V2 viewer scene — slip-ratio / slip-angle sweep for each tire model drawn as gizmo line strips",
            },
            Scene::VehicleOffroad => ScenePhysicsAudit {
                status: CpuOnly,
                reference_case: "wheeled vehicle on a HeightfieldTerrain",
                cpu_proof: "molla-vehicle Vehicle + HeightfieldTerrain unit tests",
                gpu_proof: "CPU-only vehicle dynamics; same path as Scene::Vehicle but with a heightfield",
                notes: "V3-V5 viewer scene — Vehicle::car driven by arrow keys over a procedural heightfield",
            },
            Scene::BekkerSinkage => ScenePhysicsAudit {
                status: CpuOnly,
                reference_case: "SCM Bekker pressure-sinkage on DeformableTerrain",
                cpu_proof: "molla-vehicle DeformableTerrain rutting tests + new bekker_n exponent",
                gpu_proof: "CPU-only — SCM is a per-cell phenomenological model",
                notes: "V5 viewer scene — loading plate ramps fz, curve drawn as a 3D line strip",
            },
            Scene::SoilCompaction => ScenePhysicsAudit {
                status: CpuOnly,
                reference_case: "DEM granular bed under repeated roller passes",
                cpu_proof: "molla-vehicle GranularBed::step + wheel_obstacles tests",
                gpu_proof: "CPU-only DEM — rut-depth-vs-pass measured in the viewer loop",
                notes: "V5-deep viewer scene — roller sweeps the bed; per-pass rut depths plotted",
            },
            Scene::TireDynamics => ScenePhysicsAudit {
                status: ManualGpu,
                reference_case: "volumetric tetra-FEM tire carcass + pressure-dependent contact patch",
                cpu_proof: "molla-vehicle real_tire tetra FEM tests cover volumetric mesh creation, floor contact, and lower-pressure wider patches",
                gpu_proof: "molla-kernels tire_fem.wgsl node-step kernel; make test-gpu P=molla-kernels T=tire_fem_gpu",
                notes: "T1-T3 viewer scene defaults to GPU tetra FEM nodes; CPU remains a right-rail fallback/debug toggle",
            },
            Scene::TireTraction => ScenePhysicsAudit {
                status: CpuOnly,
                reference_case: "brush tire contact with pressure/material-dependent patch traction under heavy payload",
                cpu_proof: "molla-vehicle real_tire brush-contact tests cover pressure, material friction, and low-mu traction response",
                gpu_proof: "CPU-only viewer comparison; GPU brush/contact kernel is tracked in PLAN_REALTIRE.md",
                notes: "T4 viewer scene — fixed red/blue cars plus user-controlled green car with pressure, payload, and material sliders",
            },
            Scene::WarpRaymarch => ScenePhysicsAudit {
                status: Partial,
                reference_case: "Warp volume/raymarch",
                cpu_proof: "warp_raymarch_sdf_ray_hits_signed_volume",
                gpu_proof: "no GPU volume raymarch kernel yet",
                notes: "CPU signed-volume sphere tracing exists; GPU raymarch parity remains future work",
            },
            Scene::WarpNvdb => ScenePhysicsAudit {
                status: Partial,
                reference_case: "Warp NanoVDB / sparse-SDF lookup",
                cpu_proof: "warp_nvdb_sparse_sdf_lookup_has_active_band_and_normals",
                gpu_proof: "make test-sparse-grid-gpu",
                notes: "sparse-grid CPU/GPU lookup parity exists; this is not true NanoVDB file/brick-layout parity",
            },
            Scene::WarpTileNbody => ScenePhysicsAudit {
                status: CpuOnly,
                reference_case: "Warp tile N-body",
                cpu_proof: "warp_tile_nbody_conserves_pairwise_momentum_and_center_of_mass",
                gpu_proof: "CPU-only host O(N^2) demo; no reusable N-body GPU primitive exists in molla-compute-primitives",
                notes: "all-pairs inverse-square gravity is tested as a reduced viewer physics demo",
            },
            Scene::WarpTileFft => ScenePhysicsAudit {
                status: Complete,
                reference_case: "Warp tile FFT",
                cpu_proof: "viewer host DFT readout + FFT CPU/GPU parity references",
                gpu_proof: "make test-fft-gpu",
                notes: "viewer pane uses host DFT for a tiny live readout; reusable GPU FFT is covered by the FFT GPU lane",
            },
            Scene::WarpTileMlp => ScenePhysicsAudit {
                status: Complete,
                reference_case: "Warp tile MLP",
                cpu_proof: "MLP forward routed through tile_matmul CPU primitive",
                gpu_proof: "make test-tile-primitives-gpu / make test-actuator-gpu",
                notes: "tiny demo is CPU due dispatch overhead; reusable tiled GEMM and neural actuator GPU paths are covered",
            },
            Scene::WarpTileCholesky
            | Scene::WarpTileFilter
            | Scene::WarpTileCompaction
            | Scene::WarpTileMatmul
            | Scene::WarpTileConvolution => ScenePhysicsAudit {
                status: Complete,
                reference_case: "Warp tile compute primitives",
                cpu_proof: "molla-compute-primitives tests",
                gpu_proof: "make test-tile-primitives-gpu",
                notes: "tiled Cholesky/filter/compaction/matmul/convolution have CPU oracles and explicit GPU parity",
            },
            Scene::WarpTileMcgp => ScenePhysicsAudit {
                status: CpuOnly,
                reference_case: "Monte Carlo Laplace / MCGP",
                cpu_proof: "molla-compute-primitives tile_mcgp CPU tests",
                gpu_proof: "CPU-only stochastic oracle; no reusable GPU MCGP kernel exists",
                notes: "viewer runs a reduced walk-on-spheres Laplace solve; GPU promotion is future work",
            },
            Scene::WarpParticleRepulsion | Scene::WarpDiffRay | Scene::AdamLandscape => {
                ScenePhysicsAudit {
                    status: Complete,
                    reference_case: "optimization/differentiable geometry",
                    cpu_proof: "autograd/optim/raycast-adjoint tests",
                    gpu_proof: "make test-optim-gpu / make test-robust-loss-gpu where applicable",
                    notes: "reduced optimization demos have CPU gradient/optimizer oracles and explicit GPU optimizer/loss lanes",
                }
            }
            Scene::SplatPoseRecovery => ScenePhysicsAudit {
                status: Complete,
                reference_case: "differentiable 3D Gaussian splat pose recovery",
                cpu_proof: "splat SH/basis CPU pieces in kernel tests",
                gpu_proof: "make test-splat-gpu / make test-splat-pose-gpu",
                notes: "3DGS forward/backward/SH kernels and pose recovery have explicit GPU acceptance lanes",
            },
            Scene::WarpFftFluid | Scene::WarpFftPoisson2d => ScenePhysicsAudit {
                status: Complete,
                reference_case: "FFT Poisson / fractional-step fluid",
                cpu_proof: "fft_ns_coupling analytic Poisson/Taylor-Green tests",
                gpu_proof: "make test-fem-fft-ns-gpu / make test-fft-gpu",
                notes: "GPU FFT pressure solve and fractional-step Taylor-Green decay have explicit parity gates",
            },
            Scene::WarpWave => ScenePhysicsAudit {
                status: Complete,
                reference_case: "FEM wave equation",
                cpu_proof: "fem_wave tests",
                gpu_proof: "make test-fem-wave-gpu",
                notes: "wave stepper has CPU oracle and GPU parity/nonzero-Dirichlet gates",
            },
            Scene::WarpApicFluid => ScenePhysicsAudit {
                status: Complete,
                reference_case: "APIC FEM/MPM transfer",
                cpu_proof: "fem_apic tests",
                gpu_proof: "make test-fem-apic-gpu",
                notes: "APIC transfer and falling-particle GPU parity are explicit; full fluid solve remains a higher-level coupling feature",
            },
            Scene::WarpStableFluid => ScenePhysicsAudit {
                status: Complete,
                reference_case: "Stam stable fluid",
                cpu_proof: "stam_fluid tests",
                gpu_proof: "CPU-owned reduced stable-fluid demo; spectral/NS GPU path covered by make test-fem-fft-ns-gpu",
                notes: "stable-fluid CPU solver has mass/source/force/diffusion/projection gates; GPU fluid acceleration is covered by the FFT/NS lane",
            },
            Scene::DiffsimDrone => ScenePhysicsAudit {
                status: Partial,
                reference_case: "Newton differentiable drone/MPC",
                cpu_proof: "diffsim_drone_mpc_gradient_matches_finite_difference / diffsim_drone_mpc_adam_reduces_final_position_loss",
                gpu_proof: "CPU-only reduced point-mass MPC; no GPU-owned autodiff/control path yet",
                notes: "procedural point-mass MPC now uses an analytic CPU gradient; not full Featherstone drone or Newton parity",
            },
            Scene::FluidCheckpointOptimization => ScenePhysicsAudit {
                status: Partial,
                reference_case: "Warp optim/example_fluid_checkpoint.py",
                cpu_proof: "fluid_checkpoint_bounds_saved_windows + fluid_checkpoint_optimization_reduces_loss + fluid_checkpoint_replay_is_deterministic",
                gpu_proof: "make test-differentiable for checkpoint/autograd lanes; make test-fem-ns-gpu / make test-fft-gpu cover related forward fluid/PDE GPU paths",
                notes: "reduced scalar fluid-like checkpoint optimization exists; full GPU fluid adjoint/checkpoint replay remains future work",
            },
            Scene::NavierStokesPerturbationOptimization => ScenePhysicsAudit {
                status: Partial,
                reference_case: "Warp optim/example_navier_stokes_perturbation.py",
                cpu_proof: "ns_perturbation_parameter_changes_flow_response + ns_perturbation_gradient_matches_finite_difference + ns_perturbation_optimization_reduces_loss",
                gpu_proof: "make test-fem-ns-gpu / make test-fem-fft-ns-gpu for related forward Navier-Stokes GPU lanes; reduced perturbation gradient remains CPU",
                notes: "reduced scalar perturbation optimizer over a procedural flow response; full GPU NS adjoint/perturbation optimization remains future work",
            },
        }
    }
}

#[derive(Resource)]
struct PendingSceneSwitch(Option<Scene>);

/// Muli-style activity visualisation. When `enabled`, every body that
/// is below the kinematic threshold is rendered in `sleeping_mat`
/// (uniform grey), while moving bodies keep their original palette
/// colour. Toggled by the `T` key.
#[derive(Resource, Default)]
struct ActivityView {
    enabled: bool,
    /// Cached count of "active" bodies — refreshed each frame in
    /// `update_activity_colors` and shown in the panel.
    active_count: usize,
    sleeping_mat: Option<Handle<StandardMaterial>>,
}

/// Per-body cache of the body's "awake" (palette) material so we can
/// flip back to it when the body wakes up again. The active material
/// lives in `MeshMaterial3d` like normal — this is just a snapshot of
/// the original colour assigned at spawn time.
#[derive(Component)]
struct AwakeMaterial(Handle<StandardMaterial>);

/// Tag for the deforming cloth mesh entity. Stage-12 cloth is rendered
/// as a single Bevy `Mesh` whose vertex positions are rewritten each
/// frame from `state.particle_q`.
#[derive(Component)]
struct ClothMesh;

/// Tag for static obstacles in the cloth scene (the sphere the
/// cloth drapes over). Rendered as a regular `Mesh3d` but doesn't
/// participate in molla's body/state pipeline.
#[derive(Component)]
struct ClothObstacle;

/// Whether to draw wireframes over the physics meshes (not the ground
/// grid). Toggled by `W`. We use the per-entity `Wireframe` component
/// rather than `WireframeConfig.global` so bevy_glacial's ground grid
/// is left alone.
#[derive(Resource, Default)]
struct WireframeOn(bool);

/// Per-frame configuration for the Lidar example. Driven by the
/// right-rail panel sliders. The `update_lidar` system rebuilds N
/// rays from this config every frame, runs the GPU raycaster, and
/// stashes the results in [`LidarState`] for the gizmo system.
#[derive(Resource, Clone, Copy)]
struct LidarConfig {
    /// World-space lidar origin.
    origin: Vec3,
    /// Heading angle around +Y, radians.
    yaw: f32,
    /// Up/down tilt, radians (0 = horizontal).
    pitch: f32,
    /// Horizontal FOV (radians).
    fov_h: f32,
    /// Vertical FOV (radians).
    fov_v: f32,
    /// Horizontal ray count (azimuth).
    ray_count_h: u32,
    /// Vertical ray count (elevation).
    ray_count_v: u32,
    /// Max ray distance (m).
    max_range: f32,
    /// Auto-rotate the yaw each frame (rad/s).
    auto_rotate_speed: f32,
}

impl Default for LidarConfig {
    fn default() -> Self {
        Self {
            origin: Vec3::new(0.0, 1.5, 0.0),
            yaw: 0.0,
            pitch: 0.0,
            fov_h: 120f32.to_radians(),
            fov_v: 30f32.to_radians(),
            ray_count_h: 64,
            ray_count_v: 8,
            max_range: 20.0,
            auto_rotate_speed: 0.6,
        }
    }
}

/// Solid box principal moments at the COM, body frame.
fn box_inertia(mass: f64, half: f64) -> molla::math::Mat3 {
    let h2 = half * half;
    let s = mass / 3.0;
    molla::math::Mat3::from_diagonal(MVec3::new(s * (h2 + h2), s * (h2 + h2), s * (h2 + h2)))
}

/// Solid capsule (cylinder + 2 hemispheres) approximate inertia
/// about the capsule axis (+Y) and perpendicular axes.
fn capsule_inertia(mass: f64, half_height: f64, radius: f64) -> molla::math::Mat3 {
    let r2 = radius * radius;
    let h2 = half_height * half_height;
    let i_y = 0.5 * mass * r2;
    let i_perp = mass * (h2 / 3.0 + r2 / 4.0);
    molla::math::Mat3::from_diagonal(MVec3::new(i_perp, i_y, i_perp))
}

/// Build the requested solver backend, falling back gracefully when
/// the requested kind isn't available at runtime.
///
/// Every solver kind is native Rust/wgpu; former external-wrapper scenes are
/// routed to Featherstone to keep the viewer physics-only.
fn build_solver(engine: &Engine, model: &mp::Model, kind: SolverKind) -> SolverBackend {
    match kind {
        SolverKind::Xpbd => SolverBackend::Xpbd(
            mp::XpbdSolver::new_gpu(engine, model.body_count.max(1) as u32)
                .expect("XPBD GPU solver init")
                .with_iterations(10),
        ),
        SolverKind::Featherstone => SolverBackend::Featherstone(
            FeatherstoneSolver::new_gpu(engine, model).expect("Featherstone GPU solver init"),
        ),
        SolverKind::Vbd => SolverBackend::Vbd(
            mp::VbdSolver::new_gpu(engine, model)
                .expect("VBD GPU solver init")
                .with_iterations(10)
                .with_contact_stiffness(VBD_CLOTH_CONTACT_STIFFNESS),
        ),
    }
}

impl Sim {
    fn build_scene_model(scene: Scene) -> (mp::Model, Vec<BodyVisual>) {
        match scene {
            Scene::MeshTools => build_mesh_tools_model(),
            Scene::Pile => build_pile_model(),
            Scene::Rope => build_rope_model(),
            Scene::Tower => build_tower_model(),
            Scene::Cloth => build_cloth_model(),
            Scene::Lidar => build_lidar_model(),
            Scene::Imu => build_imu_model(),
            Scene::Contact => build_contact_model(),
            // Stage 32: when the right-rail "Mimic joint" toggle is on
            // (consumed via the global atomic flag set by `arm_pane`),
            // append the mimic side-chain so the user can see the
            // coupled joint move in lockstep with the shoulder.
            Scene::Arm => build_arm_model_with(arm_mimic_flag()),
            Scene::Shapes => build_shapes_model(),
            Scene::GpuIkArm => build_gpu_ik_arm_model(),
            Scene::GpuNarrowPhase => build_gpu_narrow_phase_model(),
            Scene::Camera => build_camera_model(),
            Scene::Actuator => build_actuator_model(),
            Scene::Quadrotor => build_quadrotor_model(),
            Scene::BroadPhase => build_broad_phase_model(),
            Scene::VbdCloth => build_vbd_cloth_model(),
            Scene::ReferencePendulum => build_reference_pendulum_model(),
            Scene::DiffArm => build_diff_arm_model(),
            Scene::DiffArm6 => build_diff_arm6_model(),
            Scene::NeuralPolicy => build_neural_policy_model(),
            // Stage 31 LSTM demo reuses the cart-pole topology — the
            // controller swap is the whole point of the demo.
            Scene::NeuralLstm => build_neural_policy_model(),
            Scene::Cable => build_cable_model(),
            Scene::Mpm => build_mpm_model(),
            Scene::Jelly => build_jelly_model(),
            Scene::Pendulum => build_pendulum_model(),
            Scene::JointSampler => build_joint_sampler_model(),
            Scene::Heightfield => build_heightfield_model(),
            Scene::Conveyor => build_conveyor_model(),
            Scene::Pyramid => build_pyramid_model(),
            Scene::ClothBending => build_cloth_bending_model(),
            Scene::ClothRollers => build_cloth_rollers_model(),
            Scene::ClothTwist => build_cloth_twist_model(),
            Scene::CablePile => build_cable_pile_model(),
            Scene::CableYJunction => build_cable_y_junction_model(),
            Scene::CableBundleHysteresis => build_cable_bundle_hysteresis_model(),
            Scene::NutBoltSdf => build_nut_bolt_sdf_model(),
            Scene::NutBoltHydro => build_nut_bolt_hydro_model(),
            Scene::MpmSnowBall => build_mpm_model(),
            Scene::MpmTwoWay => build_mpm_model(),
            Scene::MpmMultiMaterial => build_mpm_model(),
            Scene::MpmBeamTwist => build_mpm_model(),
            Scene::CartPole => build_cartpole_model(),
            Scene::CartPoleSwarm => build_cartpole_swarm_model(),
            Scene::FourBar => build_fourbar_model(),
            Scene::DiffBall => build_diff_ball_model(),
            Scene::DiffSpring => build_diff_spring_model(),
            Scene::DiffsimCloth => build_diffsim_cloth_model(),
            Scene::DiffsimSoftBody => build_diffsim_softbody_model(),
            Scene::BrickWall => build_brick_wall_model(),
            Scene::SoftbodyDrop => build_softbody_drop_model(),
            Scene::ClothCollar => build_cloth_collar_model(),
            Scene::FastSoftImpact => build_fast_soft_impact_model(),
            Scene::SoftBrake => build_soft_brake_model(),
            Scene::FemHeat => build_fem_heat_model(),
            Scene::FemHeatP2 => build_fem_heat_p2_model(),
            Scene::FemBeam => build_fem_beam_model(),
            Scene::FemDiffBeam => build_fem_diff_beam_model(),
            Scene::FemStokes => build_fem_stokes_model(),
            // Warp-port scenes — most are host-driven and use the
            // raycast / kinematic primitive build helpers below.
            Scene::WarpRaycast => build_warp_raycast_model(),
            Scene::WarpMesh => build_warp_mesh_model(),
            Scene::WarpMeshIntersect => build_warp_mesh_intersect_model(),
            Scene::WarpSampleMesh => build_warp_empty_model(),
            Scene::WarpMarchingCubes => build_warp_empty_model(),
            Scene::WarpNvdb => build_warp_empty_model(),
            Scene::WarpDem => build_mpm_model(),
            Scene::WarpFluid => build_mpm_model(),
            Scene::WarpSph => build_mpm_model(),
            Scene::WarpTileNbody => build_warp_empty_model(),
            // The rest (raymarch / tile FFT / MLP / Cholesky / repulsion
            // / FEM ports) are pure-CPU compute demos with no scene
            // bodies — pane and gizmos do all the work.
            Scene::WarpRaymarch
            | Scene::WarpTileFft
            | Scene::WarpTileMlp
            | Scene::WarpTileCholesky
            | Scene::WarpParticleRepulsion
            | Scene::WarpFemDiffusion
            | Scene::WarpFemDiffusion3D
            | Scene::WarpFemConvDiff
            | Scene::WarpFemConvDiffDg
            | Scene::WarpFemTaylorGreen
            | Scene::WarpFemKelvinHelmholtz
            | Scene::WarpFemStreamlines
            | Scene::WarpFemStokesTransfer
            | Scene::WarpFemDeformedGeometry
            | Scene::WarpFemDistortion
            | Scene::WarpFemElasticOpt
            | Scene::WarpFemMixedElasticity
            | Scene::WarpFemNonconformingContact
            | Scene::WarpFftFluid
            | Scene::WarpDistortion3D
            | Scene::WarpApicFluid
            | Scene::WarpWave
            | Scene::WarpDiffRay
            | Scene::WarpTileFilter
            | Scene::WarpTileCompaction
            | Scene::AdamLandscape
            | Scene::DiffsimDrone
            | Scene::FluidCheckpointOptimization
            | Scene::NavierStokesPerturbationOptimization
            | Scene::WarpFftPoisson2d
            | Scene::WarpTileMatmul
            | Scene::WarpTileConvolution
            | Scene::WarpTileMcgp
            | Scene::ImplicitMpmBingham
            | Scene::WarpStableFluid
            | Scene::DgBurgersShock
            | Scene::ShallowWaterDamBreak
            | Scene::ImplicitMpmFull27
            | Scene::SplatPoseRecovery
            | Scene::AmrHeat
            | Scene::Magnetostatics
            | Scene::PhaseFieldFracture => build_warp_empty_model(),
            Scene::Vehicle => build_vehicle_model(),
            Scene::SphDamBreak => build_mpm_model(),
            Scene::LidarV2 => build_lidar_v2_model(),
            Scene::AerialLidarPass => build_aerial_lidar_model(),
            Scene::SolarSky => build_solar_sky_model(),
            Scene::SyntheticAnnotation => build_synth_annotation_model(),
            Scene::TireModelLab => build_tire_lab_model(),
            Scene::VehicleOffroad => build_vehicle_model(), // reuse the chassis + wheels
            Scene::BekkerSinkage => build_bekker_sinkage_model(),
            Scene::SoilCompaction => build_soil_compaction_model(),
            Scene::TireDynamics => build_tire_dynamics_model(),
            Scene::TireTraction => build_tire_traction_model(),
            Scene::ClothPokerCards => build_cloth_poker_cards_model(),
            Scene::SoftbodyHanging => build_softbody_hanging_model(),
            Scene::CableTwist => build_cable_twist_model(),
            Scene::IkCustom | Scene::IkCubeStacking => build_gpu_ik_arm_model(),
            Scene::JellyDampingFall => build_jelly_damping_fall_model(),
        }
    }

    /// Dispatch on `scene` and assemble `(Sim, BodyVisuals)`.
    fn build_scene(scene: Scene) -> (Self, Vec<BodyVisual>) {
        let (model, visuals) = Self::build_scene_model(scene);
        Self::finalize(scene, model, visuals)
    }

    /// Common upload + GPU solver wiring shared by every scene.
    fn finalize(
        scene: Scene,
        mut model: mp::Model,
        visuals: Vec<BodyVisual>,
    ) -> (Self, Vec<BodyVisual>) {
        model.default_friction = PLANE_FRICTION;
        model.default_restitution = PLANE_RESTITUTION;
        model.contact_margin = 0.02;
        model.linear_damping = LINEAR_DAMPING;

        // Stage hardening: per-shape friction/restitution were baked
        // at `b.build()` from the BUILDER's default (0.5 / 0.0), not
        // the viewer's preferred PLANE_FRICTION / PLANE_RESTITUTION.
        // Overwrite any default-valued slot so the constants above
        // actually reach the contact resolver. Per-shape overrides
        // (e.g. `Shapes` scene's rolling friction, BrickWall's
        // explicit values via `set_shape_friction*`) are preserved
        // by skipping the rewrite when the slot diverges from 0.5
        // (the builder default). We can't tell "user set 0.5
        // explicitly" from "default 0.5", but no scene currently does
        // that, so the heuristic is safe.
        if let Ok(fric) = model.shape_friction.host_mut() {
            for v in fric.iter_mut() {
                if (*v - 0.5).abs() < 1e-9 {
                    *v = PLANE_FRICTION;
                }
            }
        }
        if let Ok(rest) = model.shape_restitution.host_mut() {
            for v in rest.iter_mut() {
                if v.abs() < 1e-9 {
                    *v = PLANE_RESTITUTION;
                }
            }
        }

        let mut state_a = model.state().expect("state");
        let mut state_b = model.state().expect("state");
        let mut control = model.control();
        let mut contacts = mp::Contacts::with_capacity(8192);

        // Stage 33.4 — FastSoftImpact: launch the tet cube downward at
        // the user-controlled impact speed (default 5 m/s) so the
        // first substep already needs CCD to avoid tunneling. The
        // cube particles are the trailing batch in `particle_q` (one
        // cloth grid + one tet cube), addressed by their flat index
        // range `[cloth_count..particle_count]`.
        if matches!(scene, Scene::FastSoftImpact) {
            let v_target = -fast_soft_impact_speed();
            let cloth_count = (FAST_SOFT_IMPACT_CLOTH_NX * FAST_SOFT_IMPACT_CLOTH_NX) as usize;
            let p_total = model.particle_count;
            if let Ok(qd_a) = state_a.particle_qd.host_mut() {
                for i in cloth_count..p_total {
                    qd_a[i] = MVec3::new(0.0, v_target, 0.0);
                }
            }
            if let Ok(qd_b) = state_b.particle_qd.host_mut() {
                for i in cloth_count..p_total {
                    qd_b[i] = MVec3::new(0.0, v_target, 0.0);
                }
            }
        }

        // GPU-first: bring up wgpu and bind the XPBD GPU solver.
        let engine = Engine::new_gpu().expect(
            "molla viewer is GPU-first; wgpu adapter required. \
             Run via `make run RELEASE=1` (nixVulkan-wrapped).",
        );
        let adapter_label = engine
            .adapter_info()
            .map(|i| format!("{} ({})", i.name, i.backend))
            .unwrap_or_else(|| "unknown".into());
        let device = engine.device().unwrap().clone();
        let queue = engine.queue().unwrap().clone();

        model
            .upload_to_device(&device, &queue)
            .expect("model upload");
        state_a
            .upload_to_device(&device, &queue)
            .expect("state_a upload");
        state_b
            .upload_to_device(&device, &queue)
            .expect("state_b upload");
        control
            .upload_to_device(&device, &queue)
            .expect("control upload");
        contacts
            .upload_to_device(&device, &queue)
            .expect("contacts upload");

        // Pick the solver this scene wants. XPBD for everything but
        // the Arm scene; Featherstone for proper articulated dynamics
        // (joint_q, joint_qd, joint_force, etc.).
        let mut solver = build_solver(&engine, &model, scene.solver_kind());
        // Stage 33 — when multiple soft surfaces are registered (the
        // SoftbodyDrop scene), tune the VT contact pipeline for the
        // typical viewer cell size (~5cm). Without this the default
        // 1cm margin can let fast-moving vertices skip over the
        // contact.
        if matches!(scene, Scene::SoftbodyDrop) {
            if let SolverBackend::Vbd(s) = &mut solver {
                s.vt_contact_margin = 0.05;
                s.vt_contact_sweeps = 8;
                s.contact_stiffness = 1.0e5;
            }
        }
        // Stage 33.3 — cloth-collar leans on self-collision.
        if matches!(scene, Scene::ClothCollar) {
            if let SolverBackend::Vbd(s) = &mut solver {
                s.vt_contact_margin = 0.03;
                s.vt_contact_sweeps = 8;
                s.contact_stiffness = 2.0e5;
            }
        }
        // Stage 33.4 — FastSoftImpact uses CPU VBD because CCD is
        // host-side only in v1. Stage 33.2.1 ships GPU friction +
        // restitution, but the CCD pass for soft contact is still
        // CPU-only (TODO Stage 33.4.1). Without CCD the cube launched
        // at 5 m/s would tunnel through the cloth on the very first
        // substep, so we stay on the CPU backend until the GPU CCD
        // lands.
        if matches!(scene, Scene::FastSoftImpact) {
            solver = SolverBackend::Vbd(
                mp::VbdSolver::new_cpu()
                    .with_iterations(8)
                    .with_contact_stiffness(1.0e5)
                    .with_vt_contact_margin(0.03)
                    .with_vt_contact_sweeps(8)
                    .with_ccd(true)
                    .with_ccd_velocity_threshold(0.5),
            );
        }
        // Stage 33.5 — SoftBrake friction sweep. Stage 33.2.1 brought
        // Coulomb friction + restitution to the GPU kernel so the
        // scene now runs on the default GPU VBD backend (no CPU
        // override needed). Just tune the contact pipeline knobs.
        if matches!(scene, Scene::SoftBrake) {
            if let SolverBackend::Vbd(s) = &mut solver {
                s.vt_contact_margin = 0.02;
                s.vt_contact_sweeps = 10;
                s.contact_stiffness = 2.0e5;
            }
        }

        (
            Self {
                scene,
                model,
                state_a,
                state_b,
                control,
                contacts,
                solver,
                pipeline: CollisionPipeline::new(),
                engine,
                adapter_label,
                mpm: None,
                mpm_particles: Vec::new(),
                mpm_rigid_spheres: Vec::new(),
            },
            visuals,
        )
    }

    /// Replace the active solver while keeping the same model + state
    /// — used by the VBD-cloth toggle for side-by-side comparison.
    fn swap_solver_to(&mut self, kind: SolverKind) {
        self.solver = build_solver(&self.engine, &self.model, kind);
    }

    fn reset(&mut self) {
        let initial = self
            .model
            .initial_body_q
            .host()
            .expect("initial poses")
            .to_vec();

        let device = self.engine.device().unwrap().clone();
        let queue = self.engine.queue().unwrap().clone();
        let _ = self.state_a.download_from_device(&device, &queue);
        let _ = self.state_b.download_from_device(&device, &queue);

        self.state_a
            .body_q
            .host_mut()
            .expect("body_q")
            .copy_from_slice(&initial);
        for v in self.state_a.body_qd.host_mut().expect("body_qd") {
            *v = mp::SpatialVector::ZERO;
        }

        if self.model.particle_count > 0 {
            let initial_p = self
                .model
                .initial_particle_q
                .host()
                .expect("initial particles")
                .to_vec();
            self.state_a
                .particle_q
                .host_mut()
                .expect("particle_q")
                .copy_from_slice(&initial_p);
            for v in self.state_a.particle_qd.host_mut().expect("particle_qd") {
                *v = MVec3::ZERO;
            }
            // Stage 33.4 — FastSoftImpact: re-launch the cube on reset
            // so the user can repeatedly trigger the impact without
            // rebuilding the scene.
            if matches!(self.scene, Scene::FastSoftImpact) {
                let v_target = -fast_soft_impact_speed();
                let cloth_count = (FAST_SOFT_IMPACT_CLOTH_NX * FAST_SOFT_IMPACT_CLOTH_NX) as usize;
                let p_total = self.model.particle_count;
                if let Ok(qd_a) = self.state_a.particle_qd.host_mut() {
                    for i in cloth_count..p_total {
                        qd_a[i] = MVec3::new(0.0, v_target, 0.0);
                    }
                }
            }
        }

        let _ = self.state_a.upload_to_device(&device, &queue);
        let _ = self.state_b.upload_to_device(&device, &queue);
    }
}

// ---- Scene builders -----------------------------------------------

/// 12×12 mixed sphere / box / capsule rain falling onto a static plane.
///
/// Bodies start with **random orientations** and **staggered heights**
/// — so when they hit the ground they tip onto edges/corners and
/// produce off-axis impulses, instead of all bouncing straight up
/// along +Y. Without random rotation a flat-faced box hitting a flat
/// plane only gets a normal impulse, so it just bounces vertically;
/// the tumbling behaviour is what makes it look like a real pile.
/// projection. A small horizontal kick at the bottom makes it swing.
/// Each column is independent — settles immediately, smooth impact.
/// works — drag any particle to deform the drape.
/// for the difference (XPBD wobbles, VBD settles).

/// interior with stiff corner-cell pillars.

/// Lidar example — a static stage with kinematic obstacles for a GPU
/// [`mp::sensors::Raycaster`] to scan. The lidar config + state lives
/// in the [`LidarConfig`] / [`LidarState`] resources; the right-rail
/// pane edits the config and `update_lidar` re-runs the GPU sensor
/// each frame.
///
/// Obstacles are kinematic (mass = 0 + KINEMATIC flag) so they don't
/// fall under gravity — the lidar wants stable targets, not a
/// settling pile.

// --- Arm scene -----------------------------------------------------

//   • body 0 — cart (small box)

// --- Stage-13 specialty-shapes scene --------------------------------

/// Contact-sensor example — a small stack of bodies settling on the
/// plane. The GPU `ContactSensor` runs every frame and the right-rail
/// pane shows per-body contact counts + max depth + net push
/// direction. Gizmo arrows visualise the net push direction at each
/// body's contact centroid.

/// IMU example — a single dynamic sphere bouncing on the floor with
/// an `ImuSensor` mounted at its COM. Restitution is high so the
/// accelerometer reading is dramatic on each impact (specific force
/// spikes when the contact impulse fires).

/// Build a Bevy mesh from a molla `Heightfield` for visualisation.
fn build_heightfield_mesh(hf: &mp::Heightfield) -> Mesh {
    use bevy::asset::RenderAssetUsages;
    use bevy::mesh::{Indices, PrimitiveTopology};
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    let nx = hf.nx as usize;
    let nz = hf.nz as usize;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(nx * nz);
    let mut normals: Vec<[f32; 3]> = vec![[0.0, 1.0, 0.0]; nx * nz];
    for j in 0..nz {
        for i in 0..nx {
            let x = hf.origin.x + i as f64 * hf.cell_size;
            let z = hf.origin.z + j as f64 * hf.cell_size;
            let y = hf.heights[j * nx + i];
            positions.push([x as f32, y as f32, z as f32]);
        }
    }
    let mut indices: Vec<u32> = Vec::with_capacity((nx - 1) * (nz - 1) * 6);
    for j in 0..nz - 1 {
        for i in 0..nx - 1 {
            let v00 = (j * nx + i) as u32;
            let v10 = (j * nx + i + 1) as u32;
            let v01 = ((j + 1) * nx + i) as u32;
            let v11 = ((j + 1) * nx + i + 1) as u32;
            indices.extend_from_slice(&[v00, v10, v11, v00, v11, v01]);
        }
    }
    // Compute per-vertex normals via face-normal averaging.
    for chunk in indices.chunks(3) {
        let i0 = chunk[0] as usize;
        let i1 = chunk[1] as usize;
        let i2 = chunk[2] as usize;
        let p0 = Vec3::from_array(positions[i0]);
        let p1 = Vec3::from_array(positions[i1]);
        let p2 = Vec3::from_array(positions[i2]);
        let n = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        for &k in &[i0, i1, i2] {
            let cur = normals[k];
            normals[k] = [cur[0] + n.x, cur[1] + n.y, cur[2] + n.z];
        }
    }
    for n in normals.iter_mut() {
        let v = Vec3::from_array(*n).normalize_or_zero();
        *n = [v.x, v.y, v.z];
    }
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Build a Bevy `Mesh` (TriangleList) from a molla [`molla::sim::TriMesh`].
/// Computes per-vertex face-averaged normals on the fly. Used by the
/// face data — without faces we'd be back to dot markers.
fn build_hull_wireframe_mesh(hull: &molla::geometry::ConvexHullMesh) -> Mesh {
    use bevy::asset::RenderAssetUsages;
    use bevy::mesh::{Indices, PrimitiveTopology};

    let positions: Vec<[f32; 3]> = hull
        .vertices
        .iter()
        .map(|v| [v.x as f32, v.y as f32, v.z as f32])
        .collect();
    let edges = hull.edges();
    let mut idx_flat: Vec<u32> = Vec::with_capacity(edges.len() * 2);
    for (a, b) in edges {
        idx_flat.push(a);
        idx_flat.push(b);
    }
    let mut mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::RENDER_WORLD);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(Indices::U32(idx_flat));
    mesh
}

/// Build a single Bevy `Mesh` from molla's particle positions and
/// triangle topology. Called once per scene-switch — `sync_cloth_mesh`
/// then mutates the same `Mesh` asset's vertex positions every frame.
fn build_cloth_mesh(sim: &Sim) -> Mesh {
    use bevy::asset::RenderAssetUsages;
    use bevy::mesh::{Indices, PrimitiveTopology};

    let particle_q = sim.state_a.particle_q.host().expect("particle_q host");
    let tri_indices = sim
        .model
        .tri_indices
        .host()
        .expect("tri_indices host")
        .to_vec();

    let positions: Vec<[f32; 3]> = particle_q
        .iter()
        .map(|p| [p.x as f32, p.y as f32, p.z as f32])
        .collect();
    // Provide a flat default UV — good enough for solid-colour shading,
    // can be replaced with proper grid UVs if we ever want a texture.
    let uvs: Vec<[f32; 2]> = (0..positions.len()).map(|_| [0.0, 0.0]).collect();
    let normals: Vec<[f32; 3]> = (0..positions.len()).map(|_| [0.0, 1.0, 0.0]).collect();

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        // MAIN_WORLD so we can mutate vertex positions at runtime.
        // Default usage is RENDER_WORLD only and rejects further writes.
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(tri_indices));
    // Flat per-face normals are fine for cloth — recomputed each
    // frame in `sync_cloth_mesh` so sagging looks lit.
    mesh.compute_normals();
    mesh
}

fn handle_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut exit: MessageWriter<AppExit>,
    mut sim: ResMut<Sim>,
    mut run: ResMut<SimRunState>,
    _pending: ResMut<PendingSceneSwitch>,
    mut activity: ResMut<ActivityView>,
    mut wireframe: ResMut<WireframeOn>,
) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
    if keys.just_pressed(KeyCode::KeyR) {
        sim.reset();
    }
    if keys.just_pressed(KeyCode::Space) {
        run.paused = !run.paused;
    }
    if keys.just_pressed(KeyCode::KeyT) {
        activity.enabled = !activity.enabled;
    }
    if keys.just_pressed(KeyCode::KeyW) {
        wireframe.0 = !wireframe.0;
    }
}

/// If a scene switch was requested this frame, despawn old body
/// entities, build a new `Sim`, replace the resources, and spawn the
/// new body entities. Runs after `handle_input` so a key press in the
/// same frame takes effect immediately.
fn apply_scene_switch(
    mut commands: Commands,
    mut pending: ResMut<PendingSceneSwitch>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    bodies_q: Query<Entity, With<MollaBody>>,
    cloth_q: Query<Entity, With<ClothMesh>>,
    obstacle_q: Query<Entity, With<ClothObstacle>>,
    mut grab: ResMut<MouseGrab>,
    mut scene_states: SceneStates,
) {
    let Some(scene) = pending.0.take() else {
        return;
    };
    // Aliases for ergonomics — the SceneStates SystemParam keeps us
    // under Bevy's 16-system-param tuple ceiling.
    let lidar = &mut *scene_states.lidar;
    let imu = &mut *scene_states.imu;
    let contact = &mut *scene_states.contact;
    let arm = &mut *scene_states.arm;
    let actuator = &mut *scene_states.actuator;
    let gpu_ik = &mut *scene_states.gpu_ik;
    let camera = &mut *scene_states.camera;
    let stats = &mut *scene_states.stats;
    let gpu_np = &mut *scene_states.gpu_np;
    let gpu_bp = &mut *scene_states.gpu_bp;
    let quad = &mut *scene_states.quad;
    // Reference pendulum: clear the per-frame timing on scene switch
    // so the readout doesn't carry stale values from the previous
    // backend. The `notified_unavailable` flag is intentionally
    // preserved so we don't re-spam the log if the user toggles
    // back into the scene.
    scene_states.reference_pendulum.smoothed_step_us = 0.0;
    // DiffArm: reset the optimisation state so the new entry starts
    // from a clean torque sequence (the target survives, since the
    // user's pane sliders persist across scene switches).
    let diff_arm = &mut *scene_states.diff_arm;
    diff_arm.tau = vec![[0.0; 2]; diff_arm.arm.horizon];
    diff_arm.iter_ee_trace.clear();
    diff_arm.last_residual = f64::INFINITY;
    diff_arm.converged_iter = None;
    diff_arm.iters_run = 0;
    diff_arm.playback_t = 0.0;
    // Schedule a re-run on the next frame so the user immediately
    // sees the optimised motion when entering the scene.
    diff_arm.rerun = true;
    // DiffArm6: same reset on entry — fresh tau, clear iter trace,
    // schedule re-run.
    let diff_arm6 = &mut *scene_states.diff_arm6;
    let n6 = diff_arm6.chain.links.len();
    let h6 = diff_arm6.chain.horizon;
    diff_arm6.tau = vec![vec![0.0; n6]; h6];
    diff_arm6.iter_ee_trace.clear();
    diff_arm6.last_residual = f64::INFINITY;
    diff_arm6.converged_iter = None;
    diff_arm6.iters_run = 0;
    diff_arm6.playback_t = 0.0;
    diff_arm6.rerun = true;
    stats.arm_logged = false;
    // Stage 32 — clear the mimic-toggle dirty bit so the rebuild
    // notice in `arm_pane` disappears once the user has acted on it.
    arm.mimic_dirty = false;
    // GPU IK: re-seed the target on next entry from the new arm's rest EE.
    *gpu_ik = GpuIkArmState::default();
    // Stage 18 actuator pipeline: drop the GPU pipeline so the next
    // entry rebuilds against the new sim. `update_actuator_example`
    // also clears it on scene exit; this is belt-and-braces.
    actuator.actuator = None;
    // Stage 18.1 neural-policy demo: drop the MLP pipeline on scene
    // switch so the next entry rebuilds against the new sim's wgpu
    // device. `update_neural_policy_example` also clears it on exit.
    scene_states.neural.actuator = None;
    scene_states.neural.last_inputs = [0.0; 4];
    scene_states.neural.last_output = 0.0;
    scene_states.neural.last_delivered = 0.0;
    scene_states.neural.perturb_timer = 0.0;
    // Stage 31 neural-LSTM demo: same rationale + the LSTM's
    // hidden / cell state must reset on scene switch (otherwise the
    // recurrence carries garbage across episodes). Dropping the
    // `Actuator` discards the controller (and its on-device buffers)
    // entirely, so the next entry starts from a fresh zero state.
    scene_states.neural_lstm.actuator = None;
    scene_states.neural_lstm.last_inputs = [0.0; 4];
    scene_states.neural_lstm.last_output = 0.0;
    scene_states.neural_lstm.last_delivered = 0.0;
    scene_states.neural_lstm.last_hidden = [0.0; LSTM_HID_DIM];
    scene_states.neural_lstm.perturb_timer = 0.0;
    // Quadrotor: reset cached mass / arm-length on every scene switch.
    *quad = QuadrotorState::default();
    // Reset GPU narrow-phase counters on every scene switch — the
    // pipeline that tracked them is rebuilt with the new Sim.
    gpu_np.gpu_enabled_now = false;
    gpu_np.last_cpu_calls = 0;
    gpu_np.last_contacts = 0;
    gpu_np.last_pairs = 0;
    // Same for GPU broad-phase counters / pipeline-state cache.
    // Reset `use_gpu` to match the freshly-cleared `gpu_enabled_now`
    // so re-entering the BroadPhase scene doesn't trigger a one-frame
    // pipeline rebuild from a stale-but-different toggle (the delta
    // check at `step_simulation` rebuilds whenever `use_gpu !=
    // gpu_enabled_now`). The user can flip the toggle back on from
    // the pane to opt into the GPU path.
    gpu_bp.use_gpu = false;
    gpu_bp.gpu_enabled_now = false;
    gpu_bp.last_pairs = 0;
    gpu_bp.smoothed_ms = 0.0;
    gpu_bp.last_cpu_ms = 0.0;
    gpu_bp.last_gpu_ms = 0.0;

    // Stage 20.1 cable scene: reset the orbit time so the puller
    // starts from the same place each entry.
    scene_states.cable.sim_time = 0.0;
    // Stage 20.2 jelly scene: re-stamp the per-tet material params
    // from the slider state on entry. The model was just rebuilt
    // with default values; the slider state may have been moved
    // since last entry.
    let _ = &scene_states.jelly;

    // Stage 24d MPM: clear the particle pour so re-entering the
    // scene drops a fresh column.
    scene_states.mpm.needs_reset = true;
    scene_states.mpm.particle_count = 0;
    // MPM-port scenes: pre-set rheology so the user doesn't have to
    // dig the dropdown for the obvious default.
    if scene == Scene::MpmSnowBall {
        scene_states.mpm.rheology_idx = 1;
    } else if scene == Scene::MpmTwoWay {
        scene_states.mpm.rheology_idx = 0;
    } else if scene == Scene::MpmMultiMaterial {
        // The runtime is still single-rheology, but the scene lays out three
        // material regions and points at the branch-dispatch CPU oracle.
        scene_states.mpm.rheology_idx = 0;
        scene_states.mpm.friction_angle_deg = 30.0;
        scene_states.mpm.pour_axis_count = 8;
    } else if scene == Scene::MpmBeamTwist {
        scene_states.mpm.rheology_idx = 1;
        scene_states.mpm.youngs = 8.0e4;
        scene_states.mpm.pour_axis_count = 12;
    } else if scene == Scene::WarpDem {
        // High-friction sand to settle at angle of repose ~45°.
        scene_states.mpm.rheology_idx = 0;
        scene_states.mpm.friction_angle_deg = 45.0;
    } else if scene == Scene::WarpFluid || scene == Scene::WarpSph {
        // Very low friction — fluid-like flow.
        scene_states.mpm.rheology_idx = 0;
        scene_states.mpm.friction_angle_deg = 5.0;
    }
    // Newton-port scenes that need a per-frame state reset on entry.
    scene_states.cartpole.controller = None;
    scene_states.cartpole.last_x = 0.0;
    scene_states.cartpole.last_theta = 0.0;
    scene_states.cartpole.last_force = 0.0;
    scene_states.cartpole.perturb_timer = 0.0;
    scene_states.cloth_twist.angle = 0.0;
    scene_states.cloth_twist.top_indices.clear();
    scene_states.cloth_twist.bottom_indices.clear();
    scene_states.cloth_twist.bottom_rest.clear();
    scene_states.diff_ball.rerun = true;
    scene_states.diff_ball.iters_run = 0;
    scene_states.diff_ball.history.clear();
    scene_states.diff_ball.trajectory.clear();
    scene_states.diff_ball.playback_t = 0.0;
    scene_states.diff_spring.rerun = true;
    scene_states.diff_spring.iters_run = 0;
    scene_states.fourbar.driver_angle = 0.0;
    let _ = &scene_states.pendulum;
    let _ = &scene_states.conveyor;
    let _ = &scene_states.cloth_bend;
    // Stage 36 — reset the temperature field on entry so re-entering
    // FemHeat starts from a clean slate. The mesh + assembled
    // matrices survive the switch (rebuilding 16³ tets every time
    // would be wasteful) — only the per-vertex `u` resets.
    scene_states.fem_heat.needs_reset = true;
    // Stage 36.3 — reset the P2 FEM heat scene on entry (same as P1).
    scene_states.fem_heat_p2.needs_reset = true;
    // Stage 36.1 — same for the elasticity beam.
    scene_states.fem_beam.needs_reset = true;
    // Stage 36.4 — differentiable elasticity beam.
    scene_states.fem_diff_beam.needs_reset = true;
    // Stage 36.2 — re-solve the lid-driven cavity on entry.
    scene_states.fem_stokes.needs_reset = true;

    // Drop any in-flight grab — its body/particle index belongs to the
    // outgoing model and would dangle against the new one.
    grab.state = None;
    // The Raycaster's pipeline + GPU buffers were tied to the previous
    // model's wgpu device. Drop it so the next `update_lidar` rebuilds
    // it against the new sim.
    lidar.raycaster = None;
    lidar.raycaster_capacity = 0;
    lidar.rays.clear();
    // Same for the IMU sensor — its prev_qd buffer is sized to the old
    // model's body_count.
    imu.sensor = None;
    imu.last = None;
    imu.smoothed_accel_mag = 0.0;
    imu.smoothed_gyro_mag = 0.0;
    // Contact sensor is sized to body_count too — drop on switch.
    contact.sensor = None;
    contact.readings.clear();
    // Arm PD controller is bound to a specific joint_dof_total —
    // drop and rebuild on the next arm-scene step.
    arm.controller = None;
    arm.elapsed = 0.0;
    // Camera sensor's TiledCamera owns wgpu pipelines + buffers tied
    // to the previous Sim's device — drop it; the next
    // `update_camera_example` will rebuild against the new sim.
    camera.sensor = None;
    camera.elapsed = 0.0;
    camera.last_pose = None;
    // Drop the per-channel Bevy Image handles + cached egui texture
    // ids — the next `update_camera_example` will reallocate against
    // the new sim. `bevy_egui::free_egui_textures_system` cleans up the
    // egui-side mapping when the strong handle drops.
    camera.image_color = None;
    camera.image_depth = None;
    camera.image_normal = None;
    camera.image_albedo = None;
    camera.image_index = None;
    camera.tex_id_color = None;
    camera.tex_id_depth = None;
    camera.tex_id_normal = None;
    camera.tex_id_albedo = None;
    camera.tex_id_index = None;
    camera.last_depth_range = None;

    for e in bodies_q.iter() {
        commands.entity(e).despawn();
    }
    for e in cloth_q.iter() {
        commands.entity(e).despawn();
    }
    for e in obstacle_q.iter() {
        commands.entity(e).despawn();
    }

    let (sim, visuals) = Sim::build_scene(scene);
    let visuals = BodyVisuals(visuals);
    spawn_body_entities(&mut commands, &mut meshes, &mut materials, &sim, &visuals);

    // Stage-21 mesh-tools demo: build the visualisation entities
    // (original / decimated / MC / hull) once on scene entry. They're
    // tagged with `ClothObstacle` so the next scene switch despawns
    // them via the obstacle_q above.
    if scene == Scene::MeshTools {
        let mesh_state = &mut *scene_states.mesh;
        spawn_mesh_tools_entities(&mut commands, &mut meshes, &mut materials, mesh_state);
    }

    commands.insert_resource(sim);
    commands.insert_resource(visuals);

    info!("scene switched → {}", scene.label());
}

fn step_simulation(
    mut sim: ResMut<Sim>,
    mut run: ResMut<SimRunState>,
    mut stats: ResMut<SimStats>,
    mut gpu_toggles: (ResMut<GpuNpExampleState>, ResMut<GpuBpExampleState>),
    mut reference_pendulum_state: ResMut<ReferencePendulumState>,
) {
    let (gpu_np, gpu_bp) = (&mut gpu_toggles.0, &mut gpu_toggles.1);
    if run.paused && !run.step_once {
        return;
    }
    run.step_once = false;
    // GPU IK arm scene is purely kinematic; `update_gpu_ik_arm`
    // already wrote `joint_q` + `body_q` on the host. Skipping the
    // physics step here keeps the IK pose stable across frames.
    if sim.scene == Scene::GpuIkArm {
        return;
    }
    // DiffArm replays the optimised trajectory in `update_diff_arm`
    // (host-side; writes body_q directly). Same skip as GpuIkArm.
    if sim.scene == Scene::DiffArm {
        return;
    }
    // Stage-23.1 DiffArm6 — same host-driven playback model.
    if sim.scene == Scene::DiffArm6 {
        return;
    }
    // Stage-21 mesh-tools is purely visual — no bodies, no contacts,
    // no physics step. Bail before allocating the device handles.
    if sim.scene == Scene::MeshTools {
        return;
    }
    // Stage-36 FEM heat-equation demo runs its own CPU FEM solver in
    // `update_fem_heat_example` (P1) or `update_fem_heat_p2_example`
    // (Stage 36.3 P2); bypass the standard pipeline (no bodies, no
    // contacts).
    if sim.scene == Scene::FemHeat || sim.scene == Scene::FemHeatP2 {
        return;
    }
    // Stage-36.1 FEM elasticity beam — same rationale: pure-CPU FEM
    // solver in `update_fem_beam_example`; bypass the standard
    // pipeline (no bodies, no contacts).
    if sim.scene == Scene::FemBeam {
        return;
    }
    // Stage-36.4 differentiable FEM beam — runs gradient descent on E
    // in `update_fem_diff_beam_example`; same bypass as `FemBeam`.
    if sim.scene == Scene::FemDiffBeam {
        return;
    }
    // Stage-36.2 FEM Stokes lid-driven cavity — pure-CPU Uzawa-CG.
    if sim.scene == Scene::FemStokes {
        return;
    }
    // Stage-24d MPM scene runs its own particle/grid sim outside the
    // standard solver pipeline; `update_mpm_example` drives it.
    if sim.scene == Scene::Mpm
        || sim.scene == Scene::MpmSnowBall
        || sim.scene == Scene::MpmTwoWay
        || sim.scene == Scene::WarpDem
        || sim.scene == Scene::WarpFluid
        || sim.scene == Scene::WarpSph
        || sim.scene == Scene::MpmBeamTwist
    {
        return;
    }
    // Warp-port host-driven scenes — every Warp scene except the
    // MPM-backed ones and the few that reuse the standard collision
    // pipeline (raycast / mesh / mesh-intersect) skips physics.
    if matches!(
        sim.scene,
        Scene::WarpRaymarch
            | Scene::WarpSampleMesh
            | Scene::WarpMarchingCubes
            | Scene::WarpNvdb
            | Scene::WarpTileFft
            | Scene::WarpTileMlp
            | Scene::WarpTileNbody
            | Scene::WarpTileCholesky
            | Scene::WarpParticleRepulsion
            | Scene::WarpFemDiffusion
            | Scene::WarpFemDiffusion3D
            | Scene::WarpFemConvDiff
            | Scene::WarpFemConvDiffDg
            | Scene::WarpFemTaylorGreen
            | Scene::WarpFemKelvinHelmholtz
            | Scene::WarpFemStreamlines
            | Scene::WarpFemDeformedGeometry
            | Scene::WarpFemDistortion
            | Scene::WarpFemElasticOpt
            | Scene::WarpFemMixedElasticity
            | Scene::WarpFemNonconformingContact
            | Scene::WarpFftFluid
            | Scene::WarpDistortion3D
            | Scene::WarpApicFluid
            | Scene::WarpWave
            | Scene::WarpDiffRay
            | Scene::WarpTileFilter
            | Scene::WarpTileCompaction
            | Scene::AdamLandscape
            | Scene::WarpTileMatmul
            | Scene::WarpTileConvolution
            | Scene::WarpTileMcgp
            | Scene::IkCustom
            | Scene::IkCubeStacking
            | Scene::DiffsimDrone
            | Scene::FluidCheckpointOptimization
            | Scene::NavierStokesPerturbationOptimization
            | Scene::WarpFftPoisson2d
            | Scene::WarpRaycast
            | Scene::WarpMesh
            | Scene::WarpMeshIntersect
            | Scene::ImplicitMpmBingham
            | Scene::WarpStableFluid
            | Scene::DgBurgersShock
            | Scene::ShallowWaterDamBreak
            | Scene::ImplicitMpmFull27
            | Scene::SplatPoseRecovery
            | Scene::AmrHeat
            | Scene::Magnetostatics
            | Scene::PhaseFieldFracture
            | Scene::Vehicle
            | Scene::SphDamBreak
            | Scene::LidarV2
            | Scene::AerialLidarPass
            | Scene::SolarSky
            | Scene::SyntheticAnnotation
            | Scene::TireModelLab
            | Scene::VehicleOffroad
            | Scene::BekkerSinkage
            | Scene::SoilCompaction
            | Scene::TireDynamics
            | Scene::TireTraction
    ) {
        return;
    }
    // DiffBall + DiffSpring are host-driven (analytical projectile +
    // gradient-descent on rest lengths); the update systems write
    // body_q on the host directly.
    if sim.scene == Scene::DiffBall || sim.scene == Scene::DiffSpring {
        return;
    }
    let frame_start = std::time::Instant::now();

    let device = sim.engine.device().unwrap().clone();
    let queue = sim.engine.queue().unwrap().clone();

    // Apply Stage-15 GPU narrow-phase toggle for the demo scene.
    // Outside that scene, leave the pipeline configuration alone.
    if sim.scene == Scene::GpuNarrowPhase && gpu_np.use_gpu != gpu_np.gpu_enabled_now {
        // Need to call methods on `sim.pipeline` while borrowing
        // `sim.engine`. Borrow them as separate mutable / immutable
        // refs through pattern destructuring to keep the compiler
        // happy.
        let Sim {
            pipeline, engine, ..
        } = &mut *sim;
        *pipeline = molla::geometry::CollisionPipeline::new();
        if gpu_np.use_gpu {
            pipeline
                .enable_full_gpu_narrow_phase(engine, 256)
                .expect("enable_full_gpu_narrow_phase");
        }
        gpu_np.gpu_enabled_now = gpu_np.use_gpu;
    }
    // Apply Stage-19 GPU broad-phase toggle.
    if sim.scene == Scene::BroadPhase && gpu_bp.use_gpu != gpu_bp.gpu_enabled_now {
        let Sim {
            pipeline, engine, ..
        } = &mut *sim;
        *pipeline = molla::geometry::CollisionPipeline::new();
        if gpu_bp.use_gpu {
            pipeline
                .enable_full_gpu_broad_phase(engine, 8192, 65_536, false)
                .expect("enable_full_gpu_broad_phase");
        }
        gpu_bp.gpu_enabled_now = gpu_bp.use_gpu;
    }

    let Sim {
        model,
        state_a,
        state_b,
        control,
        contacts,
        solver,
        pipeline,
        ..
    } = &mut *sim;

    let mut total_collide_us = 0.0_f64;
    let mut total_download_us = 0.0_f64;
    let solve_start = std::time::Instant::now();
    // Featherstone keeps state on the GPU across substeps and writes
    // body_q from forward kinematics — re-uploading from host would
    // clobber that with stale (post-swap, pre-step) data. Only upload
    // body buffers for XPBD scenes where the mouse-grab system writes
    // body_qd on the host.
    let solver_is_xpbd = matches!(solver, SolverBackend::Xpbd(_));
    for _ in 0..SUBSTEPS_PER_FRAME {
        let c0 = std::time::Instant::now();
        pipeline.collide(model, state_a, contacts).expect("collide");
        contacts
            .upload_to_device(&device, &queue)
            .expect("contacts upload");
        total_collide_us += c0.elapsed().as_secs_f64() * 1_000_000.0;

        if solver_is_xpbd {
            let _ = state_a.body_q.upload(&queue);
            let _ = state_a.body_qd.upload(&queue);
        }

        solver
            .step(model, state_a, control, contacts, FIXED_DT, state_b)
            .expect("solver step");
        std::mem::swap(state_a, state_b);

        let d0 = std::time::Instant::now();
        let _ = state_a.body_q.download(&device, &queue);
        let _ = state_a.body_qd.download(&device, &queue);
        if model.joint_dof_total > 0 {
            let _ = state_a.joint_q.download(&device, &queue);
            let _ = state_a.joint_qd.download(&device, &queue);
        }
        // Cloth scenes (XPBD or VBD) write `particle_q` on the device
        // and the renderer reads `particle_q.host()` in
        // `sync_cloth_mesh`. Download per-substep so the visualisation
        // tracks every solver iteration.
        if model.particle_count > 0 {
            let _ = state_a.particle_q.download(&device, &queue);
            let _ = state_a.particle_qd.download(&device, &queue);
        }
        total_download_us += d0.elapsed().as_secs_f64() * 1_000_000.0;
    }
    let solve_us = solve_start.elapsed().as_secs_f64() * 1_000_000.0;
    let total_us = frame_start.elapsed().as_secs_f64() * 1_000_000.0;

    stats.smoothed_step_us = 0.9 * stats.smoothed_step_us + 0.1 * total_us;
    stats.smoothed_solve_us = 0.9 * stats.smoothed_solve_us + 0.1 * solve_us;
    stats.smoothed_collide_us = 0.9 * stats.smoothed_collide_us + 0.1 * total_collide_us;
    stats.smoothed_download_us = 0.9 * stats.smoothed_download_us + 0.1 * total_download_us;
    stats.last_contacts = contacts.count;

    // Surface reference-pendulum timing for the right-rail pane. The solver
    // dispatch slot in the main stats block also captures it, but
    // the scene pane wants its own EMA for a stable readout.
    if sim.scene == Scene::ReferencePendulum {
        reference_pendulum_state.smoothed_step_us =
            0.9 * reference_pendulum_state.smoothed_step_us + 0.1 * solve_us;
    }

    // Surface Stage-15 GPU narrow-phase counters for the right-rail
    // pane (only meaningful for the GpuNarrowPhase scene; harmless
    // elsewhere — `last_cpu_narrow_phase_calls()` returns the most
    // recent invocation's count regardless of pipeline mode).
    gpu_np.last_cpu_calls = sim.pipeline.last_cpu_narrow_phase_calls();
    gpu_np.last_contacts = sim.contacts.count;
    // Approx. SAP candidate count = CPU calls + contacts (close
    // enough — every successful pair becomes either a contact or a
    // CPU pair_contact call). The real count isn't exposed; this is
    // a UI estimate.
    gpu_np.last_pairs = gpu_np.last_cpu_calls + sim.contacts.count as usize;

    // Surface Stage-19 broad-phase counters. `last_broad_phase_pair_count`
    // is the post-filter pair list size — same metric for CPU SAP and
    // GPU LBVH. The smoothed ms/frame mirrors `smoothed_collide_us` but
    // re-baselines on toggle so the user sees "current backend" timing.
    if sim.scene == Scene::BroadPhase {
        gpu_bp.last_pairs = sim.pipeline.last_broad_phase_pair_count();
        let collide_ms = stats.smoothed_collide_us / 1000.0;
        gpu_bp.smoothed_ms = 0.9 * gpu_bp.smoothed_ms + 0.1 * collide_ms;
        if gpu_bp.gpu_enabled_now {
            gpu_bp.last_gpu_ms = gpu_bp.smoothed_ms;
        } else {
            gpu_bp.last_cpu_ms = gpu_bp.smoothed_ms;
        }
    }

    // One-shot debug dump after the first frame's substeps for the
    // Arm scene — surfaces whether Featherstone's FK pass is actually
    // touching `body_q` on the device. If body_q matches the initial
    // build poses (4.7, 4.1, 3.5, ...), FK never landed; if it shows
    // FK output (5.3, 5.9, 6.5, ...), the kernel is running and the
    // visual issue is downstream (Bevy transforms, mesh orientation).
    if sim.scene == Scene::Arm && !stats.arm_logged {
        if let Ok(bq) = sim.state_a.body_q.host() {
            let ys: Vec<f32> = bq.iter().map(|p| p.position.y as f32).collect();
            if let Ok(q) = sim.state_a.joint_q.host() {
                let qs: Vec<f32> = q.iter().map(|v| *v as f32).collect();
                info!("arm: post-frame body.y = {ys:?}  joint_q = {qs:?}");
            } else {
                info!("arm: post-frame body.y = {ys:?}");
            }
        }
        stats.arm_logged = true;
    }
}

fn sync_transforms(sim: Res<Sim>, mut q: Query<(&MollaBody, &mut Transform)>) {
    let body_q = sim.state_a.body_q.host().expect("body_q");
    for (body, mut tf) in q.iter_mut() {
        let i = body.0.index();
        if i >= body_q.len() {
            continue;
        }
        let pose = body_q[i];
        tf.translation = Vec3::new(
            pose.position.x as f32,
            pose.position.y as f32,
            pose.position.z as f32,
        );
        let r: MQuat = pose.rotation;
        tf.rotation = Quat::from_xyzw(r.x as f32, r.y as f32, r.z as f32, r.w as f32);
    }
}

/// Update the cloth `Mesh` asset's vertex positions from the latest
/// `state.particle_q`. Recomputes flat normals so lighting tracks the
/// deforming surface. No-op when there's no cloth in the scene.
fn sync_cloth_mesh(
    sim: Res<Sim>,
    mut meshes: ResMut<Assets<Mesh>>,
    q: Query<&Mesh3d, With<ClothMesh>>,
) {
    if sim.model.particle_count == 0 {
        return;
    }
    let particle_q = match sim.state_a.particle_q.host() {
        Ok(p) => p,
        Err(_) => return,
    };

    for mesh3d in q.iter() {
        let Some(mesh) = meshes.get_mut(&mesh3d.0) else {
            continue;
        };
        let positions: Vec<[f32; 3]> = particle_q
            .iter()
            .map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.compute_normals();
    }
}

// ---- Wireframe overlay --------------------------------------------

/// Each frame, ensure every body / cloth entity carries a `Wireframe`
/// component iff `WireframeOn` is set. Per-entity tagging instead of
/// `WireframeConfig.global` so the ground grid (owned by bevy_glacial)
/// is unaffected. Cheap on bevy 0.18 — `Has<T>` skips component
/// archetype edits when nothing needs to change.
fn apply_wireframe(
    wf: Res<WireframeOn>,
    mut commands: Commands,
    bodies: Query<
        (Entity, bevy::ecs::query::Has<Wireframe>),
        bevy::ecs::query::Or<(With<MollaBody>, With<ClothMesh>)>,
    >,
) {
    let want = wf.0;
    for (e, has) in bodies.iter() {
        match (want, has) {
            (true, false) => {
                commands.entity(e).insert(Wireframe);
            }
            (false, true) => {
                commands.entity(e).remove::<Wireframe>();
            }
            _ => {}
        }
    }
}

// ---- Activity view (sleeping vs moving) ---------------------------

/// Each frame, classify every body as "active" (kinematic norm above
/// threshold) or "sleeping". When the view is enabled, sleeping bodies
/// get the shared grey material; active bodies get their original
/// palette material back. When disabled, every body always wears its
/// palette material — this still costs one material-swap per body the
/// frame the view is toggled off, after which the system is a cheap
/// O(n) read of `body_qd`.
fn update_activity_colors(
    mut commands: Commands,
    mut activity: ResMut<ActivityView>,
    sim: Res<Sim>,
    q: Query<(
        Entity,
        &MollaBody,
        &AwakeMaterial,
        &MeshMaterial3d<StandardMaterial>,
    )>,
) {
    let Ok(qd) = sim.state_a.body_qd.host() else {
        return;
    };
    let Some(sleeping) = activity.sleeping_mat.clone() else {
        return;
    };
    let enabled = activity.enabled;

    let mut active_count = 0usize;
    for (entity, body, awake, current) in q.iter() {
        let i = body.0.index();
        if i >= qd.len() {
            continue;
        }
        let v = qd[i].linear.length();
        let w = qd[i].angular.length();
        let kin = v + w * ACTIVITY_OMEGA_RADIUS;
        let is_active = kin > ACTIVITY_THRESHOLD;
        if is_active {
            active_count += 1;
        }
        let want = if !enabled || is_active {
            awake.0.clone()
        } else {
            sleeping.clone()
        };
        // Re-inserting `MeshMaterial3d` via Commands is the supported
        // way to swap a body's material in Bevy 0.18 — directly
        // mutating the component through `Query<&mut>` triggers a
        // material-bookkeeping panic in some paths.
        if current.0.id() != want.id() {
            commands.entity(entity).insert(MeshMaterial3d(want));
        }
    }
    activity.active_count = active_count;
}

// ---- Mouse grab + ray-cast picking --------------------------------

#[derive(Default, Resource)]
struct MouseGrab {
    state: Option<GrabState>,
}

#[derive(Copy, Clone, Debug)]
enum GrabKind {
    /// Grab a rigid body at a body-local anchor point. We solve
    /// `v_anchor = v_lin + ω × r` for `v_lin` so the anchor moves at
    /// the desired velocity while preserving the body's spin.
    Body {
        body: mp::BodyId,
        local_anchor: MVec3,
    },
    /// Grab a single cloth particle. While held, the particle's
    /// `particle_inv_mass` is forced to 0 so the XPBD constraint
    /// solver treats it as kinematic — the cloth has to deform around
    /// our position instead of dragging us back. We save the original
    /// inv_mass so we can restore it on release.
    Particle { index: u32, saved_inv_mass: f64 },
}

#[derive(Copy, Clone, Debug)]
struct GrabState {
    kind: GrabKind,
    /// Distance along the ray at the moment of pick. Reused as the
    /// drag depth so the picked point stays at the same camera-distance
    /// while dragging — otherwise mouse motion would change pick depth.
    drag_distance: f32,
    /// Cached for gizmo rendering.
    last_target: MVec3,
}

/// Run before the physics step. Picks a body on LMB-press, applies a
/// spring-pull external force on the picked body while LMB is held,
/// clears state on release. Yields to the camera-orbit gesture (LMB +
/// RMB) so dragging doesn't accidentally grab a body.
fn update_mouse_grab(
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    primary_window: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    mut sim: ResMut<Sim>,
    visuals: Res<BodyVisuals>,
    mut grab: ResMut<MouseGrab>,
) {
    let Ok(window) = primary_window.single() else {
        return;
    };
    let Ok((cam, cam_tr)) = cameras.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok(ray) = cam.viewport_to_world(cam_tr, cursor) else {
        return;
    };

    let lmb_just = mouse.just_pressed(MouseButton::Left);
    let lmb_held = mouse.pressed(MouseButton::Left);
    let rmb_held = mouse.pressed(MouseButton::Right);
    let mmb_held = mouse.pressed(MouseButton::Middle);
    let alt_held = keys.pressed(KeyCode::AltLeft) || keys.pressed(KeyCode::AltRight);

    // Camera gestures or alt = let the camera have it. RMB starts an
    // orbit; MMB pans / lifts. Anything held with another button is
    // not a grab gesture.
    if rmb_held || mmb_held || alt_held {
        release_grab(&mut sim, &mut grab);
        return;
    }

    if !lmb_held {
        release_grab(&mut sim, &mut grab);
        return;
    }

    let ro = MVec3::new(
        ray.origin.x as f64,
        ray.origin.y as f64,
        ray.origin.z as f64,
    );
    let rd = MVec3::new(
        ray.direction.x as f64,
        ray.direction.y as f64,
        ray.direction.z as f64,
    );

    // Acquire a grab on the press edge — try bodies and particles,
    // pick whichever has the closer ray-hit.
    if lmb_just && grab.state.is_none() {
        let body_hit = pick_body(&sim, &visuals, ro, rd);
        let particle_hit = pick_particle(&sim, ro, rd);
        let best = match (body_hit, particle_hit) {
            (Some((bid, bt, banchor)), Some((pid, pt))) => {
                if bt <= pt {
                    Some((
                        GrabKind::Body {
                            body: bid,
                            local_anchor: banchor,
                        },
                        bt,
                    ))
                } else {
                    Some((begin_particle_grab(&mut sim, pid), pt))
                }
            }
            (Some((bid, bt, banchor)), None) => Some((
                GrabKind::Body {
                    body: bid,
                    local_anchor: banchor,
                },
                bt,
            )),
            (None, Some((pid, pt))) => Some((begin_particle_grab(&mut sim, pid), pt)),
            (None, None) => None,
        };
        if let Some((kind, t)) = best {
            grab.state = Some(GrabState {
                kind,
                drag_distance: t as f32,
                last_target: ro + rd * t,
            });
        }
    }

    // While grabbing, recompute target and drive the picked
    // body/particle toward it.
    if let Some(state) = grab.state {
        let target = ro + rd * state.drag_distance as f64;
        grab.state = Some(GrabState {
            last_target: target,
            ..state
        });
        match state.kind {
            GrabKind::Body { body, local_anchor } => {
                apply_grab_velocity(&mut sim, body, local_anchor, target);
            }
            GrabKind::Particle { index, .. } => {
                apply_particle_grab_pin(&mut sim, index, target);
            }
        }
    }
}

/// Mark a cloth particle as kinematic for the duration of a grab and
/// stash its original inv_mass into the returned `GrabKind`. Caller
/// must pair this with `release_grab` to restore the inv_mass.
fn begin_particle_grab(sim: &mut Sim, index: u32) -> GrabKind {
    let i = index as usize;
    let inv = sim
        .model
        .particle_inv_mass
        .host()
        .expect("particle_inv_mass")[i];
    sim.model
        .particle_inv_mass
        .host_mut()
        .expect("particle_inv_mass_mut")[i] = 0.0;
    GrabKind::Particle {
        index,
        saved_inv_mass: inv,
    }
}

/// End the current grab. For `Particle` grabs we restore the original
/// inverse mass so the cloth resumes simulating the released vertex.
fn release_grab(sim: &mut Sim, grab: &mut MouseGrab) {
    if let Some(state) = grab.state.take() {
        if let GrabKind::Particle {
            index,
            saved_inv_mass,
        } = state.kind
        {
            let i = index as usize;
            if let Ok(slice) = sim.model.particle_inv_mass.host_mut() {
                if i < slice.len() {
                    slice[i] = saved_inv_mass;
                }
            }
        }
    }
}

/// Walk every body's collider, ray-cast against its body-local primitive,
/// return the closest positive hit as `(body, t, local_anchor)`.
fn pick_body(
    sim: &Sim,
    visuals: &BodyVisuals,
    ro: MVec3,
    rd: MVec3,
) -> Option<(mp::BodyId, f64, MVec3)> {
    let q = sim.state_a.body_q.host().ok()?;
    let flags = sim.model.body_flags.host().ok()?;
    let mut best: Option<(f64, mp::BodyId, MVec3)> = None;
    for (i, vis) in visuals.0.iter().enumerate() {
        // Don't pick kinematic anchors — they're not meant to move.
        if (flags[i] & mp::BodyFlags::KINEMATIC) != 0 {
            continue;
        }
        let pose = q[i];
        let lo = pose.inverse_transform_point(ro);
        let ld = pose.rotation.conjugate() * rd;
        let t = match *vis {
            BodyVisual::Sphere(r) => ray_sphere_local(lo, ld, r as f64),
            BodyVisual::Box(h) => {
                ray_box_local(lo, ld, MVec3::new(h.x as f64, h.y as f64, h.z as f64))
            }
            // Capsule pick uses a tight axis-aligned-box approximation
            // in body frame — accurate enough for grabbing.
            BodyVisual::Capsule { half_h, radius } => ray_box_local(
                lo,
                ld,
                MVec3::new(radius as f64, (half_h + radius) as f64, radius as f64),
            ),
            // Cone pick: tight AABB approximation in body frame.
            BodyVisual::Cone { half_h, radius } => ray_box_local(
                lo,
                ld,
                MVec3::new(radius as f64, half_h as f64, radius as f64),
            ),
            // Ellipsoid pick: bounding-sphere of `max(half_axes)` —
            // ergonomic for grabbing without solving the quadratic.
            BodyVisual::Ellipsoid { half_axes } => {
                let r = half_axes.max_element() as f64;
                ray_sphere_local(lo, ld, r)
            }
            // GaussianSplat pick: stub — fixed picking radius so the
            // user can still drag a splat as a debug handle.
            BodyVisual::GaussianSplat { .. } => ray_sphere_local(lo, ld, 0.25),
        };
        if let Some(t) = t {
            if t > 0.0 && best.is_none_or(|(bt, _, _)| t < bt) {
                let local_hit = lo + ld * t;
                best = Some((t, mp::BodyId::from_index(i), local_hit));
            }
        }
    }
    best.map(|(t, id, l)| (id, t, l))
}

/// Ray-cast against every cloth particle treated as a small sphere.
/// Returns `(particle_index, t)` of the nearest hit. Picking radius is
/// inflated above `particle_radius` so the user doesn't have to land
/// the cursor on a 2 cm dot — clicking near any vertex on the cloth
/// reliably grabs it.
fn pick_particle(sim: &Sim, ro: MVec3, rd: MVec3) -> Option<(u32, f64)> {
    if sim.model.particle_count == 0 {
        return None;
    }
    let q = sim.state_a.particle_q.host().ok()?;
    let inv_mass = sim.model.particle_inv_mass.host().ok()?;
    // Pick radius matches the apparent visual cell size — close to
    // half the rest spacing so any click between two vertices snaps
    // to one of them. Without inflation, only direct hits on a 2 cm
    // dot would register.
    let pick_r: f64 = 0.05;
    let mut best: Option<(f64, u32)> = None;
    for i in 0..sim.model.particle_count {
        // Don't grab pinned anchors — they can't move anyway.
        if inv_mass[i] <= 0.0 {
            continue;
        }
        let centre = q[i];
        let lo = ro - centre;
        if let Some(t) = ray_sphere_local(lo, rd, pick_r) {
            if t > 0.0 && best.is_none_or(|(bt, _)| t < bt) {
                best = Some((t, i as u32));
            }
        }
    }
    best.map(|(t, i)| (i, t))
}

/// Move a grabbed cloth particle directly to `target` (clamped per
/// frame so a fast cursor doesn't teleport it across the world).
///
/// Only valid while the particle's inv_mass is 0 — see
/// `begin_particle_grab`. With inv_mass = 0, the integrator skips the
/// particle (predict pass `if inv_mass <= 0 continue`) and the
/// constraint solver sends 100% of every distance correction to the
/// other side. Result: the cloth mesh follows the cursor, the rest of
/// the cloth gets dragged along by the constraint chain, and the user
/// can pull as far as they want — the only resistance comes from
/// stretching the surrounding fabric.
fn apply_particle_grab_pin(sim: &mut Sim, index: u32, target: MVec3) {
    let i = index as usize;
    if i >= sim.model.particle_count {
        return;
    }
    let prev = sim.state_a.particle_q.host().expect("particle_q")[i];
    let to_target = target - prev;
    let dist = to_target.length();
    // Per-frame step cap — at 60 fps × 14 m/s = 0.23 m/frame, plenty
    // for natural drag, low enough not to tunnel a particle through
    // collisions.
    let max_step = GRAB_MAX_SPEED / 60.0;
    let new_pos = if dist > max_step && dist > 1e-6 {
        prev + to_target * (max_step / dist)
    } else {
        target
    };
    sim.state_a.particle_q.host_mut().expect("particle_q_mut")[i] = new_pos;
    // Zero velocity — the constraint solver doesn't touch this
    // particle (inv_mass = 0), and we don't want stale velocity to
    // confuse the velocity-recovery pass.
    sim.state_a.particle_qd.host_mut().expect("particle_qd_mut")[i] = MVec3::ZERO;
}

/// Ray vs unit sphere centred at origin (radius `r`).
fn ray_sphere_local(o: MVec3, d: MVec3, r: f64) -> Option<f64> {
    let b = o.dot(d);
    let c = o.dot(o) - r * r;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let sd = disc.sqrt();
    let t1 = -b - sd;
    if t1 > 0.0 {
        return Some(t1);
    }
    let t2 = -b + sd;
    if t2 > 0.0 { Some(t2) } else { None }
}

/// Ray vs box (centred at origin, half-extents `h`). The `h: Into<MVec3>`
/// trick lets callers pass either a uniform `f64` (cube) or a per-axis
/// vector (capsule bounding box).
trait IntoHalfExtents {
    fn to_half_extents(self) -> MVec3;
}
impl IntoHalfExtents for f64 {
    fn to_half_extents(self) -> MVec3 {
        MVec3::splat(self)
    }
}
impl IntoHalfExtents for MVec3 {
    fn to_half_extents(self) -> MVec3 {
        self
    }
}
fn ray_box_local<H: IntoHalfExtents>(o: MVec3, d: MVec3, h: H) -> Option<f64> {
    let h = h.to_half_extents();
    let mut tmin = f64::NEG_INFINITY;
    let mut tmax = f64::INFINITY;
    for k in 0..3 {
        let oi = match k {
            0 => o.x,
            1 => o.y,
            _ => o.z,
        };
        let di = match k {
            0 => d.x,
            1 => d.y,
            _ => d.z,
        };
        let hi = match k {
            0 => h.x,
            1 => h.y,
            _ => h.z,
        };
        if di.abs() < 1e-12 {
            if oi.abs() > hi {
                return None;
            }
        } else {
            let t1 = (-hi - oi) / di;
            let t2 = (hi - oi) / di;
            let (lo, hi) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
            tmin = tmin.max(lo);
            tmax = tmax.min(hi);
            if tmax < tmin {
                return None;
            }
        }
    }
    if tmax < 0.0 {
        return None;
    }
    Some(if tmin > 0.0 { tmin } else { tmax })
}

/// Set the grabbed body's linear velocity so its anchor point moves
/// toward `target` at a bounded speed. Mass-independent: same speed
/// for a 50 kg wrecking ball or a 0.4 kg rope link.
///
/// We solve `v_anchor = v_lin + ω × r` for `v_lin` so that the anchor
/// has the desired velocity, while keeping the body's existing angular
/// velocity. The XPBD solver then handles all downstream behaviour:
/// CONNECT constraints pull neighbouring bodies along, contacts brake
/// against other bodies, gravity continues to act. Critically there's
/// no force overshoot — `v_anchor_want` is bounded by `MAX_SPEED`, so
/// neighbours can never see an impulse larger than that.
fn apply_grab_velocity(sim: &mut Sim, body: mp::BodyId, local_anchor: MVec3, target: MVec3) {
    let i = body.index();

    let pose = sim.state_a.body_q.host().expect("body_q")[i];
    let qd = sim.state_a.body_qd.host().expect("body_qd")[i];
    let body_com = sim.model.body_com.host().expect("body_com")[i];

    let anchor_world = pose.transform_point(local_anchor);
    let com_world = pose.position + pose.rotation * body_com;
    let r = anchor_world - com_world;

    let to_target = target - anchor_world;
    let dist = to_target.length();
    let v_anchor_want = if dist > 1e-6 {
        let dir = to_target / dist;
        let speed = (dist / GRAB_LAG_TIME).min(GRAB_MAX_SPEED);
        dir * speed
    } else {
        MVec3::ZERO
    };

    // Keep current angular velocity. Solve linear so the anchor moves
    // at `v_anchor_want`: v_anchor = v_lin + ω × r ⇒ v_lin = v_anchor − ω × r.
    let v_lin = v_anchor_want - qd.angular.cross(r);

    // Soft angular damping so the body doesn't keep spinning forever
    // while held. Keeps natural rotations from contacts but bleeds off
    // accumulated spin if the user holds it still.
    let w = qd.angular * 0.85;

    sim.state_a.body_qd.host_mut().expect("body_qd_mut")[i] = mp::SpatialVector::new(v_lin, w);
}

/// Lidar example — runs the GPU [`Raycaster`] each frame against the
/// current scene and stashes the (origin, dir, hit) triplets in
/// [`LidarState`] for `draw_lidar_gizmos` to render. No-op for any
/// scene other than [`Scene::Lidar`].
///
/// Ray construction: for each `(iy, ix)` cell in a `ray_count_v ×
/// ray_count_h` grid covering `fov_v × fov_h`, build a unit direction
/// from yaw + pitch + per-cell offsets, applied to the lidar origin.
/// readout. No-op for any other scene.
/// the IK pose remains stable across frames.
/// No-op for any other scene.
/// [`molla::autograd::featherstone_adjoint`].
/// any other scene.
/// any other scene.
/// sync with the right-rail sliders.
/// next solver step).

/// driven by native Featherstone.
/// scene.
#[allow(clippy::too_many_arguments)]
/// the contact scene isn't active.
///     direction of the lower arm).
/// `draw_diff_arm_gizmos` — target marker + EE trajectory polyline.
/// yellow sphere so you can see where the sensor is.
/// particles when `state.show_pinned` is on.
///      drawn as a yellow polyline above the cube for A/B feel.
/// they're holding.
fn draw_grab_gizmo(grab: Res<MouseGrab>, sim: Res<Sim>, mut gizmos: Gizmos) {
    let Some(state) = grab.state else { return };
    let anchor_world = match state.kind {
        GrabKind::Body { body, local_anchor } => {
            let Ok(q) = sim.state_a.body_q.host() else {
                return;
            };
            let i = body.index();
            if i >= q.len() {
                return;
            }
            q[i].transform_point(local_anchor)
        }
        GrabKind::Particle { index, .. } => {
            let Ok(q) = sim.state_a.particle_q.host() else {
                return;
            };
            let i = index as usize;
            if i >= q.len() {
                return;
            }
            q[i]
        }
    };
    let target = state.last_target;
    let a = Vec3::new(
        anchor_world.x as f32,
        anchor_world.y as f32,
        anchor_world.z as f32,
    );
    let t = Vec3::new(target.x as f32, target.y as f32, target.z as f32);
    gizmos.line(a, t, Color::srgb(1.0, 0.85, 0.2));
    // Small + at the anchor and target so they're easier to see.
    let cross = 0.08;
    gizmos.line(
        a - Vec3::X * cross,
        a + Vec3::X * cross,
        Color::srgb(1.0, 0.85, 0.2),
    );
    gizmos.line(
        a - Vec3::Y * cross,
        a + Vec3::Y * cross,
        Color::srgb(1.0, 0.85, 0.2),
    );
    gizmos.line(
        t - Vec3::X * cross,
        t + Vec3::X * cross,
        Color::srgb(0.5, 0.95, 0.5),
    );
    gizmos.line(
        t - Vec3::Y * cross,
        t + Vec3::Y * cross,
        Color::srgb(0.5, 0.95, 0.5),
    );
}

fn control_pane(
    mut contexts: EguiContexts,
    accent: Res<AccentColor>,
    diagnostics: Res<DiagnosticsStore>,
    stats: Res<SimStats>,
    mut sim: ResMut<Sim>,
    mut run: ResMut<SimRunState>,
    mut pending: ResMut<PendingSceneSwitch>,
    mut activity: ResMut<ActivityView>,
    mut lidar_config: ResMut<LidarConfig>,
    sensor_states: (
        Res<LidarState>,
        Res<ImuExampleState>,
        Res<ContactExampleState>,
    ),
    mut description_pane: ResMut<SceneDescriptionPaneState>,
    mut pane_states: PaneSceneStates,
    mut ribbons: RibbonParams,
    mut warp_states: ResMut<WarpExampleStates>,
) {
    let (lidar_state, imu_state, contact_state) =
        (&*sensor_states.0, &*sensor_states.1, &*sensor_states.2);
    let mut ribbon_open = ribbons.open.reborrow();
    let mut ribbon_placement = ribbons.placement.reborrow();
    let mut ribbon_drag = ribbons.drag.reborrow();
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let accent_color = accent.0;
    let scene = sim.scene;

    // Draw the ribbon strips (left + right) FIRST so panes layout
    // around them. The third closure decides whether each ribbon
    // button is greyed out — we disable the lidar pane's button when
    // the active scene isn't Lidar (no settings to show).
    let _clicks = draw_assembly(
        ctx,
        accent_color,
        RIBBONS,
        RIBBON_ITEMS,
        &mut ribbon_open,
        &mut ribbon_placement,
        &mut ribbon_drag,
        |id| match id {
            // Every scene now has a right-rail description, even if it
            // has no extra per-example controls.
            PANE_SCENE => false,
            _ => false,
        },
    );

    let is_open = |id: &'static str| -> bool {
        let Some(item) = find_item(RIBBON_ITEMS, id) else {
            return false;
        };
        let (rid, _, _) = ribbon_placement.resolve(item);
        ribbon_open.is_open(rid, id)
    };

    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);
    let frame_ms = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);

    let body_count = sim.model.body_count;

    // ── Left rail: main control pane. Toggle via the left ribbon. ─
    if is_open(PANE_MAIN) {
        Pane2::new(
            PANE_MAIN,
            "molla",
            PaneAnchor::LeftRail(RailZone::Start),
            accent_color,
        )
        .show(ctx, |ui| {
            ui.add_space(2.0);

            // Scene dropdown — replaces the row of number-keyed buttons.
            ui.horizontal(|ui| {
                ui.label("Example");
                egui::ComboBox::from_id_salt("scene_dropdown")
                    .selected_text(scene.label())
                    .show_ui(ui, |ui| {
                        for &candidate in Scene::ALL {
                            if ui
                                .selectable_label(scene == candidate, candidate.label())
                                .clicked()
                                && scene != candidate
                            {
                                pending.0 = Some(candidate);
                            }
                        }
                    });
            });
            ui.separator();

            let audit = scene.physics_audit();
            readout(ui, "Audit", audit.status.label());
            ui.collapsing("Physics audit", |ui| {
                readout(ui, "Reference", audit.reference_case);
                readout(ui, "CPU proof", audit.cpu_proof);
                readout(ui, "GPU proof", audit.gpu_proof);
                readout(ui, "Notes", audit.notes);
            });
            ui.separator();

            readout(ui, "FPS", &format!("{fps:>5.0}  ({frame_ms:>4.1} ms)"));
            readout(
                ui,
                "Bodies",
                &format!(
                    "{body_count}  ({SUBSTEPS_PER_FRAME}× @ {:.0} Hz)",
                    1.0 / FIXED_DT
                ),
            );
            ui.separator();

            readout(
                ui,
                "Phys total",
                &format!("{:>4.0} µs", stats.smoothed_step_us),
            );
            readout(
                ui,
                "  · collide",
                &format!("{:>4.0} µs", stats.smoothed_collide_us),
            );
            readout(
                ui,
                "  · dispatch",
                &format!("{:>4.0} µs", stats.smoothed_solve_us),
            );
            readout(
                ui,
                "  · download",
                &format!("{:>4.0} µs", stats.smoothed_download_us),
            );
            readout(ui, "Contacts", &format!("{}", stats.last_contacts));
            ui.separator();

            readout(ui, "Solver", "XPBD-GPU");
            readout(ui, "Adapter", &sim.adapter_label);
            readout(ui, "Friction", &format!("{PLANE_FRICTION:.2}"));
            readout(ui, "Restitution", &format!("{PLANE_RESTITUTION:.2}"));
            ui.separator();

            // Activity view toggle.
            let view_label = if activity.enabled {
                "View: ACTIVITY"
            } else {
                "View: NORMAL"
            };
            readout(ui, view_label, &format!("{} active", activity.active_count));
            ui.horizontal(|ui| {
                let label = if activity.enabled {
                    "Hide sleep"
                } else {
                    "Show sleep (T)"
                };
                if button(ui, label, accent_color).clicked() {
                    activity.enabled = !activity.enabled;
                }
            });
            ui.separator();

            ui.horizontal(|ui| {
                let label = if run.paused { "▶ Play" } else { "⏸ Pause" };
                if button(ui, label, accent_color).clicked() {
                    run.paused = !run.paused;
                }
                if run.paused && button(ui, "Step", accent_color).clicked() {
                    run.step_once = true;
                }
                if button(ui, "Reset", accent_color).clicked() {
                    sim.reset();
                }
            });

            ui.add_space(4.0);
            ui.small("LMB drag · T activity · W wireframe");
            ui.small("Space pause · R reset · MMB pan · LMB+RMB orbit");
        });
    } // end is_open(PANE_MAIN)

    // ── Right rail: a single pane whose body switches on `scene`.
    // The ribbon button toggles whether the pane is shown; whether
    // the button is enabled at all is decided by the disabled-closure
    // passed to `draw_assembly` above (only enabled for scenes that
    // have something to show here).
    let scene_pane_title: Option<&'static str> = match scene {
        Scene::Lidar => Some("Lidar"),
        Scene::Imu => Some("IMU"),
        Scene::Contact => Some("Contacts"),
        Scene::Arm => Some("Arm (PD)"),
        Scene::Shapes => Some("Shapes (Stage 13)"),
        Scene::GpuIkArm => Some("GPU IK"),
        Scene::GpuNarrowPhase => Some("GPU narrow phase"),
        Scene::Camera => Some("Camera"),
        Scene::Actuator => Some("Actuator (Stage 18)"),
        Scene::Quadrotor => Some("Quadrotor (Stage 17)"),
        Scene::BroadPhase => Some("GPU broad phase"),
        Scene::VbdCloth => Some("VBD cloth (Stage 20)"),
        Scene::ReferencePendulum => Some("Native double pendulum"),
        Scene::DiffArm => Some("Diff arm (Stage 23)"),
        Scene::DiffArm6 => Some("Diff Featherstone (Stage 23.1)"),
        Scene::MeshTools => Some("Mesh tools (Stage 21)"),
        Scene::NeuralPolicy => Some("Neural policy (Stage 18.1)"),
        Scene::NeuralLstm => Some("Neural LSTM (Stage 31)"),
        Scene::Cable => Some("Cable (Stage 20.1)"),
        Scene::Mpm => Some("MPM sand (Stage 24d)"),
        Scene::Jelly => Some("Jelly (Stage 20.3)"),
        Scene::Pendulum => Some("Pendulum"),
        Scene::Conveyor => Some("Conveyor"),
        Scene::ClothBending => Some("Cloth bending"),
        Scene::ClothRollers => Some("Cloth rollers"),
        Scene::ClothTwist => Some("Cloth twist"),
        Scene::MpmSnowBall => Some("MPM snow ball"),
        Scene::MpmTwoWay => Some("MPM two-way"),
        Scene::MpmMultiMaterial => Some("MPM multi-material"),
        Scene::MpmBeamTwist => Some("MPM beam twist"),
        Scene::CableBundleHysteresis => Some("Cable bundle hysteresis"),
        Scene::CartPole => Some("Cart-pole (PD)"),
        Scene::CartPoleSwarm => Some("Cart-pole swarm"),
        Scene::FourBar => Some("Four-bar"),
        Scene::DiffBall => Some("Diff ball"),
        Scene::DiffSpring => Some("Diff spring"),
        Scene::SoftbodyDrop => Some("Softbody → cloth"),
        Scene::ClothCollar => Some("Cloth collar (self-coll)"),
        Scene::FastSoftImpact => Some("Fast soft impact (CCD)"),
        Scene::SoftBrake => Some("Soft brake (friction)"),
        Scene::FemHeat => Some("FEM heat (Stage 36)"),
        Scene::FemHeatP2 => Some("FEM heat P2 (Stage 36.3)"),
        Scene::FemBeam => Some("FEM beam (Stage 36.1)"),
        Scene::FemDiffBeam => Some("FEM diff beam (Stage 36.4)"),
        Scene::FemStokes => Some("FEM Stokes (Stage 36.2)"),
        Scene::WarpRaycast => Some("Warp raycast"),
        Scene::WarpRaymarch => Some("Warp raymarch"),
        Scene::WarpMesh => Some("Warp mesh (animated)"),
        Scene::WarpMeshIntersect => Some("Warp mesh intersect (overlap)"),
        Scene::WarpSampleMesh => Some("Warp sample mesh"),
        Scene::WarpMarchingCubes => Some("Warp marching cubes"),
        Scene::WarpNvdb => Some("Warp NanoVDB sparse grid"),
        Scene::WarpDem => Some("Warp DEM (MPM sand)"),
        Scene::WarpFluid => Some("Warp fluid (MPM)"),
        Scene::WarpSph => Some("Warp SPH (MPM)"),
        Scene::WarpTileFft => Some("Warp tile FFT"),
        Scene::WarpTileMlp => Some("Warp tile MLP"),
        Scene::WarpTileNbody => Some("Warp tile N-body"),
        Scene::WarpTileCholesky => Some("Warp tile Cholesky"),
        Scene::WarpParticleRepulsion => Some("Warp particle repulsion"),
        Scene::WarpFemDiffusion => Some("Warp FEM diffusion 2D"),
        Scene::WarpFemDiffusion3D => Some("Warp FEM diffusion 3D"),
        Scene::WarpFemConvDiff => Some("Warp FEM conv-diff"),
        Scene::WarpFemConvDiffDg => Some("Warp FEM conv-diff DG"),
        Scene::WarpFemTaylorGreen => Some("Warp FEM Taylor-Green"),
        Scene::WarpFemKelvinHelmholtz => Some("Warp FEM Kelvin-Helmholtz"),
        Scene::WarpFemStreamlines => Some("Warp FEM streamlines"),
        Scene::WarpFemStokesTransfer => Some("Warp FEM Stokes transfer"),
        Scene::WarpFemDeformedGeometry => Some("Warp FEM deformed geometry"),
        Scene::WarpFemDistortion => Some("Warp FEM distortion energy"),
        Scene::WarpFemElasticOpt => Some("Warp FEM elastic shape opt"),
        Scene::WarpFemMixedElasticity => Some("Warp FEM mixed elasticity"),
        Scene::WarpFemNonconformingContact => Some("Warp FEM nonconforming contact"),
        Scene::WarpFftFluid => Some("Warp FFT fluid"),
        Scene::WarpDistortion3D => Some("Warp distortion 3D"),
        Scene::WarpApicFluid => Some("Warp APIC fluid (Stage 37.6)"),
        Scene::WarpWave => Some("Warp wave (Stage 37.1)"),
        Scene::WarpDiffRay => Some("Warp diff raycast (Stage 37.7)"),
        Scene::WarpTileFilter => Some("Warp tile filter (Stage 37.8)"),
        Scene::WarpTileCompaction => Some("Warp tile compaction (Stage 37.8)"),
        Scene::AdamLandscape => Some("Adam optimization landscape (Stage 38c)"),
        Scene::WarpTileMatmul => Some("Warp tile matmul (Stage 37.8)"),
        Scene::WarpTileConvolution => Some("Warp tile convolution (Stage 37.8)"),
        Scene::WarpTileMcgp => Some("Warp tile MCGP (Stage 37.8)"),
        Scene::ImplicitMpmBingham => Some("Implicit MPM Bingham (Stage 39A)"),
        Scene::WarpStableFluid => Some("Warp stable fluid (Stage 39C)"),
        Scene::JellyDampingFall => Some("Jelly damping sweep (Stage 36/41)"),
        Scene::DgBurgersShock => Some("DG Burgers shock (Stage 43)"),
        Scene::ShallowWaterDamBreak => Some("Shallow water dam break (Stage 43 v2)"),
        Scene::ImplicitMpmFull27 => Some("Implicit MPM 27-stencil (Stage 40)"),
        Scene::SplatPoseRecovery => Some("Splat pose recovery (Stage 44 v2)"),
        Scene::AmrHeat => Some("AMR heat (Stage 45 v2)"),
        Scene::Magnetostatics => Some("Magnetostatics (Stage 46)"),
        Scene::PhaseFieldFracture => Some("Phase-field fracture (Tier C3)"),
        Scene::ClothPokerCards => Some("Cloth poker cards (Newton port)"),
        Scene::SoftbodyHanging => Some("Softbody hanging (E sweep)"),
        Scene::CableTwist => Some("Cable twist (tangent rotate)"),
        Scene::IkCustom => Some("IK custom (collision avoidance)"),
        Scene::IkCubeStacking => Some("IK cube stacking"),
        Scene::DiffsimDrone => Some("Diffsim drone (Adam MPC)"),
        Scene::FluidCheckpointOptimization => Some("Fluid checkpoint opt"),
        Scene::NavierStokesPerturbationOptimization => Some("NS perturbation opt"),
        Scene::WarpFftPoisson2d => Some("Warp FFT Poisson 2D"),
        _ => None,
    };
    let scene_pane_title = scene_pane_title.or(Some(scene.label()));
    if let Some(title) = scene_pane_title {
        if is_open(PANE_SCENE) {
            let anchor = PaneAnchor::RightRail(RailZone::Start);
            Pane2::new(PANE_SCENE, title, anchor, accent_color)
                .resize(PaneResize::BOTH)
                .show(ctx, |ui| {
                    scene_description_pane(ui, scene, &mut description_pane);
                    match scene {
                        Scene::Lidar => lidar_pane(ui, &mut lidar_config, &lidar_state),
                        Scene::Imu => imu_pane(ui, &imu_state),
                        Scene::Contact => contact_pane(ui, &contact_state),
                        Scene::Arm => arm_pane(ui, &mut pane_states.arm, &sim, &mut pending),
                        Scene::Shapes => shapes_pane(ui, &sim),
                        Scene::GpuIkArm => gpu_ik_pane(ui, &mut pane_states.gpu_ik, &sim),
                        Scene::GpuNarrowPhase => {
                            gpu_np_pane(ui, &mut pane_states.gpu_np, accent_color)
                        }
                        Scene::Camera => {
                            camera_pane(ui, &mut pane_states.camera, anchor, accent_color)
                        }
                        Scene::Actuator => actuator_pane(ui, &mut pane_states.actuator),
                        Scene::Quadrotor => quadrotor_pane(ui, &mut pane_states.quad),
                        Scene::BroadPhase => gpu_bp_pane(ui, &mut pane_states.gpu_bp, accent_color),
                        Scene::VbdCloth => {
                            vbd_cloth_pane(ui, &mut pane_states.vbd, &mut sim, accent_color)
                        }
                        Scene::ReferencePendulum => {
                            reference_pendulum_pane(ui, &mut pane_states.reference_pendulum, &sim)
                        }
                        Scene::DiffArm => diff_arm_pane(ui, &mut pane_states.diff_arm),
                        Scene::DiffArm6 => diff_arm6_pane(ui, &mut pane_states.diff_arm6),
                        Scene::MeshTools => {
                            mesh_tools_pane(ui, &mut pane_states.mesh, &mut pending, scene)
                        }
                        Scene::NeuralPolicy => neural_policy_pane(ui, &mut pane_states.neural),
                        Scene::NeuralLstm => neural_lstm_pane(ui, &mut pane_states.neural_lstm),
                        Scene::Cable => {
                            cable_pane(ui, &mut pane_states.cable, &mut sim, accent_color)
                        }
                        Scene::Mpm => mpm_pane(ui, &mut pane_states.mpm, &mut sim, accent_color),
                        Scene::Jelly => jelly_pane(
                            ui,
                            &mut pane_states.jelly,
                            &mut sim,
                            &mut pending,
                            accent_color,
                        ),
                        Scene::Pendulum => pendulum_pane(ui, &pane_states.pendulum),
                        Scene::Conveyor => conveyor_pane(ui, &mut pane_states.conveyor),
                        Scene::ClothBending => {
                            cloth_bending_pane(ui, &mut pane_states.cloth_bend, &mut pending, scene)
                        }
                        Scene::ClothRollers => {
                            cloth_rollers_pane(ui, &mut warp_states.cloth_rollers)
                        }
                        Scene::ClothTwist => cloth_twist_pane(ui, &mut pane_states.cloth_twist),
                        Scene::CableBundleHysteresis => {
                            cable_pane(ui, &mut pane_states.cable, &mut sim, accent_color)
                        }
                        Scene::MpmSnowBall => {
                            mpm_pane(ui, &mut pane_states.mpm, &mut sim, accent_color)
                        }
                        Scene::MpmTwoWay => {
                            mpm_pane(ui, &mut pane_states.mpm, &mut sim, accent_color)
                        }
                        Scene::MpmMultiMaterial => {
                            mpm_pane(ui, &mut pane_states.mpm, &mut sim, accent_color)
                        }
                        Scene::MpmBeamTwist => {
                            mpm_pane(ui, &mut pane_states.mpm, &mut sim, accent_color)
                        }
                        Scene::CartPole => {
                            cartpole_pane(ui, &mut pane_states.cartpole, accent_color)
                        }
                        Scene::CartPoleSwarm => cartpole_swarm_pane(ui, &sim),
                        Scene::FourBar => fourbar_pane(ui, &mut pane_states.fourbar),
                        Scene::DiffBall => {
                            diff_ball_pane(ui, &mut pane_states.diff_ball, accent_color)
                        }
                        Scene::DiffSpring => {
                            diff_spring_pane(ui, &mut pane_states.diff_spring, accent_color)
                        }
                        Scene::SoftbodyDrop => softbody_drop_pane(ui, &mut sim),
                        Scene::ClothCollar => cloth_collar_pane(ui, &mut sim),
                        Scene::FastSoftImpact => fast_soft_impact_pane(ui, &mut sim),
                        Scene::SoftBrake => soft_brake_pane(ui, &mut sim),
                        Scene::FemHeat => fem_heat_pane(ui, &mut pane_states.fem_heat),
                        Scene::FemHeatP2 => fem_heat_p2_pane(ui, &mut pane_states.fem_heat_p2),
                        Scene::FemBeam => fem_beam_pane(ui, &mut pane_states.fem_beam),
                        Scene::FemDiffBeam => {
                            fem_diff_beam_pane(ui, &mut pane_states.fem_diff_beam)
                        }
                        Scene::FemStokes => fem_stokes_pane(ui, &mut pane_states.fem_stokes),
                        Scene::TireDynamics => {
                            tire_dynamics_pane(ui, &mut pane_states.tire_dynamics)
                        }
                        Scene::TireTraction => {
                            tire_traction_pane(ui, &mut pane_states.tire_traction, accent_color)
                        }
                        Scene::WarpRaycast => warp_raycast_pane(ui, &mut warp_states.raycast),
                        Scene::WarpRaymarch => warp_raymarch_pane(ui, &mut warp_states.raymarch),
                        Scene::WarpMesh => warp_mesh_pane(ui, &mut warp_states.mesh),
                        Scene::WarpMeshIntersect => {
                            warp_mesh_intersect_pane(ui, &mut warp_states.mesh_intersect)
                        }
                        Scene::WarpSampleMesh => {
                            warp_sample_mesh_pane(ui, &mut warp_states.sample_mesh)
                        }
                        Scene::WarpMarchingCubes => {
                            warp_marching_cubes_pane(ui, &mut warp_states.marching_cubes)
                        }
                        Scene::WarpNvdb => warp_nvdb_pane(ui, &mut warp_states.nvdb),
                        Scene::WarpDem => warp_dem_pane(ui, &mut warp_states),
                        Scene::WarpFluid => warp_fluid_pane(ui, &mut warp_states),
                        Scene::WarpSph => warp_sph_pane(ui, &mut warp_states),
                        Scene::WarpTileFft => warp_tile_fft_pane(ui, &mut warp_states.tile_fft),
                        Scene::WarpTileMlp => warp_tile_mlp_pane(ui, &mut warp_states.tile_mlp),
                        Scene::WarpTileNbody => {
                            warp_tile_nbody_pane(ui, &mut warp_states.tile_nbody)
                        }
                        Scene::WarpTileCholesky => {
                            warp_tile_cholesky_pane(ui, &mut warp_states.tile_cholesky)
                        }
                        Scene::WarpParticleRepulsion => {
                            warp_repulsion_pane(ui, &mut warp_states.repulsion)
                        }
                        Scene::WarpFemDiffusion => {
                            warp_fem_diffusion_pane(ui, &mut warp_states.fem_diff)
                        }
                        Scene::WarpFemDiffusion3D => {
                            warp_fem_diffusion_3d_pane(ui, &mut warp_states.fem_diff3d)
                        }
                        Scene::WarpFemConvDiff => {
                            warp_fem_convdiff_pane(ui, &mut warp_states.fem_convdiff)
                        }
                        Scene::WarpFemConvDiffDg => {
                            warp_fem_convdiff_dg_pane(ui, &mut warp_states.fem_convdiff_dg)
                        }
                        Scene::WarpFemTaylorGreen => {
                            warp_fem_taylor_pane(ui, &mut warp_states.fem_taylor)
                        }
                        Scene::WarpFemKelvinHelmholtz => {
                            warp_fem_kh_pane(ui, &mut warp_states.fem_kh)
                        }
                        Scene::WarpFemStreamlines => {
                            warp_fem_streamlines_pane(ui, &mut warp_states.fem_streamlines)
                        }
                        Scene::WarpFemStokesTransfer => {
                            warp_fem_stokes_transfer_pane(ui, &mut warp_states.fem_stokes_transfer)
                        }
                        Scene::WarpFemDeformedGeometry => {
                            warp_fem_deformed_pane(ui, &mut warp_states.fem_deformed)
                        }
                        Scene::WarpFemDistortion => {
                            warp_fem_distortion_pane(ui, &mut warp_states.fem_distortion)
                        }
                        Scene::WarpFemElasticOpt => {
                            warp_fem_elastic_opt_pane(ui, &mut warp_states.fem_elastic_opt)
                        }
                        Scene::WarpFemMixedElasticity => warp_fem_mixed_elasticity_pane(
                            ui,
                            &mut warp_states.fem_mixed_elasticity,
                        ),
                        Scene::WarpFemNonconformingContact => warp_fem_nonconforming_contact_pane(
                            ui,
                            &mut warp_states.fem_nonconforming_contact,
                        ),
                        Scene::WarpFftFluid => warp_fft_fluid_pane(ui, &mut warp_states.fft_fluid),
                        Scene::WarpDistortion3D => {
                            warp_distortion_3d_pane(ui, &mut warp_states.distortion_3d)
                        }
                        Scene::WarpApicFluid => {
                            warp_apic_fluid_pane(ui, &mut warp_states.apic_fluid)
                        }
                        Scene::WarpWave => warp_wave_pane(ui, &mut warp_states.wave),
                        Scene::WarpDiffRay => warp_diff_ray_pane(ui, &mut warp_states.diff_ray),
                        Scene::WarpTileFilter => {
                            warp_tile_filter_pane(ui, &mut warp_states.tile_filter)
                        }
                        Scene::WarpTileCompaction => {
                            warp_tile_compaction_pane(ui, &mut warp_states.tile_compaction)
                        }
                        Scene::AdamLandscape => {
                            adam_landscape_pane(ui, &mut warp_states.adam_landscape)
                        }
                        Scene::WarpTileMatmul => {
                            warp_tile_matmul_pane(ui, &mut warp_states.tile_matmul)
                        }
                        Scene::WarpTileConvolution => {
                            warp_tile_convolution_pane(ui, &mut warp_states.tile_convolution)
                        }
                        Scene::WarpTileMcgp => warp_tile_mcgp_pane(ui, &mut warp_states.tile_mcgp),
                        Scene::ImplicitMpmBingham => {
                            implicit_mpm_bingham_pane(ui, &mut warp_states.implicit_mpm_bingham)
                        }
                        Scene::WarpStableFluid => {
                            warp_stable_fluid_pane(ui, &mut warp_states.stable_fluid)
                        }
                        Scene::JellyDampingFall => {
                            jelly_damping_fall_pane(ui, &mut warp_states.jelly_damping)
                        }
                        Scene::DgBurgersShock => dg_burgers_pane(ui, &mut warp_states.dg_burgers),
                        Scene::ShallowWaterDamBreak => {
                            swe_dam_break_pane(ui, &mut warp_states.swe_dam_break)
                        }
                        Scene::ImplicitMpmFull27 => {
                            implicit_mpm_full27_pane(ui, &mut warp_states.implicit_mpm_full27)
                        }
                        Scene::SplatPoseRecovery => {
                            splat_pose_recovery_pane(ui, &mut warp_states.splat_pose)
                        }
                        Scene::AmrHeat => amr_heat_pane(ui, &mut warp_states.amr_heat),
                        Scene::Magnetostatics => {
                            magnetostatics_pane(ui, &mut warp_states.magnetostatics)
                        }
                        Scene::ClothPokerCards => {
                            cloth_poker_cards_pane(ui, &mut warp_states.cloth_poker_cards)
                        }
                        Scene::SoftbodyHanging => {
                            softbody_hanging_pane(ui, &mut warp_states.softbody_hanging)
                        }
                        Scene::CableTwist => cable_twist_pane(ui, &mut warp_states.cable_twist),
                        Scene::IkCustom => ik_custom_pane(ui, &mut warp_states.ik_custom),
                        Scene::IkCubeStacking => {
                            ik_cube_stacking_pane(ui, &mut warp_states.ik_cube_stacking)
                        }
                        Scene::DiffsimDrone => {
                            diffsim_drone_pane(ui, &mut warp_states.diffsim_drone)
                        }
                        Scene::FluidCheckpointOptimization => {
                            fluid_checkpoint_pane(ui, &mut warp_states.fluid_checkpoint)
                        }
                        Scene::NavierStokesPerturbationOptimization => {
                            ns_perturbation_pane(ui, &mut warp_states.ns_perturbation)
                        }
                        Scene::WarpFftPoisson2d => {
                            warp_fft_poisson_2d_pane(ui, &mut warp_states.fft_poisson_2d)
                        }
                        _ => {}
                    }
                });
        }
    }
}

fn scene_description_pane(ui: &mut egui::Ui, scene: Scene, state: &mut SceneDescriptionPaneState) {
    // Master toggle for the whole description block. No `ui.horizontal`
    // and no inline slogan: a horizontal row grows to the combined
    // intrinsic width of its children, so a non-wrapping string next
    // to the button used to stretch the right-rail pane past its
    // anchor and clip everything.
    let label = if state.visible {
        "Hide description"
    } else {
        "Show description"
    };
    if ui.button(label).clicked() {
        state.visible = !state.visible;
    }

    if state.visible {
        // The per-scene description lives inside a collapsing section
        // — its header is a button directly under the toggle. Click
        // to fold/unfold; defaults open so the text shows straight
        // away. The body is a `Label::wrap()` so it reflows to the
        // pane width and renders multi-line instead of clipping.
        egui::CollapsingHeader::new("Expected behavior")
            .default_open(true)
            .show(ui, |ui| {
                ui.add(egui::Label::new(scene.description()).wrap());
            });
    }
    ui.separator();
}

fn build_warp_empty_model() -> (mp::Model, Vec<BodyVisual>) {
    let b = mp::ModelBuilder::new();
    (b.build().expect("warp empty model"), Vec::new())
}

/// Build a unit-icosahedron-ish triangle mesh (12 verts / 20 tris) at
/// the requested radius. Cheap to construct and used by `WarpMesh`,
/// `WarpMeshIntersect`, `WarpSampleMesh`.
fn warp_icosphere_trimesh(r: f64) -> mp::TriMesh {
    let phi = (1.0 + 5.0_f64.sqrt()) * 0.5;
    let inv = (1.0 + phi * phi).sqrt().recip() * r;
    let v = |x: f64, y: f64, z: f64| MVec3::new(x * inv, y * inv, z * inv);
    let vertices = vec![
        v(-1.0, phi, 0.0),
        v(1.0, phi, 0.0),
        v(-1.0, -phi, 0.0),
        v(1.0, -phi, 0.0),
        v(0.0, -1.0, phi),
        v(0.0, 1.0, phi),
        v(0.0, -1.0, -phi),
        v(0.0, 1.0, -phi),
        v(phi, 0.0, -1.0),
        v(phi, 0.0, 1.0),
        v(-phi, 0.0, -1.0),
        v(-phi, 0.0, 1.0),
    ];
    let indices = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    let mut m = mp::TriMesh {
        vertices,
        indices,
        ..Default::default()
    };
    m.build_bvh(4);
    m
}

/// Bundle of every Warp-port scene's runtime state, registered as a
/// single Bevy resource so we don't push `SceneStates` /
/// `PaneSceneStates` past Bevy's 16-tuple SystemParam ceiling.
#[derive(Resource)]
struct WarpExampleStates {
    raycast: WarpRaycastState,
    raymarch: WarpRaymarchState,
    mesh: WarpMeshState,
    mesh_intersect: WarpMeshIntersectState,
    sample_mesh: WarpSampleMeshState,
    marching_cubes: WarpMarchingCubesState,
    nvdb: WarpNvdbState,
    tile_fft: WarpTileFftState,
    tile_mlp: WarpTileMlpState,
    tile_nbody: WarpTileNbodyState,
    tile_cholesky: WarpTileCholeskyState,
    repulsion: WarpRepulsionState,
    fem_diff: WarpFemDiffusionState,
    fem_diff3d: WarpFemDiffusion3DState,
    fem_convdiff: WarpFemConvDiffState,
    fem_convdiff_dg: WarpFemConvDiffDgState,
    fem_taylor: WarpFemTaylorGreenState,
    fem_kh: WarpFemKhState,
    fem_streamlines: WarpFemStreamlinesState,
    fem_stokes_transfer: WarpFemStokesTransferState,
    fem_deformed: WarpFemDeformedState,
    fem_distortion: WarpFemDistortionState,
    fem_elastic_opt: WarpFemElasticOptState,
    fem_mixed_elasticity: WarpFemMixedElasticityState,
    fem_nonconforming_contact: WarpFemNonconformingContactState,
    fft_fluid: WarpFftFluidState,
    distortion_3d: WarpDistortion3DState,
    apic_fluid: WarpApicFluidState,
    wave: WarpWaveExampleState,
    diff_ray: WarpDiffRayExampleState,
    tile_filter: WarpTileFilterState,
    tile_compaction: WarpTileCompactionState,
    adam_landscape: AdamLandscapeState,
    tile_matmul: WarpTileMatmulState,
    tile_convolution: WarpTileConvolutionState,
    tile_mcgp: WarpTileMcgpState,
    // Newton + Warp follow-ups (Stage 39).
    cloth_rollers: ClothRollersState,
    cloth_poker_cards: ClothPokerCardsState,
    softbody_hanging: SoftbodyHangingState,
    cable_twist: CableTwistState,
    ik_custom: IkCustomState,
    ik_cube_stacking: IkCubeStackingState,
    diffsim_drone: DiffsimDroneState,
    fluid_checkpoint: FluidCheckpointState,
    ns_perturbation: NsPerturbationState,
    fft_poisson_2d: WarpFftPoisson2dState,
    implicit_mpm_bingham: ImplicitMpmBinghamState,
    stable_fluid: WarpStableFluidState,
    jelly_damping: JellyDampingFallState,
    dg_burgers: DgBurgersState,
    swe_dam_break: SweDamBreakState,
    implicit_mpm_full27: ImplicitMpmFull27State,
    splat_pose: SplatPoseRecoveryState,
    amr_heat: AmrHeatState,
    magnetostatics: MagnetostaticsState,
}

impl Default for WarpExampleStates {
    fn default() -> Self {
        Self {
            raycast: WarpRaycastState::default(),
            raymarch: WarpRaymarchState::default(),
            mesh: WarpMeshState::default(),
            mesh_intersect: WarpMeshIntersectState::default(),
            sample_mesh: WarpSampleMeshState::default(),
            marching_cubes: WarpMarchingCubesState::default(),
            nvdb: WarpNvdbState::default(),
            tile_fft: WarpTileFftState::default(),
            tile_mlp: WarpTileMlpState::default(),
            tile_nbody: WarpTileNbodyState::default(),
            tile_cholesky: WarpTileCholeskyState::default(),
            repulsion: WarpRepulsionState::default(),
            fem_diff: WarpFemDiffusionState::default(),
            fem_diff3d: WarpFemDiffusion3DState::default(),
            fem_convdiff: WarpFemConvDiffState::default(),
            fem_convdiff_dg: WarpFemConvDiffDgState::default(),
            fem_taylor: WarpFemTaylorGreenState::default(),
            fem_kh: WarpFemKhState::default(),
            fem_streamlines: WarpFemStreamlinesState::default(),
            fem_stokes_transfer: WarpFemStokesTransferState::default(),
            fem_deformed: WarpFemDeformedState::default(),
            fem_distortion: WarpFemDistortionState::default(),
            fem_elastic_opt: WarpFemElasticOptState::default(),
            fem_mixed_elasticity: WarpFemMixedElasticityState::default(),
            fem_nonconforming_contact: WarpFemNonconformingContactState::default(),
            fft_fluid: WarpFftFluidState::default(),
            distortion_3d: WarpDistortion3DState::default(),
            apic_fluid: WarpApicFluidState::default(),
            wave: WarpWaveExampleState::default(),
            diff_ray: WarpDiffRayExampleState::default(),
            tile_filter: WarpTileFilterState::default(),
            tile_compaction: WarpTileCompactionState::default(),
            adam_landscape: AdamLandscapeState::default(),
            tile_matmul: WarpTileMatmulState::default(),
            tile_convolution: WarpTileConvolutionState::default(),
            tile_mcgp: WarpTileMcgpState::default(),
            cloth_rollers: ClothRollersState::default(),
            cloth_poker_cards: ClothPokerCardsState::default(),
            softbody_hanging: SoftbodyHangingState::default(),
            cable_twist: CableTwistState::default(),
            ik_custom: IkCustomState::default(),
            ik_cube_stacking: IkCubeStackingState::default(),
            diffsim_drone: DiffsimDroneState::default(),
            fluid_checkpoint: FluidCheckpointState::default(),
            ns_perturbation: NsPerturbationState::default(),
            fft_poisson_2d: WarpFftPoisson2dState::default(),
            implicit_mpm_bingham: ImplicitMpmBinghamState::default(),
            stable_fluid: WarpStableFluidState::default(),
            jelly_damping: JellyDampingFallState::default(),
            dg_burgers: DgBurgersState::default(),
            swe_dam_break: SweDamBreakState::default(),
            implicit_mpm_full27: ImplicitMpmFull27State::default(),
            splat_pose: SplatPoseRecoveryState::default(),
            amr_heat: AmrHeatState::default(),
            magnetostatics: MagnetostaticsState::default(),
        }
    }
}

include!("examples/52_warpraycast.rs");
include!("examples/53_warpraymarch.rs");
include!("examples/54_warpmesh.rs");
include!("examples/55_warpmeshintersect.rs");
include!("examples/56_warpsamplemesh.rs");
include!("examples/57_warpmarchingcubes.rs");
include!("examples/58_warpnvdb.rs");
include!("examples/59_warp_dem.rs");
include!("examples/60_warp_fluid.rs");
include!("examples/61_warp_sph.rs");
include!("examples/62_warptilefft.rs");
include!("examples/63_warptilemlp.rs");
include!("examples/64_warptilenbody.rs");
include!("examples/65_warptilecholesky.rs");
include!("examples/66_warpparticlerepulsion.rs");
include!("examples/67_warpfemdiffusion.rs");
include!("examples/68_warpfemdiffusion3d.rs");
include!("examples/69_warpfemconvdiff.rs");
include!("examples/109_warpfemconvdiffdg.rs");
include!("examples/70_warpfemtaylorgreen.rs");
include!("examples/71_warpfemkelvinhelmholtz.rs");
include!("examples/72_warpfemstreamlines.rs");
include!("examples/112_warpfemstokestransfer.rs");
include!("examples/73_warpfemdeformedgeometry.rs");
include!("examples/74_warpfemdistortion.rs");
include!("examples/75_warpfemelasticopt.rs");
include!("examples/110_warpfemmixedelasticity.rs");
include!("examples/111_warpfemnonconformingcontact.rs");
include!("examples/76_warpfftfluid.rs");
include!("examples/77_warpdistortion3d.rs");
include!("examples/79_warpapicfluid.rs");
include!("examples/78_warpwave.rs");
include!("examples/80_warpdiffray.rs");
include!("examples/81_warptilefilter.rs");
include!("examples/82_warptilecompaction.rs");
include!("examples/83_adamlandscape.rs");
include!("examples/90_warptilematmul.rs");
include!("examples/91_warptileconvolution.rs");
include!("examples/92_warptilemcgp.rs");
include!("examples/84_clothpokercards.rs");
include!("examples/85_softbodyhanging.rs");
include!("examples/86_cabletwist.rs");
include!("examples/87_ikcustom.rs");
include!("examples/113_ikcubestacking.rs");
include!("examples/88_diffsimdrone.rs");
include!("examples/116_fluidcheckpointoptimization.rs");
include!("examples/117_navierstokesperturbationoptimization.rs");
include!("examples/89_warpfftpoisson2d.rs");
include!("examples/93_implicitmpmbingham.rs");
include!("examples/94_warpstablefluid.rs");
include!("examples/95_jellydampingfall.rs");
include!("examples/96_dgburgersshock.rs");
include!("examples/97_shallowwaterdambreak.rs");
include!("examples/98_implicitmpmfull27.rs");
include!("examples/99_splatposerecovery.rs");
include!("examples/103_mpm_multi_material.rs");
include!("examples/100_amrheat.rs");
include!("examples/101_magnetostatics.rs");
include!("examples/102_phase_field_fracture.rs");
include!("examples/115_mpm_beam_twist.rs");

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn scene_registry_has_unique_labels() {
        let mut seen = HashSet::new();
        for scene in Scene::ALL {
            let label = scene.label();
            assert!(seen.insert(label), "duplicate scene label: {label}");
        }
    }

    #[test]
    fn all_viewer_scenes_have_descriptions() {
        for scene in Scene::ALL {
            let description = scene.description().trim();
            assert!(
                description.len() >= 80,
                "{}: description is missing or too short",
                scene.label()
            );
            assert!(
                !description.contains("TODO"),
                "{}: description still has placeholder text",
                scene.label()
            );
        }
    }

    #[test]
    fn all_viewer_scenes_build_initial_model_without_gpu() {
        for scene in Scene::ALL {
            let (model, visuals) = Sim::build_scene_model(*scene);
            let state = model
                .state()
                .unwrap_or_else(|e| panic!("{}: state build failed: {e}", scene.label()));
            assert_eq!(
                state.body_q.host().unwrap().len(),
                model.body_count,
                "{}: body_q count mismatch",
                scene.label()
            );
            assert_eq!(
                state.particle_q.host().unwrap().len(),
                model.particle_count,
                "{}: particle_q count mismatch",
                scene.label()
            );
            assert!(
                visuals.len() <= model.body_count,
                "{}: more body visuals ({}) than bodies ({})",
                scene.label(),
                visuals.len(),
                model.body_count
            );
            for (i, q) in state.body_q.host().unwrap().iter().enumerate() {
                let p = q.position;
                assert!(
                    p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
                    "{}: body {i} starts at non-finite position {p:?}",
                    scene.label()
                );
            }
            for (i, p) in state.particle_q.host().unwrap().iter().enumerate() {
                assert!(
                    p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
                    "{}: particle {i} starts at non-finite position {p:?}",
                    scene.label()
                );
            }
        }
    }

    #[test]
    fn all_viewer_scenes_have_physics_audit_entries() {
        for scene in Scene::ALL {
            let audit = scene.physics_audit();
            assert!(
                !audit.reference_case.trim().is_empty(),
                "{}: missing audit reference case",
                scene.label()
            );
            assert!(
                !audit.cpu_proof.trim().is_empty(),
                "{}: missing audit CPU proof",
                scene.label()
            );
            assert!(
                !audit.gpu_proof.trim().is_empty(),
                "{}: missing audit GPU proof/status",
                scene.label()
            );
            assert!(
                !audit.notes.trim().is_empty(),
                "{}: missing audit notes",
                scene.label()
            );
        }
    }

    #[test]
    fn physics_audit_keeps_placeholders_explicit() {
        let placeholders = Scene::ALL
            .iter()
            .copied()
            .filter(|scene| scene.physics_audit().status == PhysicsAuditStatus::Placeholder)
            .collect::<Vec<_>>();
        assert_eq!(
            placeholders,
            Vec::<Scene>::new(),
            "placeholder scenes changed; update the physics audit honestly"
        );
        for scene in placeholders {
            let audit = scene.physics_audit();
            assert!(
                audit.notes.contains("needs"),
                "{}: placeholder audit must say what is missing",
                scene.label()
            );
        }
    }

    #[test]
    fn warp_raymarch_sdf_ray_hits_signed_volume() {
        let inside = warp_raymarch_sdf(bevy::math::Vec3::new(0.0, 0.75, 0.0));
        let sphere_surface = warp_raymarch_sdf(bevy::math::Vec3::new(0.0, 1.5, 0.0));
        let plane_surface = warp_raymarch_sdf(bevy::math::Vec3::new(2.0, -1.0, 2.0));
        assert!(
            inside < -0.1,
            "center should be inside the signed volume: {inside}"
        );
        assert!(
            sphere_surface.abs() < 1.0e-5,
            "top point should sit on the SDF surface: {sphere_surface}"
        );
        assert!(
            plane_surface.abs() < 1.0e-5,
            "ground point should sit on the SDF plane: {plane_surface}"
        );

        let cam = bevy::math::Vec3::new(3.0, 1.5, 0.0);
        let target = bevy::math::Vec3::new(0.0, 0.75, 0.0);
        let dir = (target - cam).normalize_or_zero();
        let mut t = 0.0_f32;
        let mut hit = None;
        for _ in 0..64 {
            let p = cam + dir * t;
            let d = warp_raymarch_sdf(p);
            assert!(d.is_finite(), "raymarch SDF returned non-finite distance");
            if d < 0.001 {
                hit = Some((p, d, t));
                break;
            }
            t += d;
        }
        let (p, d, t_hit) = hit.expect("central ray should hit the procedural SDF volume");
        assert!(
            p.is_finite() && t_hit > 1.0 && t_hit < 4.0,
            "unexpected raymarch hit p={p:?} t={t_hit}"
        );
        assert!(d.abs() < 0.001, "hit should land near the SDF surface: {d}");
    }

    #[test]
    fn warp_nvdb_sparse_sdf_lookup_has_active_band_and_normals() {
        let sphere_mesh = warp_icosphere_trimesh(0.4);
        let sparse = molla::geometry::bake_mesh_to_sparse_sdf(&sphere_mesh, 32, 0.05, 4);
        assert!(sparse.grid.active_voxel_count() > 0);
        assert!(sparse.grid.leaf_count() > 0);
        assert!(
            sparse.grid.active_voxel_count() as u64 <= sparse.logical_voxel_count(),
            "active sparse voxels cannot exceed the logical dense grid"
        );

        let cell_tol = 4.0 * sparse.cell_size;
        let surface = MVec3::new(0.4, 0.0, 0.0);
        let just_inside = MVec3::new(0.35, 0.0, 0.0);
        let just_outside = MVec3::new(0.45, 0.0, 0.0);
        let d_surface = sparse.sample(surface);
        let d_inside = sparse.sample(just_inside);
        let d_outside = sparse.sample(just_outside);
        assert!(
            d_inside < d_outside,
            "sparse SDF should increase from inside to outside: inside={d_inside}, outside={d_outside}"
        );
        assert!(
            d_surface.abs() < cell_tol,
            "surface sample should be within voxel tolerance: d={d_surface}, tol={cell_tol}"
        );

        let normal = sparse.normal(surface);
        assert!(
            normal.x > 0.5 && normal.length().is_finite(),
            "surface normal should point roughly outward along +X: {normal:?}"
        );
    }

    #[test]
    fn warp_tile_nbody_conserves_pairwise_momentum_and_center_of_mass() {
        let mut pos = vec![
            Vec3::new(-0.8, 1.0, 0.0),
            Vec3::new(0.8, 1.0, 0.0),
            Vec3::new(0.0, 1.0, -0.8),
            Vec3::new(0.0, 1.0, 0.8),
        ];
        let mut vel = vec![
            Vec3::new(0.0, 0.0, 0.15),
            Vec3::new(0.0, 0.0, -0.15),
            Vec3::new(-0.15, 0.0, 0.0),
            Vec3::new(0.15, 0.0, 0.0),
        ];
        let initial_momentum = vel.iter().copied().fold(Vec3::ZERO, |a, v| a + v);
        let initial_com = pos.iter().copied().fold(Vec3::ZERO, |a, p| a + p) / pos.len() as f32;
        assert!(initial_momentum.length() < 1.0e-6);

        for _ in 0..120 {
            warp_tile_nbody_step(&mut pos, &mut vel, 0.4, 0.05, 1.0 / 120.0);
            assert!(pos.iter().all(|p| p.is_finite()));
            assert!(vel.iter().all(|v| v.is_finite()));
        }

        let final_momentum = vel.iter().copied().fold(Vec3::ZERO, |a, v| a + v);
        let final_com = pos.iter().copied().fold(Vec3::ZERO, |a, p| a + p) / pos.len() as f32;
        assert!(
            final_momentum.length() < 1.0e-5,
            "pairwise all-pairs gravity should conserve total momentum: {final_momentum:?}"
        );
        assert!(
            (final_com - initial_com).length() < 1.0e-5,
            "center of mass should stay fixed without external forces: initial={initial_com:?}, final={final_com:?}"
        );
        assert!(
            pos.iter().all(|p| (*p - initial_com).length() < 2.5),
            "softened N-body step should remain bounded for the reduced viewer case: {pos:?}"
        );
    }

    #[test]
    fn ik_custom_collision_projection_pushes_only_inside_margin() {
        let c = MVec3::new(0.30, 1.10, 0.05);
        let r_obs = 0.15;
        let r_link = 0.06;
        let margin = 0.04;
        let r_min = r_obs + r_link + margin;

        let clear = c + MVec3::X * (r_min + 0.20);
        let (clearance, target) = ik_custom_collision_projection(clear, c, r_obs, r_link, margin);
        assert!(clearance > margin);
        assert!(
            target.is_none(),
            "clear link should not emit a residual row"
        );

        let near = c + MVec3::X * (r_obs + r_link + margin * 0.25);
        let (clearance, target) = ik_custom_collision_projection(near, c, r_obs, r_link, margin);
        let target = target.expect("link inside safety margin should be projected outward");
        assert!(clearance > 0.0 && clearance < margin);
        assert!(((target - c).length() - r_min).abs() < 1.0e-12);
        assert!(target.x > near.x, "projection should push radially outward");

        let penetrating = c + MVec3::X * (r_obs + r_link - 0.02);
        let (clearance, target) =
            ik_custom_collision_projection(penetrating, c, r_obs, r_link, margin);
        let target = target.expect("penetrating link should be projected outward");
        assert!(clearance < 0.0);
        assert!(((target - c).length() - r_min).abs() < 1.0e-12);

        let (center_clearance, center_target) =
            ik_custom_collision_projection(c, c, r_obs, r_link, margin);
        let center_target = center_target.expect("degenerate center case needs a fallback normal");
        assert!(center_clearance < 0.0);
        assert!(((center_target - c).length() - r_min).abs() < 1.0e-12);
        assert!(center_target.y > c.y);
    }

    #[test]
    fn ik_cube_stacking_targets_are_reachable_and_audited() {
        let cubes = ik_cube_stacking_initial_centers();
        assert!(
            ik_cube_stacking_targets_are_reachable(&cubes),
            "reduced cube stacking waypoints should be inside the procedural arm workspace"
        );

        let audit = Scene::IkCubeStacking.physics_audit();
        assert_eq!(audit.status, PhysicsAuditStatus::Partial);
        assert!(
            audit.gpu_proof.contains("make test-articulation-gpu"),
            "cube stacking IK audit must name the existing GPU IK/articulation lane"
        );
        assert!(
            audit.notes.contains("full gripper contact"),
            "audit must stay honest about missing coupled manipulation contact"
        );
    }

    #[test]
    fn ik_cube_stacking_stack_is_supported_and_finite() {
        let mut cubes = ik_cube_stacking_initial_centers();
        for (level, cube) in cubes.iter_mut().enumerate() {
            *cube = ik_cube_stacking_place_center(level);
        }
        let err = ik_cube_stacking_stack_support_error(&cubes, IK_STACK_CUBE_COUNT);
        assert!(
            err < 1.0e-12,
            "perfect procedural stack should have zero support error: {err}"
        );
        cubes[2].y += 0.01;
        let err_bad = ik_cube_stacking_stack_support_error(&cubes, IK_STACK_CUBE_COUNT);
        assert!(
            err_bad >= 0.009,
            "support oracle should catch vertical gap errors: {err_bad}"
        );
    }

    #[test]
    fn ik_cube_stacking_reports_obstacle_residuals() {
        let st = IkCubeStackingState::default();
        let inside = st.obstacle_center + MVec3::X * (st.obstacle_radius * 0.5);
        let clear = st.obstacle_center + MVec3::X * (st.obstacle_radius + st.obstacle_margin + 0.2);
        let inside_res = ik_cube_stacking_obstacle_residual(
            inside,
            st.obstacle_center,
            st.obstacle_radius,
            st.obstacle_margin,
        );
        let clear_res = ik_cube_stacking_obstacle_residual(
            clear,
            st.obstacle_center,
            st.obstacle_radius,
            st.obstacle_margin,
        );
        assert!(
            inside_res > 0.0,
            "inside/marginal waypoint should report positive obstacle residual"
        );
        assert_eq!(clear_res, 0.0);
    }

    #[test]
    fn quadrotor_pd_mixer_balances_hover_and_attitude_torques() {
        let mass = QUAD_FRAME_MASS + 4.0 * QUAD_ROTOR_MASS;
        let hover = quadrotor_pd_thrusts(
            mass,
            QUAD_ARM,
            QUAD_INITIAL_HEIGHT,
            QUAD_INITIAL_HEIGHT,
            0.0,
            MVec3::Y,
            MVec3::ZERO,
            30.0,
            8.0,
            4.0,
            0.6,
        );
        let per_rotor = mass * molla::math::GRAVITY / 4.0;
        for t in hover {
            assert!(
                (t - per_rotor).abs() < 1.0e-12,
                "hover thrust mismatch: {t}"
            );
        }

        let low = quadrotor_pd_thrusts(
            mass,
            QUAD_ARM,
            QUAD_INITIAL_HEIGHT + 0.5,
            QUAD_INITIAL_HEIGHT,
            0.0,
            MVec3::Y,
            MVec3::ZERO,
            30.0,
            8.0,
            4.0,
            0.6,
        );
        assert!(low.iter().all(|t| *t > per_rotor));
        assert!((low.iter().sum::<f64>() - (mass * molla::math::GRAVITY + 15.0)).abs() < 1.0e-12);

        let tilted_up = MVec3::new(0.10, 0.99, -0.20).normalize();
        let mixed = quadrotor_pd_thrusts(
            mass,
            QUAD_ARM,
            QUAD_INITIAL_HEIGHT,
            QUAD_INITIAL_HEIGHT,
            0.0,
            tilted_up,
            MVec3::ZERO,
            30.0,
            8.0,
            4.0,
            0.6,
        );
        assert!(mixed.iter().all(|t| t.is_finite() && *t >= 0.0));
        assert!(
            (mixed.iter().sum::<f64>() - mass * molla::math::GRAVITY).abs() < 1.0e-12,
            "attitude-only correction should preserve total thrust before clamping"
        );
        assert!(
            (mixed[0] - mixed[1]).abs() > 1.0e-3,
            "pitch correction should split +X/-X rotor thrusts"
        );
        assert!(
            (mixed[2] - mixed[3]).abs() > 1.0e-3,
            "roll correction should split +Z/-Z rotor thrusts"
        );

        // The attitude loop must produce a *restoring* torque: the net
        // torque from the four +Y rotor forces has to point along the
        // tilt error `body_y × world_y` so the frame rotates back to
        // level. A sign error in the roll/pitch mixer makes this an
        // anti-restoring torque (positive feedback → tumble).
        let l = QUAD_ARM;
        // Rotor lever arms: 0=+X, 1=-X, 2=+Z, 3=-Z.
        let rotor_pos = [
            MVec3::new(l, 0.0, 0.0),
            MVec3::new(-l, 0.0, 0.0),
            MVec3::new(0.0, 0.0, l),
            MVec3::new(0.0, 0.0, -l),
        ];
        let mut torque = MVec3::ZERO;
        for (p, &f) in rotor_pos.iter().zip(mixed.iter()) {
            torque += p.cross(MVec3::new(0.0, f, 0.0));
        }
        let tilt = tilted_up.cross(MVec3::Y);
        assert!(
            torque.dot(tilt) > 1.0e-6,
            "attitude torque must oppose the tilt (restoring); got torque={torque:?} tilt={tilt:?}"
        );
        assert!(
            torque.x * tilt.x > 0.0 && torque.z * tilt.z > 0.0,
            "roll (X) and pitch (Z) torques must each be restoring; \
             torque={torque:?} tilt={tilt:?}"
        );
    }

    #[test]
    fn diffsim_drone_mpc_gradient_matches_finite_difference() {
        let state = DiffsimDroneState::default();
        let mut controls = state.controls.clone();
        for k in 0..DRONE_HORIZON {
            controls[k * 3] += 0.05 * ((k as f64) * 0.31).sin();
            controls[k * 3 + 1] = 0.12 * ((k as f64) * 0.17).cos();
            controls[k * 3 + 2] = -0.08 * ((k as f64) * 0.23).sin();
        }

        let (loss, grads) = drone_loss_and_gradient(state.p0, state.target, &controls);
        assert!(loss.is_finite());
        assert_eq!(grads.len(), controls.len());
        assert!(grads.iter().all(|g| g.is_finite()));

        for i in [0, 1, 2, 9, 37, controls.len() - 1] {
            let saved = controls[i];
            controls[i] = saved + DRONE_FD_EPS;
            let lp = drone_loss(state.p0, state.target, &controls);
            controls[i] = saved - DRONE_FD_EPS;
            let lm = drone_loss(state.p0, state.target, &controls);
            controls[i] = saved;
            let fd = (lp - lm) / (2.0 * DRONE_FD_EPS);
            assert!(
                (grads[i] - fd).abs() < 1.0e-7,
                "gradient mismatch at {i}: analytic={}, fd={}",
                grads[i],
                fd
            );
        }
    }

    #[test]
    fn diffsim_drone_mpc_adam_reduces_final_position_loss() {
        let state = DiffsimDroneState::default();
        let mut controls = state.controls.clone();
        let mut adam = molla::optim::AdamState::new(controls.len());
        let cfg = state.cfg;
        let initial_loss = drone_loss(state.p0, state.target, &controls);
        assert!(initial_loss.is_finite());
        assert!(initial_loss > 0.1);

        for _ in 0..DRONE_OPT_ITERS {
            let (_loss, grads) = drone_loss_and_gradient(state.p0, state.target, &controls);
            assert!(grads.iter().all(|g| g.is_finite()));
            adam.step(&mut controls, &grads, &cfg);
            assert!(controls.iter().all(|u| u.is_finite()));
        }

        let final_traj = drone_rollout(state.p0, &controls);
        assert_eq!(final_traj.len(), DRONE_HORIZON + 1);
        assert!(
            final_traj
                .iter()
                .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
        );

        let final_loss = drone_loss(state.p0, state.target, &controls);
        let final_error = (final_traj[DRONE_HORIZON] - state.target).length();
        assert!(
            final_loss < initial_loss * 0.02,
            "MPC did not materially reduce final position loss: initial={initial_loss}, final={final_loss}"
        );
        assert!(
            final_error < 0.15,
            "MPC did not reach target closely enough: final_error={final_error}"
        );
    }

    #[test]
    fn fluid_checkpoint_bounds_saved_windows() {
        let target = fluid_checkpoint_target(FLUID_CHECKPOINT_N, 0.85);
        let result = fluid_checkpoint_rollout(0.4, &target, FLUID_CHECKPOINT_EVERY);
        assert_eq!(result.final_field.len(), FLUID_CHECKPOINT_N);
        assert!(
            result.checkpoints.len() <= FLUID_CHECKPOINT_HORIZON / FLUID_CHECKPOINT_EVERY + 2,
            "checkpoint count should stay window-bounded"
        );
        assert_eq!(result.checkpoints.first().unwrap().step, 0);
        assert_eq!(
            result.checkpoints.last().unwrap().step,
            FLUID_CHECKPOINT_HORIZON
        );
        assert!(
            result
                .checkpoints
                .iter()
                .all(|c| c.field.len() == FLUID_CHECKPOINT_N)
        );
    }

    #[test]
    fn fluid_checkpoint_optimization_reduces_loss() {
        let target = fluid_checkpoint_target(FLUID_CHECKPOINT_N, 0.85);
        let initial = fluid_checkpoint_loss(0.05, &target);
        let (_control, final_loss) = fluid_checkpoint_optimize(0.05, &target, 40, 0.8);
        assert!(
            final_loss < initial * 0.95,
            "checkpointed optimization should reduce loss: initial={initial}, final={final_loss}"
        );

        let audit = Scene::FluidCheckpointOptimization.physics_audit();
        assert_eq!(audit.status, PhysicsAuditStatus::Partial);
        assert!(
            audit.gpu_proof.contains("make test-differentiable"),
            "fluid checkpoint audit must name differentiable/checkpoint verification"
        );
    }

    #[test]
    fn fluid_checkpoint_replay_is_deterministic() {
        let target = fluid_checkpoint_target(FLUID_CHECKPOINT_N, 0.65);
        let a = fluid_checkpoint_rollout(0.42, &target, FLUID_CHECKPOINT_EVERY);
        let b = fluid_checkpoint_rollout(0.42, &target, FLUID_CHECKPOINT_EVERY);
        assert_eq!(a.checkpoints.len(), b.checkpoints.len());
        assert!((a.loss - b.loss).abs() < 1.0e-15);
        for (pa, pb) in a.final_field.iter().zip(b.final_field.iter()) {
            assert!((pa - pb).abs() < 1.0e-15);
        }
    }

    #[test]
    fn ns_perturbation_parameter_changes_flow_response() {
        let r0 = ns_perturbation_response(&ns_perturbation_rollout(0.0, NS_PERT_N));
        let r1 = ns_perturbation_response(&ns_perturbation_rollout(1.0, NS_PERT_N));
        assert!(
            r1 > r0 + 1.0e-4,
            "positive perturbation should increase projected flow response: r0={r0}, r1={r1}"
        );
    }

    #[test]
    fn ns_perturbation_gradient_matches_finite_difference() {
        let amp = 0.45;
        let target = 0.20;
        let (_loss, grad, _response) = ns_perturbation_loss_and_gradient(amp, target);
        let eps = 1.0e-5;
        let lp = ns_perturbation_loss_and_gradient(amp + eps, target).0;
        let lm = ns_perturbation_loss_and_gradient(amp - eps, target).0;
        let fd = (lp - lm) / (2.0 * eps);
        assert!(
            (grad - fd).abs() < 1.0e-8,
            "NS perturbation gradient mismatch: analytic={grad}, fd={fd}"
        );
    }

    #[test]
    fn ns_perturbation_optimization_reduces_loss() {
        let target = 0.22;
        let initial = ns_perturbation_loss_and_gradient(0.0, target).0;
        let (_amp, final_loss, response) = ns_perturbation_optimize(0.0, target, 2.0, 60);
        assert!(response.is_finite());
        assert!(
            final_loss < initial * 0.98,
            "NS perturbation optimization should reduce loss: initial={initial}, final={final_loss}"
        );

        let audit = Scene::NavierStokesPerturbationOptimization.physics_audit();
        assert_eq!(audit.status, PhysicsAuditStatus::Partial);
        assert!(
            audit.gpu_proof.contains("make test-fem-ns-gpu"),
            "NS perturbation audit must name forward NS GPU lane"
        );
    }

    #[test]
    fn physics_audit_tracks_explicit_gpu_followup_scenes() {
        let manual_gpu = Scene::ALL
            .iter()
            .copied()
            .filter(|scene| scene.physics_audit().status == PhysicsAuditStatus::ManualGpu)
            .collect::<Vec<_>>();
        assert!(
            manual_gpu.contains(&Scene::Mpm),
            "MPM family must stay marked as explicit GPU follow-up"
        );
        assert!(
            manual_gpu.contains(&Scene::ImplicitMpmFull27),
            "implicit MPM must stay marked as explicit GPU follow-up"
        );
        for scene in manual_gpu {
            let audit = scene.physics_audit();
            assert!(
                audit.gpu_proof.contains("make test-"),
                "{}: manual-GPU audit must name a make target",
                scene.label()
            );
        }
    }

    #[test]
    fn mpm_multi_material_region_splits_sand_snow_viscous_bands() {
        assert_eq!(mpm_multi_material_region([-0.30, 0.4, 0.0]), 0);
        assert_eq!(mpm_multi_material_region([0.0, 0.4, 0.0]), 1);
        assert_eq!(mpm_multi_material_region([0.30, 0.4, 0.0]), 2);

        let audit = Scene::MpmMultiMaterial.physics_audit();
        assert_eq!(audit.status, PhysicsAuditStatus::ManualGpu);
        assert!(
            audit
                .cpu_proof
                .contains("reduced_multi_material_oracle_dispatches_distinct_branches"),
            "multi-material scene must name the branch-dispatch CPU oracle"
        );
        assert!(
            audit.notes.contains("per-particle material-id"),
            "audit must stay honest that runtime material ids are future work"
        );
    }

    #[test]
    fn mpm_beam_twist_initialization_has_bounded_torsion() {
        let left = [-0.45, 0.76, 0.04];
        let right = [0.45, 0.76, 0.04];
        let left_twisted = mpm_beam_twist_position(left);
        let right_twisted = mpm_beam_twist_position(right);
        for p in [left_twisted, right_twisted] {
            assert!(p.iter().all(|v| v.is_finite()));
            let radius = ((p[1] - 0.70).powi(2) + p[2].powi(2)).sqrt();
            let original_radius = ((left[1] - 0.70).powi(2) + left[2].powi(2)).sqrt();
            assert!(
                (radius - original_radius).abs() < 1.0e-6,
                "twist initialization should preserve cross-section radius"
            );
        }
        assert!(
            (left_twisted[1] - right_twisted[1]).abs() > 1.0e-3
                || (left_twisted[2] - right_twisted[2]).abs() > 1.0e-3,
            "opposite beam ends should receive different twist phases"
        );
    }

    #[test]
    fn mpm_beam_twist_color_tracks_angle() {
        let neg = mpm_beam_twist_angle([-0.5, 0.70, 0.0]);
        let mid = mpm_beam_twist_angle([0.0, 0.70, 0.0]);
        let pos = mpm_beam_twist_angle([0.5, 0.70, 0.0]);
        assert!(neg < mid && mid < pos);
        assert!(neg >= -std::f32::consts::PI && pos <= std::f32::consts::PI);

        let audit = Scene::MpmBeamTwist.physics_audit();
        assert_eq!(audit.status, PhysicsAuditStatus::Partial);
        assert!(
            audit.gpu_proof.contains("make test-mpm-gpu"),
            "MPM beam twist audit must name existing MPM GPU lanes"
        );
    }

    #[test]
    fn mpm_beam_twist_audit_is_partial() {
        let audit = Scene::MpmBeamTwist.physics_audit();
        assert!(
            audit.notes.contains("true torsional material response"),
            "audit must not claim full torsional beam parity"
        );
    }

    #[test]
    fn cable_bundle_hysteresis_scene_has_multiple_pinned_strands() {
        let (model, _visuals) = build_cable_bundle_hysteresis_model();
        assert_eq!(model.cable_count, 7);
        assert_eq!(model.particle_count, 7 * 20);
        let inv_mass = model.particle_inv_mass.host().expect("particle inv mass");
        let offsets = model.cable_particle_offsets.host().expect("cable offsets");
        let particles = model.cable_particles.host().expect("cable particles");
        for c in 0..model.cable_count {
            let start = offsets[c] as usize;
            let end = offsets[c + 1] as usize;
            assert_eq!(end - start, 20);
            let first = particles[start] as usize;
            let last = particles[end - 1] as usize;
            assert_eq!(inv_mass[first], 0.0, "strand {c} first endpoint pinned");
            assert_eq!(inv_mass[last], 0.0, "strand {c} last endpoint pinned");
        }
    }

    #[test]
    fn cable_bundle_hysteresis_loop_has_positive_area() {
        let cfg = molla::kernels::DahlConfig {
            sigma0: 1.0e3,
            ..molla::kernels::DahlConfig::default()
        };
        let normal = MVec3::Y;
        let dt = 1.0 / 1000.0;
        let mu = 0.45;
        let mut total_dissipated = 0.0;
        let mut delayed_reverse_count = 0;

        for strand in 0..7 {
            let f_n = 8.0 + strand as f64;
            let mut z = MVec3::ZERO;
            let mut f_t = MVec3::ZERO;
            for _ in 0..1600 {
                let v_t = MVec3::X * 0.25;
                let (z_new, f_new) =
                    molla::kernels::dahl_oracle_step(z, v_t, normal, f_n, mu, dt, cfg);
                total_dissipated += -(f_new.dot(v_t)) * dt;
                z = z_new;
                f_t = f_new;
            }
            assert!(
                f_t.x < 0.0,
                "forward slip should generate negative friction force: {f_t:?}"
            );

            let (z_after_first_reverse, f_first_reverse) =
                molla::kernels::dahl_oracle_step(z, -MVec3::X * 0.25, normal, f_n, mu, dt, cfg);
            if f_first_reverse.x < 0.0 {
                delayed_reverse_count += 1;
            }
            z = z_after_first_reverse;
            total_dissipated += -(f_first_reverse.dot(-MVec3::X * 0.25)) * dt;

            for _ in 0..1600 {
                let v_t = -MVec3::X * 0.25;
                let (z_new, f_new) =
                    molla::kernels::dahl_oracle_step(z, v_t, normal, f_n, mu, dt, cfg);
                total_dissipated += -(f_new.dot(v_t)) * dt;
                z = z_new;
                f_t = f_new;
            }
            assert!(
                f_t.x > 0.0,
                "sustained reverse slip should eventually flip friction force: {f_t:?}"
            );
            assert!(f_t.length() <= mu * f_n * 1.000_001);
        }

        assert_eq!(
            delayed_reverse_count, 7,
            "Dahl memory should delay the first force flip for every strand"
        );
        assert!(
            total_dissipated > 0.1,
            "bundle Dahl loop should dissipate positive work/area, got {total_dissipated}"
        );
    }

    #[test]
    fn cloth_rollers_scene_has_kinematic_rollers_and_cloth() {
        let (model, visuals) = build_cloth_rollers_model();
        assert!(visuals.is_empty());
        assert_eq!(model.body_count, CLOTH_ROLLER_COUNT);
        assert_eq!(
            model.particle_count,
            (CLOTH_ROLLERS_GRID_X * CLOTH_ROLLERS_GRID_Z) as usize
        );
        let flags = model.body_flags.host().expect("body flags");
        for i in 0..CLOTH_ROLLER_COUNT {
            assert!(
                (flags[i] & mp::BodyFlags::KINEMATIC) != 0,
                "roller {i} should be kinematic"
            );
        }
        let audit = Scene::ClothRollers.physics_audit();
        assert_eq!(audit.status, PhysicsAuditStatus::Partial);
        assert!(
            audit.gpu_proof.contains("make test-vbd-cloth-gpu"),
            "cloth rollers audit must name the forward cloth GPU lane"
        );
    }

    #[test]
    fn cloth_roller_surface_velocity_transfers_tangential_speed() {
        let omega = 4.0;
        let v = cloth_roller_surface_velocity(omega, CLOTH_ROLLER_RADIUS);
        assert!(
            (v.length() - omega * CLOTH_ROLLER_RADIUS).abs() < 1.0e-12,
            "surface velocity magnitude should equal ωr"
        );
        assert!(
            v.x < 0.0 && v.y == 0.0 && v.z == 0.0,
            "positive +Z roller spin should drive top contact along -X: {v:?}"
        );
        let reverse = cloth_roller_surface_velocity(-omega, CLOTH_ROLLER_RADIUS);
        assert!(
            reverse.x > 0.0,
            "reversing roller spin should reverse tangential velocity"
        );
    }

    #[test]
    fn cloth_rollers_support_gap_oracle_is_finite() {
        let gap =
            cloth_rollers_support_gap(CLOTH_ROLLERS_TOP_Y, CLOTH_ROLLER_Y, CLOTH_ROLLER_RADIUS);
        assert!(gap.is_finite() && gap > 0.0);
        assert!(
            gap < 0.40,
            "initial cloth should start close enough to drape onto rollers: gap={gap}"
        );
    }

    #[test]
    fn nut_bolt_sdf_scene_has_socket_bolt_and_thread_sdfs() {
        let (model, visuals) = build_nut_bolt_sdf_model();
        assert_eq!(model.body_count, NUT_BOLT_SOCKET_WALLS + 1);
        assert_eq!(visuals.len(), model.body_count);
        assert_eq!(model.sdfs.len(), NUT_BOLT_THREAD_RINGS);

        let boxes = model
            .shape_geometry
            .iter()
            .filter(|geom| matches!(geom, mp::ShapeGeometry::Box { .. }))
            .count();
        let cylinders = model
            .shape_geometry
            .iter()
            .filter(|geom| matches!(geom, mp::ShapeGeometry::Cylinder { .. }))
            .count();
        let sdfs = model
            .shape_geometry
            .iter()
            .filter(|geom| matches!(geom, mp::ShapeGeometry::Sdf { .. }))
            .count();
        let planes = model
            .shape_geometry
            .iter()
            .filter(|geom| matches!(geom, mp::ShapeGeometry::Plane { .. }))
            .count();

        assert_eq!(boxes, NUT_BOLT_SOCKET_WALLS);
        assert_eq!(cylinders, 1);
        assert_eq!(sdfs, NUT_BOLT_THREAD_RINGS);
        assert_eq!(planes, 1);
        assert!(model.contact_margin >= 0.02);

        let audit = Scene::NutBoltSdf.physics_audit();
        assert_eq!(audit.status, PhysicsAuditStatus::Partial);
        assert!(
            audit
                .cpu_proof
                .contains("procedural_socket_bolt_dense_contact_stays_finite_and_caged"),
            "nut/bolt SDF audit must name the dense-contact CPU oracle"
        );
        assert!(
            audit
                .notes
                .contains("full threaded SDF GPU contact remains future work"),
            "audit must stay honest about missing full GPU SDF contact"
        );
    }

    #[test]
    fn nut_bolt_sdf_thread_ring_has_expected_signs() {
        let sdf = mp::SdfShape::Torus {
            major: NUT_BOLT_THREAD_MAJOR,
            minor: NUT_BOLT_THREAD_MINOR,
        };

        let tube_center = MVec3::new(NUT_BOLT_THREAD_MAJOR, 0.0, 0.0);
        let outer_surface = MVec3::new(NUT_BOLT_THREAD_MAJOR + NUT_BOLT_THREAD_MINOR, 0.0, 0.0);
        let outside = MVec3::new(
            NUT_BOLT_THREAD_MAJOR + 2.0 * NUT_BOLT_THREAD_MINOR,
            0.0,
            0.0,
        );
        let bore_clearance = MVec3::new(0.0, 0.0, 0.0);

        assert!(
            sdf.signed_distance(tube_center) < -0.9 * NUT_BOLT_THREAD_MINOR,
            "thread tube center should be inside the torus SDF"
        );
        assert!(
            sdf.signed_distance(outer_surface).abs() < 1.0e-12,
            "outer thread surface should be on the SDF zero set"
        );
        assert!(
            sdf.signed_distance(outside) > 0.9 * NUT_BOLT_THREAD_MINOR,
            "outside point should have positive SDF distance"
        );
        assert!(
            sdf.signed_distance(bore_clearance) > NUT_BOLT_BOLT_RADIUS * 0.1,
            "center bore stays outside the thread material"
        );

        let n = sdf.normal(outer_surface);
        assert!(
            n.x > 0.99 && n.y.abs() < 1.0e-6 && n.z.abs() < 1.0e-6,
            "outer torus normal should point radially outward: {n:?}"
        );
    }

    fn hydro_unit_cube_mesh() -> mp::TriMesh {
        let vertices = vec![
            MVec3::new(-0.5, -0.5, -0.5),
            MVec3::new(0.5, -0.5, -0.5),
            MVec3::new(0.5, 0.5, -0.5),
            MVec3::new(-0.5, 0.5, -0.5),
            MVec3::new(-0.5, -0.5, 0.5),
            MVec3::new(0.5, -0.5, 0.5),
            MVec3::new(0.5, 0.5, 0.5),
            MVec3::new(-0.5, 0.5, 0.5),
        ];
        let indices = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 6, 2],
            [3, 7, 6],
            [0, 7, 3],
            [0, 4, 7],
            [1, 2, 6],
            [1, 6, 5],
        ];
        mp::TriMesh { vertices, indices }
    }

    #[test]
    fn nut_bolt_hydro_scene_reuses_procedural_contact_and_names_hydro_oracles() {
        let (model, visuals) = build_nut_bolt_hydro_model();
        assert_eq!(model.body_count, NUT_BOLT_SOCKET_WALLS + 1);
        assert_eq!(visuals.len(), model.body_count);
        assert_eq!(model.sdfs.len(), NUT_BOLT_THREAD_RINGS);

        let audit = Scene::NutBoltHydro.physics_audit();
        assert_eq!(audit.status, PhysicsAuditStatus::Partial);
        assert!(
            audit
                .cpu_proof
                .contains("plane_patch_hydroelastic_force_grows_with_penetration_depth"),
            "hydro scene must name the force-growth CPU oracle"
        );
        assert!(
            audit
                .cpu_proof
                .contains("hydroelastic_offset_patch_reduction_preserves_force_and_moment"),
            "hydro scene must name the force/moment reduction oracle"
        );
        assert!(
            audit.gpu_proof.contains("make test-hydroelastic-gpu"),
            "hydro scene must point at the dedicated GPU hydroelastic target"
        );
        assert!(
            audit
                .notes
                .contains("until hydroelastic forces are coupled"),
            "audit must stay honest that live solver coupling is not complete"
        );
    }

    #[test]
    fn nut_bolt_hydro_reduced_patch_force_grows_and_preserves_moment() {
        let sdf = molla::geometry::bake_mesh_to_sdf(&hydro_unit_cube_mesh(), 40, 0.05);
        let samples = molla::geometry::sample_plane_patch(0.18, 0.14, 9, 7);
        let cfg = molla::geometry::HydroelasticConfig {
            hydroelastic_modulus: 7.5e4,
            culling_distance: 0.0,
            ..molla::geometry::HydroelasticConfig::default()
        };

        let force_at_depth = |depth: f64| {
            let patch_pose = mp::Transform::from_translation(MVec3::new(0.18, 0.5 - depth, -0.12));
            let manifold = molla::geometry::compute_hydroelastic_contact(
                &sdf,
                &samples,
                mp::Transform::IDENTITY,
                patch_pose,
                &cfg,
            );
            assert!(
                !manifold.contacts.is_empty(),
                "expected hydroelastic contacts at depth {depth}"
            );
            let patch: Vec<molla::geometry::PatchSample> = manifold
                .contacts
                .iter()
                .map(|contact| molla::geometry::PatchSample {
                    point: contact.point,
                    normal: contact.normal,
                    force_magnitude: contact.normal_force(),
                })
                .collect();
            let force_ref: MVec3 = patch.iter().map(|sample| sample.force()).sum();
            let moment_ref: MVec3 = patch
                .iter()
                .map(|sample| sample.point.cross(sample.force()))
                .sum();
            let reduced = molla::geometry::reduce_with_anchor(&patch)
                .expect("hydro patch should reduce to anchor contact");
            let force_err =
                (reduced.net_force() - force_ref).length() / force_ref.length().max(1.0e-12);
            let moment_err =
                (reduced.net_moment() - moment_ref).length() / moment_ref.length().max(1.0e-12);
            assert!(
                force_err < 1.0e-9,
                "hydro reduced force mismatch: rel={force_err:e}"
            );
            assert!(
                moment_err < 0.01,
                "hydro reduced moment mismatch: rel={moment_err:e}"
            );
            manifold.total_normal_force
        };

        let shallow = force_at_depth(0.02);
        let deep = force_at_depth(0.06);
        assert!(
            deep > shallow * 2.0,
            "hydro force should grow with penetration: shallow={shallow}, deep={deep}"
        );
    }

    fn diffsim_cloth_reduced_height(control: f64) -> f64 {
        // Reduced one-parameter cloth oracle: a pinned-edge lift/control raises
        // the center of mass, with a small quadratic softening term that mimics
        // cloth strain saturation. It is intentionally tiny so finite
        // differences can pin the gradient contract without running a long
        // adjoint simulation inside the viewer unit suite.
        DIFFSIM_CLOTH_TOP_Y - 0.35 + 0.18 * control - 0.025 * control * control
    }

    fn diffsim_cloth_reduced_loss_and_gradient(control: f64, target_y: f64) -> (f64, f64) {
        let y = diffsim_cloth_reduced_height(control);
        let dy_dc = 0.18 - 0.05 * control;
        let err = y - target_y;
        let reg = 0.01 * control * control;
        let loss = 0.5 * err * err + reg;
        let grad = err * dy_dc + 0.02 * control;
        (loss, grad)
    }

    #[test]
    fn diffsim_cloth_scene_has_pinned_edge_and_partial_audit() {
        let (model, visuals) = build_diffsim_cloth_model();
        assert!(visuals.is_empty());
        assert_eq!(
            model.particle_count,
            (DIFFSIM_CLOTH_NX * DIFFSIM_CLOTH_NX) as usize
        );
        assert!(model.tri_count > 0, "diffsim cloth should render as a mesh");

        let inv_mass = model.particle_inv_mass.host().expect("particle inv mass");
        for iy in 0..DIFFSIM_CLOTH_NX as usize {
            let pinned = iy * DIFFSIM_CLOTH_NX as usize;
            assert_eq!(inv_mass[pinned], 0.0, "left edge particle {iy} pinned");
        }
        let free_corner = (DIFFSIM_CLOTH_NX * DIFFSIM_CLOTH_NX - 1) as usize;
        assert!(
            inv_mass[free_corner] > 0.0,
            "opposite corner should remain dynamic"
        );

        let audit = Scene::DiffsimCloth.physics_audit();
        assert_eq!(audit.status, PhysicsAuditStatus::Partial);
        assert!(
            audit.gpu_proof.contains("make test-vbd-cloth-gpu"),
            "diffsim cloth audit must name the forward GPU cloth target"
        );
        assert!(
            audit.notes.contains("full cloth adjoint"),
            "audit must stay honest about missing full cloth adjoint parity"
        );
    }

    #[test]
    fn diffsim_cloth_reduced_pin_gradient_matches_finite_difference() {
        let control = 0.65;
        let target_y = DIFFSIM_CLOTH_TOP_Y - 0.18;
        let (loss, grad) = diffsim_cloth_reduced_loss_and_gradient(control, target_y);
        assert!(loss.is_finite() && grad.is_finite());

        let eps = 1.0e-6;
        let (lp, _) = diffsim_cloth_reduced_loss_and_gradient(control + eps, target_y);
        let (lm, _) = diffsim_cloth_reduced_loss_and_gradient(control - eps, target_y);
        let fd = (lp - lm) / (2.0 * eps);
        assert!(
            (grad - fd).abs() < 1.0e-9,
            "diffsim cloth reduced gradient mismatch: analytic={grad}, fd={fd}"
        );
    }

    #[test]
    fn diffsim_cloth_reduced_optimization_reduces_loss() {
        let target_y = DIFFSIM_CLOTH_TOP_Y - 0.18;
        let mut control = 0.0;
        let (initial_loss, _) = diffsim_cloth_reduced_loss_and_gradient(control, target_y);
        assert!(initial_loss.is_finite() && initial_loss > 0.0);

        for _ in 0..80 {
            let (_, grad) = diffsim_cloth_reduced_loss_and_gradient(control, target_y);
            control -= 2.0 * grad;
            assert!(control.is_finite());
        }

        let (final_loss, _) = diffsim_cloth_reduced_loss_and_gradient(control, target_y);
        assert!(
            final_loss < initial_loss * 0.50,
            "diffsim cloth reduced optimization did not reduce loss enough: initial={initial_loss}, final={final_loss}, control={control}"
        );
    }

    fn diffsim_softbody_reduced_height(initial_velocity: f64, material_scale: f64) -> f64 {
        let material_lift = 0.06 * material_scale.max(1.0e-6).ln();
        DIFFSIM_SOFTBODY_TOP_Y + 0.30 * initial_velocity + material_lift - 0.42
    }

    fn diffsim_softbody_reduced_loss_and_gradient(
        initial_velocity: f64,
        material_scale: f64,
        target_y: f64,
    ) -> (f64, f64) {
        let y = diffsim_softbody_reduced_height(initial_velocity, material_scale);
        let err = y - target_y;
        let reg = 0.015 * initial_velocity * initial_velocity;
        let loss = 0.5 * err * err + reg;
        let grad = err * 0.30 + 0.03 * initial_velocity;
        (loss, grad)
    }

    #[test]
    fn diffsim_softbody_scene_has_tet_block_and_partial_audit() {
        let (model, visuals) = build_diffsim_softbody_model();
        assert!(visuals.is_empty());
        assert_eq!(
            model.particle_count,
            (DIFFSIM_SOFTBODY_GRID * DIFFSIM_SOFTBODY_GRID * DIFFSIM_SOFTBODY_GRID) as usize
        );
        assert_eq!(
            model.tet_count,
            ((DIFFSIM_SOFTBODY_GRID - 1).pow(3) * 6) as usize
        );
        assert!(
            model.tet_indices.host().expect("tet indices").len() >= model.tet_count * 4,
            "softbody should expose tet topology for wireframe gizmos"
        );

        let audit = Scene::DiffsimSoftBody.physics_audit();
        assert_eq!(audit.status, PhysicsAuditStatus::Partial);
        assert!(
            audit.gpu_proof.contains("make test-vbd-tets-gpu"),
            "diffsim softbody audit must name the forward GPU tet target"
        );
        assert!(
            audit.notes.contains("full tet adjoint"),
            "audit must stay honest about missing full tet adjoint parity"
        );
    }

    #[test]
    fn diffsim_softbody_reduced_velocity_gradient_matches_finite_difference() {
        let initial_velocity = 0.45;
        let material_scale = 1.8;
        let target_y = DIFFSIM_SOFTBODY_TOP_Y - 0.18;
        let (loss, grad) =
            diffsim_softbody_reduced_loss_and_gradient(initial_velocity, material_scale, target_y);
        assert!(loss.is_finite() && grad.is_finite());

        let eps = 1.0e-6;
        let (lp, _) = diffsim_softbody_reduced_loss_and_gradient(
            initial_velocity + eps,
            material_scale,
            target_y,
        );
        let (lm, _) = diffsim_softbody_reduced_loss_and_gradient(
            initial_velocity - eps,
            material_scale,
            target_y,
        );
        let fd = (lp - lm) / (2.0 * eps);
        assert!(
            (grad - fd).abs() < 1.0e-9,
            "diffsim softbody reduced gradient mismatch: analytic={grad}, fd={fd}"
        );
    }

    #[test]
    fn diffsim_softbody_reduced_optimization_reduces_loss() {
        let material_scale = 1.4;
        let target_y = DIFFSIM_SOFTBODY_TOP_Y - 0.15;
        let mut initial_velocity = 0.0;
        let (initial_loss, _) =
            diffsim_softbody_reduced_loss_and_gradient(initial_velocity, material_scale, target_y);
        assert!(initial_loss.is_finite() && initial_loss > 0.0);

        for _ in 0..80 {
            let (_, grad) = diffsim_softbody_reduced_loss_and_gradient(
                initial_velocity,
                material_scale,
                target_y,
            );
            initial_velocity -= 2.0 * grad;
            assert!(initial_velocity.is_finite());
        }

        let (final_loss, _) =
            diffsim_softbody_reduced_loss_and_gradient(initial_velocity, material_scale, target_y);
        assert!(
            final_loss < initial_loss * 0.35,
            "diffsim softbody reduced optimization did not reduce loss enough: initial={initial_loss}, final={final_loss}, initial_velocity={initial_velocity}"
        );
    }

    #[test]
    fn warp_fem_convdiff_dg_reduced_step_stays_bounded_and_monotone() {
        let nx = 24;
        let ny = 10;
        let mut u = vec![0.0; nx * ny];
        dg_convdiff_reset(&mut u, nx, ny);
        for _ in 0..240 {
            u = dg_convdiff_p0_step(&u, nx, ny, 0.01, 1.0, 5.0e-4, true);
        }
        let min_u = u.iter().copied().fold(f64::INFINITY, f64::min);
        let max_u = u.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            min_u >= -1.0e-12 && max_u <= 1.0 + 1.0e-12,
            "DG conv-diff limiter should keep solution bounded: min={min_u}, max={max_u}"
        );

        let left_avg = (0..ny).map(|j| u[dg_convdiff_idx(0, j, nx)]).sum::<f64>() / ny as f64;
        let right_avg = (0..ny)
            .map(|j| u[dg_convdiff_idx(nx - 1, j, nx)])
            .sum::<f64>()
            / ny as f64;
        assert!(
            left_avg > right_avg + 0.05,
            "positive-x advection with left inflow should keep left concentration higher: left={left_avg}, right={right_avg}"
        );
    }

    #[test]
    fn warp_fem_convdiff_dg_limiter_clamps_overshoot() {
        let nx = 4;
        let ny = 2;
        let mut u = vec![0.0; nx * ny];
        u[dg_convdiff_idx(1, 0, nx)] = 1.25;
        u[dg_convdiff_idx(2, 1, nx)] = -0.20;
        let limited = dg_convdiff_p0_step(&u, nx, ny, 0.0, 0.0, 0.0, true);
        assert!(limited.iter().all(|v| *v >= 0.0 && *v <= 1.0));
        assert_eq!(limited[dg_convdiff_idx(1, 0, nx)], 1.0);
        assert_eq!(limited[dg_convdiff_idx(2, 1, nx)], 0.0);

        let audit = Scene::WarpFemConvDiffDg.physics_audit();
        assert_eq!(audit.status, PhysicsAuditStatus::Partial);
        assert!(
            audit.gpu_proof.contains("make test-fem-dg-swe-gpu"),
            "DG conv-diff scene must name the reusable DG GPU lane"
        );
        assert!(
            audit.notes.contains("full DG convection-diffusion"),
            "audit must stay honest about missing full scalar DG conv-diff parity"
        );
    }

    #[test]
    fn warp_fem_mixed_elasticity_reduced_near_incompressible_stays_finite() {
        let (disp, mixed, pres) = mixed_elasticity_reduced_response(0.499, 1.0);
        assert!(disp.is_finite() && mixed.is_finite() && pres.is_finite());
        assert!(disp > 0.0 && mixed > 0.0);
        assert!(
            pres < 1.0e-2,
            "pressure residual should stay small near incompressible limit: {pres}"
        );

        let audit = Scene::WarpFemMixedElasticity.physics_audit();
        assert_eq!(audit.status, PhysicsAuditStatus::Partial);
        assert!(
            audit.gpu_proof.contains("make test-fem-elasticity-gpu"),
            "mixed elasticity audit must name the reusable elasticity GPU lane"
        );
    }

    #[test]
    fn warp_fem_mixed_elasticity_pressure_split_beats_locking() {
        let (disp_low, mixed_low, _) = mixed_elasticity_reduced_response(0.20, 1.0);
        let (disp_high, mixed_high, _) = mixed_elasticity_reduced_response(0.49, 1.0);

        assert!(
            disp_high < disp_low * 0.35,
            "displacement-only proxy should lock as ν→0.5: low={disp_low}, high={disp_high}"
        );
        assert!(
            mixed_high > mixed_low * 0.75,
            "mixed pressure split should preserve most of the deviatoric response: low={mixed_low}, high={mixed_high}"
        );
        assert!(
            mixed_high > disp_high * 4.0,
            "mixed response should avoid the severe locking shown by displacement-only response"
        );
    }

    #[test]
    fn warp_fem_nonconforming_contact_patch_preserves_force_and_moment() {
        let tri = [
            MVec3::new(-0.55, 0.0, -0.35),
            MVec3::new(0.65, 0.0, -0.20),
            MVec3::new(-0.15, 0.0, 0.55),
        ];
        let u = 0.32;
        let v = 0.28;
        let force = MVec3::new(0.15, 1.0, -0.10);
        let p = nonconforming_barycentric_point(tri, u, v);
        let node_forces = nonconforming_patch_transfer(tri, u, v, force);
        let (force_err, moment_err) = nonconforming_force_moment_errors(tri, p, force, node_forces);
        assert!(
            force_err < 1.0e-12,
            "nonconforming force transfer lost resultant: err={force_err}"
        );
        assert!(
            moment_err < 1.0e-12,
            "nonconforming force transfer lost moment: err={moment_err}"
        );
    }

    #[test]
    fn warp_fem_nonconforming_contact_scene_names_cpu_oracles() {
        let mut st = WarpFemNonconformingContactState::default();
        st.contact_u = 0.40;
        st.contact_v = 0.20;
        st.load = 1.4;
        st.user_reset = true;
        st.recompute();
        assert!(st.net_force_error < 1.0e-12);
        assert!(st.moment_error < 1.0e-12);

        let audit = Scene::WarpFemNonconformingContact.physics_audit();
        assert_eq!(audit.status, PhysicsAuditStatus::Partial);
        assert!(
            audit.cpu_proof.contains("fem_nonconforming"),
            "nonconforming contact audit must point to nonconforming CPU tests"
        );
        assert!(
            audit.cpu_proof.contains("fem_surface_contact"),
            "nonconforming contact audit must point to surface-contact transfer tests"
        );
        assert!(
            audit.gpu_proof.contains("CPU oracle only"),
            "nonconforming contact audit must not pretend GPU parity exists"
        );
    }

    #[test]
    fn warp_fem_stokes_transfer_preserves_constant_field() {
        let constant = Vec3::new(0.17, 0.0, -0.09);
        let grid = vec![constant; STOKES_TRANSFER_GRID_NX * STOKES_TRANSFER_GRID_NZ];
        for p in [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.25, 0.0, 0.75),
            Vec3::new(0.51, 0.0, 0.49),
            Vec3::new(1.0, 0.0, 1.0),
        ] {
            let v = stokes_transfer_sample_grid(
                &grid,
                STOKES_TRANSFER_GRID_NX,
                STOKES_TRANSFER_GRID_NZ,
                p,
            );
            assert!(
                (v - constant).length() < 1.0e-7,
                "constant field transfer should be exact at {p:?}: got {v:?}"
            );
        }
    }

    #[test]
    fn warp_fem_stokes_transfer_linear_rotation_is_bounded() {
        let grid =
            stokes_transfer_build_grid(STOKES_TRANSFER_GRID_NX, STOKES_TRANSFER_GRID_NZ, 1.1, 0.05);
        let mut max_speed = 0.0_f32;
        for j in 0..21 {
            for i in 0..21 {
                let p = Vec3::new(i as f32 / 20.0, 0.0, j as f32 / 20.0);
                let v = stokes_transfer_sample_grid(
                    &grid,
                    STOKES_TRANSFER_GRID_NX,
                    STOKES_TRANSFER_GRID_NZ,
                    p,
                );
                assert!(v.is_finite(), "sampled transfer velocity must stay finite");
                max_speed = max_speed.max(v.length());
            }
        }
        assert!(
            max_speed < 0.90,
            "linear rotational field should stay bounded on the unit square: max={max_speed}"
        );

        let mut p = Vec3::new(0.72, 0.0, 0.50);
        let center = Vec3::new(0.5, 0.0, 0.5);
        for _ in 0..600 {
            p = stokes_transfer_advect_particle(
                p,
                &grid,
                STOKES_TRANSFER_GRID_NX,
                STOKES_TRANSFER_GRID_NZ,
                0.005,
                false,
            );
            assert!(p.is_finite(), "advected tracer should stay finite");
        }
        assert!(
            (p - center).length() < 0.30,
            "Euler-advected rotational tracer should remain bounded around the center: p={p:?}"
        );

        let audit = Scene::WarpFemStokesTransfer.physics_audit();
        assert_eq!(audit.status, PhysicsAuditStatus::Partial);
        assert!(
            audit.gpu_proof.contains("make test-fem-apic-gpu"),
            "Stokes transfer audit must name the reusable APIC/transfer GPU lane"
        );
    }

    #[test]
    fn warp_fem_stokes_transfer_tracers_follow_velocity() {
        let grid =
            stokes_transfer_build_grid(STOKES_TRANSFER_GRID_NX, STOKES_TRANSFER_GRID_NZ, 0.8, 0.11);
        let p = Vec3::new(0.30, 0.0, 0.35);
        let dt = 0.0125;
        let v =
            stokes_transfer_sample_grid(&grid, STOKES_TRANSFER_GRID_NX, STOKES_TRANSFER_GRID_NZ, p);
        let next = stokes_transfer_advect_particle(
            p,
            &grid,
            STOKES_TRANSFER_GRID_NX,
            STOKES_TRANSFER_GRID_NZ,
            dt,
            false,
        );
        assert!(
            (next - (p + v * dt)).length() < 1.0e-7,
            "tracer Euler step must follow transferred velocity"
        );

        let mut st = WarpFemStokesTransferState::default();
        st.reset_tracers();
        assert_eq!(
            st.tracers.len(),
            STOKES_TRANSFER_TRACERS_X * STOKES_TRANSFER_TRACERS_Z
        );
        assert!(st.max_speed.is_finite() && st.max_speed > 0.0);
    }
}
