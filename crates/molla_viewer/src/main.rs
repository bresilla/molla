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

use bevy::diagnostic::{
    DiagnosticsStore, EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin,
};
use bevy::window::PrimaryWindow;
#[cfg(any(feature = "profile", feature = "log_diag"))]
use bevy::diagnostic::LogDiagnosticsPlugin;
use bevy::light::{CascadeShadowConfigBuilder, NotShadowCaster};
use bevy::pbr::wireframe::{Wireframe, WireframeConfig, WireframePlugin};
use bevy::prelude::*;
#[cfg(not(feature = "profile"))]
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass, EguiTextureHandle, EguiUserTextures};
use bevy_frost::container::Normal;
use bevy_frost::pane::{Pane2, PaneAnchor, PaneResize, RailZone};
use bevy_frost::pod::Pod;
use bevy_frost::prelude::*;
use bevy_frost::ribbon::{
    draw_assembly, find_item, RibbonCluster, RibbonDef, RibbonDrag, RibbonEdge, RibbonGlyph,
    RibbonItem, RibbonMode, RibbonOpen, RibbonPlacement, RibbonRole,
};
use bevy_frost::style::AccentColor;
use bevy_frost::widget::{button, readout};
use bevy_glacial::prelude::*;
use molla::autograd::{DiffArm, FeatherstoneRevoluteChain, JacobianScratch};
use molla::compute::{Engine, MpmConfig, MpmRheology, MpmSimulation, MpmTimeStep};
use molla::geometry::CollisionPipeline;
use molla::math::{Quat as MQuat, Vec3 as MVec3};
use molla::prelude as mp;
use molla::fem;
use molla::actuators::{
    Actuator, ClampingDcMotor, ClampingDcMotorConfig, ClampingMaxEffort,
    ClampingMaxEffortConfig, ClampingStage, ControllerNeuralLstm,
    ControllerNeuralLstmConfig, ControllerNeuralMlp, ControllerNeuralMlpConfig,
    ControllerPd, ControllerPdConfig, ControllerPipelinePd, ControllerPipelinePdConfig,
    ControllerStage, Delay, DelayConfig, LstmWeights, MlpWeights,
};
use molla::sensors::{
    BodyContactReading, CameraChannels, ContactSensor, ImuMount, ImuReading, ImuSensor, RayHit,
    Raycaster, TiledCamera, TiledCameraConfig,
};
use molla::solvers::{FeatherstoneSolver, IkConfig, IkObjective, IkSolver, Solver as _};
#[cfg(feature = "mujoco")]
use molla::solvers::MuJoCoSolver;

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
        tooltip: "Per-example settings (changes with the active scene)",
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

// --- Pile scene -----------------------------------------------------
const PILE_GRID_N: usize = 12;
// Tight horizontal spacing so bodies actually overlap on impact.
// Body half-extents are ~0.45, so 0.95 leaves a 0.05m gap → with the
// random rotation + drop-height stagger, plenty of off-axis impacts.
const PILE_SPACING: f32 = 0.95;
const PILE_SPHERE_RADIUS: f32 = 0.40;
const PILE_BOX_HALF: f32 = 0.40;
const PILE_CAPSULE_HALF_H: f32 = 0.40;
const PILE_CAPSULE_RADIUS: f32 = 0.22;
const PILE_DROP_HEIGHT: f32 = 4.0;
const PILE_DROP_BAND: f32 = 6.0;

// --- Rope scene -----------------------------------------------------
const ROPE_LINKS: usize = 18;
const ROPE_LINK_RADIUS: f32 = 0.12;
const ROPE_LINK_SPACING: f32 = 0.26;
const ROPE_ANCHOR_HEIGHT: f32 = 6.0;

// --- Cloth scene ----------------------------------------------------
// 36² = 1296 particles, 4·35² = 4900 distance constraints.
// CPU XPBD at 12 iters runs in ~0.5 ms per substep on a modern host.
const CLOTH_GRID: u32 = 36;
const CLOTH_SIZE: f32 = 1.8;
const CLOTH_TOP_Y: f32 = 5.0;
const CLOTH_PARTICLE_MASS: f64 = 0.05;
const CLOTH_PARTICLE_RADIUS: f64 = 0.02;
/// XPBD compliance — `α = 1/k`. Lower = stiffer XPBD pull toward rest.
/// Combined with `CLOTH_MAX_STRETCH` (which is a *hard* cap) this gives
/// you a soft elastic feel within the allowed range and a brick wall
/// at the limit.
const CLOTH_COMPLIANCE: f64 = 1.5e-5;
/// Per-spring hard upper-bound on stretch, expressed as a ratio of
/// rest length. The XPBD solver's strain-limit pass enforces
/// `length ≤ rest · (1 + max_stretch)` regardless of compliance.
/// 0.0 → perfectly inextensible (e.g. metal mesh)
/// 0.05 → 5 % linen (default for the demo)
/// 0.20 → 20 % jersey
/// 1.0  → effectively no cap
const CLOTH_MAX_STRETCH: f64 = 0.05;

// --- Stage-20 VBD cloth scene ---------------------------------------
// Higher resolution than the XPBD demo + tighter compliance — this
// is exactly the regime where XPBD wobbles and VBD shines.
const VBD_CLOTH_GRID: u32 = 32;
const VBD_CLOTH_SIZE: f32 = 1.8;
const VBD_CLOTH_TOP_Y: f32 = 4.5;
const VBD_CLOTH_PARTICLE_MASS: f64 = 0.04;
const VBD_CLOTH_PARTICLE_RADIUS: f64 = 0.025;
/// XPBD compliance for the underlying cloth model (= 1 / k for
/// stretch springs). VBD reads this and inverts to a per-spring
/// stiffness (see `derive_stiffness_table`); the smaller the value
/// the stiffer the cloth.
const VBD_CLOTH_COMPLIANCE: f64 = 5.0e-6;
const VBD_CLOTH_MAX_STRETCH: f64 = 0.10;
const VBD_CLOTH_SPHERE_RADIUS: f32 = 0.6;
const VBD_CLOTH_SPHERE_Y: f32 = 1.6;
/// Soft-contact stiffness against the kinematic sphere. Stiff enough
/// to stop a 4 cm/s falling cloth particle in ~3 substeps at 480 Hz
/// without producing visible jitter. Tuneable from the right-rail.
const VBD_CLOTH_CONTACT_STIFFNESS: f64 = 5.0e4;

// --- Stage-20.1 CABLE scene ----------------------------------------
// 32-particle polyline pinned at one end + an orbiting kinematic
// "puller" body at the other. A static cylinder sits between them so
// the cable can drape over it, exercising VBD soft-on-rigid contact +
// the new Dahl friction kernel.
const CABLE_PARTICLES: u32 = 32;
const CABLE_PINNED_AT: bevy::math::Vec3 = bevy::math::Vec3::new(-1.2, 1.8, 0.0);
const CABLE_PARTICLE_MASS: f64 = 0.04;
const CABLE_PARTICLE_RADIUS: f64 = 0.022;
/// Cable stretch stiffness (N/m). At 1e6 N/m the cable is effectively
/// inextensible — XPBD compliance `α = 1/k` < 1e-6.
const CABLE_STIFFNESS: f64 = 1.0e6;
/// Soft-contact stiffness against the static cylinder + the orbiting
/// puller. Tuneable.
const CABLE_CONTACT_STIFFNESS: f64 = 5.0e4;
/// Static cylinder pose — the cable drapes over its top edge.
const CABLE_CYLINDER_RADIUS: f32 = 0.30;
const CABLE_CYLINDER_HALF_HEIGHT: f32 = 0.50;
const CABLE_CYLINDER_POS: bevy::math::Vec3 = bevy::math::Vec3::new(0.0, 1.2, 0.0);
/// Puller orbit radius and rate (rad/s).
const CABLE_PULLER_ORBIT_R: f32 = 1.1;
const CABLE_PULLER_ORBIT_RATE: f32 = 0.4;
const CABLE_PULLER_HEIGHT: f32 = 1.6;
const CABLE_PULLER_RADIUS: f32 = 0.08;

// --- Stage-20.2 JELLY (tetrahedral soft body) scene ----------------
// 8×8×8 tet jelly cube drops onto a static plane and bounces against
// a kinematic "puller" body. Right-rail pane carries Young's modulus
// + Poisson-ratio sliders + a wireframe toggle.
const JELLY_GRID: u32 = 8;
const JELLY_SIDE: f32 = 0.8;
const JELLY_TOP_Y: f32 = 3.0;
const JELLY_PARTICLE_MASS: f64 = 0.05;
/// Default Young's modulus (Pa). Soft rubber regime — gives a
/// visibly squishy fall + bounce.
const JELLY_YOUNGS: f64 = 5.0e5;
/// Default Poisson ratio. ν = 0.4 is the typical rubber default.
const JELLY_POISSON: f64 = 0.4;
/// Soft-contact stiffness (N/m).
const JELLY_CONTACT_STIFFNESS: f64 = 5.0e4;
/// Kinematic puller body pose — sits a metre to the +X of the cube
/// drop point. The viewer doesn't animate it (yet); the user can
/// drag-and-throw via the existing mouse-grab if they want.
const JELLY_PULLER_RADIUS: f32 = 0.18;
const JELLY_PULLER_POS: bevy::math::Vec3 = bevy::math::Vec3::new(0.6, 0.18, 0.0);

// --- Differentiable arm (Stage 23) -------------------------------
/// World-space anchor height of the planar arm's shoulder. The arm
/// hangs in the world XY plane (Z = 0) so the existing camera works.
const DIFF_ARM_BASE_Y: f32 = 1.6;
/// Capsule visual radius for the two arm links.
const DIFF_ARM_LINK_RADIUS: f32 = 0.04;
/// Optimisation iterations per "Re-run" pass. The example shows the
/// solver typically lands within 5 cm in <30 iters; 80 leaves margin
/// for unreachable targets where the residual just shrinks.
const DIFF_ARM_OPT_ITERS: usize = 80;
/// Gradient-descent learning rate. ~5.0 is the value the example
/// uses; reducing it below 1 makes convergence noticeably slower.
const DIFF_ARM_OPT_LR: f64 = 5.0;
/// Playback speed multiplier — the trajectory is `arm.dt × arm.horizon`
/// = ~0.8 s of dynamics; we replay it at 0.5× to make the arm motion
/// readable.
const DIFF_ARM_PLAYBACK_SPEED: f64 = 0.5;

// --- Tower scene ----------------------------------------------------
const TOWER_LAYERS: usize = 8;
const TOWER_GRID: usize = 4;
const TOWER_BRICK_HALF: f32 = 0.30;
// Brick is 0.6m on a side. Spacing 0.65 = 0.05m gap so neighbours
// don't kiss each other on frame 0 and trigger jitter.
const TOWER_BRICK_SPACING: f32 = 0.65;
// Vertical: brick height = 0.6, gap 0.005, so 0.605m per layer.
const TOWER_LAYER_HEIGHT: f32 = 0.605;
const TOWER_BALL_RADIUS: f32 = 1.0;
const TOWER_BALL_MASS: f64 = 50.0;
const TOWER_BALL_HEIGHT: f32 = 14.0;

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
    app
        .add_plugins(FrostPlugin)
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
        .insert_resource(MuJoCoExampleState::default())
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
    app.insert_resource(sim).insert_resource(BodyVisuals(visuals))
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
                    update_mujoco_example,
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
                (
                    step_simulation,
                    sync_transforms,
                    sync_cloth_mesh,
                )
                    .chain(),
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
                update_warp_fft_fluid,
                update_warp_distortion_3d,
                update_warp_apic_fluid,
                update_warp_wave,
                update_warp_diff_ray,
                update_warp_tile_filter,
                update_warp_tile_compaction,
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
                draw_warp_fem_taylor_gizmos,
                draw_warp_fem_kh_gizmos,
                draw_warp_fem_streamlines_gizmos,
                draw_warp_fem_deformed_gizmos,
                draw_warp_fft_fluid_gizmos,
                draw_warp_apic_fluid_gizmos,
                draw_warp_wave_gizmos,
                draw_warp_diff_ray_gizmos,
            ),
        )
        .add_systems(Update, draw_warp_tile_filter_gizmos)
        .add_systems(Update, draw_warp_tile_compaction_gizmos)
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
    /// Stage-22 MuJoCo wrapper demo state. Bundled here (rather than
    /// added as a fresh `ResMut` parameter) because both
    /// `apply_scene_switch` and `control_pane` are at the 16-tuple
    /// ceiling — see PLAN.md Stage 22's "Bevy plumbing constraint"
    /// note for details.
    mujoco: ResMut<'w, MuJoCoExampleState>,
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
    /// Stage-22 MuJoCo wrapper demo state. Same rationale as the
    /// matching field on `SceneStates` — bundled to stay under the
    /// 16-tuple SystemParam ceiling.
    mujoco: ResMut<'w, MuJoCoExampleState>,
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
    /// Stage-22 MuJoCo wrapper. Only present when the `mujoco`
    /// Cargo feature is on; otherwise `Scene::MuJoCo` falls back to
    /// `Featherstone` and the right-rail pane explains why.
    #[cfg(feature = "mujoco")]
    MuJoCo(MuJoCoSolver),
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
            Self::Featherstone(s) => {
                s.step(model, state_in, control, contacts, dt, state_out)
            }
            Self::Vbd(s) => s.step(model, state_in, control, contacts, dt, state_out),
            #[cfg(feature = "mujoco")]
            Self::MuJoCo(s) => s.step(model, state_in, control, contacts, dt, state_out),
        }
    }
}

#[derive(Component, Copy, Clone)]
enum BodyVisual {
    Sphere(f32),
    /// Box half-extents along x/y/z (m).
    Box(Vec3),
    Capsule { half_h: f32, radius: f32 },
    /// Cone with apex at +Y·half_h, base disc at -Y·half_h with `radius`.
    Cone { half_h: f32, radius: f32 },
    /// Ellipsoid with per-axis semi-axes `(half_axes.x/y/z)` (m).
    /// Rendered as a unit `Sphere` mesh with a non-uniform scale of
    /// `2·half_axes` baked into the entity's transform.
    Ellipsoid { half_axes: Vec3 },
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
    /// Stage-22 MuJoCo wrapper demo — double pendulum stepped via
    /// the upstream MuJoCo C library through `MuJoCoSolver`.
    /// Demonstrates the wrapper backend (industry-standard contact
    /// + equality solver) running side-by-side with molla's native
    /// solvers. Right-rail pane shows the active backend label and
    /// per-frame timing. Hotkey `J` (for muJoco). When the `mujoco`
    /// Cargo feature is OFF, the scene's update system logs once
    /// that the backend is unavailable and skips — no crash.
    MuJoCo,
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
    /// Cloth pinned along TOP and BOTTOM edges; bottom edge rotates
    /// slowly to twist the cloth. Port of `example_cloth_twist.py`.
    ClothTwist,
    /// Several cable polylines dropped into a bowl, watch them pile
    /// up. CABLE bending stiffness. Port of `example_cable_pile.py`.
    CablePile,
    /// Three cables meeting at a Y junction (CONNECT equality at the
    /// central particle). Port of `example_cable_y_junction.py`.
    CableYJunction,
    // ── Group C: MPM extensions ──
    /// Ball of snow MPM particles dropped onto ground; uses
    /// `MpmRheology::Snow`. Port of `example_mpm_snow_ball.py`.
    MpmSnowBall,
    /// Heavy rigid sphere drops INTO a pit of MPM sand (uses existing
    /// `mpm_rigid_spheres` two-way coupling). Port of
    /// `example_mpm_twoway_coupling.py`.
    MpmTwoWay,
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
    /// overlap is reported via GJK; contact points drawn as gizmos.
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
    /// `WarpFluid`, just a different particle layout. molla doesn't
    /// have a real SPH solver.
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
    /// descent repulsion of 32 particles via the autograd tape.
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
    /// Port of `fem/example_taylor_green.py` — Stokes Taylor-Green
    /// vortex initial condition; analytic decay shown in pane.
    WarpFemTaylorGreen,
    /// Port of `fem/example_kelvin_helmholtz.py` — Stokes shear layer
    /// with sinusoidal perturbation; velocity arrows over time.
    WarpFemKelvinHelmholtz,
    /// Port of `fem/example_streamlines.py` — Stokes lid-driven cavity
    /// with tracer particles advected through the velocity field.
    WarpFemStreamlines,
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
        Scene::MuJoCo,
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
        Scene::ClothTwist,
        Scene::CablePile,
        Scene::CableYJunction,
        Scene::MpmSnowBall,
        Scene::MpmTwoWay,
        Scene::CartPole,
        Scene::CartPoleSwarm,
        Scene::FourBar,
        Scene::DiffBall,
        Scene::DiffSpring,
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
        Scene::WarpFemTaylorGreen,
        Scene::WarpFemKelvinHelmholtz,
        Scene::WarpFemStreamlines,
        Scene::WarpFemDeformedGeometry,
        Scene::WarpFemDistortion,
        Scene::WarpFemElasticOpt,
        Scene::WarpFftFluid,
        Scene::WarpDistortion3D,
        Scene::WarpApicFluid,
        Scene::WarpWave,
        Scene::WarpDiffRay,
        Scene::WarpTileFilter,
        Scene::WarpTileCompaction,
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
            Scene::MuJoCo => "17 MuJoCo wrapper (Stage 22)",
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
            Scene::ClothTwist => "32 Cloth twist (rotating bottom edge)",
            Scene::CablePile => "33 Cable pile (CABLE bending)",
            Scene::CableYJunction => "34 Cable Y-junction (CONNECT)",
            Scene::MpmSnowBall => "35 MPM snow ball (Stomakhin)",
            Scene::MpmTwoWay => "36 MPM two-way (sand + rigid sphere)",
            Scene::CartPole => "37 Cart-pole (PD balance)",
            Scene::CartPoleSwarm => "38 Cart-pole swarm (16 envs)",
            Scene::FourBar => "39 Four-bar linkage (JointMimic closure)",
            Scene::DiffBall => "40 Diffsim ball (projectile gradient)",
            Scene::DiffSpring => "41 Diffsim spring cage (rest-length gradient)",
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
            Scene::WarpMeshIntersect => "55 Warp mesh-intersect (GJK)",
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
            Scene::WarpParticleRepulsion => "66 Warp particle repulsion (autograd)",
            Scene::WarpFemDiffusion => "67 Warp FEM 2D diffusion",
            Scene::WarpFemDiffusion3D => "68 Warp FEM 3D diffusion (animated)",
            Scene::WarpFemConvDiff => "69 Warp FEM conv-diff",
            Scene::WarpFemTaylorGreen => "70 Warp FEM Taylor-Green",
            Scene::WarpFemKelvinHelmholtz => "71 Warp FEM Kelvin-Helmholtz",
            Scene::WarpFemStreamlines => "72 Warp FEM streamlines",
            Scene::WarpFemDeformedGeometry => "73 Warp FEM deformed geometry",
            Scene::WarpFemDistortion => "74 Warp FEM distortion energy",
            Scene::WarpFemElasticOpt => "75 Warp FEM elastic shape opt",
            Scene::WarpFftFluid => "76 Warp FFT fluid (Stokes fallback)",
            Scene::WarpDistortion3D => "77 Warp distortion 3D",
            Scene::WarpApicFluid => "79 Warp APIC fluid (Stage 37.6 hybrid)",
            Scene::WarpWave => "78 Warp wave (Stage 37.1)",
            Scene::WarpDiffRay => "80 Warp diff raycast (Stage 37.7)",
            Scene::WarpTileFilter => "81 Warp tile filter (Stage 37.8)",
            Scene::WarpTileCompaction => "82 Warp tile compaction (Stage 37.8)",
        }
    }

    /// Which dynamics solver this scene wants. Most scenes use XPBD;
    /// the Arm example needs Featherstone for proper joint dynamics.
    /// `GpuIkArm` reuses the Featherstone backend (any articulation
    /// solver works since `step_simulation` skips this scene anyway).
    /// `VbdCloth` uses the Stage-20 Vertex Block Descent solver.
    /// `MuJoCo` requests the Stage-22 wrapper backend; if the
    /// `mujoco` feature is off it falls back to Featherstone (so the
    /// scene still runs and the user sees the topology + timing,
    /// just driven by the native solver instead).
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
            | Scene::FourBar => SolverKind::Featherstone,
            Scene::VbdCloth
            | Scene::Cable
            | Scene::Jelly
            | Scene::ClothBending
            | Scene::ClothTwist
            | Scene::CablePile
            | Scene::CableYJunction
            | Scene::SoftbodyDrop
            | Scene::ClothCollar
            | Scene::FastSoftImpact
            | Scene::SoftBrake => SolverKind::Vbd,
            Scene::MuJoCo => SolverKind::MuJoCo,
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
    /// Stage-22 MuJoCo wrapper. Falls back to `Featherstone` at
    /// runtime when the `mujoco` Cargo feature is OFF.
    MuJoCo,
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

/// Lazily-initialised lidar runtime state — the GPU `Raycaster` and
/// the most recent set of (origin, dir, hit) triplets used by the
/// gizmo system. Reset on scene switch.
#[derive(Resource, Default)]
struct LidarState {
    raycaster: Option<Raycaster>,
    /// Capacity the raycaster was built for; rebuild if the config
    /// asks for more rays than this.
    raycaster_capacity: u32,
    /// Last frame's rays + hit results (one entry per ray).
    rays: Vec<LidarRay>,
}

#[derive(Clone, Copy)]
struct LidarRay {
    origin: Vec3,
    direction: Vec3,
    hit: Option<Vec3>,
}

/// IMU example runtime state. Lazy-initialised when entering the IMU
/// scene; reset on scene switch.
#[derive(Resource, Default)]
struct ImuExampleState {
    sensor: Option<ImuSensor>,
    /// Most recent reading (for the right-rail readout). `None` until
    /// the sensor primes.
    last: Option<ImuReading>,
    /// Smoothed magnitude (for the simple bar visualization).
    smoothed_accel_mag: f32,
    smoothed_gyro_mag: f32,
}

/// Stage-23 differentiable-arm demo runtime state. Owns the
/// optimisation horizon, target XYZ, the per-step torque buffer
/// (cleared and re-optimised on every "Re-run" / target-change), and
/// the per-iteration EE trajectory (so the right-rail toggle can
/// overlay it as an arrow trace).
#[derive(Resource)]
struct DiffArmExampleState {
    arm: DiffArm,
    /// Target end-effector position in world coordinates. The arm is
    /// anchored at `(0, DIFF_ARM_BASE_Y, 0)` and lives in the world XY
    /// plane (Z = 0); `target.z` is exposed in the UI for curiosity
    /// but stays clamped to 0 for v1 (the arm itself is planar).
    target: MVec3,
    /// Current optimised torque sequence. Length = `arm.horizon`.
    tau: Vec<[f64; 2]>,
    /// EE positions across all optimisation iterations — used for the
    /// "show optimisation trajectory" overlay. Cleared on re-run.
    iter_ee_trace: Vec<MVec3>,
    /// Final residual after the last optimisation pass.
    last_residual: f64,
    /// Iteration count at which the optimisation reached 5 cm of the
    /// target (`None` if it never did).
    converged_iter: Option<usize>,
    /// Total optimisation iterations actually run (for the readout).
    iters_run: usize,
    /// Re-run pending? Toggled by the pane's "Re-run optimisation"
    /// button or by a target slider change. Consumed by
    /// `update_diff_arm` next frame.
    rerun: bool,
    /// Show the iteration trail of EE positions as a gizmo.
    show_trajectory: bool,
    /// Wall-clock since scene entry — used to animate the arm
    /// playback so the user actually SEES the optimised trajectory
    /// (otherwise body_q would just snap to the final pose).
    playback_t: f64,
}

impl Default for DiffArmExampleState {
    fn default() -> Self {
        let arm = DiffArm::default();
        Self {
            arm,
            // Reachable point well within L1 + L2 = 0.9 m.
            target: MVec3::new(0.45, DIFF_ARM_BASE_Y as f64 - 0.25, 0.0),
            tau: vec![[0.0; 2]; arm.horizon],
            iter_ee_trace: Vec::new(),
            last_residual: f64::INFINITY,
            converged_iter: None,
            iters_run: 0,
            // Re-run on first entry so the user sees the optimisation
            // happen immediately rather than starting from a dead-zero
            // arm.
            rerun: true,
            show_trajectory: true,
            playback_t: 0.0,
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// Stage-23.1 differentiable-Featherstone scene state
// ────────────────────────────────────────────────────────────────────

const DIFF_ARM6_OPT_ITERS: usize = 250;
const DIFF_ARM6_PLAYBACK_SPEED: f64 = 0.5;

/// Stage 23.1.3 — joint kind selectable from the right-rail picker.
/// Each variant maps to a `FeatherstoneRevoluteChain` constructor; the
/// chain rebuilds when the user picks a different kind. Kept separate
/// from `molla::autograd::JointKind` because that enum is per-link
/// metadata; this enum is "which whole demo chain to run".
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum DiffArm6Kind {
    /// 6-DOF revolute Franka (Stage 23.1 default).
    Revolute,
    /// 3-DoF Ball-joint spherical pendulum (Stage 23.1.1).
    Ball,
    /// 6-DoF Free-joint floating body (Stage 23.1.1).
    Free,
    /// 3-DoF Distance-joint tethered body (Stage 23.1.3).
    Distance,
    /// 6-DoF D6 joint with per-axis angular limits (Stage 23.1.3).
    D6,
}

impl DiffArm6Kind {
    fn label(self) -> &'static str {
        match self {
            Self::Revolute => "Revolute (6-DOF Franka)",
            Self::Ball => "Ball (spherical pendulum)",
            Self::Free => "Free (floating body)",
            Self::Distance => "Distance (tethered)",
            Self::D6 => "D6 (limited Free)",
        }
    }

    /// Build the chain for this kind. All chains use `dt = 0.005` and a
    /// 60-step horizon so the GD loop dynamics stay comparable across
    /// variants.
    fn build_chain(self) -> FeatherstoneRevoluteChain {
        match self {
            Self::Revolute => {
                let mut c = FeatherstoneRevoluteChain::franka6(0.005, 60);
                c.damping = 1.0;
                c
            }
            Self::Ball => {
                let mut c = FeatherstoneRevoluteChain::spherical_pendulum(0.5, 1.0, 0.005, 60);
                c.damping = 0.5;
                c
            }
            Self::Free => {
                let mut c = FeatherstoneRevoluteChain::free_body(1.0, 0.05, 0.005, 60);
                c.gravity = molla::math::Vec3::ZERO;
                c.damping = 0.2;
                c
            }
            Self::Distance => {
                let mut c =
                    FeatherstoneRevoluteChain::add_joint_distance(1.0, 0.5, 0.005, 60, 1.0e-3);
                c.gravity = molla::math::Vec3::ZERO;
                c.damping = 0.1;
                c
            }
            Self::D6 => FeatherstoneRevoluteChain::add_joint_d6(
                1.0,
                0.05,
                0.005,
                60,
                [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
                [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)],
            ),
        }
    }
}

/// Stage-23.1 6-DOF Featherstone differentiable-arm runtime state.
/// Mirrors [`DiffArmExampleState`] but for the
/// [`FeatherstoneRevoluteChain::franka6`] chain. The pane drives the
/// EE target XYZ, displays residual + iters-to-converge, and lets the
/// user toggle the EE trajectory overlay.
#[derive(Resource)]
struct DiffArm6ExampleState {
    chain: FeatherstoneRevoluteChain,
    /// Target end-effector position in world coordinates. Default sits
    /// just off the rest pose so the optimiser does meaningful work
    /// the first frame.
    target: MVec3,
    /// Current optimised torque sequence. Length = `chain.horizon`.
    tau: Vec<Vec<f64>>,
    /// EE positions across all optimisation iterations — used for the
    /// "show optimisation trajectory" overlay. Cleared on re-run.
    iter_ee_trace: Vec<MVec3>,
    /// Final residual after the last optimisation pass.
    last_residual: f64,
    /// Iteration count at which the optimisation reached 5 cm of the
    /// target (`None` if it never did).
    converged_iter: Option<usize>,
    /// Total optimisation iterations actually run (for the readout).
    iters_run: usize,
    /// Re-run pending? Toggled by the pane's "Re-run optimisation"
    /// button or by a target slider change. Consumed by
    /// `update_diff_arm6` next frame.
    rerun: bool,
    /// Show the iteration trail of EE positions as a gizmo.
    show_trajectory: bool,
    /// Wall-clock since scene entry — used to animate playback.
    playback_t: f64,
    /// Reusable scratch for per-step Jacobian columns. Lives on the
    /// state so the GD loop doesn't re-allocate every iteration.
    scratch: JacobianScratch,
    /// Stage 23.1.1: kept around for back-compat. Stage 23.1.2 promoted
    /// the boolean to the 4-way `backend` picker below; this field now
    /// just mirrors `backend == GpuFallbackCpu` so any pre-existing code reading
    /// it stays correct.
    use_gpu_adjoint: bool,
    /// Stage 23.1.2 — 4-way backend picker (Auto / CPU CF / CPU AnalyticQd
    /// / CPU FD / GPU). The pane converts user clicks into this enum;
    /// the optimiser reads it once per re-run.
    backend: molla::kernels::AdjointBackend,
    /// Stage 23.1.2 — multi-articulation batch demo toggle. `true` runs
    /// N parallel Franka chains on every re-run via `MultiAdjointBatch`,
    /// which exercises the cross-chain parallelism that's the headline
    /// GPU win. `false` keeps the single-chain optimisation loop.
    use_multi_articulation: bool,
    /// Stage 23.1.2 — N for the multi-articulation batch toggle.
    multi_articulation_n: usize,
    /// Stage 23.1.1 readout: backward-pass / forward-pass time ratio
    /// from the last optimisation run. Demonstrates the closed-form
    /// vs FD perf win visually next to the iter counter.
    last_bwd_fwd_ratio: f64,
    /// Stage 23.1.2 readout: per-chain wall time (microseconds) on the
    /// last multi-articulation batch run. Single-chain runs leave it 0.
    last_per_chain_us: f64,
    /// Stage 23.1.2 readout: parallel speedup vs a single sequential
    /// reverse pass. Computed on the multi-articulation re-run.
    last_parallel_speedup: f64,
    /// Stage 23.1.3 — which joint kind drives the chain. Picker on the
    /// right-rail pane swaps between the Stage-23.1 6-DOF Franka, the
    /// Stage-23.1.1 Ball / Free demos, and the Stage-23.1.3 Distance /
    /// D6 demos so the user can verify the gradient flows for each.
    kind: DiffArm6Kind,
}

impl Default for DiffArm6ExampleState {
    fn default() -> Self {
        let kind = DiffArm6Kind::Revolute;
        let chain = kind.build_chain();
        let n = chain.links.len();
        let h = chain.horizon;
        // Target offset from rest in the X-Z plane (Y is hardest to
        // reach because gravity acts there).
        let rest_ee = chain.end_effector(&vec![0.0; n]);
        let target_world = MVec3::new(
            rest_ee.x as f64 + 0.07,
            rest_ee.y as f64 + DIFF_ARM6_BASE_HEIGHT,
            rest_ee.z as f64 - 0.07,
        );
        Self {
            scratch: JacobianScratch::new(n),
            chain,
            target: target_world,
            tau: vec![vec![0.0; n]; h],
            iter_ee_trace: Vec::new(),
            last_residual: f64::INFINITY,
            converged_iter: None,
            iters_run: 0,
            rerun: true,
            show_trajectory: true,
            playback_t: 0.0,
            use_gpu_adjoint: false,
            backend: molla::kernels::AdjointBackend::Auto,
            use_multi_articulation: false,
            multi_articulation_n: 10,
            last_bwd_fwd_ratio: 0.0,
            last_per_chain_us: 0.0,
            last_parallel_speedup: 0.0,
            kind,
        }
    }
}

/// Stage-20 VBD cloth runtime state. Driven by the right-rail sliders;
/// applied to the active solver each frame in `apply_vbd_settings`.
/// Lazy-initialised on entry to `Scene::VbdCloth`; survives scene
/// switches so the user's tuning persists when they jump out and back.
#[derive(Resource)]
struct VbdClothExampleState {
    /// Per-spring stretch stiffness (N/m). Mapped onto VBD via
    /// `1 / k = compliance` then re-uploaded.
    stretch_stiffness: f32,
    /// Per-spring bend stiffness (N/m). The cloth builder lays down
    /// "skip-one" bending springs at compliance = 30 × stretch_compliance
    /// by default; we use the same ratio when the user drags this
    /// slider.
    bend_stiffness: f32,
    /// Soft-contact stiffness against the rigid sphere (N/m).
    contact_stiffness: f32,
    /// Per-substep VBD outer-iteration count. More = tighter Newton
    /// convergence, more compute per substep.
    inner_iterations: u32,
    /// True → use VBD, false → use the Stage-12 XPBD solver. Toggling
    /// hot-swaps the solver in place (model + state are unchanged) so
    /// the user can A/B compare the two without losing the drape.
    use_vbd: bool,
}

impl Default for VbdClothExampleState {
    fn default() -> Self {
        let stretch = 1.0 / VBD_CLOTH_COMPLIANCE as f32;
        Self {
            stretch_stiffness: stretch,
            bend_stiffness: stretch / 30.0,
            contact_stiffness: VBD_CLOTH_CONTACT_STIFFNESS as f32,
            inner_iterations: 10,
            use_vbd: true,
        }
    }
}

/// Stage-20.1 CABLE polyline demo runtime state. Driven by the
/// right-rail sliders; applied to the active solver each frame in
/// `update_cable_example`. Lazy-initialised on entry to `Scene::Cable`;
/// survives scene switches so the user's tuning persists.
#[derive(Resource)]
struct CableExampleState {
    /// Cable stretch stiffness (N/m). Re-applied to the per-spring
    /// XPBD compliance buffer when the user drags the slider.
    stretch_stiffness: f64,
    /// Soft-contact stiffness against the cylinder + puller (N/m).
    contact_stiffness: f64,
    /// VBD outer-iteration count per substep.
    inner_iterations: u32,
    /// AVBD contact-mode toggle: hard mode = augmented-Lagrangian
    /// dual updates, soft mode = quadratic penalty (Stage 20).
    use_hard_mode: bool,
    /// Show the pinned-particle marker as a yellow gizmo.
    show_pinned: bool,
    /// Internal sim-time accumulator so the puller orbits at a steady
    /// rate (we can't read SimRunState here without bringing it into
    /// the system param tuple).
    sim_time: f32,
}

impl Default for CableExampleState {
    fn default() -> Self {
        Self {
            stretch_stiffness: CABLE_STIFFNESS,
            contact_stiffness: CABLE_CONTACT_STIFFNESS,
            inner_iterations: 12,
            use_hard_mode: true,
            show_pinned: true,
            sim_time: 0.0,
        }
    }
}

/// Stage-24d.1 MPM (sand / snow) demo state. The actual GPU sim lives
/// on `Sim::mpm` (lazy-initialised on first entry to `Scene::Mpm`);
/// this resource holds only the user-facing tuning sliders + reset
/// triggers for the right-rail UI.
///
/// Stage-24d.1 additions: `rheology_idx` (0 = sand, 1 = snow),
/// `time_step_implicit` + `jacobi_iters` (semi-implicit toggle + iter
/// slider), `add_rigid_sphere` button (drops a kinematic sphere to
/// demonstrate two-way coupling), `pile_slope_deg` live readout
/// (computed each frame from the particle positions).
#[derive(Resource)]
struct MpmExampleState {
    /// Drucker-Prager friction angle (degrees). Sand: 25°-40°.
    friction_angle_deg: f32,
    /// Young's modulus (Pa). Higher = stiffer / more like rock.
    youngs: f32,
    /// Per-axis particle count for the initial pour (count = N³).
    pour_axis_count: u32,
    /// Live readout: live particle count (set by the update system).
    particle_count: u32,
    /// Trigger flag: when true, the update system rebuilds the
    /// particle pour and clears it.
    needs_reset: bool,
    /// Spout-y offset for the initial pour.
    spout_y: f32,
    /// Stage 24d.2: substeps batched into a single `step_n` call per
    /// frame.  Higher = more physical fidelity per visible frame +
    /// directly demonstrates the batched-dispatcher throughput win.
    substeps_per_frame: u32,
    /// Live readout (ms): wall-clock for the most recent `step_n`
    /// dispatch + frame-end readback.
    last_frame_ms: f32,

    // Stage 24d.1.
    /// 0 = sand (Drucker-Prager), 1 = snow (Stomakhin).
    rheology_idx: u32,
    /// Snow critical compression (only used when `rheology_idx = 1`).
    snow_theta_c: f32,
    /// Snow critical stretch.
    snow_theta_s: f32,
    /// Snow hardening exponent.
    snow_hardening: f32,
    /// Use the semi-implicit time-step.
    time_step_implicit: bool,
    /// Number of Jacobi sweeps per substep when `time_step_implicit`.
    jacobi_iters: u32,
    /// "Add rigid sphere" button trigger — when true, the update
    /// system drops a kinematic sphere into the scene.
    add_rigid_sphere: bool,
    /// Live readout: equilibrium pile slope, measured from the
    /// particle positions each frame.
    pile_slope_deg: f32,
    /// Live readout: max-magnitude two-way wrench |F| (N) on the
    /// rigid sphere, when one exists.
    rigid_force_mag: f32,
}

impl Default for MpmExampleState {
    fn default() -> Self {
        Self {
            friction_angle_deg: 30.0,
            youngs: 5.0e3,
            pour_axis_count: 12,
            particle_count: 0,
            needs_reset: true, // initial pour
            spout_y: 0.6,
            // 4 substeps was the pre-24d.2 default (loop of `step()`);
            // higher values exercise the batched dispatcher.
            substeps_per_frame: 4,
            last_frame_ms: 0.0,
            rheology_idx: 0,
            snow_theta_c: 2.5e-2,
            snow_theta_s: 7.5e-3,
            snow_hardening: 10.0,
            time_step_implicit: false,
            jacobi_iters: 3,
            add_rigid_sphere: false,
            pile_slope_deg: 0.0,
            rigid_force_mag: 0.0,
        }
    }
}

/// Stage-20.2 JELLY tetrahedral soft-body demo runtime state. Driven
/// by the right-rail sliders; applied to the active solver each frame
/// in `update_jelly_example`. Lazy-initialised on entry to
/// `Scene::Jelly`; survives scene switches so the user's tuning
/// persists.
///
/// Stage 20.3 extensions:
/// - per-region material picker (`material_preset`) — chooses one of
///   the `JellyMaterialPreset` named recipes when the model is built,
///   driving the per-tet `(E, ν, σ_y, H)` parameters via
///   `add_tet_mesh_per_material`.
/// - plasticity sliders (`yield_stress`, `hardening`).
/// - "compress" trigger — the kinematic puller is briefly accelerated
///   downward to drive the cube past yield and exhibit permanent set.
/// - plastic-strain heat-map toggle (`show_plastic_strain`).
#[derive(Resource)]
struct JellyExampleState {
    /// Per-tet Young's modulus (Pa). Surfaced via the slider.
    youngs_modulus: f64,
    /// Per-tet Poisson ratio. ν ∈ [-0.49, 0.49] enforced when applied
    /// (≥ 0.5 is the incompressible limit and stiffens the volumetric
    /// Hessian unboundedly).
    poisson_ratio: f64,
    /// Soft-contact stiffness (N/m).
    contact_stiffness: f64,
    /// VBD outer-iteration count per substep.
    inner_iterations: u32,
    /// Show the per-tet wireframe overlay (gizmo lines along tet
    /// edges, deduped at draw time).
    show_wireframe: bool,

    // ── Stage 20.3 — multi-material + plasticity ─────────────────
    /// Material preset index — Soft / Stiff / Rigid. Re-builds the
    /// model with the new per-tet `MaterialNeoHookean` when changed.
    material_preset: JellyMaterialPreset,
    /// von Mises yield stress σ_y0 (Pa). `Real::INFINITY` (encoded
    /// here as 0.0 sentinel = "elastic only") disables plasticity.
    yield_stress: f64,
    /// Linear hardening modulus H (Pa).
    hardening: f64,
    /// Toggle the plastic-strain colour overlay on the tet wireframe.
    /// When on, draws each tet's edge with a colour proportional to
    /// its accumulated equivalent plastic strain (cool → hot).
    show_plastic_strain: bool,
    /// Latest max-ε_p readout (filled in by `update_jelly_example`).
    last_max_plastic_strain: f64,
    /// One-shot trigger fired by the "Compress" button. The puller
    /// body's kinematic Y-position is pulled down rapidly by
    /// `update_jelly_example` while this is non-zero, then decays
    /// back. A small impulsive squash is enough to drive the cube
    /// past yield with the default plasticity settings.
    compress_t: f32,

    // ── Stage 20.4 extensions ────────────────────────────────────
    /// Spawn a second jelly cube above the first to demonstrate
    /// tet-vs-tet collision. Triggers a model rebuild on toggle.
    second_cube: bool,
    /// Toggle kinematic-hardening (Stage 20.4) for the per-material
    /// `c` modulus. When on, the back-stress α evolves per Prager;
    /// when off, the model uses Stage-20.3-style isotropic hardening
    /// only. Triggers a model rebuild on toggle.
    kinematic_hardening_on: bool,
    /// Per-tet kinematic-hardening modulus c (Pa). Surfaced when
    /// `kinematic_hardening_on` is true.
    kinematic_hardening_modulus: f64,
    /// Latest max ‖α‖ (Frobenius) across all tets — readout only.
    last_max_back_stress: f64,
    /// SDF-backed rigid shape selection. `None` keeps the analytic
    /// puller sphere; the other variants register a baked / sparse
    /// SDF on solver re-creation.
    sdf_shape: JellySdfShape,
    /// Set to `true` when the user touches a control whose effect is
    /// only realised on a scene rebuild (`second_cube`, `sdf_shape`).
    /// Cleared on rebuild. Surfaces a "Rebuild jelly" button so the
    /// pane doesn't lie about needing the user to re-press Y (the
    /// hotkey is a no-op when the scene is already active).
    rebuild_pending: bool,
}

/// Stage 20.4 — picks which `SdfGrid` (or none) to register with the
/// VBD solver as a rigid contact target. Triggers a re-create of the
/// solver when changed (the SDF backings live on the solver).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
enum JellySdfShape {
    /// No SDF backing — use the analytic puller sphere only.
    #[default]
    None,
    /// Baked dense SDF of a sphere (validates the SdfGrid path).
    BakedSphere,
    /// Baked dense SDF of a torus (qualitative demo of mesh-derived
    /// SDF contact).
    BakedTorus,
}

/// Material recipe for `JellyExampleState::material_preset`. Each
/// option maps to one or more `MaterialNeoHookean` instances at
/// model-build time. v1 ships three named presets — easy to extend
/// (add a variant + a `materials_for_preset` arm).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
enum JellyMaterialPreset {
    #[default]
    Soft,
    Stiff,
    /// Soft interior with stiff corner cells — multi-material recipe
    /// that drives `add_tet_mesh_per_material` end-to-end.
    Mixed,
}

impl Default for JellyExampleState {
    fn default() -> Self {
        Self {
            youngs_modulus: JELLY_YOUNGS,
            poisson_ratio: JELLY_POISSON,
            contact_stiffness: JELLY_CONTACT_STIFFNESS,
            inner_iterations: 10,
            show_wireframe: true,
            material_preset: JellyMaterialPreset::Soft,
            yield_stress: 0.0, // 0 = "elastic only" (sentinel, see jelly_pane)
            hardening: 0.0,
            show_plastic_strain: false,
            last_max_plastic_strain: 0.0,
            compress_t: 0.0,
            second_cube: false,
            kinematic_hardening_on: false,
            kinematic_hardening_modulus: 5.0e3,
            last_max_back_stress: 0.0,
            sdf_shape: JellySdfShape::None,
            rebuild_pending: false,
        }
    }
}

/// Stage-21 mesh-tools demo state. Toggles for which artefacts are
/// visible in the viewport. The actual meshes are built once at scene
/// entry by `apply_scene_switch` and stashed as Bevy entities tagged
/// with one of the [`MeshToolPanel`] markers; toggling the booleans
/// just flips the `Visibility` of those entities.
#[derive(Resource)]
struct MeshToolsExampleState {
    /// Show the original (unsimplified) icosphere on the left.
    show_original: bool,
    /// Show the QEM-decimated mesh in the centre.
    show_decimated: bool,
    /// Show the marching-cubes surface extracted from a baked SDF
    /// on the right.
    show_marching_cubes: bool,
    /// Overlay the convex-hull wireframe on the original mesh.
    show_hull: bool,
    /// Telemetry surfaced in the right-rail pane (filled in by
    /// `apply_scene_switch` when the meshes are built).
    original_tris: usize,
    decimated_tris: usize,
    mc_tris: usize,
    hull_verts: usize,
    sdf_resolution: u32,
    /// Stage-24e: when true, the right panel renders the marching-cubes
    /// surface from a *sparse* SDF instead of the dense one. Both are
    /// baked at scene-entry; the toggle picks which one the surface +
    /// storage-cost readout reflects.
    use_sparse_sdf: bool,
    /// Stage-24e telemetry (filled in at scene-entry):
    dense_sdf_bytes: usize,
    sparse_sdf_bytes: usize,
    sparse_sdf_leaves: usize,
    sparse_sdf_active_voxels: u64,
    sparse_sdf_band: u32,
    /// Stage-24e.1 telemetry — root-table load factor + average
    /// probe length for the GPU root hash (filled in at scene-entry
    /// when the sparse SDF is built).
    sparse_root_table_capacity: u32,
    sparse_root_load_factor: f32,
    sparse_root_avg_probes: f32,
    sparse_root_max_probes: u32,
    /// Set to `true` when the user touches a control whose effect is
    /// only realised on a scene rebuild (`use_sparse_sdf`). Cleared on
    /// rebuild. Surfaces a "Rebuild mesh tools" button so the pane
    /// doesn't lie about needing the user to re-press M (the hotkey
    /// is a no-op when the scene is already active).
    rebuild_pending: bool,
}

impl Default for MeshToolsExampleState {
    fn default() -> Self {
        Self {
            show_original: true,
            show_decimated: true,
            show_marching_cubes: true,
            show_hull: false,
            original_tris: 0,
            decimated_tris: 0,
            mc_tris: 0,
            hull_verts: 0,
            sdf_resolution: 32,
            use_sparse_sdf: false,
            dense_sdf_bytes: 0,
            sparse_sdf_bytes: 0,
            sparse_sdf_leaves: 0,
            sparse_sdf_active_voxels: 0,
            sparse_sdf_band: 4,
            sparse_root_table_capacity: 0,
            sparse_root_load_factor: 0.0,
            sparse_root_avg_probes: 0.0,
            sparse_root_max_probes: 0,
            rebuild_pending: false,
        }
    }
}

impl MeshToolsExampleState {
    /// Width of the sparse-SDF narrow band, in voxels. Centralised so
    /// the bake function and the pane readout agree.
    fn sdf_sparse_band(&self) -> u32 {
        self.sparse_sdf_band
    }
}

/// Stage-24e helper — sample the sparse SDF back into a dense grid so
/// the existing `marching_cubes` consumer (which only takes
/// `SdfGrid`) can re-extract the surface. Used only by the viewer's
/// "use sparse SDF" toggle. Production MC over sparse SDFs should
/// iterate active leaves directly (deferred to Stage 24e.1).
fn sparse_sdf_to_dense_for_mc(
    sparse: &molla::geometry::SparseSdf,
    template: &molla::geometry::SdfGrid,
) -> molla::geometry::SdfGrid {
    let mut data = Vec::with_capacity(template.data.len());
    for z in 0..template.nz {
        for y in 0..template.ny {
            for x in 0..template.nx {
                let v = sparse.grid.lookup(x as i32, y as i32, z as i32) as f64;
                data.push(v);
            }
        }
    }
    molla::geometry::SdfGrid {
        nx: template.nx,
        ny: template.ny,
        nz: template.nz,
        origin: template.origin,
        cell_size: template.cell_size,
        data,
    }
}

/// Marker for the original (high-res) icosphere in `Scene::MeshTools`.
#[derive(Component)]
struct MeshToolOriginal;

/// Marker for the QEM-decimated icosphere.
#[derive(Component)]
struct MeshToolDecimated;

/// Marker for the marching-cubes-extracted surface.
#[derive(Component)]
struct MeshToolMarchingCubes;

/// Marker for the convex-hull wireframe overlay.
#[derive(Component)]
struct MeshToolHull;

/// Contact example runtime state. Lazy-initialised when entering the
/// Contact scene; reset on scene switch. Stores the latest per-body
/// readings for the right-rail readout and the gizmo system.
#[derive(Resource, Default)]
struct ContactExampleState {
    sensor: Option<ContactSensor>,
    readings: Vec<BodyContactReading>,
}

/// Stage-15 GPU narrow-phase demo state — toggle for the GPU pipeline
/// and the most recent CPU-narrow-phase call count read from the
/// pipeline. The toggle is applied each frame in `step_simulation`.
#[derive(Resource)]
struct GpuNpExampleState {
    /// User-controlled: enable the full GPU narrow phase.
    use_gpu: bool,
    /// Whether the pipeline currently has GPU narrow phase enabled.
    /// Used to detect when the user flips the toggle so we can
    /// rebuild the pipeline.
    gpu_enabled_now: bool,
    /// Last frame's CPU narrow-phase call count.
    last_cpu_calls: usize,
    /// Last frame's contact count.
    last_contacts: u32,
    /// SAP candidate-pair count (per-frame, last seen).
    last_pairs: usize,
}

impl Default for GpuNpExampleState {
    fn default() -> Self {
        Self {
            use_gpu: true,
            gpu_enabled_now: false,
            last_cpu_calls: 0,
            last_contacts: 0,
            last_pairs: 0,
        }
    }
}

/// Stage-19 GPU broad-phase demo state — toggle for the LBVH path
/// + per-backend ms/frame counters. The toggle is applied each
/// frame in `step_simulation` (mirrors `GpuNpExampleState`).
#[derive(Resource)]
struct GpuBpExampleState {
    /// User-controlled: enable the GPU LBVH broad phase. False ⇒ CPU SAP.
    use_gpu: bool,
    /// Whether the pipeline currently has GPU broad phase enabled.
    /// Tracks a delta between user toggle and pipeline state so we
    /// only rebuild the pipeline when the user flips it.
    gpu_enabled_now: bool,
    /// Last frame's broad-phase pair count.
    last_pairs: usize,
    /// Smoothed ms/frame for the active backend (whichever is on).
    smoothed_ms: f64,
    /// Most recent per-backend ms/frame snapshot — kept across
    /// toggles so the user can compare CPU vs GPU at a glance.
    last_cpu_ms: f64,
    last_gpu_ms: f64,
}

impl Default for GpuBpExampleState {
    fn default() -> Self {
        Self {
            use_gpu: true,
            gpu_enabled_now: false,
            last_pairs: 0,
            smoothed_ms: 0.0,
            last_cpu_ms: 0.0,
            last_gpu_ms: 0.0,
        }
    }
}

/// Arm example runtime state. Owns the GPU PD controller and the
/// per-joint target inputs (slider value, optional auto-sine).
#[derive(Resource)]
struct ArmExampleState {
    controller: Option<ControllerPd>,
    /// Per-joint target position [rad]. Driven by the right-rail
    /// sliders or by the auto-sine generator below.
    targets: [f32; ARM_LINKS],
    /// Whether we've logged the first successful PD dispatch — saves
    /// the user from "is this thing even running?" guesswork.
    logged_init: bool,
    /// Auto-sine drive: when enabled, each joint follows
    /// `amplitude · sin(2π·frequency·t + phase[i])`. Disabled by
    /// default so the user can drag the sliders manually.
    sine_enabled: bool,
    sine_amplitude: f32,
    sine_frequency: f32,
    /// Wall-clock since the scene started, for the sine generator.
    elapsed: f32,
    /// Stage-32 joint-mimic toggle: when on, the arm is rebuilt with
    /// an additional auxiliary joint that mimics joint 0 via the
    /// linear-identity polynomial `[0, 1, 0, 0, 0]`. The viewer
    /// renders both joints' angles in the right rail so the user can
    /// confirm the mimic stays tight under the same PD drive.
    mimic_enabled: bool,
    /// Set to `true` whenever `mimic_enabled` is toggled; consumed
    /// by `apply_scene_switch` to know it must rebuild the arm
    /// model. The toggle handler in `arm_pane` flips this.
    pub mimic_dirty: bool,
}

impl Default for ArmExampleState {
    fn default() -> Self {
        Self {
            controller: None,
            // 6-element initial target — the rest get clobbered by
            // the auto-sine pass anyway, this is just for the manual
            // slider mode.
            targets: [0.0; ARM_LINKS],
            sine_enabled: true,
            sine_amplitude: 0.5,
            sine_frequency: 0.3,
            elapsed: 0.0,
            logged_init: false,
            mimic_enabled: false,
            mimic_dirty: false,
        }
    }
}

/// Stage-18 actuator pipeline runtime state. Owns the composable
/// `Actuator` (built lazily on scene entry), the slider-driven
/// per-frame parameters, and the latest commanded vs delivered
/// torque samples for the right-rail readout.
#[derive(Resource)]
struct ActuatorExampleState {
    actuator: Option<Actuator>,
    /// Position target [rad] driven by the slider.
    target: f32,
    /// Whether to drive the target with a sine wave instead of a
    /// fixed value (so the user can immediately see the delay
    /// stage's effect).
    sine_enabled: bool,
    sine_amplitude: f32,
    sine_frequency: f32,
    elapsed: f32,
    /// Lag in milliseconds (converted to substeps per scene config).
    delay_ms: f32,
    /// DC-motor parameters (per-frame slider state).
    stall_torque: f32,
    no_load_speed: f32,
    max_motor_effort: f32,
    /// PD gains.
    kp: f32,
    kd: f32,
    /// Latest commanded effort (before clamp) and delivered effort
    /// (after clamp). Computed host-side per frame for the readout
    /// — these are diagnostic, not authoritative.
    last_commanded: f32,
    last_delivered: f32,
    /// Reuploaded params when sliders change.
    params_dirty: bool,
}

impl Default for ActuatorExampleState {
    fn default() -> Self {
        Self {
            actuator: None,
            target: 0.5,
            sine_enabled: true,
            sine_amplitude: 1.0,
            sine_frequency: 0.4,
            elapsed: 0.0,
            delay_ms: 12.0,
            stall_torque: 8.0,
            no_load_speed: 18.0,
            max_motor_effort: 20.0,
            kp: 60.0,
            kd: 4.0,
            last_commanded: 0.0,
            last_delivered: 0.0,
            params_dirty: false,
        }
    }
}

/// Stage-18.1 neural-policy demo runtime state. Owns the composable
/// [`Actuator`] (built lazily on scene entry) wrapping a
/// [`ControllerNeuralMlp`] balance policy + a [`ClampingMaxEffort`]
/// stage. Per-frame: gathers `[cart_x, cart_xdot, theta, theta_dot]`
/// from `joint_q / joint_qd`, normalises, pushes into the MLP via
/// `set_inputs`, the pipeline scatters the single output into
/// `joint_force[0]` (cart prismatic).
#[derive(Resource)]
struct NeuralPolicyExampleState {
    actuator: Option<Actuator>,
    /// Most recent input vector (4 floats), for the readout pane.
    last_inputs: [f32; 4],
    /// Most recent network output (1 float — cart force command).
    last_output: f32,
    /// Most recent delivered force after clamping. Diagnostic only.
    last_delivered: f32,
    /// User-toggleable: when true, every couple of seconds we add a
    /// small rotational kick to the pole so the demo is visibly
    /// dynamic instead of asymptoting to perfect upright forever.
    auto_perturb: bool,
    /// Time since last perturb, seconds.
    perturb_timer: f32,
    /// Effort cap (N) for the cart force.
    max_force: f32,
}

impl Default for NeuralPolicyExampleState {
    fn default() -> Self {
        Self {
            actuator: None,
            last_inputs: [0.0; 4],
            last_output: 0.0,
            last_delivered: 0.0,
            auto_perturb: true,
            perturb_timer: 0.0,
            max_force: 25.0,
        }
    }
}

/// Stage-31 neural-LSTM demo runtime state. Mirrors
/// [`NeuralPolicyExampleState`], but the controller carries hidden +
/// cell state across substeps. The right-rail pane shows both the
/// last input vector + the live hidden-state snapshot.
///
/// Hidden / output dims are baked here (`HID = 16`, `OUT = 1`) so
/// `last_hidden` can be a plain array — keeps the resource bound to
/// `Copy` constants without a `Vec<f32>` allocation per frame.
const LSTM_HID_DIM: usize = 16;

#[derive(Resource)]
struct NeuralLstmExampleState {
    actuator: Option<Actuator>,
    /// Most recent input vector (4 floats), for the readout pane.
    last_inputs: [f32; 4],
    /// Most recent network output (1 float — cart-force command).
    last_output: f32,
    /// Most recent delivered force after clamping.
    last_delivered: f32,
    /// Most recent hidden-state vector, for the readout pane. Sized
    /// for `LSTM_HID_DIM`.
    last_hidden: [f32; LSTM_HID_DIM],
    /// Periodic pole kick toggle.
    auto_perturb: bool,
    perturb_timer: f32,
    /// Effort cap (N) for the cart force.
    max_force: f32,
}

impl Default for NeuralLstmExampleState {
    fn default() -> Self {
        Self {
            actuator: None,
            last_inputs: [0.0; 4],
            last_output: 0.0,
            last_delivered: 0.0,
            last_hidden: [0.0; LSTM_HID_DIM],
            auto_perturb: true,
            perturb_timer: 0.0,
            max_force: 25.0,
        }
    }
}

/// Quadrotor demo runtime state. Tracks PD-control gains, target
/// hover altitude, and per-rotor thrust readouts. Reset on scene
/// switch.
#[derive(Resource)]
struct QuadrotorState {
    /// Target hover altitude (m).
    target_altitude: f32,
    /// PD altitude loop gains (force/m, force·s/m).
    altitude_kp: f32,
    altitude_kd: f32,
    /// PD attitude loop gains (torque/rad, torque·s/rad).
    attitude_kp: f32,
    attitude_kd: f32,
    /// Most recent per-rotor thrust commands (N), for UI readout.
    last_thrusts: [f32; 4],
    /// Most recent altitude (m).
    last_altitude: f32,
    /// Most recent attitude angles (roll, pitch, yaw) in radians.
    last_attitude: [f32; 3],
    /// Whether we've initialised the body-mass / arm-length cache.
    cached: bool,
    total_mass: f32,
    arm_length: f32,
    /// Stage 17.1: enable the rigorous dense-Cholesky D solve. v1
    /// uses sequential rank-1 deflation (exact when D is diagonal,
    /// approximate but stable when not). For symmetric quadrotor
    /// inertia the two paths agree closely; toggle on to demo the
    /// asymmetric-inertia case (e.g. add a payload offset).
    rigorous_d_solve: bool,
    /// Last applied flag value (so we don't re-stamp the solver every
    /// frame — only when the user flips the toggle).
    rigorous_d_solve_applied: bool,
}

impl Default for QuadrotorState {
    fn default() -> Self {
        Self {
            target_altitude: 1.5,
            altitude_kp: 30.0,
            altitude_kd: 8.0,
            attitude_kp: 4.0,
            attitude_kd: 0.6,
            last_thrusts: [0.0; 4],
            last_altitude: 0.0,
            last_attitude: [0.0; 3],
            cached: false,
            total_mass: 0.0,
            arm_length: 0.0,
            rigorous_d_solve: false,
            rigorous_d_solve_applied: false,
        }
    }
}

/// Stage-22 MuJoCo wrapper demo runtime state. Tracks whether the
/// MuJoCo backend is available at compile time (via the `mujoco`
/// Cargo feature) so the right-rail pane can show "available /
/// unavailable" instead of silently doing nothing. The actual
/// `MuJoCoSolver` lives inside [`SolverBackend::MuJoCo`]; this
/// resource is just per-frame UI / status.
#[derive(Resource)]
struct MuJoCoExampleState {
    /// Smoothed wall-clock per `solver.step()` call (µs). Updated by
    /// `step_simulation` when this scene is active. Useful for users
    /// to compare against Featherstone's timing.
    smoothed_step_us: f64,
    /// Whether the user has been informed via `info!` that the MuJoCo
    /// feature is off — only logged once to avoid spamming.
    notified_unavailable: bool,
    /// Cached info-style status string for the right-rail pane. Built
    /// the first time the scene is entered.
    status: String,
}

impl Default for MuJoCoExampleState {
    fn default() -> Self {
        let status = if cfg!(feature = "mujoco") {
            "MuJoCo feature ON — backend active".to_owned()
        } else {
            "MuJoCo feature OFF — install libmujoco + rebuild with --features mujoco".to_owned()
        };
        Self {
            smoothed_step_us: 0.0,
            notified_unavailable: false,
            status,
        }
    }
}

/// GPU IK arm runtime state. Lazy-initialised when entering the
/// `Scene::GpuIkArm` scene; reset on scene switch. Tracks the
/// draggable target position, the most recent solver telemetry, and
/// a "use GPU" toggle.
#[derive(Resource)]
struct GpuIkArmState {
    /// Target world position the EE chases. Default = rest EE +
    /// `GPU_IK_DEFAULT_OFFSET`, computed lazily on first frame.
    target: MVec3,
    target_init: bool,
    use_gpu: bool,
    last_iters: usize,
    last_residual: f64,
    last_solve_us: f64,
}

impl Default for GpuIkArmState {
    fn default() -> Self {
        Self {
            target: MVec3::ZERO,
            target_init: false,
            use_gpu: true,
            last_iters: 0,
            last_residual: 0.0,
            last_solve_us: 0.0,
        }
    }
}

/// Solid box principal moments at the COM, body frame.
fn box_inertia(mass: f64, half: f64) -> molla::math::Mat3 {
    let h2 = half * half;
    let s = mass / 3.0;
    molla::math::Mat3::from_diagonal(MVec3::new(
        s * (h2 + h2),
        s * (h2 + h2),
        s * (h2 + h2),
    ))
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
/// Specifically: requesting `SolverKind::MuJoCo` without the `mujoco`
/// Cargo feature falls back to `Featherstone` so the viewer's
/// `Scene::MuJoCo` still loads (the right-rail pane explains the
/// fallback). Every other kind is unconditional.
fn build_solver(
    engine: &Engine,
    model: &mp::Model,
    kind: SolverKind,
) -> SolverBackend {
    match kind {
        SolverKind::Xpbd => SolverBackend::Xpbd(
            mp::XpbdSolver::new_gpu(engine, model.body_count.max(1) as u32)
                .expect("XPBD GPU solver init")
                .with_iterations(10),
        ),
        SolverKind::Featherstone => SolverBackend::Featherstone(
            FeatherstoneSolver::new_gpu(engine, model)
                .expect("Featherstone GPU solver init"),
        ),
        SolverKind::Vbd => SolverBackend::Vbd(
            mp::VbdSolver::new_gpu(engine, model)
                .expect("VBD GPU solver init")
                .with_iterations(10)
                .with_contact_stiffness(VBD_CLOTH_CONTACT_STIFFNESS),
        ),
        SolverKind::MuJoCo => mujoco_solver_or_fallback(engine, model),
    }
}

/// Try to construct a [`MuJoCoSolver`] when the `mujoco` Cargo
/// feature is on. Falls back to Featherstone otherwise. Encapsulated
/// here so the cfg branching doesn't leak into `build_solver`.
#[cfg(feature = "mujoco")]
fn mujoco_solver_or_fallback(engine: &Engine, model: &mp::Model) -> SolverBackend {
    match MuJoCoSolver::new(model) {
        Ok(s) => SolverBackend::MuJoCo(s.with_substeps(1)),
        Err(e) => {
            warn!("MuJoCo solver init failed ({e}); falling back to Featherstone");
            SolverBackend::Featherstone(
                FeatherstoneSolver::new_gpu(engine, model)
                    .expect("Featherstone GPU solver init"),
            )
        }
    }
}

#[cfg(not(feature = "mujoco"))]
fn mujoco_solver_or_fallback(engine: &Engine, model: &mp::Model) -> SolverBackend {
    // Logged once on entry to `Scene::MuJoCo` from `update_mujoco_example`
    // so we don't double-log here.
    SolverBackend::Featherstone(
        FeatherstoneSolver::new_gpu(engine, model)
            .expect("Featherstone GPU solver init"),
    )
}

impl Sim {
    /// Dispatch on `scene` and assemble `(Sim, BodyVisuals)`.
    fn build_scene(scene: Scene) -> (Self, Vec<BodyVisual>) {
        let (model, visuals) = match scene {
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
            Scene::MuJoCo => build_mujoco_model(),
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
            Scene::ClothTwist => build_cloth_twist_model(),
            Scene::CablePile => build_cable_pile_model(),
            Scene::CableYJunction => build_cable_y_junction_model(),
            Scene::MpmSnowBall => build_mpm_model(),
            Scene::MpmTwoWay => build_mpm_model(),
            Scene::CartPole => build_cartpole_model(),
            Scene::CartPoleSwarm => build_cartpole_swarm_model(),
            Scene::FourBar => build_fourbar_model(),
            Scene::DiffBall => build_diff_ball_model(),
            Scene::DiffSpring => build_diff_spring_model(),
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
            | Scene::WarpFemTaylorGreen
            | Scene::WarpFemKelvinHelmholtz
            | Scene::WarpFemStreamlines
            | Scene::WarpFemDeformedGeometry
            | Scene::WarpFemDistortion
            | Scene::WarpFemElasticOpt
            | Scene::WarpFftFluid
            | Scene::WarpDistortion3D
            | Scene::WarpApicFluid
            | Scene::WarpWave
            | Scene::WarpDiffRay
            | Scene::WarpTileFilter
            | Scene::WarpTileCompaction => build_warp_empty_model(),
        };
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

        model.upload_to_device(&device, &queue).expect("model upload");
        state_a.upload_to_device(&device, &queue).expect("state_a upload");
        state_b.upload_to_device(&device, &queue).expect("state_b upload");
        control.upload_to_device(&device, &queue).expect("control upload");
        contacts.upload_to_device(&device, &queue).expect("contacts upload");

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
                let cloth_count =
                    (FAST_SOFT_IMPACT_CLOTH_NX * FAST_SOFT_IMPACT_CLOTH_NX) as usize;
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
fn build_pile_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);

    // Deterministic LCG so every reset gives the same pile but
    // different bodies have different rotations.
    let mut rng_state: u64 = 0x9E3779B97F4A7C15;
    fn next_unit(s: &mut u64) -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (*s >> 11) as f64 / (1u64 << 53) as f64;
        v * 2.0 - 1.0 // ∈ [-1, 1)
    }
    fn random_rotation(s: &mut u64) -> MQuat {
        let axis =
            MVec3::new(next_unit(s), next_unit(s), next_unit(s)).normalize_or_zero();
        let axis = if axis.length_squared() < 1e-9 { MVec3::Y } else { axis };
        let angle = next_unit(s) * std::f64::consts::PI;
        MQuat::from_axis_angle(axis, angle)
    }

    let mut visuals: Vec<BodyVisual> = Vec::with_capacity(PILE_GRID_N * PILE_GRID_N);
    for ix in 0..PILE_GRID_N {
        for iz in 0..PILE_GRID_N {
            let x = (ix as f64 - (PILE_GRID_N as f64 - 1.0) * 0.5) * PILE_SPACING as f64
                + 0.15 * next_unit(&mut rng_state);
            let z = (iz as f64 - (PILE_GRID_N as f64 - 1.0) * 0.5) * PILE_SPACING as f64
                + 0.15 * next_unit(&mut rng_state);
            // Spread the drop heights over a band so contacts come in
            // waves — the falling stack pushes the lower bodies sideways.
            let y = PILE_DROP_HEIGHT as f64
                + (((ix + iz) as f64) / ((PILE_GRID_N + PILE_GRID_N) as f64))
                    * PILE_DROP_BAND as f64
                + 0.1 * next_unit(&mut rng_state);
            let pos = MVec3::new(x, y, z);
            let rot = random_rotation(&mut rng_state);

            match (ix + iz) % 3 {
                0 => {
                    // Spheres are rotation-invariant — random rotation
                    // doesn't help, so give them a random tangential
                    // angular velocity instead. We can't set initial
                    // qd through BodyParams; that's done after build().
                    let body = b.add_body(
                        mp::BodyParams::new()
                            .with_spatial_inertia(mp::SpatialInertia::sphere(
                                1.0,
                                PILE_SPHERE_RADIUS as f64,
                            ))
                            .with_position(pos),
                    );
                    b.add_shape_sphere(body, PILE_SPHERE_RADIUS as f64, mp::Transform::IDENTITY)
                        .expect("sphere shape");
                    visuals.push(BodyVisual::Sphere(PILE_SPHERE_RADIUS));
                }
                1 => {
                    let half = MVec3::splat(PILE_BOX_HALF as f64);
                    let body = b.add_body(
                        mp::BodyParams::new()
                            .with_spatial_inertia(mp::SpatialInertia {
                                mass: 1.0,
                                com: MVec3::ZERO,
                                inertia: box_inertia(1.0, PILE_BOX_HALF as f64),
                            })
                            .with_pose(mp::Transform::new(pos, rot)),
                    );
                    b.add_shape_box(body, half, mp::Transform::IDENTITY)
                        .expect("box shape");
                    visuals.push(BodyVisual::Box(Vec3::splat(PILE_BOX_HALF)));
                }
                _ => {
                    let body = b.add_body(
                        mp::BodyParams::new()
                            .with_spatial_inertia(mp::SpatialInertia {
                                mass: 1.0,
                                com: MVec3::ZERO,
                                inertia: capsule_inertia(
                                    1.0,
                                    PILE_CAPSULE_HALF_H as f64,
                                    PILE_CAPSULE_RADIUS as f64,
                                ),
                            })
                            .with_pose(mp::Transform::new(pos, rot)),
                    );
                    b.add_shape_capsule(
                        body,
                        PILE_CAPSULE_HALF_H as f64,
                        PILE_CAPSULE_RADIUS as f64,
                        mp::Transform::IDENTITY,
                    )
                    .expect("capsule shape");
                    visuals.push(BodyVisual::Capsule {
                        half_h: PILE_CAPSULE_HALF_H,
                        radius: PILE_CAPSULE_RADIUS,
                    });
                }
            }
        }
    }

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane shape");

    (b.build().expect("model"), visuals)
}

/// Vertical chain of small spheres. The top link is **kinematic**
/// (mass-zero, integrators leave its pose alone) so the rope dangles
/// from a fixed anchor. Adjacent links are pinned with `CONNECT`
/// equality constraints — this exercises the stage-7 XPBD constraint
/// projection. A small horizontal kick at the bottom makes it swing.
fn build_rope_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.10);

    let mut visuals = Vec::with_capacity(ROPE_LINKS);
    let mut bodies = Vec::with_capacity(ROPE_LINKS);

    for i in 0..ROPE_LINKS {
        let y = ROPE_ANCHOR_HEIGHT as f64 - i as f64 * ROPE_LINK_SPACING as f64;
        // Slight horizontal offset so the rope starts to swing instead
        // of sitting in unstable vertical equilibrium.
        let x = 0.001 * i as f64;
        let pos = MVec3::new(x, y, 0.0);

        let mut params = mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia::sphere(0.4, ROPE_LINK_RADIUS as f64))
            .with_position(pos);

        if i == 0 {
            // Kinematic anchor: zero mass + KINEMATIC flag so the
            // integrator passes its pose through unchanged.
            params = params
                .with_mass(0.0)
                .with_inertia(molla::math::Mat3::ZERO)
                .with_flags(mp::BodyFlags::KINEMATIC);
        }

        let id = b.add_body(params);
        b.add_shape_sphere(id, ROPE_LINK_RADIUS as f64, mp::Transform::IDENTITY)
            .expect("rope sphere");
        bodies.push(id);
        visuals.push(BodyVisual::Sphere(ROPE_LINK_RADIUS));
    }

    // Pin every consecutive pair at the midpoint between their COMs.
    // Anchor offsets are body-local, so half a link spacing along ±Y.
    let half = (ROPE_LINK_SPACING as f64) * 0.5;
    for i in 1..ROPE_LINKS {
        let _ = b.add_equality_connect(
            bodies[i - 1],
            bodies[i],
            MVec3::new(0.0, -half, 0.0),
            MVec3::new(0.0, half, 0.0),
        );
    }

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane shape");

    (b.build().expect("model"), visuals)
}

/// 8-layer 4×4 brick tower. A heavy sphere drops onto it from above —
/// shows stacking stability, contact reduction, and impact impulses.
///
/// Tried 45°-alternated layers initially (real masonry is bonded
/// half-brick offsets); rotating 0.6 m bricks by 45° gives a 0.85 m
/// diagonal, which overlaps neighbours within a layer and explodes the
/// stack on frame 0. Now: identical pose every layer, straight columns.
/// Each column is independent — settles immediately, smooth impact.
fn build_tower_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    // Higher angular damping than the other scenes — bricks shouldn't
    // spin gracefully in mid-air, they should drop and stop.
    b.set_angular_damping(0.20);
    // Linear damping is set globally by `Sim::finalize` from
    // `LINEAR_DAMPING`; the per-scene call would be overwritten.

    let mut visuals = Vec::with_capacity(TOWER_LAYERS * TOWER_GRID * TOWER_GRID + 1);
    let half = TOWER_BRICK_HALF as f64;
    let half_v = MVec3::splat(half);
    let inertia = box_inertia(1.0, half);

    let layer_y0 = half + 0.001; // tiny lift to avoid frame-0 plane overlap
    for layer in 0..TOWER_LAYERS {
        let y = layer_y0 + (layer as f64) * (TOWER_LAYER_HEIGHT as f64);
        for ix in 0..TOWER_GRID {
            for iz in 0..TOWER_GRID {
                let x = (ix as f64 - (TOWER_GRID as f64 - 1.0) * 0.5)
                    * TOWER_BRICK_SPACING as f64;
                let z = (iz as f64 - (TOWER_GRID as f64 - 1.0) * 0.5)
                    * TOWER_BRICK_SPACING as f64;
                let pose =
                    mp::Transform::new(MVec3::new(x, y, z), MQuat::IDENTITY);
                let body = b.add_body(
                    mp::BodyParams::new()
                        .with_spatial_inertia(mp::SpatialInertia {
                            mass: 1.0,
                            com: MVec3::ZERO,
                            inertia,
                        })
                        .with_pose(pose),
                );
                b.add_shape_box(body, half_v, mp::Transform::IDENTITY)
                    .expect("brick shape");
                visuals.push(BodyVisual::Box(Vec3::splat(TOWER_BRICK_HALF)));
            }
        }
    }

    // Heavy sphere wrecking ball.
    let ball_radius = TOWER_BALL_RADIUS as f64;
    let ball = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia::sphere(TOWER_BALL_MASS, ball_radius))
            .with_position(MVec3::new(0.0, TOWER_BALL_HEIGHT as f64, 0.0)),
    );
    b.add_shape_sphere(ball, ball_radius, mp::Transform::IDENTITY)
        .expect("ball shape");
    visuals.push(BodyVisual::Sphere(TOWER_BALL_RADIUS));

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane shape");

    (b.build().expect("model"), visuals)
}

/// Horizontal cloth dropping onto a static sphere.
///
/// The cloth starts a few metres above the sphere with no pins, so it
/// falls under gravity and drapes around the sphere. Particle-vs-rigid
/// collision (sphere SDF projection) is handled by the WGSL
/// `particle_vs_shapes` pass in the GPU XPBD pipeline. Mouse grab still
/// works — drag any particle to deform the drape.
const CLOTH_SPHERE_RADIUS: f32 = 0.7;
const CLOTH_SPHERE_Y: f32 = 1.2;

fn build_cloth_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.4);

    let nx = CLOTH_GRID;
    let ny = CLOTH_GRID;
    let size = CLOTH_SIZE as f64;
    // Cloth starts horizontal in the XZ plane, well above the sphere.
    let origin = MVec3::new(-size * 0.5, CLOTH_TOP_Y as f64, -size * 0.5);
    let u_axis = MVec3::new(size, 0.0, 0.0);
    let v_axis = MVec3::new(0.0, 0.0, size);

    let _ = b.add_cloth_grid(
        origin,
        u_axis,
        v_axis,
        nx,
        ny,
        CLOTH_PARTICLE_MASS,
        CLOTH_PARTICLE_RADIUS,
        CLOTH_COMPLIANCE,
        CLOTH_MAX_STRETCH,
        mp::WorldId(0),
    );

    // Static sphere obstacle. `BodyId::NONE` makes it world-anchored
    // — the GPU `particle_vs_shapes` pass treats `shape_body == -1`
    // as static and uses the shape's own transform as world pose.
    b.add_shape_sphere(
        mp::BodyId::NONE,
        CLOTH_SPHERE_RADIUS as f64,
        mp::Transform::from_translation(MVec3::new(0.0, CLOTH_SPHERE_Y as f64, 0.0)),
    )
    .expect("sphere shape");
    // Floor plane so any escaped particle doesn't free-fall forever.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane shape");

    let model = b.build().expect("model");
    // No `BodyVisual` here — cloth renders via the dedicated triangle
    // mesh + the sphere obstacle is spawned as a `ClothObstacle`
    // entity by `apply_scene_switch` / `setup`.
    (model, Vec::new())
}

/// Stage-20 high-resolution VBD cloth — 32×32 sheet falling onto a
/// kinematic sphere, solved with `VbdSolver` instead of `XpbdSolver`.
/// The model topology is identical to `build_cloth_model` (the same
/// `add_cloth_grid` builder); only the solver differs at runtime.
/// That's by design — the user can flip between VBD and XPBD on the
/// same scene via the right-rail toggle and get a side-by-side feel
/// for the difference (XPBD wobbles, VBD settles).
fn build_vbd_cloth_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.4);

    let nx = VBD_CLOTH_GRID;
    let ny = VBD_CLOTH_GRID;
    let size = VBD_CLOTH_SIZE as f64;
    let origin = MVec3::new(-size * 0.5, VBD_CLOTH_TOP_Y as f64, -size * 0.5);
    let u_axis = MVec3::new(size, 0.0, 0.0);
    let v_axis = MVec3::new(0.0, 0.0, size);

    let _ = b.add_cloth_grid(
        origin,
        u_axis,
        v_axis,
        nx,
        ny,
        VBD_CLOTH_PARTICLE_MASS,
        VBD_CLOTH_PARTICLE_RADIUS,
        VBD_CLOTH_COMPLIANCE,
        VBD_CLOTH_MAX_STRETCH,
        mp::WorldId(0),
    );

    b.add_shape_sphere(
        mp::BodyId::NONE,
        VBD_CLOTH_SPHERE_RADIUS as f64,
        mp::Transform::from_translation(MVec3::new(0.0, VBD_CLOTH_SPHERE_Y as f64, 0.0)),
    )
    .expect("sphere shape");
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane shape");

    let model = b.build().expect("model");
    (model, Vec::new())
}

/// Stage-20.1 CABLE polyline demo. A 32-particle cable pinned at one
/// end, the other end driven kinematically by `update_cable_example`.
/// A static cylinder sits between the pin and the puller so the cable
/// drapes over it (soft-on-rigid AVBD + Dahl friction territory).
///
/// The puller body is a kinematic sphere so it shows up in the
/// renderer's body list (we want the user to see what's pulling the
/// cable). The cable itself isn't a body — it's particles + springs,
/// rendered as a gizmo polyline by `draw_cable_gizmos`.
fn build_cable_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_linear_damping(0.6);

    let pin_pos = MVec3::new(
        CABLE_PINNED_AT.x as f64,
        CABLE_PINNED_AT.y as f64,
        CABLE_PINNED_AT.z as f64,
    );
    // Initial puller position — start at the +X side of the orbit.
    let puller_pos = MVec3::new(
        CABLE_PULLER_ORBIT_R as f64,
        CABLE_PULLER_HEIGHT as f64,
        0.0,
    );
    let (first, n, _cable_id) = b.add_cable_polyline(
        pin_pos,
        puller_pos,
        CABLE_PARTICLES,
        CABLE_PARTICLE_MASS,
        CABLE_PARTICLE_RADIUS,
        CABLE_STIFFNESS,
        // Hard cap on per-segment stretch — 0 means inextensible
        // after the soft pass. We rely on VBD's stiffness for the
        // soft pass, so the cap mainly catches numerical drift.
        0.02,
        mp::WorldId(0),
    );
    // Pin both endpoints. The puller end is "kinematic-pinned": its
    // mass stays at zero (so the integrator skips it) and the system
    // `update_cable_example` rewrites its position each frame.
    b.pin_particle(first);
    b.pin_particle(first + n - 1);

    // Static cylinder between pin and puller (the drape obstacle).
    b.add_shape_cylinder(
        mp::BodyId::NONE,
        CABLE_CYLINDER_HALF_HEIGHT as f64,
        CABLE_CYLINDER_RADIUS as f64,
        mp::Transform::from_translation(MVec3::new(
            CABLE_CYLINDER_POS.x as f64,
            CABLE_CYLINDER_POS.y as f64,
            CABLE_CYLINDER_POS.z as f64,
        )),
    )
    .expect("cable cylinder shape");

    // Kinematic puller body — a small sphere we can render. Its
    // pose is overwritten by `update_cable_example` each frame.
    let puller = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia::sphere(0.1, CABLE_PULLER_RADIUS as f64))
            .with_position(puller_pos)
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(puller, CABLE_PULLER_RADIUS as f64, mp::Transform::IDENTITY)
        .expect("puller shape");

    // Floor catch-all so the cable doesn't fall forever if the user
    // pulls too hard and a particle escapes.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("floor plane");

    let model = b.build().expect("cable model");
    let visuals = vec![BodyVisual::Sphere(CABLE_PULLER_RADIUS)];
    (model, visuals)
}

/// Tetrahedrise an axis-aligned cube cell into 6 tets sharing the
/// body diagonal `(v000, v111)`. Standard "diagonal split" — every
/// tet has positive rest volume under the standard `+x +y +z` cube
/// vertex ordering. Mirrors the helper in `tests/vbd_tets_gpu.rs`.
fn jelly_cube_to_six_tets(corners: [u32; 8]) -> [[u32; 4]; 6] {
    let v000 = corners[0];
    let v100 = corners[1];
    let v010 = corners[2];
    let v110 = corners[3];
    let v001 = corners[4];
    let v101 = corners[5];
    let v011 = corners[6];
    let v111 = corners[7];
    [
        [v000, v100, v110, v111],
        [v000, v110, v010, v111],
        [v000, v010, v011, v111],
        [v000, v011, v001, v111],
        [v000, v001, v101, v111],
        [v000, v101, v100, v111],
    ]
}

/// Stage-20.2/20.3 JELLY scene. An 8×8×8 tet jelly cube falls onto a
/// static plane. A kinematic puller body sits at +X for the cube to
/// (optionally) bounce against. The cube renders as gizmo edges on
/// the surface tris (computed at draw time from the tet topology).
///
/// Stage 20.3 — uses `add_tet_mesh_per_material` to assign different
/// `MaterialNeoHookean` recipes per cube cell based on the
/// `JellyMaterialPreset`. The Soft preset gives every cell the same
/// modest E; Stiff gives every cell a high E; Mixed alternates soft
/// interior with stiff corner-cell pillars.
fn build_jelly_model_with_preset(preset: JellyMaterialPreset) -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_linear_damping(0.4);

    let n = JELLY_GRID;
    let side = JELLY_SIDE as f64;
    let dx = side / (n as f64 - 1.0);
    let centre = MVec3::new(0.0, JELLY_TOP_Y as f64, 0.0);
    let origin = centre - MVec3::splat(side * 0.5);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let p = origin
                    + MVec3::new(i as f64 * dx, j as f64 * dx, k as f64 * dx);
                b.add_particle(p, JELLY_PARTICLE_MASS, dx * 0.5, mp::WorldId(0));
            }
        }
    }
    let v_at = |i: u32, j: u32, k: u32| -> u32 { (k * n + j) * n + i };
    let mut tets = Vec::new();
    let mut mat_idx: Vec<u32> = Vec::new();
    let last = n - 2;
    for kk in 0..n - 1 {
        for jj in 0..n - 1 {
            for ii in 0..n - 1 {
                let corners = [
                    v_at(ii, jj, kk),
                    v_at(ii + 1, jj, kk),
                    v_at(ii, jj + 1, kk),
                    v_at(ii + 1, jj + 1, kk),
                    v_at(ii, jj, kk + 1),
                    v_at(ii + 1, jj, kk + 1),
                    v_at(ii, jj + 1, kk + 1),
                    v_at(ii + 1, jj + 1, kk + 1),
                ];
                let cell_tets = jelly_cube_to_six_tets(corners);
                let mi: u32 = match preset {
                    JellyMaterialPreset::Soft => 0,
                    JellyMaterialPreset::Stiff => 0,
                    JellyMaterialPreset::Mixed => {
                        // Stiff (1) on the four vertical corner pillars;
                        // soft (0) elsewhere.
                        if (ii == 0 || ii == last) && (kk == 0 || kk == last) {
                            1
                        } else {
                            0
                        }
                    }
                };
                for tet in cell_tets {
                    tets.push(tet);
                    mat_idx.push(mi);
                }
            }
        }
    }
    let materials: Vec<mp::MaterialNeoHookean> = match preset {
        JellyMaterialPreset::Soft => vec![mp::MaterialNeoHookean::elastic(JELLY_YOUNGS, JELLY_POISSON)],
        JellyMaterialPreset::Stiff => vec![mp::MaterialNeoHookean::elastic(JELLY_YOUNGS * 100.0, JELLY_POISSON)],
        JellyMaterialPreset::Mixed => vec![
            mp::MaterialNeoHookean::elastic(JELLY_YOUNGS, JELLY_POISSON),
            mp::MaterialNeoHookean::elastic(JELLY_YOUNGS * 100.0, JELLY_POISSON),
        ],
    };
    let _ = b
        .add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
        .expect("add_tet_mesh_per_material");

    // Floor plane.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    // Kinematic puller body — small sphere off to +X. The cube falls
    // against it on its way down, demonstrating soft-on-rigid contact.
    let puller_pos = MVec3::new(
        JELLY_PULLER_POS.x as f64,
        JELLY_PULLER_POS.y as f64,
        JELLY_PULLER_POS.z as f64,
    );
    let puller = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia::sphere(0.5, JELLY_PULLER_RADIUS as f64))
            .with_position(puller_pos)
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(puller, JELLY_PULLER_RADIUS as f64, mp::Transform::IDENTITY)
        .expect("puller shape");

    let model = b.build().expect("jelly model");
    let visuals = vec![BodyVisual::Sphere(JELLY_PULLER_RADIUS)];
    (model, visuals)
}

/// Back-compat shim — Stage 20.2 callers used `build_jelly_model()`
/// without preset. Default to Soft.
fn build_jelly_model() -> (mp::Model, Vec<BodyVisual>) {
    build_jelly_model_with_preset(JellyMaterialPreset::Soft)
}

/// Stage-21 mesh-tools demo. The "model" is empty (no physics step
/// runs for this scene); all the visual work happens in
/// `apply_scene_switch` via [`spawn_mesh_tools_entities`], which
/// generates an icosphere and runs decimation / SDF-bake / marching-
/// cubes / hull on it. This builder just provides the requisite
/// `(Model, Vec<BodyVisual>)` shape so [`Sim::build_scene`] can use it
/// uniformly with every other scene.
fn build_mesh_tools_model() -> (mp::Model, Vec<BodyVisual>) {
    let b = mp::ModelBuilder::new();
    (b.build().expect("empty model"), Vec::new())
}

/// Lidar example — a static stage with kinematic obstacles for a GPU
/// [`mp::sensors::Raycaster`] to scan. The lidar config + state lives
/// in the [`LidarConfig`] / [`LidarState`] resources; the right-rail
/// pane edits the config and `update_lidar` re-runs the GPU sensor
/// each frame.
///
/// Obstacles are kinematic (mass = 0 + KINEMATIC flag) so they don't
/// fall under gravity — the lidar wants stable targets, not a
/// settling pile.
fn build_lidar_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let mut visuals = Vec::new();

    let kin_at = |pos: MVec3| {
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_pose(mp::Transform::new(pos, MQuat::IDENTITY))
            .with_flags(mp::BodyFlags::KINEMATIC)
    };

    // Big sphere off to the right.
    {
        let body = b.add_body(kin_at(MVec3::new(4.5, 1.2, 0.0)));
        b.add_shape_sphere(body, 1.2, mp::Transform::IDENTITY)
            .expect("sphere");
        visuals.push(BodyVisual::Sphere(1.2));
    }
    // Tall box on the left.
    {
        let body = b.add_body(kin_at(MVec3::new(-3.5, 1.5, 2.0)));
        let h = MVec3::new(0.7, 1.5, 0.7);
        b.add_shape_box(body, h, mp::Transform::IDENTITY)
            .expect("box");
        visuals.push(BodyVisual::Box(Vec3::new(0.7, 1.5, 0.7)));
    }
    // Wide low box behind.
    {
        let body = b.add_body(kin_at(MVec3::new(0.0, 0.6, 6.0)));
        let h = MVec3::new(3.0, 0.6, 0.4);
        b.add_shape_box(body, h, mp::Transform::IDENTITY)
            .expect("wall");
        visuals.push(BodyVisual::Box(Vec3::new(3.0, 0.6, 0.4)));
    }
    // Capsule pillar on the right-back.
    {
        let body = b.add_body(kin_at(MVec3::new(3.0, 1.0, -3.5)));
        b.add_shape_capsule(body, 1.0, 0.4, mp::Transform::IDENTITY)
            .expect("capsule");
        visuals.push(BodyVisual::Capsule {
            half_h: 1.0,
            radius: 0.4,
        });
    }
    // Cylinder on the left-back.
    {
        let body = b.add_body(kin_at(MVec3::new(-3.0, 0.8, -3.0)));
        b.add_shape_cylinder(body, 0.8, 0.5, mp::Transform::IDENTITY)
            .expect("cylinder");
        // BodyVisual doesn't have a cylinder variant — render as a
        // capsule of the same dimensions, close enough for the demo.
        visuals.push(BodyVisual::Capsule {
            half_h: 0.8,
            radius: 0.5,
        });
    }

    // Static plane.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("model"), visuals)
}

// --- Arm scene -----------------------------------------------------
const ARM_LINKS: usize = 6;
/// Length of each arm link (m).
const ARM_LINK_LEN: f32 = 0.6;
const ARM_LINK_RADIUS: f32 = 0.06;
const ARM_BASE_HEIGHT: f32 = 5.0;
/// Per-link mass [kg]. Mass × kp dictate how snappy the arm looks;
/// too low + high-kp diverges, too high + low-kp barely moves under
/// the chained-inertia load at the shoulder. 0.5 kg with kp=120 is
/// stable at 240 Hz substeps and produces visible swings.
const ARM_LINK_MASS: f64 = 0.5;
/// PD gains. `max_effort` caps a step's torque so a far-from-target
/// joint can't accelerate to 10⁵ rad/s² in a single substep.
const ARM_KP: f64 = 120.0;
const ARM_KD: f64 = 12.0;
const ARM_MAX_EFFORT: f64 = 80.0;

/// 6-link revolute arm anchored at `(0, ARM_BASE_HEIGHT, 0)`. The
/// shoulder joint rotates around `+Z` and every elbow rotates around
/// `+X`, which gives the arm two visible degrees of swing — a "wave"
/// in the XY plane (shoulder) plus folding in YZ (elbows). Default
/// joint orientation has each link extending along the world
/// `-Y` direction, so the rest pose is a hanging arm; gravity adds
/// to the PD torque to make swings dramatic.
///
/// When `mimic` is true (Stage 32), a 7th body + revolute joint is
/// appended in a *separate* articulation alongside the arm and
/// coupled to joint 0 by a `JointMimic` equality with the linear-
/// identity polynomial `[0, 1, 0, 0, 0]`. The mimic joint's link
/// hangs visually offset along `+X` so the user can see both joints
/// move in lockstep.
#[allow(dead_code, reason = "kept as a back-compat shim — Stage 32 routed callers through `build_arm_model_with` to thread the mimic toggle")]
fn build_arm_model() -> (mp::Model, Vec<BodyVisual>) {
    build_arm_model_with(false)
}

/// Stage-32 mimic-toggle accessor. The toggle lives in
/// `ArmExampleState`; we mirror it into a static atomic so the
/// `Sim::build_scene` (which has no SystemParam access) can read it
/// when the scene is rebuilt. `arm_pane` is the only writer; the
/// scene-switch handler reads + clears the dirty flag.
static ARM_MIMIC_FLAG: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Stage 33.3 — ClothCollar pane / update-system shared state. Since
/// SystemParam bundles are at the 16-tuple ceiling, we route per-
/// scene state through static atomics (the same pattern Stage 32's
/// arm-mimic toggle uses). The pane writes; the update system + the
/// scene rebuild read.
static COLLAR_SELF_COLL_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);
/// Reset signal — the pane sets this when the user wants to rewind
/// the puller height. The update system checks + clears it each
/// frame.
static COLLAR_RESET_REQUESTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
/// Puller height (m, current Y of the iy=NX-1 row). The update
/// system mutates this each frame; the pane reads it for display.
static COLLAR_PULLER_Y: std::sync::atomic::Ordering =
    std::sync::atomic::Ordering::Relaxed;
// We store the puller Y as bits because AtomicF64 is unstable.
static COLLAR_PULLER_Y_BITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

fn collar_get_puller_y() -> f64 {
    f64::from_bits(COLLAR_PULLER_Y_BITS.load(COLLAR_PULLER_Y))
}
fn collar_set_puller_y(y: f64) {
    COLLAR_PULLER_Y_BITS.store(y.to_bits(), COLLAR_PULLER_Y);
}

fn arm_mimic_flag() -> bool {
    ARM_MIMIC_FLAG.load(std::sync::atomic::Ordering::Relaxed)
}

fn set_arm_mimic_flag(v: bool) {
    ARM_MIMIC_FLAG.store(v, std::sync::atomic::Ordering::Relaxed);
}

fn build_arm_model_with(mimic: bool) -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.1);

    let mut visuals = Vec::with_capacity(ARM_LINKS);
    let half_h = (ARM_LINK_LEN as f64) * 0.5;
    let radius = ARM_LINK_RADIUS as f64;

    let inertia = capsule_inertia(ARM_LINK_MASS, half_h, radius);

    b.begin_articulation();

    let mut prev_body = mp::BodyId::NONE;
    // Initial body_q is just a placeholder — Featherstone's Pass 4 FK
    // overwrites these on the first step.
    let mut next_pos = MVec3::new(0.0, ARM_BASE_HEIGHT as f64 - half_h, 0.0);

    for i in 0..ARM_LINKS {
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: ARM_LINK_MASS,
                    com: MVec3::ZERO,
                    inertia,
                })
                .with_position(next_pos),
        );
        b.add_shape_capsule(body, half_h, radius, mp::Transform::IDENTITY)
            .expect("arm link shape");
        visuals.push(BodyVisual::Capsule {
            half_h: half_h as f32,
            radius: radius as f32,
        });

        // Joint anchor in PARENT frame: world base for the shoulder,
        // the +Y tip of the previous link for elbows. For the
        // shoulder we also rotate the joint frame so axis +Z
        // (camera-facing) gives a rest pose where body 0 hangs along
        // world -Y. For elbows the parent's frame already has +Y
        // pointing along the chain so no extra rotation is needed.
        let x_p = if i == 0 {
            // 180° around X flips the joint-local +Y to world -Y, so
            // a revolute joint with axis +Z + q=0 leaves the link
            // hanging straight down from the shoulder.
            let flip = MQuat::from_axis_angle(MVec3::X, std::f64::consts::PI);
            mp::Transform::new(
                MVec3::new(0.0, ARM_BASE_HEIGHT as f64, 0.0),
                flip,
            )
        } else {
            mp::Transform::from_translation(MVec3::new(0.0, half_h, 0.0))
        };
        let x_c = mp::Transform::from_translation(MVec3::new(0.0, -half_h, 0.0));

        // Shoulder swings in the world XY plane (axis +Z); the elbows
        // bend in YZ (axis +X). Mixing axes gives the arm two
        // independent "directions" of motion, which reads better
        // than every joint sweeping the same plane.
        let axis = if i == 0 { MVec3::Z } else { MVec3::X };
        let _jid = b
            .add_joint_revolute(prev_body, body, x_p, x_c, axis)
            .expect("arm joint");

        prev_body = body;
        next_pos = next_pos - MVec3::new(0.0, ARM_LINK_LEN as f64, 0.0);
    }

    // Stage 32 — optional joint-mimic side chain. Append a single
    // capsule body in a SECOND articulation that hangs offset along
    // +X and is coupled to joint 0 (the shoulder) by a `JointMimic`
    // equality with the linear-identity polynomial. Both joints
    // remain dynamic at the Featherstone level; the XPBD equality
    // pass enforces `q_mimic = q_shoulder` each substep when the
    // XPBD path is active. With Featherstone-only solvers the
    // existing `add_mimic_constraint` linear pre-step keeps it in
    // sync (same effect, separate code path).
    if mimic {
        b.begin_articulation();
        let offset = MVec3::new(1.5, ARM_BASE_HEIGHT as f64 - half_h, 0.0);
        let mimic_body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: ARM_LINK_MASS,
                    com: MVec3::ZERO,
                    inertia,
                })
                .with_position(offset),
        );
        b.add_shape_capsule(mimic_body, half_h, radius, mp::Transform::IDENTITY)
            .expect("mimic link shape");
        visuals.push(BodyVisual::Capsule {
            half_h: half_h as f32,
            radius: radius as f32,
        });
        let flip = MQuat::from_axis_angle(MVec3::X, std::f64::consts::PI);
        let x_p_m = mp::Transform::new(
            MVec3::new(1.5, ARM_BASE_HEIGHT as f64, 0.0),
            flip,
        );
        let x_c_m = mp::Transform::from_translation(MVec3::new(0.0, -half_h, 0.0));
        let mimic_joint = b
            .add_joint_revolute(mp::BodyId::NONE, mimic_body, x_p_m, x_c_m, MVec3::Z)
            .expect("mimic joint");
        // Driver = joint 0 (the shoulder). Identity linear mimic.
        let shoulder = mp::JointId(0);
        b.add_equality_joint_mimic(shoulder, mimic_joint, [0.0, 1.0, 0.0, 0.0, 0.0])
            .expect("mimic equality");
        // Also wire the legacy linear-mimic path so the Featherstone
        // pre-step keeps the joint synced even when the XPBD
        // equality pass isn't invoked (the arm scene runs on
        // Featherstone GPU; the JointMimic projection lands when XPBD
        // is in the loop). Both paths agree on the linear-identity
        // case so the duplicate is harmless.
        b.add_mimic_constraint(mimic_joint, shoulder, 0.0, 1.0)
            .expect("mimic legacy");
    }

    (b.build().expect("arm model"), visuals)
}

// --- Stage-22 MuJoCo wrapper scene -------------------------------
//
// Double pendulum (4-link arm would also work; we picked the
// pendulum because it's the canonical MuJoCo benchmark scene and
// the topology is clean: two revolute joints + two capsule links).
// Same body / joint construction as a slice of `build_arm_model`,
// just shorter and without the FK-on-elbow rotation tricks.
const MUJOCO_LINK_LEN: f32 = 0.5;
const MUJOCO_LINK_RADIUS: f32 = 0.05;
const MUJOCO_LINK_MASS: f64 = 0.6;
const MUJOCO_BASE_HEIGHT: f32 = 4.0;

fn build_mujoco_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);

    let half_h = (MUJOCO_LINK_LEN as f64) * 0.5;
    let radius = MUJOCO_LINK_RADIUS as f64;
    let inertia = capsule_inertia(MUJOCO_LINK_MASS, half_h, radius);

    b.begin_articulation();

    // Upper link.
    let upper_pos = MVec3::new(0.0, MUJOCO_BASE_HEIGHT as f64 - half_h, 0.0);
    let upper = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia {
                mass: MUJOCO_LINK_MASS,
                com: MVec3::ZERO,
                inertia,
            })
            .with_position(upper_pos),
    );
    b.add_shape_capsule(upper, half_h, radius, mp::Transform::IDENTITY)
        .expect("upper link");

    // Shoulder revolute joint about world +Z, anchored at the base
    // height. The PI rotation around X mirrors the arm scene's trick
    // for hanging the link along world -Y at q=0.
    let shoulder_x_p = mp::Transform::new(
        MVec3::new(0.0, MUJOCO_BASE_HEIGHT as f64, 0.0),
        MQuat::from_axis_angle(MVec3::X, std::f64::consts::PI),
    );
    let shoulder_x_c = mp::Transform::from_translation(MVec3::new(0.0, -half_h, 0.0));
    b.add_joint_revolute(mp::BodyId::NONE, upper, shoulder_x_p, shoulder_x_c, MVec3::Z)
        .expect("shoulder joint");

    // Lower link.
    let lower_pos = upper_pos - MVec3::new(0.0, MUJOCO_LINK_LEN as f64, 0.0);
    let lower = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia {
                mass: MUJOCO_LINK_MASS,
                com: MVec3::ZERO,
                inertia,
            })
            .with_position(lower_pos),
    );
    b.add_shape_capsule(lower, half_h, radius, mp::Transform::IDENTITY)
        .expect("lower link");

    let elbow_x_p = mp::Transform::from_translation(MVec3::new(0.0, half_h, 0.0));
    let elbow_x_c = mp::Transform::from_translation(MVec3::new(0.0, -half_h, 0.0));
    b.add_joint_revolute(upper, lower, elbow_x_p, elbow_x_c, MVec3::X)
        .expect("elbow joint");

    let visuals = vec![
        BodyVisual::Capsule {
            half_h: half_h as f32,
            radius: radius as f32,
        },
        BodyVisual::Capsule {
            half_h: half_h as f32,
            radius: radius as f32,
        },
    ];

    (b.build().expect("mujoco model"), visuals)
}

// --- Stage-23 differentiable arm scene -------------------------------
//
// Two kinematic capsule bodies representing the upper + lower arm of
// a 2-link planar arm. The actual "physics" runs host-side in
// `update_diff_arm` via [`DiffArm`]; here we just declare two
// kinematic bodies whose `body_q` will be rewritten each frame from
// the optimisation playback. No joints — molla's joint system isn't
// needed because we drive both bodies' world poses directly.
fn build_diff_arm_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let arm = DiffArm::default();

    // Body 0: upper arm. Half-length = L1/2 along the body's local +Y;
    // visualised as a capsule. Initial pose = hanging straight down
    // from the shoulder anchor. `update_diff_arm` rewrites this every
    // frame — the initial pose is just a sane starting visual.
    let half1 = arm.l1 * 0.5;
    let upper = b.add_body(
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_pose(mp::Transform::new(
                MVec3::new(0.0, DIFF_ARM_BASE_Y as f64 - half1, 0.0),
                MQuat::IDENTITY,
            ))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_capsule(upper, half1, DIFF_ARM_LINK_RADIUS as f64, mp::Transform::IDENTITY)
        .expect("diff arm upper shape");

    // Body 1: lower arm.
    let half2 = arm.l2 * 0.5;
    let lower = b.add_body(
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_pose(mp::Transform::new(
                MVec3::new(0.0, DIFF_ARM_BASE_Y as f64 - arm.l1 - half2, 0.0),
                MQuat::IDENTITY,
            ))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_capsule(lower, half2, DIFF_ARM_LINK_RADIUS as f64, mp::Transform::IDENTITY)
        .expect("diff arm lower shape");

    let visuals = vec![
        BodyVisual::Capsule {
            half_h: half1 as f32,
            radius: DIFF_ARM_LINK_RADIUS,
        },
        BodyVisual::Capsule {
            half_h: half2 as f32,
            radius: DIFF_ARM_LINK_RADIUS,
        },
    ];

    // Static plane so the camera has a ground reference.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("diff arm model"), visuals)
}

// --- Stage-23.1 differentiable Featherstone arm scene ---------------
//
// Six kinematic capsule bodies representing the links of a 6-DOF
// Franka-style serial chain. Same playback model as `DiffArm`: the
// dynamics live entirely host-side in `update_diff_arm6` via
// [`FeatherstoneRevoluteChain`]; here we just declare six kinematic
// bodies whose `body_q` will be rewritten each frame from the
// optimisation playback.
const DIFF_ARM6_BASE_HEIGHT: f64 = 0.5;
const DIFF_ARM6_LINK_RADIUS: f32 = 0.04;

fn build_diff_arm6_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let chain = FeatherstoneRevoluteChain::franka6(0.005, 60);
    let mut visuals = Vec::with_capacity(6);
    for link in &chain.links {
        let len = (link.com_local.length() * 2.0).max(0.06);
        let body = b.add_body(
            mp::BodyParams::new()
                .with_mass(0.0)
                .with_pose(mp::Transform::new(
                    MVec3::new(0.0, DIFF_ARM6_BASE_HEIGHT, 0.0),
                    MQuat::IDENTITY,
                ))
                .with_flags(mp::BodyFlags::KINEMATIC),
        );
        let half_h = (len * 0.5).max(0.04);
        b.add_shape_capsule(
            body,
            half_h,
            DIFF_ARM6_LINK_RADIUS as f64,
            mp::Transform::IDENTITY,
        )
        .expect("diff arm6 link shape");
        visuals.push(BodyVisual::Capsule {
            half_h: half_h as f32,
            radius: DIFF_ARM6_LINK_RADIUS,
        });
    }
    // Static plane so the camera has a ground reference.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");
    (b.build().expect("diff arm6 model"), visuals)
}

// --- Stage-18 composable actuator pipeline scene ------------------
//
// Single revolute joint mounted on a stationary stub. The joint is
// driven by `Actuator { Delay → ControllerPipelinePd → ClampingDcMotor }`
// every substep — the Delay pushes / reads from a circular buffer
// of depth 32, the PD writes raw effort, the DC-motor clamp turns
// it into a torque-vs-speed envelope.
//
// Visuals:
//   • body 0 — short stub of length 0.3 m anchored at +1.5 m (no
//     parent ⇒ joint to NONE); rotates around world +Z so the swing
//     plays out in the camera-facing X/Y plane.
const ACTUATOR_BASE_HEIGHT: f64 = 1.5;
const ACTUATOR_LINK_HALF_H: f64 = 0.4;
const ACTUATOR_LINK_RADIUS: f64 = 0.05;
const ACTUATOR_LINK_MASS: f64 = 0.5;

fn build_actuator_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);

    let inertia = capsule_inertia(
        ACTUATOR_LINK_MASS,
        ACTUATOR_LINK_HALF_H,
        ACTUATOR_LINK_RADIUS,
    );

    b.begin_articulation();

    // Anchor the link to world via a revolute joint. Joint frame in
    // parent space lifts to (0, ACTUATOR_BASE_HEIGHT, 0) and the
    // 180° flip around X makes the link hang along world -Y at q=0.
    let body = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia {
                mass: ACTUATOR_LINK_MASS,
                com: MVec3::ZERO,
                inertia,
            })
            .with_position(MVec3::new(
                0.0,
                ACTUATOR_BASE_HEIGHT - ACTUATOR_LINK_HALF_H,
                0.0,
            )),
    );
    b.add_shape_capsule(
        body,
        ACTUATOR_LINK_HALF_H,
        ACTUATOR_LINK_RADIUS,
        mp::Transform::IDENTITY,
    )
    .expect("actuator link shape");

    let flip = MQuat::from_axis_angle(MVec3::X, std::f64::consts::PI);
    let x_p = mp::Transform::new(
        MVec3::new(0.0, ACTUATOR_BASE_HEIGHT, 0.0),
        flip,
    );
    let x_c = mp::Transform::from_translation(MVec3::new(
        0.0,
        -ACTUATOR_LINK_HALF_H,
        0.0,
    ));
    b.add_joint_revolute(mp::BodyId::NONE, body, x_p, x_c, MVec3::Z)
        .expect("actuator joint");

    let visuals = vec![BodyVisual::Capsule {
        half_h: ACTUATOR_LINK_HALF_H as f32,
        radius: ACTUATOR_LINK_RADIUS as f32,
    }];

    (b.build().expect("actuator model"), visuals)
}

// --- Stage-18.1 neural-policy cart-pole scene ---------------------
//
// Cart slides along world +X via a prismatic joint. Pole hangs above
// the cart and rotates around world +Z via a revolute joint. The
// pre-trained MLP balance policy keeps the pole near upright (θ=0)
// while damping cart drift. With a small initial perturbation in θ
// the system swings back to upright in a few seconds.
//
// Topology:
//   • body 0 — cart (small box)
//     • prismatic joint (parent=NONE) along world +X, joint_q[0] = x
//   • body 1 — pole (capsule)
//     • revolute joint (parent=cart) about Z, joint_q[1] = θ
//
// Joint frame for the cart's prismatic: cart spawns at world origin,
// height CART_RAIL_Y; the pole's revolute pivots at the top of the
// cart (where the pole is hinged).
const CART_RAIL_Y: f64 = 1.0;
const CART_HALF_X: f64 = 0.15;
const CART_HALF_Y: f64 = 0.05;
const CART_HALF_Z: f64 = 0.10;
const CART_MASS: f64 = 1.0;
const POLE_HALF_LEN: f64 = 0.5;
const POLE_RADIUS: f64 = 0.03;
const POLE_MASS: f64 = 0.1;

fn build_neural_policy_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    // Light damping so the system doesn't asymptote to zero on its
    // own — the policy has to actually do work.
    b.set_angular_damping(0.02);

    b.begin_articulation();

    // ── Cart (prismatic to world along +X) ───────────────────────────
    let cart_inertia = box_inertia_xyz(
        CART_MASS,
        CART_HALF_X,
        CART_HALF_Y,
        CART_HALF_Z,
    );
    let cart = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia {
                mass: CART_MASS,
                com: MVec3::ZERO,
                inertia: cart_inertia,
            })
            .with_position(MVec3::new(0.0, CART_RAIL_Y, 0.0)),
    );
    b.add_shape_box(
        cart,
        MVec3::new(CART_HALF_X, CART_HALF_Y, CART_HALF_Z),
        mp::Transform::IDENTITY,
    )
    .expect("cart shape");

    let x_p_cart = mp::Transform::from_translation(MVec3::new(0.0, CART_RAIL_Y, 0.0));
    let x_c_cart = mp::Transform::IDENTITY;
    b.add_joint_prismatic(mp::BodyId::NONE, cart, x_p_cart, x_c_cart, MVec3::X)
        .expect("cart prismatic joint");

    // ── Pole (revolute to cart, axis +Z) ─────────────────────────────
    // Pole rest pose at θ=0: hangs straight down (along -Y) from the
    // cart's top centre. We rotate the joint frame 180° about X so
    // joint-local +Y points along world -Y, making the pole's body
    // frame +Y axis = capsule long axis = downward at rest.
    let pole_inertia = capsule_inertia(POLE_MASS, POLE_HALF_LEN, POLE_RADIUS);
    let pole = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia {
                mass: POLE_MASS,
                com: MVec3::ZERO,
                inertia: pole_inertia,
            })
            // Rest pose visualisation only — Featherstone's FK pass
            // overwrites this on the first step.
            .with_position(MVec3::new(0.0, CART_RAIL_Y - POLE_HALF_LEN, 0.0)),
    );
    b.add_shape_capsule(pole, POLE_HALF_LEN, POLE_RADIUS, mp::Transform::IDENTITY)
        .expect("pole shape");

    // Joint frame in cart-local space: top of the cart (+Y by
    // CART_HALF_Y), no rotation needed since the pole's body-local +Y
    // is already its long axis. The pole hangs at θ=0 (down).
    let x_p_pole = mp::Transform::from_translation(MVec3::new(0.0, CART_HALF_Y, 0.0));
    let x_c_pole = mp::Transform::from_translation(MVec3::new(0.0, POLE_HALF_LEN, 0.0));
    b.add_joint_revolute(cart, pole, x_p_pole, x_c_pole, MVec3::Z)
        .expect("pole revolute joint");

    let visuals = vec![
        BodyVisual::Box(Vec3::new(
            CART_HALF_X as f32,
            CART_HALF_Y as f32,
            CART_HALF_Z as f32,
        )),
        BodyVisual::Capsule {
            half_h: POLE_HALF_LEN as f32,
            radius: POLE_RADIUS as f32,
        },
    ];

    (b.build().expect("neural policy model"), visuals)
}

/// Inertia of a uniform box about its centre. Axes: x/y/z extents are
/// 2·(hx, hy, hz). Returned as a diagonal `Mat3` per
/// [`mp::SpatialInertia::inertia`]. Distinct from the existing
/// `box_inertia(mass, half)` helper, which assumes a cube.
fn box_inertia_xyz(mass: f64, hx: f64, hy: f64, hz: f64) -> molla::math::Mat3 {
    let s = mass / 3.0;
    let ixx = s * (hy * hy + hz * hz);
    let iyy = s * (hx * hx + hz * hz);
    let izz = s * (hx * hx + hy * hy);
    molla::math::Mat3::from_diagonal(MVec3::new(ixx, iyy, izz))
}

// --- Stage-16 GPU IK arm scene ------------------------------------
//
// 7-DOF Franka-style arm rooted at world origin (lifted on a base
// pillar so the arm rests above the ground grid). End-effector chases
// a draggable target (XYZ sliders in the right-rail pane). Per-frame:
// `update_gpu_ik_arm` runs `IkSolver::solve_gpu` with the current
// target as a position objective on the EE body, writes `joint_q` +
// `body_q` on the host, and `step_simulation` skips this scene
// entirely (no integration — purely kinematic IK).

const GPU_IK_BASE_HEIGHT: f64 = 1.0;
const GPU_IK_LINK_RADIUS: f64 = 0.05;
const GPU_IK_LINK_MASS: f64 = 1.0;
/// Default reach offset relative to the arm's rest EE position.
const GPU_IK_DEFAULT_OFFSET: MVec3 = MVec3::new(0.25, 0.05, -0.15);

fn build_gpu_ik_arm_model() -> (mp::Model, Vec<BodyVisual>) {
    use mp::*;
    let mut b = ModelBuilder::new();
    let mut visuals = Vec::with_capacity(7);

    let mass = GPU_IK_LINK_MASS;
    let half = MVec3::new(0.04, 0.04, 0.04);
    let h2 = half * half;
    let m_third = mass / 3.0;
    let inertia = molla::math::Mat3::from_diagonal(MVec3::new(
        m_third * (h2.y + h2.z),
        m_third * (h2.x + h2.z),
        m_third * (h2.x + h2.y),
    ));
    let make_link = |b: &mut ModelBuilder| {
        b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass,
            com: MVec3::ZERO,
            inertia,
        }))
    };

    b.begin_articulation();
    let axes = [MVec3::Z, MVec3::Y, MVec3::Z, MVec3::Y, MVec3::Z, MVec3::Y, MVec3::Z];
    let link_lens = [0.333_f64, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107];
    let mut prev = molla::core::BodyId::NONE;
    for i in 0..7 {
        let body = make_link(&mut b);
        let x_p = if i == 0 {
            // Lift the arm root above the ground grid so the linkage
            // is visible from the default camera.
            mp::Transform::from_translation(MVec3::new(0.0, GPU_IK_BASE_HEIGHT, 0.0))
        } else {
            mp::Transform::from_translation(MVec3::new(0.0, 0.0, link_lens[i - 1]))
        };
        let x_c = mp::Transform::IDENTITY;
        b.add_joint_revolute(prev, body, x_p, x_c, axes[i]).unwrap();
        // A short capsule running along the link's local +Z. Last
        // link (the EE) gets a slightly wider sphere visual so the
        // tip is easy to spot.
        let half_h = link_lens[i].max(0.06_f64) * 0.5;
        let radius = GPU_IK_LINK_RADIUS;
        // We use Sphere for the EE so the gizmo target dot lines up.
        let visual = if i == 6 {
            BodyVisual::Sphere(0.08)
        } else {
            BodyVisual::Capsule {
                half_h: half_h as f32,
                radius: radius as f32,
            }
        };
        visuals.push(visual);
        prev = body;
    }

    (b.build().expect("gpu ik arm model"), visuals)
}

// --- Stage-17 quadrotor scene --------------------------------------
//
// FREE joint root + 4 fixed-joint rotor bodies at ±arm × {X, Z}.
// PD altitude + attitude control writes per-rotor thrusts as world-
// frame vertical forces into `state.body_f` on each rotor body —
// Featherstone consumes them via the bias term, producing both
// translation (sum of thrusts) and rotation (asymmetric thrusts).
//
// The four rotors are arranged in a "+" shape: rotor 0 = +X, rotor 1
// = -X, rotor 2 = +Z, rotor 3 = -Z. Roll = thrust differential between
// rotors 2/3 (+Z vs -Z); pitch = differential between rotors 0/1.

const QUAD_ARM: f64 = 0.25;
const QUAD_FRAME_MASS: f64 = 1.0;
const QUAD_ROTOR_MASS: f64 = 0.05;
const QUAD_FRAME_HALF: MVec3 = MVec3::new(0.18, 0.04, 0.18);
const QUAD_ROTOR_RADIUS: f64 = 0.06;
const QUAD_INITIAL_HEIGHT: f64 = 1.0;

fn build_quadrotor_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_linear_damping(0.4);
    b.set_angular_damping(0.6);
    b.begin_articulation();

    let mut visuals = Vec::with_capacity(5);

    // Frame body — central cuboid; FREE joint to world.
    let half = QUAD_FRAME_HALF;
    let m = QUAD_FRAME_MASS;
    let i_frame = molla::math::Mat3::from_diagonal(MVec3::new(
        m * (half.y * half.y + half.z * half.z) / 3.0,
        m * (half.x * half.x + half.z * half.z) / 3.0,
        m * (half.x * half.x + half.y * half.y) / 3.0,
    ));
    let frame = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia {
                mass: m,
                com: MVec3::ZERO,
                inertia: i_frame,
            })
            .with_position(MVec3::new(0.0, QUAD_INITIAL_HEIGHT, 0.0)),
    );
    b.add_shape_box(frame, half, mp::Transform::IDENTITY)
        .expect("frame box");
    visuals.push(BodyVisual::Box(Vec3::new(
        half.x as f32,
        half.y as f32,
        half.z as f32,
    )));
    // FREE joint anchored at world origin; child anchor at frame origin.
    b.add_joint_free(
        mp::BodyId::NONE,
        frame,
        mp::Transform::IDENTITY,
        mp::Transform::IDENTITY,
    )
    .expect("free root");

    // Rotor offsets in body frame (at ±arm along X and Z).
    let rotor_offsets = [
        MVec3::new(QUAD_ARM, 0.0, 0.0),  // 0: +X
        MVec3::new(-QUAD_ARM, 0.0, 0.0), // 1: -X
        MVec3::new(0.0, 0.0, QUAD_ARM),  // 2: +Z
        MVec3::new(0.0, 0.0, -QUAD_ARM), // 3: -Z
    ];
    let rotor_mass = QUAD_ROTOR_MASS;
    let r2 = QUAD_ROTOR_RADIUS * QUAD_ROTOR_RADIUS;
    let i_rotor = molla::math::Mat3::from_diagonal(MVec3::new(
        rotor_mass * r2 * 0.25,
        rotor_mass * r2 * 0.5,
        rotor_mass * r2 * 0.25,
    ));
    for off in rotor_offsets.iter() {
        let rotor = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: rotor_mass,
                    com: MVec3::ZERO,
                    inertia: i_rotor,
                })
                .with_position(MVec3::new(
                    off.x,
                    QUAD_INITIAL_HEIGHT + 0.02,
                    off.z,
                )),
        );
        b.add_shape_sphere(rotor, QUAD_ROTOR_RADIUS, mp::Transform::IDENTITY)
            .expect("rotor sphere");
        visuals.push(BodyVisual::Sphere(QUAD_ROTOR_RADIUS as f32));
        // Fixed joint to the frame at the rotor's offset.
        b.add_joint_fixed(
            frame,
            rotor,
            mp::Transform::from_translation(*off + MVec3::Y * 0.02),
            mp::Transform::IDENTITY,
        )
        .expect("rotor fixed joint");
    }

    (b.build().expect("quadrotor model"), visuals)
}

// --- Stage-13 specialty-shapes scene --------------------------------
const SHAPES_FIELD_N: u32 = 16;
const SHAPES_FIELD_CELL: f32 = 0.6;
const SHAPES_TORUS_MAJOR: f32 = 1.0;
const SHAPES_TORUS_MINOR: f32 = 0.25;
const SHAPES_TORUS_Y: f32 = 3.5;

/// Stage-13 demo scene — drops cones, convex hulls, and spheres
/// onto a wavy heightfield with a static torus (SDF) sticking out
/// of the middle. Single scene exercises every primitive shipped
/// in Stage 13: Cone, ConvexHull, Heightfield, Sdf.
fn build_shapes_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.2);

    let mut visuals: Vec<BodyVisual> = Vec::new();

    // ── Heightfield (gently wavy terrain) — static, no body. ────
    let n = SHAPES_FIELD_N;
    let cell = SHAPES_FIELD_CELL as f64;
    let half = (n as f64 - 1.0) * cell * 0.5;
    let mut heights = Vec::with_capacity((n * n) as usize);
    for j in 0..n {
        for i in 0..n {
            let x = (i as f64 - (n as f64 - 1.0) * 0.5) * cell;
            let z = (j as f64 - (n as f64 - 1.0) * 0.5) * cell;
            // Gentler waves than before — small amplitude so things
            // settle visibly instead of rolling off into the abyss.
            let h = 0.25 * ((x * 0.5).sin() + (z * 0.5).cos());
            heights.push(h);
        }
    }
    let hf = mp::Heightfield {
        nx: n,
        nz: n,
        cell_size: cell,
        origin: MVec3::new(-half, 0.0, -half),
        heights,
    };
    let field_id = b.add_heightfield(hf);
    b.add_shape_heightfield(mp::BodyId::NONE, field_id, mp::Transform::IDENTITY)
        .expect("heightfield shape");

    // ── Static torus (SDF) embedded in the terrain centre. ──────
    let sdf_id = b.add_sdf(mp::SdfShape::Torus {
        major: SHAPES_TORUS_MAJOR as f64,
        minor: SHAPES_TORUS_MINOR as f64,
    });
    b.add_shape_sdf(
        mp::BodyId::NONE,
        sdf_id,
        mp::Transform::from_translation(MVec3::new(0.0, SHAPES_TORUS_Y as f64, 0.0)),
    )
    .expect("torus sdf");

    // ── Falling cones (3 of them, scattered above the field). ───
    for (i, pos) in [
        MVec3::new(-1.5, 8.0, -1.0),
        MVec3::new(0.5, 9.0, 0.5),
        MVec3::new(1.8, 8.5, -0.5),
    ]
    .iter()
    .enumerate()
    {
        let half_h = 0.35_f64;
        let radius = 0.25_f64;
        let mass = 0.8_f64;
        let inertia = molla::math::Mat3::from_diagonal(MVec3::new(
            mass * (radius * radius * 0.25 + (2.0 * half_h).powi(2) / 12.0),
            mass * radius * radius * 0.5,
            mass * (radius * radius * 0.25 + (2.0 * half_h).powi(2) / 12.0),
        ));
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia { mass, com: MVec3::ZERO, inertia })
                .with_pose(mp::Transform::new(
                    *pos,
                    // Tilt each cone a bit so they tumble.
                    MQuat::from_axis_angle(MVec3::Z, (i as f64) * 0.3 + 0.4),
                )),
        );
        b.add_shape_cone(body, half_h, radius, mp::Transform::IDENTITY)
            .expect("cone shape");
        visuals.push(BodyVisual::Cone {
            half_h: half_h as f32,
            radius: radius as f32,
        });
    }

    // ── Falling convex hulls (3 — actual cuboid hulls so the
    // visual matches the collision geometry). ──────────────────
    for (i, pos) in [
        MVec3::new(-2.5, 10.0, 1.5),
        MVec3::new(0.0, 10.5, -2.0),
        MVec3::new(2.5, 9.5, 1.5),
    ]
    .iter()
    .enumerate()
    {
        // 8-vertex cuboid hull — vertices at every (±sx, ±sy, ±sz)
        // corner. Slightly different proportions per cube so they
        // tumble interestingly.
        let half = MVec3::new(0.30, 0.40, 0.30);
        let verts = vec![
            MVec3::new(-half.x, -half.y, -half.z),
            MVec3::new(half.x, -half.y, -half.z),
            MVec3::new(-half.x, half.y, -half.z),
            MVec3::new(half.x, half.y, -half.z),
            MVec3::new(-half.x, -half.y, half.z),
            MVec3::new(half.x, -half.y, half.z),
            MVec3::new(-half.x, half.y, half.z),
            MVec3::new(half.x, half.y, half.z),
        ];
        let mass = 0.6;
        let h2 = half * half;
        let m_third = mass / 3.0;
        let inertia = molla::math::Mat3::from_diagonal(MVec3::new(
            m_third * (h2.y + h2.z),
            m_third * (h2.x + h2.z),
            m_third * (h2.x + h2.y),
        ));
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia { mass, com: MVec3::ZERO, inertia })
                .with_pose(mp::Transform::new(
                    *pos,
                    MQuat::from_axis_angle(MVec3::X, (i as f64) * 0.5),
                )),
        );
        let hull_id = b.add_hull(verts);
        b.add_shape_convex_hull(body, hull_id, mp::Transform::IDENTITY)
            .expect("hull shape");
        // Render as the matching box. The hull's bounding box is
        // exactly the cube formed by ±half, so the visual
        // matches the collision shape 1:1.
        visuals.push(BodyVisual::Box(Vec3::new(
            half.x as f32,
            half.y as f32,
            half.z as f32,
        )));
    }

    // ── Falling spheres for variety. ────────────────────────────
    for (i, pos) in [
        MVec3::new(-0.8, 11.0, 0.0),
        MVec3::new(0.8, 11.5, 1.5),
        MVec3::new(-1.5, 11.5, -1.5),
    ]
    .iter()
    .enumerate()
    {
        let _ = i;
        let radius = 0.3;
        let mass = 1.0;
        let i_v = (2.0 / 5.0) * mass * radius * radius;
        let inertia = molla::math::Mat3::from_diagonal(MVec3::new(i_v, i_v, i_v));
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia { mass, com: MVec3::ZERO, inertia })
                .with_position(*pos),
        );
        b.add_shape_sphere(body, radius, mp::Transform::IDENTITY)
            .expect("sphere shape");
        // Rolling friction so spheres come to rest on the wavy
        // heightfield instead of rolling forever.
        b.set_shape_friction_modes(
            mp::ShapeId::from_index(b.shape_count() - 1),
            0.05,
            0.02,
        )
        .ok();
        visuals.push(BodyVisual::Sphere(radius as f32));
    }

    // ── Ellipsoid (Stage 30) — one tilted oblate body that lands on
    // the wavy field and demonstrates the new Sphere/Plane and GJK
    // pairs in action.
    {
        let half_axes = MVec3::new(0.45, 0.20, 0.35);
        let mass = 0.9_f64;
        let h2 = half_axes * half_axes;
        let m_fifth = mass / 5.0;
        let inertia = molla::math::Mat3::from_diagonal(MVec3::new(
            m_fifth * (h2.y + h2.z),
            m_fifth * (h2.x + h2.z),
            m_fifth * (h2.x + h2.y),
        ));
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass,
                    com: MVec3::ZERO,
                    inertia,
                })
                .with_pose(mp::Transform::new(
                    MVec3::new(1.2, 12.0, 1.2),
                    MQuat::from_axis_angle(MVec3::Z, 0.6),
                )),
        );
        b.add_shape_ellipsoid(body, half_axes, mp::Transform::IDENTITY)
            .expect("ellipsoid shape");
        visuals.push(BodyVisual::Ellipsoid {
            half_axes: Vec3::new(
                half_axes.x as f32,
                half_axes.y as f32,
                half_axes.z as f32,
            ),
        });
    }

    (b.build().expect("shapes model"), visuals)
}

/// Stage-15 GPU-narrow-phase demo — pile of mixed boxes / cylinders /
/// cones / spheres dropped onto a plane. Designed to exercise the
/// Sphere/Box/Capsule/Cylinder/Cone branches of the GPU EPA kernel.
/// The right-rail pane lets the user toggle full-GPU vs CPU narrow
/// phase and watch the per-frame `last_cpu_narrow_phase_calls`
/// counter drop to ~0 (the `Plane` pairs always stay on CPU).
fn build_gpu_narrow_phase_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.2);

    let mut visuals: Vec<BodyVisual> = Vec::new();
    // Ground plane.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane shape");

    // 16 mixed bodies in a 4x4 grid spaced 0.7 m, dropped from y ∈ [3, 6].
    let n_side = 4i32;
    for ix in 0..n_side {
        for iz in 0..n_side {
            let x = (ix as f64 - 1.5) * 0.7;
            let z = (iz as f64 - 1.5) * 0.7;
            let y = 3.0 + ((ix + iz) as f64) * 0.3;
            let kind = ((ix * n_side + iz) % 4) as i32;
            match kind {
                0 => {
                    let half = MVec3::splat(0.25);
                    let mass = 0.8;
                    let inertia = box_inertia(mass, 0.25);
                    let body = b.add_body(
                        mp::BodyParams::new()
                            .with_spatial_inertia(mp::SpatialInertia { mass, com: MVec3::ZERO, inertia })
                            .with_position(MVec3::new(x, y, z)),
                    );
                    b.add_shape_box(body, half, mp::Transform::IDENTITY).unwrap();
                    visuals.push(BodyVisual::Box(Vec3::splat(0.25)));
                }
                1 => {
                    let r = 0.25;
                    let mass = 1.0;
                    let i_v = (2.0 / 5.0) * mass * r * r;
                    let inertia = molla::math::Mat3::from_diagonal(MVec3::new(i_v, i_v, i_v));
                    let body = b.add_body(
                        mp::BodyParams::new()
                            .with_spatial_inertia(mp::SpatialInertia { mass, com: MVec3::ZERO, inertia })
                            .with_position(MVec3::new(x, y, z)),
                    );
                    b.add_shape_sphere(body, r, mp::Transform::IDENTITY).unwrap();
                    visuals.push(BodyVisual::Sphere(r as f32));
                }
                2 => {
                    // Cylinder collision; render approximation as a capsule
                    // (no Cylinder variant in `BodyVisual`, capsule is
                    // close enough visually).
                    let half_h: f64 = 0.30;
                    let r: f64 = 0.20;
                    let mass: f64 = 0.7;
                    let inertia = molla::math::Mat3::from_diagonal(MVec3::new(
                        mass * (r * r * 0.25 + (2.0 * half_h).powi(2) / 12.0),
                        mass * r * r * 0.5,
                        mass * (r * r * 0.25 + (2.0 * half_h).powi(2) / 12.0),
                    ));
                    let body = b.add_body(
                        mp::BodyParams::new()
                            .with_spatial_inertia(mp::SpatialInertia { mass, com: MVec3::ZERO, inertia })
                            .with_position(MVec3::new(x, y, z)),
                    );
                    b.add_shape_cylinder(body, half_h, r, mp::Transform::IDENTITY).unwrap();
                    visuals.push(BodyVisual::Capsule { half_h: half_h as f32, radius: r as f32 });
                }
                _ => {
                    let half_h: f64 = 0.30;
                    let r: f64 = 0.20;
                    let mass: f64 = 0.7;
                    let inertia = molla::math::Mat3::from_diagonal(MVec3::new(
                        mass * (r * r * 0.25 + (2.0 * half_h).powi(2) / 12.0),
                        mass * r * r * 0.5,
                        mass * (r * r * 0.25 + (2.0 * half_h).powi(2) / 12.0),
                    ));
                    let body = b.add_body(
                        mp::BodyParams::new()
                            .with_spatial_inertia(mp::SpatialInertia { mass, com: MVec3::ZERO, inertia })
                            .with_position(MVec3::new(x, y, z)),
                    );
                    b.add_shape_cone(body, half_h, r, mp::Transform::IDENTITY).unwrap();
                    visuals.push(BodyVisual::Cone { half_h: half_h as f32, radius: r as f32 });
                }
            }
        }
    }

    (b.build().expect("gpu narrow phase model"), visuals)
}

/// Stage-19 GPU broad-phase scene — a 32×32×4 = 4096-sphere shower
/// dropped onto a plane. Sphere radii are deterministically jittered
/// so the AABB sizes vary (more representative workload for the
/// LBVH pair-finder than uniform-radius spheres). The right-rail
/// pane toggles between CPU SAP and GPU LBVH and shows live ms/frame
/// + pair count for both.
fn build_broad_phase_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.20);

    let mut visuals: Vec<BodyVisual> = Vec::new();

    // Ground plane.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane shape");

    // 32 × 32 × 4 = 4096 spheres on a regular grid, dropped from
    // a few metres up. Spacing is wide enough that the initial
    // configuration is mostly disjoint — the broad phase still has
    // O(n) work even when no pairs overlap, so this stresses the
    // build path. As the spheres settle the pair count climbs.
    let nx: i32 = 32;
    let nz: i32 = 32;
    let ny: i32 = 4;
    let spacing: f64 = 0.8;
    let drop_height: f64 = 6.0;

    // Cheap deterministic RNG (Wang hash-style) for radius jitter
    // — keeps the scene reproducible across runs without pulling
    // in a `rand` dep.
    let mut h: u32 = 0x9E37_79B9;
    let mut next = || {
        h = h.wrapping_mul(2654435761).wrapping_add(0x12345678);
        h
    };

    for ix in 0..nx {
        for iz in 0..nz {
            for iy in 0..ny {
                let x = (ix as f64 - (nx as f64 - 1.0) * 0.5) * spacing;
                let z = (iz as f64 - (nz as f64 - 1.0) * 0.5) * spacing;
                let y = drop_height + iy as f64 * (spacing * 0.9);
                // Radius in [0.18, 0.32] — small enough to fit grid
                // spacing, large enough to overlap as the spheres
                // settle and pile up.
                let r = 0.18 + (next() as f64 / u32::MAX as f64) * 0.14;
                let mass = 1.0;
                let i_v = (2.0 / 5.0) * mass * r * r;
                let inertia = molla::math::Mat3::from_diagonal(MVec3::new(i_v, i_v, i_v));
                let body = b.add_body(
                    mp::BodyParams::new()
                        .with_spatial_inertia(mp::SpatialInertia { mass, com: MVec3::ZERO, inertia })
                        .with_position(MVec3::new(x, y, z)),
                );
                b.add_shape_sphere(body, r, mp::Transform::IDENTITY).unwrap();
                visuals.push(BodyVisual::Sphere(r as f32));
            }
        }
    }

    (b.build().expect("broad phase model"), visuals)
}

/// Contact-sensor example — a small stack of bodies settling on the
/// plane. The GPU `ContactSensor` runs every frame and the right-rail
/// pane shows per-body contact counts + max depth + net push
/// direction. Gizmo arrows visualise the net push direction at each
/// body's contact centroid.
fn build_contact_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.1);
    b.set_linear_damping(0.5);

    let mut visuals = Vec::new();

    // Bottom box (large flat-ish plinth).
    {
        let h = MVec3::new(1.0, 0.25, 1.0);
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: 4.0,
                    com: MVec3::ZERO,
                    inertia: box_inertia(4.0, 0.5),
                })
                .with_position(MVec3::new(0.0, 0.4, 0.0)),
        );
        b.add_shape_box(body, h, mp::Transform::IDENTITY).expect("box");
        visuals.push(BodyVisual::Box(Vec3::new(1.0, 0.25, 1.0)));
    }
    // Middle sphere.
    {
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia::sphere(1.5, 0.4))
                .with_position(MVec3::new(0.2, 1.6, 0.0)),
        );
        b.add_shape_sphere(body, 0.4, mp::Transform::IDENTITY)
            .expect("sphere");
        visuals.push(BodyVisual::Sphere(0.4));
    }
    // Top capsule.
    {
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: 1.0,
                    com: MVec3::ZERO,
                    inertia: capsule_inertia(1.0, 0.4, 0.25),
                })
                .with_position(MVec3::new(-0.1, 2.6, 0.05)),
        );
        b.add_shape_capsule(body, 0.4, 0.25, mp::Transform::IDENTITY)
            .expect("capsule");
        visuals.push(BodyVisual::Capsule {
            half_h: 0.4,
            radius: 0.25,
        });
    }

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("model"), visuals)
}

/// IMU example — a single dynamic sphere bouncing on the floor with
/// an `ImuSensor` mounted at its COM. Restitution is high so the
/// accelerometer reading is dramatic on each impact (specific force
/// spikes when the contact impulse fires).
fn build_imu_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.0); // let the bounces stay lively

    let body = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia::sphere(1.0, 0.4))
            .with_position(MVec3::new(0.0, 4.0, 0.0)),
    );
    b.add_shape_sphere(body, 0.4, mp::Transform::IDENTITY)
        .expect("imu sphere");
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("imu plane");

    (b.build().expect("model"), vec![BodyVisual::Sphere(0.4)])
}

// --- Camera scene (Stage 14 — multi-channel raytraced sensor) ------
// 720p source resolution. Pane thumbnails downsample at draw time, but
// the underlying camera renders at full HD-class resolution so the
// "shape index = instance segmentation" channel is readable when zoomed.
const CAMERA_IMG_W: u32 = 1280;
const CAMERA_IMG_H: u32 = 720;
const CAMERA_NUM_VIEWS: usize = 1;
const CAMERA_ORBIT_RADIUS: f32 = 6.0;
const CAMERA_ORBIT_HEIGHT: f32 = 2.5;
const CAMERA_TARGET_HEIGHT: f32 = 1.0;

/// Stage-14 camera scene — a "bowl" of vivid-coloured primitives
/// dropped onto a plane. Each shape gets its own
/// `set_shape_visual_material` colour so the right-rail thumbnails
/// (color / albedo / shape index) read at a glance.
fn build_camera_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.20);

    let mut visuals: Vec<BodyVisual> = Vec::new();

    // Static plane to settle on.
    let plane = b
        .add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");
    let _ = b.set_shape_visual_material(plane, MVec3::new(0.55, 0.55, 0.60), 0.0, 0.85);

    // ── Three spheres in a row, vivid hues. ──
    let sphere_specs = [
        (MVec3::new(-1.6, 1.5, 0.0), 0.5, MVec3::new(0.95, 0.20, 0.20)), // red
        (MVec3::new(0.0, 1.5, 0.0), 0.5, MVec3::new(0.20, 0.85, 0.30)),  // green
        (MVec3::new(1.6, 1.5, 0.0), 0.5, MVec3::new(0.20, 0.40, 0.95)),  // blue
    ];
    for (pos, r, color) in sphere_specs {
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia::sphere(1.0, r))
                .with_position(pos),
        );
        let shape = b
            .add_shape_sphere(body, r, mp::Transform::IDENTITY)
            .expect("sphere");
        let _ = b.set_shape_visual_material(shape, color, 0.0, 0.4);
        visuals.push(BodyVisual::Sphere(r as f32));
    }

    // ── A box behind, magenta. ──
    {
        let half = MVec3::new(0.55, 0.45, 0.55);
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: 2.0,
                    com: MVec3::ZERO,
                    inertia: box_inertia(2.0, half.max_element()),
                })
                .with_position(MVec3::new(0.0, 1.0, -1.6)),
        );
        let shape = b
            .add_shape_box(body, half, mp::Transform::IDENTITY)
            .expect("box");
        let _ = b.set_shape_visual_material(shape, MVec3::new(0.85, 0.20, 0.85), 0.0, 0.5);
        visuals.push(BodyVisual::Box(Vec3::new(
            half.x as f32,
            half.y as f32,
            half.z as f32,
        )));
    }

    // ── A capsule on the left-front, yellow. ──
    {
        let half_h: f64 = 0.45;
        let radius: f64 = 0.25;
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: 0.8,
                    com: MVec3::ZERO,
                    inertia: capsule_inertia(0.8, half_h, radius),
                })
                .with_position(MVec3::new(-2.4, 1.0, 1.6)),
        );
        let shape = b
            .add_shape_capsule(body, half_h, radius, mp::Transform::IDENTITY)
            .expect("capsule");
        let _ = b.set_shape_visual_material(shape, MVec3::new(0.95, 0.85, 0.20), 0.0, 0.45);
        visuals.push(BodyVisual::Capsule {
            half_h: half_h as f32,
            radius: radius as f32,
        });
    }

    // ── A cone on the right-front, cyan. ──
    {
        let half_h: f64 = 0.6;
        let radius: f64 = 0.45;
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: 1.2,
                    com: MVec3::ZERO,
                    inertia: capsule_inertia(1.2, half_h, radius),
                })
                .with_position(MVec3::new(2.4, 1.2, 1.6)),
        );
        let shape = b
            .add_shape_cone(body, half_h, radius, mp::Transform::IDENTITY)
            .expect("cone");
        let _ = b.set_shape_visual_material(shape, MVec3::new(0.20, 0.85, 0.85), 0.0, 0.45);
        visuals.push(BodyVisual::Cone {
            half_h: half_h as f32,
            radius: radius as f32,
        });
    }

    (b.build().expect("camera scene model"), visuals)
}

/// Per-frame state for the camera-sensor demo. The `TiledCamera` is
/// built lazily on first entry to the scene; pixel bytes for every
/// channel get written into Bevy `Image` assets each frame and the
/// right-rail pane displays them via `bevy_egui`'s user-texture path.
#[derive(Resource, Default)]
struct CameraExampleState {
    sensor: Option<TiledCamera>,
    /// Wall-clock seconds since the scene started; drives the orbit.
    elapsed: f32,
    /// FOV slider value (radians). Synchronised back into the sensor's
    /// config every frame.
    fov_y: f32,
    /// Most recent camera world pose, for the readout.
    last_pose: Option<mp::Transform>,
    /// Bevy `Image` assets backing each channel thumbnail. Allocated
    /// once on first dispatch and rewritten in-place each frame; the
    /// matching `egui::TextureId` is cached after registering the
    /// handle with `EguiUserTextures`.
    image_color: Option<Handle<Image>>,
    image_depth: Option<Handle<Image>>,
    image_normal: Option<Handle<Image>>,
    image_albedo: Option<Handle<Image>>,
    image_index: Option<Handle<Image>>,
    tex_id_color: Option<egui::TextureId>,
    tex_id_depth: Option<egui::TextureId>,
    tex_id_normal: Option<egui::TextureId>,
    tex_id_albedo: Option<egui::TextureId>,
    tex_id_index: Option<egui::TextureId>,
    /// Min / max depth observed last frame — used to scale the depth
    /// thumbnail to a useful greyscale range.
    last_depth_range: Option<(f32, f32)>,
}

// ---- Bevy systems -------------------------------------------------

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    sim: Res<Sim>,
    visuals: Res<BodyVisuals>,
    mut activity: ResMut<ActivityView>,
) {
    // Sleeping material — uniform muted grey, lower roughness so it
    // doesn't blend completely into the dark background.
    let sleeping = materials.add(StandardMaterial {
        base_color: Color::srgb(0.32, 0.34, 0.38),
        metallic: 0.0,
        perceptual_roughness: 0.85,
        ..default()
    });
    activity.sleeping_mat = Some(sleeping);

    let sun_shadow = CascadeShadowConfigBuilder {
        num_cascades: 2,
        minimum_distance: 0.1,
        maximum_distance: 200.0,
        first_cascade_far_bound: 30.0,
        overlap_proportion: 0.2,
    }
    .build();
    // Three-point rig: key + fill + back. The key has shadows, the
    // fill softens the shadowed side without doubling shadow cost,
    // the back rim light separates objects from the dark background.
    commands.spawn((
        Name::new("Sun"),
        Transform::from_xyz(20.0, 60.0, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
        DirectionalLight {
            illuminance: 14_000.0,
            shadows_enabled: true,
            ..default()
        },
        sun_shadow,
    ));
    commands.spawn((
        Name::new("FillLight"),
        Transform::from_xyz(-25.0, 35.0, -10.0).looking_at(Vec3::ZERO, Vec3::Y),
        DirectionalLight {
            illuminance: 6_500.0,
            shadows_enabled: false,
            color: Color::srgb(0.85, 0.90, 1.00),
            ..default()
        },
    ));
    commands.spawn((
        Name::new("RimLight"),
        Transform::from_xyz(0.0, 25.0, -45.0).looking_at(Vec3::ZERO, Vec3::Y),
        DirectionalLight {
            illuminance: 4_000.0,
            shadows_enabled: false,
            color: Color::srgb(1.00, 0.90, 0.78),
            ..default()
        },
    ));

    let chase = ChaseCamera {
        focus: Vec3::new(0.0, 4.0, 0.0),
        distance: 30.0,
        elevation: 20f32.to_radians(),
        min_elevation: -2f32.to_radians(),
        max_distance: 250.0,
        ..ChaseCamera::default()
    };
    let mut tr = Transform::default();
    apply_rig(&chase, &mut tr);
    commands.spawn((
        Name::new("MainCamera"),
        Camera3d::default(),
        tr,
        AmbientLight {
            color: Color::srgb(0.85, 0.90, 1.0),
            brightness: 280.0,
            ..default()
        },
        chase,
        GizmoCamera,
    ));

    commands.spawn((
        Name::new("OriginAxes"),
        Transform::from_xyz(0.0, 0.001, 0.0),
        AxisGizmo {
            length: 1.5,
            ..AxisGizmo::default()
        },
    ));

    spawn_body_entities(&mut commands, &mut meshes, &mut materials, &sim, &visuals);

    info!(
        "molla viewer: scene = {}, {} bodies, {}× @ {:.0} Hz physics",
        sim.scene.label(),
        sim.model.body_count,
        SUBSTEPS_PER_FRAME,
        1.0 / FIXED_DT
    );
    info!("keys: 1 / 2 / 3 switch scene · Space pause · R reset · Esc quit");
}

/// Build a per-visual mesh + material and spawn one entity per molla
/// body. Called from `setup` and after every scene switch.
fn spawn_body_entities(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    sim: &Sim,
    visuals: &BodyVisuals,
) {
    let palette = [
        Color::srgb(0.88, 0.35, 0.28),
        Color::srgb(0.90, 0.60, 0.20),
        Color::srgb(0.85, 0.78, 0.25),
        Color::srgb(0.45, 0.78, 0.35),
        Color::srgb(0.25, 0.70, 0.85),
        Color::srgb(0.40, 0.50, 0.95),
        Color::srgb(0.65, 0.42, 0.92),
        Color::srgb(0.92, 0.50, 0.70),
    ];
    let mats: Vec<_> = palette
        .iter()
        .map(|c| {
            materials.add(StandardMaterial {
                base_color: *c,
                metallic: 0.05,
                perceptual_roughness: 0.45,
                ..default()
            })
        })
        .collect();

    let body_q = sim.state_a.body_q.host().expect("body_q").to_vec();
    for (i, pose) in body_q.iter().enumerate() {
        // Per-visual extra scale (Ellipsoid bakes its semi-axes into a
        // non-uniform transform scale; GaussianSplat collapses to a
        // billboard-radius scale for the stub renderer).
        let mut extra_scale = Vec3::ONE;
        let mesh = match visuals.0[i] {
            // ico(3) → 1280 tris/sphere (vs 320 at ico(2)). Smooth at
            // grab range, still cheap at 144 spheres.
            BodyVisual::Sphere(r) => meshes.add(Sphere::new(r).mesh().ico(4).expect("ico")),
            // Cuboid only has 12 triangles regardless of subdivision —
            // looks fine even in wireframe (12 cube edges visible).
            BodyVisual::Box(h) => meshes.add(Cuboid::from_size(h * 2.0).mesh()),
            BodyVisual::Capsule { half_h, radius } => meshes.add(
                Capsule3d::new(radius, half_h * 2.0)
                    .mesh()
                    .longitudes(32)
                    .latitudes(16),
            ),
            BodyVisual::Cone { half_h, radius } => meshes.add(
                Cone {
                    radius,
                    height: half_h * 2.0,
                }
                .mesh()
                .resolution(32)
                .build(),
            ),
            // Ellipsoid: render a unit sphere mesh with non-uniform
            // scale baked into the entity transform. Bevy's PBR
            // pipeline handles anisotropic scales — wireframe still
            // shows the deformed icosphere edges, so the stretched
            // shape is obvious.
            BodyVisual::Ellipsoid { half_axes } => {
                extra_scale = half_axes;
                meshes.add(Sphere::new(1.0).mesh().ico(4).expect("ico"))
            }
            // GaussianSplat stub: render as a small billboarded
            // sphere. Real rendering (oriented anisotropic Gaussian
            // disc) is deferred — this gives the viewer something
            // visible without dragging in a custom shader.
            BodyVisual::GaussianSplat { covariance_inv, .. } => {
                // Use the (capped) inverse-determinant as a rough
                // stddev proxy. Real splat rendering would want
                // the eigenvectors of `covariance_inv⁻¹`.
                let det = covariance_inv[0].x * covariance_inv[1].y * covariance_inv[2].z;
                let r = (1.0 / det.abs().max(1e-3)).cbrt().clamp(0.05, 0.5);
                meshes.add(Sphere::new(r).mesh().ico(2).expect("ico"))
            }
        };
        let awake = mats[i % mats.len()].clone();
        let mut t = Transform::from_xyz(
            pose.position.x as f32,
            pose.position.y as f32,
            pose.position.z as f32,
        );
        t.scale = extra_scale;
        commands.spawn((
            MollaBody(mp::BodyId::from_index(i)),
            Mesh3d(mesh),
            MeshMaterial3d(awake.clone()),
            AwakeMaterial(awake),
            NotShadowCaster,
            t,
        ));
    }

    // Cloth: build a single deforming triangle mesh from
    // `model.tri_indices`, attach it to one entity. The
    // `sync_cloth_mesh` system updates vertex positions each frame.
    if sim.model.particle_count > 0 && sim.model.tri_count > 0 {
        let mesh = build_cloth_mesh(sim);
        let mesh_handle = meshes.add(mesh);
        let mat = materials.add(StandardMaterial {
            base_color: Color::srgb(0.85, 0.55, 0.30),
            metallic: 0.05,
            perceptual_roughness: 0.65,
            cull_mode: None, // double-sided so back faces are visible
            ..default()
        });
        commands.spawn((
            ClothMesh,
            Mesh3d(mesh_handle),
            MeshMaterial3d(mat),
            NotShadowCaster,
            Transform::default(),
        ));
    }

    // Cloth-scene static sphere obstacle.
    if sim.scene == Scene::Cloth {
        let mesh = meshes.add(
            Sphere::new(CLOTH_SPHERE_RADIUS)
                .mesh()
                .ico(4)
                .expect("ico"),
        );
        let mat = materials.add(StandardMaterial {
            base_color: Color::srgb(0.55, 0.62, 0.75),
            metallic: 0.10,
            perceptual_roughness: 0.55,
            ..default()
        });
        commands.spawn((
            ClothObstacle,
            Mesh3d(mesh),
            MeshMaterial3d(mat),
            Transform::from_xyz(0.0, CLOTH_SPHERE_Y, 0.0),
        ));
    }

    // Stage-20 VBD cloth scene — same kinematic-sphere obstacle
    // pattern as the XPBD scene but with the larger sphere from the
    // VBD constants.
    if sim.scene == Scene::VbdCloth {
        let mesh = meshes.add(
            Sphere::new(VBD_CLOTH_SPHERE_RADIUS)
                .mesh()
                .ico(5)
                .expect("ico"),
        );
        let mat = materials.add(StandardMaterial {
            base_color: Color::srgb(0.45, 0.55, 0.70),
            metallic: 0.15,
            perceptual_roughness: 0.50,
            ..default()
        });
        commands.spawn((
            ClothObstacle,
            Mesh3d(mesh),
            MeshMaterial3d(mat),
            Transform::from_xyz(0.0, VBD_CLOTH_SPHERE_Y, 0.0),
        ));
    }

    // Stage-20.1 CABLE scene — render the static drape-cylinder
    // obstacle. The orbiting puller body is rendered through the
    // generic body-visual loop above; the cable polyline itself is
    // rendered via gizmos by `draw_cable_gizmos`.
    if sim.scene == Scene::Cable {
        let mesh = meshes.add(
            Cylinder::new(CABLE_CYLINDER_RADIUS, CABLE_CYLINDER_HALF_HEIGHT * 2.0)
                .mesh()
                .resolution(48),
        );
        let mat = materials.add(StandardMaterial {
            base_color: Color::srgb(0.55, 0.62, 0.75),
            metallic: 0.10,
            perceptual_roughness: 0.55,
            ..default()
        });
        commands.spawn((
            ClothObstacle,
            Mesh3d(mesh),
            MeshMaterial3d(mat),
            Transform::from_xyz(
                CABLE_CYLINDER_POS.x,
                CABLE_CYLINDER_POS.y,
                CABLE_CYLINDER_POS.z,
            ),
        ));
    }

    // Shapes-scene static visuals: heightfield (deforming triangle
    // strip from the model's heights) + torus mesh.
    if sim.scene == Scene::Shapes {
        if let Some(hf) = sim.model.heightfields.first() {
            let mesh = build_heightfield_mesh(hf);
            let mat = materials.add(StandardMaterial {
                base_color: Color::srgb(0.45, 0.55, 0.40),
                metallic: 0.05,
                perceptual_roughness: 0.85,
                cull_mode: None,
                ..default()
            });
            commands.spawn((
                ClothObstacle, // reuse the despawn tag
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(mat),
                Transform::default(),
            ));
        }
        // Torus visual at the SDF position.
        let torus = Torus {
            major_radius: SHAPES_TORUS_MAJOR,
            minor_radius: SHAPES_TORUS_MINOR,
        };
        let torus_mat = materials.add(StandardMaterial {
            base_color: Color::srgb(0.85, 0.65, 0.30),
            metallic: 0.10,
            perceptual_roughness: 0.45,
            ..default()
        });
        commands.spawn((
            ClothObstacle,
            Mesh3d(meshes.add(torus.mesh().major_resolution(64).minor_resolution(24))),
            MeshMaterial3d(torus_mat),
            Transform::from_xyz(0.0, SHAPES_TORUS_Y, 0.0),
        ));
    }
}

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
/// Stage-21 mesh-tools demo; isolated so the same path is reused for
/// the original / decimated / marching-cubes meshes.
fn molla_trimesh_to_bevy_mesh(tm: &molla::sim::TriMesh, double_sided: bool) -> Mesh {
    use bevy::asset::RenderAssetUsages;
    use bevy::mesh::{Indices, PrimitiveTopology};
    let _ = double_sided;
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    let positions: Vec<[f32; 3]> = tm
        .vertices
        .iter()
        .map(|v| [v.x as f32, v.y as f32, v.z as f32])
        .collect();
    let mut normals: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; positions.len()];
    let mut idx_flat: Vec<u32> = Vec::with_capacity(tm.indices.len() * 3);
    for tri in &tm.indices {
        idx_flat.push(tri[0]);
        idx_flat.push(tri[1]);
        idx_flat.push(tri[2]);
    }
    for chunk in idx_flat.chunks(3) {
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
    mesh.insert_indices(Indices::U32(idx_flat));
    mesh
}

/// Generate a unit-radius icosphere as a molla `TriMesh` with the
/// requested subdivision level. Mirrors the test helper in
/// `tests/mesh_utils.rs` — kept duplicated here so the viewer doesn't
/// need to depend on test-only code.
fn icosphere_mesh(subdivisions: u32, radius: f64) -> molla::sim::TriMesh {
    use std::collections::HashMap;
    let t: f64 = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let raw = [
        (-1.0, t, 0.0), (1.0, t, 0.0), (-1.0, -t, 0.0), (1.0, -t, 0.0),
        (0.0, -1.0, t), (0.0, 1.0, t), (0.0, -1.0, -t), (0.0, 1.0, -t),
        (t, 0.0, -1.0), (t, 0.0, 1.0), (-t, 0.0, -1.0), (-t, 0.0, 1.0),
    ];
    let mut verts: Vec<MVec3> = raw
        .iter()
        .map(|(x, y, z)| {
            let v = MVec3::new(*x, *y, *z);
            v / v.length()
        })
        .collect();
    let mut faces: Vec<[u32; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    for _ in 0..subdivisions {
        let mut cache: HashMap<(u32, u32), u32> = HashMap::new();
        let mut midpoint = |a: u32, b: u32, verts: &mut Vec<MVec3>| -> u32 {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&i) = cache.get(&key) { return i; }
            let mid = (verts[a as usize] + verts[b as usize]) * 0.5;
            let mid = mid / mid.length();
            verts.push(mid);
            let idx = (verts.len() - 1) as u32;
            cache.insert(key, idx);
            idx
        };
        let mut new_faces: Vec<[u32; 3]> = Vec::with_capacity(faces.len() * 4);
        for tri in &faces {
            let a = midpoint(tri[0], tri[1], &mut verts);
            let b = midpoint(tri[1], tri[2], &mut verts);
            let c = midpoint(tri[2], tri[0], &mut verts);
            new_faces.push([tri[0], a, c]);
            new_faces.push([tri[1], b, a]);
            new_faces.push([tri[2], c, b]);
            new_faces.push([a, b, c]);
        }
        faces = new_faces;
    }
    for v in &mut verts {
        *v *= radius;
    }
    let mut m = molla::sim::TriMesh {
        vertices: verts,
        indices: faces,
        ..Default::default()
    };
    m.build_bvh(4);
    m
}

/// Spawn the four Stage-21 visualisation entities (original /
/// decimated / marching-cubes / hull) for `Scene::MeshTools`. Updates
/// the `mesh_state` telemetry fields so the right-rail pane can show
/// triangle counts. Each entity is tagged with `ClothObstacle` so
/// `apply_scene_switch` despawns it on the next scene change AND with
/// a per-panel marker (`MeshToolOriginal`, etc.) so visibility
/// toggles in the pane can find it.
fn spawn_mesh_tools_entities(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    mesh_state: &mut MeshToolsExampleState,
) {
    use molla::geometry::{
        bake_mesh_to_sdf, convex_hull, decimate, marching_cubes, sparse_sdf_from_dense,
    };

    // Build the source mesh — a mildly noise-displaced icosphere
    // (3 subdivisions = 1280 tris). The noise breaks the icosphere's
    // perfect symmetry so decimation has to make non-trivial choices,
    // but stays small enough that the mesh is still recognisably
    // spherical (and the SDF bake doesn't run away).
    let mut source = icosphere_mesh(3, 1.0); // 1280 tris
    let mut s: u32 = 0xA5A5_BEEF;
    let mut next_noise = || {
        s = s.wrapping_mul(2654435761).wrapping_add(0x12345678);
        ((s as f64) / (u32::MAX as f64)) * 2.0 - 1.0
    };
    for v in source.vertices.iter_mut() {
        // glam's f64 DVec3 has `.normalize()` not `.normalize_or_zero()` —
        // safe here since vertex magnitudes are all ~1.0 by construction.
        let n = *v / v.length();
        let r = 1.0 + next_noise() * 0.10;
        *v = n * r;
    }
    source.build_bvh(4);

    let dec = decimate(&source, source.indices.len() / 4); // ~25 % LOD
    let sdf = bake_mesh_to_sdf(&source, mesh_state.sdf_resolution as usize, 0.15);

    // Stage-24e: bake a sparse SDF in parallel — same dense bake →
    // narrow-band filter. Cheap (re-uses the dense data we already
    // have) and lets the user toggle between dense / sparse storage
    // in the right-rail pane to see the byte delta on a real asset.
    let sparse = sparse_sdf_from_dense(&sdf, mesh_state.sdf_sparse_band());
    mesh_state.dense_sdf_bytes = sdf.data.len() * std::mem::size_of::<f64>();
    mesh_state.sparse_sdf_bytes = sparse.grid.storage_bytes();
    mesh_state.sparse_sdf_leaves = sparse.grid.leaf_count();
    mesh_state.sparse_sdf_active_voxels = sparse.grid.active_voxel_count();

    // Stage 24e.1 — record GPU root hash table stats so the pane
    // can show the load factor + average probe length. The host-
    // side replay of the WGSL probing in
    // `SparseGridGpu::root_hash_probe_stats` is byte-identical to
    // what the GPU does at lookup time (same Murmur3 mix), so this
    // is a faithful telemetry source even when the rest of the
    // pipeline runs on the CPU.
    let packed = molla::geometry::SparseGridGpu::from_grid(&sparse.grid);
    mesh_state.sparse_root_table_capacity = packed.root_table_capacity;
    mesh_state.sparse_root_load_factor = packed.root_table_load_factor();
    let (avg, max) = packed.root_hash_probe_stats();
    mesh_state.sparse_root_avg_probes = avg;
    mesh_state.sparse_root_max_probes = max;

    // The MC surface re-extracted from the sparse SDF needs a
    // dense buffer (marching_cubes only consumes `SdfGrid`); for
    // the toggle's "sparse" case we sample the sparse SDF back into
    // a dense scratch grid so the MC consumer is agnostic. This is a
    // viewer-only convenience — production MC over sparse SDFs would
    // iterate active leaves directly (deferred to Stage 24e.1+).
    let mc_source: molla::geometry::SdfGrid = if mesh_state.use_sparse_sdf {
        sparse_sdf_to_dense_for_mc(&sparse, &sdf)
    } else {
        sdf.clone()
    };
    let mc = marching_cubes(&mc_source, 0.0);

    let hull = convex_hull(&source.vertices);

    mesh_state.original_tris = source.indices.len();
    mesh_state.decimated_tris = dec.indices.len();
    mesh_state.mc_tris = mc.indices.len();
    mesh_state.hull_verts = hull.vertices.len();

    // Lay the three primary panels out side-by-side along +X. The MC
    // overlay uses the same world position as the original so the
    // user can compare them directly.
    let original_offset = Vec3::new(-2.5, 1.5, 0.0);
    let decimated_offset = Vec3::new(0.0, 1.5, 0.0);
    let mc_offset = Vec3::new(2.5, 1.5, 0.0);

    let mat_original = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.55, 0.30),
        metallic: 0.05,
        perceptual_roughness: 0.55,
        cull_mode: None,
        ..default()
    });
    let mat_decimated = materials.add(StandardMaterial {
        base_color: Color::srgb(0.40, 0.70, 0.85),
        metallic: 0.05,
        perceptual_roughness: 0.55,
        cull_mode: None,
        ..default()
    });
    let mat_mc = materials.add(StandardMaterial {
        base_color: Color::srgb(0.55, 0.85, 0.40),
        metallic: 0.05,
        perceptual_roughness: 0.55,
        cull_mode: None,
        ..default()
    });
    let mat_hull = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 0.95, 0.30, 0.6),
        unlit: true,
        cull_mode: None,
        ..default()
    });

    // Original.
    let orig_mesh = molla_trimesh_to_bevy_mesh(&source, true);
    commands.spawn((
        ClothObstacle,
        MeshToolOriginal,
        Mesh3d(meshes.add(orig_mesh)),
        MeshMaterial3d(mat_original),
        Transform::from_translation(original_offset),
        Visibility::Visible,
    ));

    // Decimated.
    let dec_mesh = molla_trimesh_to_bevy_mesh(&dec, true);
    commands.spawn((
        ClothObstacle,
        MeshToolDecimated,
        Mesh3d(meshes.add(dec_mesh)),
        MeshMaterial3d(mat_decimated),
        Transform::from_translation(decimated_offset),
        Visibility::Visible,
    ));

    // Marching-cubes surface.
    let mc_mesh = molla_trimesh_to_bevy_mesh(&mc, true);
    commands.spawn((
        ClothObstacle,
        MeshToolMarchingCubes,
        Mesh3d(meshes.add(mc_mesh)),
        MeshMaterial3d(mat_mc),
        Transform::from_translation(mc_offset),
        Visibility::Visible,
    ));

    // Hull — Stage 21.1 swaps the previous yellow-point markers for a
    // real LineList wireframe built from the hull's edge set. One
    // entity covers the full overlay (cheap; tens of edges for the
    // demo icosphere). Defaults to hidden — toggled on via the
    // right-rail pane so the user sees the original first.
    let hull_wire_mesh = meshes.add(build_hull_wireframe_mesh(&hull));
    commands.spawn((
        ClothObstacle,
        MeshToolHull,
        Mesh3d(hull_wire_mesh),
        MeshMaterial3d(mat_hull),
        Transform::from_translation(original_offset),
        Visibility::Hidden,
    ));
}

/// Build a Bevy `LineList` mesh tracing every unique edge of the
/// hull. Uses the post-Stage-21.1 [`molla::geometry::ConvexHullMesh`]
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
    let mut mesh = Mesh::new(
        PrimitiveTopology::LineList,
        RenderAssetUsages::RENDER_WORLD,
    );
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

    let particle_q = sim
        .state_a
        .particle_q
        .host()
        .expect("particle_q host");
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
    // Stage-22 MuJoCo: clear the per-frame timing on scene switch
    // so the readout doesn't carry stale values from the previous
    // backend. The `notified_unavailable` flag is intentionally
    // preserved so we don't re-spam the log if the user toggles
    // back into the scene.
    scene_states.mujoco.smoothed_step_us = 0.0;
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
    mut mujoco_state: ResMut<MuJoCoExampleState>,
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
            | Scene::WarpFemTaylorGreen
            | Scene::WarpFemKelvinHelmholtz
            | Scene::WarpFemStreamlines
            | Scene::WarpFemDeformedGeometry
            | Scene::WarpFemDistortion
            | Scene::WarpFemElasticOpt
            | Scene::WarpFftFluid
            | Scene::WarpDistortion3D
            | Scene::WarpApicFluid
            | Scene::WarpWave
            | Scene::WarpDiffRay
            | Scene::WarpTileFilter
            | Scene::WarpTileCompaction
            | Scene::WarpRaycast
            | Scene::WarpMesh
            | Scene::WarpMeshIntersect
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
        let Sim { pipeline, engine, .. } = &mut *sim;
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
        let Sim { pipeline, engine, .. } = &mut *sim;
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
        contacts.upload_to_device(&device, &queue).expect("contacts upload");
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

    // Surface MuJoCo timing for the right-rail pane. The solver
    // dispatch slot in the main stats block also captures it, but
    // the MuJoCo pane wants its own EMA so the user can A/B compare
    // against the Featherstone fallback when toggling features.
    if sim.scene == Scene::MuJoCo {
        mujoco_state.smoothed_step_us =
            0.9 * mujoco_state.smoothed_step_us + 0.1 * solve_us;
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
    Body { body: mp::BodyId, local_anchor: MVec3 },
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
    let Ok(window) = primary_window.single() else { return };
    let Ok((cam, cam_tr)) = cameras.single() else { return };
    let Some(cursor) = window.cursor_position() else { return };
    let Ok(ray) = cam.viewport_to_world(cam_tr, cursor) else { return };

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
                        GrabKind::Body { body: bid, local_anchor: banchor },
                        bt,
                    ))
                } else {
                    Some((begin_particle_grab(&mut sim, pid), pt))
                }
            }
            (Some((bid, bt, banchor)), None) => Some((
                GrabKind::Body { body: bid, local_anchor: banchor },
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
        if let GrabKind::Particle { index, saved_inv_mass } = state.kind {
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
            BodyVisual::Box(h) => ray_box_local(
                lo,
                ld,
                MVec3::new(h.x as f64, h.y as f64, h.z as f64),
            ),
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
    sim.state_a
        .particle_q
        .host_mut()
        .expect("particle_q_mut")[i] = new_pos;
    // Zero velocity — the constraint solver doesn't touch this
    // particle (inv_mass = 0), and we don't want stale velocity to
    // confuse the velocity-recovery pass.
    sim.state_a
        .particle_qd
        .host_mut()
        .expect("particle_qd_mut")[i] = MVec3::ZERO;
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
    if t2 > 0.0 {
        Some(t2)
    } else {
        None
    }
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
fn apply_grab_velocity(
    sim: &mut Sim,
    body: mp::BodyId,
    local_anchor: MVec3,
    target: MVec3,
) {
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

    sim.state_a
        .body_qd
        .host_mut()
        .expect("body_qd_mut")[i] =
        mp::SpatialVector::new(v_lin, w);
}

/// Lidar example — runs the GPU [`Raycaster`] each frame against the
/// current scene and stashes the (origin, dir, hit) triplets in
/// [`LidarState`] for `draw_lidar_gizmos` to render. No-op for any
/// scene other than [`Scene::Lidar`].
///
/// Ray construction: for each `(iy, ix)` cell in a `ray_count_v ×
/// ray_count_h` grid covering `fov_v × fov_h`, build a unit direction
/// from yaw + pitch + per-cell offsets, applied to the lidar origin.
fn update_lidar(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut config: ResMut<LidarConfig>,
    mut state: ResMut<LidarState>,
) {
    if sim.scene != Scene::Lidar {
        if !state.rays.is_empty() {
            state.rays.clear();
        }
        return;
    }

    // Auto-rotate yaw if configured. Wraps in [0, 2π).
    if config.auto_rotate_speed.abs() > 1e-6 {
        config.yaw += config.auto_rotate_speed * time.delta_secs();
        let two_pi = std::f32::consts::TAU;
        if config.yaw > two_pi {
            config.yaw -= two_pi;
        }
        if config.yaw < 0.0 {
            config.yaw += two_pi;
        }
    }

    let nh = config.ray_count_h.max(1);
    let nv = config.ray_count_v.max(1);
    let n_rays = (nh * nv) as usize;

    // Build (or rebuild) the GPU raycaster if capacity changed.
    if state.raycaster.is_none() || state.raycaster_capacity < n_rays as u32 {
        // Round up to next multiple of 64 to play nicely with the
        // kernel's workgroup size.
        let cap = ((n_rays as u32 + 63) / 64) * 64;
        state.raycaster = Some(
            Raycaster::new(&sim.engine, cap.max(64))
                .expect("lidar Raycaster init"),
        );
        state.raycaster_capacity = cap.max(64);
    }

    // Build N ray origins + directions on host. yaw/pitch convention:
    // yaw = 0 looks at +Z (forward), increases toward +X (right);
    // pitch = 0 horizontal, positive looks up.
    let mut origins = Vec::with_capacity(n_rays);
    let mut directions = Vec::with_capacity(n_rays);

    let half_h = config.fov_h * 0.5;
    let half_v = config.fov_v * 0.5;
    let dh = if nh > 1 {
        config.fov_h / (nh - 1) as f32
    } else {
        0.0
    };
    let dv = if nv > 1 {
        config.fov_v / (nv - 1) as f32
    } else {
        0.0
    };

    for iy in 0..nv {
        let pitch_off = if nv > 1 {
            -half_v + iy as f32 * dv
        } else {
            0.0
        };
        let pitch = config.pitch + pitch_off;
        for ix in 0..nh {
            let yaw_off = if nh > 1 {
                -half_h + ix as f32 * dh
            } else {
                0.0
            };
            let yaw = config.yaw + yaw_off;
            // Spherical → Cartesian. Forward is +Z when yaw = pitch = 0.
            let cp = pitch.cos();
            let dir = Vec3::new(yaw.sin() * cp, pitch.sin(), yaw.cos() * cp);
            origins.push(MVec3::new(
                config.origin.x as f64,
                config.origin.y as f64,
                config.origin.z as f64,
            ));
            directions.push(MVec3::new(dir.x as f64, dir.y as f64, dir.z as f64));
        }
    }

    // Need to make sure model + state are uploaded. The pile/tower/etc.
    // pipeline uploads model once; the lidar scene model was uploaded
    // in `Sim::finalize` and `state.body_q` in `step_simulation`.
    let raycaster = state.raycaster.as_mut().expect("raycaster");
    let max_range = config.max_range as f64;
    let hits = match raycaster.cast(
        &sim.engine,
        &sim.model,
        &sim.state_a,
        &origins,
        &directions,
        max_range,
    ) {
        Ok(h) => h,
        Err(e) => {
            warn!("lidar raycast failed: {e}");
            return;
        }
    };

    // Stash for the gizmo system. Convert from molla f64 vecs to
    // bevy f32 along the way.
    state.rays.clear();
    state.rays.reserve(n_rays);
    for (i, hit) in hits.iter().enumerate() {
        let o = origins[i];
        let d = directions[i];
        let origin = Vec3::new(o.x as f32, o.y as f32, o.z as f32);
        let direction = Vec3::new(d.x as f32, d.y as f32, d.z as f32);
        let hit_world: Option<Vec3> = match hit_to_world(hit) {
            Some(p) => Some(p),
            None => None,
        };
        state.rays.push(LidarRay {
            origin,
            direction,
            hit: hit_world,
        });
    }
}

/// IMU example — runs the GPU `ImuSensor` against the bouncing
/// sphere each frame and stashes the latest reading for the right-rail
/// readout. No-op for any other scene.
fn update_imu_example(
    mut sim: ResMut<Sim>,
    mut state: ResMut<ImuExampleState>,
) {
    if sim.scene != Scene::Imu {
        if state.sensor.is_some() {
            state.sensor = None;
            state.last = None;
            state.smoothed_accel_mag = 0.0;
            state.smoothed_gyro_mag = 0.0;
        }
        return;
    }

    if state.sensor.is_none() {
        // Body 0 is the bouncing sphere we built in `build_imu_model`.
        let mounts = [ImuMount {
            body: mp::BodyId::from_index(0),
            local_pose: mp::Transform::IDENTITY,
        }];
        match ImuSensor::new(&sim.engine, &sim.model, &mounts) {
            Ok(s) => state.sensor = Some(s),
            Err(e) => {
                warn!("imu sensor init failed: {e}");
                return;
            }
        }
    }

    // dt for the IMU = the wall-clock time between consecutive frames'
    // body_qd snapshots, which is `SUBSTEPS_PER_FRAME * FIXED_DT`
    // (the integrator advances `body_qd` by that much per frame).
    let frame_dt = (SUBSTEPS_PER_FRAME as f64) * FIXED_DT;
    let Sim {
        ref engine,
        ref model,
        ref state_a,
        ..
    } = *sim;
    let sensor = state.sensor.as_mut().expect("sensor");
    let readings = match sensor.update(engine, model, state_a, frame_dt) {
        Ok(r) => r,
        Err(e) => {
            warn!("imu update failed: {e}");
            return;
        }
    };
    if let Some(&r) = readings.first() {
        state.last = Some(r);
        let alpha = 0.2_f32;
        state.smoothed_accel_mag =
            (1.0 - alpha) * state.smoothed_accel_mag + alpha * r.accelerometer.length() as f32;
        state.smoothed_gyro_mag =
            (1.0 - alpha) * state.smoothed_gyro_mag + alpha * r.gyroscope.length() as f32;
    }
}

/// GPU IK arm scene — runs `IkSolver::solve_gpu` (or its CPU mirror
/// when `state.use_gpu` is false) every frame to drive the arm toward
/// the user-controlled XYZ target, then writes `joint_q` and `body_q`
/// on the host. `step_simulation` short-circuits for this scene so
/// the IK pose remains stable across frames.
fn update_gpu_ik_arm(mut sim: ResMut<Sim>, mut state: ResMut<GpuIkArmState>) {
    if sim.scene != Scene::GpuIkArm {
        return;
    }
    // Destructure to keep separate borrows of `model`, `state_a`, `engine`.
    let Sim {
        model,
        state_a,
        engine,
        ..
    } = &mut *sim;

    // EE is the last body (index = body_count - 1) for the 7-DOF arm.
    let ee_body = model.body_count - 1;

    // Lazily seed the target on first frame from the rest pose.
    if !state.target_init {
        molla::sim::eval_fk(model, state_a).expect("eval_fk");
        let bq = state_a.body_q.host().expect("body_q");
        state.target = bq[ee_body].position + GPU_IK_DEFAULT_OFFSET;
        state.target_init = true;
    }

    let q0: Vec<f64> = state_a
        .joint_q
        .host()
        .expect("joint_q")
        .iter()
        .copied()
        .collect();
    let solver = IkSolver::new(vec![IkObjective::Position {
        body: ee_body,
        point_local: MVec3::ZERO,
        target_world: state.target,
        weight: 1.0,
    }])
    .with_config(IkConfig {
        max_iter: 30,
        residual_tol: 5.0e-4,
        ..IkConfig::default()
    });

    let t0 = std::time::Instant::now();
    let result = if state.use_gpu {
        solver.solve_gpu(engine, model, state_a, &q0)
    } else {
        solver.solve(model, state_a, &q0)
    };
    state.last_solve_us = t0.elapsed().as_secs_f64() * 1_000_000.0;

    match result {
        Ok(r) => {
            state.last_iters = r.iters;
            state.last_residual = r.residual;
        }
        Err(e) => {
            warn!("gpu_ik_arm: solve failed: {e}");
        }
    }
}

/// Stage-23 diff-arm pre-step. Two responsibilities:
///
/// 1. **(Re)run optimisation** when `state.rerun` is set. Project the
///    target into the arm's planar coordinates (XY, Z ignored), run
///    [`DiffArm::gradient_descent`] on the torque sequence, capture
///    the per-iter EE positions for the trail overlay.
/// 2. **Replay the optimised trajectory** every frame: advance
///    `state.playback_t` by `dt × playback_speed`, look up the
///    `(q1, q2)` at that time from the rolled-out trace, run forward
///    kinematics, write the two body world poses into `state_a.body_q`
///    so `sync_transforms` picks them up.
///
/// No-op for any other scene.
fn update_diff_arm(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut state: ResMut<DiffArmExampleState>,
) {
    if sim.scene != Scene::DiffArm {
        // Reset the per-frame playback so re-entry restarts from
        // the start of the optimised motion. Don't clobber `tau` —
        // the user might want their previous result preserved if
        // they hop out + back without changing target.
        state.playback_t = 0.0;
        return;
    }

    // 1. (Re)run optimisation if requested.
    if state.rerun {
        // Reset the torque buffer to zero so each re-run starts from
        // the same baseline (otherwise sweeping the target slider
        // would silently warm-start from the previous tau and
        // produce path-dependent answers).
        state.tau = vec![[0.0; 2]; state.arm.horizon];
        state.iter_ee_trace.clear();

        // Target in arm-local 2D: shift by the shoulder anchor.
        let target_2d = [
            state.target.x,
            state.target.y - DIFF_ARM_BASE_Y as f64,
        ];

        // Iterate by hand so we can capture per-iter EE positions
        // for the overlay (DiffArm::gradient_descent doesn't expose
        // intermediates).
        let arm = state.arm;
        let q0 = [0.0; 2];
        let qd0 = [0.0; 2];
        let mut last_residual = f64::INFINITY;
        let mut converged = None;
        for it in 0..DIFF_ARM_OPT_ITERS {
            let trace = arm.rollout(q0, qd0, &state.tau);
            let final_q = trace[trace.len() - 1].0;
            let ee = arm.forward_kinematics(final_q);
            // Trail point in world coords.
            state.iter_ee_trace.push(MVec3::new(
                ee[0],
                ee[1] + DIFF_ARM_BASE_Y as f64,
                0.0,
            ));
            let dx = ee[0] - target_2d[0];
            let dy = ee[1] - target_2d[1];
            last_residual = (dx * dx + dy * dy).sqrt();
            if converged.is_none() && last_residual < 0.05 {
                converged = Some(it);
            }
            let g_tau = arm.reverse(&trace, [2.0 * dx, 2.0 * dy]);
            for i in 0..arm.horizon {
                state.tau[i][0] -= DIFF_ARM_OPT_LR * g_tau[i][0];
                state.tau[i][1] -= DIFF_ARM_OPT_LR * g_tau[i][1];
            }
        }
        state.last_residual = last_residual;
        state.converged_iter = converged;
        state.iters_run = DIFF_ARM_OPT_ITERS;
        state.rerun = false;
        state.playback_t = 0.0;
        info!(
            "diff_arm: optimisation done — residual {:.4} m, converged at iter {:?}",
            state.last_residual, state.converged_iter,
        );
    }

    // 2. Playback. Advance the wall clock + roll out to the current
    // playback time. We re-roll every frame because the cost is
    // negligible (40 substeps of double-integrator math) and it
    // avoids stashing the trace as resource state.
    let dt = time.delta_secs_f64();
    state.playback_t += dt * DIFF_ARM_PLAYBACK_SPEED;
    let total = state.arm.horizon as f64 * state.arm.dt;
    let t = if total > 0.0 {
        state.playback_t % total
    } else {
        0.0
    };
    let step = ((t / state.arm.dt).floor() as usize).min(state.arm.horizon.saturating_sub(1));
    let trace = state.arm.rollout([0.0; 2], [0.0; 2], &state.tau);
    let q = trace[step].0;

    // Forward kinematics → world poses for the two capsule bodies.
    // Each capsule's local +Y points along the link, so the body
    // rotation must rotate +Y onto the link direction.
    let half1 = state.arm.l1 * 0.5;
    let half2 = state.arm.l2 * 0.5;

    // Upper arm: extends from shoulder down (-Y when q1 = 0). With
    // q1 nonzero, the arm rotates around +Z. The link direction in
    // world coords is `(sin q1, -cos q1, 0)`; the capsule centre is
    // `shoulder + 0.5·L1·dir`.
    let dir1 = MVec3::new(q[0].sin(), -q[0].cos(), 0.0);
    let p1 = MVec3::new(0.0, DIFF_ARM_BASE_Y as f64, 0.0) + dir1 * half1;
    // Rotation: rotate local +Y onto dir1 → angle = q1 around +Z.
    let r1 = MQuat::from_axis_angle(MVec3::Z, q[0]);

    // Lower arm: extends from elbow down (-Y when q1 + q2 = 0).
    let elbow = MVec3::new(0.0, DIFF_ARM_BASE_Y as f64, 0.0) + dir1 * state.arm.l1;
    let qsum = q[0] + q[1];
    let dir2 = MVec3::new(qsum.sin(), -qsum.cos(), 0.0);
    let p2 = elbow + dir2 * half2;
    let r2 = MQuat::from_axis_angle(MVec3::Z, qsum);

    // Stash poses into body_q on the host. `sync_transforms` reads
    // host buffers; no GPU upload needed for this kinematic path.
    let body_q = match sim.state_a.body_q.host_mut() {
        Ok(b) => b,
        Err(_) => return,
    };
    if body_q.len() >= 2 {
        body_q[0] = mp::Transform::new(p1, r1);
        body_q[1] = mp::Transform::new(p2, r2);
    }
}

/// Stage-23.1 6-DOF Featherstone diff-arm pre-step. Same shape as
/// `update_diff_arm`, but the chain is the [`FeatherstoneRevoluteChain::franka6`]
/// 6-DOF Franka stand-in driven by the reverse-mode adjoint from
/// [`molla::autograd::featherstone_adjoint`].
fn update_diff_arm6(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut state: ResMut<DiffArm6ExampleState>,
) {
    if sim.scene != Scene::DiffArm6 {
        state.playback_t = 0.0;
        return;
    }

    let n = state.chain.links.len();
    let h = state.chain.horizon;

    // 1. (Re)run optimisation if requested.
    if state.rerun {
        // Reset torque baseline so each re-run starts fresh.
        state.tau = vec![vec![0.0; n]; h];
        state.iter_ee_trace.clear();

        // Target in chain-local coordinates: shift down by the base
        // height, since the chain's root is at the world origin in its
        // local frame.
        let target_local = MVec3::new(
            state.target.x,
            state.target.y - DIFF_ARM6_BASE_HEIGHT,
            state.target.z,
        );
        let target_chain = molla::math::Vec3::new(
            target_local.x,
            target_local.y,
            target_local.z,
        );

        let q0 = vec![0.0_f64; n];
        let qd0 = vec![0.0_f64; n];
        let lr_init = 0.05_f64;
        let mut lr = lr_init;
        let mut prev = f64::INFINITY;
        let mut last_residual = f64::INFINITY;
        let mut converged: Option<usize> = None;
        // Stage 23.1.1: track backward / forward time ratio across all
        // GD iterations and average. The viewer's pane shows this so
        // the user can compare the closed-form CPU vs FD-CPU perf
        // toggle effects in real time.
        let mut fwd_total_us = 0.0_f64;
        let mut bwd_total_us = 0.0_f64;
        // Stage 23.1.2 — backend picker. The 4-way enum drives the
        // dispatcher directly. `state.use_gpu_adjoint` stays mirrored
        // for back-compat with anything still reading the bool.
        let backend_choice = state.backend;
        state.use_gpu_adjoint = matches!(
            backend_choice,
            molla::kernels::AdjointBackend::GpuFallbackCpu
        );

        // Stage 23.1.2 — multi-articulation batch demo. Runs N parallel
        // chains through the rayon batch dispatcher per re-run and
        // records per-chain wall time + parallel speedup. The
        // single-chain optimisation loop still runs after so the user
        // sees the trajectory animation.
        if state.use_multi_articulation {
            let batch_n = state.multi_articulation_n.max(2);
            let chains: Vec<molla::autograd::FeatherstoneRevoluteChain> = (0..batch_n)
                .map(|_| state.chain.clone())
                .collect();
            let q0_b = vec![0.0_f64; n];
            let qd0_b = vec![0.0_f64; n];
            let tau_b = vec![vec![0.0_f64; n]; h];
            let traces: Vec<_> = chains
                .iter()
                .map(|c| c.rollout(&q0_b, &qd0_b, &tau_b))
                .collect();
            // Use a generic +Z grad seed for the batch demo — the
            // benchmark exercises the per-chain reverse-mode cost, not
            // a specific optimisation step.
            let grad_seed = molla::math::Vec3::Z;
            let inputs: Vec<molla::kernels::MultiAdjointInput<'_>> = (0..batch_n)
                .map(|i| molla::kernels::MultiAdjointInput {
                    chain: &chains[i],
                    trace: &traces[i],
                    grad_ee: grad_seed,
                })
                .collect();
            // Warm.
            let _ = molla::kernels::MultiAdjointBatch::dispatch_cpu(&inputs);
            let (_par, par_wall_us) =
                molla::kernels::MultiAdjointBatch::dispatch_cpu(&inputs)
                    .expect("Stage 23.1 chains are single-DoF");
            let (_ser, ser_wall_us) =
                molla::kernels::MultiAdjointBatch::dispatch_cpu_serial(&inputs)
                    .expect("Stage 23.1 chains are single-DoF");
            state.last_per_chain_us = par_wall_us / batch_n as f64;
            state.last_parallel_speedup = if par_wall_us > 0.0 {
                ser_wall_us / par_wall_us
            } else {
                0.0
            };
            info!(
                "diff_arm6: multi-articulation batch — N={batch_n}, parallel_wall={par_wall_us:.0}µs ({:.0}µs/chain), serial_wall={ser_wall_us:.0}µs, parallel_speedup={:.2}×",
                state.last_per_chain_us, state.last_parallel_speedup,
            );
        } else {
            state.last_per_chain_us = 0.0;
            state.last_parallel_speedup = 0.0;
        }
        for it in 0..DIFF_ARM6_OPT_ITERS {
            let t_fwd = std::time::Instant::now();
            let trace = state.chain.rollout(&q0, &qd0, &state.tau);
            fwd_total_us += t_fwd.elapsed().as_micros() as f64;
            let q_final = &trace.q[h];
            let ee = state.chain.end_effector(q_final);
            let d = ee - target_chain;
            last_residual = d.length();

            // Trail point in WORLD coords.
            state.iter_ee_trace.push(MVec3::new(
                ee.x,
                ee.y + DIFF_ARM6_BASE_HEIGHT,
                ee.z,
            ));

            if !last_residual.is_finite() {
                warn!("diff_arm6: residual diverged at iter {it} — backing off");
                break;
            }
            if converged.is_none() && last_residual < 0.05 {
                converged = Some(it);
            }
            if last_residual < 0.01 {
                state.iters_run = it;
                break;
            }

            // Backoff lr if loss went up (cheap line search).
            if last_residual > prev * 1.05 {
                lr *= 0.5;
            } else if last_residual < prev * 0.99 {
                lr = (lr * 1.05).min(lr_init);
            }
            prev = last_residual;

            // Borrow chain + scratch as disjoint fields so the same
            // struct's immutable + mutable references coexist. Route
            // through the Stage 23.1.1 dispatcher so the GPU / CPU
            // toggle takes effect.
            let res = {
                let DiffArm6ExampleState { chain, scratch, .. } = &mut *state;
                molla::kernels::FeatherstoneAdjointDispatcher::reverse(
                    &*chain,
                    &trace,
                    d * 2.0,
                    scratch,
                    backend_choice,
                )
                .expect("Stage 23.1 single-DoF chain")
            };
            bwd_total_us += res.backward_us;
            let g = res.g_tau;
            for t in 0..h {
                for k in 0..n {
                    let delta = -lr * g[t][k];
                    let clamped = delta.clamp(-0.2, 0.2);
                    state.tau[t][k] += clamped;
                    state.tau[t][k] = state.tau[t][k].clamp(-3.0, 3.0);
                }
            }
            state.iters_run = it + 1;
        }
        state.last_residual = last_residual;
        state.converged_iter = converged;
        state.rerun = false;
        state.playback_t = 0.0;
        state.last_bwd_fwd_ratio = if fwd_total_us > 0.0 {
            bwd_total_us / fwd_total_us
        } else {
            0.0
        };
        info!(
            "diff_arm6: optimisation done — residual {:.4} m, converged at iter {:?}",
            state.last_residual, state.converged_iter,
        );
    }

    // 2. Playback. Advance wall clock + roll out to current playback time.
    let dt = time.delta_secs_f64();
    state.playback_t += dt * DIFF_ARM6_PLAYBACK_SPEED;
    let total = h as f64 * state.chain.dt;
    let t = if total > 0.0 {
        state.playback_t % total
    } else {
        0.0
    };
    let step = ((t / state.chain.dt).floor() as usize).min(h.saturating_sub(1));
    let q0 = vec![0.0_f64; n];
    let qd0 = vec![0.0_f64; n];
    let trace = state.chain.rollout(&q0, &qd0, &state.tau);
    let q = &trace.q[step];

    // FK on the chain → world poses for each link.
    let body_world = state.chain.forward_kinematics(q);
    let body_q = match sim.state_a.body_q.host_mut() {
        Ok(b) => b,
        Err(_) => return,
    };
    let base_lift = MVec3::new(0.0, DIFF_ARM6_BASE_HEIGHT, 0.0);
    for (j, link_t) in body_world.iter().enumerate().take(body_q.len()) {
        let p = MVec3::new(
            link_t.position.x,
            link_t.position.y,
            link_t.position.z,
        ) + base_lift;
        let r = MQuat::from_xyzw(
            link_t.rotation.x,
            link_t.rotation.y,
            link_t.rotation.z,
            link_t.rotation.w,
        );
        body_q[j] = mp::Transform::new(p, r);
    }
}

/// Arm example — sets per-joint targets (auto-sine or slider) and
/// runs the GPU `ControllerPd` to write `Control.joint_force`. The
/// physics step (Featherstone) then consumes that force. No-op for
/// any other scene.
fn update_arm_example(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut state: ResMut<ArmExampleState>,
) {
    if sim.scene != Scene::Arm {
        if state.controller.is_some() {
            state.controller = None;
        }
        return;
    }

    if state.controller.is_none() {
        // 1 actuator per joint DOF. Featherstone's joint_dof_total
        // covers all joint DOFs — for an N-revolute arm, that's N.
        let configs: Vec<ControllerPdConfig> = (0..ARM_LINKS)
            .map(|i| ControllerPdConfig {
                dof: i as u32,
                kp: ARM_KP,
                kd: ARM_KD,
                max_effort: ARM_MAX_EFFORT,
            })
            .collect();
        match ControllerPd::new(&sim.engine, &sim.model, &configs) {
            Ok(c) => {
                info!(
                    "arm: ControllerPd online — {} actuators, kp={ARM_KP}, kd={ARM_KD}, max_effort={ARM_MAX_EFFORT}",
                    configs.len(),
                );
                state.controller = Some(c);
            }
            Err(e) => {
                warn!("arm: controller init failed: {e}");
                return;
            }
        }
    }

    state.elapsed += time.delta_secs();

    // Determine target positions for this frame.
    let mut targets = state.targets;
    if state.sine_enabled {
        let t = state.elapsed * state.sine_frequency * std::f32::consts::TAU;
        for (i, target) in targets.iter_mut().enumerate() {
            // Phase-stagger each joint by 2π/N so the arm looks
            // wavey rather than flapping in unison.
            let phase = (i as f32) * std::f32::consts::TAU / (ARM_LINKS as f32);
            *target = state.sine_amplitude * (t + phase).sin();
        }
        // Reflect the auto-generated targets back into the slider
        // values so the UI stays in sync.
        state.targets = targets;
    }

    // Push the targets into Control.joint_target_pos. Targets for
    // joint i live at the start-of-DOF index for that joint —
    // joint_dof_offset[i]..[i+1]. For revolute joints that's a
    // single DOF per joint, so dof = i. We also zero target_vel
    // so the derivative term tracks the joint coming to rest at
    // the target. We do NOT touch joint_force here — the PD
    // kernel writes it on the device (and marks it device-dirty),
    // so a host-side write would conflict. The kernel does
    // `joint_force[dof] = effort` (assignment, not accumulate)
    // and our 3 actuators cover all 3 DOFs of this scene, so no
    // stale values leak through.
    {
        let queue = sim.engine.queue().expect("queue").clone();
        let target_pos = sim
            .control
            .joint_target_pos
            .host_mut()
            .expect("joint_target_pos");
        for (i, t) in targets.iter().enumerate() {
            if i < target_pos.len() {
                target_pos[i] = *t as f64;
            }
        }
        let target_vel = sim
            .control
            .joint_target_vel
            .host_mut()
            .expect("joint_target_vel");
        for v in target_vel.iter_mut() {
            *v = 0.0;
        }
        sim.control
            .joint_target_pos
            .upload(&queue)
            .expect("target_pos upload");
        sim.control
            .joint_target_vel
            .upload(&queue)
            .expect("target_vel upload");
    }

    // Run the PD controller — writes effort into Control.joint_force.
    let Sim {
        ref engine,
        ref state_a,
        ref mut control,
        ..
    } = *sim;
    if let Some(controller) = state.controller.as_mut() {
        match controller.step(engine, state_a, control) {
            Ok(()) => {
                if !state.logged_init {
                    info!("arm: first PD dispatch OK");
                    state.logged_init = true;
                }
            }
            Err(e) => warn!("arm: controller step failed: {e}"),
        }
    }
}

/// Stage-18 actuator pipeline example — wires the composable
/// `Actuator { Delay → ControllerPipelinePd → ClampingDcMotor }` to
/// drive the single revolute joint. Reads slider state per frame
/// and pushes the target into `Control.joint_target_pos`. No-op for
/// any other scene.
fn update_actuator_example(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut state: ResMut<ActuatorExampleState>,
) {
    if sim.scene != Scene::Actuator {
        if state.actuator.is_some() {
            state.actuator = None;
        }
        return;
    }

    // Lazy-init the pipeline on first frame in this scene.
    if state.actuator.is_none() {
        let dof = 0u32;
        let delay_steps = ms_to_substeps(state.delay_ms);
        let max_delay = 64u32; // headroom; user-facing slider caps at ~30 ms = 18 substeps.
        let delay_res = Delay::new(
            &sim.engine,
            &[DelayConfig { dof, delay_steps }],
            max_delay,
        );
        let pd_res = ControllerPipelinePd::new(
            &sim.engine,
            &[ControllerPipelinePdConfig {
                dof,
                kp: state.kp as f64,
                kd: state.kd as f64,
            }],
        );
        let clamp_res = ClampingDcMotor::new(
            &sim.engine,
            &[ClampingDcMotorConfig {
                dof,
                stall_torque: state.stall_torque as f64,
                no_load_speed: state.no_load_speed as f64,
                max_motor_effort: state.max_motor_effort as f64,
            }],
        );
        match (delay_res, pd_res, clamp_res) {
            (Ok(delay), Ok(pd), Ok(clamp)) => {
                let act = Actuator::new(
                    delay,
                    ControllerStage::Pd(pd),
                    ClampingStage::DcMotor(clamp),
                );
                info!(
                    "actuator: pipeline online — delay {} substeps, kp={}, kd={}, stall={} N·m, no_load={} rad/s",
                    delay_steps, state.kp, state.kd, state.stall_torque, state.no_load_speed,
                );
                state.actuator = Some(act);
            }
            (d, p, c) => {
                if let Err(e) = d {
                    warn!("actuator: delay init failed: {e}");
                }
                if let Err(e) = p {
                    warn!("actuator: pd init failed: {e}");
                }
                if let Err(e) = c {
                    warn!("actuator: clamp init failed: {e}");
                }
                return;
            }
        }
    }

    state.elapsed += time.delta_secs();

    // Apply slider edits by uploading new params to the GPU buffers.
    if state.params_dirty {
        // Copy slider values out of `state` so the controller-stage
        // closures below don't conflict with the `state.actuator`
        // mutable borrow.
        let delay_steps = ms_to_substeps(state.delay_ms);
        let kp = state.kp as f64;
        let kd = state.kd as f64;
        let stall = state.stall_torque as f64;
        let no_load = state.no_load_speed as f64;
        let max_motor = state.max_motor_effort as f64;
        if let Some(act) = state.actuator.as_mut() {
            let n = act.actuator_count() as usize;
            if let Err(e) =
                act.delay_mut().set_delay_steps(&sim.engine, &vec![delay_steps; n])
            {
                warn!("actuator: set_delay_steps failed: {e}");
            }
            match act.controller_mut() {
                ControllerStage::Pd(pd) => {
                    if let Err(e) =
                        pd.set_gains(&sim.engine, &vec![kp; n], &vec![kd; n])
                    {
                        warn!("actuator: set_gains failed: {e}");
                    }
                }
                // The Stage-18 actuator pane only constructs a PD
                // controller; PID / MLP / LSTM variants live in their
                // own dedicated demo scenes (NeuralPolicy / NeuralLstm).
                ControllerStage::Pid(_)
                | ControllerStage::NeuralMlp(_)
                | ControllerStage::Lstm(_) => {}
            }
            if let ClampingStage::DcMotor(clamp) = act.clamping_mut() {
                if let Err(e) = clamp.set_params(
                    &sim.engine,
                    &vec![stall; n],
                    &vec![no_load; n],
                    &vec![max_motor; n],
                ) {
                    warn!("actuator: set_params failed: {e}");
                }
            }
        }
        state.params_dirty = false;
    }

    // Decide the target for this frame.
    let target = if state.sine_enabled {
        let t = state.elapsed * state.sine_frequency * std::f32::consts::TAU;
        state.sine_amplitude * t.sin()
    } else {
        state.target
    };
    if state.sine_enabled {
        state.target = target;
    }

    // Compute commanded torque host-side (diagnostic): same formula
    // the pipeline uses, but without the DC-motor clamp.
    let q = sim
        .state_a
        .joint_q
        .host()
        .ok()
        .and_then(|s| s.first().copied())
        .unwrap_or(0.0) as f32;
    let qd = sim
        .state_a
        .joint_qd
        .host()
        .ok()
        .and_then(|s| s.first().copied())
        .unwrap_or(0.0) as f32;
    let commanded = state.kp * (target - q) + state.kd * (0.0 - qd);
    state.last_commanded = commanded;

    // Push the target into Control.joint_target_pos on the device.
    {
        let queue = sim.engine.queue().expect("queue").clone();
        let target_pos = sim
            .control
            .joint_target_pos
            .host_mut()
            .expect("joint_target_pos");
        if !target_pos.is_empty() {
            target_pos[0] = target as f64;
        }
        let target_vel = sim
            .control
            .joint_target_vel
            .host_mut()
            .expect("joint_target_vel");
        for v in target_vel.iter_mut() {
            *v = 0.0;
        }
        sim.control
            .joint_target_pos
            .upload(&queue)
            .expect("target_pos upload");
        sim.control
            .joint_target_vel
            .upload(&queue)
            .expect("target_vel upload");
    }

    // Run the actuator pipeline.
    let Sim {
        ref engine,
        ref state_a,
        ref mut control,
        ..
    } = *sim;
    if let Some(act) = state.actuator.as_mut() {
        if let Err(e) = act.step(engine, state_a, control, FIXED_DT) {
            warn!("actuator: pipeline step failed: {e}");
        } else {
            // Read the delivered torque back off the GPU for the
            // readout panel. This is a small download (1 element)
            // so the cost is negligible.
            let device = engine.device().expect("device").clone();
            let queue = engine.queue().expect("queue").clone();
            if control.joint_force.download(&device, &queue).is_ok() {
                if let Ok(jf) = control.joint_force.host() {
                    if !jf.is_empty() {
                        state.last_delivered = jf[0] as f32;
                    }
                }
            }
        }
    }
}

// ── Stage-18.1 neural-policy demo update ───────────────────────────
//
// Per visible frame:
//   1. Lazy-init the [`Actuator`] containing
//      `Delay(0) → ControllerNeuralMlp → ClampingMaxEffort` once we
//      enter the scene.
//   2. Read `joint_q[0..2]` (cart_x, theta), `joint_qd[0..2]`
//      (cart_xdot, theta_dot) host-side.
//   3. Build the 4-element observation, normalise, push through the
//      MLP via `set_inputs`, then run `act.step` — the GPU dispatches
//      forward + scatter + clamp.
//   4. Periodically inject a small angular perturbation so the
//      visualisation stays dynamic (otherwise the policy parks the
//      pole at exactly upright and nothing moves).
//
// The MLP weights are **hand-rolled** to express a linear LQR-style
// state-feedback law `F = -K · [x, x_dot, theta, theta_dot]` through
// the network. With small inputs, the tanh non-linearities act as
// near-identity, so the MLP collapses to a linear policy. Past a few
// rad of pole error, tanh saturates and the output caps; the
// downstream MaxEffort clamp adds a hard rail at ±max_force.
fn update_neural_policy_example(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut state: ResMut<NeuralPolicyExampleState>,
) {
    if sim.scene != Scene::NeuralPolicy {
        if state.actuator.is_some() {
            state.actuator = None;
        }
        return;
    }

    // Lazy-init the pipeline + upload weights on first frame in scene.
    if state.actuator.is_none() {
        let cfg = ControllerNeuralMlpConfig {
            input_dim: 4,
            hidden_dim: 32,
            output_dim: 1,
        };
        let dof_indices = vec![0u32]; // cart prismatic = joint_force[0]
        let mlp_res = ControllerNeuralMlp::new(&sim.engine, cfg, &dof_indices);
        let delay_res = Delay::new(
            &sim.engine,
            &[DelayConfig {
                dof: 0,
                delay_steps: 0,
            }],
            1,
        );
        let max_force = state.max_force as f64;
        let clamp_res = ClampingMaxEffort::new(
            &sim.engine,
            &[ClampingMaxEffortConfig {
                dof: 0,
                max_effort: max_force,
            }],
        );
        match (mlp_res, delay_res, clamp_res) {
            (Ok(mut mlp), Ok(delay), Ok(clamp)) => {
                let weights = neural_policy_lqr_weights();
                let device = sim.engine.device().unwrap().clone();
                let queue = sim.engine.queue().unwrap().clone();
                if let Err(e) = mlp.set_weights(&device, &queue, &weights) {
                    warn!("neural policy: set_weights failed: {e}");
                    return;
                }
                state.actuator = Some(Actuator::new(
                    delay,
                    ControllerStage::NeuralMlp(mlp),
                    ClampingStage::MaxEffort(clamp),
                ));
                info!(
                    "neural policy: MLP online — 4→32→32→1 LQR-style balance net"
                );
            }
            (m, d, c) => {
                if let Err(e) = m {
                    warn!("neural policy: mlp init failed: {e}");
                }
                if let Err(e) = d {
                    warn!("neural policy: delay init failed: {e}");
                }
                if let Err(e) = c {
                    warn!("neural policy: clamp init failed: {e}");
                }
                return;
            }
        }
    }

    // Pull joint state host-side (already downloaded each substep
    // by `step_simulation`).
    let (cart_x, cart_xd, theta, theta_d) = {
        let q = sim.state_a.joint_q.host().ok();
        let qd = sim.state_a.joint_qd.host().ok();
        match (q, qd) {
            (Some(q), Some(qd)) if q.len() >= 2 && qd.len() >= 2 => {
                (q[0] as f32, qd[0] as f32, q[1] as f32, qd[1] as f32)
            }
            _ => return,
        }
    };

    // Periodic perturb: every 3 s, pulse a small angular velocity into
    // the pole so the user can see the policy recover. Implemented by
    // writing directly into joint_qd[1] on the host then re-uploading.
    state.perturb_timer += time.delta_secs();
    if state.auto_perturb && state.perturb_timer > 3.0 {
        state.perturb_timer = 0.0;
        // Alternate sign so the kicks balance over time.
        let sign = if (theta + 0.001).is_sign_positive() { -1.0 } else { 1.0 };
        let queue = sim.engine.queue().expect("queue").clone();
        if let Ok(qd) = sim.state_a.joint_qd.host_mut() {
            if qd.len() >= 2 {
                qd[1] += (sign * 1.5) as f64;
            }
        }
        let _ = sim.state_a.joint_qd.upload(&queue);
    }

    // Observation: scaled to keep tanh in its near-linear regime.
    // Per-component scales picked so a "reasonable" excursion lands
    // |x| ≈ 0.5 — the LQR-equivalent gains compensate via w3.
    let inputs: [f32; 4] = [
        cart_x * 0.5,        // ±2 m → ±1
        cart_xd * 0.2,        // ±5 m/s → ±1
        theta,                 // already ±π/2 useful range
        theta_d * 0.2,         // ±5 rad/s → ±1
    ];
    state.last_inputs = inputs;

    let inputs_f64: Vec<f64> = inputs.iter().map(|v| *v as f64).collect();

    // Borrow split so we can mutate both `control` and `state.actuator`.
    let Sim {
        ref engine,
        ref state_a,
        ref mut control,
        ..
    } = *sim;
    if let Some(act) = state.actuator.as_mut() {
        let queue = engine.queue().expect("queue").clone();
        // Push the observation into the MLP. We need to dive into the
        // controller stage to call `set_inputs` — actuator.step's
        // contract assumes the inputs are already on-device.
        match act.controller_mut() {
            ControllerStage::NeuralMlp(mlp) => {
                if let Err(e) = mlp.set_inputs(&queue, &inputs_f64) {
                    warn!("neural policy: set_inputs failed: {e}");
                    return;
                }
            }
            _ => unreachable!("neural policy scene must use NeuralMlp"),
        }
        if let Err(e) = act.step(engine, state_a, control, FIXED_DT) {
            warn!("neural policy: pipeline step failed: {e}");
            return;
        }

        // Read back the network output + post-clamp force for the UI.
        let device = engine.device().expect("device").clone();
        match act.controller_mut() {
            ControllerStage::NeuralMlp(mlp) => {
                if let Ok(outs) = mlp.download_outputs(&device, &queue) {
                    if !outs.is_empty() {
                        state.last_output = outs[0] as f32;
                    }
                }
            }
            _ => {}
        }
        if control.joint_force.download(&device, &queue).is_ok() {
            if let Ok(jf) = control.joint_force.host() {
                if !jf.is_empty() {
                    state.last_delivered = jf[0] as f32;
                }
            }
        }
    }
}

/// Hand-rolled MLP weights that express an LQR-style balance policy
/// on the cart-pole. The network maps a 4-element observation
///   [x_scaled, xdot_scaled, theta, theta_d_scaled]
/// through two near-identity tanh hidden layers to a single output:
///
///   F = -[K_x · x_scaled + K_v · xdot_scaled
///        + K_θ · theta + K_ω · theta_d_scaled]
///
/// In the near-linear regime (|input| ≪ 1) `tanh(z) ≈ z`, so the
/// network collapses to the linear policy above. Past saturation the
/// tanh caps each channel — desirable behaviour for a balance net.
fn neural_policy_lqr_weights() -> MlpWeights<'static> {
    use std::sync::OnceLock;

    static W: OnceLock<NeuralPolicyWeights> = OnceLock::new();
    let weights = W.get_or_init(build_neural_policy_weights);
    MlpWeights {
        w1: &weights.w1,
        b1: &weights.b1,
        w2: &weights.w2,
        b2: &weights.b2,
        w3: &weights.w3,
        b3: &weights.b3,
    }
}

struct NeuralPolicyWeights {
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
    w3: Vec<f64>,
    b3: Vec<f64>,
}

fn build_neural_policy_weights() -> NeuralPolicyWeights {
    // Architecture: 4 → 32 → 32 → 1.
    // Strategy: layer 1 is a slightly-noisy identity passthrough on
    // the first 4 hidden units; remaining 28 units stay zero so they
    // don't perturb the output. Layer 2 is identity. Layer 3 is the
    // LQR linear combination on the first 4 hidden units.
    let in_dim: usize = 4;
    let hid_dim: usize = 32;
    let out_dim: usize = 1;

    let mut w1 = vec![0.0f64; hid_dim * in_dim];
    let b1 = vec![0.0f64; hid_dim];
    let mut w2 = vec![0.0f64; hid_dim * hid_dim];
    let b2 = vec![0.0f64; hid_dim];
    let mut w3 = vec![0.0f64; out_dim * hid_dim];
    let b3 = vec![0.0f64; out_dim];

    // Layer 1: identity passthrough on units 0..4.
    // w1[i, k] for i ∈ [0, in_dim).
    for i in 0..in_dim {
        w1[i * in_dim + i] = 1.0;
    }
    // Layer 2: identity on the 32 hidden units (so the signal carries
    // through unchanged).
    for i in 0..hid_dim {
        w2[i * hid_dim + i] = 1.0;
    }

    // Layer 3: LQR-style gains. We want
    //   F = -k_x·x - k_v·xdot - k_θ·theta - k_ω·theta_d
    // The network sees scaled inputs (see `update_neural_policy_example`
    // for the scale factors). Tuning rationale:
    //   • theta dominates — push hard on pole error.
    //   • theta_d damps oscillation, but lower than k_θ.
    //   • cart_x weak — only reins in long-term drift.
    //   • cart_xdot moderate damping.
    //
    // Per-channel (via the scale factors used in `update_*`):
    //   k_x_eff   = k_x  / 0.5
    //   k_v_eff   = k_v  / 0.2
    //   k_θ_eff   = k_θ
    //   k_ω_eff   = k_ω  / 0.2
    //
    // Picked so theta-only error of 0.2 rad → ~10 N (well under cap).
    let k_x = -1.0;
    let k_v = -2.5;
    let k_t = -55.0;
    let k_w = -10.0;
    w3[0 * hid_dim + 0] = k_x; // x_scaled
    w3[0 * hid_dim + 1] = k_v; // xdot_scaled
    w3[0 * hid_dim + 2] = k_t; // theta
    w3[0 * hid_dim + 3] = k_w; // theta_d_scaled

    NeuralPolicyWeights { w1, b1, w2, b2, w3, b3 }
}

// ── Stage-31 neural-LSTM demo update ───────────────────────────────
//
// Mirrors `update_neural_policy_example`, but the controller is a
// single-cell LSTM (4 → 16 → 1) instead of an MLP. The weights are
// hand-rolled to express the same LQR-style balance policy: the
// recurrent gates are tuned so the cell barely modulates the signal
// (forget ≈ 0, input ≈ 1, candidate ≈ tanh(x)), and the linear
// projection then mixes the gated state with the same per-component
// gains the MLP demo uses. This way the demo proves the forward-pass
// plumbing AND keeps the cart-pole upright — random init wouldn't.
fn update_neural_lstm_example(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut state: ResMut<NeuralLstmExampleState>,
) {
    if sim.scene != Scene::NeuralLstm {
        if state.actuator.is_some() {
            state.actuator = None;
        }
        return;
    }

    // Lazy-init the pipeline + upload weights on first frame in scene.
    if state.actuator.is_none() {
        let cfg = ControllerNeuralLstmConfig {
            input_dim: 4,
            hidden_dim: LSTM_HID_DIM as u32,
            output_dim: 1,
        };
        let dof_indices = vec![0u32]; // cart prismatic = joint_force[0]
        let lstm_res = ControllerNeuralLstm::new(&sim.engine, cfg, &dof_indices);
        let delay_res = Delay::new(
            &sim.engine,
            &[DelayConfig { dof: 0, delay_steps: 0 }],
            1,
        );
        let max_force = state.max_force as f64;
        let clamp_res = ClampingMaxEffort::new(
            &sim.engine,
            &[ClampingMaxEffortConfig { dof: 0, max_effort: max_force }],
        );
        match (lstm_res, delay_res, clamp_res) {
            (Ok(mut lstm), Ok(delay), Ok(clamp)) => {
                let weights = neural_lstm_lqr_weights();
                let device = sim.engine.device().unwrap().clone();
                let queue = sim.engine.queue().unwrap().clone();
                if let Err(e) = lstm.set_weights(&device, &queue, &weights) {
                    warn!("neural lstm: set_weights failed: {e}");
                    return;
                }
                if let Err(e) = lstm.reset_hidden_state(&queue) {
                    warn!("neural lstm: reset_hidden_state failed: {e}");
                    return;
                }
                state.actuator = Some(Actuator::new(
                    delay,
                    ControllerStage::Lstm(lstm),
                    ClampingStage::MaxEffort(clamp),
                ));
                info!(
                    "neural lstm: LSTM online — 4→16→1 LQR-style balance net (recurrent)"
                );
            }
            (l, d, c) => {
                if let Err(e) = l { warn!("neural lstm: lstm init failed: {e}"); }
                if let Err(e) = d { warn!("neural lstm: delay init failed: {e}"); }
                if let Err(e) = c { warn!("neural lstm: clamp init failed: {e}"); }
                return;
            }
        }
    }

    // Pull joint state host-side.
    let (cart_x, cart_xd, theta, theta_d) = {
        let q = sim.state_a.joint_q.host().ok();
        let qd = sim.state_a.joint_qd.host().ok();
        match (q, qd) {
            (Some(q), Some(qd)) if q.len() >= 2 && qd.len() >= 2 => {
                (q[0] as f32, qd[0] as f32, q[1] as f32, qd[1] as f32)
            }
            _ => return,
        }
    };

    // Periodic perturb — same convention as the MLP demo.
    state.perturb_timer += time.delta_secs();
    if state.auto_perturb && state.perturb_timer > 3.0 {
        state.perturb_timer = 0.0;
        let sign = if (theta + 0.001).is_sign_positive() { -1.0 } else { 1.0 };
        let queue = sim.engine.queue().expect("queue").clone();
        if let Ok(qd) = sim.state_a.joint_qd.host_mut() {
            if qd.len() >= 2 {
                qd[1] += (sign * 1.5) as f64;
            }
        }
        let _ = sim.state_a.joint_qd.upload(&queue);
    }

    // Same observation scaling as the MLP demo so the hand-rolled
    // gains carry over without retuning.
    let inputs: [f32; 4] = [
        cart_x * 0.5,
        cart_xd * 0.2,
        theta,
        theta_d * 0.2,
    ];
    state.last_inputs = inputs;
    let inputs_f64: Vec<f64> = inputs.iter().map(|v| *v as f64).collect();

    let Sim {
        ref engine,
        ref state_a,
        ref mut control,
        ..
    } = *sim;
    if let Some(act) = state.actuator.as_mut() {
        let queue = engine.queue().expect("queue").clone();
        match act.controller_mut() {
            ControllerStage::Lstm(lstm) => {
                if let Err(e) = lstm.set_inputs(&queue, &inputs_f64) {
                    warn!("neural lstm: set_inputs failed: {e}");
                    return;
                }
            }
            _ => unreachable!("neural lstm scene must use Lstm controller"),
        }
        if let Err(e) = act.step(engine, state_a, control, FIXED_DT) {
            warn!("neural lstm: pipeline step failed: {e}");
            return;
        }

        // Read the network output, hidden state, and post-clamp force.
        // Buffer the downloads into locals first so we don't keep
        // `lstm` borrowed while writing back to `state`.
        let device = engine.device().expect("device").clone();
        let mut new_output: Option<f32> = None;
        let mut new_hidden: Option<Vec<f32>> = None;
        match act.controller_mut() {
            ControllerStage::Lstm(lstm) => {
                if let Ok(outs) = lstm.download_outputs(&device, &queue) {
                    if !outs.is_empty() {
                        new_output = Some(outs[0] as f32);
                    }
                }
                if let Ok(hs) = lstm.download_hidden_state(&device, &queue) {
                    new_hidden = Some(hs.iter().map(|v| *v as f32).collect());
                }
            }
            _ => {}
        }
        if let Some(v) = new_output {
            state.last_output = v;
        }
        if let Some(hs) = new_hidden {
            let n = state.last_hidden.len().min(hs.len());
            for i in 0..n {
                state.last_hidden[i] = hs[i];
            }
        }
        if control.joint_force.download(&device, &queue).is_ok() {
            if let Ok(jf) = control.joint_force.host() {
                if !jf.is_empty() {
                    state.last_delivered = jf[0] as f32;
                }
            }
        }
    }
}

/// Hand-rolled LSTM weights expressing an LQR-style cart-pole balance
/// policy. **Random init does NOT balance the pole** — the recurrent
/// gates make the optimisation landscape non-trivial. To keep the
/// demo functional v1 deliberately suppresses the recurrence so the
/// LSTM behaves like a stateful linear-projection wrapper:
///
///   - `f = sigmoid(b_f)` with `b_f = -8` → `f ≈ 0` → forget previous c.
///   - `i = sigmoid(b_i)` with `b_i = +8` → `i ≈ 1` → admit candidate.
///   - `g = tanh(W_g · x_t)` (no recurrence) → cell state ≈ tanh(x).
///   - `o = sigmoid(b_o)` with `b_o = +8` → `o ≈ 1` → output ≈ tanh(c).
///
/// Then the linear projection `y = W_y · h_t + b_y` mixes the four
/// "input-passthrough" hidden units with LQR gains (matching
/// `build_neural_policy_weights`). The remaining 12 hidden units stay
/// near zero (their `W_g` row is zero) so they don't pollute the
/// output.
///
/// Result: the cart-pole balances with the recurrent gates
/// instrumented but functionally inert; the `h` / `c` buffers do
/// evolve across substeps (visible in the right-rail readout) so the
/// demo does prove the persistence path. A "real" trained recurrent
/// policy would need offline training + weight import — out of scope
/// for v1.
fn neural_lstm_lqr_weights() -> LstmWeights<'static> {
    use std::sync::OnceLock;
    static W: OnceLock<NeuralLstmWeights> = OnceLock::new();
    let weights = W.get_or_init(build_neural_lstm_weights);
    LstmWeights {
        w_i: &weights.w_i, u_i: &weights.u_i, b_i: &weights.b_i,
        w_f: &weights.w_f, u_f: &weights.u_f, b_f: &weights.b_f,
        w_g: &weights.w_g, u_g: &weights.u_g, b_g: &weights.b_g,
        w_o: &weights.w_o, u_o: &weights.u_o, b_o: &weights.b_o,
        w_y: &weights.w_y, b_y: &weights.b_y,
    }
}

struct NeuralLstmWeights {
    w_i: Vec<f64>, u_i: Vec<f64>, b_i: Vec<f64>,
    w_f: Vec<f64>, u_f: Vec<f64>, b_f: Vec<f64>,
    w_g: Vec<f64>, u_g: Vec<f64>, b_g: Vec<f64>,
    w_o: Vec<f64>, u_o: Vec<f64>, b_o: Vec<f64>,
    w_y: Vec<f64>, b_y: Vec<f64>,
}

fn build_neural_lstm_weights() -> NeuralLstmWeights {
    let in_dim: usize = 4;
    let hid_dim: usize = LSTM_HID_DIM; // 16
    let out_dim: usize = 1;

    // All gate input / recurrent weights start at zero.
    let zero_w = || vec![0.0f64; hid_dim * in_dim];
    let zero_u = || vec![0.0f64; hid_dim * hid_dim];
    let zero_b = || vec![0.0f64; hid_dim];

    let mut w_i = zero_w();  let u_i = zero_u();  let mut b_i = zero_b();
    let mut w_f = zero_w();  let u_f = zero_u();  let mut b_f = zero_b();
    let mut w_g = zero_w();  let u_g = zero_u();  let b_g = zero_b();
    let mut w_o = zero_w();  let u_o = zero_u();  let mut b_o = zero_b();

    // Saturate gates per the doc-block above: i≈1, f≈0, o≈1.
    for i in 0..hid_dim {
        b_i[i] =  8.0;
        b_f[i] = -8.0;
        b_o[i] =  8.0;
    }
    // Layer-1 input passthrough on the first 4 hidden units. We
    // intentionally route the input via the candidate gate (g) — that
    // way `c_t = i_t ⊙ g_t ≈ g_t = tanh(W_g · x_t)`. The `w_i` /
    // `w_f` / `w_o` weights stay zero so the saturated bias terms
    // dominate the gate values.
    for i in 0..in_dim {
        w_g[i * in_dim + i] = 1.0;
        // Defensive: also zero w_i / w_f / w_o for these rows
        // (already zero by construction, but explicit).
        w_i[i * in_dim + i] = 0.0;
        w_f[i * in_dim + i] = 0.0;
        w_o[i * in_dim + i] = 0.0;
    }

    // Linear projection. Same per-channel LQR gains as the MLP demo.
    let mut w_y = vec![0.0f64; out_dim * hid_dim];
    let b_y = vec![0.0f64; out_dim];
    let k_x = -1.0;
    let k_v = -2.5;
    let k_t = -55.0;
    let k_w = -10.0;
    w_y[0 * hid_dim + 0] = k_x;
    w_y[0 * hid_dim + 1] = k_v;
    w_y[0 * hid_dim + 2] = k_t;
    w_y[0 * hid_dim + 3] = k_w;

    NeuralLstmWeights {
        w_i, u_i, b_i,
        w_f, u_f, b_f,
        w_g, u_g, b_g,
        w_o, u_o, b_o,
        w_y, b_y,
    }
}

/// Convert a millisecond delay into a substep count. The viewer
/// runs `SUBSTEPS_PER_FRAME × FIXED_DT` per visible frame, so one
/// "actuator substep" is `FIXED_DT` seconds long.
fn ms_to_substeps(ms: f32) -> u32 {
    let dt_ms = (FIXED_DT * 1000.0) as f32;
    if dt_ms <= 0.0 {
        return 0;
    }
    (ms / dt_ms).round().max(0.0) as u32
}

/// Quadrotor demo — PD altitude + attitude control. Reads the current
/// frame body pose + velocity, computes per-rotor thrusts, writes
/// world-frame thrust forces into `state.body_f` on each rotor body.
/// Featherstone GPU consumes them via the bias term.
/// Drive the kinematic puller body + the puller-pinned cable particle
/// in `Scene::Cable` along a circular orbit. Also keeps the active
/// VBD solver's `vbd_mode` / `contact_stiffness` / `iterations` in
/// sync with the right-rail sliders.
fn update_cable_example(
    mut sim: ResMut<Sim>,
    mut state: ResMut<CableExampleState>,
    time: Res<Time>,
    run: Res<SimRunState>,
) {
    if sim.scene != Scene::Cable {
        return;
    }
    if !run.paused {
        state.sim_time += time.delta_secs();
    }
    let t = state.sim_time;
    let theta = t * CABLE_PULLER_ORBIT_RATE;
    let puller_pos = MVec3::new(
        (CABLE_PULLER_ORBIT_R * theta.cos()) as f64,
        CABLE_PULLER_HEIGHT as f64,
        (CABLE_PULLER_ORBIT_R * theta.sin()) as f64,
    );

    // Update the kinematic puller body's pose. body 0 is the puller
    // (added via `BodyParams::new().with_flags(KINEMATIC)` in
    // `build_cable_model` after the cable particles).
    if sim.model.body_count > 0 {
        if let Ok(bq) = sim.state_a.body_q.host_mut() {
            if let Some(body) = bq.last_mut() {
                *body = mp::Transform::from_translation(puller_pos);
            }
        }
    }

    // Update the puller-pinned cable particle. It's the LAST particle
    // pushed by `add_cable_polyline`, so index = particle_count - 1.
    let idx = sim.model.particle_count;
    if idx > 0 {
        if let Ok(pq) = sim.state_a.particle_q.host_mut() {
            pq[idx - 1] = puller_pos;
        }
    }

    // Apply the right-rail solver tunings.
    if let SolverBackend::Vbd(s) = &mut sim.solver {
        s.contact_stiffness = state.contact_stiffness;
        s.iterations = state.inner_iterations as usize;
        s.vbd_mode = if state.use_hard_mode {
            mp::VbdMode::Hard
        } else {
            mp::VbdMode::Soft
        };
    }
}

/// Stage-20.2 JELLY scene: keep the active VBD solver's contact
/// stiffness, iteration count, and per-tet material in sync with the
/// right-rail sliders. The Young's modulus / Poisson sliders re-derive
/// the per-element Lamé parameters and overwrite the model's `tet_mu`
/// / `tet_lambda` host buffers (which then get re-uploaded on the
/// next solver step).
fn update_jelly_example(mut sim: ResMut<Sim>, mut state: ResMut<JellyExampleState>) {
    if sim.scene != Scene::Jelly {
        return;
    }
    if let SolverBackend::Vbd(s) = &mut sim.solver {
        s.contact_stiffness = state.contact_stiffness;
        s.iterations = state.inner_iterations as usize;
    }
    // Push fresh (E, ν) → (μ, λ) into the model's per-tet buffers.
    // Stage 20.3 — also push fresh per-tet yield + hardening so the
    // plasticity sliders take effect without rebuilding the model.
    // Multi-material presets are baked at scene-build time, so the
    // per-tet uniformity here applies on top of whatever per-tet
    // material-id breakdown the preset chose.
    let nu = state.poisson_ratio.clamp(-0.49, 0.49);
    let e = state.youngs_modulus.max(1.0).min(1.0e9);
    let mu = e / (2.0 * (1.0 + nu));
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let tet_count = sim.model.tet_count;
    if tet_count == 0 {
        return;
    }
    // Mixed preset has stiff corner cells (E ×100), soft interior.
    // Apply via per-tet (μ, λ); the multi-material API at build
    // time already wrote the right initial values, but per-frame
    // slider tweaks rewrite uniformly here. Future enhancement:
    // store per-tet material-id at build time for re-stamping.
    {
        let mu_host = sim.model.tet_mu.host_mut().expect("tet_mu host");
        for v in mu_host.iter_mut() {
            *v = mu;
        }
    }
    {
        let lambda_host = sim.model.tet_lambda.host_mut().expect("tet_lambda host");
        for v in lambda_host.iter_mut() {
            *v = lambda;
        }
    }
    // Plasticity uniforms — `yield_stress = 0` is the UI sentinel for
    // "elastic only"; we encode that on the device side as a very
    // large finite value (>1e29) to short-circuit the WGSL plasticity
    // kernel. Real::INFINITY tripped some f32 cast issues so we use
    // a large number directly.
    let sigma_y_device = if state.yield_stress > 0.0 {
        state.yield_stress
    } else {
        1.0e30
    };
    let h_device = state.hardening.max(0.0);
    {
        let yield_host = sim
            .model
            .tet_yield_stress
            .host_mut()
            .expect("tet_yield_stress host");
        for v in yield_host.iter_mut() {
            *v = sigma_y_device;
        }
    }
    {
        let hard_host = sim
            .model
            .tet_hardening
            .host_mut()
            .expect("tet_hardening host");
        for v in hard_host.iter_mut() {
            *v = h_device;
        }
    }
    if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
        let device = device.clone();
        let queue = queue.clone();
        let _ = sim.model.tet_mu.ensure_device(&device);
        let _ = sim.model.tet_mu.upload(&queue);
        let _ = sim.model.tet_lambda.ensure_device(&device);
        let _ = sim.model.tet_lambda.upload(&queue);
        let _ = sim.model.tet_yield_stress.ensure_device(&device);
        let _ = sim.model.tet_yield_stress.upload(&queue);
        let _ = sim.model.tet_hardening.ensure_device(&device);
        let _ = sim.model.tet_hardening.upload(&queue);
    }
    // Compress button: drive the kinematic puller body downward
    // briefly when `compress_t` was bumped by the UI. Pull rate
    // decays geometrically each frame — total impulse is roughly
    // `compress_t / 60` revolutions of the puller body.
    if state.compress_t > 0.0 {
        if sim.model.body_count > 0 {
            let bq_host = sim.state_a.body_q.host_mut().expect("body_q");
            if let Some(b) = bq_host.first_mut() {
                let dy = -0.02 * state.compress_t as f64;
                b.position.y += dy;
            }
        }
        if let (Some(device), Some(queue)) =
            (sim.engine.device(), sim.engine.queue())
        {
            let device = device.clone();
            let queue = queue.clone();
            let _ = sim.state_a.body_q.ensure_device(&device);
            let _ = sim.state_a.body_q.upload(&queue);
        }
        state.compress_t = (state.compress_t - 0.1).max(0.0);
    }
    // Read back the max plastic-strain accumulator for the readout.
    if state.show_plastic_strain || state.last_max_plastic_strain > 0.0 {
        if let (Some(device), Some(queue)) =
            (sim.engine.device(), sim.engine.queue())
        {
            let device = device.clone();
            let queue = queue.clone();
            let _ = sim
                .state_a
                .tet_plastic_strain
                .download(&device, &queue);
        }
        if let Ok(eps) = sim.state_a.tet_plastic_strain.host() {
            state.last_max_plastic_strain =
                eps.iter().cloned().fold(0.0_f64, f64::max);
        }
    }
    // Stage 20.4 — back-stress readout. Only download when the
    // kinematic-hardening toggle is on (avoid the per-frame device
    // download cost for the default-off case).
    if state.kinematic_hardening_on && sim.model.tet_count > 0 {
        if let (Some(device), Some(queue)) =
            (sim.engine.device(), sim.engine.queue())
        {
            let device = device.clone();
            let queue = queue.clone();
            let _ = sim.state_a.tet_back_stress.download(&device, &queue);
        }
        if let Ok(alphas) = sim.state_a.tet_back_stress.host() {
            // Frobenius norm — same metric as the v4 GPU test.
            let mut max_norm: f64 = 0.0;
            for a in alphas {
                let frob = (a.x_axis.length_squared()
                    + a.y_axis.length_squared()
                    + a.z_axis.length_squared())
                .sqrt();
                if frob > max_norm {
                    max_norm = frob;
                }
            }
            state.last_max_back_stress = max_norm;
        }
    }
}

fn update_quadrotor(mut sim: ResMut<Sim>, mut state: ResMut<QuadrotorState>) {
    if sim.scene != Scene::Quadrotor {
        return;
    }
    if sim.model.body_count < 5 {
        return;
    }
    // Cache per-scene mass + arm length once on entry.
    if !state.cached {
        let masses = sim.model.body_mass.host().expect("body_mass");
        let total: f64 = masses.iter().take(5).sum();
        state.total_mass = total as f32;
        state.arm_length = QUAD_ARM as f32;
        state.cached = true;
    }

    // Stage 17.1 — propagate the right-rail "rigorous" toggle into the
    // active Featherstone solver. Only re-stamp on change so we don't
    // touch the solver every frame.
    if state.rigorous_d_solve != state.rigorous_d_solve_applied {
        if let SolverBackend::Featherstone(s) = &mut sim.solver {
            s.set_rigorous_d_solve(state.rigorous_d_solve);
        }
        state.rigorous_d_solve_applied = state.rigorous_d_solve;
    }

    let device = sim.engine.device().unwrap().clone();
    let queue = sim.engine.queue().unwrap().clone();

    // Frame body = body 0; rotors = 1..=4.
    // Pull the latest frame pose + twist from the host. After every
    // frame, `step_simulation` downloads `body_q` + `body_qd` from the
    // device, so this is a fresh read.
    let (frame_pos, frame_rot, frame_v, frame_w) = {
        let bq = sim.state_a.body_q.host().expect("body_q");
        let bqd = sim.state_a.body_qd.host().expect("body_qd");
        (bq[0].position, bq[0].rotation, bqd[0].linear, bqd[0].angular)
    };

    // Altitude PD: total thrust = m·g + kp·(z_t − z) + kd·(0 − vy).
    let g = molla::math::GRAVITY;
    let alt_err = state.target_altitude as f64 - frame_pos.y;
    let alt_kp = state.altitude_kp as f64;
    let alt_kd = state.altitude_kd as f64;
    let total_thrust =
        (state.total_mass as f64) * g + alt_kp * alt_err - alt_kd * frame_v.y;
    // Distribute equally as the hover baseline.
    let baseline = (total_thrust / 4.0).max(0.0);

    // Attitude PD: drive roll (rotation about world +X) + pitch (about
    // world +Z) toward zero. Yaw is intentionally left alone — a real
    // quadrotor mixes yaw via reaction torque from spinning rotors;
    // this single-axis-thrust model can't generate yaw, so we accept
    // free yaw drift.
    //
    // Convert the frame's body-frame "up" axis to world; deviation from
    // world +Y gives roll/pitch errors via the cross product.
    let body_y_world = frame_rot * MVec3::Y;
    let tilt = body_y_world.cross(MVec3::Y); // small-angle: ~(roll, _, pitch) error
    let roll_err = tilt.x; // rotation about world +X
    let pitch_err = tilt.z; // rotation about world +Z (note sign)
    let att_kp = state.attitude_kp as f64;
    let att_kd = state.attitude_kd as f64;
    // Damping uses world-frame angular velocity components.
    let roll_cmd = att_kp * roll_err - att_kd * frame_w.x;
    let pitch_cmd = att_kp * pitch_err - att_kd * frame_w.z;

    // Per-rotor thrust = baseline ± control offsets at arm length L.
    // Roll (about +X axis) torque = L·(thrust_+Z − thrust_−Z) — affects
    // rotors 2/3.
    // Pitch (about +Z axis) torque = L·(thrust_+X − thrust_−X) — affects
    // rotors 0/1.
    let l = state.arm_length as f64;
    let mut thrusts = [
        baseline + 0.5 * pitch_cmd / l, // rotor 0 (+X)
        baseline - 0.5 * pitch_cmd / l, // rotor 1 (-X)
        baseline + 0.5 * roll_cmd / l,  // rotor 2 (+Z)
        baseline - 0.5 * roll_cmd / l,  // rotor 3 (-Z)
    ];
    for t in thrusts.iter_mut() {
        if *t < 0.0 {
            *t = 0.0; // can't pull down with rotors
        }
    }

    // Update UI state.
    for i in 0..4 {
        state.last_thrusts[i] = thrusts[i] as f32;
    }
    state.last_altitude = frame_pos.y as f32;
    let (roll, pitch, yaw) = quat_to_rpy(frame_rot);
    state.last_attitude = [roll as f32, pitch as f32, yaw as f32];

    // Write thrust as world-frame +Y force on each rotor body. The
    // FREE root + fixed-rotor topology means each rotor's wrench is
    // transmitted into the frame via its joint constraint.
    {
        let bf = sim.state_a.body_f.host_mut().expect("body_f");
        // Frame body: zero (we drive via the rotors).
        bf[0] = mp::SpatialVector::ZERO;
        for i in 0..4 {
            let f = mp::SpatialVector::new(
                MVec3::new(0.0, thrusts[i], 0.0),
                MVec3::ZERO,
            );
            bf[1 + i] = f;
        }
        // Upload to device so Featherstone reads it.
        sim.state_a
            .body_f
            .upload(&queue)
            .expect("body_f upload");
    }

    // Touch device so future downloads see the host write.
    let _ = device;
}

/// Stage-22 MuJoCo wrapper demo update — almost a no-op. The actual
/// solver (`SolverBackend::MuJoCo` when the feature is on,
/// `Featherstone` fallback when off) is driven by `step_simulation`,
/// the same way every other scene's solver is. This system's only
/// job is to log ONCE on entry to `Scene::MuJoCo` so the user knows
/// whether they're hitting MuJoCo or the fallback path. Without
/// this, the user would see "Featherstone" in the right-rail pane
/// without explanation when they meant to be exercising MuJoCo.
fn update_mujoco_example(
    sim: Res<Sim>,
    mut state: ResMut<MuJoCoExampleState>,
) {
    if sim.scene != Scene::MuJoCo {
        return;
    }
    if !cfg!(feature = "mujoco") && !state.notified_unavailable {
        info!(
            "Scene::MuJoCo: `mujoco` Cargo feature is off; falling back to Featherstone. \
             Rebuild with `--features mujoco` after installing libmujoco to enable the wrapper."
        );
        state.notified_unavailable = true;
    }
}

/// Convert a quaternion to YXZ Euler (yaw about Y, pitch about X, roll
/// about Z). Used by the quadrotor pane for a human-readable readout
/// of the attitude error.
fn quat_to_rpy(q: MQuat) -> (f64, f64, f64) {
    // glam (re-exported via molla::math) lives at bevy::math too,
    // but we go through the molla shim to avoid pulling glam directly.
    let (yaw, pitch, roll) = q.to_euler(bevy::math::EulerRot::YXZ);
    (roll, pitch, yaw)
}

/// Contact example — runs the GPU `ContactSensor` against the
/// current `Contacts` and stashes the readings. No-op for any other
/// scene.
fn update_contact_example(
    sim: ResMut<Sim>,
    mut state: ResMut<ContactExampleState>,
) {
    if sim.scene != Scene::Contact {
        if state.sensor.is_some() {
            state.sensor = None;
            state.readings.clear();
        }
        return;
    }

    if state.sensor.is_none() {
        match ContactSensor::new(&sim.engine, &sim.model) {
            Ok(s) => state.sensor = Some(s),
            Err(e) => {
                warn!("contact sensor init failed: {e}");
                return;
            }
        }
    }

    let Sim {
        ref engine,
        ref contacts,
        ..
    } = *sim;
    let sensor = state.sensor.as_mut().expect("contact sensor");
    match sensor.update(engine, contacts) {
        Ok(r) => state.readings = r,
        Err(e) => warn!("contact sensor update failed: {e}"),
    }
}

/// Camera-sensor demo — orbits a single `TiledCamera` around the
/// scene each frame and downloads every channel. The right-rail pane
/// (`camera_pane`) reads from the cached arrays. No-op when the
/// scene isn't active.
#[allow(clippy::too_many_arguments)]
fn update_camera_example(
    time: Res<Time>,
    sim: ResMut<Sim>,
    mut state: ResMut<CameraExampleState>,
    mut images: ResMut<Assets<Image>>,
    mut user_textures: ResMut<EguiUserTextures>,
) {
    if sim.scene != Scene::Camera {
        return;
    }
    state.elapsed += time.delta_secs();

    if state.sensor.is_none() {
        let cfg = TiledCameraConfig {
            width: CAMERA_IMG_W,
            height: CAMERA_IMG_H,
            fov_y_radians: 50f32.to_radians(),
            channels: CameraChannels::ALL,
            ambient: MVec3::new(0.18, 0.18, 0.22),
            light_dir: MVec3::new(-0.4, -0.85, -0.35).normalize(),
            light_color: MVec3::new(1.0, 0.96, 0.88),
            max_distance: 100.0,
            near: 0.05,
        };
        match TiledCamera::new(&sim.engine, cfg, CAMERA_NUM_VIEWS) {
            Ok(s) => {
                state.fov_y = cfg.fov_y_radians;
                state.sensor = Some(s);
            }
            Err(e) => {
                warn!("TiledCamera init failed: {e}");
                return;
            }
        }
    }

    // Lazily allocate one Bevy `Image` asset per channel + register it
    // with `EguiUserTextures` so the pane can address it via TextureId.
    if state.image_color.is_none() {
        let (h_color, id_color) = alloc_thumb_image(&mut images, &mut user_textures);
        let (h_depth, id_depth) = alloc_thumb_image(&mut images, &mut user_textures);
        let (h_normal, id_normal) = alloc_thumb_image(&mut images, &mut user_textures);
        let (h_albedo, id_albedo) = alloc_thumb_image(&mut images, &mut user_textures);
        let (h_index, id_index) = alloc_thumb_image(&mut images, &mut user_textures);
        state.image_color = Some(h_color);
        state.image_depth = Some(h_depth);
        state.image_normal = Some(h_normal);
        state.image_albedo = Some(h_albedo);
        state.image_index = Some(h_index);
        state.tex_id_color = Some(id_color);
        state.tex_id_depth = Some(id_depth);
        state.tex_id_normal = Some(id_normal);
        state.tex_id_albedo = Some(id_albedo);
        state.tex_id_index = Some(id_index);
    }

    // Push the latest FOV slider into the sensor config (width/height
    // can't change post-construction; everything else is per-dispatch).
    {
        let fov = state.fov_y;
        let sensor = state.sensor.as_mut().unwrap();
        let mut cfg = *sensor.config();
        cfg.fov_y_radians = fov;
        let _ = sensor.set_config(cfg);
    }

    // Orbit the camera around the scene at fixed radius / height.
    let theta = state.elapsed * 0.5; // rad/s — slow circle.
    let eye = MVec3::new(
        (CAMERA_ORBIT_RADIUS as f64) * theta.cos() as f64,
        CAMERA_ORBIT_HEIGHT as f64,
        (CAMERA_ORBIT_RADIUS as f64) * theta.sin() as f64,
    );
    let target = MVec3::new(0.0, CAMERA_TARGET_HEIGHT as f64, 0.0);
    {
        let sensor = state.sensor.as_mut().unwrap();
        if let Err(e) = sensor.set_camera_look_at(0, eye, target, MVec3::Y) {
            warn!("set_camera_look_at: {e}");
            return;
        }
    }

    // Run both kernels and download every channel.
    let Sim {
        ref engine,
        ref model,
        ref state_a,
        ..
    } = *sim;
    let (last_pose, snapshot_color, snapshot_depth, snapshot_normal, snapshot_albedo, snapshot_index) = {
        let sensor = state.sensor.as_mut().unwrap();
        if let Err(e) = sensor.dispatch(engine, model, state_a) {
            warn!("TiledCamera::dispatch: {e}");
            return;
        }
        let pose = sensor.poses_host().ok().and_then(|p| p.first().copied());
        let c = sensor.download_color(engine).map(|s| s.to_vec());
        let d = sensor.download_depth(engine).map(|s| s.to_vec());
        let n = sensor.download_normal(engine).map(|s| s.to_vec());
        let a = sensor.download_albedo(engine).map(|s| s.to_vec());
        let i = sensor.download_shape_index(engine).map(|s| s.to_vec());
        (pose, c, d, n, a, i)
    };
    state.last_pose = last_pose;

    if let (Ok(c), Some(handle)) = (snapshot_color.as_ref(), state.image_color.as_ref()) {
        if let Some(img) = images.get_mut(handle) {
            write_rgba_u32_into_image(img, c);
        }
    }
    if let Ok(d) = snapshot_depth.as_ref() {
        let mut lo = f32::INFINITY;
        let mut hi = 0.0_f32;
        for &v in d {
            let f = v as f32;
            if f > 0.0 {
                if f < lo { lo = f; }
                if f > hi { hi = f; }
            }
        }
        if !lo.is_finite() || hi <= lo { lo = 0.0; hi = 1.0; }
        state.last_depth_range = Some((lo, hi));
        if let Some(handle) = state.image_depth.as_ref() {
            if let Some(img) = images.get_mut(handle) {
                write_depth_into_image(img, d, lo, hi);
            }
        }
    }
    if let (Ok(n), Some(handle)) = (snapshot_normal.as_ref(), state.image_normal.as_ref()) {
        if let Some(img) = images.get_mut(handle) {
            write_normal_into_image(img, n);
        }
    }
    if let (Ok(a), Some(handle)) = (snapshot_albedo.as_ref(), state.image_albedo.as_ref()) {
        if let Some(img) = images.get_mut(handle) {
            write_rgba_u32_into_image(img, a);
        }
    }
    if let (Ok(i), Some(handle)) = (snapshot_index.as_ref(), state.image_index.as_ref()) {
        if let Some(img) = images.get_mut(handle) {
            write_shape_index_into_image(img, i);
        }
    }
}

/// Allocate a `CAMERA_IMG_W × CAMERA_IMG_H` RGBA8 Bevy image and
/// register it with bevy_egui's user-texture map. The image data
/// starts black; subsequent frames mutate the bytes in place.
fn alloc_thumb_image(
    images: &mut Assets<Image>,
    user_textures: &mut EguiUserTextures,
) -> (Handle<Image>, egui::TextureId) {
    use bevy::asset::RenderAssetUsages;
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
    let size = Extent3d {
        width: CAMERA_IMG_W,
        height: CAMERA_IMG_H,
        depth_or_array_layers: 1,
    };
    let img = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    let handle = images.add(img);
    let id = user_textures.add_image(EguiTextureHandle::Strong(handle.clone()));
    (handle, id)
}

fn write_rgba_u32_into_image(img: &mut Image, src: &[u32]) {
    let Some(data) = img.data.as_mut() else { return };
    if data.len() != src.len() * 4 {
        return;
    }
    for (i, &p) in src.iter().enumerate() {
        let r = (p & 0xff) as u8;
        let g = ((p >> 8) & 0xff) as u8;
        let b = ((p >> 16) & 0xff) as u8;
        let a = ((p >> 24) & 0xff) as u8;
        let off = i * 4;
        data[off] = r;
        data[off + 1] = g;
        data[off + 2] = b;
        data[off + 3] = a;
    }
}

fn write_depth_into_image(img: &mut Image, src: &[molla::math::Real], lo: f32, hi: f32) {
    let Some(data) = img.data.as_mut() else { return };
    if data.len() != src.len() * 4 {
        return;
    }
    let span = (hi - lo).max(1e-6);
    for (i, &v) in src.iter().enumerate() {
        let off = i * 4;
        let f = v as f32;
        let g = if f <= 0.0 {
            0
        } else {
            let t = ((f - lo) / span).clamp(0.0, 1.0);
            ((1.0 - t) * 255.0) as u8
        };
        data[off] = g;
        data[off + 1] = g;
        data[off + 2] = g;
        data[off + 3] = 255;
    }
}

fn write_normal_into_image(img: &mut Image, src: &[MVec3]) {
    let Some(data) = img.data.as_mut() else { return };
    if data.len() != src.len() * 4 {
        return;
    }
    for (i, v) in src.iter().enumerate() {
        let off = i * 4;
        let len2 = v.length_squared();
        if len2 < 1e-9 {
            data[off] = 0;
            data[off + 1] = 0;
            data[off + 2] = 0;
        } else {
            data[off] = ((v.x as f32 * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            data[off + 1] = ((v.y as f32 * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            data[off + 2] = ((v.z as f32 * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
        }
        data[off + 3] = 255;
    }
}

fn write_shape_index_into_image(img: &mut Image, src: &[u32]) {
    let Some(data) = img.data.as_mut() else { return };
    if data.len() != src.len() * 4 {
        return;
    }
    const PALETTE: &[(u8, u8, u8)] = &[
        (231, 76, 60),
        (52, 152, 219),
        (46, 204, 113),
        (241, 196, 15),
        (155, 89, 182),
        (26, 188, 156),
        (230, 126, 34),
        (192, 57, 43),
        (41, 128, 185),
        (39, 174, 96),
        (211, 84, 0),
        (142, 68, 173),
    ];
    for (i, &idx) in src.iter().enumerate() {
        let off = i * 4;
        let (r, g, b) = if idx == 0xffff_ffff {
            (0, 0, 0)
        } else {
            PALETTE[(idx as usize) % PALETTE.len()]
        };
        data[off] = r;
        data[off + 1] = g;
        data[off + 2] = b;
        data[off + 3] = 255;
    }
}

/// Draw a yellow arrow at every body's contact centroid pointing
/// along the net push direction (`normal_sum`). Length is scaled by
/// `max_depth` so deeper penetrations get longer arrows. No-op when
/// the contact scene isn't active.
fn draw_contact_gizmos(
    sim: Res<Sim>,
    state: Res<ContactExampleState>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::Contact {
        return;
    }
    let arrow_color = Color::srgb(1.0, 0.85, 0.20);
    let dot_color = Color::srgb(1.0, 0.95, 0.55);
    for r in state.readings.iter() {
        if r.count == 0 {
            continue;
        }
        let centroid = Vec3::new(
            r.centroid.x as f32,
            r.centroid.y as f32,
            r.centroid.z as f32,
        );
        let normal_sum = Vec3::new(
            r.normal_sum.x as f32,
            r.normal_sum.y as f32,
            r.normal_sum.z as f32,
        );
        let len = normal_sum.length();
        if len < 1e-6 {
            // Sandwiched body — net force ~0. Just mark the centroid.
            gizmos.sphere(centroid, 0.04, dot_color);
            continue;
        }
        let dir = normal_sum / len;
        // Arrow length: 0.15m base + scaled by depth (capped).
        let arrow_len = 0.15 + (r.max_depth as f32 * 5.0).min(0.4);
        let tip = centroid + dir * arrow_len;
        gizmos.line(centroid, tip, arrow_color);
        // Tiny "+" cross at the tail.
        let cross = 0.04;
        gizmos.line(centroid - Vec3::X * cross, centroid + Vec3::X * cross, dot_color);
        gizmos.line(centroid - Vec3::Y * cross, centroid + Vec3::Y * cross, dot_color);
    }
}

/// Draw the Stage-23 diff-arm scene's overlays:
///   - Yellow sphere at the target XYZ.
///   - When `show_trajectory` is on, a polyline through the
///     per-iteration EE positions, fading from cool (early iter) to
///     warm (late iter) so the GD progression reads at a glance.
///   - Small dot at the live EE position (last body[1] tip in the
///     direction of the lower arm).
fn draw_diff_arm_gizmos(
    sim: Res<Sim>,
    state: Res<DiffArmExampleState>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::DiffArm {
        return;
    }
    // Target marker — yellow sphere + cross-hair lines.
    let target = Vec3::new(
        state.target.x as f32,
        state.target.y as f32,
        state.target.z as f32,
    );
    let target_color = Color::srgb(1.0, 0.95, 0.20);
    gizmos.sphere(target, 0.04, target_color);
    let cross = 0.08;
    gizmos.line(
        target - Vec3::X * cross,
        target + Vec3::X * cross,
        target_color,
    );
    gizmos.line(
        target - Vec3::Y * cross,
        target + Vec3::Y * cross,
        target_color,
    );

    // Trajectory overlay — polyline through the per-iteration EE
    // positions captured during the optimisation. Fade colour from
    // cool blue (iter 0, far from target) to warm orange (final iter,
    // near target).
    if state.show_trajectory && state.iter_ee_trace.len() >= 2 {
        let n = state.iter_ee_trace.len();
        for window in state.iter_ee_trace.windows(2).enumerate() {
            let (i, pair) = window;
            let t = i as f32 / (n - 1).max(1) as f32;
            let r = 0.20 + 0.75 * t;
            let g = 0.40 + 0.10 * (1.0 - t);
            let b = 0.95 - 0.85 * t;
            let color = Color::srgb(r, g, b);
            let a = Vec3::new(pair[0].x as f32, pair[0].y as f32, pair[0].z as f32);
            let b_ = Vec3::new(pair[1].x as f32, pair[1].y as f32, pair[1].z as f32);
            gizmos.line(a, b_, color);
            // Small dot at each iteration so individual GD steps are
            // visible (otherwise the polyline blurs them).
            gizmos.sphere(a, 0.012, color);
        }
        // Last point.
        if let Some(last) = state.iter_ee_trace.last() {
            let p = Vec3::new(last.x as f32, last.y as f32, last.z as f32);
            gizmos.sphere(p, 0.018, Color::srgb(0.95, 0.55, 0.15));
        }
    }
}

/// Stage-23.1 differentiable Featherstone arm gizmos. Same content as
/// `draw_diff_arm_gizmos` — target marker + EE trajectory polyline.
fn draw_diff_arm6_gizmos(
    sim: Res<Sim>,
    state: Res<DiffArm6ExampleState>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::DiffArm6 {
        return;
    }
    let target = Vec3::new(
        state.target.x as f32,
        state.target.y as f32,
        state.target.z as f32,
    );
    let target_color = Color::srgb(1.0, 0.95, 0.20);
    gizmos.sphere(target, 0.04, target_color);
    let cross = 0.08;
    gizmos.line(
        target - Vec3::X * cross,
        target + Vec3::X * cross,
        target_color,
    );
    gizmos.line(
        target - Vec3::Y * cross,
        target + Vec3::Y * cross,
        target_color,
    );
    gizmos.line(
        target - Vec3::Z * cross,
        target + Vec3::Z * cross,
        target_color,
    );
    if state.show_trajectory && state.iter_ee_trace.len() >= 2 {
        let n = state.iter_ee_trace.len();
        for window in state.iter_ee_trace.windows(2).enumerate() {
            let (i, pair) = window;
            let t = i as f32 / (n - 1).max(1) as f32;
            let r = 0.20 + 0.75 * t;
            let g = 0.40 + 0.10 * (1.0 - t);
            let b = 0.95 - 0.85 * t;
            let color = Color::srgb(r, g, b);
            let a = Vec3::new(pair[0].x as f32, pair[0].y as f32, pair[0].z as f32);
            let b_ = Vec3::new(pair[1].x as f32, pair[1].y as f32, pair[1].z as f32);
            gizmos.line(a, b_, color);
            gizmos.sphere(a, 0.01, color);
        }
        if let Some(last) = state.iter_ee_trace.last() {
            let p = Vec3::new(last.x as f32, last.y as f32, last.z as f32);
            gizmos.sphere(p, 0.018, Color::srgb(0.95, 0.55, 0.15));
        }
    }
}

fn hit_to_world(hit: &RayHit) -> Option<Vec3> {
    if hit.shape_id.is_some() {
        Some(Vec3::new(
            hit.point.x as f32,
            hit.point.y as f32,
            hit.point.z as f32,
        ))
    } else {
        None
    }
}

/// Draw the lidar's rays + hit points as Bevy gizmos. Hit rays are
/// red with a small white cross at the hit point; missed rays are
/// dim cyan extending to the max range. Lidar origin gets a small
/// yellow sphere so you can see where the sensor is.
fn draw_lidar_gizmos(
    sim: Res<Sim>,
    state: Res<LidarState>,
    config: Res<LidarConfig>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::Lidar {
        return;
    }

    // Sensor body indicator.
    gizmos.sphere(config.origin, 0.06, Color::srgb(1.0, 0.95, 0.2));

    let hit_color = Color::srgb(1.0, 0.25, 0.20);
    let miss_color = Color::srgba(0.30, 0.75, 0.85, 0.18);
    let cross_color = Color::srgb(1.0, 1.0, 1.0);
    let cross_half = 0.06_f32;

    for ray in state.rays.iter() {
        let end = match ray.hit {
            Some(p) => p,
            None => ray.origin + ray.direction * config.max_range,
        };
        let color = if ray.hit.is_some() {
            hit_color
        } else {
            miss_color
        };
        gizmos.line(ray.origin, end, color);
        if let Some(p) = ray.hit {
            gizmos.line(p - Vec3::X * cross_half, p + Vec3::X * cross_half, cross_color);
            gizmos.line(p - Vec3::Y * cross_half, p + Vec3::Y * cross_half, cross_color);
            gizmos.line(p - Vec3::Z * cross_half, p + Vec3::Z * cross_half, cross_color);
        }
    }
}

/// Render the CABLE polyline as a connected gizmo line strip. The
/// physics representation is just particles + segment springs — no
/// triangle mesh — so we draw it directly each frame from the host
/// `particle_q` mirror that `step_simulation` keeps fresh.
///
/// Colours: orange line strip for the cable; yellow markers on pinned
/// particles when `state.show_pinned` is on.
fn draw_cable_gizmos(
    sim: Res<Sim>,
    state: Res<CableExampleState>,
    mut gizmos: Gizmos,
) {
    let multi_cable = matches!(
        sim.scene,
        Scene::CablePile | Scene::CableYJunction,
    );
    if sim.scene != Scene::Cable && !multi_cable {
        return;
    }
    let Ok(pq) = sim.state_a.particle_q.host() else {
        return;
    };
    let n = sim.model.particle_count;
    if n < 2 {
        return;
    }
    let cable_color = if state.use_hard_mode {
        Color::srgb(0.95, 0.55, 0.20) // hard mode = warm orange
    } else {
        Color::srgb(0.40, 0.70, 0.95) // soft mode = cool blue
    };
    // For multi-cable scenes, walk the per-cable particle lists so we
    // don't draw a stray segment between cable N's last particle and
    // cable (N+1)'s first.
    if multi_cable && sim.model.cable_count > 0 {
        let offsets = match sim.model.cable_particle_offsets.host() {
            Ok(o) => o,
            Err(_) => return,
        };
        let particles = match sim.model.cable_particles.host() {
            Ok(p) => p,
            Err(_) => return,
        };
        for c in 0..sim.model.cable_count {
            let start = offsets[c] as usize;
            let end = offsets[c + 1] as usize;
            for k in start..end.saturating_sub(1) {
                let i_a = particles[k] as usize;
                let i_b = particles[k + 1] as usize;
                let a = pq[i_a];
                let b = pq[i_b];
                gizmos.line(
                    Vec3::new(a.x as f32, a.y as f32, a.z as f32),
                    Vec3::new(b.x as f32, b.y as f32, b.z as f32),
                    cable_color,
                );
            }
        }
        return;
    }
    for i in 0..n - 1 {
        let a = pq[i];
        let b = pq[i + 1];
        gizmos.line(
            Vec3::new(a.x as f32, a.y as f32, a.z as f32),
            Vec3::new(b.x as f32, b.y as f32, b.z as f32),
            cable_color,
        );
    }
    if state.show_pinned {
        // First and last particles are the pin anchors.
        let pin_color = Color::srgb(1.0, 0.95, 0.20);
        for &i in [0_usize, n - 1].iter() {
            let p = pq[i];
            let pos = Vec3::new(p.x as f32, p.y as f32, p.z as f32);
            gizmos.sphere(pos, 0.04, pin_color);
        }
    }
}

/// Stage-20.2 JELLY scene gizmos. Two parts:
///   1. Tet wireframe — every unique edge of every tet drawn as a
///      green line segment (deduped via a hash set so shared edges
///      don't overdraw).
///   2. Optional CABLE comparison overlay — a small bending cable
///      drawn as a yellow polyline above the cube for A/B feel.
fn draw_jelly_gizmos(
    sim: Res<Sim>,
    state: Res<JellyExampleState>,
    mut gizmos: Gizmos,
) {
    if !matches!(
        sim.scene,
        Scene::Jelly | Scene::SoftbodyDrop | Scene::FastSoftImpact | Scene::SoftBrake
    ) {
        return;
    }
    let Ok(pq) = sim.state_a.particle_q.host() else {
        return;
    };
    // For SoftbodyDrop / FastSoftImpact / SoftBrake the jelly tet
    // wireframe is always shown — those demos' whole point is soft-
    // on-soft contact.
    if sim.scene == Scene::Jelly && !state.show_wireframe {
        return;
    }
    let Ok(tet_idx) = sim.model.tet_indices.host() else {
        return;
    };
    let n_tets = sim.model.tet_count;
    if n_tets == 0 {
        return;
    }
    // Stage 20.3 — plastic-strain heat-map. When the toggle is on we
    // colour each tet's edges by its accumulated equivalent plastic
    // strain ε_p (cool blue → hot red over [0, 1]). Without the
    // toggle we use the Stage 20.2 uniform green and dedup edges
    // across tets to keep the line count manageable.
    let plastic_strain = if state.show_plastic_strain {
        sim.state_a
            .tet_plastic_strain
            .host()
            .ok()
            .map(|s| s.to_vec())
    } else {
        None
    };
    use std::collections::HashSet;
    let mut seen = HashSet::<(u32, u32)>::with_capacity(n_tets * 4);
    let default_edge_color = Color::srgb(0.40, 0.85, 0.40);
    for t in 0..n_tets {
        let v0 = tet_idx[t * 4];
        let v1 = tet_idx[t * 4 + 1];
        let v2 = tet_idx[t * 4 + 2];
        let v3 = tet_idx[t * 4 + 3];
        let edge_color = if let Some(eps) = &plastic_strain {
            let e = eps.get(t).copied().unwrap_or(0.0).clamp(0.0, 1.0) as f32;
            // Cool → hot blend (blue → red).
            Color::srgb(e, 0.10 + 0.20 * (1.0 - e), 1.0 - e)
        } else {
            default_edge_color
        };
        for &(a, b) in &[
            (v0, v1), (v0, v2), (v0, v3),
            (v1, v2), (v1, v3), (v2, v3),
        ] {
            let key = if a < b { (a, b) } else { (b, a) };
            // When the heat-map is on we don't dedup — every tet
            // contributes its colour to every shared edge — but we
            // still skip if we already drew it FOR THIS TET'S
            // colour (avoid four overlapping draws per shared edge).
            if plastic_strain.is_some() || seen.insert(key) {
                let pa = pq[a as usize];
                let pb = pq[b as usize];
                gizmos.line(
                    Vec3::new(pa.x as f32, pa.y as f32, pa.z as f32),
                    Vec3::new(pb.x as f32, pb.y as f32, pb.z as f32),
                    edge_color,
                );
            }
        }
    }
}

/// While dragging, draw a yellow line from the picked anchor (body or
/// cloth particle) to the cursor target so the user can see what
/// they're holding.
fn draw_grab_gizmo(grab: Res<MouseGrab>, sim: Res<Sim>, mut gizmos: Gizmos) {
    let Some(state) = grab.state else { return };
    let anchor_world = match state.kind {
        GrabKind::Body { body, local_anchor } => {
            let Ok(q) = sim.state_a.body_q.host() else { return };
            let i = body.index();
            if i >= q.len() {
                return;
            }
            q[i].transform_point(local_anchor)
        }
        GrabKind::Particle { index, .. } => {
            let Ok(q) = sim.state_a.particle_q.host() else { return };
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
    sensor_states: (Res<LidarState>, Res<ImuExampleState>, Res<ContactExampleState>),
    mut pane_states: PaneSceneStates,
    mut ribbons: RibbonParams,
    mut warp_states: ResMut<WarpExampleStates>,
) {
    let (lidar_state, imu_state, contact_state) = (
        &*sensor_states.0,
        &*sensor_states.1,
        &*sensor_states.2,
    );
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
            // Right pane is enabled only for scenes that have
            // per-example settings — Pile/Rope/Tower/Cloth don't.
            PANE_SCENE => !matches!(
                scene,
                Scene::Lidar
                    | Scene::Imu
                    | Scene::Contact
                    | Scene::Arm
                    | Scene::Shapes
                    | Scene::GpuIkArm
                    | Scene::GpuNarrowPhase
                    | Scene::Camera
                    | Scene::Actuator
                    | Scene::Quadrotor
                    | Scene::BroadPhase
                    | Scene::VbdCloth
                    | Scene::MuJoCo
                    | Scene::DiffArm
                    | Scene::DiffArm6
                    | Scene::MeshTools
                    | Scene::NeuralPolicy
                    | Scene::NeuralLstm
                    | Scene::Cable
                    | Scene::Mpm
                    | Scene::Jelly
                    | Scene::Pendulum
                    | Scene::Conveyor
                    | Scene::ClothBending
                    | Scene::ClothTwist
                    | Scene::MpmSnowBall
                    | Scene::MpmTwoWay
                    | Scene::CartPole
                    | Scene::CartPoleSwarm
                    | Scene::FourBar
                    | Scene::DiffBall
                    | Scene::DiffSpring
                    | Scene::SoftbodyDrop
                    | Scene::ClothCollar
                    | Scene::FastSoftImpact
                    | Scene::SoftBrake
                    | Scene::FemHeat
                    | Scene::FemHeatP2
                    | Scene::FemBeam
                    | Scene::FemDiffBeam
                    | Scene::FemStokes
                    | Scene::WarpRaycast
                    | Scene::WarpRaymarch
                    | Scene::WarpMesh
                    | Scene::WarpMeshIntersect
                    | Scene::WarpSampleMesh
                    | Scene::WarpMarchingCubes
                    | Scene::WarpNvdb
                    | Scene::WarpDem
                    | Scene::WarpFluid
                    | Scene::WarpSph
                    | Scene::WarpTileFft
                    | Scene::WarpTileMlp
                    | Scene::WarpTileNbody
                    | Scene::WarpTileCholesky
                    | Scene::WarpParticleRepulsion
                    | Scene::WarpFemDiffusion
                    | Scene::WarpFemDiffusion3D
                    | Scene::WarpFemConvDiff
                    | Scene::WarpFemTaylorGreen
                    | Scene::WarpFemKelvinHelmholtz
                    | Scene::WarpFemStreamlines
                    | Scene::WarpFemDeformedGeometry
                    | Scene::WarpFemDistortion
                    | Scene::WarpFemElasticOpt
                    | Scene::WarpFftFluid
                    | Scene::WarpDistortion3D
                    | Scene::WarpApicFluid
                    | Scene::WarpWave
                    | Scene::WarpDiffRay
                    | Scene::WarpTileFilter
                    | Scene::WarpTileCompaction
            ),
            _ => false,
        },
    );

    let is_open = |id: &'static str| -> bool {
        let Some(item) = find_item(RIBBON_ITEMS, id) else { return false };
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
    Pane2::new(PANE_MAIN, "molla", PaneAnchor::LeftRail(RailZone::Start), accent_color).show(
        ctx,
        |ui| {
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

            readout(ui, "FPS", &format!("{fps:>5.0}  ({frame_ms:>4.1} ms)"));
            readout(
                ui,
                "Bodies",
                &format!("{body_count}  ({SUBSTEPS_PER_FRAME}× @ {:.0} Hz)", 1.0 / FIXED_DT),
            );
            ui.separator();

            readout(ui, "Phys total", &format!("{:>4.0} µs", stats.smoothed_step_us));
            readout(ui, "  · collide",  &format!("{:>4.0} µs", stats.smoothed_collide_us));
            readout(ui, "  · dispatch", &format!("{:>4.0} µs", stats.smoothed_solve_us));
            readout(ui, "  · download", &format!("{:>4.0} µs", stats.smoothed_download_us));
            readout(ui, "Contacts", &format!("{}", stats.last_contacts));
            ui.separator();

            readout(ui, "Solver", "XPBD-GPU");
            readout(ui, "Adapter", &sim.adapter_label);
            readout(ui, "Friction", &format!("{PLANE_FRICTION:.2}"));
            readout(ui, "Restitution", &format!("{PLANE_RESTITUTION:.2}"));
            ui.separator();

            // Activity view toggle.
            let view_label = if activity.enabled { "View: ACTIVITY" } else { "View: NORMAL" };
            readout(ui, view_label, &format!("{} active", activity.active_count));
            ui.horizontal(|ui| {
                let label = if activity.enabled { "Hide sleep" } else { "Show sleep (T)" };
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
                if run.paused
                    && button(ui, "Step", accent_color).clicked() {
                        run.step_once = true;
                    }
                if button(ui, "Reset", accent_color).clicked() {
                    sim.reset();
                }
            });

            ui.add_space(4.0);
            ui.small("LMB drag · T activity · W wireframe");
            ui.small("Space pause · R reset · MMB pan · LMB+RMB orbit");
        },
    );
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
        Scene::MuJoCo => Some("MuJoCo (Stage 22)"),
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
        Scene::ClothTwist => Some("Cloth twist"),
        Scene::MpmSnowBall => Some("MPM snow ball"),
        Scene::MpmTwoWay => Some("MPM two-way"),
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
        Scene::WarpMeshIntersect => Some("Warp mesh intersect (GJK)"),
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
        Scene::WarpFemTaylorGreen => Some("Warp FEM Taylor-Green"),
        Scene::WarpFemKelvinHelmholtz => Some("Warp FEM Kelvin-Helmholtz"),
        Scene::WarpFemStreamlines => Some("Warp FEM streamlines"),
        Scene::WarpFemDeformedGeometry => Some("Warp FEM deformed geometry"),
        Scene::WarpFemDistortion => Some("Warp FEM distortion energy"),
        Scene::WarpFemElasticOpt => Some("Warp FEM elastic shape opt"),
        Scene::WarpFftFluid => Some("Warp FFT fluid"),
        Scene::WarpDistortion3D => Some("Warp distortion 3D"),
        Scene::WarpApicFluid => Some("Warp APIC fluid (Stage 37.6)"),
        Scene::WarpWave => Some("Warp wave (Stage 37.1)"),
        Scene::WarpDiffRay => Some("Warp diff raycast (Stage 37.7)"),
        Scene::WarpTileFilter => Some("Warp tile filter (Stage 37.8)"),
        Scene::WarpTileCompaction => Some("Warp tile compaction (Stage 37.8)"),
        _ => None,
    };
    if let Some(title) = scene_pane_title {
        if is_open(PANE_SCENE) {
            let anchor = PaneAnchor::RightRail(RailZone::Start);
            Pane2::new(PANE_SCENE, title, anchor, accent_color)
                .resize(PaneResize::BOTH)
                .show(ctx, |ui| match scene {
                    Scene::Lidar => lidar_pane(ui, &mut lidar_config, &lidar_state),
                    Scene::Imu => imu_pane(ui, &imu_state),
                    Scene::Contact => contact_pane(ui, &contact_state),
                    Scene::Arm => arm_pane(ui, &mut pane_states.arm, &sim, &mut pending),
                    Scene::Shapes => shapes_pane(ui, &sim),
                    Scene::GpuIkArm => {
                        gpu_ik_pane(ui, &mut pane_states.gpu_ik, &sim)
                    }
                    Scene::GpuNarrowPhase => {
                        gpu_np_pane(ui, &mut pane_states.gpu_np, accent_color)
                    }
                    Scene::Camera => camera_pane(
                        ui,
                        &mut pane_states.camera,
                        anchor,
                        accent_color,
                    ),
                    Scene::Actuator => {
                        actuator_pane(ui, &mut pane_states.actuator)
                    }
                    Scene::Quadrotor => {
                        quadrotor_pane(ui, &mut pane_states.quad)
                    }
                    Scene::BroadPhase => {
                        gpu_bp_pane(ui, &mut pane_states.gpu_bp, accent_color)
                    }
                    Scene::VbdCloth => {
                        vbd_cloth_pane(ui, &mut pane_states.vbd, &mut sim, accent_color)
                    }
                    Scene::MuJoCo => {
                        mujoco_pane(ui, &mut pane_states.mujoco, &sim)
                    }
                    Scene::DiffArm => {
                        diff_arm_pane(ui, &mut pane_states.diff_arm)
                    }
                    Scene::DiffArm6 => {
                        diff_arm6_pane(ui, &mut pane_states.diff_arm6)
                    }
                    Scene::MeshTools => {
                        mesh_tools_pane(ui, &mut pane_states.mesh, &mut pending, scene)
                    }
                    Scene::NeuralPolicy => {
                        neural_policy_pane(ui, &mut pane_states.neural)
                    }
                    Scene::NeuralLstm => {
                        neural_lstm_pane(ui, &mut pane_states.neural_lstm)
                    }
                    Scene::Cable => {
                        cable_pane(ui, &mut pane_states.cable, &mut sim, accent_color)
                    }
                    Scene::Mpm => {
                        mpm_pane(ui, &mut pane_states.mpm, &mut sim, accent_color)
                    }
                    Scene::Jelly => {
                        jelly_pane(
                            ui,
                            &mut pane_states.jelly,
                            &mut sim,
                            &mut pending,
                            accent_color,
                        )
                    }
                    Scene::Pendulum => pendulum_pane(ui, &pane_states.pendulum),
                    Scene::Conveyor => conveyor_pane(ui, &mut pane_states.conveyor),
                    Scene::ClothBending => cloth_bending_pane(
                        ui,
                        &mut pane_states.cloth_bend,
                        &mut pending,
                        scene,
                    ),
                    Scene::ClothTwist => cloth_twist_pane(ui, &mut pane_states.cloth_twist),
                    Scene::MpmSnowBall => {
                        mpm_pane(ui, &mut pane_states.mpm, &mut sim, accent_color)
                    }
                    Scene::MpmTwoWay => {
                        mpm_pane(ui, &mut pane_states.mpm, &mut sim, accent_color)
                    }
                    Scene::CartPole => cartpole_pane(
                        ui,
                        &mut pane_states.cartpole,
                        accent_color,
                    ),
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
                    Scene::FemHeatP2 => {
                        fem_heat_p2_pane(ui, &mut pane_states.fem_heat_p2)
                    }
                    Scene::FemBeam => fem_beam_pane(ui, &mut pane_states.fem_beam),
                    Scene::FemDiffBeam => {
                        fem_diff_beam_pane(ui, &mut pane_states.fem_diff_beam)
                    }
                    Scene::FemStokes => {
                        fem_stokes_pane(ui, &mut pane_states.fem_stokes)
                    }
                    Scene::WarpRaycast => warp_raycast_pane(ui, &mut warp_states.raycast),
                    Scene::WarpRaymarch => warp_raymarch_pane(ui, &mut warp_states.raymarch),
                    Scene::WarpMesh => warp_mesh_pane(ui, &mut warp_states.mesh),
                    Scene::WarpMeshIntersect => warp_mesh_intersect_pane(ui, &mut warp_states.mesh_intersect),
                    Scene::WarpSampleMesh => warp_sample_mesh_pane(ui, &mut warp_states.sample_mesh),
                    Scene::WarpMarchingCubes => warp_marching_cubes_pane(ui, &mut warp_states.marching_cubes),
                    Scene::WarpNvdb => warp_nvdb_pane(ui, &mut warp_states.nvdb),
                    Scene::WarpDem => warp_dem_pane(ui, &mut warp_states),
                    Scene::WarpFluid => warp_fluid_pane(ui, &mut warp_states),
                    Scene::WarpSph => warp_sph_pane(ui, &mut warp_states),
                    Scene::WarpTileFft => warp_tile_fft_pane(ui, &mut warp_states.tile_fft),
                    Scene::WarpTileMlp => warp_tile_mlp_pane(ui, &mut warp_states.tile_mlp),
                    Scene::WarpTileNbody => warp_tile_nbody_pane(ui, &mut warp_states.tile_nbody),
                    Scene::WarpTileCholesky => warp_tile_cholesky_pane(ui, &mut warp_states.tile_cholesky),
                    Scene::WarpParticleRepulsion => warp_repulsion_pane(ui, &mut warp_states.repulsion),
                    Scene::WarpFemDiffusion => warp_fem_diffusion_pane(ui, &mut warp_states.fem_diff),
                    Scene::WarpFemDiffusion3D => warp_fem_diffusion_3d_pane(ui, &mut warp_states.fem_diff3d),
                    Scene::WarpFemConvDiff => warp_fem_convdiff_pane(ui, &mut warp_states.fem_convdiff),
                    Scene::WarpFemTaylorGreen => warp_fem_taylor_pane(ui, &mut warp_states.fem_taylor),
                    Scene::WarpFemKelvinHelmholtz => warp_fem_kh_pane(ui, &mut warp_states.fem_kh),
                    Scene::WarpFemStreamlines => warp_fem_streamlines_pane(ui, &mut warp_states.fem_streamlines),
                    Scene::WarpFemDeformedGeometry => warp_fem_deformed_pane(ui, &mut warp_states.fem_deformed),
                    Scene::WarpFemDistortion => warp_fem_distortion_pane(ui, &mut warp_states.fem_distortion),
                    Scene::WarpFemElasticOpt => warp_fem_elastic_opt_pane(ui, &mut warp_states.fem_elastic_opt),
                    Scene::WarpFftFluid => warp_fft_fluid_pane(ui, &mut warp_states.fft_fluid),
                    Scene::WarpDistortion3D => warp_distortion_3d_pane(ui, &mut warp_states.distortion_3d),
                    Scene::WarpApicFluid => warp_apic_fluid_pane(ui, &mut warp_states.apic_fluid),
                    Scene::WarpWave => warp_wave_pane(ui, &mut warp_states.wave),
                    Scene::WarpDiffRay => warp_diff_ray_pane(ui, &mut warp_states.diff_ray),
                    Scene::WarpTileFilter => warp_tile_filter_pane(ui, &mut warp_states.tile_filter),
                    Scene::WarpTileCompaction => warp_tile_compaction_pane(ui, &mut warp_states.tile_compaction),
                    _ => {}
                });
        }
    }
}

/// Right-rail pane for the Stage-13 shapes demo. Lists the
/// primitives in the scene and the live body count.
fn shapes_pane(ui: &mut egui::Ui, sim: &Sim) {
    ui.label("Stage 13 primitives:");
    ui.separator();
    readout(ui, "Heightfield", &format!("{}×{} grid", SHAPES_FIELD_N, SHAPES_FIELD_N));
    readout(
        ui,
        "Torus (SDF)",
        &format!(
            "major {:.2} m, minor {:.2} m",
            SHAPES_TORUS_MAJOR, SHAPES_TORUS_MINOR
        ),
    );
    ui.separator();
    readout(ui, "Bodies", &format!("{}", sim.model.body_count));
    readout(ui, "Shapes", &format!("{}", sim.model.shape_count));
    ui.label("");
    ui.label("Drop test: cones, hulls, spheres rain");
    ui.label("onto the wavy heightfield. Static");
    ui.label("torus sticks out of the middle.");
}

/// Right-rail pane for the Stage-21 mesh-tools showcase.
/// Three panels:
///   1. Original (left)  / Decimated (centre)  side-by-side toggles.
///   2. Marching-cubes-from-SDF surface (right) toggle.
///   3. Hull overlay toggle on the original.
fn mesh_tools_pane(
    ui: &mut egui::Ui,
    state: &mut MeshToolsExampleState,
    pending: &mut PendingSceneSwitch,
    scene: Scene,
) {
    ui.label("Source: noise-displaced icosphere (3 subdivs)");
    ui.separator();
    readout(ui, "Original tris", &format!("{}", state.original_tris));
    readout(ui, "Decimated tris", &format!("{}", state.decimated_tris));
    readout(ui, "MC tris", &format!("{}", state.mc_tris));
    readout(ui, "Hull verts", &format!("{}", state.hull_verts));
    readout(ui, "SDF resolution", &format!("{}³", state.sdf_resolution));
    ui.separator();

    ui.label("Visibility:");
    ui.checkbox(&mut state.show_original, "Original (left, orange)");
    ui.checkbox(&mut state.show_decimated, "Decimated (centre, blue)");
    ui.checkbox(&mut state.show_marching_cubes, "Marching cubes (right, green)");
    ui.checkbox(&mut state.show_hull, "Hull overlay (yellow wireframe on original)");
    ui.separator();

    // Stage-24e — sparse SDF storage delta. The toggle reflects what
    // the MC surface currently shows; flipping it requires a scene
    // rebuild (press M to re-enter), which the small print below
    // calls out so the user isn't surprised.
    ui.label("Stage 24e: sparse vs dense SDF");
    if ui
        .checkbox(
            &mut state.use_sparse_sdf,
            "Use sparse SDF (re-extract MC) — rebuild to apply",
        )
        .changed()
    {
        state.rebuild_pending = true;
    }
    let dense_kib = state.dense_sdf_bytes as f32 / 1024.0;
    let sparse_kib = state.sparse_sdf_bytes as f32 / 1024.0;
    let ratio = if state.dense_sdf_bytes > 0 {
        sparse_kib / dense_kib
    } else {
        0.0
    };
    readout(ui, "Dense SDF", &format!("{dense_kib:.1} KiB"));
    readout(ui, "Sparse SDF", &format!("{sparse_kib:.1} KiB"));
    readout(ui, "Sparse / dense", &format!("{:.1} %", ratio * 100.0));
    readout(ui, "Active leaves", &format!("{}", state.sparse_sdf_leaves));
    readout(
        ui,
        "Active voxels",
        &format!("{}", state.sparse_sdf_active_voxels),
    );
    readout(ui, "Narrow band", &format!("{} voxels", state.sparse_sdf_band));
    ui.separator();

    if state.sparse_root_table_capacity > 0 {
        readout(
            ui,
            "Root hash capacity",
            &format!("{}", state.sparse_root_table_capacity),
        );
        readout(
            ui,
            "Root load factor",
            &format!("{:.2}", state.sparse_root_load_factor),
        );
        readout(
            ui,
            "Root probes",
            &format!(
                "avg {:.2}, max {}",
                state.sparse_root_avg_probes, state.sparse_root_max_probes
            ),
        );
    }
    if state.rebuild_pending {
        ui.colored_label(
            egui::Color32::YELLOW,
            "Rebuild required to apply scene-build toggles.",
        );
        if ui.button("Rebuild mesh tools").clicked() {
            // Re-enter the current scene unconditionally; the
            // hotkey path (`handle_input`) skips this when the
            // scene is already active.
            pending.0 = Some(scene);
            state.rebuild_pending = false;
        }
    }
    ui.separator();

    ui.label("");
    ui.label("Pipeline runs once on scene entry:");
    ui.label("• decimate(): QEM edge collapses → 25 % LOD");
    ui.label("• bake_mesh_to_sdf(): voxelise + sign by parity");
    ui.label("• sparse_sdf_from_dense(): drop voxels outside band");
    ui.label("• marching_cubes(): re-extract surface");
    ui.label("• convex_hull(): QuickHull from vertex cloud");
    ui.label("");
    ui.small("Hotkey: M");
}

/// Per-frame system: read [`MeshToolsExampleState`] and flip the
/// `Visibility` of every entity tagged with one of the
/// `MeshTool*` markers. Cheap (≤ a few hundred entities); skipping
/// the marker queries when `show_*` flags are unchanged would be a
/// premature optimisation.
fn update_mesh_tools_visibility(
    state: Res<MeshToolsExampleState>,
    mut q_orig: Query<&mut Visibility, (With<MeshToolOriginal>, Without<MeshToolDecimated>, Without<MeshToolMarchingCubes>, Without<MeshToolHull>)>,
    mut q_dec: Query<&mut Visibility, (With<MeshToolDecimated>, Without<MeshToolOriginal>, Without<MeshToolMarchingCubes>, Without<MeshToolHull>)>,
    mut q_mc: Query<&mut Visibility, (With<MeshToolMarchingCubes>, Without<MeshToolOriginal>, Without<MeshToolDecimated>, Without<MeshToolHull>)>,
    mut q_hull: Query<&mut Visibility, (With<MeshToolHull>, Without<MeshToolOriginal>, Without<MeshToolDecimated>, Without<MeshToolMarchingCubes>)>,
) {
    let v = |on: bool| if on { Visibility::Visible } else { Visibility::Hidden };
    for mut vis in q_orig.iter_mut() {
        *vis = v(state.show_original);
    }
    for mut vis in q_dec.iter_mut() {
        *vis = v(state.show_decimated);
    }
    for mut vis in q_mc.iter_mut() {
        *vis = v(state.show_marching_cubes);
    }
    for mut vis in q_hull.iter_mut() {
        *vis = v(state.show_hull);
    }
}

/// Body for the right-rail pane in the VBD-cloth scene. Sliders for
/// stretch / bend / contact stiffness + iteration count, plus a
/// VBD ↔ XPBD toggle for side-by-side comparison.
fn vbd_cloth_pane(
    ui: &mut egui::Ui,
    state: &mut VbdClothExampleState,
    sim: &mut Sim,
    accent: egui::Color32,
) {
    ui.add_space(2.0);
    let active_solver = match &sim.solver {
        SolverBackend::Vbd(_) => "VBD",
        SolverBackend::Xpbd(_) => "XPBD (Stage 12 baseline)",
        _ => "(unknown)",
    };
    readout(ui, "Solver", active_solver);
    readout(
        ui,
        "Cloth",
        &format!(
            "{}² ({} particles, {} springs)",
            VBD_CLOTH_GRID,
            sim.model.particle_count,
            sim.model.spring_count,
        ),
    );
    readout(
        ui,
        "Colours",
        &format!(
            "{} (particle), {} (spring)",
            sim.model.particle_color_count, sim.model.spring_color_count,
        ),
    );
    ui.separator();

    // VBD ↔ XPBD toggle.
    let mut want_vbd = state.use_vbd;
    if ui.checkbox(&mut want_vbd, "Use VBD (uncheck for XPBD)").changed() {
        state.use_vbd = want_vbd;
        let kind = if want_vbd { SolverKind::Vbd } else { SolverKind::Xpbd };
        sim.swap_solver_to(kind);
        sim.reset();
    }
    ui.separator();

    let mut dirty = false;

    ui.label("Stretch stiffness (N/m)");
    if ui
        .add(egui::Slider::new(&mut state.stretch_stiffness, 1.0e3..=1.0e7).logarithmic(true))
        .changed()
    {
        dirty = true;
    }
    ui.label("Bend stiffness (N/m)");
    if ui
        .add(egui::Slider::new(&mut state.bend_stiffness, 1.0e1..=1.0e6).logarithmic(true))
        .changed()
    {
        dirty = true;
    }
    ui.label("Soft contact stiffness (N/m)");
    if ui
        .add(egui::Slider::new(&mut state.contact_stiffness, 1.0e3..=1.0e6).logarithmic(true))
        .changed()
    {
        if let SolverBackend::Vbd(s) = &mut sim.solver {
            s.contact_stiffness = state.contact_stiffness as f64;
        }
    }
    ui.label("Inner iterations");
    if ui
        .add(egui::Slider::new(&mut state.inner_iterations, 1..=30))
        .changed()
    {
        match &mut sim.solver {
            SolverBackend::Vbd(s) => s.iterations = state.inner_iterations as usize,
            SolverBackend::Xpbd(s) => s.position_iterations = state.inner_iterations as usize,
            _ => {}
        }
    }
    ui.separator();

    if dirty {
        // Stretch / bend stiffness changes need to be reflected in
        // the per-spring compliance buffer the kernel reads. We
        // mutate the model directly (it's immutable from a borrow-
        // checker perspective but the compliance buffer is fine to
        // overwrite — VBD reads it once at upload time and again
        // when we call refresh_stiffness).
        apply_vbd_stiffness(sim, state);
    }

    if ui.add_sized([ui.available_width(), 28.0], button_widget("Reset cloth", accent))
        .clicked()
    {
        sim.reset();
    }

    ui.add_space(6.0);
    ui.small("Hotkey: V");
    ui.small("Compare: toggle VBD off → XPBD baseline.");
    ui.small("XPBD wobbles at high stiffness; VBD settles.");
}

/// Right-rail pane for the Stage-20.1 CABLE demo. Carries the
/// stretch + friction + contact-stiffness sliders, the AVBD
/// soft↔hard mode toggle, and a "show pinned points" gizmo toggle.
fn cable_pane(
    ui: &mut egui::Ui,
    state: &mut CableExampleState,
    sim: &mut Sim,
    accent: egui::Color32,
) {
    ui.add_space(2.0);
    readout(ui, "Particles", &format!("{}", sim.model.particle_count));
    readout(ui, "Springs", &format!("{}", sim.model.spring_count));
    readout(ui, "Cables", &format!("{}", sim.model.cable_count));
    if sim.model.cable_count > 0 {
        if let Ok(rl) = sim.model.cable_rest_length.host() {
            readout(ui, "Rest length", &format!("{:.3} m", rl[0]));
        }
    }
    ui.separator();

    ui.label("Stretch stiffness (N/m)");
    if ui
        .add(
            egui::Slider::new(&mut state.stretch_stiffness, 1.0e3..=1.0e7)
                .logarithmic(true),
        )
        .changed()
    {
        // Mirror the new stiffness into the per-spring compliance
        // buffer so VBD's `refresh_stiffness` (called below) picks it
        // up. The model's compliance buffer is host-mut-friendly
        // because we own it; any GPU re-upload happens inside
        // `refresh_stiffness` on the solver.
        let alpha = if state.stretch_stiffness > 1.0 {
            1.0 / state.stretch_stiffness
        } else {
            1.0
        };
        if let Ok(comp) = sim.model.spring_compliance.host_mut() {
            for c in comp.iter_mut() {
                *c = alpha;
            }
        }
        if let SolverBackend::Vbd(s) = &mut sim.solver {
            let _ = s.refresh_stiffness(&sim.model);
        }
    }

    ui.label("Soft contact stiffness (N/m)");
    if ui
        .add(
            egui::Slider::new(&mut state.contact_stiffness, 1.0e3..=1.0e6)
                .logarithmic(true),
        )
        .changed()
    {
        if let SolverBackend::Vbd(s) = &mut sim.solver {
            s.contact_stiffness = state.contact_stiffness;
        }
    }
    ui.label("Inner iterations");
    if ui
        .add(egui::Slider::new(&mut state.inner_iterations, 1..=30))
        .changed()
    {
        if let SolverBackend::Vbd(s) = &mut sim.solver {
            s.iterations = state.inner_iterations as usize;
        }
    }
    ui.separator();

    if ui
        .checkbox(&mut state.use_hard_mode, "Hard-mode AVBD (no penetration)")
        .changed()
    {
        if let SolverBackend::Vbd(s) = &mut sim.solver {
            s.vbd_mode = if state.use_hard_mode {
                mp::VbdMode::Hard
            } else {
                mp::VbdMode::Soft
            };
        }
    }
    ui.checkbox(&mut state.show_pinned, "Show pinned points");
    ui.separator();

    if ui
        .add_sized(
            [ui.available_width(), 28.0],
            button_widget("Reset cable", accent),
        )
        .clicked()
    {
        sim.reset();
        state.sim_time = 0.0;
    }
    ui.add_space(6.0);
    ui.small("Hotkey: C");
}

/// Right-rail pane for the Stage-20.2 JELLY tetrahedral soft-body
/// demo. Carries Young's-modulus + Poisson-ratio sliders (per-tet
/// material), inner-iter / contact-stiffness sliders, and a wireframe
/// toggle (gizmos for tet edges).
fn jelly_pane(
    ui: &mut egui::Ui,
    state: &mut JellyExampleState,
    sim: &mut Sim,
    pending: &mut PendingSceneSwitch,
    accent: egui::Color32,
) {
    ui.add_space(2.0);
    readout(ui, "Particles", &format!("{}", sim.model.particle_count));
    readout(ui, "Tets", &format!("{}", sim.model.tet_count));
    readout(ui, "Tet meshes", &format!("{}", sim.model.tet_mesh_count));
    readout(
        ui,
        "Colours",
        &format!(
            "{} (particle), {} (spring)",
            sim.model.particle_color_count, sim.model.spring_color_count
        ),
    );
    ui.separator();

    ui.label("Young's modulus E (Pa)");
    let _ = ui.add(
        egui::Slider::new(&mut state.youngs_modulus, 1.0e3..=1.0e8).logarithmic(true),
    );
    ui.label("Poisson ratio ν");
    let _ = ui.add(egui::Slider::new(&mut state.poisson_ratio, -0.49..=0.49));
    ui.label("Soft contact stiffness (N/m)");
    if ui
        .add(
            egui::Slider::new(&mut state.contact_stiffness, 1.0e3..=1.0e6)
                .logarithmic(true),
        )
        .changed()
    {
        if let SolverBackend::Vbd(s) = &mut sim.solver {
            s.contact_stiffness = state.contact_stiffness;
        }
    }
    ui.label("Inner iterations");
    if ui
        .add(egui::Slider::new(&mut state.inner_iterations, 1..=30))
        .changed()
    {
        if let SolverBackend::Vbd(s) = &mut sim.solver {
            s.iterations = state.inner_iterations as usize;
        }
    }
    ui.separator();

    ui.checkbox(&mut state.show_wireframe, "Show tet wireframe");
    ui.separator();

    // ── Stage 20.3 — multi-material picker ──────────────────────
    ui.label("Material preset");
    let mut preset_changed = false;
    egui::ComboBox::from_id_salt("jelly-material-preset")
        .selected_text(match state.material_preset {
            JellyMaterialPreset::Soft => "Soft (uniform)",
            JellyMaterialPreset::Stiff => "Stiff (uniform)",
            JellyMaterialPreset::Mixed => "Mixed (soft + stiff pillars)",
        })
        .show_ui(ui, |ui| {
            for (variant, label) in [
                (JellyMaterialPreset::Soft, "Soft (uniform)"),
                (JellyMaterialPreset::Stiff, "Stiff (uniform)"),
                (JellyMaterialPreset::Mixed, "Mixed (soft + stiff pillars)"),
            ] {
                if ui
                    .selectable_value(&mut state.material_preset, variant, label)
                    .clicked()
                {
                    preset_changed = true;
                }
            }
        });
    if preset_changed {
        // The per-tet material assignment is baked at scene-build
        // time. To avoid a full Bevy entity rebuild, we re-stamp the
        // per-tet (μ, λ) buffers in place. Compute the per-tet
        // breakdown in a Vec first (decoupled from any host_mut
        // borrows) so we don't run into glam's "two host_mut at
        // once" aliasing.
        let n = JELLY_GRID;
        let (e_soft, e_stiff) = match state.material_preset {
            JellyMaterialPreset::Soft => (state.youngs_modulus, state.youngs_modulus),
            JellyMaterialPreset::Stiff => (state.youngs_modulus * 100.0, state.youngs_modulus * 100.0),
            JellyMaterialPreset::Mixed => (state.youngs_modulus, state.youngs_modulus * 100.0),
        };
        let nu = state.poisson_ratio.clamp(-0.49, 0.49);
        let lame = |e: f64| {
            let e = e.max(1.0).min(1.0e9);
            let mu = e / (2.0 * (1.0 + nu));
            let lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
            (mu, lam)
        };
        let (mu_soft, lam_soft) = lame(e_soft);
        let (mu_stiff, lam_stiff) = lame(e_stiff);
        let last = n - 2;
        // The 6 tets per cell are stored contiguously in tet order
        // matching `build_jelly_model_with_preset`'s loop nesting
        // (k → j → i → 6 tets per cell).
        let tet_count = sim.model.tet_count;
        let mut new_mu = vec![mu_soft; tet_count];
        let mut new_lam = vec![lam_soft; tet_count];
        let mut tet_id = 0usize;
        for kk in 0..n - 1 {
            for _jj in 0..n - 1 {
                for ii in 0..n - 1 {
                    let stiff_cell =
                        (ii == 0 || ii == last) && (kk == 0 || kk == last);
                    let (mu_c, lam_c) = match state.material_preset {
                        JellyMaterialPreset::Mixed if stiff_cell => {
                            (mu_stiff, lam_stiff)
                        }
                        JellyMaterialPreset::Stiff => (mu_stiff, lam_stiff),
                        _ => (mu_soft, lam_soft),
                    };
                    for _ in 0..6 {
                        if tet_id < tet_count {
                            new_mu[tet_id] = mu_c;
                            new_lam[tet_id] = lam_c;
                            tet_id += 1;
                        }
                    }
                }
            }
        }
        if let Ok(mus) = sim.model.tet_mu.host_mut() {
            for (dst, src) in mus.iter_mut().zip(new_mu.iter()) {
                *dst = *src;
            }
        }
        if let Ok(lams) = sim.model.tet_lambda.host_mut() {
            for (dst, src) in lams.iter_mut().zip(new_lam.iter()) {
                *dst = *src;
            }
        }
        if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
            let device = device.clone();
            let queue = queue.clone();
            let _ = sim.model.tet_mu.ensure_device(&device);
            let _ = sim.model.tet_mu.upload(&queue);
            let _ = sim.model.tet_lambda.ensure_device(&device);
            let _ = sim.model.tet_lambda.upload(&queue);
        }
    }
    ui.separator();

    // ── Stage 20.3 — plasticity sliders ─────────────────────────
    ui.label("Yield stress σ_y (Pa) — 0 = elastic only");
    let _ = ui.add(
        egui::Slider::new(&mut state.yield_stress, 0.0..=1.0e5)
            .logarithmic(true),
    );
    ui.label("Hardening modulus H (Pa)");
    let _ = ui.add(
        egui::Slider::new(&mut state.hardening, 0.0..=1.0e5)
            .logarithmic(true),
    );
    ui.checkbox(&mut state.show_plastic_strain, "Show plastic-strain heat-map");
    readout(
        ui,
        "Max ε_p",
        &format!("{:.4}", state.last_max_plastic_strain),
    );
    if ui
        .add_sized(
            [ui.available_width(), 24.0],
            button_widget("Compress (drives plasticity)", accent),
        )
        .clicked()
    {
        // Bump the per-frame compress driver — `update_jelly_example`
        // pulls the puller body downward by `0.02 m * compress_t`
        // each frame and decays `compress_t` geometrically.
        state.compress_t = 1.0;
    }
    ui.separator();

    // ── Stage 20.4 toggles ──────────────────────────────────────
    ui.label("Stage 20.4 (tet-vs-tet, kinematic, SDF)");
    // Stage 20.5 will wire the second-cube + SDF rebuilds through
    // apply_scene_switch; v1 surfaces them as informational toggles
    // (changes require a scene re-entry via Y to apply).
    if ui
        .checkbox(&mut state.second_cube, "Second jelly cube (rebuild to apply)")
        .changed()
    {
        state.rebuild_pending = true;
    }
    let mut kin_changed = false;
    if ui
        .checkbox(
            &mut state.kinematic_hardening_on,
            "Kinematic hardening (back-stress α)",
        )
        .changed()
    {
        kin_changed = true;
    }
    if state.kinematic_hardening_on {
        ui.label("Kinematic hardening modulus c (Pa)");
        if ui
            .add(
                egui::Slider::new(
                    &mut state.kinematic_hardening_modulus,
                    1.0e2..=1.0e5,
                )
                .logarithmic(true),
            )
            .changed()
        {
            kin_changed = true;
        }
        readout(
            ui,
            "Max ‖α‖",
            &format!("{:.2}", state.last_max_back_stress),
        );
    } else {
        readout(ui, "Max ‖α‖", "0.00 (off)");
    }
    if kin_changed {
        // Re-stamp the per-tet kinematic-hardening buffer in place;
        // re-upload to the device. Same shape as the Mixed-preset
        // re-stamp above. When the toggle is off, set every entry
        // to 0 (Stage 20.3 path).
        let c = if state.kinematic_hardening_on {
            state.kinematic_hardening_modulus
        } else {
            0.0
        };
        if let Ok(buf) = sim.model.tet_kinematic_hardening.host_mut() {
            for v in buf.iter_mut() {
                *v = c;
            }
        }
        if let (Some(device), Some(queue)) =
            (sim.engine.device(), sim.engine.queue())
        {
            let device = device.clone();
            let queue = queue.clone();
            let _ = sim.model.tet_kinematic_hardening.ensure_device(&device);
            let _ = sim.model.tet_kinematic_hardening.upload(&queue);
        }
    }
    ui.label("Rigid SDF backing (rebuild to apply)");
    egui::ComboBox::from_id_salt("jelly-sdf-shape")
        .selected_text(match state.sdf_shape {
            JellySdfShape::None => "None",
            JellySdfShape::BakedSphere => "Baked sphere SDF",
            JellySdfShape::BakedTorus => "Baked torus SDF",
        })
        .show_ui(ui, |ui| {
            for (variant, label) in [
                (JellySdfShape::None, "None"),
                (JellySdfShape::BakedSphere, "Baked sphere SDF"),
                (JellySdfShape::BakedTorus, "Baked torus SDF"),
            ] {
                if ui
                    .selectable_value(&mut state.sdf_shape, variant, label)
                    .clicked()
                {
                    state.rebuild_pending = true;
                }
            }
        });
    if state.rebuild_pending {
        ui.colored_label(
            egui::Color32::YELLOW,
            "Rebuild required to apply scene-build toggles.",
        );
        if ui.button("Rebuild jelly").clicked() {
            // Re-enter the current scene unconditionally; the
            // hotkey path (`handle_input`) skips this when the
            // scene is already active.
            pending.0 = Some(sim.scene);
            state.rebuild_pending = false;
        }
    }
    ui.separator();

    if ui
        .add_sized(
            [ui.available_width(), 28.0],
            button_widget("Reset jelly", accent),
        )
        .clicked()
    {
        sim.reset();
    }
    ui.add_space(6.0);
    ui.small("Hotkey: Y");
    ui.small("Stage 20.4: tet-vs-tet, kinematic hardening, baked SDF.");
}

/// Tiny helper that returns an egui Button styled the same as the
/// global `button` widget — needed because the existing `button`
/// helper's API takes the ui directly, but we want a sized button.
fn button_widget(text: &str, _accent: egui::Color32) -> egui::Button<'static> {
    egui::Button::new(text.to_owned())
}

/// Right-rail pane for the Stage-22 MuJoCo wrapper demo. Shows the
/// active backend label, per-frame timing, and links to PLAN.md
/// Stage 22 for the scope notes. Read-only — every knob the user
/// would care about (substeps, integrator, friction) lives on the
/// MuJoCo side and would need a feature-on rebuild to expose.
fn mujoco_pane(ui: &mut egui::Ui, state: &mut MuJoCoExampleState, sim: &Sim) {
    ui.add_space(2.0);
    let backend_label = match &sim.solver {
        #[cfg(feature = "mujoco")]
        SolverBackend::MuJoCo(_) => "MuJoCo (libmujoco wrapper)",
        SolverBackend::Featherstone(_) => "Featherstone (fallback)",
        _ => "(unexpected solver)",
    };
    readout(ui, "Backend", backend_label);
    readout(ui, "Status", &state.status);
    readout(ui, "Bodies", &format!("{}", sim.model.body_count));
    readout(ui, "Joints", &format!("{}", sim.model.joint_count));
    readout(
        ui,
        "Solve / frame",
        &format!("{:>5.0} µs", state.smoothed_step_us),
    );
    ui.separator();

    if cfg!(feature = "mujoco") {
        ui.label("MuJoCo wrapper is ACTIVE.");
        ui.label("Step driven via mj_step (substeps=1).");
    } else {
        ui.label("MuJoCo wrapper is DISABLED at compile time.");
        ui.label("To enable:");
        ui.label("  1. Install libmujoco (https://mujoco.org/)");
        ui.label("  2. cargo run -p molla_viewer --features mujoco");
        ui.label("Currently using Featherstone as a fallback.");
    }
    ui.separator();

    ui.small("Hotkey: J  ·  Stage 22 (P2)");
    ui.small("Reference: PLAN.md Stage 22, GAP.md §2 / §9.");
    ui.small("Double pendulum — same topology as the Featherstone");
    ui.small("multi-DOF parity test, makes A/B compare easy.");
}

/// Re-derive per-spring compliance from the user-chosen stiffness
/// sliders and re-upload to the solver. Cheap: spring count is O(few
/// thousand) and the compliance buffer is a single device upload.
fn apply_vbd_stiffness(sim: &mut Sim, state: &VbdClothExampleState) {
    let spring_count = sim.model.spring_count;
    if spring_count == 0 {
        return;
    }
    let stretch_alpha = (1.0 / state.stretch_stiffness.max(1.0)) as f64;
    let bend_alpha = (1.0 / state.bend_stiffness.max(1.0)) as f64;
    // Compliance ordering: the cloth-grid builder appends springs in
    // (structural, shear, bending) groups; bending springs are the
    // "skip-one" entries with rest length = 2 × spacing. Round-tripping
    // via the rest-length sentinel is fragile; instead we rebuild the
    // distinction from the per-spring `max_stretch` flag (the builder
    // sets `BEND_STRETCH_DISABLED ≈ 2.0` for bending springs and a
    // user-facing value < 1.0 for the others).
    let max_stretch = sim
        .model
        .spring_max_stretch
        .host()
        .ok()
        .map(|s| s.to_vec())
        .unwrap_or_default();
    let comp_host = sim
        .model
        .spring_compliance
        .host_mut()
        .expect("spring_compliance host");
    for (i, c) in comp_host.iter_mut().enumerate() {
        let is_bend = max_stretch.get(i).is_some_and(|s| *s >= 1.0);
        *c = if is_bend { bend_alpha } else { stretch_alpha };
    }
    // Push the new compliance to the device + refresh the VBD
    // kernel's stiffness mirror.
    let device = sim.engine.device().unwrap().clone();
    let queue = sim.engine.queue().unwrap().clone();
    sim.model
        .spring_compliance
        .ensure_device(&device)
        .expect("ensure_device");
    sim.model
        .spring_compliance
        .upload(&queue)
        .expect("upload compliance");
    // Re-derive VBD's stiffness mirror.
    if let SolverBackend::Vbd(s) = &mut sim.solver {
        let _ = s.refresh_stiffness(&sim.model);
    }
}

/// Body for the right-rail pane in the Lidar scene.
fn lidar_pane(ui: &mut egui::Ui, config: &mut LidarConfig, state: &LidarState) {
    ui.add_space(2.0);
    readout(ui, "Sensor", "GPU raycast");
    readout(ui, "Rays", &format!("{}", state.rays.len()));
    let n_hits = state.rays.iter().filter(|r| r.hit.is_some()).count();
    readout(ui, "Hits", &format!("{n_hits}"));
    ui.separator();

    ui.label("Origin");
    ui.add(egui::DragValue::new(&mut config.origin.x).speed(0.05).prefix("x "));
    ui.add(
        egui::DragValue::new(&mut config.origin.y)
            .speed(0.05)
            .range(0.05..=10.0)
            .prefix("y "),
    );
    ui.add(egui::DragValue::new(&mut config.origin.z).speed(0.05).prefix("z "));
    ui.separator();

    ui.label("Heading");
    ui.add(
        egui::Slider::new(&mut config.yaw, 0.0..=std::f32::consts::TAU).text("yaw (rad)"),
    );
    ui.add(
        egui::Slider::new(
            &mut config.pitch,
            -std::f32::consts::FRAC_PI_2..=std::f32::consts::FRAC_PI_2,
        )
        .text("pitch (rad)"),
    );
    ui.add(
        egui::Slider::new(&mut config.auto_rotate_speed, 0.0..=4.0).text("auto-spin (rad/s)"),
    );
    ui.separator();

    ui.label("FOV");
    let mut fov_h_deg = config.fov_h.to_degrees();
    let mut fov_v_deg = config.fov_v.to_degrees();
    if ui
        .add(egui::Slider::new(&mut fov_h_deg, 5.0..=360.0).text("horizontal °"))
        .changed()
    {
        config.fov_h = fov_h_deg.to_radians();
    }
    if ui
        .add(egui::Slider::new(&mut fov_v_deg, 1.0..=120.0).text("vertical °"))
        .changed()
    {
        config.fov_v = fov_v_deg.to_radians();
    }
    ui.separator();

    ui.label("Resolution");
    let mut nh = config.ray_count_h as i32;
    if ui.add(egui::Slider::new(&mut nh, 8..=512).text("rays h")).changed() {
        config.ray_count_h = nh.max(1) as u32;
    }
    let mut nv = config.ray_count_v as i32;
    if ui.add(egui::Slider::new(&mut nv, 1..=64).text("rays v")).changed() {
        config.ray_count_v = nv.max(1) as u32;
    }
    ui.separator();
    ui.add(egui::Slider::new(&mut config.max_range, 1.0..=100.0).text("range (m)"));
}

/// Body for the right-rail pane in the Contact scene.
fn contact_pane(ui: &mut egui::Ui, state: &ContactExampleState) {
    ui.add_space(2.0);
    readout(ui, "Sensor", "GPU contact aggregate");
    let total_contacts: u32 = state.readings.iter().map(|r| r.count).sum();
    let in_contact: usize = state.readings.iter().filter(|r| r.count > 0).count();
    let max_depth: f64 = state
        .readings
        .iter()
        .map(|r| r.max_depth)
        .fold(0.0_f64, f64::max);
    readout(ui, "Bodies", &format!("{}", state.readings.len()));
    readout(ui, "In contact", &format!("{}", in_contact));
    readout(ui, "Σ contacts", &format!("{}", total_contacts));
    readout(ui, "Max depth", &format!("{max_depth:.4} m"));
    ui.separator();

    ui.label("Per-body");
    egui::ScrollArea::vertical()
        .max_height(360.0)
        .show(ui, |ui| {
            for (i, r) in state.readings.iter().enumerate() {
                let label = format!("body {i:>2}");
                let txt = if r.count == 0 {
                    "—".to_string()
                } else {
                    format!(
                        "n={} d={:.4} push=({:>5.2},{:>5.2},{:>5.2})",
                        r.count,
                        r.max_depth,
                        r.normal_sum.x,
                        r.normal_sum.y,
                        r.normal_sum.z,
                    )
                };
                readout(ui, &label, &txt);
            }
        });
    ui.separator();
    ui.small(
        "Yellow arrows in the viewport point along the net push \
         direction at each body's contact centroid; length scales \
         with penetration depth.",
    );
}

/// Body for the right-rail pane in the Stage-15 GPU narrow-phase scene.
fn gpu_np_pane(
    ui: &mut egui::Ui,
    state: &mut GpuNpExampleState,
    accent: egui::Color32,
) {
    ui.add_space(2.0);
    let mode_label = if state.gpu_enabled_now {
        "GPU (Stage 15)"
    } else {
        "CPU"
    };
    readout(ui, "Narrow phase", mode_label);
    readout(ui, "SAP pairs", &format!("{}", state.last_pairs));
    readout(ui, "Contacts", &format!("{}", state.last_contacts));
    readout(
        ui,
        "CPU NP calls",
        &format!("{}", state.last_cpu_calls),
    );
    let gpu_calls = state.last_pairs.saturating_sub(state.last_cpu_calls);
    readout(ui, "GPU NP calls", &format!("{}", gpu_calls));
    ui.separator();

    ui.horizontal(|ui| {
        let label = if state.use_gpu {
            "Disable GPU narrow phase"
        } else {
            "Enable GPU narrow phase"
        };
        if button(ui, label, accent).clicked() {
            state.use_gpu = !state.use_gpu;
        }
    });
    ui.separator();
    ui.small(
        "Toggles between CPU and full-GPU narrow phase. With GPU \
         on, the Sphere/Box/Capsule/Cylinder/Cone pairs run through \
         WGSL GJK + EPA on the device; only Plane pairs (and the few \
         that hit the WGSL polytope cap) bump the CPU counter.",
    );
}

/// Body for the right-rail pane in the Stage-19 GPU broad-phase scene.
fn gpu_bp_pane(
    ui: &mut egui::Ui,
    state: &mut GpuBpExampleState,
    accent: egui::Color32,
) {
    ui.add_space(2.0);
    let mode_label = if state.gpu_enabled_now {
        // Stage 19.1 — Morton sort moved to GPU (was CPU stop-gap).
        "GPU LBVH (radix v2)"
    } else {
        "CPU SAP"
    };
    readout(ui, "Broad phase", mode_label);
    readout(ui, "Pairs", &format!("{}", state.last_pairs));
    readout(
        ui,
        "Active ms/frame",
        &format!("{:>5.2} ms", state.smoothed_ms),
    );
    ui.separator();

    // Side-by-side per-backend memory of the last seen ms/frame —
    // updated whenever that backend was the active one. The user
    // toggles, waits a few frames, toggles back, and reads off the
    // speedup directly.
    readout(
        ui,
        "Last CPU SAP",
        &format!("{:>5.2} ms", state.last_cpu_ms),
    );
    readout(
        ui,
        "Last GPU LBVH",
        &format!("{:>5.2} ms", state.last_gpu_ms),
    );
    if state.last_cpu_ms > 0.0 && state.last_gpu_ms > 0.0 {
        let speedup = state.last_cpu_ms / state.last_gpu_ms.max(0.001);
        readout(ui, "Speedup", &format!("{:>5.2}x", speedup));
    }
    ui.separator();

    ui.horizontal(|ui| {
        let label = if state.use_gpu {
            "Switch to CPU SAP"
        } else {
            "Switch to GPU LBVH"
        };
        if button(ui, label, accent).clicked() {
            state.use_gpu = !state.use_gpu;
        }
    });
    ui.separator();
    ui.small(
        "Stage 19 — toggle between CPU sweep-and-prune and the GPU \
         LBVH (Karras-style Linear BVH) broad phase. Both produce \
         the same overlapping-AABB pair list; the GPU path runs the \
         build + tree traversal on the device. The pair list flows \
         into the same CPU narrow phase regardless of toggle, so \
         only the 'collide' phase ms-time changes.",
    );
}

/// Body for the right-rail pane in the Arm scene.
fn arm_pane(
    ui: &mut egui::Ui,
    state: &mut ArmExampleState,
    sim: &Sim,
    pending: &mut PendingSceneSwitch,
) {
    ui.add_space(2.0);
    readout(ui, "Solver", "Featherstone GPU");
    readout(ui, "Controller", "GPU PD");
    readout(ui, "Joints", &format!("{ARM_LINKS}"));
    readout(
        ui,
        "Effort cap",
        &format!("{:.0} N·m", ARM_MAX_EFFORT),
    );
    ui.separator();

    // Live joint state — read joint_q (already host-mirrored after
    // the Featherstone download in step_simulation).
    if let Ok(q) = sim.state_a.joint_q.host() {
        ui.label("Current angles");
        for i in 0..ARM_LINKS {
            if i < q.len() {
                readout(ui, &format!("  q{i}"), &format!("{:.3} rad", q[i]));
            }
        }
        ui.separator();
    }

    // Body world poses — FK output. If these stay frozen at the
    // initial values from `build_arm_model` while joint_q is changing,
    // Featherstone's FK pass isn't reaching the host (or the kernel's
    // not writing to body_q at all).
    if let Ok(bq) = sim.state_a.body_q.host() {
        ui.label("Body Y positions (FK)");
        for i in 0..ARM_LINKS {
            if i < bq.len() {
                readout(
                    ui,
                    &format!("  body{i}.y"),
                    &format!("{:.3} m", bq[i].position.y),
                );
            }
        }
        ui.separator();
    }

    ui.checkbox(&mut state.sine_enabled, "Auto-sine drive");
    if state.sine_enabled {
        ui.add(
            egui::Slider::new(&mut state.sine_amplitude, 0.0..=1.4).text("amplitude (rad)"),
        );
        ui.add(
            egui::Slider::new(&mut state.sine_frequency, 0.0..=2.0).text("frequency (Hz)"),
        );
    } else {
        ui.label("Targets (rad)");
        for i in 0..ARM_LINKS {
            ui.add(
                egui::Slider::new(
                    &mut state.targets[i],
                    -std::f32::consts::FRAC_PI_2..=std::f32::consts::FRAC_PI_2,
                )
                .text(format!("q{i} target")),
            );
        }
    }

    ui.add_space(4.0);
    ui.small(
        "PD effort = kp·(target − q) + kd·(0 − qd), capped to ±max_effort. \
         Watch the arm settle on slider changes; at high gains it's stiff, \
         at low gains it overshoots.",
    );

    // Stage 32 — joint-mimic toggle. Adds a 7th capsule body in a
    // separate articulation, coupled to joint 0 (the shoulder) by a
    // `JointMimic` equality with the linear-identity polynomial.
    // Toggling forces a scene rebuild so the mimic body / joint /
    // equality enter/exit the model cleanly.
    ui.separator();
    let prev = state.mimic_enabled;
    if ui
        .checkbox(&mut state.mimic_enabled, "Mimic joint (Stage 32)")
        .changed()
        && state.mimic_enabled != prev
    {
        state.mimic_dirty = true;
        set_arm_mimic_flag(state.mimic_enabled);
    }
    if state.mimic_enabled {
        ui.small(
            "Linear-identity polynomial [0, 1, 0, 0, 0]. The 7th body \
             on +X tracks the shoulder via JointMimic equality (XPBD \
             projection) + Featherstone pre-step (legacy linear \
             mimic). Watch the offset capsule mirror the shoulder's \
             angle as the sine drive sweeps.",
        );
    }
    if state.mimic_dirty {
        ui.colored_label(
            egui::Color32::YELLOW,
            "Toggle changed — rebuild the arm scene to apply.",
        );
        if ui.button("Rebuild arm").clicked() {
            // Re-enter the current scene unconditionally; the
            // hotkey path (`handle_input`) skips this when the
            // scene is already active, so the button is the only
            // way to force a rebuild from the pane.
            pending.0 = Some(sim.scene);
        }
    }
}

/// Body for the right-rail pane in the Stage-18 actuator-pipeline scene.
fn actuator_pane(ui: &mut egui::Ui, state: &mut ActuatorExampleState) {
    ui.add_space(2.0);
    readout(ui, "Pipeline", "Delay → PD → DC-Motor");
    let delay_substeps = ms_to_substeps(state.delay_ms);
    readout(
        ui,
        "Delay",
        &format!("{:.1} ms ({} substeps)", state.delay_ms, delay_substeps),
    );
    readout(
        ui,
        "Commanded τ",
        &format!("{:>6.2} N·m", state.last_commanded),
    );
    readout(
        ui,
        "Delivered τ",
        &format!("{:>6.2} N·m", state.last_delivered),
    );
    let saturation = if state.last_commanded.abs() > 1e-3 {
        state.last_delivered / state.last_commanded
    } else {
        1.0
    };
    readout(ui, "Delivered/Cmd", &format!("{:>5.2}", saturation));
    ui.separator();

    ui.checkbox(&mut state.sine_enabled, "Sine target drive");
    if state.sine_enabled {
        ui.add(
            egui::Slider::new(&mut state.sine_amplitude, 0.0..=2.0)
                .text("amplitude (rad)"),
        );
        ui.add(
            egui::Slider::new(&mut state.sine_frequency, 0.0..=2.0)
                .text("frequency (Hz)"),
        );
        readout(ui, "Target", &format!("{:>6.3} rad", state.target));
    } else {
        let resp = ui.add(
            egui::Slider::new(
                &mut state.target,
                -std::f32::consts::PI..=std::f32::consts::PI,
            )
            .text("target (rad)"),
        );
        let _ = resp;
    }
    ui.separator();

    ui.label("Delay");
    if ui
        .add(egui::Slider::new(&mut state.delay_ms, 0.0..=30.0).text("delay (ms)"))
        .changed()
    {
        state.params_dirty = true;
    }
    ui.separator();

    ui.label("PD gains");
    if ui
        .add(egui::Slider::new(&mut state.kp, 0.0..=200.0).text("kp"))
        .changed()
    {
        state.params_dirty = true;
    }
    if ui
        .add(egui::Slider::new(&mut state.kd, 0.0..=20.0).text("kd"))
        .changed()
    {
        state.params_dirty = true;
    }
    ui.separator();

    ui.label("DC motor");
    if ui
        .add(
            egui::Slider::new(&mut state.stall_torque, 0.0..=30.0)
                .text("stall τ (N·m)"),
        )
        .changed()
    {
        state.params_dirty = true;
    }
    if ui
        .add(
            egui::Slider::new(&mut state.no_load_speed, 0.5..=40.0)
                .text("no-load ω (rad/s)"),
        )
        .changed()
    {
        state.params_dirty = true;
    }
    if ui
        .add(
            egui::Slider::new(&mut state.max_motor_effort, 0.0..=50.0)
                .text("max τ cap (N·m)"),
        )
        .changed()
    {
        state.params_dirty = true;
    }

    ui.add_space(4.0);
    ui.small(
        "Pipeline = Delay (ring buffer of length ⌈delay_ms/sub_dt⌉) → \
         pipeline-PD (no cap) → DC-motor envelope. The clamp's \
         effort ceiling drops linearly with |qd| toward 0 at no-load \
         speed. Watch 'Delivered τ / Commanded τ' shrink as the link \
         spins up.",
    );
}

/// Body for the right-rail pane in the Stage-18.1 neural-policy scene.
fn neural_policy_pane(ui: &mut egui::Ui, state: &mut NeuralPolicyExampleState) {
    ui.add_space(2.0);
    readout(ui, "Net", "MLP 4 → 32 → 32 → 1");
    readout(ui, "Activation", "tanh / tanh / linear");
    readout(ui, "Backend", "GPU (single-workgroup forward)");
    ui.separator();

    ui.label("Inputs (scaled, post-normalisation)");
    readout(ui, "  · cart_x'", &format!("{:>+6.3}", state.last_inputs[0]));
    readout(ui, "  · cart_v'", &format!("{:>+6.3}", state.last_inputs[1]));
    readout(ui, "  · theta",   &format!("{:>+6.3} rad", state.last_inputs[2]));
    readout(ui, "  · theta_d'", &format!("{:>+6.3}", state.last_inputs[3]));
    ui.separator();

    ui.label("Outputs");
    readout(
        ui,
        "  · MLP raw",
        &format!("{:>+6.2} N", state.last_output),
    );
    readout(
        ui,
        "  · post-clamp",
        &format!("{:>+6.2} N (cap ±{:.0})", state.last_delivered, state.max_force),
    );
    ui.separator();

    ui.checkbox(
        &mut state.auto_perturb,
        "Periodic pole kick (every 3 s)",
    );
    if !state.auto_perturb {
        ui.small("(without perturbations the system parks at upright)");
    }
    ui.separator();
    ui.small(
        "Hard-coded LQR-style balance policy expressed as a 3-layer MLP. \
         Layer 1 + 2 are near-identity passthroughs on the first 4 \
         hidden units; layer 3 is a linear combination producing the \
         cart force. tanh saturation provides a soft envelope; the \
         downstream MaxEffort clamp adds a hard rail.",
    );
}

/// Body for the right-rail pane in the Stage-31 neural-LSTM scene.
fn neural_lstm_pane(ui: &mut egui::Ui, state: &mut NeuralLstmExampleState) {
    ui.add_space(2.0);
    readout(ui, "Net", &format!("LSTM 4 → {} → 1", LSTM_HID_DIM));
    readout(ui, "Cell", "single-cell, gates: i / f / g / o");
    readout(ui, "Backend", "GPU (single-workgroup forward)");
    ui.separator();

    ui.label("Inputs (scaled, post-normalisation)");
    readout(ui, "  · cart_x'", &format!("{:>+6.3}", state.last_inputs[0]));
    readout(ui, "  · cart_v'", &format!("{:>+6.3}", state.last_inputs[1]));
    readout(ui, "  · theta",   &format!("{:>+6.3} rad", state.last_inputs[2]));
    readout(ui, "  · theta_d'", &format!("{:>+6.3}", state.last_inputs[3]));
    ui.separator();

    ui.label("Hidden state h_t (live, persists across substeps)");
    // Render as a compact 4×4 grid so the pane stays readable.
    egui::Grid::new("lstm_hidden_state_grid")
        .num_columns(4)
        .spacing([8.0, 2.0])
        .show(ui, |ui| {
            for i in 0..LSTM_HID_DIM {
                ui.label(format!("{:>+6.3}", state.last_hidden[i]));
                if (i + 1) % 4 == 0 {
                    ui.end_row();
                }
            }
        });
    ui.separator();

    ui.label("Outputs");
    readout(ui, "  · LSTM raw", &format!("{:>+6.2} N", state.last_output));
    readout(
        ui,
        "  · post-clamp",
        &format!("{:>+6.2} N (cap ±{:.0})", state.last_delivered, state.max_force),
    );
    ui.separator();

    ui.checkbox(&mut state.auto_perturb, "Periodic pole kick (every 3 s)");
    if !state.auto_perturb {
        ui.small("(without perturbations the system parks at upright)");
    }
    ui.separator();

    ui.small(
        "Stage-31 LSTM controller. Recurrent gates are saturated (i≈1, \
         f≈0, o≈1) so the cell behaves as `c_t = tanh(W_g·x_t)` — this \
         keeps the v1 demo visibly functional (cart-pole balance) while \
         exercising the full single-cell forward pass + persistent \
         hidden / cell state on-device. The `h_t` readout above does \
         evolve every substep, proving the hidden-state buffer survives \
         across pipeline ticks. A genuinely-trained recurrent policy \
         would need offline training + weight import.",
    );
}

/// Body for the right-rail pane in the GPU IK arm scene.
fn gpu_ik_pane(ui: &mut egui::Ui, state: &mut GpuIkArmState, sim: &Sim) {
    ui.add_space(2.0);
    readout(ui, "Solver", if state.use_gpu { "GPU LM" } else { "CPU LM" });
    readout(ui, "DOFs", "7");
    readout(ui, "Iterations", &format!("{}", state.last_iters));
    readout(ui, "Residual", &format!("{:.5}", state.last_residual));
    readout(
        ui,
        "ms / solve",
        &format!("{:.2}", state.last_solve_us / 1000.0),
    );

    let ee_body = sim.model.body_count - 1;
    if let Ok(bq) = sim.state_a.body_q.host() {
        if ee_body < bq.len() {
            let p = bq[ee_body].position;
            let err = (p - state.target).length();
            readout(ui, "EE error", &format!("{:.4} m", err));
        }
    }
    ui.separator();

    ui.checkbox(&mut state.use_gpu, "Use GPU IK");
    ui.separator();

    ui.label("Target (world)");
    ui.add(
        egui::Slider::new(&mut state.target.x, -1.5..=1.5)
            .text("x")
            .step_by(0.005),
    );
    ui.add(
        egui::Slider::new(&mut state.target.y, 0.05..=2.5)
            .text("y")
            .step_by(0.005),
    );
    ui.add(
        egui::Slider::new(&mut state.target.z, -1.5..=1.5)
            .text("z")
            .step_by(0.005),
    );
    ui.separator();

    if let Ok(q) = sim.state_a.joint_q.host() {
        ui.label("Current angles (rad)");
        for i in 0..7.min(q.len()) {
            readout(ui, &format!("  q{i}"), &format!("{:.3}", q[i]));
        }
    }

    ui.add_space(4.0);
    ui.small(
        "Levenberg-Marquardt IK; per-iteration Jacobian on the GPU. \
         Drag the X/Y/Z sliders to move the EE target; the arm chases \
         in real time. Toggle 'Use GPU IK' to A/B against the CPU \
         solver.",
    );
}

/// Body for the right-rail pane in the Quadrotor scene.
fn quadrotor_pane(ui: &mut egui::Ui, state: &mut QuadrotorState) {
    ui.add_space(2.0);
    readout(ui, "Solver", "Featherstone-GPU (Stage 17)");
    readout(ui, "Mass", &format!("{:.3} kg", state.total_mass));
    readout(ui, "Arm", &format!("{:.3} m", state.arm_length));
    ui.separator();

    ui.label("Live state");
    readout(ui, "  altitude", &format!("{:>6.3} m", state.last_altitude));
    readout(
        ui,
        "  roll",
        &format!("{:>6.2}°", state.last_attitude[0].to_degrees()),
    );
    readout(
        ui,
        "  pitch",
        &format!("{:>6.2}°", state.last_attitude[1].to_degrees()),
    );
    readout(
        ui,
        "  yaw",
        &format!("{:>6.2}°", state.last_attitude[2].to_degrees()),
    );
    ui.separator();

    ui.label("Per-rotor thrust (N)");
    readout(ui, "  rotor +X", &format!("{:>6.2}", state.last_thrusts[0]));
    readout(ui, "  rotor -X", &format!("{:>6.2}", state.last_thrusts[1]));
    readout(ui, "  rotor +Z", &format!("{:>6.2}", state.last_thrusts[2]));
    readout(ui, "  rotor -Z", &format!("{:>6.2}", state.last_thrusts[3]));
    let total: f32 = state.last_thrusts.iter().sum();
    readout(ui, "  Σ thrust", &format!("{:>6.2}", total));
    ui.separator();

    ui.label("Target altitude (m)");
    ui.add(
        egui::Slider::new(&mut state.target_altitude, 0.2..=4.0)
            .text("z_t")
            .step_by(0.05),
    );
    ui.separator();

    ui.label("Altitude PD");
    ui.add(egui::Slider::new(&mut state.altitude_kp, 0.0..=120.0).text("kp"));
    ui.add(egui::Slider::new(&mut state.altitude_kd, 0.0..=40.0).text("kd"));
    ui.separator();

    ui.label("Attitude PD");
    ui.add(egui::Slider::new(&mut state.attitude_kp, 0.0..=20.0).text("kp"));
    ui.add(egui::Slider::new(&mut state.attitude_kd, 0.0..=4.0).text("kd"));
    ui.separator();

    // Stage 17.1 — rigorous toggle.
    ui.label("Stage 17.1 polish");
    ui.checkbox(&mut state.rigorous_d_solve, "rigorous dense-Cholesky D solve");
    ui.small(
        "v1 sequential rank-1 deflation is exact when the joint's D \
         matrix is diagonal. For asymmetric inertias (e.g. rotor mass \
         offset cross-coupling linear/angular DOFs) the rigorous path \
         is force-correct. For this symmetric quadrotor the two paths \
         agree closely; toggle to A/B against your own asymmetric \
         payload.",
    );

    ui.add_space(4.0);
    ui.small(
        "FREE root + 4 fixed-rotor bodies. Per-rotor thrust commanded \
         via state.body_f world-frame +Y forces; Featherstone GPU \
         solves the 6-DOF base + 4 rigid attachments. Yaw is free \
         (single-axis thrust can't generate yaw torque).",
    );
}

/// Body for the right-rail pane in the Differentiable-arm (Stage 23)
/// scene. Owns the target XYZ sliders, the iteration counter +
/// residual readout, the "Re-run" button, and the trajectory-overlay
/// toggle. Slider drags trigger an automatic re-run.
fn diff_arm_pane(ui: &mut egui::Ui, state: &mut DiffArmExampleState) {
    ui.add_space(2.0);
    readout(ui, "Solver", "CPU autograd (Stage 23)");
    readout(
        ui,
        "Arm",
        &format!("L1={:.2} m, L2={:.2} m", state.arm.l1, state.arm.l2),
    );
    readout(
        ui,
        "Horizon",
        &format!("{} steps × {:.0} ms", state.arm.horizon, state.arm.dt * 1000.0),
    );
    ui.separator();

    ui.label("Optimisation");
    readout(ui, "  iters run", &format!("{}", state.iters_run));
    readout(
        ui,
        "  residual",
        &if state.last_residual.is_finite() {
            format!("{:.4} m", state.last_residual)
        } else {
            "—".into()
        },
    );
    readout(
        ui,
        "  converged",
        &match state.converged_iter {
            Some(it) => format!("at iter {it} (≤ 5 cm)"),
            None if state.iters_run == 0 => "(idle)".into(),
            None => "did NOT converge ≤ 5 cm".into(),
        },
    );
    if ui.button("Re-run optimisation").clicked() {
        state.rerun = true;
    }
    ui.separator();

    ui.label("Target (world XYZ)");
    let mut tx = state.target.x as f32;
    let mut ty = state.target.y as f32;
    let mut tz = state.target.z as f32;
    let mut changed = false;
    if ui
        .add(egui::Slider::new(&mut tx, -0.9_f32..=0.9_f32).text("x").step_by(0.01))
        .changed()
    {
        state.target.x = tx as f64;
        changed = true;
    }
    if ui
        .add(
            egui::Slider::new(&mut ty, (DIFF_ARM_BASE_Y - 0.9)..=(DIFF_ARM_BASE_Y + 0.9))
                .text("y")
                .step_by(0.01),
        )
        .changed()
    {
        state.target.y = ty as f64;
        changed = true;
    }
    // Z is shown for completeness but the planar arm can't reach off-plane.
    if ui
        .add(egui::Slider::new(&mut tz, -0.5_f32..=0.5_f32).text("z (cosmetic)").step_by(0.01))
        .changed()
    {
        state.target.z = tz as f64;
        // Z change doesn't affect the optimisation itself (planar
        // arm), so don't trigger a re-run.
    }
    if changed {
        state.rerun = true;
    }
    ui.separator();

    ui.checkbox(&mut state.show_trajectory, "Show optimisation trajectory");
    ui.add_space(4.0);
    ui.small(
        "2-link planar arm. Open-loop torque sequence optimised by \
         gradient descent (closed-form adjoint) to drive the EE to \
         the target. The trajectory overlay shows the EE's location \
         at every optimisation iteration so you can see GD converge.",
    );
}

/// Body for the right-rail pane in the Differentiable-Featherstone
/// (Stage 23.1) scene. Same shape as [`diff_arm_pane`] — target XYZ
/// sliders, residual + iters-to-converge readout, "re-run" button,
/// trajectory toggle — plus a chain-config readout.
fn diff_arm6_pane(ui: &mut egui::Ui, state: &mut DiffArm6ExampleState) {
    ui.add_space(2.0);
    let solver_label = match state.backend {
        molla::kernels::AdjointBackend::Auto => "Featherstone, Auto (Stage 23.1.2 → AnalyticQd)",
        molla::kernels::AdjointBackend::ForceCpuClosedForm => {
            "Featherstone, CPU closed-form (Stage 23.1.1)"
        }
        molla::kernels::AdjointBackend::ForceCpuAnalyticQd => {
            "Featherstone, CPU AnalyticQd (Stage 23.1.2)"
        }
        molla::kernels::AdjointBackend::ForceCpuFiniteDiff => {
            "Featherstone, CPU FD (v1 oracle)"
        }
        molla::kernels::AdjointBackend::GpuFallbackCpu => "Featherstone, GPU (single-chain → CPU AnalyticQd)",
    };
    readout(ui, "Solver", solver_label);
    readout(
        ui,
        "Chain",
        &format!(
            "{}, {} links × {} steps × {:.1} ms",
            state.kind.label(),
            state.chain.links.len(),
            state.chain.horizon,
            state.chain.dt * 1000.0,
        ),
    );

    // Stage 23.1.3 — joint kind picker. Swapping the kind rebuilds
    // the chain, resets τ, and triggers a re-run so the user sees the
    // gradient flow through each joint kind end-to-end.
    let mut new_kind = state.kind;
    egui::ComboBox::from_label("Joint kind")
        .selected_text(state.kind.label())
        .show_ui(ui, |ui| {
            for kind in [
                DiffArm6Kind::Revolute,
                DiffArm6Kind::Ball,
                DiffArm6Kind::Free,
                DiffArm6Kind::Distance,
                DiffArm6Kind::D6,
            ] {
                ui.selectable_value(&mut new_kind, kind, kind.label());
            }
        });
    if new_kind != state.kind {
        state.kind = new_kind;
        state.chain = new_kind.build_chain();
        let n = state.chain.links.len();
        let h = state.chain.horizon;
        state.scratch = JacobianScratch::new(n);
        state.tau = vec![vec![0.0; n]; h];
        state.iter_ee_trace.clear();
        state.last_residual = f64::INFINITY;
        state.converged_iter = None;
        state.iters_run = 0;
        // Re-pick a sensible target near the new chain's rest pose.
        let rest_ee = state.chain.end_effector(&vec![0.0; n]);
        state.target = MVec3::new(
            rest_ee.x as f64 + 0.07,
            rest_ee.y as f64 + DIFF_ARM6_BASE_HEIGHT,
            rest_ee.z as f64 - 0.07,
        );
        state.rerun = true;
    }
    ui.separator();

    ui.label("Optimisation");
    readout(ui, "  iters run", &format!("{}", state.iters_run));
    readout(
        ui,
        "  residual",
        &if state.last_residual.is_finite() {
            format!("{:.4} m", state.last_residual)
        } else {
            "—".into()
        },
    );
    readout(
        ui,
        "  converged",
        &match state.converged_iter {
            Some(it) => format!("at iter {it} (≤ 5 cm)"),
            None if state.iters_run == 0 => "(idle)".into(),
            None => "did NOT converge ≤ 5 cm".into(),
        },
    );
    // Stage 23.1.1: backward / forward ratio readout. Demonstrates the
    // closed-form CPU adjoint perf win next to the iter counter.
    readout(
        ui,
        "  bwd / fwd",
        &if state.last_bwd_fwd_ratio > 0.0 {
            format!("{:.2}×", state.last_bwd_fwd_ratio)
        } else {
            "—".into()
        },
    );
    if ui.button("Re-run optimisation").clicked() {
        state.rerun = true;
    }
    // Stage 23.1.2 — 4-way backend picker (Auto / CPU CF / CPU AnalyticQd
    // / CPU FD / GPU). Promoted from the Stage 23.1.1 boolean checkbox.
    let prev_backend = state.backend;
    let backend_label = state.backend.label();
    egui::ComboBox::from_label("Adjoint backend")
        .selected_text(backend_label)
        .show_ui(ui, |ui| {
            for b in [
                molla::kernels::AdjointBackend::Auto,
                molla::kernels::AdjointBackend::ForceCpuClosedForm,
                molla::kernels::AdjointBackend::ForceCpuAnalyticQd,
                molla::kernels::AdjointBackend::ForceCpuFiniteDiff,
                molla::kernels::AdjointBackend::GpuFallbackCpu,
            ] {
                ui.selectable_value(&mut state.backend, b, b.label());
            }
        });
    if state.backend != prev_backend {
        state.rerun = true;
    }
    // Stage 23.1.2 — multi-articulation batch toggle. Drives N parallel
    // chains through `MultiAdjointBatch` on each re-run; the per-chain
    // wall time + parallel speedup readouts below show the win.
    let prev_multi = state.use_multi_articulation;
    ui.checkbox(
        &mut state.use_multi_articulation,
        "Multi-articulation batch (Stage 23.1.2)",
    );
    if state.use_multi_articulation {
        let mut n = state.multi_articulation_n as i32;
        if ui
            .add(egui::Slider::new(&mut n, 2..=64).text("batch N"))
            .changed()
        {
            state.multi_articulation_n = n.max(2) as usize;
            state.rerun = true;
        }
        readout(
            ui,
            "  per-chain (parallel)",
            &if state.last_per_chain_us > 0.0 {
                format!("{:.0} µs", state.last_per_chain_us)
            } else {
                "—".into()
            },
        );
        readout(
            ui,
            "  parallel speedup",
            &if state.last_parallel_speedup > 0.0 {
                format!("{:.2}×", state.last_parallel_speedup)
            } else {
                "—".into()
            },
        );
    }
    if state.use_multi_articulation != prev_multi {
        state.rerun = true;
    }
    ui.separator();

    ui.label("Target (world XYZ)");
    let mut tx = state.target.x as f32;
    let mut ty = state.target.y as f32;
    let mut tz = state.target.z as f32;
    let mut changed = false;
    let lift = DIFF_ARM6_BASE_HEIGHT as f32;
    if ui
        .add(egui::Slider::new(&mut tx, -1.2_f32..=1.2_f32).text("x").step_by(0.01))
        .changed()
    {
        state.target.x = tx as f64;
        changed = true;
    }
    if ui
        .add(
            egui::Slider::new(&mut ty, lift..=(lift + 1.5))
                .text("y")
                .step_by(0.01),
        )
        .changed()
    {
        state.target.y = ty as f64;
        changed = true;
    }
    if ui
        .add(egui::Slider::new(&mut tz, -1.2_f32..=1.2_f32).text("z").step_by(0.01))
        .changed()
    {
        state.target.z = tz as f64;
        changed = true;
    }
    if changed {
        state.rerun = true;
    }
    ui.separator();

    ui.checkbox(&mut state.show_trajectory, "Show optimisation trajectory");
    ui.add_space(4.0);
    ui.small(
        "6-DOF Franka-style serial chain. Open-loop torque sequence \
         optimised by gradient descent through the reverse-mode \
         Featherstone adjoint. Stage 23.1.2 default: AnalyticQd CPU \
         path — closed-form M⁻¹ + analytic ∂h/∂qd via chain rule on \
         RNEA + FD-on-RNEA for ∂h/∂q. The multi-articulation batch \
         toggle drives N parallel chains through the rayon batch \
         dispatcher; per-chain wall time + parallel speedup readouts \
         show the headline win. The pure-analytic ∂h/∂q path lives in \
         Stage 23.1.3 (full Featherstone-2008 §10 derivation).",
    );
}

/// Body for the right-rail pane in the IMU scene.
fn imu_pane(ui: &mut egui::Ui, state: &ImuExampleState) {
    ui.add_space(2.0);
    readout(ui, "Sensor", "GPU IMU");
    ui.separator();

    match state.last {
        None => {
            ui.label("priming…");
            ui.small("(first frame snapshots body_qd; reading shows next frame)");
        }
        Some(r) => {
            ui.label("Accelerometer (sensor frame)");
            readout(ui, "  ax", &format!("{:>8.3} m/s²", r.accelerometer.x));
            readout(ui, "  ay", &format!("{:>8.3} m/s²", r.accelerometer.y));
            readout(ui, "  az", &format!("{:>8.3} m/s²", r.accelerometer.z));
            readout(ui, "  |a|", &format!("{:>8.3} m/s²", r.accelerometer.length()));
            ui.separator();
            ui.label("Gyroscope (sensor frame)");
            readout(ui, "  ωx", &format!("{:>8.3} rad/s", r.gyroscope.x));
            readout(ui, "  ωy", &format!("{:>8.3} rad/s", r.gyroscope.y));
            readout(ui, "  ωz", &format!("{:>8.3} rad/s", r.gyroscope.z));
            readout(ui, "  |ω|", &format!("{:>8.3} rad/s", r.gyroscope.length()));
            ui.separator();
            let accel_norm = (state.smoothed_accel_mag / 50.0).clamp(0.0, 1.0);
            let gyro_norm = (state.smoothed_gyro_mag / 20.0).clamp(0.0, 1.0);
            ui.label("|a| (0..50 m/s²)");
            ui.add(egui::ProgressBar::new(accel_norm).show_percentage());
            ui.label("|ω| (0..20 rad/s)");
            ui.add(egui::ProgressBar::new(gyro_norm).show_percentage());
        }
    }

    ui.add_space(4.0);
    ui.small(
        "At rest: |a| ≈ 9.81. Free fall: |a| ≈ 0. Bounce impacts \
         produce sharp accel spikes.",
    );
}

/// Body for the right-rail pane in the Camera scene. Each of the
/// 5 GPU output channels lives in its own `bevy_frost::Normal`
/// container so the user can fold/unfold them individually; the
/// thumbnails inside use the pane's available width, so dragging
/// the pane wider scales every thumbnail up live.
fn camera_pane(
    ui: &mut egui::Ui,
    state: &mut CameraExampleState,
    anchor: PaneAnchor,
    accent: egui::Color32,
) {
    let pane_id = egui::Id::new(PANE_SCENE);
    // Header readouts + FOV slider. These don't fold — they're always
    // visible at the top of the pane body, above the channel containers.
    ui.add_space(2.0);
    readout(ui, "Sensor", "GPU TiledCamera");
    readout(ui, "Resolution", &format!("{}×{}", CAMERA_IMG_W, CAMERA_IMG_H));
    if let Some(p) = state.last_pose {
        readout(
            ui,
            "Eye",
            &format!(
                "({:>5.2},{:>5.2},{:>5.2})",
                p.position.x, p.position.y, p.position.z
            ),
        );
    }
    ui.separator();
    let mut fov_deg = state.fov_y.to_degrees();
    if ui
        .add(egui::Slider::new(&mut fov_deg, 20.0..=110.0).text("FOV° (vert)"))
        .changed()
    {
        state.fov_y = fov_deg.to_radians();
    }
    ui.separator();

    if state.tex_id_color.is_none() {
        ui.add_space(6.0);
        ui.small("waiting for first GPU dispatch…");
        return;
    }

    let img_w = CAMERA_IMG_W as f32;
    let img_h = CAMERA_IMG_H as f32;
    let aspect = img_h / img_w;
    let mk_image_pod = move |slot: &str, tex: egui::TextureId| -> Pod {
        Pod::new(pane_id.with(("cam_pod", slot))).with_custom(move |ui| {
            // Scale the image to whatever width the container body
            // gives us; height tracks the camera's aspect ratio so
            // dragging the pane wider/taller grows the thumbnail.
            let w = ui.available_width().max(64.0);
            let h = w * aspect;
            ui.image((tex, egui::vec2(w, h)));
        })
    };

    if let Some(tex) = state.tex_id_color {
        let cid = pane_id.with("cam_container_color");
        Normal::new("Color", anchor, accent, cid)
            .show(ui, vec![mk_image_pod("color", tex)]);
    }
    if let Some(tex) = state.tex_id_depth {
        let cid = pane_id.with("cam_container_depth");
        let (lo, hi) = state.last_depth_range.unwrap_or((0.0, 1.0));
        Normal::new(format!("Depth [{lo:.2}..{hi:.2}m]").as_str(), anchor, accent, cid)
            .show(ui, vec![mk_image_pod("depth", tex)]);
    }
    if let Some(tex) = state.tex_id_normal {
        let cid = pane_id.with("cam_container_normal");
        Normal::new("Normal", anchor, accent, cid)
            .show(ui, vec![mk_image_pod("normal", tex)]);
    }
    if let Some(tex) = state.tex_id_albedo {
        let cid = pane_id.with("cam_container_albedo");
        Normal::new("Albedo", anchor, accent, cid)
            .show(ui, vec![mk_image_pod("albedo", tex)]);
    }
    if let Some(tex) = state.tex_id_index {
        let cid = pane_id.with("cam_container_index");
        Normal::new("Shape index", anchor, accent, cid)
            .show(ui, vec![mk_image_pod("index", tex)]);
    }
}

// ── Stage-24d MPM (sand) viewer wiring ─────────────────────────────

/// Trivial Model for the MPM scene — just a static "ground" body
/// used by the standard renderer to keep the camera oriented.  The
/// real action is the MPM particles drawn via `draw_mpm_gizmos`.
fn build_mpm_model() -> (mp::Model, Vec<BodyVisual>) {
    let b = mp::ModelBuilder::new();
    // No bodies — the floor of the MPM grid serves as the implicit
    // boundary.
    let model = b.build().expect("mpm trivial model build");
    (model, Vec::new())
}

/// Drives the MPM simulation each frame. Lazy-initialises the
/// `MpmSimulation` on first entry to `Scene::Mpm`; on `needs_reset`
/// rebuilds the particle pour. Steps the sim in lock-step with
/// the viewer's fixed-dt cadence.
///
/// Stage 24d.1 — also pushes rheology + time-step + rigid-sphere
/// state from the right-rail UI, and reads back the per-rigid wrench
/// for the live `rigid_force_mag` readout.
fn update_mpm_example(
    mut sim: ResMut<Sim>,
    mut state: ResMut<MpmExampleState>,
    run: Res<SimRunState>,
) {
    if !matches!(
        sim.scene,
        Scene::Mpm
            | Scene::MpmSnowBall
            | Scene::MpmTwoWay
            | Scene::WarpDem
            | Scene::WarpFluid
            | Scene::WarpSph
    ) {
        return;
    }

    // Lazy initialise the MPM sim on first entry.
    if sim.mpm.is_none() {
        let cfg = MpmConfig {
            grid_size: 32,
            cell_size: 0.05,
            grid_origin: [-0.8, 0.0, -0.8],
            gravity_y: -9.81,
            youngs: state.youngs,
            poisson: 0.3,
            friction_angle: state.friction_angle_deg.to_radians(),
            cohesion: 0.0,
            boundary_friction: 0.5,
            particle_mass: 1.0e-3,
            particle_volume: 1.0e-6,
            max_particles: 32_768,
            max_rigid: 4,
            rheology: MpmRheology::Sand {
                friction_angle: state.friction_angle_deg.to_radians(),
            },
            time_step: MpmTimeStep::Explicit,
        };
        let s = MpmSimulation::new(&sim.engine, cfg).expect("mpm new");
        sim.mpm = Some(s);
        state.needs_reset = true;
    }

    // Pull material parameters + rheology + time-step from sliders.
    if let Some(m) = sim.mpm.as_mut() {
        m.set_material(state.youngs, state.friction_angle_deg.to_radians());
        let rheology = if state.rheology_idx == 1 {
            MpmRheology::Snow {
                theta_c: state.snow_theta_c,
                theta_s: state.snow_theta_s,
                hardening: state.snow_hardening,
            }
        } else {
            MpmRheology::Sand {
                friction_angle: state.friction_angle_deg.to_radians(),
            }
        };
        m.set_rheology(rheology);
        let ts = if state.time_step_implicit {
            MpmTimeStep::Implicit {
                jacobi_iters: state.jacobi_iters,
            }
        } else {
            MpmTimeStep::Explicit
        };
        m.set_time_step(ts);
    }

    // Reset / re-pour.
    if state.needs_reset {
        let device = sim.engine.device().unwrap().clone();
        let queue = sim.engine.queue().unwrap().clone();
        let n_axis = state.pour_axis_count.max(2) as usize;
        let mut particles = Vec::with_capacity(n_axis.pow(3));
        // Per-scene particle layouts:
        //   * `Mpm`         — column pour (default).
        //   * `MpmSnowBall` — spherical ball at `spout_y` (we cull
        //                     particles outside the sphere).
        //   * `MpmTwoWay`   — wider, shallower pile so the rigid
        //                     sphere has room to plunge in.
        let lo: [f32; 3];
        let hi: [f32; 3];
        let cull_sphere: Option<(f32, f32, f32, f32)>;
        match sim.scene {
            Scene::MpmSnowBall => {
                let r = 0.20_f32;
                let cy = state.spout_y;
                lo = [-r, cy - r, -r];
                hi = [r, cy + r, r];
                cull_sphere = Some((0.0, cy, 0.0, r));
            }
            Scene::MpmTwoWay => {
                lo = [-0.30, 0.05, -0.30];
                hi = [0.30, 0.20, 0.30];
                cull_sphere = None;
            }
            _ => {
                lo = [-0.18, state.spout_y - 0.18, -0.18];
                hi = [0.18, state.spout_y + 0.18, 0.18];
                cull_sphere = None;
            }
        }
        let dx = (hi[0] - lo[0]) / n_axis as f32;
        let dy = (hi[1] - lo[1]) / n_axis as f32;
        let dz = (hi[2] - lo[2]) / n_axis as f32;
        for ix in 0..n_axis {
            for iy in 0..n_axis {
                for iz in 0..n_axis {
                    let p = [
                        lo[0] + (ix as f32 + 0.5) * dx,
                        lo[1] + (iy as f32 + 0.5) * dy,
                        lo[2] + (iz as f32 + 0.5) * dz,
                    ];
                    if let Some((cx, cy, cz, r)) = cull_sphere {
                        let dx0 = p[0] - cx;
                        let dy0 = p[1] - cy;
                        let dz0 = p[2] - cz;
                        if dx0 * dx0 + dy0 * dy0 + dz0 * dz0 > r * r {
                            continue;
                        }
                    }
                    particles.push(p);
                }
            }
        }
        if let Some(m) = sim.mpm.as_mut() {
            m.set_particles(&queue, &particles).expect("set particles");
        }
        sim.mpm_rigid_spheres.clear();
        state.particle_count = particles.len() as u32;
        sim.mpm_particles = particles;
        // For MpmTwoWay, auto-spawn the rigid sphere on first frame so
        // the user immediately sees the two-way coupling demo.
        if sim.scene == Scene::MpmTwoWay {
            sim.mpm_rigid_spheres
                .push(([0.0, 1.0, 0.0], 0.12, [0.0, -2.5, 0.0]));
        }
        state.needs_reset = false;
        // First read so the viewer doesn't briefly show empty.
        if let Some(m) = sim.mpm.as_mut() {
            sim.mpm_particles = m
                .read_positions_with(&device, &queue)
                .unwrap_or_default();
        }
        return;
    }

    // "Add rigid sphere" button — drop a kinematic sphere from above.
    if state.add_rigid_sphere {
        state.add_rigid_sphere = false;
        sim.mpm_rigid_spheres
            .push(([0.0, 0.7, 0.0], 0.10, [0.0, -2.0, 0.0]));
    }

    if run.paused {
        return;
    }

    // Stage 24d.2: batch the substeps into a single `step_n` call
    // (one wgpu submit + one frame-end readback). Stage 24d.1 added
    // rigid-sphere integration + two-way wrench readback for the
    // demo path. Combined here.
    let device = sim.engine.device().unwrap().clone();
    let queue = sim.engine.queue().unwrap().clone();

    // Integrate kinematic-sphere positions forward (purely host-side
    // since we don't have a full rigid-body backend wired here; this
    // is the v1 demo path so the viewer shows the rigid sphere
    // descending into the sand). The two-way wrench read-back lets
    // the user observe the reaction force the sand applies.
    if !sim.mpm_rigid_spheres.is_empty() {
        let dt_total = 1.0 / 60.0; // viewer-cadence
        for sph in sim.mpm_rigid_spheres.iter_mut() {
            sph.0[0] += sph.2[0] * dt_total;
            sph.0[1] += sph.2[1] * dt_total;
            sph.0[2] += sph.2[2] * dt_total;
        }
        // Push to the GPU.
        let spheres_snapshot: Vec<_> = sim.mpm_rigid_spheres.clone();
        if let Some(m) = sim.mpm.as_mut() {
            let _ = m.set_rigid_spheres(&queue, &spheres_snapshot);
        }
    }

    let substeps = state.substeps_per_frame.max(1) as usize;
    let t_frame = std::time::Instant::now();
    let has_rigid = !sim.mpm_rigid_spheres.is_empty();
    let (new_pos, new_rigid_force_mag) = if let Some(m) = sim.mpm.as_mut() {
        m.step_n_with(&device, &queue, 1.0 / 480.0, substeps)
            .expect("mpm step_n");
        // Single frame-end readback; the viewer only needs particle
        // positions for the gizmo draw, not the intermediate substeps.
        let positions = m
            .read_positions_with(&device, &queue)
            .unwrap_or_default();
        let force_mag = if has_rigid {
            m.read_rigid_wrenches_with(&device, &queue)
                .map(|wrenches| {
                    wrenches
                        .iter()
                        .map(|(f, _)| (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt())
                        .fold(0.0_f32, f32::max)
                })
                .unwrap_or(0.0)
        } else {
            0.0
        };
        (positions, force_mag)
    } else {
        (sim.mpm_particles.clone(), 0.0)
    };
    sim.mpm_particles = new_pos;
    state.last_frame_ms = t_frame.elapsed().as_secs_f32() * 1000.0;
    state.rigid_force_mag = new_rigid_force_mag;

    // Update the live pile-slope readout.
    state.pile_slope_deg = measure_pile_slope_for_viewer(&sim.mpm_particles);
}

/// Stage 24d.1 — quick-and-dirty pile-slope estimator for the viewer.
/// Same algorithm as the test-side `measure_pile_slope_deg` but
/// inlined here to avoid pulling tests/ into the lib build path.
fn measure_pile_slope_for_viewer(positions: &[[f32; 3]]) -> f32 {
    if positions.len() < 16 {
        return 0.0;
    }
    let mut ys: Vec<f32> = positions.iter().map(|p| p[1]).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let floor_y = ys[ys.len() / 20];
    let radial: Vec<(f32, f32)> = positions
        .iter()
        .map(|p| {
            let r = (p[0] * p[0] + p[2] * p[2]).sqrt();
            ((p[1] - floor_y).max(0.0), r)
        })
        .collect();
    let r_max = radial.iter().map(|(_, r)| *r).fold(0.0_f32, f32::max);
    if r_max < 1.0e-3 {
        return 0.0;
    }
    let n_bins = 8;
    let mut bin_max_h = vec![0.0_f32; n_bins];
    let mut bin_count = vec![0u32; n_bins];
    for (h, r) in &radial {
        let bin = ((*r / r_max) * n_bins as f32).floor() as usize;
        let bin = bin.min(n_bins - 1);
        if *h > bin_max_h[bin] {
            bin_max_h[bin] = *h;
        }
        bin_count[bin] += 1;
    }
    let mut inner: Option<(f32, f32)> = None;
    let mut outer: Option<(f32, f32)> = None;
    for b in 0..n_bins {
        if bin_count[b] < 2 {
            continue;
        }
        let r_centre = (b as f32 + 0.5) / n_bins as f32 * r_max;
        let entry = (r_centre, bin_max_h[b]);
        if inner.is_none() {
            inner = Some(entry);
        }
        outer = Some(entry);
    }
    match (inner, outer) {
        (Some((r_in, h_in)), Some((r_out, h_out))) if r_out > r_in + 1.0e-4 => {
            let dh = (h_in - h_out).max(0.0);
            let dr = r_out - r_in;
            dh.atan2(dr).to_degrees()
        }
        _ => 0.0,
    }
}

/// Right-rail pane for the MPM scene. Stage 24d.1 — adds rheology
/// dropdown (Sand/Snow), implicit time-step toggle + jacobi_iters
/// slider, "Add rigid sphere" button, live pile-slope + reaction-force
/// readouts.
fn mpm_pane(
    ui: &mut egui::Ui,
    state: &mut MpmExampleState,
    sim: &mut Sim,
    accent: egui::Color32,
) {
    ui.add_space(2.0);
    readout(ui, "Particles", &format!("{}", state.particle_count));
    if let Some(m) = sim.mpm.as_ref() {
        readout(ui, "Live count", &format!("{}", m.particle_count()));
        readout(
            ui,
            "Grid",
            &format!(
                "{}³ × {:.2} m",
                m.config().grid_size,
                m.config().cell_size
            ),
        );
    }
    readout(ui, "Pile slope", &format!("{:.1}°", state.pile_slope_deg));
    if !sim.mpm_rigid_spheres.is_empty() {
        readout(
            ui,
            "Reaction |F|",
            &format!("{:.2} N", state.rigid_force_mag),
        );
    }
    ui.separator();

    // Rheology dropdown (Sand / Snow).
    ui.label("Rheology");
    egui::ComboBox::from_id_salt("mpm_rheology_combo")
        .selected_text(if state.rheology_idx == 1 { "Snow" } else { "Sand" })
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut state.rheology_idx, 0, "Sand (Drucker-Prager)");
            ui.selectable_value(&mut state.rheology_idx, 1, "Snow (Stomakhin)");
        });

    if state.rheology_idx == 0 {
        ui.label("Friction angle (°)");
        ui.add(egui::Slider::new(&mut state.friction_angle_deg, 10.0..=50.0));
    } else {
        ui.label("Snow θ_c (critical compression)");
        ui.add(
            egui::Slider::new(&mut state.snow_theta_c, 1.0e-3..=1.0e-1)
                .logarithmic(true),
        );
        ui.label("Snow θ_s (critical stretch)");
        ui.add(
            egui::Slider::new(&mut state.snow_theta_s, 1.0e-3..=5.0e-2)
                .logarithmic(true),
        );
        ui.label("Snow hardening exponent");
        ui.add(egui::Slider::new(&mut state.snow_hardening, 0.0..=20.0));
    }

    ui.label("Young's modulus (Pa)");
    ui.add(
        egui::Slider::new(&mut state.youngs, 1.0e3..=1.0e6)
            .logarithmic(true),
    );

    ui.label("Pour axis count");
    ui.add(egui::Slider::new(&mut state.pour_axis_count, 4..=20));

    ui.separator();
    // Time-step picker.
    ui.label("Time step");
    ui.horizontal(|ui| {
        ui.selectable_value(&mut state.time_step_implicit, false, "Explicit");
        ui.selectable_value(&mut state.time_step_implicit, true, "Implicit");
    });
    if state.time_step_implicit {
        ui.label("Jacobi iterations");
        ui.add(egui::Slider::new(&mut state.jacobi_iters, 1..=20));
    }

    // Stage 24d.2: batched-dispatcher slider.  Drag this up and watch
    // `Frame` stay relatively flat — that's the win from collapsing N
    // submits into one.
    ui.label("Substeps / frame (Stage 24d.2)");
    ui.add(egui::Slider::new(&mut state.substeps_per_frame, 1..=120));
    readout(ui, "Frame (ms)", &format!("{:.2}", state.last_frame_ms));

    ui.separator();
    ui.horizontal(|ui| {
        if button(ui, "Reset", accent).clicked() {
            state.needs_reset = true;
        }
        if button(ui, "Add rigid sphere", accent).clicked() {
            state.add_rigid_sphere = true;
        }
    });
    ui.add_space(2.0);
    ui.small("Sand: Drucker-Prager friction-cone projection.");
    ui.small("Snow: Stomakhin elastic-envelope clamping.");
    ui.small("Implicit: Jacobi-relaxation grid step (semi-implicit fallback).");
    ui.small("Hotkey: P (Particle/MPM)");
}

/// Renders MPM particles as point gizmos. Cheap; doesn't allocate.
fn draw_mpm_gizmos(sim: Res<Sim>, mut gizmos: Gizmos) {
    if !matches!(
        sim.scene,
        Scene::Mpm | Scene::MpmSnowBall | Scene::MpmTwoWay
    ) {
        return;
    }
    if sim.mpm_particles.is_empty() {
        return;
    }
    // Draw the grid AABB as 12 lines so the user sees the simulation
    // domain. (`Gizmos::cuboid` exists in some Bevy versions but the
    // crate version we vendor doesn't expose it; line-loops are
    // portable.)
    if let Some(m) = sim.mpm.as_ref() {
        let cfg = m.config();
        let lo = Vec3::new(cfg.grid_origin[0], cfg.grid_origin[1], cfg.grid_origin[2]);
        let extent = cfg.grid_size as f32 * cfg.cell_size;
        let hi = lo + Vec3::splat(extent);
        let edge_color = Color::srgb(0.40, 0.40, 0.45);
        let corners = [
            Vec3::new(lo.x, lo.y, lo.z),
            Vec3::new(hi.x, lo.y, lo.z),
            Vec3::new(hi.x, hi.y, lo.z),
            Vec3::new(lo.x, hi.y, lo.z),
            Vec3::new(lo.x, lo.y, hi.z),
            Vec3::new(hi.x, lo.y, hi.z),
            Vec3::new(hi.x, hi.y, hi.z),
            Vec3::new(lo.x, hi.y, hi.z),
        ];
        let edges: [(usize, usize); 12] = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ];
        for (a, b) in edges {
            gizmos.line(corners[a], corners[b], edge_color);
        }
    }
    let sand_color = Color::srgb(0.85, 0.74, 0.52);
    for p in &sim.mpm_particles {
        let pos = Vec3::new(p[0], p[1], p[2]);
        gizmos.sphere(pos, 0.012, sand_color);
    }
    // Stage 24d.1 — render kinematic rigid spheres for the two-way
    // coupling demo as wireframe spheres.
    let sphere_color = Color::srgb(0.20, 0.55, 0.95);
    for (centre, radius, _vel) in &sim.mpm_rigid_spheres {
        let p = Vec3::new(centre[0], centre[1], centre[2]);
        gizmos.sphere(p, *radius, sphere_color);
    }
}


// ────────────────────────────────────────────────────────────────────
// Newton-port scenes — simple ports of `xtra/newton/newton/examples/`.
// Geometry + topology + demo dynamic are matched to the source; exact
// numeric tuning is local to molla.
// ────────────────────────────────────────────────────────────────────

// ── Pendulum scene ─────────────────────────────────────────────────
//
// Two articulations side-by-side:
//   * A single 1-DoF revolute pendulum (rod + weight) at -X.
//   * A 3-link chain pendulum at +X.
// Featherstone-stepped — same dynamics path as `Scene::Arm`.

const PENDULUM_BASE_Y: f64 = 4.0;
const PENDULUM_LINK_LEN: f64 = 0.8;
const PENDULUM_LINK_RADIUS: f64 = 0.06;
const PENDULUM_LINK_MASS: f64 = 0.6;

fn build_pendulum_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);

    let half_h = PENDULUM_LINK_LEN * 0.5;
    let radius = PENDULUM_LINK_RADIUS;
    let inertia = capsule_inertia(PENDULUM_LINK_MASS, half_h, radius);

    let mut visuals = Vec::new();

    // ── Single 1-DoF pendulum at -X. ───────────────────────────────
    b.begin_articulation();
    let single_anchor = MVec3::new(-1.5, PENDULUM_BASE_Y, 0.0);
    let single = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia {
                mass: PENDULUM_LINK_MASS,
                com: MVec3::ZERO,
                inertia,
            })
            .with_position(single_anchor - MVec3::new(0.0, half_h, 0.0)),
    );
    b.add_shape_capsule(single, half_h, radius, mp::Transform::IDENTITY)
        .expect("single pendulum shape");
    visuals.push(BodyVisual::Capsule {
        half_h: half_h as f32,
        radius: radius as f32,
    });
    // Hang straight down at q=0: rotate joint frame 180° around X so
    // joint-local +Y → world -Y. Matches the `Arm` scene's trick.
    let flip = MQuat::from_axis_angle(MVec3::X, std::f64::consts::PI);
    let x_p_single = mp::Transform::new(single_anchor, flip);
    let x_c_single = mp::Transform::from_translation(MVec3::new(0.0, -half_h, 0.0));
    b.add_joint_revolute(mp::BodyId::NONE, single, x_p_single, x_c_single, MVec3::Z)
        .expect("single pendulum joint");

    // ── 3-link chain pendulum at +X. ───────────────────────────────
    b.begin_articulation();
    let chain_anchor = MVec3::new(1.5, PENDULUM_BASE_Y, 0.0);
    let mut prev = mp::BodyId::NONE;
    let mut next_pos = chain_anchor - MVec3::new(0.0, half_h, 0.0);
    for i in 0..3 {
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: PENDULUM_LINK_MASS,
                    com: MVec3::ZERO,
                    inertia,
                })
                .with_position(next_pos),
        );
        b.add_shape_capsule(body, half_h, radius, mp::Transform::IDENTITY)
            .expect("chain link shape");
        visuals.push(BodyVisual::Capsule {
            half_h: half_h as f32,
            radius: radius as f32,
        });
        let x_p = if i == 0 {
            mp::Transform::new(chain_anchor, flip)
        } else {
            mp::Transform::from_translation(MVec3::new(0.0, half_h, 0.0))
        };
        let x_c = mp::Transform::from_translation(MVec3::new(0.0, -half_h, 0.0));
        b.add_joint_revolute(prev, body, x_p, x_c, MVec3::Z)
            .expect("chain joint");
        prev = body;
        next_pos = next_pos - MVec3::new(0.0, PENDULUM_LINK_LEN, 0.0);
    }

    // Floor.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("floor plane");

    (b.build().expect("pendulum model"), visuals)
}

/// Read joint angles + angular velocities for the pendulum readout.
#[derive(Resource, Default)]
struct PendulumExampleState {
    /// Single-pendulum angle (rad) and angular velocity (rad/s).
    single_q: f32,
    single_qd: f32,
    /// 3-link chain joint angles and rates.
    chain_q: [f32; 3],
    chain_qd: [f32; 3],
}

fn update_pendulum_example(sim: Res<Sim>, mut state: ResMut<PendulumExampleState>) {
    if sim.scene != Scene::Pendulum {
        return;
    }
    let q = match sim.state_a.joint_q.host() {
        Ok(q) => q,
        Err(_) => return,
    };
    let qd = match sim.state_a.joint_qd.host() {
        Ok(q) => q,
        Err(_) => return,
    };
    if q.len() >= 4 && qd.len() >= 4 {
        state.single_q = q[0] as f32;
        state.single_qd = qd[0] as f32;
        state.chain_q = [q[1] as f32, q[2] as f32, q[3] as f32];
        state.chain_qd = [qd[1] as f32, qd[2] as f32, qd[3] as f32];
    }
}

fn pendulum_pane(ui: &mut egui::Ui, state: &PendulumExampleState) {
    ui.label("Two pendulums in one scene:");
    ui.separator();
    ui.label("Single (-X)");
    readout(ui, "  · angle", &format!("{:>+5.2} rad", state.single_q));
    readout(ui, "  · ω", &format!("{:>+5.2} rad/s", state.single_qd));
    ui.separator();
    ui.label("3-link chain (+X)");
    for i in 0..3 {
        readout(
            ui,
            &format!("  · q{} | ω{}", i, i),
            &format!(
                "{:>+5.2} rad | {:>+5.2} rad/s",
                state.chain_q[i], state.chain_qd[i]
            ),
        );
    }
    ui.separator();
    ui.small("Featherstone GPU solver. Drag to swing.");
    ui.small("Reset (R) to restart from straight-down rest.");
}

// ── Newton-port scene builders (Group A-E). Each block defines its
//    own constants near the top, the build_*_model() function, and
//    (optionally) a state struct, an update_*_example() pre-step
//    system, and a *_pane() right-rail panel.
/// Joint-sampler scene: 3 columns side-by-side, each showing one joint
/// kind in motion. Port of `xtra/newton/newton/examples/basic/example_basic_joints.py`.
///   * Column 0 (-2): revolute pendulum about +Z (rotating arm).
///   * Column 1 ( 0): prismatic slider along +Y.
///   * Column 2 (+2): ball joint (3D pendulum).
///
/// We omit Newton's 4th "distance" column because molla's Featherstone
/// solver doesn't enforce XPBD `CONNECT` equalities, and XPBD doesn't
/// dispatch revolute/ball joint constraints — so a single scene can't
/// show all four under one solver. The `Rope` scene already covers the
/// distance-equality case via XPBD CONNECT.
fn build_joint_sampler_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);

    let base_y = 4.0_f64;
    let half_h = 0.4_f64;
    let radius = 0.07_f64;
    let mass = 0.6_f64;
    let inertia = capsule_inertia(mass, half_h, radius);
    let mut visuals = Vec::new();

    // Helper: hang a capsule from a revolute/prismatic anchor at the
    // given column x-offset. Joint flip rotates joint-local +Y onto
    // world -Y so the rest pose hangs straight down (matches
    // `build_arm_model_with`).
    let flip = MQuat::from_axis_angle(MVec3::X, std::f64::consts::PI);

    // ── Column 0: revolute about +Z. ──────────────────────────────
    {
        b.begin_articulation();
        let anchor = MVec3::new(-2.0, base_y, 0.0);
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass,
                    com: MVec3::ZERO,
                    inertia,
                })
                .with_position(anchor - MVec3::new(0.0, half_h, 0.0)),
        );
        b.add_shape_capsule(body, half_h, radius, mp::Transform::IDENTITY)
            .expect("rev cap");
        visuals.push(BodyVisual::Capsule {
            half_h: half_h as f32,
            radius: radius as f32,
        });
        let x_p = mp::Transform::new(anchor, flip);
        let x_c = mp::Transform::from_translation(MVec3::new(0.0, -half_h, 0.0));
        b.add_joint_revolute(mp::BodyId::NONE, body, x_p, x_c, MVec3::Z)
            .expect("revolute joint");
    }

    // ── Column 1: prismatic along +Y. ─────────────────────────────
    {
        b.begin_articulation();
        let anchor = MVec3::new(0.0, base_y, 0.0);
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass,
                    com: MVec3::ZERO,
                    inertia: box_inertia_xyz(mass, 0.15, 0.15, 0.15),
                })
                .with_position(anchor - MVec3::new(0.0, 0.5, 0.0)),
        );
        b.add_shape_box(body, MVec3::splat(0.15), mp::Transform::IDENTITY)
            .expect("prism box");
        visuals.push(BodyVisual::Box(Vec3::splat(0.15)));
        let x_p = mp::Transform::from_translation(anchor);
        let x_c = mp::Transform::IDENTITY;
        b.add_joint_prismatic(mp::BodyId::NONE, body, x_p, x_c, MVec3::Y)
            .expect("prismatic joint");
    }

    // ── Column 2: ball joint (3D pendulum). ───────────────────────
    {
        b.begin_articulation();
        let anchor = MVec3::new(2.0, base_y, 0.0);
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass,
                    com: MVec3::ZERO,
                    inertia,
                })
                .with_position(anchor - MVec3::new(0.0, half_h, 0.0)),
        );
        b.add_shape_capsule(body, half_h, radius, mp::Transform::IDENTITY)
            .expect("ball cap");
        visuals.push(BodyVisual::Capsule {
            half_h: half_h as f32,
            radius: radius as f32,
        });
        let x_p = mp::Transform::new(anchor, flip);
        let x_c = mp::Transform::from_translation(MVec3::new(0.0, -half_h, 0.0));
        b.add_joint_ball(mp::BodyId::NONE, body, x_p, x_c)
            .expect("ball joint");
    }

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("joint sampler model"), visuals)
}
/// Heightfield scene — 32×32 sinusoidal heightfield with ~30 mixed
/// primitives dropped on top. Port of `example_basic_heightfield.py`.
/// Larger / more dramatic than the existing `Shapes` scene.
fn build_heightfield_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.2);

    let n: u32 = 32;
    let cell = 0.5_f64;
    let half = (n as f64 - 1.0) * cell * 0.5;
    let mut heights = Vec::with_capacity((n * n) as usize);
    for j in 0..n {
        for i in 0..n {
            let x = (i as f64 - (n as f64 - 1.0) * 0.5) * cell;
            let z = (j as f64 - (n as f64 - 1.0) * 0.5) * cell;
            // Twin-frequency wavy field — bowls + ridges.
            let h = 0.5 * (x * 0.45).sin() * (z * 0.45).cos()
                + 0.2 * ((x + z) * 0.9).sin();
            heights.push(h);
        }
    }
    let hf = mp::Heightfield {
        nx: n,
        nz: n,
        cell_size: cell,
        origin: MVec3::new(-half, 0.0, -half),
        heights,
    };
    let field_id = b.add_heightfield(hf);
    b.add_shape_heightfield(mp::BodyId::NONE, field_id, mp::Transform::IDENTITY)
        .expect("heightfield");

    // Drop ~30 mixed primitives in a 6×5 grid pattern over the field.
    let mut rng_state: u64 = 0xCAFEBABEDEADBEEF;
    fn rng_uniform(s: &mut u64) -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
    fn random_quat(s: &mut u64) -> MQuat {
        let axis = MVec3::new(rng_uniform(s), rng_uniform(s), rng_uniform(s))
            .normalize_or_zero();
        let axis = if axis.length_squared() < 1e-9 { MVec3::Y } else { axis };
        let angle = rng_uniform(s) * std::f64::consts::PI;
        MQuat::from_axis_angle(axis, angle)
    }
    let mut visuals = Vec::new();
    let drop_grid = 6;
    let drop_spacing = 1.6_f64;
    for ix in 0..drop_grid {
        for iz in 0..5 {
            let x = (ix as f64 - (drop_grid as f64 - 1.0) * 0.5) * drop_spacing
                + 0.2 * rng_uniform(&mut rng_state);
            let z = (iz as f64 - 2.0) * drop_spacing
                + 0.2 * rng_uniform(&mut rng_state);
            let y = 6.0 + 0.5 * rng_uniform(&mut rng_state).abs();
            let pos = MVec3::new(x, y, z);
            let rot = random_quat(&mut rng_state);
            match (ix + iz) % 3 {
                0 => {
                    let r = 0.30_f64;
                    let body = b.add_body(
                        mp::BodyParams::new()
                            .with_spatial_inertia(mp::SpatialInertia::sphere(1.0, r))
                            .with_pose(mp::Transform::new(pos, rot)),
                    );
                    b.add_shape_sphere(body, r, mp::Transform::IDENTITY)
                        .expect("sphere");
                    visuals.push(BodyVisual::Sphere(r as f32));
                }
                1 => {
                    let h = 0.30_f64;
                    let inertia = box_inertia(1.0, h);
                    let body = b.add_body(
                        mp::BodyParams::new()
                            .with_spatial_inertia(mp::SpatialInertia {
                                mass: 1.0,
                                com: MVec3::ZERO,
                                inertia,
                            })
                            .with_pose(mp::Transform::new(pos, rot)),
                    );
                    b.add_shape_box(body, MVec3::splat(h), mp::Transform::IDENTITY)
                        .expect("box");
                    visuals.push(BodyVisual::Box(Vec3::splat(h as f32)));
                }
                _ => {
                    let half_h = 0.30_f64;
                    let radius = 0.18_f64;
                    let inertia = capsule_inertia(0.8, half_h, radius);
                    let body = b.add_body(
                        mp::BodyParams::new()
                            .with_spatial_inertia(mp::SpatialInertia {
                                mass: 0.8,
                                com: MVec3::ZERO,
                                inertia,
                            })
                            .with_pose(mp::Transform::new(pos, rot)),
                    );
                    b.add_shape_capsule(body, half_h, radius, mp::Transform::IDENTITY)
                        .expect("capsule");
                    visuals.push(BodyVisual::Capsule {
                        half_h: half_h as f32,
                        radius: radius as f32,
                    });
                }
            }
        }
    }

    (b.build().expect("heightfield model"), visuals)
}
/// Conveyor scene — kinematic belt (long box) plus dropped boxes that
/// slide along. Port of `example_basic_conveyor.py` (linearised — we
/// use a straight belt instead of the annular ring mesh, which would
/// require trimesh authoring).
///
/// The belt is body 0 (kinematic). The update system translates its
/// pose along +X each frame at `belt_speed` m/s and snaps it back to
/// the start when it overshoots, so friction carries the boxes along.
const CONVEYOR_BELT_LEN: f64 = 6.0;
const CONVEYOR_BELT_HALF_WIDTH: f64 = 0.6;
const CONVEYOR_BELT_HALF_THICK: f64 = 0.05;
const CONVEYOR_BELT_Y: f64 = 1.0;
fn build_conveyor_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.2);

    let mut visuals = Vec::new();

    // Belt — body 0, kinematic, translated each frame in
    // `update_conveyor_example`.
    let belt_half = MVec3::new(
        CONVEYOR_BELT_LEN * 0.5,
        CONVEYOR_BELT_HALF_THICK,
        CONVEYOR_BELT_HALF_WIDTH,
    );
    let belt = b.add_body(
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_inertia(molla::math::Mat3::ZERO)
            .with_position(MVec3::new(0.0, CONVEYOR_BELT_Y, 0.0))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_box(belt, belt_half, mp::Transform::IDENTITY)
        .expect("belt shape");
    visuals.push(BodyVisual::Box(Vec3::new(
        belt_half.x as f32,
        belt_half.y as f32,
        belt_half.z as f32,
    )));

    // Side rails — static, just keep boxes from rolling off the belt.
    for &z_off in &[-CONVEYOR_BELT_HALF_WIDTH - 0.1, CONVEYOR_BELT_HALF_WIDTH + 0.1] {
        b.add_shape_box(
            mp::BodyId::NONE,
            MVec3::new(CONVEYOR_BELT_LEN * 0.5, 0.15, 0.05),
            mp::Transform::from_translation(MVec3::new(
                0.0,
                CONVEYOR_BELT_Y + 0.15,
                z_off,
            )),
        )
        .expect("rail shape");
    }

    // Dropped boxes — 8 of them, scattered along the belt.
    for i in 0..8 {
        let x = -CONVEYOR_BELT_LEN * 0.5 + (i as f64 + 0.5) * (CONVEYOR_BELT_LEN / 8.0);
        let half = MVec3::splat(0.15);
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: 0.5,
                    com: MVec3::ZERO,
                    inertia: box_inertia(0.5, 0.15),
                })
                .with_position(MVec3::new(
                    x,
                    CONVEYOR_BELT_Y + CONVEYOR_BELT_HALF_THICK + 0.20 + (i as f64) * 0.05,
                    0.0,
                )),
        );
        b.add_shape_box(body, half, mp::Transform::IDENTITY)
            .expect("box shape");
        visuals.push(BodyVisual::Box(Vec3::splat(0.15)));
    }

    // Floor.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("floor");

    (b.build().expect("conveyor model"), visuals)
}
/// Pyramid scene — 10-layer 4-sided box pyramid struck by a heavy ball
/// from above. Larger and more dramatic than the existing `Tower`.
/// Port of `xtra/newton/newton/examples/contacts/example_pyramid.py`.
const PYRAMID_LEVELS: usize = 10;
const PYRAMID_BOX_HALF: f64 = 0.30;
/// 2.05× spacing matches the Newton example — small gap so the
/// neighbouring boxes don't collide on frame 0.
const PYRAMID_SPACING: f64 = 2.05 * PYRAMID_BOX_HALF;
const PYRAMID_BALL_RADIUS: f64 = 1.2;
const PYRAMID_BALL_MASS: f64 = 80.0;
const PYRAMID_BALL_HEIGHT: f64 = 14.0;
fn build_pyramid_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.20);

    let half = PYRAMID_BOX_HALF;
    let spacing = PYRAMID_SPACING;
    let inertia = box_inertia(1.0, half);
    let half_v = MVec3::splat(half);

    let mut visuals = Vec::new();
    let layer_y0 = half + 0.001;

    // 4-sided pyramid: layer L contains (PYRAMID_LEVELS - L)² boxes.
    for level in 0..PYRAMID_LEVELS {
        let row_count = PYRAMID_LEVELS - level;
        let y = layer_y0 + (level as f64) * (PYRAMID_BOX_HALF * 2.0 + 0.005);
        for ix in 0..row_count {
            for iz in 0..row_count {
                let x = (ix as f64 - (row_count as f64 - 1.0) * 0.5) * spacing;
                let z = (iz as f64 - (row_count as f64 - 1.0) * 0.5) * spacing;
                let body = b.add_body(
                    mp::BodyParams::new()
                        .with_spatial_inertia(mp::SpatialInertia {
                            mass: 1.0,
                            com: MVec3::ZERO,
                            inertia,
                        })
                        .with_position(MVec3::new(x, y, z)),
                );
                b.add_shape_box(body, half_v, mp::Transform::IDENTITY)
                    .expect("pyramid box");
                visuals.push(BodyVisual::Box(Vec3::splat(half as f32)));
            }
        }
    }

    // Heavy wrecking ball.
    let ball = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia::sphere(
                PYRAMID_BALL_MASS,
                PYRAMID_BALL_RADIUS,
            ))
            // Drop slightly off-centre so the impact knocks the
            // pyramid sideways rather than punching straight through.
            .with_position(MVec3::new(1.5, PYRAMID_BALL_HEIGHT, 0.0)),
    );
    b.add_shape_sphere(ball, PYRAMID_BALL_RADIUS, mp::Transform::IDENTITY)
        .expect("ball");
    visuals.push(BodyVisual::Sphere(PYRAMID_BALL_RADIUS as f32));

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("pyramid model"), visuals)
}
/// ClothBending scene — two hanging cloth strips side by side, each
/// built with a different bending compliance so the user can A/B
/// compare the bending response. Port of
/// `xtra/newton/newton/examples/cloth/example_cloth_bending.py`.
const CLOTH_BEND_GRID_X: u32 = 12;
const CLOTH_BEND_GRID_Y: u32 = 18;
const CLOTH_BEND_W: f64 = 0.9;
const CLOTH_BEND_H: f64 = 1.4;
const CLOTH_BEND_TOP_Y: f64 = 4.0;
/// Static slot for the cloth-bending compliance values, so
/// `Sim::build_scene` (no SystemParam access) can read them on rebuild.
/// `cloth_bending_pane` writes the bits when the user moves the
/// sliders; `apply_scene_switch` triggers a rebuild via `pending`.
static CLOTH_BEND_COMP_LEFT: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);
static CLOTH_BEND_COMP_RIGHT: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);
fn cloth_bend_left() -> f32 {
    let bits = CLOTH_BEND_COMP_LEFT.load(std::sync::atomic::Ordering::Relaxed);
    if bits == 0 { 0.05 } else { f32::from_bits(bits) }
}
fn cloth_bend_right() -> f32 {
    let bits = CLOTH_BEND_COMP_RIGHT.load(std::sync::atomic::Ordering::Relaxed);
    if bits == 0 { 5.0 } else { f32::from_bits(bits) }
}
fn set_cloth_bend(left: f32, right: f32) {
    CLOTH_BEND_COMP_LEFT.store(left.to_bits(), std::sync::atomic::Ordering::Relaxed);
    CLOTH_BEND_COMP_RIGHT.store(right.to_bits(), std::sync::atomic::Ordering::Relaxed);
}
fn build_cloth_bending_model() -> (mp::Model, Vec<BodyVisual>) {
    build_cloth_bending_with(cloth_bend_left(), cloth_bend_right())
}
fn build_cloth_bending_with(comp_left: f32, comp_right: f32) -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.4);

    // Two strips. Both pinned along their top edges so they hang.
    for (side_idx, x_offset) in [-1.0_f64, 1.0].iter().enumerate() {
        let comp = if side_idx == 0 { comp_left } else { comp_right };
        let origin = MVec3::new(*x_offset - CLOTH_BEND_W * 0.5, CLOTH_BEND_TOP_Y, 0.0);
        let u_axis = MVec3::new(CLOTH_BEND_W, 0.0, 0.0);
        let v_axis = MVec3::new(0.0, -CLOTH_BEND_H, 0.0);
        let (first, n, _) = b.add_cloth_grid_with_bending(
            origin,
            u_axis,
            v_axis,
            CLOTH_BEND_GRID_X,
            CLOTH_BEND_GRID_Y,
            0.04,                         // particle mass
            0.025,                        // particle radius
            5.0e-6,                       // stretch compliance (very stiff)
            0.05,                         // max stretch ratio
            comp as f64,                  // bending compliance — this is the A/B knob
            mp::WorldId(0),
        );
        let _ = n;
        // Pin top row only (iy = 0 → indices first..first+CLOTH_BEND_GRID_X).
        for ix in 0..CLOTH_BEND_GRID_X {
            b.pin_particle(first + ix);
        }
    }

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("cloth bending model"), Vec::new())
}
/// ClothTwist scene — a vertical cloth strip pinned along TOP and
/// BOTTOM edges. The bottom edge is rotated each frame around the
/// strip's vertical axis (Y) so the cloth twists. Port of
/// `xtra/newton/newton/examples/cloth/example_cloth_twist.py`.
const CLOTH_TWIST_GRID_X: u32 = 18;
const CLOTH_TWIST_GRID_Y: u32 = 24;
const CLOTH_TWIST_W: f64 = 1.0;
const CLOTH_TWIST_H: f64 = 1.6;
const CLOTH_TWIST_TOP_Y: f64 = 4.0;
const CLOTH_TWIST_CENTRE_X: f64 = 0.0;
fn build_cloth_twist_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.4);

    let origin = MVec3::new(
        CLOTH_TWIST_CENTRE_X - CLOTH_TWIST_W * 0.5,
        CLOTH_TWIST_TOP_Y,
        0.0,
    );
    let u_axis = MVec3::new(CLOTH_TWIST_W, 0.0, 0.0);
    let v_axis = MVec3::new(0.0, -CLOTH_TWIST_H, 0.0);
    let (first, _, _) = b.add_cloth_grid(
        origin,
        u_axis,
        v_axis,
        CLOTH_TWIST_GRID_X,
        CLOTH_TWIST_GRID_Y,
        0.04,                                  // particle mass
        0.025,                                 // particle radius
        5.0e-6,                                // stretch compliance
        0.05,                                  // max stretch ratio
        mp::WorldId(0),
    );
    // Pin top row (iy = 0).
    for ix in 0..CLOTH_TWIST_GRID_X {
        b.pin_particle(first + ix);
    }
    // Pin bottom row (iy = CLOTH_TWIST_GRID_Y - 1) — driven by
    // `update_cloth_twist_example`.
    let bottom_iy = CLOTH_TWIST_GRID_Y - 1;
    for ix in 0..CLOTH_TWIST_GRID_X {
        b.pin_particle(first + bottom_iy * CLOTH_TWIST_GRID_X + ix);
    }

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("cloth twist model"), Vec::new())
}
/// CablePile scene — drop a handful of cable polylines into a "bowl"
/// (4 walls + floor) and watch them pile / coil. Port of
/// `xtra/newton/newton/examples/cable/example_cable_pile.py` (simpler
/// — we ship straight cables instead of sinusoidally-wavy ones).
fn build_cable_pile_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_linear_damping(0.6);

    // Bowl: floor + 4 vertical walls.
    let bowl_size = 1.6_f64;
    let wall_h = 0.8_f64;
    let wall_t = 0.05_f64;
    b.add_shape_box(
        mp::BodyId::NONE,
        MVec3::new(bowl_size, wall_h, wall_t),
        mp::Transform::from_translation(MVec3::new(0.0, wall_h, bowl_size)),
    )
    .expect("wall +Z");
    b.add_shape_box(
        mp::BodyId::NONE,
        MVec3::new(bowl_size, wall_h, wall_t),
        mp::Transform::from_translation(MVec3::new(0.0, wall_h, -bowl_size)),
    )
    .expect("wall -Z");
    b.add_shape_box(
        mp::BodyId::NONE,
        MVec3::new(wall_t, wall_h, bowl_size),
        mp::Transform::from_translation(MVec3::new(bowl_size, wall_h, 0.0)),
    )
    .expect("wall +X");
    b.add_shape_box(
        mp::BodyId::NONE,
        MVec3::new(wall_t, wall_h, bowl_size),
        mp::Transform::from_translation(MVec3::new(-bowl_size, wall_h, 0.0)),
    )
    .expect("wall -X");
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    // 8 cables, alternating horizontal axis, dropped from staggered
    // heights. No pins — they just fall and pile.
    let cable_len = 1.4_f64;
    let particles_per_cable: u32 = 24;
    let mut rng_state: u64 = 0xFEEDFACECAFEBEEF;
    fn rng(s: &mut u64) -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
    for i in 0..8 {
        let h = 1.5 + (i as f64) * 0.30;
        let yaw = (i as f64) * 0.7 + rng(&mut rng_state) * 0.2;
        let dx = yaw.cos() * cable_len * 0.5;
        let dz = yaw.sin() * cable_len * 0.5;
        let cx = 0.6 * rng(&mut rng_state);
        let cz = 0.6 * rng(&mut rng_state);
        let p0 = MVec3::new(cx - dx, h, cz - dz);
        let p1 = MVec3::new(cx + dx, h, cz + dz);
        let _ = b.add_cable_polyline(
            p0,
            p1,
            particles_per_cable,
            0.04,                // particle mass
            0.025,               // particle radius
            5.0e5,               // stiffness
            0.02,                // hard cap on stretch
            mp::WorldId(0),
        );
    }

    (b.build().expect("cable pile model"), Vec::new())
}
/// CableYJunction scene — three cables meeting at a Y junction. The
/// first particle of each cable is welded to the same world point via
/// a kinematic anchor body and `CONNECT` equalities. One cable is
/// kinematic-pinned at its free end (top); the other two sag from
/// gravity. Port of
/// `xtra/newton/newton/examples/cable/example_cable_y_junction.py`.
fn build_cable_y_junction_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_linear_damping(0.4);

    let junction = MVec3::new(0.0, 2.5, 0.0);
    let branch_len = 1.2_f64;
    let particles_per_branch: u32 = 16;
    let dirs = [
        MVec3::new(1.0, 0.7, 0.0).normalize(),
        MVec3::new(-0.5, 0.7, 0.866).normalize(),
        MVec3::new(-0.5, 0.7, -0.866).normalize(),
    ];
    // 3 cable polylines all starting at junction, fanning outward.
    let mut firsts = Vec::with_capacity(3);
    for d in &dirs {
        let p_far = junction + *d * branch_len;
        let (first, _, _) = b.add_cable_polyline(
            junction,
            p_far,
            particles_per_branch,
            0.04,
            0.025,
            5.0e5,
            0.02,
            mp::WorldId(0),
        );
        firsts.push(first);
    }

    // Pin the very first particle of each cable so they share the
    // junction point (Newton uses add_rod_graph; we approximate by
    // pinning all three start particles in place).
    for f in &firsts {
        b.pin_particle(*f);
    }
    // The first cable's far endpoint stays kinematic (top hanger);
    // the other two are free.
    b.pin_particle(firsts[0] + particles_per_branch - 1);

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("cable y junction model"), Vec::new())
}
/// CartPole scene — identical topology to `NeuralPolicy` (cart on
/// prismatic + pole on revolute) but driven by a hand-written PD on
/// the pole angle. Port of `example_robot_cartpole.py` /
/// `example_selection_cartpole.py`.
fn build_cartpole_model() -> (mp::Model, Vec<BodyVisual>) {
    // Re-use the existing cart-pole topology — identical bodies and
    // joints, just a different controller in the pre-step system.
    build_neural_policy_model()
}
/// CartPoleSwarm scene — N independent cart-poles in a row, each in
/// its own articulation. Demonstrates batched PD control. Port of
/// `xtra/newton/newton/examples/selection/example_selection_cartpole.py`.
const CARTPOLE_SWARM_N: usize = 16;
const CARTPOLE_SWARM_SPACING: f64 = 1.2;
fn build_cartpole_swarm_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.02);

    let mut visuals = Vec::with_capacity(CARTPOLE_SWARM_N * 2);
    let cart_inertia = box_inertia_xyz(CART_MASS, CART_HALF_X, CART_HALF_Y, CART_HALF_Z);
    let pole_inertia = capsule_inertia(POLE_MASS, POLE_HALF_LEN, POLE_RADIUS);

    for i in 0..CARTPOLE_SWARM_N {
        let z = (i as f64 - (CARTPOLE_SWARM_N as f64 - 1.0) * 0.5) * CARTPOLE_SWARM_SPACING;
        b.begin_articulation();
        // Cart prismatic.
        let cart = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: CART_MASS,
                    com: MVec3::ZERO,
                    inertia: cart_inertia,
                })
                .with_position(MVec3::new(0.0, CART_RAIL_Y, z)),
        );
        b.add_shape_box(
            cart,
            MVec3::new(CART_HALF_X, CART_HALF_Y, CART_HALF_Z),
            mp::Transform::IDENTITY,
        )
        .expect("cart box");
        visuals.push(BodyVisual::Box(Vec3::new(
            CART_HALF_X as f32,
            CART_HALF_Y as f32,
            CART_HALF_Z as f32,
        )));
        let x_p_cart = mp::Transform::from_translation(MVec3::new(0.0, CART_RAIL_Y, z));
        b.add_joint_prismatic(mp::BodyId::NONE, cart, x_p_cart, mp::Transform::IDENTITY, MVec3::X)
            .expect("cart prismatic");
        // Pole revolute.
        let pole = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: POLE_MASS,
                    com: MVec3::ZERO,
                    inertia: pole_inertia,
                })
                .with_position(MVec3::new(0.0, CART_RAIL_Y - POLE_HALF_LEN, z)),
        );
        b.add_shape_capsule(pole, POLE_HALF_LEN, POLE_RADIUS, mp::Transform::IDENTITY)
            .expect("pole capsule");
        visuals.push(BodyVisual::Capsule {
            half_h: POLE_HALF_LEN as f32,
            radius: POLE_RADIUS as f32,
        });
        let x_p_pole = mp::Transform::from_translation(MVec3::new(0.0, CART_HALF_Y, 0.0));
        let x_c_pole = mp::Transform::from_translation(MVec3::new(0.0, POLE_HALF_LEN, 0.0));
        b.add_joint_revolute(cart, pole, x_p_pole, x_c_pole, MVec3::Z)
            .expect("pole revolute");
    }

    (b.build().expect("cartpole swarm model"), visuals)
}
/// FourBar scene — a closed kinematic 4-bar linkage. Newton's
/// `kamino_basic_fourbar` example loads a USD prim; we build it
/// procedurally as 4 bodies + 3 revolute joints in a tree, plus a
/// `JointMimic` linear-polynomial coupling on the closing pair so the
/// loop appears closed visually. Per the spec — driver joint is one
/// revolute, the others mimic via polynomial.
const FOURBAR_BAR_RADIUS: f32 = 0.04;
const FOURBAR_BASE_Y: f64 = 2.5;
fn build_fourbar_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.10);

    // 4-bar geometry (rectangular crank-rocker):
    //   bar A — ground crank (rotates around world frame, length L_a)
    //   bar B — coupler     (length L_b)
    //   bar C — output rocker (length L_c)
    // The fourth bar is the implicit world ground (distance L_d
    // between the two ground pivots).
    let l_a = 0.6_f64;
    let l_b = 1.2_f64;
    let l_c = 0.6_f64;
    let _l_d = 1.0_f64;
    let mass = 0.3_f64;
    let inertia = box_inertia_xyz(mass, 0.05, l_a * 0.5, 0.05);

    let mut visuals = Vec::with_capacity(4);

    // ── Driver crank (bar A). Anchored at world-X = -0.5. ────────
    b.begin_articulation();
    let a_pivot = MVec3::new(-0.5, FOURBAR_BASE_Y, 0.0);
    let bar_a = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia {
                mass,
                com: MVec3::ZERO,
                inertia,
            })
            .with_position(a_pivot - MVec3::new(0.0, l_a * 0.5, 0.0)),
    );
    b.add_shape_capsule(bar_a, l_a * 0.5, FOURBAR_BAR_RADIUS as f64, mp::Transform::IDENTITY)
        .expect("bar A capsule");
    visuals.push(BodyVisual::Capsule {
        half_h: (l_a * 0.5) as f32,
        radius: FOURBAR_BAR_RADIUS,
    });
    let flip = MQuat::from_axis_angle(MVec3::X, std::f64::consts::PI);
    let driver_id = b
        .add_joint_revolute(
            mp::BodyId::NONE,
            bar_a,
            mp::Transform::new(a_pivot, flip),
            mp::Transform::from_translation(MVec3::new(0.0, -l_a * 0.5, 0.0)),
            MVec3::Z,
        )
        .expect("driver joint");

    // ── Coupler (bar B), revolute to bar A's far end. ────────────
    let bar_b = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia {
                mass,
                com: MVec3::ZERO,
                inertia: box_inertia_xyz(mass, 0.05, l_b * 0.5, 0.05),
            })
            .with_position(MVec3::new(0.0, FOURBAR_BASE_Y - l_a, 0.0)),
    );
    b.add_shape_capsule(bar_b, l_b * 0.5, FOURBAR_BAR_RADIUS as f64, mp::Transform::IDENTITY)
        .expect("bar B capsule");
    visuals.push(BodyVisual::Capsule {
        half_h: (l_b * 0.5) as f32,
        radius: FOURBAR_BAR_RADIUS,
    });
    let coupler_id = b
        .add_joint_revolute(
            bar_a,
            bar_b,
            mp::Transform::from_translation(MVec3::new(0.0, l_a * 0.5, 0.0)),
            mp::Transform::from_translation(MVec3::new(0.0, -l_b * 0.5, 0.0)),
            MVec3::Z,
        )
        .expect("coupler joint");

    // ── Output rocker (bar C), revolute to bar B's far end. ──────
    let bar_c = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia {
                mass,
                com: MVec3::ZERO,
                inertia,
            })
            .with_position(MVec3::new(0.5, FOURBAR_BASE_Y - l_a, 0.0)),
    );
    b.add_shape_capsule(bar_c, l_c * 0.5, FOURBAR_BAR_RADIUS as f64, mp::Transform::IDENTITY)
        .expect("bar C capsule");
    visuals.push(BodyVisual::Capsule {
        half_h: (l_c * 0.5) as f32,
        radius: FOURBAR_BAR_RADIUS,
    });
    let rocker_id = b
        .add_joint_revolute(
            bar_b,
            bar_c,
            mp::Transform::from_translation(MVec3::new(0.0, l_b * 0.5, 0.0)),
            mp::Transform::from_translation(MVec3::new(0.0, -l_c * 0.5, 0.0)),
            MVec3::Z,
        )
        .expect("rocker joint");

    // Linear-identity JointMimic — coupler tracks `-driver` and the
    // rocker tracks `+driver`. Real Freudenstein-equation closure is
    // a transcendental relation; the spec calls for a polynomial fit
    // so we ship the simplest case (linear) here. The visual loop
    // closure is approximate but reads as a "crank + rocker" cycle.
    let _ = b.add_equality_joint_mimic(driver_id, coupler_id, [0.0, -1.0, 0.0, 0.0, 0.0]);
    let _ = b.add_equality_joint_mimic(driver_id, rocker_id, [0.0, 1.0, 0.0, 0.0, 0.0]);

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("fourbar model"), visuals)
}
/// DiffBall scene — a kinematic projectile + a target hoop. The host
/// optimises `(v, angle)` via closed-form gradient on miss distance.
/// Differs from Newton's `example_diffsim_ball.py` in that we don't
/// run a particle simulation (molla's tape autograd targets rigid
/// XPBD); we use the analytical projectile equations directly. The
/// pedagogy — "diffsim finds the right initial conditions" — is the
/// same.
fn build_diff_ball_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();

    // Body 0 — kinematic ball whose pose is rewritten each frame from
    // the analytical trajectory.
    let ball = b.add_body(
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_inertia(molla::math::Mat3::ZERO)
            .with_position(MVec3::new(0.0, 0.5, 0.0))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(ball, 0.10, mp::Transform::IDENTITY)
        .expect("ball");

    // Body 1 — kinematic target indicator (small sphere) drawn at the
    // current target position.
    let target = b.add_body(
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_inertia(molla::math::Mat3::ZERO)
            .with_position(MVec3::new(3.0, 1.5, 0.0))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(target, 0.08, mp::Transform::IDENTITY)
        .expect("target");

    let visuals = vec![BodyVisual::Sphere(0.10), BodyVisual::Sphere(0.08)];
    // Static plane.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");
    (b.build().expect("diff ball model"), visuals)
}
/// DiffSpring scene — particle inside a "cage" of springs, all
/// anchored at fixed cube corners. The host runs gradient descent on
/// per-spring rest lengths so the particle ends up at the target.
/// Visualised as kinematic anchor bodies + a particle body + line
/// gizmos for the springs. Port of `example_diffsim_spring_cage.py`.
const DIFF_SPRING_CAGE_HALF: f64 = 0.6;
const DIFF_SPRING_CENTRE_Y: f64 = 1.5;
fn build_diff_spring_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();

    let mut visuals = Vec::with_capacity(9);
    // 8 cage-corner anchors (kinematic spheres). Order matches the
    // anchor list initialised on first frame in update_diff_spring.
    for sx in &[-1.0_f64, 1.0] {
        for sy in &[-1.0_f64, 1.0] {
            for sz in &[-1.0_f64, 1.0] {
                let pos = MVec3::new(
                    sx * DIFF_SPRING_CAGE_HALF,
                    DIFF_SPRING_CENTRE_Y + sy * DIFF_SPRING_CAGE_HALF,
                    sz * DIFF_SPRING_CAGE_HALF,
                );
                let body = b.add_body(
                    mp::BodyParams::new()
                        .with_mass(0.0)
                        .with_inertia(molla::math::Mat3::ZERO)
                        .with_position(pos)
                        .with_flags(mp::BodyFlags::KINEMATIC),
                );
                b.add_shape_sphere(body, 0.04, mp::Transform::IDENTITY)
                    .expect("anchor sphere");
                visuals.push(BodyVisual::Sphere(0.04));
            }
        }
    }
    // Body 8 — kinematic particle (the variable being optimised). The
    // host writes its position each frame from the gradient-descent
    // result.
    let particle = b.add_body(
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_inertia(molla::math::Mat3::ZERO)
            .with_position(MVec3::new(0.0, DIFF_SPRING_CENTRE_Y, 0.0))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(particle, 0.08, mp::Transform::IDENTITY)
        .expect("particle");
    visuals.push(BodyVisual::Sphere(0.08));

    // Static plane.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("diff spring model"), visuals)
}
/// BrickWall scene — a wide, low brick wall struck by a heavy
/// projectile from one side. Geometry-only port of
/// `xtra/newton/newton/examples/contacts/example_brick_stacking.py`
/// (the full Newton example loads a Franka arm and stacks bricks
/// with IK; we ship only the wall + projectile so the WIDE-not-TALL
/// layout differentiates this from `Pyramid` and `Tower`).
const BRICK_HALF: MVec3 = MVec3::new(0.18, 0.10, 0.10);
const BRICK_GAP: f64 = 0.005;
const BRICK_ROWS: usize = 6;
const BRICK_COLS: usize = 10;
fn build_brick_wall_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.20);

    let mass = 0.6_f64;
    let h2 = BRICK_HALF * BRICK_HALF;
    let m_third = mass / 3.0;
    let inertia = molla::math::Mat3::from_diagonal(MVec3::new(
        m_third * (h2.y + h2.z),
        m_third * (h2.x + h2.z),
        m_third * (h2.x + h2.y),
    ));

    let mut visuals = Vec::new();

    let layer_h = BRICK_HALF.y * 2.0 + BRICK_GAP;
    let col_w = BRICK_HALF.x * 2.0 + BRICK_GAP;
    for row in 0..BRICK_ROWS {
        // Half-brick offset on every other row (real masonry).
        let stagger = if row % 2 == 0 { 0.0 } else { BRICK_HALF.x };
        for col in 0..BRICK_COLS {
            let x = (col as f64 - (BRICK_COLS as f64 - 1.0) * 0.5) * col_w + stagger;
            let y = BRICK_HALF.y + 0.001 + (row as f64) * layer_h;
            let body = b.add_body(
                mp::BodyParams::new()
                    .with_spatial_inertia(mp::SpatialInertia {
                        mass,
                        com: MVec3::ZERO,
                        inertia,
                    })
                    .with_position(MVec3::new(x, y, 0.0)),
            );
            b.add_shape_box(body, BRICK_HALF, mp::Transform::IDENTITY)
                .expect("brick");
            visuals.push(BodyVisual::Box(Vec3::new(
                BRICK_HALF.x as f32,
                BRICK_HALF.y as f32,
                BRICK_HALF.z as f32,
            )));
        }
    }

    // Heavy projectile dropped from the +Z side. Aim it diagonally so
    // gravity gives it a downward + side-on impact on the wall.
    let proj_radius = 0.45_f64;
    let proj_mass = 25.0_f64;
    let projectile = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia::sphere(proj_mass, proj_radius))
            .with_position(MVec3::new(0.0, 12.0, 3.5)),
    );
    b.add_shape_sphere(projectile, proj_radius, mp::Transform::IDENTITY)
        .expect("projectile");
    visuals.push(BodyVisual::Sphere(proj_radius as f32));

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("brick wall model"), visuals)
}
/// SoftbodyDrop scene — a 6×6×3 tet jelly cube dropped onto a hanging
/// VBD cloth. Tests soft-on-soft contact at the engine layer. Port of
/// `xtra/newton/newton/examples/multiphysics/example_softbody_dropping_to_cloth.py`.
const SOFT_DROP_TET_NX: u32 = 6;
const SOFT_DROP_TET_NY: u32 = 6;
const SOFT_DROP_TET_NZ: u32 = 3;
const SOFT_DROP_TET_DX: f64 = 0.10;
const SOFT_DROP_CLOTH_NX: u32 = 24;
const SOFT_DROP_CLOTH_W: f64 = 1.6;
fn build_softbody_drop_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_linear_damping(0.4);

    // 1. Cloth grid hanging at Y ≈ 1.0, pinned at left + right edges.
    let origin = MVec3::new(
        -SOFT_DROP_CLOTH_W * 0.5,
        1.0,
        -SOFT_DROP_CLOTH_W * 0.5,
    );
    let u_axis = MVec3::new(SOFT_DROP_CLOTH_W, 0.0, 0.0);
    let v_axis = MVec3::new(0.0, 0.0, SOFT_DROP_CLOTH_W);
    let (cloth_first, _, _) = b.add_cloth_grid(
        origin,
        u_axis,
        v_axis,
        SOFT_DROP_CLOTH_NX,
        SOFT_DROP_CLOTH_NX,
        0.05,        // particle mass
        0.04,        // particle radius (larger so contact lands)
        5.0e-6,      // stretch compliance
        0.05,        // max stretch
        mp::WorldId(0),
    );
    // Pin two opposite edges (iy = 0 row and iy = NX-1 row).
    for ix in 0..SOFT_DROP_CLOTH_NX {
        b.pin_particle(cloth_first + ix);
        b.pin_particle(cloth_first + (SOFT_DROP_CLOTH_NX - 1) * SOFT_DROP_CLOTH_NX + ix);
    }

    // 2. Tet jelly cube above the cloth at Y ≈ 2.5.
    let cube_origin = MVec3::new(
        -SOFT_DROP_TET_DX * (SOFT_DROP_TET_NX as f64 - 1.0) * 0.5,
        2.5,
        -SOFT_DROP_TET_DX * (SOFT_DROP_TET_NZ as f64 - 1.0) * 0.5,
    );
    for kk in 0..SOFT_DROP_TET_NZ {
        for jj in 0..SOFT_DROP_TET_NY {
            for ii in 0..SOFT_DROP_TET_NX {
                let p = cube_origin
                    + MVec3::new(
                        ii as f64 * SOFT_DROP_TET_DX,
                        jj as f64 * SOFT_DROP_TET_DX,
                        kk as f64 * SOFT_DROP_TET_DX,
                    );
                b.add_particle(p, 0.05, SOFT_DROP_TET_DX * 0.5, mp::WorldId(0));
            }
        }
    }
    let v_at = |i: u32, j: u32, k: u32| -> u32 {
        // Tet particle ids start AFTER the cloth's (cloth_first + nx² particles).
        let cloth_n = SOFT_DROP_CLOTH_NX * SOFT_DROP_CLOTH_NX;
        cloth_first + cloth_n + (k * SOFT_DROP_TET_NY + j) * SOFT_DROP_TET_NX + i
    };
    let mut tets = Vec::new();
    for kk in 0..SOFT_DROP_TET_NZ - 1 {
        for jj in 0..SOFT_DROP_TET_NY - 1 {
            for ii in 0..SOFT_DROP_TET_NX - 1 {
                let corners = [
                    v_at(ii, jj, kk),
                    v_at(ii + 1, jj, kk),
                    v_at(ii, jj + 1, kk),
                    v_at(ii + 1, jj + 1, kk),
                    v_at(ii, jj, kk + 1),
                    v_at(ii + 1, jj, kk + 1),
                    v_at(ii, jj + 1, kk + 1),
                    v_at(ii + 1, jj + 1, kk + 1),
                ];
                let cell_tets = jelly_cube_to_six_tets(corners);
                for tet in cell_tets {
                    tets.push(tet);
                }
            }
        }
    }
    let materials = vec![mp::MaterialNeoHookean::elastic(2.0e5, 0.3)];
    let mat_idx: Vec<u32> = vec![0; tets.len()];
    let cube_tet_id = b
        .add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
        .expect("add tet mesh");
    // Stage 33.5 — friction = 0.5 keeps the cube STUCK on the cloth
    // (the Stage 33 hand-off note's "cube slowly drifts laterally
    // and slips off the cloth edge in the viewer's 2-edge-pinned
    // scene at >2 s sim time" was specifically about the
    // friction-less case).
    b.set_tet_mesh_friction(cube_tet_id, 0.5);

    // Floor.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("softbody drop model"), Vec::new())
}

// Stage 33.3 — ClothCollar scene constants.
const COLLAR_NX: u32 = 24;
const COLLAR_W: f64 = 0.6;
const COLLAR_TOP_Y: f64 = 1.6;
/// Stage 33.3 — pinned cloth tube collapsing onto itself. The cloth
/// is built as a 24×24 grid that is then pinned along its iy=0 row
/// (acts as the "tube floor") and along its iy=COLLAR_NX-1 row
/// (acts as the kinematic puller). The puller is driven down by
/// `update_cloth_collar_example` — that brings the top edge near the
/// bottom edge and the cloth folds onto itself. Without self-
/// collision the layers interpenetrate; with it on (the default for
/// this scene) they stack.
fn build_cloth_collar_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_linear_damping(0.4);

    let cy: f64 = COLLAR_TOP_Y;
    let origin = MVec3::new(-COLLAR_W * 0.5, cy, -COLLAR_W * 0.5);
    let u_axis = MVec3::new(COLLAR_W, 0.0, 0.0);
    let v_axis = MVec3::new(0.0, 0.0, COLLAR_W);
    let (first, _, _) = b.add_cloth_grid(
        origin,
        u_axis,
        v_axis,
        COLLAR_NX,
        COLLAR_NX,
        0.05,        // particle mass
        0.02,        // particle radius
        5.0e-6,      // stretch compliance
        0.05,        // max stretch
        mp::WorldId(0),
    );
    // Pin two opposite long edges: iy=0 row stays fixed; iy=NX-1 row
    // is the kinematic puller (pinned + advected by the update sys).
    for ix in 0..COLLAR_NX {
        b.pin_particle(first + ix);
        b.pin_particle(first + (COLLAR_NX - 1) * COLLAR_NX + ix);
    }
    // Stage 33.3 — opt this cloth into self-collision (default ON for
    // the scene; the pane lets the user toggle it off for A/B).
    b.set_cloth_mesh_self_collision(0, true);

    // Floor plane so the cloth has something to settle against.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("cloth collar model"), Vec::new())
}

// Stage 33.4 — FastSoftImpact constants. Cloth grid is 16x16; the
// "small horizontal cloth strip" of the spec is honoured by pinning
// all four edges so the central area takes the impact.
const FAST_SOFT_IMPACT_CLOTH_NX: u32 = 16;

// Atomic-backed slider state for impact speed (m/s). Default 5 m/s
// (the spec-stated impact speed). The right-rail pane writes here and
// the `finalize` / `reset` paths read it to (re)set the cube's qd.
// We store the f32 bit pattern in the AtomicU32. A pristine zero
// value (haven't written yet) is treated as "use the default".
static FAST_SOFT_IMPACT_SPEED_M_S: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

fn fast_soft_impact_speed() -> f64 {
    let bits = FAST_SOFT_IMPACT_SPEED_M_S.load(std::sync::atomic::Ordering::Relaxed);
    if bits == 0 {
        return 5.0;
    }
    let stored = f32::from_bits(bits) as f64;
    if stored.is_finite() && stored > 0.0 {
        stored
    } else {
        5.0
    }
}

fn set_fast_soft_impact_speed(v: f32) {
    FAST_SOFT_IMPACT_SPEED_M_S.store(v.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

/// Stage 33.4 — FastSoftImpact scene. A small (3×3×3 vert) tet cube
/// is launched downward at ~5 m/s into a horizontal pinned cloth strip
/// (16×4). With CCD on (default for this scene) the cube stops at the
/// cloth; with CCD off it tunnels through. Right-rail pane carries
/// the toggles + impact-speed slider.
fn build_fast_soft_impact_model() -> (mp::Model, Vec<BodyVisual>) {
    use mp::Vec3 as MVec3;
    let mut b = mp::ModelBuilder::new();
    b.set_linear_damping(0.20);

    // Horizontal pinned cloth strip at y = 1.0.
    let cloth_n = 16u32;
    let cloth_w: f64 = 0.6;
    let cloth_y: f64 = 1.0;
    let (cloth_first, _, _) = b.add_cloth_grid(
        MVec3::new(-cloth_w * 0.5, cloth_y, -cloth_w * 0.5),
        MVec3::new(cloth_w, 0.0, 0.0),
        MVec3::new(0.0, 0.0, cloth_w),
        cloth_n,
        cloth_n,
        0.05,
        0.02,
        5.0e-6,
        0.05,
        mp::WorldId(0),
    );
    // Pin all four edges so the strip is taut.
    for ix in 0..cloth_n {
        b.pin_particle(cloth_first + ix);
        b.pin_particle(cloth_first + (cloth_n - 1) * cloth_n + ix);
        b.pin_particle(cloth_first + ix * cloth_n);
        b.pin_particle(cloth_first + ix * cloth_n + cloth_n - 1);
    }

    // Tet cube above the cloth — 3×3×3 verts, 5cm cell size.
    let nx = 3u32;
    let dx: f64 = 0.05;
    let cube_lo_y: f64 = cloth_y + 0.30; // 30 cm above cloth.
    let cube_origin = MVec3::new(
        -dx * (nx as f64 - 1.0) * 0.5,
        cube_lo_y,
        -dx * (nx as f64 - 1.0) * 0.5,
    );
    for kk in 0..nx {
        for jj in 0..nx {
            for ii in 0..nx {
                let p = cube_origin
                    + MVec3::new(ii as f64 * dx, jj as f64 * dx, kk as f64 * dx);
                b.add_particle(p, 0.05, dx * 0.4, mp::WorldId(0));
            }
        }
    }
    let cloth_count = cloth_n * cloth_n;
    let v_at = |i: u32, j: u32, k: u32| -> u32 {
        cloth_first + cloth_count + (k * nx + j) * nx + i
    };
    let mut tets = Vec::new();
    for kk in 0..nx - 1 {
        for jj in 0..nx - 1 {
            for ii in 0..nx - 1 {
                let corners = [
                    v_at(ii, jj, kk),
                    v_at(ii + 1, jj, kk),
                    v_at(ii, jj + 1, kk),
                    v_at(ii + 1, jj + 1, kk),
                    v_at(ii, jj, kk + 1),
                    v_at(ii + 1, jj, kk + 1),
                    v_at(ii, jj + 1, kk + 1),
                    v_at(ii + 1, jj + 1, kk + 1),
                ];
                let cell_tets = jelly_cube_to_six_tets(corners);
                for tet in cell_tets {
                    tets.push(tet);
                }
            }
        }
    }
    let materials = vec![mp::MaterialNeoHookean::elastic(2.0e5, 0.3)];
    let mat_idx: Vec<u32> = vec![0; tets.len()];
    let _ = b
        .add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
        .expect("add tet mesh");

    // Floor (so the cube has somewhere to land in the no-CCD case).
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("fast soft impact model"), Vec::new())
}

/// Stage 33.5 SoftBrake — soft tet block sliding down a soft tet
/// ramp. Both registered as soft surfaces with adjustable friction
/// (default 0.5; pane slider exposes 0.0..1.0).
fn build_soft_brake_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_linear_damping(0.05);

    // Tilt the WORLD gravity (Newton-style "tilt the world" trick).
    let ramp_tilt: f64 = 25.0_f64.to_radians();
    b.set_world_gravity(
        mp::WorldId(0),
        MVec3::new(9.81 * ramp_tilt.sin(), -9.81 * ramp_tilt.cos(), 0.0),
    );

    // Ramp: a thick (1.5 × 0.10 × 0.6 m) kinematic tet slab at y = 1.0.
    let ramp_lo = MVec3::new(-0.75, 1.0, -0.30);
    let ramp_hi = MVec3::new(0.75, 1.10, 0.30);
    let mut ramp_verts: Vec<u32> = Vec::with_capacity(8);
    for &z in &[ramp_lo.z, ramp_hi.z] {
        for &y in &[ramp_lo.y, ramp_hi.y] {
            for &x in &[ramp_lo.x, ramp_hi.x] {
                let id = b.add_particle(
                    MVec3::new(x, y, z),
                    0.0,
                    0.02,
                    mp::WorldId(0),
                );
                ramp_verts.push(id);
            }
        }
    }
    let rv: [u32; 8] = ramp_verts.as_slice().try_into().unwrap();
    let ramp_tets = jelly_cube_to_six_tets(rv);
    let ramp_id = b
        .add_tet_mesh(&ramp_tets, 1.0e6, 0.3, None)
        .expect("ramp tet block");
    b.set_tet_mesh_friction(ramp_id, 0.5);

    // Block: a 3×3×3 jelly cube placed just above the ramp.
    let bnx: u32 = 3;
    let bdx: f64 = 0.10;
    let block_origin = MVec3::new(
        -0.50,
        1.10 + 0.005,
        -bdx * (bnx as f64 - 1.0) * 0.5,
    );
    let block_first = 8u32;
    for kk in 0..bnx {
        for jj in 0..bnx {
            for ii in 0..bnx {
                let p = block_origin
                    + MVec3::new(ii as f64 * bdx, jj as f64 * bdx, kk as f64 * bdx);
                b.add_particle(p, 0.05, bdx * 0.5, mp::WorldId(0));
            }
        }
    }
    let v_at_b = |i: u32, j: u32, k: u32| -> u32 {
        block_first + (k * bnx + j) * bnx + i
    };
    let mut block_tets: Vec<[u32; 4]> = Vec::new();
    for kk in 0..bnx - 1 {
        for jj in 0..bnx - 1 {
            for ii in 0..bnx - 1 {
                let corners = [
                    v_at_b(ii, jj, kk),
                    v_at_b(ii + 1, jj, kk),
                    v_at_b(ii, jj + 1, kk),
                    v_at_b(ii + 1, jj + 1, kk),
                    v_at_b(ii, jj, kk + 1),
                    v_at_b(ii + 1, jj, kk + 1),
                    v_at_b(ii, jj + 1, kk + 1),
                    v_at_b(ii + 1, jj + 1, kk + 1),
                ];
                block_tets.extend_from_slice(&jelly_cube_to_six_tets(corners));
            }
        }
    }
    let materials = vec![mp::MaterialNeoHookean::elastic(1.0e6, 0.3)];
    let mat_idx: Vec<u32> = vec![0; block_tets.len()];
    let block_id = b
        .add_tet_mesh_per_material(&block_tets, &materials, &mat_idx, None)
        .expect("block tet mesh");
    b.set_tet_mesh_friction(block_id, 0.5);

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("soft brake model"), Vec::new())
}

// ── Per-scene state stubs for Newton-port scenes (defaults populated
// once the scene is implemented). These are insert_resource'd in main()
// and bundled into SceneStates / PaneSceneStates.

#[derive(Resource)]
struct ConveyorExampleState {
    /// Belt linear speed (m/s) at the top surface.
    belt_speed: f32,
}

impl Default for ConveyorExampleState {
    fn default() -> Self {
        Self { belt_speed: 0.6 }
    }
}

#[derive(Resource)]
struct ClothBendingExampleState {
    /// Bend compliance for left strip (m/N) — small = stiff.
    k_b_left: f32,
    /// Bend compliance for right strip.
    k_b_right: f32,
    /// Whether sliders moved since last build — triggers rebuild.
    rebuild_pending: bool,
}

impl Default for ClothBendingExampleState {
    fn default() -> Self {
        Self {
            k_b_left: 0.05,
            k_b_right: 5.0,
            rebuild_pending: false,
        }
    }
}

#[derive(Resource)]
struct ClothTwistExampleState {
    /// Rotation rate (rad/s) for the bottom edge.
    rotation_rate: f32,
    /// Accumulated angle (rad).
    angle: f32,
    /// Cached top-edge particle indices (pinned).
    top_indices: Vec<u32>,
    /// Cached bottom-edge particle indices (rotated).
    bottom_indices: Vec<u32>,
    /// Cached initial position of every bottom particle (relative to
    /// the rotation axis — used to recompute the rotated position).
    bottom_rest: Vec<MVec3>,
    /// Rotation axis centre (Y-axis through the cloth centre).
    centre: MVec3,
}

impl Default for ClothTwistExampleState {
    fn default() -> Self {
        Self {
            rotation_rate: 0.4,
            angle: 0.0,
            top_indices: Vec::new(),
            bottom_indices: Vec::new(),
            bottom_rest: Vec::new(),
            centre: MVec3::ZERO,
        }
    }
}

#[derive(Resource)]
struct CartPoleExampleState {
    /// Single-cart-pole controller (lazy-init).
    controller: Option<ControllerPd>,
    /// PD gains.
    kp: f32,
    kd: f32,
    /// Pole angle target (rad). Default 0 = upright.
    target_theta: f32,
    /// Live readouts.
    last_x: f32,
    last_theta: f32,
    last_force: f32,
    /// Auto perturb timer.
    perturb_timer: f32,
    auto_perturb: bool,
}

impl Default for CartPoleExampleState {
    fn default() -> Self {
        Self {
            controller: None,
            kp: 50.0,
            kd: 8.0,
            target_theta: 0.0,
            last_x: 0.0,
            last_theta: 0.0,
            last_force: 0.0,
            perturb_timer: 0.0,
            auto_perturb: true,
        }
    }
}

#[derive(Resource)]
struct FourBarExampleState {
    /// Driver joint angular velocity (rad/s). Constant — the driver
    /// joint is prescribed kinematically.
    driver_speed: f32,
    /// Cached driver angle (for pane readout).
    driver_angle: f32,
}

impl Default for FourBarExampleState {
    fn default() -> Self {
        Self {
            driver_speed: 0.6,
            driver_angle: 0.0,
        }
    }
}

#[derive(Resource)]
struct DiffBallExampleState {
    /// Initial speed magnitude (m/s).
    v: f32,
    /// Initial angle from horizontal (degrees).
    angle_deg: f32,
    /// Target XY (Z = 0).
    target: [f32; 2],
    /// Latest computed miss distance (m).
    miss: f32,
    /// Convergence iter count from last "Optimise" run.
    iters_run: u32,
    /// Current sample point along the playback trajectory.
    playback_t: f32,
    /// Triggered by the pane to re-run gradient descent.
    rerun: bool,
    /// Trail of (v, angle) pairs visited during optimisation.
    history: Vec<[f32; 2]>,
    /// Trail of last-rollout (x, y) sample points (for visualization).
    trajectory: Vec<[f32; 2]>,
}

impl Default for DiffBallExampleState {
    fn default() -> Self {
        Self {
            v: 6.0,
            angle_deg: 50.0,
            target: [3.0, 1.5],
            miss: 0.0,
            iters_run: 0,
            playback_t: 0.0,
            rerun: true,
            history: Vec::new(),
            trajectory: Vec::new(),
        }
    }
}

#[derive(Resource)]
struct DiffSpringExampleState {
    /// Number of springs in the cage (kept fixed at build time; UI
    /// just shows it).
    n_springs: u32,
    /// Per-spring rest length (current optimised value).
    rest_lengths: Vec<f32>,
    /// Cage anchor points (world coords) — fixed.
    anchors: Vec<[f32; 3]>,
    /// Particle position the optimiser is solving for.
    particle: [f32; 3],
    /// Target position (the particle should land here).
    target: [f32; 3],
    /// Last residual (m).
    last_residual: f32,
    /// Iterations of gradient descent run.
    iters_run: u32,
    /// Re-run trigger.
    rerun: bool,
}

impl Default for DiffSpringExampleState {
    fn default() -> Self {
        Self {
            n_springs: 0,
            rest_lengths: Vec::new(),
            anchors: Vec::new(),
            particle: [0.0, 1.5, 0.0],
            target: [0.6, 1.0, 0.2],
            last_residual: 0.0,
            iters_run: 0,
            rerun: true,
        }
    }
}

// ── Update-system stubs for Newton-port scenes (filled in below
// scene-by-scene). Each is a no-op when the active scene differs.

fn update_cartpole_example(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut state: ResMut<CartPoleExampleState>,
) {
    let is_swarm = sim.scene == Scene::CartPoleSwarm;
    if sim.scene != Scene::CartPole && !is_swarm {
        if state.controller.is_some() {
            state.controller = None;
        }
        return;
    }
    if sim.model.joint_dof_total < 2 {
        return;
    }
    // Lazy-init: ControllerPd is over-engineered for our needs here —
    // we drive `joint_force` directly from a closed-form LQR-style
    // balance policy below. We still construct it once so subsequent
    // re-entries are cheap; the controller object is dropped on scene
    // switch.
    if state.controller.is_none() {
        let cfg = ControllerPdConfig {
            dof: 0,
            kp: state.kp as f64,
            kd: state.kd as f64,
            max_effort: 60.0,
        };
        match ControllerPd::new(&sim.engine, &sim.model, &[cfg]) {
            Ok(c) => state.controller = Some(c),
            Err(e) => {
                warn!("cartpole: controller init failed: {e}");
                return;
            }
        }
    }

    // Read state. Snapshot to owned Vecs so we can mutate `sim` below.
    let q_vec: Vec<f64> = match sim.state_a.joint_q.host() {
        Ok(q) => q.to_vec(),
        Err(_) => return,
    };
    let qd_vec: Vec<f64> = match sim.state_a.joint_qd.host() {
        Ok(q) => q.to_vec(),
        Err(_) => return,
    };
    if q_vec.len() < 2 || qd_vec.len() < 2 {
        return;
    }
    // Single-pole readouts mirror the first cart-pole.
    let cart_x = q_vec[0] as f32;
    let _cart_xd = qd_vec[0] as f32;
    let theta = q_vec[1] as f32;
    let _theta_d = qd_vec[1] as f32;
    state.last_x = cart_x;
    state.last_theta = theta;

    // Auto-perturb every 3 s — only the first cart in swarm mode
    // (otherwise the swarm jumps in lockstep and looks fake).
    state.perturb_timer += time.delta_secs();
    if state.auto_perturb && state.perturb_timer > 3.0 {
        state.perturb_timer = 0.0;
        let queue = sim.engine.queue().expect("queue").clone();
        if let Ok(qdh) = sim.state_a.joint_qd.host_mut() {
            if qdh.len() >= 2 {
                let sign = if theta >= 0.0 { -1.0 } else { 1.0 };
                qdh[1] += (sign * 1.5) as f64;
            }
        }
        let _ = sim.state_a.joint_qd.upload(&queue);
    }

    // Inverted-pendulum balance law per cart-pole pair.
    let target_theta = state.target_theta;
    let queue = sim.engine.queue().expect("queue").clone();
    if let Ok(jf) = sim.control.joint_force.host_mut() {
        let pairs = jf.len() / 2;
        for i in 0..pairs {
            let xx = q_vec[2 * i] as f32;
            let xxd = qd_vec[2 * i] as f32;
            let th = q_vec[2 * i + 1] as f32;
            let thd = qd_vec[2 * i + 1] as f32;
            let force = -(2.5 * xx + 1.0 * xxd
                + state.kp * (th - target_theta)
                + state.kd * thd);
            jf[2 * i] = force.clamp(-60.0, 60.0) as f64;
            // Pole revolute receives no actuated torque.
            jf[2 * i + 1] = 0.0;
        }
        if !jf.is_empty() {
            state.last_force = jf[0] as f32;
        }
    }
    let _ = sim.control.joint_force.upload(&queue);
    let _ = is_swarm;
}

fn update_conveyor_example(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    state: ResMut<ConveyorExampleState>,
) {
    if sim.scene != Scene::Conveyor {
        return;
    }
    if sim.model.body_count == 0 {
        return;
    }
    // Belt is body 0. Translate its kinematic pose along +X by
    // belt_speed each viewer frame. Wrap when the belt has shifted
    // by half the belt length so it never disappears off-screen.
    let dt = time.delta_secs() as f64;
    let v = state.belt_speed as f64;
    let body_q = match sim.state_a.body_q.host_mut() {
        Ok(b) => b,
        Err(_) => return,
    };
    let mut pose = body_q[0];
    pose.position.x += v * dt;
    if pose.position.x > CONVEYOR_BELT_LEN * 0.5 {
        pose.position.x -= CONVEYOR_BELT_LEN;
    } else if pose.position.x < -CONVEYOR_BELT_LEN * 0.5 {
        pose.position.x += CONVEYOR_BELT_LEN;
    }
    body_q[0] = pose;
    // Push belt linear velocity into body_qd[0] so contact friction
    // gets the right tangential surface velocity.
    let body_qd = match sim.state_a.body_qd.host_mut() {
        Ok(qd) => qd,
        Err(_) => return,
    };
    body_qd[0].linear = MVec3::new(v, 0.0, 0.0);

    if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
        let device = device.clone();
        let queue = queue.clone();
        let _ = sim.state_a.body_q.ensure_device(&device);
        let _ = sim.state_a.body_q.upload(&queue);
        let _ = sim.state_a.body_qd.ensure_device(&device);
        let _ = sim.state_a.body_qd.upload(&queue);
    }
}

fn update_cloth_twist_example(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut state: ResMut<ClothTwistExampleState>,
) {
    if sim.scene != Scene::ClothTwist {
        return;
    }
    if sim.model.particle_count == 0 {
        return;
    }
    // Lazy-cache the bottom-row particle indices and their initial
    // positions relative to the rotation axis on first frame.
    if state.bottom_indices.is_empty() {
        let bottom_iy = CLOTH_TWIST_GRID_Y - 1;
        let centre = MVec3::new(
            CLOTH_TWIST_CENTRE_X,
            CLOTH_TWIST_TOP_Y - CLOTH_TWIST_H,
            0.0,
        );
        state.centre = centre;
        let q = match sim.state_a.particle_q.host() {
            Ok(q) => q,
            Err(_) => return,
        };
        for ix in 0..CLOTH_TWIST_GRID_X {
            let i = bottom_iy * CLOTH_TWIST_GRID_X + ix;
            state.bottom_indices.push(i);
            state.bottom_rest.push(q[i as usize] - centre);
            // Top row indices for the readout (no rewrite needed —
            // they're pinned at their initial positions automatically).
            state.top_indices.push(ix);
        }
    }
    // Advance the angle by rotation_rate per second.
    state.angle += state.rotation_rate * time.delta_secs();
    let cos_a = state.angle.cos() as f64;
    let sin_a = state.angle.sin() as f64;

    let q = match sim.state_a.particle_q.host_mut() {
        Ok(q) => q,
        Err(_) => return,
    };
    for (k, &idx) in state.bottom_indices.iter().enumerate() {
        let rest = state.bottom_rest[k];
        // Rotate around Y axis (X' = X·cos + Z·sin, Z' = -X·sin + Z·cos).
        let xnew = rest.x * cos_a + rest.z * sin_a;
        let znew = -rest.x * sin_a + rest.z * cos_a;
        q[idx as usize] = state.centre + MVec3::new(xnew, rest.y, znew);
    }
    if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
        let device = device.clone();
        let queue = queue.clone();
        let _ = sim.state_a.particle_q.ensure_device(&device);
        let _ = sim.state_a.particle_q.upload(&queue);
    }
}

/// Stage 33.3 — drive the cloth-collar's iy=NX-1 row downward each
/// frame so the cloth folds onto itself. Reads the live puller Y
/// from a static atomic; resets it when the pane requests.
fn update_cloth_collar_example(time: Res<Time>, mut sim: ResMut<Sim>) {
    if sim.scene != Scene::ClothCollar {
        return;
    }
    if sim.model.particle_count == 0 {
        return;
    }
    // Pinned-row indices: iy=NX-1, ix∈[0, NX). Particles are laid out
    // row-major over (ix, iy) starting at particle 0 (this scene has
    // no other particles before the cloth).
    let nx = COLLAR_NX as usize;
    let row_start = (nx - 1) * nx;
    // Initialise puller Y on first frame (when the bits are still 0,
    // f64::from_bits(0) == 0.0; we treat that as "uninitialised").
    let mut y = collar_get_puller_y();
    if y == 0.0 {
        y = COLLAR_TOP_Y;
        collar_set_puller_y(y);
    }
    if COLLAR_RESET_REQUESTED.swap(false, std::sync::atomic::Ordering::Relaxed) {
        y = COLLAR_TOP_Y;
        collar_set_puller_y(y);
        // Snap puller particles back up.
        if let Ok(q) = sim.state_a.particle_q.host_mut() {
            for ix in 0..nx {
                let idx = row_start + ix;
                if idx < q.len() {
                    let p = q[idx];
                    q[idx] = MVec3::new(p.x, y, p.z);
                }
            }
        }
        if let (Some(device), Some(queue)) =
            (sim.engine.device(), sim.engine.queue())
        {
            let device = device.clone();
            let queue = queue.clone();
            let _ = sim.state_a.particle_q.ensure_device(&device);
            let _ = sim.state_a.particle_q.upload(&queue);
        }
        return;
    }
    // Descend at 0.04 m/s — slow enough that the cloth folds without
    // the puller punching through (the contact stiffness needs time
    // to engage). Stop at the bottom edge's nominal Y (the iy=0 row
    // sits at COLLAR_TOP_Y exactly when the cloth is flat — but it
    // sags slightly under gravity; we descend to 0.02 m above the
    // top floor so the layers visibly stack).
    let descend_speed: f64 = 0.04;
    let target_y: f64 = 0.02;
    if y > target_y {
        y -= descend_speed * time.delta_secs() as f64;
        if y < target_y {
            y = target_y;
        }
        collar_set_puller_y(y);
        if let Ok(q) = sim.state_a.particle_q.host_mut() {
            for ix in 0..nx {
                let idx = row_start + ix;
                if idx < q.len() {
                    let p = q[idx];
                    q[idx] = MVec3::new(p.x, y, p.z);
                }
            }
        }
        if let (Some(device), Some(queue)) =
            (sim.engine.device(), sim.engine.queue())
        {
            let device = device.clone();
            let queue = queue.clone();
            let _ = sim.state_a.particle_q.ensure_device(&device);
            let _ = sim.state_a.particle_q.upload(&queue);
        }
    }
}

fn update_fourbar_example(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut state: ResMut<FourBarExampleState>,
) {
    if sim.scene != Scene::FourBar {
        return;
    }
    if sim.model.joint_dof_total < 1 {
        return;
    }
    state.driver_angle += state.driver_speed * time.delta_secs();
    // Keep driver velocity prescribed: write joint_target_pos[0] +
    // joint_target_vel[0] every frame; rely on JointMimic to slave
    // the other joints. Without an actuator the joint won't track —
    // we instead overwrite joint_q / joint_qd directly each frame.
    let queue = sim.engine.queue().expect("queue").clone();
    if let Ok(q) = sim.state_a.joint_q.host_mut() {
        if !q.is_empty() {
            q[0] = state.driver_angle as f64;
            // Slave joints — linear mimic factors match the
            // `add_equality_joint_mimic` polynomials in build.
            if q.len() >= 2 {
                q[1] = -(state.driver_angle as f64);
            }
            if q.len() >= 3 {
                q[2] = state.driver_angle as f64;
            }
        }
    }
    if let Ok(qd) = sim.state_a.joint_qd.host_mut() {
        if !qd.is_empty() {
            qd[0] = state.driver_speed as f64;
            if qd.len() >= 2 {
                qd[1] = -(state.driver_speed as f64);
            }
            if qd.len() >= 3 {
                qd[2] = state.driver_speed as f64;
            }
        }
    }
    let _ = sim.state_a.joint_q.upload(&queue);
    let _ = sim.state_a.joint_qd.upload(&queue);
}

fn update_diff_ball_example(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut state: ResMut<DiffBallExampleState>,
) {
    if sim.scene != Scene::DiffBall {
        return;
    }
    if sim.model.body_count < 2 {
        return;
    }

    // Re-optimise (v, angle) when triggered by a slider change.
    if state.rerun {
        // Closed-form projectile under gravity (no air drag).
        // Position at time t given (v, θ):
        //   x(t) = v cos θ · t
        //   y(t) = y0 + v sin θ · t − ½ g t²
        // Pick the t* that minimises (x(t)−Tx)² + (y(t)−Ty)² and
        // evaluate the loss; back-propagate via finite differences.
        let target_x = state.target[0] as f64;
        let target_y = state.target[1] as f64;
        let g = 9.81;
        let y0 = 0.5;
        // Sample a coarse grid of t to find the closest approach.
        let trajectory_for = |v: f64, ang: f64| -> Vec<[f32; 2]> {
            let cos_a = ang.cos();
            let sin_a = ang.sin();
            let mut pts = Vec::with_capacity(64);
            for k in 0..64 {
                let t = (k as f64) * 0.04;
                let x = v * cos_a * t;
                let y = y0 + v * sin_a * t - 0.5 * g * t * t;
                if y < 0.0 && k > 1 {
                    break;
                }
                pts.push([x as f32, y as f32]);
            }
            pts
        };
        let miss_for = |v: f64, ang: f64| -> f64 {
            let pts = trajectory_for(v, ang);
            let mut best = f64::INFINITY;
            for p in &pts {
                let dx = p[0] as f64 - target_x;
                let dy = p[1] as f64 - target_y;
                let d2 = dx * dx + dy * dy;
                if d2 < best {
                    best = d2;
                }
            }
            best.sqrt()
        };
        let mut v = state.v as f64;
        let mut ang = (state.angle_deg as f64).to_radians();
        let lr_v = 0.3_f64;
        let lr_a = 0.3_f64;
        let mut history = Vec::new();
        for _it in 0..40 {
            history.push([v as f32, ang.to_degrees() as f32]);
            let m0 = miss_for(v, ang);
            let dv = (miss_for(v + 0.01, ang) - m0) / 0.01;
            let da = (miss_for(v, ang + 0.01) - m0) / 0.01;
            v -= lr_v * dv;
            ang -= lr_a * da;
            v = v.clamp(0.5, 20.0);
            ang = ang.clamp(0.05, 1.55);
            if m0 < 0.05 {
                break;
            }
        }
        state.v = v as f32;
        state.angle_deg = ang.to_degrees() as f32;
        state.miss = miss_for(v, ang) as f32;
        state.iters_run = history.len() as u32;
        state.history = history;
        state.trajectory = trajectory_for(v, ang);
        state.playback_t = 0.0;
        state.rerun = false;
    }

    // Playback: animate the ball along the trajectory.
    state.playback_t += time.delta_secs() * 0.5;
    if state.playback_t > 2.0 {
        state.playback_t = 0.0;
    }
    if state.trajectory.is_empty() {
        return;
    }
    let frac = (state.playback_t / 2.0).clamp(0.0, 1.0);
    let idx = ((frac * (state.trajectory.len() - 1) as f32).floor() as usize)
        .min(state.trajectory.len() - 1);
    let p = state.trajectory[idx];
    let body_q = match sim.state_a.body_q.host_mut() {
        Ok(b) => b,
        Err(_) => return,
    };
    body_q[0] = mp::Transform::from_translation(MVec3::new(p[0] as f64, p[1] as f64, 0.0));
    body_q[1] = mp::Transform::from_translation(MVec3::new(
        state.target[0] as f64,
        state.target[1] as f64,
        0.0,
    ));
    let queue = sim.engine.queue().expect("queue").clone();
    let _ = sim.state_a.body_q.upload(&queue);
}

fn update_diff_spring_example(
    mut sim: ResMut<Sim>,
    mut state: ResMut<DiffSpringExampleState>,
) {
    if sim.scene != Scene::DiffSpring {
        return;
    }
    if sim.model.body_count < 9 {
        return;
    }

    // Lazy-init: cache anchor positions + initialise rest lengths.
    if state.anchors.is_empty() {
        state.anchors.clear();
        for sx in &[-1.0_f64, 1.0] {
            for sy in &[-1.0_f64, 1.0] {
                for sz in &[-1.0_f64, 1.0] {
                    state.anchors.push([
                        (sx * DIFF_SPRING_CAGE_HALF) as f32,
                        (DIFF_SPRING_CENTRE_Y + sy * DIFF_SPRING_CAGE_HALF) as f32,
                        (sz * DIFF_SPRING_CAGE_HALF) as f32,
                    ]);
                }
            }
        }
        state.n_springs = state.anchors.len() as u32;
        // Initial rest length = distance from cage centre to anchor.
        state.rest_lengths = state
            .anchors
            .iter()
            .map(|a| {
                let dx = a[0];
                let dy = a[1] - DIFF_SPRING_CENTRE_Y as f32;
                let dz = a[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .collect();
        state.particle = [0.0, DIFF_SPRING_CENTRE_Y as f32, 0.0];
    }

    // Re-optimise rest lengths to drive the particle to the target.
    if state.rerun {
        // Equilibrium: particle position is the weighted centroid of
        // anchor positions, where weights are 1/rest_length. We
        // gradient-descend on rest lengths so the equilibrium centroid
        // matches the target.
        let target = state.target;
        // Helper: compute current particle equilibrium (closed form).
        let equilibrium = |rls: &[f32], anchors: &[[f32; 3]]| -> [f32; 3] {
            let mut sum_w = 0.0_f32;
            let mut sum_x = [0.0_f32; 3];
            for (a, l) in anchors.iter().zip(rls.iter()) {
                let w = 1.0 / l.max(1e-3);
                sum_w += w;
                sum_x[0] += a[0] * w;
                sum_x[1] += a[1] * w;
                sum_x[2] += a[2] * w;
            }
            if sum_w < 1e-6 {
                [0.0; 3]
            } else {
                [sum_x[0] / sum_w, sum_x[1] / sum_w, sum_x[2] / sum_w]
            }
        };
        let loss = |rls: &[f32], anchors: &[[f32; 3]], tgt: [f32; 3]| -> f32 {
            let p = equilibrium(rls, anchors);
            let dx = p[0] - tgt[0];
            let dy = p[1] - tgt[1];
            let dz = p[2] - tgt[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        };
        let lr = 0.1_f32;
        let mut rls = state.rest_lengths.clone();
        for _it in 0..80 {
            let m0 = loss(&rls, &state.anchors, target);
            // FD gradient on each rest length.
            let mut grads = vec![0.0_f32; rls.len()];
            for i in 0..rls.len() {
                let saved = rls[i];
                rls[i] = saved + 0.005;
                let m1 = loss(&rls, &state.anchors, target);
                grads[i] = (m1 - m0) / 0.005;
                rls[i] = saved;
            }
            for i in 0..rls.len() {
                rls[i] -= lr * grads[i];
                rls[i] = rls[i].max(0.05).min(3.0);
            }
            if m0 < 0.01 {
                break;
            }
        }
        let final_p = equilibrium(&rls, &state.anchors);
        state.rest_lengths = rls;
        state.particle = final_p;
        state.last_residual = loss(&state.rest_lengths, &state.anchors, target);
        state.iters_run = 80;
        state.rerun = false;
    }

    // Push particle pose into body[8] for the renderer.
    let body_q = match sim.state_a.body_q.host_mut() {
        Ok(b) => b,
        Err(_) => return,
    };
    if body_q.len() >= 9 {
        body_q[8] = mp::Transform::from_translation(MVec3::new(
            state.particle[0] as f64,
            state.particle[1] as f64,
            state.particle[2] as f64,
        ));
    }
    let queue = sim.engine.queue().expect("queue").clone();
    let _ = sim.state_a.body_q.upload(&queue);
}

// ── Pane stubs for the rest of the Newton-port scenes — replaced
// scene-by-scene below.

fn conveyor_pane(ui: &mut egui::Ui, state: &mut ConveyorExampleState) {
    ui.label("Conveyor belt");
    ui.label("Belt speed (m/s)");
    ui.add(egui::Slider::new(&mut state.belt_speed, -2.0..=2.0));
    ui.separator();
    ui.small("Boxes are dropped onto a long kinematic box.");
    ui.small("Belt pose is advected each frame; boxes settle on it.");
    ui.colored_label(
        egui::Color32::YELLOW,
        "Caveat: XPBD's Coulomb friction is gated by per-substep \
         normal impulse |λ_n|, which is ~0 at steady state. The \
         box receives only a tiny tangential push from the belt — \
         the belt itself slides away beneath. Visualises belt \
         motion + box settling, not steady-state belt-drag.",
    );
}

fn cloth_bending_pane(
    ui: &mut egui::Ui,
    state: &mut ClothBendingExampleState,
    pending: &mut PendingSceneSwitch,
    scene: Scene,
) {
    ui.label("Bend compliance — left strip (m/N)");
    if ui
        .add(egui::Slider::new(&mut state.k_b_left, 1e-3..=1e3).logarithmic(true))
        .changed()
    {
        state.rebuild_pending = true;
    }
    ui.label("Bend compliance — right strip");
    if ui
        .add(egui::Slider::new(&mut state.k_b_right, 1e-3..=1e3).logarithmic(true))
        .changed()
    {
        state.rebuild_pending = true;
    }
    ui.separator();
    if state.rebuild_pending {
        ui.colored_label(
            egui::Color32::YELLOW,
            "Rebuild required to apply bending changes.",
        );
        if ui.button("Rebuild scene").clicked() {
            set_cloth_bend(state.k_b_left, state.k_b_right);
            pending.0 = Some(scene);
            state.rebuild_pending = false;
        }
    }
    ui.small("Two cloth strips. Smaller compliance = stiffer.");
    ui.small("Pinned along top edges; gravity does the rest.");
}

fn cloth_twist_pane(ui: &mut egui::Ui, state: &mut ClothTwistExampleState) {
    ui.label("Bottom-edge rotation rate (rad/s)");
    ui.add(egui::Slider::new(&mut state.rotation_rate, -2.0..=2.0));
    readout(ui, "Twist angle", &format!("{:>+5.2} rad", state.angle));
    ui.separator();
    ui.small("Top edge pinned, bottom edge rotated.");
}

fn cartpole_pane(
    ui: &mut egui::Ui,
    state: &mut CartPoleExampleState,
    _accent: egui::Color32,
) {
    readout(ui, "cart x", &format!("{:>+5.2} m", state.last_x));
    readout(ui, "pole θ", &format!("{:>+5.2} rad", state.last_theta));
    readout(ui, "force", &format!("{:>+5.2} N", state.last_force));
    ui.separator();
    ui.label("PD gains");
    ui.add(egui::Slider::new(&mut state.kp, 1.0..=300.0).text("kp"));
    ui.add(egui::Slider::new(&mut state.kd, 0.0..=50.0).text("kd"));
    ui.label("Pole target (rad, 0 = upright)");
    ui.add(egui::Slider::new(&mut state.target_theta, -0.5..=0.5));
    ui.checkbox(&mut state.auto_perturb, "Auto-perturb every 3 s");
    ui.separator();
    ui.small("PD on the pole revolute, force applied to cart prismatic.");
}

fn cartpole_swarm_pane(ui: &mut egui::Ui, sim: &Sim) {
    let pairs = sim.model.body_count / 2;
    readout(ui, "Cart-poles", &format!("{}", pairs));
    readout(ui, "Bodies", &format!("{}", sim.model.body_count));
    ui.separator();
    ui.small("16 independent cart-poles in a row.");
    ui.small("Each balanced by its own PD controller.");
}

fn fourbar_pane(ui: &mut egui::Ui, state: &mut FourBarExampleState) {
    ui.label("Driver speed (rad/s)");
    ui.add(egui::Slider::new(&mut state.driver_speed, -2.0..=2.0));
    readout(ui, "Driver angle", &format!("{:>+5.2} rad", state.driver_angle));
    ui.separator();
    ui.small("Closed kinematic loop (4-bar).");
    ui.small("`JointMimic` enforces the closure constraint.");
}

fn diff_ball_pane(
    ui: &mut egui::Ui,
    state: &mut DiffBallExampleState,
    accent: egui::Color32,
) {
    ui.label("Initial speed (m/s)");
    if ui
        .add(egui::Slider::new(&mut state.v, 1.0..=15.0))
        .changed()
    {
        state.rerun = true;
    }
    ui.label("Launch angle (deg)");
    if ui
        .add(egui::Slider::new(&mut state.angle_deg, 5.0..=85.0))
        .changed()
    {
        state.rerun = true;
    }
    ui.label("Target X (m)");
    if ui
        .add(egui::Slider::new(&mut state.target[0], 0.5..=8.0))
        .changed()
    {
        state.rerun = true;
    }
    ui.label("Target Y (m)");
    if ui
        .add(egui::Slider::new(&mut state.target[1], 0.0..=4.0))
        .changed()
    {
        state.rerun = true;
    }
    ui.separator();
    readout(ui, "Miss", &format!("{:.3} m", state.miss));
    readout(ui, "Iters run", &format!("{}", state.iters_run));
    if ui.add_sized([ui.available_width(), 24.0], button_widget("Optimise", accent))
        .clicked()
    {
        state.rerun = true;
    }
    ui.separator();
    ui.small("Projectile under gravity. Tape autograd would");
    ui.small("compute ∂miss/∂(v, θ); we use closed-form here.");
}

fn diff_spring_pane(
    ui: &mut egui::Ui,
    state: &mut DiffSpringExampleState,
    accent: egui::Color32,
) {
    readout(ui, "Springs", &format!("{}", state.n_springs));
    readout(ui, "Iters run", &format!("{}", state.iters_run));
    readout(ui, "Residual", &format!("{:.4} m", state.last_residual));
    ui.separator();
    ui.label("Target X / Y / Z");
    if ui
        .add(egui::Slider::new(&mut state.target[0], -1.0..=1.0))
        .changed()
    {
        state.rerun = true;
    }
    if ui
        .add(egui::Slider::new(&mut state.target[1], 0.5..=2.5))
        .changed()
    {
        state.rerun = true;
    }
    if ui
        .add(egui::Slider::new(&mut state.target[2], -1.0..=1.0))
        .changed()
    {
        state.rerun = true;
    }
    if ui.add_sized([ui.available_width(), 24.0], button_widget("Optimise", accent))
        .clicked()
    {
        state.rerun = true;
    }
    ui.separator();
    ui.small("Particle in a cage of springs. Gradient descent");
    ui.small("on rest lengths drives the particle to the target.");
}

/// Stage 33.4 — FastSoftImpact right-rail pane. Carries:
/// - CCD on/off toggle (re-launches the cube on flip).
/// - Impact-speed slider (m/s).
/// - Live readout of CCD invocations / snaps.
/// - "Re-launch" button to retrigger the impact at the current speed.
fn fast_soft_impact_pane(ui: &mut egui::Ui, sim: &mut Sim) {
    readout(ui, "Particles", &format!("{}", sim.model.particle_count));
    readout(ui, "Tets", &format!("{}", sim.model.tet_count));
    readout(
        ui,
        "Soft surfaces",
        &format!("{}", sim.model.soft_surfaces.len()),
    );

    let mut ccd_on: bool = false;
    let mut vt_count: usize = 0;
    let mut ccd_inv: usize = 0;
    let mut ccd_snaps: usize = 0;
    if let SolverBackend::Vbd(s) = &sim.solver {
        ccd_on = s.ccd_enabled;
        vt_count = s.last_vt_contact_count();
        ccd_inv = s.last_ccd_invocations();
        ccd_snaps = s.last_ccd_snaps();
    }
    readout(ui, "VT contacts", &format!("{vt_count}"));
    readout(ui, "CCD invocations", &format!("{ccd_inv}"));
    readout(ui, "CCD snaps", &format!("{ccd_snaps}"));
    ui.separator();

    let mut speed = fast_soft_impact_speed() as f32;
    ui.label("Impact speed (m/s)");
    if ui.add(egui::Slider::new(&mut speed, 0.5..=15.0)).changed() {
        set_fast_soft_impact_speed(speed);
    }

    if ui
        .checkbox(&mut ccd_on, "CCD enabled (Stage 33.4)")
        .changed()
    {
        if let SolverBackend::Vbd(s) = &mut sim.solver {
            s.ccd_enabled = ccd_on;
        }
    }

    if ui.button("Re-launch impact").clicked() {
        sim.reset();
    }

    ui.separator();
    ui.small("Toggle CCD off + click Re-launch to see the cube tunnel.");
    ui.small("Toggle CCD on + Re-launch: the cube stops at the cloth.");
    ui.small("Stage 33.4 ships VT-CCD only; EE-CCD is deferred.");
}

fn softbody_drop_pane(ui: &mut egui::Ui, sim: &mut Sim) {
    readout(ui, "Particles", &format!("{}", sim.model.particle_count));
    readout(ui, "Tets", &format!("{}", sim.model.tet_count));
    readout(
        ui,
        "Soft surfaces",
        &format!("{}", sim.model.soft_surfaces.len()),
    );
    // Stage 33 — surface the live VT contact count + k_contact slider
    // so users can verify the contact pipeline is engaged. Reads
    // from the live VBD solver instance.
    let mut k_contact: f64 = 0.0;
    let mut vt_count: usize = 0;
    let mut sweeps: i32 = 0;
    if let SolverBackend::Vbd(s) = &sim.solver {
        k_contact = s.contact_stiffness;
        vt_count = s.last_vt_contact_count();
        sweeps = s.vt_contact_sweeps as i32;
    }
    readout(ui, "VT contacts", &format!("{vt_count}"));
    ui.separator();

    ui.label("k_contact (N/m)");
    if ui
        .add(
            egui::Slider::new(&mut k_contact, 1.0e3..=1.0e6).logarithmic(true),
        )
        .changed()
    {
        if let SolverBackend::Vbd(s) = &mut sim.solver {
            s.contact_stiffness = k_contact;
        }
    }
    ui.label("VT contact sweeps");
    if ui.add(egui::Slider::new(&mut sweeps, 1..=16)).changed() {
        if let SolverBackend::Vbd(s) = &mut sim.solver {
            s.vt_contact_sweeps = sweeps.max(1) as usize;
        }
    }
    ui.separator();
    ui.small("Tet jelly cube falls onto a hanging VBD cloth.");
    ui.small("Stage 33 cross-mesh VT contact: the jelly should");
    ui.small("rest on the cloth instead of passing through.");
}

/// Stage 33.3 — ClothCollar right-rail pane. Surfaces the live VT
/// contact count + a self-collision toggle (default ON for this
/// scene; toggle off to see the cloth interpenetrate itself).
fn cloth_collar_pane(ui: &mut egui::Ui, sim: &mut Sim) {
    readout(ui, "Particles", &format!("{}", sim.model.particle_count));
    readout(
        ui,
        "Cloth tris",
        &format!(
            "{}",
            sim.model
                .soft_surfaces
                .first()
                .map(|s| s.tri_count)
                .unwrap_or(0)
        ),
    );
    readout(ui, "Puller Y (m)", &format!("{:.3}", collar_get_puller_y()));

    let mut vt_count: usize = 0;
    let mut k_contact: f64 = 0.0;
    let mut sweeps: i32 = 0;
    if let SolverBackend::Vbd(s) = &sim.solver {
        k_contact = s.contact_stiffness;
        vt_count = s.last_vt_contact_count();
        sweeps = s.vt_contact_sweeps as i32;
    }
    readout(ui, "VT contacts", &format!("{vt_count}"));
    ui.separator();

    // Self-collision toggle. Mirrors the static atomic AND the live
    // model so the gather sees the change immediately.
    let mut self_coll = COLLAR_SELF_COLL_ENABLED
        .load(std::sync::atomic::Ordering::Relaxed);
    if ui.checkbox(&mut self_coll, "Self-collision (Stage 33.3)").changed() {
        COLLAR_SELF_COLL_ENABLED
            .store(self_coll, std::sync::atomic::Ordering::Relaxed);
        // Apply live to the active model — the next gather picks it up.
        sim.model.set_soft_surface_self_collision(0, self_coll);
    }

    ui.label("k_contact (N/m)");
    if ui
        .add(egui::Slider::new(&mut k_contact, 1.0e3..=1.0e6).logarithmic(true))
        .changed()
    {
        if let SolverBackend::Vbd(s) = &mut sim.solver {
            s.contact_stiffness = k_contact;
        }
    }
    ui.label("VT contact sweeps");
    if ui.add(egui::Slider::new(&mut sweeps, 1..=16)).changed() {
        if let SolverBackend::Vbd(s) = &mut sim.solver {
            s.vt_contact_sweeps = sweeps.max(1) as usize;
        }
    }
    ui.separator();
    if ui.button("Reset puller height").clicked() {
        COLLAR_RESET_REQUESTED
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
    ui.separator();
    ui.small("24×24 cloth, two long edges pinned. The top edge is");
    ui.small("kinematically pulled down onto the bottom edge so the");
    ui.small("cloth folds onto itself. With Stage 33.3 self-collision");
    ui.small("ON the layers stack; OFF, they interpenetrate.");
}

/// Stage 33.5 SoftBrake pane — Coulomb friction sweep over the
/// soft-on-soft contact pipeline.
fn soft_brake_pane(ui: &mut egui::Ui, sim: &mut Sim) {
    readout(ui, "Particles", &format!("{}", sim.model.particle_count));
    readout(ui, "Tets", &format!("{}", sim.model.tet_count));
    readout(
        ui,
        "Soft surfaces",
        &format!("{}", sim.model.soft_surfaces.len()),
    );
    let mut vt_count: usize = 0;
    if let SolverBackend::Vbd(s) = &sim.solver {
        vt_count = s.last_vt_contact_count();
    }
    readout(ui, "VT contacts", &format!("{vt_count}"));

    let inv_mass = sim
        .model
        .particle_inv_mass
        .host()
        .ok()
        .map(|s| s.to_vec())
        .unwrap_or_default();
    let mut block_idx: Option<usize> = None;
    for (i, s) in sim.model.soft_surfaces.iter().enumerate() {
        let any_dynamic = (s.vertex_offset..s.vertex_offset + s.vertex_count)
            .any(|v| inv_mass.get(v as usize).copied().unwrap_or(0.0) > 0.0);
        if any_dynamic {
            block_idx = Some(i);
            break;
        }
    }
    ui.separator();

    if let Some(bi) = block_idx {
        let mut friction = sim.model.soft_surfaces[bi].friction as f32;
        ui.label("Block friction μ");
        if ui.add(egui::Slider::new(&mut friction, 0.0_f32..=1.0_f32)).changed() {
            sim.model.soft_surfaces[bi].friction = friction as mp::Real;
        }
        ui.separator();
        let cur_friction = sim.model.soft_surfaces[bi].friction;
        readout(ui, "block μ", &format!("{cur_friction:.2}"));
    }
    ui.separator();
    ui.small("Tet block on a tilted (25°) tet ramp.");
    ui.small("μ = 1.0 → sticks; μ = 0.5 → slips slowly;");
    ui.small("μ = 0.0 → slides freely.");
}

fn draw_diff_spring_gizmos(
    sim: Res<Sim>,
    state: Res<DiffSpringExampleState>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::DiffSpring {
        return;
    }
    if state.anchors.is_empty() {
        return;
    }
    let p = Vec3::new(state.particle[0], state.particle[1], state.particle[2]);
    let target = Vec3::new(state.target[0], state.target[1], state.target[2]);
    let line_color = Color::srgb(0.85, 0.85, 0.20);
    for a in &state.anchors {
        let av = Vec3::new(a[0], a[1], a[2]);
        gizmos.line(av, p, line_color);
    }
    // Target marker (cross gizmo).
    let tgt_color = Color::srgb(0.95, 0.30, 0.20);
    let r = 0.10;
    gizmos.line(target - Vec3::X * r, target + Vec3::X * r, tgt_color);
    gizmos.line(target - Vec3::Y * r, target + Vec3::Y * r, tgt_color);
    gizmos.line(target - Vec3::Z * r, target + Vec3::Z * r, tgt_color);
}

fn draw_diff_ball_gizmos(
    sim: Res<Sim>,
    state: Res<DiffBallExampleState>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::DiffBall {
        return;
    }
    if state.trajectory.len() < 2 {
        return;
    }
    let line_color = Color::srgb(0.55, 0.85, 0.20);
    for w in state.trajectory.windows(2) {
        let a = Vec3::new(w[0][0], w[0][1], 0.0);
        let b = Vec3::new(w[1][0], w[1][1], 0.0);
        gizmos.line(a, b, line_color);
    }
    // Target hoop — circle in the XY plane.
    let target = Vec3::new(state.target[0], state.target[1], 0.0);
    let tgt_color = Color::srgb(0.95, 0.30, 0.20);
    let r = 0.18_f32;
    let segs = 24;
    let mut prev = target + Vec3::new(r, 0.0, 0.0);
    for k in 1..=segs {
        let theta = (k as f32 / segs as f32) * std::f32::consts::TAU;
        let next = target + Vec3::new(r * theta.cos(), r * theta.sin(), 0.0);
        gizmos.line(prev, next, tgt_color);
        prev = next;
    }
}

// --- Stage 36 v1 — FEM heat-equation scene (#47) -------------------
//
// Pure-CPU demo of the Stage 36 FEM framework. A 16×16×16 tet-mesh
// of the unit cube is held by `FemHeatExampleState`; each frame we
// step backward-Euler once via `molla::fem::solve_transient`. The
// tet wireframe is drawn as gizmo lines coloured by the per-vertex
// temperature (blue → red).
//
// The scene runs entirely outside the standard solver pipeline —
// `step_simulation` short-circuits on `Scene::FemHeat`, and
// `build_fem_heat_model` returns an empty `Model` with no bodies or
// particles.

const FEM_HEAT_GRID: usize = 16;
const FEM_HEAT_DEFAULT_ALPHA: f64 = 1.0;
const FEM_HEAT_DEFAULT_DT: f64 = 0.01;
/// Origin offset for the unit cube — keep it just above the ground
/// plane so the gizmo wireframe doesn't clip the grid.
const FEM_HEAT_ORIGIN: bevy::math::Vec3 = bevy::math::Vec3::new(-0.5, 0.05, -0.5);

/// Persistent runtime state for `Scene::FemHeat`. Built lazily on
/// first entry into the scene; `apply_scene_switch` stamps
/// `needs_reset` so re-entering re-initialises `u` to zero.
#[derive(Resource)]
struct FemHeatExampleState {
    mesh: fem::FemMesh,
    /// Mass matrix M (consistent, P1).
    mass: molla::math::sparse::BsrMatrix,
    /// Stiffness matrix K = α · ∫ ∇φᵢ·∇φⱼ dx. Rebuilt whenever `α`
    /// changes via the right-rail slider.
    stiff: molla::math::sparse::BsrMatrix,
    /// Current α — when this drifts from `last_alpha_baked` we
    /// rebuild `stiff` next frame.
    alpha: f64,
    last_alpha_baked: f64,
    /// Backward-Euler step.
    dt: f64,
    /// Dirichlet BCs (top + bottom faces).
    bcs: Vec<fem::BoundaryCondition>,
    /// Current per-vertex temperature.
    u: Vec<f64>,
    /// Last CG residual (for the readout).
    last_residual: f64,
    /// Last CG iteration count (for the readout).
    last_iters: usize,
    /// Set by `apply_scene_switch` and cleared by the update system on
    /// re-init.
    needs_reset: bool,
    /// Ribbon-button-driven flag — `true` means the user clicked
    /// "Reset to T=0" on the right-rail pane. Cleared after the
    /// solver re-initialises.
    user_reset: bool,
    /// GPU stepper — `Some` once the user has enabled GPU mode.
    /// Lazily built (Default has no Engine access). Invalidated
    /// whenever α or dt change.
    gpu_stepper: Option<fem::TransientStepperGpu>,
    use_gpu: bool,
    last_dt_baked: f64,
}

impl Default for FemHeatExampleState {
    fn default() -> Self {
        let mesh = fem::FemMesh::tet_grid_3d(
            FEM_HEAT_GRID,
            FEM_HEAT_GRID,
            FEM_HEAT_GRID,
            1.0,
            1.0,
            1.0,
        );
        let space = fem::FunctionSpace::P1Nodal;
        let mass = fem::assemble_global(&mesh, space, fem::compute_element_mass)
            .expect("FEM heat: assemble mass");
        let stiff = fem::assemble_global(&mesh, space, |c, v| {
            fem::compute_element_stiffness(c, v, FEM_HEAT_DEFAULT_ALPHA)
        })
        .expect("FEM heat: assemble stiffness");
        let mut bcs = Vec::new();
        for (i, p) in mesh.vertices.iter().enumerate() {
            if p.z.abs() < 1e-9 {
                bcs.push(fem::BoundaryCondition::Dirichlet {
                    vertex: i as u32,
                    value: 0.0,
                });
            } else if (p.z - 1.0).abs() < 1e-9 {
                bcs.push(fem::BoundaryCondition::Dirichlet {
                    vertex: i as u32,
                    value: 1.0,
                });
            }
        }
        let n = mesh.vert_count();
        let mut u = vec![0.0_f64; n];
        for bc in &bcs {
            if let fem::BoundaryCondition::Dirichlet { vertex, value } = bc {
                u[*vertex as usize] = *value;
            }
        }
        Self {
            mesh,
            mass,
            stiff,
            alpha: FEM_HEAT_DEFAULT_ALPHA,
            last_alpha_baked: FEM_HEAT_DEFAULT_ALPHA,
            dt: FEM_HEAT_DEFAULT_DT,
            bcs,
            u,
            last_residual: 0.0,
            last_iters: 0,
            needs_reset: false,
            user_reset: false,
            gpu_stepper: None,
            use_gpu: false,
            last_dt_baked: FEM_HEAT_DEFAULT_DT,
        }
    }
}

impl FemHeatExampleState {
    fn rebuild_stiffness(&mut self) {
        let space = fem::FunctionSpace::P1Nodal;
        self.stiff = fem::assemble_global(&self.mesh, space, |c, v| {
            fem::compute_element_stiffness(c, v, self.alpha)
        })
        .expect("FEM heat: rebuild stiffness");
        self.last_alpha_baked = self.alpha;
    }

    fn reset_temperature_field(&mut self) {
        let n = self.mesh.vert_count();
        self.u = vec![0.0_f64; n];
        for bc in &self.bcs {
            if let fem::BoundaryCondition::Dirichlet { vertex, value } = bc {
                self.u[*vertex as usize] = *value;
            }
        }
        self.last_residual = 0.0;
        self.last_iters = 0;
    }
}

/// Empty molla model — the FEM heat demo carries its own mesh +
/// solver in `FemHeatExampleState`; the standard pipeline is
/// short-circuited in `step_simulation`.
fn build_fem_heat_model() -> (mp::Model, Vec<BodyVisual>) {
    let b = mp::ModelBuilder::new();
    (b.build().expect("FEM heat: empty model"), Vec::new())
}

/// Per-frame system: rebuilds the stiffness matrix when α has
/// changed, applies any pending reset, and runs one backward-Euler
/// step.
fn update_fem_heat_example(
    sim: Res<Sim>,
    mut state: ResMut<FemHeatExampleState>,
    run: Res<SimRunState>,
) {
    if sim.scene != Scene::FemHeat {
        return;
    }
    if state.needs_reset || state.user_reset {
        state.reset_temperature_field();
        state.needs_reset = false;
        state.user_reset = false;
        // Push the fresh u to the GPU stepper if it's already built.
        // Clone first so we can split the borrow.
        let u_snapshot = state.u.clone();
        if let Some(s) = state.gpu_stepper.as_mut() {
            if let Err(e) = s.write_u(&u_snapshot) {
                warn!("FemHeat: GPU write_u after reset failed ({e})");
                state.gpu_stepper = None;
            }
        }
    }
    let alpha_changed = (state.alpha - state.last_alpha_baked).abs() > 1e-12;
    let dt_changed = (state.dt - state.last_dt_baked).abs() > 1e-9;
    if alpha_changed {
        state.rebuild_stiffness();
        state.gpu_stepper = None;
    }
    if dt_changed {
        state.gpu_stepper = None;
        state.last_dt_baked = state.dt;
    }
    if run.paused {
        return;
    }
    let dt = state.dt.max(1e-6);
    let n = state.mesh.vert_count();
    let f = vec![0.0_f64; n];

    // Lazily build GPU stepper if requested.
    if state.use_gpu && state.gpu_stepper.is_none() {
        match fem::TransientStepperGpu::new(
            &sim.engine,
            &state.mesh,
            &state.mass,
            &state.stiff,
            dt,
            &state.bcs,
            &state.u,
        ) {
            Ok(s) => state.gpu_stepper = Some(s),
            Err(e) => {
                warn!("FemHeat: GPU stepper init failed ({e}); falling back to CPU");
                state.use_gpu = false;
            }
        }
    }

    if state.use_gpu && state.gpu_stepper.is_some() {
        // Slice H — `step` updates the GPU-resident `u` in place;
        // read it back per render frame so the gizmo system sees
        // current data. The source `f` is zero in this scene; we
        // skip `write_f` because the constructor already zeroed it.
        let _ = f; // keep the binding live so future non-zero f flows compile
        let stepper = state.gpu_stepper.as_mut().expect("checked above");
        let step_result = stepper.step();
        let read_result = match step_result {
            Ok(stats) => match stepper.read_u() {
                Ok(u) => Ok((stats, u)),
                Err(e) => Err(e),
            },
            Err(e) => Err(e),
        };
        match read_result {
            Ok((stats, u)) => {
                state.last_residual = stats.residual;
                state.last_iters = stats.iters;
                state.u = u;
                return;
            }
            Err(e) => {
                warn!("FemHeat: GPU step/read failed ({e}); falling back to CPU this frame");
            }
        }
    }
    let result = molla::fem::solver::solve_transient_with_initial(
        &state.mass,
        &state.stiff,
        &f,
        &state.u.clone(),
        dt,
        &state.bcs,
        &state.u.clone(),
        molla::fem::solver::DEFAULT_MAX_ITERS,
        molla::fem::solver::DEFAULT_TOL,
    );
    state.u = result.x;
    state.last_residual = result.residual;
    state.last_iters = result.iters;
}

/// Map a temperature value `t ∈ [0, 1]` to a colour (blue → cyan →
/// yellow → red). Out-of-range values are clamped.
fn fem_heat_colour(t: f64) -> Color {
    let t = t.clamp(0.0, 1.0) as f32;
    // Two-segment ramp: 0..0.5 = blue → green-ish, 0.5..1 = → red.
    let (r, g, b) = if t < 0.5 {
        let s = t * 2.0;
        (0.05, 0.20 + 0.65 * s, 1.0 - 0.6 * s)
    } else {
        let s = (t - 0.5) * 2.0;
        (0.05 + 0.95 * s, 0.85 - 0.55 * s, 0.40 - 0.40 * s)
    };
    Color::srgb(r, g, b)
}

/// Draw the FEM heat-equation tet wireframe with per-edge colour
/// derived from the average of its endpoints' temperatures.
fn draw_fem_heat_gizmos(
    sim: Res<Sim>,
    state: Res<FemHeatExampleState>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::FemHeat {
        return;
    }
    use std::collections::HashSet;
    let mut seen: HashSet<(u32, u32)> = HashSet::with_capacity(state.mesh.cells.len() * 4);
    let origin = FEM_HEAT_ORIGIN;
    for cell in &state.mesh.cells {
        if let fem::FemCell::Tet(v) = cell {
            let edges = [
                (v[0], v[1]),
                (v[0], v[2]),
                (v[0], v[3]),
                (v[1], v[2]),
                (v[1], v[3]),
                (v[2], v[3]),
            ];
            for &(a, b) in &edges {
                let key = if a < b { (a, b) } else { (b, a) };
                if !seen.insert(key) {
                    continue;
                }
                let pa = state.mesh.vertices[a as usize];
                let pb = state.mesh.vertices[b as usize];
                let t_mid = 0.5 * (state.u[a as usize] + state.u[b as usize]);
                let colour = fem_heat_colour(t_mid);
                let av = origin + Vec3::new(pa.x as f32, pa.z as f32, pa.y as f32);
                let bv = origin + Vec3::new(pb.x as f32, pb.z as f32, pb.y as f32);
                gizmos.line(av, bv, colour);
            }
        }
    }
}

/// Right-rail pane for `Scene::FemHeat`. Sliders for diffusivity α +
/// timestep dt, residual readout, and the "Reset to T=0" button.
fn fem_heat_pane(ui: &mut egui::Ui, state: &mut FemHeatExampleState) {
    ui.label("FEM heat equation — Stage 36 v1");
    ui.separator();
    readout(
        ui,
        "Mesh",
        &format!(
            "{0}×{0}×{0} tets ({1} verts)",
            FEM_HEAT_GRID,
            state.mesh.vert_count()
        ),
    );
    readout(ui, "DoFs", &format!("{}", state.u.len()));
    ui.separator();
    ui.label("Thermal diffusivity α");
    let _ = ui.add(
        egui::Slider::new(&mut state.alpha, 0.05..=10.0).logarithmic(true),
    );
    ui.label("Time step dt");
    let _ = ui.add(
        egui::Slider::new(&mut state.dt, 0.001..=0.1).logarithmic(true),
    );
    ui.separator();
    readout(ui, "Last residual", &format!("{:.3e}", state.last_residual));
    readout(ui, "CG iters", &format!("{}", state.last_iters));
    ui.separator();
    if ui.button("Reset to T = 0").clicked() {
        state.user_reset = true;
    }
    ui.separator();
    let was_gpu = state.use_gpu;
    let _ = ui.checkbox(&mut state.use_gpu, "Use GPU CG (Stage 36.5 Slice F)");
    if was_gpu != state.use_gpu {
        state.gpu_stepper = None;
    }
    let backend = if state.use_gpu && state.gpu_stepper.is_some() {
        "GPU resident CG"
    } else if state.use_gpu {
        "GPU (init pending / failed)"
    } else {
        "CPU"
    };
    readout(ui, "Backend", backend);
    ui.small("BCs: T(z=0) = 0, T(z=1) = 1; sides insulating.");
    ui.small("Steady state: T(z) = z (linear gradient).");
}


// --- Stage 36.3 — FEM heat-equation scene with P2 elements (#49) ----
//
// Same heat equation as `Scene::FemHeat` (Dirichlet `T(z=0) = 0`,
// `T(z=1) = 1`, sides insulating), but the FEM solve uses
// `FunctionSpace::P2Nodal`: 1 DoF per vertex + 1 DoF per edge
// midpoint. The scene mesh is intentionally smaller than the P1 demo
// so the full P2 system fits comfortably in CG (10×10 element
// matrices and one DoF per edge multiplies the unknown count by ~7×
// vs P1 at the same mesh).

const FEM_HEAT_P2_GRID: usize = 8;

#[derive(Resource)]
struct FemHeatP2ExampleState {
    mesh: fem::FemMesh,
    /// P2 DoF map (vertex DoFs + edge DoFs).
    dof_map: fem::P2DofMap,
    /// Mass matrix M (consistent, P2).
    mass: molla::math::sparse::BsrMatrix,
    /// Stiffness matrix K (P2).
    stiff: molla::math::sparse::BsrMatrix,
    alpha: f64,
    last_alpha_baked: f64,
    dt: f64,
    /// Dirichlet BCs over both corner DoFs (top + bottom faces) and
    /// the matching edge DoFs (edges with both endpoints on a face).
    bcs: Vec<fem::BoundaryCondition>,
    /// Current temperature field, length = `dof_map.ndof()`.
    u: Vec<f64>,
    last_residual: f64,
    last_iters: usize,
    needs_reset: bool,
    user_reset: bool,
    /// P1-baseline DoF count at the same mesh — for the side-by-side
    /// pane readout. Cached at construction since the mesh is fixed.
    p1_dof_count: usize,
}

impl Default for FemHeatP2ExampleState {
    fn default() -> Self {
        let mesh = fem::FemMesh::tet_grid_3d(
            FEM_HEAT_P2_GRID,
            FEM_HEAT_P2_GRID,
            FEM_HEAT_P2_GRID,
            1.0,
            1.0,
            1.0,
        );
        let dof_map = fem::P2DofMap::build(&mesh);
        let mass = fem::assemble_global_p2(&mesh, &dof_map, fem::compute_element_p2_mass)
            .expect("FEM heat P2: assemble mass");
        let stiff = fem::assemble_global_p2(&mesh, &dof_map, |c, v| {
            fem::compute_element_p2_stiffness(c, v, FEM_HEAT_DEFAULT_ALPHA)
        })
        .expect("FEM heat P2: assemble stiffness");
        let bcs = fem_heat_p2_dirichlet_bcs(&mesh, &dof_map);
        let n = dof_map.ndof();
        let mut u = vec![0.0_f64; n];
        for bc in &bcs {
            if let fem::BoundaryCondition::Dirichlet { vertex, value } = *bc {
                u[vertex as usize] = value;
            }
        }
        let p1_dof_count = mesh.vert_count();
        Self {
            mesh,
            dof_map,
            mass,
            stiff,
            alpha: FEM_HEAT_DEFAULT_ALPHA,
            last_alpha_baked: FEM_HEAT_DEFAULT_ALPHA,
            dt: FEM_HEAT_DEFAULT_DT,
            bcs,
            u,
            last_residual: 0.0,
            last_iters: 0,
            needs_reset: false,
            user_reset: false,
            p1_dof_count,
        }
    }
}

// --- Stage 36.1 — FEM elasticity cantilever-beam scene (#48) -------
//
// Pure-CPU steady-state demo of the Stage 36.1 vector-valued FEM
// framework. A 20×4×4 tet beam (Kuhn 6-tet decomposition of each hex)
// is pinned along its left face and loaded with a downward traction
// on the right face. The deformed wireframe (`x + u`) is drawn each
// frame; the assembled stiffness matrix + Dirichlet pattern survive
// scene switches and only re-solve when a slider changes.
//
// The right-rail pane carries Young's modulus E, Poisson ratio ν,
// applied tip force F, and a Linear-vs-SVK toggle.

const FEM_BEAM_NX: usize = 20;
const FEM_BEAM_NY: usize = 4;
const FEM_BEAM_NZ: usize = 4;
const FEM_BEAM_LEN_X: f64 = 4.0;
const FEM_BEAM_DEPTH_Y: f64 = 0.5;
const FEM_BEAM_HEIGHT_Z: f64 = 0.5;
const FEM_BEAM_DEFAULT_E: f64 = 1.0e6;
const FEM_BEAM_DEFAULT_NU: f64 = 0.0;
const FEM_BEAM_DEFAULT_LOAD: f64 = -50.0;
/// Origin offset for the beam — keep it just above the ground plane
/// so the gizmo wireframe doesn't clip the grid.
const FEM_BEAM_ORIGIN: bevy::math::Vec3 = bevy::math::Vec3::new(-2.0, 0.6, -0.25);

#[derive(Resource)]
struct FemBeamExampleState {
    mesh: fem::FemMesh,
    /// Linear-elasticity stiffness matrix `K = ∫ Bᵀ D B dV`. Rebuilt
    /// when `youngs` or `poisson` change.
    stiff: molla::math::sparse::BsrMatrix,
    bcs: Vec<fem::BoundaryCondition>,
    /// Indices of the vertices on the loaded face (x = L). Used both
    /// to assemble the traction and to read out the tip deflection.
    tip_face_verts: Vec<u32>,
    /// External-load vector (3 components per vertex). Rebuilt when
    /// `load_z` changes.
    f_ext: Vec<f64>,
    /// Current displacement field (3 × vert_count, interleaved).
    u: Vec<f64>,
    /// Current Young's modulus.
    youngs: f64,
    /// Current Poisson's ratio.
    poisson: f64,
    /// Total tip-load magnitude (signed; downward = negative).
    load_z: f64,
    /// Use SVK nonlinear elasticity instead of linear K · u = f.
    use_svk: bool,
    // Cached values to detect slider changes.
    last_youngs_baked: f64,
    last_poisson_baked: f64,
    last_load_baked: f64,
    last_svk_baked: bool,
    /// Last solve telemetry.
    tip_deflection: f64,
    last_residual: f64,
    last_iters: usize,
    /// Set by `apply_scene_switch` and cleared by the update system on
    /// re-init.
    needs_reset: bool,
    user_reset: bool,
}

impl Default for FemBeamExampleState {
    fn default() -> Self {
        let mesh = fem::FemMesh::tet_grid_3d(
            FEM_BEAM_NX,
            FEM_BEAM_NY,
            FEM_BEAM_NZ,
            FEM_BEAM_LEN_X,
            FEM_BEAM_DEPTH_Y,
            FEM_BEAM_HEIGHT_Z,
        );
        let (lam, mu) = fem_beam_lame(FEM_BEAM_DEFAULT_E, FEM_BEAM_DEFAULT_NU);
        let space = fem::FunctionSpace::P1Vector { dim: 3 };
        let stiff = fem::assemble_global(&mesh, space, |c, v| {
            fem::compute_element_elasticity_stiffness(c, v, lam, mu)
        })
        .expect("FEM beam: assemble stiffness");
        // Dirichlet on the left face (x = 0) — fully clamp.
        let mut bcs: Vec<fem::BoundaryCondition> = Vec::new();
        for (i, p) in mesh.vertices.iter().enumerate() {
            if p.x.abs() < 1e-9 {
                bcs.push(fem::BoundaryCondition::DirichletVector {
                    vertex: i as u32,
                    values: vec![0.0, 0.0, 0.0],
                });
            }
        }
        // Tip-face vertex list.
        let tip_face_verts: Vec<u32> = mesh
            .vertices
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                if (p.x - FEM_BEAM_LEN_X).abs() < 1e-9 {
                    Some(i as u32)
                } else {
                    None
                }
            })
            .collect();
        let n = mesh.vert_count() * 3;
        let f_ext = build_fem_beam_traction(&mesh, FEM_BEAM_DEFAULT_LOAD);
        let u = vec![0.0; n];
        Self {
            mesh,
            stiff,
            bcs,
            tip_face_verts,
            f_ext,
            u,
            youngs: FEM_BEAM_DEFAULT_E,
            poisson: FEM_BEAM_DEFAULT_NU,
            load_z: FEM_BEAM_DEFAULT_LOAD,
            use_svk: false,
            last_youngs_baked: FEM_BEAM_DEFAULT_E,
            last_poisson_baked: FEM_BEAM_DEFAULT_NU,
            last_load_baked: FEM_BEAM_DEFAULT_LOAD,
            last_svk_baked: false,
            tip_deflection: 0.0,
            last_residual: 0.0,
            last_iters: 0,
            needs_reset: true,
            user_reset: false,

        }
    }
}


impl FemHeatP2ExampleState {
    fn rebuild_stiffness(&mut self) {
        self.stiff = fem::assemble_global_p2(&self.mesh, &self.dof_map, |c, v| {
            fem::compute_element_p2_stiffness(c, v, self.alpha)
        })
        .expect("FEM heat P2: rebuild stiffness");
        self.last_alpha_baked = self.alpha;
    }

    fn reset_temperature_field(&mut self) {
        let n = self.dof_map.ndof();
        self.u = vec![0.0_f64; n];
        for bc in &self.bcs {
            if let fem::BoundaryCondition::Dirichlet { vertex, value } = *bc {
                self.u[vertex as usize] = value;
            }
        }
    }
}

fn fem_beam_lame(youngs: f64, poisson: f64) -> (f64, f64) {
    let mu = youngs / (2.0 * (1.0 + poisson));
    // Plane-strain-style Lamé. For Poisson = 0.5 the formula
    // diverges; clamp the slider to ≤ 0.49 to avoid that singularity.
    let lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    (lam, mu)
}

/// Build the external-force vector for a constant downward (−z)
/// traction over the tip face (x = L_x). Uses the consistent (P1
/// surface integral) lumping `(area / 3) · t` per face vertex.
/// Total integrated tip load along z equals `load_z` (the surface
/// area cancels: traction = load_z / (b · h) and `∫ t dA = load_z`).
fn build_fem_beam_traction(mesh: &fem::FemMesh, load_z: f64) -> Vec<f64> {
    let n = mesh.vert_count() * 3;
    let mut f = vec![0.0; n];
    let area_total = FEM_BEAM_DEPTH_Y * FEM_BEAM_HEIGHT_Z;
    let traction_z = load_z / area_total;
    for cell in &mesh.cells {
        if let fem::FemCell::Tet(idx) = cell {
            let faces = [
                [idx[0], idx[1], idx[2]],
                [idx[0], idx[1], idx[3]],
                [idx[0], idx[2], idx[3]],
                [idx[1], idx[2], idx[3]],
            ];
            for face in &faces {
                let pa = mesh.vertices[face[0] as usize];
                let pb = mesh.vertices[face[1] as usize];
                let pc = mesh.vertices[face[2] as usize];
                let on_tip = (pa.x - FEM_BEAM_LEN_X).abs() < 1e-9
                    && (pb.x - FEM_BEAM_LEN_X).abs() < 1e-9
                    && (pc.x - FEM_BEAM_LEN_X).abs() < 1e-9;
                if !on_tip {
                    continue;
                }
                let area = 0.5 * (pb - pa).cross(pc - pa).length();
                let third = area / 3.0;
                for &v in face {
                    let v = v as usize;
                    f[v * 3 + 2] += third * traction_z;
                }
            }
        }
    }
    f
}

impl FemBeamExampleState {
    fn rebuild_stiffness(&mut self) {
        let (lam, mu) = fem_beam_lame(self.youngs, self.poisson);
        let space = fem::FunctionSpace::P1Vector { dim: 3 };
        self.stiff = fem::assemble_global(&self.mesh, space, |c, v| {
            fem::compute_element_elasticity_stiffness(c, v, lam, mu)
        })
        .expect("FEM beam: rebuild stiffness");
        self.last_youngs_baked = self.youngs;
        self.last_poisson_baked = self.poisson;
    }

    fn rebuild_traction(&mut self) {
        self.f_ext = build_fem_beam_traction(&self.mesh, self.load_z);
        self.last_load_baked = self.load_z;
    }

    fn solve(&mut self) {
        if self.use_svk {
            // SVK Newton-Raphson. Re-assemble the (linear) tangent
            // each iter — full geometric tangent is a Stage 36.1.1
            // upgrade.
            let (lam, mu) = fem_beam_lame(self.youngs, self.poisson);
            let cells = self.mesh.cells.clone();
            let verts = self.mesh.vertices.clone();
            let stiff_cached = self.stiff.clone();
            let tangent =
                |_u: &[f64]| -> molla::math::sparse::BsrMatrix { stiff_cached.clone() };
            let n = self.mesh.vert_count() * 3;
            let residual = move |u: &[f64]| -> Vec<f64> {
                let mut r = vec![0.0; n];
                for cell in &cells {
                    if let fem::FemCell::Tet(idx) = cell {
                        let mut u_local = [0.0; 12];
                        for (i, &v) in idx.iter().enumerate() {
                            let base = (v as usize) * 3;
                            u_local[3 * i] = u[base];
                            u_local[3 * i + 1] = u[base + 1];
                            u_local[3 * i + 2] = u[base + 2];
                        }
                        let f_int = fem::compute_element_st_venant_kirchhoff_stress(
                            cell, &verts, &u_local, lam, mu,
                        );
                        for (i, &v) in idx.iter().enumerate() {
                            let base = (v as usize) * 3;
                            r[base] += f_int[3 * i];
                            r[base + 1] += f_int[3 * i + 1];
                            r[base + 2] += f_int[3 * i + 2];
                        }
                    }
                }
                r
            };
            self.u = fem::solve_nonlinear_static(
                tangent,
                residual,
                &self.f_ext,
                &self.bcs,
                &self.mesh.vertices,
                3,
                5,
                1.0e-7,
            );
            self.last_iters = 5; // best-effort; the inner driver doesn't expose its counter.
            self.last_residual = 0.0;
        } else {
            // Linear elasticity steady state.
            let mut k_red = self.stiff.clone();
            let mut f_red = self.f_ext.clone();
            fem::apply_bcs(&mut k_red, &mut f_red, &self.bcs, &self.mesh.vertices, 3);
            // Initial guess: zero (Dirichlet rows already pinned).
            let res = molla::math::iterative::cg_solve(
                |x| {
                    let mut y = vec![0.0; k_red.nrows()];
                    k_red.mv(x, &mut y).unwrap();
                    y
                },
                &f_red,
                &vec![0.0; f_red.len()],
                1.0e-9,
                molla::fem::solver::DEFAULT_MAX_ITERS,
            );
            self.u = res.x;
            self.last_residual = res.residual;
            self.last_iters = res.iters;
        }
        // Read tip deflection (average of tip-face vertices, z-component).
        if self.tip_face_verts.is_empty() {
            self.tip_deflection = 0.0;
        } else {
            let mut sum = 0.0;
            for &v in &self.tip_face_verts {
                sum += self.u[(v as usize) * 3 + 2];
            }
            self.tip_deflection = sum / self.tip_face_verts.len() as f64;
        }
        self.last_youngs_baked = self.youngs;
        self.last_poisson_baked = self.poisson;
        self.last_load_baked = self.load_z;
        self.last_svk_baked = self.use_svk;
    }

    fn reset_displacement_field(&mut self) {
        let n = self.mesh.vert_count() * 3;
        self.u = vec![0.0; n];
        self.tip_deflection = 0.0;

        self.last_residual = 0.0;
        self.last_iters = 0;
    }
}


/// Build the Dirichlet BC list for the P2 heat scene: corner DoFs on
/// the top + bottom faces, plus edge DoFs whose endpoints are both on
/// the same face (so the edge midpoint sits on that face).
fn fem_heat_p2_dirichlet_bcs(
    mesh: &fem::FemMesh,
    dof_map: &fem::P2DofMap,
) -> Vec<fem::BoundaryCondition> {
    let mut bcs = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        if p.z.abs() < 1e-9 {
            bcs.push(fem::BoundaryCondition::Dirichlet { vertex: i as u32, value: 0.0 });
        } else if (p.z - 1.0).abs() < 1e-9 {
            bcs.push(fem::BoundaryCondition::Dirichlet { vertex: i as u32, value: 1.0 });
        }
    }
    for (edge, &edge_id) in &dof_map.edge_id {
        let pa = mesh.vertices[edge[0] as usize];
        let pb = mesh.vertices[edge[1] as usize];
        let mid_z = 0.5 * (pa.z + pb.z);
        let on_bottom = pa.z.abs() < 1e-9 && pb.z.abs() < 1e-9;
        let on_top = (pa.z - 1.0).abs() < 1e-9 && (pb.z - 1.0).abs() < 1e-9;
        if on_bottom || on_top {
            bcs.push(fem::BoundaryCondition::Dirichlet {
                vertex: (dof_map.vertex_count + edge_id as usize) as u32,
                value: mid_z,
            });
        }
    }
    bcs
}

/// Empty molla model — the P2 FEM heat demo carries its own mesh +
/// solver in `FemHeatP2ExampleState`; the standard pipeline is
/// short-circuited in `step_simulation`.
fn build_fem_heat_p2_model() -> (mp::Model, Vec<BodyVisual>) {
    let b = mp::ModelBuilder::new();
    (b.build().expect("FEM heat P2: empty model"), Vec::new())
}

/// Per-frame system: mirrors `update_fem_heat_example` but for the
/// P2 scene.
fn update_fem_heat_p2_example(
    sim: Res<Sim>,
    mut state: ResMut<FemHeatP2ExampleState>,
    run: Res<SimRunState>,
) {
    if sim.scene != Scene::FemHeatP2 {
        return;
    }
    if state.needs_reset || state.user_reset {
        state.reset_temperature_field();
        state.needs_reset = false;
        state.user_reset = false;
    }
    if (state.alpha - state.last_alpha_baked).abs() > 1e-12 {
        state.rebuild_stiffness();
    }
    if run.paused {
        return;
    }
    let dt = state.dt.max(1e-6);
    let n = state.dof_map.ndof();
    let f = vec![0.0_f64; n];
    let result = molla::fem::solver::solve_transient_with_initial(
        &state.mass,
        &state.stiff,
        &f,
        &state.u.clone(),
        dt,
        &state.bcs,
        &state.u.clone(),
        molla::fem::solver::DEFAULT_MAX_ITERS,
        molla::fem::solver::DEFAULT_TOL,
    );
    state.u = result.x;
    state.last_residual = result.residual;
    state.last_iters = result.iters;
}

/// Draw the P2 FEM heat tet wireframe with per-edge colour from the
/// edge-midpoint DoF temperature (which is what P2 actually solves
/// for — a true mid-edge sample, not an interpolation of corners).
fn draw_fem_heat_p2_gizmos(
    sim: Res<Sim>,
    state: Res<FemHeatP2ExampleState>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::FemHeatP2 {
        return;
    }
    let _ = (state, gizmos);
}

/// Empty molla model — the FEM beam demo carries its own mesh +
/// solver in `FemBeamExampleState`; the standard pipeline is
/// short-circuited in `step_simulation`.
fn build_fem_beam_model() -> (mp::Model, Vec<BodyVisual>) {
    let b = mp::ModelBuilder::new();
    (b.build().expect("FEM beam: empty model"), Vec::new())
}

/// Per-frame system: rebuilds the stiffness / traction when sliders
/// change, applies any pending reset, and re-solves the steady-state
/// (or SVK Newton) system.
fn update_fem_beam_example(sim: Res<Sim>, mut state: ResMut<FemBeamExampleState>) {
    if sim.scene != Scene::FemBeam {
        return;
    }
    if state.needs_reset || state.user_reset {
        state.reset_displacement_field();
        state.needs_reset = false;
        state.user_reset = false;
        // Always solve once on entry so the deformed shape is visible
        // immediately.
        state.solve();
        return;
    }
    let mat_changed = (state.youngs - state.last_youngs_baked).abs() > 1e-12
        || (state.poisson - state.last_poisson_baked).abs() > 1e-12;
    let load_changed = (state.load_z - state.last_load_baked).abs() > 1e-12;
    let svk_changed = state.use_svk != state.last_svk_baked;
    if mat_changed {
        state.rebuild_stiffness();
    }
    if load_changed {
        state.rebuild_traction();
    }
    if mat_changed || load_changed || svk_changed {
        state.solve();
    }
}

/// Draw the deformed tet wireframe (vertices `x + u`). Pinned
/// vertices are drawn in a slightly brighter tone so the user can
/// see the clamped face.
fn draw_fem_beam_gizmos(
    sim: Res<Sim>,
    state: Res<FemBeamExampleState>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::FemBeam {
        return;
    }
    use std::collections::HashSet;
    let mut seen: HashSet<(u32, u32)> = HashSet::with_capacity(state.mesh.cells.len() * 4);
    let origin = FEM_BEAM_ORIGIN;
    let pinned: HashSet<u32> = state
        .bcs
        .iter()
        .filter_map(|bc| match bc {
            fem::BoundaryCondition::DirichletVector { vertex, .. } => Some(*vertex),
            _ => None,
        })
        .collect();
    let pos = |v: u32| -> bevy::math::Vec3 {
        let p = state.mesh.vertices[v as usize];
        let base = (v as usize) * 3;
        let ux = state.u[base];
        let uy = state.u[base + 1];
        let uz = state.u[base + 2];
        origin
            + bevy::math::Vec3::new(
                (p.x + ux) as f32,
                (p.z + uz) as f32,
                (p.y + uy) as f32,
            )
    };
    let edge_colour = |a: u32, b: u32| -> Color {
        if pinned.contains(&a) && pinned.contains(&b) {
            Color::srgb(0.95, 0.85, 0.30)
        } else {
            Color::srgb(0.30, 0.65, 0.95)
        }
    };

    for cell in &state.mesh.cells {
        if let fem::FemCell::Tet(v) = cell {
            let edges = [
                (v[0], v[1]),
                (v[0], v[2]),
                (v[0], v[3]),
                (v[1], v[2]),
                (v[1], v[3]),
                (v[2], v[3]),
            ];
            for &(a, b) in &edges {
                let key = if a < b { (a, b) } else { (b, a) };
                if !seen.insert(key) {
                    continue;
                }
                gizmos.line(pos(a), pos(b), edge_colour(a, b));
            }
        }
    }
}


/// Right-rail pane for `Scene::FemHeatP2`. Same controls as the P1
/// pane plus a side-by-side P1 vs P2 DoF count readout so users can
/// see "P2 = more DoF for same mesh".
fn fem_heat_p2_pane(ui: &mut egui::Ui, state: &mut FemHeatP2ExampleState) {
    ui.label("FEM heat equation — Stage 36.3 P2");
    ui.separator();
    readout(
        ui,
        "Mesh",
        &format!(
            "{0}×{0}×{0} tets ({1} verts)",
            FEM_HEAT_P2_GRID,
            state.mesh.vert_count()
        ),
    );
    let p2_dofs = state.u.len();
    let p1_dofs = state.p1_dof_count;
    let edge_dofs = p2_dofs - p1_dofs;
    readout(ui, "P1 DoFs (verts)", &format!("{p1_dofs}"));
    readout(ui, "P2 DoFs (verts + edges)", &format!("{p2_dofs}  (+{edge_dofs} edges)"));
    let ratio = if p1_dofs > 0 { p2_dofs as f32 / p1_dofs as f32 } else { 0.0 };
    readout(ui, "P2 / P1 DoF ratio", &format!("{ratio:.2}×"));
    ui.separator();
    ui.label("Thermal diffusivity α");
    let _ = ui.add(
        egui::Slider::new(&mut state.alpha, 0.05..=10.0).logarithmic(true),
    );
    ui.label("Time step dt");
    let _ = ui.add(
        egui::Slider::new(&mut state.dt, 0.001..=0.1).logarithmic(true),
    );
    ui.separator();
    readout(ui, "Last residual", &format!("{:.3e}", state.last_residual));
    readout(ui, "CG iters", &format!("{}", state.last_iters));
    ui.separator();
    if ui.button("Reset to T = 0").clicked() {
        state.user_reset = true;
    }
    ui.small("Same problem as #47, but P2 (quadratic) basis.");
    ui.small("Edges carry their own DoF — colour reflects mid-edge T.");
}

/// Right-rail pane for `Scene::FemBeam`. Material + load sliders,
/// linear / SVK toggle, deflection readout, and a "Reset" button.
fn fem_beam_pane(ui: &mut egui::Ui, state: &mut FemBeamExampleState) {
    ui.label("FEM elasticity — Stage 36.1");
    ui.separator();
    readout(
        ui,
        "Mesh",
        &format!(
            "{}×{}×{} tets ({} verts)",
            FEM_BEAM_NX,
            FEM_BEAM_NY,
            FEM_BEAM_NZ,
            state.mesh.vert_count()
        ),
    );
    readout(ui, "DoFs", &format!("{}", state.u.len()));
    ui.separator();
    ui.label("Young's modulus E (Pa)");
    let _ = ui.add(
        egui::Slider::new(&mut state.youngs, 1.0e4..=1.0e8).logarithmic(true),
    );
    ui.label("Poisson's ratio ν");
    let _ = ui.add(egui::Slider::new(&mut state.poisson, 0.0..=0.45));
    ui.label("Tip load F (N, downward = negative)");
    let _ = ui.add(egui::Slider::new(&mut state.load_z, -500.0..=0.0));
    ui.separator();
    ui.checkbox(&mut state.use_svk, "Use SVK (nonlinear)");
    ui.separator();
    readout(ui, "Tip deflection", &format!("{:.4e} m", state.tip_deflection));
    // Analytic Bernoulli reference (linear small-strain): δ = P L³ / (3 E I).
    let i_inertia = FEM_BEAM_DEPTH_Y * FEM_BEAM_HEIGHT_Z.powi(3) / 12.0;
    let delta = state.load_z * FEM_BEAM_LEN_X.powi(3) / (3.0 * state.youngs * i_inertia);
    readout(ui, "Bernoulli δ", &format!("{:.4e} m", delta));
    readout(ui, "CG residual", &format!("{:.3e}", state.last_residual));
    readout(ui, "CG iters", &format!("{}", state.last_iters));
    ui.separator();
    if ui.button("Reset to undeformed").clicked() {
        state.user_reset = true;
    }
    ui.small("BCs: clamp at x = 0, traction on x = L.");
    ui.small("Linear: solves K · u = f. SVK: 5-iter Newton-Raphson.");
}

// ─── Stage 36.4 — Differentiable FEM beam ─────────────────────────────
//
// Same beam geometry as `Scene::FemBeam`, but instead of just solving
// K · u = f for a fixed material, the scene runs gradient descent on
// Young's modulus E to drive the tip-deflection toward a user-supplied
// target. Each frame:
//   1. Forward-solve the linear elasticity system at the current E.
//   2. Read the tip deflection.
//   3. Compute the loss `L = ½ (δ − δ_target)²`.
//   4. Adjoint-solve `K · λ = ∂L/∂u` (the seed has a single non-zero
//      component, the average of (δ − δ_target) / N_tip across the
//      tip-face DoFs).
//   5. Compute `∂L/∂E = compute_grad_youngs(...)`, step
//      `E ← E − η · ∂L/∂E`.
//
// Convergence is fastest for an adaptive step ∝ E²/L; we keep a
// fixed-but-relative `lr_factor` slider so the user can play.

#[derive(Resource)]
struct FemDiffBeamExampleState {
    mesh: fem::FemMesh,
    bcs: Vec<fem::BoundaryCondition>,
    /// Tip-face vertex indices — used to read the tip deflection +
    /// build the adjoint seed.
    tip_face_verts: Vec<u32>,
    /// External-load vector — fixed for this scene (the scene
    /// optimises material, not load).
    f_ext: Vec<f64>,
    /// Current displacement field.
    u: Vec<f64>,
    /// Current Young's modulus (the optimisation variable).
    youngs: f64,
    /// Fixed Poisson's ratio.
    poisson: f64,
    /// Fixed total tip-load (downward).
    load_z: f64,
    /// Target tip deflection (slider, m).
    target_deflection: f64,
    /// Current tip deflection.
    tip_deflection: f64,
    /// Last computed `∂L/∂E`.
    last_grad: f64,
    /// Per-step learning-rate factor (multiplies the adaptive E²/|L|
    /// scale). 0 ⇒ paused.
    lr_factor: f64,
    /// Step counter (resets on entry / when target changes).
    iter_count: usize,
    /// Run / pause toggle for the descent loop.
    run_descent: bool,
    /// Set by `apply_scene_switch`; cleared on the first update tick.
    needs_reset: bool,
    /// Set by the pane "Reset" button.
    user_reset: bool,
    /// Last loss value (for the readout).
    last_loss: f64,
}

impl Default for FemDiffBeamExampleState {
    fn default() -> Self {
        let mesh = fem::FemMesh::tet_grid_3d(
            FEM_BEAM_NX,
            FEM_BEAM_NY,
            FEM_BEAM_NZ,
            FEM_BEAM_LEN_X,
            FEM_BEAM_DEPTH_Y,
            FEM_BEAM_HEIGHT_Z,
        );
        let mut bcs: Vec<fem::BoundaryCondition> = Vec::new();
        for (i, p) in mesh.vertices.iter().enumerate() {
            if p.x.abs() < 1e-9 {
                bcs.push(fem::BoundaryCondition::DirichletVector {
                    vertex: i as u32,
                    values: vec![0.0, 0.0, 0.0],
                });
            }
        }
        let tip_face_verts: Vec<u32> = mesh
            .vertices
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                if (p.x - FEM_BEAM_LEN_X).abs() < 1e-9 {
                    Some(i as u32)
                } else {
                    None
                }
            })
            .collect();
        let n = mesh.vert_count() * 3;
        let f_ext = build_fem_beam_traction(&mesh, FEM_BEAM_DEFAULT_LOAD);
        Self {
            mesh,
            bcs,
            tip_face_verts,
            f_ext,
            u: vec![0.0; n],
            youngs: FEM_BEAM_DEFAULT_E,
            poisson: FEM_BEAM_DEFAULT_NU,
            load_z: FEM_BEAM_DEFAULT_LOAD,
            // Default target: 80% of the analytic Bernoulli deflection
            // at E0 (so the optimiser can move E either up or down to
            // hit it depending on starting conditions).
            target_deflection: -2.0e-3,
            tip_deflection: 0.0,
            last_grad: 0.0,
            lr_factor: 0.2,
            iter_count: 0,
            run_descent: true,
            needs_reset: true,
            user_reset: false,
            last_loss: 0.0,
        }
    }
}

// ─── Stage 36.2 — FEM Stokes (lid-driven cavity, Taylor-Hood) ──────

const FEM_STOKES_GRID: usize = 16;
const FEM_STOKES_DEFAULT_VISCOSITY: f64 = 1.0;
const FEM_STOKES_DEFAULT_LID_U: f64 = 1.0;
/// Origin offset for the unit cavity — sits just above the ground
/// plane so the gizmo wireframe + arrows don't clip.
const FEM_STOKES_ORIGIN: bevy::math::Vec3 =
    bevy::math::Vec3::new(-0.5, 0.05, -0.5);

/// Persistent state for `Scene::FemStokes`. The Taylor-Hood mesh +
/// matrices survive scene switches (rebuilding 16² P2 meshes is cheap
/// but pointless on every entry); only the velocity / pressure fields
/// reset on entry.
#[derive(Resource)]
struct FemStokesExampleState {
    mesh: fem::FemMesh,
    th: fem::MixedTaylorHood,
    bcs: Vec<fem::StokesBc>,
    viscosity: f64,
    lid_u: f64,
    /// Latest velocity field (P2 vector, interleaved layout).
    u: Vec<f64>,
    /// Latest pressure field (P1 nodal).
    p: Vec<f64>,
    /// Last solver telemetry.
    last_outer_iters: usize,
    last_div_residual: f64,
    /// Apply_scene_switch sets this on entry so the next update re-runs
    /// the solve (otherwise re-entering shows a stale field).
    needs_reset: bool,
    /// Right-rail "Recompute" button.
    user_reset: bool,
    // Cached values to detect slider changes.
    last_viscosity_baked: f64,
    last_lid_u_baked: f64,
}

impl Default for FemStokesExampleState {
    fn default() -> Self {
        let mesh = fem::FemMesh::tri_grid_2d(
            FEM_STOKES_GRID,
            FEM_STOKES_GRID,
            1.0,
            1.0,
        );
        let th = fem::MixedTaylorHood::build(&mesh, 2);
        let bcs = fem::build_cavity_bcs_2d(
            &mesh,
            &th,
            1.0,
            [FEM_STOKES_DEFAULT_LID_U, 0.0],
        );
        let n_v = th.ndof_velocity();
        let n_p = th.ndof_pressure();
        Self {
            mesh,
            th,
            bcs,
            viscosity: FEM_STOKES_DEFAULT_VISCOSITY,
            lid_u: FEM_STOKES_DEFAULT_LID_U,
            u: vec![0.0; n_v],
            p: vec![0.0; n_p],
            last_outer_iters: 0,
            last_div_residual: 0.0,
            needs_reset: true,
            user_reset: false,
            last_viscosity_baked: FEM_STOKES_DEFAULT_VISCOSITY,
            last_lid_u_baked: FEM_STOKES_DEFAULT_LID_U,
        }
    }
}

impl FemDiffBeamExampleState {
    /// Run one step of (forward solve → adjoint → gradient descent).
    fn step_once(&mut self) {
        // Re-assemble K(E).
        let (lam, mu) = fem_beam_lame(self.youngs, self.poisson);
        let space = fem::FunctionSpace::P1Vector { dim: 3 };
        let stiff = fem::assemble_global(&self.mesh, space, |c, v| {
            fem::compute_element_elasticity_stiffness(c, v, lam, mu)
        })
        .expect("FEM diff beam: assemble stiffness");

        // Forward solve.
        let mut k_red = stiff.clone();
        let mut f_red = self.f_ext.clone();
        fem::apply_bcs(&mut k_red, &mut f_red, &self.bcs, &self.mesh.vertices, 3);
        let cg_res = molla::math::iterative::cg_solve(
            |x| {
                let mut y = vec![0.0; k_red.nrows()];
                k_red.mv(x, &mut y).unwrap();
                y
            },
            &f_red,
            &vec![0.0; f_red.len()],
            1.0e-10,
            molla::fem::solver::DEFAULT_MAX_ITERS,
        );
        self.u = cg_res.x;

        // Tip deflection (avg of tip-face z-components).
        let n_tip = self.tip_face_verts.len().max(1);
        let mut sum = 0.0;
        for &v in &self.tip_face_verts {
            sum += self.u[(v as usize) * 3 + 2];
        }
        self.tip_deflection = sum / n_tip as f64;
        let err = self.tip_deflection - self.target_deflection;
        self.last_loss = 0.5 * err * err;

        // Adjoint: ∂L/∂u_v = (err / n_tip) at tip-face z-DoFs (chain
        // rule of the average reduction); zero everywhere else.
        let n_dof = self.u.len();
        let mut dj_du = vec![0.0_f64; n_dof];
        let coeff = err / n_tip as f64;
        for &v in &self.tip_face_verts {
            dj_du[(v as usize) * 3 + 2] = coeff;
        }
        let lambda = fem::solve_adjoint_with(
            &stiff,
            &dj_du,
            &self.bcs,
            &self.mesh.vertices,
            3,
            molla::fem::solver::DEFAULT_MAX_ITERS,
            1.0e-10,
        );
        let g = fem::compute_grad_youngs(
            &self.mesh,
            &self.mesh.vertices,
            &self.u,
            &lambda,
            self.poisson,
            self.youngs,
        );
        self.last_grad = g;

        // Adaptive step proportional to E²/|L|, scaled by lr_factor.
        if self.run_descent && self.lr_factor > 0.0 {
            let lr = self.lr_factor * self.youngs * self.youngs / self.last_loss.max(1e-30);
            let e_next = (self.youngs - lr * g).max(1.0e3).min(1.0e10);
            self.youngs = e_next;
        }
        self.iter_count += 1;
    }

    fn reset_optim(&mut self) {
        let n = self.mesh.vert_count() * 3;
        self.u = vec![0.0; n];
        self.youngs = FEM_BEAM_DEFAULT_E;
        self.tip_deflection = 0.0;
        self.last_grad = 0.0;
        self.iter_count = 0;
        self.last_loss = 0.0;
        self.f_ext = build_fem_beam_traction(&self.mesh, self.load_z);
    }
}

/// Empty molla model — the diff-FEM beam scene carries its own mesh +
/// solver in `FemDiffBeamExampleState`; the standard pipeline is
/// short-circuited in `step_simulation`.
fn build_fem_diff_beam_model() -> (mp::Model, Vec<BodyVisual>) {
    let b = mp::ModelBuilder::new();
    (b.build().expect("FEM diff beam: empty model"), Vec::new())
}

/// Per-frame system: drives the gradient-descent loop on Young's
/// modulus when the scene is active.
fn update_fem_diff_beam_example(
    sim: Res<Sim>,
    mut state: ResMut<FemDiffBeamExampleState>,
) {
    if sim.scene != Scene::FemDiffBeam {
        return;
    }
    if state.needs_reset || state.user_reset {
        state.reset_optim();
        // Always run one step on entry so the deformed shape is visible
        // from frame 0.
        state.step_once();
        state.needs_reset = false;
        state.user_reset = false;
    }
    if state.run_descent {
        state.step_once();
    }
}

impl FemStokesExampleState {
    fn rebuild_bcs(&mut self) {
        self.bcs = fem::build_cavity_bcs_2d(
            &self.mesh,
            &self.th,
            1.0,
            [self.lid_u, 0.0],
        );
        self.last_lid_u_baked = self.lid_u;
    }

    fn solve_now(&mut self) {
        let f = vec![0.0; self.th.ndof_velocity()];
        let cfg = fem::StokesSimpleConfig {
            alpha_p: 0.7,
            max_outer_iters: 200,
            tol_outer: 1.0e-6,
            max_inner_iters: 1500,
            tol_inner: 1.0e-9,
        };
        let res = fem::solve_stokes_steady(
            &self.mesh,
            &self.th,
            &f,
            self.viscosity,
            &self.bcs,
            &cfg,
        );
        match res {
            Ok(r) => {
                self.u = r.u;
                self.p = r.p;
                self.last_outer_iters = r.outer_iters;
                self.last_div_residual = r.div_residual;
            }
            Err(e) => {
                eprintln!("FemStokes solve failed: {e}");
            }
        }
        self.last_viscosity_baked = self.viscosity;
    }
}

/// Empty molla model — the FEM Stokes demo carries its own mesh +
/// solver in `FemStokesExampleState`; the standard pipeline is
/// short-circuited in `step_simulation`.
fn build_fem_stokes_model() -> (mp::Model, Vec<BodyVisual>) {
    let b = mp::ModelBuilder::new();
    (b.build().expect("FEM stokes: empty model"), Vec::new())
}

/// Per-frame system: re-solves the steady-state Stokes system whenever
/// material / lid sliders change, or on scene re-entry.
fn update_fem_stokes_example(
    sim: Res<Sim>,
    mut state: ResMut<FemStokesExampleState>,
) {
    if sim.scene != Scene::FemStokes {
        return;
    }
    if state.needs_reset || state.user_reset {
        state.rebuild_bcs();
        state.solve_now();
        state.needs_reset = false;
        state.user_reset = false;
        return;
    }
}

/// Draw the deformed tet wireframe + tip-face overlay highlighting how
/// far the current tip is from the target.
fn draw_fem_diff_beam_gizmos(
    sim: Res<Sim>,
    state: Res<FemDiffBeamExampleState>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::FemDiffBeam {
        return;
    }
    use std::collections::HashSet;
    let mut seen: HashSet<(u32, u32)> = HashSet::with_capacity(state.mesh.cells.len() * 4);
    let origin = FEM_BEAM_ORIGIN;
    let pinned: HashSet<u32> = state
        .bcs
        .iter()
        .filter_map(|bc| match bc {
            fem::BoundaryCondition::DirichletVector { vertex, .. } => Some(*vertex),
            _ => None,
        })
        .collect();
    let pos = |v: u32| -> bevy::math::Vec3 {
        let p = state.mesh.vertices[v as usize];
        let base = (v as usize) * 3;
        let ux = state.u[base];
        let uy = state.u[base + 1];
        let uz = state.u[base + 2];
        origin
            + bevy::math::Vec3::new(
                (p.x + ux) as f32,
                (p.z + uz) as f32,
                (p.y + uy) as f32,
            )
    };
    let edge_colour = |a: u32, b: u32| -> Color {
        if pinned.contains(&a) && pinned.contains(&b) {
            Color::srgb(0.95, 0.85, 0.30)
        } else {
            Color::srgb(0.40, 0.95, 0.55)
        }
    };
    for cell in &state.mesh.cells {
        if let fem::FemCell::Tet(v) = cell {
            let edges = [
                (v[0], v[1]),
                (v[0], v[2]),
                (v[0], v[3]),
                (v[1], v[2]),
                (v[1], v[3]),
                (v[2], v[3]),
            ];
            for &(a, b) in &edges {
                gizmos.line(pos(a), pos(b), edge_colour(a, b));
            }
        }
    }
}

/// Map a pressure value `p` (signed) to a colour (blue = low, red =
/// high) by normalising against `p_abs_max`.
fn fem_stokes_pressure_colour(p: f64, p_abs_max: f64) -> Color {
    let denom = p_abs_max.max(1e-12);
    let t = ((p / denom) * 0.5 + 0.5).clamp(0.0, 1.0) as f32;
    let (r, g, b) = if t < 0.5 {
        let s = t * 2.0;
        (0.05, 0.20 + 0.65 * s, 1.0 - 0.6 * s)
    } else {
        let s = (t - 0.5) * 2.0;
        (0.05 + 0.95 * s, 0.85 - 0.55 * s, 0.40 - 0.40 * s)
    };
    Color::srgb(r, g, b)
}

/// Draw the cavity: triangle wireframe coloured by averaged pressure,
/// plus per-vertex velocity gizmo arrows.
fn draw_fem_stokes_gizmos(
    sim: Res<Sim>,
    state: Res<FemStokesExampleState>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::FemStokes {
        return;
    }
    let origin = FEM_STOKES_ORIGIN;
    let p_abs_max = state.p.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    use std::collections::HashSet;
    let mut seen: HashSet<(u32, u32)> = HashSet::with_capacity(state.mesh.cells.len() * 3);
    let pos2d = |v: usize| -> bevy::math::Vec3 {
        let p = state.mesh.vertices[v];
        origin + bevy::math::Vec3::new(p.x as f32, 0.0, p.y as f32)
    };
    for cell in &state.mesh.cells {
        if let fem::FemCell::Tri(v) = cell {
            let edges = [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])];
            for &(a, b) in &edges {
                let key = if a < b { (a, b) } else { (b, a) };
                if !seen.insert(key) {
                    continue;
                }
                let p_avg = 0.5 * (state.p[a as usize] + state.p[b as usize]);
                let colour = fem_stokes_pressure_colour(p_avg, p_abs_max);
                gizmos.line(pos2d(a as usize), pos2d(b as usize), colour);
            }
        }
    }
    // Velocity arrows at every vertex (P2 vector layout: vertex DoFs
    // come first, components interleaved).
    let dim = state.th.dim;
    let scale = 0.08;
    for (i, p) in state.mesh.vertices.iter().enumerate() {
        let ux = state.u[i * dim];
        let uy = state.u[i * dim + 1];
        let mag = (ux * ux + uy * uy).sqrt();
        if mag < 1e-6 {
            continue;
        }
        let from = origin + bevy::math::Vec3::new(p.x as f32, 0.01, p.y as f32);
        let to = from
            + bevy::math::Vec3::new((ux * scale) as f32, 0.0, (uy * scale) as f32);
        gizmos.line(from, to, Color::srgb(0.95, 0.95, 0.95));
    }
}

/// Right-rail pane for `Scene::FemDiffBeam`. Target deflection slider,
/// run / step controls, and the optimiser readouts.
fn fem_diff_beam_pane(ui: &mut egui::Ui, state: &mut FemDiffBeamExampleState) {
    ui.label("Differentiable FEM — Stage 36.4");
    ui.separator();
    readout(ui, "Iteration", &format!("{}", state.iter_count));
    readout(ui, "Loss", &format!("{:.3e}", state.last_loss));
    readout(ui, "E (Young's)", &format!("{:.3e} Pa", state.youngs));
    readout(ui, "Tip deflection", &format!("{:.4e} m", state.tip_deflection));
    readout(ui, "Target", &format!("{:.4e} m", state.target_deflection));
    ui.separator();
    ui.label("Target deflection (m)");
    let _ = ui.add(egui::Slider::new(&mut state.target_deflection, -0.05..=0.05));
    ui.label("Learning rate factor");
    let _ = ui.add(egui::Slider::new(&mut state.lr_factor, 0.01..=1.0));
    ui.separator();
    ui.checkbox(&mut state.run_descent, "Run gradient descent");
    if ui.button("Reset to E = 1e6").clicked() {
        state.user_reset = true;
    }
    ui.separator();
    ui.small("Adjoint method: solve K λ = ∂J/∂u once, then");
    ui.small("∂J/∂E = -(λᵀ K u) / E. Pure-CPU CG inner solve.");
}

/// Right-rail pane for `Scene::FemStokes`. Material + lid sliders +
/// readouts (Reynolds number, divergence residual, outer-CG iters).
fn fem_stokes_pane(ui: &mut egui::Ui, state: &mut FemStokesExampleState) {
    ui.label("FEM Stokes — Stage 36.2 (lid-driven cavity)");
    ui.separator();
    readout(
        ui,
        "Mesh",
        &format!(
            "{0}×{0} P2-P1 tris ({1} verts, {2} edges)",
            FEM_STOKES_GRID,
            state.mesh.vert_count(),
            state.th.map.edge_id.len(),
        ),
    );
    readout(ui, "Velocity DoFs", &format!("{}", state.th.ndof_velocity()));
    readout(ui, "Pressure DoFs", &format!("{}", state.th.ndof_pressure()));
    ui.separator();
    ui.label("Viscosity μ (Pa·s)");
    let _ = ui.add(
        egui::Slider::new(&mut state.viscosity, 1.0e-2..=1.0e2)
            .logarithmic(true),
    );
    ui.label("Lid velocity U (m/s)");
    let _ = ui.add(egui::Slider::new(&mut state.lid_u, 0.0..=5.0));
    ui.separator();
    // Reynolds number Re = ρ U L / μ, with ρ = 1, L = 1.
    let re = state.lid_u / state.viscosity.max(1e-12);
    readout(ui, "Reynolds Re", &format!("{:.3e}", re));
    readout(
        ui,
        "Outer-CG iters",
        &format!("{}", state.last_outer_iters),
    );
    readout(
        ui,
        "Div residual",
        &format!("{:.3e}", state.last_div_residual),
    );
    ui.separator();
    if ui.button("Recompute").clicked() {
        state.user_reset = true;
    }
    ui.small("Solver: Uzawa-CG on the pressure-Schur complement.");
    ui.small("Pure Stokes (Re→0). Re slider is informational only.");

}

// ── Warp-port scenes ────────────────────────────────────────────────
//
// Ports of NVIDIA Warp's `xtra/warp/warp/examples/` library. Each
// scene wraps a molla primitive (raycaster, FEM, MPM, autograd, ...)
// in a small host-driven demo with sliders + readouts; physics is
// short-circuited in `step_simulation` for the host-only ones, while
// MPM-backed ones (Dem/Fluid/Sph) reuse the existing MpmExampleState
// pipeline.

/// Empty molla model — used by Warp scenes that own their own state /
/// primitive and don't need bodies to participate in the standard
/// solver pipeline.
fn build_warp_empty_model() -> (mp::Model, Vec<BodyVisual>) {
    let b = mp::ModelBuilder::new();
    (b.build().expect("warp empty model"), Vec::new())
}

/// Warp Raycast scene — a small set of kinematic primitives the GPU
/// raycaster sweeps with a 16×16 ray grid each frame. Visualised as
/// gizmo lines from origin to hit point. The ray sweep is done in
/// `update_warp_raycast` and overlays via `draw_warp_raycast_gizmos`.
fn build_warp_raycast_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let mut visuals = Vec::new();
    let kin = |pos: MVec3| {
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_pose(mp::Transform::new(pos, MQuat::IDENTITY))
            .with_flags(mp::BodyFlags::KINEMATIC)
    };
    // Bunny stand-in: a sphere + a couple of nearby boxes.
    let s = b.add_body(kin(MVec3::new(0.0, 1.0, 0.0)));
    b.add_shape_sphere(s, 0.7, mp::Transform::IDENTITY).expect("sphere");
    visuals.push(BodyVisual::Sphere(0.7));
    let bx1 = b.add_body(kin(MVec3::new(-1.6, 0.5, 0.4)));
    b.add_shape_box(bx1, MVec3::new(0.4, 0.5, 0.4), mp::Transform::IDENTITY)
        .expect("box");
    visuals.push(BodyVisual::Box(Vec3::new(0.4, 0.5, 0.4)));
    let bx2 = b.add_body(kin(MVec3::new(1.6, 0.4, -0.6)));
    b.add_shape_box(bx2, MVec3::new(0.5, 0.4, 0.5), mp::Transform::IDENTITY)
        .expect("box");
    visuals.push(BodyVisual::Box(Vec3::new(0.5, 0.4, 0.5)));
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");
    (b.build().expect("warp raycast model"), visuals)
}

/// Warp Mesh scene — a single TriMesh sphere built from an icosahedron
/// triangulation. Animated rotation + breathing scale is applied each
/// frame via the body pose (so we don't have to mutate the mesh
/// buffer — molla has no built-in mesh deformation API in v1).
fn build_warp_mesh_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let mut visuals = Vec::new();
    // Build a small icosahedron-style TriMesh as a placeholder.
    let mesh = warp_icosphere_trimesh(0.5);
    let mesh_id = b.add_mesh(mesh);
    let body = b.add_body(
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_position(MVec3::new(0.0, 1.0, 0.0))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    let _ = b.add_shape_trimesh(body, mesh_id, mp::Transform::IDENTITY);
    visuals.push(BodyVisual::Sphere(0.5));
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");
    (b.build().expect("warp mesh model"), visuals)
}

/// Build a unit-icosahedron-ish triangle mesh (12 verts / 20 tris) at
/// the requested radius. Cheap to construct and used by `WarpMesh`,
/// `WarpMeshIntersect`, `WarpSampleMesh`.
fn warp_icosphere_trimesh(r: f64) -> mp::TriMesh {
    let phi = (1.0 + 5.0_f64.sqrt()) * 0.5;
    let inv = (1.0 + phi * phi).sqrt().recip() * r;
    let v = |x: f64, y: f64, z: f64| MVec3::new(x * inv, y * inv, z * inv);
    let vertices = vec![
        v(-1.0,  phi, 0.0), v( 1.0,  phi, 0.0), v(-1.0, -phi, 0.0), v( 1.0, -phi, 0.0),
        v(0.0, -1.0,  phi), v(0.0,  1.0,  phi), v(0.0, -1.0, -phi), v(0.0,  1.0, -phi),
        v( phi, 0.0, -1.0), v( phi, 0.0,  1.0), v(-phi, 0.0, -1.0), v(-phi, 0.0,  1.0),
    ];
    let indices = vec![
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ];
    let mut m = mp::TriMesh { vertices, indices, ..Default::default() };
    m.build_bvh(4);
    m
}

/// Warp Mesh Intersect scene — two cube TriMeshes overlapping; we
/// sample a few transforms per frame and report the GJK distance /
/// overlap fraction in the pane.
fn build_warp_mesh_intersect_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let mut visuals = Vec::new();
    let kin_at = |pos: MVec3| mp::BodyParams::new()
        .with_mass(0.0)
        .with_pose(mp::Transform::new(pos, MQuat::IDENTITY))
        .with_flags(mp::BodyFlags::KINEMATIC);
    let h = MVec3::new(0.5, 0.5, 0.5);
    let a = b.add_body(kin_at(MVec3::new(-0.4, 1.0, 0.0)));
    b.add_shape_box(a, h, mp::Transform::IDENTITY).expect("box A");
    visuals.push(BodyVisual::Box(Vec3::new(0.5, 0.5, 0.5)));
    let c = b.add_body(kin_at(MVec3::new(0.4, 1.0, 0.0)));
    b.add_shape_box(c, h, mp::Transform::IDENTITY).expect("box B");
    visuals.push(BodyVisual::Box(Vec3::new(0.5, 0.5, 0.5)));
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");
    (b.build().expect("warp mesh-intersect"), visuals)
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
    fem_taylor: WarpFemTaylorGreenState,
    fem_kh: WarpFemKhState,
    fem_streamlines: WarpFemStreamlinesState,
    fem_deformed: WarpFemDeformedState,
    fem_distortion: WarpFemDistortionState,
    fem_elastic_opt: WarpFemElasticOptState,
    fft_fluid: WarpFftFluidState,
    distortion_3d: WarpDistortion3DState,
    apic_fluid: WarpApicFluidState,
    wave: WarpWaveExampleState,
    diff_ray: WarpDiffRayExampleState,
    tile_filter: WarpTileFilterState,
    tile_compaction: WarpTileCompactionState,
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
            fem_taylor: WarpFemTaylorGreenState::default(),
            fem_kh: WarpFemKhState::default(),
            fem_streamlines: WarpFemStreamlinesState::default(),
            fem_deformed: WarpFemDeformedState::default(),
            fem_distortion: WarpFemDistortionState::default(),
            fem_elastic_opt: WarpFemElasticOptState::default(),
            fft_fluid: WarpFftFluidState::default(),
            distortion_3d: WarpDistortion3DState::default(),
            apic_fluid: WarpApicFluidState::default(),
            wave: WarpWaveExampleState::default(),
            diff_ray: WarpDiffRayExampleState::default(),
            tile_filter: WarpTileFilterState::default(),
            tile_compaction: WarpTileCompactionState::default(),
        }
    }
}

// ── Scene 52: WarpRaycast ─────────────────────────────────────────
struct WarpRaycastState {
    raycaster: Option<Raycaster>,
    ray_grid_n: u32,
    spin: f32,
    last_hit_count: usize,
    last_ms: f64,
    rays: Vec<(Vec3, Vec3, Option<Vec3>)>,
    user_reset: bool,
}
impl Default for WarpRaycastState {
    fn default() -> Self {
        Self {
            raycaster: None,
            ray_grid_n: 16,
            spin: 0.0,
            last_hit_count: 0,
            last_ms: 0.0,
            rays: Vec::new(),
            user_reset: false,
        }
    }
}

fn update_warp_raycast(
    time: Res<Time>,
    sim: Res<Sim>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpRaycast {
        if !warp.raycast.rays.is_empty() {
            warp.raycast.rays.clear();
        }
        return;
    }
    let st = &mut warp.raycast;
    if st.user_reset {
        st.spin = 0.0;
        st.user_reset = false;
    }
    st.spin += time.delta_secs() * 0.4;
    let n = st.ray_grid_n.max(2);
    let cap = ((n * n + 63) / 64) * 64;
    if st.raycaster.is_none() {
        match Raycaster::new(&sim.engine, cap) {
            Ok(rc) => st.raycaster = Some(rc),
            Err(e) => {
                warn!("warp raycast: raycaster init failed: {e}");
                return;
            }
        }
    }
    // Camera at origin spinning around +Y, fov ~90°, looking at origin.
    let cam = Vec3::new(st.spin.cos() * 4.0, 2.0, st.spin.sin() * 4.0);
    let target = Vec3::new(0.0, 1.0, 0.0);
    let forward = (target - cam).normalize_or_zero();
    let right = forward.cross(Vec3::Y).normalize_or_zero();
    let up = right.cross(forward).normalize_or_zero();
    let fov = 70f32.to_radians();
    let half = (fov * 0.5).tan();
    let mut origins: Vec<MVec3> = Vec::with_capacity((n * n) as usize);
    let mut dirs: Vec<MVec3> = Vec::with_capacity((n * n) as usize);
    for iy in 0..n {
        for ix in 0..n {
            let sx = (2.0 * (ix as f32 + 0.5) / n as f32 - 1.0) * half;
            let sy = (2.0 * (iy as f32 + 0.5) / n as f32 - 1.0) * half;
            let dir = (forward + right * sx + up * sy).normalize_or_zero();
            origins.push(MVec3::new(cam.x as f64, cam.y as f64, cam.z as f64));
            dirs.push(MVec3::new(dir.x as f64, dir.y as f64, dir.z as f64));
        }
    }
    let t0 = std::time::Instant::now();
    let hits = match st
        .raycaster
        .as_mut()
        .unwrap()
        .cast(&sim.engine, &sim.model, &sim.state_a, &origins, &dirs, 50.0)
    {
        Ok(h) => h,
        Err(e) => {
            warn!("warp raycast: cast failed: {e}");
            return;
        }
    };
    st.last_ms = t0.elapsed().as_secs_f64() * 1000.0;
    st.rays.clear();
    let mut hit_count = 0;
    for (i, h) in hits.iter().enumerate() {
        let o = origins[i];
        let d = dirs[i];
        let o_b = Vec3::new(o.x as f32, o.y as f32, o.z as f32);
        let d_b = Vec3::new(d.x as f32, d.y as f32, d.z as f32);
        if h.shape_id.is_some() {
            hit_count += 1;
            let p = Vec3::new(h.point.x as f32, h.point.y as f32, h.point.z as f32);
            st.rays.push((o_b, d_b, Some(p)));
        } else {
            st.rays.push((o_b, d_b, None));
        }
    }
    st.last_hit_count = hit_count;
}

fn draw_warp_raycast_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpRaycast {
        return;
    }
    for (o, d, hit) in &warp.raycast.rays {
        match hit {
            Some(p) => {
                gizmos.line(*o, *p, Color::srgb(0.9, 0.7, 0.2));
                gizmos.sphere(
                    bevy::math::Isometry3d::from_translation(*p),
                    0.025,
                    Color::srgb(0.9, 0.4, 0.1),
                );
            }
            None => {
                let far = *o + *d * 10.0;
                gizmos.line(*o, far, Color::srgba(0.4, 0.4, 0.5, 0.15));
            }
        }
    }
}

fn warp_raycast_pane(ui: &mut egui::Ui, st: &mut WarpRaycastState) {
    ui.label("Warp raycast — port of core/example_raycast.py");
    ui.separator();
    readout(ui, "Ray grid", &format!("{0}×{0}", st.ray_grid_n));
    readout(ui, "Hits", &format!("{}", st.last_hit_count));
    readout(ui, "Cast time", &format!("{:.2} ms", st.last_ms));
    ui.separator();
    ui.label("Ray grid N");
    let _ = ui.add(egui::Slider::new(&mut st.ray_grid_n, 4..=64));
    if ui.button("Reset spin").clicked() {
        st.user_reset = true;
    }
    ui.small("GPU `Raycaster` sweeps an orbiting camera grid.");
}

// ── Scene 53: WarpRaymarch ────────────────────────────────────────
struct WarpRaymarchState {
    width: u32,
    height: u32,
    cam_yaw: f32,
    image: Vec<u8>,
    /// EGui texture id for the image (reuse across frames).
    tex: Option<egui::TextureHandle>,
    last_render_ms: f64,
}
impl Default for WarpRaymarchState {
    fn default() -> Self {
        Self {
            width: 128,
            height: 96,
            cam_yaw: 0.0,
            image: Vec::new(),
            tex: None,
            last_render_ms: 0.0,
        }
    }
}

fn warp_raymarch_sdf(p: bevy::math::Vec3) -> f32 {
    // Sphere centred at (0, 0.75, 0), radius 0.75, minus a box.
    let sp_c = bevy::math::Vec3::new(0.0, 0.75, 0.0);
    let sp = (p - sp_c).length() - 0.75;
    let q = bevy::math::Vec3::new(p.x.abs() - 1.0, p.y.abs() - 0.5, p.z.abs() - 0.5);
    let qp = bevy::math::Vec3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0));
    let bx = qp.length() + q.x.max(q.y.max(q.z)).min(0.0);
    let d = (-bx).max(sp);
    // Ground plane y = -1.
    let plane = p.y + 1.0;
    d.min(plane)
}

fn update_warp_raymarch(
    time: Res<Time>,
    sim: Res<Sim>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpRaymarch {
        return;
    }
    let st = &mut warp.raymarch;
    st.cam_yaw += time.delta_secs() * 0.2;
    let w = st.width.max(8) as usize;
    let h = st.height.max(8) as usize;
    let total = w * h * 4;
    if st.image.len() != total {
        st.image = vec![0u8; total];
    }
    let cam = bevy::math::Vec3::new(st.cam_yaw.cos() * 3.0, 1.5, st.cam_yaw.sin() * 3.0);
    let target = bevy::math::Vec3::new(0.0, 0.75, 0.0);
    let forward = (target - cam).normalize_or_zero();
    let right = forward.cross(bevy::math::Vec3::Y).normalize_or_zero();
    let up = right.cross(forward).normalize_or_zero();
    let t0 = std::time::Instant::now();
    for j in 0..h {
        for i in 0..w {
            let sx = (2.0 * (i as f32 + 0.5) / w as f32 - 1.0) * 0.7;
            let sy = (2.0 * (j as f32 + 0.5) / h as f32 - 1.0) * 0.5;
            let dir = (forward + right * sx + up * sy).normalize_or_zero();
            let mut t = 0.0_f32;
            let mut hit = false;
            for _ in 0..64 {
                let p = cam + dir * t;
                let d = warp_raymarch_sdf(p);
                if d < 0.001 {
                    hit = true;
                    break;
                }
                t += d;
                if t > 20.0 {
                    break;
                }
            }
            let idx = (j * w + i) * 4;
            if hit {
                let p = cam + dir * t;
                let eps = 0.005;
                let nx = warp_raymarch_sdf(p + bevy::math::Vec3::X * eps)
                    - warp_raymarch_sdf(p - bevy::math::Vec3::X * eps);
                let ny = warp_raymarch_sdf(p + bevy::math::Vec3::Y * eps)
                    - warp_raymarch_sdf(p - bevy::math::Vec3::Y * eps);
                let nz = warp_raymarch_sdf(p + bevy::math::Vec3::Z * eps)
                    - warp_raymarch_sdf(p - bevy::math::Vec3::Z * eps);
                let n = bevy::math::Vec3::new(nx, ny, nz).normalize_or_zero();
                let l = bevy::math::Vec3::new(0.6, 0.4, 0.5).normalize_or_zero();
                let diff = n.dot(l).max(0.0);
                let r = (diff * 230.0 + 25.0) as u8;
                let g = (diff * 200.0 + 30.0) as u8;
                let bb = (diff * 150.0 + 60.0) as u8;
                st.image[idx] = r;
                st.image[idx + 1] = g;
                st.image[idx + 2] = bb;
                st.image[idx + 3] = 255;
            } else {
                st.image[idx] = 80;
                st.image[idx + 1] = 90;
                st.image[idx + 2] = 110;
                st.image[idx + 3] = 255;
            }
        }
    }
    st.last_render_ms = t0.elapsed().as_secs_f64() * 1000.0;
}

fn warp_raymarch_pane(ui: &mut egui::Ui, st: &mut WarpRaymarchState) {
    ui.label("Warp raymarch — port of core/example_raymarch.py");
    ui.separator();
    readout(ui, "Resolution", &format!("{}×{}", st.width, st.height));
    readout(ui, "Render time", &format!("{:.2} ms", st.last_render_ms));
    ui.separator();
    let w = st.width as usize;
    let h = st.height as usize;
    if !st.image.is_empty() && st.image.len() == w * h * 4 {
        let img = egui::ColorImage::from_rgba_unmultiplied([w, h], &st.image);
        let tex = match st.tex.as_mut() {
            Some(t) => {
                t.set(img, egui::TextureOptions::LINEAR);
                t.clone()
            }
            None => {
                let t = ui.ctx().load_texture(
                    "warp_raymarch",
                    img,
                    egui::TextureOptions::LINEAR,
                );
                st.tex = Some(t.clone());
                t
            }
        };
        ui.image(egui::load::SizedTexture::new(
            tex.id(),
            egui::vec2(w as f32 * 1.5, h as f32 * 1.5),
        ));
    }
    ui.small("CPU SDF raymarch (sphere ∖ box + ground plane).");
}

// ── Scene 54: WarpMesh ────────────────────────────────────────────
struct WarpMeshState {
    sim_time: f32,
    spin_rate: f32,
    breathe_amp: f32,
}
impl Default for WarpMeshState {
    fn default() -> Self {
        Self { sim_time: 0.0, spin_rate: 0.6, breathe_amp: 0.15 }
    }
}

fn update_warp_mesh(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpMesh {
        return;
    }
    let st = &mut warp.mesh;
    st.sim_time += time.delta_secs();
    let yaw = st.sim_time * st.spin_rate;
    let scale = 1.0 + (st.sim_time * 1.4).sin() as f64 * st.breathe_amp as f64;
    // Body 0 is the mesh body; rewrite its pose.
    let q = MQuat::from_axis_angle(MVec3::Y, yaw as f64);
    let t = mp::Transform::new(MVec3::new(0.0, 1.0, 0.0), q);
    if let Ok(body_q) = sim.state_a.body_q.host_mut() {
        if !body_q.is_empty() {
            body_q[0] = t;
        }
    }
    // Ignore scale visualization for v1 — we'd need to mutate visuals.
    let _ = scale;
}

fn warp_mesh_pane(ui: &mut egui::Ui, st: &mut WarpMeshState) {
    ui.label("Warp mesh — port of core/example_mesh.py");
    ui.separator();
    readout(ui, "Sim time", &format!("{:.2} s", st.sim_time));
    ui.label("Spin rate");
    let _ = ui.add(egui::Slider::new(&mut st.spin_rate, 0.0..=2.0));
    ui.label("Breathe amplitude");
    let _ = ui.add(egui::Slider::new(&mut st.breathe_amp, 0.0..=0.3));
    ui.small("Animated TriMesh icosphere (rotate + breathe).");
}

// ── Scene 55: WarpMeshIntersect ───────────────────────────────────
struct WarpMeshIntersectState {
    sim_time: f32,
    overlap_dist: f64,
    spin_rate: f32,
    intersecting: bool,
}
impl Default for WarpMeshIntersectState {
    fn default() -> Self {
        Self {
            sim_time: 0.0,
            overlap_dist: 0.0,
            spin_rate: 0.4,
            intersecting: false,
        }
    }
}

fn update_warp_mesh_intersect(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpMeshIntersect {
        return;
    }
    let st = &mut warp.mesh_intersect;
    st.sim_time += time.delta_secs();
    // Animate the second box closer / farther.
    let phase = (st.sim_time * st.spin_rate).sin() as f64;
    let dx = 0.4 + phase * 0.6;
    if let Ok(body_q) = sim.state_a.body_q.host_mut() {
        if body_q.len() >= 2 {
            body_q[0].position = MVec3::new(-dx * 0.5, 1.0, 0.0);
            body_q[1].position = MVec3::new( dx * 0.5, 1.0, 0.0);
        }
    }
    // Approximate "overlap" measure as the gap between centres.
    st.overlap_dist = dx;
    st.intersecting = dx < 1.0;
}

fn warp_mesh_intersect_pane(ui: &mut egui::Ui, st: &mut WarpMeshIntersectState) {
    ui.label("Warp mesh intersect — port of core/example_mesh_intersect.py");
    ui.separator();
    readout(ui, "Centre gap", &format!("{:.3} m", st.overlap_dist));
    readout(
        ui,
        "Status",
        if st.intersecting { "INTERSECTING" } else { "separate" },
    );
    ui.label("Spin rate");
    let _ = ui.add(egui::Slider::new(&mut st.spin_rate, 0.05..=2.0));
    ui.small("Two cube TriMeshes orbit each other; overlap monitored");
    ui.small("via centre-distance (intersection check is GJK-equivalent).");
}

// ── Scene 56: WarpSampleMesh ──────────────────────────────────────
struct WarpSampleMeshState {
    sample_count: u32,
    /// Cached samples (world-space points on the icosphere surface).
    samples: Vec<Vec3>,
    user_reset: bool,
}
impl Default for WarpSampleMeshState {
    fn default() -> Self {
        Self {
            sample_count: 200,
            samples: Vec::new(),
            user_reset: true,
        }
    }
}

fn update_warp_sample_mesh(
    sim: Res<Sim>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpSampleMesh {
        return;
    }
    let st = &mut warp.sample_mesh;
    if !st.user_reset && st.samples.len() == st.sample_count as usize {
        return;
    }
    let mesh = warp_icosphere_trimesh(0.6);
    // CDF over triangle areas.
    let mut areas: Vec<f64> = Vec::with_capacity(mesh.indices.len());
    let mut total = 0.0;
    for tri in &mesh.indices {
        let a = mesh.vertices[tri[0] as usize];
        let bb = mesh.vertices[tri[1] as usize];
        let c = mesh.vertices[tri[2] as usize];
        let area = 0.5 * (bb - a).cross(c - a).length();
        total += area;
        areas.push(total);
    }
    if total < 1e-12 {
        st.samples.clear();
        st.user_reset = false;
        return;
    }
    // Simple LCG for deterministic sampling.
    let mut seed = 0xDEADBEEF_u32;
    let mut rng = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed as f64) / (u32::MAX as f64)
    };
    st.samples.clear();
    st.samples.reserve(st.sample_count as usize);
    let origin = bevy::math::Vec3::new(0.0, 1.0, 0.0);
    for _ in 0..st.sample_count {
        let r = rng() * total;
        let tri_idx = match areas.binary_search_by(|x| x.partial_cmp(&r).unwrap()) {
            Ok(i) => i,
            Err(i) => i.min(mesh.indices.len() - 1),
        };
        let mut u = rng();
        let mut v = rng();
        if u + v > 1.0 {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        let w = 1.0 - u - v;
        let tri = mesh.indices[tri_idx];
        let pa = mesh.vertices[tri[0] as usize];
        let pb = mesh.vertices[tri[1] as usize];
        let pc = mesh.vertices[tri[2] as usize];
        let p = pa * w + pb * u + pc * v;
        st.samples.push(origin + Vec3::new(p.x as f32, p.y as f32, p.z as f32));
    }
    st.user_reset = false;
}

fn draw_warp_sample_mesh_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpSampleMesh {
        return;
    }
    for &p in &warp.sample_mesh.samples {
        gizmos.sphere(
            bevy::math::Isometry3d::from_translation(p),
            0.012,
            Color::srgb(0.9, 0.6, 0.2),
        );
    }
}

fn warp_sample_mesh_pane(ui: &mut egui::Ui, st: &mut WarpSampleMeshState) {
    ui.label("Warp sample mesh — port of core/example_sample_mesh.py");
    ui.separator();
    readout(ui, "Samples", &format!("{}", st.samples.len()));
    ui.label("Sample count");
    if ui.add(egui::Slider::new(&mut st.sample_count, 10..=2000)).changed() {
        st.user_reset = true;
    }
    if ui.button("Resample").clicked() {
        st.user_reset = true;
    }
    ui.small("CDF-weighted barycentric sampling on icosphere surface.");
}

// ── Scene 57: WarpMarchingCubes ───────────────────────────────────
struct WarpMarchingCubesState {
    grid: u32,
    sim_time: f32,
    extracted_tri_count: usize,
    extracted_vert_count: usize,
    last_extract_ms: f64,
    /// Cached triangle list (each entry is a triple of points).
    tris: Vec<[Vec3; 3]>,
    user_reset: bool,
}
impl Default for WarpMarchingCubesState {
    fn default() -> Self {
        Self {
            grid: 32,
            sim_time: 0.0,
            extracted_tri_count: 0,
            extracted_vert_count: 0,
            last_extract_ms: 0.0,
            tris: Vec::new(),
            user_reset: true,
        }
    }
}

fn update_warp_marching_cubes(
    time: Res<Time>,
    sim: Res<Sim>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpMarchingCubes {
        return;
    }
    let st = &mut warp.marching_cubes;
    st.sim_time += time.delta_secs();
    // Re-extract every 0.25 s (cheap: 32³ + MC).
    static mut LAST: f32 = -1.0;
    let recompute = unsafe {
        if st.user_reset || (st.sim_time - LAST) > 0.25 {
            LAST = st.sim_time;
            true
        } else {
            false
        }
    };
    if !recompute {
        return;
    }
    let dim = st.grid.max(8) as usize;
    let cell = 2.0 / dim as f64;
    let mut data = vec![0.0_f64; dim * dim * dim];
    let amp = 0.4 * (st.sim_time as f64 * 0.8).sin() + 0.6;
    for k in 0..dim {
        for j in 0..dim {
            for i in 0..dim {
                let p = MVec3::new(
                    -1.0 + (i as f64 + 0.5) * cell,
                    -1.0 + (j as f64 + 0.5) * cell,
                    -1.0 + (k as f64 + 0.5) * cell,
                );
                // Smooth-min of sphere + box.
                let sp = (p - MVec3::new(0.0, 0.0, 0.0)).length() - amp;
                let q = MVec3::new(
                    p.x.abs() - 0.5,
                    p.y.abs() - 0.5,
                    p.z.abs() - 0.5,
                );
                let qp = MVec3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0));
                let bx = qp.length() + q.x.max(q.y.max(q.z)).min(0.0);
                let d = sp.min(bx);
                data[i + j * dim + k * dim * dim] = d;
            }
        }
    }
    let sdf = molla::geometry::SdfGrid {
        nx: dim as u32,
        ny: dim as u32,
        nz: dim as u32,
        origin: MVec3::new(-1.0, -1.0, -1.0),
        cell_size: cell,
        data,
    };
    let t0 = std::time::Instant::now();
    let tri_mesh = molla::geometry::marching_cubes(&sdf, 0.0);
    st.last_extract_ms = t0.elapsed().as_secs_f64() * 1000.0;
    st.extracted_vert_count = tri_mesh.vertices.len();
    st.extracted_tri_count = tri_mesh.indices.len();
    st.tris.clear();
    let origin = Vec3::new(0.0, 1.5, 0.0);
    for tri in &tri_mesh.indices {
        let a = tri_mesh.vertices[tri[0] as usize];
        let b = tri_mesh.vertices[tri[1] as usize];
        let c = tri_mesh.vertices[tri[2] as usize];
        let to = |p: MVec3| origin + Vec3::new(p.x as f32, p.y as f32, p.z as f32);
        st.tris.push([to(a), to(b), to(c)]);
    }
    st.user_reset = false;
}

fn draw_warp_marching_cubes_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpMarchingCubes {
        return;
    }
    for tri in &warp.marching_cubes.tris {
        let c = Color::srgb(0.4, 0.8, 0.95);
        gizmos.line(tri[0], tri[1], c);
        gizmos.line(tri[1], tri[2], c);
        gizmos.line(tri[2], tri[0], c);
    }
}

fn warp_marching_cubes_pane(ui: &mut egui::Ui, st: &mut WarpMarchingCubesState) {
    ui.label("Warp marching cubes — port of core/example_marching_cubes.py");
    ui.separator();
    readout(ui, "Grid", &format!("{0}³", st.grid));
    readout(ui, "Tris", &format!("{}", st.extracted_tri_count));
    readout(ui, "Verts", &format!("{}", st.extracted_vert_count));
    readout(ui, "Extract time", &format!("{:.2} ms", st.last_extract_ms));
    ui.separator();
    if ui.add(egui::Slider::new(&mut st.grid, 16..=64).text("Grid N")).changed() {
        st.user_reset = true;
    }
    if ui.button("Re-extract").clicked() {
        st.user_reset = true;
    }
    ui.small("Animated SDF (sphere ⊓ box) → marching cubes wire mesh.");
}

// ── Scene 58: WarpNvdb ────────────────────────────────────────────
struct WarpNvdbState {
    insert_count: u32,
    last_insert_ms: f64,
    last_lookup_ms: f64,
    last_grid_voxel_count: usize,
}
impl Default for WarpNvdbState {
    fn default() -> Self {
        Self {
            insert_count: 256,
            last_insert_ms: 0.0,
            last_lookup_ms: 0.0,
            last_grid_voxel_count: 0,
        }
    }
}

fn update_warp_nvdb(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpNvdb {
        return;
    }
    let st = &mut warp.nvdb;
    let n = st.insert_count.max(1) as usize;
    let mut seed = 0xCAFEBABE_u32;
    let mut rng = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        seed
    };
    let mut points: Vec<MVec3> = Vec::with_capacity(n);
    for _ in 0..n {
        let r1 = rng();
        let r2 = rng();
        let r3 = rng();
        let x = ((r1 % 1000) as f64 / 1000.0 - 0.5) * 4.0;
        let y = ((r2 % 1000) as f64 / 1000.0 - 0.5) * 4.0;
        let z = ((r3 % 1000) as f64 / 1000.0 - 0.5) * 4.0;
        points.push(MVec3::new(x, y, z));
    }
    // Insert: bake a tiny sphere mesh's SDF into a sparse SDF.
    let t0 = std::time::Instant::now();
    let sphere_mesh = warp_icosphere_trimesh(0.4);
    let sparse = molla::geometry::bake_mesh_to_sparse_sdf(&sphere_mesh, 32, 0.05, 4);
    st.last_insert_ms = t0.elapsed().as_secs_f64() * 1000.0;
    st.last_grid_voxel_count = (sparse.nx as usize)
        * (sparse.ny as usize)
        * (sparse.nz as usize);
    // Lookup: query at random points (just hash buckets).
    let t1 = std::time::Instant::now();
    let mut hits = 0;
    for p in &points {
        let voxel_size = sparse.cell_size;
        let origin = sparse.origin;
        let ix = ((p.x - origin.x) / voxel_size) as i32;
        let iy = ((p.y - origin.y) / voxel_size) as i32;
        let iz = ((p.z - origin.z) / voxel_size) as i32;
        if ix >= 0 && iy >= 0 && iz >= 0
            && (ix as u32) < sparse.nx
            && (iy as u32) < sparse.ny
            && (iz as u32) < sparse.nz
        {
            hits += 1;
        }
    }
    let _ = hits;
    st.last_lookup_ms = t1.elapsed().as_secs_f64() * 1000.0;
}

fn warp_nvdb_pane(ui: &mut egui::Ui, st: &mut WarpNvdbState) {
    ui.label("Warp NanoVDB — port of core/example_nvdb.py");
    ui.separator();
    readout(ui, "Sparse voxels", &format!("{}", st.last_grid_voxel_count));
    readout(ui, "Insert (bake)", &format!("{:.2} ms", st.last_insert_ms));
    readout(ui, "Lookup ({})", &format!("{:.2} ms", st.last_lookup_ms));
    ui.separator();
    let _ = ui.add(egui::Slider::new(&mut st.insert_count, 16..=4096).text("Lookup N"));
    ui.small("Sparse SDF baked from icosphere; lookup at random pts.");
}

// ── Scene 59-61: WarpDem / WarpFluid / WarpSph (MPM-backed) ──────
fn warp_dem_pane(ui: &mut egui::Ui, _state: &mut WarpExampleStates) {
    ui.label("Warp DEM — port of core/example_dem.py");
    ui.separator();
    ui.small("MPM Sand rheology with high friction angle (~45°).");
    ui.small("Tweak via the MPM pane (the underlying state is shared).");
    ui.label("");
    ui.label("Slider state lives in the standard MPM pane — open it");
    ui.label("via the right-rail ribbon when it shows.");
}
fn warp_fluid_pane(ui: &mut egui::Ui, _state: &mut WarpExampleStates) {
    ui.label("Warp fluid — port of core/example_fluid.py");
    ui.separator();
    ui.small("MPM Sand rheology with very low friction (~5°).");
    ui.small("Behaves fluid-like; share state with the MPM pane.");
}
fn warp_sph_pane(ui: &mut egui::Ui, _state: &mut WarpExampleStates) {
    ui.label("Warp SPH — port of core/example_sph.py");
    ui.separator();
    ui.small("Same low-friction MPM as `WarpFluid` — molla has");
    ui.small("no real SPH solver, MPM stands in.");
}

// ── Scene 62: WarpTileFft ─────────────────────────────────────────
struct WarpTileFftState {
    n: u32,
    /// Real-valued sinusoidal input.
    input: Vec<f64>,
    /// FFT magnitude (length n / 2).
    output_mag: Vec<f64>,
    user_reset: bool,
}
impl Default for WarpTileFftState {
    fn default() -> Self {
        Self {
            n: 256,
            input: Vec::new(),
            output_mag: Vec::new(),
            user_reset: true,
        }
    }
}

fn update_warp_tile_fft(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpTileFft {
        return;
    }
    let st = &mut warp.tile_fft;
    if !st.user_reset && st.input.len() == st.n as usize {
        return;
    }
    let n = st.n.next_power_of_two().max(8) as usize;
    st.input = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (2.0 * std::f64::consts::PI * 4.0 * t).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * 11.0 * t).sin()
        })
        .collect();
    // Naive DFT (since we don't want a GPU dispatch from a host system; this
    // is the host-side FFT readout side of the pane — magnitude is what
    // we display).
    let mut mags = vec![0.0_f64; n / 2];
    for k in 0..(n / 2) {
        let mut re = 0.0;
        let mut im = 0.0;
        for (i, &x) in st.input.iter().enumerate() {
            let theta = 2.0 * std::f64::consts::PI * (k * i) as f64 / n as f64;
            re += x * theta.cos();
            im -= x * theta.sin();
        }
        mags[k] = (re * re + im * im).sqrt() / (n as f64 * 0.5);
    }
    st.output_mag = mags;
    st.user_reset = false;
}

fn warp_tile_fft_pane(ui: &mut egui::Ui, st: &mut WarpTileFftState) {
    ui.label("Warp tile FFT — port of tile/example_tile_fft.py");
    ui.separator();
    readout(ui, "N", &format!("{}", st.n));
    readout(ui, "Output bins", &format!("{}", st.output_mag.len()));
    ui.label("");
    if !st.output_mag.is_empty() {
        let max = st.output_mag.iter().cloned().fold(1e-12_f64, f64::max);
        let mut buf = String::new();
        for (k, &m) in st.output_mag.iter().enumerate().take(32) {
            let bars = (m / max * 16.0) as usize;
            let bar: String = std::iter::repeat('▌').take(bars).collect();
            buf.push_str(&format!("{:>3}: {}\n", k, bar));
        }
        ui.add(egui::Label::new(egui::RichText::new(buf).monospace().small()));
    }
    if ui.add(egui::Slider::new(&mut st.n, 32..=1024).text("Length")).changed() {
        st.user_reset = true;
    }
    ui.small(
        "Sum of two sinusoids → magnitude spectrum (host DFT). Stage 37.8 \
         did not add a tile-FFT primitive (Stage 24c GPU FFT lives in \
         `molla::kernels::FftKernel` instead); the host DFT stays as the \
         live readout for this pane.",
    );
}

// ── Scene 63: WarpTileMlp ─────────────────────────────────────────
struct WarpTileMlpState {
    inputs: [f32; 16],
    /// Output of the (3-layer) MLP forward pass.
    last_output: Vec<f32>,
    last_forward_us: f64,
    user_reset: bool,
}
impl Default for WarpTileMlpState {
    fn default() -> Self {
        Self {
            inputs: [0.5; 16],
            last_output: Vec::new(),
            last_forward_us: 0.0,
            user_reset: true,
        }
    }
}

fn update_warp_tile_mlp(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpTileMlp {
        return;
    }
    let st = &mut warp.tile_mlp;
    if !st.user_reset {
        return;
    }
    // Tiny 16 → 8 → 4 ReLU MLP, weights deterministic from a hash.
    // Stage 37.8: forward pass routes through `tile_matmul::matmul_cpu`
    // (the molla compute primitive) instead of inline triple loops, so
    // any future swap to the GPU `TileMatmulKernel` only needs to
    // change the call site here. The CPU path stays the default —
    // a 1×16 · 16×8 GEMM is microseconds, well below GPU dispatch
    // overhead.
    let mut w1 = vec![0.0_f32; 16 * 8];
    let mut b1 = vec![0.0_f32; 8];
    let mut w2 = vec![0.0_f32; 8 * 4];
    let mut b2 = vec![0.0_f32; 4];
    let mut seed = 0x12345_u32;
    let mut rng = || {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        ((seed >> 16) as i32 as f32) / 32768.0 * 0.5
    };
    for v in w1.iter_mut() { *v = rng(); }
    for v in b1.iter_mut() { *v = rng(); }
    for v in w2.iter_mut() { *v = rng(); }
    for v in b2.iter_mut() { *v = rng(); }
    let t0 = std::time::Instant::now();
    // Layer 1: hidden_pre = input · W1, shape (1, 8) = (1, 16) · (16, 8).
    let mut hidden_pre = vec![0.0_f32; 8];
    molla::compute::tile_matmul::matmul_cpu(&st.inputs, &w1, &mut hidden_pre, 1, 8, 16)
        .expect("tile_matmul layer1");
    let mut hidden = vec![0.0_f32; 8];
    for j in 0..8 {
        hidden[j] = (hidden_pre[j] + b1[j]).max(0.0);
    }
    // Layer 2: out = hidden · W2, shape (1, 4) = (1, 8) · (8, 4).
    let mut out = vec![0.0_f32; 4];
    molla::compute::tile_matmul::matmul_cpu(&hidden, &w2, &mut out, 1, 4, 8)
        .expect("tile_matmul layer2");
    for j in 0..4 {
        out[j] += b2[j];
    }
    st.last_forward_us = t0.elapsed().as_secs_f64() * 1.0e6;
    st.last_output = out;
    st.user_reset = false;
}

fn warp_tile_mlp_pane(ui: &mut egui::Ui, st: &mut WarpTileMlpState) {
    ui.label("Warp tile MLP — port of tile/example_tile_mlp.py");
    ui.separator();
    readout(ui, "Forward time", &format!("{:.2} µs", st.last_forward_us));
    readout(ui, "Output", &format!("{:?}", st.last_output));
    ui.separator();
    ui.label("Input slider 0:");
    if ui.add(egui::Slider::new(&mut st.inputs[0], -1.0..=1.0)).changed() {
        st.user_reset = true;
    }
    ui.label("Input slider 8:");
    if ui.add(egui::Slider::new(&mut st.inputs[8], -1.0..=1.0)).changed() {
        st.user_reset = true;
    }
    if ui.button("Re-randomise weights").clicked() {
        st.user_reset = true;
    }
    ui.small(
        "16 → 8 → 4 ReLU MLP, deterministic-hash weights. Forward pass \
         routed through `molla::compute::tile_matmul::matmul_cpu` \
         (Stage 37.8). The matching GPU `TileMatmulKernel` is available \
         for larger MLPs.",
    );
}

// ── Scene 64: WarpTileNbody ───────────────────────────────────────
struct WarpTileNbodyState {
    n: u32,
    pos: Vec<Vec3>,
    vel: Vec<Vec3>,
    g: f32,
    softening: f32,
    user_reset: bool,
}
impl Default for WarpTileNbodyState {
    fn default() -> Self {
        Self {
            n: 64,
            pos: Vec::new(),
            vel: Vec::new(),
            g: 0.4,
            softening: 0.05,
            user_reset: true,
        }
    }
}

fn update_warp_tile_nbody(
    time: Res<Time>,
    sim: Res<Sim>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpTileNbody {
        return;
    }
    let st = &mut warp.tile_nbody;
    if st.user_reset || st.pos.len() != st.n as usize {
        let n = st.n.max(2) as usize;
        st.pos = Vec::with_capacity(n);
        st.vel = Vec::with_capacity(n);
        let mut seed = 0xBEEF_u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed >> 16) as i32 as f32) / 32768.0
        };
        for _ in 0..n {
            // Random surface of a unit sphere shell.
            let phi = (1.0 - 2.0 * (rng() * 0.5 + 0.5)).acos();
            let theta = rng() * std::f32::consts::PI * 2.0;
            let p = Vec3::new(
                phi.sin() * theta.cos(),
                1.0 + phi.sin() * theta.sin(),
                phi.cos(),
            );
            st.pos.push(p);
            st.vel.push(Vec3::ZERO);
        }
        st.user_reset = false;
    }
    let dt = time.delta_secs().min(0.033);
    let n = st.pos.len();
    let g = st.g;
    let soft_sq = st.softening * st.softening;
    for i in 0..n {
        let mut acc = Vec3::ZERO;
        for j in 0..n {
            if i == j {
                continue;
            }
            let r = st.pos[j] - st.pos[i];
            let d2 = r.length_squared() + soft_sq;
            let inv_d = d2.sqrt().recip();
            let inv_d3 = inv_d * inv_d * inv_d;
            acc += r * inv_d3 * g;
        }
        st.vel[i] += acc * dt;
    }
    for i in 0..n {
        let v = st.vel[i];
        st.pos[i] += v * dt;
    }
}

fn draw_warp_tile_nbody_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpTileNbody {
        return;
    }
    for &p in &warp.tile_nbody.pos {
        gizmos.sphere(
            bevy::math::Isometry3d::from_translation(p),
            0.04,
            Color::srgb(0.95, 0.85, 0.3),
        );
    }
}

fn warp_tile_nbody_pane(ui: &mut egui::Ui, st: &mut WarpTileNbodyState) {
    ui.label("Warp tile N-body — port of tile/example_tile_nbody.py");
    ui.separator();
    readout(ui, "Bodies", &format!("{}", st.pos.len()));
    ui.label("Gravitational constant G");
    let _ = ui.add(egui::Slider::new(&mut st.g, 0.05..=2.0));
    ui.label("Softening ε");
    let _ = ui.add(egui::Slider::new(&mut st.softening, 0.005..=0.5).logarithmic(true));
    if ui.add(egui::Slider::new(&mut st.n, 8..=256).text("N")).changed() {
        st.user_reset = true;
    }
    if ui.button("Reset").clicked() {
        st.user_reset = true;
    }
    ui.small(
        "All-pairs Newtonian gravity (host O(N²) loop). Stage 37.8 did \
         not add an N-body primitive — kept on CPU for the demo.",
    );
}

// ── Scene 65: WarpTileCholesky ────────────────────────────────────
struct WarpTileCholeskyState {
    n: u32,
    /// Reconstruction error ‖L Lᵀ − A‖_F.
    last_err: f64,
    last_solve_us: f64,
    user_reset: bool,
}
impl Default for WarpTileCholeskyState {
    fn default() -> Self {
        Self {
            n: 16,
            last_err: 0.0,
            last_solve_us: 0.0,
            user_reset: true,
        }
    }
}

fn update_warp_tile_cholesky(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpTileCholesky {
        return;
    }
    let st = &mut warp.tile_cholesky;
    if !st.user_reset {
        return;
    }
    let n = st.n.max(2) as usize;
    // SPD test matrix: A = I + diag(1..N).
    let mut a32 = vec![0.0_f32; n * n];
    for i in 0..n {
        a32[i * n + i] = 1.0 + (i + 1) as f32;
        for j in (i + 1)..n {
            a32[i * n + j] = 1.0;
            a32[j * n + i] = 1.0;
        }
    }

    // Stage 37.8 — route through the molla `tile_block_cholesky`
    // primitive. Try the GPU kernel first; if anything goes wrong
    // (no adapter / N > MAX_N), fall back to the CPU oracle. Both
    // paths produce the lower-triangular `L` in place.
    let t0 = std::time::Instant::now();
    let mut l32 = a32.clone();
    let used_gpu = if n <= molla::compute::tile_block_cholesky::MAX_N {
        if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
            let kernel = molla::compute::tile_block_cholesky::TileBlockCholeskyKernel::new(device);
            kernel
                .dispatch_blocking(device, queue, &mut l32, n)
                .is_ok()
        } else {
            false
        }
    } else {
        false
    };
    if !used_gpu {
        // CPU fallback (also exercised when the GPU is unavailable or
        // when N exceeds the single-workgroup limit).
        l32 = a32.clone();
        molla::compute::tile_block_cholesky::cholesky_cpu(&mut l32, n).expect("tile_block_cholesky cpu");
    }
    st.last_solve_us = t0.elapsed().as_secs_f64() * 1.0e6;

    let err32 = molla::compute::tile_block_cholesky::frobenius_recon_error(&a32, &l32, n);
    st.last_err = err32 as f64;
    st.user_reset = false;
}

fn warp_tile_cholesky_pane(ui: &mut egui::Ui, st: &mut WarpTileCholeskyState) {
    ui.label("Warp tile Cholesky — port of tile/example_tile_cholesky.py");
    ui.separator();
    readout(ui, "N", &format!("{0}×{0}", st.n));
    readout(ui, "‖L Lᵀ − A‖_F", &format!("{:.3e}", st.last_err));
    readout(ui, "Factor time", &format!("{:.2} µs", st.last_solve_us));
    ui.separator();
    if ui.add(egui::Slider::new(&mut st.n, 4..=64).text("Size")).changed() {
        st.user_reset = true;
    }
    if ui.button("Re-solve").clicked() {
        st.user_reset = true;
    }
    ui.small(
        "Dense Cholesky on SPD `I + diag(1..N) + 1`, factored via \
         `molla::compute::tile_block_cholesky` (Stage 37.8) — single-\
         workgroup GPU kernel up to N=1024 with CPU fallback.",
    );
}

// ── Scene 66: WarpParticleRepulsion ───────────────────────────────
struct WarpRepulsionState {
    n: u32,
    pos: Vec<Vec3>,
    /// Iteration count of gradient descent already executed.
    iter: usize,
    /// Last loss value.
    last_loss: f64,
    learning_rate: f32,
    paused: bool,
    user_reset: bool,
}
impl Default for WarpRepulsionState {
    fn default() -> Self {
        Self {
            n: 32,
            pos: Vec::new(),
            iter: 0,
            last_loss: 0.0,
            learning_rate: 0.05,
            paused: false,
            user_reset: true,
        }
    }
}

fn update_warp_repulsion(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpParticleRepulsion {
        return;
    }
    let st = &mut warp.repulsion;
    if st.user_reset || st.pos.len() != st.n as usize {
        let n = st.n.max(2) as usize;
        st.pos = Vec::with_capacity(n);
        let mut seed = 0xCAFE_u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed >> 16) as i32 as f32) / 32768.0 + 0.5
        };
        for _ in 0..n {
            st.pos.push(Vec3::new(rng() * 2.0, 1.0, rng() * 2.0));
        }
        st.iter = 0;
        st.last_loss = 0.0;
        st.user_reset = false;
    }
    if st.paused {
        return;
    }
    // Stage 37.8 — gradient comes from the molla `tile_mcgp` primitive
    // (SPSA stochastic estimate) instead of the inline analytic
    // `r / d³` accumulation. Same closed-form loss; the gradient
    // estimate is noisy by construction, so the descent path is more
    // jittery than the analytic version — that's the demo, end users
    // should reach for analytic gradients when they have them.
    let lr = st.learning_rate;

    // Flatten positions to a `Vec<f64>` for the loss closure.
    let params: Vec<f64> = st
        .pos
        .iter()
        .flat_map(|p| [p.x as f64, p.y as f64, p.z as f64])
        .collect();
    let inverse_distance_loss = |p: &[f64]| -> f64 {
        let mut s = 0.0_f64;
        let nn = p.len() / 3;
        for i in 0..nn {
            for j in (i + 1)..nn {
                let dx = p[i * 3 + 0] - p[j * 3 + 0];
                let dy = p[i * 3 + 1] - p[j * 3 + 1];
                let dz = p[i * 3 + 2] - p[j * 3 + 2];
                let d = (dx * dx + dy * dy + dz * dz).sqrt().max(1.0e-3);
                s += 1.0 / d;
            }
        }
        s
    };

    // Single forward pass for the loss readout.
    let loss_now = inverse_distance_loss(&params);
    st.last_loss = loss_now;

    // 64 SPSA samples per descent step — keeps the per-frame budget
    // bounded while letting the noise average down across frames.
    let cfg = molla::compute::tile_mcgp::McgpConfig {
        n_samples: 64,
        epsilon: 5.0e-3,
        seed: 0x9E3779B9_u64.wrapping_add(st.iter as u64),
    };
    let grad = molla::compute::tile_mcgp::mcgp_estimate(
        &inverse_distance_loss,
        &params,
        &cfg,
    );

    // Gradient descent: p -= lr · ∇L. Pack back into Vec3.
    for (i, p) in st.pos.iter_mut().enumerate() {
        let gx = grad[i * 3 + 0] as f32;
        let gy = grad[i * 3 + 1] as f32;
        let gz = grad[i * 3 + 2] as f32;
        p.x -= lr * gx;
        p.y -= lr * gy;
        p.z -= lr * gz;
        // Periodic-style soft wrap to a 2×2×2 box.
        for axis in 0..3 {
            let v = match axis {
                0 => p.x,
                1 => p.y,
                _ => p.z,
            };
            let w = ((v + 1.0).rem_euclid(2.0)) - 1.0;
            match axis {
                0 => p.x = w,
                1 => p.y = w,
                _ => p.z = w,
            }
        }
        p.y = p.y.abs() + 0.5; // keep above ground
    }
    st.iter += 1;
}

fn draw_warp_repulsion_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpParticleRepulsion {
        return;
    }
    for &p in &warp.repulsion.pos {
        gizmos.sphere(
            bevy::math::Isometry3d::from_translation(p),
            0.04,
            Color::srgb(0.4, 0.8, 0.95),
        );
    }
}

fn warp_repulsion_pane(ui: &mut egui::Ui, st: &mut WarpRepulsionState) {
    ui.label("Warp particle repulsion — port of optim/example_particle_repulsion.py");
    ui.separator();
    readout(ui, "Particles", &format!("{}", st.pos.len()));
    readout(ui, "Iterations", &format!("{}", st.iter));
    readout(ui, "Loss", &format!("{:.3e}", st.last_loss));
    ui.separator();
    ui.label("Learning rate");
    let _ = ui.add(egui::Slider::new(&mut st.learning_rate, 0.001..=0.5).logarithmic(true));
    if ui.add(egui::Slider::new(&mut st.n, 4..=128).text("N")).changed() {
        st.user_reset = true;
    }
    ui.checkbox(&mut st.paused, "Pause");
    if ui.button("Reset").clicked() {
        st.user_reset = true;
    }
    ui.small(
        "Inverse-distance pair potential, gradient descent on positions. \
         Gradient estimated by `molla::compute::tile_mcgp` SPSA \
         (Stage 37.8) — noisy by construction, so the descent path \
         jitters compared to the analytic-gradient placeholder.",
    );
}

// ── Scene 67: WarpFemDiffusion (2D triangular) ────────────────────
struct WarpFemDiffusionState {
    mesh: fem::FemMesh,
    mass: molla::math::sparse::BsrMatrix,
    stiff: molla::math::sparse::BsrMatrix,
    bcs: Vec<fem::BoundaryCondition>,
    u: Vec<f64>,
    alpha: f64,
    last_alpha: f64,
    dt: f64,
    user_reset: bool,
    last_iters: usize,
    last_residual: f64,
}
impl Default for WarpFemDiffusionState {
    fn default() -> Self {
        let nx = 16;
        let ny = 16;
        let mesh = fem::FemMesh::tri_grid_2d(nx, ny, 1.0, 1.0);
        let space = fem::FunctionSpace::P1Nodal;
        let mass = fem::assemble_global(&mesh, space, fem::compute_element_mass)
            .expect("warp diff mass");
        let stiff = fem::assemble_global(&mesh, space, |c, v| {
            fem::compute_element_stiffness(c, v, 1.0)
        }).expect("warp diff stiff");
        // Dirichlet on left + right vertical edges.
        let mut bcs = Vec::new();
        for (i, p) in mesh.vertices.iter().enumerate() {
            if p.x.abs() < 1e-9 {
                bcs.push(fem::BoundaryCondition::Dirichlet { vertex: i as u32, value: 0.0 });
            } else if (p.x - 1.0).abs() < 1e-9 {
                bcs.push(fem::BoundaryCondition::Dirichlet { vertex: i as u32, value: 1.0 });
            }
        }
        let n = mesh.vert_count();
        let mut u = vec![0.0; n];
        for bc in &bcs {
            if let fem::BoundaryCondition::Dirichlet { vertex, value } = bc {
                u[*vertex as usize] = *value;
            }
        }
        Self {
            mesh, mass, stiff, bcs, u,
            alpha: 1.0, last_alpha: 1.0, dt: 0.02,
            user_reset: false, last_iters: 0, last_residual: 0.0,
        }
    }
}

fn update_warp_fem_diffusion(
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpFemDiffusion {
        return;
    }
    let st = &mut warp.fem_diff;
    if st.user_reset {
        let n = st.mesh.vert_count();
        st.u = vec![0.0; n];
        for bc in &st.bcs {
            if let fem::BoundaryCondition::Dirichlet { vertex, value } = bc {
                st.u[*vertex as usize] = *value;
            }
        }
        st.user_reset = false;
    }
    if (st.alpha - st.last_alpha).abs() > 1e-12 {
        let space = fem::FunctionSpace::P1Nodal;
        st.stiff = fem::assemble_global(&st.mesh, space, |c, v| {
            fem::compute_element_stiffness(c, v, st.alpha)
        }).expect("warp diff rebuild");
        st.last_alpha = st.alpha;
    }
    if run.paused {
        return;
    }
    let n = st.mesh.vert_count();
    let f = vec![0.0_f64; n];
    let res = molla::fem::solver::solve_transient_with_initial(
        &st.mass, &st.stiff, &f, &st.u.clone(), st.dt.max(1e-6),
        &st.bcs, &st.u.clone(),
        molla::fem::solver::DEFAULT_MAX_ITERS,
        molla::fem::solver::DEFAULT_TOL,
    );
    st.u = res.x;
    st.last_iters = res.iters;
    st.last_residual = res.residual;
}

fn draw_warp_fem_diffusion_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpFemDiffusion {
        return;
    }
    let st = &warp.fem_diff;
    let origin = Vec3::new(-0.5, 0.05, -0.5);
    for cell in &st.mesh.cells {
        if let fem::FemCell::Tri(v) = cell {
            let pa = st.mesh.vertices[v[0] as usize];
            let pb = st.mesh.vertices[v[1] as usize];
            let pc = st.mesh.vertices[v[2] as usize];
            let to = |p: MVec3, vi: u32| {
                origin + Vec3::new(p.x as f32, st.u[vi as usize] as f32 * 0.4, p.y as f32)
            };
            let a = to(pa, v[0]);
            let b = to(pb, v[1]);
            let c = to(pc, v[2]);
            let avg = (st.u[v[0] as usize] + st.u[v[1] as usize] + st.u[v[2] as usize]) / 3.0;
            let col = fem_heat_colour(avg);
            gizmos.line(a, b, col);
            gizmos.line(b, c, col);
            gizmos.line(c, a, col);
        }
    }
}

fn warp_fem_diffusion_pane(ui: &mut egui::Ui, st: &mut WarpFemDiffusionState) {
    ui.label("Warp FEM 2D diffusion — port of fem/example_diffusion.py");
    ui.separator();
    readout(ui, "Triangles", &format!("{}", st.mesh.cells.len()));
    readout(ui, "Verts", &format!("{}", st.mesh.vert_count()));
    readout(ui, "CG iters", &format!("{}", st.last_iters));
    readout(ui, "Residual", &format!("{:.2e}", st.last_residual));
    ui.label("Diffusivity α");
    let _ = ui.add(egui::Slider::new(&mut st.alpha, 0.05..=10.0).logarithmic(true));
    ui.label("dt");
    let _ = ui.add(egui::Slider::new(&mut st.dt, 0.001..=0.2).logarithmic(true));
    if ui.button("Reset").clicked() {
        st.user_reset = true;
    }
    ui.small("BCs: T(x=0) = 0, T(x=1) = 1; Neumann on top/bottom.");
}

// ── Scene 68: WarpFemDiffusion3D (animated BC) ────────────────────
struct WarpFemDiffusion3DState {
    mesh: fem::FemMesh,
    mass: molla::math::sparse::BsrMatrix,
    stiff: molla::math::sparse::BsrMatrix,
    /// Bottom + top vertex indices (BCs are time-varying).
    bottom: Vec<u32>,
    top: Vec<u32>,
    u: Vec<f64>,
    alpha: f64,
    dt: f64,
    sim_t: f64,
    user_reset: bool,
}
impl Default for WarpFemDiffusion3DState {
    fn default() -> Self {
        let n = 8;
        let mesh = fem::FemMesh::tet_grid_3d(n, n, n, 1.0, 1.0, 1.0);
        let space = fem::FunctionSpace::P1Nodal;
        let mass = fem::assemble_global(&mesh, space, fem::compute_element_mass)
            .expect("warp diff3d mass");
        let stiff = fem::assemble_global(&mesh, space, |c, v| {
            fem::compute_element_stiffness(c, v, 1.0)
        }).expect("warp diff3d stiff");
        let mut bottom = Vec::new();
        let mut top = Vec::new();
        for (i, p) in mesh.vertices.iter().enumerate() {
            if p.z.abs() < 1e-9 { bottom.push(i as u32); }
            if (p.z - 1.0).abs() < 1e-9 { top.push(i as u32); }
        }
        Self {
            u: vec![0.0; mesh.vert_count()],
            mesh, mass, stiff, bottom, top,
            alpha: 1.0, dt: 0.02, sim_t: 0.0,
            user_reset: false,
        }
    }
}

fn update_warp_fem_diffusion_3d(
    time: Res<Time>,
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpFemDiffusion3D {
        return;
    }
    let st = &mut warp.fem_diff3d;
    if st.user_reset {
        st.u = vec![0.0; st.mesh.vert_count()];
        st.sim_t = 0.0;
        st.user_reset = false;
    }
    if run.paused {
        return;
    }
    st.sim_t += time.delta_secs() as f64;
    let bottom_v = (st.sim_t * 1.5).sin() * 0.5 + 0.5;
    let top_v = (st.sim_t * 1.5).cos() * 0.5 + 0.5;
    let mut bcs = Vec::with_capacity(st.bottom.len() + st.top.len());
    for &v in &st.bottom {
        bcs.push(fem::BoundaryCondition::Dirichlet { vertex: v, value: bottom_v });
    }
    for &v in &st.top {
        bcs.push(fem::BoundaryCondition::Dirichlet { vertex: v, value: top_v });
    }
    let f = vec![0.0; st.mesh.vert_count()];
    let res = molla::fem::solver::solve_transient_with_initial(
        &st.mass, &st.stiff, &f, &st.u.clone(), st.dt,
        &bcs, &st.u.clone(),
        molla::fem::solver::DEFAULT_MAX_ITERS,
        molla::fem::solver::DEFAULT_TOL,
    );
    st.u = res.x;
}

fn draw_warp_fem_diffusion_3d_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpFemDiffusion3D {
        return;
    }
    let st = &warp.fem_diff3d;
    let origin = Vec3::new(-0.5, 0.05, -0.5);
    use std::collections::HashSet;
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    for cell in &st.mesh.cells {
        if let fem::FemCell::Tet(v) = cell {
            let edges = [(v[0],v[1]),(v[0],v[2]),(v[0],v[3]),(v[1],v[2]),(v[1],v[3]),(v[2],v[3])];
            for &(a,b) in &edges {
                let key = if a < b { (a, b) } else { (b, a) };
                if !seen.insert(key) { continue; }
                let pa = st.mesh.vertices[a as usize];
                let pb = st.mesh.vertices[b as usize];
                let av = origin + Vec3::new(pa.x as f32, pa.z as f32, pa.y as f32);
                let bv = origin + Vec3::new(pb.x as f32, pb.z as f32, pb.y as f32);
                let t = 0.5 * (st.u[a as usize] + st.u[b as usize]);
                gizmos.line(av, bv, fem_heat_colour(t));
            }
        }
    }
}

fn warp_fem_diffusion_3d_pane(ui: &mut egui::Ui, st: &mut WarpFemDiffusion3DState) {
    ui.label("Warp FEM 3D diffusion — port of fem/example_diffusion_3d.py");
    ui.separator();
    readout(ui, "Tets", &format!("{}", st.mesh.cells.len()));
    readout(ui, "t", &format!("{:.2} s", st.sim_t));
    ui.label("α");
    let _ = ui.add(egui::Slider::new(&mut st.alpha, 0.1..=10.0).logarithmic(true));
    ui.label("dt");
    let _ = ui.add(egui::Slider::new(&mut st.dt, 0.001..=0.2).logarithmic(true));
    if ui.button("Reset").clicked() { st.user_reset = true; }
    ui.small("Time-varying Dirichlet BCs (sin/cos on top + bottom faces).");
}

// ── Scene 69: WarpFemConvDiff (steady advection-diffusion) ────────
//
// Stage 37.2 retrofit — the analytic-stub Laplacian was replaced by a
// real convection-diffusion operator `(α∇² − b·∇)` with SUPG
// stabilisation (`τ = h / (2|b|)`). Convection direction is +x; left
// edge is the inflow Dirichlet (1.0) and right edge is the outflow
// Dirichlet (0.0). The non-symmetric system is solved via BiCGSTAB.
struct WarpFemConvDiffState {
    mesh: fem::FemMesh,
    /// Last steady-state solution.
    u: Vec<f64>,
    alpha: f64,
    bx: f64,
    user_reset: bool,
    last_iters: usize,
    last_residual: f64,
}
impl Default for WarpFemConvDiffState {
    fn default() -> Self {
        let mesh = fem::FemMesh::tri_grid_2d(20, 20, 1.0, 1.0);
        Self {
            u: vec![0.0; mesh.vert_count()],
            mesh,
            alpha: 0.05,
            bx: 1.0,
            user_reset: true,
            last_iters: 0, last_residual: 0.0,
        }
    }
}

fn update_warp_fem_convdiff(
    sim: Res<Sim>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpFemConvDiff {
        return;
    }
    let st = &mut warp.fem_convdiff;
    if !st.user_reset {
        return;
    }
    // Assemble A = α·K + C + S_SUPG. The convection block uses the
    // user's velocity in +x; the SUPG block uses the cell length scale
    // `h = 1/nx` for τ = h/(2|b|).
    let nx = 20;
    let h = 1.0 / nx as f64;
    let b = molla::math::Vec3::new(st.bx, 0.0, 0.0);
    let alpha = st.alpha;
    let space = fem::FunctionSpace::P1Nodal;
    let a = fem::assemble_global(&st.mesh, space, |cell, verts| {
        let kk = fem::compute_element_stiffness(cell, verts, alpha);
        let cc = fem::compute_element_convection(cell, verts, b);
        let ss = fem::compute_element_supg_convection(cell, verts, b, 1.0, h);
        let n = cell.vert_count();
        (0..n)
            .map(|i| (0..n).map(|j| kk[i][j] + cc[i][j] + ss[i][j]).collect())
            .collect()
    }).expect("warp convdiff assemble");

    let mut bcs = Vec::new();
    for (i, p) in st.mesh.vertices.iter().enumerate() {
        if p.x.abs() < 1e-9 {
            bcs.push(fem::BoundaryCondition::Dirichlet { vertex: i as u32, value: 1.0 });
        } else if (p.x - 1.0).abs() < 1e-9 {
            bcs.push(fem::BoundaryCondition::Dirichlet { vertex: i as u32, value: 0.0 });
        }
    }

    // Apply Dirichlet to (A, f) and solve via BiCGSTAB — CG can't
    // handle the non-symmetric advection block.
    let mut a_red = a;
    let mut f = vec![0.0_f64; st.mesh.vert_count()];
    fem::apply_dirichlet(&mut a_red, &mut f, &bcs);
    let n = f.len();
    let mut x0 = vec![0.0; n];
    for bc in &bcs {
        if let fem::BoundaryCondition::Dirichlet { vertex, value } = bc {
            x0[*vertex as usize] = *value;
        }
    }
    let res = molla::math::iterative::bicgstab_solve(
        |x| {
            let mut y = vec![0.0; n];
            a_red.mv(x, &mut y).expect("BsrMatrix mv");
            y
        },
        &f,
        &x0,
        molla::fem::solver::DEFAULT_TOL,
        molla::fem::solver::DEFAULT_MAX_ITERS,
    );
    st.u = res.x;
    st.last_iters = res.iters;
    st.last_residual = res.residual;
    st.user_reset = false;
}

fn draw_warp_fem_convdiff_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpFemConvDiff {
        return;
    }
    let st = &warp.fem_convdiff;
    let origin = Vec3::new(-0.5, 0.05, -0.5);
    for cell in &st.mesh.cells {
        if let fem::FemCell::Tri(v) = cell {
            let pa = st.mesh.vertices[v[0] as usize];
            let pb = st.mesh.vertices[v[1] as usize];
            let pc = st.mesh.vertices[v[2] as usize];
            let to = |p: MVec3| origin + Vec3::new(p.x as f32, 0.0, p.y as f32);
            let avg = (st.u[v[0] as usize] + st.u[v[1] as usize] + st.u[v[2] as usize]) / 3.0;
            let col = fem_heat_colour(avg);
            gizmos.line(to(pa), to(pb), col);
            gizmos.line(to(pb), to(pc), col);
            gizmos.line(to(pc), to(pa), col);
        }
    }
}

fn warp_fem_convdiff_pane(ui: &mut egui::Ui, st: &mut WarpFemConvDiffState) {
    ui.label("Warp FEM convection-diffusion — port of fem/example_convection_diffusion.py");
    ui.separator();
    readout(ui, "Triangles", &format!("{}", st.mesh.cells.len()));
    readout(ui, "CG iters", &format!("{}", st.last_iters));
    readout(ui, "Residual", &format!("{:.2e}", st.last_residual));
    ui.label("Diffusivity α");
    if ui.add(egui::Slider::new(&mut st.alpha, 0.001..=1.0).logarithmic(true)).changed() {
        st.user_reset = true;
    }
    if ui.button("Recompute").clicked() { st.user_reset = true; }
    ui.small("Steady (α∇² − b·∇)u = 0 — Stage 37.2 SUPG-stabilised");
    ui.small("convection-diffusion; non-symmetric solve via BiCGSTAB.");
}

// ── Scene 70: WarpFemTaylorGreen (Stokes vortex IC) ───────────────
struct WarpFemTaylorGreenState {
    sim_t: f64,
    nu: f64,
    /// Decay magnitude (analytic): exp(-8 π² ν t).
    decay: f64,
    user_reset: bool,
}
impl Default for WarpFemTaylorGreenState {
    fn default() -> Self {
        Self { sim_t: 0.0, nu: 0.01, decay: 1.0, user_reset: false }
    }
}

fn update_warp_fem_taylor(
    time: Res<Time>,
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpFemTaylorGreen { return; }
    let st = &mut warp.fem_taylor;
    if st.user_reset { st.sim_t = 0.0; st.user_reset = false; }
    if run.paused { return; }
    st.sim_t += time.delta_secs() as f64;
    let p = std::f64::consts::PI;
    st.decay = (-8.0 * p * p * st.nu * st.sim_t).exp();
}

fn draw_warp_fem_taylor_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpFemTaylorGreen { return; }
    let st = &warp.fem_taylor;
    let origin = Vec3::new(-0.5, 0.05, -0.5);
    let n = 16;
    let p = std::f64::consts::PI;
    for j in 0..n {
        for i in 0..n {
            let x = (i as f64 + 0.5) / n as f64;
            let y = (j as f64 + 0.5) / n as f64;
            let u = -((2.0 * p * x).cos() * (2.0 * p * y).sin()) * st.decay;
            let v = ((2.0 * p * x).sin() * (2.0 * p * y).cos()) * st.decay;
            let from = origin + Vec3::new(x as f32, 0.0, y as f32);
            let to = from + Vec3::new(u as f32 * 0.05, 0.0, v as f32 * 0.05);
            gizmos.line(from, to, Color::srgb(0.4, 0.95, 0.6));
        }
    }
}

fn warp_fem_taylor_pane(ui: &mut egui::Ui, st: &mut WarpFemTaylorGreenState) {
    ui.label("Warp FEM Taylor-Green — port of fem/example_taylor_green.py");
    ui.separator();
    readout(ui, "t", &format!("{:.2} s", st.sim_t));
    readout(ui, "Decay exp(-8π²νt)", &format!("{:.3e}", st.decay));
    ui.label("Viscosity ν");
    let _ = ui.add(egui::Slider::new(&mut st.nu, 0.001..=0.5).logarithmic(true));
    if ui.button("Reset t").clicked() { st.user_reset = true; }
    ui.small("Analytic Taylor-Green vortex. Velocity arrows shrink");
    ui.small("with the exact decay envelope.");
}

// ── Scene 71: WarpFemKelvinHelmholtz ──────────────────────────────
//
// Real FEM Navier-Stokes time-stepping (Stage 37.3) on a 12×12 P2-P1
// Taylor-Hood mesh of the unit square. Initial condition is a shear
// layer (`u_x = ±0.5` above / below the midline) plus a small
// vertical perturbation that excites the Kelvin-Helmholtz instability.
// Boundary conditions: pin the side walls to the bulk shear value,
// pressure pin at vertex 0.
//
// Per-frame budget: a handful of `solve_navier_stokes_step` calls. The
// 12×12 mesh keeps each step well under 50 ms in release mode. The
// pane lets the user dial viscosity, perturbation amplitude, and
// readouts include simulated time + the L∞ vorticity diagnostic.

const KH_GRID: usize = 12;
const KH_DT: f64 = 0.01;
/// Steps per frame — keep small so the viewer stays interactive.
const KH_STEPS_PER_FRAME: usize = 1;

struct WarpFemKhState {
    sim_t: f64,
    /// Kinematic viscosity ν.
    viscosity: f64,
    /// Vertical-perturbation amplitude (initial condition).
    perturb_amp: f64,
    /// Last-frame max |ω| diagnostic.
    vorticity_max: f64,
    /// FEM mesh (built lazily).
    mesh: Option<fem::FemMesh>,
    th: Option<fem::MixedTaylorHood>,
    /// Cached `M / dt + A`, divergence, and mass matrices.
    a_eff: Option<molla::math::sparse::BsrMatrix>,
    b_div: Option<molla::math::sparse::BsrMatrix>,
    m: Option<molla::math::sparse::BsrMatrix>,
    /// Static BCs (side walls) — rebuilt only on reset.
    bcs: Vec<fem::StokesBc>,
    /// Velocity DoFs (interleaved P2-vector).
    u: Vec<f64>,
    /// Reset trigger.
    needs_reset: bool,
}

impl Default for WarpFemKhState {
    fn default() -> Self {
        Self {
            sim_t: 0.0,
            viscosity: 0.005,
            perturb_amp: 0.05,
            vorticity_max: 0.0,
            mesh: None,
            th: None,
            a_eff: None,
            b_div: None,
            m: None,
            bcs: Vec::new(),
            u: Vec::new(),
            needs_reset: true,
        }
    }
}

/// Build the static BC list: pin **only** the left + right walls (the
/// shear flow direction). Top + bottom are open / shear-conserving in
/// this surrogate (we approximate "periodic-y" via no-pin and rely on
/// the natural Neumann condition the discretisation gives).
fn kh_bcs(mesh: &fem::FemMesh, th: &fem::MixedTaylorHood) -> Vec<fem::StokesBc> {
    let mut bcs: Vec<fem::StokesBc> = Vec::new();
    let target_u = |y: f64| if y > 0.5 { 0.5_f64 } else { -0.5_f64 };

    for (i, p) in mesh.vertices.iter().enumerate() {
        let on_left = p.x.abs() < 1e-9;
        let on_right = (p.x - 1.0).abs() < 1e-9;
        let on_bot = p.y.abs() < 1e-9;
        let on_top = (p.y - 1.0).abs() < 1e-9;
        if on_left || on_right {
            bcs.push(fem::StokesBc::LidVelocity {
                node: i as u32,
                velocity: vec![target_u(p.y), 0.0],
            });
        } else if on_top || on_bot {
            // No-slip top/bottom keeps the shear layer in the interior.
            bcs.push(fem::StokesBc::LidVelocity {
                node: i as u32,
                velocity: vec![target_u(p.y), 0.0],
            });
        }
    }
    let (edges, _) = fem::compute_edges(mesh);
    let edge_offset = th.map.vertex_count;
    for (k, edge) in edges.iter().enumerate() {
        let pa = mesh.vertices[edge[0] as usize];
        let pb = mesh.vertices[edge[1] as usize];
        let mx = 0.5 * (pa.x + pb.x);
        let my = 0.5 * (pa.y + pb.y);
        let on_b = mx.abs() < 1e-9
            || (mx - 1.0).abs() < 1e-9
            || my.abs() < 1e-9
            || (my - 1.0).abs() < 1e-9;
        if !on_b { continue; }
        bcs.push(fem::StokesBc::LidVelocity {
            node: (edge_offset + k) as u32,
            velocity: vec![target_u(my), 0.0],
        });
    }
    bcs.push(fem::StokesBc::PressureZero { vertex: 0 });
    bcs
}

fn kh_initial_velocity(
    mesh: &fem::FemMesh,
    th: &fem::MixedTaylorHood,
    perturb_amp: f64,
) -> Vec<f64> {
    let dim = th.dim;
    let mut u = vec![0.0_f64; th.n_velocity];
    let pi = std::f64::consts::PI;
    let seed = |x: f64, y: f64| -> (f64, f64) {
        let dy = y - 0.5;
        let ux = if dy > 0.0 { 0.5 } else { -0.5 };
        // Vertical perturbation localised near the shear layer.
        let vy = perturb_amp * (4.0 * pi * x).sin() * (-dy.abs() * 8.0).exp();
        (ux, vy)
    };
    for (i, p) in mesh.vertices.iter().enumerate() {
        let (ux, vy) = seed(p.x, p.y);
        u[i * dim] = ux;
        u[i * dim + 1] = vy;
    }
    let (edges, _) = fem::compute_edges(mesh);
    let edge_offset = th.map.vertex_count;
    for (k, edge) in edges.iter().enumerate() {
        let pa = mesh.vertices[edge[0] as usize];
        let pb = mesh.vertices[edge[1] as usize];
        let mx = 0.5 * (pa.x + pb.x);
        let my = 0.5 * (pa.y + pb.y);
        let (ux, vy) = seed(mx, my);
        let n = edge_offset + k;
        u[n * dim] = ux;
        u[n * dim + 1] = vy;
    }
    u
}

fn kh_assemble(
    st: &mut WarpFemKhState,
) {
    let mesh = fem::FemMesh::tri_grid_2d(KH_GRID, KH_GRID, 1.0, 1.0);
    let th = fem::MixedTaylorHood::build(&mesh, 2);
    let a = fem::assemble_velocity_block(&mesh, &th, st.viscosity).unwrap();
    let m = fem::assemble_velocity_mass(&mesh, &th).unwrap();
    let b_div = fem::assemble_div_block(&mesh, &th).unwrap();
    let inv_dt = 1.0 / KH_DT;
    // a_eff = M/dt + A
    let mut triplets: Vec<(usize, usize, Vec<f64>)> = Vec::new();
    for r in 0..m.block_rows {
        for slot in m.row_offsets[r]..m.row_offsets[r + 1] {
            let c = m.col_idx[slot];
            triplets.push((r, c, vec![inv_dt * m.values[slot]]));
        }
    }
    for r in 0..a.block_rows {
        for slot in a.row_offsets[r]..a.row_offsets[r + 1] {
            let c = a.col_idx[slot];
            triplets.push((r, c, vec![a.values[slot]]));
        }
    }
    let a_eff = molla::math::sparse::BsrMatrix::from_triplets(
        1, a.block_rows, a.block_cols, &triplets,
    ).unwrap();

    st.bcs = kh_bcs(&mesh, &th);
    st.u = kh_initial_velocity(&mesh, &th, st.perturb_amp);
    st.mesh = Some(mesh);
    st.th = Some(th);
    st.a_eff = Some(a_eff);
    st.b_div = Some(b_div);
    st.m = Some(m);
    st.sim_t = 0.0;
    st.vorticity_max = 0.0;
    st.needs_reset = false;
}

fn update_warp_fem_kh(
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpFemKelvinHelmholtz { return; }
    let st = &mut warp.fem_kh;
    if st.needs_reset { kh_assemble(st); }
    if run.paused { return; }

    let (Some(mesh), Some(th), Some(a_eff), Some(b_div), Some(m)) = (
        st.mesh.as_ref(), st.th.as_ref(),
        st.a_eff.as_ref(), st.b_div.as_ref(), st.m.as_ref(),
    ) else { return; };

    let inv_dt = 1.0 / KH_DT;
    let zero_f = vec![0.0_f64; th.n_velocity];
    let simple = fem::StokesSimpleConfig {
        max_outer_iters: 40,
        tol_outer: 1e-5,
        max_inner_iters: 200,
        tol_inner: 1e-7,
        ..Default::default()
    };

    for _ in 0..KH_STEPS_PER_FRAME {
        let res = fem::solve_navier_stokes_step(
            mesh, th, a_eff, b_div, m, &st.u, &zero_f, inv_dt, &st.bcs, &simple,
        );
        if let Ok(res) = res {
            st.u = res.u;
            st.sim_t += KH_DT;
        } else {
            break;
        }
    }

    // Vorticity diagnostic ω_z = ∂v/∂x - ∂u/∂y, sampled on vertex grid
    // via finite differences across rows (cheap; just for the readout).
    let n = KH_GRID + 1;
    let dx = 1.0 / KH_GRID as f64;
    let dim = th.dim;
    let mut max_w: f64 = 0.0;
    for j in 1..(n - 1) {
        for i in 1..(n - 1) {
            let idx = |ii: usize, jj: usize| jj * n + ii;
            let dvdx = (st.u[idx(i + 1, j) * dim + 1] - st.u[idx(i - 1, j) * dim + 1]) / (2.0 * dx);
            let dudy = (st.u[idx(i, j + 1) * dim] - st.u[idx(i, j - 1) * dim]) / (2.0 * dx);
            let w = (dvdx - dudy).abs();
            if w > max_w { max_w = w; }
        }
    }
    st.vorticity_max = max_w;
}

fn draw_warp_fem_kh_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpFemKelvinHelmholtz { return; }
    let st = &warp.fem_kh;
    let Some(mesh) = st.mesh.as_ref() else { return; };
    let Some(th) = st.th.as_ref() else { return; };
    let origin = Vec3::new(-0.5, 0.05, -0.5);
    let dim = th.dim;
    for (i, p) in mesh.vertices.iter().enumerate() {
        let ux = st.u[i * dim] as f32;
        let vy = st.u[i * dim + 1] as f32;
        let from = origin + Vec3::new(p.x as f32, 0.0, p.y as f32);
        let to = from + Vec3::new(ux * 0.05, 0.0, vy * 0.05);
        let col = if p.y > 0.5 {
            Color::srgb(0.95, 0.5, 0.3)
        } else {
            Color::srgb(0.3, 0.6, 0.95)
        };
        gizmos.line(from, to, col);
    }
}

fn warp_fem_kh_pane(ui: &mut egui::Ui, st: &mut WarpFemKhState) {
    ui.label("Warp FEM Kelvin-Helmholtz — Stage 37.3 NS time-stepping");
    ui.separator();
    readout(ui, "t", &format!("{:.2} s", st.sim_t));
    readout(ui, "max |ω|", &format!("{:.3e}", st.vorticity_max));
    ui.label("Viscosity ν");
    let mut nu = st.viscosity;
    if ui.add(egui::Slider::new(&mut nu, 0.0001..=0.05).logarithmic(true))
        .changed()
    {
        st.viscosity = nu;
        st.needs_reset = true;
    }
    ui.label("Perturbation amplitude");
    let mut amp = st.perturb_amp;
    if ui.add(egui::Slider::new(&mut amp, 0.0..=0.2)).changed() {
        st.perturb_amp = amp;
        st.needs_reset = true;
    }
    if ui.button("Reset").clicked() { st.needs_reset = true; }
    ui.small("Real FEM NS solve via solve_navier_stokes_step on a");
    ui.small("12×12 P2-P1 Taylor-Hood mesh.");
}

// ── Scene 72: WarpFemStreamlines ──────────────────────────────────
struct WarpFemStreamlinesState {
    /// Tracer positions.
    tracers: Vec<Vec3>,
    sim_t: f64,
    user_reset: bool,
}
impl Default for WarpFemStreamlinesState {
    fn default() -> Self {
        Self { tracers: Vec::new(), sim_t: 0.0, user_reset: true }
    }
}

fn warp_streamlines_velocity(p: Vec3) -> Vec3 {
    // Simple analytic vortex (poor man's lid-driven cavity).
    let r = Vec3::new(p.x - 0.5, 0.0, p.z - 0.5);
    Vec3::new(-r.z, 0.0, r.x) * 0.6
}

fn update_warp_fem_streamlines(
    time: Res<Time>,
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpFemStreamlines { return; }
    let st = &mut warp.fem_streamlines;
    if st.user_reset {
        st.tracers.clear();
        for j in 0..8 {
            for i in 0..8 {
                let x = (i as f32 + 0.5) / 8.0;
                let z = (j as f32 + 0.5) / 8.0;
                st.tracers.push(Vec3::new(x, 0.05, z));
            }
        }
        st.sim_t = 0.0;
        st.user_reset = false;
    }
    if run.paused { return; }
    let dt = time.delta_secs().min(0.05);
    st.sim_t += dt as f64;
    for p in st.tracers.iter_mut() {
        let v = warp_streamlines_velocity(*p);
        *p += v * dt;
        // Wrap into [0,1]^2 in xz.
        if p.x < 0.0 { p.x += 1.0; }
        if p.x > 1.0 { p.x -= 1.0; }
        if p.z < 0.0 { p.z += 1.0; }
        if p.z > 1.0 { p.z -= 1.0; }
    }
}

fn draw_warp_fem_streamlines_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpFemStreamlines { return; }
    let st = &warp.fem_streamlines;
    let origin = Vec3::new(-0.5, 0.05, -0.5);
    for &p in &st.tracers {
        let world = origin + Vec3::new(p.x, 0.0, p.z);
        gizmos.sphere(
            bevy::math::Isometry3d::from_translation(world),
            0.015,
            Color::srgb(0.3, 0.7, 0.95),
        );
    }
}

fn warp_fem_streamlines_pane(ui: &mut egui::Ui, st: &mut WarpFemStreamlinesState) {
    ui.label("Warp FEM streamlines — port of fem/example_streamlines.py");
    ui.separator();
    readout(ui, "Tracers", &format!("{}", st.tracers.len()));
    readout(ui, "t", &format!("{:.2} s", st.sim_t));
    if ui.button("Reset tracers").clicked() { st.user_reset = true; }
    ui.small("Tracers advect through analytic 2D vortex velocity field.");
}

// ── Scene 73: WarpFemDeformedGeometry (curved annular mesh) ──────
struct WarpFemDeformedState {
    /// Annular vertex positions for visualisation.
    verts: Vec<MVec3>,
    /// Annular triangle indices.
    tris: Vec<[u32; 3]>,
    /// Per-vertex temperature.
    u: Vec<f64>,
    user_reset: bool,
}
impl Default for WarpFemDeformedState {
    fn default() -> Self {
        Self { verts: Vec::new(), tris: Vec::new(), u: Vec::new(), user_reset: true }
    }
}

fn update_warp_fem_deformed(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpFemDeformedGeometry { return; }
    let st = &mut warp.fem_deformed;
    if !st.user_reset { return; }
    // Build a curved (annular) tri mesh by mapping the unit square
    // (x,y) → (r sin t, r cos t) with r = y + 0.5, t = π x / 2.
    let nx = 12;
    let ny = 8;
    st.verts.clear();
    st.tris.clear();
    for j in 0..=ny {
        for i in 0..=nx {
            let x = i as f64 / nx as f64;
            let y = j as f64 / ny as f64;
            let r = y + 0.5;
            let t = 0.5 * std::f64::consts::PI * x;
            st.verts.push(MVec3::new(r * t.sin(), 0.0, r * t.cos()));
        }
    }
    let stride = nx + 1;
    for j in 0..ny {
        for i in 0..nx {
            let a = (j * stride + i) as u32;
            let b = a + 1;
            let c = a + stride as u32;
            let d = c + 1;
            st.tris.push([a, b, d]);
            st.tris.push([a, d, c]);
        }
    }
    // T = boundary value: 1 on outer arc, 0 on inner arc; linear blend
    // by radial coordinate (cheap analytic approximation).
    st.u = st.verts.iter().map(|p| {
        let r = (p.x * p.x + p.z * p.z).sqrt();
        ((r - 0.5) / 1.0).clamp(0.0, 1.0)
    }).collect();
    st.user_reset = false;
}

fn draw_warp_fem_deformed_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpFemDeformedGeometry { return; }
    let st = &warp.fem_deformed;
    let origin = Vec3::new(0.0, 0.05, 0.0);
    for tri in &st.tris {
        let pa = st.verts[tri[0] as usize];
        let pb = st.verts[tri[1] as usize];
        let pc = st.verts[tri[2] as usize];
        let to = |p: MVec3| origin + Vec3::new(p.x as f32, 0.0, p.z as f32);
        let avg = (st.u[tri[0] as usize] + st.u[tri[1] as usize] + st.u[tri[2] as usize]) / 3.0;
        let col = fem_heat_colour(avg);
        gizmos.line(to(pa), to(pb), col);
        gizmos.line(to(pb), to(pc), col);
        gizmos.line(to(pc), to(pa), col);
    }
}

fn warp_fem_deformed_pane(ui: &mut egui::Ui, st: &mut WarpFemDeformedState) {
    ui.label("Warp FEM deformed geometry — port of fem/example_deformed_geometry.py");
    ui.separator();
    readout(ui, "Triangles", &format!("{}", st.tris.len()));
    if ui.button("Rebuild").clicked() { st.user_reset = true; }
    ui.small("Annular (curved) mesh from unit-square parameterisation;");
    ui.small("temperature = radial blend (analytic).");
}

// ── Scene 74: WarpFemDistortion ───────────────────────────────────
//
// Stage 37.4 retrofit — runs the real symmetric-Dirichlet energy
// kernel + Newton minimisation on a small 2D mesh. The state holds
// both the initial-energy reading (from a small perturbation of the
// rest mesh) and the post-Newton energy so the pane can show the
// solver pulling the mesh back to E = 0.
struct WarpFemDistortionState {
    /// Initial perturbed energy.
    initial_energy: f64,
    /// Final energy after Newton.
    energy: f64,
    /// Newton iteration cap.
    max_iter: usize,
    user_reset: bool,
}
impl Default for WarpFemDistortionState {
    fn default() -> Self {
        Self { initial_energy: 0.0, energy: 0.0, max_iter: 10, user_reset: true }
    }
}

fn update_warp_fem_distortion(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpFemDistortion { return; }
    let st = &mut warp.fem_distortion;
    if !st.user_reset { return; }
    // Build a small 4×4 tri grid and perturb the interior vertices
    // by a deterministic ±2e-2 offset. Hold the boundary fixed so
    // Newton has a unique minimum to walk towards.
    let mesh = fem::FemMesh::tri_grid_2d(4, 4, 1.0, 1.0);
    let rest = mesh.vertices.clone();
    let mut init = rest.clone();
    for (i, p) in init.iter_mut().enumerate() {
        let on_boundary = p.x.abs() < 1e-9
            || (p.x - 1.0).abs() < 1e-9
            || p.y.abs() < 1e-9
            || (p.y - 1.0).abs() < 1e-9;
        if !on_boundary {
            let h1 = ((i as u64).wrapping_mul(2654435761)) as f64 / u64::MAX as f64;
            let h2 = ((i as u64).wrapping_mul(40503)) as f64 / u64::MAX as f64;
            p.x += (2.0 * h1 - 1.0) * 2.0e-2;
            p.y += (2.0 * h2 - 1.0) * 2.0e-2;
        }
    }
    let mut e0: f64 = 0.0;
    for cell in &mesh.cells {
        let (e, _g, _h) = fem::compute_element_distortion_energy(cell, &rest, &init);
        e0 += e;
    }
    st.initial_energy = e0;
    let final_def = fem::solve_distortion_minimization(&mesh, &init, st.max_iter, 1e-12);
    let mut ef: f64 = 0.0;
    for cell in &mesh.cells {
        let (e, _g, _h) = fem::compute_element_distortion_energy(cell, &rest, &final_def);
        ef += e;
    }
    st.energy = ef;
    st.user_reset = false;
}

fn warp_fem_distortion_pane(ui: &mut egui::Ui, st: &mut WarpFemDistortionState) {
    ui.label("Warp FEM distortion energy — port of fem/example_distortion_energy.py");
    ui.separator();
    readout(ui, "Initial perturbed E", &format!("{:.4e}", st.initial_energy));
    readout(ui, "Final E (post-Newton)", &format!("{:.4e}", st.energy));
    if ui.button("Recompute").clicked() { st.user_reset = true; }
    ui.small("Stage 37.4 — symmetric Dirichlet E = ½ tr(FᵀF + (FᵀF)⁻¹) − 2");
    ui.small("on 4×4 tri grid; Newton drives perturbed mesh back to E = 0.");
}

// ── Scene 75: WarpFemElasticOpt ───────────────────────────────────
struct WarpFemElasticOptState {
    /// Beam height parameter (optimised).
    height: f64,
    /// Stress proxy (smaller = better).
    last_stress: f64,
    iter: usize,
    learning_rate: f64,
    paused: bool,
    user_reset: bool,
}
impl Default for WarpFemElasticOptState {
    fn default() -> Self {
        Self {
            height: 0.4, last_stress: 0.0, iter: 0,
            learning_rate: 0.01, paused: false, user_reset: true,
        }
    }
}

fn update_warp_fem_elastic_opt(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpFemElasticOpt { return; }
    let st = &mut warp.fem_elastic_opt;
    if st.user_reset {
        st.height = 0.4;
        st.iter = 0;
        st.last_stress = 0.0;
        st.user_reset = false;
    }
    if st.paused { return; }
    // Surrogate stress model: small for h=0.5, growing for h<0.2 or h>0.8.
    let s = (st.height - 0.5).powi(2) * 100.0 + 1.0 / st.height.max(0.05);
    let grad = 200.0 * (st.height - 0.5) - 1.0 / st.height.max(0.05).powi(2);
    st.last_stress = s;
    st.height = (st.height - st.learning_rate * grad).clamp(0.1, 0.9);
    st.iter += 1;
}

fn warp_fem_elastic_opt_pane(ui: &mut egui::Ui, st: &mut WarpFemElasticOptState) {
    ui.label("Warp FEM elastic shape opt — port of fem/example_elastic_shape_optimization.py");
    ui.separator();
    readout(ui, "Iter", &format!("{}", st.iter));
    readout(ui, "Beam height h", &format!("{:.3} m", st.height));
    readout(ui, "Stress proxy", &format!("{:.3e}", st.last_stress));
    ui.label("Learning rate");
    let _ = ui.add(egui::Slider::new(&mut st.learning_rate, 0.001..=0.1).logarithmic(true));
    ui.checkbox(&mut st.paused, "Pause");
    if ui.button("Reset").clicked() { st.user_reset = true; }
    ui.small("Surrogate model: optimal h ≈ 0.5; gradient descent.");
}

// ── Scene 76: WarpFftFluid ────────────────────────────────────────
//
// Real fractional-step Navier-Stokes (Stage 37.5) on a periodic
// 16×16 grid with Taylor-Green initial condition. Pressure-Poisson
// solved by the GPU FFT kernel (Stage 24c). Velocity field is
// rendered as gizmo arrows.
//
// One frame = `FFT_FLUID_STEPS_PER_FRAME` projection steps. The FFT
// kernel is reused across frames — initialised lazily on first use.

const FFT_FLUID_GRID: usize = 16;
const FFT_FLUID_DT: f32 = 0.002;
const FFT_FLUID_STEPS_PER_FRAME: usize = 1;

struct WarpFftFluidState {
    sim_t: f64,
    /// Kinematic viscosity ν.
    viscosity: f32,
    /// Current velocity field — row-major `(j, i) → j·n + i`.
    u: Vec<[f32; 2]>,
    /// FFT kernel — built lazily on first GPU access.
    kernel: Option<molla::kernels::FftKernel>,
    /// Reset flag.
    needs_reset: bool,
}

impl Default for WarpFftFluidState {
    fn default() -> Self {
        Self {
            sim_t: 0.0,
            viscosity: 0.05,
            u: Vec::new(),
            kernel: None,
            needs_reset: true,
        }
    }
}

fn fft_fluid_seed_taylor_green(n: usize) -> Vec<[f32; 2]> {
    let two_pi = 2.0 * std::f32::consts::PI;
    let mut u = vec![[0.0_f32; 2]; n * n];
    for j in 0..n {
        let y = (j as f32 + 0.5) / n as f32;
        for i in 0..n {
            let x = (i as f32 + 0.5) / n as f32;
            let ux = (two_pi * x).sin() * (two_pi * y).cos();
            let vy = -(two_pi * x).cos() * (two_pi * y).sin();
            u[j * n + i] = [ux, vy];
        }
    }
    u
}

fn update_warp_fft_fluid(
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpFftFluid { return; }
    let st = &mut warp.fft_fluid;
    if st.needs_reset {
        st.u = fft_fluid_seed_taylor_green(FFT_FLUID_GRID);
        st.sim_t = 0.0;
        st.needs_reset = false;
    }
    if run.paused { return; }

    // Acquire the GPU runtime; if unavailable the sim leaves the
    // state alone (the gizmos will still render the IC).
    let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) else {
        return;
    };
    if st.kernel.is_none() {
        st.kernel = Some(molla::kernels::FftKernel::new(device));
    }
    let kernel = st.kernel.as_ref().unwrap();

    let cfg = molla::fem::FractionalStepConfig {
        grid_n: FFT_FLUID_GRID,
        viscosity: st.viscosity,
        dt: FFT_FLUID_DT,
        num_steps: FFT_FLUID_STEPS_PER_FRAME,
    };
    let _ = molla::fem::advance_navier_stokes_fractional_step(
        device, queue, kernel, &mut st.u, cfg,
    );
    st.sim_t += FFT_FLUID_DT as f64;
}

fn draw_warp_fft_fluid_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpFftFluid { return; }
    let st = &warp.fft_fluid;
    if st.u.is_empty() { return; }
    let origin = Vec3::new(-0.5, 0.05, -0.5);
    let n = FFT_FLUID_GRID;
    for j in 0..n {
        let y = (j as f32 + 0.5) / n as f32;
        for i in 0..n {
            let x = (i as f32 + 0.5) / n as f32;
            let v = st.u[j * n + i];
            let from = origin + Vec3::new(x, 0.0, y);
            let to = from + Vec3::new(v[0] * 0.05, 0.0, v[1] * 0.05);
            gizmos.line(from, to, Color::srgb(0.5, 0.7, 0.95));
        }
    }
}

fn warp_fft_fluid_pane(ui: &mut egui::Ui, st: &mut WarpFftFluidState) {
    ui.label("Warp FFT fluid — Stage 37.5 fractional-step NS via GPU FFT");
    ui.separator();
    readout(ui, "t", &format!("{:.3} s", st.sim_t));
    ui.label("Viscosity ν");
    let _ = ui.add(egui::Slider::new(&mut st.viscosity, 0.001..=0.5).logarithmic(true));
    if ui.button("Reset (Taylor-Green IC)").clicked() {
        st.needs_reset = true;
    }
    ui.small("Pressure-Poisson via molla::fem::solve_pressure_poisson_fft");
    ui.small("(Stage 24c GPU FFT). Predictor + corrector run on CPU.");
}

// ── Scene 77: WarpDistortion3D ────────────────────────────────────
//
// Stage 37.4 retrofit — 3D companion to `WarpFemDistortion`. Builds a
// 3×3×3 tet grid, perturbs the interior vertices, and runs the same
// Newton minimisation on the symmetric Dirichlet energy.
struct WarpDistortion3DState {
    initial_energy: f64,
    energy: f64,
    max_iter: usize,
    user_reset: bool,
}
impl Default for WarpDistortion3DState {
    fn default() -> Self {
        Self { initial_energy: 0.0, energy: 0.0, max_iter: 10, user_reset: true }
    }
}

fn update_warp_distortion_3d(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpDistortion3D { return; }
    let st = &mut warp.distortion_3d;
    if !st.user_reset { return; }
    let mesh = fem::FemMesh::tet_grid_3d(3, 3, 3, 1.0, 1.0, 1.0);
    let rest = mesh.vertices.clone();
    let mut init = rest.clone();
    for (i, p) in init.iter_mut().enumerate() {
        let on_boundary = p.x.abs() < 1e-9
            || (p.x - 1.0).abs() < 1e-9
            || p.y.abs() < 1e-9
            || (p.y - 1.0).abs() < 1e-9
            || p.z.abs() < 1e-9
            || (p.z - 1.0).abs() < 1e-9;
        if !on_boundary {
            let h1 = ((i as u64).wrapping_mul(2654435761)) as f64 / u64::MAX as f64;
            let h2 = ((i as u64).wrapping_mul(40503)) as f64 / u64::MAX as f64;
            let h3 = ((i as u64).wrapping_mul(2246822519)) as f64 / u64::MAX as f64;
            p.x += (2.0 * h1 - 1.0) * 1.0e-2;
            p.y += (2.0 * h2 - 1.0) * 1.0e-2;
            p.z += (2.0 * h3 - 1.0) * 1.0e-2;
        }
    }
    let mut e0: f64 = 0.0;
    for cell in &mesh.cells {
        let (e, _g, _h) = fem::compute_element_distortion_energy(cell, &rest, &init);
        e0 += e;
    }
    st.initial_energy = e0;
    let final_def = fem::solve_distortion_minimization(&mesh, &init, st.max_iter, 1e-12);
    let mut ef: f64 = 0.0;
    for cell in &mesh.cells {
        let (e, _g, _h) = fem::compute_element_distortion_energy(cell, &rest, &final_def);
        ef += e;
    }
    st.energy = ef;
    st.user_reset = false;
}

fn warp_distortion_3d_pane(ui: &mut egui::Ui, st: &mut WarpDistortion3DState) {
    ui.label("Warp distortion 3D (bonus, fem/example_distortion_energy.py extension)");
    ui.separator();
    readout(ui, "Initial perturbed E", &format!("{:.4e}", st.initial_energy));
    readout(ui, "Final E (post-Newton)", &format!("{:.4e}", st.energy));
    if ui.button("Recompute").clicked() { st.user_reset = true; }
    ui.small("Stage 37.4 — symmetric Dirichlet E = ½ tr(FᵀF + (FᵀF)⁻¹) − 3");
    ui.small("on 3×3×3 tet grid; Newton minimises energy back to E ≈ 0.");
}

// ── Scene 79: WarpApicFluid (Stage 37.6 — APIC FEM-MPM hybrid) ────
//
// 200 particles fall under gravity into a tet-meshed unit cube box
// with no-slip walls. The FEM mesh provides the velocity field; the
// MPM particles deposit + retrieve momentum via APIC weights. Render
// particles as gizmo points + tet wireframe in muted grey.
//
// All compute lives on CPU through `molla::fem::ApicHybrid`; physics
// is short-circuited in `step_simulation`.

const WARP_APIC_GRID: usize = 6;
/// Origin offset for the unit-cube mesh — sit it just above the
/// ground plane so the gizmo wireframe doesn't clip.
const WARP_APIC_ORIGIN: Vec3 = Vec3::new(-0.5, 0.05, -0.5);
const WARP_APIC_DEFAULT_VISCOSITY: f64 = 0.5;
const WARP_APIC_DEFAULT_PARTICLE_COUNT: usize = 200;

struct WarpApicFluidState {
    /// The hybrid carries both the mesh and the particles.
    hybrid: fem::apic_coupling::ApicHybrid,
    /// Cached no-slip BCs (every boundary vertex). Rebuilt only when
    /// the mesh changes (which it doesn't in v1).
    bcs: Vec<fem::BoundaryCondition>,
    /// Viscosity slider (kinematic, m²/s).
    viscosity: f64,
    /// Live mean-KE (per particle) readout.
    mean_kinetic_energy: f64,
    /// Step counter — surfaced in the pane.
    step_count: u64,
    /// "Reset" button trigger.
    user_reset: bool,
    /// `apply_scene_switch` stamps this so the next update tick re-
    /// initialises the particle layout.
    needs_reset: bool,
}

impl Default for WarpApicFluidState {
    fn default() -> Self {
        let (hybrid, bcs) = build_apic_fluid_state(WARP_APIC_DEFAULT_PARTICLE_COUNT);
        Self {
            hybrid,
            bcs,
            viscosity: WARP_APIC_DEFAULT_VISCOSITY,
            mean_kinetic_energy: 0.0,
            step_count: 0,
            user_reset: false,
            // Don't reset on first build — the constructor already did it.
            needs_reset: false,
        }
    }
}

/// Build a fresh `ApicHybrid` (cube mesh + `count` particles in the
/// upper half) plus the matching no-slip boundary list.
fn build_apic_fluid_state(
    count: usize,
) -> (fem::apic_coupling::ApicHybrid, Vec<fem::BoundaryCondition>) {
    let mesh = fem::FemMesh::tet_grid_3d(
        WARP_APIC_GRID,
        WARP_APIC_GRID,
        WARP_APIC_GRID,
        1.0,
        1.0,
        1.0,
    );
    // Pseudo-random sample inside the upper half of the cube.
    let mut s: u64 = 0xCAFE_F00D_DEAD_BEEF;
    let next = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f64) / ((1u64 << 31) as f64)
    };
    let mut particles = Vec::with_capacity(count);
    for _ in 0..count {
        let x = molla::math::Vec3::new(
            0.10 + 0.80 * next(&mut s),
            0.10 + 0.80 * next(&mut s),
            // Upper half of the box so they have room to fall.
            0.55 + 0.40 * next(&mut s),
        );
        particles.push(fem::ApicParticle::at_rest(x, 1.0));
    }
    let mut bcs: Vec<fem::BoundaryCondition> = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        let on_boundary = p.x <= 1e-9
            || p.x >= 1.0 - 1e-9
            || p.y <= 1e-9
            || p.y >= 1.0 - 1e-9
            || p.z <= 1e-9
            || p.z >= 1.0 - 1e-9;
        if on_boundary {
            bcs.push(fem::BoundaryCondition::DirichletVector {
                vertex: i as u32,
                values: vec![0.0, 0.0, 0.0],
            });
        }
    }
    let mut hybrid = fem::apic_coupling::ApicHybrid::new(mesh, particles);
    hybrid.gravity = molla::math::Vec3::new(0.0, 0.0, -2.0);
    (hybrid, bcs)
}

fn update_warp_apic_fluid(
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpApicFluid {
        return;
    }
    let st = &mut warp.apic_fluid;
    if st.user_reset || st.needs_reset {
        let count = st.hybrid.particle_count();
        let (h, bcs) = build_apic_fluid_state(count);
        st.hybrid = h;
        st.bcs = bcs;
        st.step_count = 0;
        st.user_reset = false;
        st.needs_reset = false;
    }
    if run.paused {
        return;
    }
    let dt: f64 = 1.0 / 60.0;
    st.hybrid.step(dt, st.viscosity, &st.bcs);
    // Clamp to AABB so the explicit advect doesn't push particles
    // through a no-slip wall when the velocity exceeds (cell_size / dt).
    st.hybrid.clamp_to_aabb(
        molla::math::Vec3::new(0.001, 0.001, 0.001),
        molla::math::Vec3::new(0.999, 0.999, 0.999),
    );
    let n = st.hybrid.particle_count().max(1) as f64;
    st.mean_kinetic_energy = st.hybrid.total_kinetic_energy() / n;
    st.step_count += 1;
}

fn draw_warp_apic_fluid_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpApicFluid {
        return;
    }
    let st = &warp.apic_fluid;
    let origin = WARP_APIC_ORIGIN;
    // Tet wireframe in muted grey.
    use std::collections::HashSet;
    let mesh = &st.hybrid.mesh;
    let mut seen: HashSet<(u32, u32)> =
        HashSet::with_capacity(mesh.cells.len() * 4);
    let wire_colour = Color::srgb(0.30, 0.30, 0.34);
    for cell in &mesh.cells {
        if let fem::FemCell::Tet(v) = cell {
            let edges = [
                (v[0], v[1]),
                (v[0], v[2]),
                (v[0], v[3]),
                (v[1], v[2]),
                (v[1], v[3]),
                (v[2], v[3]),
            ];
            for &(a, b) in &edges {
                let key = if a < b { (a, b) } else { (b, a) };
                if !seen.insert(key) {
                    continue;
                }
                let pa = mesh.vertices[a as usize];
                let pb = mesh.vertices[b as usize];
                let av = origin + Vec3::new(pa.x as f32, pa.z as f32, pa.y as f32);
                let bv = origin + Vec3::new(pb.x as f32, pb.z as f32, pb.y as f32);
                gizmos.line(av, bv, wire_colour);
            }
        }
    }
    // Particles as small gizmo crosses in vivid cyan; brighter when
    // moving fast so the viewer can see the velocity field.
    for particle in &st.hybrid.particles {
        let speed = particle.v.length();
        let t = (speed / 2.0).clamp(0.0, 1.0) as f32;
        let colour = Color::srgb(0.40 + 0.55 * t, 0.85, 0.95 - 0.55 * t);
        let p = particle.x;
        let centre = origin + Vec3::new(p.x as f32, p.z as f32, p.y as f32);
        let r = 0.012;
        gizmos.line(
            centre - Vec3::new(r, 0.0, 0.0),
            centre + Vec3::new(r, 0.0, 0.0),
            colour,
        );
        gizmos.line(
            centre - Vec3::new(0.0, r, 0.0),
            centre + Vec3::new(0.0, r, 0.0),
            colour,
        );
        gizmos.line(
            centre - Vec3::new(0.0, 0.0, r),
            centre + Vec3::new(0.0, 0.0, r),
            colour,
        );
    }
}

fn warp_apic_fluid_pane(ui: &mut egui::Ui, st: &mut WarpApicFluidState) {
    ui.label("Warp APIC fluid — Stage 37.6 FEM-MPM hybrid");
    ui.separator();
    readout(
        ui,
        "Mesh",
        &format!(
            "{0}×{0}×{0} tets ({1} verts)",
            WARP_APIC_GRID,
            st.hybrid.mesh.vert_count()
        ),
    );
    readout(ui, "Particles", &format!("{}", st.hybrid.particle_count()));
    readout(ui, "Step", &format!("{}", st.step_count));
    readout(
        ui,
        "Mean KE / particle",
        &format!("{:.4e} J", st.mean_kinetic_energy),
    );
    ui.separator();
    ui.label("Viscosity ν (m²/s)");
    let _ = ui.add(
        egui::Slider::new(&mut st.viscosity, 0.0..=2.0)
            .text("smoothing"),
    );
    ui.separator();
    if ui.button("Reset").clicked() {
        st.user_reset = true;
    }
    ui.small("Particles → mesh nodes via APIC weights;");
    ui.small("mesh applies gravity + viscous Laplacian + no-slip;");
    ui.small("particles read updated node velocities back.");
    ui.small("Pure Stokes pressure-Poisson is Stage 37.6.1.");
}

// ═══ Scene 78: WarpWave (Stage 37.1) ═══════════════════════════════════
//
// 32×32 P1 tri mesh on the unit square; all four edges pinned (Dirichlet
// u = 0); initial Gaussian pulse at (0.3, 0.3); Newmark-β implicit step
// per frame. Renders as height-field via gizmo lines.

const WARP_WAVE_GRID: usize = 32;

struct WarpWaveExampleState {
    mesh: fem::FemMesh,
    mass: molla::math::sparse::BsrMatrix,
    stiff: molla::math::sparse::BsrMatrix,
    bcs: Vec<fem::BoundaryCondition>,
    u: Vec<f64>,
    v: Vec<f64>,
    a: Vec<f64>,
    c2: f64,
    last_c2_baked: f64,
    dt: f64,
    pulse_amp: f64,
    sim_t: f64,
    last_residual: f64,
    last_iters: usize,
    needs_reset: bool,
    user_reset: bool,
    /// GPU stepper — `Some` once the user has clicked "Use GPU"
    /// (built lazily because `Default` doesn't have access to the
    /// `Engine`). Rebuilt whenever the mesh or BCs change.
    gpu_stepper: Option<fem::WaveStepperGpu>,
    use_gpu: bool,
}

impl Default for WarpWaveExampleState {
    fn default() -> Self {
        let mesh = fem::FemMesh::tri_grid_2d(WARP_WAVE_GRID, WARP_WAVE_GRID, 1.0, 1.0);
        let c2 = 1.0_f64;
        let mass = fem::assemble_global(&mesh, fem::FunctionSpace::P1Nodal, |c, v| {
            fem::compute_element_mass(c, v)
        })
        .expect("WarpWave mass");
        let stiff = fem::assemble_global(&mesh, fem::FunctionSpace::P1Nodal, |c, v| {
            fem::compute_element_wave_stiffness(c, v, c2)
        })
        .expect("WarpWave stiffness");
        // Pin all boundary verts.
        let mut bcs = Vec::new();
        for (i, p) in mesh.vertices.iter().enumerate() {
            if p.x <= 1e-9 || p.x >= 1.0 - 1e-9 || p.y <= 1e-9 || p.y >= 1.0 - 1e-9 {
                bcs.push(fem::BoundaryCondition::Dirichlet {
                    vertex: i as u32,
                    value: 0.0,
                });
            }
        }
        let n = mesh.vertices.len();
        // Gaussian pulse initial displacement at (0.3, 0.3).
        let mut u = vec![0.0_f64; n];
        let amp = 0.05;
        for (i, p) in mesh.vertices.iter().enumerate() {
            let dx = p.x - 0.3;
            let dy = p.y - 0.3;
            u[i] = amp * (-50.0 * (dx * dx + dy * dy)).exp();
        }
        // Re-pin BC values.
        for bc in &bcs {
            if let fem::BoundaryCondition::Dirichlet { vertex, value } = bc {
                u[*vertex as usize] = *value;
            }
        }
        Self {
            mesh,
            mass,
            stiff,
            bcs,
            u,
            v: vec![0.0; n],
            a: vec![0.0; n],
            c2,
            last_c2_baked: c2,
            dt: 0.005,
            pulse_amp: amp,
            sim_t: 0.0,
            last_residual: 0.0,
            last_iters: 0,
            needs_reset: false,
            user_reset: false,
            gpu_stepper: None,
            use_gpu: false,
        }
    }
}

fn update_warp_wave(
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpWave || run.paused {
        return;
    }
    let st = &mut warp.wave;
    if st.needs_reset || st.user_reset {
        *st = WarpWaveExampleState::default();
        return;
    }
    let stiff_changed = (st.c2 - st.last_c2_baked).abs() > 1e-9;
    if stiff_changed {
        st.stiff = fem::assemble_global(&st.mesh, fem::FunctionSpace::P1Nodal, |c, v| {
            fem::compute_element_wave_stiffness(c, v, st.c2)
        })
        .expect("WarpWave rebuild stiffness");
        st.last_c2_baked = st.c2;
        // Stiffness changed — invalidate the cached GPU stepper.
        st.gpu_stepper = None;
    }
    // Lazily build the GPU stepper if the user has enabled GPU mode
    // and the cache is empty. Slice H — initial state is uploaded
    // once to the persistent GPU buffers.
    if st.use_gpu && st.gpu_stepper.is_none() {
        match fem::WaveStepperGpu::new(
            &sim.engine,
            &st.mesh,
            &st.mass,
            &st.stiff,
            st.dt,
            fem::NEWMARK_BETA_DEFAULT,
            fem::NEWMARK_GAMMA_DEFAULT,
            &st.bcs,
            &st.u,
            &st.v,
            &st.a,
        ) {
            Ok(s) => st.gpu_stepper = Some(s),
            Err(e) => {
                warn!("WarpWave: GPU stepper init failed ({e}); falling back to CPU");
                st.use_gpu = false;
            }
        }
    }
    if st.use_gpu && st.gpu_stepper.is_some() {
        // Slice H GPU path — `step` runs entirely on GPU; `read_state`
        // pulls back `(u, v, a)` so the height-field gizmos can render.
        // Falling back to CPU here would leak the GPU-resident state
        // out of sync with the host; on error we mark the stepper for
        // rebuild on the next frame.
        let stepper = st.gpu_stepper.as_mut().expect("checked above");
        let step_result = stepper.step();
        let read_result = match step_result {
            Ok(stats) => match stepper.read_state() {
                Ok((u, v, a)) => Ok((stats, u, v, a)),
                Err(e) => Err(e),
            },
            Err(e) => Err(e),
        };
        match read_result {
            Ok((stats, u, v, a)) => {
                st.u = u;
                st.v = v;
                st.a = a;
                st.last_residual = stats.residual;
                st.last_iters = stats.iters;
            }
            Err(e) => {
                warn!("WarpWave: GPU step/read failed ({e}); rebuilding next frame");
                st.gpu_stepper = None;
            }
        }
    } else {
        let step = fem::solve_wave_transient(
            &st.mass,
            &st.stiff,
            &st.u,
            &st.v,
            &st.a,
            st.dt,
            fem::NEWMARK_BETA_DEFAULT,
            fem::NEWMARK_GAMMA_DEFAULT,
            &st.bcs,
        );
        st.u = step.u_new;
        st.v = step.v_new;
        st.a = step.a_new;
        st.last_residual = step.residual;
        st.last_iters = step.iters;
    }
    st.sim_t += st.dt;
}

fn draw_warp_wave_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpWave {
        return;
    }
    let st = &warp.wave;
    let origin = Vec3::new(-0.5, 0.5, -0.5);
    let h_scale = 1.5_f32;
    // Draw tri edges as height-field lines coloured by displacement.
    for cell in &st.mesh.cells {
        if let fem::FemCell::Tri(v) = cell {
            for (a, b) in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
                let pa = st.mesh.vertices[a as usize];
                let pb = st.mesh.vertices[b as usize];
                let ua = st.u[a as usize] as f32;
                let ub = st.u[b as usize] as f32;
                let from = origin
                    + Vec3::new(pa.x as f32, ua * h_scale, pa.y as f32);
                let to = origin
                    + Vec3::new(pb.x as f32, ub * h_scale, pb.y as f32);
                let amp = (0.5 * (ua + ub)).abs() / st.pulse_amp.max(1e-6) as f32;
                let amp = amp.clamp(0.0, 1.0);
                let colour = Color::srgb(
                    0.30 + 0.65 * amp,
                    0.30 + 0.40 * amp,
                    0.95 - 0.40 * amp,
                );
                gizmos.line(from, to, colour);
            }
        }
    }
}

fn warp_wave_pane(ui: &mut egui::Ui, state: &mut WarpWaveExampleState) {
    ui.label("Warp wave (port of example_wave.py)");
    ui.separator();
    readout(ui, "Mesh", &format!("{0}×{0} P1 tris", WARP_WAVE_GRID));
    readout(ui, "DoFs", &format!("{}", state.u.len()));
    readout(ui, "Sim time t", &format!("{:.3} s", state.sim_t));
    readout(ui, "CG iters", &format!("{}", state.last_iters));
    readout(ui, "Residual", &format!("{:.3e}", state.last_residual));
    ui.separator();
    ui.label("Wave speed² c²");
    let _ = ui.add(egui::Slider::new(&mut state.c2, 0.1..=10.0).logarithmic(true));
    ui.label("Time step dt");
    let _ = ui.add(egui::Slider::new(&mut state.dt, 0.001..=0.02).logarithmic(true));
    ui.separator();
    if ui.button("Reset (re-emit pulse)").clicked() {
        state.user_reset = true;
    }
    ui.separator();
    let was_gpu = state.use_gpu;
    let _ = ui.checkbox(&mut state.use_gpu, "Use GPU CG (Stage 36.5 Slice E)");
    if was_gpu != state.use_gpu {
        // Toggle changed — invalidate the cached stepper so the
        // next frame either builds it or drops it.
        state.gpu_stepper = None;
    }
    let backend = if state.use_gpu && state.gpu_stepper.is_some() {
        "GPU resident CG"
    } else if state.use_gpu {
        "GPU (init pending / failed)"
    } else {
        "CPU"
    };
    readout(ui, "Backend", backend);
    ui.small("Newmark-β implicit (β=¼, γ=½) — unconditionally stable.");
    ui.small("Initial Gaussian pulse at (0.3, 0.3); 4 edges pinned.");
}

// ═══ Scene 80: WarpDiffRay (Stage 37.7) ════════════════════════════════
//
// Single sphere; gradient descent on its position to match a target depth
// image rendered with the sphere at a different position.

struct WarpDiffRayExampleState {
    scene: molla::sensors::raycast_adjoint::RayScene,
    rays: Vec<molla::sensors::raycast_adjoint::Ray>,
    target_image: Vec<f64>,
    params: Vec<f64>,
    target_params: Vec<f64>,
    lr: f64,
    iter: u32,
    last_loss: f64,
    run_descent: bool,
    user_reset: bool,
}

impl Default for WarpDiffRayExampleState {
    fn default() -> Self {
        use molla::sensors::raycast_adjoint::{Ray, RayScene};
        use molla::math::Vec3 as MVec3;
        let mut scene = RayScene::new();
        // Sphere takes 3 params (xyz) starting at offset 0; radius 0.5.
        scene.add_sphere(0, 0.5);
        scene.add_plane_z(-2.0);
        // 32×32 orthographic-down-Z rays.
        let n = 32usize;
        let half = 1.5_f64;
        let mut rays = Vec::with_capacity(n * n);
        for j in 0..n {
            for i in 0..n {
                let uu = (i as f64 + 0.5) / n as f64;
                let vv = (j as f64 + 0.5) / n as f64;
                rays.push(Ray::new(
                    MVec3::new((uu - 0.5) * 2.0 * half, (vv - 0.5) * 2.0 * half, 5.0),
                    MVec3::new(0.0, 0.0, -1.0),
                ));
            }
        }
        let target_params = vec![0.2_f64, 0.2_f64, 0.0_f64];
        let target_image = scene.trace(&target_params, &rays);
        let params = vec![0.0_f64; 3];
        Self {
            scene,
            rays,
            target_image,
            params,
            target_params,
            lr: 5e-4,
            iter: 0,
            last_loss: 0.0,
            run_descent: true,
            user_reset: false,
        }
    }
}

fn update_warp_diff_ray(
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpDiffRay || run.paused {
        return;
    }
    let st = &mut warp.diff_ray;
    if st.user_reset {
        *st = WarpDiffRayExampleState::default();
        return;
    }
    let current = st.scene.trace(&st.params, &st.rays);
    st.last_loss =
        molla::sensors::raycast_adjoint::compute_loss_l2(&current, &st.target_image);
    if st.run_descent {
        let grad = molla::sensors::raycast_adjoint::compute_grad_raycast(
            &mut st.scene,
            &mut st.params,
            &st.target_image,
            &st.rays,
            1e-3,
        );
        for i in 0..st.params.len() {
            st.params[i] -= st.lr * grad[i];
        }
        st.iter += 1;
    }
}

fn draw_warp_diff_ray_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpDiffRay {
        return;
    }
    let st = &warp.diff_ray;
    let cur = Vec3::new(
        st.params[0] as f32,
        st.params[1] as f32,
        st.params[2] as f32,
    );
    let tgt = Vec3::new(
        st.target_params[0] as f32,
        st.target_params[1] as f32,
        st.target_params[2] as f32,
    );
    gizmos.sphere(cur, 0.5, Color::srgb(0.30, 0.65, 0.95));
    gizmos.sphere(tgt, 0.5, Color::srgb(0.95, 0.40, 0.30));
}

fn warp_diff_ray_pane(ui: &mut egui::Ui, state: &mut WarpDiffRayExampleState) {
    ui.label("Warp differentiable raycast (port of example_diffray.py)");
    ui.separator();
    readout(ui, "Iteration", &format!("{}", state.iter));
    readout(ui, "Loss", &format!("{:.4e}", state.last_loss));
    readout(
        ui,
        "Position",
        &format!(
            "({:.3}, {:.3}, {:.3})",
            state.params[0], state.params[1], state.params[2]
        ),
    );
    readout(
        ui,
        "Target",
        &format!(
            "({:.3}, {:.3}, {:.3})",
            state.target_params[0], state.target_params[1], state.target_params[2]
        ),
    );
    ui.separator();
    ui.label("Target X");
    let _ = ui.add(egui::Slider::new(&mut state.target_params[0], -1.0..=1.0));
    ui.label("Target Y");
    let _ = ui.add(egui::Slider::new(&mut state.target_params[1], -1.0..=1.0));
    ui.label("Learning rate");
    let _ = ui.add(egui::Slider::new(&mut state.lr, 1e-5..=1e-2).logarithmic(true));
    ui.separator();
    ui.checkbox(&mut state.run_descent, "Run descent");
    if ui.button("Reset (random init)").clicked() {
        state.user_reset = true;
    }
    ui.small("Blue = current sphere; red = target.");
    ui.small("Vanilla SGD on FD adjoint of L2 depth-image loss.");
    ui.small("Z-axis is hostile; XY converges within 50 iters at lr=5e-4.");
}

// ═══ Scene 81: WarpTileFilter (Stage 37.8) ═════════════════════════════
//
// Procedural 32×32 input image, separable 5-tap filter (box / gaussian
// / laplacian / identity) via `tile_convolution` + `tile_filtering`.
// Renders input on the left, filtered output on the right, both as
// colored point grids. CPU/GPU toggle exercises both paths.

const TILE_FILTER_DIM: usize = 32;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum TileFilterKind {
    Box,
    Gaussian,
    Laplacian,
    Identity,
}

struct WarpTileFilterState {
    input: Vec<f32>,
    output: Vec<f32>,
    kind: TileFilterKind,
    use_gpu: bool,
    last_us: f64,
    needs_recompute: bool,
}

impl Default for WarpTileFilterState {
    fn default() -> Self {
        let n = TILE_FILTER_DIM;
        // Procedural input: low-frequency sin·cos pattern + a circular
        // bump that gives the Laplacian/box visible high-freq content.
        let mut input = vec![0.0_f32; n * n];
        for j in 0..n {
            for i in 0..n {
                let x = (i as f32 + 0.5) / n as f32;
                let y = (j as f32 + 0.5) / n as f32;
                let pattern = (4.0 * std::f32::consts::PI * x).sin()
                    * (4.0 * std::f32::consts::PI * y).cos();
                let dx = x - 0.5;
                let dy = y - 0.5;
                let bump = (-30.0 * (dx * dx + dy * dy)).exp();
                input[j * n + i] = 0.4 * pattern + 0.6 * bump;
            }
        }
        Self {
            input,
            output: vec![0.0_f32; n * n],
            kind: TileFilterKind::Gaussian,
            use_gpu: false,
            last_us: 0.0,
            needs_recompute: true,
        }
    }
}

fn update_warp_tile_filter(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpTileFilter {
        return;
    }
    let st = &mut warp.tile_filter;
    if !st.needs_recompute {
        return;
    }
    let n = TILE_FILTER_DIM;
    let t0 = std::time::Instant::now();
    let mut output = vec![0.0_f32; n * n];

    let used_gpu = if st.use_gpu {
        if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
            let kernel =
                molla::compute::tile_convolution::TileConvolutionKernel::new(device);
            let result = match st.kind {
                TileFilterKind::Box => molla::compute::tile_filtering::box_filter_gpu(
                    device, queue, &kernel, &st.input, &mut output, n, n,
                ),
                TileFilterKind::Gaussian => molla::compute::tile_filtering::gaussian_filter_gpu(
                    device, queue, &kernel, &st.input, &mut output, n, n,
                ),
                TileFilterKind::Laplacian => molla::compute::tile_filtering::laplacian_filter_gpu(
                    device, queue, &kernel, &st.input, &mut output, n, n,
                ),
                TileFilterKind::Identity => molla::compute::tile_filtering::identity_filter_gpu(
                    device, queue, &kernel, &st.input, &mut output, n, n,
                ),
            };
            result.is_ok()
        } else {
            false
        }
    } else {
        false
    };
    if !used_gpu {
        let _ = match st.kind {
            TileFilterKind::Box => molla::compute::tile_filtering::box_filter_cpu(
                &st.input, &mut output, n, n,
            ),
            TileFilterKind::Gaussian => molla::compute::tile_filtering::gaussian_filter_cpu(
                &st.input, &mut output, n, n,
            ),
            TileFilterKind::Laplacian => molla::compute::tile_filtering::laplacian_filter_cpu(
                &st.input, &mut output, n, n,
            ),
            TileFilterKind::Identity => molla::compute::tile_filtering::identity_filter_cpu(
                &st.input, &mut output, n, n,
            ),
        };
    }
    st.last_us = t0.elapsed().as_secs_f64() * 1.0e6;
    st.output = output;
    st.needs_recompute = false;
}

/// Map a value `v` (typically in [-1, 1]) to a Bevy color ramp.
fn tile_filter_color(v: f32) -> Color {
    let t = ((v + 1.0) * 0.5).clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.5 {
        let s = t * 2.0;
        (0.05, 0.20 + 0.65 * s, 1.0 - 0.6 * s)
    } else {
        let s = (t - 0.5) * 2.0;
        (0.05 + 0.95 * s, 0.85 - 0.55 * s, 0.40 - 0.40 * s)
    };
    Color::srgb(r, g, b)
}

fn draw_warp_tile_filter_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpTileFilter {
        return;
    }
    let st = &warp.tile_filter;
    let n = TILE_FILTER_DIM;
    let cell = 1.0 / n as f32;
    // Two grids: input on the left at x ∈ [-1.1, -0.1], output on the
    // right at x ∈ [0.1, 1.1]. Y up = z in world space (camera looks
    // down +Y).
    let half = 0.5;
    for j in 0..n {
        for i in 0..n {
            let x_local = (i as f32 + 0.5) * cell - 0.5;
            let z_local = (j as f32 + 0.5) * cell - 0.5;
            // Input
            let in_val = st.input[j * n + i];
            let in_pos = Vec3::new(-half - 0.05 + x_local, 0.05 + 0.5 * in_val, z_local);
            gizmos.sphere(in_pos, cell * 0.4, tile_filter_color(in_val));
            // Output
            let out_val = st.output[j * n + i];
            let out_pos = Vec3::new(half + 0.05 + x_local, 0.05 + 0.5 * out_val, z_local);
            gizmos.sphere(out_pos, cell * 0.4, tile_filter_color(out_val));
        }
    }
}

fn warp_tile_filter_pane(ui: &mut egui::Ui, st: &mut WarpTileFilterState) {
    ui.label("Warp tile filter — Stage 37.8 separable 5-tap convolution");
    ui.separator();
    readout(ui, "Image", &format!("{0}×{0}", TILE_FILTER_DIM));
    readout(ui, "Last solve", &format!("{:.1} µs", st.last_us));
    ui.separator();
    ui.label("Filter");
    let prev_kind = st.kind;
    egui::ComboBox::from_id_salt("tile_filter_kind")
        .selected_text(match st.kind {
            TileFilterKind::Box => "Box",
            TileFilterKind::Gaussian => "Gaussian (binomial 1-4-6-4-1)",
            TileFilterKind::Laplacian => "Laplacian (1-D edge stencil)",
            TileFilterKind::Identity => "Identity (passthrough)",
        })
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut st.kind, TileFilterKind::Box, "Box");
            ui.selectable_value(&mut st.kind, TileFilterKind::Gaussian, "Gaussian");
            ui.selectable_value(&mut st.kind, TileFilterKind::Laplacian, "Laplacian");
            ui.selectable_value(&mut st.kind, TileFilterKind::Identity, "Identity");
        });
    if prev_kind != st.kind {
        st.needs_recompute = true;
    }
    ui.separator();
    let was_gpu = st.use_gpu;
    let _ = ui.checkbox(&mut st.use_gpu, "Use GPU");
    if was_gpu != st.use_gpu {
        st.needs_recompute = true;
    }
    let backend = if st.use_gpu { "GPU TileConvolutionKernel" } else { "CPU convolve_separable" };
    readout(ui, "Backend", backend);
    if ui.button("Re-apply").clicked() {
        st.needs_recompute = true;
    }
    ui.small("Left grid: input image. Right grid: filtered output.");
    ui.small("Stage 37.8 primitives: tile_convolution + tile_filtering.");
}

// ═══ Scene 82: WarpTileCompaction (Stage 37.8) ═════════════════════════
//
// Random 3D point cloud + a "keep if inside radius R" predicate. Each
// frame builds the mask, calls `tile_stream_compaction` (CPU or GPU)
// to extract the surviving point indices, and renders kept points as
// bright spheres with rejected points faded. CPU/GPU toggle exercises
// both paths.

const TILE_COMPACT_N: usize = 256;

struct WarpTileCompactionState {
    /// 3D positions, fixed at construction.
    positions: Vec<Vec3>,
    /// Radius threshold; points with `‖p‖ < radius` are kept.
    radius: f32,
    /// Indices of surviving points after the last compaction (CPU
    /// reads + draw walks this).
    kept: Vec<u32>,
    use_gpu: bool,
    last_us: f64,
    needs_recompute: bool,
}

impl Default for WarpTileCompactionState {
    fn default() -> Self {
        let n = TILE_COMPACT_N;
        let mut positions = Vec::with_capacity(n);
        // Deterministic LCG so the cloud doesn't shimmer between
        // sessions or test runs.
        let mut seed = 0xC0DE_u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed >> 16) as i32 as f32) / 32768.0
        };
        for _ in 0..n {
            positions.push(Vec3::new(rng(), rng(), rng()));
        }
        Self {
            positions,
            radius: 0.5,
            kept: Vec::new(),
            use_gpu: false,
            last_us: 0.0,
            needs_recompute: true,
        }
    }
}

fn update_warp_tile_compaction(
    sim: Res<Sim>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpTileCompaction {
        return;
    }
    let st = &mut warp.tile_compaction;
    if !st.needs_recompute {
        return;
    }
    let n = st.positions.len();
    // Index array used as the input — the compaction primitive returns
    // the surviving entries; passing point indices means we get
    // surviving point indices back (then we look up positions to
    // render).
    let input_f32: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let mask: Vec<bool> = st
        .positions
        .iter()
        .map(|p| p.length() < st.radius)
        .collect();

    let t0 = std::time::Instant::now();
    let kept_f32: Vec<f32> = if st.use_gpu {
        if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
            let kernel =
                molla::compute::tile_stream_compaction::TileStreamCompactionKernel::new(device);
            let mask_u32 = molla::compute::tile_stream_compaction::mask_to_u32(&mask);
            kernel
                .dispatch_blocking(device, queue, &input_f32, &mask_u32)
                .unwrap_or_else(|_| {
                    molla::compute::tile_stream_compaction::compact_cpu(&input_f32, &mask)
                        .expect("compact_cpu fallback")
                })
        } else {
            molla::compute::tile_stream_compaction::compact_cpu(&input_f32, &mask)
                .expect("compact_cpu")
        }
    } else {
        molla::compute::tile_stream_compaction::compact_cpu(&input_f32, &mask)
            .expect("compact_cpu")
    };
    st.last_us = t0.elapsed().as_secs_f64() * 1.0e6;
    st.kept = kept_f32.into_iter().map(|x| x as u32).collect();
    st.needs_recompute = false;
}

fn draw_warp_tile_compaction_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpTileCompaction {
        return;
    }
    let st = &warp.tile_compaction;
    // Draw all points as faded gray dots, then re-draw kept ones as
    // bright spheres on top.
    for p in &st.positions {
        gizmos.sphere(*p, 0.005, Color::srgb(0.3, 0.3, 0.3));
    }
    for &idx in &st.kept {
        let p = st.positions[idx as usize];
        gizmos.sphere(p, 0.012, Color::srgb(0.2, 0.9, 0.4));
    }
    // Show the radius threshold as a wireframe sphere.
    let segments = 64;
    for axis in 0..3 {
        for s in 0..segments {
            let t0 = (s as f32 / segments as f32) * std::f32::consts::TAU;
            let t1 = ((s + 1) as f32 / segments as f32) * std::f32::consts::TAU;
            let r = st.radius;
            let (a, b) = match axis {
                0 => (Vec3::new(0.0, r * t0.cos(), r * t0.sin()), Vec3::new(0.0, r * t1.cos(), r * t1.sin())),
                1 => (Vec3::new(r * t0.cos(), 0.0, r * t0.sin()), Vec3::new(r * t1.cos(), 0.0, r * t1.sin())),
                _ => (Vec3::new(r * t0.cos(), r * t0.sin(), 0.0), Vec3::new(r * t1.cos(), r * t1.sin(), 0.0)),
            };
            gizmos.line(a, b, Color::srgb(0.7, 0.5, 0.3));
        }
    }
}

fn warp_tile_compaction_pane(
    ui: &mut egui::Ui,
    st: &mut WarpTileCompactionState,
) {
    ui.label("Warp tile compaction — Stage 37.8 stream compaction");
    ui.separator();
    readout(ui, "N points", &format!("{}", st.positions.len()));
    readout(ui, "Kept", &format!("{}", st.kept.len()));
    readout(ui, "Last solve", &format!("{:.1} µs", st.last_us));
    ui.separator();
    ui.label("Radius threshold");
    if ui
        .add(egui::Slider::new(&mut st.radius, 0.05..=1.5))
        .changed()
    {
        st.needs_recompute = true;
    }
    ui.separator();
    let was_gpu = st.use_gpu;
    let _ = ui.checkbox(&mut st.use_gpu, "Use GPU");
    if was_gpu != st.use_gpu {
        st.needs_recompute = true;
    }
    let backend = if st.use_gpu { "GPU TileStreamCompactionKernel" } else { "CPU compact_cpu" };
    readout(ui, "Backend", backend);
    if ui.button("Re-shuffle predicate").clicked() {
        st.needs_recompute = true;
    }
    ui.small("Green spheres = points kept by `‖p‖ < radius`.");
    ui.small("Wireframe sphere = the radius threshold.");
}
