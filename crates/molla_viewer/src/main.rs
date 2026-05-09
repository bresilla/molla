//! `molla_viewer` — Bevy app that runs a molla scene and visualises it.
//!
//! Four switchable demo scenes (hotkeys 1/2/3/4):
//! * **1 Pile**   — 12×12 mixed sphere/box/capsule rain (collision +
//!                  restitution + friction).
//! * **2 Rope**   — chain of small spheres pinned by CONNECT equality
//!                  constraints, top link held kinematic (stage 7).
//! * **3 Tower**  — 8-layer 4×4 box pyramid getting smashed by a heavy
//!                  sphere dropped from above (stacking + impact).
//! * **4 Cloth**  — 24×24 hanging curtain pinned along its top edge,
//!                  sags and swings via XPBD distance constraints
//!                  (stage 12).
//!
//! Mouse:
//!   * **LMB-click+drag**   grab a body and drag it (spring-pull force)
//!   * MMB-drag             pan the camera focus (XZ plane)
//!   * MMB+LMB / MMB+RMB    lift the focus (vertical Y)
//!   * LMB+RMB-drag         orbit yaw + pitch
//!   * Scroll               zoom
//!
//! Keys:
//!   * 1 / 2 / 3 / 4   switch demo scene
//!   * Q           Quadrotor (Stage 17 multi-DOF Featherstone)
//!   * T           toggle "activity view" — sleeping bodies turn grey,
//!                 only active (moving) bodies keep their colour
//!   * W           toggle wireframe view (every mesh outlined)
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
use molla::compute::Engine;
use molla::geometry::CollisionPipeline;
use molla::math::{Quat as MQuat, Vec3 as MVec3};
use molla::prelude as mp;
use molla::actuators::{
    Actuator, ClampingDcMotor, ClampingDcMotorConfig, ClampingStage, ControllerPd,
    ControllerPdConfig, ControllerPipelinePd, ControllerPipelinePdConfig, ControllerStage,
    Delay, DelayConfig,
};
use molla::sensors::{
    BodyContactReading, CameraChannels, ContactSensor, ImuMount, ImuReading, ImuSensor, RayHit,
    Raycaster, TiledCamera, TiledCameraConfig,
};
use molla::solvers::{FeatherstoneSolver, IkConfig, IkObjective, IkSolver, Solver as _};

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
                    update_gpu_ik_arm,
                    update_quadrotor,
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
                    apply_wireframe,
                    draw_grab_gizmo,
                    draw_lidar_gizmos,
                    draw_contact_gizmos,
                )
                    .chain(),
            )
                .chain(),
        )
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
}

/// Per-scene solver dispatch. Both variants implement the same
/// `Solver::step` signature, so callers don't need to care which
/// backend the active scene picked.
enum SolverBackend {
    Xpbd(mp::XpbdSolver),
    Featherstone(FeatherstoneSolver),
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
    ];

    fn label(self) -> &'static str {
        match self {
            Scene::Pile => "Pile",
            Scene::Rope => "Rope",
            Scene::Tower => "Tower",
            Scene::Cloth => "Cloth",
            Scene::Lidar => "Lidar (raycast)",
            Scene::Imu => "IMU (accel + gyro)",
            Scene::Contact => "Contact (per-body)",
            Scene::Arm => "Arm (PD actuator)",
            Scene::Shapes => "Shapes (cone / hull / heightfield / SDF)",
            Scene::GpuIkArm => "GPU IK arm (Stage 16)",
            Scene::GpuNarrowPhase => "GPU narrow phase (Stage 15)",
            Scene::Camera => "Camera (5-channel raytrace)",
            Scene::Actuator => "Actuator pipeline (Stage 18)",
            Scene::Quadrotor => "Quadrotor (Stage 17 multi-DOF)",
            Scene::BroadPhase => "GPU broad phase (Stage 19)",
        }
    }

    /// Which dynamics solver this scene wants. Most scenes use XPBD;
    /// the Arm example needs Featherstone for proper joint dynamics.
    /// `GpuIkArm` reuses the Featherstone backend (any articulation
    /// solver works since `step_simulation` skips this scene anyway).
    fn solver_kind(self) -> SolverKind {
        match self {
            Scene::Arm | Scene::GpuIkArm | Scene::Actuator | Scene::Quadrotor => {
                SolverKind::Featherstone
            }
            _ => SolverKind::Xpbd,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum SolverKind {
    Xpbd,
    Featherstone,
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

impl Sim {
    /// Dispatch on `scene` and assemble `(Sim, BodyVisuals)`.
    fn build_scene(scene: Scene) -> (Self, Vec<BodyVisual>) {
        let (model, visuals) = match scene {
            Scene::Pile => build_pile_model(),
            Scene::Rope => build_rope_model(),
            Scene::Tower => build_tower_model(),
            Scene::Cloth => build_cloth_model(),
            Scene::Lidar => build_lidar_model(),
            Scene::Imu => build_imu_model(),
            Scene::Contact => build_contact_model(),
            Scene::Arm => build_arm_model(),
            Scene::Shapes => build_shapes_model(),
            Scene::GpuIkArm => build_gpu_ik_arm_model(),
            Scene::GpuNarrowPhase => build_gpu_narrow_phase_model(),
            Scene::Camera => build_camera_model(),
            Scene::Actuator => build_actuator_model(),
            Scene::Quadrotor => build_quadrotor_model(),
            Scene::BroadPhase => build_broad_phase_model(),
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

        let mut state_a = model.state().expect("state");
        let mut state_b = model.state().expect("state");
        let mut control = model.control();
        let mut contacts = mp::Contacts::with_capacity(8192);

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
        let solver = match scene.solver_kind() {
            SolverKind::Xpbd => SolverBackend::Xpbd(
                mp::XpbdSolver::new_gpu(&engine, model.body_count as u32)
                    .expect("XPBD GPU solver init")
                    .with_iterations(10),
            ),
            SolverKind::Featherstone => SolverBackend::Featherstone(
                FeatherstoneSolver::new_gpu(&engine, &model)
                    .expect("Featherstone GPU solver init"),
            ),
        };

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
            },
            visuals,
        )
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
fn build_arm_model() -> (mp::Model, Vec<BodyVisual>) {
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
        b.add_joint_revolute(prev_body, body, x_p, x_c, axis)
            .expect("arm joint");

        prev_body = body;
        next_pos = next_pos - MVec3::new(0.0, ARM_LINK_LEN as f64, 0.0);
    }

    (b.build().expect("arm model"), visuals)
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
        };
        let awake = mats[i % mats.len()].clone();
        commands.spawn((
            MollaBody(mp::BodyId::from_index(i)),
            Mesh3d(mesh),
            MeshMaterial3d(awake.clone()),
            AwakeMaterial(awake),
            NotShadowCaster,
            Transform::from_xyz(
                pose.position.x as f32,
                pose.position.y as f32,
                pose.position.z as f32,
            ),
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
    mut pending: ResMut<PendingSceneSwitch>,
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
    if keys.just_pressed(KeyCode::Digit1) && sim.scene != Scene::Pile {
        pending.0 = Some(Scene::Pile);
    }
    if keys.just_pressed(KeyCode::Digit2) && sim.scene != Scene::Rope {
        pending.0 = Some(Scene::Rope);
    }
    if keys.just_pressed(KeyCode::Digit3) && sim.scene != Scene::Tower {
        pending.0 = Some(Scene::Tower);
    }
    if keys.just_pressed(KeyCode::Digit4) && sim.scene != Scene::Cloth {
        pending.0 = Some(Scene::Cloth);
    }
    if keys.just_pressed(KeyCode::Digit5) && sim.scene != Scene::Lidar {
        pending.0 = Some(Scene::Lidar);
    }
    if keys.just_pressed(KeyCode::Digit6) && sim.scene != Scene::Imu {
        pending.0 = Some(Scene::Imu);
    }
    if keys.just_pressed(KeyCode::Digit7) && sim.scene != Scene::Contact {
        pending.0 = Some(Scene::Contact);
    }
    if keys.just_pressed(KeyCode::Digit8) && sim.scene != Scene::Arm {
        pending.0 = Some(Scene::Arm);
    }
    if keys.just_pressed(KeyCode::Digit9) && sim.scene != Scene::Shapes {
        pending.0 = Some(Scene::Shapes);
    }
    if keys.just_pressed(KeyCode::Digit0) && sim.scene != Scene::GpuIkArm {
        pending.0 = Some(Scene::GpuIkArm);
    }
    if keys.just_pressed(KeyCode::Minus) && sim.scene != Scene::GpuNarrowPhase {
        pending.0 = Some(Scene::GpuNarrowPhase);
    }
    if keys.just_pressed(KeyCode::Equal) && sim.scene != Scene::Camera {
        pending.0 = Some(Scene::Camera);
    }
    if keys.just_pressed(KeyCode::KeyA) && sim.scene != Scene::Actuator {
        pending.0 = Some(Scene::Actuator);
    }
    if keys.just_pressed(KeyCode::KeyQ) && sim.scene != Scene::Quadrotor {
        pending.0 = Some(Scene::Quadrotor);
    }
    if keys.just_pressed(KeyCode::KeyB) && sim.scene != Scene::BroadPhase {
        pending.0 = Some(Scene::BroadPhase);
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
    stats.arm_logged = false;
    // GPU IK: re-seed the target on next entry from the new arm's rest EE.
    *gpu_ik = GpuIkArmState::default();
    // Stage 18 actuator pipeline: drop the GPU pipeline so the next
    // entry rebuilds against the new sim. `update_actuator_example`
    // also clears it on scene exit; this is belt-and-braces.
    actuator.actuator = None;
    // Quadrotor: reset cached mass / arm-length on every scene switch.
    *quad = QuadrotorState::default();
    // Reset GPU narrow-phase counters on every scene switch — the
    // pipeline that tracked them is rebuilt with the new Sim.
    gpu_np.gpu_enabled_now = false;
    gpu_np.last_cpu_calls = 0;
    gpu_np.last_contacts = 0;
    gpu_np.last_pairs = 0;
    // Same for GPU broad-phase counters / pipeline-state cache.
    gpu_bp.gpu_enabled_now = false;
    gpu_bp.last_pairs = 0;
    gpu_bp.smoothed_ms = 0.0;
    gpu_bp.last_cpu_ms = 0.0;
    gpu_bp.last_gpu_ms = 0.0;

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
    commands.insert_resource(sim);
    commands.insert_resource(visuals);

    info!("scene switched → {}", scene.label());
}

fn step_simulation(
    mut sim: ResMut<Sim>,
    mut run: ResMut<SimRunState>,
    mut stats: ResMut<SimStats>,
    mut gpu_toggles: (ResMut<GpuNpExampleState>, ResMut<GpuBpExampleState>),
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
        total_download_us += d0.elapsed().as_secs_f64() * 1_000_000.0;
    }
    let solve_us = solve_start.elapsed().as_secs_f64() * 1_000_000.0;
    let total_us = frame_start.elapsed().as_secs_f64() * 1_000_000.0;

    stats.smoothed_step_us = 0.9 * stats.smoothed_step_us + 0.1 * total_us;
    stats.smoothed_solve_us = 0.9 * stats.smoothed_solve_us + 0.1 * solve_us;
    stats.smoothed_collide_us = 0.9 * stats.smoothed_collide_us + 0.1 * total_collide_us;
    stats.smoothed_download_us = 0.9 * stats.smoothed_download_us + 0.1 * total_download_us;
    stats.last_contacts = contacts.count;

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
        if let Err(e) = act.step(engine, state_a, control) {
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
    mut sim: ResMut<Sim>,
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
    mut sim: ResMut<Sim>,
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
                    Scene::Arm => arm_pane(ui, &mut pane_states.arm, &sim),
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
        "GPU LBVH (Stage 19)"
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
fn arm_pane(ui: &mut egui::Ui, state: &mut ArmExampleState, sim: &Sim) {
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

    ui.add_space(4.0);
    ui.small(
        "FREE root + 4 fixed-rotor bodies. Per-rotor thrust commanded \
         via state.body_f world-frame +Y forces; Featherstone GPU \
         solves the 6-DOF base + 4 rigid attachments. Yaw is free \
         (single-axis thrust can't generate yaw torque).",
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

