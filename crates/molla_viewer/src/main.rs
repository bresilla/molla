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
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};
use bevy_frost::pane::{Pane2, PaneAnchor, RailZone};
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
use molla::actuators::{ControllerPd, ControllerPdConfig};
use molla::sensors::{
    BodyContactReading, ContactSensor, ImuMount, ImuReading, ImuSensor, RayHit, Raycaster,
};
use molla::solvers::{FeatherstoneSolver, Solver as _};

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
        .insert_resource(ArmExampleState::default())
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
                handle_input,
                apply_scene_switch,
                update_mouse_grab,
                // Arm controller runs BEFORE the physics step so its
                // `joint_force` writes are picked up by the solver.
                update_arm_example,
                step_simulation,
                sync_transforms,
                sync_cloth_mesh,
                update_lidar,
                update_imu_example,
                update_contact_example,
                update_activity_colors,
                apply_wireframe,
                draw_grab_gizmo,
                draw_lidar_gizmos,
                draw_contact_gizmos,
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
        }
    }

    /// Which dynamics solver this scene wants. Most scenes use XPBD;
    /// the Arm example needs Featherstone for proper joint dynamics.
    fn solver_kind(self) -> SolverKind {
        match self {
            Scene::Arm => SolverKind::Featherstone,
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
    mut lidar: ResMut<LidarState>,
    mut imu: ResMut<ImuExampleState>,
    mut contact: ResMut<ContactExampleState>,
    mut arm: ResMut<ArmExampleState>,
    mut stats: ResMut<SimStats>,
) {
    let Some(scene) = pending.0.take() else {
        return;
    };
    stats.arm_logged = false;

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
) {
    if run.paused && !run.step_once {
        return;
    }
    run.step_once = false;
    let frame_start = std::time::Instant::now();

    let device = sim.engine.device().unwrap().clone();
    let queue = sim.engine.queue().unwrap().clone();

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

#[allow(clippy::too_many_arguments)]
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
    lidar_state: Res<LidarState>,
    imu_state: Res<ImuExampleState>,
    contact_state: Res<ContactExampleState>,
    mut arm_state: ResMut<ArmExampleState>,
    mut ribbon_open: ResMut<RibbonOpen>,
    mut ribbon_placement: ResMut<RibbonPlacement>,
    mut ribbon_drag: ResMut<RibbonDrag>,
) {
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
                Scene::Lidar | Scene::Imu | Scene::Contact | Scene::Arm
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
        _ => None,
    };
    if let Some(title) = scene_pane_title {
        if is_open(PANE_SCENE) {
            Pane2::new(
                PANE_SCENE,
                title,
                PaneAnchor::RightRail(RailZone::Start),
                accent_color,
            )
            .show(ctx, |ui| match scene {
                Scene::Lidar => lidar_pane(ui, &mut lidar_config, &lidar_state),
                Scene::Imu => imu_pane(ui, &imu_state),
                Scene::Contact => contact_pane(ui, &contact_state),
                Scene::Arm => arm_pane(ui, &mut arm_state, &sim),
                _ => {}
            });
        }
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
