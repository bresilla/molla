//! `molla_viewer` — Bevy app that runs a molla scene and visualises it.
//!
//! - **`bevy_glacial`** drives the camera (mouse pan / orbit / zoom),
//!   the LOD ground grid, and an axis triad at the world origin.
//! - **`bevy_frost`** themes egui, hosts the `molla` control pane on
//!   the left rail (frost `Pane2` + corekit widgets), and runs the
//!   pointer-event firewall so clicks over the panel don't leak into
//!   the viewport.
//!
//! Stage 4 viewer: XPBD CPU solver with a real static-plane collision
//! shape. The GPU toggle from the previous build is parked until XPBD
//! has a GPU port (next sub-stage).
//!
//! Mouse:
//!   * MMB-drag             pan the camera focus (XZ plane)
//!   * MMB+LMB / MMB+RMB    lift the focus (vertical Y)
//!   * LMB+RMB-drag         orbit yaw + pitch
//!   * Scroll               zoom
//!
//! Keys:
//!   * Space   pause / resume the simulation
//!   * R       reset poses to the initial drop
//!   * Esc     quit

use bevy::diagnostic::{
    DiagnosticsStore, EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin,
};
#[cfg(any(feature = "profile", feature = "log_diag"))]
use bevy::diagnostic::LogDiagnosticsPlugin;
use bevy::light::{CascadeShadowConfigBuilder, NotShadowCaster};
use bevy::prelude::*;
#[cfg(not(feature = "profile"))]
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass};
use bevy_frost::pane::{Pane2, PaneAnchor, RailZone};
use bevy_frost::prelude::*;
use bevy_frost::style::AccentColor;
use bevy_frost::widget::{button, readout};
use bevy_glacial::prelude::*;
use molla::compute::Engine;
use molla::geometry::{collide, ShapePair};
use molla::math::{Quat as MQuat, Vec3 as MVec3};
use molla::prelude as mp;
use molla::solvers::Solver as _;

#[cfg(not(target_arch = "wasm32"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const GRID_N: usize = 50; // 50×50 = 2500 spheres
const GRID_SPACING: f32 = 1.2;
const SPHERE_RADIUS: f32 = 0.5;
const DROP_HEIGHT: f32 = 15.0;
const FIXED_DT: f64 = 1.0 / 120.0;
const SUBSTEPS_PER_FRAME: usize = 2;

const PLANE_RESTITUTION: f64 = 0.35;
const PLANE_FRICTION: f64 = 0.6;

const PANE_ID: &str = "molla_panel";

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
        .add_plugins(EntityCountDiagnosticsPlugin::default());
    // RenderDiagnosticsPlugin is auto-added by Bevy when the `trace`
    // feature is on (i.e. our `profile` feature). Add it ourselves
    // only when profiling is OFF.
    #[cfg(not(feature = "profile"))]
    app.add_plugins(RenderDiagnosticsPlugin);
    // Stdout diagnostics — enabled by either `profile` or `log_diag`.
    // Filter to fps + frame_time so the periodic log call doesn't
    // itself become a frame stall.
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
        .insert_resource(Sim::build())
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (handle_input, step_simulation, sync_transforms).chain(),
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
}

/// Owns the molla simulation. **GPU-first** per the project rule —
/// `Engine::new_gpu()` is hard-required at startup, no CPU fallback.
#[derive(Resource)]
struct Sim {
    model: mp::Model,
    state_a: mp::State,
    state_b: mp::State,
    control: mp::Control,
    contacts: mp::Contacts,
    solver: mp::XpbdSolver,
    pair_buf: Vec<ShapePair>,
    engine: Engine,
    adapter_label: String,
}

impl Sim {
    fn build() -> Self {
        let mut b = mp::ModelBuilder::new();
        b.set_angular_damping(0.05);

        for ix in 0..GRID_N {
            for iz in 0..GRID_N {
                let x = (ix as f64 - (GRID_N as f64 - 1.0) * 0.5) * GRID_SPACING as f64;
                let z = (iz as f64 - (GRID_N as f64 - 1.0) * 0.5) * GRID_SPACING as f64;
                let y = DROP_HEIGHT as f64 + 0.4 * ((ix + iz) as f64 / 2.0).sin();
                let pos = MVec3::new(x, y, z);

                let body = b.add_body(
                    mp::BodyParams::new()
                        .with_spatial_inertia(mp::SpatialInertia::sphere(
                            1.0,
                            SPHERE_RADIUS as f64,
                        ))
                        .with_position(pos),
                );
                b.add_shape_sphere(body, SPHERE_RADIUS as f64, mp::Transform::IDENTITY)
                    .expect("sphere shape");
            }
        }

        b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
            .expect("plane shape");

        let mut model = b.build().expect("model");
        model.default_friction = PLANE_FRICTION;
        model.default_restitution = PLANE_RESTITUTION;
        model.contact_margin = 0.02;

        let mut state_a = model.state().expect("state");
        let mut state_b = model.state().expect("state");
        let mut control = model.control();
        let mut contacts = mp::Contacts::with_capacity(8192);

        // GPU-first: bring up wgpu and bind the XPBD GPU solver. We
        // hard-fail if there's no adapter rather than silently
        // falling back to CPU.
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

        // One-time uploads of model + initial state + control. The
        // step loop re-uploads contacts + state_a each substep.
        model.upload_to_device(&device, &queue).expect("model upload");
        state_a.upload_to_device(&device, &queue).expect("state_a upload");
        state_b.upload_to_device(&device, &queue).expect("state_b upload");
        control.upload_to_device(&device, &queue).expect("control upload");
        contacts.upload_to_device(&device, &queue).expect("contacts upload");

        let solver = mp::XpbdSolver::new_gpu(&engine, model.body_count as u32)
            .expect("XPBD GPU solver init")
            .with_iterations(4);

        Self {
            model,
            state_a,
            state_b,
            control,
            contacts,
            solver,
            pair_buf: Vec::with_capacity(16384),
            engine,
            adapter_label,
        }
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
        // Host may be stale after a GPU step — pull fresh data first
        // so host_mut doesn't trip the DeviceDirty guard.
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
        let _ = self.state_a.upload_to_device(&device, &queue);
        let _ = self.state_b.upload_to_device(&device, &queue);
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    sim: Res<Sim>,
) {
    // Sun + cascade shadows.
    let sun_shadow = CascadeShadowConfigBuilder {
        num_cascades: 2,
        minimum_distance: 0.1,
        maximum_distance: 200.0,
        first_cascade_far_bound: 30.0,
        overlap_proportion: 0.2,
    }
    .build();
    commands.spawn((
        Name::new("Sun"),
        Transform::from_xyz(20.0, 60.0, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
        DirectionalLight {
            illuminance: 12_000.0,
            shadows_enabled: true,
            ..default()
        },
        sun_shadow,
    ));

    let chase = ChaseCamera {
        focus: Vec3::new(0.0, 6.0, 0.0),
        distance: 80.0,
        elevation: 28f32.to_radians(),
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
            brightness: 100.0,
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

    // 8-colour palette. Spheres are NotShadowCaster — at 2500
    // instances the shadow pass would dominate frame time and the
    // visual gain is negligible against the LOD grid.
    let sphere_mesh = meshes.add(Sphere::new(SPHERE_RADIUS).mesh().ico(1).expect("ico"));
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
        commands.spawn((
            MollaBody(mp::BodyId::from_index(i)),
            Mesh3d(sphere_mesh.clone()),
            MeshMaterial3d(mats[i % mats.len()].clone()),
            NotShadowCaster,
            Transform::from_xyz(
                pose.position.x as f32,
                pose.position.y as f32,
                pose.position.z as f32,
            ),
        ));
    }

    info!(
        "molla viewer: {} bodies + plane, {}× @ {:.0} Hz physics — Space pause, R reset, Esc quit",
        sim.model.body_count,
        SUBSTEPS_PER_FRAME,
        1.0 / FIXED_DT
    );
}

fn handle_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut exit: MessageWriter<AppExit>,
    mut sim: ResMut<Sim>,
    mut run: ResMut<SimRunState>,
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
        pair_buf,
        ..
    } = &mut *sim;

    let mut total_collide_us = 0.0_f64;
    let mut total_download_us = 0.0_f64;
    let solve_start = std::time::Instant::now();
    for _ in 0..SUBSTEPS_PER_FRAME {
        // Collide on host using the freshest body poses (downloaded
        // at the end of the previous substep, or set up by the
        // initial state for the very first substep).
        let c0 = std::time::Instant::now();
        collide(model, state_a, contacts, pair_buf).expect("collide");
        contacts.upload_to_device(&device, &queue).expect("contacts upload");
        total_collide_us += c0.elapsed().as_secs_f64() * 1_000_000.0;

        // Push host-mutated state up to the device before dispatch.
        let _ = state_a.body_q.upload(&queue);
        let _ = state_a.body_qd.upload(&queue);

        solver
            .step(model, state_a, control, contacts, FIXED_DT, state_b)
            .expect("solver step");
        std::mem::swap(state_a, state_b);

        // Pull the new state back so the next substep's collide and
        // sync_transforms see real numbers.
        let d0 = std::time::Instant::now();
        let _ = state_a.body_q.download(&device, &queue);
        let _ = state_a.body_qd.download(&device, &queue);
        total_download_us += d0.elapsed().as_secs_f64() * 1_000_000.0;
    }
    let solve_us = solve_start.elapsed().as_secs_f64() * 1_000_000.0;
    let total_us = frame_start.elapsed().as_secs_f64() * 1_000_000.0;

    stats.smoothed_step_us = 0.9 * stats.smoothed_step_us + 0.1 * total_us;
    stats.smoothed_solve_us = 0.9 * stats.smoothed_solve_us + 0.1 * solve_us;
    stats.smoothed_collide_us = 0.9 * stats.smoothed_collide_us + 0.1 * total_collide_us;
    stats.smoothed_download_us = 0.9 * stats.smoothed_download_us + 0.1 * total_download_us;
    stats.last_contacts = contacts.count;
}

fn sync_transforms(sim: Res<Sim>, mut q: Query<(&MollaBody, &mut Transform)>) {
    let body_q = sim.state_a.body_q.host().expect("body_q");
    for (body, mut tf) in q.iter_mut() {
        let pose = body_q[body.0.index()];
        tf.translation = Vec3::new(
            pose.position.x as f32,
            pose.position.y as f32,
            pose.position.z as f32,
        );
        let r: MQuat = pose.rotation;
        tf.rotation = Quat::from_xyzw(r.x as f32, r.y as f32, r.z as f32, r.w as f32);
    }
}

fn control_pane(
    mut contexts: EguiContexts,
    accent: Res<AccentColor>,
    diagnostics: Res<DiagnosticsStore>,
    stats: Res<SimStats>,
    mut sim: ResMut<Sim>,
    mut run: ResMut<SimRunState>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let accent_color = accent.0;

    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);
    let frame_ms = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);

    let body_count = sim.model.body_count;

    Pane2::new(PANE_ID, "molla", PaneAnchor::LeftRail(RailZone::Start), accent_color).show(
        ctx,
        |ui| {
            ui.add_space(2.0);

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
            ui.small("Space · R · MMB pan · LMB+RMB orbit · wheel zoom");
        },
    );
}
