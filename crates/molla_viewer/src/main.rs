//! `molla_viewer` — Bevy app that runs a molla scene and visualises it.
//!
//! - **`bevy_glacial`** drives the camera (mouse pan / orbit / zoom),
//!   the LOD ground grid, and an axis triad at the world origin.
//! - **`bevy_frost`** themes egui, hosts the `molla` control pane on
//!   the left rail (frost `Pane2` + corekit widgets), and runs the
//!   pointer-event firewall so clicks over the panel don't leak into
//!   the viewport.
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

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::light::{CascadeShadowConfigBuilder, NotShadowCaster};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass};

// mimalloc — egui's tessellator allocates a lot per frame; mimalloc
// keeps fragmentation low and small-alloc throughput high. Frost's
// own notes show ~10× speedup vs the system allocator on heavy egui
// scenes.
#[cfg(not(target_arch = "wasm32"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
use bevy_frost::pane::{Pane2, PaneAnchor, RailZone};
use bevy_frost::prelude::*;
use bevy_frost::style::AccentColor;
use bevy_frost::widget::{button, readout, toggle};
use bevy_glacial::prelude::*;
use molla::compute::Engine;
use molla::math::{Quat as MQuat, Vec3 as MVec3};
use molla::prelude as mp;
use molla::solvers::Solver as _;

// 50×50 = 2500 spheres. Stage-2 bench showed the GPU pulls ahead of
// CPU around N≈500-1000 bodies; this scene size makes the win visible.
const GRID_N: usize = 50;
const GRID_SPACING: f32 = 1.2;
const SPHERE_RADIUS: f32 = 0.5;
const DROP_HEIGHT: f32 = 15.0;
const FIXED_DT: f64 = 1.0 / 120.0;
const SUBSTEPS_PER_FRAME: usize = 2;

const FLOOR_Y: f64 = 0.0;
const FLOOR_RESTITUTION: f64 = 0.45;
const FLOOR_FRICTION: f64 = 0.6;

const PANE_ID: &str = "molla_panel";

fn main() {
    let geometry = WindowGeometry::load("molla_viewer");
    App::new()
        .add_plugins(
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
        .add_plugins(FrostPlugin)
        // Drop the transform gizmo + selection ring — we don't need
        // per-body manipulation in the demo.
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

/// Marker linking a Bevy entity to its molla `BodyId`.
#[derive(Component)]
struct MollaBody(mp::BodyId);

/// Pause / step state.
#[derive(Resource, Default)]
struct SimRunState {
    paused: bool,
    /// One-shot used by the "step once" button while paused.
    step_once: bool,
}

/// Per-frame physics timing surfaced in the control pane. Kept on its
/// own resource so the panel can read it without holding a borrow on
/// the `Sim` itself.
#[derive(Resource, Default)]
struct SimStats {
    /// Smoothed total time spent inside `step_simulation`, in
    /// microseconds. Includes GPU upload + dispatch + download when
    /// the GPU backend is active.
    smoothed_step_us: f64,
    /// Smoothed time spent only in the substep loop (just
    /// `solver.step` calls — excludes upload / download / floor
    /// clamp). On GPU this measures host-side encoding cost, not the
    /// async kernel runtime.
    smoothed_solve_us: f64,
}

/// Which solver backend the sim is currently running on.
#[derive(Clone, Debug)]
enum ActiveBackend {
    Cpu,
    Gpu,
    /// User asked for GPU but it's unavailable. Sticky reason string
    /// shown in the panel; the toggle stays off.
    GpuUnavailable(String),
}

impl ActiveBackend {
    fn is_gpu(&self) -> bool {
        matches!(self, Self::Gpu)
    }
}

/// Owns the molla simulation, the active solver, and (lazily) the
/// `Engine` that hosts the GPU backend.
#[derive(Resource)]
struct Sim {
    model: mp::Model,
    state_a: mp::State,
    state_b: mp::State,
    control: mp::Control,
    contacts: mp::Contacts,
    solver: mp::SemiImplicitSolver,
    backend: ActiveBackend,
    engine: Option<Engine>,
    /// Adapter string for display in the panel. None when CPU-only.
    adapter_label: Option<String>,
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

        let model = b.build().expect("model");
        let state_a = model.state().expect("state");
        let state_b = model.state().expect("state");
        let control = model.control();
        let contacts = mp::Contacts::with_capacity(0);

        Self {
            model,
            state_a,
            state_b,
            control,
            contacts,
            solver: mp::SemiImplicitSolver::new_cpu(),
            backend: ActiveBackend::Cpu,
            engine: None,
            adapter_label: None,
        }
    }

    fn switch_to_gpu(&mut self) {
        if self.engine.is_none() {
            match Engine::new_gpu() {
                Ok(e) => {
                    self.adapter_label = e.adapter_info().map(|i| {
                        format!("{} ({})", i.name, i.backend)
                    });
                    self.engine = Some(e);
                }
                Err(e) => {
                    self.backend = ActiveBackend::GpuUnavailable(format!("{e}"));
                    return;
                }
            }
        }
        let engine = self.engine.as_ref().unwrap();
        let device = engine.device().unwrap().clone();
        let queue = engine.queue().unwrap().clone();

        if let Err(e) = self
            .model
            .upload_to_device(&device, &queue)
            .and_then(|_| self.state_a.upload_to_device(&device, &queue))
            .and_then(|_| self.state_b.upload_to_device(&device, &queue))
            .and_then(|_| self.control.upload_to_device(&device, &queue))
        {
            self.backend = ActiveBackend::GpuUnavailable(format!("upload failed: {e}"));
            return;
        }

        match mp::SemiImplicitSolver::new_gpu(engine) {
            Ok(s) => {
                self.solver = s;
                self.backend = ActiveBackend::Gpu;
            }
            Err(e) => {
                self.backend = ActiveBackend::GpuUnavailable(format!("solver init: {e}"));
            }
        }
    }

    fn switch_to_cpu(&mut self) {
        // Pull the latest pose data off the device so subsequent CPU
        // steps and `sync_transforms` see fresh values. Both `state_a`
        // and `state_b` need to come back: the kernel marked each as
        // `DeviceDirty` during dispatch, and the next CPU step will
        // call `host_mut` on `state_b` (which would otherwise panic).
        if self.backend.is_gpu()
            && let Some(engine) = &self.engine
            && let (Some(device), Some(queue)) = (engine.device(), engine.queue())
        {
            let _ = self.state_a.download_from_device(device, queue);
            let _ = self.state_b.download_from_device(device, queue);
        }
        self.solver = mp::SemiImplicitSolver::new_cpu();
        self.backend = ActiveBackend::Cpu;
    }

    fn reset(&mut self) {
        let initial = self
            .model
            .initial_body_q
            .host()
            .expect("initial poses")
            .to_vec();
        self.state_a
            .body_q
            .host_mut()
            .expect("body_q")
            .copy_from_slice(&initial);
        for v in self.state_a.body_qd.host_mut().expect("body_qd") {
            *v = mp::SpatialVector::ZERO;
        }
        // If running on GPU, push the reset state back to the device.
        if self.backend.is_gpu()
            && let Some(engine) = &self.engine
                && let (Some(device), Some(queue)) = (engine.device(), engine.queue()) {
                    let _ = self.state_a.upload_to_device(device, queue);
                }
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    sim: Res<Sim>,
) {
    // Sun + cascade shadows. Cascade tuned for the 60m-wide scene.
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

    // Camera with the chase rig + ambient + GizmoCamera marker.
    // Tuned for a 50×50 sphere grid at 1.2m spacing (~60m wide).
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

    // World-origin axis triad — drawn by `AxisGizmoPlugin`.
    commands.spawn((
        Name::new("OriginAxes"),
        Transform::from_xyz(0.0, 0.001, 0.0),
        AxisGizmo {
            length: 1.5,
            ..AxisGizmo::default()
        },
    ));

    // Body meshes. Two perf decisions:
    //   * Low-poly icosphere (subdivision 1 = 80 tris). At 2500
    //     instances we'd have 2M tris with subdiv 2 — most of those
    //     get sub-pixel coverage and just thrash the rasteriser.
    //   * `NotShadowCaster` on every sphere. The directional light's
    //     shadow pass would otherwise re-draw all 2500 spheres into
    //     the cascade map, doubling the geometry cost. Spheres still
    //     RECEIVE shadows from anything that does cast.
    // Eight materials rendered behind one mesh handle still allow
    // Bevy's render-graph batching by-bin.
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
        "molla viewer: {} bodies, {}Hz physics — Space: pause, R: reset, Esc: quit",
        GRID_N * GRID_N,
        (1.0 / FIXED_DT) as u32
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

    // Snapshot the device handles up front to avoid borrowing
    // `sim.engine` while we mutably borrow the rest of the struct.
    let gpu_handles = if sim.backend.is_gpu() {
        sim.engine.as_ref().and_then(|e| {
            let d = e.device()?.clone();
            let q = e.queue()?.clone();
            Some((d, q))
        })
    } else {
        None
    };

    let Sim {
        model,
        state_a,
        state_b,
        control,
        contacts,
        solver,
        ..
    } = &mut *sim;

    // GPU mode: state_a's device buffer is stale because the
    // previous frame's floor clamp dirtied the host copy. Upload
    // once, then run all substeps device-side without round-tripping.
    if let Some((_d, q)) = gpu_handles.as_ref() {
        let _ = state_a.body_q.upload(q);
        let _ = state_a.body_qd.upload(q);
    }

    let solve_start = std::time::Instant::now();
    for _ in 0..SUBSTEPS_PER_FRAME {
        solver
            .step(model, state_a, control, contacts, FIXED_DT, state_b)
            .expect("solver step");
        std::mem::swap(state_a, state_b);
    }
    let solve_us = solve_start.elapsed().as_secs_f64() * 1_000_000.0;

    // GPU mode: bring the final state_a back for the floor clamp +
    // `sync_transforms`. Only one device→host transfer per frame
    // regardless of substep count, instead of one per substep.
    if let Some((d, q)) = gpu_handles.as_ref() {
        let _ = state_a.body_q.download(d, q);
        let _ = state_a.body_qd.download(d, q);
    }

    clamp_to_floor(state_a);

    // EMA smoothing — α = 0.1 keeps the readout legible while still
    // tracking drift.
    let total_us = frame_start.elapsed().as_secs_f64() * 1_000_000.0;
    stats.smoothed_step_us = 0.9 * stats.smoothed_step_us + 0.1 * total_us;
    stats.smoothed_solve_us = 0.9 * stats.smoothed_solve_us + 0.1 * solve_us;
}

/// Viewer-side floor clamp. Stage-1 molla has no contact pipeline; this
/// keeps spheres above `y = 0` until stage 4 lands proper collisions.
fn clamp_to_floor(state: &mut mp::State) {
    let q = state.body_q.host_mut().expect("body_q");
    let mut hits = vec![false; q.len()];
    for (i, t) in q.iter_mut().enumerate() {
        let radius = SPHERE_RADIUS as f64;
        let min_y = FLOOR_Y + radius;
        if t.position.y < min_y {
            t.position.y = min_y;
            hits[i] = true;
        }
    }
    let qd = state.body_qd.host_mut().expect("body_qd");
    for (i, v) in qd.iter_mut().enumerate() {
        if hits[i] && v.linear.y < 0.0 {
            v.linear.y = -v.linear.y * FLOOR_RESTITUTION;
            v.linear.x *= 1.0 - FLOOR_FRICTION;
            v.linear.z *= 1.0 - FLOOR_FRICTION;
            v.angular *= 1.0 - 0.5 * FLOOR_FRICTION;
        }
    }
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

/// Frost `Pane2` on the left rail with stats + controls.
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

    let backend_label = match &sim.backend {
        ActiveBackend::Cpu => "CPU".to_string(),
        ActiveBackend::Gpu => "GPU".to_string(),
        ActiveBackend::GpuUnavailable(_) => "CPU (GPU N/A)".to_string(),
    };
    let adapter = sim.adapter_label.clone().unwrap_or_else(|| "—".into());
    let backend_err = match &sim.backend {
        ActiveBackend::GpuUnavailable(msg) => Some(msg.clone()),
        _ => None,
    };
    let mut want_gpu = sim.backend.is_gpu();

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

            // Physics timing — the headline indicator of CPU vs GPU.
            // `total` includes the GPU-mode upload + download tax;
            // `solve` is just the dispatch loop (host-side encoding
            // on GPU, true compute on CPU).
            readout(ui, "Phys total", &format!("{:>4.0} µs", stats.smoothed_step_us));
            readout(ui, "Phys solve", &format!("{:>4.0} µs", stats.smoothed_solve_us));
            ui.separator();

            readout(ui, "Backend", &backend_label);
            readout(ui, "Adapter", &adapter);

            let prev = want_gpu;
            toggle(ui, "Use GPU", &mut want_gpu, accent_color);
            if want_gpu != prev {
                if want_gpu {
                    sim.switch_to_gpu();
                } else {
                    sim.switch_to_cpu();
                }
            }
            if let Some(err) = backend_err {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 90, 90),
                    egui::RichText::new(err).small(),
                );
            }
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

// `egui` re-import for the in-function colored_label call.
use bevy_egui::egui;
