//! `molla_viewer` — Bevy app that runs a molla scene and visualises it.
//!
//! - **`bevy_glacial`** drives the camera (mouse pan / orbit / zoom),
//!   the LOD ground grid, and an axis triad at the world origin.
//! - **`bevy_frost`** themes egui and provides the input firewall so
//!   pointer events over UI panels don't leak into the viewport.
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
use bevy::light::CascadeShadowConfigBuilder;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};
use bevy_frost::prelude::*;
use bevy_glacial::prelude::*;
use molla::math::{Quat as MQuat, Vec3 as MVec3};
use molla::prelude as mp;
use molla::solvers::Solver as _;

const GRID_N: usize = 10; // 10×10 = 100 spheres
const GRID_SPACING: f32 = 1.5;
const SPHERE_RADIUS: f32 = 0.5;
const DROP_HEIGHT: f32 = 12.0;
const FIXED_DT: f64 = 1.0 / 240.0;
const SUBSTEPS_PER_FRAME: usize = 4;

const FLOOR_Y: f64 = 0.0;
const FLOOR_RESTITUTION: f64 = 0.45;
const FLOOR_FRICTION: f64 = 0.6;

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
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (handle_input, step_simulation, sync_transforms).chain(),
        )
        .add_systems(EguiPrimaryContextPass, hud_panel)
        .run();
}

/// Marker linking a Bevy entity to its molla `BodyId`.
#[derive(Component)]
struct MollaBody(mp::BodyId);

/// Marker for the world-origin axis triad so we can draw it each frame.
#[derive(Component)]
struct OriginAxes;

/// Pause / step state.
#[derive(Resource, Default)]
struct SimRunState {
    paused: bool,
    /// Sticky one-shot used by the "step once" button while paused.
    step_once: bool,
}

/// Owns the molla simulation. Lives as a Bevy `Resource`.
#[derive(Resource)]
struct Sim {
    model: mp::Model,
    state_a: mp::State,
    state_b: mp::State,
    control: mp::Control,
    contacts: mp::Contacts,
    solver: mp::SemiImplicitSolver,
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
        let solver = mp::SemiImplicitSolver::new();

        Self {
            model,
            state_a,
            state_b,
            control,
            contacts,
            solver,
        }
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
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Sun + cascade shadows.
    let sun_shadow = CascadeShadowConfigBuilder {
        num_cascades: 1,
        minimum_distance: 0.1,
        maximum_distance: 100.0,
        first_cascade_far_bound: 100.0,
        overlap_proportion: 0.0,
    }
    .build();
    commands.spawn((
        Name::new("Sun"),
        Transform::from_xyz(8.0, 25.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
        DirectionalLight {
            illuminance: 12_000.0,
            shadows_enabled: true,
            ..default()
        },
        sun_shadow,
    ));

    // Camera with the chase rig + ambient + GizmoCamera marker.
    let chase = ChaseCamera {
        min_elevation: -5f32.to_radians(),
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

    // World-origin axis triad. `AxisGizmoPlugin` draws gizmos every
    // frame for any entity tagged with `AxisGizmo`.
    commands.spawn((
        Name::new("OriginAxes"),
        OriginAxes,
        Transform::from_xyz(0.0, 0.001, 0.0),
        AxisGizmo {
            length: 1.5,
            ..AxisGizmo::default()
        },
    ));

    // Sim + Bevy entities.
    let sim = Sim::build();
    let sphere_mesh = meshes.add(Sphere::new(SPHERE_RADIUS).mesh().ico(3).expect("ico"));
    let body_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.35, 0.25),
        metallic: 0.1,
        perceptual_roughness: 0.4,
        ..default()
    });

    let body_q = sim.state_a.body_q.host().expect("body_q").to_vec();
    for (i, pose) in body_q.iter().enumerate() {
        commands.spawn((
            MollaBody(mp::BodyId::from_index(i)),
            Mesh3d(sphere_mesh.clone()),
            MeshMaterial3d(body_mat.clone()),
            Transform::from_xyz(
                pose.position.x as f32,
                pose.position.y as f32,
                pose.position.z as f32,
            ),
        ));
    }

    commands.insert_resource(sim);

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

fn step_simulation(mut sim: ResMut<Sim>, mut run: ResMut<SimRunState>) {
    if run.paused && !run.step_once {
        return;
    }
    run.step_once = false;

    let Sim {
        model,
        state_a,
        state_b,
        control,
        contacts,
        solver,
    } = &mut *sim;
    for _ in 0..SUBSTEPS_PER_FRAME {
        solver
            .step(model, state_a, control, contacts, FIXED_DT, state_b)
            .expect("solver step");
        std::mem::swap(state_a, state_b);
        clamp_to_floor(state_a);
    }
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

/// Top-left HUD: FPS, body count, paused state, control buttons.
fn hud_panel(
    mut contexts: EguiContexts,
    diagnostics: Res<DiagnosticsStore>,
    mut sim: ResMut<Sim>,
    mut run: ResMut<SimRunState>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };

    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);
    let frame_ms = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);

    let body_count = sim.model.body_count;
    let avg_y = {
        let qs = sim.state_a.body_q.host().ok();
        qs.map(|qs| {
            if qs.is_empty() {
                0.0
            } else {
                qs.iter().map(|q| q.position.y).sum::<f64>() / qs.len() as f64
            }
        })
        .unwrap_or(0.0)
    };

    egui::Window::new("molla")
        .anchor(egui::Align2::LEFT_TOP, egui::vec2(10.0, 10.0))
        .resizable(false)
        .collapsible(true)
        .show(ctx, |ui| {
            ui.label(format!("FPS:        {fps:>6.1}"));
            ui.label(format!("Frame:      {frame_ms:>6.2} ms"));
            ui.separator();
            ui.label(format!("Bodies:     {body_count}"));
            ui.label(format!("Substeps:   {SUBSTEPS_PER_FRAME} @ {:>3.0} Hz", 1.0 / FIXED_DT));
            ui.label(format!("Avg y:      {avg_y:>6.2} m"));
            ui.separator();
            ui.horizontal(|ui| {
                let label = if run.paused { "▶  Play" } else { "⏸  Pause" };
                if ui.button(label).clicked() {
                    run.paused = !run.paused;
                }
                if ui
                    .add_enabled(run.paused, egui::Button::new("Step"))
                    .clicked()
                {
                    run.step_once = true;
                }
                if ui.button("Reset").clicked() {
                    sim.reset();
                }
            });
            ui.separator();
            ui.small("Space: pause · R: reset · MMB: pan · LMB+RMB: orbit · Wheel: zoom");
        });
}
