//! `molla_viewer` — Bevy app that runs a molla scene and visualises it.
//!
//! Stage 1 demo: a 10×10 grid of spheres falling under gravity. The molla
//! solver runs each Bevy frame; per-body world poses are copied into Bevy
//! `Transform` components for rendering.
//!
//! Press **R** to reset, **ESC** to quit.

use bevy::prelude::*;
use molla::math::{Quat as MQuat, Vec3 as MVec3};
use molla::prelude as mp;
use molla::solvers::Solver as _;

const GRID_N: usize = 10; // 10×10 = 100 spheres
const GRID_SPACING: f32 = 1.5;
const SPHERE_RADIUS: f32 = 0.5;
const DROP_HEIGHT: f32 = 12.0;
const FIXED_DT: f64 = 1.0 / 240.0; // 240 Hz physics
const SUBSTEPS_PER_FRAME: usize = 4; // ~60Hz frame, 4 sub-steps = 240Hz

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "molla viewer — stage 1".into(),
                        resolution: (1280u32, 800u32).into(),
                        ..default()
                    }),
                    ..default()
                }),
        )
        .insert_resource(ClearColor(Color::srgb(0.05, 0.07, 0.12)))
        .insert_resource(GlobalAmbientLight {
            color: Color::srgb(0.8, 0.85, 1.0),
            brightness: 80.0,
            ..default()
        })
        .add_systems(Startup, setup)
        .add_systems(Update, (handle_input, step_simulation, sync_transforms).chain())
        .run();
}

/// Marker linking a Bevy entity to its molla `BodyId`.
#[derive(Component)]
struct MollaBody(mp::BodyId);

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
    fn build() -> (Self, Vec<MVec3>) {
        let mut b = mp::ModelBuilder::new();
        b.set_angular_damping(0.05);

        let mut initial_positions = Vec::with_capacity(GRID_N * GRID_N);
        for ix in 0..GRID_N {
            for iz in 0..GRID_N {
                let x = (ix as f64 - (GRID_N as f64 - 1.0) * 0.5) * GRID_SPACING as f64;
                let z = (iz as f64 - (GRID_N as f64 - 1.0) * 0.5) * GRID_SPACING as f64;
                let y = DROP_HEIGHT as f64 + 0.4 * ((ix + iz) as f64 / 2.0).sin();
                let pos = MVec3::new(x, y, z);
                initial_positions.push(pos);

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

        (
            Self {
                model,
                state_a,
                state_b,
                control,
                contacts,
                solver,
            },
            initial_positions,
        )
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
    // Camera — looks across the grid from above, slight tilt.
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 14.0, 22.0).looking_at(Vec3::new(0.0, 4.0, 0.0), Vec3::Y),
    ));

    // Sun-like directional light with shadows.
    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(8.0, 20.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Ground plane (visual only — stage 1 has no contacts; spheres will
    // pass through. Stage 4 wires it up as a collision plane.)
    let ground_mesh = meshes.add(Cuboid::new(50.0, 0.1, 50.0));
    let ground_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.18, 0.22, 0.28),
        perceptual_roughness: 0.9,
        ..default()
    });
    commands.spawn((
        Mesh3d(ground_mesh),
        MeshMaterial3d(ground_mat),
        Transform::from_xyz(0.0, -0.05, 0.0),
    ));

    // Spawn the molla scene + matching Bevy entities.
    let (sim, _initial_positions) = Sim::build();

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
        "molla viewer: {} bodies, {}Hz physics, press R to reset, ESC to quit",
        GRID_N * GRID_N,
        (1.0 / FIXED_DT) as u32
    );
}

fn handle_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut exit: MessageWriter<AppExit>,
    mut sim: ResMut<Sim>,
) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
    if keys.just_pressed(KeyCode::KeyR) {
        sim.reset();
        info!("molla: scene reset");
    }
}

fn step_simulation(mut sim: ResMut<Sim>) {
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
