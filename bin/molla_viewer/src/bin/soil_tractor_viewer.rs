use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, PrimaryEguiContext, egui};
use molla_solvers::soil_tractor::{
    CHASSIS_HALF_EXTENTS, TINE_HALF_EXTENTS, WHEEL_HALF_WIDTH, WHEEL_RADIUS,
};
use std::{io::Write, path::PathBuf, sync::mpsc::TryRecvError};

#[path = "../soil_tractor_live.rs"]
mod live;

#[derive(Resource)]
struct State {
    worker: live::Worker,
    schedule: live::Schedule,
    snapshot: Option<live::Snapshot>,
    status: String,
    soil_mesh: Handle<Mesh>,
    particle_view: bool,
    transfer: live::Transfer,
    prediction: live::Prediction,
    output: Option<PathBuf>,
    limit: Option<u64>,
    settled_render_frames: u32,
    shot_requested: bool,
    yaw: f32,
    distance: f32,
    headless: bool,
    render_image: Option<Handle<Image>>,
    delayed_shot_frame: Option<u64>,
    profile_stages: bool,
    profile_frame: Option<u64>,
}

#[derive(Component)]
struct Machine(usize);

fn options(
    args: impl Iterator<Item = String>,
) -> Result<
    (
        Option<PathBuf>,
        Option<u64>,
        bool,
        bool,
        live::Transfer,
        live::Prediction,
        bool,
    ),
    String,
> {
    let mut args = args;
    let mut output = None;
    let mut frames = None;
    let mut particles = false;
    let mut headless = false;
    let mut profile_stages = false;
    let mut expected_profile = false;
    let mut expected_preparation = false;
    let mut expected_submissions = false;
    let mut expected_drift = false;
    let mut expected_geometry = false;
    let mut transfer = None;
    let mut prediction = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--particles" if !particles => particles = true,
            "--headless" if !headless => headless = true,
            "--profile-stages" if !profile_stages => profile_stages = true,
            "--machine-prediction" if prediction.is_none() => {
                prediction = Some(live::Prediction::parse(
                    &args.next().ok_or("missing machine prediction")?,
                )?);
            }
            "--soil-transfer" if transfer.is_none() => {
                transfer = Some(live::Transfer::parse(
                    &args.next().ok_or("missing soil transfer")?,
                )?);
            }
            "--expect-geometry" if !expected_geometry => {
                let profile = args.next().ok_or("missing drift geometry profile")?;
                if profile != live::geometry_profile() {
                    return Err(format!(
                        "requested {profile} drift geometry, binary is {}",
                        live::geometry_profile()
                    ));
                }
                expected_geometry = true;
            }
            "--expect-drift" if !expected_drift => {
                let profile = args.next().ok_or("missing native drift profile")?;
                if profile != live::drift_profile() {
                    return Err(format!(
                        "requested {profile} native drift, binary is {}",
                        live::drift_profile()
                    ));
                }
                expected_drift = true;
            }
            "--expect-profile" if !expected_profile => {
                let profile = args.next().ok_or("missing solver profile")?;
                if profile != live::solver_profile() {
                    return Err(format!(
                        "requested {profile} profile, binary is {}",
                        live::solver_profile()
                    ));
                }
                expected_profile = true;
            }
            "--expect-preparation" if !expected_preparation => {
                let profile = args.next().ok_or("missing preparation profile")?;
                if profile != live::preparation_profile() {
                    return Err(format!(
                        "requested {profile} preparation, binary is {}",
                        live::preparation_profile()
                    ));
                }
                live::check_preparation_backend(live::compiled_preparation_backend())?;
                expected_preparation = true;
            }
            "--expect-submissions" if !expected_submissions => {
                let profile = args.next().ok_or("missing submission profile")?;
                if profile != live::submission_profile() {
                    return Err(format!(
                        "requested {profile} submissions, binary is {}",
                        live::submission_profile()
                    ));
                }
                expected_submissions = true;
            }
            "--capture-dir" if output.is_none() => {
                output = Some(PathBuf::from(
                    args.next().ok_or("missing capture directory")?,
                ))
            }
            "--frames" if frames.is_none() => {
                let n = args
                    .next()
                    .ok_or("missing frame count")?
                    .parse::<u64>()
                    .map_err(|_| "invalid frame count")?;
                if !(1..=330).contains(&n) {
                    return Err("frame count must be 1..330".into());
                }
                frames = Some(n);
            }
            _ => return Err(format!("unknown or duplicate argument {arg}")),
        }
    }
    if output.is_some() != frames.is_some() {
        return Err("capture requires both --capture-dir and --frames".into());
    }
    if headless && output.is_none() {
        return Err("headless mode requires bounded capture".into());
    }
    if profile_stages && output.is_none() {
        return Err("stage profiling requires bounded capture".into());
    }
    if expected_drift
        && live::drift_profile() == "deferred"
        && prediction != Some(live::Prediction::Native)
    {
        return Err("deferred native drift requires --machine-prediction native".into());
    }
    Ok((
        output,
        frames,
        particles,
        headless,
        transfer.unwrap_or_default(),
        prediction.unwrap_or_default(),
        profile_stages,
    ))
}

fn main() -> AppExit {
    let (output, limit, particle_view, headless, transfer, prediction, profile_stages) =
        match options(std::env::args().skip(1)) {
            Ok(options) => options,
            Err(error) => {
                eprintln!("{error}");
                return AppExit::error();
            }
        };
    if let Some(path) = &output {
        if let Err(error) = std::fs::create_dir(path) {
            eprintln!("fresh capture directory required: {error}");
            return AppExit::error();
        }
    }
    let worker = match live::Worker::spawn(
        live::ObservationPolicy {
            particles: particle_view,
            capture_end: limit,
        },
        transfer,
        prediction,
        profile_stages,
    ) {
        Ok(worker) => worker,
        Err(error) => {
            eprintln!("{error}");
            return AppExit::error();
        }
    };
    let mut plugins = DefaultPlugins.set(WindowPlugin {
        primary_window: (!headless).then(|| Window {
            title: "Live tractor + tine — physically uncalibrated".into(),
            resolution: (1280, 800).into(),
            ..default()
        }),
        exit_condition: if headless {
            bevy::window::ExitCondition::DontExit
        } else {
            bevy::window::ExitCondition::OnAllClosed
        },
        ..default()
    });
    if headless {
        plugins = plugins.disable::<bevy::winit::WinitPlugin>();
    }
    let mut app = App::new();
    app.add_plugins(plugins);
    if headless {
        app.add_plugins(bevy::app::ScheduleRunnerPlugin::run_loop(
            std::time::Duration::from_secs_f64(1.0 / 60.0),
        ));
    }
    app.insert_resource(ClearColor(Color::srgb(0.12, 0.16, 0.20)))
        .insert_resource(State {
            worker,
            schedule: default(),
            snapshot: None,
            status: "Constructing actual GPU solver; simulation time has not advanced".into(),
            soil_mesh: default(),
            particle_view,
            transfer,
            prediction,
            output,
            limit,
            settled_render_frames: 0,
            shot_requested: false,
            yaw: 2.35,
            distance: 8.0,
            headless,
            render_image: None,
            delayed_shot_frame: None,
            profile_stages,
            profile_frame: None,
        })
        .add_systems(Startup, setup)
        .add_systems(Update, update)
        .add_systems(Last, shutdown_on_exit)
        .add_plugins(EguiPlugin::default())
        .add_systems(EguiPrimaryContextPass, overlay)
        .run()
}

fn grain_vertices(points: &[[f32; 3]]) -> Vec<[f32; 3]> {
    points
        .iter()
        .flat_map(|p| GRAIN_OFFSETS.map(|v| [p[0] + v[0], p[1] + v[1], p[2] + v[2]]))
        .collect()
}

const GRAIN_OFFSETS: [[f32; 3]; 4] = [
    [0.012, 0.012, 0.012],
    [-0.012, -0.012, 0.012],
    [-0.012, 0.012, -0.012],
    [0.012, -0.012, -0.012],
];

fn grain_mesh() -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0f32; 3]; 50_000 * 4]);
    let normals: Vec<_> = (0..50_000)
        .flat_map(|_| GRAIN_OFFSETS.map(|v| Vec3::from(v).normalize().to_array()))
        .collect();
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    let indices: Vec<u32> = (0..50_000)
        .flat_map(|i| [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3].map(|v| i * 4 + v))
        .collect();
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn surface_mesh(heights: &[f32], counts: &[u32]) -> Mesh {
    let c = live::SURFACE;
    assert_eq!(heights.len(), (c.nx * c.nz) as usize);
    assert_eq!(counts.len(), heights.len());
    let positions: Vec<[f32; 3]> = heights
        .iter()
        .enumerate()
        .map(|(i, &height)| {
            [
                c.origin[0] + (i as u32 % c.nx) as f32 * c.spacing + c.spacing * 0.5,
                height,
                c.origin[1] + (i as u32 / c.nx) as f32 * c.spacing + c.spacing * 0.5,
            ]
        })
        .collect();
    let mut indices = Vec::with_capacity(((c.nx - 1) * (c.nz - 1) * 6) as usize);
    let mut normals = vec![Vec3::ZERO; heights.len()];
    for z in 0..c.nz - 1 {
        for x in 0..c.nx - 1 {
            let a = z * c.nx + x;
            for triangle in [[a, a + c.nx, a + 1], [a + 1, a + c.nx, a + c.nx + 1]] {
                if triangle.iter().all(|&i| counts[i as usize] > 0) {
                    let [p, q, r] = triangle.map(|i| Vec3::from(positions[i as usize]));
                    let normal = (q - p).cross(r - p);
                    for i in triangle {
                        normals[i as usize] += normal;
                    }
                    indices.extend_from_slice(&triangle);
                } else {
                    indices.extend_from_slice(&[0; 3]);
                }
            }
        }
    }
    let normals: Vec<_> = normals
        .into_iter()
        .map(|n| n.try_normalize().unwrap_or(Vec3::Y).to_array())
        .collect();
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut state: ResMut<State>,
) {
    let target = if state.headless {
        let mut image = Image::new_target_texture(
            1280,
            800,
            bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
            None,
        );
        image.texture_descriptor.usage |= bevy::render::render_resource::TextureUsages::COPY_SRC;
        let handle = images.add(image);
        state.render_image = Some(handle.clone());
        bevy::camera::RenderTarget::Image(handle.into())
    } else {
        default()
    };
    commands.spawn((
        Camera3d::default(),
        target,
        PrimaryEguiContext,
        Transform::from_xyz(6.0, 4.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(3.0, 7.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(7.0, 0.1, 4.2))),
        MeshMaterial3d(materials.add(Color::srgb(0.19, 0.22, 0.17))),
        Transform::from_xyz(0.0, -0.55, 0.0),
    ));
    for i in 0..6 {
        let mesh = if i == 0 || i == 5 {
            let half = if i == 0 {
                CHASSIS_HALF_EXTENTS
            } else {
                TINE_HALF_EXTENTS
            };
            meshes.add(Cuboid::new(
                2.0 * half.x as f32,
                2.0 * half.y as f32,
                2.0 * half.z as f32,
            ))
        } else {
            meshes.add(Cylinder::new(
                WHEEL_RADIUS as f32,
                2.0 * WHEEL_HALF_WIDTH as f32,
            ))
        };
        let color = if i == 0 {
            Color::srgb(0.17, 0.55, 0.24)
        } else if i == 5 {
            Color::srgb(0.85, 0.35, 0.08)
        } else {
            Color::srgb(0.07, 0.075, 0.08)
        };
        commands.spawn((
            Machine(i),
            Mesh3d(mesh),
            MeshMaterial3d(materials.add(color)),
            Transform::default(),
            Visibility::Hidden,
        ));
    }
    state.soil_mesh = meshes.add(if state.particle_view {
        grain_mesh()
    } else {
        surface_mesh(&vec![live::SURFACE.floor_height; 2560], &vec![0; 2560])
    });
    commands.spawn((
        Mesh3d(state.soil_mesh.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.48, 0.28, 0.12),
            cull_mode: None,
            ..default()
        })),
        bevy::camera::visibility::NoFrustumCulling,
    ));
}

fn record(state: &State, value: serde_json::Value) -> Result<(), String> {
    eprintln!("soil-live {value}");
    if let Some(path) = &state.output {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.join("frames.jsonl"))
            .map_err(|e| e.to_string())?;
        writeln!(file, "{value}").map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn capture_target(state: &State) -> Screenshot {
    match &state.render_image {
        Some(image) => Screenshot::image(image.clone()),
        None => Screenshot::primary_window(),
    }
}

fn fail(state: &mut State, error: String) {
    state.schedule.failed = true;
    state.schedule.busy = false;
    state.status = format!("FAILED — last committed state retained: {error}");
    let _ = record(
        state,
        serde_json::json!({"event":"failure", "error":error, "acceptance":false}),
    );
}

fn update(
    mut commands: Commands,
    mut state: ResMut<State>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut machine: Query<(&Machine, &mut Transform, &mut Visibility), Without<Camera3d>>,
    mut camera: Query<&mut Transform, (With<Camera3d>, Without<Machine>)>,
    mut exit: MessageWriter<AppExit>,
) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
        return;
    }
    if keys.just_pressed(KeyCode::Space) {
        state.schedule.paused = !state.schedule.paused;
    }
    let turn = u8::from(keys.pressed(KeyCode::ArrowRight)) as f32
        - u8::from(keys.pressed(KeyCode::ArrowLeft)) as f32;
    let zoom = u8::from(keys.pressed(KeyCode::ArrowDown)) as f32
        - u8::from(keys.pressed(KeyCode::ArrowUp)) as f32;
    state.yaw += turn * time.delta_secs();
    state.distance = (state.distance + zoom * time.delta_secs() * 3.0).clamp(3.0, 14.0);
    if let Ok(mut camera) = camera.single_mut() {
        *camera = Transform::from_xyz(
            state.distance * state.yaw.cos(),
            state.distance * 0.5,
            state.distance * state.yaw.sin(),
        )
        .looking_at(Vec3::new(0.0, 0.25, 0.0), Vec3::Y);
    }
    loop {
        let event = state.worker.events.lock().unwrap().try_recv();
        match event {
            Ok(live::Event::Ready {
                adapter,
                construction_ms,
                newton_backend,
                transfer,
                transfer_info,
                prediction,
                native_prediction,
                compiled_preparation_backend,
                composed_submissions,
                profile_stages,
                native_drift_deferred,
                unrestricted_drift_backend,
            }) => {
                if transfer != state.transfer
                    || prediction != state.prediction
                    || native_prediction != (prediction == live::Prediction::Native)
                    || live::check_preparation_backend(compiled_preparation_backend).is_err()
                    || live::check_submission_backend(Some(composed_submissions)).is_err()
                    || profile_stages != state.profile_stages
                    || live::check_drift_backend(prediction, native_drift_deferred).is_err()
                    || live::check_geometry_backend(Some(unrestricted_drift_backend)).is_err()
                {
                    fail(
                        &mut state,
                        "worker backend differs from requested transfer or machine prediction"
                            .into(),
                    );
                    break;
                }
                if let Err(error) = record(
                    &state,
                    serde_json::json!({"event":"ready", "adapter":adapter, "construction_ms":construction_ms,"solver_profile":live::solver_profile(),"newton_backend":newton_backend,
                    "soil_transfer":transfer.name(),"transfer_info":{"fixed_order":transfer_info.fixed_order,"shared_chunks":transfer_info.shared_chunks,"grid_reuse":transfer_info.grid_reuse,"particle_ordering":transfer_info.particle_ordering,"parallel_sort":transfer_info.parallel_sort,"gather_bytes":transfer_info.gather_bytes,"ordering_bytes":transfer_info.ordering_bytes},
                    "machine_prediction":prediction.name(),"native_prediction":native_prediction,
                    "drift_geometry_profile":live::geometry_profile(),"prepared_unrestricted_drift_backend":unrestricted_drift_backend,"drift_geometry_scope":"prepared paired unrestricted drift pipeline; not prediction, momentum or refresh arithmetic",
                    "native_drift_profile":live::drift_profile(),"native_drift_deferred":native_drift_deferred,"native_drift_scope":"rates-only native prediction when deferred; soil projection followed by one final corrected pose drift; otherwise full prediction plus corrected drift",
                    "preparation_profile":live::preparation_profile(),"compiled_preparation_backend":compiled_preparation_backend,
                    "submission_profile":live::submission_profile(),"composed_submissions":composed_submissions,"submission_scope":"projected guard/solve/publication sequence, not whole-frame queue traffic",
                    "stage_profiling":profile_stages,"stage_profile_version":1,"stage_profile_scope":"opt-in blocking diagnostic reads after committed physics; disabled in ordinary runs",
                    "particles":50000, "dt":live::DT, "wheel_torque_nm":live::TORQUE, "physical_calibration":"uncalibrated", "acceptance":false,"surface_observation_version":2,"load_observation_version":1,"presentation":if state.headless {"offscreen Bevy image"} else {"interactive window"},
                    "renderer":if state.particle_view { "explicit particle glyph diagnostic" } else { "geometric point-envelope surface" },
                    "surface_grid":{"origin":live::SURFACE.origin,"nx":live::SURFACE.nx,"nz":live::SURFACE.nz,"spacing":live::SURFACE.spacing,"floor_height":live::SURFACE.floor_height},
                    "observation":"asynchronous capacity-one surface and force bridges; initial/final surfaces and final loads wait for exact frames; normal geometry and force samples are labelled by captured frame; full particles only for glyph mode or endpoint capture; not a fully resident or constitutive contact surface"}),
                ) {
                    fail(&mut state, error);
                }
            }
            Ok(live::Event::Profile(profile)) => {
                if !state.profile_stages
                    || Some(profile.frame()) == state.profile_frame
                    || state.schedule.frame.and_then(|f| f.checked_add(1)) != Some(profile.frame())
                {
                    fail(
                        &mut state,
                        "unexpected or out-of-order stage profile".into(),
                    );
                    break;
                }
                if let Err(error) = record(&state, profile.record()) {
                    fail(&mut state, error);
                    break;
                }
                state.profile_frame = Some(profile.frame());
            }
            Ok(live::Event::Frame(snapshot)) => {
                if state.profile_stages
                    && snapshot.frame > 0
                    && state.profile_frame != Some(snapshot.frame)
                {
                    fail(&mut state, "missing committed-frame stage profile".into());
                    break;
                }
                if let Err(error) = state.schedule.accept(snapshot.frame) {
                    fail(&mut state, error);
                    break;
                }
                if let Some(root) = &state.output {
                    if snapshot.frame == 0 || state.limit == Some(snapshot.frame) {
                        if snapshot.particles.len() != 50_000
                            || snapshot.surface_frame != snapshot.frame
                            || !snapshot.surface_waited
                            || (snapshot.frame > 0
                                && (!snapshot.load_waited
                                    || snapshot
                                        .load
                                        .as_ref()
                                        .is_none_or(|s| s.frame != snapshot.frame)))
                        {
                            fail(&mut state, "missing requested particle endpoint".into());
                            break;
                        }
                        let surface_path = root.join(format!("surface-{:06}.json", snapshot.frame));
                        let surface_data = serde_json::json!({"frame":snapshot.surface_frame,"generation":snapshot.surface_generation,"simulated_seconds":snapshot.seconds,"heights":snapshot.surface.heights,"counts":snapshot.surface.counts,"outside_particles":snapshot.surface.outside_particles});
                        if let Err(error) =
                            std::fs::File::create_new(surface_path).and_then(|mut file| {
                                file.write_all(surface_data.to_string().as_bytes())
                            })
                        {
                            fail(&mut state, format!("surface capture failed: {error}"));
                            break;
                        }
                        let path = root.join(format!("particles-{:06}.bin", snapshot.frame));
                        let bytes: Vec<_> = snapshot
                            .particles
                            .iter()
                            .flatten()
                            .flat_map(|v| v.to_le_bytes())
                            .collect();
                        if let Err(error) = std::fs::File::create_new(path)
                            .and_then(|mut file| file.write_all(&bytes))
                        {
                            fail(&mut state, format!("particle capture failed: {error}"));
                            break;
                        }
                    }
                }
                if let Err(error) = record(
                    &state,
                    serde_json::json!({"event":"frame", "committed_frame":snapshot.frame, "simulated_seconds":snapshot.seconds,
                    "physics_ms":snapshot.physics_ms,"observation_ms":snapshot.observation_ms,"nominal_slip":snapshot.slip,"tine_world_force_n":snapshot.load.as_ref().map(|s|s.force),
                    "load_frame":snapshot.load.as_ref().map(|s|s.frame),"load_interval_seconds":snapshot.load.as_ref().map(|s|s.seconds),"load_age_frames":snapshot.load.as_ref().map(|s|snapshot.frame-s.frame),"load_readback_bytes":snapshot.load_readback_bytes,"load_request_poll_ms":snapshot.load_ms,"load_waited":snapshot.load_waited,
                    "surface_request_poll_ms":snapshot.surface_ms,"surface_readback_bytes":snapshot.surface_readback_bytes,"surface_frame":snapshot.surface_frame,"surface_generation":snapshot.surface_generation,"surface_age_frames":snapshot.frame-snapshot.surface_frame,"surface_waited":snapshot.surface_waited,"particle_snapshot_points":snapshot.particles.len(),"outside_surface_particles":snapshot.surface.outside_particles,
                    "load_scope":"frame-integrated world impulse / sum of soil substep dt; not fixed-origin torque or work",
                    "bodies":snapshot.poses.iter().map(|p| serde_json::json!({"position":p.position,"rotation":p.rotation})).collect::<Vec<_>>()}),
                ) {
                    fail(&mut state, error);
                    break;
                }
                for (body, mut transform, mut visible) in &mut machine {
                    let pose = &snapshot.poses[body.0];
                    transform.translation = Vec3::from(pose.position);
                    transform.rotation = Quat::from_array(pose.rotation);
                    *visible = Visibility::Visible;
                }
                if let Some(mesh) = meshes.get_mut(&state.soil_mesh) {
                    if !state.particle_view {
                        if state.snapshot.as_ref().is_none_or(|previous| {
                            previous.surface_frame != snapshot.surface_frame
                                || previous.surface_generation != snapshot.surface_generation
                        }) {
                            *mesh =
                                surface_mesh(&snapshot.surface.heights, &snapshot.surface.counts);
                        }
                    } else if let Some(VertexAttributeValues::Float32x3(vertices)) =
                        mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
                    {
                        *vertices = grain_vertices(&snapshot.particles);
                    }
                }
                state.status = "Committed state from actual coupled GPU solver".into();
                state.snapshot = Some(snapshot);
            }
            Ok(live::Event::Surface(update)) => {
                let Some(snapshot) = state.snapshot.as_mut() else {
                    fail(&mut state, "surface arrived before committed state".into());
                    break;
                };
                if let Err(error) = snapshot.apply_surface(update) {
                    fail(&mut state, error);
                    break;
                }
                let entry = serde_json::json!({"event":"surface", "committed_frame":snapshot.frame,
                    "surface_frame":snapshot.surface_frame,"surface_generation":snapshot.surface_generation,
                    "surface_age_frames":snapshot.frame-snapshot.surface_frame,
                    "surface_readback_bytes":snapshot.surface_readback_bytes,"surface_request_poll_ms":snapshot.surface_ms,
                    "surface_waited":false});
                if !state.particle_view {
                    if let Some(mesh) = meshes.get_mut(&state.soil_mesh) {
                        let snapshot = state.snapshot.as_ref().unwrap();
                        *mesh = surface_mesh(&snapshot.surface.heights, &snapshot.surface.counts);
                    }
                }
                if let Err(error) = record(&state, entry) {
                    fail(&mut state, error);
                    break;
                }
            }
            Ok(live::Event::Loads(update)) => {
                let Some(snapshot) = state.snapshot.as_mut() else {
                    fail(&mut state, "force arrived before committed state".into());
                    break;
                };
                if let Err(error) = snapshot.apply_load(update) {
                    fail(&mut state, error);
                    break;
                }
                let sample = snapshot.load.as_ref().unwrap();
                let entry = serde_json::json!({"event":"loads","committed_frame":snapshot.frame,
                    "load_frame":sample.frame,"load_interval_seconds":sample.seconds,
                    "load_age_frames":snapshot.frame-sample.frame,"tine_world_force_n":sample.force,
                    "load_readback_bytes":snapshot.load_readback_bytes,"load_request_poll_ms":snapshot.load_ms,
                    "load_waited":false});
                if let Err(error) = record(&state, entry) {
                    fail(&mut state, error);
                    break;
                }
            }
            Ok(live::Event::Failed(error)) => {
                fail(&mut state, error);
                break;
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                if !state.schedule.failed {
                    fail(&mut state, "physics worker disconnected".into());
                }
                break;
            }
        }
    }
    if state.output.is_some() && state.delayed_shot_frame.is_none() {
        if let Some(snapshot) = state
            .snapshot
            .as_ref()
            .filter(|s| s.surface_frame < s.frame)
        {
            let frame = snapshot.frame;
            if let Err(error) = record(
                &state,
                serde_json::json!({"event":"delayed_capture_requested", "committed_frame":frame, "surface_frame":snapshot.surface_frame, "screenshot":"delayed-frame.png", "acceptance":false}),
            ) {
                fail(&mut state, error);
            } else {
                commands.spawn(capture_target(&state)).observe(save_to_disk(
                    state.output.as_ref().unwrap().join("delayed-frame.png"),
                ));
                state.delayed_shot_frame = Some(frame);
            }
        }
    }
    let limit = state.limit;
    if state
        .schedule
        .request(keys.just_pressed(KeyCode::Enter), limit)
    {
        if let Err(error) = state.worker.step() {
            fail(&mut state, error);
        }
    }
    if state.output.is_some() && state.schedule.failed {
        exit.write(AppExit::error());
        return;
    }
    if limit.is_some_and(|limit| state.schedule.frame == Some(limit)) {
        state.settled_render_frames += 1;
        let path = state.output.as_ref().unwrap().join("committed-frame.png");
        if state.settled_render_frames == 30 {
            commands
                .spawn(capture_target(&state))
                .observe(save_to_disk(path.clone()));
            state.shot_requested = true;
        }
        let delayed_ready = state.delayed_shot_frame.is_none()
            || state
                .output
                .as_ref()
                .unwrap()
                .join("delayed-frame.png")
                .metadata()
                .is_ok_and(|meta| meta.len() > 0);
        if state.shot_requested && path.metadata().is_ok_and(|meta| meta.len() > 0) && delayed_ready
        {
            if let Err(error) = record(
                &state,
                serde_json::json!({"event":"capture_complete","committed_frame":state.schedule.frame,"screenshot":"committed-frame.png","acceptance":false}),
            ) {
                fail(&mut state, error);
                exit.write(AppExit::error());
            } else {
                exit.write(AppExit::Success);
            }
        }
    }
}

fn overlay(mut contexts: EguiContexts, state: Res<State>) -> Result {
    egui::Window::new("50k live soil + tractor + tine — PHYSICALLY UNCALIBRATED")
        .fixed_pos([16.0, 16.0])
        .collapsible(false)
        .show(contexts.ctx_mut()?, |ui| {
            ui.label(&state.status);
            ui.label(format!(
                "Unrestricted drift geometry: {}",
                live::geometry_profile()
            ));
            if state.prediction == live::Prediction::Native {
                ui.label(format!(
                    "Native predicted pose: {} drift",
                    live::drift_profile()
                ));
            }
            if state.profile_stages {
                ui.label("Stage profiling ON: blocking diagnostic reads; not production timing");
            }
            if let Some(s) = &state.snapshot {
                ui.label(format!(
                    "Committed frame {} | simulated {:.6} s",
                    s.frame, s.seconds
                ));
                ui.label(format!(
                    "Physics {:.1} ms | observation {:.1} ms",
                    s.physics_ms, s.observation_ms
                ));
                ui.label(format!(
                    "Surface frame {} | generation {} | age {} frames{}",
                    s.surface_frame,
                    s.surface_generation,
                    s.frame - s.surface_frame,
                    if s.surface_waited {
                        " | exact capture"
                    } else {
                        " | asynchronous"
                    }
                ));
                ui.label(format!(
                    "Nominal slip {:?} | tine force {:?} N",
                    s.slip.map(|v| (v * 100.0).round() / 100.0),
                    s.load.as_ref().map(|sample| sample.force)
                ));
                if let Some(sample) = &s.load {
                    ui.label(format!(
                        "Force frame {} | age {} frames | interval {:.6} s | {}",
                        sample.frame,
                        s.frame - sample.frame,
                        sample.seconds,
                        if s.load_waited {
                            "exact capture"
                        } else {
                            "asynchronous"
                        }
                    ));
                } else {
                    ui.label("Force observation pending; no measured sample yet");
                }
            }
            ui.label("Space: pause/resume | Enter: one frame | arrows: orbit/zoom | Esc: quit");
            ui.label(format!("Newton profile: {}", live::solver_profile()));
            ui.label(format!(
                "Contact preparation: {}",
                live::preparation_profile()
            ));
            ui.label(format!("Soil transfer: {}", state.transfer.name()));
            ui.label(format!(
                "Projected submissions: {}",
                live::submission_profile()
            ));
            ui.label(format!("Machine prediction: {}", state.prediction.name()));
            ui.label(if state.particle_view {
                "Explicit particle diagnostic; full position readback enabled."
            } else {
                "Geometric point envelope; bounded 61,440-byte terrain bridge."
            });
            ui.label("Not a calibrated contact surface or real-time acceptance.");
            if state.schedule.busy {
                ui.label("Physics pending");
            } else if state.schedule.paused {
                ui.label("Paused");
            }
        });
    Ok(())
}

fn shutdown_on_exit(mut exits: MessageReader<AppExit>, mut state: ResMut<State>) {
    if exits.read().next().is_some() {
        state.worker.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn capture_arguments_are_bounded_and_paired() {
        let parse = |v: &[&str]| options(v.iter().map(|s| s.to_string()));
        assert_eq!(
            parse(&[]).unwrap(),
            (
                None,
                None,
                false,
                false,
                live::Transfer::Atomic,
                live::Prediction::Featherstone,
                false
            )
        );
        assert!(parse(&["--expect-profile", live::solver_profile()]).is_ok());
        assert!(parse(&["--expect-geometry", live::geometry_profile()]).is_ok());
        for args in [
            vec!["--expect-geometry"],
            vec!["--expect-geometry", "unknown"],
            vec![
                "--expect-geometry",
                if live::geometry_profile() == "eager" {
                    "native-lazy"
                } else {
                    "eager"
                },
            ],
            vec![
                "--expect-geometry",
                live::geometry_profile(),
                "--expect-geometry",
                live::geometry_profile(),
            ],
        ] {
            assert!(parse(&args).is_err());
        }
        assert!(
            parse(&[
                "--expect-drift",
                live::drift_profile(),
                "--machine-prediction",
                "native"
            ])
            .is_ok()
        );
        for args in [
            vec!["--expect-drift"],
            vec!["--expect-drift", "unknown"],
            vec![
                "--expect-drift",
                if live::drift_profile() == "full" {
                    "deferred"
                } else {
                    "full"
                },
            ],
            vec![
                "--expect-drift",
                live::drift_profile(),
                "--expect-drift",
                live::drift_profile(),
                "--machine-prediction",
                "native",
            ],
        ] {
            assert!(parse(&args).is_err());
        }
        if live::drift_profile() == "deferred" {
            assert!(parse(&["--expect-drift", "deferred"]).is_err());
            assert!(
                parse(&[
                    "--expect-drift",
                    "deferred",
                    "--machine-prediction",
                    "featherstone"
                ])
                .is_err()
            );
        }
        assert!(parse(&["--expect-preparation", live::preparation_profile()]).is_ok());
        assert!(parse(&["--expect-preparation"]).is_err());
        assert!(parse(&["--expect-preparation", "unknown"]).is_err());
        assert!(parse(&["--expect-submissions", live::submission_profile()]).is_ok());
        assert!(parse(&["--expect-submissions"]).is_err());
        assert!(parse(&["--expect-submissions", "unknown"]).is_err());
        assert!(
            parse(&[
                "--expect-submissions",
                if live::submission_profile() == "separate" {
                    "composed"
                } else {
                    "separate"
                }
            ])
            .is_err()
        );
        assert!(
            parse(&[
                "--expect-submissions",
                live::submission_profile(),
                "--expect-submissions",
                live::submission_profile()
            ])
            .is_err()
        );
        let other = if live::preparation_profile() == "default" {
            "cooperative"
        } else {
            "default"
        };
        assert!(parse(&["--expect-preparation", other]).is_err());
        assert!(
            parse(&[
                "--expect-preparation",
                live::preparation_profile(),
                "--expect-preparation",
                live::preparation_profile()
            ])
            .is_err()
        );
        assert!(
            parse(&[
                "--expect-profile",
                if live::solver_profile() == "soa" {
                    "volatile"
                } else {
                    "soa"
                }
            ])
            .is_err()
        );
        assert!(
            parse(&[
                "--expect-profile",
                live::solver_profile(),
                "--expect-profile",
                live::solver_profile()
            ])
            .is_err()
        );
        assert_eq!(parse(&[]).unwrap().4, live::Transfer::Atomic);
        for transfer in [
            live::Transfer::Atomic,
            live::Transfer::FixedOrder,
            live::Transfer::GridReuse,
        ] {
            assert_eq!(
                parse(&["--soil-transfer", transfer.name()]).unwrap().4,
                transfer
            );
        }
        assert_eq!(
            parse(&["--particles"]).unwrap(),
            (
                None,
                None,
                true,
                false,
                live::Transfer::Atomic,
                live::Prediction::Featherstone,
                false
            )
        );
        assert_eq!(
            parse(&["--capture-dir", "fresh", "--frames", "33"]).unwrap(),
            (
                Some("fresh".into()),
                Some(33),
                false,
                false,
                live::Transfer::Atomic,
                live::Prediction::Featherstone,
                false
            )
        );
        assert_eq!(
            parse(&["--capture-dir", "fresh", "--frames", "4", "--headless"]).unwrap(),
            (
                Some("fresh".into()),
                Some(4),
                false,
                true,
                live::Transfer::Atomic,
                live::Prediction::Featherstone,
                false
            )
        );
        for args in [
            vec!["--profile-stages"],
            vec!["--profile-stages", "--profile-stages"],
            vec!["--machine-prediction"],
            vec!["--machine-prediction", "unknown"],
            vec![
                "--machine-prediction",
                "native",
                "--machine-prediction",
                "featherstone",
            ],
            vec!["--frames", "1"],
            vec!["--capture-dir", "fresh"],
            vec!["--frames", "0"],
            vec!["--frames", "331"],
            vec!["--frames", "2", "--frames", "3"],
            vec!["--wat"],
            vec!["--particles", "--particles"],
            vec!["--headless"],
            vec!["--headless", "--headless"],
            vec!["--expect-profile"],
            vec!["--expect-profile", "unknown"],
            vec!["--soil-transfer"],
            vec!["--soil-transfer", "shared-chunk"],
            vec![
                "--soil-transfer",
                "atomic",
                "--soil-transfer",
                "fixed-order",
            ],
        ] {
            assert!(parse(&args).is_err());
        }
        for prediction in [live::Prediction::Featherstone, live::Prediction::Native] {
            assert_eq!(
                parse(&["--machine-prediction", prediction.name()])
                    .unwrap()
                    .5,
                prediction
            );
        }
        assert!(
            parse(&[
                "--capture-dir",
                "fresh",
                "--frames",
                "4",
                "--profile-stages"
            ])
            .unwrap()
            .6
        );
    }
    #[test]
    fn particle_glyphs_preserve_centres_and_body_local_geometry() {
        let points = [[1.0, 2.0, 3.0], [-2.0, 0.0, 1.0]];
        let vertices = grain_vertices(&points);
        assert_eq!(vertices.len(), 8);
        for (chunk, point) in vertices.chunks_exact(4).zip(points) {
            for axis in 0..3 {
                assert!(
                    (chunk.iter().map(|p| p[axis]).sum::<f32>() / 4.0 - point[axis]).abs() < 1e-6
                );
            }
        }
        let mesh = grain_mesh();
        assert_eq!(mesh.count_vertices(), 200_000);
        assert_eq!(mesh.indices().unwrap().len(), 600_000);
    }

    #[test]
    fn surface_mesh_keeps_fixed_capacity_without_bridging_empty_cells() {
        let heights = vec![-0.25; 2560];
        let mut counts = vec![0; 2560];
        let empty = surface_mesh(&heights, &counts);
        assert_eq!(empty.count_vertices(), 2560);
        let Some(Indices::U32(empty_indices)) = empty.indices() else {
            panic!("indices");
        };
        assert!(empty_indices.iter().all(|&i| i == 0));
        counts[0] = 1;
        counts[1] = 1;
        counts[64] = 1;
        let mesh = surface_mesh(&heights, &counts);
        let Some(Indices::U32(indices)) = mesh.indices() else {
            panic!("indices");
        };
        assert_eq!(indices.len(), empty_indices.len());
        assert_eq!(&indices[..3], &[0, 64, 1]);
        assert!(indices[3..].iter().all(|&i| i == 0));
        let Some(VertexAttributeValues::Float32x3(normals)) =
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
        else {
            panic!("normals");
        };
        assert_eq!(normals[0], [0.0, 1.0, 0.0]);
    }
}
