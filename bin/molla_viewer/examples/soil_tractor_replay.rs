use bevy::{
    app::ScheduleRunnerPlugin,
    asset::RenderAssetUsages,
    camera::RenderTarget,
    mesh::PrimitiveTopology,
    prelude::*,
    render::render_resource::TextureFormat,
    render::view::screenshot::{Screenshot, save_to_disk},
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, PrimaryEguiContext, egui};
use std::{path::PathBuf, time::Instant};

#[path = "soil_tractor_replay/data.rs"]
mod data;

#[derive(Resource)]
struct Replay {
    snapshot: data::Snapshot,
    output: PathBuf,
    frames: u32,
    started: Instant,
}

#[derive(Resource)]
struct CaptureTarget(Handle<Image>);

fn main() {
    if let Err(error) = run() {
        eprintln!("soil replay: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if !(2..=3).contains(&args.len()) {
        return Err(
            "usage: soil_tractor_replay RUN_DIRECTORY NEW_OUTPUT_DIRECTORY [final|COMMITTED_FRAME]"
                .into(),
        );
    }
    let run = PathBuf::from(&args[0]);
    let output = PathBuf::from(&args[1]);
    let frame = match args
        .get(2)
        .map(|v| v.to_str().ok_or("frame is not UTF-8"))
        .transpose()?
    {
        None | Some("final") => None,
        Some(value) => Some(value.parse::<u64>().map_err(|e| e.to_string())?),
    };
    let snapshot = data::Snapshot::load_frame(&run, frame)?;
    std::fs::create_dir(&output).map_err(|e| format!("output must be a new directory: {e}"))?;
    let mut metadata = snapshot.metadata();
    metadata["camera"] = serde_json::json!({"eye": [5.8, 3.8, 6.5], "target": [0.0, 0.15, 0.0], "width": 1280, "height": 900});
    metadata["implement_camera"] = serde_json::json!({"eye": [-5.8, 2.5, 4.5], "target": [-0.6, -0.05, 0.0], "width": 1280, "height": 900});
    metadata["display_floor_y_m"] = serde_json::json!(-0.55);
    metadata["render_target"] = serde_json::json!("headless GPU image, no window or compositor");
    std::fs::write(
        output.join("replay.json"),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    let exit = App::new()
        .insert_resource(ClearColor(Color::srgb(0.10, 0.13, 0.17)))
        .insert_resource(Replay {
            snapshot,
            output,
            frames: 0,
            started: Instant::now(),
        })
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(
            std::time::Duration::from_secs_f64(1.0 / 60.0),
        ))
        .add_plugins(EguiPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(Update, capture)
        .add_systems(EguiPrimaryContextPass, labels)
        .run();
    if exit == AppExit::Success {
        Ok(())
    } else {
        Err("renderer did not complete successfully".into())
    }
}

fn point_mesh(points: &[[f32; 3]]) -> Mesh {
    let vertices = [
        Vec3::new(1., 1., 1.),
        Vec3::new(-1., -1., 1.),
        Vec3::new(-1., 1., -1.),
        Vec3::new(1., -1., -1.),
    ]
    .map(|v| v.normalize() * 0.018);
    let mut positions = Vec::with_capacity(points.len() * 12);
    let mut normals = Vec::with_capacity(points.len() * 12);
    for point in points {
        let center = Vec3::from_array(*point);
        for face in [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]] {
            let normal = (vertices[face[1]] - vertices[face[0]])
                .cross(vertices[face[2]] - vertices[face[0]])
                .normalize();
            for index in face {
                positions.push((center + vertices[index]).to_array());
                normals.push(normal.to_array());
            }
        }
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
}

fn setup(
    mut commands: Commands,
    replay: Res<Replay>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let target = images.add(Image::new_target_texture(
        1280,
        900,
        TextureFormat::Rgba8UnormSrgb,
        None,
    ));
    commands.insert_resource(CaptureTarget(target.clone()));
    commands.spawn((
        Camera3d::default(),
        RenderTarget::Image(target.into()),
        PrimaryEguiContext,
        Transform::from_xyz(5.8, 3.8, 6.5).looking_at(Vec3::new(0., 0.15, 0.), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 16000.,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4., 8., 3.).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(9., 7.))),
        MeshMaterial3d(materials.add(Color::srgb(0.22, 0.26, 0.30))),
        Transform::from_xyz(0., -0.55, 0.),
    ));
    commands.spawn((
        Mesh3d(meshes.add(point_mesh(&replay.snapshot.particles))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.48, 0.29, 0.12),
            perceptual_roughness: 1.,
            ..default()
        })),
    ));
    for body in &replay.snapshot.bodies {
        let mesh: Mesh = match body.shape {
            data::Shape::Box(half) => Cuboid::new(half[0] * 2., half[1] * 2., half[2] * 2.).into(),
            data::Shape::Cylinder { radius, half_width } => Cylinder::new(radius, half_width * 2.)
                .mesh()
                .resolution(40)
                .build(),
        };
        let color = match body.role.as_str() {
            "chassis" => Color::srgb(0.18, 0.52, 0.25),
            "wheel" => Color::srgb(0.07, 0.08, 0.09),
            _ => Color::srgb(0.82, 0.33, 0.09),
        };
        let transform = Transform {
            translation: Vec3::from_array(body.position),
            rotation: Quat::from_array(body.rotation),
            ..default()
        };
        commands.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: color,
                perceptual_roughness: 0.85,
                ..default()
            })),
            transform,
        ));
    }
}

fn labels(mut contexts: EguiContexts, replay: Res<Replay>) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    egui::Area::new(egui::Id::new("replay provenance")).fixed_pos(egui::pos2(18., 18.)).show(ctx, |ui| {
        egui::Frame::new().fill(egui::Color32::from_black_alpha(225)).inner_margin(14.).show(ui, |ui| {
            ui.heading("TRACTOR + TINE / RECORDED GPU SOIL");
            ui.label(format!("t = {:.6} s  |  committed frame {}  |  {} material points", replay.snapshot.seconds, replay.snapshot.frame, replay.snapshot.particles.len()));
            ui.colored_label(egui::Color32::YELLOW, "UNCALIBRATED — OFFLINE REPLAY, NOT REAL-TIME ACCEPTANCE");
            ui.label("Recorded body poses and analytic shapes. Points are display markers, not grains.");
            ui.label("No solver stepping, terrain reconstruction or live-render performance measurement.");
        });
    });
}

fn capture(
    mut commands: Commands,
    mut replay: ResMut<Replay>,
    mut exit: MessageWriter<AppExit>,
    target: Res<CaptureTarget>,
    mut camera: Single<&mut Transform, With<Camera3d>>,
) {
    replay.frames += 1;
    if replay.frames == 40 {
        commands
            .spawn(Screenshot::image(target.0.clone()))
            .observe(save_to_disk(replay.output.join("snapshot.png")));
    }
    if replay.frames == 60 {
        **camera =
            Transform::from_xyz(-5.8, 2.5, 4.5).looking_at(Vec3::new(-0.6, -0.05, 0.0), Vec3::Y);
    }
    if replay.frames == 80 {
        commands
            .spawn(Screenshot::image(target.0.clone()))
            .observe(save_to_disk(replay.output.join("implement.png")));
    }
    if replay.frames > 80
        && replay.output.join("snapshot.png").is_file()
        && replay.output.join("implement.png").is_file()
    {
        exit.write(AppExit::Success);
    } else if replay.started.elapsed().as_secs() > 90 {
        eprintln!("soil replay screenshot deadline exceeded");
        exit.write(AppExit::error());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_marker_preserves_centroid_and_outward_faces() {
        let mesh = point_mesh(&[[2., 3., 4.]]);
        let bevy::mesh::VertexAttributeValues::Float32x3(positions) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap()
        else {
            panic!("positions");
        };
        let bevy::mesh::VertexAttributeValues::Float32x3(normals) =
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL).unwrap()
        else {
            panic!("normals");
        };
        assert_eq!(positions.len(), 12);
        for (p, normal) in positions.iter().zip(normals) {
            assert!(
                (Vec3::from_array(*p) - Vec3::new(2., 3., 4.)).dot(Vec3::from_array(*normal)) > 0.
            );
        }
        let center = positions.iter().map(|p| Vec3::from_array(*p)).sum::<Vec3>() / 12.;
        assert!((center - Vec3::new(2., 3., 4.)).length() < 1e-6);
    }
}
