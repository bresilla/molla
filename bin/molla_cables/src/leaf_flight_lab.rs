use crate::{draw, leafy_draw, leafy_model::Leaf, studio, video_capture};
use eframe::egui::{self, Color32};
use molla_compute::Engine;
use molla_math::{Quat, Transform, Vec3};
use molla_solvers::{
    airflow::{AirConfig, Grid},
    attachments::{Anchor, Attachment, BladeFrame, Config, GpuAttachments, LeafBody, Particle},
};
use std::{f64::consts::PI, path::Path};

pub fn capture(args: &[String]) -> video_capture::Result<()> {
    let value = |name: &str| {
        args.windows(2)
            .find(|p| p[0] == name)
            .map(|p| p[1].as_str())
    };
    let path = Path::new(value("--video").ok_or("leaf lab requires --video PATH")?);
    let seconds: f64 = value("--seconds").unwrap_or("8").parse()?;
    if !seconds.is_finite() || !(0.1..=120.0).contains(&seconds) {
        return Err("leaf lab duration must be between 0.1 and 120 seconds".into());
    }
    let still_air = args.iter().any(|s| s == "--still-air");
    let mixed = args.iter().any(|s| s == "--mixed-wind");
    let engine = Engine::new_gpu()?;
    let mut coefficients = Vec::new();
    let mut particles = Vec::new();
    let mut leaves = Vec::new();
    for i in 0..9 {
        let length = 0.1 + 0.012 * (i % 3) as f64;
        let mass = 0.0003;
        let inertia = [1.1, 1.4, 2.2][i % 3];
        let q = Quat::from_rotation_y((i / 3) as f64 * 2.1)
            * Quat::from_rotation_z(0.6 + 0.1 * (i / 3) as f64);
        let frame = BladeFrame {
            direction: q * Vec3::X,
            normal: q * Vec3::Y,
            angular_velocity: if i < 3 {
                Vec3::ZERO
            } else {
                q * Vec3::new(0.7, 0.2, 0.0)
            },
        };
        let mut p = Particle::attached(Vec3::new(
            (i % 3) as f64 * 0.35 - 0.35,
            3.0,
            (i / 3) as f64 * 0.3 - 0.3,
        ));
        p.attached = false;
        p.leaf = Some(LeafBody {
            frame,
            length,
            width: length * 0.6,
            area: if mixed {
                mass * (8.0 + 0.6 * i as f64)
            } else {
                4.0 * mass / (PI * 1.225 * length * inertia)
            },
            camber: if mixed {
                length * (0.03 + 0.006 * i as f64)
            } else {
                0.0
            },
        });
        coefficients.push(Attachment {
            mass,
            stiffness: 0.6,
            damping: 0.015,
            strength: 0.02,
            drag: 0.00066,
            radius: 0.002,
            petiole: None,
        });
        particles.push(p);
        leaves.push(Leaf {
            owner: 0,
            rest: Transform::IDENTITY,
            anchor: Vec3::ZERO,
            offset: Vec3::ZERO,
            direction: frame.direction,
            normal: frame.normal,
            length,
            tint: i as f64 / 8.0,
            released_direction: frame.direction,
            released_normal: frame.normal,
        });
    }
    let anchors: Vec<_> = particles
        .iter()
        .map(|p| Anchor {
            position: p.position,
            velocity: Vec3::ZERO,
            valid: false,
        })
        .collect();
    let mut gpu = GpuAttachments::new(&engine, coefficients.clone(), &particles)?;
    let mut video =
        video_capture::Capture::new_with_title(path, "Molla | Falling-leaf circulation")?;
    let mut camera = draw::Camera {
        yaw: 0.15,
        pitch: 0.18,
        distance: 4.6,
        target: Vec3::new(0.0, 1.6, 0.0),
    };
    let mut tracks = vec![Vec::<[f64; 10]>::new(); particles.len()];
    let frames = (seconds * video_capture::FPS as f64).ceil() as usize;
    let c = Config {
        dt: 1.0 / 120.0,
        ..Default::default()
    };
    let air = AirConfig::new(Grid::new([4, 4, 4], 1.0, Vec3::ZERO)?);
    for frame in 0..frames {
        for substep in 0..4 {
            let time = (frame * 4 + substep) as f64 * c.dt;
            let background = AirConfig {
                ambient: Vec3::new(0.65, 0.0, 0.2),
                gust_strength: 0.5,
                time,
                turbulence_strength: if mixed {
                    crate::orchard_sim::TURBULENCE_STRENGTH
                } else {
                    0.0
                },
                ..air
            };
            let wind: Vec<_> = particles
                .iter()
                .map(|p| {
                    if still_air {
                        Vec3::ZERO
                    } else {
                        background.background(p.position) + background.turbulence(p.position)
                    }
                })
                .collect();
            gpu.step_spatial(&engine, &anchors, c, &wind)?;
            particles = gpu.snapshot(&engine)?;
        }
        if particles.iter().any(|p| {
            !p.position.is_finite()
                || !p.velocity.is_finite()
                || !p.leaf.unwrap().frame.angular_velocity.is_finite()
        }) {
            return Err("non-finite leaf flight".into());
        }
        for (track, p) in tracks.iter_mut().zip(&particles) {
            let d = p.leaf.unwrap().frame.direction;
            track.push([
                (frame + 1) as f64 / 30.0,
                p.position.x,
                p.position.y,
                p.position.z,
                p.velocity.x,
                p.velocity.y,
                p.velocity.z,
                d.x,
                d.y,
                d.z,
            ]);
        }
        let center = particles.iter().map(|p| p.position).sum::<Vec3>() / particles.len() as f64;
        camera.target.x += (center.x - camera.target.x) * 0.035;
        camera.target.z += (center.z - camera.target.z) * 0.035;
        let still = path.with_extension("png");
        video.frame(
            |ui| {
                let (response, painter) =
                    ui.allocate_painter(ui.available_size(), egui::Sense::hover());
                painter.rect_filled(response.rect, 0.0, Color32::from_rgb(15, 21, 31));
                let mesh = leafy_draw::free_mesh(&leaves, &particles, &coefficients);
                studio::paint(&painter, &camera, response.rect, mesh);
            },
            if frame == 45 { Some(&still) } else { None },
        )?;
    }
    video.finish()?;
    std::fs::write(
        path.with_extension("json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "solver":"APW circulation, 3D reduced leaf", "still_air":still_air,
            "seconds":frames as f64/30.0, "frames":frames, "width":1920,"height":1080,
            "track_columns":["time","x","y","z","vx","vy","vz","direction_x","direction_y","direction_z"],
            "tracks":tracks, "source":"https://doi.org/10.1017/S0022112005005847"
        }))?,
    )?;
    eprintln!(
        "Leaf-flight capture complete: {} ({frames} frames)",
        path.display()
    );
    Ok(())
}
