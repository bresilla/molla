use crate::{draw, drone_asset, studio, video_capture};
use eframe::egui::{self, Color32};
use molla_math::Vec3;
use molla_solvers::airflow::{
    FlightReference, FlightSegment, MotionLimits, Quadrotor, QuadrotorConfig,
};
use std::path::Path;

const SECONDS: f64 = 30.0;
const DT: f64 = 1.0 / 120.0;

fn reference(t: f64) -> (FlightReference, Vec3) {
    let points = [
        Vec3::new(-2.0, 2.0, 0.0),
        Vec3::new(2.0, 2.0, 0.0),
        Vec3::new(2.0, 2.0, 3.0),
        Vec3::new(-2.0, 2.0, 0.0),
        Vec3::new(-2.0, 3.0, 0.0),
    ];
    let legs = [(3.0, 4.0), (9.0, 4.0), (15.0, 5.0), (22.0, 4.0)];
    let headings = [Vec3::Z, Vec3::Z, -Vec3::X, -Vec3::X];
    if t < 3.0 {
        return (FlightReference::hold(points[0]), Vec3::Z);
    }
    let i = legs.iter().rposition(|(begin, _)| t >= *begin).unwrap();
    let (begin, duration) = legs[i];
    let limits = MotionLimits {
        speed: if i == 3 { 0.8 } else { 1.5 },
        ..Default::default()
    };
    let reference = FlightSegment::new(points[i], points[i + 1], limits)
        .unwrap()
        .with_duration(duration)
        .unwrap()
        .sample(t - begin)
        .unwrap();
    (reference, headings[i])
}

pub fn capture(args: &[String]) -> video_capture::Result<()> {
    let path = args
        .windows(2)
        .find(|p| p[0] == "--video")
        .map(|p| Path::new(&p[1]))
        .ok_or("flight lab requires --video PATH")?;
    let (initial, heading) = reference(0.0);
    let mut drone = Quadrotor::new(
        initial.position,
        QuadrotorConfig {
            mass: 5.6,
            ..Default::default()
        },
    )?;
    for _ in 0..480 {
        drone.track(initial, heading)?;
        drone.step(DT, Vec3::ZERO)?;
    }
    let mut capture =
        video_capture::Capture::new_with_title(path, "Molla | Multirotor flight dynamics")?;
    let mut camera = draw::Camera {
        yaw: 0.85,
        pitch: 0.30,
        distance: 3.8,
        target: initial.position,
    };
    let mut samples = Vec::new();
    let mut peak_error = 0.0_f64;
    let mut peak_tilt = 0.0_f64;
    let mut peak_wrench_error = 0.0_f64;
    for frame in 0..(SECONDS * 30.0) as usize {
        for substep in 0..4 {
            let time = (frame * 4 + substep) as f64 * DT;
            let (r, heading) = reference(time);
            let wind = if time > 26.0 {
                Vec3::new(2.5, 0.0, -1.2) * ((time - 26.0) * 1.8).sin().powi(2)
            } else {
                Vec3::ZERO
            };
            drone.track(r, heading)?;
            drone.step(DT, wind)?;
            peak_error = peak_error.max(drone.pose()?.position.distance(r.position));
        }
        let time = (frame + 1) as f64 / 30.0;
        let pose = drone.pose()?;
        let velocity = drone.velocity()?;
        let axis = pose.rotation * Vec3::Z;
        let tilt = axis.y.clamp(-1.0, 1.0).acos();
        peak_tilt = peak_tilt.max(tilt);
        let air_force = drone.air_loads()?.iter().map(|l| l.force).sum::<Vec3>();
        let body_force = drone
            .propellers
            .iter()
            .map(|id| {
                drone
                    .world
                    .devices
                    .propellers
                    .reading(*id)
                    .unwrap()
                    .world_force
            })
            .sum::<Vec3>();
        peak_wrench_error = peak_wrench_error.max((air_force + body_force).length());
        if !pose.position.is_finite() || !velocity.is_finite() || !air_force.is_finite() {
            return Err("non-finite flight state".into());
        }
        samples.push(serde_json::json!({"time": time, "position": pose.position.to_array(),
            "velocity": velocity.to_array(), "orientation_xyzw": pose.rotation.to_array(),
            "thrust_axis": axis.to_array(), "air_force": air_force.to_array(),
            "reference": reference(time).0.position.to_array(),
            "rpm": drone.propellers.map(|id| drone.world.devices.propellers.reading(id).unwrap().omega * 60.0 / std::f64::consts::TAU)}));
        camera.target = camera.target.lerp(pose.position, 0.08);
        let still = [44, 104, 134, 194, 224, 284, 344, 374, 464, 569, 899]
            .contains(&frame)
            .then(|| path.with_file_name(format!("flight-{:04}.png", frame + 1)));
        capture.frame(
            |ui| {
                let (response, painter) =
                    ui.allocate_painter(ui.available_size(), egui::Sense::hover());
                painter.rect_filled(response.rect, 0.0, Color32::from_rgb(15, 21, 31));
                studio::paint(
                    &painter,
                    &camera,
                    response.rect,
                    drone_asset::mesh(&drone, time),
                );
            },
            still.as_deref(),
        )?;
    }
    let frames = capture.frames;
    capture.finish()?;
    let final_error = drone
        .pose()?
        .position
        .distance(reference(SECONDS).0.position);
    let final_speed = drone.velocity()?.length();
    let passed = peak_error < 0.5
        && final_error < 0.08
        && final_speed < 0.08
        && peak_tilt > 0.1
        && peak_wrench_error < 1e-9;
    std::fs::write(
        path.with_extension("json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "seconds": SECONDS, "frames": frames, "width": 1920, "height": 1080,
            "mass_kg": drone.config.mass, "peak_tracking_error_m": peak_error,
            "peak_tilt_degrees": peak_tilt.to_degrees(), "final_error_m": final_error,
            "final_speed_m_s": final_speed, "peak_rotor_air_force_balance_error_n": peak_wrench_error,
            "verified": passed, "samples": samples,
            "stages": ["lateral translation and braking", "forward translation and braking", "diagonal return with heading change", "climb", "gust rejection"],
            "limitations": "Illustrative parameters; no sensor noise or hardware calibration. Air wrench reported, not fluid grid evolved in this isolated flight lab."
        }))?,
    )?;
    if !passed {
        return Err("flight lab motion acceptance failed; inspect report".into());
    }
    Ok(())
}
