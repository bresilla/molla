use crate::{ag_drone, draw, studio, video_capture};
use eframe::egui::{self, Color32};
use molla_math::Vec3;
use molla_solvers::airflow::Quadrotor;
use std::path::Path;

pub fn capture(args: &[String]) -> video_capture::Result<()> {
    let path = args
        .windows(2)
        .find(|p| p[0] == "--video")
        .map(|p| Path::new(&p[1]))
        .ok_or("AG model lab requires --video PATH")?;
    let drone = Quadrotor::new(Vec3::Y * 1.1, Default::default())?;
    let mut capture =
        video_capture::Capture::new_with_title(path, "Molla | Agricultural aircraft")?;
    for frame in 0..360 {
        let angle = frame as f64 / 360.0 * std::f64::consts::TAU;
        let camera = draw::Camera {
            yaw: 0.75 + angle,
            pitch: 0.36 + 0.18 * angle.sin(),
            distance: 1.85,
            target: Vec3::Y * 0.97,
        };
        let still = [0, 90, 180, 270]
            .contains(&frame)
            .then(|| path.with_file_name(format!("model-{frame:04}.png")));
        capture.frame(
            |ui| {
                let (response, painter) =
                    ui.allocate_painter(ui.available_size(), egui::Sense::hover());
                painter.rect_filled(response.rect, 0.0, Color32::from_rgb(15, 21, 31));
                studio::paint(&painter, &camera, response.rect, ag_drone::mesh(&drone));
            },
            still.as_deref(),
        )?;
    }
    capture.finish()?;
    Ok(())
}
