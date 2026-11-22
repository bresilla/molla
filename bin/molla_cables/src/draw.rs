use crate::scene::Scene;
use eframe::egui::{self, Color32, Pos2, Rect, Stroke};
use molla_math::Vec3;

pub const COLORS: [Color32; 3] = [
    Color32::from_rgb(57, 212, 211),
    Color32::from_rgb(255, 184, 91),
    Color32::from_rgb(176, 143, 255),
];

#[derive(Clone, Copy)]
pub struct Camera {
    pub yaw: f64,
    pub pitch: f64,
    pub distance: f64,
    pub target: Vec3,
}
impl Default for Camera {
    fn default() -> Self {
        Self {
            yaw: 0.5,
            pitch: 0.33,
            distance: 6.8,
            target: Vec3::new(0.0, 1.25, 0.0),
        }
    }
}

impl Camera {
    pub fn ray(&self, p: Pos2, rect: Rect) -> (Vec3, Vec3) {
        let forward = Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        );
        let right = Vec3::Y.cross(forward).normalize();
        let up = forward.cross(right);
        let scale = f64::from(rect.height()) * 1.25;
        let direction = -forward + right * (f64::from(p.x - rect.center().x) / scale)
            - up * (f64::from(p.y - rect.center().y) / scale);
        (self.target + forward * self.distance, direction.normalize())
    }

    pub fn project(&self, p: Vec3, rect: Rect) -> (Pos2, f64) {
        let target = self.target;
        let forward = Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        );
        let right = Vec3::Y.cross(forward).normalize();
        let up = forward.cross(right);
        let relative = p - target;
        let depth = self.distance - relative.dot(forward);
        let scale = f64::from(rect.height()) * 1.25 / depth.max(0.2);
        (
            Pos2::new(
                rect.center().x + (relative.dot(right) * scale) as f32,
                rect.center().y - (relative.dot(up) * scale) as f32,
            ),
            depth,
        )
    }
}

#[cfg(test)]
mod camera_tests {
    use super::*;

    #[test]
    fn translated_camera_projects_back_onto_its_pick_ray() {
        let rect = Rect::from_min_size(Pos2::ZERO, egui::vec2(1920.0, 1080.0));
        for target in [Vec3::new(0.0, 1.25, 0.0), Vec3::new(-0.2, 0.8, 0.4)] {
            let camera = Camera {
                target,
                distance: 2.5,
                ..Default::default()
            };
            let point = target + Vec3::new(0.3, -0.4, 0.1);
            let pixel = camera.project(point, rect).0;
            let (origin, direction) = camera.ray(pixel, rect);
            assert!((point - origin).cross(direction).length() < 1e-6);
        }
    }
}

pub(crate) struct Triangle {
    pub p: [Vec3; 3],
    pub color: Color32,
    pub material: f32,
    pub surface: Option<Surface>,
}

#[derive(Clone, Copy)]
pub(crate) struct Surface {
    pub normals: [Vec3; 3],
    pub uv: [[f32; 2]; 3],
}

pub(crate) fn tube(tris: &mut Vec<Triangle>, a: Vec3, b: Vec3, radius: f64, color: Color32) {
    let axis = (b - a).normalize_or_zero();
    let helper = if axis.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let u = axis.cross(helper).normalize_or_zero() * radius;
    let v = axis.cross(u);
    for i in 0..12 {
        let ring = |j: usize| {
            let t = std::f64::consts::TAU * j as f64 / 12.0;
            u * t.cos() + v * t.sin()
        };
        let p = ring(i);
        let q = ring(i + 1);
        tris.push(Triangle {
            p: [a + p, b + p, b + q],
            color,
            material: 0.0,
            surface: None,
        });
        tris.push(Triangle {
            p: [a + p, b + q, a + q],
            color,
            material: 0.0,
            surface: None,
        });
    }
}

pub(crate) fn sphere(
    tris: &mut Vec<Triangle>,
    center: Vec3,
    radius: f64,
    color: Color32,
    segments: usize,
) {
    let rings = segments / 2;
    let point = |r: usize, c: usize| {
        let latitude = std::f64::consts::PI * r as f64 / rings as f64;
        let longitude = std::f64::consts::TAU * c as f64 / segments as f64;
        center
            + Vec3::new(
                latitude.sin() * longitude.cos(),
                latitude.cos(),
                latitude.sin() * longitude.sin(),
            ) * radius
    };
    for r in 0..rings {
        for c in 0..segments {
            let a = point(r, c);
            let b = point(r + 1, c);
            let d = point(r, c + 1);
            let e = point(r + 1, c + 1);
            tris.push(Triangle {
                p: [a, b, e],
                color,
                material: 0.0,
                surface: None,
            });
            tris.push(Triangle {
                p: [a, e, d],
                color,
                material: 0.0,
                surface: None,
            });
        }
    }
}

pub fn paint(ui: &mut egui::Ui, scene: &Scene, camera: &mut Camera, show_nodes: bool) {
    let (response, painter) = ui.allocate_painter(ui.available_size(), egui::Sense::drag());
    let rect = response.rect;
    painter.rect_filled(rect, 0.0, Color32::from_rgb(15, 21, 31));
    if response.dragged() {
        let delta = ui.input(|i| i.pointer.delta());
        camera.yaw -= f64::from(delta.x) * 0.006;
        camera.pitch = (camera.pitch + f64::from(delta.y) * 0.006).clamp(0.05, 1.35);
    }
    if response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        camera.distance = (camera.distance * (-f64::from(scroll) * 0.002).exp()).clamp(3.8, 12.0);
    }
    grid(&painter, camera, rect);
    let mut triangles = Vec::new();
    if let Some((c, r)) = scene.sphere {
        sphere(&mut triangles, c, r, Color32::from_rgb(92, 110, 132), 32);
    }
    for (c, ids) in scene.cables.iter().enumerate() {
        for edge in ids.windows(2) {
            tube(
                &mut triangles,
                scene.points[edge[0] as usize],
                scene.points[edge[1] as usize],
                0.035,
                COLORS[c],
            );
        }
        for (i, &id) in ids.iter().enumerate() {
            if show_nodes || i == 0 || i + 1 == ids.len() {
                sphere(
                    &mut triangles,
                    scene.points[id as usize],
                    if show_nodes { 0.043 } else { 0.035 },
                    COLORS[c],
                    8,
                );
            }
        }
    }
    for &id in &scene.anchors {
        let p = scene.points[id as usize];
        tube(
            &mut triangles,
            Vec3::new(p.x, 0.0, p.z),
            p,
            0.018,
            Color32::from_rgb(95, 104, 122),
        );
        sphere(
            &mut triangles,
            p,
            0.075,
            Color32::from_rgb(237, 242, 252),
            12,
        );
    }
    paint_triangles(&painter, camera, rect, triangles);
    painter.text(
        rect.left_bottom() + egui::vec2(20.0, -20.0),
        egui::Align2::LEFT_BOTTOM,
        "DRAG TO ORBIT   /   SCROLL TO ZOOM",
        egui::FontId::monospace(12.0),
        Color32::from_gray(145),
    );
}

pub(crate) fn grid(painter: &egui::Painter, camera: &Camera, rect: Rect) {
    for i in -8..=8 {
        let s = i as f64 * 0.5;
        for (a, b) in [
            (Vec3::new(s, 0.0, -4.0), Vec3::new(s, 0.0, 4.0)),
            (Vec3::new(-4.0, 0.0, s), Vec3::new(4.0, 0.0, s)),
        ] {
            painter.line_segment(
                [camera.project(a, rect).0, camera.project(b, rect).0],
                Stroke::new(1.0, Color32::from_rgb(35, 44, 57)),
            );
        }
    }
}

pub(crate) fn paint_triangles(
    painter: &egui::Painter,
    camera: &Camera,
    rect: Rect,
    triangles: Vec<Triangle>,
) {
    crate::studio::paint(painter, camera, rect, triangles);
}
