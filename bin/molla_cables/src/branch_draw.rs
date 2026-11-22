use crate::{
    branch::Branch,
    draw::{self, Camera, Surface, Triangle},
};
use eframe::egui::{self, Color32, Pos2, Rect, Stroke};
use molla_geometry::convex_cut::FaceOrigin;
use molla_math::Vec3;
use molla_solvers::strands::{Fiber, StrandModel};
use std::f64::consts::TAU;

#[path = "junction_surface.rs"]
mod junction_surface;

use crate::foliage_layout as layout;

/// Cylindrical bark parameterization of one wood cell in its local frame (axis +Z).
struct Bark {
    wraps: f64,
    along: f64,
    offset: f64,
    half_length: f64,
    taper: f64,
    thinness: f64,
}

/// Bark frames with texture offsets accumulated from each rooted cell along the bond graph.
fn bark_frames(model: &StrandModel) -> Vec<Bark> {
    let fibers = model.fibers();
    let index: std::collections::HashMap<u64, usize> =
        fibers.iter().enumerate().map(|(i, f)| (f.id, i)).collect();
    let mut neighbors = vec![Vec::new(); fibers.len()];
    for bond in model.bonds() {
        if let (Some(&a), Some(&b)) = (index.get(&bond.fibers[0]), index.get(&bond.fibers[1])) {
            neighbors[a].push(b);
            neighbors[b].push(a);
        }
    }
    let mut frames: Vec<Bark> = fibers
        .iter()
        .map(|f| {
            let vertices = f.shape.vertices();
            let radial = |below: bool| {
                let ring: Vec<f64> = vertices
                    .iter()
                    .filter(|p| (p.z < 0.0) == below)
                    .map(|p| p.x.hypot(p.y))
                    .collect();
                ring.iter().sum::<f64>() / ring.len().max(1) as f64
            };
            let (bottom, top) = (radial(true), radial(false));
            let z = vertices.iter().map(|p| p.z);
            let length = z.clone().fold(f64::NEG_INFINITY, f64::max) - z.fold(f64::INFINITY, f64::min);
            let radius = (0.5 * (bottom + top)).max(1e-4);
            let wraps = layout::bark_wraps(radius);
            Bark {
                wraps,
                along: wraps / (TAU * radius),
                offset: f64::NAN,
                half_length: 0.5 * length,
                taper: (top - bottom) / length.max(1e-6),
                thinness: layout::bark_thinness(radius),
            }
        })
        .collect();
    let pinned = (0..fibers.len()).filter(|&i| fibers[i].pin.is_some());
    let mut queue = std::collections::VecDeque::new();
    for seed in pinned.chain(0..fibers.len()) {
        if frames[seed].offset.is_finite() {
            continue;
        }
        frames[seed].offset = (seed as f64 * 0.618_034).fract() * 7.0;
        queue.push_back(seed);
        while let Some(i) = queue.pop_front() {
            for &j in &neighbors[i] {
                if !frames[j].offset.is_finite() {
                    frames[j].offset = frames[i].offset
                        + frames[i].half_length * frames[i].along
                        + frames[j].half_length * frames[j].along;
                    queue.push_back(j);
                }
            }
        }
    }
    frames
}

/// Smooth cone normals and wrapped bark coordinates of one world-space triangle on `fiber`.
fn bark_surface(fiber: &Fiber, bark: &Bark, p: [Vec3; 3]) -> Surface {
    let pose = &fiber.motion.pose;
    let local = p.map(|w| pose.inverse_transform_point(w));
    let face = (local[1] - local[0]).cross(local[2] - local[0]).normalize_or_zero();
    if face.z.abs() > 0.7 {
        let normal = pose.rotation * face;
        return Surface {
            normals: [normal; 3],
            uv: local.map(|l| [(l.x * bark.along) as f32, (l.y * bark.along) as f32]),
        };
    }
    let first = local[0].y.atan2(local[0].x);
    let angles = local.map(|l| {
        let a = l.y.atan2(l.x);
        first + (a - first + std::f64::consts::PI).rem_euclid(TAU) - std::f64::consts::PI
    });
    Surface {
        normals: std::array::from_fn(|i| {
            let (s, c) = angles[i].sin_cos();
            pose.rotation * Vec3::new(c, s, -bark.taper).normalize()
        }),
        uv: std::array::from_fn(|i| {
            [
                (bark.wraps * angles[i] / TAU) as f32,
                (bark.offset + local[i].z * bark.along) as f32,
            ]
        }),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    Cut,
    Break,
    Push,
    Grab,
    Orbit,
}

pub enum Action {
    Section(Pos2, Pos2, Rect, bool),
    Click(Pos2, Rect, bool),
    Push(Pos2, Rect),
    GrabStart(Pos2, Rect),
    GrabMove(Pos2, Rect),
    GrabEnd,
}

pub struct Interaction {
    pub tool: Tool,
    pub start: Option<Pos2>,
    pub viewport: Option<Rect>,
}
impl Default for Interaction {
    fn default() -> Self {
        Self {
            tool: Tool::Cut,
            start: None,
            viewport: None,
        }
    }
}

pub fn paint(
    ui: &mut egui::Ui,
    branch: &Branch,
    camera: &mut Camera,
    state: &mut Interaction,
    show_bonds: bool,
) -> Option<Action> {
    paint_model(ui, &branch.model, camera, state, show_bonds, true)
}

pub fn paint_model(
    ui: &mut egui::Ui,
    model: &molla_solvers::strands::StrandModel,
    camera: &mut Camera,
    state: &mut Interaction,
    show_bonds: bool,
    stand: bool,
) -> Option<Action> {
    paint_scene(
        ui,
        model,
        camera,
        state,
        show_bonds,
        stand,
        true,
        (Vec::new(), Vec::new()),
    )
}

pub fn paint_foliage(
    ui: &mut egui::Ui,
    model: &molla_solvers::strands::StrandModel,
    camera: &mut Camera,
    state: &mut Interaction,
    foliage: Vec<Triangle>,
) -> Option<Action> {
    paint_scene(ui, model, camera, state, false, false, false, (foliage, Vec::new()))
}

pub fn paint_cinematic(
    ui: &mut egui::Ui,
    model: &molla_solvers::strands::StrandModel,
    camera: &mut Camera,
) {
    paint_cinematic_foliage(ui, model, camera, Vec::new());
}

pub fn paint_cinematic_foliage(
    ui: &mut egui::Ui,
    model: &molla_solvers::strands::StrandModel,
    camera: &mut Camera,
    foliage: Vec<Triangle>,
) {
    paint_cinematic_leaves(ui, model, camera, foliage, Vec::new());
}

/// Cinematic view with foliage triangles and GPU-instanced leaves.
pub fn paint_cinematic_leaves(
    ui: &mut egui::Ui,
    model: &molla_solvers::strands::StrandModel,
    camera: &mut Camera,
    foliage: Vec<Triangle>,
    leaves: Vec<crate::studio::LeafInstance>,
) {
    let mut state = Interaction {
        tool: Tool::Orbit,
        ..Default::default()
    };
    paint_scene(ui, model, camera, &mut state, false, false, false, (foliage, leaves));
}

fn paint_scene(
    ui: &mut egui::Ui,
    model: &molla_solvers::strands::StrandModel,
    camera: &mut Camera,
    state: &mut Interaction,
    show_bonds: bool,
    stand: bool,
    labels: bool,
    (foliage, leaves): (Vec<Triangle>, Vec<crate::studio::LeafInstance>),
) -> Option<Action> {
    let (response, painter) =
        ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());
    let rect = response.rect;
    state.viewport = Some(rect);
    painter.rect_filled(rect, 0.0, Color32::from_rgb(15, 21, 31));
    if response.dragged_by(egui::PointerButton::Secondary)
        || (state.tool == Tool::Orbit && response.dragged())
    {
        let delta = ui.input(|i| i.pointer.delta());
        camera.yaw -= f64::from(delta.x) * 0.006;
        camera.pitch = (camera.pitch + f64::from(delta.y) * 0.006).clamp(0.05, 1.35);
    }
    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
    if response.hovered() && scroll != 0.0 {
        camera.distance = (camera.distance * (-f64::from(scroll) * 0.002).exp()).clamp(0.6, 64.0);
    }
    draw::grid(&painter, camera, rect);
    let mut triangles = foliage;
    let junctions = (!stand).then(|| junction_surface::JunctionSurface::new(model));
    let mut damage_by_fiber = std::collections::BTreeMap::<u64, f64>::new();
    for bond in model.bonds().iter().filter(|b| b.damage > 0.0) {
        for id in bond.fibers {
            let damage = damage_by_fiber.entry(id).or_default();
            *damage = damage.max(bond.damage);
        }
    }
    let barks = bark_frames(model);
    for (fiber, bark) in model.fibers().iter().zip(&barks) {
        let points: Vec<_> = fiber
            .shape
            .vertices()
            .iter()
            .map(|&v| fiber.motion.pose.transform_point(v))
            .collect();
        let damage = damage_by_fiber.get(&fiber.id).copied().unwrap_or(0.0);
        for face in fiber.shape.faces() {
            let shade = ((fiber.rest.position.z * 85.0).sin() * 15.0) as i16;
            let color = match face.origin {
                FaceOrigin::Cut { .. } => Color32::from_rgb(
                    (240 + shade) as u8,
                    (215 + shade) as u8,
                    (150 + shade) as u8,
                ),
                FaceOrigin::Fracture { .. } | FaceOrigin::Interface(_) => Color32::from_rgb(
                    (233 + shade) as u8,
                    (184 + shade) as u8,
                    (104 + shade) as u8,
                ),
                _ if damage > 0.01 => Color32::from_rgb(230, (155.0 * (1.0 - damage)) as u8, 50),
                _ => {
                    let s = shade / 3;
                    Color32::from_rgb((130 + s) as u8, (127 + s) as u8, (122 + s) as u8)
                }
            };
            let exposed = matches!(
                face.origin,
                FaceOrigin::Cut { .. } | FaceOrigin::Fracture { .. } | FaceOrigin::Interface(_)
            );
            let polygon = face.vertices.iter().map(|&i| points[i as usize]).collect();
            let patches = if let Some(junctions) = &junctions {
                junctions.patches(fiber.id, face.origin, polygon)
            } else {
                vec![polygon]
            };
            for patch in patches {
                for j in 1..patch.len() - 1 {
                    let p = [patch[0], patch[j], patch[j + 1]];
                    triangles.push(Triangle {
                        p,
                        color,
                        surface: (!exposed).then(|| bark_surface(fiber, bark, p)),
                        material: if exposed {
                            2.0
                        } else {
                            1.0 + 0.45 * bark.thinness as f32
                        },
                    });
                }
            }
        }
    }
    let root = model
        .fibers()
        .iter()
        .find(|f| f.pin.is_some())
        .map(|f| f.center());
    if let Some(root) = root.filter(|_| stand) {
        draw::tube(
            &mut triangles,
            Vec3::new(-1.24, 0.0, root.z),
            Vec3::new(-1.24, root.y, root.z),
            0.055,
            Color32::from_rgb(95, 110, 130),
        );
        draw::tube(
            &mut triangles,
            Vec3::new(-1.24, root.y, root.z),
            root,
            0.07,
            Color32::from_rgb(95, 110, 130),
        );
    }
    crate::studio::paint_leaves(&painter, camera, rect, triangles, leaves);
    if show_bonds {
        for bond in model.bonds().iter().filter(|b| b.intact()) {
            let points = [0, 1].map(|i| {
                camera
                    .project(model.fiber(bond.fibers[i]).unwrap().center(), rect)
                    .0
            });
            painter.line_segment(
                points,
                Stroke::new(
                    1.0,
                    if bond.damage > 0.0 {
                        Color32::RED
                    } else {
                        Color32::from_rgba_unmultiplied(60, 210, 190, 95)
                    },
                ),
            );
        }
    }
    let mut action = None;
    if state.tool == Tool::Grab {
        if response.hovered()
            && ui.input(|i| i.pointer.primary_pressed())
            && let Some(p) = response.interact_pointer_pos()
        {
            action = Some(Action::GrabStart(p, rect));
        } else if response.dragged_by(egui::PointerButton::Primary)
            && let Some(p) = response.interact_pointer_pos()
        {
            action = Some(Action::GrabMove(p, rect));
            painter.circle_stroke(p, 8.0, Stroke::new(2.0, Color32::from_rgb(255, 190, 90)));
        }
        if ui.input(|i| i.pointer.primary_released())
            || ui.input(|i| i.key_pressed(egui::Key::Escape))
            || !ui.input(|i| i.focused)
        {
            action = Some(Action::GrabEnd);
        }
    }
    if matches!(state.tool, Tool::Cut | Tool::Break)
        && response.clicked_by(egui::PointerButton::Primary)
        && let Some(p) = response.interact_pointer_pos()
    {
        action = Some(Action::Click(p, rect, state.tool == Tool::Break));
    }
    if matches!(state.tool, Tool::Cut | Tool::Break) {
        if response.drag_started_by(egui::PointerButton::Primary) {
            state.start = ui.input(|i| i.pointer.press_origin());
        }
        if let (Some(a), Some(b)) = (state.start, ui.input(|i| i.pointer.interact_pos())) {
            painter.line_segment([a, b], Stroke::new(2.0, Color32::from_rgb(80, 240, 220)));
            painter.circle_filled(a, 4.0, Color32::WHITE);
            painter.circle_filled(b, 4.0, Color32::WHITE);
            if response.drag_stopped_by(egui::PointerButton::Primary) {
                action = Some(Action::Section(a, b, rect, state.tool == Tool::Break));
                state.start = None;
            }
        }
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) || !ui.input(|i| i.focused) {
            state.start = None;
            action = None;
        }
    } else {
        state.start = None;
        if state.tool == Tool::Push
            && response.clicked_by(egui::PointerButton::Primary)
            && let Some(p) = response.interact_pointer_pos()
        {
            action = Some(Action::Push(p, rect));
        }
    }
    if labels {
        painter.text(
            rect.left_top() + egui::vec2(18.0, 18.0),
            egui::Align2::LEFT_TOP,
            "FINITE FIBER VOLUMES  /  EXACT CUT FACES",
            egui::FontId::monospace(12.0),
            Color32::from_gray(180),
        );
        painter.text(
            rect.left_bottom() + egui::vec2(18.0, -18.0),
            egui::Align2::LEFT_BOTTOM,
            if state.tool == Tool::Grab {
                "HOLD + DRAG: BEND WOOD   /   RELEASE: LET GO   /   ESC: RELEASE"
            } else {
                "CLICK: CUT / BREAK AT HIT   /   RIGHT DRAG: ORBIT   /   SCROLL: ZOOM"
            },
            egui::FontId::monospace(11.0),
            Color32::from_gray(150),
        );
    }
    action
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_geometry::convex_cut::ConvexCell;
    use molla_math::{Quat, Transform};
    use molla_solvers::strands::{Bond, BondMaterial, Pin};
    use std::f64::consts::PI;

    #[test]
    fn bark_coordinates_wrap_seamlessly_and_continue_across_bonded_cells() {
        let (radius, length) = (0.02, 0.3);
        let fibers: Vec<_> = [0u64, 1]
            .into_iter()
            .map(|id| {
                let pose = Transform::new(Vec3::Z * (length * (id as f64 + 0.5)), Quat::IDENTITY);
                let mut f =
                    Fiber::new(id, ConvexCell::prism(radius, length, 8).unwrap(), 650.0, pose)
                        .unwrap();
                if id == 0 {
                    f.pin = Some(Pin {
                        local_point: Vec3::ZERO,
                        pose,
                    });
                }
                f
            })
            .collect();
        let material = BondMaterial {
            translation: Vec3::splat(1e-6),
            rotation: Vec3::splat(0.01),
            force_limit: Vec3::ONE,
            torque_limit: Vec3::ONE,
            damage_rate: 0.0,
            damping: 0.0,
        };
        let anchors = [Vec3::Z * (0.5 * length), -Vec3::Z * (0.5 * length)];
        let bond = Bond::new(0, [&fibers[0], &fibers[1]], anchors, material).unwrap();
        let model = StrandModel::new(fibers, vec![bond]).unwrap();
        let barks = bark_frames(&model);
        assert_eq!(barks[0].wraps, 1.0);
        assert!((barks[0].thinness - layout::bark_thinness(radius)).abs() < 1e-12);
        let boundary = Vec3::new(radius, 0.0, length);
        let along = |i: usize| {
            let local = model.fibers()[i].motion.pose.inverse_transform_point(boundary);
            barks[i].offset + local.z * barks[i].along
        };
        assert!((along(0) - along(1)).abs() < 1e-9);
        let f = &model.fibers()[0];
        let ring = [(-PI + 0.1, 0.0), (PI - 0.1, 0.0), (PI - 0.05, 0.05)].map(|(a, z)| {
            f.motion
                .pose
                .transform_point(Vec3::new(radius * a.cos(), radius * a.sin(), z))
        });
        let surface = bark_surface(f, &barks[0], ring);
        let u = surface.uv.map(|uv| uv[0]);
        let span = u.iter().copied().fold(f32::MIN, f32::max) - u.iter().copied().fold(f32::MAX, f32::min);
        assert!(span < 0.05, "seam span {span}");
        for (n, p) in surface.normals.iter().zip(ring) {
            let radial = (p - f.center()).with_z(0.0).normalize();
            assert!(n.dot(radial) > 0.99);
        }
    }
}
