use crate::{
    draw::{Camera, Surface, Triangle},
    leafy_draw::{self, LeafSurface},
    orchard_model::TREE_COUNT,
    orchard_sim::Orchard,
    orchard_spray::Deposit,
};
use eframe::egui::{self, Color32, FontId, Pos2, Rect, Stroke};
use molla_math::Vec3;

pub const LEAD_IN_S: f64 = 0.6;
pub const ZOOM_S: f64 = 3.2;
pub const HOLD_S: f64 = 9.0;
const VIEW_LENGTHS: f64 = 2.3;
const BEAD_VIEW_DISTANCE_M: f64 = 1.6;
const BEAD_RADIUS_M: f64 = 0.45;
const MAX_BEADS: usize = 8000;
const WATER: Color32 = Color32::from_rgb(208, 228, 234);
const WATER_MATERIAL: f32 = 13.0;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tally {
    pub parcels: usize,
    pub litres: f64,
    pub droplets: f64,
}

pub fn tally(deposits: &[Deposit], leaf: usize) -> Tally {
    deposits
        .iter()
        .filter(|d| d.leaf == leaf)
        .fold(Tally::default(), |mut t, d| {
            t.parcels += 1;
            t.litres += d.litres;
            t.droplets += d.droplets;
            t
        })
}

pub fn tree_leaves(leaves: usize, tree: usize) -> std::ops::Range<usize> {
    let per_tree = leaves / TREE_COUNT;
    tree * per_tree..(tree + 1) * per_tree
}

/// The leaf of `tree` grown at the same place as `leaf` of `from`; every tree is the same model.
pub fn twin(world: &Orchard, leaf: usize, from: usize, tree: usize) -> usize {
    let count = world.particles.len();
    tree_leaves(count, tree).start + (leaf - tree_leaves(count, from).start)
}

/// Water on the inspected leaf and on its whole tree.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Summary {
    pub tree: usize,
    pub leaf: Tally,
    pub tree_parcels: usize,
    pub wet_leaves: usize,
}

pub fn summary(world: &Orchard, focus: &Focus) -> Summary {
    let Some(spray) = &world.spray else {
        return Summary::default();
    };
    let range = tree_leaves(world.particles.len(), focus.tree);
    Summary {
        tree: focus.tree,
        leaf: tally(&spray.deposits, focus.leaf),
        tree_parcels: spray.deposits.iter().filter(|d| range.contains(&d.leaf)).count(),
        wet_leaves: range.filter(|&i| spray.wet_litres[i] > 0.0).count(),
    }
}

fn surface(world: &Orchard, leaf: usize) -> Option<LeafSurface> {
    leafy_draw::surface(
        &world.leaves[leaf],
        &world.particles[leaf],
        Some(&world.tree.model),
    )
}

/// Signs mapping the collision blade frame onto the drawn blade: along, lateral, normal.
fn orientation(world: &Orchard, leaf: usize, s: &LeafSurface) -> [f64; 3] {
    let Some(blade) = world.particles[leaf].leaf else {
        return [1.0; 3];
    };
    let (d, n) = (blade.frame.direction, blade.frame.normal);
    [d.dot(s.direction), d.cross(n).dot(s.side), n.dot(s.normal)].map(|v| v.signum())
}

/// Attached, roughly horizontal leaf of `tree` with the most parcels landed on it.
pub fn select(world: &Orchard, tree: usize) -> Option<usize> {
    let spray = world.spray.as_ref()?;
    let range = tree_leaves(world.particles.len(), tree);
    let mut counts = vec![0usize; range.len()];
    for d in spray.deposits.iter().filter(|d| range.contains(&d.leaf)) {
        counts[d.leaf - range.start] += 1;
    }
    range
        .clone()
        .filter(|&i| counts[i - range.start] > 0 && world.particles[i].attached)
        .filter(|&i| surface(world, i).is_some_and(|s| s.normal.y.abs() > 0.5))
        .max_by_key(|&i| (counts[i - range.start], std::cmp::Reverse(i)))
}

fn smooth(u: f64) -> f64 {
    let u = u.clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

fn ramp(u: f64, lo: f64, hi: f64) -> f64 {
    smooth((u - lo) / (hi - lo))
}

fn wrap(angle: f64) -> f64 {
    (angle + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}

pub struct Focus {
    pub leaf: usize,
    pub tree: usize,
    pub start: f64,
    pub length: f64,
    /// Panel heading after the tree number.
    pub title: String,
    /// Earlier inspection shown beneath this one for comparison.
    pub reference: Option<(Summary, String)>,
    yaw: f64,
    pitch: f64,
}

impl Focus {
    /// Frames the wetted face of `leaf`, leaning toward the overview heading `yaw_hint`.
    pub fn new(world: &Orchard, leaf: usize, tree: usize, start: f64, yaw_hint: f64) -> Option<Self> {
        let s = surface(world, leaf)?;
        let spray = world.spray.as_ref()?;
        let flip = orientation(world, leaf, &s)[2];
        let wetted: f64 = spray
            .deposits
            .iter()
            .filter(|d| d.leaf == leaf)
            .map(|d| d.side * flip)
            .sum();
        let face = s.normal * if wetted != 0.0 { wetted.signum() } else { s.normal.y.signum() };
        let heading = Vec3::new(yaw_hint.sin(), 0.0, yaw_hint.cos());
        let forward = (face + heading * 0.55).normalize();
        Some(Self {
            leaf,
            tree,
            start,
            length: s.length,
            title: "ONE LEAF UP CLOSE".into(),
            reference: None,
            yaw: forward.x.atan2(forward.z),
            pitch: forward.y.asin().clamp(0.35, 1.2),
        })
    }

    pub fn end(&self) -> f64 {
        self.start + 2.0 * ZOOM_S + HOLD_S
    }

    /// Blend weights toward the leaf: (target, framing).
    pub fn weights(&self, time: f64) -> (f64, f64) {
        let into = (time - self.start) / ZOOM_S;
        let out = (time - self.start - ZOOM_S - HOLD_S) / ZOOM_S;
        (
            ramp(into, 0.0, 0.6).min(1.0 - ramp(out, 0.4, 1.0)),
            ramp(into, 0.0, 1.0).min(1.0 - ramp(out, 0.0, 1.0)),
        )
    }

    pub fn camera(&self, world: &Orchard, base: Camera, time: f64) -> Camera {
        let (target, view) = self.weights(time);
        let Some(s) = surface(world, self.leaf) else {
            return base;
        };
        let distance = VIEW_LENGTHS * s.length;
        Camera {
            yaw: base.yaw + wrap(self.yaw - base.yaw) * view,
            pitch: base.pitch + (self.pitch - base.pitch) * view,
            distance: (base.distance.ln() + (distance.ln() - base.distance.ln()) * view).exp(),
            target: base.target.lerp(s.point(0.5, 0.0), target),
        }
    }
}

fn bead(out: &mut Vec<Triangle>, center: Vec3, up: Vec3, radius: f64) {
    const AROUND: usize = 8;
    const HEIGHT: f64 = 0.62;
    let u = up.any_orthonormal_vector();
    let v = up.cross(u);
    let ring = |k: usize, elevation: f64| {
        let angle = k as f64 / AROUND as f64 * std::f64::consts::TAU;
        let radial = u * angle.cos() + v * angle.sin();
        let (c, s) = (elevation.cos(), elevation.sin());
        (
            center + radial * (radius * c) + up * (radius * HEIGHT * s),
            (radial * (c * HEIGHT) + up * s).normalize(),
        )
    };
    let apex = (center + up * (radius * HEIGHT), up);
    let mut face = |a: (Vec3, Vec3), b: (Vec3, Vec3), c: (Vec3, Vec3)| {
        out.push(Triangle {
            p: [a.0, b.0, c.0],
            color: WATER,
            material: WATER_MATERIAL,
            surface: Some(Surface {
                normals: [a.1, b.1, c.1],
                uv: [[0.0; 2]; 3],
            }),
        })
    };
    for k in 0..AROUND {
        let (r0, r1) = (ring(k, 0.0), ring(k + 1, 0.0));
        let (s0, s1) = (ring(k, 0.9), ring(k + 1, 0.9));
        face(r0, r1, s0);
        face(r1, s1, s0);
        face(s0, s1, apex);
    }
}

/// Water beads for every deposit on attached leaves near the camera target.
pub fn beads(world: &Orchard, camera: &Camera) -> Vec<Triangle> {
    let mut out = Vec::new();
    let Some(spray) = &world.spray else {
        return out;
    };
    if camera.distance > BEAD_VIEW_DISTANCE_M {
        return out;
    }
    let mut cache: Option<(usize, LeafSurface, [f64; 3])> = None;
    let mut count = 0;
    for d in &spray.deposits {
        let p = &world.particles[d.leaf];
        if !p.attached || p.position.distance(camera.target) > BEAD_RADIUS_M {
            continue;
        }
        if cache.as_ref().is_none_or(|c| c.0 != d.leaf) {
            cache = surface(world, d.leaf).map(|s| (d.leaf, s, orientation(world, d.leaf, &s)));
        }
        let Some((_, s, [along, lateral, normal])) = &cache else {
            continue;
        };
        let t = (0.5 + (d.t - 0.5) * along).clamp(0.04, 0.96);
        let limit = 0.85 * leafy_draw::half_width(t);
        let x = (d.x * lateral).clamp(-limit, limit);
        let up = s.normal * (d.side * normal);
        let radius = (3.0 * d.litres * 1e-3 / (4.0 * std::f64::consts::PI)).cbrt();
        bead(&mut out, s.point(t, x) + up * 0.00015, up, radius);
        count += 1;
        if count == MAX_BEADS {
            break;
        }
    }
    out
}

fn thousands(value: f64) -> String {
    let digits = format!("{:.0}", value.max(0.0));
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

pub fn overlay(painter: &egui::Painter, rect: Rect, camera: &Camera, world: &Orchard, focus: &Focus) {
    if world.spray.is_none() {
        return;
    }
    let (_, view) = focus.weights(world.time());
    let ring = view * (1.0 - ramp(view, 0.55, 0.9));
    if ring > 0.01
        && let Some(s) = surface(world, focus.leaf)
    {
        let (center, _) = camera.project(s.point(0.5, 0.0), rect);
        let (tip, _) = camera.project(s.point(1.0, 0.0), rect);
        let radius = (center.distance(tip) * 1.6).max(14.0);
        let alpha = (ring * 255.0) as u8;
        painter.circle_stroke(
            center,
            radius,
            Stroke::new(2.0, Color32::from_rgba_unmultiplied(255, 255, 255, alpha)),
        );
    }
    let alpha = ramp(view, 0.6, 0.95);
    if alpha <= 0.0 {
        return;
    }
    let fade = |c: Color32| c.gamma_multiply(alpha as f32);
    if let Some(s) = surface(world, focus.leaf) {
        const STEPS: usize = 16;
        let edge = |i: usize, sign: f64| {
            let t = i as f64 / STEPS as f64;
            camera.project(s.point(t, sign * leafy_draw::half_width(t)), rect).0
        };
        let outline: Vec<_> = (0..=STEPS)
            .map(|i| edge(i, 1.0))
            .chain((1..STEPS).rev().map(|i| edge(i, -1.0)))
            .collect();
        painter.add(egui::Shape::closed_line(
            outline,
            Stroke::new(2.0, fade(Color32::from_rgba_unmultiplied(255, 255, 255, 150))),
        ));
    }
    let current = summary(world, focus);
    let t = current.leaf;
    let (width, height) = if focus.reference.is_some() {
        (820.0, 214.0)
    } else {
        (620.0, 176.0)
    };
    let panel = Rect::from_min_size(
        Pos2::new(rect.left() + 56.0, rect.bottom() - 56.0 - height),
        egui::vec2(width, height),
    );
    painter.rect_filled(panel, 14.0, fade(Color32::from_rgba_unmultiplied(8, 12, 16, 190)));
    let left = panel.left() + 28.0;
    let line = |y: f32, text: String, size: f32, color: Color32| {
        painter.text(
            Pos2::new(left, panel.top() + y),
            egui::Align2::LEFT_TOP,
            text,
            FontId::proportional(size),
            fade(color),
        );
    };
    line(
        20.0,
        format!("TREE {} · {}", focus.tree + 1, focus.title),
        17.0,
        Color32::from_rgb(150, 196, 170),
    );
    line(
        46.0,
        format!("≈ {} droplets landed", thousands(t.droplets)),
        40.0,
        Color32::WHITE,
    );
    line(
        100.0,
        format!(
            "{} spray parcels · {:.0} µL on this leaf",
            t.parcels,
            t.litres * 1e6
        ),
        20.0,
        Color32::from_rgb(214, 222, 228),
    );
    line(
        130.0,
        format!(
            "tree: {} parcels on {} wet leaves · 160–240 µm water",
            thousands(current.tree_parcels as f64),
            current.wet_leaves
        ),
        16.0,
        Color32::from_rgb(150, 160, 170),
    );
    if let Some((before, label)) = &focus.reference {
        line(
            166.0,
            format!(
                "{label}: ≈ {} droplets · tree: {} parcels on {} wet leaves",
                thousands(before.leaf.droplets),
                thousands(before.tree_parcels as f64),
                before.wet_leaves
            ),
            17.0,
            Color32::from_rgb(242, 196, 120),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deposit(leaf: usize, litres: f64) -> Deposit {
        Deposit {
            leaf,
            t: 0.5,
            x: 0.0,
            side: 1.0,
            time: 0.0,
            litres,
            droplets: litres * 1000.0,
        }
    }

    #[test]
    fn tally_sums_only_the_chosen_leaf() {
        let deposits = [deposit(3, 1e-6), deposit(4, 2e-6), deposit(3, 3e-6)];
        let t = tally(&deposits, 3);
        assert_eq!(t.parcels, 2);
        assert!((t.litres - 4e-6).abs() < 1e-18);
        assert!((t.droplets - 4e-3).abs() < 1e-15);
        assert_eq!(tally(&deposits, 9), Tally::default());
    }

    #[test]
    fn focus_blend_zooms_in_holds_and_returns() {
        let focus = Focus {
            leaf: 0,
            tree: 0,
            start: 10.0,
            length: 0.1,
            title: String::new(),
            reference: None,
            yaw: 0.0,
            pitch: 1.0,
        };
        assert_eq!(focus.weights(9.0), (0.0, 0.0));
        let (target, view) = focus.weights(10.0 + ZOOM_S * 0.5);
        assert!(target > view && view > 0.0 && view < 1.0);
        assert_eq!(focus.weights(10.0 + ZOOM_S + HOLD_S * 0.5), (1.0, 1.0));
        let (target, view) = focus.weights(10.0 + ZOOM_S * 1.5 + HOLD_S);
        assert!(target > view && view > 0.0 && view < 1.0);
        assert_eq!(focus.weights(focus.end() + 0.01), (0.0, 0.0));
        assert!((wrap(3.0 - -3.0) - (6.0 - std::f64::consts::TAU)).abs() < 1e-12);
    }

    #[test]
    fn water_bead_is_closed_outward_and_sized() {
        let mut out = Vec::new();
        let up = Vec3::new(0.2, 1.0, -0.1).normalize();
        bead(&mut out, Vec3::ONE, up, 0.001);
        assert_eq!(out.len(), 24);
        for t in &out {
            let n = (t.p[1] - t.p[0]).cross(t.p[2] - t.p[0]);
            let centroid = (t.p[0] + t.p[1] + t.p[2]) / 3.0 - Vec3::ONE;
            assert!(n.dot(centroid) > 0.0);
            assert!(t.p.iter().all(|p| (*p - Vec3::ONE).dot(up) > -1e-12));
            assert!(t.p.iter().all(|p| p.distance(Vec3::ONE) <= 0.001 + 1e-12));
        }
        assert_eq!(thousands(34519.6), "34,520");
        assert_eq!(thousands(999.0), "999");
    }
}
