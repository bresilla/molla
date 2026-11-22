use crate::{
    draw::{Surface, Triangle},
    leafy_model::{Leaf, random},
};
use eframe::egui::Color32;
use molla_math::{Quat, Vec3};
use molla_solvers::{
    attachments::{Attachment, Particle},
    strands::StrandModel,
};

use crate::foliage_layout as layout;

/// Upper (adaxial) and lower leaf faces sample the scanned textures; plain leaves use vertex color.
const UPPER: f32 = 4.0;
const LOWER: f32 = 5.0;
const PLAIN: f32 = 6.0;
/// Lateral growth of strip and hexagon leaf polygons beyond the blade outline.
const MARGIN: [f64; 2] = [1.06, 1.14];
/// Strength of the per-leaf tint relative to the scanned texture colors.
const TINT_STRENGTH: f64 = 0.5;

fn leaf_color(tint: f64) -> Color32 {
    let palette = [
        [30.0, 76.0, 42.0],
        [42.0, 105.0, 54.0],
        [68.0, 132.0, 43.0],
        [106.0, 145.0, 67.0],
        [137.0, 160.0, 48.0],
        [171.0, 166.0, 62.0],
    ];
    let position = tint.clamp(0.0, 1.0) * (palette.len() - 1) as f64;
    let index = (position.floor() as usize).min(palette.len() - 2);
    let u = position - index as f64;
    let rgb: [u8; 3] =
        std::array::from_fn(|c| (palette[index][c] * (1.0 - u) + palette[index + 1][c] * u) as u8);
    Color32::from_rgb(rgb[0], rgb[1], rgb[2])
}

fn triangle(out: &mut Vec<Triangle>, p: [Vec3; 3], color: Color32) {
    out.push(Triangle {
        p,
        color,
        material: PLAIN,
        surface: None,
    });
    out.push(Triangle {
        p: [p[2], p[1], p[0]],
        color,
        material: PLAIN,
        surface: None,
    });
}

/// Texture selection of one leaf: scanned layer, palette donor, mirroring and tint.
#[derive(Clone, Copy)]
struct Look {
    code: f32,
    mirror: bool,
    color: Color32,
    textured: bool,
}

fn look(index: usize, leaf: &Leaf) -> Look {
    let count = layout::LEAF_TEXTURES;
    let pick = |salt| ((random(index, salt) * count as f64) as usize).min(count - 1);
    let (layer, palette) = (pick(0x1eaf), pick(0xc010));
    let tint = leaf_color(leaf.tint).to_array();
    let middle = leaf_color(0.5).to_array();
    let channel = |c: usize| {
        let ratio = f64::from(tint[c]) / f64::from(middle[c]);
        (127.5 * (1.0 + TINT_STRENGTH * (ratio - 1.0))).clamp(0.0, 255.0) as u8
    };
    Look {
        code: (layer + 64 * palette) as f32 / 4096.0,
        mirror: random(index, 0x5eed) < 0.5,
        color: Color32::from_rgb(channel(0), channel(1), channel(2)),
        textured: true,
    }
}

/// Pushes both faces of one blade triangle given positions, normals and blade coordinates.
fn blade_triangle(
    out: &mut Vec<Triangle>,
    look: &Look,
    p: [Vec3; 3],
    normals: [Vec3; 3],
    coordinates: [(f64, f64); 3],
    up: Vec3,
) {
    let uv = coordinates.map(|(t, x)| layout::leaf_uv(t, x, look.mirror));
    let upper = (p[1] - p[0]).cross(p[2] - p[0]).dot(up) >= 0.0;
    for flip in [false, true] {
        let order = if flip { [2, 1, 0] } else { [0, 1, 2] };
        let facing_up = upper != flip;
        let sign = if facing_up { 1.0 } else { -1.0 };
        let material = match (look.textured, facing_up) {
            (false, _) => PLAIN,
            (true, true) => UPPER + look.code,
            (true, false) => LOWER + look.code,
        };
        out.push(Triangle {
            p: order.map(|i| p[i]),
            color: look.color,
            material,
            surface: Some(Surface {
                normals: order.map(|i| normals[i] * sign),
                uv: order.map(|i| uv[i]),
            }),
        });
    }
}

/// Drawn blade surface of one leaf: `point(t, x)` maps the fraction `t` along
/// the blade from its base and the lateral offset `x` (in blade lengths) to world space.
#[derive(Clone, Copy)]
pub struct LeafSurface {
    pub base: Vec3,
    pub direction: Vec3,
    pub side: Vec3,
    pub normal: Vec3,
    pub length: f64,
    droop: bool,
    camber: bool,
}

impl LeafSurface {
    pub fn point(&self, t: f64, x: f64) -> Vec3 {
        let droop = if self.droop { 0.12 * t * t } else { 0.0 };
        let camber = if self.camber { 0.12 * x * x * 4.0 } else { 0.0 };
        self.base
            + self.direction * (t * self.length)
            + self.side * (x * self.length)
            + self.normal * ((camber - droop) * self.length)
    }

    /// Unit surface normal at `(t, x)` on the side of `normal`.
    pub fn normal_at(&self, t: f64, x: f64) -> Vec3 {
        let h = 0.02;
        let (t0, t1) = ((t - h).max(0.0), (t + h).min(1.0));
        let along = self.point(t1, x) - self.point(t0, x);
        let across = self.point(t, x + h) - self.point(t, x - h);
        let n = across.cross(along).normalize_or(self.normal);
        if n.dot(self.normal) < 0.0 { -n } else { n }
    }
}

/// Half-width of the drawn blade outline, in blade lengths, at fraction `t`.
pub fn half_width(t: f64) -> f64 {
    layout::half_width(t)
}

pub fn surface(leaf: &Leaf, p: &Particle, wood: Option<&StrandModel>) -> Option<LeafSurface> {
    let grounded = !p.attached && p.position.y < 0.0021;
    let (direction, normal) = if p.attached
        && let Some(blade) = p.leaf
    {
        wood?.fiber(leaf.owner)?;
        (blade.frame.direction, blade.frame.normal)
    } else if p.attached {
        let f = wood?.fiber(leaf.owner)?;
        let base = f.motion.pose.transform_point(leaf.anchor);
        let direction = (p.position - base).normalize_or_zero();
        let n = f.motion.pose.rotation * leaf.normal;
        (
            direction,
            (n - direction * n.dot(direction)).normalize_or_zero(),
        )
    } else if let Some(blade) = p.leaf {
        (blade.frame.direction, blade.frame.normal)
    } else if grounded {
        (
            (leaf.released_direction * Vec3::new(1.0, 0.0, 1.0)).normalize_or_zero(),
            Vec3::Y,
        )
    } else {
        let q = Quat::from_axis_angle(
            leaf.released_direction,
            (p.age * 8.0 + leaf.tint * 6.0).sin() * 0.6,
        ) * Quat::from_rotation_y(p.age * (leaf.tint - 0.5) * 2.0);
        (q * leaf.released_direction, q * leaf.released_normal)
    };
    let rigid_blade = p.leaf.is_some() && !p.attached;
    Some(LeafSurface {
        base: p.position - direction * (leaf.length * 0.5),
        direction,
        side: direction.cross(normal).normalize_or_zero(),
        normal,
        length: leaf.length,
        droop: !(grounded || rigid_blade),
        camber: !rigid_blade,
    })
}

pub fn mesh(
    leaves: &[Leaf],
    particles: &[Particle],
    wood: &StrandModel,
    stress: bool,
    strengths: &[Attachment],
    strength_scale: f64,
) -> Vec<Triangle> {
    mesh_with_segments(
        leaves,
        particles,
        wood,
        stress,
        strengths,
        strength_scale,
        12,
    )
}

pub fn mesh_with_segments(
    leaves: &[Leaf],
    particles: &[Particle],
    wood: &StrandModel,
    stress: bool,
    strengths: &[Attachment],
    strength_scale: f64,
    segments: usize,
) -> Vec<Triangle> {
    mesh_inner(
        leaves,
        particles,
        Some(wood),
        stress,
        strengths,
        strength_scale,
        segments,
    )
}

/// GPU instances of every drawable leaf: blade surface, texture look and petiole anchor.
pub fn instances(
    leaves: &[Leaf],
    particles: &[Particle],
    wood: &StrandModel,
) -> Vec<crate::studio::LeafInstance> {
    use crate::studio::leaves as flag;
    let xyz = |v: Vec3, w: f64| [v.x as f32, v.y as f32, v.z as f32, w as f32];
    leaves
        .iter()
        .zip(particles)
        .enumerate()
        .filter_map(|(index, (leaf, p))| {
            let s = surface(leaf, p, Some(wood))?;
            let look = look(index, leaf);
            let mut flags = 0;
            for (set, bit) in [
                (look.mirror, flag::MIRROR),
                (s.droop, flag::DROOP),
                (s.camber, flag::CAMBER),
            ] {
                if set {
                    flags |= bit;
                }
            }
            let plate = p.leaf.filter(|b| !p.attached && b.camber > 0.0);
            if plate.is_some() {
                flags |= flag::PLATE;
            }
            let center = s.base + s.direction * (0.5 * s.length);
            let anchor = p
                .attached
                .then(|| wood.fiber(leaf.owner))
                .flatten()
                .map(|f| f.motion.pose.transform_point(leaf.anchor));
            if anchor.is_some() {
                flags |= flag::PETIOLE;
            }
            let tint = look.color.to_array().map(|c| f64::from(c) / 255.0);
            Some(crate::studio::LeafInstance {
                center_length: xyz(center, s.length),
                direction_code: xyz(s.direction, f64::from(look.code)),
                normal_flags: xyz(s.normal, f64::from(flags)),
                anchor_bulge: xyz(anchor.unwrap_or(center), plate.map_or(0.0, |b| b.camber)),
                tint_span: xyz(
                    Vec3::new(tint[0], tint[1], tint[2]),
                    plate.map_or(0.6 * s.length, |b| b.width),
                ),
            })
        })
        .collect()
}

pub fn free_mesh(
    leaves: &[Leaf],
    particles: &[Particle],
    strengths: &[Attachment],
) -> Vec<Triangle> {
    mesh_inner(leaves, particles, None, false, strengths, 1.0, 12)
}

fn mesh_inner(
    leaves: &[Leaf],
    particles: &[Particle],
    wood: Option<&StrandModel>,
    stress: bool,
    strengths: &[Attachment],
    strength_scale: f64,
    segments: usize,
) -> Vec<Triangle> {
    let segments = segments.clamp(3, 12);
    let mut result = Vec::with_capacity(leaves.len() * if segments == 3 { 14 } else { 4 * segments + 2 });
    for (index, ((leaf, p), a)) in leaves.iter().zip(particles).zip(strengths).enumerate() {
        let owner = wood.and_then(|model| model.fiber(leaf.owner));
        let Some(blade_surface) = surface(leaf, p, wood) else {
            continue;
        };
        let LeafSurface {
            base, side, normal, ..
        } = blade_surface;
        let mut look = look(index, leaf);
        if stress && p.attached {
            look.textured = false;
            look.color = Color32::from_rgb(
                (80.0 + 175.0 * (p.force.length() / (a.strength * strength_scale)).min(1.0)) as u8,
                110,
                40,
            );
        }
        let point = |t: f64, x: f64| blade_surface.point(t, x);
        let width = half_width;
        if segments == 3 || (p.leaf.is_some() && !p.attached) {
            let w = [width(1.0 / 3.0), width(2.0 / 3.0)];
            let mut coordinates = [
                (0.0, 0.0),
                (1.0 / 3.0, w[0]),
                (2.0 / 3.0, w[1]),
                (1.0, 0.0),
                (2.0 / 3.0, -w[1]),
                (1.0 / 3.0, -w[0]),
                (0.5, 0.0),
            ];
            let mut points = coordinates.map(|(t, x)| point(t, x));
            let mut normals = coordinates.map(|(t, x)| blade_surface.normal_at(t, x));
            if !p.attached
                && let Some(blade) = p.leaf
                && blade.camber > 0.0
            {
                let offsets = blade.contact_offsets();
                points = std::array::from_fn(|i| p.position + offsets[i]);
                normals = [blade.frame.normal; 7];
                let lateral = 0.5 * blade.width / blade.length;
                coordinates = coordinates
                    .map(|(t, x)| (t, if x == 0.0 { 0.0 } else { lateral.copysign(x) }));
            }
            let uv_coordinates = coordinates.map(|(t, x)| (t, x * MARGIN[1]));
            for i in 0..6 {
                let j = (i + 1) % 6;
                blade_triangle(
                    &mut result,
                    &look,
                    [points[6], points[i], points[j]],
                    [normals[6], normals[i], normals[j]],
                    [uv_coordinates[6], uv_coordinates[i], uv_coordinates[j]],
                    normal,
                );
            }
        } else {
            for i in 0..segments {
                let a = i as f64 / segments as f64;
                let b = (i + 1) as f64 / segments as f64;
                for sign in [-1.0, 1.0] {
                    let corners = [
                        (a, 0.0),
                        (a, width(a) * sign * MARGIN[0]),
                        (b, width(b) * sign * MARGIN[0]),
                        (b, 0.0),
                    ];
                    let points = corners.map(|(t, x)| point(t, x));
                    let normals = corners.map(|(t, x)| blade_surface.normal_at(t, x));
                    for [i, j, k] in [[0, 1, 2], [0, 2, 3]] {
                        blade_triangle(
                            &mut result,
                            &look,
                            [points[i], points[j], points[k]],
                            [normals[i], normals[j], normals[k]],
                            [corners[i], corners[j], corners[k]],
                            normal,
                        );
                    }
                }
            }
        }
        if p.attached
            && let Some(f) = owner
        {
            let anchor = f.motion.pose.transform_point(leaf.anchor);
            triangle(
                &mut result,
                [anchor - side * 0.0007, anchor + side * 0.0007, base],
                Color32::from_rgb(113, 127, 48),
            );
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_leaf_mesh_matches_integrated_frame_and_floor_extent() {
        use molla_solvers::attachments::{BladeFrame, LeafBody};
        let c = crate::leafy_model::build().unwrap();
        let leaf = &c.leaves[0];
        let q = Quat::from_rotation_z(0.6);
        let direction = q * Vec3::X;
        let normal = q * Vec3::Y;
        let mut p = c.particles[0];
        p.attached = false;
        p.leaf = Some(LeafBody {
            frame: BladeFrame {
                direction,
                normal,
                angular_velocity: Vec3::ZERO,
            },
            length: leaf.length,
            width: leaf.length * 0.6,
            area: c.attachments[0].mass * 9.0,
            camber: 0.0,
        });
        p.position = Vec3::Y * (0.002 + 0.5 * leaf.length * direction.y.abs());
        let mesh = mesh_with_segments(
            &c.leaves[..1],
            &[p],
            &c.tree.model,
            false,
            &c.attachments[..1],
            1.0,
            3,
        );
        assert!(
            mesh.iter()
                .flat_map(|t| t.p)
                .all(|v| (v - p.position).dot(normal).abs() < 1e-10 && v.y >= 0.002 - 1e-10)
        );
    }

    #[test]
    fn curved_leaf_render_uses_the_solver_contact_surface() {
        use molla_solvers::attachments::{BladeFrame, LeafBody};
        let c = crate::leafy_model::build().unwrap();
        for angle in [0.0, 0.6, 2.7] {
            let q = Quat::from_rotation_z(angle);
            let mut p = c.particles[0];
            p.attached = false;
            let blade = LeafBody {
                frame: BladeFrame {
                    direction: q * Vec3::X,
                    normal: q * Vec3::Y,
                    angular_velocity: Vec3::ZERO,
                },
                length: c.leaves[0].length,
                width: c.leaves[0].length * 0.6,
                area: c.attachments[0].mass * 9.0,
                camber: 0.004,
            };
            p.leaf = Some(blade);
            let offsets = blade.contact_offsets();
            p.position =
                Vec3::Y * (0.002 - offsets.iter().map(|r| r.y).fold(f64::INFINITY, f64::min));
            let mesh = mesh_with_segments(
                &c.leaves[..1],
                &[p],
                &c.tree.model,
                false,
                &c.attachments[..1],
                1.0,
                8,
            );
            assert!(mesh.iter().flat_map(|t| t.p).all(|v| v.y >= 0.002 - 1e-10
                && offsets.iter().any(|r| v.distance(p.position + *r) < 1e-10)));
        }
    }

    #[test]
    fn distant_leaf_mesh_preserves_particles_with_fourteen_triangles_each() {
        let c = crate::leafy_model::build().unwrap();
        let triangles = mesh_with_segments(
            &c.leaves,
            &c.particles,
            &c.tree.model,
            false,
            &c.attachments,
            1.0,
            3,
        );
        assert_eq!(triangles.len(), c.leaves.len() * 14);
        assert!(triangles.iter().flat_map(|t| t.p).all(|p| p.is_finite()));
        assert!(c.particles.iter().all(|p| p.attached));
    }

    #[test]
    fn palette_varies_hue_and_brightness_without_changing_strength() {
        let canopy = crate::leafy_model::build().unwrap();
        let colors: std::collections::BTreeSet<_> = canopy
            .leaves
            .iter()
            .map(|leaf| leaf_color(leaf.tint).to_array())
            .collect();
        assert!(colors.len() > 150);
        assert!(colors.iter().any(|c| c[0] < 45 && c[1] < 110));
        assert!(colors.iter().any(|c| c[0] > 150 && c[1] > 155));
        assert_eq!(leaf_color(-1.0), leaf_color(0.0));
        assert_eq!(leaf_color(2.0), leaf_color(1.0));
    }

    #[test]
    fn scanned_leaf_looks_encode_valid_layers_and_both_faces() {
        let c = crate::leafy_model::build().unwrap();
        let count = layout::LEAF_TEXTURES as u32;
        let mut combinations = std::collections::BTreeSet::new();
        let mut layers = std::collections::BTreeSet::new();
        for (i, leaf) in c.leaves.iter().enumerate() {
            let look = look(i, leaf);
            let code = (look.code * 4096.0).round() as u32;
            assert!(code % 64 < count && code / 64 < count && UPPER + look.code < LOWER);
            combinations.insert((code, look.mirror));
            layers.insert(code % 64);
        }
        assert_eq!(layers.len(), layout::LEAF_TEXTURES);
        assert!(
            combinations.len() * 10 > c.leaves.len() * 8,
            "{} combinations for {} leaves",
            combinations.len(),
            c.leaves.len()
        );
        let mesh = mesh_with_segments(
            &c.leaves[..8],
            &c.particles[..8],
            &c.tree.model,
            false,
            &c.attachments[..8],
            1.0,
            8,
        );
        let faces = |low: f32, high: f32| {
            mesh.iter()
                .filter(|t| (low..high).contains(&t.material))
                .count()
        };
        assert!(faces(UPPER, LOWER) > 0 && faces(UPPER, LOWER) == faces(LOWER, PLAIN));
        for t in mesh.iter().filter(|t| t.material < PLAIN) {
            let s = t.surface.unwrap();
            let geometric = (t.p[1] - t.p[0]).cross(t.p[2] - t.p[0]);
            assert!(s.normals.iter().all(|n| n.dot(geometric) > 0.0));
            assert!(s.uv.iter().flatten().all(|v| (-0.01..1.01).contains(v)));
        }
    }
}
