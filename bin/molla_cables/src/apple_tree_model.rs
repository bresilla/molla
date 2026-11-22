use molla_core::{Error, Result};
use molla_geometry::convex_cut::ConvexCell;
use molla_math::{Quat, Transform, Vec3};
use molla_solvers::strands::{Bond, BondMaterial, Fiber, Pin, StrandModel};
use serde_json::Value;
use std::collections::BTreeSet;

pub const SOURCE: &str = include_str!("../data/apple_tree1.json");
pub const DEFAULT_LIMBS: usize = 20;

struct Limb {
    points: Vec<Vec3>,
    parent: Option<(usize, usize)>,
    length: f64,
    radius: f64,
}

pub struct TreeModel {
    pub model: StrandModel,
    pub trunk_cells: Vec<u64>,
    #[allow(dead_code)]
    pub limb_cells: Vec<Vec<u64>>,
    pub tree_id: String,
    pub source_limbs: usize,
    pub simulated_limbs: usize,
}

fn invalid(message: &str) -> Error {
    Error::Build(format!("apple skeleton: {message}"))
}

fn points(value: &Value) -> Result<Vec<Vec3>> {
    let mut points = Vec::new();
    for value in value
        .as_array()
        .ok_or_else(|| invalid("missing centerline"))?
    {
        let xyz = value
            .as_array()
            .filter(|a| a.len() == 3)
            .ok_or_else(|| invalid("centerline point must have three coordinates"))?;
        let xyz = xyz
            .iter()
            .map(|x| {
                x.as_f64()
                    .filter(|x| x.is_finite())
                    .ok_or_else(|| invalid("nonfinite centerline"))
            })
            .collect::<Result<Vec<_>>>()?;
        let point = Vec3::new(xyz[0], xyz[2], -xyz[1]);
        if points
            .last()
            .is_none_or(|p: &Vec3| p.distance(point) > 1e-5)
        {
            points.push(point);
        }
    }
    if points.len() < 2 {
        return Err(invalid("centerline needs two distinct points"));
    }
    Ok(points)
}

fn length(points: &[Vec3]) -> f64 {
    points.windows(2).map(|p| p[0].distance(p[1])).sum()
}

fn smooth(points: &mut Vec<Vec3>) {
    for _ in 0..4 {
        let mut next = points.clone();
        for i in 1..points.len() - 1 {
            next[i] = points[i] * 0.5 + (points[i - 1] + points[i + 1]) * 0.25;
        }
        *points = next;
    }
}

fn sampled(points: &[Vec3]) -> Vec<Vec3> {
    let mut result = vec![points[0]];
    let mut distance = 0.0;
    for i in 1..points.len() - 1 {
        distance += points[i].distance(points[i - 1]);
        if distance >= 0.28 && points[i].distance(*points.last().unwrap()) > 0.08 {
            result.push(points[i]);
            distance = 0.0;
        }
    }
    result.push(*points.last().unwrap());
    result
}

pub(crate) fn tapered_cell(a: f64, b: f64, length: f64) -> Result<ConvexCell> {
    let prism = ConvexCell::prism(1.0, length, 8)?;
    let vertices = prism
        .vertices()
        .iter()
        .map(|v| {
            let radius = if v.z < 0.0 { a } else { b };
            Vec3::new(v.x * radius, v.y * radius, v.z)
        })
        .collect();
    ConvexCell::new(vertices, prism.faces().to_vec())
}

fn nearest_segment(point: Vec3, segments: &[(usize, Vec3, Vec3)]) -> (usize, Vec3) {
    segments
        .iter()
        .map(|&(id, a, b)| {
            let d = b - a;
            let p = a + d * ((point - a).dot(d) / d.length_squared()).clamp(0.0, 1.0);
            (id, p)
        })
        .min_by(|a, b| {
            a.1.distance_squared(point)
                .total_cmp(&b.1.distance_squared(point))
        })
        .unwrap()
}

pub(crate) fn connect(
    bonds: &mut Vec<Bond>,
    fibers: &[Fiber],
    a: usize,
    b: usize,
    point: Vec3,
    radius: f64,
) -> Result<()> {
    let [a, b] = [&fibers[a], &fibers[b]];
    let dual = a.center().distance(b.center()).max(0.03);
    let area = std::f64::consts::PI * radius.powi(2);
    let moment = std::f64::consts::PI * radius.powi(4) / 4.0;
    let material = BondMaterial {
        translation: Vec3::splat(dual / (2e7 * area)),
        rotation: Vec3::splat(dual / (2e9 * moment)),
        force_limit: Vec3::splat(2e7 * area),
        torque_limit: Vec3::splat(1e7 * moment / radius),
        damage_rate: 0.8,
        damping: 1.2,
    };
    let anchors = [a, b].map(|f| f.rest.inverse_transform_point(point));
    bonds.push(Bond::new(bonds.len() as u64, [a, b], anchors, material)?);
    Ok(())
}

pub fn build(source: &str, limit: usize) -> Result<TreeModel> {
    if !(1..=80).contains(&limit) {
        return Err(invalid("limb budget must be 1..80"));
    }
    let json: Value = serde_json::from_str(source).map_err(|e| invalid(&e.to_string()))?;
    let geometry = &json["geometry"];
    let mut trunk = points(&geometry["trunk_points"])?;
    if trunk[0].y > trunk.last().unwrap().y {
        trunk.reverse();
    }
    let base = trunk[0];
    let radii = geometry["trunk_radii"]
        .as_array()
        .ok_or_else(|| invalid("missing trunk radii"))?;
    let mut radii = radii
        .iter()
        .map(|r| {
            r.as_f64()
                .filter(|r| r.is_finite() && *r > 0.0)
                .ok_or_else(|| invalid("invalid trunk radius"))
        })
        .collect::<Result<Vec<_>>>()?;
    if radii.is_empty() {
        return Err(invalid("empty trunk radii"));
    }
    radii.sort_by(f64::total_cmp);
    let root_radius = radii[(radii.len() * 9 / 10).min(radii.len() - 1)].clamp(0.015, 0.08);
    let mut pending = geometry["branches"]
        .as_object()
        .ok_or_else(|| invalid("missing branches"))?
        .iter()
        .map(|(id, b)| Ok((id.clone(), points(&b["skeleton"])?)))
        .collect::<Result<Vec<_>>>()?;
    pending.sort_by(|a, b| a.0.cmp(&b.0));
    let source_limbs = pending.len();
    let mut limbs = vec![Limb {
        length: length(&trunk),
        points: trunk,
        parent: None,
        radius: root_radius,
    }];
    while !pending.is_empty() {
        let mut best = (f64::INFINITY, 0, false, 0, 0);
        for (i, (_, path)) in pending.iter().enumerate() {
            for (end, flip) in [(path[0], false), (*path.last().unwrap(), true)] {
                for (parent, limb) in limbs.iter().enumerate() {
                    for (vertex, &p) in limb.points.iter().enumerate() {
                        let distance = end.distance_squared(p);
                        if distance < best.0 {
                            best = (distance, i, flip, parent, vertex);
                        }
                    }
                }
            }
        }
        let (_, i, flip, parent, vertex) = best;
        let (_, mut path) = pending.remove(i);
        if flip {
            path.reverse();
        }
        path[0] = limbs[parent].points[vertex];
        let fraction = length(&limbs[parent].points[..=vertex]) / limbs[parent].length;
        let radius = (limbs[parent].radius * (1.0 - 0.8 * fraction) * 0.65).max(0.0035);
        limbs.push(Limb {
            length: length(&path),
            points: path,
            parent: Some((parent, vertex)),
            radius,
        });
    }
    let mut ranked: Vec<_> = (1..limbs.len()).collect();
    ranked.sort_by(|&a, &b| limbs[b].length.total_cmp(&limbs[a].length));
    let mut selected = BTreeSet::from([0]);
    for id in ranked {
        let mut chain = Vec::new();
        let mut current = id;
        while !selected.contains(&current) {
            chain.push(current);
            current = limbs[current].parent.unwrap().0;
        }
        if selected.len() + chain.len() <= limit + 1 {
            selected.extend(chain);
        }
    }
    for i in 0..limbs.len() {
        smooth(&mut limbs[i].points);
        if let Some((parent, vertex)) = limbs[i].parent {
            limbs[i].points[0] = limbs[parent].points[vertex];
        }
    }
    let mut fibers = Vec::new();
    let mut bonds = Vec::new();
    let mut segments = vec![Vec::new(); limbs.len()];
    for &id in &selected {
        let limb = &limbs[id];
        let mut path: Vec<_> = sampled(&limb.points)
            .into_iter()
            .map(|p| p - base + Vec3::Y * 0.015)
            .collect();
        let attachment = limb
            .parent
            .map(|(parent, _)| nearest_segment(path[0], &segments[parent]));
        if let Some((_, point)) = attachment {
            path[0] = point;
        }
        let total = length(&path);
        let mut traveled = 0.0;
        for pair in path.windows(2) {
            let [a, b] = [pair[0], pair[1]];
            let size = a.distance(b);
            if size < 1e-5 {
                continue;
            }
            let r0 = (limb.radius * (1.0 - 0.8 * traveled / total)).max(0.0035);
            traveled += size;
            let r1 = (limb.radius * (1.0 - 0.8 * traveled / total)).max(0.0035);
            let rotation = Quat::from_rotation_arc(Vec3::Z, (b - a) / size);
            let pose = Transform::new((a + b) * 0.5, rotation);
            let index = fibers.len();
            let mut fiber = Fiber::new(index as u64, tapered_cell(r0, r1, size)?, 650.0, pose)?;
            if index == 0 {
                fiber.pin = Some(Pin {
                    local_point: -Vec3::Z * size * 0.5,
                    pose,
                });
            }
            fibers.push(fiber);
            if let Some(&(previous, _, _)) = segments[id].last() {
                connect(&mut bonds, &fibers, previous, index, a, r0)?;
            } else if let Some((parent, point)) = attachment {
                connect(&mut bonds, &fibers, parent, index, point, r0)?;
            }
            segments[id].push((index, a, b));
        }
    }
    let model = StrandModel::new(fibers, bonds)?;
    if model.components().len() != 1 {
        return Err(invalid("disconnected tree"));
    }
    Ok(TreeModel {
        model,
        trunk_cells: segments[0].iter().map(|s| s.0 as u64).collect(),
        limb_cells: selected
            .iter()
            .filter(|&&id| id != 0)
            .map(|&id| segments[id].iter().map(|s| s.0 as u64).collect())
            .collect(),
        tree_id: json["tree_id"].as_str().unwrap_or("apple tree").to_owned(),
        source_limbs,
        simulated_limbs: selected.len() - 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_geometry::convex_cut::{CutPlane, FaceOrigin};

    #[test]
    fn measured_tree_is_connected_grounded_and_bounded() {
        let tree = build(SOURCE, DEFAULT_LIMBS).unwrap();
        assert_eq!(tree.tree_id, "tree1");
        assert_eq!(tree.source_limbs, 162);
        assert_eq!(tree.simulated_limbs, DEFAULT_LIMBS);
        assert!(tree.model.fibers().len() < 128);
        assert_eq!(tree.model.components().len(), 1);
        assert_eq!(
            tree.model
                .fibers()
                .iter()
                .filter(|f| f.pin.is_some())
                .count(),
            1
        );
        assert!(
            tree.model
                .fibers()
                .iter()
                .map(|f| f.center().y)
                .fold(0.0, f64::max)
                > 2.3
        );
        assert!(
            tree.model
                .bonds()
                .iter()
                .all(|b| b.material.damage_rate == 0.8)
        );
    }

    #[test]
    fn off_node_cut_conserves_mass_and_detaches_a_subtree() {
        let tree = build(SOURCE, DEFAULT_LIMBS).unwrap();
        let fiber = &tree.model.fibers()[3];
        let hit = fiber.rest.transform_point(Vec3::new(0.0, 0.0, 0.0317));
        let plane = CutPlane::new(hit, fiber.rest.rotation * Vec3::Z).unwrap();
        let edit = tree.model.prepare_cut(plane, &[fiber.id], 1).unwrap();
        assert_eq!(edit.report().replacements.len(), 1);
        assert_eq!(edit.candidate().components().len(), 2);
        let mass = |m: &StrandModel| m.fibers().iter().map(|f| f.mass.mass).sum::<f64>();
        assert!((mass(&tree.model) - mass(edit.candidate())).abs() < 1e-10);
        for f in edit.candidate().fibers() {
            for face in f
                .shape
                .faces()
                .iter()
                .filter(|f| matches!(f.origin, FaceOrigin::Cut { .. }))
            {
                assert!(face.vertices.iter().all(|&v| {
                    plane
                        .distance(f.rest.transform_point(f.shape.vertices()[v as usize]))
                        .abs()
                        < 1e-10
                }));
            }
        }
    }

    #[test]
    fn junction_volumes_share_overlapping_local_neighborhoods() {
        let tree = build(SOURCE, DEFAULT_LIMBS).unwrap();
        let neighbors = |id| {
            tree.model
                .bonds()
                .iter()
                .filter_map(move |b| {
                    if b.fibers[0] == id {
                        Some(b.fibers[1])
                    } else if b.fibers[1] == id {
                        Some(b.fibers[0])
                    } else {
                        None
                    }
                })
                .collect::<BTreeSet<_>>()
        };
        let mut count = 0;
        for (i, a) in tree.model.fibers().iter().enumerate() {
            for b in tree.model.fibers().iter().skip(i + 1) {
                if neighbors(a.id).contains(&b.id) {
                    continue;
                }
                if let Some(m) = molla_geometry::cell_contact::cell_manifold(
                    &a.shape, a.rest, &b.shape, b.rest, 0.0,
                )
                .unwrap()
                {
                    let depth = m
                        .points
                        .iter()
                        .map(|p| (p.a - p.b).dot(m.normal))
                        .fold(0.0, f64::max);
                    if depth > 1e-6 {
                        let shared = !neighbors(a.id).is_disjoint(&neighbors(b.id));
                        eprintln!(
                            "Rest junction overlap {}-{} depth={depth:.6} shared_neighbor={shared}",
                            a.id, b.id
                        );
                        count += 1;
                        assert!(
                            shared,
                            "authored overlap outside the local junction neighborhood"
                        );
                    }
                }
            }
        }
        eprintln!("Unfiltered junction overlaps: {count}");
        assert_eq!(count, 2);
    }

    #[test]
    fn rejects_bad_source_and_invalid_budget() {
        assert!(build("{}", DEFAULT_LIMBS).is_err());
        assert!(build(SOURCE, 0).is_err());
        assert!(build(SOURCE, 81).is_err());
    }
}
