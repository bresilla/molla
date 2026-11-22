use molla_geometry::convex_cut::FaceOrigin;
use molla_math::Vec3;
use molla_solvers::strands::StrandModel;
use std::collections::BTreeMap;

struct Plane {
    point: Vec3,
    normal: Vec3,
}

pub struct JunctionSurface {
    planes: BTreeMap<u64, Vec<Plane>>,
    neighbors: BTreeMap<u64, Vec<u64>>,
}

fn halfspace(polygon: &[Vec3], plane: &Plane, outside: bool) -> Vec<Vec3> {
    let mut result = Vec::new();
    for i in 0..polygon.len() {
        let a = polygon[i];
        let b = polygon[(i + 1) % polygon.len()];
        let da = plane.normal.dot(a - plane.point) - 1e-9;
        let db = plane.normal.dot(b - plane.point) - 1e-9;
        let keep_a = (da > 0.0) == outside;
        let keep_b = (db > 0.0) == outside;
        if keep_a {
            result.push(a);
        }
        if keep_a != keep_b {
            let t = (da / (da - db)).clamp(0.0, 1.0);
            result.push(a + (b - a) * t);
        }
    }
    result
}

fn area(polygon: &[Vec3]) -> f64 {
    if polygon.len() < 3 {
        return 0.0;
    }
    (1..polygon.len() - 1)
        .map(|i| {
            (polygon[i] - polygon[0])
                .cross(polygon[i + 1] - polygon[0])
                .length()
                * 0.5
        })
        .sum()
}

impl JunctionSurface {
    pub fn new(model: &StrandModel) -> Self {
        let planes = model
            .fibers()
            .iter()
            .map(|f| {
                let planes = f
                    .shape
                    .faces()
                    .iter()
                    .enumerate()
                    .map(|(i, face)| Plane {
                        point: f
                            .motion
                            .pose
                            .transform_point(f.shape.vertices()[face.vertices[0] as usize]),
                        normal: f.motion.pose.rotation
                            * molla_geometry::cell_contact::face_normal(&f.shape, i),
                    })
                    .collect();
                (f.id, planes)
            })
            .collect();
        let mut neighbors = BTreeMap::<u64, Vec<u64>>::new();
        for b in model.bonds().iter().filter(|b| b.intact()) {
            neighbors.entry(b.fibers[0]).or_default().push(b.fibers[1]);
            neighbors.entry(b.fibers[1]).or_default().push(b.fibers[0]);
        }
        let direct = neighbors.clone();
        for (&id, group) in &mut neighbors {
            for neighbor in &direct[&id] {
                group.extend(
                    direct
                        .get(neighbor)
                        .into_iter()
                        .flatten()
                        .copied()
                        .filter(|&other| other != id),
                );
            }
            group.sort_unstable();
            group.dedup();
        }
        Self { planes, neighbors }
    }

    /// Remove surfaces buried inside intact neighbors, retaining cut and fracture caps.
    pub fn patches(&self, id: u64, origin: FaceOrigin, polygon: Vec<Vec3>) -> Vec<Vec<Vec3>> {
        if matches!(origin, FaceOrigin::Cut { .. } | FaceOrigin::Fracture { .. }) {
            return vec![polygon];
        }
        let normal = (1..polygon.len() - 1)
            .map(|i| (polygon[i] - polygon[0]).cross(polygon[i + 1] - polygon[0]))
            .sum::<Vec3>()
            .normalize_or_zero();
        let mut patches = vec![polygon];
        for &neighbor in self.neighbors.get(&id).into_iter().flatten() {
            let planes = &self.planes[&neighbor];
            let mut visible = Vec::new();
            for patch in patches {
                if id < neighbor
                    && planes.iter().any(|p| {
                        p.normal.dot(normal) > 1.0 - 1e-8
                            && patch
                                .iter()
                                .all(|v| p.normal.dot(*v - p.point).abs() < 1e-9)
                    })
                {
                    visible.push(patch);
                    continue;
                }
                let mut inside = patch;
                for plane in planes {
                    if inside.len() < 3 {
                        break;
                    }
                    let outside = halfspace(&inside, plane, true);
                    if area(&outside) > 1e-16 {
                        visible.push(outside);
                    }
                    inside = halfspace(&inside, plane, false);
                }
            }
            patches = visible;
        }
        patches
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_geometry::convex_cut::ConvexCell;
    use molla_math::{Quat, Transform};
    use molla_solvers::strands::{Bond, BondMaterial, Fiber};

    fn model(overlap: bool, broken: bool) -> StrandModel {
        let fibers: Vec<_> = [0u64, 1]
            .into_iter()
            .map(|id| {
                Fiber::new(
                    id,
                    ConvexCell::prism(0.1, 1.0, 8).unwrap(),
                    650.0,
                    Transform::new(
                        Vec3::Z * if overlap { 0.0 } else { id as f64 - 0.5 },
                        Quat::IDENTITY,
                    ),
                )
                .unwrap()
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
        let anchors = if overlap {
            [Vec3::ZERO; 2]
        } else {
            [Vec3::Z * 0.5, -Vec3::Z * 0.5]
        };
        let mut bond = Bond::new(0, [&fibers[0], &fibers[1]], anchors, material).unwrap();
        if broken {
            bond.damage = 1.0;
        }
        StrandModel::new(fibers, vec![bond]).unwrap()
    }

    fn patches(model: &StrandModel, cap: bool) -> Vec<Vec<Vec3>> {
        let surface = JunctionSurface::new(model);
        model
            .fibers()
            .iter()
            .flat_map(|f| {
                f.shape.faces().iter().flat_map(|face| {
                    let polygon = face
                        .vertices
                        .iter()
                        .map(|&id| {
                            f.motion
                                .pose
                                .transform_point(f.shape.vertices()[id as usize])
                        })
                        .collect();
                    surface.patches(
                        f.id,
                        if cap {
                            FaceOrigin::Cut {
                                id: 1,
                                positive_side: false,
                            }
                        } else {
                            face.origin
                        },
                        polygon,
                    )
                })
            })
            .collect()
    }

    #[test]
    fn internal_join_caps_are_removed_but_detached_caps_remain() {
        assert_eq!(patches(&model(false, false), false).len(), 18);
        assert_eq!(patches(&model(false, true), false).len(), 20);
        assert_eq!(patches(&model(false, false), true).len(), 20);
    }

    #[test]
    fn coincident_outer_faces_have_one_owner_without_holes() {
        let patches = patches(&model(true, false), false);
        assert_eq!(patches.len(), 10);
        let actual: f64 = patches.iter().map(|p| area(p)).sum();
        let expected: f64 = self::patches(&model(true, true), false)
            .iter()
            .map(|p| area(p))
            .sum::<f64>()
            * 0.5;
        assert!((actual - expected).abs() < 1e-8);
    }
}
