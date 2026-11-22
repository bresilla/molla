use crate::{
    draw::{Surface, Triangle},
    studio::model_asset,
};
use eframe::egui::Color32;
use molla_math::{Mat3, Quat, Vec3};
use molla_solvers::airflow::Quadrotor;
use std::sync::OnceLock;

struct Face {
    p: [Vec3; 3],
    normals: [Vec3; 3],
    uv: [[f32; 2]; 3],
    rotor: Option<usize>,
    texture: usize,
}

fn hub(i: usize) -> Vec3 {
    Vec3::new(
        if i < 2 { -1.0 } else { 1.0 },
        if i % 2 == 0 { -1.0 } else { 1.0 },
        0.0,
    )
}

fn body_point(p: Vec3, pivots: [Vec3; 4]) -> Vec3 {
    let [a, b, c, d] = pivots;
    let mut u = (p.x - a.x) / (c.x - a.x);
    let mut v = (p.y - a.y) / (b.y - a.y);
    for _ in 0..6 {
        let point = a.lerp(b, v).lerp(c.lerp(d, v), u);
        let du = c.lerp(d, v) - a.lerp(b, v);
        let dv = b.lerp(d, u) - a.lerp(c, u);
        let delta = p - point;
        let det = du.x * dv.y - du.y * dv.x;
        u += (delta.x * dv.y - delta.y * dv.x) / det;
        v += (du.x * delta.y - du.y * delta.x) / det;
    }
    let height = a.lerp(b, v).lerp(c.lerp(d, v), u).z;
    let span = ((b - a).length() + (d - c).length()) * 0.25;
    Vec3::new(2.0 * u - 1.0, 2.0 * v - 1.0, (p.z - height) / span)
}

fn faces() -> &'static [Face] {
    static FACES: OnceLock<Vec<Face>> = OnceLock::new();
    FACES.get_or_init(|| {
        let model = model_asset::model();
        let rotors: [(f64, Mat3); 4] = std::array::from_fn(|i| {
            let mut gram = Mat3::ZERO;
            let mut rhs = Vec3::ZERO;
            let mut radius = 0.0_f64;
            for part in &model.parts {
                if part.rotor != Some(i) {
                    continue;
                }
                for p in &part.positions {
                    let p = *p - model.pivots[i];
                    radius = radius.max(p.x.hypot(p.y));
                    let row = Vec3::new(p.x, p.y, 1.0);
                    gram += Mat3::from_cols(row * row.x, row * row.y, row * row.z);
                    rhs += row * p.z;
                }
            }
            let plane = gram.inverse() * rhs;
            let flatten = Mat3::from_cols(
                Vec3::new(1.0, 0.0, -plane.x),
                Vec3::new(0.0, 1.0, -plane.y),
                Vec3::Z,
            );
            (radius, flatten)
        });
        let mut out = Vec::new();
        for part in &model.parts {
            let mut positions = Vec::new();
            let mut normals = Vec::new();
            for (&p, &n) in part.positions.iter().zip(&part.normals) {
                let (position, normal) = if let Some(i) = part.rotor {
                    let (radius, flatten) = rotors[i];
                    (
                        flatten * (p - model.pivots[i]) / radius,
                        flatten.inverse().transpose() * n,
                    )
                } else {
                    let mapped = body_point(p, model.pivots);
                    let derivative =
                        |axis: Vec3| (body_point(p + axis * 1e-4, model.pivots) - mapped) * 1e4;
                    let jacobian = Mat3::from_cols(
                        derivative(Vec3::X),
                        derivative(Vec3::Y),
                        derivative(Vec3::Z),
                    );
                    (mapped, jacobian.inverse().transpose() * n)
                };
                positions.push(position);
                normals.push(normal.normalize());
            }
            for triangle in part.indices.chunks_exact(3) {
                let ids = [triangle[0], triangle[1], triangle[2]];
                out.push(Face {
                    p: ids.map(|i| positions[i]),
                    normals: ids.map(|i| normals[i]),
                    uv: ids.map(|i| part.uv[i]),
                    rotor: part.rotor,
                    texture: part.texture,
                });
            }
        }
        out
    })
}

pub fn mesh(drone: &Quadrotor, _time: f64) -> Vec<Triangle> {
    let pose = drone.pose().unwrap();
    let phases: [Quat; 4] = std::array::from_fn(|i| {
        Quat::from_rotation_z(
            drone
                .world
                .devices
                .propellers
                .reading(drone.propellers[i])
                .unwrap()
                .phase,
        )
    });
    let mut out = Vec::with_capacity(faces().len() * 2);
    for face in faces() {
        let rotation = pose.rotation * face.rotor.map_or(Quat::IDENTITY, |i| phases[i]);
        let positions = face.p.map(|p| {
            let local = face.rotor.map_or(p * drone.config.arm, |i| {
                hub(i) * drone.config.arm + phases[i] * p * drone.config.rotor_radius
            });
            pose.transform_point(local)
        });
        let normals = face.normals.map(|n| rotation * n);
        for ids in [[0, 1, 2], [2, 1, 0]] {
            let sign = if ids[0] == 0 { 1.0 } else { -1.0 };
            out.push(Triangle {
                p: ids.map(|i| positions[i]),
                color: Color32::WHITE,
                material: 8.0 + face.texture as f32,
                surface: Some(Surface {
                    normals: ids.map(|i| normals[i] * sign),
                    uv: ids.map(|i| face.uv[i]),
                }),
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mavic_asset_is_textured_finite_and_registered_to_the_physical_rotors() {
        let model = model_asset::model();
        assert_eq!(faces().len(), 12_701);
        for i in 0..4 {
            assert!(body_point(model.pivots[i], model.pivots).distance(hub(i)) < 1e-9);
            let rotor: Vec<_> = faces().iter().filter(|f| f.rotor == Some(i)).collect();
            assert!(!rotor.is_empty());
            let radius = rotor
                .iter()
                .flat_map(|f| f.p)
                .map(|p| p.x.hypot(p.y))
                .fold(0.0_f64, f64::max);
            assert!((radius - 1.0).abs() < 1e-9);
        }
        assert_eq!(model_asset::textures().len(), 2);
        let drone = Quadrotor::new(Vec3::Y * 3.0, Default::default()).unwrap();
        let triangles = mesh(&drone, 0.0);
        let mut keys = std::collections::BTreeSet::new();
        for f in faces() {
            let mut key = f.p.map(|p| p.to_array().map(|v| (v * 1e6).round() as i64));
            key.sort();
            assert!(keys.insert((f.rotor, key)));
        }
        assert_eq!(triangles.len(), 25_402);
        assert!(
            triangles
                .iter()
                .flat_map(|f| f.p)
                .all(|p| p.is_finite() && p.y > 2.3 && p.y < 3.3)
        );
        assert!(triangles.iter().all(|f| {
            f.surface
                .unwrap()
                .normals
                .iter()
                .all(|n| n.is_finite() && (n.length() - 1.0).abs() < 1e-8)
        }));
        assert!(triangles.iter().all(|f| {
            f.surface
                .unwrap()
                .uv
                .iter()
                .flatten()
                .all(|v| v.is_finite())
        }));
    }
}
