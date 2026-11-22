//! Loader for the superseded Innerscene quadcopter visualization.
use crate::draw::Triangle;
use eframe::egui::Color32;
use molla_math::{Quat, Vec3};
use molla_solvers::airflow::Quadrotor;
use serde_json::Value;
use std::sync::OnceLock;

const ASSET: &[u8] = include_bytes!("../data/drone/quadcopter-original.glb");

struct Part {
    triangles: Vec<[Vec3; 3]>,
    color: Color32,
    rotor: Option<usize>,
    double_sided: bool,
}

fn word(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn vector(value: &Value, default: Vec3) -> Vec3 {
    value.as_array().map_or(default, |v| {
        Vec3::new(
            v[0].as_f64().unwrap(),
            v[1].as_f64().unwrap(),
            v[2].as_f64().unwrap(),
        )
    })
}

fn pivot(rotor: usize) -> Vec3 {
    Vec3::new(
        if rotor < 2 { -0.22 } else { 0.22 },
        if rotor % 2 == 0 { -0.22 } else { 0.22 },
        0.0,
    )
}

fn load() -> Vec<Part> {
    assert_eq!(&ASSET[..4], b"glTF");
    assert_eq!(word(ASSET, 4), 2);
    assert_eq!(word(ASSET, 8) as usize, ASSET.len());
    assert_eq!(&ASSET[16..20], b"JSON");
    let json_end = 20 + word(ASSET, 12) as usize;
    let document: Value = serde_json::from_slice(&ASSET[20..json_end]).unwrap();
    assert_eq!(&ASSET[json_end + 4..json_end + 8], b"BIN\0");
    let binary = &ASSET[json_end + 8..];
    assert!(document.get("extensionsRequired").is_none());
    let accessor = |index: usize, element_size: usize| {
        let a = &document["accessors"][index];
        let view = &document["bufferViews"][a["bufferView"].as_u64().unwrap() as usize];
        assert_eq!(view["buffer"].as_u64().unwrap(), 0);
        assert!(a.get("sparse").is_none());
        let offset = (view["byteOffset"].as_u64().unwrap_or(0)
            + a["byteOffset"].as_u64().unwrap_or(0)) as usize;
        let stride = view["byteStride"].as_u64().unwrap_or(element_size as u64) as usize;
        (a, offset, stride)
    };
    let mut parts = Vec::new();
    for node in document["nodes"].as_array().unwrap() {
        assert!(node.get("children").is_none() && node.get("matrix").is_none());
        let translation = vector(&node["translation"], Vec3::ZERO);
        let scale = vector(&node["scale"], Vec3::ONE);
        let rotation = node["rotation"].as_array().map_or(Quat::IDENTITY, |v| {
            Quat::from_array(std::array::from_fn(|i| v[i].as_f64().unwrap())).normalize()
        });
        let name = node["name"].as_str().unwrap();
        for primitive in document["meshes"][node["mesh"].as_u64().unwrap() as usize]["primitives"]
            .as_array()
            .unwrap()
        {
            assert_eq!(primitive["mode"].as_u64().unwrap_or(4), 4);
            let (a, offset, stride) = accessor(
                primitive["attributes"]["POSITION"].as_u64().unwrap() as usize,
                12,
            );
            assert_eq!(a["componentType"], 5126);
            assert_eq!(a["type"], "VEC3");
            let points: Vec<_> = (0..a["count"].as_u64().unwrap() as usize)
                .map(|i| {
                    let p = Vec3::from_array(std::array::from_fn(|axis| {
                        f32::from_bits(word(binary, offset + i * stride + axis * 4)) as f64
                    }));
                    let p = translation + rotation * (p * scale);
                    Vec3::new(p.x, -p.z, p.y - 0.231)
                })
                .collect();
            let index_id = primitive["indices"].as_u64().unwrap() as usize;
            let component = document["accessors"][index_id]["componentType"]
                .as_u64()
                .unwrap();
            assert!(component == 5123 || component == 5125);
            let (a, offset, stride) = accessor(index_id, if component == 5123 { 2 } else { 4 });
            assert_eq!(a["type"], "SCALAR");
            let count = a["count"].as_u64().unwrap() as usize;
            assert_eq!(count % 3, 0);
            let index = |i: usize| {
                let at = offset + i * stride;
                if component == 5123 {
                    u16::from_le_bytes(binary[at..at + 2].try_into().unwrap()) as usize
                } else {
                    word(binary, at) as usize
                }
            };
            let triangles = (0..count / 3)
                .map(|i| std::array::from_fn(|j| points[index(i * 3 + j)]))
                .collect();
            let material = &document["materials"][primitive["material"].as_u64().unwrap() as usize];
            let rgb = vector(
                &material["pbrMetallicRoughness"]["baseColorFactor"],
                Vec3::ONE,
            );
            let channel = |x: f64| (x.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
            let center = points.iter().copied().sum::<Vec3>() / points.len() as f64;
            let rotor = (name.starts_with("Rotor") || name.starts_with("Motor bell"))
                .then_some(usize::from(center.x >= 0.0) * 2 + usize::from(center.y >= 0.0));
            parts.push(Part {
                triangles,
                color: Color32::from_rgb(channel(rgb.x), channel(rgb.y), channel(rgb.z)),
                rotor,
                double_sided: material["doubleSided"].as_bool().unwrap_or(false),
            });
        }
    }
    parts
}

fn parts() -> &'static [Part] {
    static PARTS: OnceLock<Vec<Part>> = OnceLock::new();
    PARTS.get_or_init(load)
}

pub fn mesh(drone: &Quadrotor, time: f64) -> Vec<Triangle> {
    let pose = drone.pose().unwrap();
    let scale = drone.config.arm / 0.22;
    let rotations: [Quat; 4] = std::array::from_fn(|i| {
        let omega = drone
            .world
            .devices
            .propellers
            .reading(drone.propellers[i])
            .unwrap()
            .omega;
        Quat::from_rotation_z(time * omega)
    });
    let mut out = Vec::with_capacity(10072);
    for part in parts() {
        for triangle in &part.triangles {
            let p = triangle.map(|mut p| {
                if let Some(i) = part.rotor {
                    let mut offset = p - pivot(i);
                    let radial_scale = drone.config.rotor_radius / (0.124 * scale);
                    offset.x *= radial_scale;
                    offset.y *= radial_scale;
                    p = pivot(i) + rotations[i] * offset;
                }
                pose.transform_point(p * scale)
            });
            out.push(Triangle {
                p,
                color: part.color,
                material: 0.0,
                surface: None,
            });
            if part.double_sided {
                out.push(Triangle {
                    p: [p[2], p[1], p[0]],
                    color: part.color,
                    material: 0.0,
                    surface: None,
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bundled_drone_preserves_materials_and_four_separate_rotors() {
        let parts = parts();
        assert_eq!(parts.iter().map(|p| p.triangles.len()).sum::<usize>(), 5036);
        for i in 0..4 {
            assert!(parts.iter().filter(|p| p.rotor == Some(i)).count() >= 4);
        }
        assert!(
            parts
                .iter()
                .all(|p| p.triangles.iter().flatten().all(|v| v.is_finite()))
        );
        let colors: std::collections::BTreeSet<_> =
            parts.iter().map(|p| p.color.to_array()).collect();
        assert!(colors.len() >= 6);
        let drone = Quadrotor::new(Vec3::Y * 3.0, Default::default()).unwrap();
        let mesh = mesh(&drone, 0.0);
        assert!(!mesh.is_empty());
        assert!(
            mesh.iter()
                .flat_map(|t| t.p)
                .all(|p| p.is_finite() && p.y > 2.5 && p.y < 3.5)
        );
    }
}
