use crate::{
    draw::{Surface, Triangle},
    orchard_spray::nozzle_local,
};
use eframe::egui::Color32;
use molla_math::{Quat, Vec3};
use molla_solvers::airflow::Quadrotor;
use std::sync::OnceLock;

const FRAME: Color32 = Color32::from_rgb(43, 48, 52);
const SHELL: Color32 = Color32::from_rgb(116, 125, 124);
const TANK: Color32 = Color32::from_rgb(219, 223, 211);
const RUBBER: Color32 = Color32::from_rgb(22, 27, 29);
const METAL: Color32 = Color32::from_rgb(142, 153, 161);
const SAFETY: Color32 = Color32::from_rgb(218, 145, 43);
const HOSE: Color32 = Color32::from_rgb(38, 104, 115);
const GLASS: Color32 = Color32::from_rgb(22, 44, 54);

fn face(out: &mut Vec<Triangle>, p: [Vec3; 3], color: Color32, normals: Option<[Vec3; 3]>) {
    out.push(Triangle {
        p,
        color,
        material: if color == METAL || color == GLASS {
            12.0
        } else if color == TANK {
            11.0
        } else {
            10.0
        },
        surface: normals.map(|normals| Surface {
            normals,
            uv: [[0.0; 2]; 3],
        }),
    });
}

fn cylinder(out: &mut Vec<Triangle>, a: Vec3, b: Vec3, radius: f64, color: Color32) {
    let axis = (b - a).normalize();
    let u = axis
        .cross(if axis.z.abs() < 0.9 { Vec3::Z } else { Vec3::X })
        .normalize();
    let v = axis.cross(u);
    let sides = if radius < 0.006 { 8 } else { 16 };
    for i in 0..sides {
        let n = |j: usize| {
            let t = j as f64 * std::f64::consts::TAU / sides as f64;
            u * t.cos() + v * t.sin()
        };
        let (n0, n1) = (n(i), n(i + 1));
        let (p, q) = (n0 * radius, n1 * radius);
        face(out, [a + p, a + q, b + q], color, Some([n0, n1, n1]));
        face(out, [a + p, b + q, b + p], color, Some([n0, n1, n0]));
        face(out, [a, a + q, a + p], color, Some([-axis; 3]));
        face(out, [b, b + p, b + q], color, Some([axis; 3]));
    }
}

fn pipe(out: &mut Vec<Triangle>, points: &[Vec3], radius: f64, color: Color32) {
    for pair in points.windows(2) {
        cylinder(out, pair[0], pair[1], radius, color);
    }
}

fn loft(out: &mut Vec<Triangle>, levels: &[(f64, f64, f64)], center: Vec3, color: Color32) {
    loft_sides(out, levels, center, color, 32);
}

fn loft_sides(
    out: &mut Vec<Triangle>,
    levels: &[(f64, f64, f64)],
    center: Vec3,
    color: Color32,
    sides: usize,
) {
    let ring = |j: usize, level: (f64, f64, f64)| {
        let t = j as f64 * std::f64::consts::TAU / sides as f64;
        let rounded = |x: f64| x.signum() * x.abs().powf(0.42);
        center
            + Vec3::new(
                rounded(t.cos()) * level.1,
                rounded(t.sin()) * level.2,
                level.0,
            )
    };
    for pair in levels.windows(2) {
        let normal = |j: usize, level| {
            let tangent = ring((j + 1) % sides, level) - ring((j + sides - 1) % sides, level);
            tangent
                .cross(ring(j, pair[1]) - ring(j, pair[0]))
                .normalize()
        };
        for j in 0..sides {
            let k = (j + 1) % sides;
            let p = [
                ring(j, pair[0]),
                ring(k, pair[0]),
                ring(k, pair[1]),
                ring(j, pair[1]),
            ];
            let n = [
                normal(j, pair[0]),
                normal(k, pair[0]),
                normal(k, pair[1]),
                normal(j, pair[1]),
            ];
            face(out, [p[0], p[1], p[2]], color, Some([n[0], n[1], n[2]]));
            face(out, [p[0], p[2], p[3]], color, Some([n[0], n[2], n[3]]));
        }
    }
    for (end, flip) in [(levels[0], true), (*levels.last().unwrap(), false)] {
        for j in 0..sides {
            let p = [
                center + Vec3::Z * end.0,
                ring(j, end),
                ring((j + 1) % sides, end),
            ];
            face(
                out,
                if flip { [p[0], p[2], p[1]] } else { p },
                color,
                Some([if flip { -Vec3::Z } else { Vec3::Z }; 3]),
            );
        }
    }
}

fn housing(out: &mut Vec<Triangle>, center: Vec3, half: Vec3, color: Color32) {
    let bevel = 0.006_f64.min(half.min_element() * 0.35);
    loft_sides(
        out,
        &[
            (-half.z, half.x - bevel, half.y - bevel),
            (-half.z + bevel, half.x, half.y),
            (half.z - bevel, half.x, half.y),
            (half.z, half.x - bevel, half.y - bevel),
        ],
        center,
        color,
        16,
    );
}

fn bolt(out: &mut Vec<Triangle>, p: Vec3, axis: Vec3) {
    cylinder(out, p, p + axis * 0.003, 0.004, METAL);
}

fn body() -> &'static [Triangle] {
    static BODY: OnceLock<Vec<Triangle>> = OnceLock::new();
    BODY.get_or_init(|| {
        let mut out = Vec::new();
        loft(
            &mut out,
            &[
                (-0.255, 0.090, 0.092),
                (-0.233, 0.137, 0.126),
                (-0.105, 0.158, 0.138),
                (-0.062, 0.143, 0.127),
            ],
            Vec3::new(-0.025, 0.0, 0.0),
            TANK,
        );
        housing(
            &mut out,
            Vec3::new(0.0, 0.0, -0.065),
            Vec3::new(0.213, 0.157, 0.022),
            FRAME,
        );
        housing(
            &mut out,
            Vec3::new(0.028, 0.0, -0.037),
            Vec3::new(0.184, 0.134, 0.014),
            SHELL,
        );
        housing(
            &mut out,
            Vec3::new(0.033, 0.0, 0.005),
            Vec3::new(0.104, 0.094, 0.034),
            FRAME,
        );
        housing(
            &mut out,
            Vec3::new(0.033, 0.0, 0.041),
            Vec3::new(0.101, 0.091, 0.004),
            RUBBER,
        );
        pipe(
            &mut out,
            &[
                Vec3::new(-0.015, -0.052, 0.046),
                Vec3::new(-0.015, -0.052, 0.063),
                Vec3::new(-0.015, 0.052, 0.063),
                Vec3::new(-0.015, 0.052, 0.046),
            ],
            0.006,
            SHELL,
        );
        for s in [-1.0, 1.0] {
            for x in [-0.050, 0.100] {
                housing(
                    &mut out,
                    Vec3::new(x, s * 0.096, 0.011),
                    Vec3::new(0.009, 0.005, 0.018),
                    SHELL,
                );
            }
            for i in 0..6 {
                housing(
                    &mut out,
                    Vec3::new(-0.032 + i as f64 * 0.023, s * 0.092, 0.032),
                    Vec3::new(0.006, 0.003, 0.008),
                    RUBBER,
                );
            }
            for x in [-0.110, 0.071] {
                pipe(
                    &mut out,
                    &[
                        Vec3::new(x, s * 0.129, -0.086),
                        Vec3::new(x, s * 0.139, -0.113),
                        Vec3::new(x, s * 0.130, -0.193),
                        Vec3::new(x, s * 0.121, -0.232),
                    ],
                    0.004,
                    TANK,
                );
            }
            for x in [-0.155, 0.155] {
                let top = Vec3::new(x, s * 0.140, -0.069);
                let foot = Vec3::new(x * 1.39, s * 0.210, -0.365);
                cylinder(&mut out, top, foot, 0.0105, FRAME);
                cylinder(&mut out, top, top.lerp(foot, 0.19), 0.014, SHELL);
                cylinder(
                    &mut out,
                    top.lerp(foot, 0.78),
                    top.lerp(foot, 0.84),
                    0.0125,
                    RUBBER,
                );
                bolt(&mut out, top + Vec3::new(0.0, s * 0.015, 0.0), Vec3::Y * s);
                housing(
                    &mut out,
                    Vec3::new(x * 1.39, s * 0.210, -0.378),
                    Vec3::new(0.027, 0.016, 0.009),
                    RUBBER,
                );
            }
            pipe(
                &mut out,
                &[
                    Vec3::new(-0.261, s * 0.210, -0.335),
                    Vec3::new(-0.253, s * 0.210, -0.359),
                    Vec3::new(-0.238, s * 0.210, -0.373),
                    Vec3::new(0.238, s * 0.210, -0.373),
                    Vec3::new(0.253, s * 0.210, -0.359),
                    Vec3::new(0.261, s * 0.210, -0.335),
                ],
                0.011,
                FRAME,
            );
            housing(
                &mut out,
                Vec3::new(0.15, s * 0.086, -0.111),
                Vec3::new(0.034, 0.029, 0.035),
                SHELL,
            );
            for i in 0..4 {
                housing(
                    &mut out,
                    Vec3::new(0.186, s * 0.086, -0.13 + i as f64 * 0.012),
                    Vec3::new(0.003, 0.022, 0.002),
                    FRAME,
                );
            }
            cylinder(
                &mut out,
                Vec3::new(-0.185, s * 0.100, -0.024),
                Vec3::new(-0.185, s * 0.100, 0.030),
                0.005,
                FRAME,
            );
            cylinder(
                &mut out,
                Vec3::new(-0.185, s * 0.100, 0.025),
                Vec3::new(-0.185, s * 0.100, 0.048),
                0.010,
                SHELL,
            );
        }
        cylinder(
            &mut out,
            Vec3::new(-0.150, 0.0, -0.041),
            Vec3::new(-0.150, 0.0, -0.016),
            0.033,
            TANK,
        );
        cylinder(
            &mut out,
            Vec3::new(-0.150, 0.0, -0.016),
            Vec3::new(-0.150, 0.0, -0.004),
            0.038,
            RUBBER,
        );
        for i in 0..12 {
            let t = i as f64 * std::f64::consts::TAU / 12.0;
            let p = Vec3::new(-0.150 + 0.038 * t.cos(), 0.038 * t.sin(), -0.014);
            cylinder(&mut out, p, p + Vec3::Z * 0.009, 0.002, SHELL);
        }
        housing(
            &mut out,
            Vec3::new(0.200, 0.0, -0.047),
            Vec3::new(0.022, 0.055, 0.024),
            SHELL,
        );
        housing(
            &mut out,
            Vec3::new(0.223, 0.0, -0.044),
            Vec3::new(0.003, 0.042, 0.013),
            RUBBER,
        );
        cylinder(
            &mut out,
            Vec3::new(0.223, 0.0, -0.044),
            Vec3::new(0.228, 0.0, -0.044),
            0.010,
            GLASS,
        );
        housing(
            &mut out,
            Vec3::new(0.115, 0.0, -0.197),
            Vec3::new(0.034, 0.050, 0.018),
            FRAME,
        );
        housing(
            &mut out,
            Vec3::new(0.145, 0.0, -0.199),
            Vec3::new(0.006, 0.041, 0.012),
            SHELL,
        );
        let corners = [0, 1, 2, 3].map(|i| nozzle_local(i) + Vec3::Z * 0.055);
        for i in 0..4 {
            cylinder(&mut out, corners[i], corners[(i + 1) % 4], 0.007, FRAME);
        }
        for s in [-1.0, 1.0] {
            let side = s * crate::orchard_spray::BOOM_HALF_SIDE_M;
            pipe(
                &mut out,
                &[
                    Vec3::new(0.149, s * 0.086, -0.140),
                    Vec3::new(0.157, s * 0.120, -0.195),
                    Vec3::new(0.117, s * 0.157, -0.258),
                    Vec3::new(0.04, side, -0.285),
                ],
                0.0045,
                HOSE,
            );
            for x in [-0.15, 0.15] {
                cylinder(
                    &mut out,
                    Vec3::new(x, side, -0.285),
                    Vec3::new(x, s * 0.14, -0.108),
                    0.005,
                    FRAME,
                );
            }
        }
        for i in 0..4 {
            let p = nozzle_local(i);
            let feed = (nozzle_local((i + 1) % 4) - p).normalize();
            cylinder(
                &mut out,
                p + Vec3::Z * 0.061,
                p + Vec3::Z * 0.042,
                0.012,
                SHELL,
            );
            cylinder(
                &mut out,
                p + Vec3::Z * 0.042,
                p + Vec3::Z * 0.016,
                0.009,
                FRAME,
            );
            cylinder(
                &mut out,
                p + Vec3::Z * 0.016,
                p + Vec3::Z * 0.008,
                0.013,
                SAFETY,
            );
            cylinder(&mut out, p + Vec3::Z * 0.008, p, 0.006, METAL);
            pipe(
                &mut out,
                &[
                    p + feed * 0.036 + Vec3::Z * 0.055,
                    p + feed * 0.018 + Vec3::Z * 0.048,
                    p + feed * 0.010 + Vec3::Z * 0.025,
                ],
                0.003,
                HOSE,
            );
        }
        out
    })
}

fn hub(drone: &Quadrotor, i: usize) -> Vec3 {
    Vec3::new(
        if i < 2 { -1.0 } else { 1.0 },
        if i % 2 == 0 { -1.0 } else { 1.0 },
        0.0,
    ) * drone.config.arm
}

fn arm(out: &mut Vec<Triangle>, hub: Vec3) {
    let shoulder = Vec3::new(hub.x.signum() * 0.162, hub.y.signum() * 0.123, -0.060);
    let tip = hub - Vec3::Z * 0.059;
    let axis = (tip - shoulder).normalize();
    cylinder(out, shoulder, tip, 0.016, FRAME);
    cylinder(out, shoulder, shoulder + axis * 0.061, 0.024, SHELL);
    cylinder(
        out,
        shoulder + axis * 0.060,
        shoulder + axis * 0.069,
        0.022,
        RUBBER,
    );
    cylinder(
        out,
        shoulder - Vec3::Z * 0.025,
        shoulder + Vec3::Z * 0.026,
        0.010,
        METAL,
    );
    housing(
        out,
        shoulder + Vec3::Z * 0.026,
        Vec3::new(0.027, 0.019, 0.005),
        FRAME,
    );
    housing(
        out,
        shoulder + axis * 0.037 + Vec3::Z * 0.024,
        Vec3::new(0.014, 0.010, 0.003),
        SAFETY,
    );
    cylinder(
        out,
        shoulder + axis * 0.072 - Vec3::Z * 0.017,
        tip - Vec3::Z * 0.017,
        0.0035,
        RUBBER,
    );
    cylinder(out, tip - axis * 0.044, tip + axis * 0.019, 0.021, SHELL);
    loft(
        out,
        &[
            (-0.095, 0.025, 0.028),
            (-0.075, 0.039, 0.039),
            (-0.042, 0.043, 0.043),
        ],
        hub,
        FRAME,
    );
    cylinder(
        out,
        hub - Vec3::Z * 0.042,
        hub - Vec3::Z * 0.034,
        0.047,
        METAL,
    );
    cylinder(
        out,
        hub - Vec3::Z * 0.034,
        hub - Vec3::Z * 0.012,
        0.045,
        FRAME,
    );
    cylinder(
        out,
        hub - Vec3::Z * 0.012,
        hub - Vec3::Z * 0.006,
        0.048,
        SHELL,
    );
    for j in 0..12 {
        let a = j as f64 * std::f64::consts::TAU / 12.0;
        let p = hub + Vec3::new(a.cos(), a.sin(), 0.0) * 0.045;
        cylinder(out, p - Vec3::Z * 0.032, p - Vec3::Z * 0.016, 0.0015, METAL);
    }
    cylinder(
        out,
        hub - Vec3::Z * 0.006,
        hub + Vec3::Z * 0.008,
        0.018,
        METAL,
    );
    for s in [-1.0, 1.0] {
        bolt(out, hub + Vec3::new(0.011 * s, 0.0, 0.008), Vec3::Z);
    }
}

fn blades(out: &mut Vec<Triangle>, hub: Vec3, phase: f64, radius: f64, spin: f64) {
    for side in [0.0, std::f64::consts::PI] {
        let rotation = Quat::from_rotation_z(phase + side);
        for i in 0..20 {
            let point = |j: usize, s: f64, skin: f64| {
                let t = j as f64 / 20.0;
                let r = radius * (0.11 + 0.89 * t);
                let width =
                    radius * (0.095 * (std::f64::consts::PI * t).sin() + 0.018 * (1.0 - t) + 0.004);
                let chord = radius * 0.055 * t * t + s * width;
                let x = (r * r - chord * chord).max(0.0).sqrt();
                hub + rotation
                    * Vec3::new(
                        x,
                        chord * spin,
                        s * width * (0.36 - 0.24 * t) + skin * 0.0009,
                    )
            };
            let color = if i == 19 { SAFETY } else { FRAME };
            for skin in [-1.0, 1.0] {
                let p = [
                    point(i, -1.0, skin),
                    point(i, 1.0, skin),
                    point(i + 1, 1.0, skin),
                    point(i + 1, -1.0, skin),
                ];
                let orders = if skin * spin > 0.0 {
                    [[2, 1, 0], [3, 2, 0]]
                } else {
                    [[0, 1, 2], [0, 2, 3]]
                };
                for ids in orders {
                    face(out, ids.map(|j| p[j]), color, None);
                }
            }
            for s in [-1.0, 1.0] {
                let p = [
                    point(i, s, -1.0),
                    point(i + 1, s, -1.0),
                    point(i + 1, s, 1.0),
                    point(i, s, 1.0),
                ];
                for ids in [[0, 1, 2], [0, 2, 3]] {
                    face(out, ids.map(|j| p[j]), color, None);
                }
            }
        }
    }
}

pub fn mesh(drone: &Quadrotor) -> Vec<Triangle> {
    static AIRFRAME: OnceLock<(f64, Vec<Triangle>)> = OnceLock::new();
    let copy = |t: &Triangle| Triangle {
        p: t.p,
        color: t.color,
        material: t.material,
        surface: t.surface,
    };
    let assemble = || {
        let mut out: Vec<_> = body().iter().map(copy).collect();
        for i in 0..4 {
            arm(&mut out, hub(drone, i));
        }
        out
    };
    let (arm_length, cached) = AIRFRAME.get_or_init(|| (drone.config.arm, assemble()));
    let mut out = if *arm_length == drone.config.arm {
        cached.iter().map(copy).collect()
    } else {
        assemble()
    };
    for i in 0..4 {
        let hub = hub(drone, i);
        let phase = drone
            .world
            .devices
            .propellers
            .reading(drone.propellers[i])
            .unwrap()
            .phase;
        blades(
            &mut out,
            hub,
            phase,
            drone.config.rotor_radius,
            if i == 0 || i == 3 { 1.0 } else { -1.0 },
        );
    }
    let pose = drone.pose().unwrap();
    for t in &mut out {
        t.p = t.p.map(|p| pose.transform_point(p));
        if let Some(s) = &mut t.surface {
            s.normals = s.normals.map(|n| pose.rotation * n);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ag_model_has_finite_geometry_and_downward_nozzles_with_clearance() {
        let drone = Quadrotor::new(Vec3::Y * 3.6, Default::default()).unwrap();
        let mesh = mesh(&drone);
        assert!(mesh.len() < 25_000, "{} triangles", mesh.len());
        for t in &mesh {
            assert!(t.p.iter().all(|p| p.is_finite() && p.y > 3.2));
            assert!((t.p[1] - t.p[0]).cross(t.p[2] - t.p[0]).length() > 1e-12);
            if let Some(s) = t.surface {
                assert!(
                    s.normals
                        .iter()
                        .all(|n| n.is_finite() && (n.length() - 1.0).abs() < 1e-9)
                );
            }
        }
        for i in 0..4 {
            let outlet = drone.pose().unwrap().transform_point(nozzle_local(i));
            assert!((outlet.y - 3.26).abs() < 1e-8);
            assert!(
                mesh.iter()
                    .flat_map(|t| t.p)
                    .any(|p| p.distance(outlet) < 1e-9)
            );
        }
        let ymin = mesh
            .iter()
            .flat_map(|t| t.p)
            .map(|p| p.y)
            .fold(f64::INFINITY, f64::min);
        let ymax = mesh
            .iter()
            .flat_map(|t| t.p)
            .map(|p| p.y)
            .fold(f64::NEG_INFINITY, f64::max);
        eprintln!(
            "AG mesh: {} triangles, height {:.3} m, swept width {:.3} m",
            mesh.len(),
            ymax - ymin,
            2.0 * (drone.config.arm + drone.config.rotor_radius)
        );
        assert!((ymax - ymin) / (2.0 * (drone.config.arm + drone.config.rotor_radius)) < 0.33);
    }

    #[test]
    fn cached_airframe_tracks_changed_physical_arm_lengths() {
        for length in [0.42, 0.53, 0.42] {
            let drone = Quadrotor::new(
                Vec3::Y * 3.6,
                molla_solvers::airflow::QuadrotorConfig {
                    arm: length,
                    ..Default::default()
                },
            )
            .unwrap();
            let triangles = mesh(&drone);
            for i in 0..4 {
                let motor_center = drone
                    .pose()
                    .unwrap()
                    .transform_point(hub(&drone, i) - Vec3::Z * 0.034);
                assert!(
                    triangles
                        .iter()
                        .flat_map(|t| t.p)
                        .any(|p| p.distance(motor_center) < 1e-9)
                );
            }
        }
    }

    #[test]
    fn visual_blades_match_physical_rotor_disks_at_every_phase() {
        let drone = Quadrotor::new(Vec3::Y * 3.6, Default::default()).unwrap();
        let pose = drone.pose().unwrap();
        let loads = drone.air_loads().unwrap();
        for (i, load) in loads.iter().enumerate() {
            assert!(
                pose.transform_point(hub(&drone, i))
                    .distance(load.disk.center)
                    < 1e-9
            );
            for phase in [0.0, 0.71, 2.5] {
                let mut triangles = Vec::new();
                blades(
                    &mut triangles,
                    hub(&drone, i),
                    phase,
                    load.disk.radius,
                    if i == 0 || i == 3 { 1.0 } else { -1.0 },
                );
                let mut radius = 0.0_f64;
                for p in triangles.iter().flat_map(|t| t.p) {
                    let d = p - hub(&drone, i);
                    radius = radius.max(d.x.hypot(d.y));
                    assert!(d.x.hypot(d.y) <= load.disk.radius + 1e-10);
                    assert!(d.z.abs() < 0.012);
                }
                assert!((radius - load.disk.radius).abs() < 1e-9);
            }
        }
    }
}
