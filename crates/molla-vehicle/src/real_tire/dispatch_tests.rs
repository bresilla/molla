use super::*;
use crate::{FlatTerrain, Terrain};

macro_rules! same_bits {
    ($a:expr, $b:expr, $($field:ident),+ $(,)?) => {
        $(assert_eq!($a.$field.to_bits(), $b.$field.to_bits(), stringify!($field));)+
    };
}

pub(super) fn compare_contact(a: &RealTireContact, b: &RealTireContact) {
    assert_eq!(a.in_contact, b.in_contact);
    assert_eq!(a.contact_count, b.contact_count);
    assert_eq!(a.sliding_count, b.sliding_count);
    assert_eq!(a.force.map(f64::to_bits), b.force.map(f64::to_bits));
    same_bits!(
        a,
        b,
        normal_load,
        longitudinal_force,
        lateral_force,
        aligning_moment_z,
        camber_angle_rad,
        contact_center_x,
        force_resultant_x,
        pneumatic_trail_m,
        rolling_resistance_force,
        slip_ratio,
        slip_speed,
        lateral_slip_speed,
        slip_angle_rad,
        patch_length,
        adhesion_fraction,
        sliding_fraction,
        available_friction_budget,
        consumed_friction_budget,
        effective_longitudinal_stiffness,
        effective_cornering_stiffness,
        dissipated_power
    );
    assert_eq!(a.cells.len(), b.cells.len());
    for (a, b) in a.cells.iter().zip(&b.cells) {
        assert_eq!(a.node_index, b.node_index);
        assert_eq!(a.sliding, b.sliding);
        assert_eq!(a.normal.map(f64::to_bits), b.normal.map(f64::to_bits));
        same_bits!(
            a,
            b,
            local_x,
            local_z,
            penetration,
            area,
            normal_load,
            tangential_force,
            lateral_force
        );
    }
}

pub(super) fn compare_history(a: &TireBrushState, b: &TireBrushState) {
    assert_eq!(a.geometry, b.geometry);
    assert_eq!(a.profile.len(), b.profile.len());
    for (a, b) in a.profile.iter().zip(b.profile.iter()) {
        assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
    }
    assert_eq!(a.cells.len(), b.cells.len());
    for (a, b) in a.cells.iter().zip(&b.cells) {
        assert_eq!(
            a.displacement().map(f64::to_bits),
            b.displacement().map(f64::to_bits)
        );
    }
}

pub(super) struct MixedTerrain;

impl Terrain for MixedTerrain {
    fn height(&self, x: f64, z: f64) -> f64 {
        0.02 * x.sin() + 0.01 * z
    }
    fn normal(&self, _: f64, _: f64) -> [f64; 3] {
        [0.0, 1.0, 0.0]
    }
    fn friction(&self) -> f64 {
        1.2
    }
    fn friction_at(&self, x: f64, z: f64) -> f64 {
        if x + z < 0.0 { 0.0 } else { 0.7 }
    }
    fn surface_velocity(&self, x: f64, z: f64) -> [f64; 3] {
        [0.2 * x.cos(), 0.0, 0.1 * z]
    }
    fn water_film_depth(&self, x: f64, _: f64) -> f64 {
        if x > 0.0 { 0.002 } else { 0.0 }
    }
}

fn compare_dispatch<T: Terrain>(terrain: &T) {
    let erased: &dyn Terrain = std::hint::black_box(terrain as &dyn Terrain);
    let material = SurfaceMaterial::dry_asphalt();
    for segments in [8, 32, 128, 512] {
        for width_segments in [1, 3, 16] {
            let mut params = RealTireParams {
                segments,
                width_segments,
                ..Default::default()
            };
            let mut typed = TireBrushState::default();
            let mut dynamic = TireBrushState::default();
            for phase in 0..18 {
                params.pressure_pa = [50_000.0, 180_000.0, 400_000.0][phase % 3];
                if phase == 12 {
                    params.width *= 1.1;
                }
                let wheel = RealWheelState {
                    hub_pos: [
                        0.1 * phase as f64,
                        params.radius * if phase % 5 == 0 { 1.2 } else { 0.9 },
                        -0.02,
                    ],
                    hub_vel: [if phase % 2 == 0 { -1.0 } else { 2.0 }, 0.0, 0.15],
                    heading: [0.8, 0.0, 0.6],
                    omega: phase as f64 - 4.0,
                };
                let matrix = [[0.8, -0.6], [0.6, 0.8]];
                typed.project_frame(matrix).unwrap();
                dynamic.project_frame(matrix).unwrap();
                let a = evaluate_planar_brush_tire_persistent(
                    &params, &wheel, terrain, &material, &mut typed, 0.001,
                )
                .unwrap();
                let b = evaluate_planar_brush_tire_persistent(
                    &params,
                    &wheel,
                    erased,
                    &material,
                    &mut dynamic,
                    0.001,
                )
                .unwrap();
                compare_contact(&a, &b);
                compare_history(&typed, &dynamic);
                let before = typed.clone();
                for invalid_dt in [0.0, -1.0, f64::NAN, f64::INFINITY] {
                    assert_eq!(
                        evaluate_planar_brush_tire_persistent(
                            &params, &wheel, terrain, &material, &mut typed, invalid_dt
                        )
                        .unwrap_err(),
                        evaluate_planar_brush_tire_persistent(
                            &params,
                            &wheel,
                            erased,
                            &material,
                            &mut dynamic,
                            invalid_dt
                        )
                        .unwrap_err()
                    );
                    compare_history(&typed, &before);
                    compare_history(&dynamic, &before);
                }
            }
        }
    }
}

#[test]
fn concrete_and_erased_terrain_match_bitwise() {
    compare_dispatch(&FlatTerrain::new(0.0, 0.8));
    compare_dispatch(&MixedTerrain);
}
