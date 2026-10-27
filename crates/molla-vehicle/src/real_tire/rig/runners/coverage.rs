#[derive(Clone, Copy, Debug, Default)]
struct Integral {
    length: f64,
    load: f64,
    moment: f64,
    min: f64,
    max: f64,
}

fn positive_segment(mut x0: f64, mut p0: f64, mut x1: f64, mut p1: f64) -> Integral {
    if p0 <= 0.0 && p1 <= 0.0 {
        return Integral::default();
    }
    if p0 < 0.0 {
        x0 += (x1 - x0) * (-p0 / (p1 - p0));
        p0 = 0.0;
    } else if p1 < 0.0 {
        x1 = x0 + (x1 - x0) * (p0 / (p0 - p1));
        p1 = 0.0;
    }
    let length = (x1 - x0).abs();
    let load = length * (p0 + p1) * 0.5;
    let moment = length * (x0 * (2.0 * p0 + p1) + x1 * (p0 + 2.0 * p1)) / 6.0;
    Integral {
        length,
        load,
        moment,
        min: x0.min(x1),
        max: x0.max(x1),
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Cell {
    pub index: usize,
    pub x: f64,
    pub z: f64,
    pub penetration: f64,
    pub length: f64,
    pub min: f64,
    pub max: f64,
}

/// Integrate positive, piecewise-linear penetration over a sample's dual cell.
pub(super) fn cell(profile: &[[f64; 3]], depths: &[f64], strips: usize, i: usize) -> Option<Cell> {
    let prev = (i + profile.len() - strips) % profile.len();
    let next = (i + strips) % profile.len();
    let x = profile[i][0];
    let p = depths[i];
    let halves = [
        positive_segment((profile[prev][0] + x) * 0.5, (depths[prev] + p) * 0.5, x, p),
        positive_segment(x, p, (profile[next][0] + x) * 0.5, (depths[next] + p) * 0.5),
    ];
    let length: f64 = halves.iter().map(|h| h.length).sum();
    let load: f64 = halves.iter().map(|h| h.load).sum();
    if length <= 0.0 || load <= 0.0 {
        return None;
    }
    Some(Cell {
        index: i,
        x: halves.iter().map(|h| h.moment).sum::<f64>() / load,
        z: profile[i][2],
        penetration: load / length,
        length,
        min: halves
            .iter()
            .filter(|h| h.length > 0.0)
            .map(|h| h.min)
            .fold(f64::INFINITY, f64::min),
        max: halves
            .iter()
            .filter(|h| h.length > 0.0)
            .map(|h| h.max)
            .fold(f64::NEG_INFINITY, f64::max),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clipped_linear_pressure_integrates_area_load_and_centroid() {
        for (x0, p0, x1, p1) in [(0.0, -1.0, 2.0, 1.0), (2.0, 1.0, 0.0, -1.0)] {
            let value = positive_segment(x0, p0, x1, p1);
            assert_eq!(value.length, 1.0);
            assert_eq!(value.load, 0.5);
            assert!((value.moment / value.load - 5.0 / 3.0).abs() < 1e-14);
            assert_eq!((value.min, value.max), (1.0, 2.0));
        }
        assert_eq!(positive_segment(0.0, -1.0, 2.0, 0.0).length, 0.0);
        let full = positive_segment(-1.0, 2.0, 1.0, 2.0);
        assert_eq!((full.length, full.load, full.moment), (2.0, 4.0, 0.0));
    }

    fn patch(radius: f64, depth: f64, segments: usize, strips: usize) -> (f64, f64) {
        let profile = super::super::brush_profile(radius, 0.58, segments, strips);
        let depths: Vec<_> = profile.iter().map(|p| depth - radius - p[1]).collect();
        let mut area = 0.0;
        let mut load = 0.0;
        for i in 0..profile.len() {
            if let Some(c) = cell(&profile, &depths, strips, i) {
                assert!(c.x >= c.min - 1e-14 && c.x <= c.max + 1e-14);
                let a = c.length * 0.58 / strips as f64;
                area += a;
                load += a * c.penetration;
            }
        }
        (area, load)
    }

    #[test]
    fn subsegment_compression_grows_area_and_load_without_plateaus() {
        let mut previous = (0.0, 0.0);
        for i in 0..401 {
            let current = patch(0.88820508, 0.04 + i as f64 * 0.0001, 48, 3);
            assert!(current.0 > previous.0 && current.1 > previous.1);
            previous = current;
        }
        assert_eq!(patch(0.88820508, 0.0, 48, 3), (0.0, 0.0));
        assert_eq!(patch(0.88820508, -0.1, 48, 3), (0.0, 0.0));
    }

    #[test]
    fn refinement_converges_to_circle_plane_area_and_normal_integral() {
        let r = 0.88820508_f64;
        let depth = 0.054;
        let h = r - depth;
        let a = (r * r - h * h).sqrt();
        let exact = (2.0 * a * 0.58, (r * r * (a / r).asin() - h * a) * 0.58);
        let mut previous_error = (f64::INFINITY, f64::INFINITY);
        for segments in [16, 32, 64, 128, 256, 512] {
            let value = patch(r, depth, segments, 3);
            let error = ((value.0 - exact.0).abs(), (value.1 - exact.1).abs());
            assert!(error.0 < previous_error.0 && error.1 < previous_error.1);
            previous_error = error;
        }
        assert!(previous_error.0 < exact.0 * 0.001);
        assert!(previous_error.1 < exact.1 * 0.001);
        for strips in [1, 2, 3, 8, 16] {
            let value = patch(r, depth, 48, strips);
            let base = patch(r, depth, 48, 1);
            assert!((value.0 - base.0).abs() < 1e-12);
            assert!((value.1 - base.1).abs() < 1e-12);
        }
    }

    #[test]
    fn area_and_load_are_continuous_when_sample_enters_contact() {
        let r = 0.88820508;
        for i in 1..8 {
            let depth = r * (1.0 - (i as f64 * std::f64::consts::TAU / 48.0).cos());
            let before = patch(r, depth - 1e-9, 48, 3);
            let after = patch(r, depth + 1e-9, 48, 3);
            assert!(after.0 > before.0 && after.1 > before.1);
            assert!(after.0 - before.0 < 1e-7);
            assert!(after.1 - before.1 < 1e-8);
        }
    }

    #[test]
    fn mixed_slip_moment_is_odd_under_lateral_reversal() {
        use crate::FlatTerrain;
        use crate::real_tire::{RealTireParams, SurfaceMaterial, tire_rig_point};

        let material = SurfaceMaterial {
            rolling_resistance: 0.0,
            cohesion: 0.0,
            ..SurfaceMaterial::dry_asphalt()
        };
        let terrain = FlatTerrain::new(0.0, material.mu_static);
        for segments in [64, 128, 256, 512] {
            let params = RealTireParams::passenger_radial().with_segments(segments);
            let mut moments = Vec::new();
            for angle in [0.01, 0.02, 0.035, 0.05, 0.07, 0.095, 0.125, 0.16] {
                let positive =
                    tire_rig_point(&params, &terrain, &material, 4200.0, 11.0, 0.08, angle, 0.0)
                        .telemetry;
                let negative = tire_rig_point(
                    &params, &terrain, &material, 4200.0, 11.0, 0.08, -angle, 0.0,
                )
                .telemetry;
                assert!((positive.normal_load_n - 4200.0).abs() < 1e-6);
                assert!((negative.normal_load_n - 4200.0).abs() < 1e-6);
                assert!(
                    (positive.longitudinal_force_n - negative.longitudinal_force_n).abs() < 1e-8
                );
                assert!((positive.lateral_force_n + negative.lateral_force_n).abs() < 1e-8);
                assert!((positive.aligning_moment_nm + negative.aligning_moment_nm).abs() < 1e-8);
                assert!(
                    positive.consumed_friction_budget_n
                        <= positive.available_friction_budget_n + 1e-8
                );
                moments.push(positive.aligning_moment_nm);
            }
            eprintln!("partial coverage mixed-slip moments segments={segments}: {moments:?}");
        }
    }

    #[test]
    fn static_dynamic_friction_gap_controls_moment_reversal_not_footprint() {
        use crate::FlatTerrain;
        use crate::real_tire::{RealTireParams, SurfaceMaterial, tire_rig_point};

        let material = SurfaceMaterial {
            rolling_resistance: 0.0,
            cohesion: 0.0,
            ..SurfaceMaterial::dry_asphalt()
        };
        let equal = SurfaceMaterial {
            mu_dynamic: material.mu_static,
            ..material
        };
        let terrain = FlatTerrain::new(0.0, material.mu_static);
        for segments in [64, 128, 256, 512] {
            let params = RealTireParams::passenger_radial().with_segments(segments);
            let mut has_positive = false;
            let mut has_negative = false;
            for angle in [0.01, 0.02, 0.035, 0.05, 0.07, 0.095, 0.125, 0.16] {
                let variable =
                    tire_rig_point(&params, &terrain, &material, 4200.0, 11.0, 0.08, angle, 0.0)
                        .telemetry;
                let constant =
                    tire_rig_point(&params, &terrain, &equal, 4200.0, 11.0, 0.08, angle, 0.0)
                        .telemetry;
                assert_eq!(variable.normal_load_n, constant.normal_load_n);
                assert_eq!(
                    variable.patch.contact_area_m2,
                    constant.patch.contact_area_m2
                );
                assert_eq!(variable.patch.patch_length_m, constant.patch.patch_length_m);
                assert!(
                    constant.aligning_moment_nm < 0.0,
                    "constant-friction moment {} at {angle}, segments={segments}",
                    constant.aligning_moment_nm
                );
                has_positive |= variable.aligning_moment_nm > 0.1;
                has_negative |= variable.aligning_moment_nm < -0.1;
            }
            assert!(has_positive && has_negative);
        }
    }

    #[test]
    fn matched_load_footprint_obeys_inscribed_polygon_and_circle_bounds() {
        use crate::FlatTerrain;
        use crate::real_tire::{RealTireParams, SurfaceMaterial, tire_rig_point};

        let material = SurfaceMaterial::dry_asphalt();
        let terrain = FlatTerrain::new(0.0, material.mu_static);
        for segments in [16, 48, 128, 512] {
            for pressure_pa in [120_000.0, 220_000.0, 320_000.0] {
                let params = RealTireParams {
                    segments,
                    pressure_pa,
                    ..Default::default()
                };
                for load in [1000.0, 3000.0, 5000.0, 8000.0] {
                    let point =
                        tire_rig_point(&params, &terrain, &material, load, 0.0, 0.0, 0.0, 0.0);
                    let h = point.hub_height_m;
                    let inner = params.radius * (std::f64::consts::PI / segments as f64).cos();
                    let lower = 2.0 * (inner * inner - h * h).max(0.0).sqrt();
                    let upper = 2.0 * (params.radius * params.radius - h * h).max(0.0).sqrt();
                    let patch = point.telemetry.patch;
                    assert!(patch.patch_length_m >= lower - 1e-12);
                    assert!(patch.patch_length_m <= upper + 1e-12);
                    assert!(
                        (patch.contact_area_m2 - params.width * patch.patch_length_m).abs() < 1e-12
                    );
                    assert!((point.telemetry.normal_load_n - load).abs() < 1e-6);
                }
            }
        }
    }
}
