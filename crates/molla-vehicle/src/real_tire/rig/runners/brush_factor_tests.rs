use super::*;
use crate::FlatTerrain;
use crate::real_tire::dispatch_tests::{MixedTerrain, compare_contact, compare_history};

#[test]
fn repeated_and_changed_inputs_match_fresh_factors_bitwise() {
    let pressures = [
        0.0,
        -0.0,
        1e-310,
        -1.0,
        50_000.0,
        180_000.0,
        400_000.0,
        f64::MAX,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    for inflation in pressures {
        let mut cache = BrushFactors::<true>::new(inflation);
        for normal in pressures.into_iter().flat_map(|v| [v, v, v, 220_000.0, v]) {
            let expected = brush_load_sensitivity(normal);
            let actual = cache.load(normal);
            assert_eq!(actual.0.to_bits(), expected.0.to_bits());
            assert_eq!(actual.1.to_bits(), expected.1.to_bits());
            assert_eq!(
                cache.inflation().to_bits(),
                brush_pressure_stiffness_factor(inflation).to_bits()
            );
        }
    }
}

fn exercise<T: Terrain>(terrain: &T) {
    let material = SurfaceMaterial::dry_asphalt();
    for segments in [8, 128, 512] {
        for width_segments in [1, 3, 16] {
            let mut params = RealTireParams {
                segments,
                width_segments,
                ..Default::default()
            };
            let mut cached = TireBrushState::default();
            let mut reference = TireBrushState::default();
            for step in 0..18 {
                params.pressure_pa = [50_000.0, 180_000.0, 400_000.0][step % 3];
                if step == 12 {
                    params.radius *= 1.1;
                }
                let wheel = RealWheelState {
                    hub_pos: [
                        0.03 * step as f64,
                        params.radius * if step % 5 == 0 { 1.2 } else { 0.85 },
                        0.0,
                    ],
                    hub_vel: [step as f64 * 0.3 - 2.0, 0.01, -0.2],
                    heading: [0.8, 0.0, 0.6],
                    omega: step as f64 - 6.0,
                };
                cached.project_frame([[0.8, -0.6], [0.6, 0.8]]).unwrap();
                reference.project_frame([[0.8, -0.6], [0.6, 0.8]]).unwrap();
                let a = evaluate_persistent_inner::<T, true>(
                    &params,
                    &wheel,
                    terrain,
                    &material,
                    &mut cached,
                    0.001,
                )
                .unwrap();
                let b = evaluate_persistent_inner::<T, false>(
                    &params,
                    &wheel,
                    terrain,
                    &material,
                    &mut reference,
                    0.001,
                )
                .unwrap();
                compare_contact(&a, &b);
                compare_history(&cached, &reference);
                let orientation = RealWheelOrientation {
                    wheel_up: [0.0, 0.98, 0.2],
                };
                let a = evaluate_brush_inner::<T, true>(
                    &params,
                    &wheel,
                    terrain,
                    &material,
                    &orientation,
                    None,
                )
                .unwrap();
                let b = evaluate_brush_inner::<T, false>(
                    &params,
                    &wheel,
                    terrain,
                    &material,
                    &orientation,
                    None,
                )
                .unwrap();
                compare_contact(&a, &b);
                let before = cached.clone();
                let mut invalid = wheel;
                invalid.hub_vel[0] = f64::NAN;
                let a = evaluate_persistent_inner::<T, true>(
                    &params,
                    &invalid,
                    terrain,
                    &material,
                    &mut cached,
                    0.001,
                );
                let b = evaluate_persistent_inner::<T, false>(
                    &params,
                    &invalid,
                    terrain,
                    &material,
                    &mut reference,
                    0.001,
                );
                assert_eq!(a.unwrap_err(), b.unwrap_err());
                compare_history(&cached, &before);
                compare_history(&reference, &before);
            }
        }
    }
}

#[test]
fn complete_cached_brush_matches_uncached_oracle() {
    exercise(&FlatTerrain::new(0.0, 0.8));
    exercise(&MixedTerrain);
}
