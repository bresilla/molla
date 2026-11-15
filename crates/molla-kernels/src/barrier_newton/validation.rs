use super::{BarrierGpuObstacle, BarrierGpuParams, BarrierGpuSpring};
use molla_core::{Error, Result};

fn invalid(field: &str) -> Error {
    Error::Solver(format!("barrier_newton: invalid {field}"))
}

impl BarrierGpuParams {
    /// Validate solver coefficients and iteration budgets.
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("dhat", self.dhat),
            ("kappa_scale", self.kappa_scale),
            ("newton_tol", self.newton_tol),
            ("friction_epsv", self.friction_epsv),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid(name));
            }
        }
        for (name, value) in [
            ("friction_mu", self.friction_mu),
            ("strain_barrier_scale", self.strain_barrier_scale),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(invalid(name));
            }
        }
        if !self.gravity.iter().all(|v| v.is_finite()) {
            return Err(invalid("gravity"));
        }
        if !self.friction_mu_s_scale.is_finite() || self.friction_mu_s_scale < 1.0 {
            return Err(invalid("friction_mu_s_scale"));
        }
        if !self.strain_limit.is_finite() || (self.strain_limit != 0.0 && self.strain_limit <= 1.0)
        {
            return Err(invalid("strain_limit"));
        }
        if self.max_newton_iters == 0 || self.cg_iters == 0 || self.cg_block == 0 {
            return Err(invalid("iteration budget"));
        }
        Ok(())
    }
}

pub(super) fn timestep(h: f32) -> Result<()> {
    let h2 = h * h;
    if !h.is_finite() || h <= 0.0 || !h2.is_finite() || !h2.recip().is_finite() {
        return Err(invalid("timestep"));
    }
    Ok(())
}

pub(super) fn strain_limit(params: &BarrierGpuParams, has_zero_rest_springs: bool) -> Result<()> {
    if params.strain_limit > 0.0 && has_zero_rest_springs {
        return Err(invalid("strain limit on zero-rest spring"));
    }
    Ok(())
}

pub(super) fn mesh(
    positions: &[[f32; 3]],
    masses: &[f32],
    springs: &[BarrierGpuSpring],
    obstacles: &[BarrierGpuObstacle],
    triangles: &[[u32; 3]],
    tets: &[([u32; 4], f32)],
    pinned: &[u32],
) -> Result<()> {
    let n = positions.len();
    if n > u32::MAX as usize || masses.len() != n {
        return Err(invalid("mass length or vertex count"));
    }
    if !positions.iter().flatten().all(|v| v.is_finite()) {
        return Err(invalid("position"));
    }
    if masses.iter().any(|m| !m.is_finite() || *m <= 0.0) {
        return Err(invalid("mass"));
    }
    let valid_ids = |ids: &[u32]| {
        ids.iter()
            .enumerate()
            .all(|(i, &v)| (v as usize) < n && !ids[..i].contains(&v))
    };
    for s in springs {
        if !valid_ids(&[s.a, s.b])
            || !s.rest.is_finite()
            || s.rest < 0.0
            || !s.k.is_finite()
            || s.k < 0.0
        {
            return Err(invalid("spring"));
        }
    }
    let p = |i: u32| {
        molla_math::Vec3::new(
            f64::from(positions[i as usize][0]),
            f64::from(positions[i as usize][1]),
            f64::from(positions[i as usize][2]),
        )
    };
    for t in triangles {
        if !valid_ids(t) {
            return Err(invalid("triangle indices"));
        }
        let cross = (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0]));
        let area2 = cross.length_squared() as f32;
        if !area2.is_finite() || area2 <= 0.0 {
            return Err(invalid("triangle area"));
        }
    }
    for &(ids, kv) in tets {
        if !valid_ids(&ids) || !kv.is_finite() || kv < 0.0 {
            return Err(invalid("tet indices or stiffness"));
        }
        let a = p(ids[0]);
        let volume = (p(ids[1]) - a).dot((p(ids[2]) - a).cross(p(ids[3]) - a)) as f32;
        if !volume.is_finite() || volume == 0.0 {
            return Err(invalid("tet volume"));
        }
    }
    if pinned.iter().any(|&i| i as usize >= n) {
        return Err(invalid("pinned index"));
    }
    for obstacle in obstacles {
        match *obstacle {
            BarrierGpuObstacle::HalfSpace { n, offset } => {
                let length2: f32 = n.iter().map(|v| v * v).sum();
                if !length2.is_finite() || (length2 - 1.0).abs() > 1.0e-4 || !offset.is_finite() {
                    return Err(invalid("plane normal or offset"));
                }
            }
            BarrierGpuObstacle::Sphere { center, radius } => {
                if !center.iter().all(|v| v.is_finite()) || !radius.is_finite() || radius <= 0.0 {
                    return Err(invalid("sphere"));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_parameters_and_nonzero_budgets_are_required() {
        let default = BarrierGpuParams::default();
        default.validate().unwrap();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for params in [
                BarrierGpuParams {
                    dhat: value,
                    ..default
                },
                BarrierGpuParams {
                    kappa_scale: value,
                    ..default
                },
                BarrierGpuParams {
                    newton_tol: value,
                    ..default
                },
                BarrierGpuParams {
                    gravity: [value, 0.0, 0.0],
                    ..default
                },
                BarrierGpuParams {
                    friction_mu: value,
                    ..default
                },
                BarrierGpuParams {
                    friction_mu_s_scale: value,
                    ..default
                },
                BarrierGpuParams {
                    friction_epsv: value,
                    ..default
                },
                BarrierGpuParams {
                    strain_limit: value,
                    ..default
                },
                BarrierGpuParams {
                    strain_barrier_scale: value,
                    ..default
                },
            ] {
                assert!(params.validate().is_err(), "{params:?}");
            }
        }
        for params in [
            BarrierGpuParams {
                max_newton_iters: 0,
                ..default
            },
            BarrierGpuParams {
                cg_iters: 0,
                ..default
            },
            BarrierGpuParams {
                cg_block: 0,
                ..default
            },
            BarrierGpuParams {
                strain_limit: 1.0,
                ..default
            },
            BarrierGpuParams {
                strain_limit: -1.0,
                ..default
            },
            BarrierGpuParams {
                strain_barrier_scale: -1.0,
                ..default
            },
            BarrierGpuParams {
                friction_mu: -1.0,
                ..default
            },
        ] {
            assert!(params.validate().is_err(), "{params:?}");
        }
    }

    #[test]
    fn timestep_must_support_finite_inverse_squared_arithmetic() {
        timestep(1.0 / 240.0).unwrap();
        for h in [
            0.0,
            -1.0,
            f32::NAN,
            f32::INFINITY,
            f32::MAX,
            f32::MIN_POSITIVE,
        ] {
            assert!(timestep(h).is_err(), "{h}");
        }
    }

    #[test]
    fn topology_and_geometry_are_checked_before_indexing() {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let spring = BarrierGpuSpring {
            a: 0,
            b: 1,
            rest: 1.0,
            k: 1.0,
        };
        mesh(
            &positions,
            &[1.0; 4],
            &[BarrierGpuSpring {
                rest: 0.0,
                ..spring
            }],
            &[],
            &[],
            &[],
            &[],
        )
        .unwrap();
        mesh(
            &positions,
            &[1.0; 4],
            &[spring],
            &[],
            &[[0, 1, 2]],
            &[([0, 1, 2, 3], 1.0)],
            &[0],
        )
        .unwrap();
        for s in [
            BarrierGpuSpring { a: 99, ..spring },
            BarrierGpuSpring { b: 0, ..spring },
            BarrierGpuSpring {
                rest: -1.0,
                ..spring
            },
            BarrierGpuSpring {
                k: f32::NAN,
                ..spring
            },
        ] {
            assert!(mesh(&positions, &[1.0; 4], &[s], &[], &[], &[], &[]).is_err());
        }
        for t in [[0, 1, 99], [0, 1, 1]] {
            assert!(mesh(&positions, &[1.0; 4], &[], &[], &[t], &[], &[]).is_err());
        }
        for ids in [[0, 1, 2, 99], [0, 1, 2, 2]] {
            assert!(mesh(&positions, &[1.0; 4], &[], &[], &[], &[(ids, 1.0)], &[]).is_err());
        }
        assert!(mesh(&positions, &[1.0; 4], &[], &[], &[], &[], &[99]).is_err());
        assert!(mesh(&[[0.0; 3]; 4], &[1.0; 4], &[], &[], &[[0, 1, 2]], &[], &[]).is_err());
        assert!(
            mesh(
                &[[0.0; 3]; 4],
                &[1.0; 4],
                &[],
                &[],
                &[],
                &[([0, 1, 2, 3], 1.0)],
                &[]
            )
            .is_err()
        );
    }

    #[test]
    fn invalid_state_and_obstacles_are_refused() {
        for value in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(mesh(&[[0.0; 3]], &[value], &[], &[], &[], &[], &[]).is_err());
        }
        assert!(mesh(&[[f32::NAN, 0.0, 0.0]], &[1.0], &[], &[], &[], &[], &[]).is_err());
        assert!(mesh(&[[0.0; 3]], &[], &[], &[], &[], &[], &[]).is_err());
        for obstacle in [
            BarrierGpuObstacle::HalfSpace {
                n: [0.0; 3],
                offset: 0.0,
            },
            BarrierGpuObstacle::HalfSpace {
                n: [0.0, 1.0, 0.0],
                offset: f32::NAN,
            },
            BarrierGpuObstacle::Sphere {
                center: [0.0; 3],
                radius: 0.0,
            },
            BarrierGpuObstacle::Sphere {
                center: [f32::INFINITY; 3],
                radius: 1.0,
            },
        ] {
            assert!(mesh(&[[0.0; 3]], &[1.0], &[], &[obstacle], &[], &[], &[]).is_err());
        }
    }

    #[test]
    fn zero_rest_springs_require_disabled_strain_limit() {
        let mut params = BarrierGpuParams::default();
        strain_limit(&params, true).unwrap();
        params.strain_limit = 1.1;
        assert!(strain_limit(&params, true).is_err());
        strain_limit(&params, false).unwrap();
    }
}
