use molla_math::{
    Mat3, Quat, Real, Vec3,
    constitutive::{ElasticMaterial, ElasticModel},
};
use molla_sim::ElasticElement;
use molla_solvers::barrier_contact::{BarrierObstacle, BarrierParams, BarrierShell};

fn material(model: ElasticModel) -> ElasticMaterial {
    ElasticMaterial {
        model,
        mu: 1000.0,
        lambda: 1000.0,
    }
}
fn models() -> [ElasticModel; 4] {
    [
        ElasticModel::Corotated,
        ElasticModel::QuarticStretch,
        ElasticModel::StableNeoHookean,
        ElasticModel::BaraffWitkin,
    ]
}
fn shell(x: Vec<Vec3>, element: ElasticElement) -> BarrierShell {
    let n = x.len();
    BarrierShell::new(
        x,
        vec![0.05; n],
        vec![],
        vec![],
        BarrierParams {
            gravity: Vec3::ZERO,
            ..Default::default()
        },
    )
    .unwrap()
    .with_elastic_elements(vec![element])
    .unwrap()
}

#[test]
fn element_force_and_coupled_tangent_follow_the_position_chain_rule() {
    let rest = vec![
        Vec3::new(0.2, 0.3, 0.1),
        Vec3::new(1.3, 0.4, 0.1),
        Vec3::new(0.4, 1.6, 0.2),
        Vec3::new(0.3, 0.5, 1.2),
    ];
    for model in models() {
        for n in [2, 3] {
            if n == 3 && model == ElasticModel::BaraffWitkin {
                continue;
            }
            let element = if n == 2 {
                ElasticElement::membrane(&rest, [0, 1, 2], 0.03, material(model)).unwrap()
            } else {
                ElasticElement::tetrahedron(&rest, [0, 1, 2, 3], material(model)).unwrap()
            };
            let rot = Mat3::from_quat(Quat::from_rotation_y(0.4));
            let affine = rot
                * Mat3::from_cols(
                    Vec3::new(1.1, 0.0, 0.1),
                    Vec3::new(0.2, 0.9, 0.0),
                    Vec3::Z * 1.2,
                );
            let x: Vec<_> = rest
                .iter()
                .map(|p| affine * *p + Vec3::splat(2.0))
                .collect();
            let r = element.response(&x, false).unwrap();
            let weights = element.shape_gradients();
            let eps = 1.0e-6;
            let mut net = Vec3::ZERO;
            for a in 0..n + 1 {
                let gradient = r.gradient * weights[a];
                net += gradient;
                for axis in 0..3 {
                    let mut plus = x.clone();
                    let mut minus = x.clone();
                    plus[a][axis] += eps;
                    minus[a][axis] -= eps;
                    let fd = (element.energy(&plus).unwrap() - element.energy(&minus).unwrap())
                        / (2.0 * eps);
                    assert!(
                        (fd - gradient[axis]).abs() < 1.0e-5 * (1.0 + gradient[axis].abs()),
                        "{model:?} n={n}: {fd} != {}",
                        gradient[axis]
                    );
                    let gp = element.response(&plus, false).unwrap().gradient;
                    let gm = element.response(&minus, false).unwrap().gradient;
                    let dir = [Vec3::X, Vec3::Y, Vec3::Z][axis];
                    let df =
                        Mat3::from_cols(dir * weights[a].x, dir * weights[a].y, dir * weights[a].z);
                    let hp = r.hessian_mul(df);
                    for b in 0..n + 1 {
                        let fd = (gp - gm) * weights[b] * (0.5 / eps);
                        assert!((fd - hp * weights[b]).length() < 2.0e-4 * (1.0 + fd.length()));
                    }
                }
            }
            assert!(net.length() < 1.0e-10);
            let sh = shell(x.clone(), element.clone());
            let mut g = vec![Vec3::ZERO; x.len()];
            sh.fd_probe_gradient(&x, &x, 0.1, &mut g);
            for a in 0..n + 1 {
                assert!((g[a] - r.gradient * weights[a] * 0.01).length() < 1.0e-10);
            }
        }
    }
}

#[test]
fn membranes_relax_with_no_springs_and_preserve_contact() {
    let rest = vec![Vec3::Y, Vec3::Y + Vec3::X, Vec3::Y + Vec3::Z];
    for model in models() {
        let element = ElasticElement::membrane(&rest, [0, 1, 2], 0.05, material(model)).unwrap();
        let mut x = rest.clone();
        x[1] += Vec3::new(0.3, 0.0, 0.2);
        let before = element.energy(&x).unwrap();
        let mut sh = shell(x, element.clone()).with_triangles(vec![[0, 1, 2]]);
        sh.pin_vertices(&[0, 2]);
        sh.obstacles.push(BarrierObstacle::HalfSpace {
            n: Vec3::Y,
            offset: 0.0,
        });
        for _ in 0..60 {
            sh.step(0.02).unwrap();
            assert!(sh.min_gap() > 0.0);
            assert_eq!(sh.positions[0], rest[0]);
            assert_eq!(sh.positions[2], rest[2]);
        }
        let after = element.energy(&sh.positions).unwrap();
        assert!(after < 0.7 * before, "{model:?}: {before} -> {after}");
        assert!(sh.springs.is_empty());
    }
}

#[test]
fn tet_materials_resist_volume_preserving_shear_without_springs() {
    let rest = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
    for model in models().into_iter().take(3) {
        let element = ElasticElement::tetrahedron(&rest, [0, 1, 2, 3], material(model)).unwrap();
        let mut x = rest.clone();
        x[2] += Vec3::X * 0.35;
        assert!((element.deformation(&x).unwrap().determinant() - 1.0).abs() < 1.0e-12);
        let before = element.energy(&x).unwrap();
        assert!(before > 0.0);
        let mut sh = shell(x, element.clone());
        sh.pin_vertices(&[0, 1, 3]);
        for _ in 0..60 {
            sh.step(0.02).unwrap();
            assert!(element.deformation(&sh.positions).unwrap().determinant() > 0.0);
        }
        let after = element.energy(&sh.positions).unwrap();
        assert!(after < 0.7 * before, "{model:?}: {before} -> {after}");
        assert!(sh.springs.is_empty() && sh.tets.is_empty());
    }
}

#[test]
fn rest_validation_and_rank_bound_refuse_hidden_inversion_paths() {
    let rest = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
    let m = material(ElasticModel::StableNeoHookean);
    assert!(ElasticElement::membrane(&rest, [0, 1, 9], 0.1, m).is_err());
    assert!(ElasticElement::membrane(&rest, [0, 1, 1], 0.1, m).is_err());
    assert!(ElasticElement::tetrahedron(&rest, [0, 1, 2, 2], m).is_err());
    assert!(ElasticElement::membrane(&rest, [0, 1, 2], Real::INFINITY, m).is_err());
    let element = ElasticElement::tetrahedron(&rest, [0, 1, 2, 3], m).unwrap();
    let direction = vec![Vec3::ZERO, Vec3::X * (-3.0), Vec3::Y * (-3.0), Vec3::ZERO];
    let target: Vec<_> = rest.iter().zip(&direction).map(|(x, d)| *x + *d).collect();
    assert!(element.deformation(&target).unwrap().determinant() > 0.0);
    let bound = element.step_bound(&rest, &direction).unwrap();
    assert!(bound > 0.0 && bound < 1.0 / 3.0);
    for alpha in [bound * 0.5, bound] {
        let x: Vec<_> = rest
            .iter()
            .zip(&direction)
            .map(|(x, d)| *x + *d * alpha)
            .collect();
        assert!(element.deformation(&x).unwrap().determinant() > 0.0);
    }
    let mut sh = shell(rest, element);
    sh.positions.swap(1, 2);
    let before = sh.positions.clone();
    assert!(sh.step(0.02).is_err());
    assert_eq!(sh.positions, before);
}
