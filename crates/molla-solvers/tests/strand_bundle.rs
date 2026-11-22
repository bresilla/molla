use molla_geometry::convex_cut::CutPlane;
use molla_math::Vec3;
use molla_solvers::strands::{BundleDesc, StrandModel};

fn description() -> BundleDesc {
    BundleDesc {
        start: Vec3::new(0.0, 1.5, 0.0),
        end: Vec3::new(2.0, 1.5, 0.0),
        strand_radius: 0.015,
        rings: 1,
        segments: 8,
        density: 650.0,
        young_modulus: 1e8,
        shear_modulus: 4e7,
        tensile_strength: 1e6,
        shear_strength: 5e5,
        interface_ratio: 1.0,
        damage_rate: 1.0,
        damping: 0.5,
        pin_start: true,
    }
}

fn mass(model: &StrandModel) -> f64 {
    model.fibers().iter().map(|f| f.mass.mass).sum()
}

#[test]
fn prescribed_fiber_fracture_is_nonplanar_conservative_and_keeps_two_pieces() {
    use molla_geometry::convex_cut::FaceOrigin;
    let mut model = description().build().unwrap().model;
    for fiber in model.fibers().to_vec() {
        if fiber.pin.is_none() {
            let mut motion = fiber.motion;
            motion.linear_velocity = Vec3::new(0.3, -0.2, 0.7);
            motion.angular_velocity = Vec3::new(0.2, 0.4, -0.1);
            model.set_motion(fiber.id, motion).unwrap();
        }
    }
    let totals = |m: &StrandModel| {
        m.fibers()
            .iter()
            .fold((0.0, Vec3::ZERO, Vec3::ZERO, 0.0), |t, f| {
                let (p, l, k) = f.motion.momentum(f.mass).unwrap();
                (t.0 + f.mass.mass, t.1 + p, t.2 + l, t.3 + k)
            })
    };
    let before = totals(&model);
    let plane = CutPlane::new(Vec3::new(1.137, 1.5, 0.0), Vec3::X).unwrap();
    let surfaces: Vec<_> = model
        .fibers()
        .iter()
        .map(|f| {
            let offset = 0.06 * ((f.center().y - 1.5) * 70.0).sin() * (f.center().z * 50.0).cos();
            (
                f.id,
                CutPlane::new(plane.point() + Vec3::X * offset, Vec3::X).unwrap(),
            )
        })
        .collect();
    let transaction = model.prepare_fracture(plane, &surfaces, 91).unwrap();
    assert_eq!(transaction.candidate().components().len(), 2);
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &(parent, children) in &transaction.report().replacements {
        let surface = surfaces.iter().find(|(id, _)| *id == parent).unwrap().1;
        for id in children {
            let f = transaction.candidate().fiber(id).unwrap();
            for face in f.shape.faces() {
                if matches!(face.origin, FaceOrigin::Fracture { id: 91, .. }) {
                    for &index in &face.vertices {
                        let p = f
                            .motion
                            .pose
                            .transform_point(f.shape.vertices()[index as usize]);
                        assert!(surface.distance(p).abs() < 1e-12);
                        lo = lo.min(p.x);
                        hi = hi.max(p.x);
                    }
                }
            }
        }
    }
    assert!(hi - lo > 0.02, "fracture was flattened into a single plane");
    let after = totals(transaction.candidate());
    assert!((before.0 - after.0).abs() < before.0 * 1e-12);
    assert!(before.1.distance(after.1) < 1e-11);
    assert!(before.2.distance(after.2) < 1e-11);
    assert!((before.3 - after.3).abs() < 1e-11);
    for bond in transaction.candidate().bonds() {
        let a = transaction.candidate().fiber(bond.fibers[0]).unwrap();
        let b = transaction.candidate().fiber(bond.fibers[1]).unwrap();
        for row in molla_solvers::strands::bond_rows(bond, a, b) {
            assert!(row.value.abs() < 1e-12);
        }
    }
    model.commit_cut(transaction).unwrap();
    assert!(
        model
            .prepare_fracture(plane, &[(0, plane), (0, plane)], 92)
            .is_err()
    );
}

#[test]
fn coupled_oracle_resolves_loaded_bundle_constitutive_equations() {
    use molla_solvers::strands::{CoupledConfig, StepConfig, bond_rows, step_cpu};
    let mut desc = description();
    desc.damage_rate = 0.0;
    let mut model = desc.build().unwrap().model;
    let config = StepConfig {
        ground_height: None,
        coupled: Some(CoupledConfig {
            nonlinear_iterations: 6,
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut max_gap: f64 = 0.0;
    let mut max_residual: f64 = 0.0;
    for _ in 0..24 {
        let report = step_cpu(&mut model, config).unwrap();
        for bond in model.bonds() {
            let rows = bond_rows(
                bond,
                model.fiber(bond.fibers[0]).unwrap(),
                model.fiber(bond.fibers[1]).unwrap(),
            );
            let gap = Vec3::new(rows[0].value, rows[1].value, rows[2].value).length();
            max_gap = max_gap.max(gap);
            let (_, force, torque) = report
                .bond_loads
                .iter()
                .find(|(id, _, _)| *id == bond.id)
                .unwrap();
            for (k, row) in rows.iter().enumerate() {
                let load = if k < 3 { force[k] } else { torque[k - 3] };
                max_residual = max_residual.max((row.value + row.compliance * load).abs());
            }
        }
    }
    eprintln!("coupled CPU bundle: 24 steps, gap={max_gap}, constitutive residual={max_residual}");
    assert!(max_residual < 1e-9);
    assert!(
        max_gap > 0.0001,
        "finite axial compliance must not be replaced by rigidity"
    );
}

#[test]
fn hex_bundle_has_volume_based_mass_staggered_fibers_and_material_scaled_bonds() {
    let desc = description();
    let bundle = desc.build().unwrap();
    assert_eq!(bundle.strands.len(), 7);
    assert_eq!(bundle.model.fibers().len(), 56);
    assert_eq!(bundle.model.components().len(), 1);
    let expected = 7.0
        * 1.5
        * 3.0_f64.sqrt()
        * desc.strand_radius.powi(2)
        * (desc.end - desc.start).length()
        * desc.density;
    assert!((mass(&bundle.model) - expected).abs() < expected * 1e-12);
    assert_eq!(
        bundle
            .model
            .fibers()
            .iter()
            .filter(|f| f.pin.is_some())
            .count(),
        7
    );
    let inner = &bundle.model.fibers()[bundle.strands[0][2] as usize];
    let outer = &bundle.model.fibers()[bundle.strands[1][2] as usize];
    assert!((inner.rest.position.x - outer.rest.position.x).abs() > 0.01);
    assert!(bundle.model.bonds().len() > 49);
    for group in bundle.model.colors() {
        let mut ids = std::collections::BTreeSet::new();
        for &index in group {
            for &id in &bundle.model.bonds()[index].fibers {
                assert!(ids.insert(id));
            }
        }
    }
}

#[test]
fn whole_bundle_cut_is_coplanar_mass_conserving_and_disconnects_every_fiber() {
    let mut bundle = description().build().unwrap();
    let before = mass(&bundle.model);
    let plane = CutPlane::new(Vec3::new(1.137, 1.5, 0.0), Vec3::new(1.0, 0.3, -0.2)).unwrap();
    let ids: Vec<_> = bundle.model.fibers().iter().map(|f| f.id).collect();
    let cut = bundle.model.prepare_cut(plane, &ids, 71).unwrap();
    assert!(!cut.report().split_bonds.is_empty());
    for (old, children) in &cut.report().split_bonds {
        let original = bundle.model.bonds().iter().find(|b| b.id == *old).unwrap();
        let mut stiffness = Vec3::ZERO;
        let mut strength = Vec3::ZERO;
        for &id in children {
            let child = cut.candidate().bonds().iter().find(|b| b.id == id).unwrap();
            stiffness += Vec3::ONE / child.material.translation;
            strength += child.material.force_limit;
        }
        assert!(
            (stiffness * original.material.translation - Vec3::ONE)
                .abs()
                .max_element()
                < 1e-9
        );
        assert!(
            (strength / original.material.force_limit - Vec3::ONE)
                .abs()
                .max_element()
                < 1e-9
        );
    }
    let split_ids: std::collections::BTreeSet<_> = cut
        .report()
        .replacements
        .iter()
        .map(|(id, _)| *id)
        .collect();
    for strand in &bundle.strands {
        assert!(strand.iter().any(|id| split_ids.contains(id)));
    }
    bundle.model.commit_cut(cut).unwrap();
    assert_eq!(bundle.model.components().len(), 2);
    assert!((mass(&bundle.model) - before).abs() < before * 1e-12);
    for bond in bundle.model.bonds() {
        let p = [0, 1].map(|i| {
            bundle
                .model
                .fiber(bond.fibers[i])
                .unwrap()
                .motion
                .pose
                .transform_point(bond.anchors[i])
        });
        assert!(plane.distance(p[0]) * plane.distance(p[1]) >= 0.0);
    }
}

#[test]
fn repeated_oblique_cuts_preserve_material_interfaces_without_loose_wedges() {
    let mut bundle = description().build().unwrap();
    let before = mass(&bundle.model);
    for (id, x) in [0.371, 0.793, 1.137, 1.629].into_iter().enumerate() {
        let plane = CutPlane::new(Vec3::new(x, 1.5, 0.0), Vec3::new(1.0, 0.31, -0.27)).unwrap();
        let selected: Vec<_> = bundle.model.fibers().iter().map(|f| f.id).collect();
        let cut = bundle
            .model
            .prepare_cut(plane, &selected, id as u64)
            .unwrap();
        bundle.model.commit_cut(cut).unwrap();
        assert_eq!(bundle.model.components().len(), id + 2);
        for bond in bundle.model.bonds() {
            let a = bundle.model.fiber(bond.fibers[0]).unwrap();
            let b = bundle.model.fiber(bond.fibers[1]).unwrap();
            for row in molla_solvers::strands::bond_rows(bond, a, b) {
                assert!(row.value.abs() < 1e-12);
            }
        }
    }
    assert!((mass(&bundle.model) - before).abs() < before * 1e-12);
}

#[test]
fn cutting_only_one_fiber_leaves_the_bundle_partially_attached() {
    let mut bundle = description().build().unwrap();
    let plane = CutPlane::new(Vec3::new(1.137, 1.5, 0.0), Vec3::X).unwrap();
    let cut = bundle
        .model
        .prepare_cut(plane, &bundle.strands[1], 2)
        .unwrap();
    assert_eq!(cut.report().replacements.len(), 1);
    bundle.model.commit_cut(cut).unwrap();
    assert_eq!(bundle.model.components().len(), 1);
}

#[test]
fn bundle_description_rejects_degenerate_and_unbounded_inputs() {
    for rings in [4, usize::MAX] {
        assert!(
            BundleDesc {
                rings,
                ..description()
            }
            .build()
            .is_err()
        );
    }
    for segments in [0, 1, 2, 513] {
        assert!(
            BundleDesc {
                segments,
                ..description()
            }
            .build()
            .is_err()
        );
    }
    for young_modulus in [0.0, -1.0, f64::NAN] {
        assert!(
            BundleDesc {
                young_modulus,
                ..description()
            }
            .build()
            .is_err()
        );
    }
    assert!(
        BundleDesc {
            end: description().start,
            ..description()
        }
        .build()
        .is_err()
    );
}
