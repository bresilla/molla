use molla_geometry::{
    HydroelasticConfig, PatchSample, bake_mesh_to_sdf, compute_hydroelastic_contact,
    reduce_with_anchor, sample_plane_patch,
};
use molla_math::{Real, Transform, Vec3};
use molla_sim::TriMesh;

fn unit_cube_mesh() -> TriMesh {
    let vertices = vec![
        Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(0.5, 0.5, -0.5),
        Vec3::new(-0.5, 0.5, -0.5),
        Vec3::new(-0.5, -0.5, 0.5),
        Vec3::new(0.5, -0.5, 0.5),
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(-0.5, 0.5, 0.5),
    ];
    let indices = vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [3, 6, 2],
        [3, 7, 6],
        [0, 7, 3],
        [0, 4, 7],
        [1, 2, 6],
        [1, 6, 5],
    ];
    TriMesh { vertices, indices }
}

fn finite_manifold_force(label: &str, force: Real, net_force: Vec3) {
    assert!(
        force.is_finite() && force > 0.0,
        "{label}: total normal force must be positive and finite, got {force}",
    );
    assert!(
        net_force.is_finite(),
        "{label}: net force must remain finite, got {net_force:?}",
    );
    assert!(
        net_force.y > 0.0,
        "{label}: top-face patch should push upward, got {net_force:?}",
    );
}

#[test]
fn plane_patch_hydroelastic_force_grows_with_penetration_depth() {
    let sdf = bake_mesh_to_sdf(&unit_cube_mesh(), 40, 0.05);
    let samples = sample_plane_patch(0.20, 0.20, 10, 10);
    let cfg = HydroelasticConfig {
        hydroelastic_modulus: 1.0e5,
        culling_distance: 0.0,
        ..HydroelasticConfig::default()
    };

    let force_at_depth = |depth: Real| {
        let patch_pose = Transform::from_translation(Vec3::new(0.0, 0.5 - depth, 0.0));
        let manifold =
            compute_hydroelastic_contact(&sdf, &samples, Transform::IDENTITY, patch_pose, &cfg);
        assert!(
            !manifold.contacts.is_empty(),
            "expected contacts for patch penetration depth {depth}",
        );
        finite_manifold_force(
            "penetrating patch",
            manifold.total_normal_force,
            manifold.net_force,
        );
        manifold.total_normal_force
    };

    let shallow = force_at_depth(0.02);
    let deep = force_at_depth(0.06);
    assert!(
        deep > shallow * 2.0,
        "hydroelastic force should grow with penetration depth: shallow={shallow}, deep={deep}",
    );
}

#[test]
fn hydroelastic_max_pressure_caps_integrated_force() {
    let sdf = bake_mesh_to_sdf(&unit_cube_mesh(), 40, 0.05);
    let samples = sample_plane_patch(0.25, 0.25, 8, 8);
    let max_pressure: Real = 250.0;
    let cfg = HydroelasticConfig {
        hydroelastic_modulus: 1.0e8,
        max_pressure,
        culling_distance: 0.0,
        ..HydroelasticConfig::default()
    };
    let patch_pose = Transform::from_translation(Vec3::new(0.0, 0.40, 0.0));
    let manifold =
        compute_hydroelastic_contact(&sdf, &samples, Transform::IDENTITY, patch_pose, &cfg);

    assert!(
        !manifold.contacts.is_empty(),
        "expected capped hydroelastic contacts for a deep patch",
    );
    finite_manifold_force(
        "capped patch",
        manifold.total_normal_force,
        manifold.net_force,
    );
    for contact in &manifold.contacts {
        assert!(
            contact.pressure <= max_pressure + 1.0e-9,
            "pressure cap violated: {} > {}",
            contact.pressure,
            max_pressure,
        );
    }

    let total_area: Real = samples.iter().map(|sample| sample.area).sum();
    let upper_bound = max_pressure * total_area;
    assert!(
        manifold.total_normal_force <= upper_bound + 1.0e-6,
        "capped integrated force {} exceeded max_pressure * area {}",
        manifold.total_normal_force,
        upper_bound,
    );
}

#[test]
fn hydroelastic_offset_patch_reduction_preserves_force_and_moment() {
    let sdf = bake_mesh_to_sdf(&unit_cube_mesh(), 40, 0.05);
    let samples = sample_plane_patch(0.18, 0.14, 9, 7);
    let cfg = HydroelasticConfig {
        hydroelastic_modulus: 7.5e4,
        culling_distance: 0.0,
        ..HydroelasticConfig::default()
    };
    let patch_pose = Transform::from_translation(Vec3::new(0.18, 0.46, -0.12));
    let manifold =
        compute_hydroelastic_contact(&sdf, &samples, Transform::IDENTITY, patch_pose, &cfg);
    assert!(
        !manifold.contacts.is_empty(),
        "expected offset hydroelastic patch contacts"
    );
    finite_manifold_force(
        "offset patch",
        manifold.total_normal_force,
        manifold.net_force,
    );

    let patch: Vec<PatchSample> = manifold
        .contacts
        .iter()
        .map(|contact| PatchSample {
            point: contact.point,
            normal: contact.normal,
            force_magnitude: contact.normal_force(),
        })
        .collect();
    let force_ref: Vec3 = patch.iter().map(|sample| sample.force()).sum();
    let moment_ref: Vec3 = patch
        .iter()
        .map(|sample| sample.point.cross(sample.force()))
        .sum();
    assert!(
        moment_ref.length() > 1.0e-3,
        "offset patch should generate a non-trivial moment, got {moment_ref:?}"
    );

    let reduced = reduce_with_anchor(&patch).expect("hydro patch reduction");
    let force_err = (reduced.net_force() - force_ref).length() / force_ref.length().max(1.0e-12);
    let moment_err =
        (reduced.net_moment() - moment_ref).length() / moment_ref.length().max(1.0e-12);

    assert!(
        force_err < 1.0e-9,
        "hydro patch reduced force mismatch: ref={force_ref:?}, reduced={:?}, rel={force_err:e}",
        reduced.net_force()
    );
    assert!(
        moment_err < 0.01,
        "hydro patch reduced moment mismatch: ref={moment_ref:?}, reduced={:?}, rel={moment_err:e}",
        reduced.net_moment()
    );
}
