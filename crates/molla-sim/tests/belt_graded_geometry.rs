use molla_core::WorldId;
use molla_math::{Real, Vec3};
use molla_sim::soft_belt::path::{BeltPath, BeltPulley};
use molla_sim::soft_belt::{RubberBeltMesh, RubberBeltParams, RubberCordParams, RubberLugParams};
use molla_sim::{ModelBuilder, compute_tet_surface_triangles};
use std::collections::HashMap;

fn base() -> (RubberBeltMesh, BeltPath) {
    let path = BeltPath::around_pulleys(&[
        BeltPulley {
            center: Vec3::new(0.0, -0.52, -0.11),
            radius: 0.144,
        },
        BeltPulley {
            center: Vec3::new(0.0, 0.54, -0.11),
            radius: 0.144,
        },
        BeltPulley {
            center: Vec3::new(0.0, 0.405, 0.194),
            radius: 0.133,
        },
    ])
    .unwrap();
    let points = (0..256)
        .map(|i| {
            path.sample(path.length() * i as Real / 256.0)
                .unwrap()
                .position
        })
        .collect::<Vec<_>>();
    let neutral = BeltPath::from_closed_polyline(&points, 0.0105).unwrap();
    let mesh = RubberBeltMesh::circular(RubberBeltParams {
        radius: neutral.length() / std::f64::consts::TAU,
        segments: 224,
        width_cells: 5,
        thickness: 0.021,
        ..Default::default()
    })
    .unwrap();
    (mesh, neutral)
}

#[test]
fn graded_width_preserves_mass_cord_energy_and_exact_guide_columns() {
    let (original, _) = base();
    let mesh = original
        .with_width_stations(&[-0.09, -0.025, -0.013, 0.013, 0.025, 0.09])
        .unwrap();
    assert!(
        (original.masses.iter().sum::<Real>() - mesh.masses.iter().sum::<Real>()).abs() < 1e-10
    );
    for column in 0..6 {
        assert_eq!(
            mesh.positions[column * 3].x,
            [-0.09, -0.025, -0.013, 0.013, 0.025, 0.09][column]
        );
    }
    for prestrain in [0.0, 0.005] {
        let cord = RubberCordParams {
            prestrain,
            ..Default::default()
        };
        let mut builder = ModelBuilder::new();
        let handles = mesh
            .append_reinforced_to(&mut builder, WorldId(0), Vec3::ZERO, cord)
            .unwrap();
        let model = builder.build().unwrap();
        let mut energy = 0.0;
        let length = model.spring_rest_length.host().unwrap()[0] * 224.0;
        for index in handles.cords {
            let a = model.spring_a.host().unwrap()[index] as usize;
            let b = model.spring_b.host().unwrap()[index] as usize;
            let extension = (mesh.positions[b] - mesh.positions[a]).length() * 1.02
                - model.spring_rest_length.host().unwrap()[index];
            energy += 0.5 * extension * extension / model.spring_compliance.host().unwrap()[index];
        }
        let strain = 1.02 * (1.0 + prestrain) - 1.0;
        assert!((energy - 0.5 * cord.axial_rigidity * length * strain * strain).abs() < 1e-8);
        let mass = model.particle_mass.host().unwrap().iter().sum::<Real>()
            - mesh.masses.iter().sum::<Real>();
        assert!((mass - cord.linear_density * length).abs() < 1e-10);
    }
    for bad in [
        [-0.1, -0.025, -0.013, 0.013, 0.025, 0.09],
        [-0.09, -0.025, -0.025, 0.013, 0.025, 0.09],
        [-0.09, -0.025, 0.013, -0.013, 0.025, 0.09],
        [-0.09, -0.025, Real::NAN, 0.013, 0.025, 0.09],
    ] {
        assert!(original.with_width_stations(&bad).is_err());
    }
}

#[test]
fn paired_guides_share_base_nodes_and_preserve_positive_installed_volume() {
    let (base, path) = base();
    let base = base
        .with_width_stations(&[-0.09, -0.025, -0.013, 0.013, 0.025, 0.09])
        .unwrap();
    let lug = RubberLugParams {
        count: 56,
        segment_span: 2,
        width_start: 1,
        width_cells: 1,
        depth: 0.017,
        depth_cells: 2,
    };
    let mesh = base
        .with_drive_lug_rows(&[
            lug,
            RubberLugParams {
                width_start: 3,
                ..lug
            },
        ])
        .unwrap();
    assert_eq!(
        mesh.positions.len(),
        base.positions.len() + 2 * 56 * 3 * 2 * 2
    );
    let mut edges = HashMap::new();
    for tri in compute_tet_surface_triangles(&mesh.tets) {
        for mut edge in [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]] {
            edge.sort();
            *edges.entry(edge).or_insert(0) += 1;
        }
    }
    assert!(edges.values().all(|&count| count == 2));
    let mut parent: Vec<usize> = (0..mesh.positions.len()).collect();
    fn root(parent: &[usize], mut node: usize) -> usize {
        while parent[node] != node {
            node = parent[node];
        }
        node
    }
    for tet in &mesh.tets {
        for &node in &tet[1..] {
            let a = root(&parent, tet[0] as usize);
            let b = root(&parent, node as usize);
            parent[b] = a;
        }
    }
    assert!((0..parent.len()).all(|node| root(&parent, node) == root(&parent, 0)));
    for phase in [0.0, path.length() / 112.0] {
        let installed = mesh.installed_positions(&path, phase, Vec3::ZERO).unwrap();
        let ratio = mesh.minimum_volume_ratio(&installed).unwrap();
        assert!(ratio > 0.5, "installed paired-guide minimum J={ratio}");
    }
    assert!(base.with_drive_lug_rows(&[lug, lug]).is_err());
    assert!(
        base.with_drive_lug_rows(&[
            lug,
            RubberLugParams {
                width_start: 2,
                ..lug
            }
        ])
        .is_err()
    );
    assert!(base.with_drive_lug_rows(&[]).is_err());
    assert!(
        mesh.with_width_stations(&[-0.09, -0.025, -0.013, 0.013, 0.025, 0.09])
            .is_err()
    );
}
