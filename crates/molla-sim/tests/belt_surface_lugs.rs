use molla_core::WorldId;
use molla_math::{Real, Vec3};
use molla_sim::soft_belt::path::{BeltPath, BeltPulley};
use molla_sim::soft_belt::{RubberBeltMesh, RubberBeltParams, RubberCordParams, RubberLugParams};
use molla_sim::{ModelBuilder, compute_tet_surface_triangles};
use std::collections::HashMap;

const WIDTHS: [Real; 14] = [
    -0.09, -0.088, -0.060, -0.034, -0.025, -0.013, -0.006, 0.006, 0.013, 0.025, 0.034, 0.060,
    0.088, 0.09,
];

fn footprint(x: Real) -> (Real, Real) {
    let left = [
        (-0.088, -0.016, 0.005),
        (-0.060, -0.016, 0.018),
        (-0.034, -0.016, 0.018),
        (-0.006, -0.004, 0.018),
    ];
    let clamped = -x.abs().clamp(0.006, 0.088);
    let pair = left
        .windows(2)
        .find(|w| clamped >= w[0].0 && clamped <= w[1].0)
        .unwrap();
    let t = (clamped - pair[0].0) / (pair[1].0 - pair[0].0);
    let low = pair[0].1 + t * (pair[1].1 - pair[0].1);
    let high = pair[0].2 + t * (pair[1].2 - pair[0].2);
    if x < 0.0 { (low, high) } else { (-high, -low) }
}

fn fixture() -> (RubberBeltMesh, BeltPath, Vec<Real>) {
    let outer = BeltPath::around_pulleys(&[
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
            radius: 0.1377096631429759,
        },
    ])
    .unwrap();
    let samples = (0..512)
        .map(|i| {
            outer
                .sample(outer.length() * i as Real / 512.0)
                .unwrap()
                .position
        })
        .collect::<Vec<_>>();
    let path = BeltPath::from_closed_polyline(&samples, 0.0105).unwrap();
    let pitch = path.length() / 56.0;
    let base = RubberBeltMesh::circular(RubberBeltParams {
        radius: path.length() / std::f64::consts::TAU,
        segments: 224,
        width_cells: 13,
        thickness: 0.021,
        ..Default::default()
    })
    .unwrap()
    .with_width_stations(&WIDTHS)
    .unwrap();
    let offsets = WIDTHS
        .iter()
        .flat_map(|&x| {
            let (low, high) = footprint(x);
            [
                low + pitch / 4.0,
                (low + high) / 2.0,
                high - pitch / 4.0,
                (high + pitch + low) / 2.0 - pitch / 2.0,
            ]
        })
        .collect();
    (base, path, offsets)
}

fn rows() -> ([RubberLugParams; 2], [RubberLugParams; 2]) {
    let inner = RubberLugParams {
        count: 56,
        segment_span: 2,
        width_start: 4,
        width_cells: 1,
        depth: 0.017,
        depth_cells: 2,
    };
    let outer = RubberLugParams {
        width_start: 1,
        width_cells: 5,
        depth: 0.014,
        ..inner
    };
    (
        [
            inner,
            RubberLugParams {
                width_start: 8,
                ..inner
            },
        ],
        [
            outer,
            RubberLugParams {
                width_start: 7,
                ..outer
            },
        ],
    )
}

fn assert_closed_connected(mesh: &RubberBeltMesh) {
    let mut faces = HashMap::new();
    for &[a, b, c, d] in &mesh.tets {
        for mut face in [[a, b, c], [a, b, d], [a, c, d], [b, c, d]] {
            face.sort();
            *faces.entry(face).or_insert(0usize) += 1;
        }
    }
    assert!(faces.values().all(|&n| n == 1 || n == 2));
    let mut edges = HashMap::new();
    for tri in compute_tet_surface_triangles(&mesh.tets) {
        for mut edge in [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]] {
            edge.sort();
            *edges.entry(edge).or_insert(0usize) += 1;
        }
    }
    assert!(edges.values().all(|&n| n == 2));
    let mut parent: Vec<_> = (0..mesh.positions.len()).collect();
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
    assert!((0..parent.len()).all(|i| root(&parent, i) == root(&parent, 0)));
}

#[test]
fn phased_chevrons_are_bonded_closed_and_positive_when_installed() {
    let (base, path, offsets) = fixture();
    let phased = base.with_outer_phase_offsets(56, &offsets).unwrap();
    let (inner, outer) = rows();
    let mesh = phased.with_surface_lug_rows(&inner, &outer).unwrap();
    assert_eq!(
        mesh.positions.len(),
        base.positions.len() + 2 * 56 * 3 * 2 * 2 + 2 * 56 * 3 * 6 * 2
    );
    assert_closed_connected(&mesh);
    assert!(mesh.volumes.iter().all(|v| v.is_finite() && *v > 0.0));
    assert!(mesh.masses.iter().all(|m| m.is_finite() && *m > 0.0));
    assert!(
        (mesh.masses.iter().sum::<Real>() - 1100.0 * mesh.volumes.iter().sum::<Real>()).abs()
            < 1e-10
    );
    let radius = path.length() / std::f64::consts::TAU;
    for s in 0..224 {
        for w in 0..14 {
            for r in 0..=1 {
                let i = (s * 14 + w) * 3 + r;
                assert_eq!(base.positions[i], mesh.positions[i]);
            }
            let point = phased.positions[(s * 14 + w) * 3 + 2];
            let angle = point.z.atan2(point.y).rem_euclid(std::f64::consts::TAU);
            let local = (angle * radius - path.length() / 224.0).rem_euclid(path.length() / 56.0);
            let (low, high) = footprint(WIDTHS[w]);
            let expected = [
                low,
                (low + high) / 2.0,
                high,
                (high + path.length() / 56.0 + low) / 2.0,
            ][s % 4];
            let error = (local - expected + path.length() / 112.0).rem_euclid(path.length() / 56.0)
                - path.length() / 112.0;
            assert!(error.abs() < 1e-12, "s={s} w={w} error={error}");
        }
    }
    let max_radius = mesh
        .positions
        .iter()
        .map(|p| p.y.hypot(p.z))
        .fold(0.0, Real::max);
    assert!((max_radius - radius - 0.0105 - 0.014).abs() < 1e-12);
    for fraction in [0.0, 0.25, 0.5, 0.75] {
        let installed = mesh
            .installed_positions(
                &path,
                -path.length() / 224.0 + fraction * path.length() / 56.0,
                Vec3::ZERO,
            )
            .unwrap();
        let min_j = mesh.minimum_volume_ratio(&installed).unwrap();
        assert!(min_j > 0.5, "phase={fraction} minJ={min_j}");
    }
}

#[test]
fn outer_phases_preserve_cord_rest_lengths_stiffness_and_mass() {
    let (base, _, offsets) = fixture();
    let phased = base.with_outer_phase_offsets(56, &offsets).unwrap();
    let (inner, outer) = rows();
    let mesh = phased.with_surface_lug_rows(&inner, &outer).unwrap();
    let build = |mesh: &RubberBeltMesh| {
        let mut builder = ModelBuilder::new();
        mesh.append_reinforced_to(
            &mut builder,
            WorldId(0),
            Vec3::ZERO,
            RubberCordParams::default(),
        )
        .unwrap();
        builder.build().unwrap()
    };
    let original = build(&base);
    let changed = build(&mesh);
    assert_eq!(
        original.spring_rest_length.host().unwrap(),
        changed.spring_rest_length.host().unwrap()
    );
    assert_eq!(
        original.spring_compliance.host().unwrap(),
        changed.spring_compliance.host().unwrap()
    );
    let original_cord_mass = original.particle_mass.host().unwrap().iter().sum::<Real>()
        - base.masses.iter().sum::<Real>();
    let changed_cord_mass = changed.particle_mass.host().unwrap().iter().sum::<Real>()
        - mesh.masses.iter().sum::<Real>();
    assert!((original_cord_mass - changed_cord_mass).abs() < 1e-10);
}

#[test]
fn rejects_invalid_phase_grids_and_surface_rows() {
    let (base, path, offsets) = fixture();
    for count in [0, 1, 3, 225, usize::MAX] {
        assert!(base.with_outer_phase_offsets(count, &offsets).is_err());
    }
    assert!(
        base.with_outer_phase_offsets(56, &offsets[..offsets.len() - 1])
            .is_err()
    );
    for (index, value) in [
        (0, Real::NAN),
        (0, Real::INFINITY),
        (0, path.length()),
        (3, path.length() / 56.0),
    ] {
        let mut bad = offsets.clone();
        bad[index] = value;
        assert!(base.with_outer_phase_offsets(56, &bad).is_err());
    }
    let (inner, outer) = rows();
    assert!(base.with_surface_lug_rows(&[], &[]).is_err());
    assert!(
        base.with_surface_lug_rows(&inner, &[outer[0], outer[0]])
            .is_err()
    );
    assert!(
        base.with_surface_lug_rows(
            &inner,
            &[
                outer[0],
                RubberLugParams {
                    width_start: 6,
                    ..outer[1]
                }
            ]
        )
        .is_err()
    );
    for bad in [
        RubberLugParams {
            depth: 0.0,
            ..outer[0]
        },
        RubberLugParams {
            depth: Real::NAN,
            ..outer[0]
        },
        RubberLugParams {
            width_start: usize::MAX,
            ..outer[0]
        },
        RubberLugParams {
            depth_cells: usize::MAX,
            ..outer[0]
        },
        RubberLugParams {
            segment_span: 4,
            ..outer[0]
        },
    ] {
        assert!(base.with_surface_lug_rows(&inner, &[bad]).is_err());
    }
    let mesh = base.with_surface_lug_rows(&inner, &outer).unwrap();
    assert!(mesh.with_outer_phase_offsets(56, &offsets).is_err());
    assert!(mesh.with_surface_lug_rows(&inner, &outer).is_err());
    assert!(base.with_surface_lug_rows(&[], &outer).is_ok());
}
