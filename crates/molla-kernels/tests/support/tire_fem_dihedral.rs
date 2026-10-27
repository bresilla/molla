use super::*;

type Quad = [[f64; 3]; 4];

fn angle(p: &Quad) -> f64 {
    let sub = |a: [f64; 3], b: [f64; 3]| std::array::from_fn(|i| a[i] - b[i]);
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: [f64; 3], b: [f64; 3]| (0..3).map(|i| a[i] * b[i]).sum::<f64>();
    let a = cross(sub(p[1], p[0]), sub(p[2], p[0]));
    let b = cross(sub(p[2], p[0]), sub(p[3], p[0]));
    let c = cross(a, b);
    dot(c, c).sqrt().atan2(dot(a, b))
}

fn fixture(size: f64, bend: f64, compliance: f32) -> TireFemGpuInput {
    let mut input = simple_tire_fem_input(0.);
    macro_rules! clear {
        ($($field:ident),* $(,)?) => { $(input.$field.clear();)* };
    }
    clear!(
        tet_indices,
        tet_grads,
        tet_volumes,
        tet_mu,
        tet_lambda,
        tet_fiber_dir,
        tet_fiber_k1,
        tet_fiber_k2,
        tet_fiber_dir_b,
        tet_fiber_k1_b,
        tet_fiber_k2_b,
        tet_fiber_dir_c,
        tet_fiber_k1_c,
        tet_fiber_k2_c,
        tet_fiber_dir_d,
        tet_fiber_k1_d,
        tet_fiber_k2_d,
        node_tet_entries,
        tread_ids
    );
    input.node_tet_offsets = vec![0; 5];
    input.positions = [
        [0., 0., 0.],
        [0.3, 1., 0.],
        [1., 0., 0.],
        [0.8, -bend.cos(), bend.sin()],
    ]
    .map(|p| {
        [
            (size * p[0]) as f32,
            (size * p[1] + 4.) as f32,
            (size * p[2]) as f32,
            0.,
        ]
    })
    .to_vec();
    input.rest_positions = input.positions.clone();
    input.masses = vec![1., 2., 3., 4.];
    input.params.gravity_y = 0.;
    input.params.pressure_pa = 0.;
    input.params.node_damping = 0.;
    input.params.xpbd_volume_compliance = 1e30;
    input.params.xpbd_bead_compliance = 1e30;
    input.params.xpbd_sidewall_ply_compliance = 1e30;
    input.xpbd_dihedral_indices = vec![[0, 1, 2, 3]];
    input.xpbd_dihedral_params = vec![[0.4, compliance, 0., 0.]];
    input.node_dihedral_offsets = vec![0, 1, 2, 3, 4];
    input.node_dihedral_entries = vec![0, 1, 2, 3];
    input
}

#[test]
#[ignore = "native GPU; single/atlas bending projection vs independent five-point derivative"]
fn analytic_dihedral_projection_gpu() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireFemGpuKernel::new(device);
    let mut inputs = Vec::new();
    for size in [0.05, 0.5, 2.] {
        for bend in [
            -3.140592653589793,
            -0.7,
            -0.001,
            0.001,
            0.7,
            3.140592653589793,
        ] {
            for compliance in [0., 1e-4] {
                inputs.push(fixture(size, bend, compliance));
            }
        }
    }
    let atlas = kernel
        .dispatch_batch_atlas_blocking(device, queue, &inputs)
        .unwrap();
    let mut maximum = [0f64; 2];
    for (case, input) in inputs.iter().enumerate() {
        let p: Quad =
            std::array::from_fn(|a| std::array::from_fn(|i| input.positions[a][i] as f64));
        let h = (p[2][0] - p[0][0]).abs() * 1e-6;
        let g: Quad = std::array::from_fn(|a| {
            std::array::from_fn(|i| {
                let sample = |offset: f64| {
                    let mut p = p;
                    p[a][i] += h * offset;
                    angle(&p)
                };
                ((sample(-2.) - 8. * sample(-1.) + 8. * sample(1.) - sample(2.)) / (12. * h))
                    .clamp(-1e4, 1e4)
            })
        });
        let dt = input.params.dt as f64;
        let denom = input.xpbd_dihedral_params[0][1] as f64 / (dt * dt).max(1e-12)
            + (0..4)
                .map(|a| g[a].iter().map(|v| v * v).sum::<f64>() / input.masses[a] as f64)
                .sum::<f64>();
        let lambda = -(angle(&p) - input.xpbd_dihedral_params[0][0] as f64) / denom;
        let single = kernel.dispatch_blocking(device, queue, input).unwrap();
        for (route, result) in [(0, &single), (1, &atlas[case])] {
            assert!(
                result
                    .positions
                    .iter()
                    .chain(&result.velocities)
                    .chain(&result.forces)
                    .flatten()
                    .all(|v| v.is_finite())
            );
            assert!(result.forces.iter().flatten().all(|v| *v == 0.));
            for a in 0..4 {
                for i in 0..3 {
                    let dp = g[a][i] * lambda / input.masses[a] as f64;
                    for (actual, expected) in [
                        (result.positions[a][i], p[a][i] + dp),
                        (result.velocities[a][i], (dp / dt).clamp(-12., 12.)),
                    ] {
                        let error = (actual as f64 - expected).abs() / (1. + expected.abs());
                        maximum[route] = maximum[route].max(error);
                        assert!(
                            error <= 2e-5,
                            "case {case} route {route} node {a} axis {i}: {error:e}"
                        );
                    }
                }
            }
        }
    }
    println!(
        "36 isolated bending projections, nonuniform masses, two compliances, three scales, both fold orientations: scaled position/velocity errors {maximum:?}"
    );
}
