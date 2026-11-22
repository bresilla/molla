use molla_core::WorldId;
use molla_math::{
    Vec3,
    constitutive::{ElasticMaterial, ElasticModel, MembraneStrainLimit},
};
use molla_sim::{Contacts, ElasticElement, Model, ModelBuilder, State};
use molla_solvers::{BarrierSolver, Solver};

fn scene(shrink: [f64; 2]) -> (Model, ElasticElement) {
    let rest = [Vec3::Y, Vec3::Y + Vec3::X, Vec3::Y * 2.0];
    let mut b = ModelBuilder::new();
    for p in rest {
        b.add_particle(p, 1.0, 0.01, WorldId(0));
    }
    let e = ElasticElement::membrane(
        &rest,
        [0, 1, 2],
        0.01,
        ElasticMaterial {
            model: ElasticModel::StableNeoHookean,
            mu: 1.0,
            lambda: 1.0,
        },
    )
    .unwrap()
    .with_strain_limit(MembraneStrainLimit {
        strain: 0.05,
        shrink,
    }, 3.0)
    .unwrap();
    b.add_elastic_elements(vec![e.clone()]).unwrap();
    (b.build().unwrap(), e)
}

fn step(
    solver: &mut BarrierSolver,
    m: &Model,
    input: &State,
    output: &mut State,
) -> molla_core::Result<()> {
    solver.step(
        m,
        input,
        &m.control(),
        &Contacts::with_capacity(0),
        0.01,
        output,
    )
}

fn kick(m: &Model) -> State {
    let mut state = m.state().unwrap();
    state.particle_qd.host_mut().unwrap()[2] = Vec3::new(20.0, (0.96f64.sqrt() - 1.0) / 0.01, 0.0);
    state
}

#[test]
fn public_cpu_solver_caps_principal_shear_and_preserves_rejected_output() {
    for shrink in [[1.0; 2], [0.8, 0.9]] {
        let (m, e) = scene(shrink);
        let limit = e.strain_limit().unwrap();
        let mut solver = BarrierSolver::new();
        solver.params.gravity = Vec3::ZERO;
        let mut input = kick(&m);
        let mut output = m.state().unwrap();
        let mut peak = 0.0f64;
        for _ in 0..20 {
            step(&mut solver, &m, &input, &mut output).unwrap();
            let stretch = MembraneStrainLimit::largest_stretch(
                e.deformation(output.particle_q.host().unwrap()).unwrap(),
            )
            .unwrap();
            assert!(stretch < limit.maximum_stretch().unwrap());
            peak = peak.max(stretch);
            std::mem::swap(&mut input, &mut output);
        }
        assert!(
            peak > limit.maximum_stretch().unwrap() - 0.002,
            "peak={peak}"
        );
        input.particle_q.host_mut().unwrap()[1] = Vec3::Y + Vec3::X * 2.0;
        let before = output.particle_q.host().unwrap().to_vec();
        let velocity = output.particle_qd.host().unwrap().to_vec();
        assert!(step(&mut solver, &m, &input, &mut output).is_err());
        assert_eq!(output.particle_q.host().unwrap(), before);
        assert_eq!(output.particle_qd.host().unwrap(), velocity);
    }
}

#[test]
fn rest_and_runtime_strain_domains_are_checked_separately_from_rank() {
    let (mut m, e) = scene([1.0; 2]);
    let rest = m.initial_particle_q.host().unwrap();
    let mut shear = rest.to_vec();
    shear[2] = rest[0] + Vec3::new(0.2, 0.96f64.sqrt(), 0.0);
    assert!(
        e.deformation(&shear)
            .unwrap()
            .x_axis
            .cross(e.deformation(&shear).unwrap().y_axis)
            .length()
            > 0.0
    );
    assert!(e.validate(&shear).is_err());
    assert!(e.energy(&shear).is_err());
    assert!(e.response(&shear, true).is_err());
    let rest = rest.to_vec();
    m.initial_particle_q
        .host_mut()
        .unwrap()
        .copy_from_slice(&shear);
    let old = m.elastic_elements().clone();
    assert!(m.set_elastic_elements(vec![e.clone()]).is_err());
    assert!(old.same_revision(m.elastic_elements()));
    let mut direction = vec![Vec3::ZERO; 3];
    direction[1] = Vec3::X * 1000.0;
    let alpha = e.step_bound(&rest, &direction).unwrap();
    assert!(alpha > 0.0 && alpha < 0.001);
    for i in 0..=100 {
        let x: Vec<_> = rest
            .iter()
            .zip(&direction)
            .map(|(x, d)| *x + *d * (alpha * i as f64 / 100.0))
            .collect();
        e.validate(&x).unwrap();
    }
}

#[cfg(feature = "resource-counters")]
#[test]
#[ignore = "requires GPU; run via make test-ppf-strain-gpu"]
fn public_gpu_principal_caps_match_cpu_with_shrink_and_external_edits() {
    use molla_compute::Engine;
    let engine = Engine::new_gpu().unwrap();
    for shrink in [[1.0; 2], [0.8, 0.9]] {
        let (m, e) = scene(shrink);
        let mut gpu = BarrierSolver::new_gpu(&engine).unwrap();
        gpu.params.gravity = Vec3::ZERO;
        gpu.params.max_newton_iters = 12;
        let mut cpu = BarrierSolver::new().with_params(gpu.params);
        let (mut a, mut b) = (kick(&m), m.state().unwrap());
        let (mut c, mut d) = (kick(&m), m.state().unwrap());
        let mut peak = 0.0f64;
        let mut worst = 0.0f64;
        for _ in 0..80 {
            step(&mut gpu, &m, &a, &mut b).unwrap();
            step(&mut cpu, &m, &c, &mut d).unwrap();
            let stretch = MembraneStrainLimit::largest_stretch(
                e.deformation(b.particle_q.host().unwrap()).unwrap(),
            )
            .unwrap();
            assert!(stretch < e.strain_limit().unwrap().maximum_stretch().unwrap());
            peak = peak.max(stretch);
            for (x, y) in b
                .particle_q
                .host()
                .unwrap()
                .iter()
                .zip(d.particle_q.host().unwrap())
            {
                worst = worst.max(x.distance(*y));
            }
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut c, &mut d);
        }
        eprintln!("principal cap shrink={shrink:?} peak={peak:.9} trajectory_error={worst:.3e}");
        assert!(peak > e.strain_limit().unwrap().maximum_stretch().unwrap() - 0.002);
        assert!(worst < 0.001);
        a.particle_q.host_mut().unwrap()[1] = Vec3::Y + Vec3::X * 2.0;
        let before = b.particle_q.host().unwrap().to_vec();
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        assert!(step(&mut gpu, &m, &a, &mut b).is_err());
        assert_eq!(allocations.finish().observations, 0);
        assert_eq!(b.particle_q.host().unwrap(), before);
        let fresh = kick(&m);
        step(&mut gpu, &m, &fresh, &mut b).unwrap();
    }
}

#[cfg(feature = "resource-counters")]
#[test]
#[ignore = "requires GPU; run via make test-ppf-strain-gpu"]
fn resident_principal_caps_add_no_step_buffer_allocations() {
    use molla_compute::Engine;
    use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
    use molla_kernels::{BarrierGpuParams, BarrierNewtonGpu};
    let engine = Engine::new_gpu().unwrap();
    let (m, e) = scene([1.0; 2]);
    let rest: Vec<_> = m
        .initial_particle_q
        .host()
        .unwrap()
        .iter()
        .map(|v| [v.x as f32, v.y as f32, v.z as f32])
        .collect();
    let mut gpu = BarrierNewtonGpu::new_with_elastic_elements(
        engine.device().unwrap(),
        engine.queue().unwrap(),
        &rest,
        &[1.0; 3],
        &[],
        &[],
        &[[0, 1, 2]],
        &[],
        &[],
        &[e.clone()],
        BarrierGpuParams {
            gravity: [0.0; 3],
            ..Default::default()
        },
    )
    .unwrap();
    gpu.set_velocities(&[[0.0; 3], [0.0; 3], [20.0, -2.0204103, 0.0]])
        .unwrap();
    let allocations = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    let dispatches = DispatchScope::start().unwrap();
    let now = std::time::Instant::now();
    for _ in 0..20 {
        gpu.step(0.01).unwrap();
    }
    engine
        .device()
        .unwrap()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let elapsed = now.elapsed().as_secs_f64();
    let a = allocations.finish();
    let t = transfers.finish();
    let d = dispatches.finish();
    assert_eq!(a.observations, 0);
    assert!(!t.overflowed && !d.overflowed);
    let x: Vec<_> = gpu
        .read_positions()
        .unwrap()
        .into_iter()
        .map(|v| Vec3::from_array(v.map(f64::from)))
        .collect();
    e.validate(&x).unwrap();
    eprintln!(
        "principal cap resident steps=20 elapsed_ms={:.3} allocations=0 uploads={} readbacks={} dispatches={}",
        elapsed * 1000.0,
        t.upload_bytes,
        t.readback_bytes,
        d.encoded_dispatches
    );
}
