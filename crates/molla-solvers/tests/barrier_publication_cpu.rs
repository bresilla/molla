use molla_core::{BodyId, WorldId};
use molla_math::{Mat3, Real, SpatialVector, Transform, Vec3};
use molla_sim::{BodyParams, Contacts, Model, ModelBuilder, State};
use molla_solvers::{BarrierSolver, Solver};
use molla_storage::{Buffer, BufferUsage};

fn model() -> Model {
    let mut b = ModelBuilder::new();
    b.add_particle(Vec3::Y, 1.0, 0.01, WorldId(0));
    b.add_body(BodyParams::new().with_mass(1.0));
    b.build().unwrap()
}

fn snapshot(s: &State) -> (Vec<Vec3>, Vec<Vec3>, Vec<Transform>, Vec<SpatialVector>) {
    (
        s.particle_q.host().unwrap().to_vec(),
        s.particle_qd.host().unwrap().to_vec(),
        s.body_q.host().unwrap().to_vec(),
        s.body_qd.host().unwrap().to_vec(),
    )
}

fn sentinels(model: &Model) -> State {
    let mut s = model.state().unwrap();
    s.particle_q.host_mut().unwrap().fill(Vec3::splat(41.0));
    s.particle_qd.host_mut().unwrap().fill(Vec3::splat(42.0));
    s.body_q
        .host_mut()
        .unwrap()
        .fill(Transform::from_translation(Vec3::splat(43.0)));
    s.body_qd.host_mut().unwrap().fill(SpatialVector {
        linear: Vec3::splat(44.0),
        angular: Vec3::splat(45.0),
    });
    s
}

fn step(
    solver: &mut BarrierSolver,
    model: &Model,
    input: &State,
    output: &mut State,
) -> molla_core::Result<()> {
    solver.step(
        model,
        input,
        &model.control(),
        &Contacts::with_capacity(0),
        1.0 / 240.0,
        output,
    )
}

#[test]
fn every_output_length_is_checked_before_publication() {
    let m = model();
    let input = m.state().unwrap();
    for buffer in 0..4 {
        let mut solver = BarrierSolver::new();
        solver.last_newton_iters = 17;
        solver.last_min_gap = 19.0;
        let mut output = sentinels(&m);
        let usage = BufferUsage::StorageReadWrite;
        match buffer {
            0 => output.particle_q = Buffer::from_vec(vec![], usage),
            1 => output.particle_qd = Buffer::from_vec(vec![], usage),
            2 => output.body_q = Buffer::from_vec(vec![], usage),
            _ => output.body_qd = Buffer::from_vec(vec![], usage),
        }
        let before = snapshot(&output);
        assert!(
            step(&mut solver, &m, &input, &mut output)
                .unwrap_err()
                .to_string()
                .contains("output")
        );
        assert_eq!(snapshot(&output), before);
        assert_eq!((solver.last_newton_iters, solver.last_min_gap), (17, 19.0));
        let mut retry = m.state().unwrap();
        let mut expected = m.state().unwrap();
        step(&mut solver, &m, &input, &mut retry).unwrap();
        step(&mut BarrierSolver::new(), &m, &input, &mut expected).unwrap();
        assert_eq!(snapshot(&retry), snapshot(&expected));
    }
}

#[test]
fn malformed_model_arrays_and_shape_references_are_refused() {
    for case in 0..6 {
        let mut m = model();
        let input = m.state().unwrap();
        let mut output = sentinels(&m);
        match case {
            0 => m.particle_mass = Buffer::from_vec(vec![], BufferUsage::Storage),
            1 => m.spring_count = 1,
            2 => m.shape_count = 1,
            3 => m.soft_surface_tris.push([0, 1, u32::MAX]),
            4 => m.body_inv_inertia = Buffer::from_vec(vec![], BufferUsage::Storage),
            _ => m.particle_inv_mass.host_mut().unwrap()[0] = Real::NAN,
        }
        let before = snapshot(&output);
        assert!(step(&mut BarrierSolver::new(), &m, &input, &mut output).is_err());
        assert_eq!(snapshot(&output), before);
    }
    let mut b = ModelBuilder::new();
    b.add_particle(Vec3::Y, 1.0, 0.01, WorldId(0));
    b.add_shape_sphere(BodyId::NONE, 0.1, Transform::IDENTITY)
        .unwrap();
    let mut m = b.build().unwrap();
    let input = m.state().unwrap();
    let mut output = sentinels(&m);
    m.shape_body.host_mut().unwrap()[0] = 0;
    let before = snapshot(&output);
    assert!(step(&mut BarrierSolver::new(), &m, &input, &mut output).is_err());
    assert_eq!(snapshot(&output), before);
}

#[test]
fn post_solve_body_overflow_rolls_back_affine_state_and_all_output() {
    let mut b = ModelBuilder::new();
    b.add_particle(Vec3::new(1.0, 1.0005, 0.0), 1.0, 0.01, WorldId(0));
    let ball = b.add_body(BodyParams::new().with_mass(1.0));
    b.add_shape_sphere(ball, 1.0, Transform::from_translation(Vec3::X))
        .unwrap();
    let cube = b.add_body(
        BodyParams::new()
            .with_mass(1.0)
            .with_pose(Transform::from_translation(Vec3::new(5.0, 3.0, 0.0))),
    );
    b.add_shape_box(cube, Vec3::splat(0.1), Transform::IDENTITY)
        .unwrap();
    let mut m = b.build().unwrap();
    let mut initial = m.state().unwrap();
    initial.body_qd.host_mut().unwrap()[cube.0 as usize].linear = Vec3::X;
    let mut solver = BarrierSolver::new();
    solver.params.gravity = Vec3::ZERO;
    solver.params.dhat = 0.01;
    let mut reference = BarrierSolver::new().with_params(solver.params);
    let mut input = m.state().unwrap();
    let mut ref_input = m.state().unwrap();
    step(&mut solver, &m, &initial, &mut input).unwrap();
    step(&mut reference, &m, &initial, &mut ref_input).unwrap();
    assert_eq!(snapshot(&input), snapshot(&ref_input));
    let stats = (solver.last_newton_iters, solver.last_min_gap);
    let original_inertia = m.body_inv_inertia.host().unwrap()[ball.0 as usize];
    m.body_inv_inertia.host_mut().unwrap()[ball.0 as usize] = Mat3::IDENTITY * Real::MAX;
    let mut output = sentinels(&m);
    let before = snapshot(&output);
    let err = step(&mut solver, &m, &input, &mut output)
        .unwrap_err()
        .to_string();
    assert!(err.contains("body motion"), "{err}");
    assert_eq!(snapshot(&output), before);
    assert_eq!((solver.last_newton_iters, solver.last_min_gap), stats);
    m.body_inv_inertia.host_mut().unwrap()[ball.0 as usize] = original_inertia;
    let mut expected = m.state().unwrap();
    step(&mut solver, &m, &input, &mut output).unwrap();
    step(&mut reference, &m, &ref_input, &mut expected).unwrap();
    assert_eq!(snapshot(&output), snapshot(&expected));
}

#[test]
fn removing_an_affine_body_rebuilds_the_cached_backend() {
    let mut b = ModelBuilder::new();
    b.add_particle(Vec3::Y, 1.0, 0.01, WorldId(0));
    let cube = b.add_body(
        BodyParams::new()
            .with_mass(1.0)
            .with_pose(Transform::from_translation(Vec3::splat(5.0))),
    );
    b.add_shape_box(cube, Vec3::splat(0.1), Transform::IDENTITY)
        .unwrap();
    let old = b.build().unwrap();
    let mut solver = BarrierSolver::new();
    step(
        &mut solver,
        &old,
        &old.state().unwrap(),
        &mut old.state().unwrap(),
    )
    .unwrap();
    let mut b = ModelBuilder::new();
    b.add_particle(Vec3::Y, 1.0, 0.01, WorldId(0));
    let new = b.build().unwrap();
    let input = new.state().unwrap();
    let mut output = new.state().unwrap();
    let mut expected = new.state().unwrap();
    step(&mut solver, &new, &input, &mut output).unwrap();
    step(&mut BarrierSolver::new(), &new, &input, &mut expected).unwrap();
    assert_eq!(snapshot(&output), snapshot(&expected));
}
