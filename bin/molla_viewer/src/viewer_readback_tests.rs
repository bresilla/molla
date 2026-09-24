use super::*;
use bevy::ecs::system::RunSystemOnce;

#[test]
#[ignore = "requires a GPU adapter"]
fn viewer_readback_scene_switches_preserve_solver_output() {
    let (model, visuals) = Sim::build_scene_model(Scene::Pile);
    let (sim, _) = Sim::finalize(Scene::Pile, model, visuals);
    let mut world = World::new();
    world.insert_resource(sim);
    world.insert_resource(SimRunState::default());
    world.insert_resource(SimStats::default());
    world.insert_resource(GpuNpExampleState::default());
    world.insert_resource(GpuBpExampleState::default());
    world.insert_resource(ReferencePendulumState::default());
    let scenes = std::iter::once(Scene::PpfWoven)
        .chain(Scene::ALL.iter().copied().filter(|scene| matches!(
            scene.solver_kind(),
            SolverKind::Barrier | SolverKind::BarrierCpu | SolverKind::BarrierCpuChains
        )))
        .chain([Scene::VbdCloth, Scene::Arm, Scene::Pile, Scene::PpfWoven]);
    for scene in scenes {
        eprintln!("viewer readback: {}", scene.label());
        world.resource_mut::<Sim>().rebuild_scene(scene);
        let initial = world.resource::<Sim>().state_a.particle_q.host().unwrap().to_vec();
        for _ in 0..2 {
            world.run_system_once(step_simulation).unwrap();
        }
        let sim = world.resource::<Sim>();
        let particles = sim.state_a.particle_q.host().unwrap();
        assert!(particles.iter().all(|p| p.is_finite()));
        assert!(sim.state_a.body_q.host().unwrap().iter().all(|p| p.position.is_finite()));
        if matches!(scene, Scene::PpfWoven | Scene::VbdCloth) {
            assert!(initial.iter().zip(particles).any(|(a, b)| (*a - *b).length() > 1e-8));
        }
    }
}

#[test]
#[ignore = "requires a GPU adapter"]
fn viewer_readback_respects_buffer_authority() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut buffer = mp::Buffer::from_vec(vec![7u32], mp::BufferUsage::StorageReadWrite);
    readback_buffer(&mut buffer, device, queue).unwrap();
    assert_eq!(buffer.host().unwrap(), &[7]);
    buffer.ensure_device(device).unwrap();
    readback_buffer(&mut buffer, device, queue).unwrap();
    assert_eq!(buffer.host().unwrap(), &[7]);
    buffer.upload(queue).unwrap();
    readback_buffer(&mut buffer, device, queue).unwrap();
    assert_eq!(buffer.host().unwrap(), &[7]);
    buffer.host_mut().unwrap()[0] = 42;
    readback_buffer(&mut buffer, device, queue).unwrap();
    assert_eq!(buffer.host().unwrap(), &[42]);
    buffer.upload(queue).unwrap();
    queue.write_buffer(buffer.device_buffer().unwrap(), 0, &99u32.to_ne_bytes());
    buffer.mark_device_dirty();
    readback_buffer(&mut buffer, device, queue).unwrap();
    assert_eq!(buffer.host().unwrap(), &[99]);
}
