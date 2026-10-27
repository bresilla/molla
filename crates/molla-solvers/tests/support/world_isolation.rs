use molla_math::{SpatialVector, Vec3};
use molla_sim::runtime::{BodyDesc, RigidScene};
use molla_solvers::rigid_world::RigidWorld;

pub fn verify(mut make_world: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    let create = |gravity: Vec3, position: Vec3, velocity: Vec3| {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(gravity).unwrap();
        let mut desc = BodyDesc::default();
        desc.params.pose.position = position;
        let body = scene.insert_body(desc).unwrap();
        scene
            .set_body_velocity(body, SpatialVector::new(velocity, Vec3::ZERO))
            .unwrap();
        (scene, body)
    };
    let mut gravity_a = Vec3::new(1., -3., 2.);
    let gravity_b = Vec3::new(-2., 1., -4.);
    let mut velocity_a = Vec3::new(0.2, 0.1, -0.3);
    let mut velocity_b = Vec3::new(-0.4, 0.3, 0.1);
    let (scene_a, body_a) = create(gravity_a, Vec3::new(2., 3., -1.), velocity_a);
    let (scene_b, body_b) = create(gravity_b, Vec3::new(-3., 1., 2.), velocity_b);
    let mut a = make_world(scene_a);
    let mut b = make_world(scene_b);
    a.solver.substeps = 3;
    b.solver.substeps = 3;
    let stamp_b = b.scene.model_stamp();
    let mut extra = None;
    let dt = 1. / 120.;
    for frame in 0..24 {
        let pose_b = b.scene.body_pose(body_b).unwrap();
        let motion_b = b.scene.body_velocity(body_b).unwrap();
        #[cfg(feature = "gpu")]
        let diagnostics_b = b.gpu_diagnostics();
        if frame == 4 {
            a.set_reaction_readout(true).unwrap();
        }
        if frame == 8 {
            gravity_a = Vec3::new(3., 2., -1.);
            a.scene.set_gravity(gravity_a).unwrap();
            assert!(a.scene.set_gravity(Vec3::splat(f64::NAN)).is_err());
        }
        if frame == 12 {
            extra = Some(a.scene.insert_body(BodyDesc::default()).unwrap());
        }
        if frame == 16 {
            a.scene.remove_body(extra.take().unwrap()).unwrap();
        }
        for _ in 0..2 {
            a.step(dt).unwrap();
            velocity_a += gravity_a * dt;
        }
        assert_eq!(b.scene.model_stamp(), stamp_b);
        assert_eq!(b.scene.body_pose(body_b).unwrap(), pose_b);
        assert_eq!(b.scene.body_velocity(body_b).unwrap(), motion_b);
        assert_eq!(b.completed_steps(), frame);
        #[cfg(feature = "gpu")]
        if let Some(before) = diagnostics_b {
            let after = b.gpu_diagnostics().unwrap();
            assert_eq!(after.model_compilations, before.model_compilations);
            assert_eq!(after.completed_frames, before.completed_frames);
        }
        b.step(dt).unwrap();
        velocity_b += gravity_b * dt;
        for (world, body, expected) in [(&a, body_a, velocity_a), (&b, body_b, velocity_b)] {
            let actual = world.scene.body_velocity(body).unwrap();
            let error = (actual.linear - expected).length();
            assert!(error <= tolerance, "frame={frame} velocity_error={error}");
            assert!(actual.angular.length() <= tolerance);
            assert!(world.scene.body_pose(body).unwrap().position.is_finite());
        }
    }
    drop(a);
    for _ in 0..8 {
        b.step(dt).unwrap();
        velocity_b += gravity_b * dt;
        let error = (b.scene.body_velocity(body_b).unwrap().linear - velocity_b).length();
        assert!(error <= tolerance, "surviving_world_velocity_error={error}");
    }
    assert_eq!(b.scene.model_stamp(), stamp_b);
    assert_eq!(b.completed_steps(), 32);
}
