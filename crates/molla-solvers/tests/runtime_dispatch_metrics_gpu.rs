#![cfg(feature = "resource-counters")]

use molla_compute::Engine;
use molla_core::gpu_metrics::DispatchScope;
use molla_math::{Mat3, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, JointDesc, RigidScene};
use molla_sim::{JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;
use std::sync::Arc;

#[test]
#[ignore = "requires GPU; run via make test-primitives-dispatch-metrics-gpu"]
fn dispatch_counts_follow_substeps_and_batched_reaction_stages_not_body_count() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    for kind in [JointType::Fixed, JointType::Free, JointType::D6] {
        for bodies in [1, 32] {
            let mut scene = RigidScene::new().unwrap();
            for _ in 0..bodies {
                let body = scene
                    .insert_body(BodyDesc {
                        additional_mass: Some(MassProperties {
                            mass: 1.,
                            local_com: Vec3::ZERO,
                            inertia: Mat3::IDENTITY,
                        }),
                        ..Default::default()
                    })
                    .unwrap();
                scene
                    .insert_joint(JointDesc {
                        parent: None,
                        child: body,
                        kind,
                        axis: Vec3::X,
                        frame_parent: Transform::IDENTITY,
                        frame_child: Transform::IDENTITY,
                    })
                    .unwrap();
            }
            let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
            for substeps in [1, 4] {
                world.solver.substeps = substeps;
                for reactions in [false, true] {
                    world.set_reaction_readout(reactions).unwrap();
                    world.step(0.004).unwrap();
                    let compilations = world.gpu_diagnostics().unwrap().model_compilations;
                    let scope = DispatchScope::start().unwrap();
                    world.step(0.004).unwrap();
                    let counts = scope.finish();
                    assert!(!counts.overflowed);
                    assert_eq!(
                        counts.encoded_dispatches,
                        1 + substeps as u64
                            * (if reactions { 5 } else { 1 } + u64::from(kind != JointType::Fixed)),
                        "kind={kind:?} bodies={bodies} substeps={substeps} reactions={reactions}"
                    );
                    assert_eq!(
                        world.gpu_diagnostics().unwrap().model_compilations,
                        compilations
                    );
                }
            }
        }
    }
}
