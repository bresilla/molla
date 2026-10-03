//! Prepare live primitive GPU wiring, optionally advancing one CPU/GPU frame.

use molla_compute::Engine;
use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxes, JointAxis, JointMotor, JointSoftness};
use molla_sim::runtime::{
    BodyDesc, ColliderDesc, ColliderGeometry, JointBreakLimits, JointDesc, RigidScene,
};
use molla_sim::{BodyParams, ColliderMass, JointTargetMode, JointType, MassProperties};
use molla_solvers::rigid_world::{GpuContactCapacity, RigidWorld};
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn fixture(compliant: bool, loaded: bool) -> Result<RigidScene> {
    let mut scene = RigidScene::new()?;
    let mut bodies = Vec::new();
    for position in [Vec3::Y, Vec3::Y + Vec3::X * 0.1] {
        bodies.push(scene.insert_body(BodyDesc {
            params: BodyParams::new().with_position(position),
            additional_mass: Some(MassProperties {
                mass: 2.,
                local_com: Vec3::new(0.01, 0.02, -0.03),
                inertia: Mat3::from_cols(
                    Vec3::new(1., 0.1, 0.05),
                    Vec3::new(0.1, 1.2, 0.08),
                    Vec3::new(0.05, 0.08, 0.9),
                ),
            }),
            ..Default::default()
        })?);
    }
    scene.insert_joint(JointDesc {
        parent: None,
        child: bodies[0],
        kind: JointType::Free,
        axis: Vec3::X,
        frame_parent: Transform::from_translation(Vec3::Y),
        frame_child: Transform::IDENTITY,
    })?;
    let joint = scene.insert_joint(JointDesc {
        parent: Some(bodies[1]),
        child: bodies[0],
        kind: JointType::D6,
        axis: Vec3::X,
        frame_parent: Transform::new(Vec3::new(0.02, -0.03, 0.01), Quat::from_rotation_z(0.3)),
        frame_child: Transform::new(Vec3::new(-0.01, 0.02, 0.03), Quat::from_rotation_x(-0.2)),
    })?;
    for axis in JointAxis::ALL {
        scene.set_joint_limits(joint, axis, Some([-0.5, 0.5]))?;
        scene.set_joint_velocity_limit(joint, axis, Some(0.4))?;
    }
    let softness = JointSoftness {
        natural_frequency: 3.,
        damping_ratio: 0.7,
    };
    if compliant {
        scene.set_joint_softness(joint, Some(softness))?;
    }
    if loaded {
        scene.set_joint_limits(joint, JointAxis::AngX, Some([0., 0.5]))?;
    }
    let motor = JointMotor {
        mode: JointTargetMode::PositionVelocity,
        target_position: 0.1,
        stiffness: 30.,
        damping: 2.,
        max_force: 5.,
        ..Default::default()
    };
    scene.set_joint_motor(joint, JointAxis::AngX, motor)?;
    scene.set_joint_brake(joint, JointAxis::AngX, 2.)?;
    scene.set_joint_break_limits(
        joint,
        JointBreakLimits {
            force: Some(100.),
            torque: Some(100.),
        },
    )?;
    let external = scene.insert_loop_joint(
        JointDesc {
            parent: Some(bodies[1]),
            child: bodies[0],
            kind: JointType::D6,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        },
        softness,
    )?;
    scene.set_joint_locked_axes(
        external,
        JointAxes(JointAxes::ALL.0 & !(1 << JointAxis::AngX as u32)),
    )?;
    scene.set_joint_motor(external, JointAxis::AngX, motor)?;
    scene.set_joint_brake(external, JointAxis::AngX, 1.)?;
    scene.set_joint_velocity_limit(external, JointAxis::AngX, Some(0.3))?;
    scene.set_body_damping(bodies[0], 0.2, 0.3)?;
    scene.set_body_velocity(
        bodies[0],
        SpatialVector::new(Vec3::new(0.01, 0.02, -0.03), Vec3::new(0.02, 0.01, 0.03)),
    )?;
    let mut collider = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.1 });
    collider.parent = Some(bodies[0]);
    collider.mass = ColliderMass::Mass(0.);
    scene.insert_collider(collider)?;
    let mut floor = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(2., 0.1, 2.),
    });
    floor.pose = Transform::from_translation(-Vec3::Y * 0.1);
    scene.insert_collider(floor)?;
    if !scene.model().joint_reversed.host()?.contains(&1) {
        return Err("fixture did not compile an authored reversed tree joint".into());
    }
    if loaded {
        let index = scene
            .joint_index(joint)
            .ok_or("missing loaded tree joint")?
            .index();
        let coord = scene.model().joint_coord_offset.host()?[index] as usize;
        let dof = scene.model().joint_dof_offset.host()?[index] as usize;
        let q = scene.state().joint_q.host()?;
        let rotation = Quat::from_xyzw(q[coord + 3], q[coord + 4], q[coord + 5], q[coord + 6]);
        let (x, _, _) = rotation.to_euler(glam::EulerRot::XYZ);
        let record = scene.model().joint_compliance.host()?[dof + 3];
        if record.enabled == 0 || (x >= record.lower && x <= record.upper) {
            return Err("loaded fixture has no active tree compliance".into());
        }
    }
    Ok(scene)
}

fn probe(world: &mut RigidWorld, compliant: bool, loaded: bool) -> Result<()> {
    const DT: f64 = 1. / 240.;
    const TOLERANCE: f64 = 1e-3;
    let mut reference = RigidWorld::from_scene(fixture(compliant, loaded)?)?;
    reference.solver.substeps = world.solver.substeps;
    reference
        .step(DT)
        .map_err(|e| format!("CPU one-frame wiring: {e}"))?;
    world
        .step(DT)
        .map_err(|e| format!("GPU one-frame wiring: {e}"))?;
    if !world.quarantined_bodies().is_empty()
        || !reference.quarantined_bodies().is_empty()
        || !world.joint_break_events().is_empty()
        || !reference.joint_break_events().is_empty()
    {
        return Err("one-frame wiring produced quarantine or unexpected break events".into());
    }
    let mut position: f64 = 0.;
    let mut orientation: f64 = 0.;
    let mut linear: f64 = 0.;
    let mut angular: f64 = 0.;
    for (body, _) in world.scene.bodies() {
        let pose = world.scene.body_pose(body)?;
        let velocity = world.scene.body_velocity(body)?;
        let expected = reference.scene.body_pose(body)?;
        let rate = reference.scene.body_velocity(body)?;
        println!(
            "gpu_frame_body\tcompliant_tree={compliant}\tloaded={loaded}\tbody={body:?}\tgpu_velocity={velocity:?}\tcpu_velocity={rate:?}"
        );
        if !pose.position.is_finite()
            || !pose.rotation.is_finite()
            || !velocity.linear.is_finite()
            || !velocity.angular.is_finite()
            || !expected.position.is_finite()
            || !expected.rotation.is_finite()
            || !rate.linear.is_finite()
            || !rate.angular.is_finite()
        {
            return Err("non-finite one-frame wiring state".into());
        }
        position = position.max((pose.position - expected.position).length());
        orientation = orientation.max(pose.rotation.angle_between(expected.rotation));
        linear = linear.max((velocity.linear - rate.linear).length());
        angular = angular.max((velocity.angular - rate.angular).length());
    }
    println!(
        "gpu_frame_probe\tcompliant_tree={compliant}\tloaded={loaded}\tdt={DT}\tposition_error={position:.9e}\torientation_error={orientation:.9e}\tlinear_rate_error={linear:.9e}\tangular_rate_error={angular:.9e}\ttolerance={TOLERANCE}\tphysics_steps=1"
    );
    println!(
        "gpu_frame_coordinates\tgpu_qd={:?}\tcpu_qd={:?}",
        world.scene.state().joint_qd.host()?,
        reference.scene.state().joint_qd.host()?
    );
    if [position, orientation, linear, angular]
        .into_iter()
        .any(|v| !v.is_finite() || v > TOLERANCE)
    {
        return Err("one-frame CPU/GPU wiring comparison exceeded its declared tolerance".into());
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let step = match args.as_slice() {
        [] => false,
        [arg] if arg == "--step" => true,
        [arg] if arg == "--help" => {
            println!(
                "prepare_primitives_gpu [--step]: prepare two GPU worlds, optionally run three CPU/GPU frame fixtures"
            );
            return Ok(());
        }
        _ => return Err("expected no arguments, --step or --help".into()),
    };
    let engine = Arc::new(Engine::new_gpu()?);
    let adapter = engine
        .adapter_info()
        .ok_or("missing GPU adapter information")?;
    if adapter.is_software {
        return Err("primitive GPU wiring requires a hardware adapter".into());
    }
    println!("adapter\t{adapter:?}");
    let mut fixtures = 0;
    for (compliant, loaded) in [(false, false), (true, false), (true, true)] {
        if loaded && !step {
            continue;
        }
        let mut world = RigidWorld::from_scene_gpu_with_contact_capacity(
            fixture(compliant, loaded)?,
            engine.clone(),
            GpuContactCapacity {
                pairs: 8,
                contacts_per_island: 16,
            },
        )?;
        let diagnostics = world
            .gpu_diagnostics()
            .ok_or("missing GPU runtime diagnostics")?;
        if diagnostics.model_compilations != 1 {
            return Err("expected one prepared GPU model".into());
        }
        println!(
            "gpu_wiring_ok\tcompliant_tree={compliant}\tloaded={loaded}\treversed_tree=true\tsoft_loop=true\taxis_caps=true\tbrakes=true\tbody_damping=true\tbreak_thresholds=true\tphysics_steps=0\tdiagnostics={diagnostics:?}"
        );
        if step {
            probe(&mut world, compliant, loaded)?;
        }
        fixtures += 1;
    }
    println!(
        "gpu_wiring_complete\tfixtures={fixtures}\tphysics_steps={}\tfull_plan_acceptance=false",
        u32::from(step)
    );
    Ok(())
}
